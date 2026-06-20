"""The framework-agnostic consensus gate.

This is what the package ships: a small, dependency-free runtime that takes a
state snapshot + a proposed action + a checkable claim, fans out N read-only
juror calls in parallel, applies a red-flag filter, and returns a verdict via
first-to-ahead-by-K voting.

The migration demo's `ConsensusGate` (in `jurors.py`) is now a thin domain-aware
wrapper around `Gate` that knows how to localize root causes against a
`MigrationWorld`. Everything generic lives here.

Design contract:

    juror_client: a callable (system_prompt: str, user_prompt: str) -> str
                  Returns the juror's raw response text. The Gate tolerates any
                  shape — malformed responses are caught by the red-flag filter.

    state: dict   The observable post-action world state. The Gate serializes it
                  into the juror prompt so each juror can re-derive the truth.

    action: dict  The brain's proposed irreversible action.

    claim: str    The brain's checkable post-state assertion.

    claim_kind:   An optional discriminator the juror prompt can route on (e.g.
                  "row_count", "schema", "diff_semantics"). Generic — the Gate
                  itself doesn't interpret it.

The Gate is intentionally NOT migration-aware. Adapters wire it into a host
framework (Claude Agent SDK, LangGraph, OpenAI Agents SDK) by mapping that
framework's "irreversible tool" surface onto these four arguments.
"""

from __future__ import annotations

import functools
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, TypeVar


# Stable juror system prompt used by the demo + adapters. Adapters can override
# it via Gate(juror_system=...) when domain-specific guidance helps.
DEFAULT_JUROR_SYSTEM = """You are an independent verification juror in a consensus gate.

You are READ-ONLY. You did not plan this action and you cannot see the planner's
reasoning. Your only job: decide whether the planner's CLAIM about the post-action
world state is actually TRUE, by inspecting the observable state yourself.

You MUST gather your own evidence — re-derive the relevant facts from the state
given to you. Do not take the claim on faith.

Respond with ONLY a JSON object, no prose around it:
{"vote": "PASS" | "FAIL", "evidence": "<the specific facts you checked>", "confidence": <0..1>}

PASS means the claim is fully supported by the state. FAIL means the state
contradicts the claim. When in doubt, look harder before voting; cite specifics
in evidence.
"""


class JurorClient(Protocol):
    """Anything the Gate can ask for a juror verdict.

    A `(system, user) -> str` callable also satisfies this protocol — see
    `as_juror_client` below for the adapter.
    """

    def vote(self, system: str, user: str) -> str: ...


JurorCallable = Callable[[str, str], str]


def as_juror_client(fn: JurorCallable) -> JurorClient:
    """Wrap a plain callable so it satisfies JurorClient."""

    class _Adapter:
        def vote(self, system: str, user: str) -> str:
            return fn(system, user)

    return _Adapter()


# --------------------------------------------------------------------------------------
# Vote / GateResult
# --------------------------------------------------------------------------------------


@dataclass
class Vote:
    juror_id: int
    raw: str
    vote: Optional[str] = None        # "PASS" | "FAIL" | None (unparsed)
    evidence: str = ""
    confidence: float = 0.0
    red_flagged: bool = False
    red_flag_reason: str = ""

    @property
    def counted(self) -> bool:
        return not self.red_flagged and self.vote in ("PASS", "FAIL")


@dataclass
class GateResult:
    decision: str                      # "PASS" | "FAIL" | "NO_CONSENSUS"
    pass_votes: int
    fail_votes: int
    red_flagged: int
    jurors_polled: int
    votes: list[Vote] = field(default_factory=list)
    root_cause: str = ""               # populated by the caller's localizer

    @property
    def blocked(self) -> bool:
        return self.decision != "PASS"


class GateBlocked(Exception):
    """Raised by `@gate.protect` when the consensus gate blocks an irreversible call.

    The result is attached as `.result` so callers can rollback / re-plan with full
    juror evidence (votes, red-flags, root_cause).
    """

    def __init__(self, action: dict, claim: str, result: "GateResult"):
        self.action = action
        self.claim = claim
        self.result = result
        super().__init__(
            f"quorum-py blocked {action.get('tool', '<callable>')}: "
            f"{result.decision} (PASS={result.pass_votes} FAIL={result.fail_votes} "
            f"polled={result.jurors_polled})"
            + (f" — {result.root_cause}" if result.root_cause else "")
        )


# --------------------------------------------------------------------------------------
# Red-flag filter
# --------------------------------------------------------------------------------------


def red_flag(
    raw: str, *, min_evidence_chars: int = 15, max_chars: int = 4000
) -> tuple[Optional[dict], str]:
    """Validate a juror's raw output. Returns (parsed_dict | None, reason).

    Drops malformed / over-long / no-evidence / no-vote responses BEFORE they count.
    This is MAKER's red-flagging, pointed at verification.
    """
    if not raw or len(raw) > max_chars:
        return None, "empty or over-long response"
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, "no JSON object found"
    try:
        obj = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None, "malformed JSON"
    if not isinstance(obj, dict):
        return None, "not a JSON object"
    vote = obj.get("vote")
    if vote not in ("PASS", "FAIL"):
        return None, "missing or invalid 'vote'"
    evidence = obj.get("evidence", "")
    if not isinstance(evidence, str) or len(evidence.strip()) < min_evidence_chars:
        return None, "no/insufficient evidence cited"
    return obj, ""


# --------------------------------------------------------------------------------------
# Gate — framework-agnostic
# --------------------------------------------------------------------------------------


class Gate:
    """A read-only consensus gate. Polls jurors in parallel waves, votes, returns a
    verdict via first-to-ahead-by-K. Knows nothing about your domain."""

    def __init__(
        self,
        juror_client: JurorClient | JurorCallable,
        *,
        k: int = 3,
        max_jurors: int = 24,
        batch_size: int = 6,
        max_workers: int = 8,
        juror_system: str = DEFAULT_JUROR_SYSTEM,
        on_vote: Optional[Callable[[Vote], None]] = None,
        prompt_builder: Optional[Callable[[dict, dict, str, str], str]] = None,
    ):
        self.juror_client: JurorClient = (
            juror_client if hasattr(juror_client, "vote") else as_juror_client(juror_client)
        )
        self.k = k
        self.max_jurors = max_jurors
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.juror_system = juror_system
        self.on_vote = on_vote
        self.prompt_builder = prompt_builder or _default_prompt_builder

    def evaluate(
        self,
        state: dict[str, Any],
        action: dict[str, Any],
        claim: str,
        claim_kind: str = "",
    ) -> GateResult:
        """Run the gate. Returns a verdict after at most `max_jurors` polls."""
        prompt = self.prompt_builder(state, action, claim, claim_kind)
        votes: list[Vote] = []
        pass_n = fail_n = redflag_n = 0
        polled = 0

        while polled < self.max_jurors:
            wave = min(self.batch_size, self.max_jurors - polled)
            batch = self._poll_batch(prompt, start_id=polled, n=wave)
            for v in batch:
                polled += 1
                votes.append(v)
                if self.on_vote:
                    self.on_vote(v)
                if v.red_flagged:
                    redflag_n += 1
                elif v.vote == "PASS":
                    pass_n += 1
                elif v.vote == "FAIL":
                    fail_n += 1
            # First-to-ahead-by-K, checked after each wave.
            if pass_n - fail_n >= self.k:
                return GateResult("PASS", pass_n, fail_n, redflag_n, polled, votes)
            if fail_n - pass_n >= self.k:
                return GateResult("FAIL", pass_n, fail_n, redflag_n, polled, votes)

        decision = (
            "PASS" if pass_n > fail_n
            else "FAIL" if fail_n > pass_n
            else "NO_CONSENSUS"
        )
        return GateResult(decision, pass_n, fail_n, redflag_n, polled, votes)

    def _poll_batch(self, prompt: str, start_id: int, n: int) -> list[Vote]:
        results: dict[int, Vote] = {}

        def one(jid: int) -> Vote:
            raw = self.juror_client.vote(self.juror_system, prompt)
            parsed, reason = red_flag(raw)
            if parsed is None:
                return Vote(juror_id=jid, raw=raw, red_flagged=True, red_flag_reason=reason)
            return Vote(
                juror_id=jid,
                raw=raw,
                vote=parsed["vote"],
                evidence=parsed.get("evidence", ""),
                confidence=float(parsed.get("confidence", 0.0)),
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(one, start_id + i): start_id + i for i in range(n)}
            for fut in as_completed(futs):
                v = fut.result()
                results[v.juror_id] = v
        return [results[k] for k in sorted(results)]

    # ----- decorator API ----------------------------------------------------

    def protect(
        self,
        claim: str | Callable[..., str],
        *,
        snapshot_state: Callable[..., dict[str, Any]],
        claim_kind: str | Callable[..., str] = "",
        rollback: Optional[Callable[..., None]] = None,
        raise_on_block: bool = True,
    ):
        """Decorator. Gate any callable on its post-state.

            gate = Gate(juror_client)

            @gate.protect(
                claim="after this, the file compiles and tests pass",
                snapshot_state=lambda *a, **kw: read_repo_state(),
                claim_kind="diff_semantics",
                rollback=lambda *a, **kw: git_reset_hard(),
            )
            def write_file(path, contents): ...

        Behavior: the wrapped function runs first (its effect must be sandboxed
        or rollback-able), then `snapshot_state(*args, **kwargs)` produces the
        post-state, the gate evaluates against `claim`, and on FAIL the
        decorator calls `rollback(*args, **kwargs)` (if given) and either
        raises `GateBlocked` (default) or returns the `GateResult`.

        `claim` and `claim_kind` may be strings (constant) or callables
        receiving the same args/kwargs as the wrapped function.
        """

        def _resolve(value, args, kwargs) -> str:
            return value(*args, **kwargs) if callable(value) else value

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                func_result = func(*args, **kwargs)
                state = snapshot_state(*args, **kwargs)
                resolved_claim = _resolve(claim, args, kwargs)
                resolved_kind = _resolve(claim_kind, args, kwargs)
                action = {
                    "tool": func.__name__,
                    "args": _safe_repr_args(args, kwargs),
                }
                gate_result = self.evaluate(
                    state, action, resolved_claim, resolved_kind
                )
                if gate_result.decision == "PASS":
                    return func_result
                if rollback is not None:
                    rollback(*args, **kwargs)
                if raise_on_block:
                    raise GateBlocked(action, resolved_claim, gate_result)
                return gate_result

            wrapper.__quorum_gate__ = self  # type: ignore[attr-defined]
            return wrapper

        return decorator


def _safe_repr_args(args: tuple, kwargs: dict) -> dict:
    """Best-effort serialization of call args for the juror prompt — never raises."""
    try:
        json.dumps({"a": args, "kw": kwargs})
        return {"args": list(args), "kwargs": dict(kwargs)}
    except (TypeError, ValueError):
        return {"args": [repr(a) for a in args],
                "kwargs": {k: repr(v) for k, v in kwargs.items()}}


F = TypeVar("F", bound=Callable[..., Any])


def _default_prompt_builder(
    state: dict[str, Any], action: dict[str, Any], claim: str, claim_kind: str
) -> str:
    return (
        "An agent proposes an IRREVERSIBLE action. Verify its claim.\n\n"
        f"PROPOSED ACTION: {json.dumps(action)}\n\n"
        f"PLANNER CLAIM: {claim}\n\n"
        "OBSERVABLE WORLD STATE (post-action). Re-derive the facts yourself:\n"
        f"STATE_JSON={json.dumps(state)} "
        f"CLAIM_KIND={claim_kind}\n\n"
        "Run your own checks against the state above and vote. Cite specifics."
    )
