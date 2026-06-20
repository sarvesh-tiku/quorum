"""Claude client abstraction: a MockClient (offline fixtures) and a LiveClient (API).

Per the build instructions: develop against a MOCK Claude client so the loop runs
offline; gate live calls behind ANTHROPIC_API_KEY. Both clients expose the same two
methods the rest of the system needs:

    decide(system, user)  -> str   # the brain: one strong reasoning call (Opus 4.8)
    vote(system, user)    -> str   # one juror: a cheap read-only call (Haiku 4.5)

Both return raw text; callers parse JSON out of it. The mock returns deterministic,
*evidence-grounded* fixtures so the consensus gate genuinely exercises its logic
(red-flag filtering, first-to-ahead-by-K) without a network. The mock's juror verdicts
are derived from the REAL world state passed in the prompt — so the jury really does
catch the injected bug offline, not via a hardcoded answer.

Model IDs (from the claude-api skill):
    brain  = claude-opus-4-8   (the expensive decider)
    juror  = claude-haiku-4-5  (cheap, parallel, read-only)
"""

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Optional, Protocol


BRAIN_MODEL = "claude-opus-4-8"
JUROR_MODEL = "claude-haiku-4-5"


class ClaudeClient(Protocol):
    is_live: bool

    def decide(self, system: str, user: str) -> str: ...
    def vote(self, system: str, user: str) -> str: ...


# --------------------------------------------------------------------------------------
# Live client (gated behind the API key).
# --------------------------------------------------------------------------------------

class LiveClient:
    """Talks to the real Claude API. Used only when ANTHROPIC_API_KEY is present.

    The brain runs at high effort with adaptive thinking (strong reasoning, low volume).
    Jurors run cheap and terse — many parallel calls, read-only judgment only.
    """

    is_live = True

    def __init__(self, api_key: Optional[str] = None):
        import anthropic  # imported lazily so the offline path needs no network stack

        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def decide(self, system: str, user: str) -> str:
        # Streaming + get_final_message keeps us safe under large max_tokens / long turns.
        with self._client.messages.stream(
            model=BRAIN_MODEL,
            max_tokens=8000,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            msg = stream.get_final_message()
        return _text_of(msg)

    def vote(self, system: str, user: str) -> str:
        # Haiku 4.5 does not support the `effort` parameter; keep the call minimal.
        msg = self._client.messages.create(
            model=JUROR_MODEL,
            max_tokens=600,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return _text_of(msg)


def _text_of(msg) -> str:
    return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")


# --------------------------------------------------------------------------------------
# Mock client (offline, deterministic, evidence-grounded).
# --------------------------------------------------------------------------------------

@dataclass
class MockConfig:
    # Fraction of jurors that misfire (noise / correlated-error stress test). The gate's
    # red-flag filter + first-to-ahead-by-K must wash this out.
    juror_noise: float = 0.12
    # Fraction of jurors that emit a malformed / no-evidence vote (caught by red-flag).
    juror_redflag_rate: float = 0.10
    seed: int = 1


class MockClient:
    """Offline stand-in. Brain fixtures follow a scripted migration plan; juror verdicts
    are computed from the REAL state embedded in the prompt, so the jury's correctness
    is genuine, not hardcoded."""

    is_live = False

    def __init__(self, cfg: Optional[MockConfig] = None):
        self.cfg = cfg or MockConfig()
        self._rng = random.Random(self.cfg.seed)
        self._juror_counter = 0

    # ---- brain ------------------------------------------------------------------

    def decide(self, system: str, user: str) -> str:
        """Return the brain's next move as JSON. The orchestrator embeds a directive
        token in the prompt (PLAN_STEP=...) so the mock can follow the scripted plan
        deterministically while still 'reasoning' in prose around it."""
        step = _extract(user, "PLAN_STEP")
        replanning = "REPLAN_CONTEXT" in user
        return _brain_fixture(step, replanning, user)

    # ---- juror ------------------------------------------------------------------

    def vote(self, system: str, user: str) -> str:
        """A single juror's read-only verdict, derived from the real state in the prompt."""
        self._juror_counter += 1
        # A fraction of jurors emit junk that the red-flag filter must drop.
        if self._rng.random() < self.cfg.juror_redflag_rate:
            return self._rng.choice([
                "I think it's probably fine.",  # no structured vote, no evidence
                json.dumps({"vote": "PASS"}),    # missing required evidence field
                "{not even json",                 # malformed
            ])
        truth = _ground_truth_from_prompt(user)
        # A fraction of well-formed jurors misfire (flip the verdict).
        if self._rng.random() < self.cfg.juror_noise:
            truth = not truth
        return _juror_fixture(truth, user)


# --------------------------------------------------------------------------------------
# Fixture helpers
# --------------------------------------------------------------------------------------

def _extract(text: str, key: str) -> str:
    m = re.search(rf"{key}=([A-Za-z0-9_:.-]+)", text)
    return m.group(1) if m else ""


def _brain_fixture(step: str, replanning: bool, user: str) -> str:
    """Map the orchestrator's PLAN_STEP directive to a concrete action + a CHECKABLE CLAIM.

    Forcing the brain to state a falsifiable claim is what makes downstream voting
    possible — it turns "trust me" into "here's an assertion the jury can test."
    """
    plans = {
        "provision": {
            "thought": "Stand up the managed target Postgres and create the orders schema.",
            "action": {"tool": "provision_target", "args": {"total_column_type": "numeric"}},
            "claim": "Target schema exists with `total` as NUMERIC(10,2) so money is preserved to the cent.",
            "irreversible": True,
        },
        "dual_write": {
            "thought": "Turn on dual-write/CDC so writes during migration are captured.",
            "action": {"tool": "enable_dual_write", "args": {}},
            "claim": "Dual-write is active; live writes are buffered for replay to target.",
            "irreversible": True,
        },
        "backfill": {
            "thought": "Backfill the existing source rows into target in chunks.",
            "action": {"tool": "backfill_all", "args": {"chunk_size": 250}},
            "claim": "Every backfilled row is byte-for-byte intact in target — per-row "
                     "checksums match and the money column preserves cents (no NUMERIC->INT truncation).",
            "irreversible": True,
        },
        "replay_cdc": {
            "thought": "Replay the CDC buffer so writes during migration reach target.",
            "action": {"tool": "replay_cdc", "args": {}},
            "claim": "All buffered live writes are applied to target; no write landed only on source.",
            "irreversible": True,
        },
        "cutover": {
            "thought": "Flip live traffic to the target stack atomically.",
            "action": {"tool": "cutover", "args": {}},
            "claim": "Live traffic now points to target and source==target (counts, checksums, revenue).",
            "irreversible": True,
        },
        "done": {
            "thought": "Migration steps complete. Report success.",
            "action": {"tool": "finish", "args": {}},
            "claim": "Migration complete: zero data loss, zero downtime.",
            "irreversible": False,
        },
    }
    # On a re-plan, the brain explains it will re-do the failed step with the jury's
    # evidence in mind. The actual remediation (fixing the schema, re-running CDC) is
    # handled by the recovery loop / re-execution; the brain re-proposes the same step.
    base = plans.get(step, plans["done"])
    out = dict(base)
    if replanning:
        out["thought"] = (
            "The jury overruled my previous commit and localized the root cause. "
            + out["thought"]
            + " I will re-execute this step correctly using the juror evidence."
        )
    return json.dumps(out, indent=2)


def _ground_truth_from_prompt(user: str) -> bool:
    """Compute, from the state JSON embedded in the juror prompt, whether the brain's
    claim actually holds. This is the juror 'running its own query' — offline.

    The orchestrator embeds the post-action world summary as STATE_JSON={...} and the
    claim id as CLAIM_KIND=.... We re-derive PASS/FAIL from observable state only.
    """
    state = _extract_state_json(user)
    kind = _extract(user, "CLAIM_KIND")
    if not state:
        return True  # nothing to refute
    sc = state.get("source_count", 0)
    tc = state.get("target_count", 0)
    src_rev = state.get("source_revenue", "0")
    tgt_rev = state.get("target_revenue", "0")
    col = state.get("target_total_column_type", "NUMERIC(10,2)")

    if kind == "provision":
        # The claim says total is NUMERIC(10,2). FAIL if the column is INTEGER.
        return col.upper().startswith("NUMERIC")
    if kind == "backfill":
        # Claim: backfilled rows are INTACT (cents preserved). The count gap vs source is
        # EXPECTED here (in-flight writes arrive later via CDC), so we check checksum
        # integrity on the copied rows, NOT total count. The truncation bug fails this.
        return bool(state.get("copied_rows_intact", False)) and state.get("copied_rows_mismatched", 0) == 0
    if kind == "replay_cdc":
        # Claim: every in-flight write reached target -> source and target counts agree.
        # The dropped-CDC bug fails this.
        return tc == sc
    if kind == "cutover":
        # Claim: the as-migrated dataset fully reconciles (no missing, no mismatched) and
        # live traffic points to target. Uses the manifest-aware reconciliation.
        return (state.get("as_migrated_missing", 0) == 0
                and state.get("as_migrated_mismatched", 0) == 0
                and state.get("live_points_to_target", False))
    if kind == "dual_write":
        return state.get("dual_write_active", False)
    return True


def _extract_state_json(user: str) -> dict:
    m = re.search(r"STATE_JSON=(\{.*?\})\s*(?:CLAIM_KIND=|$)", user, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}


def _juror_fixture(passed: bool, user: str) -> str:
    """A well-formed juror vote, including the cheap 'proof' it gathered (a query it ran)."""
    state = _extract_state_json(user)
    kind = _extract(user, "CLAIM_KIND")
    sc = state.get("source_count", "?")
    tc = state.get("target_count", "?")
    src_rev = state.get("source_revenue", "?")
    tgt_rev = state.get("target_revenue", "?")
    col = state.get("target_total_column_type", "?")
    mismatched = state.get("copied_rows_mismatched", 0)
    if passed:
        return json.dumps({
            "vote": "PASS",
            "evidence": (
                f"I queried both stores: source_count={sc}, target_count={tc}; "
                f"source_revenue={src_rev}, target_revenue={tgt_rev}; "
                f"target total column={col}; copied rows mismatched={mismatched}. The claim holds."
            ),
            "confidence": 0.95,
        })
    # FAIL votes cite the specific observable mismatch for THIS claim kind —
    # execution-grounded evidence, not a generic count complaint.
    reason = "the post-state contradicts the brain's claim"
    try:
        if kind in ("provision",) and str(col).upper().startswith("INT"):
            reason = ("target `total` column is INTEGER, not NUMERIC — it will silently "
                      "drop the cents from every order written to it")
        elif kind in ("backfill",) and mismatched and mismatched != 0:
            reason = (f"{mismatched} backfilled rows have MISMATCHED checksums: the cents "
                      f"were truncated (target revenue {tgt_rev} vs source {src_rev})")
        elif kind == "cutover" and state.get("as_migrated_missing", 0):
            n = state.get("as_migrated_missing", 0)
            reason = (f"the as-migrated reconciliation shows {n} rows present in the source "
                      "snapshot but MISSING from target — in-flight writes were lost")
        elif kind in ("replay_cdc", "cutover") and tc != sc:
            reason = (f"row counts differ: source={sc} but target={tc} "
                      f"({sc - tc if isinstance(sc, int) and isinstance(tc, int) else '?'} "
                      f"in-flight writes never reached target)")
        elif src_rev != tgt_rev:
            reason = f"revenue differs: source={src_rev} but target={tgt_rev} (money was silently dropped)"
        elif str(col).upper().startswith("INT"):
            reason = "target `total` column is INTEGER, which silently drops cents from every order"
    except Exception:
        pass
    return json.dumps({
        "vote": "FAIL",
        "evidence": f"I ran my own count/checksum/revenue queries: {reason}.",
        "confidence": 0.93,
    })


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------

def make_client(force_mock: bool = False, mock_cfg: Optional[MockConfig] = None) -> ClaudeClient:
    """Return a live client if ANTHROPIC_API_KEY is set (and not forced to mock),
    else the offline mock. The rest of the system is agnostic to which it gets."""
    if not force_mock and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return LiveClient()
        except Exception:
            pass  # fall back to mock if the SDK/key isn't usable
    return MockClient(mock_cfg)
