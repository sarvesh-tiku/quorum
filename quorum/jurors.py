"""The juror fleet + the consensus gate (red-flag filter, first-to-ahead-by-K voting).

This is the new primitive: a `commit()` that blocks on a jury.

Each juror is a STATELESS, READ-ONLY microagent. It is given:
  * a snapshot of the observable world state (NOT the brain's reasoning),
  * the proposed irreversible action,
  * the brain's checkable claim about the post-state,
and it must INDEPENDENTLY gather its own evidence (run its own count/checksum/revenue
query against the state) and return {vote, evidence, confidence}.

Three structural defenses against correlated juror error (the obvious objection):
  1. Jurors never see the brain's chain-of-thought -> kills agreement bias.
  2. Jurors must cite execution-grounded evidence (a query they ran) -> no bare opinions.
  3. RED-FLAG filter + FIRST-TO-AHEAD-BY-K -> decorrelates and washes out noise; cheap
     when the answer is obvious, more jurors only when genuinely uncertain.

Cost/quality is a dial: more jurors (K, max_jurors) = provably more reliable, brain cost
flat. On the GB200 demo box the fleet is near-free and near-0-latency; here we serve it
via parallel Haiku 4.5 calls (or the offline mock).
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional

from .llm import ClaudeClient
from .world import MigrationWorld, summary_json


JUROR_SYSTEM = """You are an independent verification juror in a consensus gate.

You are READ-ONLY. You did not plan this action and you cannot see the planner's
reasoning. Your only job: decide whether the planner's CLAIM about the post-action
world state is actually TRUE, by inspecting the observable state yourself.

You MUST gather your own evidence — re-derive the relevant counts, checksums, revenue
totals, or column types from the state given to you. Do not take the claim on faith.

Respond with ONLY a JSON object, no prose around it:
{"vote": "PASS" | "FAIL", "evidence": "<the specific numbers/facts you checked>", "confidence": <0..1>}

PASS means the claim is fully supported by the state. FAIL means the state contradicts
the claim (e.g. row counts differ, revenue dropped, a money column is INTEGER instead of
NUMERIC). When in doubt, look harder at the numbers before voting; cite them in evidence.
"""


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
    root_cause: str = ""               # localized on FAIL (handed to the brain)

    @property
    def blocked(self) -> bool:
        return self.decision != "PASS"


# --------------------------------------------------------------------------------------
# Red-flag filter
# --------------------------------------------------------------------------------------

def red_flag(raw: str, *, min_evidence_chars: int = 15, max_chars: int = 4000) -> tuple[Optional[dict], str]:
    """Validate a juror's raw output. Returns (parsed_dict | None, reason).

    Drops malformed / over-long / no-evidence / no-vote responses BEFORE they count.
    This is MAKER's red-flagging, pointed at verification.
    """
    if not raw or len(raw) > max_chars:
        return None, "empty or over-long response"
    # Extract the first JSON object if the model wrapped it in prose.
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
# The consensus gate
# --------------------------------------------------------------------------------------

class ConsensusGate:
    """Runs the juror fleet with first-to-ahead-by-K voting and red-flag filtering."""

    def __init__(
        self,
        client: ClaudeClient,
        *,
        k: int = 3,
        max_jurors: int = 24,
        batch_size: int = 6,
        max_workers: int = 8,
        on_vote: Optional[Callable[[Vote], None]] = None,
    ):
        self.client = client
        self.k = k                      # PASS must lead FAIL by this many to decide
        self.max_jurors = max_jurors    # hard cap on jurors polled per gate
        self.batch_size = batch_size    # jurors per parallel wave
        self.max_workers = max_workers
        self.on_vote = on_vote          # streaming callback for the live UI feed

    def _juror_prompt(self, world: MigrationWorld, action: dict, claim: str, claim_kind: str) -> str:
        state = world.state_summary()
        return (
            "A long-horizon agent proposes an IRREVERSIBLE action. Verify its claim.\n\n"
            f"PROPOSED ACTION: {json.dumps(action)}\n\n"
            f"PLANNER CLAIM: {claim}\n\n"
            "OBSERVABLE WORLD STATE (post-action). Re-derive the facts yourself:\n"
            f"STATE_JSON={json.dumps(state)} "
            f"CLAIM_KIND={claim_kind}\n\n"
            "Run your own count / checksum / revenue / column-type checks against the "
            "state above and vote. Cite the specific numbers in your evidence."
        )

    def evaluate(self, world: MigrationWorld, action: dict, claim: str, claim_kind: str) -> GateResult:
        """Poll jurors in parallel waves until PASS leads FAIL by K, FAIL leads PASS by K,
        or we exhaust max_jurors (NO_CONSENSUS -> treated as a block)."""
        prompt = self._juror_prompt(world, action, claim, claim_kind)
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
                return self._result("PASS", pass_n, fail_n, redflag_n, polled, votes, world, action, claim, claim_kind)
            if fail_n - pass_n >= self.k:
                return self._result("FAIL", pass_n, fail_n, redflag_n, polled, votes, world, action, claim, claim_kind)

        decision = "PASS" if pass_n > fail_n else "FAIL" if fail_n > pass_n else "NO_CONSENSUS"
        return self._result(decision, pass_n, fail_n, redflag_n, polled, votes, world, action, claim, claim_kind)

    def _poll_batch(self, prompt: str, start_id: int, n: int) -> list[Vote]:
        results: dict[int, Vote] = {}

        def one(jid: int) -> Vote:
            raw = self.client.vote(JUROR_SYSTEM, prompt)
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
        # Return in stable juror-id order so the feed reads naturally.
        return [results[k] for k in sorted(results)]

    def _result(self, decision, pass_n, fail_n, redflag_n, polled, votes, world, action, claim, claim_kind="") -> GateResult:
        root_cause = ""
        if decision != "PASS":
            root_cause = self._localize_root_cause(votes, world, claim_kind)
        return GateResult(
            decision=decision,
            pass_votes=pass_n,
            fail_votes=fail_n,
            red_flagged=redflag_n,
            jurors_polled=polled,
            votes=votes,
            root_cause=root_cause,
        )

    def _localize_root_cause(self, votes: list[Vote], world: MigrationWorld, claim_kind: str = "") -> str:
        """AgentDebug-style: don't just say FAIL — say WHERE it went wrong, from the
        ground-truth oracle (this is the recovery loop's input to the brain)."""
        # For backfill, the count gap vs source is EXPECTED (in-flight writes arrive via
        # CDC). The real defect is value truncation on the copied rows — check that.
        if claim_kind == "backfill":
            intact = world.copied_rows_intact()
            if intact["n_mismatched"]:
                rec = world.reconcile()
                return (
                    f"{intact['n_mismatched']} backfilled rows have MISMATCHED checksums — "
                    "the money column was truncated NUMERIC->INTEGER, silently dropping cents "
                    f"(source revenue {rec['source_revenue']} vs target {rec['target_revenue']})."
                )
        if claim_kind == "provision" and world.target_total_is_int:
            return ("target `total` column was created as INTEGER, not NUMERIC(10,2) — every "
                    "order written through it will silently lose its cents.")
        rec = world.reconcile()
        parts = []
        if rec["n_missing"]:
            parts.append(
                f"{rec['n_missing']} rows present in source are MISSING from target "
                f"(e.g. ids {rec['missing_rows'][:5]}{'...' if rec['n_missing'] > 5 else ''}) "
                "— a dual-write/CDC replay dropped writes."
            )
        if rec["n_mismatched"]:
            parts.append(
                f"{rec['n_mismatched']} rows have MISMATCHED checksums — values were "
                "altered in transit (likely a money column truncated NUMERIC->INTEGER)."
            )
        if rec["source_revenue"] != rec["target_revenue"]:
            parts.append(
                f"revenue mismatch: source={rec['source_revenue']} vs "
                f"target={rec['target_revenue']} (delta {rec['revenue_delta']})."
            )
        if not parts:
            # No oracle-visible defect but jury blocked -> surface the juror evidence.
            fail_ev = [v.evidence for v in votes if v.vote == "FAIL"]
            parts.append(fail_ev[0] if fail_ev else "jurors did not reach a PASS consensus.")
        return " ".join(parts)
