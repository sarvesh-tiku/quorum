"""Migration-aware juror layer that wraps the framework-agnostic `Gate`.

This module is what the QUORUM migration demo uses. The juror prompt is built
from a `MigrationWorld.state_summary()`, and on FAIL the gate result's
root_cause is populated via the world's reconciliation oracle.

Anything generic (red-flag filter, first-to-ahead-by-K, parallel polling, the
Vote / GateResult dataclasses) lives in `gate.py`. This module just adds
domain knowledge: how to summarize the world for the prompt and how to
localize the root cause from the world's truth.
"""

import json
from typing import Callable, Optional

from .gate import (
    DEFAULT_JUROR_SYSTEM,
    Gate,
    GateResult,
    JurorClient,
    Vote,
    red_flag,
)
from .llm import ClaudeClient
from .world import MigrationWorld, summary_json


# Re-export the migration-flavored juror system prompt for clarity.
JUROR_SYSTEM = DEFAULT_JUROR_SYSTEM


__all__ = [
    "JUROR_SYSTEM",
    "Vote",
    "GateResult",
    "ConsensusGate",
    "red_flag",
]


class ConsensusGate:
    """Migration-demo gate. Thin wrapper over `Gate` that knows about MigrationWorld."""

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
        self._client: JurorClient = client          # ClaudeClient already exposes .vote()
        self._gate = Gate(
            self._client,
            k=k,
            max_jurors=max_jurors,
            batch_size=batch_size,
            max_workers=max_workers,
            juror_system=JUROR_SYSTEM,
            on_vote=on_vote,
            prompt_builder=self._build_prompt,
        )
        # Surface knobs callers used to set on us directly.
        self.client = client
        self.k = k
        self.max_jurors = max_jurors
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.on_vote = on_vote

    def _build_prompt(self, state, action, claim, claim_kind) -> str:
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

    def evaluate(
        self,
        world: MigrationWorld,
        action: dict,
        claim: str,
        claim_kind: str,
    ) -> GateResult:
        state = world.state_summary()
        result = self._gate.evaluate(state, action, claim, claim_kind)
        if result.decision != "PASS":
            result.root_cause = self._localize_root_cause(result.votes, world, claim_kind)
        return result

    def _localize_root_cause(
        self, votes: list[Vote], world: MigrationWorld, claim_kind: str = ""
    ) -> str:
        """AgentDebug-style: don't just say FAIL — say WHERE it went wrong, from the
        ground-truth oracle (this is the recovery loop's input to the brain)."""
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
            return (
                "target `total` column was created as INTEGER, not NUMERIC(10,2) — every "
                "order written through it will silently lose its cents."
            )
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
            fail_ev = [v.evidence for v in votes if v.vote == "FAIL"]
            parts.append(fail_ev[0] if fail_ev else "jurors did not reach a PASS consensus.")
        return " ".join(parts)
