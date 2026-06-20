"""The single-threaded decider (brain) + the orchestrator that wires in the gate.

Architecture (from the design doc):

    GOAL -> DECIDER (single-threaded, owns ALL decisions, Claude Opus 4.8)
              | proposes ONE irreversible action + a checkable claim
              v
            CONSENSUS GATE (jurors vote; red-flag + first-to-ahead-by-K)
              |--PASS--> commit; snapshot; append to decision journal
              |--FAIL--> RECOVERY LOOP: localize root cause -> rollback to last
                         good snapshot -> brain re-plans WITH the juror evidence

The brain stays single-threaded (Cognition's "don't build multi-agents": no conflicting
decisions, full context). Only VERIFICATION is fanned out — read-only, the one
parallelism Cognition endorses.

Two run modes share this orchestrator:
  * baseline (gate OFF): the brain acts and reports "done"; silent bugs ship.
  * quorum   (gate ON):  every irreversible commit blocks on the jury; bugs are caught
                          and recovered before they compound.
"""

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from .jurors import ConsensusGate, GateResult, Vote
from .llm import ClaudeClient, BRAIN_MODEL
from .tools import CLAIM_KIND, IRREVERSIBLE, TERMINAL, execute
from .world import MigrationWorld


BRAIN_SYSTEM = """You are the single-threaded decider for a long-horizon migration agent.

You own every decision. For each step you emit (a) ONE action and (b) a machine-checkable
CLAIM about the resulting world state — a falsifiable assertion an independent verifier
can test (e.g. "target row count == source count; per-row checksums match"). Stating a
checkable claim is mandatory: it turns 'trust me' into a testable assertion.

Respond with ONLY a JSON object:
{"thought": "...", "action": {"tool": "...", "args": {...}}, "claim": "...", "irreversible": true|false}
"""

# The scripted plan the brain follows (the mock brain reads PLAN_STEP; a live brain is
# guided by the same ordering). This is the long-horizon, dependent-step task.
PLAN = ["provision", "dual_write", "backfill", "replay_cdc", "cutover", "done"]
STEP_TO_TOOL = {
    "provision": "provision_target",
    "dual_write": "enable_dual_write",
    "backfill": "backfill_all",
    "replay_cdc": "replay_cdc",
    "cutover": "cutover",
    "done": "finish",
}


@dataclass
class JournalEntry:
    """Append-only decision journal — NOT lossy context compaction. The brain re-reads
    the head; recovery rewinds it. Answers 'agents lose track of what's been done.'"""
    step: str
    thought: str
    action: dict
    claim: str
    irreversible: bool
    gate: Optional[GateResult] = None
    committed: bool = False
    result: str = ""
    attempt: int = 1


# Event types emitted to the live verification feed.
@dataclass
class Event:
    kind: str            # step_start | claim | vote | gate_result | commit | rollback |
                         # replan | finish | baseline_report
    payload: dict = field(default_factory=dict)


EventSink = Callable[[Event], None]


class Orchestrator:
    def __init__(
        self,
        client: ClaudeClient,
        world: MigrationWorld,
        *,
        gate_enabled: bool = True,
        gate: Optional[ConsensusGate] = None,
        max_attempts_per_step: int = 3,
        sink: Optional[EventSink] = None,
    ):
        self.client = client
        self.world = world
        self.gate_enabled = gate_enabled
        self.sink = sink or (lambda e: None)
        self.max_attempts = max_attempts_per_step
        self.journal: list[JournalEntry] = []
        self.snapshots: list[dict] = []  # last-good snapshots (restore points)
        # The gate streams individual votes to the same feed.
        self.gate = gate or ConsensusGate(
            client,
            on_vote=lambda v: self.sink(Event("vote", _vote_payload(v))),
        )

    # ---- brain call -------------------------------------------------------------

    def _ask_brain(self, step: str, replan_context: str = "") -> dict:
        journal_head = self._journal_head_text()
        user = (
            "GOAL: Migrate the `orders` service + Postgres from legacy to target stack "
            "with zero data loss and zero downtime, and prove it.\n\n"
            f"DECISION JOURNAL (what is already done):\n{journal_head}\n\n"
            f"PLAN_STEP={step}\n"
        )
        if replan_context:
            user += f"\nREPLAN_CONTEXT: {replan_context}\n"
        raw = self.client.decide(BRAIN_SYSTEM, user)
        return _parse_brain(raw, step)

    def _journal_head_text(self) -> str:
        if not self.journal:
            return "(nothing done yet)"
        lines = []
        for e in self.journal:
            status = "committed" if e.committed else "BLOCKED/rolled-back"
            lines.append(f"- {e.action.get('tool')}: {status} (claim: {e.claim})")
        return "\n".join(lines)

    # ---- main loop --------------------------------------------------------------

    def run(self) -> dict:
        """Execute the migration end-to-end. Returns the final reconciliation + stats."""
        # Inject some live write traffic up front (the 'zero downtime' pressure).
        for _ in range(30):
            self.world.emit_live_write()

        for step in PLAN:
            if step == "done":
                break
            self._run_step(step)
            # Live writes keep arriving between steps (pressure on CDC/dual-write).
            for _ in range(8):
                self.world.emit_live_write()

        # Final replay of any trailing CDC the loop captured, then settle.
        # (The brain already proposed replay_cdc as a gated step; we don't silently fix.)

        result = self._finish()
        return result

    def _run_step(self, step: str) -> None:
        replan_context = ""
        for attempt in range(1, self.max_attempts + 1):
            decision = self._ask_brain(step, replan_context)
            tool = decision["action"]["tool"]
            entry = JournalEntry(
                step=step,
                thought=decision.get("thought", ""),
                action=decision["action"],
                claim=decision.get("claim", ""),
                irreversible=tool in IRREVERSIBLE,
                attempt=attempt,
            )
            self.sink(Event("step_start", {
                "step": step, "tool": tool, "thought": entry.thought, "attempt": attempt,
            }))
            self.sink(Event("claim", {"tool": tool, "claim": entry.claim}))

            # Take a candidate snapshot BEFORE executing the irreversible action so we
            # can roll back if the jury blocks the commit.
            pre_snapshot = self.world.snapshot()

            # Execute the action (it mutates the world; in a real system the PreToolUse
            # hook would block first, but to *verify post-state* the jurors need the
            # effect applied to a sandboxed copy — here we apply, then gate, then
            # rollback on FAIL. The world is sandboxed, so this is safe.)
            entry.result = execute(self.world, tool, decision["action"].get("args", {}))

            if not entry.irreversible or not self.gate_enabled:
                # Baseline mode (gate off) OR reversible action: commit unconditionally.
                entry.committed = True
                self.journal.append(entry)
                self.snapshots.append(self.world.snapshot())
                self.sink(Event("commit", {
                    "tool": tool, "gated": False,
                    "reason": "gate disabled" if self.gate_enabled is False else "reversible",
                }))
                return

            # GATE: jurors verify the post-state against the claim.
            gate_res = self.gate.evaluate(
                self.world, decision["action"], entry.claim, CLAIM_KIND.get(tool, "done")
            )
            entry.gate = gate_res
            self.sink(Event("gate_result", _gate_payload(gate_res, tool)))

            if gate_res.decision == "PASS":
                entry.committed = True
                self.journal.append(entry)
                self.snapshots.append(self.world.snapshot())
                self.sink(Event("commit", {"tool": tool, "gated": True}))
                return

            # FAIL / NO_CONSENSUS -> recovery loop.
            entry.committed = False
            self.journal.append(entry)
            self.sink(Event("rollback", {
                "tool": tool,
                "root_cause": gate_res.root_cause,
                "to_snapshot": len(self.snapshots) - 1,
            }))
            # Roll back to the last GOOD snapshot (state before this failed action).
            self.world.restore(pre_snapshot)
            replan_context = (
                f"Your commit of `{tool}` was OVERRULED by the jury "
                f"({gate_res.fail_votes} FAIL vs {gate_res.pass_votes} PASS). "
                f"Root cause: {gate_res.root_cause} "
                f"Re-do this step correctly."
            )
            self.sink(Event("replan", {"tool": tool, "context": replan_context}))
            # On the retry we REMEDIATE the underlying fault for this step, then re-run.
            self._remediate(step)

        # Exhausted attempts on a gated step -> escalate (still better than shipping a bug).
        self.sink(Event("step_start", {"step": step, "tool": "ESCALATE", "thought":
                  "Max attempts reached; halting before committing a bad irreversible step.",
                  "attempt": self.max_attempts}))

    def _remediate(self, step: str) -> None:
        """The brain, now armed with the jury's root-cause evidence, fixes the underlying
        defect before re-executing the step. In a live system the brain would issue the
        corrective tool calls; here we apply the corresponding fix to the sandbox so the
        re-execution succeeds. (The fault injector is disabled for the remediated path —
        modeling 'the agent learned what was wrong and did it right the second time'.)"""
        f = self.world.faults
        if step == "provision":
            f.truncate_total_to_int = False           # re-provision as NUMERIC
            self.world.provision_target(total_column_type="numeric")
        elif step == "backfill":
            f.truncate_total_to_int = False
            self.world.target_total_is_int = False
            self.world.provision_target(total_column_type="numeric")
        elif step == "replay_cdc":
            f.drop_rows_during_cdc = 0                 # stop dropping CDC writes
            f.skip_dual_write = False
        elif step == "cutover":
            # Cutover failing means an upstream defect leaked through. The brain, armed
            # with the jury's root-cause evidence, clears all faults and rebuilds target
            # cleanly from source so the final reconciliation can pass.
            f.truncate_total_to_int = False
            f.drop_rows_during_cdc = 0
            f.skip_dual_write = False
            self.world.target_total_is_int = False
            self.world.target = {}
            self.world.cdc_buffer = []
            for oid in sorted(self.world.source.keys()):
                self.world._apply_write_to_target(self.world.source[oid])

    def _finish(self) -> dict:
        decision = self._ask_brain("done")
        self.sink(Event("step_start", {"step": "done", "tool": "finish",
                  "thought": decision.get("thought", ""), "attempt": 1}))
        # The brain reports completion. Now the INDEPENDENT oracle has the final say.
        recon = self.world.reconcile()
        self.sink(Event("finish", {
            "brain_claim": decision.get("claim", "Migration complete."),
            "reconciliation": recon,
        }))
        stats = self.stats()
        return {"reconciliation": recon, "stats": stats}

    # ---- stats ------------------------------------------------------------------

    def stats(self) -> dict:
        total_jurors = sum(e.gate.jurors_polled for e in self.journal if e.gate)
        total_redflag = sum(e.gate.red_flagged for e in self.journal if e.gate)
        blocks = sum(1 for e in self.journal if e.gate and e.gate.blocked)
        rollbacks = sum(1 for e in self.journal if e.gate and e.gate.blocked and not e.committed)
        return {
            "gate_enabled": self.gate_enabled,
            "brain_model": BRAIN_MODEL,
            "steps_attempted": len(self.journal),
            "jurors_polled_total": total_jurors,
            "votes_red_flagged": total_redflag,
            "gate_blocks": blocks,
            "rollbacks": rollbacks,
            "live": self.client.is_live,
        }


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _parse_brain(raw: str, step: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(raw[start:end + 1])
            if isinstance(obj, dict) and "action" in obj:
                return obj
        except json.JSONDecodeError:
            pass
    # Fallback: synthesize the planned action so a flaky live response can't stall the loop.
    tool = STEP_TO_TOOL.get(step, "finish")
    return {
        "thought": f"(fallback) executing planned step {step}",
        "action": {"tool": tool, "args": {} if tool != "backfill_all" else {"chunk_size": 250}},
        "claim": f"Step {step} produces a consistent source==target post-state.",
        "irreversible": tool in IRREVERSIBLE,
    }


def _vote_payload(v: Vote) -> dict:
    return {
        "juror_id": v.juror_id,
        "vote": v.vote,
        "evidence": v.evidence,
        "confidence": v.confidence,
        "red_flagged": v.red_flagged,
        "red_flag_reason": v.red_flag_reason,
    }


def _gate_payload(g: GateResult, tool: str) -> dict:
    return {
        "tool": tool,
        "decision": g.decision,
        "pass_votes": g.pass_votes,
        "fail_votes": g.fail_votes,
        "red_flagged": g.red_flagged,
        "jurors_polled": g.jurors_polled,
        "root_cause": g.root_cause,
    }
