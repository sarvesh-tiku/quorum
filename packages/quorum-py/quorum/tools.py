"""The tool surface the brain drives, and which actions are IRREVERSIBLE.

A real implementation would expose these as Claude Agent SDK tools (write/provision/
snapshot for the brain; read-only Postgres queries for jurors via MCP). Here they map
directly onto MigrationWorld operations.

The key design decision (per the doc): irreversible tools are TAGGED. In the SDK this is
a PreToolUse hook that intercepts anything tagged `irreversible` and routes it through the
consensus gate before it's allowed to run. Here the orchestrator consults IRREVERSIBLE.
"""

from typing import Any, Callable

from .world import MigrationWorld


# Tools whose effects are hard/impossible to undo without a snapshot rewind.
# These are the ones a single silent error makes catastrophic — so these are gated.
IRREVERSIBLE = {
    "provision_target",
    "enable_dual_write",
    "backfill_all",
    "backfill_chunk",
    "replay_cdc",
    "cutover",
}

# Terminal / no-op tools.
TERMINAL = {"finish"}


def execute(world: MigrationWorld, tool: str, args: dict[str, Any]) -> str:
    """Run a brain-proposed tool against the world. Returns a short result string."""
    fn: Callable[..., str]
    if tool == "provision_target":
        return world.provision_target(**args)
    if tool == "enable_dual_write":
        return world.enable_dual_write()
    if tool == "backfill_all":
        return world.backfill_all(**args)
    if tool == "backfill_chunk":
        return world.backfill_chunk(**args)
    if tool == "replay_cdc":
        return world.replay_cdc()
    if tool == "cutover":
        return world.cutover()
    if tool == "finish":
        return "migration reported complete"
    raise ValueError(f"unknown tool: {tool}")


# Map a tool to the "claim kind" the jurors verify (keeps the juror prompt focused).
CLAIM_KIND = {
    "provision_target": "provision",
    "enable_dual_write": "dual_write",
    "backfill_all": "backfill",
    "backfill_chunk": "backfill",
    "replay_cdc": "replay_cdc",
    "cutover": "cutover",
    "finish": "done",
}
