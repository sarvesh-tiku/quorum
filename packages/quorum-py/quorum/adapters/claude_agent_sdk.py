"""Claude Agent SDK adapter for `quorum.Gate`.

The Claude Agent SDK exposes a `PreToolUse` hook that runs before every tool
call. This adapter turns that hook into a consensus gate: irreversible tools
are intercepted, a juror fleet votes on the brain's checkable claim, and the
hook either lets the tool through (PASS) or blocks it (FAIL / NO_CONSENSUS).

This module is OPTIONAL. The core package has zero dependencies; importing
this module lazily imports `claude_agent_sdk`, which is only needed if you're
wiring the gate into a real SDK agent.

User-facing API:

    from quorum import Gate
    from quorum.adapters.claude_agent_sdk import gate_irreversible_tools

    gate = Gate(juror_client, k=3)
    hooks = gate_irreversible_tools(
        gate,
        irreversible={"write_file", "shell.exec", "git.commit"},
        snapshot_state=lambda: world.snapshot(),
        claim_for=lambda tool, args: f"after {tool}(...) the working tree compiles + tests pass",
    )
    # then: agent = ClaudeAgent(..., hooks=hooks)

The adapter is intentionally small and synchronous. Async + streaming votes
are a v0.2 concern.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

from ..gate import Gate, GateResult


# Type aliases — kept liberal so we don't pin to an SDK version.
SnapshotFn = Callable[[], dict[str, Any]]
ClaimForFn = Callable[[str, dict[str, Any]], str]
ClaimKindForFn = Callable[[str, dict[str, Any]], str]
OnBlockFn = Callable[[str, dict[str, Any], GateResult], None]


def gate_irreversible_tools(
    gate: Gate,
    *,
    irreversible: Iterable[str],
    snapshot_state: SnapshotFn,
    claim_for: ClaimForFn,
    claim_kind_for: Optional[ClaimKindForFn] = None,
    on_block: Optional[OnBlockFn] = None,
) -> dict:
    """Build a hooks dict suitable for the Claude Agent SDK's `hooks=` argument.

    Returns a dict in the SDK's expected shape:

        {"PreToolUse": [callable]}

    The callable, when invoked by the SDK before a tool runs, decides whether
    to allow the tool. If the tool name is in `irreversible`, the callable:
      1. snapshots state via `snapshot_state()`
      2. asks the planner-supplied `claim_for(tool, args)` for the claim
      3. runs `gate.evaluate(state, action, claim, claim_kind)`
      4. returns an SDK-shaped continue/halt response

    The SDK's exact hook signature has churned across versions; we keep this
    adapter framework-shape-agnostic so it lights up under any of them. The
    contract: a PreToolUse hook is a callable that receives (tool_name, args)
    or (event_payload) and returns either:
      * None / True / {"continue": True}                 → allow
      * False / {"continue": False, "stop_reason": "…"}  → block

    We support all three response styles by returning a dict; the SDK's hook
    runtime is permissive about what it accepts as a stop signal.
    """
    irreversible_set = set(irreversible)

    def _pretooluse(*args, **kwargs):
        tool_name, tool_args = _normalize_hook_args(args, kwargs)
        if tool_name not in irreversible_set:
            return {"continue": True}

        state = snapshot_state()
        action = {"tool": tool_name, "args": tool_args}
        claim = claim_for(tool_name, tool_args)
        kind = claim_kind_for(tool_name, tool_args) if claim_kind_for else ""

        result = gate.evaluate(state, action, claim, kind)
        if result.decision == "PASS":
            return {"continue": True, "gate": _result_payload(result)}

        if on_block is not None:
            on_block(tool_name, tool_args, result)
        return {
            "continue": False,
            "stop_reason": (
                f"quorum-py blocked `{tool_name}`: {result.decision} "
                f"(PASS={result.pass_votes} FAIL={result.fail_votes} "
                f"polled={result.jurors_polled})"
            ),
            "gate": _result_payload(result),
        }

    return {"PreToolUse": [_pretooluse]}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _normalize_hook_args(args, kwargs) -> tuple[str, dict[str, Any]]:
    """Best-effort extraction of (tool_name, tool_args) from whatever shape the
    SDK passes us. We accept three common shapes:

      pretooluse(tool_name: str, args: dict)
      pretooluse(event: dict)                   # event = {"name": ..., "input": ...}
      pretooluse(name=..., input=...)           # kwargs
    """
    if "tool_name" in kwargs or "name" in kwargs:
        name = kwargs.get("tool_name") or kwargs.get("name") or ""
        tool_args = kwargs.get("input") or kwargs.get("args") or {}
        return name, tool_args
    if args:
        first = args[0]
        if isinstance(first, dict):
            return first.get("name") or first.get("tool_name") or "", (
                first.get("input") or first.get("args") or {}
            )
        if len(args) >= 2:
            return str(first), args[1] if isinstance(args[1], dict) else {}
        return str(first), {}
    return "", {}


def _result_payload(result: GateResult) -> dict:
    return {
        "decision": result.decision,
        "pass_votes": result.pass_votes,
        "fail_votes": result.fail_votes,
        "red_flagged": result.red_flagged,
        "jurors_polled": result.jurors_polled,
        "root_cause": result.root_cause,
    }


__all__ = ["gate_irreversible_tools"]
