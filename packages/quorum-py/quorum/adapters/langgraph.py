"""LangGraph adapter for `quorum.Gate`.

LangGraph models agents as graphs of nodes. Tool calls are nodes; state lives
in a typed dict that flows through the graph. This adapter gives you two ways
to interpose the gate:

1. `gate_tool(tool, ...)` — wrap a single LangChain `Tool` (or any callable)
   so each invocation runs the tool, evaluates the post-state, and either
   returns the tool's result (PASS) or raises GateBlocked (FAIL).

2. `make_gate_node(gate, ...)` — produce a LangGraph node fn that you can wire
   directly into `StateGraph.add_node`. Useful when you want the gate to be a
   distinct graph step that decides whether to route to a `commit` node or a
   `rollback` node.

Both are lazy: this module imports nothing from langgraph at import time, so
the package's core install never pulls langgraph in.

Usage with `add_conditional_edges` (sketch):

    from quorum import Gate
    from quorum.adapters.langgraph import make_gate_node

    gate = Gate(juror_client, k=3)
    graph.add_node("propose",  propose_node)
    graph.add_node("gate",     make_gate_node(gate,
        snapshot_state=lambda s: s["world"],
        claim_for=lambda s: s["claim"],
        claim_kind_for=lambda s: s["claim_kind"],
    ))
    graph.add_node("commit",   commit_node)
    graph.add_node("rollback", rollback_node)
    graph.add_edge("propose", "gate")
    graph.add_conditional_edges("gate",
        lambda s: "commit" if s["gate"]["decision"] == "PASS" else "rollback")
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from ..gate import Gate, GateBlocked, GateResult


# Type aliases — kept liberal so we don't pin to a langgraph version.
StateLike = Any                                          # whatever your graph's state type is
SnapshotFromState = Callable[[StateLike], dict]
ClaimFromState = Callable[[StateLike], str]


def make_gate_node(
    gate: Gate,
    *,
    snapshot_state: SnapshotFromState,
    claim_for: ClaimFromState,
    claim_kind_for: Optional[ClaimFromState] = None,
    action_from: Optional[Callable[[StateLike], dict]] = None,
    output_key: str = "gate",
) -> Callable[[StateLike], dict]:
    """Return a LangGraph-compatible node function.

    The node reads the world from the graph state via `snapshot_state(state)`,
    asks `claim_for(state)` for the brain's checkable claim, runs the gate, and
    returns a state-update dict containing the GateResult under `output_key`.

    Conditional edges can then route off `state[output_key]["decision"]`:

        graph.add_conditional_edges(
            "gate",
            lambda s: "commit" if s["gate"]["decision"] == "PASS" else "rollback",
        )
    """

    def node(state: StateLike) -> dict:
        world = snapshot_state(state)
        claim = claim_for(state)
        kind = claim_kind_for(state) if claim_kind_for else ""
        action = action_from(state) if action_from else {"tool": "<gate>"}
        result = gate.evaluate(world, action, claim, kind)
        return {output_key: _result_payload(result)}

    return node


def gate_tool(
    tool: Any,
    gate: Gate,
    *,
    snapshot_state: Callable[..., dict],
    claim_for: Callable[..., str],
    claim_kind_for: Optional[Callable[..., str]] = None,
    rollback: Optional[Callable[..., None]] = None,
):
    """Wrap a LangChain-style tool (or any callable) in the consensus gate.

    Returns a new callable with the same signature. On PASS, returns the tool's
    result; on FAIL, optionally calls `rollback(*args, **kwargs)` and raises
    `GateBlocked`.

    Works on:
      * a plain function
      * a LangChain `BaseTool` (wraps its `.invoke` / `.run` method)
      * anything with a `.func` or `.invoke` attribute the SDK uses
    """
    func = _resolve_tool_callable(tool)

    def wrapped(*args, **kwargs):
        result_value = func(*args, **kwargs)
        state = snapshot_state(*args, **kwargs)
        claim = claim_for(*args, **kwargs)
        kind = claim_kind_for(*args, **kwargs) if claim_kind_for else ""
        action = {
            "tool": getattr(tool, "name", None) or func.__name__,
            "args": _safe_args_dict(args, kwargs),
        }
        gate_result = gate.evaluate(state, action, claim, kind)
        if gate_result.decision == "PASS":
            return result_value
        if rollback is not None:
            rollback(*args, **kwargs)
        raise GateBlocked(action, claim, gate_result)

    wrapped.__name__ = getattr(tool, "name", None) or func.__name__
    wrapped.__quorum_gate__ = gate          # type: ignore[attr-defined]
    return wrapped


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _resolve_tool_callable(tool: Any) -> Callable[..., Any]:
    """Best-effort extraction of the underlying callable from a LangChain tool."""
    if callable(tool) and not hasattr(tool, "invoke") and not hasattr(tool, "run"):
        return tool
    for attr in ("func", "_run", "invoke", "run"):
        candidate = getattr(tool, attr, None)
        if callable(candidate):
            return candidate
    if callable(tool):
        return tool
    raise TypeError(f"can't find a callable on tool: {tool!r}")


def _safe_args_dict(args: tuple, kwargs: dict) -> dict:
    import json as _json

    try:
        _json.dumps({"a": args, "kw": kwargs})
        return {"args": list(args), "kwargs": dict(kwargs)}
    except (TypeError, ValueError):
        return {
            "args": [repr(a) for a in args],
            "kwargs": {k: repr(v) for k, v in kwargs.items()},
        }


def _result_payload(result: GateResult) -> dict:
    return {
        "decision": result.decision,
        "pass_votes": result.pass_votes,
        "fail_votes": result.fail_votes,
        "red_flagged": result.red_flagged,
        "jurors_polled": result.jurors_polled,
        "root_cause": result.root_cause,
    }


__all__ = ["gate_tool", "make_gate_node"]
