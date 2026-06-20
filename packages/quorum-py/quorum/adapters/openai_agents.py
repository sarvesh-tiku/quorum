"""OpenAI Agents SDK adapter for `quorum.Gate`.

The OpenAI Agents SDK exposes `output_guardrails` and `input_guardrails` on
agents, plus `function_tool`-decorated callables that drive side effects.
This adapter gates a function tool the same way the Claude Agent SDK adapter
does: wrap the tool callable so each invocation runs through the gate.

The module is OPTIONAL — it does not import `openai-agents` at module load.
Pass any tool object that has a callable surface (or a plain function) and we
discover the underlying callable.

Usage:

    from quorum import Gate
    from quorum.adapters.openai_agents import gate_function_tool

    @gate_function_tool(
        gate,
        snapshot_state=lambda *a, **kw: read_world(),
        claim_for=lambda *a, **kw: "post-state holds",
        claim_kind="diff_semantics",
    )
    @function_tool                           # OpenAI's decorator
    def write_file(path: str, contents: str) -> str: ...

You can also build an output_guardrail-shaped check that runs after a tool
call and returns a tripwire if the gate blocks. See `make_output_guardrail`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..gate import Gate, GateBlocked, GateResult


# Type aliases — kept liberal so we don't pin to an openai-agents version.
SnapshotFn = Callable[..., dict]
ClaimFn = Callable[..., str]


def gate_function_tool(
    gate: Gate,
    *,
    snapshot_state: SnapshotFn,
    claim_for: ClaimFn,
    claim_kind: str | ClaimFn = "",
    rollback: Optional[Callable[..., None]] = None,
):
    """Decorator that wraps an OpenAI Agents `function_tool`-style callable in
    the consensus gate. Use it OUTSIDE the SDK's `@function_tool` so the SDK
    sees a still-typed signature; or apply it to a plain function.

    The wrapped function runs first; the gate evaluates the post-state; on
    FAIL we call `rollback` (if given) and raise `GateBlocked`. The exception
    surfaces through the SDK's tool-call machinery as a tool error, which the
    agent can then re-plan against.
    """
    def _resolve(value, args, kwargs) -> str:
        return value(*args, **kwargs) if callable(value) else value

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        underlying = _resolve_callable(func)

        def wrapper(*args, **kwargs):
            value = underlying(*args, **kwargs)
            state = snapshot_state(*args, **kwargs)
            claim = claim_for(*args, **kwargs)
            kind = _resolve(claim_kind, args, kwargs)
            action = {
                "tool": getattr(func, "name", None) or underlying.__name__,
                "args": _safe_args_dict(args, kwargs),
            }
            result = gate.evaluate(state, action, claim, kind)
            if result.decision == "PASS":
                return value
            if rollback is not None:
                rollback(*args, **kwargs)
            raise GateBlocked(action, claim, result)

        wrapper.__name__ = getattr(func, "name", None) or underlying.__name__
        wrapper.__doc__ = underlying.__doc__
        wrapper.__quorum_gate__ = gate    # type: ignore[attr-defined]
        return wrapper

    return decorator


# --------------------------------------------------------------------------------------
# Output guardrail
# --------------------------------------------------------------------------------------


@dataclass
class GuardrailOutput:
    """Shape compatible with OpenAI Agents output guardrails.

    The SDK expects guardrails to return an object with `.tripwire_triggered`
    (bool) and arbitrary `.output_info`. We provide that shape here without
    importing the SDK.
    """

    tripwire_triggered: bool
    output_info: dict


def make_output_guardrail(
    gate: Gate,
    *,
    snapshot_state: Callable[[Any, Any, Any], dict],
    claim_for: Callable[[Any, Any, Any], str],
    claim_kind_for: Optional[Callable[[Any, Any, Any], str]] = None,
):
    """Return a guardrail callable shaped like the OpenAI Agents output
    guardrail signature: `(context, agent, output) -> GuardrailOutput`.

    The guardrail snapshots the world after the agent's output, runs the gate
    on the brain's claim, and trips on FAIL. The SDK then halts the run and
    surfaces the gate's evidence to the orchestrator.
    """
    def guardrail(context, agent, output) -> GuardrailOutput:
        state = snapshot_state(context, agent, output)
        claim = claim_for(context, agent, output)
        kind = claim_kind_for(context, agent, output) if claim_kind_for else ""
        action = {"tool": "<agent_output>", "args": {}}
        result = gate.evaluate(state, action, claim, kind)
        return GuardrailOutput(
            tripwire_triggered=result.decision != "PASS",
            output_info=_result_payload(result),
        )

    return guardrail


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _resolve_callable(tool: Any) -> Callable[..., Any]:
    """Find the underlying callable on whatever the SDK handed us."""
    if callable(tool):
        return tool
    for attr in ("func", "_func", "invoke", "run"):
        candidate = getattr(tool, attr, None)
        if callable(candidate):
            return candidate
    raise TypeError(f"can't resolve callable from {tool!r}")


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


__all__ = ["gate_function_tool", "make_output_guardrail", "GuardrailOutput"]
