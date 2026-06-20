"""Tests for the framework adapters and the @gate.protect decorator.

These run without any host-framework dependency installed — adapters are
shape-agnostic by design.
"""

import json

import pytest

from quorum import Gate, GateBlocked
from quorum.adapters.langgraph import gate_tool, make_gate_node
from quorum.adapters.openai_agents import (
    GuardrailOutput,
    gate_function_tool,
    make_output_guardrail,
)


def _ok_vote(verdict: str, ev: str = "I checked the state directly: counts match."):
    return json.dumps({"vote": verdict, "evidence": ev, "confidence": 0.9})


def make_juror(verdict: str):
    class _J:
        def vote(self, _s, _u):
            return _ok_vote(verdict)
    return _J()


# --------------------------------------------------------------------------------------
# @gate.protect decorator
# --------------------------------------------------------------------------------------


def test_protect_returns_value_on_pass():
    gate = Gate(make_juror("PASS"), k=3, max_jurors=6, batch_size=3)

    @gate.protect(claim="greet returns a string", snapshot_state=lambda *a, **kw: {"ok": True})
    def greet(name):
        return f"hi {name}"

    assert greet("world") == "hi world"


def test_protect_raises_and_rolls_back_on_fail():
    gate = Gate(make_juror("FAIL"), k=3, max_jurors=6, batch_size=3)
    rolled_back = []

    @gate.protect(
        claim="must hold",
        snapshot_state=lambda *a, **kw: {"x": 1},
        rollback=lambda *a, **kw: rolled_back.append(True),
    )
    def do_it():
        return "shipped"

    with pytest.raises(GateBlocked) as excinfo:
        do_it()

    assert excinfo.value.result.decision == "FAIL"
    assert rolled_back == [True]


def test_protect_returns_result_when_raise_disabled():
    gate = Gate(make_juror("FAIL"), k=3, max_jurors=6, batch_size=3)

    @gate.protect(
        claim="x", snapshot_state=lambda *a, **kw: {}, raise_on_block=False
    )
    def do_it():
        return "ignored"

    out = do_it()
    assert hasattr(out, "decision") and out.decision == "FAIL"


def test_protect_callable_claim_resolves_per_call():
    gate = Gate(make_juror("PASS"), k=3, max_jurors=6, batch_size=3)
    seen_claims = []

    def claim_fn(name):
        c = f"greeting for {name} returns OK"
        seen_claims.append(c)
        return c

    @gate.protect(claim=claim_fn, snapshot_state=lambda *a, **kw: {})
    def greet(name):
        return f"hi {name}"

    assert greet("alice") == "hi alice"
    assert greet("bob") == "hi bob"
    assert seen_claims == ["greeting for alice returns OK", "greeting for bob returns OK"]


# --------------------------------------------------------------------------------------
# LangGraph adapter
# --------------------------------------------------------------------------------------


def test_gate_tool_wraps_a_plain_function():
    gate = Gate(make_juror("PASS"), k=3, max_jurors=6, batch_size=3)

    def write_file(path, contents):
        return f"wrote {len(contents)} bytes to {path}"

    wrapped = gate_tool(
        write_file,
        gate,
        snapshot_state=lambda *a, **kw: {"target_total_column_type": "NUMERIC(10,2)"},
        claim_for=lambda *a, **kw: "post-state holds",
        claim_kind_for=lambda *a, **kw: "provision",
    )
    assert wrapped("/tmp/x", "hello") == "wrote 5 bytes to /tmp/x"


def test_gate_tool_blocks_and_rolls_back():
    gate = Gate(make_juror("FAIL"), k=3, max_jurors=6, batch_size=3)
    rolled_back = []

    def commit():
        return "ok"

    wrapped = gate_tool(
        commit,
        gate,
        snapshot_state=lambda *a, **kw: {},
        claim_for=lambda *a, **kw: "must hold",
        rollback=lambda *a, **kw: rolled_back.append(True),
    )
    with pytest.raises(GateBlocked):
        wrapped()
    assert rolled_back == [True]


def test_gate_tool_resolves_langchain_style_invoke():
    gate = Gate(make_juror("PASS"), k=3, max_jurors=6, batch_size=3)

    class FakeLCTool:
        name = "fake_tool"

        def invoke(self, x):
            return x * 2

    wrapped = gate_tool(
        FakeLCTool(),
        gate,
        snapshot_state=lambda *a, **kw: {},
        claim_for=lambda *a, **kw: "doubles input",
    )
    assert wrapped(3) == 6


def test_make_gate_node_returns_graph_compatible_state_update():
    gate = Gate(make_juror("PASS"), k=3, max_jurors=6, batch_size=3)
    node = make_gate_node(
        gate,
        snapshot_state=lambda s: s["world"],
        claim_for=lambda s: s["claim"],
        claim_kind_for=lambda s: s["kind"],
    )
    update = node({"world": {"x": 1}, "claim": "x is 1", "kind": "provision"})
    assert "gate" in update
    assert update["gate"]["decision"] == "PASS"
    assert "pass_votes" in update["gate"]


# --------------------------------------------------------------------------------------
# OpenAI Agents SDK adapter
# --------------------------------------------------------------------------------------


def test_gate_function_tool_passes_through_on_pass():
    gate = Gate(make_juror("PASS"), k=3, max_jurors=6, batch_size=3)

    @gate_function_tool(
        gate,
        snapshot_state=lambda *a, **kw: {"target_total_column_type": "NUMERIC(10,2)"},
        claim_for=lambda *a, **kw: "post-state OK",
        claim_kind="provision",
    )
    def write_file(path: str, contents: str) -> str:
        return f"wrote {len(contents)}@{path}"

    assert write_file("/tmp/x", "abcd") == "wrote 4@/tmp/x"


def test_gate_function_tool_blocks_on_fail():
    gate = Gate(make_juror("FAIL"), k=3, max_jurors=6, batch_size=3)

    @gate_function_tool(
        gate,
        snapshot_state=lambda *a, **kw: {},
        claim_for=lambda *a, **kw: "x",
    )
    def commit():
        return "shipped"

    with pytest.raises(GateBlocked) as excinfo:
        commit()
    assert excinfo.value.result.decision == "FAIL"


def test_make_output_guardrail_trips_on_fail():
    gate = Gate(make_juror("FAIL"), k=3, max_jurors=6, batch_size=3)
    guardrail = make_output_guardrail(
        gate,
        snapshot_state=lambda ctx, agent, out: {},
        claim_for=lambda ctx, agent, out: "output reflects truth",
    )
    result = guardrail(context=None, agent=None, output="The answer is 42")
    assert isinstance(result, GuardrailOutput)
    assert result.tripwire_triggered is True
    assert result.output_info["decision"] == "FAIL"


def test_make_output_guardrail_passes_on_pass():
    gate = Gate(make_juror("PASS"), k=3, max_jurors=6, batch_size=3)
    guardrail = make_output_guardrail(
        gate,
        snapshot_state=lambda ctx, agent, out: {"target_total_column_type": "NUMERIC(10,2)"},
        claim_for=lambda ctx, agent, out: "OK",
        claim_kind_for=lambda ctx, agent, out: "provision",
    )
    result = guardrail(None, None, "out")
    assert result.tripwire_triggered is False
    assert result.output_info["decision"] == "PASS"
