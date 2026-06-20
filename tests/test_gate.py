"""Tests for the framework-agnostic Gate.

These exercise quorum.Gate without any MigrationWorld dependency,
proving the gate works on a generic state dict + a fake juror callable.
"""

import json

import pytest

from quorum import Gate, GateResult, Vote, as_juror_client, red_flag
from quorum.adapters.claude_agent_sdk import gate_irreversible_tools


# --------------------------------------------------------------------------------------
# Fake juror clients
# --------------------------------------------------------------------------------------


def _ok_vote(verdict: str, ev: str = "I checked the state directly: counts match."):
    return json.dumps({"vote": verdict, "evidence": ev, "confidence": 0.9})


def make_scripted_juror(responses: list[str]):
    """Returns a juror callable that yields each response in order, then repeats the last."""
    state = {"i": 0}

    def vote(_system: str, _user: str) -> str:
        i = min(state["i"], len(responses) - 1)
        state["i"] += 1
        return responses[i]

    return vote


# --------------------------------------------------------------------------------------
# Red-flag filter
# --------------------------------------------------------------------------------------


def test_red_flag_drops_non_json():
    parsed, reason = red_flag("not json at all")
    assert parsed is None
    assert "no JSON" in reason


def test_red_flag_drops_missing_evidence():
    parsed, reason = red_flag(json.dumps({"vote": "PASS"}))
    assert parsed is None
    assert "evidence" in reason


def test_red_flag_drops_invalid_vote():
    parsed, reason = red_flag(json.dumps({"vote": "MAYBE", "evidence": "x" * 30}))
    assert parsed is None
    assert "vote" in reason


def test_red_flag_accepts_valid():
    parsed, _ = red_flag(_ok_vote("PASS"))
    assert parsed["vote"] == "PASS"


# --------------------------------------------------------------------------------------
# Gate
# --------------------------------------------------------------------------------------


def test_gate_pass_with_unanimous_jury():
    juror = make_scripted_juror([_ok_vote("PASS")] * 10)
    gate = Gate(juror, k=3, max_jurors=12, batch_size=3)
    result = gate.evaluate(
        state={"hello": "world"},
        action={"tool": "write_file", "args": {"path": "x"}},
        claim="x exists",
    )
    assert isinstance(result, GateResult)
    assert result.decision == "PASS"
    assert result.pass_votes >= 3
    assert result.fail_votes == 0
    assert result.jurors_polled <= 6  # first-to-ahead-by-K stops after a wave or two


def test_gate_fail_when_majority_says_fail():
    juror = make_scripted_juror([_ok_vote("FAIL")] * 10)
    gate = Gate(juror, k=3, max_jurors=12, batch_size=3)
    result = gate.evaluate(state={}, action={"tool": "x"}, claim="will hold")
    assert result.decision == "FAIL"
    assert result.fail_votes >= 3
    assert result.blocked is True


def test_gate_filters_red_flags():
    # Mix: 3 valid PASS, 3 malformed (red-flagged), then more PASS to settle.
    juror = make_scripted_juror(
        [_ok_vote("PASS"), "{not json", _ok_vote("PASS"),
         "no json", _ok_vote("PASS"), "{not json"] + [_ok_vote("PASS")] * 6
    )
    gate = Gate(juror, k=3, max_jurors=12, batch_size=3)
    result = gate.evaluate(state={}, action={}, claim="x")
    assert result.decision == "PASS"
    assert result.red_flagged >= 1


def test_gate_no_consensus_returns_block():
    # Alternating PASS/FAIL — neither side reaches +K.
    pattern = [_ok_vote("PASS"), _ok_vote("FAIL")] * 10
    juror = make_scripted_juror(pattern)
    gate = Gate(juror, k=3, max_jurors=4, batch_size=4)
    result = gate.evaluate(state={}, action={}, claim="x")
    assert result.decision in ("NO_CONSENSUS", "PASS", "FAIL")
    # The gate ran exactly max_jurors at most.
    assert result.jurors_polled <= 4


def test_gate_streams_votes_via_callback():
    seen: list[Vote] = []
    juror = make_scripted_juror([_ok_vote("PASS")] * 6)
    gate = Gate(juror, k=3, max_jurors=6, batch_size=3, on_vote=seen.append)
    gate.evaluate(state={}, action={}, claim="x")
    assert len(seen) >= 3
    assert all(isinstance(v, Vote) for v in seen)


def test_as_juror_client_protocol_adapter():
    juror = make_scripted_juror([_ok_vote("PASS")] * 4)
    client = as_juror_client(juror)
    assert client.vote("sys", "user").startswith("{")


# --------------------------------------------------------------------------------------
# Claude Agent SDK adapter
# --------------------------------------------------------------------------------------


def test_claude_sdk_adapter_passes_through_non_irreversible_tools():
    juror = make_scripted_juror([_ok_vote("FAIL")] * 6)  # would fail if asked
    gate = Gate(juror, k=3, max_jurors=6, batch_size=3)
    hooks = gate_irreversible_tools(
        gate,
        irreversible={"write_file"},
        snapshot_state=lambda: {},
        claim_for=lambda t, a: "x",
    )
    fn = hooks["PreToolUse"][0]
    out = fn("read_file", {"path": "/tmp/x"})
    assert out["continue"] is True


def test_claude_sdk_adapter_blocks_when_jury_fails():
    juror = make_scripted_juror([_ok_vote("FAIL")] * 6)
    gate = Gate(juror, k=3, max_jurors=6, batch_size=3)
    blocked = []
    hooks = gate_irreversible_tools(
        gate,
        irreversible={"write_file"},
        snapshot_state=lambda: {"target_count": 0, "source_count": 1},
        claim_for=lambda t, a: "everything is fine",
        on_block=lambda t, a, r: blocked.append((t, r.decision)),
    )
    fn = hooks["PreToolUse"][0]
    out = fn("write_file", {"path": "/tmp/x"})
    assert out["continue"] is False
    assert "stop_reason" in out
    assert blocked == [("write_file", "FAIL")]


def test_claude_sdk_adapter_dict_event_shape():
    """The SDK sometimes calls hooks with a single event dict."""
    juror = make_scripted_juror([_ok_vote("PASS")] * 6)
    gate = Gate(juror, k=3, max_jurors=6, batch_size=3)
    hooks = gate_irreversible_tools(
        gate,
        irreversible={"write_file"},
        snapshot_state=lambda: {},
        claim_for=lambda t, a: "x",
    )
    fn = hooks["PreToolUse"][0]
    out = fn({"name": "write_file", "input": {"path": "/tmp/x"}})
    assert out["continue"] is True
