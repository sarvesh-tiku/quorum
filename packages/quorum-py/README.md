# quorum-py

> A `commit()` that blocks on a jury — drop-in consensus gating for irreversible
> agent actions.

`quorum-py` is a small, framework-agnostic Python library that wraps the
**irreversible-tool boundary** of any LLM agent and runs a **read-only consensus
gate** before each side-effectful action. A swarm of cheap juror calls votes on
whether the action is actually correct; the gate either lets the action through
or triggers rollback.

The package gives you:

- `Gate` — a small, dependency-free runtime: red-flag filter + first-to-ahead-by-K voting + parallel polling.
- `@gate.protect(...)` — decorate any callable to run a gate on its post-state.
- Adapters for **Claude Agent SDK**, **LangGraph**, and **OpenAI Agents SDK** — each is a thin shim over the same `Gate`.

The implementation comes from QUORUM's hackathon design (a long-horizon migration
agent whose silent bugs the gate catches). The migration demo lives in
[`../../demo/run_demo.py`](../../demo/run_demo.py); this package is the runtime
distilled out of it.

## Install

```bash
pip install quorum-py                   # core only, zero deps, offline
pip install "quorum-py[live]"           # + the Anthropic SDK for live runs
pip install "quorum-py[dev]"            # + pytest / build tooling
```

For local hacking from a checkout:

```bash
pip install -e ./packages/quorum-py[dev]
```

Adapters do **not** pull their host frameworks. You install the host framework
yourself; the adapter module talks to it via duck typing so version drift
doesn't break this package.

## Quick start (no framework)

```python
from quorum import Gate, make_client, GateBlocked

juror_client = make_client()                    # MockClient offline; LiveClient if ANTHROPIC_API_KEY set
gate = Gate(juror_client, k=3, max_jurors=8)

@gate.protect(
    claim="after this call, the file exists and contains the new contents verbatim",
    snapshot_state=lambda path, contents: {"path": path, "size": len(contents)},
    rollback=lambda path, contents: os.unlink(path),     # called on FAIL
)
def write_file(path: str, contents: str) -> int:
    open(path, "w").write(contents)
    return len(contents)

try:
    write_file("/tmp/foo.txt", "hello")
except GateBlocked as e:
    print(e.result.decision, e.result.root_cause)         # "FAIL" + jury evidence
```

## Claude Agent SDK

```python
from quorum import Gate, make_client
from quorum.adapters.claude_agent_sdk import gate_irreversible_tools

gate = Gate(make_client(), k=3)
hooks = gate_irreversible_tools(
    gate,
    irreversible={"write_file", "shell.exec", "git.commit"},
    snapshot_state=lambda: read_world(),
    claim_for=lambda tool, args: f"after {tool}, repo compiles + tests pass",
)
agent = ClaudeAgent(..., hooks=hooks)
```

## LangGraph

```python
from quorum import Gate
from quorum.adapters.langgraph import make_gate_node

graph.add_node("propose",  propose_node)
graph.add_node("gate",     make_gate_node(gate,
    snapshot_state=lambda s: s["world"],
    claim_for=lambda s: s["claim"],
    claim_kind_for=lambda s: s["claim_kind"]))
graph.add_node("commit",   commit_node)
graph.add_node("rollback", rollback_node)
graph.add_edge("propose", "gate")
graph.add_conditional_edges(
    "gate",
    lambda s: "commit" if s["gate"]["decision"] == "PASS" else "rollback",
)
```

Or wrap a single tool:

```python
from quorum.adapters.langgraph import gate_tool

safe_write = gate_tool(write_tool, gate,
    snapshot_state=lambda *a, **kw: read_world(),
    claim_for=lambda *a, **kw: "post-state holds",
)
```

## OpenAI Agents SDK

```python
from quorum import Gate
from quorum.adapters.openai_agents import gate_function_tool

@gate_function_tool(
    gate,
    snapshot_state=lambda *a, **kw: read_world(),
    claim_for=lambda *a, **kw: "post-state holds",
    claim_kind="diff_semantics",
)
@function_tool                                   # OpenAI's decorator
def write_file(path: str, contents: str) -> str:
    ...
```

Or use the gate as an output guardrail:

```python
from quorum.adapters.openai_agents import make_output_guardrail

guardrail = make_output_guardrail(gate,
    snapshot_state=lambda ctx, agent, out: read_world(),
    claim_for=lambda ctx, agent, out: f"output is supported by sources",
)
agent = Agent(..., output_guardrails=[guardrail])
```

## Why

The frontier problem in agents is **reliability over horizon**, not capability.
Per-step error compounds and silent failures (no stack trace, no alert) survive
self-checks because LLMs are unreliable at judging their own output. The
literature's one demonstrated escape — MAKER's first-to-ahead-by-K voting +
red-flagging — was applied only to extremely decomposed sub-tasks; QUORUM aims
the same machinery at **verifying open-ended plans at the irreversible-action
boundary**.

`quorum-py` is that boundary, packaged.

## Status

Alpha. The runtime is exercised end-to-end by the migration demo + reliability
benchmark in this repo. Public API is settling toward 0.1.0; expect breaking
changes pre-1.0.

## License

MIT.
