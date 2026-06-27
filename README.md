# QUORUM commit() by consensus

Inference-Time Compute Hackathon 2026 · Agents Track. Every irreversible action is verified by a quorum of independent jurors before it executes.

## The package: `quorum-py`
A lightweight, framework-agnostic Python library that gates every irreversible agent action behind a read-only consensus vote. Independent jurors verify the action before it commits; failed votes trigger rollback.
The runtime is published as [`quorum-py`](https://test.pypi.org/project/quorum-py/) : 

What you get:

- **`Gate`** — dependency-free runtime: red-flag filter + first-to-ahead-by-K voting + parallel polling.
- **`@gate.protect(...)`** — decorate any callable to run a gate on its post-state.
- **Adapters** for the **Claude Agent SDK**, **LangGraph**, and the **OpenAI Agents SDK** — thin shims over the same `Gate`. None of them import their host framework, so version drift won't break this package.

### Install

```bash
pip install quorum-py                   # core only, zero deps, offline
pip install "quorum-py[live]"           # + Anthropic SDK for live runs
pip install "quorum-py[dev]"            # + pytest / build tooling

# from TestPyPI today (real PyPI release pending):
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ quorum-py

# from a checkout:
pip install -e ./packages/quorum-py[dev]
```

Requires Python 3.10+. Fully typed (`py.typed` ships with the wheel).

### Quick start (no framework)

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
    print(e.result.decision, e.result.root_cause)        # "FAIL" + jury evidence
```

### Claude Agent SDK

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

### LangGraph

```python
from quorum import Gate
from quorum.adapters.langgraph import make_gate_node, gate_tool

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

# Or wrap a single tool:
safe_write = gate_tool(write_tool, gate,
    snapshot_state=lambda *a, **kw: read_world(),
    claim_for=lambda *a, **kw: "post-state holds",
)
```

### OpenAI Agents SDK

```python
from quorum import Gate
from quorum.adapters.openai_agents import gate_function_tool, make_output_guardrail

@gate_function_tool(
    gate,
    snapshot_state=lambda *a, **kw: read_world(),
    claim_for=lambda *a, **kw: "post-state holds",
    claim_kind="diff_semantics",
)
@function_tool                                   # OpenAI's decorator
def write_file(path: str, contents: str) -> str:
    ...

# Or use the gate as an output guardrail:
guardrail = make_output_guardrail(gate,
    snapshot_state=lambda ctx, agent, out: read_world(),
    claim_for=lambda ctx, agent, out: "output is supported by sources",
)
agent = Agent(..., output_guardrails=[guardrail])
```

Status: **alpha**. Public API is settling toward 0.1.0; expect breaking changes pre-1.0.
Full package docs at [`packages/quorum-py/README.md`](packages/quorum-py/README.md).

---
