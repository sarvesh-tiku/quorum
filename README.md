# QUORUM — a `commit()` that blocks on a jury

> **Inference-Time Compute Hackathon 2026 · Agents track.** A long-horizon autonomous
> agent whose every **irreversible** action must first pass a **consensus gate**: a swarm
> of cheap, independent, read-only juror microagents that vote on whether the action is
> actually correct — converting unreliable *self*-verification into structurally reliable
> *external* verification.

## The package: `quorum-py`

The runtime is published as [`quorum-py`](https://test.pypi.org/project/quorum-py/) — a
small, framework-agnostic Python library that wraps the irreversible-tool boundary of any
LLM agent and runs a read-only consensus gate before each side-effectful action. A swarm
of cheap juror calls votes on whether the action is correct; the gate either lets it
through or triggers rollback.

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

The frontier problem in agents is not capability — it's **reliability over horizon**. Task
success decays exponentially with the number of dependent steps, because a small per-step
error rate compounds and errors propagate *silently* (no stack trace, no alert) — and LLMs
are unreliable at judging the correctness of their own outputs (agreement / self-enhancement
bias; self-correction can even make things worse). The one demonstrated escape from
error-compounding — **MAKER** (a million-step, zero-error system) — works by extreme
decomposition + **first-to-ahead-by-K voting** + **red-flagging**, but it punts on
*verification of an open-ended plan*. Meanwhile Cognition argues **"don't build
multi-agents"** — parallel decision-makers make conflicting implicit decisions — while
blessing one exception: **read-only, decision-free subagents**.

**QUORUM is the unbuilt thing those threads point at.** Keep a single-threaded brain that
owns every decision (Cognition-compliant). Pour parallel, read-only verification compute
into a **quorum** that gates each irreversible step (MAKER's machinery, pointed at the
thing MAKER didn't do). Reliability becomes a **compute dial**, not a smarter brain — and
verification is read-only and embarrassingly parallel, exactly what near-free, near-0-latency
inference (GB200 / the Etched thesis) makes nearly free.

```
                 GOAL
                  │
        ┌─────────▼──────────┐
        │   DECIDER (brain)  │  single-threaded, owns ALL decisions  (Claude Opus 4.8)
        │  plan → action +   │  emits ONE irreversible action + a CHECKABLE CLAIM
        │  a falsifiable claim│  ("after this, target == source: counts, checksums, revenue")
        └─────────┬──────────┘
                  │
        ┌─────────▼─────────────────────────────────────────────┐
        │   CONSENSUS GATE   (the new primitive)                 │
        │   N independent READ-ONLY jurors (Claude Haiku 4.5)    │
        │   see {state, action, claim} — NOT the brain's         │
        │   reasoning (kills agreement bias). Each gathers its    │
        │   OWN evidence (runs its own count/checksum query).     │
        │   • RED-FLAG filter: drop malformed / no-evidence votes │
        │   • FIRST-TO-AHEAD-BY-K: cheap when obvious, more       │
        │     jurors only when genuinely uncertain                │
        └─────────┬───────────────────────────┬─────────────────┘
              PASS │                           │ FAIL / NO-CONSENSUS
                   ▼                           ▼
         commit; snapshot;           jurors localize the root cause →
         append to journal           rollback to last good snapshot →
                                     brain re-plans WITH the evidence
```
---
