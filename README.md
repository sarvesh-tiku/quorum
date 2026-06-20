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

## The demo: baseline vs QUORUM on a silent DB-migration bug

The motivating task: migrate the `orders` service + Postgres from a legacy to a new
stack with zero data loss / zero downtime. Stateful, side-effectful, mostly-irreversible
— the opposite of a coding benchmark, because a bad cast or a dropped row is permanent.

A fault injector introduces two "looks-done" bugs:

- a `NUMERIC(10,2) → INTEGER` cast that silently drops the cents from every order, and
- 12 dropped change-data-capture rows (writes that landed only on the old DB).

An independent reconciliation oracle (not part of the agent) computes source-vs-target
row counts + per-row checksums + a revenue total and prints a hard PASS / FAIL.

- **Baseline (gate OFF):** the agent reports "migration complete" — oracle says **FAIL**.
- **QUORUM (gate ON):** same world, same faults — the jury blocks the bad commit with execution-grounded evidence, the jurors localize the root cause, the brain rewinds + re-plans, retry passes — oracle says **PASS**.

```
  BASELINE → FAIL ❌   missing=20 mismatched=859 revenue_delta=-10317.60
  QUORUM   → PASS ✅   missing=0  mismatched=0   revenue_delta=0.00
            jurors_polled=48 red_flagged=5 gate_blocks=2 rollbacks=2
```

And the reliability curve QUORUM is designed to flatten (the τ-bench `pass^k` story):

```
  baseline   |                                        | 0%
  quorum     |████████████████████████████████████████| 100%
       (40 trials each, faults ON, juror noise 12%, red-flag 10%, K=3)
```

---

## Run it

Requires Python 3.10+. The whole loop runs **offline** against a mock Claude client — no
key needed.

```bash
# editable install of the package (also pulls demo deps via dev extras)
make install                                  # → pip install -e ./packages/quorum-py[dev]

# the side-by-side baseline-vs-QUORUM demo (offline, deterministic)
make demo                                     # → python3 demo/run_demo.py

# slow it down for a live audience / screen-record
make slow                                     # → python3 demo/run_demo.py --slow 0.08

# the reliability benchmark (the pass^k chart)
make reliability                              # → python3 demo/reliability.py --trials 40

# tests
make test                                     # → python3 -m pytest tests/ -q
```

### The web verification feed

A self-contained `web/index.html` replays the run as a live verification feed (votes
streaming in, the running tally, gate verdicts, rollbacks) with the side-by-side oracle
verdict and the reliability chart baked in — no server, no network.

```bash
# regenerate the embedded traces, then rebuild the page
python3 demo/run_demo.py     --json-out web/trace.json
python3 demo/reliability.py  --trials 40 --json-out web/reliability.json
python3 demo/build_web.py
open web/index.html
```

### Going live (Claude API)

The system is agnostic to mock vs live. Set the key and pass `--live`:

```bash
export ANTHROPIC_API_KEY=...          # or put it in ../.hackathon.env
python3 demo/run_demo.py --live
```

- **Brain** = `claude-opus-4-8`, one strong reasoning call per step (adaptive thinking,
  high effort), low call volume.
- **Jurors** = `claude-haiku-4-5`, many cheap parallel read-only calls — *this is the
  volume*, and exactly the workload near-free / near-0-latency inference (GB200) makes
  affordable on every step.

---

## What's real, what's mocked, what's the GPU swap-in

| Piece | Status |
|---|---|
| Single-threaded brain + decision journal + checkable claims | **Real** (offline + live) |
| Consensus gate: red-flag filter + first-to-ahead-by-K, parallel jurors | **Real** |
| Snapshot / rollback + root-cause-localized recovery loop | **Real** |
| Independent reconciliation oracle (hard PASS/FAIL) | **Real** — not part of the agent |
| Fault injector (NUMERIC→INT truncation, dropped CDC rows) | **Real** |
| Reliability benchmark (pass^k curve) | **Real** |
| Migration "world" | A faithful **in-memory** stand-in for source/target Postgres + live write traffic + CDC — so the loop is offline and deterministic. Every op maps 1:1 to a real migration step. |
| Claude calls | **Mocked by default** (evidence-grounded fixtures: juror verdicts are computed from the *real* world state in the prompt, so the jury genuinely catches the bug offline). **Live** behind `ANTHROPIC_API_KEY` + `--live`. |
| Juror fleet on GB200 | **Optional demo-time swap-in.** The `/vote` workload is identical whether served by the Haiku 4.5 API or a vLLM fleet on the GB200 box — only the client changes. The whole value prop *is* "the expensive brain decides; cheap near-infinite compute verifies." |

The architecture mirrors the Claude Agent SDK shape: irreversible tools are **tagged**
(`quorum/tools.py::IRREVERSIBLE`) so the gate sits exactly where a `PreToolUse` hook would
intercept and route to the jury — and that's exactly what the
[`quorum.adapters.claude_agent_sdk`](packages/quorum-py/quorum/adapters/claude_agent_sdk.py)
shim ships. Jurors get a read-only query surface (the MCP analogue).

---

## Layout

```
packages/quorum-py/                 the published library (PyPI: quorum-py)
  pyproject.toml                    PEP 621 metadata, optional extras: live, dev, claude-agent-sdk, langgraph, openai-agents
  README.md                         package-level docs
  CHANGELOG.md
  quorum/
    gate.py                         framework-agnostic Gate + @gate.protect + GateBlocked
    jurors.py                       migration-coupled ConsensusGate (used by the demo)
    brain.py                        single-threaded decider + orchestrator
    world.py                        snapshot-able migration world + fault injector + reconciliation oracle
    tools.py                        which tools are IRREVERSIBLE (gated)
    llm.py                          Claude client abstraction: MockClient (offline) + LiveClient
    py.typed                        PEP 561 marker — fully typed
    adapters/
      claude_agent_sdk.py           gate_irreversible_tools(...) → PreToolUse hook
      langgraph.py                  gate_tool(...), make_gate_node(...)
      openai_agents.py              gate_function_tool(...), make_output_guardrail(...)

demo/
  run_demo.py                       baseline-vs-QUORUM, live terminal feed + JSON trace export
  reliability.py                    the pass^k reliability benchmark
  build_web.py                      bakes the traces into a self-contained web/index.html
tests/
  test_quorum.py                    migration end-to-end (world, recovery loop)
  test_gate.py                      framework-agnostic Gate (no migration deps)
  test_adapters.py                  Claude Agent SDK / LangGraph / OpenAI Agents adapters
web/
  index.html                        self-contained live verification-feed UI (open directly)
```

## Why it's novel

Everyone is trying to make the *brain* more reliable (bigger models, RL, self-correction —
which the literature shows often backfires). QUORUM's claim is that **long-horizon
reliability is a property of the verification layer, not the policy** — and that layer is
the ideal place to spend inference-time compute, because verification is read-only and
embarrassingly parallel. We take the one method that beat error-compounding (MAKER's
voting + red-flag) and aim it at the one thing it left unsolved (verifying an open-ended
plan), inside the one architecture a host (Cognition) endorses, defeating agreement bias
by hiding the brain's reasoning from the jurors. Nobody has shipped *consensus-gated
irreversible actions*. That's the primitive: **a `commit()` that blocks on a jury.**
