# QUORUM — a `commit()` that blocks on a jury

> **Inference-Time Compute Hackathon 2026 · Agents track.** A long-horizon autonomous
> agent whose every **irreversible** action must first pass a **consensus gate**: a swarm
> of cheap, independent, read-only juror microagents that vote on whether the action is
> actually correct — converting unreliable *self*-verification into structurally reliable
> *external* verification.

**Now shipped as a Python package.** The runtime is published as
[`quorum-py`](https://test.pypi.org/project/quorum-py/) — a small,
framework-agnostic library you can drop in front of any agent's irreversible tools, with
adapters for the **Claude Agent SDK**, **LangGraph**, and the **OpenAI Agents SDK**. The
migration story below is the demo that motivated the package.

```bash
pip install quorum-py                       # core, zero deps, offline
pip install "quorum-py[live]"               # + Anthropic SDK for live runs
# or, from TestPyPI today:
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ quorum-py
```

```python
from quorum import Gate, GateBlocked, make_client

gate = Gate(make_client(), k=3, max_jurors=8)

@gate.protect(
    claim="after this call, the file exists with the new contents verbatim",
    snapshot_state=lambda path, contents: {"path": path, "size": len(contents)},
    rollback=lambda path, contents: os.unlink(path),
)
def write_file(path, contents):
    open(path, "w").write(contents)
```

See [`packages/quorum-py/README.md`](packages/quorum-py/README.md) for the full package
docs (Claude Agent SDK hooks, LangGraph nodes, OpenAI Agents guardrails).

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

The task is the headline from the design doc, made fully offline and deterministic:

> *Migrate the `orders` service and its Postgres database from the legacy stack to the
> new target stack, with zero data loss and zero downtime — and prove it.*

It's a stateful, side-effectful, **mostly-irreversible** operations task — the opposite of
a coding benchmark, where you can re-run tests forever. Here a wrong cast or a dropped row
is permanent and silent. A **fault injector** introduces exactly the bugs that "look done":

- a `NUMERIC(10,2) → INTEGER` cast that silently **drops the cents** from every order, and
- **12 dropped change-data-capture rows** (writes that landed only on the old DB).

An **independent reconciliation oracle** (not part of the agent) then computes source-vs-target
row counts + per-row checksums + a revenue total and prints a hard **PASS / FAIL** — no judge
subjectivity.

**Act 1 — the baseline dies.** Single agent, gate OFF, faults ON. It reports *"migration
complete."* The oracle says **FAIL**: ~20 rows missing, hundreds of rows truncated,
thousands of dollars of revenue silently corrupted.

**Act 2 — QUORUM survives.** Same world, same faults, gate ON. At the corrupted provision
and the dropped-CDC replay, the jury votes **FAIL** with execution-grounded evidence, the
gate **blocks the commit**, the jurors **localize the root cause**, QUORUM **rolls back to
the last snapshot**, the brain **re-plans**, and the retry **passes**. The oracle says
**PASS**: zero data loss.

```
  BASELINE → FAIL ❌   missing=20 mismatched=859 revenue_delta=-10317.60
  QUORUM   → PASS ✅   missing=0  mismatched=0   revenue_delta=0.00
            jurors_polled=48 red_flagged=5 gate_blocks=2 rollbacks=2

  ★ The jury overruled the brain on a silent bug the baseline shipped, rewound, and fixed it.
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
