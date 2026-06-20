# Changelog

All notable changes to `quorum-py` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows [Semantic Versioning](https://semver.org/) starting at 1.0.

## [0.1.0] — initial release

### Added

- `Gate`: framework-agnostic consensus runtime — first-to-ahead-by-K voting,
  red-flag filter, parallel polling.
- `@gate.protect(...)`: decorator API to gate any callable on its post-state
  (with optional rollback + custom claim/state functions).
- `GateBlocked` exception: surfaces full juror evidence on FAIL.
- `quorum.adapters.claude_agent_sdk.gate_irreversible_tools`: PreToolUse
  hook adapter for the Claude Agent SDK.
- `quorum.adapters.langgraph.{gate_tool, make_gate_node}`: tool-wrap and
  node-factory adapters for LangGraph.
- `quorum.adapters.openai_agents.{gate_function_tool, make_output_guardrail}`:
  tool-wrap and output-guardrail adapters for the OpenAI Agents SDK.
- Mock + live Claude clients, swappable via `make_client()`.
- 37 tests covering gate logic, red-flag filter, all three adapters, and the
  decorator.
- PEP 561 `py.typed` marker — type information ships with the wheel.

### Notes

- Migration demo (`../../demo/run_demo.py`) and reliability benchmark
  (`../../demo/reliability.py`) exercise the gate end-to-end against an
  in-memory Postgres-shaped world with injected silent faults.
- Public API is alpha; expect breaking changes pre-1.0.
