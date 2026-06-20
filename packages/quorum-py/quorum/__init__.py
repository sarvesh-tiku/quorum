"""quorum-py — a `commit()` that blocks on a jury.

Public API:

    from quorum import Gate, Vote, GateResult, red_flag

    gate = Gate(juror_client, k=3, max_jurors=24)
    result = gate.evaluate(state, action, claim, claim_kind="schema")
    if result.blocked:
        rollback()

`Gate` is framework-agnostic. Adapters that wire it into specific agent
frameworks (Claude Agent SDK, LangGraph, OpenAI Agents SDK) live under
`quorum.adapters.*` and are imported lazily — they don't pull in their
host framework unless you actually use them.

The migration demo's `ConsensusGate` is also exposed here for the existing
demo + tests; new code should prefer `Gate`.
"""

from .gate import (
    DEFAULT_JUROR_SYSTEM,
    Gate,
    GateBlocked,
    GateResult,
    JurorCallable,
    JurorClient,
    Vote,
    as_juror_client,
    red_flag,
)
from .llm import (
    BRAIN_MODEL,
    JUROR_MODEL,
    ClaudeClient,
    LiveClient,
    MockClient,
    MockConfig,
    make_client,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "Gate",
    "GateBlocked",
    "Vote",
    "GateResult",
    "red_flag",
    "DEFAULT_JUROR_SYSTEM",
    "JurorClient",
    "JurorCallable",
    "as_juror_client",
    # LLM clients
    "ClaudeClient",
    "MockClient",
    "MockConfig",
    "LiveClient",
    "make_client",
    "BRAIN_MODEL",
    "JUROR_MODEL",
    # Versioning
    "__version__",
]
