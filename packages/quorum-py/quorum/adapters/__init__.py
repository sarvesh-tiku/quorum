"""Framework adapters for `quorum.Gate`.

Each adapter wires the gate into a specific agent framework's irreversible-
action surface. Adapters are imported on demand so installing the core
package never pulls in framework dependencies you don't use:

    from quorum.adapters.claude_agent_sdk import gate_irreversible_tools
    from quorum.adapters.langgraph         import GateNode          # planned
    from quorum.adapters.openai_agents     import gate_guardrail    # planned
"""
