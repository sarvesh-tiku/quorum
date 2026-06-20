"""QUORUM — a long-horizon agent whose every irreversible action passes a consensus gate.

A single-threaded "decider" brain (Claude Opus 4.8) proposes irreversible actions.
Before any irreversible action commits, a swarm of cheap, independent, read-only
"juror" microagents (Claude Haiku 4.5) vote on whether the action is correct, using
first-to-ahead-by-K voting with a red-flag filter. On FAIL, the system rewinds to the
last good snapshot and the brain re-plans with the jury's evidence in context.

This converts unreliable SELF-verification into structurally reliable EXTERNAL
verification — the missing reliability layer for long-horizon autonomous agents.
"""

__version__ = "0.1.0"
