"""
Agent Output Runtime for Remy v3.

Owns normalization from specialist-facing ``AgentOutput`` objects into the
execution-facing ``ExecutionResult`` contract.
"""

from __future__ import annotations


class AgentOutputRuntime:
    """Normalize specialist output into execution-layer results."""

    def normalize(self, agent_output):
        return agent_output.to_execution_result()
