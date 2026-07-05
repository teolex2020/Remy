"""
Cost Runtime for Remy v3.

Owns application of execution cost to mission-level accounting so the chief
agent does not mutate financial/token state inline.
"""

from __future__ import annotations


class CostRuntime:
    """Apply normalized execution cost to mission state."""

    def apply(self, *, mission, exec_result) -> None:
        mission.add_cost(exec_result.tokens_used, exec_result.cost_usd)
