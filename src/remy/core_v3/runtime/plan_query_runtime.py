"""
Plan query runtime for Remy v3.

Read-side access to plan state so projection/telemetry do not depend on
execution-oriented mission runtime methods for step resolution.
"""

from __future__ import annotations


class PlanQueryRuntime:
    """Read-only query facade for plan state."""

    def __init__(self, *, plans):
        self._plans = plans

    def get_plan(self, mission_id: str):
        return self._plans.get(mission_id)

    def current_step(self, plan, task=None):
        if plan is None:
            return None
        if task is not None:
            expected_step_id = f"step_{task.id}"
            for step in plan.steps:
                if step.id == expected_step_id:
                    return step
        return plan.next_step()

    def plan_progress(self, plan) -> float:
        return plan.progress if plan else 0.0

    def plan_steps(self, plan) -> int:
        return len(plan.steps) if plan else 0
