"""
Mission Outcome Runtime for Remy v3.

Owns mission-level completion and escalation transitions so OutcomeRuntime does
not manipulate mission state directly.
"""

from __future__ import annotations

from ..missions.mission_models import Mission, MissionStatus
from .state_machine import can_transition, transition


class MissionOutcomeRuntime:
    """Apply post-evaluation mission-level state transitions."""

    def __init__(self, mission_runtime):
        self.mission_runtime = mission_runtime

    def should_complete(self, *, mission: Mission, plan, current_task=None) -> bool:
        if plan.is_complete:
            return True

        tasks = self.mission_runtime.mission_tasks(mission.id)
        if not tasks:
            return False

        active_statuses = {
            "pending",
            "active",
            "running",
            "waiting",
            "blocked",
            "blocked_external",
            "blocked_approval",
        }
        for task in tasks:
            if current_task is not None and task.id == current_task.id:
                continue
            if getattr(task.status, "value", str(task.status)) in active_statuses:
                return False
        return True

    def complete(self, *, mission: Mission, reason: str = "All steps done") -> None:
        if can_transition(mission.status, MissionStatus.COMPLETED):
            transition(mission, MissionStatus.COMPLETED, reason)

    def escalate(self, *, mission: Mission, reason: str) -> None:
        if can_transition(mission.status, MissionStatus.ESCALATED):
            transition(mission, MissionStatus.ESCALATED, reason)
