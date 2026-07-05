"""
Goal query runtime for Remy v3.

Read-side access to goal and task summaries so projection code does not depend
on GoalTracker for reporting concerns.
"""

from __future__ import annotations

from ..missions.mission_models import GoalStatus


class GoalQueryRuntime:
    """Read-only query facade for goals/tasks."""

    def __init__(self, *, goals, tasks):
        self._goals = goals
        self._tasks = tasks

    def mission_goals(self, mission_id: str):
        return [goal for goal in self._goals.values() if goal.mission_id == mission_id]

    def mission_tasks(self, mission_id: str):
        return [task for task in self._tasks.values() if task.mission_id == mission_id]

    def summary(self, mission_id: str = "") -> dict:
        goals = self.mission_goals(mission_id) if mission_id else list(self._goals.values())
        return {
            "total": len(goals),
            "by_status": {
                status.value: sum(1 for goal in goals if goal.status == status)
                for status in GoalStatus
            },
            "blocked": [
                {"id": goal.id, "reason": goal.metadata.get("block_reason", "")}
                for goal in goals if goal.status == GoalStatus.BLOCKED
            ],
        }
