"""
Mission query runtime for Remy v3.

Provides a read-side query layer over mission/goal/task/plan state so
projection and telemetry code do not reach into ChiefAgent directly.
"""

from __future__ import annotations


class MissionQueryRuntime:
    """Read-side query facade for mission runtime state."""

    def __init__(self, *, missions, goals, tasks, plans):
        self._missions = missions
        self._goals = goals
        self._tasks = tasks
        self._plans = plans

    def get_mission(self, mission_id: str):
        return self._missions.get(mission_id)

    def get_plan(self, mission_id: str):
        return self._plans.get(mission_id)

    def active_missions(self):
        return [mission for mission in self._missions.values() if mission.is_active()]

    def all_missions(self):
        return list(self._missions.values())

    def mission_goals(self, mission_id: str):
        return [goal for goal in self._goals.values() if goal.mission_id == mission_id]

    def mission_tasks(self, mission_id: str):
        return [task for task in self._tasks.values() if task.mission_id == mission_id]
