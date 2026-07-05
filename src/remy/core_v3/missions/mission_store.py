"""
Mission Store for Remy v3.

Loads, persists, and manages mission state.
Phase 1: Adapts v2 missions.json + autonomy_goals into v3 Mission objects.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from .mission_models import (
    Mission, MissionStatus, Goal, GoalStatus, Task, TaskStatus,
    TaskRepeat, SuccessCriterion, mission_from_json, goal_from_v2_record,
    _parse_priority,
)

log = logging.getLogger(__name__)


class MissionStore:
    """Manages mission lifecycle and persistence.

    Phase 1: reads from v2 data/missions.json and wraps
    existing autonomy_goals records.
    """

    def __init__(self, data_dir: str = ""):
        if not data_dir:
            try:
                from remy.config import settings
                data_dir = str(settings.DATA_DIR)
            except ImportError:
                data_dir = "data"
        self._data_dir = data_dir
        self._missions: dict[str, Mission] = {}
        self._goals: dict[str, Goal] = {}
        self._tasks: dict[str, Task] = {}

    # -------------------------------------------------------------------
    # Load from v2
    # -------------------------------------------------------------------

    def load_from_missions_json(self) -> list[Mission]:
        """Load missions from v2 data/missions.json."""
        path = os.path.join(self._data_dir, "missions.json")
        if not os.path.exists(path):
            log.warning("missions.json not found at %s", path)
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.error("Failed to load missions.json: %s", e)
            return []

        missions_data = data if isinstance(data, list) else data.get("missions", [])
        loaded = []
        for entry in missions_data:
            mission = mission_from_json(entry)

            # Load tasks from mission entry
            for task_data in entry.get("tasks", []):
                task = Task(
                    id=task_data.get("id", ""),
                    action=task_data.get("action", ""),
                    done_when=task_data.get("done_when", ""),
                    priority=_parse_priority(task_data.get("priority", 5)),
                    repeat=TaskRepeat(task_data.get("repeat", "once")),
                    depends_on=task_data.get("depends_on", []),
                    mission_id=mission.id,
                    metadata=task_data.get("metadata", {}),
                    success_criteria=[
                        SuccessCriterion.from_dict(c)
                        for c in task_data.get("success_criteria", [])
                    ],
                )
                self._tasks[task.id] = task

            self._missions[mission.id] = mission
            loaded.append(mission)

        log.info("Loaded %d missions from missions.json", len(loaded))
        return loaded

    def load_goals_from_brain(self) -> list[Goal]:
        """Load active goals from v2 brain records."""
        try:
            from remy.core.agent_tools import brain
            records = brain.list_records(tags=["autonomous-goal"], limit=200)
            goals = []
            for rec in records:
                goal = goal_from_v2_record(
                    self._record_to_dict(rec)
                )
                self._goals[goal.id] = goal

                # Link to mission if applicable
                if goal.mission_id and goal.mission_id in self._missions:
                    mission = self._missions[goal.mission_id]
                    if goal.id not in mission.goal_ids:
                        mission.goal_ids.append(goal.id)

                goals.append(goal)

            log.info("Loaded %d goals from brain", len(goals))
            return goals
        except ImportError:
            log.warning("v2 agent_tools not available")
            return []

    def _record_to_dict(self, record) -> dict:
        """Convert Aura record to plain dict."""
        return {
            "id": getattr(record, "id", "") or str(getattr(record, "record_id", "")),
            "content": getattr(record, "content", ""),
            "tags": list(getattr(record, "tags", []) or []),
            "metadata": dict(getattr(record, "metadata", {}) or {}),
        }

    # -------------------------------------------------------------------
    # CRUD
    # -------------------------------------------------------------------

    def get_mission(self, mission_id: str) -> Mission | None:
        return self._missions.get(mission_id)

    def get_goal(self, goal_id: str) -> Goal | None:
        return self._goals.get(goal_id)

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def active_missions(self) -> list[Mission]:
        return [
            m for m in self._missions.values()
            if m.is_active() or m.status == MissionStatus.INTAKE
        ]

    def all_missions(self) -> list[Mission]:
        return list(self._missions.values())

    def add_mission(self, mission: Mission):
        self._missions[mission.id] = mission

    def add_goal(self, goal: Goal):
        self._goals[goal.id] = goal

    def add_task(self, task: Task):
        self._tasks[task.id] = task

    def mission_goals(self, mission_id: str) -> list[Goal]:
        return [g for g in self._goals.values() if g.mission_id == mission_id]

    def mission_tasks(self, mission_id: str) -> list[Task]:
        return [t for t in self._tasks.values() if t.mission_id == mission_id]

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "missions": len(self._missions),
            "active": len(self.active_missions()),
            "goals": len(self._goals),
            "tasks": len(self._tasks),
            "by_status": {
                status.value: sum(
                    1 for m in self._missions.values() if m.status == status
                )
                for status in MissionStatus
            },
        }
