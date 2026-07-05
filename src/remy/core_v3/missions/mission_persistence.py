"""
Mission Persistence for Remy v3.

Saves and loads mission/goal/task state to disk (JSON).
Ensures the system can resume after restart without losing progress.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from typing import Any

from .mission_models import (
    Mission, MissionStatus, MissionMode, Goal, GoalStatus,
    Task, TaskStatus, TaskRepeat, BudgetEstimate, RiskLevel,
    SuccessCriterion,
)

log = logging.getLogger(__name__)

_MISSIONS_FILE = "v3_missions_state.json"


class MissionPersistence:
    """Persists v3 mission state to disk.

    File: data/v3_missions_state.json
    Format: {missions: [...], goals: [...], tasks: [...], saved_at: float}
    """

    def __init__(self, data_dir: str = ""):
        if not data_dir:
            try:
                from remy.config import settings
                data_dir = str(settings.DATA_DIR)
            except ImportError:
                data_dir = "data"
        self._path = os.path.join(data_dir, _MISSIONS_FILE)

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------

    def save(
        self,
        missions: dict[str, Mission],
        goals: dict[str, Goal],
        tasks: dict[str, Task],
    ):
        """Save all mission state to disk."""
        data = {
            "saved_at": time.time(),
            "version": 1,
            "missions": [self._serialize_mission(m) for m in missions.values()],
            "goals": [self._serialize_goal(g) for g in goals.values()],
            "tasks": [self._serialize_task(t) for t in tasks.values()],
        }
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp, self._path)
            log.debug("Saved %d missions, %d goals, %d tasks",
                      len(missions), len(goals), len(tasks))
        except Exception as e:
            log.error("Failed to save mission state: %s", e)

    # -------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------

    def load(self) -> tuple[dict[str, Mission], dict[str, Goal], dict[str, Task]]:
        """Load mission state from disk.

        Returns:
            (missions, goals, tasks) dicts keyed by id
        """
        if not os.path.exists(self._path):
            return {}, {}, {}

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.error("Failed to load mission state: %s", e)
            return {}, {}, {}

        missions = {}
        for md in data.get("missions", []):
            m = self._deserialize_mission(md)
            missions[m.id] = m

        goals = {}
        for gd in data.get("goals", []):
            g = self._deserialize_goal(gd)
            goals[g.id] = g

        tasks = {}
        for td in data.get("tasks", []):
            t = self._deserialize_task(td)
            tasks[t.id] = t

        # On startup: tasks that were RUNNING when the process died are now
        # orphaned — no specialist is executing them.  Reset to PENDING so the
        # scheduler can pick them up again.
        reset_count = 0
        for t in tasks.values():
            if t.status == TaskStatus.RUNNING:
                t.status = TaskStatus.PENDING
                reset_count += 1
        if reset_count:
            log.info("Startup reset: %d orphaned RUNNING tasks → PENDING", reset_count)

        # On startup: missions stuck in ESCALATED should resume as ACTIVE so the
        # scheduler considers them. Escalation is per-session state, not permanent.
        escalated_reset = 0
        for m in missions.values():
            if m.status == MissionStatus.ESCALATED:
                m.status = MissionStatus.ACTIVE
                escalated_reset += 1
        if escalated_reset:
            log.info("Startup reset: %d ESCALATED missions → ACTIVE", escalated_reset)

        log.info("Loaded state: %d missions, %d goals, %d tasks",
                 len(missions), len(goals), len(tasks))
        return missions, goals, tasks

    def exists(self) -> bool:
        return os.path.exists(self._path)

    # -------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------

    def _serialize_mission(self, m: Mission) -> dict:
        return {
            "id": m.id,
            "description": m.description,
            "objective": m.objective,
            "status": m.status.value,
            "mode": m.mode.value,
            "priority": m.priority,
            "tags": m.tags,
            "goal_ids": m.goal_ids,
            "plan_id": m.plan_id,
            "budget": {
                "tokens": m.budget.tokens,
                "cost_usd": m.budget.cost_usd,
                "time_sec": m.budget.time_sec,
                "tool_calls": m.budget.tool_calls,
                "risk": m.budget.risk.value,
            },
            "risk": m.risk.value,
            "requires_approval": m.requires_approval,
            "created_at": m.created_at,
            "updated_at": m.updated_at,
            "completed_at": m.completed_at,
            "cycles_run": m.cycles_run,
            "total_cost_usd": m.total_cost_usd,
            "total_tokens": m.total_tokens,
            "source": m.source,
            "immortal": m.immortal,
            "outcomes": m.outcomes,
            "lessons": m.lessons,
            "record_id": m.record_id,
        }

    def _deserialize_mission(self, d: dict) -> Mission:
        budget_d = d.get("budget", {})
        return Mission(
            id=d["id"],
            description=d.get("description", ""),
            objective=d.get("objective", ""),
            status=MissionStatus(d.get("status", "intake")),
            mode=MissionMode(d.get("mode", "quick_tactical")),
            priority=d.get("priority", 5),
            tags=d.get("tags", []),
            goal_ids=d.get("goal_ids", []),
            plan_id=d.get("plan_id", ""),
            budget=BudgetEstimate(
                tokens=budget_d.get("tokens", 0),
                cost_usd=budget_d.get("cost_usd", 0.0),
                time_sec=budget_d.get("time_sec", 0),
                tool_calls=budget_d.get("tool_calls", 0),
                risk=RiskLevel(budget_d.get("risk", "low")),
            ),
            risk=RiskLevel(d.get("risk", "low")),
            requires_approval=d.get("requires_approval", False),
            created_at=d.get("created_at", 0.0),
            updated_at=d.get("updated_at", 0.0),
            completed_at=d.get("completed_at", 0.0),
            cycles_run=d.get("cycles_run", 0),
            total_cost_usd=d.get("total_cost_usd", 0.0),
            total_tokens=d.get("total_tokens", 0),
            source=d.get("source", ""),
            immortal=d.get("immortal", False),
            outcomes=d.get("outcomes", []),
            lessons=d.get("lessons", []),
            record_id=d.get("record_id", ""),
        )

    def _serialize_goal(self, g: Goal) -> dict:
        return {
            "id": g.id,
            "description": g.description,
            "status": g.status.value,
            "priority": g.priority,
            "tags": g.tags,
            "metadata": g.metadata,
            "mission_id": g.mission_id,
            "parent_goal_id": g.parent_goal_id,
            "task_ids": g.task_ids,
            "pack_template": g.pack_template,
            "attempts": g.attempts,
            "max_attempts": g.max_attempts,
            "immortal": g.immortal,
            "success_criteria": [c.to_dict() for c in g.success_criteria],
            "created_at": g.created_at,
            "updated_at": g.updated_at,
            "completed_at": g.completed_at,
            "record_id": g.record_id,
        }

    def _deserialize_goal(self, d: dict) -> Goal:
        return Goal(
            id=d["id"],
            description=d.get("description", ""),
            status=GoalStatus(d.get("status", "pending")),
            priority=d.get("priority", 5),
            tags=d.get("tags", []),
            metadata=d.get("metadata", {}),
            mission_id=d.get("mission_id", ""),
            parent_goal_id=d.get("parent_goal_id", ""),
            task_ids=d.get("task_ids", []),
            pack_template=d.get("pack_template", ""),
            attempts=d.get("attempts", 0),
            max_attempts=d.get("max_attempts", 5),
            immortal=d.get("immortal", False),
            success_criteria=[
                SuccessCriterion.from_dict(c)
                for c in d.get("success_criteria", [])
            ],
            created_at=d.get("created_at", 0.0),
            updated_at=d.get("updated_at", 0.0),
            completed_at=d.get("completed_at", 0.0),
            record_id=d.get("record_id", ""),
        )

    def _serialize_task(self, t: Task) -> dict:
        return {
            "id": t.id,
            "action": t.action,
            "done_when": t.done_when,
            "status": t.status.value,
            "priority": t.priority,
            "repeat": t.repeat.value,
            "depends_on": t.depends_on,
            "attempts": t.attempts,
            "last_attempt_at": t.last_attempt_at,
            "completed_at": t.completed_at,
            "next_run_after": t.next_run_after,
            "error": t.error,
            "goal_id": t.goal_id,
            "mission_id": t.mission_id,
            "record_id": t.record_id,
            "success_criteria": [c.to_dict() for c in t.success_criteria],
            "blocker_reason": t.blocker_reason,
            "waiting_reason": t.waiting_reason,
            "metadata": t.metadata,
        }

    def _deserialize_task(self, d: dict) -> Task:
        return Task(
            id=d["id"],
            action=d.get("action", ""),
            done_when=d.get("done_when", ""),
            status=TaskStatus(d.get("status", "pending")),
            priority=d.get("priority", 5),
            repeat=TaskRepeat(d.get("repeat", "once")),
            depends_on=d.get("depends_on", []),
            attempts=d.get("attempts", 0),
            last_attempt_at=d.get("last_attempt_at", 0.0),
            completed_at=d.get("completed_at", 0.0),
            next_run_after=d.get("next_run_after", 0.0),
            error=d.get("error", ""),
            goal_id=d.get("goal_id", ""),
            mission_id=d.get("mission_id", ""),
            record_id=d.get("record_id", ""),
            success_criteria=[
                SuccessCriterion.from_dict(c)
                for c in d.get("success_criteria", [])
            ],
            blocker_reason=d.get("blocker_reason", ""),
            waiting_reason=d.get("waiting_reason", ""),
            metadata=d.get("metadata", {}),
        )
