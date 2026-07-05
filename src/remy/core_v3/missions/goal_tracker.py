"""
Goal Progress Tracker for Remy v3.

Manages goal lifecycle: selection, progress tracking, auto-completion,
blocking, and memory-assisted context recall.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .mission_models import (
    Mission, MissionStatus, Goal, GoalStatus,
    Task, TaskStatus, TaskRepeat,
)
from ..runtime.state_machine import transition_task

log = logging.getLogger(__name__)


class GoalTracker:
    """Tracks goal progress across cycles.

    Integrates with memory for context recall and outcome storage.
    """

    def __init__(self, goals: dict[str, Goal] | None = None):
        self._goals = goals or {}

    def bind_goals(self, goals: dict[str, Goal]):
        self._goals = goals

    # -------------------------------------------------------------------
    # Selection
    # -------------------------------------------------------------------

    def select_top_goal(self, mission: Mission) -> Goal | None:
        """Select the highest-priority runnable goal for a mission."""
        candidates = [
            g for g in self._goals.values()
            if g.mission_id == mission.id
            and g.is_runnable()
            and g.attempts < g.max_attempts
        ]
        if not candidates:
            return None

        # Prefer active over pending, then higher priority (lower number)
        candidates.sort(key=lambda g: (
            0 if g.status == GoalStatus.ACTIVE else 1,
            g.priority,
            g.created_at,
        ))
        return candidates[0]

    def get_runnable_goals(self, mission_id: str = "") -> list[Goal]:
        """Get all runnable goals, optionally filtered by mission."""
        return [
            g for g in self._goals.values()
            if g.is_runnable()
            and g.attempts < g.max_attempts
            and (not mission_id or g.mission_id == mission_id)
        ]

    # -------------------------------------------------------------------
    # Progress
    # -------------------------------------------------------------------

    def mark_attempt(self, goal_id: str):
        """Record an execution attempt on a goal."""
        goal = self._goals.get(goal_id)
        if goal:
            goal.attempts += 1
            goal.updated_at = time.time()
            if goal.status == GoalStatus.PENDING:
                goal.status = GoalStatus.ACTIVE

    def mark_completed(self, goal_id: str, reason: str = ""):
        """Mark a goal as completed."""
        goal = self._goals.get(goal_id)
        if goal and not goal.is_terminal():
            goal.status = GoalStatus.COMPLETED
            goal.completed_at = time.time()
            goal.updated_at = time.time()
            log.info("Goal completed: %s (%s)", goal_id, reason or goal.description[:50])

    def mark_failed(self, goal_id: str, reason: str = ""):
        """Mark a goal as failed."""
        goal = self._goals.get(goal_id)
        if goal and not goal.is_terminal():
            goal.status = GoalStatus.FAILED
            goal.updated_at = time.time()
            log.info("Goal failed: %s (%s)", goal_id, reason)

    def mark_blocked(self, goal_id: str, reason: str = ""):
        """Mark a goal as externally blocked."""
        goal = self._goals.get(goal_id)
        if goal and goal.status not in (GoalStatus.COMPLETED, GoalStatus.ARCHIVED):
            goal.status = GoalStatus.BLOCKED
            goal.updated_at = time.time()
            goal.metadata["block_reason"] = reason
            log.info("Goal blocked: %s (%s)", goal_id, reason)

    def unblock(self, goal_id: str):
        """Unblock a goal."""
        goal = self._goals.get(goal_id)
        if goal and goal.status == GoalStatus.BLOCKED:
            goal.status = GoalStatus.PENDING
            goal.updated_at = time.time()
            goal.metadata.pop("block_reason", None)

    # -------------------------------------------------------------------
    # Auto-completion checks
    # -------------------------------------------------------------------

    def should_auto_fail(self, goal_id: str) -> bool:
        """Check if a goal has exhausted all attempts."""
        goal = self._goals.get(goal_id)
        if not goal:
            return False
        return goal.attempts >= goal.max_attempts and goal.is_runnable()

    def check_repeated_failure(self, goal_id: str) -> bool:
        """Check if a goal is in a repeated failure pattern."""
        goal = self._goals.get(goal_id)
        if not goal:
            return False
        return goal.attempts >= 3 and goal.is_runnable()

    # -------------------------------------------------------------------
    # Task tracking
    # -------------------------------------------------------------------

    def next_task_for_goal(
        self, goal_id: str, tasks: dict[str, Task]
    ) -> Task | None:
        """Get the next runnable task for a goal.

        Deterministic ordering:
        - unblock tasks whose prerequisites are now complete
        - runnable active/pending tasks first
        - waiting tasks remain waiting until prerequisites clear
        - blocked tasks are not retried blindly
        """
        goal = self._goals.get(goal_id)
        if not goal:
            return None

        completed_ids = {
            t.id for t in tasks.values()
            if t.goal_id == goal_id and t.status == TaskStatus.COMPLETED
        }

        goal_tasks = [
            t for t in tasks.values()
            if t.goal_id == goal_id
        ]

        for task in goal_tasks:
            deps_ready = all(dep in completed_ids for dep in task.depends_on)
            if task.status == TaskStatus.WAITING and deps_ready:
                transition_task(task, TaskStatus.PENDING, "Dependencies satisfied")
            elif task.status in (TaskStatus.PENDING, TaskStatus.ACTIVE) and not deps_ready:
                transition_task(task, TaskStatus.WAITING, "Waiting for dependencies")

        candidates = [
            t for t in goal_tasks
            if t.is_runnable() and all(dep in completed_ids for dep in t.depends_on)
        ]

        if not candidates:
            return None

        candidates.sort(key=lambda t: (t.priority, t.id))
        return candidates[0]

    def activate_next_task(
        self, goal_id: str, tasks: dict[str, Task]
    ) -> Task | None:
        """Activate the next pending task for a goal."""
        task = self.next_task_for_goal(goal_id, tasks)
        if task:
            transition_task(task, TaskStatus.ACTIVE, "Activated for execution")
        return task

    def complete_task(
        self, task: Task, tasks: dict[str, Task]
    ):
        """Mark a task complete and handle repeat logic."""
        import time as _time
        transition_task(task, TaskStatus.COMPLETED, "Execution succeeded")

        # Handle repeating tasks — reset to PENDING but enforce cooldown interval
        if task.repeat != TaskRepeat.ONCE:
            _REPEAT_INTERVALS = {
                TaskRepeat.DAILY:   86400.0,
                TaskRepeat.WEEKLY:  7 * 86400.0,
                TaskRepeat.MONTHLY: 30 * 86400.0,
            }
            interval = _REPEAT_INTERVALS.get(task.repeat, 86400.0)
            now = _time.time()
            task.status = TaskStatus.PENDING
            task.completed_at = now
            task.next_run_after = now + interval
            task.attempts = 0
            task.waiting_reason = ""
            task.blocker_reason = ""
            log.info(
                "Repeat task %s (%s) scheduled next run in %.0fh",
                task.id, task.repeat.value, interval / 3600,
            )

    def mark_task_failed(self, task: Task, reason: str = ""):
        task.error = reason
        transition_task(task, TaskStatus.FAILED, reason or "Execution failed")

    def mark_task_waiting(self, task: Task, reason: str):
        transition_task(task, TaskStatus.WAITING, reason)

    def mark_task_blocked_external(self, task: Task, reason: str):
        transition_task(task, TaskStatus.BLOCKED_EXTERNAL, reason)

    def mark_task_blocked_approval(self, task: Task, reason: str):
        transition_task(task, TaskStatus.BLOCKED_APPROVAL, reason)

    def mark_task_aborted(self, task: Task, reason: str):
        task.error = reason
        transition_task(task, TaskStatus.ABORTED, reason or "Execution aborted")

    def all_tasks_done(self, goal_id: str, tasks: dict[str, Task]) -> bool:
        """Check if all tasks for a goal are complete."""
        goal_tasks = [t for t in tasks.values() if t.goal_id == goal_id]
        if not goal_tasks:
            return False  # No tasks = not auto-completable
        return all(
            t.status == TaskStatus.COMPLETED
            for t in goal_tasks
            if t.repeat == TaskRepeat.ONCE
        )

    # -------------------------------------------------------------------
    # Archival
    # -------------------------------------------------------------------

    def archive_stale(self, max_age_hours: float = 24.0) -> list[str]:
        """Archive completed/failed goals older than threshold.

        Never archives immortal goals.
        Returns list of archived goal IDs.
        """
        cutoff = time.time() - (max_age_hours * 3600)
        archived = []

        for goal in self._goals.values():
            if goal.immortal:
                continue
            if goal.is_terminal() and goal.updated_at < cutoff:
                goal.status = GoalStatus.ARCHIVED
                archived.append(goal.id)

        if archived:
            log.info("Archived %d stale goals", len(archived))
        return archived

    # -------------------------------------------------------------------
    # Memory-assisted context
    # -------------------------------------------------------------------

    def recall_goal_context(self, goal: Goal) -> list[dict]:
        """Recall memory records related to a goal.

        Used to provide context to the specialist before execution.
        """
        try:
            from ..memory.memory_api import get_memory
            memory = get_memory()

            # Recall by goal description
            records = memory.recall(
                goal.description,
                tags=["outcome", "finding", "failure"],
                limit=5,
            )

            return [
                {
                    "content": r.content[:200],
                    "type": r.record_type,
                    "score": round(r.score, 2),
                }
                for r in records
            ]
        except Exception as e:
            log.debug("Memory recall failed: %s", e)
            return []

    def store_outcome(
        self,
        goal: Goal,
        success: bool,
        summary: str,
        evidence: dict[str, Any] | None = None,
    ):
        """Store goal outcome in memory for future recall."""
        try:
            from ..memory.memory_api import get_memory, MemoryClass
            memory = get_memory()

            record_type = "outcome" if success else "failure"
            tags = [record_type, "goal_outcome"]

            memory.store(
                content=f"[{record_type.upper()}] {goal.description[:100]}: {summary}",
                tags=tags,
                metadata={
                    "goal_id": goal.id,
                    "mission_id": goal.mission_id,
                    "success": success,
                    "attempts": goal.attempts,
                    **(evidence or {}),
                },
                memory_class=MemoryClass.OUTCOME,
                # Append-log outcome (no in-memory guard on this path) — collapse
                # identical repeats at the backend instead of accumulating them.
                deduplicate=True,
            )
        except Exception as e:
            log.debug("Failed to store outcome: %s", e)

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------

    def summary(self, mission_id: str = "") -> dict[str, Any]:
        goals = self._goals.values()
        if mission_id:
            goals = [g for g in goals if g.mission_id == mission_id]
        else:
            goals = list(goals)

        return {
            "total": len(goals),
            "by_status": {
                s.value: sum(1 for g in goals if g.status == s)
                for s in GoalStatus
            },
            "blocked": [
                {"id": g.id, "reason": g.metadata.get("block_reason", "")}
                for g in goals if g.status == GoalStatus.BLOCKED
            ],
        }
