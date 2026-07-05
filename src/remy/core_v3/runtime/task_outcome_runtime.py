"""
Task Outcome Runtime for Remy v3.

Applies post-evaluation task-level state transitions through GoalTracker so
task lifecycle policy stays outside OutcomeRuntime.
"""

from __future__ import annotations

from ..missions.goal_tracker import GoalTracker
from ..missions.mission_models import Task, TaskStatus


class TaskOutcomeRuntime:
    """Own task-level transitions after execution has been evaluated."""

    def __init__(self, goal_tracker: GoalTracker, tasks: dict[str, Task]):
        self.goal_tracker = goal_tracker
        self.tasks = tasks

    def complete(self, *, task: Task | None) -> None:
        if task is None:
            return
        self.goal_tracker.complete_task(task, self.tasks)

    def partial_continue(self, *, task: Task | None) -> None:
        if task is None:
            return
        task.status = TaskStatus.PENDING
        task.waiting_reason = ""
        task.blocker_reason = ""

    def retry(self, *, task: Task | None, reason: str = "") -> None:
        if task is None:
            return
        task.status = TaskStatus.PENDING
        task.error = self._humanize_reason(reason)
        task.waiting_reason = ""
        task.blocker_reason = ""

    def skip(self, *, task: Task | None, reason: str = "") -> None:
        if task is None:
            return
        task.error = self._humanize_reason(reason)
        task.status = TaskStatus.SKIPPED
        task.waiting_reason = ""
        task.blocker_reason = ""

    def wait(self, *, task: Task | None, reason: str = "") -> None:
        if task is None:
            return
        self.goal_tracker.mark_task_waiting(task, self._humanize_reason(reason))

    def block_external(self, *, task: Task | None, reason: str = "") -> None:
        if task is None:
            return
        self.goal_tracker.mark_task_blocked_external(task, self._humanize_reason(reason))

    def abort(self, *, task: Task | None, reason: str = "") -> None:
        if task is None:
            return
        self.goal_tracker.mark_task_aborted(task, self._humanize_reason(reason))

    def fail(self, *, task: Task | None, reason: str = "") -> None:
        if task is None:
            return
        self.goal_tracker.mark_task_failed(task, self._humanize_reason(reason))

    def all_goal_tasks_done(self, *, goal_id: str) -> bool:
        return self.goal_tracker.all_tasks_done(goal_id, self.tasks)

    @staticmethod
    def _humanize_reason(reason: str) -> str:
        return (reason or "").replace("_", " ")
