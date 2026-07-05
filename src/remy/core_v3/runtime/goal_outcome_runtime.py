"""
Goal Outcome Runtime for Remy v3.

Owns goal-level completion/failure transitions and normalized outcome
storage so OutcomeRuntime does not mutate goal state directly.
"""

from __future__ import annotations

from ..missions.goal_tracker import GoalTracker
from ..missions.mission_models import Goal


class GoalOutcomeRuntime:
    """Apply post-evaluation goal-level state transitions."""

    def __init__(self, goal_tracker: GoalTracker, tasks, persistence_runtime=None):
        self.goal_tracker = goal_tracker
        self.tasks = tasks
        self.persistence_runtime = persistence_runtime

    def complete(self, *, goal: Goal | None, summary: str) -> None:
        if goal is None:
            return
        self.goal_tracker.mark_completed(goal.id, summary)
        self._store_outcome(goal, success=True, summary=summary)

    def complete_if_all_tasks_done(self, *, goal: Goal | None) -> bool:
        if goal is None:
            return False
        if not self.goal_tracker.all_tasks_done(goal.id, self.tasks):
            return False
        self.complete(goal=goal, summary="All goal tasks completed")
        return True

    def fail(self, *, goal: Goal | None, summary: str) -> None:
        if goal is None:
            return
        self.goal_tracker.mark_failed(goal.id, summary)
        self._store_outcome(goal, success=False, summary=summary)

    def _store_outcome(self, goal: Goal, *, success: bool, summary: str) -> None:
        if self.persistence_runtime is not None:
            self.persistence_runtime.store_goal_outcome(
                goal,
                success=success,
                summary=summary,
            )
            return
        self.goal_tracker.store_outcome(goal, success, summary)
