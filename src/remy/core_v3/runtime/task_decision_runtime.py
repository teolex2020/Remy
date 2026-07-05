"""
Task decision runtime for Remy v3.

Owns deterministic task/step state mutations for deny and approval-block
outcomes so execution gating stays focused on governance decisions.
"""

from __future__ import annotations

from ..missions.mission_models import TaskStatus
from .state_machine import transition_task


class TaskDecisionRuntime:
    """Applies task/step state policy for non-execution gate outcomes."""

    def __init__(self, *, goal_tracker, step_state_runtime=None):
        if step_state_runtime is None:
            from .step_state_runtime import StepStateRuntime

            step_state_runtime = StepStateRuntime()
        self.goal_tracker = goal_tracker
        self.step_state_runtime = step_state_runtime

    def deny_execution(self, *, step, task, reason: str):
        self.step_state_runtime.mark_skipped(step=step)
        if task is not None:
            self.goal_tracker.mark_task_aborted(task, reason)

    def block_for_approval(self, *, task, approval_id: str):
        if task is not None:
            self.goal_tracker.mark_task_blocked_approval(
                task, f"Awaiting approval {approval_id}"
            )

    def block_for_consequence_scar(self, *, step, task, reason: str):
        self.step_state_runtime.mark_blocked(step=step)
        if task is not None:
            transition_task(task, TaskStatus.BLOCKED, reason)
