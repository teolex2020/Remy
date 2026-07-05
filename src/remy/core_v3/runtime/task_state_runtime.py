"""
Task state runtime for Remy v3.

Owns deterministic task/step promotion into execution so execution gating does
not mutate state directly.
"""

from __future__ import annotations

from ..missions.mission_models import TaskStatus
from .state_machine import transition_task


class TaskStateRuntime:
    """State-policy helper for task and plan-step execution transitions."""

    def __init__(self, *, step_state_runtime=None):
        if step_state_runtime is None:
            from .step_state_runtime import StepStateRuntime

            step_state_runtime = StepStateRuntime()
        self.step_state_runtime = step_state_runtime

    def promote_for_execution(self, *, step, task=None):
        self.step_state_runtime.mark_running(step=step)

        if task is not None:
            task.attempts += 1
            if task.status == TaskStatus.PENDING:
                transition_task(task, TaskStatus.ACTIVE, "Selected for execution")
            transition_task(task, TaskStatus.RUNNING, "Delegated to specialist")
