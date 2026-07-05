"""
Completion Runtime for Remy v3.

Applies success-side and partial-side state transitions after a structured
post-execution decision has been produced.
"""

from __future__ import annotations

from ..evaluation.evaluation_engine import EvalVerdict
from .result_runtime import ResultRuntime


class CompletionRuntime:
    """Apply success and partial outcome state changes."""

    def __init__(
        self,
        mission_runtime,
        plan_state_runtime,
        task_outcome_runtime,
        goal_outcome_runtime,
        mission_outcome_runtime,
        result_runtime=None,
    ):
        self.mission_runtime = mission_runtime
        self.plan_state_runtime = plan_state_runtime
        self.task_outcome_runtime = task_outcome_runtime
        self.goal_outcome_runtime = goal_outcome_runtime
        self.mission_outcome_runtime = mission_outcome_runtime
        self.result_runtime = result_runtime or ResultRuntime()

    def apply_success(
        self,
        *,
        mission,
        goal,
        task,
        plan,
        step,
        exec_result,
        eval_result,
        decision,
        outcome=None,
    ):
        self.plan_state_runtime.complete_step(
            plan=plan,
            step=step,
            result=exec_result.evidence,
        )

        self.task_outcome_runtime.complete(task=task)
        if goal and eval_result.verdict == EvalVerdict.SUCCESS and not eval_result.should_continue:
            self.goal_outcome_runtime.complete(goal=goal, summary=eval_result.reason)
        elif goal and task:
            self.goal_outcome_runtime.complete_if_all_tasks_done(goal=goal)

        if decision.mission_completed:
            self.mission_outcome_runtime.complete(mission=mission, reason="All steps done")
            return self._finalize(
                outcome,
                self.result_runtime.complete(decision.reason or "all_steps_complete"),
            )

        next_task = self.mission_runtime.select_task_for_mission(mission)
        return self._finalize(
            outcome,
            self.result_runtime.execute_step(
                next_action=next_task.action[:80] if next_task else "",
            ),
        )

    def apply_partial(
        self,
        *,
        plan,
        task,
        step,
        decision,
        outcome=None,
    ):
        self.task_outcome_runtime.partial_continue(task=task)
        self.plan_state_runtime.reset_step_for_retry(plan=plan, step=step)
        return self._finalize(
            outcome,
            self.result_runtime.execute_step(
                reason=decision.reason or "Partial evidence collected",
            ),
        )

    @staticmethod
    def _finalize(outcome, result):
        if outcome is not None:
            outcome.decision = result.decision
            outcome.reason = result.reason
            outcome.next_action = result.next_action
            return outcome
        return result
