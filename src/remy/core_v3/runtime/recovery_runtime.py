"""
Recovery Runtime for Remy v3.

Applies failure-side recovery policy after a structured outcome decision has
already been made.
"""

from __future__ import annotations

from ..planning.replan_engine import ReplanAction
from .result_runtime import ResultRuntime


class RecoveryRuntime:
    """Apply retry/fallback/wait/escalate/abort/replan state changes."""

    def __init__(
        self,
        plan_state_runtime,
        task_outcome_runtime,
        goal_outcome_runtime,
        mission_outcome_runtime,
        result_runtime=None,
    ):
        self.plan_state_runtime = plan_state_runtime
        self.task_outcome_runtime = task_outcome_runtime
        self.goal_outcome_runtime = goal_outcome_runtime
        self.mission_outcome_runtime = mission_outcome_runtime
        self.result_runtime = result_runtime or ResultRuntime()

    def apply_failure(
        self,
        *,
        mission,
        goal,
        task,
        plan,
        step,
        decision,
        outcome=None,
    ):
        replan_decision = decision.replan_decision

        if decision.replan_action in (ReplanAction.RETRY, ReplanAction.FALLBACK):
            if replan_decision.action == ReplanAction.FALLBACK and replan_decision.modified_step is not None:
                self.plan_state_runtime.activate_fallback(
                    plan=plan,
                    failed_step=step,
                    fallback_step=replan_decision.modified_step,
                )
            else:
                self.plan_state_runtime.reset_step_for_retry(plan=plan, step=step)
            self.task_outcome_runtime.retry(task=task, reason=replan_decision.reason or "Retrying task")
            return self._finalize(outcome, self.result_runtime.execute_step(reason=replan_decision.reason))

        if replan_decision.action == ReplanAction.SKIP:
            self.plan_state_runtime.skip_step(plan=plan, step=step)
            self.task_outcome_runtime.skip(task=task, reason=replan_decision.reason or "Skipping task")
            return self._finalize(outcome, self.result_runtime.execute_step(reason=replan_decision.reason))

        if replan_decision.action == ReplanAction.WAIT:
            self.task_outcome_runtime.wait(task=task, reason=replan_decision.reason)
            return self._finalize(outcome, self.result_runtime.pause(replan_decision.reason))

        if replan_decision.action == ReplanAction.ESCALATE:
            self.task_outcome_runtime.block_external(task=task, reason=replan_decision.reason)
            self.mission_outcome_runtime.escalate(
                mission=mission,
                reason=replan_decision.reason,
            )
            return self._finalize(outcome, self.result_runtime.escalate(replan_decision.reason))

        if replan_decision.action == ReplanAction.ABORT:
            self.plan_state_runtime.fail_step(plan=plan, step=step)
            self.task_outcome_runtime.abort(task=task, reason=replan_decision.reason)
            self.goal_outcome_runtime.fail(goal=goal, summary=replan_decision.reason)
            return self._finalize(outcome, self.result_runtime.abort(replan_decision.reason))

        self.plan_state_runtime.fail_step(plan=plan, step=step)
        self.task_outcome_runtime.fail(task=task, reason=replan_decision.reason)
        return self._finalize(outcome, self.result_runtime.replan(replan_decision.reason))

    @staticmethod
    def _finalize(outcome, result):
        if outcome is not None:
            outcome.decision = result.decision
            outcome.reason = result.reason
            outcome.next_action = result.next_action
            return outcome
        return result
