"""
Decision Runtime for Remy v3.

Normalizes evaluation + replanning into a structured post-execution decision
contract before any runtime state is mutated.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..evaluation.evaluation_engine import EvalVerdict
from ..planning.replan_engine import ReplanAction


@dataclass
class OutcomeDecision:
    phase: str = "pause"
    reason: str = ""
    next_action: str = ""
    task_completed: bool = False
    mission_completed: bool = False
    replan_action: str = ""
    replan_decision: object | None = None


class DecisionRuntime:
    """Build deterministic post-execution decisions from eval + plan state."""

    def __init__(self, replanner, mission_outcome_runtime):
        self.replanner = replanner
        self.mission_outcome_runtime = mission_outcome_runtime

    def decide(
        self,
        *,
        mission,
        plan,
        step,
        task=None,
        eval_result,
    ) -> OutcomeDecision:
        success = eval_result.verdict in (EvalVerdict.SUCCESS, EvalVerdict.PARTIAL)
        task_completed = success and (
            eval_result.verdict == EvalVerdict.SUCCESS or not eval_result.should_continue
        )

        if task_completed:
            mission_completed = self.mission_outcome_runtime.should_complete(
                mission=mission,
                plan=plan,
                current_task=task,
            )
            return OutcomeDecision(
                phase="success",
                reason="all_steps_complete" if mission_completed else (eval_result.reason or ""),
                task_completed=True,
                mission_completed=mission_completed,
            )

        if success:
            return OutcomeDecision(
                phase="partial",
                reason=eval_result.reason or "Partial evidence collected",
            )

        replan_decision = self.replanner.decide(plan, step, eval_result)
        return OutcomeDecision(
            phase="failure",
            reason=replan_decision.reason,
            replan_action=replan_decision.action,
            replan_decision=replan_decision,
        )

    @staticmethod
    def is_retry_like(decision: OutcomeDecision) -> bool:
        return decision.replan_action in (ReplanAction.RETRY, ReplanAction.FALLBACK)
