"""
Outcome Runtime for Remy v3.

Applies evaluated execution results to task, plan, goal, and mission state.
This keeps runtime transitions deterministic and avoids re-implementing the
same stop/go mapping inside ChiefAgent.
"""

from __future__ import annotations

from ..evaluation.evaluation_engine import EvalVerdict
from ..missions.goal_tracker import GoalTracker
from ..missions.mission_models import Goal, Mission, Task
from ..missions.mission_runtime import MissionRuntime
from ..planning.plan_models import Plan, PlanStep
from ..planning.replan_engine import ReplanEngine
from .decision_runtime import OutcomeDecision
from .result_runtime import OutcomeResult


class OutcomeRuntime:
    """Deterministically map eval + replan into persisted runtime state."""

    def __init__(
        self,
        goal_tracker: GoalTracker,
        mission_runtime: MissionRuntime,
        replanner: ReplanEngine,
        decision_runtime,
        completion_runtime,
        recovery_runtime,
        plan_state_runtime,
        task_outcome_runtime,
        goal_outcome_runtime,
        mission_outcome_runtime,
        persistence_runtime=None,
    ):
        self.goal_tracker = goal_tracker
        self.mission_runtime = mission_runtime
        self.replanner = replanner
        self.decision_runtime = decision_runtime
        self.completion_runtime = completion_runtime
        self.recovery_runtime = recovery_runtime
        self.plan_state_runtime = plan_state_runtime
        self.task_outcome_runtime = task_outcome_runtime
        self.goal_outcome_runtime = goal_outcome_runtime
        self.mission_outcome_runtime = mission_outcome_runtime
        self.persistence_runtime = persistence_runtime

    def apply(
        self,
        *,
        mission: Mission,
        goal: Goal | None,
        task: Task | None,
        plan: Plan,
        step: PlanStep,
        exec_result,
        eval_result,
    ) -> OutcomeResult:
        decision = self.decision_runtime.decide(
            mission=mission,
            plan=plan,
            step=step,
            task=task,
            eval_result=eval_result,
        )

        self.plan_state_runtime.add_execution_cost(plan=plan, exec_result=exec_result)

        if decision.phase == "success":
            return self.completion_runtime.apply_success(
                mission=mission,
                goal=goal,
                task=task,
                plan=plan,
                step=step,
                exec_result=exec_result,
                eval_result=eval_result,
                decision=decision,
            )

        if decision.phase == "partial":
            return self.completion_runtime.apply_partial(
                plan=plan,
                task=task,
                step=step,
                decision=decision,
            )

        return self._apply_failure(
            mission=mission,
            goal=goal,
            task=task,
            plan=plan,
            step=step,
            decision=decision,
        )

    def _apply_failure(
        self,
        *,
        mission: Mission,
        goal: Goal | None,
        task: Task | None,
        plan: Plan,
        step: PlanStep,
        decision: OutcomeDecision,
    ):
        return self.recovery_runtime.apply_failure(
            mission=mission,
            goal=goal,
            task=task,
            plan=plan,
            step=step,
            decision=decision,
        )
