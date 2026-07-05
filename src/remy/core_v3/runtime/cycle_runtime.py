"""
Cycle Runtime for Remy v3.

Resolves the beginning of a mission cycle deterministically:
- budget gate
- runnable task / goal selection
- plan and step resolution
- early terminal decisions before delegation
"""

from __future__ import annotations

from dataclasses import dataclass

from ..governance.budget_engine import BudgetAction, BudgetEngine
from ..missions.goal_tracker import GoalTracker
from ..missions.mission_models import Goal, Mission, MissionStatus, Task
from ..missions.mission_runtime import MissionRuntime
from ..planning.plan_models import Plan, PlanStep
from .state_machine import can_transition, transition


@dataclass
class CyclePreparation:
    proceed: bool = False
    decision: str = "pause"
    reason: str = ""
    goal: Goal | None = None
    task: Task | None = None
    plan: Plan | None = None
    step: PlanStep | None = None
    memory_context: list[dict] | None = None


class CycleRuntime:
    """Deterministic mission-cycle resolver."""

    def __init__(
        self,
        *,
        budget: BudgetEngine,
        goal_tracker: GoalTracker,
        mission_runtime: MissionRuntime,
        memory_runtime,
        activate_mission,
    ):
        self.budget = budget
        self.goal_tracker = goal_tracker
        self.mission_runtime = mission_runtime
        self.memory_runtime = memory_runtime
        self.activate_mission = activate_mission

    def prepare(self, mission: Mission) -> CyclePreparation:
        budget_action, budget_reason = self.budget.check_budget(
            estimated_cost_usd=0.05,
            mission_id=mission.id,
        )
        if budget_action == BudgetAction.DENY:
            return CyclePreparation(
                proceed=False,
                decision="pause",
                reason=budget_reason,
            )

        current_task = self.mission_runtime.select_task_for_mission(mission)
        goal = self.mission_runtime.resolve_goal_for_cycle(mission, current_task)

        if goal is None and current_task is None:
            if self.mission_runtime.mission_tasks_complete(mission.id):
                if can_transition(mission.status, MissionStatus.COMPLETED):
                    transition(mission, MissionStatus.COMPLETED, "All mission tasks done")
                return CyclePreparation(
                    proceed=False,
                    decision="complete",
                    reason="All mission tasks completed",
                )
            return CyclePreparation(
                proceed=False,
                decision="pause",
                reason="No runnable task or goal",
            )

        if goal:
            self.goal_tracker.mark_attempt(goal.id)
        memory_context = self.memory_runtime.build_cycle_context(
            mission,
            goal=goal,
            task=current_task,
        ) if goal or current_task else []

        plan = self.mission_runtime.ensure_plan_for_cycle(mission, goal)
        if mission.status in (MissionStatus.INTAKE, MissionStatus.PLANNING):
            self.activate_mission(mission)

        step = self.mission_runtime.select_step_for_cycle(plan, current_task)
        if step is None:
            if self.mission_runtime.mission_tasks_complete(mission.id):
                if can_transition(mission.status, MissionStatus.COMPLETED):
                    transition(mission, MissionStatus.COMPLETED, "Mission runtime complete")
                return CyclePreparation(
                    proceed=False,
                    decision="complete",
                    reason="No runnable tasks remain",
                    goal=goal,
                    task=current_task,
                    plan=plan,
                    memory_context=memory_context,
                )
            return CyclePreparation(
                proceed=False,
                decision="replan",
                reason="No executable step available",
                goal=goal,
                task=current_task,
                plan=plan,
                memory_context=memory_context,
            )

        return CyclePreparation(
            proceed=True,
            decision="execute_step",
            reason="ready_for_execution",
            goal=goal,
            task=current_task,
            plan=plan,
            step=step,
            memory_context=memory_context,
        )
