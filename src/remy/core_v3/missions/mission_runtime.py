"""
Deterministic mission runtime for Remy v3.

This module owns mission/task orchestration decisions:
- which task is runnable now
- how a mission derives or reuses a goal container
- how plans are created from atomic tasks
- how the current step is resolved from runtime truth

It keeps these decisions out of ChiefAgent so v3 remains runtime-first.
"""

from __future__ import annotations

import time
from typing import Callable

from .goal_tracker import GoalTracker
from .mission_models import Goal, GoalStatus, Mission, Task, TaskStatus
from ..planning.plan_models import Plan, PlanStep


class MissionRuntime:
    """Task-first mission orchestration over persisted state."""

    def __init__(
        self,
        goals: dict[str, Goal],
        tasks: dict[str, Task],
        plans: dict[str, Plan],
        goal_tracker: GoalTracker,
        plan_loader: Callable[[str], Plan | None],
        plan_creator: Callable[[Mission, list[PlanStep] | None], Plan],
        plan_builder_runtime,
    ):
        self._goals = goals
        self._tasks = tasks
        self._plans = plans
        self._goal_tracker = goal_tracker
        self._plan_loader = plan_loader
        self._plan_creator = plan_creator
        self.plan_builder_runtime = plan_builder_runtime

    def mission_tasks(self, mission_id: str) -> list[Task]:
        tasks = [task for task in self._tasks.values() if task.mission_id == mission_id]
        tasks.sort(key=lambda task: (task.priority, task.id))
        return tasks

    @property
    def tasks(self) -> dict[str, Task]:
        return self._tasks

    def mission_tasks_complete(self, mission_id: str) -> bool:
        tasks = self.mission_tasks(mission_id)
        if not tasks:
            return False
        for task in tasks:
            if task.status in (
                TaskStatus.PENDING,
                TaskStatus.ACTIVE,
                TaskStatus.RUNNING,
                TaskStatus.WAITING,
                TaskStatus.BLOCKED,
                TaskStatus.BLOCKED_EXTERNAL,
                TaskStatus.BLOCKED_APPROVAL,
            ):
                return False
        return True

    def select_task_for_mission(self, mission: Mission) -> Task | None:
        tasks = self.mission_tasks(mission.id)
        if not tasks:
            return None

        candidates: list[tuple[int, int, float, Task]] = []
        grouped_goal_ids = {task.goal_id for task in tasks if task.goal_id}
        for goal_id in grouped_goal_ids:
            goal = self._goals.get(goal_id)
            if goal is None or not goal.is_runnable() or goal.attempts >= goal.max_attempts:
                continue
            task = self._goal_tracker.next_task_for_goal(goal_id, self._tasks)
            if task is not None:
                candidates.append((goal.priority, task.priority, goal.created_at, task))

        for task in tasks:
            if task.goal_id:
                continue
            if task.is_runnable():
                if all(
                    self._tasks.get(dep) and self._tasks[dep].status == TaskStatus.COMPLETED
                    for dep in task.depends_on
                ):
                    candidates.append((mission.priority, task.priority, mission.created_at, task))
                else:
                    self._goal_tracker.mark_task_waiting(task, "Waiting for dependencies")

        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3].id))
        return candidates[0][3]

    def resolve_goal_for_cycle(self, mission: Mission, task: Task | None) -> Goal | None:
        if task and task.goal_id and task.goal_id in self._goals:
            return self._goals[task.goal_id]

        goal = self._goal_tracker.select_top_goal(mission)
        if goal is not None:
            return goal
        if task is None:
            return None

        synthetic_goal = Goal(
            description=mission.description,
            mission_id=mission.id,
            pack_template=self.task_specialist(task),
            status=GoalStatus.ACTIVE,
            created_at=time.time(),
            updated_at=time.time(),
        )
        self._goals[synthetic_goal.id] = synthetic_goal
        self._goal_tracker.bind_goals(self._goals)
        task.goal_id = synthetic_goal.id
        return synthetic_goal

    def ensure_plan_for_cycle(self, mission: Mission, goal: Goal | None) -> Plan:
        plan = self._plans.get(mission.id) or self._plan_loader(mission.id)
        if plan is not None:
            self._plans[mission.id] = plan
            return plan

        mission_tasks = self.mission_tasks(mission.id)
        if mission_tasks:
            steps = self.plan_builder_runtime.steps_from_tasks(mission_tasks)
            plan = self._plan_creator(mission, steps)
            self._plans[mission.id] = plan
            return plan

        plan = self._plan_creator(mission, self.plan_builder_runtime.fallback_steps(mission=mission, goal=goal))
        self._plans[mission.id] = plan
        return plan

    def select_step_for_cycle(self, plan: Plan, task: Task | None) -> PlanStep | None:
        if task is not None:
            expected_step_id = f"step_{task.id}"
            for step in plan.steps:
                if step.id == expected_step_id:
                    return step
        return plan.next_step()

    def task_specialist(self, task: Task) -> str:
        return self.plan_builder_runtime.task_specialist(task)
