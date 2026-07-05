"""
Plan builder runtime for Remy v3.

Owns construction of plan steps from mission tasks and fallback single-step
plans so this logic no longer lives inside ChiefAgent or MissionRuntime.
"""

from __future__ import annotations

from ..planning.plan_models import PlanStep


class PlanBuilderRuntime:
    """Build plan steps for mission execution."""

    def __init__(self, specialist_inference_runtime):
        self.specialist_inference_runtime = specialist_inference_runtime

    def task_specialist(self, task) -> str:
        template = str(getattr(task, "metadata", {}).get("goal_template", "") or "")
        return template or self.specialist_inference_runtime.infer(task.action)

    def steps_from_tasks(self, tasks) -> list[PlanStep]:
        steps = []
        for task in tasks:
            specialist = self.task_specialist(task)
            steps.append(
                PlanStep(
                    id=f"step_{task.id}",
                    description=task.done_when or task.action[:80],
                    instruction=task.action,
                    specialist=specialist,
                    completion_criteria=task.done_when,
                    expected_evidence=task.done_when,
                    depends_on=[f"step_{dep}" for dep in task.depends_on],
                )
            )
        return steps

    def fallback_steps(self, *, mission, goal) -> list[PlanStep]:
        instruction = goal.description if goal else mission.description
        specialist = (
            goal.pack_template
            if goal and goal.pack_template
            else self.specialist_inference_runtime.infer(instruction)
        )
        return [PlanStep(instruction=instruction, specialist=specialist)]
