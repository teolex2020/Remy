"""
Mission-level learning runtime for Remy v3.

Observes finished cycles, learns from outcomes, and promotes repeated
successful specialist paths into reusable playbooks.
"""

from __future__ import annotations

from typing import Any

from ..agents.specialist_registry import SpecialistProfile
from ..evaluation.evaluation_engine import EvalResult, EvalVerdict
from ..execution.execution_runtime import ExecutionResult, ExecutionStatus
from ..improvement.outcome_learner import OutcomeLearner
from ..improvement.playbook_engine import PlaybookEngine
from ..missions.mission_models import Goal, Mission, MissionMode, Task
from ..planning.plan_models import PlanStep


class LearningRuntime:
    """Post-cycle learning hook."""

    def __init__(self, learner: OutcomeLearner, playbooks: PlaybookEngine):
        self.learner = learner
        self.playbooks = playbooks

    def observe_cycle(
        self,
        *,
        mission: Mission,
        goal: Goal | None,
        task: Task | None,
        step: PlanStep,
        specialist: SpecialistProfile,
        exec_result: ExecutionResult,
        eval_result: EvalResult,
        decision: str,
    ) -> None:
        goal_description = goal.description if goal else mission.description
        tools_used = self._extract_tools(exec_result.session_log)
        status = self._normalize_status(exec_result, eval_result)

        self.learner.observe_outcome(
            goal_id=goal.id if goal else mission.id,
            goal_description=goal_description,
            specialist=specialist.id,
            status=status,
            tools_used=tools_used,
            blocker=self._blocker(eval_result, task),
            duration_ms=exec_result.duration_ms,
            cost_usd=exec_result.cost_usd,
            unsupported_observed_claims=getattr(exec_result, "unsupported_observed_claims", 0),
        )

        new_insights = self.learner.analyze()
        if new_insights:
            self.learner.store_insights_to_memory()

        if self._should_create_playbook(mission, specialist, eval_result, tools_used):
            self.playbooks.create_from_execution(
                name=f"{specialist.label} {self._infer_domain(mission, specialist).title()} path",
                goal_description=goal_description,
                domain=self._infer_domain(mission, specialist),
                steps=[
                    {
                        "action": step.instruction or (task.action if task else goal_description),
                        "specialist": specialist.id,
                        "tools": tools_used,
                        "outcome": eval_result.reason,
                        "decision": decision,
                    }
                ],
                cost_usd=exec_result.cost_usd,
                duration_ms=exec_result.duration_ms,
            )
            self.playbooks.store_to_memory()

    def _normalize_status(self, exec_result: ExecutionResult, eval_result: EvalResult) -> str:
        if eval_result.verdict == EvalVerdict.SUCCESS or exec_result.status == ExecutionStatus.SUCCESS:
            return "success"
        if eval_result.verdict == EvalVerdict.PARTIAL or exec_result.status == ExecutionStatus.PARTIAL:
            return "partial"
        if eval_result.verdict == EvalVerdict.BLOCKED or exec_result.status == ExecutionStatus.BLOCKED:
            return "blocked"
        return "failure"

    def _blocker(self, eval_result: EvalResult, task: Task | None) -> str:
        if eval_result.blocker_type is not None:
            return eval_result.blocker_type.value
        if task is not None and task.blocker_reason:
            return task.blocker_reason
        return ""

    def _extract_tools(self, session_log: list[dict[str, Any]] | None) -> list[str]:
        if not session_log:
            return []
        tools: list[str] = []
        for entry in session_log:
            tool_name = entry.get("tool") or entry.get("tool_name")
            if tool_name and tool_name not in tools:
                tools.append(tool_name)
        return tools

    def _should_create_playbook(
        self,
        mission: Mission,
        specialist: SpecialistProfile,
        eval_result: EvalResult,
        tools_used: list[str],
    ) -> bool:
        if specialist.id == "researcher":
            return False
        if eval_result.verdict != EvalVerdict.SUCCESS:
            return False
        if eval_result.confidence < 0.6:
            return False
        return bool(tools_used) or mission.mode != MissionMode.QUICK_TACTICAL

    def _infer_domain(self, mission: Mission, specialist: SpecialistProfile) -> str:
        if mission.mode == MissionMode.DEEP_RESEARCH:
            return "research"
        if mission.mode == MissionMode.CONTINUOUS_MONITORING:
            return "monitoring"
        if mission.mode == MissionMode.CAMPAIGN:
            return "campaign"
        if specialist.id == "analyst":
            return "analysis"
        if specialist.id == "executor":
            return "execution"
        return specialist.id
