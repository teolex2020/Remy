"""
Shared persistence runtime for Remy v3.

Coordinates state persistence and normalized memory writes so runtime truth is
not scattered across multiple subsystems.
"""

from __future__ import annotations

from ..memory.memory_api import MemoryClass, get_memory
from ..memory.record_models import outcome_record


class PersistenceRuntime:
    """Centralized persistence coordinator for v3."""

    def __init__(self, mission_persistence, plan_persistence, memory=None):
        self.mission_persistence = mission_persistence
        self.plan_persistence = plan_persistence
        self.memory = memory or get_memory()
        self._goal_outcome_keys: set[tuple[str, bool, str]] = set()
        self._research_summary_keys: set[tuple[str, str]] = set()

    def save_runtime_state(self, missions, goals, tasks, plans) -> None:
        self.mission_persistence.save(missions, goals, tasks)
        for plan in plans.values():
            self.plan_persistence.save(plan)

    def save_plan(self, plan) -> None:
        self.plan_persistence.save(plan)

    def store_goal_outcome(
        self,
        goal,
        *,
        success: bool,
        summary: str,
        evidence: dict | None = None,
    ) -> None:
        key = (goal.id, success, summary.strip().lower())
        if key in self._goal_outcome_keys:
            return

        record_type = "outcome" if success else "failure"
        tags = [record_type, "goal_outcome"]

        self.memory.store(
            content=f"[{record_type.upper()}] {goal.description[:100]}: {summary}",
            tags=tags,
            metadata={
                "goal_id": goal.id,
                "mission_id": goal.mission_id,
                "success": success,
                "attempts": goal.attempts,
                **(evidence or {}),
            },
            memory_class=MemoryClass.OUTCOME,
            # Append-log outcome: identical content across process restarts is a
            # true repeat (the in-memory key set above resets on restart), so let
            # the backend collapse it instead of accumulating duplicates.
            deduplicate=True,
        )
        self._goal_outcome_keys.add(key)

    def store_research_summary(self, project) -> None:
        if not project.synthesis:
            return
        key = (project.id, project.synthesis.summary[:200].strip().lower())
        if key in self._research_summary_keys:
            return

        content, tags, meta, memory_class = outcome_record(
            f"[RESEARCH SESSION] {project.objective[:120]} -> {project.synthesis.summary[:280]}",
            mission_id=project.mission_id,
            extra_tags=["research-session"],
            extra_meta={
                "goal_id": project.goal_id,
                "research_project_id": project.id,
                "source_count": project.synthesis.source_count,
                "finding_count": project.synthesis.finding_count,
                "confidence": project.synthesis.confidence,
                "reused_playbook_id": project.reused_playbook_id,
            },
        )
        self.memory.store(
            content, tags=tags, metadata=meta,
            memory_class=memory_class, deduplicate=True,
        )
        self._research_summary_keys.add(key)
