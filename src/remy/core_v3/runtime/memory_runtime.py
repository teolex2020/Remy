"""
Shared memory runtime for Remy v3.

Provides a single orchestration layer for:
- cycle-time context recall
- playbook reuse hints
- research context hydration
"""

from __future__ import annotations

from ..improvement.playbook_engine import get_playbook_engine
from ..memory.memory_api import get_memory
from ..missions.mission_models import Goal, Mission, MissionMode, Task


class MemoryRuntime:
    """Central memory-aware orchestration helper."""

    def __init__(self, memory=None, playbooks=None):
        self.memory = memory or get_memory()
        self.playbooks = playbooks or get_playbook_engine()

    def build_cycle_context(
        self,
        mission: Mission,
        goal: Goal | None = None,
        task: Task | None = None,
        limit: int = 6,
    ) -> list[dict]:
        """Recall compact context for the next executable unit."""
        query = self._build_query(mission, goal, task)
        recalled = self.memory.recall(
            query,
            tags=["outcome", "finding", "failure", "playbook"],
            limit=max(limit, 8),
        )

        context: list[dict] = []
        seen: set[str] = set()
        for record in recalled:
            snippet = record.content[:220].strip()
            if not snippet:
                continue
            norm = snippet.lower()
            if norm in seen:
                continue
            seen.add(norm)
            context.append(
                {
                    "content": snippet,
                    "type": record.record_type,
                    "score": round(record.score, 2),
                    "record_id": record.id,
                }
            )

        playbook = self.playbooks.match(query, domain=self._domain_for_mission(mission))
        if playbook is not None:
            playbook_steps = " -> ".join(
                step.action[:60] for step in playbook.steps[:3] if step.action
            )
            hint = f"[PLAYBOOK] {playbook.name}: {playbook_steps}"
            if hint.lower() not in seen:
                context.insert(
                    0,
                    {
                        "content": hint,
                        "type": "playbook",
                        "score": round(playbook.success_rate, 2),
                        "record_id": playbook.id,
                    },
                )

        return context[:limit]

    def hydrate_research_project(self, project):
        """Hydrate a research project with prior memory and reusable strategy."""
        recalled = self.memory.recall(
            project.objective,
            tags=["finding", "outcome", "playbook"],
            limit=8,
        )
        project.prior_context = [record.content[:220] for record in recalled[:4]]

        playbook = self.playbooks.match(project.objective, domain="research")
        if playbook:
            project.reused_playbook_id = playbook.id
            project.strategy_hints.extend(
                step.action[:120] for step in playbook.steps[:3] if step.action
            )

        if project.prior_context:
            project.strategy_hints.append(f"{project.objective} latest updates")

        deduped: list[str] = []
        seen: set[str] = set()
        for hint in project.strategy_hints:
            norm = hint.strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                deduped.append(hint.strip())
        project.strategy_hints = deduped[:5]
        return project

    def _build_query(self, mission: Mission, goal: Goal | None, task: Task | None) -> str:
        parts = []
        if goal is not None and goal.description:
            parts.append(goal.description)
        if task is not None and task.action:
            parts.append(task.action)
        if not parts:
            parts.append(mission.description)
        return " ".join(parts)[:400]

    def _domain_for_mission(self, mission: Mission) -> str:
        if mission.mode == MissionMode.DEEP_RESEARCH:
            return "research"
        if mission.mode == MissionMode.CONTINUOUS_MONITORING:
            return "monitoring"
        if mission.mode == MissionMode.CAMPAIGN:
            return "campaign"
        if mission.mode == MissionMode.SELF_IMPROVEMENT:
            return "improvement"
        return ""
