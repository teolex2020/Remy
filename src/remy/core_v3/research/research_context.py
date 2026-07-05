"""
Research context runtime for Remy v3.

Loads reusable research context from memory and playbooks before a project
starts, and stores a compact research-session summary after completion.
"""

from __future__ import annotations

from ..improvement.playbook_engine import get_playbook_engine
from ..memory.memory_api import get_memory
from ..memory.record_models import outcome_record
from ..runtime.memory_runtime import MemoryRuntime


class ResearchContextRuntime:
    """Hydrate projects from memory and reusable playbooks."""

    def __init__(self, memory=None, playbooks=None, persistence_runtime=None):
        self.memory = memory or get_memory()
        self.playbooks = playbooks or get_playbook_engine()
        self.memory_runtime = MemoryRuntime(memory=self.memory, playbooks=self.playbooks)
        self.persistence_runtime = persistence_runtime

    def hydrate_project(self, project):
        return self.memory_runtime.hydrate_research_project(project)

    def store_project_summary(self, project):
        if self.persistence_runtime is not None:
            self.persistence_runtime.store_research_summary(project)
            return
        if not project.synthesis:
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
        self.memory.store(content, tags=tags, metadata=meta, memory_class=memory_class)
