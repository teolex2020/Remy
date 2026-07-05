"""
Researcher Specialist Agent for Remy v3.

Handles: source discovery, query expansion, evidence gathering,
note extraction, synthesis handoff.

Phase 4: Uses v3 ResearchRuntime for structured pipeline,
falls back to v2 dispatch_worker if runtime unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

from .base_agent import BaseSpecialistAgent, AgentContext, AgentOutput

log = logging.getLogger(__name__)


class ResearcherAgent(BaseSpecialistAgent):
    """Research specialist — discovers, gathers, and synthesizes information."""

    agent_id = "researcher"
    agent_label = "Researcher"

    async def execute(self, ctx: AgentContext) -> AgentOutput:
        """Execute research task.

        Phase 4: Uses v3 ResearchRuntime for structured pipeline.
        Falls back to v2 dispatch_worker if needed.
        """
        # Try v3 structured research first
        try:
            return await self._execute_v3(ctx)
        except Exception as e:
            log.warning("v3 research runtime failed (%s), falling back to v2", e)

        # Fallback: v2 dispatch_worker
        return await self._execute_v2(ctx)

    async def _execute_v3(self, ctx: AgentContext) -> AgentOutput:
        """Run structured research via v3 ResearchRuntime."""
        from ..improvement.playbook_engine import get_playbook_engine
        from ..research.research_runtime import ResearchRuntime
        from ..research.research_policy import assess_project

        runtime = ResearchRuntime(
            persistence_runtime=getattr(self, "persistence_runtime", None)
        )
        project = await runtime.research(
            objective=ctx.instruction,
            mission_id=ctx.mission_id,
            goal_id=ctx.goal_id,
        )

        # Convert to AgentOutput
        synthesis = project.synthesis
        usable_sources = [source for source in project.sources if source.is_usable()]
        assessment = assess_project(project)
        contradictions_checked = assessment.contradictions_checked

        if synthesis and project.findings:
            finding_payload = [
                {
                    "content": finding.content[:300],
                    "category": finding.category,
                    "confidence": finding.confidence.value,
                    "source_ids": list(finding.source_ids),
                }
                for finding in project.findings
            ]
            artifact_payload = [
                {
                    "type": "source",
                    "url": source.url,
                    "title": source.title[:200],
                    "credibility": source.credibility.value,
                }
                for source in usable_sources
            ]
            self._store_research_memory(ctx, project, synthesis, finding_payload)
            if assessment.is_success:
                get_playbook_engine().create_from_execution(
                    name=f"Research: {ctx.instruction[:60]}",
                    goal_description=ctx.instruction,
                    domain="research",
                    steps=[
                        {"action": query, "specialist": "researcher"}
                        for query in project.queries_executed[:5]
                    ] or [{"action": ctx.instruction, "specialist": "researcher"}],
                )
            findings = [
                {
                    "content": finding.content[:300],
                    "category": finding.category,
                    "confidence": finding.confidence.value,
                }
                for finding in project.findings
            ]
            return AgentOutput(
                status=assessment.verdict,
                response=synthesis.summary,
                findings=findings,
                artifacts=artifact_payload,
                evidence={"synthesis": {
                    "key_findings": synthesis.key_findings,
                    "recommendations": synthesis.recommendations,
                    "confidence": synthesis.confidence,
                    "source_count": synthesis.source_count,
                    "finding_count": synthesis.finding_count,
                }, "findings": finding_payload, "artifacts": artifact_payload,
                    "contradictions_checked": contradictions_checked},
                next_action=(
                    synthesis.recommendations[0]
                    if synthesis.recommendations else
                    "Review research findings"
                ),
            )

        return AgentOutput(
            status="failure",
            response=assessment.reason or f"Research produced no findings for: {ctx.instruction[:100]}",
            next_action="Retry with different search strategy",
        )

    def _store_research_memory(self, ctx: AgentContext, project, synthesis, findings: list[dict]):
        """Store research evidence only when it is concrete enough to be reusable."""
        if not findings:
            return
        try:
            from ..memory.memory_api import get_memory
            from ..memory.record_models import finding_record, outcome_record

            memory = get_memory()
            for finding in findings[:10]:
                content, tags, meta, memory_class = finding_record(
                    finding["content"],
                    mission_id=ctx.mission_id,
                    confidence=synthesis.confidence,
                    extra_meta={
                        "goal_id": ctx.goal_id,
                        "category": finding["category"],
                        "source_ids": finding["source_ids"],
                        "research_project_id": project.id,
                    },
                )
                memory.store(content, tags=tags, metadata=meta, memory_class=memory_class)

            content, tags, meta, memory_class = outcome_record(
                f"[RESEARCH] {ctx.instruction[:120]} -> {synthesis.summary[:280]}",
                mission_id=ctx.mission_id,
                extra_meta={
                    "goal_id": ctx.goal_id,
                    "research_project_id": project.id,
                    "finding_count": len(findings),
                    "source_count": len([source for source in project.sources if source.is_usable()]),
                    "confidence": synthesis.confidence,
                },
            )
            # Append-log research summary (not the individual findings above,
            # which are distinct facts): collapse identical repeats.
            memory.store(content, tags=tags, metadata=meta, memory_class=memory_class, deduplicate=True)
        except Exception as exc:
            log.debug("Failed to store research memory: %s", exc)

    async def _execute_v2(self, ctx: AgentContext) -> AgentOutput:
        """Fallback: route to v2 research_worker via orchestrator."""
        goal_dict = ctx.v2_goal_dict or self._build_goal_dict(ctx)

        try:
            from remy.core.orchestrator import dispatch_worker

            response_text, history, session_log, worker_result = (
                await dispatch_worker(
                    goal=goal_dict,
                    decision_prompt=ctx.instruction,
                    session_id=f"v3_research_{ctx.mission_id}",
                    session_log=[],
                    history=None,
                )
            )

            tool_calls = getattr(worker_result, "tool_calls", 0) if worker_result else 0
            w_status = getattr(worker_result, "status", "") if worker_result else ""
            evidence = getattr(worker_result, "evidence", {}) if worker_result else {}

            status = self._map_status(w_status, tool_calls)
            findings = self._extract_findings(session_log)

            return AgentOutput(
                status=status,
                response=response_text or "",
                findings=findings,
                evidence=evidence,
                tool_calls=tool_calls,
                session_log=session_log,
                history=history,
                next_action=self._suggest_next(status, findings),
            )

        except ImportError:
            return AgentOutput(
                status="failure",
                response="Research worker not available (v2 orchestrator missing)",
            )
        except Exception as e:
            log.exception("Researcher v2 fallback error: %s", e)
            return AgentOutput(status="failure", response=f"Research error: {e}")

    def can_handle(self, ctx: AgentContext) -> bool:
        import re
        words = set(re.findall(r'\b\w+\b', ctx.instruction.lower()))
        return bool(words & {
            "research", "find", "search", "investigate",
            "discover", "study", "gather",
        })

    def _build_goal_dict(self, ctx: AgentContext) -> dict:
        return {
            "content": ctx.instruction,
            "metadata": {
                "status": "active",
                "priority": "3",
                "goal_template": "market_research",
                "mission_id": ctx.mission_id,
            },
            "tags": ["goal", "v3_step", "research"],
        }

    def _map_status(self, worker_status: str, tool_calls: int) -> str:
        if worker_status in ("success", "completed"):
            return "success"
        if worker_status == "blocked":
            return "blocked"
        if tool_calls > 0:
            return "partial"
        return "failure"

    def _extract_findings(self, session_log: list[dict]) -> list[dict]:
        """Extract research findings from session log."""
        findings = []
        for entry in (session_log or []):
            tool = entry.get("tool", "")
            if tool in ("add_research_finding", "store"):
                content = entry.get("result", {})
                if isinstance(content, dict):
                    findings.append(content)
                elif isinstance(content, str) and content:
                    findings.append({"content": content[:200]})
        return findings

    def _suggest_next(self, status: str, findings: list) -> str:
        if status == "success" and findings:
            return "Synthesize findings into report"
        if status == "partial":
            return "Continue research with deeper queries"
        return "Retry with different search strategy"
