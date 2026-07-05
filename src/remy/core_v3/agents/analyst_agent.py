"""
Analyst Specialist Agent for Remy v3.

Handles: pattern analysis, cross-source synthesis,
confidence scoring, recommendation formation.

Phase 3: Delegates to v2 generic worker via orchestrator.
"""

from __future__ import annotations

import logging

from .base_agent import BaseSpecialistAgent, AgentContext, AgentOutput

log = logging.getLogger(__name__)


class AnalystAgent(BaseSpecialistAgent):
    """Analyst specialist — synthesizes information and produces recommendations."""

    agent_id = "analyst"
    agent_label = "Analyst"

    async def execute(self, ctx: AgentContext) -> AgentOutput:
        """Execute analysis task.

        Phase 3: Routes to v2 generic worker via orchestrator.
        """
        goal_dict = ctx.v2_goal_dict or self._build_goal_dict(ctx)

        try:
            from remy.core.orchestrator import dispatch_worker

            response_text, history, session_log, worker_result = (
                await dispatch_worker(
                    goal=goal_dict,
                    decision_prompt=ctx.instruction,
                    session_id=f"v3_analyst_{ctx.mission_id}",
                    session_log=[],
                    history=None,
                )
            )

            tool_calls = getattr(worker_result, "tool_calls", 0) if worker_result else 0
            w_status = getattr(worker_result, "status", "") if worker_result else ""

            status = self._map_status(w_status, tool_calls, response_text or "")

            return AgentOutput(
                status=status,
                response=response_text or "",
                tool_calls=tool_calls,
                session_log=session_log,
                history=history,
                findings=self._extract_analysis(response_text or ""),
            )

        except ImportError:
            return AgentOutput(
                status="failure",
                response="Analyst worker not available (v2 orchestrator missing)",
            )
        except Exception as e:
            log.exception("Analyst error: %s", e)
            return AgentOutput(status="failure", response=f"Analysis error: {e}")

    def can_handle(self, ctx: AgentContext) -> bool:
        import re
        words = set(re.findall(r'\b\w+\b', ctx.instruction.lower()))
        return bool(words & {
            "analyze", "synthesize", "compare", "summarize",
            "recommend", "evaluate", "assess", "report",
        })

    def _build_goal_dict(self, ctx: AgentContext) -> dict:
        return {
            "content": ctx.instruction,
            "metadata": {
                "status": "active",
                "priority": "3",
                "goal_template": "general",
                "mission_id": ctx.mission_id,
            },
            "tags": ["goal", "v3_step", "analysis"],
        }

    def _map_status(self, worker_status: str, tool_calls: int, response: str) -> str:
        if worker_status in ("success", "completed"):
            return "success"
        # Analyst can succeed with just a good response (no tools needed)
        if response and len(response) > 100:
            return "success"
        if tool_calls > 0:
            return "partial"
        return "failure"

    def _extract_analysis(self, response: str) -> list[dict]:
        """Extract structured analysis points from response."""
        if not response:
            return []
        # Simple extraction: each paragraph as a finding
        paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
        return [
            {"content": p[:300], "type": "analysis"}
            for p in paragraphs[:5]
        ]
