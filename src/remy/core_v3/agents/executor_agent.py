"""
Executor Specialist Agent for Remy v3.

Handles: concrete step execution, browser operations, tool invocation,
signup flows, publishing, artifact collection.

Phase 3: Delegates to v2 browser_worker/generic worker via orchestrator.
"""

from __future__ import annotations

import logging

from .base_agent import BaseSpecialistAgent, AgentContext, AgentOutput

log = logging.getLogger(__name__)


class ExecutorAgent(BaseSpecialistAgent):
    """Executor specialist — carries out concrete actions."""

    agent_id = "executor"
    agent_label = "Executor"

    async def execute(self, ctx: AgentContext) -> AgentOutput:
        """Execute a concrete task.

        Phase 3: Routes to v2 dispatch_worker (browser or generic).
        """
        goal_dict = ctx.v2_goal_dict or self._build_goal_dict(ctx)

        try:
            from remy.core.orchestrator import dispatch_worker

            response_text, history, session_log, worker_result = (
                await dispatch_worker(
                    goal=goal_dict,
                    decision_prompt=ctx.instruction,
                    session_id=f"v3_exec_{ctx.mission_id}",
                    session_log=[],
                    history=None,
                )
            )

            tool_calls = getattr(worker_result, "tool_calls", 0) if worker_result else 0
            w_status = getattr(worker_result, "status", "") if worker_result else ""
            evidence = getattr(worker_result, "evidence", {}) if worker_result else {}

            # Detect blockers
            blocker = self._detect_blocker(session_log, response_text or "")

            status = "blocked" if blocker else self._map_status(w_status, tool_calls)

            return AgentOutput(
                status=status,
                response=response_text or "",
                evidence=evidence,
                tool_calls=tool_calls,
                session_log=session_log,
                history=history,
                blocker=blocker,
                artifacts=self._extract_artifacts(session_log),
            )

        except ImportError:
            return AgentOutput(
                status="failure",
                response="Executor worker not available (v2 orchestrator missing)",
            )
        except Exception as e:
            log.exception("Executor error: %s", e)
            return AgentOutput(status="failure", response=f"Execution error: {e}")

    def can_handle(self, ctx: AgentContext) -> bool:
        import re
        words = set(re.findall(r'\b\w+\b', ctx.instruction.lower()))
        return bool(words & {
            "browse", "signup", "register", "navigate", "publish",
            "post", "click", "fill", "submit",
        })

    def _build_goal_dict(self, ctx: AgentContext) -> dict:
        # Infer pack template from instruction
        instruction_lower = ctx.instruction.lower()
        if any(w in instruction_lower for w in ("signup", "register", "account")):
            template = "signup_operator"
        elif any(w in instruction_lower for w in ("publish", "post", "draft")):
            template = "publisher"
        else:
            template = "general"

        return {
            "content": ctx.instruction,
            "metadata": {
                "status": "active",
                "priority": "3",
                "goal_template": template,
                "mission_id": ctx.mission_id,
            },
            "tags": ["goal", "v3_step", "execution"],
        }

    def _map_status(self, worker_status: str, tool_calls: int) -> str:
        if worker_status in ("success", "completed"):
            return "success"
        if worker_status == "blocked":
            return "blocked"
        if tool_calls > 0:
            return "partial"
        return "failure"

    def _detect_blocker(self, session_log: list[dict], response: str) -> str:
        """Detect hard blockers from execution."""
        response_lower = response.lower()
        blockers = {
            "captcha": "CAPTCHA detected",
            "email verification": "Email verification required",
            "payment required": "Payment required",
            "kyc": "KYC verification required",
            "rate limit": "Rate limited",
        }
        for keyword, description in blockers.items():
            if keyword in response_lower:
                return description
        return ""

    def _extract_artifacts(self, session_log: list[dict]) -> list[dict]:
        """Extract execution artifacts (URLs, screenshots, etc.)."""
        artifacts = []
        for entry in (session_log or []):
            tool = entry.get("tool", "")
            result = entry.get("result", {})
            if tool == "browse_page" and isinstance(result, dict):
                url = result.get("url", "")
                if url:
                    artifacts.append({"type": "url", "value": url})
            elif tool == "store" and isinstance(result, dict):
                rid = result.get("id", "")
                if rid:
                    artifacts.append({"type": "record", "value": rid})
        return artifacts
