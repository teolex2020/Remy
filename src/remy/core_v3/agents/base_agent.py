"""
Base Specialist Agent for Remy v3.

Defines the contract that all specialist agents must implement.
Each specialist receives a task instruction, executes it using
scoped tools, and returns a structured result.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..execution.execution_runtime import ExecutionResult, ExecutionStatus

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent context (what the specialist receives)
# ---------------------------------------------------------------------------

@dataclass
class AgentContext:
    """Context passed to a specialist agent before execution."""
    instruction: str = ""          # What to do
    mission_id: str = ""
    goal_id: str = ""
    goal_description: str = ""
    plan_step_id: str = ""

    # Constraints
    step_budget: int = 10          # Max iterations
    timeout_sec: int = 120
    tools_allowed: tuple[str, ...] = ()
    guardrails: tuple[str, ...] = ()
    approval_mode: str = "none"

    # Context from memory
    memory_context: list[dict] = field(default_factory=list)
    past_outcomes: str = ""        # Summary of recent execution history
    policy_hints: list[dict] = field(default_factory=list)

    # Budget awareness
    budget_remaining_usd: float = 1.0
    use_cheap_model: bool = False
    preferred_model: str = ""
    avoid_models: tuple[str, ...] = ()

    # V2 compat (goal dict for dispatch_worker)
    v2_goal_dict: dict | None = None


# ---------------------------------------------------------------------------
# Agent output
# ---------------------------------------------------------------------------

@dataclass
class AgentOutput:
    """Structured output from a specialist agent."""
    status: str = "failure"        # success, partial, failure, blocked, timeout
    response: str = ""             # Human-readable summary
    findings: list[dict] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    # Execution stats
    tool_calls: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    steps_used: int = 0

    # For pipeline
    session_log: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)

    # Recommendations
    next_action: str = ""          # Suggestion for next step
    needs_replan: bool = False
    blocker: str = ""              # If blocked, describe why

    @property
    def is_positive(self) -> bool:
        return self.status in ("success", "partial")

    def to_execution_result(self) -> ExecutionResult:
        """Convert to ExecutionResult for evaluation engine."""
        try:
            from remy.core.model_trace import extract_model_runtime
            model, fallback_used = extract_model_runtime(self.session_log)
        except Exception:
            model, fallback_used = "", False

        status_map = {
            "success": ExecutionStatus.SUCCESS,
            "partial": ExecutionStatus.PARTIAL,
            "failure": ExecutionStatus.FAILURE,
            "blocked": ExecutionStatus.BLOCKED,
            "timeout": ExecutionStatus.TIMEOUT,
        }
        return ExecutionResult(
            status=status_map.get(self.status, ExecutionStatus.FAILURE),
            response=self.response,
            evidence=self.evidence,
            tool_calls=self.tool_calls,
            tokens_used=self.tokens_used,
            cost_usd=self.cost_usd,
            duration_ms=self.duration_ms,
            model=model,
            fallback_used=fallback_used,
            session_log=self.session_log,
            history=self.history,
        )


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------

class BaseSpecialistAgent(ABC):
    """Abstract base class for specialist agents.

    Subclasses implement execute() with domain-specific logic.
    The base class provides common lifecycle management.
    """

    agent_id: str = "base"
    agent_label: str = "Base Agent"

    @abstractmethod
    async def execute(self, ctx: AgentContext) -> AgentOutput:
        """Execute the task described in context.

        Must be implemented by each specialist.
        """
        ...

    async def run(self, ctx: AgentContext) -> AgentOutput:
        """Run the agent with timeout and error handling.

        This is the entry point called by the delegation engine.
        """
        start = time.time()
        try:
            output = await asyncio.wait_for(
                self.execute(ctx),
                timeout=ctx.timeout_sec,
            )
            output.duration_ms = int((time.time() - start) * 1000)
            return output

        except asyncio.TimeoutError:
            log.warning("%s timed out after %ds", self.agent_id, ctx.timeout_sec)
            return AgentOutput(
                status="timeout",
                response=f"Agent {self.agent_id} timed out after {ctx.timeout_sec}s",
                duration_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:
            log.exception("%s execution error: %s", self.agent_id, e)
            return AgentOutput(
                status="failure",
                response=f"Agent {self.agent_id} error: {e}",
                duration_ms=int((time.time() - start) * 1000),
            )

    def can_handle(self, ctx: AgentContext) -> bool:
        """Check if this agent can handle the given context.

        Override for domain-specific routing logic.
        """
        return True

    def __repr__(self):
        return f"<{self.agent_label} ({self.agent_id})>"
