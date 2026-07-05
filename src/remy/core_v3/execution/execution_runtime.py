"""
Execution Runtime for Remy v3.

Responsible for tool invocation, worker lifecycle, retries,
timeout handling, and execution evidence collection.

Wraps v2 orchestrator.dispatch_worker() behind a typed contract.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------

class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial_progress"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"
    ERROR = "error"


@dataclass
class ExecutionResult:
    """Outcome of executing a plan step or task."""
    status: ExecutionStatus = ExecutionStatus.FAILURE
    response: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    tool_calls: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    specialist: str = ""
    model: str = ""
    fallback_used: bool = False
    error: str = ""
    unsupported_observed_claims: int = 0
    had_external_evidence: bool = False
    factuality_modified: bool = False
    factuality_report: Any = None

    # From v2 worker result
    session_log: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)

    @property
    def is_positive(self) -> bool:
        return self.status in (ExecutionStatus.SUCCESS, ExecutionStatus.PARTIAL)


# ---------------------------------------------------------------------------
# Execution Runtime
# ---------------------------------------------------------------------------

class ExecutionRuntime:
    """Manages execution of plan steps and tasks.

    In Phase 1, delegates to v2 orchestrator.dispatch_worker().
    Later phases will add direct specialist agent invocation.
    """

    def __init__(self):
        self._v2_orchestrator = None

    def _get_orchestrator(self):
        if self._v2_orchestrator is None:
            try:
                from remy.core import orchestrator
                self._v2_orchestrator = orchestrator
            except ImportError:
                log.error("v2 orchestrator not available")
        return self._v2_orchestrator

    async def execute_step(
        self,
        instruction: str,
        specialist: str = "",
        goal_dict: dict | None = None,
        session_id: str = "",
        timeout_sec: int = 120,
    ) -> ExecutionResult:
        """Execute a single plan step.

        Phase 1: Delegates to v2 dispatch_worker.
        """
        orch = self._get_orchestrator()
        if orch is None:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error="v2 orchestrator not available",
            )

        start = time.time()
        try:
            response_text, history, session_log, worker_result = (
                await asyncio.wait_for(
                    orch.dispatch_worker(
                        goal=goal_dict,
                        decision_prompt=instruction,
                        session_id=session_id or f"v3_{int(time.time())}",
                        session_log=[],
                        history=None,
                    ),
                    timeout=timeout_sec,
                )
            )

            duration_ms = int((time.time() - start) * 1000)

            # Extract status from worker result
            w_status = ""
            tool_count = 0
            if worker_result:
                w_status = getattr(worker_result, "status", "")
                tool_count = getattr(worker_result, "tool_calls", 0)

            status = self._derive_status(w_status, tool_count, response_text)
            model, fallback_used = self._extract_model_runtime(session_log)

            return ExecutionResult(
                status=status,
                response=response_text,
                evidence=getattr(worker_result, "evidence", {}) if worker_result else {},
                tool_calls=tool_count,
                duration_ms=duration_ms,
                specialist=specialist,
                model=model,
                fallback_used=fallback_used,
                session_log=session_log,
                history=history,
            )

        except asyncio.TimeoutError:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                error=f"Execution timed out after {timeout_sec}s",
                duration_ms=int((time.time() - start) * 1000),
                specialist=specialist,
            )
        except Exception as e:
            log.exception("Execution error: %s", e)
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=str(e),
                duration_ms=int((time.time() - start) * 1000),
                specialist=specialist,
            )

    def _derive_status(
        self, worker_status: str, tool_count: int, response: str
    ) -> ExecutionStatus:
        """Map v2 worker status to v3 ExecutionStatus."""
        if worker_status == "blocked":
            return ExecutionStatus.BLOCKED
        if worker_status == "timeout":
            if tool_count > 0:
                return ExecutionStatus.PARTIAL
            return ExecutionStatus.TIMEOUT
        if worker_status in ("success", "completed"):
            return ExecutionStatus.SUCCESS
        if tool_count > 0:
            return ExecutionStatus.PARTIAL
        if tool_count == 0 and response:
            return ExecutionStatus.FAILURE
        return ExecutionStatus.FAILURE

    def _extract_model_runtime(self, session_log: list[dict] | None) -> tuple[str, bool]:
        """Extract model routing facts from the v2/v3 session log."""
        try:
            from remy.core.model_trace import extract_model_runtime
            return extract_model_runtime(session_log)
        except Exception:
            return "", False
