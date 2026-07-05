"""
Error runtime for Remy v3.

Normalizes execution-time exceptions into structured cycle results so the loop
does not perform ad hoc error mapping.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..agents.chief_agent import ChiefDecision, CycleResult


@dataclass
class ErrorHandlingResult:
    cycle_result: CycleResult
    consecutive_failures: int


class ErrorRuntime:
    """Convert execution failures into deterministic loop outcomes."""

    def handle_cycle_exception(
        self,
        *,
        mission_id: str,
        error: Exception,
        consecutive_failures: int,
    ) -> ErrorHandlingResult:
        return ErrorHandlingResult(
            cycle_result=CycleResult(
                decision=ChiefDecision.PAUSE,
                mission_id=mission_id,
                reason=f"Cycle error: {error}",
            ),
            consecutive_failures=consecutive_failures + 1,
        )
