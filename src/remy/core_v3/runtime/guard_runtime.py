"""
Guard runtime for Remy v3.

Owns deterministic pre-execution guard policy for the autonomy loop:
- budget sync from v2
- budget exhausted handling
- maintenance-only pause
- consecutive failure cooldown
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GuardCheck:
    proceed: bool
    sleep_sec: int = 0
    reason: str = ""
    consecutive_failures: int | None = None


class GuardRuntime:
    """Encapsulate loop guard policy without owning scheduling itself."""

    def __init__(self, chief):
        self.chief = chief

    def check(
        self,
        *,
        cycle_count: int,
        consecutive_failures: int,
        max_consecutive_failures: int,
        maintenance_only: bool,
    ) -> GuardCheck:
        self.chief.budget.sync_from_v2()
        budget_status = self.chief.budget.get_status()
        if budget_status.value == "exhausted":
            self.chief.audit.log_event(
                "budget_exhausted",
                "Daily budget exhausted",
                actor="system",
            )
            return GuardCheck(
                proceed=False,
                sleep_sec=300,
                reason="budget_exhausted",
                consecutive_failures=consecutive_failures,
            )

        if maintenance_only:
            return GuardCheck(
                proceed=False,
                reason="maintenance_only",
                consecutive_failures=consecutive_failures,
            )

        if consecutive_failures >= max_consecutive_failures:
            self.chief.audit.log_event(
                "failure_cooldown",
                f"Cooldown after {max_consecutive_failures} failures",
                actor="system",
            )
            return GuardCheck(
                proceed=False,
                sleep_sec=600,
                reason="failure_cooldown",
                consecutive_failures=0,
            )

        return GuardCheck(
            proceed=True,
            consecutive_failures=consecutive_failures,
        )
