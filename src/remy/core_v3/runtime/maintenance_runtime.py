"""
Maintenance runtime for Remy v3.

Owns deterministic loop maintenance side effects so AutonomyLoop can remain a
thin scheduler shell.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


class MaintenanceRuntime:
    """Encapsulate periodic maintenance orchestration."""

    def __init__(self, chief):
        self.chief = chief

    def run(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "archived_stale_goals": True,
            "aura_maintenance": None,
            "aura_maintenance_error": "",
            "approvals_cleared": True,
            "state_persisted": True,
        }

        self.chief.goal_tracker.archive_stale()

        try:
            from ..memory.memory_api import get_memory

            memory = get_memory()
            report["aura_maintenance"] = memory.run_maintenance()
        except Exception as exc:
            report["aura_maintenance_error"] = str(exc)
            log.debug("Aura maintenance skipped: %s", exc)

        self.chief.approval.clear_decided()
        self.chief.save_state()
        return report
