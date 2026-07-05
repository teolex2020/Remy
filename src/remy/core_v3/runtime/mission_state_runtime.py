"""
Mission state runtime for Remy v3.

Owns mission activation policy so scheduler and chief do not carry direct
transition logic for INTAKE/PLANNING/ACTIVE states.
"""

from __future__ import annotations

from ..missions.mission_models import Mission, MissionStatus
from .state_machine import transition


class MissionStateRuntime:
    """Centralize mission status transitions used by the runtime."""

    def activate_for_execution(self, mission: Mission) -> Mission:
        if mission.status == MissionStatus.INTAKE:
            transition(mission, MissionStatus.PLANNING, "Mission prepared for execution")
        if mission.status == MissionStatus.PLANNING:
            transition(mission, MissionStatus.ACTIVE, "Mission activated for execution")
        return mission

    def is_schedulable(self, mission: Mission) -> bool:
        return mission.status in (
            MissionStatus.INTAKE,
            MissionStatus.PLANNING,
            MissionStatus.ACTIVE,
        )
