"""
Loop Runtime for Remy v3.

Owns deterministic post-cycle behavior:
- cycle result handling
- consecutive failure accounting
- churn / stuck pressure tracking
"""

from __future__ import annotations

import logging
import time
from collections import deque

from ..agents.chief_agent import ChiefDecision, CycleResult
log = logging.getLogger(__name__)


class LoopRuntime:
    """Deterministic result-accounting layer for the loop."""

    def __init__(self, chief):
        self.chief = chief
        self._mission_pressure: dict[str, dict[str, float | int | str]] = {}
        self._recent_decisions: deque[dict] = deque(maxlen=20)

    def _pressure_entry(self, mission_id: str) -> dict[str, float | int | str]:
        return self._mission_pressure.setdefault(
            mission_id,
            {
                "pressure": 0,
                "last_decision": "",
                "last_reason": "",
                "updated_at": 0.0,
            },
        )

    def _record_decision(self, result: CycleResult):
        self._recent_decisions.append({
            "mission_id": result.mission_id,
            "decision": result.decision,
            "reason": result.reason,
            "timestamp": time.time(),
        })

    def pressure_for(self, mission_id: str) -> int:
        entry = self._mission_pressure.get(mission_id)
        return int(entry.get("pressure", 0)) if entry else 0

    def stuck_missions(self, limit: int = 10) -> list[dict]:
        items = []
        for mission_id, entry in self._mission_pressure.items():
            pressure = int(entry.get("pressure", 0) or 0)
            if pressure <= 0:
                continue
            items.append({
                "mission_id": mission_id,
                "pressure": pressure,
                "last_decision": entry.get("last_decision", ""),
                "last_reason": entry.get("last_reason", ""),
                "updated_at": entry.get("updated_at", 0.0),
            })
        items.sort(key=lambda item: (-item["pressure"], -float(item["updated_at"])))
        return items[:limit]

    def stuck_missions_count(self) -> int:
        return len(self.stuck_missions(limit=1000))

    def recent_decisions(self, limit: int = 10) -> list[dict]:
        return list(self._recent_decisions)[-limit:]

    def handle_result(self, result: CycleResult, consecutive_failures: int) -> int:
        self._record_decision(result)
        entry = self._pressure_entry(result.mission_id) if result.mission_id else None

        if result.decision == ChiefDecision.COMPLETE:
            log.info("Mission %s completed", result.mission_id)
            if entry is not None:
                entry["pressure"] = 0
                entry["last_decision"] = result.decision
                entry["last_reason"] = result.reason
                entry["updated_at"] = time.time()
            return 0

        if result.decision == ChiefDecision.EXECUTE_STEP:
            if entry is not None:
                entry["pressure"] = 0
                entry["last_decision"] = result.decision
                entry["last_reason"] = result.reason
                entry["updated_at"] = time.time()
            return 0

        if result.decision == ChiefDecision.REPLAN:
            log.info("Mission %s needs replan: %s", result.mission_id, result.reason)
            if entry is not None:
                entry["pressure"] = min(12, int(entry["pressure"]) + 1)
                entry["last_decision"] = result.decision
                entry["last_reason"] = result.reason
                entry["updated_at"] = time.time()
            return consecutive_failures

        if result.decision == ChiefDecision.PAUSE:
            log.info("Mission %s paused: %s", result.mission_id, result.reason)
            if entry is not None:
                entry["pressure"] = min(12, int(entry["pressure"]) + 1)
                entry["last_decision"] = result.decision
                entry["last_reason"] = result.reason
                entry["updated_at"] = time.time()
            return consecutive_failures

        if result.decision == ChiefDecision.ESCALATE:
            log.warning("Mission %s escalated: %s", result.mission_id, result.reason)
            if entry is not None:
                entry["pressure"] = min(12, int(entry["pressure"]) + 1)
                entry["last_decision"] = result.decision
                entry["last_reason"] = result.reason
                entry["updated_at"] = time.time()
            return consecutive_failures

        if result.decision == ChiefDecision.ABORT:
            log.warning("Mission %s aborted: %s", result.mission_id, result.reason)
            if entry is not None:
                entry["pressure"] = min(12, int(entry["pressure"]) + 2)
                entry["last_decision"] = result.decision
                entry["last_reason"] = result.reason
                entry["updated_at"] = time.time()
            return consecutive_failures + 1

        return consecutive_failures
