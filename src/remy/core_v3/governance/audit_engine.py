"""
Audit Engine for Remy v3 Governance Layer.

Append-only audit trail for all significant system events.
Phase 5: Enhanced with query, filter, aggregation, and event categories.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class EventType:
    """Standard audit event types."""
    MISSION_STARTED = "mission_started"
    MISSION_COMPLETED = "mission_completed"
    MISSION_FAILED = "mission_failed"
    GOAL_STARTED = "goal_started"
    GOAL_COMPLETED = "goal_completed"
    GOAL_FAILED = "goal_failed"
    STEP_EXECUTED = "step_executed"
    STEP_FAILED = "step_failed"
    BUDGET_SPEND = "budget_spend"
    BUDGET_WARNING = "budget_warning"
    BUDGET_EXHAUSTED = "budget_exhausted"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    POLICY_VIOLATION = "policy_violation"
    DELEGATION = "delegation"
    REPLAN = "replan"
    ESCALATION = "escalation"
    ERROR = "error"
    SYSTEM = "system"


@dataclass
class AuditEvent:
    """Single audit trail entry."""
    timestamp: float = field(default_factory=time.time)
    event_type: str = ""
    actor: str = ""                # chief, researcher, user, system
    action: str = ""
    mission_id: str = ""
    goal_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"
    cost_usd: float = 0.0


class AuditEngine:
    """Append-only audit log with query capabilities."""

    RETENTION_DAYS = 3  # Keep only last 3 days of audit entries
    TRIM_EVERY_N = 50   # Trim old entries every N writes

    def __init__(self, log_path: str = "", buffer_max: int = 500):
        if not log_path:
            try:
                from remy.config import settings
                log_path = os.path.join(str(settings.DATA_DIR), "audit_log.jsonl")
            except ImportError:
                log_path = "data/audit_log.jsonl"
        self._path = log_path
        self._buffer: list[AuditEvent] = []
        self._buffer_max = buffer_max
        self._write_count = 0

    def record(self, event: AuditEvent):
        """Record an audit event."""
        self._buffer.append(event)
        if len(self._buffer) > self._buffer_max:
            self._buffer = self._buffer[-self._buffer_max:]
        self._flush_one(event)
        self._write_count += 1
        if self._write_count >= self.TRIM_EVERY_N:
            self._write_count = 0
            self._trim_old_entries()

    def _trim_old_entries(self):
        """Remove entries older than RETENTION_DAYS from the JSONL file."""
        try:
            if not os.path.exists(self._path):
                return
            cutoff = time.time() - (self.RETENTION_DAYS * 86400)
            kept: list[str] = []
            removed = 0
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if float(entry.get("timestamp", 0)) >= cutoff:
                            kept.append(line)
                        else:
                            removed += 1
                    except (json.JSONDecodeError, ValueError):
                        kept.append(line)  # keep unparseable lines
            if removed:
                with open(self._path, "w", encoding="utf-8") as f:
                    f.write("\n".join(kept) + ("\n" if kept else ""))
                log.debug("Audit log trimmed: removed %d entries older than %d days", removed, self.RETENTION_DAYS)
        except Exception as e:
            log.debug("Audit log trim failed: %s", e)

    def log_event(
        self,
        event_type: str,
        action: str,
        *,
        actor: str = "system",
        mission_id: str = "",
        goal_id: str = "",
        details: dict[str, Any] | None = None,
        risk_level: str = "low",
        cost_usd: float = 0.0,
    ):
        """Convenience method to record an event."""
        self.record(AuditEvent(
            event_type=event_type,
            actor=actor,
            action=action,
            mission_id=mission_id,
            goal_id=goal_id,
            details=details or {},
            risk_level=risk_level,
            cost_usd=cost_usd,
        ))

    def _flush_one(self, event: AuditEvent):
        """Append single event to JSONL file."""
        try:
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(event), default=str) + "\n")
        except Exception as e:
            log.error("Failed to write audit event: %s", e)

    # -------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------

    def recent(self, limit: int = 50) -> list[AuditEvent]:
        """Get recent events from buffer."""
        return list(reversed(self._buffer[-limit:]))

    def recent_by_mission(self, mission_id: str, limit: int = 20) -> list[AuditEvent]:
        return [
            e for e in reversed(self._buffer)
            if e.mission_id == mission_id
        ][:limit]

    def recent_by_type(self, event_type: str, limit: int = 20) -> list[AuditEvent]:
        return [
            e for e in reversed(self._buffer)
            if e.event_type == event_type
        ][:limit]

    def recent_by_actor(self, actor: str, limit: int = 20) -> list[AuditEvent]:
        return [
            e for e in reversed(self._buffer)
            if e.actor == actor
        ][:limit]

    def errors(self, limit: int = 20) -> list[AuditEvent]:
        """Get recent error events."""
        return self.recent_by_type(EventType.ERROR, limit)

    def policy_violations(self, limit: int = 20) -> list[AuditEvent]:
        """Get recent policy violations."""
        return self.recent_by_type(EventType.POLICY_VIOLATION, limit)

    # -------------------------------------------------------------------
    # Aggregation
    # -------------------------------------------------------------------

    def total_cost(self, hours: float = 24.0) -> float:
        """Total cost in the last N hours."""
        cutoff = time.time() - (hours * 3600)
        return sum(
            e.cost_usd for e in self._buffer
            if e.timestamp >= cutoff
        )

    def event_counts(self, hours: float = 24.0) -> dict[str, int]:
        """Count events by type in the last N hours."""
        cutoff = time.time() - (hours * 3600)
        counts: dict[str, int] = {}
        for e in self._buffer:
            if e.timestamp >= cutoff:
                counts[e.event_type] = counts.get(e.event_type, 0) + 1
        return counts

    def mission_cost(self, mission_id: str) -> float:
        """Total cost for a specific mission."""
        return sum(
            e.cost_usd for e in self._buffer
            if e.mission_id == mission_id
        )

    def actor_stats(self, hours: float = 24.0) -> dict[str, dict[str, Any]]:
        """Per-actor statistics in the last N hours."""
        cutoff = time.time() - (hours * 3600)
        stats: dict[str, dict[str, Any]] = {}
        for e in self._buffer:
            if e.timestamp < cutoff:
                continue
            if e.actor not in stats:
                stats[e.actor] = {"events": 0, "cost_usd": 0.0, "errors": 0}
            stats[e.actor]["events"] += 1
            stats[e.actor]["cost_usd"] += e.cost_usd
            if e.event_type in (EventType.ERROR, EventType.STEP_FAILED):
                stats[e.actor]["errors"] += 1
        return stats

    def summary(self) -> dict[str, Any]:
        """Audit summary for observability."""
        return {
            "total_events": len(self._buffer),
            "cost_24h": round(self.total_cost(24.0), 4),
            "event_counts_24h": self.event_counts(24.0),
            "recent_errors": len(self.errors(50)),
            "policy_violations": len(self.policy_violations(50)),
        }
