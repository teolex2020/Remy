"""Shared runtime failure taxonomy for operator-visible incidents."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum


class FailureCode(StrEnum):
    FORMAT_ERROR = "format_error"
    TOOL_ERROR = "tool_error"
    VALIDATION_ERROR = "validation_error"
    VERIFICATION_FAILED = "verification_failed"
    EVIDENCE_CONFLICT = "evidence_conflict"
    MEMORY_RECOVERY_APPLIED = "memory_recovery_applied"
    STORE_INTEGRITY_INCIDENT = "store_integrity_incident"
    TIMEOUT = "timeout"
    APPROVAL_BLOCKED = "approval_blocked"


class FailureSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(slots=True)
class RuntimeIncident:
    code: str
    severity: str
    message: str
    recovery_applied: bool = False
    source: str = "runtime"
    operator_action: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def get_failure_taxonomy_summary() -> dict:
    return {
        "codes": [code.value for code in FailureCode],
        "severities": [sev.value for sev in FailureSeverity],
    }


def classify_startup_incident(startup_status: dict | None) -> RuntimeIncident | None:
    startup_status = startup_status or {}
    blocked = bool(startup_status.get("startup_blocked"))
    quarantined = bool(startup_status.get("quarantined_at_startup"))
    reason = str(startup_status.get("quarantine_reason") or "").strip()
    incident = str(startup_status.get("startup_incident") or "").strip()

    if blocked:
        return RuntimeIncident(
            code=FailureCode.STORE_INTEGRITY_INCIDENT.value,
            severity=FailureSeverity.CRITICAL.value,
            message=incident or reason or "Startup was blocked by an Aura store integrity incident.",
            recovery_applied=False,
            source="startup",
            operator_action="Inspect startup diagnostics before continuing.",
        )

    if quarantined:
        return RuntimeIncident(
            code=FailureCode.MEMORY_RECOVERY_APPLIED.value,
            severity=FailureSeverity.WARNING.value,
            message=incident or reason or "Active store was quarantined and startup recovery was applied.",
            recovery_applied=True,
            source="startup",
            operator_action="Review recovery summary and verify restored memory.",
        )

    return None
