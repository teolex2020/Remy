"""Shared verification gate helpers for artifact-backed success checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from contextlib import nullcontext

from remy.core.failure_taxonomy import FailureCode


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    REPAIR_REQUIRED = "repair_required"


@dataclass(slots=True)
class VerificationResult:
    status: str
    verified: bool
    failure_code: str | None = None
    reason: str = ""
    artifact_ids: list[str] | None = None
    repair_required: bool = False

    def to_dict(self) -> dict:
        data = asdict(self)
        data["artifact_ids"] = list(self.artifact_ids or [])
        return data


def verification_incident_dedupe_key(*, source: str, artifact_label: str = "") -> str:
    """Build a stable dedupe key for verification incidents/resolutions."""
    label = artifact_label.strip() or source
    return f"verification|{source}|{label}"


def verification_action_target(source: str) -> str:
    """Map a verification source to the most relevant operator review surface."""
    normalized = str(source or "").strip()
    if normalized == "reconstruct_missing_memory":
        return "open_missing_memory_review"
    if normalized in {"generate_report", "complete_research"}:
        return "open_memory_verification"
    return ""


def _store_verification_incident_artifact(
    *,
    source: str,
    message: str,
    verification: VerificationResult,
    extra: dict | None = None,
) -> str:
    """Persist an operator-visible verification incident as a durable review artifact."""
    try:
        import remy.core.brain_tools as _bt
        from remy.core.agent_tools import Level
        from remy.core.provenance import _stamp_provenance

        brain = getattr(_bt, "brain", None)
        if brain is None or not hasattr(brain, "store"):
            return ""

        dedupe_key = verification_incident_dedupe_key(source=source, artifact_label=source)
        if hasattr(brain, "search"):
            existing = brain.search(query=dedupe_key, tags=["incident_snapshot"], limit=1) or []
            if existing:
                return str(getattr(existing[0], "id", "") or "")

        extra = dict(extra or {})
        artifact_ids = list(verification.artifact_ids or [])
        content = "\n".join(
            [
                "Operator Incident Snapshot",
                "",
                f"Message: {message}",
                f"Source: {source}",
                f"Failure code: {str(verification.failure_code or FailureCode.VERIFICATION_FAILED.value)}",
                f"Verification status: {verification.status}",
                f"Verification reason: {verification.reason}",
                f"Requested: {extra.get('requested', '')}",
                f"Applied: {extra.get('applied', '')}",
                f"Skipped: {extra.get('skipped', '')}",
                f"Dedupe key: {dedupe_key}",
                artifact_ids and "Artifacts:\n" + "\n".join(f"- {item}" for item in artifact_ids) or "",
            ]
        ).strip()
        metadata = _stamp_provenance(
            {
                "type": "incident_snapshot",
                "source": source,
                "failure_code": str(verification.failure_code or FailureCode.VERIFICATION_FAILED.value),
                "verification_status": verification.status,
                "verification_reason": verification.reason,
                "artifact_ids": artifact_ids,
                "requested": extra.get("requested"),
                "applied": extra.get("applied"),
                "skipped": extra.get("skipped"),
                "dedupe_key": dedupe_key,
            },
            "system",
            tags=["operator", "incident_snapshot", "review"],
        )
        brain_lock = getattr(_bt, "brain_lock", None) or nullcontext()
        with brain_lock:
            rec = brain.store(
                content=content,
                level=Level.DECISIONS,
                tags=["operator", "incident_snapshot", "review"],
                metadata=metadata,
            )
        return str(getattr(rec, "id", "") or "")
    except Exception:
        return ""


def emit_verification_incident(
    *,
    source: str,
    verification: VerificationResult,
    artifact_label: str = "",
    extra: dict | None = None,
) -> None:
    """Emit operator-visible incident for verification failures."""
    if verification.verified or not verification.repair_required:
        return
    try:
        from remy.core.notification_router import notify

        label = artifact_label.strip() or source
        message = f"Verification failed for {label}: {verification.reason or 'repair required'}"
        extra = dict(extra or {})
        incident_artifact_id = _store_verification_incident_artifact(
            source=source,
            message=message,
            verification=verification,
            extra=extra,
        )
        event_data = {
            "failure_code": str(verification.failure_code or FailureCode.VERIFICATION_FAILED.value),
            "verification_status": verification.status,
            "verification_reason": verification.reason,
            "artifact_ids": list(verification.artifact_ids or []) + ([incident_artifact_id] if incident_artifact_id else []),
            "dedupe_key": verification_incident_dedupe_key(source=source, artifact_label=label),
            "source": source,
            "action_target": verification_action_target(source),
        }
        if extra:
            event_data.update(extra)
        notify(
            message,
            level="warning",
            event_type="operator_alert",
            event_data=event_data,
            parse_mode="",
        )
    except Exception:
        pass


def resolve_verification_incident(
    *,
    source: str,
    artifact_label: str = "",
    message: str = "",
    extra: dict | None = None,
) -> None:
    """Resolve a prior verification warning when the same artifact later verifies cleanly."""
    try:
        from remy.core.notification_router import notify

        label = artifact_label.strip() or source
        event_data = {
            "source": source,
            "artifact_label": label,
            "action_target": verification_action_target(source),
            "resolves": [verification_incident_dedupe_key(source=source, artifact_label=label)],
        }
        if extra:
            event_data.update(extra)
        notify(
            message or f"Verification passed for {label}.",
            level="info",
            event_type="verification.resolved",
            event_data=event_data,
            parse_mode="",
        )
    except Exception:
        pass


def run_report_verification_gate(
    filepath: str | Path,
    *,
    title: str = "",
    section_count: int = 0,
) -> VerificationResult:
    """Verify a generated PDF artifact before reporting success."""
    from remy.core.report_builder import verify_generated_report

    valid_pdf, validation_note = verify_generated_report(filepath, title=title)
    if valid_pdf:
        return VerificationResult(
            status=VerificationStatus.VERIFIED.value,
            verified=True,
            reason=validation_note,
            artifact_ids=[str(Path(filepath))],
            repair_required=False,
        )

    return VerificationResult(
        status=VerificationStatus.REPAIR_REQUIRED.value,
        verified=False,
        failure_code=FailureCode.VERIFICATION_FAILED.value if section_count else None,
        reason=validation_note,
        artifact_ids=[str(Path(filepath))],
        repair_required=bool(section_count),
    )


def run_research_completion_verification_gate(
    *,
    project_id: str,
    report_record_id: str,
    stored_report_record,
    markdown_body: str,
    findings_count: int,
    pdf_result: dict | None = None,
) -> VerificationResult:
    """Verify research completion before marking the project complete."""
    artifact_ids: list[str] = []
    if report_record_id:
        artifact_ids.append(str(report_record_id))
    if isinstance(pdf_result, dict):
        pdf_record_id = str(pdf_result.get("record_id") or pdf_result.get("pdf_record_id") or "").strip()
        if pdf_record_id:
            artifact_ids.append(pdf_record_id)

    if findings_count <= 0:
        return VerificationResult(
            status=VerificationStatus.REPAIR_REQUIRED.value,
            verified=False,
            failure_code=FailureCode.VERIFICATION_FAILED.value,
            reason="Research completion requires at least one finding.",
            artifact_ids=artifact_ids,
            repair_required=True,
        )

    if not report_record_id or stored_report_record is None:
        return VerificationResult(
            status=VerificationStatus.REPAIR_REQUIRED.value,
            verified=False,
            failure_code=FailureCode.VERIFICATION_FAILED.value,
            reason=f"Research report artifact for project '{project_id}' was not stored correctly.",
            artifact_ids=artifact_ids,
            repair_required=True,
        )

    if len((markdown_body or "").strip()) < 20:
        return VerificationResult(
            status=VerificationStatus.REPAIR_REQUIRED.value,
            verified=False,
            failure_code=FailureCode.VERIFICATION_FAILED.value,
            reason="Research report artifact is too short to verify.",
            artifact_ids=artifact_ids,
            repair_required=True,
        )

    return VerificationResult(
        status=VerificationStatus.VERIFIED.value,
        verified=True,
        reason="Research report artifact stored and ready.",
        artifact_ids=artifact_ids,
        repair_required=False,
    )


def run_reconstruction_verification_gate(
    *,
    requested: int,
    applied_candidate_ids: list[str] | None = None,
    skipped_candidate_ids: list[str] | None = None,
    missing_candidate_ids: list[str] | None = None,
    tool_errors: list[dict] | None = None,
) -> VerificationResult:
    """Verify selective reconstruction before treating it as a clean success."""
    applied_candidate_ids = list(applied_candidate_ids or [])
    skipped_candidate_ids = list(skipped_candidate_ids or [])
    missing_candidate_ids = list(missing_candidate_ids or [])
    tool_errors = list(tool_errors or [])

    artifact_ids = applied_candidate_ids + skipped_candidate_ids + missing_candidate_ids

    if requested <= 0:
        return VerificationResult(
            status=VerificationStatus.REPAIR_REQUIRED.value,
            verified=False,
            failure_code=FailureCode.VALIDATION_ERROR.value,
            reason="No reconstruction candidates were selected.",
            artifact_ids=artifact_ids,
            repair_required=True,
        )

    if applied_candidate_ids and not skipped_candidate_ids and not missing_candidate_ids and not tool_errors:
        return VerificationResult(
            status=VerificationStatus.VERIFIED.value,
            verified=True,
            reason=f"Reconstruction restored {len(applied_candidate_ids)} selected candidate(s).",
            artifact_ids=artifact_ids,
            repair_required=False,
        )

    if applied_candidate_ids:
        return VerificationResult(
            status=VerificationStatus.REPAIR_REQUIRED.value,
            verified=False,
            failure_code=FailureCode.VERIFICATION_FAILED.value,
            reason=(
                f"Reconstruction only partially succeeded: restored {len(applied_candidate_ids)} of "
                f"{requested} selected candidate(s)."
            ),
            artifact_ids=artifact_ids,
            repair_required=True,
        )

    return VerificationResult(
        status=VerificationStatus.REPAIR_REQUIRED.value,
        verified=False,
        failure_code=FailureCode.VERIFICATION_FAILED.value,
        reason="Reconstruction did not restore any selected candidates.",
        artifact_ids=artifact_ids,
        repair_required=True,
    )
