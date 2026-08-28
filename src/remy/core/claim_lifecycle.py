"""Local, append-only lifecycle tracking for mutable research claims."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
from threading import RLock
from typing import Any, Mapping

from remy.config.settings import settings
from remy.core.claim_temporal_evidence import (
    classify_claim_temporality,
    parse_source_datetime,
)
from remy.core.file_utils import atomic_write


CLAIM_LIFECYCLE_DIR = settings.DATA_DIR / "claim_lifecycle"
_LOCK = RLock()
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_VALUE_RE = re.compile(
    r"(?:[$€£₴]\s?\d+(?:[.,]\d+)?)|"
    r"(?:\bv?\d+(?:\.\d+){1,3}(?:[-.][a-z0-9]+)?\b)|"
    r"(?:\b\d+(?:[.,]\d+)?%?\b)",
    re.IGNORECASE,
)
_MUTABLE_RE = re.compile(
    r"\b(?:price|cost|version|release|status|supported|available|availability|"
    r"ceo|president|prime minister|head coach)\b",
    re.IGNORECASE,
)
_STATE_RE = re.compile(
    r"\b(?:active|inactive|available|unavailable|supported|unsupported|"
    r"operational|degraded|offline|online|released|deprecated|retired)\b",
    re.IGNORECASE,
)
_SUBJECT_STOP = {
    "the", "a", "an", "is", "was", "are", "were", "now", "current",
    "currently", "latest", "today", "version", "price", "cost", "status",
    "supported", "available", "availability", "release", "as", "of", "in",
    "on", "to", "for", "and", "per",
}


def _utc(value: datetime | None = None) -> datetime:
    observed = value or datetime.now(timezone.utc)
    if observed.tzinfo is None:
        return observed.replace(tzinfo=timezone.utc)
    return observed.astimezone(timezone.utc)


def _normalized(text: str) -> str:
    return " ".join(str(text or "").casefold().split())


def is_lifecycle_claim(claim: str) -> bool:
    """Return whether a claim describes a fact expected to change over time."""
    text = _normalized(claim)
    return bool(
        classify_claim_temporality(text).get("time_sensitive")
        or _MUTABLE_RE.search(text)
    )


def claim_value_signature(claim: str) -> str:
    """Extract the changing value while keeping the result deterministic."""
    text = _normalized(claim)
    values = [match.group(0).replace(" ", "") for match in _VALUE_RE.finditer(text)]
    states = [match.group(0) for match in _STATE_RE.finditer(text)]
    signature = list(dict.fromkeys([*values, *states]))
    return "|".join(signature) if signature else hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:16]


def claim_subject_key(claim: str) -> str:
    """Build a stable subject key after removing mutable values and modifiers."""
    scrubbed = _VALUE_RE.sub(" ", _normalized(claim))
    scrubbed = _STATE_RE.sub(" ", scrubbed)
    tokens = sorted(
        {
            token
            for token in _TOKEN_RE.findall(scrubbed)
            if len(token) >= 3 and token not in _SUBJECT_STOP
        }
    )
    stable = " ".join(tokens) or scrubbed.strip() or _normalized(claim)
    return f"subject-{hashlib.sha256(stable.encode('utf-8')).hexdigest()[:16]}"


def lifecycle_scope_key(project_id: str, topic: str) -> str:
    # Topic wording often changes between runs. Subject keys provide separation;
    # the durable isolation boundary is the workspace project itself.
    boundary = _normalized(project_id) or "default"
    return hashlib.sha256(boundary.encode("utf-8")).hexdigest()[:24]


def _path(project_id: str, topic: str):
    return CLAIM_LIFECYCLE_DIR / f"{lifecycle_scope_key(project_id, topic)}.json"


def _best_relation(row: Mapping[str, Any]) -> dict[str, Any]:
    relations = [
        dict(item)
        for item in row.get("relations") or []
        if isinstance(item, Mapping) and item.get("relation") == "supports"
    ]
    if not relations:
        return {}

    def rank(item: Mapping[str, Any]) -> tuple[int, float, float]:
        parsed = parse_source_datetime(item.get("source_date"))
        timestamp = parsed.timestamp() if parsed else 0.0
        return (
            int(bool(item.get("temporally_valid"))),
            timestamp,
            float(item.get("authority_score") or 0.0),
        )

    return max(relations, key=rank)


def _snapshot(row: Mapping[str, Any], observed_at: str) -> dict[str, Any]:
    relation = _best_relation(row)
    claim = str(row.get("claim") or "").strip()
    source_date = str(relation.get("source_date") or "")
    return {
        "claim_id": str(row.get("claim_id") or ""),
        "claim": claim,
        "value_signature": claim_value_signature(claim),
        "first_seen": observed_at,
        "last_seen": observed_at,
        "observation_count": 1,
        "effective_at": source_date or observed_at,
        "source_date": source_date,
        "source_url": str(relation.get("url") or ""),
        "evidence_root": str(
            relation.get("evidence_root") or relation.get("url") or ""
        ),
        "authority_role": str(relation.get("authority_role") or "unknown"),
        "authority_score": float(relation.get("authority_score") or 0.0),
        "temporally_valid": bool(relation.get("temporally_valid", False)),
    }


def _historical_snapshot(supersession: Mapping[str, Any], observed_at: str) -> dict[str, Any]:
    claim = str(supersession.get("old_claim") or "").strip()
    effective = str(supersession.get("old_source_date") or observed_at)
    return {
        "claim_id": "",
        "claim": claim,
        "value_signature": claim_value_signature(claim),
        "first_seen": observed_at,
        "last_seen": observed_at,
        "observation_count": 1,
        "effective_at": effective,
        "source_date": effective,
        "source_url": str(supersession.get("old_source_url") or ""),
        "evidence_root": str(supersession.get("old_evidence_root") or ""),
        "authority_role": "historical",
        "authority_score": float(supersession.get("old_authority_score") or 0.0),
        "temporally_valid": False,
    }


def _event_id(subject_key: str, event_type: str, before: Mapping[str, Any] | None, after: Mapping[str, Any]) -> str:
    material = "\n".join(
        (
            subject_key,
            event_type,
            str((before or {}).get("value_signature") or ""),
            str(after.get("value_signature") or ""),
            str(after.get("effective_at") or ""),
            str(after.get("source_url") or ""),
        )
    )
    return f"cle-{hashlib.sha256(material.encode('utf-8')).hexdigest()[:16]}"


def _event(
    subject_key: str,
    event_type: str,
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any],
    observed_at: str,
    *,
    reasons: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "event_id": _event_id(subject_key, event_type, before, after),
        "event_type": event_type,
        "observed_at": observed_at,
        "effective_at": str(after.get("effective_at") or observed_at),
        "from_claim": str((before or {}).get("claim") or ""),
        "from_value": str((before or {}).get("value_signature") or ""),
        "to_claim": str(after.get("claim") or ""),
        "to_value": str(after.get("value_signature") or ""),
        "source_url": str(after.get("source_url") or ""),
        "evidence_root": str(after.get("evidence_root") or ""),
        "authority_score": float(after.get("authority_score") or 0.0),
        "decision_reasons": list(reasons or []),
        "history_preserved": True,
    }


def _transition_checks(before: Mapping[str, Any], after: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    before_date = parse_source_datetime(before.get("effective_at"))
    after_date = parse_source_datetime(after.get("effective_at"))
    if not before_date or not after_date:
        reasons.append("both_versions_must_be_dated")
    elif after_date <= before_date:
        reasons.append("candidate_is_not_newer")
    if not after.get("temporally_valid"):
        reasons.append("candidate_is_not_temporally_valid")
    before_root = str(before.get("evidence_root") or "")
    after_root = str(after.get("evidence_root") or "")
    if not before_root or not after_root or before_root == after_root:
        reasons.append("independent_evidence_root_required")
    if float(after.get("authority_score") or 0.0) + 0.10 < float(
        before.get("authority_score") or 0.0
    ):
        reasons.append("candidate_authority_is_materially_weaker")
    if not is_lifecycle_claim(str(after.get("claim") or "")):
        reasons.append("claim_is_not_mutable")
    return not reasons, reasons


def _empty_ledger(project_id: str, topic: str, observed_at: str) -> dict[str, Any]:
    return {
        "version": 1,
        "scope_key": lifecycle_scope_key(project_id, topic),
        "project_id": str(project_id or "default"),
        "topic": str(topic or ""),
        "topics": [str(topic or "")] if str(topic or "").strip() else [],
        "created_at": observed_at,
        "updated_at": observed_at,
        "subjects": {},
    }


def update_claim_lifecycle(
    ledger: Mapping[str, Any] | None,
    matrix: Mapping[str, Any],
    *,
    project_id: str = "default",
    topic: str = "",
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    """Apply a claim matrix to a lifecycle ledger without external services."""
    observed_iso = _utc(observed_at).isoformat()
    result = dict(ledger or _empty_ledger(project_id, topic, observed_iso))
    result.setdefault("version", 1)
    result.setdefault("scope_key", lifecycle_scope_key(project_id, topic))
    result.setdefault("project_id", str(project_id or "default"))
    result.setdefault("topic", str(topic or ""))
    topics = [str(item) for item in result.get("topics") or [] if str(item).strip()]
    if str(topic or "").strip() and str(topic) not in topics:
        topics.append(str(topic))
    result["topics"] = topics[-50:]
    result.setdefault("created_at", observed_iso)
    subjects = {
        str(key): dict(value)
        for key, value in dict(result.get("subjects") or {}).items()
        if isinstance(value, Mapping)
    }
    run = {
        "observed_claims": 0,
        "new_subjects": 0,
        "reaffirmed_claims": 0,
        "confirmed_transitions": 0,
        "pending_changes": 0,
    }

    rows = [
        row
        for row in matrix.get("rows") or []
        if isinstance(row, Mapping)
        and row.get("status") == "supported"
        and is_lifecycle_claim(str(row.get("claim") or ""))
    ]
    for row in rows:
        run["observed_claims"] += 1
        snapshot = _snapshot(row, observed_iso)
        subject_key = claim_subject_key(snapshot["claim"])
        subject = subjects.get(subject_key)
        if not subject:
            supersessions = [
                item
                for item in row.get("resolved_supersessions") or []
                if isinstance(item, Mapping) and item.get("old_claim")
            ]
            history: list[dict[str, Any]] = []
            if supersessions:
                old = _historical_snapshot(supersessions[0], observed_iso)
                history.append(_event(subject_key, "first_seen", None, old, observed_iso))
                history.append(
                    _event(
                        subject_key,
                        "confirmed_transition",
                        old,
                        snapshot,
                        observed_iso,
                        reasons=["resolved_by_temporal_supersession"],
                    )
                )
                run["confirmed_transitions"] += 1
            else:
                history.append(
                    _event(subject_key, "first_seen", None, snapshot, observed_iso)
                )
            subjects[subject_key] = {
                "subject_key": subject_key,
                "current": snapshot,
                "history": history,
                "pending_changes": [],
            }
            run["new_subjects"] += 1
            continue

        current = dict(subject.get("current") or {})
        history = list(subject.get("history") or [])
        pending = list(subject.get("pending_changes") or [])
        if current.get("value_signature") == snapshot.get("value_signature"):
            current["last_seen"] = observed_iso
            current["observation_count"] = int(current.get("observation_count") or 1) + 1
            if snapshot.get("source_date") and str(snapshot["source_date"]) >= str(
                current.get("source_date") or ""
            ):
                current.update(
                    {
                        key: snapshot[key]
                        for key in (
                            "claim_id", "claim", "effective_at", "source_date",
                            "source_url", "evidence_root", "authority_role",
                            "authority_score", "temporally_valid",
                        )
                    }
                )
            subject["current"] = current
            run["reaffirmed_claims"] += 1
            subjects[subject_key] = subject
            continue

        confirmed, reasons = _transition_checks(current, snapshot)
        event_type = "confirmed_transition" if confirmed else "pending_change"
        event = _event(
            subject_key,
            event_type,
            current,
            snapshot,
            observed_iso,
            reasons=reasons or ["cross_run_safety_gate_passed"],
        )
        target = history if confirmed else pending
        if not any(item.get("event_id") == event["event_id"] for item in target):
            target.append(event)
        if confirmed:
            subject["current"] = snapshot
            run["confirmed_transitions"] += 1
        else:
            run["pending_changes"] += 1
        subject["history"] = history[-100:]
        subject["pending_changes"] = pending[-25:]
        subjects[subject_key] = subject

    result["subjects"] = subjects
    result["updated_at"] = observed_iso
    result["last_run"] = run
    result["summary"] = {
        "tracked_subjects": len(subjects),
        "confirmed_transitions": sum(
            item.get("event_type") == "confirmed_transition"
            for subject in subjects.values()
            for item in subject.get("history") or []
        ),
        "pending_changes": sum(
            len(subject.get("pending_changes") or []) for subject in subjects.values()
        ),
    }
    return result


def load_claim_lifecycle(project_id: str, topic: str) -> dict[str, Any] | None:
    path = _path(project_id, topic)
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def trajectory_lifecycle_view(
    ledger: Mapping[str, Any],
    *,
    subject_limit: int = 50,
    history_limit: int = 20,
    pending_limit: int = 10,
) -> dict[str, Any]:
    """Return a bounded view so long-lived ledgers cannot slow Trajectory."""
    raw_subjects = [
        dict(subject)
        for subject in (ledger.get("subjects") or {}).values()
        if isinstance(subject, Mapping)
    ]
    raw_subjects.sort(
        key=lambda subject: str((subject.get("current") or {}).get("last_seen") or ""),
        reverse=True,
    )
    subjects = []
    for subject in raw_subjects[: max(1, int(subject_limit))]:
        subjects.append(
            {
                **subject,
                "history": list(subject.get("history") or [])[-max(1, int(history_limit)):],
                "pending_changes": list(subject.get("pending_changes") or [])[
                    -max(1, int(pending_limit)):
                ],
            }
        )
    summary = dict(ledger.get("summary") or {})
    summary["returned_subjects"] = len(subjects)
    summary["trajectory_window_truncated"] = len(raw_subjects) > len(subjects)
    return {
        "version": ledger.get("version", 1),
        "scope_key": ledger.get("scope_key", ""),
        "project_id": ledger.get("project_id", ""),
        "topic": ledger.get("topic", ""),
        "topics": list(ledger.get("topics") or []),
        "updated_at": ledger.get("updated_at", ""),
        "summary": summary,
        "last_run": dict(ledger.get("last_run") or {}),
        "subjects": subjects,
    }
def record_claim_lifecycle(
    project_id: str,
    topic: str,
    matrix: Mapping[str, Any],
    *,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    """Atomically persist and return a bounded Trajectory-facing lifecycle view."""
    with _LOCK:
        ledger = update_claim_lifecycle(
            load_claim_lifecycle(project_id, topic),
            matrix,
            project_id=project_id,
            topic=topic,
            observed_at=observed_at,
        )
        path = _path(project_id, topic)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, json.dumps(ledger, ensure_ascii=False, indent=2))
    return trajectory_lifecycle_view(ledger)
