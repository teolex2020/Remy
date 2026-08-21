"""Replayable projections over Remy's append-only session event journal."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from remy.config.settings import settings
from remy.core.trajectory_store import TrajectoryStore, _json_dump

logger = logging.getLogger(__name__)


def _latest_snapshots(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for event in events:
        subject = str(event.get("subject_event_id") or "")
        current = latest.get(subject)
        if current is None or int(event.get("event_version") or 0) > int(
            current.get("event_version") or 0
        ):
            latest[subject] = event
    return sorted(
        latest.values(),
        key=lambda row: int((row.get("payload") or {}).get("projection_sequence") or 0),
    )


def _content_text(value: Any) -> str:
    if isinstance(value, dict) and "content" in value:
        return _content_text(value.get("content"))
    if isinstance(value, str):
        return value
    if value in (None, ""):
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class SessionProjection(Protocol):
    name: str

    def project(self, events: list[dict[str, Any]]) -> Any: ...


class TranscriptProjection:
    name = "transcript"

    def __init__(self, *, include_context: bool = False):
        self.include_context = bool(include_context)

    def project(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        roles = {"USER": "user", "ASSISTANT": "assistant"}
        if self.include_context:
            roles.update({"SYSTEM": "system", "CONTEXT": "system"})
        messages = []
        for event in _latest_snapshots(events):
            payload = event.get("payload") or {}
            kind = str(payload.get("kind") or "")
            role = roles.get(kind)
            if not role:
                continue
            content = _content_text(
                payload.get("output")
                if payload.get("output") not in (None, "")
                else payload.get("input")
            )
            if not content:
                continue
            messages.append({
                "message_id": f"session-event:{event['subject_event_id']}",
                "session_id": str(payload.get("session_id") or ""),
                "owner_project_id": str(payload.get("project_id") or ""),
                "brain_id": "event-log-recovery",
                "role": role,
                "content": content,
                "message_type": "recovered_event",
                "metadata": {
                    "event_id": str(event["subject_event_id"]),
                    "event_version": int(event.get("event_version") or 0),
                    "source": payload.get("source") or {},
                    "model_visible": bool(payload.get("model_visible")),
                },
                "created_at": str(payload.get("created_at") or event.get("recorded_at") or ""),
            })
        return messages


class ToolRunProjection:
    name = "tool_runs"

    def project(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        runs = []
        for event in _latest_snapshots(events):
            payload = event.get("payload") or {}
            if str(payload.get("kind") or "") not in {"TOOL", "SUBTOOL", "ATTEMPT"}:
                continue
            details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
            source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
            runs.append({
                "event_id": str(event["subject_event_id"]),
                "kind": str(payload.get("kind") or ""),
                "name": str(details.get("name") or source.get("name") or "model-attempt"),
                "status": str(payload.get("status") or ""),
                "request_id": str(payload.get("request_id") or ""),
                "call_id": str(payload.get("call_id") or ""),
                "duration_ms": int(payload.get("duration_ms") or 0),
                "error": str(payload.get("error") or ""),
                "artifacts": details.get("artifacts") or [],
            })
        return runs


class MetricsProjection:
    name = "metrics"

    def project(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        snapshots = _latest_snapshots(events)
        requests = [row.get("payload") or {} for row in snapshots if row.get("kind") == "REQUEST"]
        tools = [
            row.get("payload") or {}
            for row in snapshots
            if row.get("kind") in {"TOOL", "SUBTOOL"}
        ]
        failures = [
            payload for payload in (*requests, *tools)
            if str(payload.get("status") or "") == "failed" or payload.get("error")
        ]
        total_tokens = 0
        for payload in requests:
            details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
            usage = details.get("usage") if isinstance(details.get("usage"), dict) else {}
            total_tokens += int(
                usage.get("total_tokens")
                or usage.get("total_token_count")
                or 0
            )
        durations = [int(payload.get("duration_ms") or 0) for payload in requests]
        return {
            "requests": len(requests),
            "tool_calls": len(tools),
            "failures": len(failures),
            "success_rate": round(1 - len(failures) / max(1, len(requests) + len(tools)), 4),
            "avg_request_ms": round(sum(durations) / max(1, len(durations)), 2),
            "total_tokens": total_tokens,
        }


class RecoveryProjection:
    name = "recovery"

    def project(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "sequence": int(event.get("sequence") or 0),
                "event_id": str(event.get("subject_event_id") or ""),
                "event_type": str(event.get("event_type") or ""),
                "status": str(event.get("status") or ""),
                "error": str((event.get("payload") or {}).get("error") or ""),
                "recorded_at": str(event.get("recorded_at") or ""),
            }
            for event in events
            if str(event.get("event_type") or "") in {
                "event.recovered", "event.interrupted"
            }
        ]


class ModelHistoryProjection:
    name = "model_history"

    def project(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        history = []
        for event in _latest_snapshots(events):
            payload = event.get("payload") or {}
            if str(payload.get("kind") or "") not in {"REQUEST", "ATTEMPT"}:
                continue
            details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
            source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
            options = details.get("options") if isinstance(details.get("options"), dict) else {}
            model = str(
                details.get("model")
                or source.get("model")
                or options.get("preferred_model")
                or ""
            )
            history.append({
                "event_id": str(event["subject_event_id"]),
                "kind": str(payload.get("kind") or ""),
                "model": model,
                "provider": str(details.get("provider") or source.get("provider") or ""),
                "status": str(payload.get("status") or ""),
                "fallback": bool(details.get("fallback") or details.get("fallback_used")),
                "routing": options,
            })
        return history


class CheckpointProjection:
    """Latest non-sensitive LangGraph checkpoint status for a session."""

    name = "checkpoint"

    def project(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        matching = [
            event for event in events
            if str((event.get("payload") or {}).get("kind") or "") == "CHECKPOINT"
        ]
        if not matching:
            return {}
        event = matching[-1]
        payload = event.get("payload") or {}
        details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
        return {
            "status": str(payload.get("status") or "empty"),
            "thread_id": str(details.get("thread_id") or ""),
            "channel": str(details.get("channel") or ""),
            "next": list(details.get("next") or []),
            "pending_tasks": list(details.get("pending_tasks") or []),
            "checkpoint_created_at": details.get("checkpoint_created_at"),
            "observed_at": str(payload.get("created_at") or event.get("recorded_at") or ""),
            "event_type": str(event.get("event_type") or ""),
            "event_version": int(event.get("event_version") or 0),
        }


class ExecutionLedgerProjection:
    """Rebuild execution attempts and their receipt stream from journal events."""

    name = "execution_ledger"

    def project(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        attempts: dict[str, dict[str, Any]] = {}
        receipts: dict[str, list[dict[str, Any]]] = {}
        order: dict[str, int] = {}
        for event in events:
            payload = event.get("payload") or {}
            if str(payload.get("kind") or "") != "EXECUTION_ATTEMPT":
                continue
            attempt = payload.get("attempt")
            if not isinstance(attempt, dict):
                continue
            attempt_id = str(attempt.get("attempt_id") or event.get("subject_event_id") or "")
            if not attempt_id:
                continue
            attempts[attempt_id] = dict(attempt)
            order[attempt_id] = int(event.get("sequence") or 0)
            receipt = payload.get("receipt")
            if isinstance(receipt, dict) and receipt:
                existing = receipts.setdefault(attempt_id, [])
                receipt_key = (
                    str(receipt.get("event") or ""),
                    str(receipt.get("created_at") or ""),
                    _json_dump(receipt),
                )
                if not any(
                    (
                        str(item.get("event") or ""),
                        str(item.get("created_at") or ""),
                        _json_dump(item),
                    ) == receipt_key
                    for item in existing
                ):
                    existing.append(dict(receipt))
        result = []
        for attempt_id, attempt in attempts.items():
            attempt["receipts"] = receipts.get(attempt_id, [])
            attempt["journal_sequence"] = order.get(attempt_id, 0)
            result.append(attempt)
        return sorted(
            result,
            key=lambda row: (
                str(row.get("claimed_at") or ""),
                int(row.get("journal_sequence") or 0),
            ),
            reverse=True,
        )


class AuditProjection:
    """Project scoped critical-action audit records from the common journal."""

    name = "audit"

    def project(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        entries = []
        for event in events:
            payload = event.get("payload") or {}
            if str(payload.get("kind") or "") != "AUDIT":
                continue
            entry = payload.get("entry")
            if isinstance(entry, dict):
                entries.append(dict(entry))
        return sorted(entries, key=lambda row: str(row.get("timestamp") or ""), reverse=True)


class SessionEventStore:
    """Read and verify the append-only journal written by ``TrajectoryStore``."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else settings.DATA_DIR / "trajectory.sqlite3"

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def append_event(
        self,
        *,
        subject_event_id: str,
        project_id: str,
        session_id: str,
        event_type: str,
        kind: str,
        status: str,
        payload: dict[str, Any] | None = None,
        turn_id: str = "",
        changed_fields: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Append one generic runtime event using the canonical hash contract."""
        subject = str(subject_event_id or "").strip()
        project = str(project_id or "").strip()
        session = str(session_id or "").strip()
        if not subject or not project or not session:
            raise ValueError("Session events require subject_event_id, project_id, and session_id")
        recorded_at = datetime.now(timezone.utc).isoformat()
        canonical_payload = dict(payload or {})
        canonical_payload.setdefault("contract_version", 1)
        canonical_payload.setdefault("project_id", project)
        canonical_payload.setdefault("session_id", session)
        canonical_payload.setdefault("turn_id", str(turn_id or ""))
        canonical_payload.setdefault("kind", str(kind or "").upper())
        canonical_payload.setdefault("status", str(status or ""))
        canonical_payload.setdefault("created_at", recorded_at)
        encoded = _json_dump(canonical_payload)
        checksum = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        ledger_event_id = f"session-event-{uuid.uuid4().hex}"
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT COALESCE(MAX(event_version), 0) AS version "
                "FROM session_event_log WHERE subject_event_id=?",
                (subject,),
            ).fetchone()
            version = int(row["version"] or 0) + 1
            cursor = conn.execute(
                """INSERT INTO session_event_log(
                       ledger_event_id, subject_event_id, event_version,
                       project_id, session_id, turn_id, event_type, kind, status,
                       changed_fields_json, payload_json, payload_sha256, recorded_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ledger_event_id,
                    subject,
                    version,
                    project,
                    session,
                    str(turn_id or ""),
                    str(event_type or "runtime.updated"),
                    str(kind or "").upper(),
                    str(status or ""),
                    _json_dump(sorted({str(field) for field in (changed_fields or [])})),
                    encoded,
                    checksum,
                    recorded_at,
                ),
            )
            sequence = int(cursor.lastrowid)
        return {
            "sequence": sequence,
            "ledger_event_id": ledger_event_id,
            "subject_event_id": subject,
            "event_version": version,
            "project_id": project,
            "session_id": session,
            "event_type": str(event_type or "runtime.updated"),
            "kind": str(kind or "").upper(),
            "status": str(status or ""),
            "payload": canonical_payload,
            "payload_sha256": checksum,
            "recorded_at": recorded_at,
            "integrity_ok": True,
        }

    def list_events(
        self,
        *,
        project_id: str,
        session_id: str,
        after_sequence: int = 0,
        limit: int = 20_000,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM session_event_log
                   WHERE project_id=? AND session_id=? AND sequence>?
                   ORDER BY sequence LIMIT ?""",
                (
                    project_id,
                    session_id,
                    max(0, int(after_sequence)),
                    max(1, min(int(limit), 50_000)),
                ),
            ).fetchall()
        events = []
        for row in rows:
            item = dict(row)
            encoded = str(item.pop("payload_json") or "{}")
            try:
                item["payload"] = json.loads(encoded)
                payload_decoded = True
            except json.JSONDecodeError:
                item["payload"] = {"corrupt_payload": True}
                payload_decoded = False
            try:
                item["changed_fields"] = json.loads(
                    item.pop("changed_fields_json") or "[]"
                )
            except json.JSONDecodeError:
                item["changed_fields"] = []
            item["integrity_ok"] = payload_decoded and (
                hashlib.sha256(encoded.encode("utf-8")).hexdigest()
                == str(item.get("payload_sha256") or "")
            )
            events.append(item)
        return events

    def list_project_events(
        self,
        *,
        project_id: str,
        kinds: set[str] | None = None,
        limit: int = 50_000,
    ) -> list[dict[str, Any]]:
        requested = {str(kind).upper() for kind in (kinds or set()) if str(kind).strip()}
        with self._connect() as conn:
            clauses = ["project_id=?"]
            args: list[Any] = [str(project_id or "")]
            if requested:
                placeholders = ",".join("?" for _ in requested)
                clauses.append(f"kind IN ({placeholders})")
                args.extend(sorted(requested))
            args.append(max(1, min(int(limit), 50_000)))
            rows = conn.execute(
                "SELECT * FROM session_event_log WHERE "
                + " AND ".join(clauses)
                + " ORDER BY sequence LIMIT ?",
                args,
            ).fetchall()
        return self._decode_rows(rows)

    def list_subject_events(self, subject_event_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM session_event_log WHERE subject_event_id=? "
                "ORDER BY event_version",
                (str(subject_event_id or ""),),
            ).fetchall()
        return self._decode_rows(rows)

    @staticmethod
    def _decode_rows(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
        events = []
        for row in rows:
            item = dict(row)
            encoded = str(item.pop("payload_json") or "{}")
            try:
                item["payload"] = json.loads(encoded)
                payload_decoded = True
            except json.JSONDecodeError:
                item["payload"] = {"corrupt_payload": True}
                payload_decoded = False
            try:
                item["changed_fields"] = json.loads(item.pop("changed_fields_json") or "[]")
            except json.JSONDecodeError:
                item["changed_fields"] = []
            item["integrity_ok"] = payload_decoded and (
                hashlib.sha256(encoded.encode("utf-8")).hexdigest()
                == str(item.get("payload_sha256") or "")
            )
            events.append(item)
        return events

    def project(
        self,
        projection: SessionProjection,
        *,
        project_id: str,
        session_id: str,
    ) -> Any:
        return projection.project(self.list_events(
            project_id=project_id,
            session_id=session_id,
        ))

    def projection_snapshot(self, *, project_id: str, session_id: str) -> dict[str, Any]:
        events = self.list_events(project_id=project_id, session_id=session_id)
        projections: tuple[SessionProjection, ...] = (
            TranscriptProjection(),
            ToolRunProjection(),
            MetricsProjection(),
            RecoveryProjection(),
            ModelHistoryProjection(),
            CheckpointProjection(),
            ExecutionLedgerProjection(),
            AuditProjection(),
        )
        return {
            "project_id": project_id,
            "session_id": session_id,
            "event_count": len(events),
            "last_sequence": int(events[-1]["sequence"] if events else 0),
            "projections": {
                projection.name: projection.project(events) for projection in projections
            },
        }

    def verify_materialized_projection(
        self,
        *,
        project_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        events = self.list_events(project_id=project_id, session_id=session_id)
        latest = [
            event for event in _latest_snapshots(events)
            if (event.get("payload") or {}).get("projection_sequence") is not None
        ]
        drifted = []
        with self._connect() as conn:
            projection_rows = conn.execute(
                """SELECT * FROM trajectory_events
                   WHERE project_id=? AND session_id=?""",
                (project_id, session_id),
            ).fetchall()
            projections_by_id = {
                str(row["event_id"]): row for row in projection_rows
            }
            for event in latest:
                row = projections_by_id.get(str(event["subject_event_id"]))
                if not row:
                    drifted.append({
                        "event_id": event["subject_event_id"],
                        "reason": "missing_projection_row",
                    })
                    continue
                current = TrajectoryStore._session_event_payload(row)
                if _json_dump(current) != _json_dump(event.get("payload") or {}):
                    drifted.append({
                        "event_id": event["subject_event_id"],
                        "reason": "projection_drift",
                    })
        corrupt = [
            str(event.get("ledger_event_id") or "")
            for event in events if not event.get("integrity_ok")
        ]
        return {
            "ok": not drifted and not corrupt,
            "event_versions": len(events),
            "subjects": len(latest),
            "drifted": drifted,
            "corrupt_ledger_events": corrupt,
        }


_STORE: SessionEventStore | None = None
_STORE_LOCK = threading.Lock()


def get_session_event_store() -> SessionEventStore:
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                # Ensure the writer has created/backfilled the ledger before a
                # read projection is requested.
                from remy.core.trajectory_store import get_trajectory_store

                trajectory = get_trajectory_store()
                _STORE = SessionEventStore(trajectory.path)
    return _STORE
