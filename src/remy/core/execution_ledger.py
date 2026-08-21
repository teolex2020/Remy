"""Durable attempt ledger and background-result inbox.

The ledger is deliberately separate from domain records.  A workflow, research
project, or scheduled job may have many attempts, while every attempt has one
immutable terminal outcome.  Interrupted attempts become ``unknown``; callers
must explicitly decide whether their idempotency class permits another attempt.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from remy.config.settings import settings

logger = logging.getLogger(__name__)

ACTIVE_STATES = {"claimed", "running"}
TERMINAL_STATES = {
    "completed",
    "completed_with_limits",
    "failed",
    "cancelled",
    "blocked",
    "unknown",
}
IDEMPOTENCY_CLASSES = {"read_only", "idempotent", "side_effecting"}

_PROCESS_FINGERPRINT = uuid.uuid4().hex
_INIT_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process_started_at(pid: int) -> str:
    try:
        import psutil

        return f"{psutil.Process(pid).create_time():.6f}"
    except Exception:
        return _PROCESS_FINGERPRINT if pid == os.getpid() else ""


def _owner_is_alive(pid: int, started_at: str) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        current = _process_started_at(pid)
        return not started_at or current == started_at
    try:
        import psutil

        process = psutil.Process(pid)
        if not process.is_running():
            return False
        return not started_at or f"{process.create_time():.6f}" == started_at
    except Exception:
        return False


class ExecutionLedger:
    """Small SQLite ledger safe for use from worker threads and async wrappers."""

    def __init__(self, path: Path | str | None = None, *, event_store=None):
        self.path = Path(path) if path else settings.DATA_DIR / "execution_ledger.sqlite3"
        self._event_store = event_store
        self._use_default_event_store = path is None and event_store is None
        self._ensure_schema()

    def _session_events(self):
        if self._event_store is not None:
            return self._event_store
        if not self._use_default_event_store:
            return None
        try:
            from remy.core.session_event_store import get_session_event_store

            self._event_store = get_session_event_store()
        except Exception as exc:
            logger.debug("Execution journal unavailable: %s", exc)
            return None
        return self._event_store

    def _mirror_attempt(
        self,
        attempt_id: str,
        event_type: str,
        *,
        changed_fields: tuple[str, ...] = (),
    ) -> None:
        store = self._session_events()
        if store is None:
            return
        try:
            attempt = self._get_legacy(attempt_id)
            if not attempt:
                return
            receipts = list(attempt.pop("receipts", []) or [])
            project_id = str(attempt.get("owner_project_id") or "").strip()
            if not project_id:
                return
            session_id = str(attempt.get("session_id") or "").strip()
            if not session_id:
                session_id = (
                    f"execution:{attempt.get('kind') or 'job'}:"
                    f"{attempt.get('job_id') or attempt_id}"
                )
            store.append_event(
                subject_event_id=attempt_id,
                project_id=project_id,
                session_id=session_id,
                event_type=event_type,
                kind="EXECUTION_ATTEMPT",
                status=str(attempt.get("state") or ""),
                payload={
                    "attempt": attempt,
                    "receipt": receipts[-1] if receipts else {},
                },
                changed_fields=changed_fields,
            )
        except Exception as exc:
            # The coordination ledger remains authoritative for mutations until
            # projection parity is proven in production.
            logger.warning("Could not mirror execution attempt %s: %s", attempt_id, exc)

    def _projected_attempt(self, attempt_id: str) -> dict[str, Any] | None:
        store = self._session_events()
        if store is None:
            return None
        try:
            from remy.core.session_event_store import ExecutionLedgerProjection

            attempts = ExecutionLedgerProjection().project(
                store.list_subject_events(attempt_id)
            )
            return attempts[0] if attempts else None
        except Exception as exc:
            logger.debug("Execution projection read failed for %s: %s", attempt_id, exc)
            return None

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _ensure_schema(self) -> None:
        with _INIT_LOCK, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS execution_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    idempotency_class TEXT NOT NULL,
                    state TEXT NOT NULL,
                    owner_pid INTEGER NOT NULL,
                    owner_started_at TEXT NOT NULL,
                    owner_project_id TEXT NOT NULL DEFAULT '',
                    brain_id TEXT NOT NULL DEFAULT '',
                    session_id TEXT NOT NULL DEFAULT '',
                    channel TEXT NOT NULL DEFAULT '',
                    claimed_at TEXT NOT NULL,
                    started_at TEXT NOT NULL DEFAULT '',
                    heartbeat_at TEXT NOT NULL DEFAULT '',
                    finished_at TEXT NOT NULL DEFAULT '',
                    output_ref TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_attempt_job
                    ON execution_attempts(kind, job_id, claimed_at DESC);
                CREATE INDEX IF NOT EXISTS idx_attempt_state
                    ON execution_attempts(state);
                CREATE TABLE IF NOT EXISTS execution_receipts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    attempt_id TEXT NOT NULL,
                    event TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    data_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_receipt_attempt
                    ON execution_receipts(attempt_id, id);
                CREATE TABLE IF NOT EXISTS continuation_inbox (
                    continuation_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    owner_project_id TEXT NOT NULL DEFAULT '',
                    brain_id TEXT NOT NULL DEFAULT '',
                    kind TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    delivered_at TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_continuation_pending
                    ON continuation_inbox(session_id, delivered_at, created_at);
                """
            )
            # Existing installations predate project-scoped MicroBrains. SQLite
            # does not apply new CREATE TABLE columns to an existing table, so
            # migrate in place and retain every attempt/continuation.
            self._ensure_column(
                conn,
                "execution_attempts",
                "owner_project_id",
                "TEXT NOT NULL DEFAULT ''",
            )
            self._ensure_column(
                conn,
                "execution_attempts",
                "brain_id",
                "TEXT NOT NULL DEFAULT ''",
            )
            self._ensure_column(
                conn,
                "continuation_inbox",
                "owner_project_id",
                "TEXT NOT NULL DEFAULT ''",
            )
            self._ensure_column(
                conn,
                "continuation_inbox",
                "brain_id",
                "TEXT NOT NULL DEFAULT ''",
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_continuation_project_pending "
                "ON continuation_inbox(owner_project_id, session_id, delivered_at, created_at)"
            )

    @staticmethod
    def _ensure_column(
        conn: sqlite3.Connection,
        table: str,
        column: str,
        declaration: str,
    ) -> None:
        allowed = {
            ("execution_attempts", "owner_project_id"),
            ("execution_attempts", "brain_id"),
            ("continuation_inbox", "owner_project_id"),
            ("continuation_inbox", "brain_id"),
        }
        if (table, column) not in allowed:
            raise ValueError(f"Unsupported ledger migration: {table}.{column}")
        columns = {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def claim(
        self,
        *,
        kind: str,
        job_id: str,
        idempotency_class: str = "side_effecting",
        owner_project_id: str = "",
        brain_id: str = "",
        session_id: str = "",
        channel: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if idempotency_class not in IDEMPOTENCY_CLASSES:
            raise ValueError(f"Unsupported idempotency class: {idempotency_class}")
        owner_project_id = str(owner_project_id or "").strip()
        brain_id = str(brain_id or "").strip()
        if not owner_project_id or not brain_id:
            raise ValueError(
                "Execution attempts require owner_project_id and brain_id"
            )
        attempt_id = f"attempt-{uuid.uuid4().hex}"
        now = _now()
        pid = os.getpid()
        started_at = _process_started_at(pid)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            active = conn.execute(
                "SELECT attempt_id FROM execution_attempts "
                "WHERE kind=? AND job_id=? AND owner_project_id=? "
                "AND state IN ('claimed','running') LIMIT 1",
                (kind, job_id, owner_project_id),
            ).fetchone()
            if active:
                raise RuntimeError(f"Job already has an active attempt: {active['attempt_id']}")
            conn.execute(
                """INSERT INTO execution_attempts(
                    attempt_id, kind, job_id, idempotency_class, state,
                    owner_pid, owner_started_at, owner_project_id, brain_id,
                    session_id, channel,
                    claimed_at, heartbeat_at, metadata_json
                ) VALUES (?, ?, ?, ?, 'claimed', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    attempt_id,
                    kind,
                    job_id,
                    idempotency_class,
                    pid,
                    started_at,
                    owner_project_id,
                    brain_id,
                    session_id,
                    channel,
                    now,
                    now,
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            self._insert_receipt(conn, attempt_id, "claimed", {"owner_pid": pid})
        self._mirror_attempt(
            attempt_id,
            "execution.claimed",
            changed_fields=("state", "claimed_at", "owner_pid"),
        )
        return self.get(attempt_id) or {}

    def mark_running(self, attempt_id: str) -> dict[str, Any]:
        now = _now()
        with self._connect() as conn:
            updated = conn.execute(
                "UPDATE execution_attempts SET state='running', started_at=CASE "
                "WHEN started_at='' THEN ? ELSE started_at END, heartbeat_at=? "
                "WHERE attempt_id=? AND state='claimed'",
                (now, now, attempt_id),
            ).rowcount
            if not updated:
                self._require_active(conn, attempt_id)
            self._insert_receipt(conn, attempt_id, "running", {})
        self._mirror_attempt(
            attempt_id,
            "execution.running",
            changed_fields=("state", "started_at", "heartbeat_at"),
        )
        return self.get(attempt_id) or {}

    def heartbeat(self, attempt_id: str, data: dict[str, Any] | None = None) -> None:
        with self._connect() as conn:
            self._require_active(conn, attempt_id)
            conn.execute(
                "UPDATE execution_attempts SET heartbeat_at=? WHERE attempt_id=?",
                (_now(), attempt_id),
            )
            self._insert_receipt(conn, attempt_id, "heartbeat", data or {})
        self._mirror_attempt(
            attempt_id,
            "execution.heartbeat",
            changed_fields=("heartbeat_at",),
        )

    def patch_metadata(
        self,
        attempt_id: str,
        metadata: dict[str, Any],
        *,
        event: str = "state_updated",
        event_data: dict[str, Any] | None = None,
        heartbeat: bool = True,
    ) -> dict[str, Any]:
        """Merge durable operator state into an active attempt.

        Run envelopes use this method for observable progress without creating a
        second execution database.  Domain-specific records (experiments,
        pipelines, and automations) remain authoritative for their own output.
        """
        now = _now()
        with self._connect() as conn:
            row = self._require_active(conn, attempt_id)
            existing = json.loads(row["metadata_json"] or "{}")
            existing.update(metadata or {})
            if heartbeat:
                conn.execute(
                    "UPDATE execution_attempts SET metadata_json=?, heartbeat_at=? "
                    "WHERE attempt_id=?",
                    (json.dumps(existing, ensure_ascii=False), now, attempt_id),
                )
            else:
                conn.execute(
                    "UPDATE execution_attempts SET metadata_json=? WHERE attempt_id=?",
                    (json.dumps(existing, ensure_ascii=False), attempt_id),
                )
            self._insert_receipt(conn, attempt_id, event, event_data or {})
        self._mirror_attempt(
            attempt_id,
            f"execution.{event}",
            changed_fields=("metadata", "heartbeat_at") if heartbeat else ("metadata",),
        )
        return self.get(attempt_id) or {}

    def append_receipt(self, attempt_id: str, event: str, data: dict[str, Any] | None = None) -> None:
        with self._connect() as conn:
            if not conn.execute(
                "SELECT 1 FROM execution_attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone():
                raise KeyError(attempt_id)
            self._insert_receipt(conn, attempt_id, event, data or {})
        self._mirror_attempt(attempt_id, f"execution.{event}")

    def finish(
        self,
        attempt_id: str,
        state: str,
        *,
        output_ref: str = "",
        error: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if state not in TERMINAL_STATES:
            raise ValueError(f"Unsupported terminal state: {state}")
        now = _now()
        with self._connect() as conn:
            row = self._require_active(conn, attempt_id)
            existing_meta = json.loads(row["metadata_json"] or "{}")
            existing_meta.update(metadata or {})
            conn.execute(
                """UPDATE execution_attempts SET state=?, finished_at=?, heartbeat_at=?,
                    output_ref=?, error=?, metadata_json=? WHERE attempt_id=?""",
                (
                    state,
                    now,
                    now,
                    output_ref,
                    error[:4000],
                    json.dumps(existing_meta, ensure_ascii=False),
                    attempt_id,
                ),
            )
            self._insert_receipt(conn, attempt_id, state, {"output_ref": output_ref, "error": error[:1000]})
        self._mirror_attempt(
            attempt_id,
            f"execution.{state}",
            changed_fields=("state", "finished_at", "output_ref", "error", "metadata"),
        )
        return self.get(attempt_id) or {}

    def replace_terminal_metadata(
        self, attempt_id: str, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge metadata after recovery has made an attempt terminal."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT metadata_json FROM execution_attempts WHERE attempt_id=?",
                (attempt_id,),
            ).fetchone()
            if not row:
                raise KeyError(attempt_id)
            existing = json.loads(row["metadata_json"] or "{}")
            existing.update(metadata or {})
            conn.execute(
                "UPDATE execution_attempts SET metadata_json=? WHERE attempt_id=?",
                (json.dumps(existing, ensure_ascii=False), attempt_id),
            )
            self._insert_receipt(conn, attempt_id, "metadata_recovered", {})
        self._mirror_attempt(
            attempt_id,
            "execution.metadata_recovered",
            changed_fields=("metadata",),
        )
        return self.get(attempt_id) or {}

    def recover_orphans(self, *, kind: str | None = None) -> list[dict[str, Any]]:
        """Mark dead-owner attempts unknown. Never retries them automatically."""
        recovered: list[dict[str, Any]] = []
        recovered_ids: list[str] = []
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            sql = "SELECT * FROM execution_attempts WHERE state IN ('claimed','running')"
            args: tuple[Any, ...] = ()
            if kind:
                sql += " AND kind=?"
                args = (kind,)
            for row in conn.execute(sql, args).fetchall():
                item = dict(row)
                if _owner_is_alive(int(row["owner_pid"]), str(row["owner_started_at"])):
                    continue
                now = _now()
                conn.execute(
                    "UPDATE execution_attempts SET state='unknown', finished_at=?, heartbeat_at=?, "
                    "error=? WHERE attempt_id=?",
                    (now, now, "Owner process ended before a terminal receipt", row["attempt_id"]),
                )
                self._insert_receipt(conn, row["attempt_id"], "unknown", {"reason": "owner_not_alive"})
                item["state"] = "unknown"
                recovered.append(self._decode_attempt(item))
                recovered_ids.append(str(row["attempt_id"]))
        for attempt_id in recovered_ids:
            self._mirror_attempt(
                attempt_id,
                "execution.recovered",
                changed_fields=("state", "finished_at", "error"),
            )
        return [self.get(attempt_id) or item for attempt_id, item in zip(recovered_ids, recovered)]

    def get(self, attempt_id: str) -> dict[str, Any] | None:
        projected = self._projected_attempt(attempt_id)
        if projected is not None:
            return projected
        legacy = self._get_legacy(attempt_id)
        if legacy is not None:
            self._mirror_attempt(attempt_id, "execution.imported")
            return self._projected_attempt(attempt_id) or legacy
        return None

    def _get_legacy(self, attempt_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM execution_attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
            if not row:
                return None
            item = self._decode_attempt(dict(row))
            receipts = conn.execute(
                "SELECT event, created_at, data_json FROM execution_receipts "
                "WHERE attempt_id=? ORDER BY id", (attempt_id,)
            ).fetchall()
            item["receipts"] = [
                {"event": r["event"], "created_at": r["created_at"], **json.loads(r["data_json"] or "{}")}
                for r in receipts
            ]
            return item

    def list_attempts(
        self,
        *,
        state: str = "",
        kind: str = "",
        owner_project_id: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return recent attempts for runtime inspection and operator UI."""
        legacy = self._list_attempts_legacy(
            state=state,
            kind=kind,
            owner_project_id=owner_project_id,
            limit=limit,
        )
        store = self._session_events() if owner_project_id else None
        if store is None:
            return legacy
        try:
            from remy.core.session_event_store import ExecutionLedgerProjection

            for attempt in legacy:
                attempt_id = str(attempt.get("attempt_id") or "")
                if attempt_id and not store.list_subject_events(attempt_id):
                    self._mirror_attempt(attempt_id, "execution.imported")
            projected = ExecutionLedgerProjection().project(
                store.list_project_events(
                    project_id=owner_project_id,
                    kinds={"EXECUTION_ATTEMPT"},
                )
            )
            if state:
                projected = [row for row in projected if row.get("state") == state]
            if kind:
                projected = [row for row in projected if row.get("kind") == kind]
            return projected[: max(1, min(int(limit), 500))]
        except Exception as exc:
            logger.warning("Execution projection list failed; using legacy ledger: %s", exc)
            return legacy

    def _list_attempts_legacy(
        self,
        *,
        state: str = "",
        kind: str = "",
        owner_project_id: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        args: list[Any] = []
        if state:
            clauses.append("state=?")
            args.append(state)
        if kind:
            clauses.append("kind=?")
            args.append(kind)
        if owner_project_id:
            clauses.append("owner_project_id=?")
            args.append(owner_project_id)
        sql = "SELECT * FROM execution_attempts"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY claimed_at DESC LIMIT ?"
        args.append(max(1, min(int(limit), 500)))
        with self._connect() as conn:
            return [self._decode_attempt(dict(row)) for row in conn.execute(sql, args).fetchall()]

    def enqueue_continuation(
        self,
        *,
        session_id: str,
        owner_project_id: str = "",
        brain_id: str = "",
        kind: str,
        source_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        owner_project_id = str(owner_project_id or "").strip()
        brain_id = str(brain_id or "").strip()
        if not owner_project_id or not brain_id:
            raise ValueError(
                "Continuations require owner_project_id and brain_id"
            )
        continuation_id = f"continuation-{uuid.uuid4().hex}"
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO continuation_inbox(
                    continuation_id, session_id, owner_project_id, brain_id,
                    kind, source_id, content,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    continuation_id,
                    session_id,
                    owner_project_id,
                    brain_id,
                    kind,
                    source_id,
                    content,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    _now(),
                ),
            )
        return continuation_id

    def consume_continuations(
        self,
        session_id: str,
        limit: int = 20,
        *,
        delivery_targets: set[str] | None = None,
        include_unmatched_sessions: bool = False,
        owner_project_id: str = "",
        brain_id: str = "",
        include_legacy_unscoped: bool = False,
    ) -> list[dict[str, Any]]:
        owner_project_id = str(owner_project_id or "").strip()
        brain_id = str(brain_id or "").strip()
        if not owner_project_id or not brain_id:
            raise ValueError(
                "Continuation delivery requires owner_project_id and brain_id"
            )
        now = _now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            clauses = ["delivered_at=''"]
            args: list[Any] = []
            if not include_unmatched_sessions:
                clauses.append("session_id=?")
                args.append(session_id)
            if owner_project_id:
                if include_legacy_unscoped:
                    clauses.append("owner_project_id IN (?, '')")
                else:
                    clauses.append("owner_project_id=?")
                args.append(owner_project_id)
            if brain_id:
                if include_legacy_unscoped:
                    clauses.append("brain_id IN (?, '')")
                else:
                    clauses.append("brain_id=?")
                args.append(brain_id)
            query_limit = max(1, min(int(limit), 100))
            args.append(query_limit)
            rows = conn.execute(
                "SELECT * FROM continuation_inbox WHERE "
                + " AND ".join(clauses)
                + " ORDER BY created_at LIMIT ?",
                args,
            ).fetchall()
            decoded = [self._decode_continuation(dict(row)) for row in rows]
            if delivery_targets:
                allowed = {str(item).lower() for item in delivery_targets}
                decoded = [
                    item for item in decoded
                    if str(item.get("metadata", {}).get("delivery_target", "web")).lower() in allowed
                ]
            decoded = decoded[: max(1, min(int(limit), 100))]
            ids = [item["continuation_id"] for item in decoded]
            if ids:
                conn.executemany(
                    "UPDATE continuation_inbox SET delivered_at=? WHERE continuation_id=?",
                    [(now, item_id) for item_id in ids],
                )
            return decoded

    @staticmethod
    def _insert_receipt(
        conn: sqlite3.Connection, attempt_id: str, event: str, data: dict[str, Any]
    ) -> None:
        conn.execute(
            "INSERT INTO execution_receipts(attempt_id, event, created_at, data_json) VALUES (?, ?, ?, ?)",
            (attempt_id, event, _now(), json.dumps(data, ensure_ascii=False)),
        )

    @staticmethod
    def _require_active(conn: sqlite3.Connection, attempt_id: str) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM execution_attempts WHERE attempt_id=?", (attempt_id,)
        ).fetchone()
        if not row:
            raise KeyError(attempt_id)
        if row["state"] not in ACTIVE_STATES:
            raise RuntimeError(f"Attempt {attempt_id} is already terminal: {row['state']}")
        return row

    @staticmethod
    def _decode_attempt(item: dict[str, Any]) -> dict[str, Any]:
        item["metadata"] = json.loads(item.pop("metadata_json", "{}") or "{}")
        return item

    @staticmethod
    def _decode_continuation(item: dict[str, Any]) -> dict[str, Any]:
        item["metadata"] = json.loads(item.pop("metadata_json", "{}") or "{}")
        return item


def get_execution_ledger() -> ExecutionLedger:
    """Return a ledger rooted in the current (possibly test-overridden) DATA_DIR."""
    return ExecutionLedger()
