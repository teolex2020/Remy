"""Durable lifecycle and inbox for continuable delegated child sessions."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from remy.config.settings import settings


CHILD_ACTIVE_STATUSES = frozenset({"starting", "running", "interrupt_requested"})
CHILD_CONTINUABLE_STATUSES = frozenset({"ready", "interrupted", "failed"})
CHILD_STATUSES = CHILD_ACTIVE_STATUSES | CHILD_CONTINUABLE_STATUSES
MESSAGE_DIRECTIONS = frozenset({"parent_to_child", "child_to_parent"})
MESSAGE_KINDS = frozenset({"initial", "follow_up", "interrupt", "report", "settlement"})
_SCHEMA_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class ChildSessionStore:
    """SQLite-backed child identity, attempt history, inbox, and lifecycle events."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else settings.DATA_DIR / "child_sessions.sqlite3"
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _ensure_schema(self) -> None:
        with _SCHEMA_LOCK, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS child_sessions (
                    child_id TEXT PRIMARY KEY,
                    owner_project_id TEXT NOT NULL,
                    brain_id TEXT NOT NULL,
                    parent_session_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    generation INTEGER NOT NULL DEFAULT 0,
                    current_run_id TEXT NOT NULL DEFAULT '',
                    latest_attempt_id TEXT NOT NULL DEFAULT '',
                    idempotency_class TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    latest_report_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_settled_at TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_child_parent
                    ON child_sessions(owner_project_id, parent_session_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS child_attempts (
                    child_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    run_id TEXT NOT NULL UNIQUE,
                    attempt_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    report_json TEXT NOT NULL DEFAULT '{}',
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(child_id, generation),
                    FOREIGN KEY(child_id) REFERENCES child_sessions(child_id)
                );
                CREATE INDEX IF NOT EXISTS idx_child_attempts
                    ON child_attempts(child_id, generation);
                CREATE TABLE IF NOT EXISTS child_inbox (
                    message_id TEXT PRIMARY KEY,
                    child_id TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    consumed_at TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(child_id) REFERENCES child_sessions(child_id)
                );
                CREATE INDEX IF NOT EXISTS idx_child_inbox_pending
                    ON child_inbox(child_id, direction, consumed_at, created_at);
                CREATE TABLE IF NOT EXISTS child_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    event_key TEXT NOT NULL UNIQUE,
                    child_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    run_id TEXT NOT NULL DEFAULT '',
                    data_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(child_id) REFERENCES child_sessions(child_id)
                );
                CREATE INDEX IF NOT EXISTS idx_child_events
                    ON child_events(child_id, sequence);
                """
            )

    @staticmethod
    def _decode_session(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        item = dict(row)
        item["spec"] = json.loads(item.pop("spec_json", "[]") or "[]")
        item["latest_report"] = json.loads(
            item.pop("latest_report_json", "{}") or "{}"
        )
        item["can_interrupt"] = item.get("status") in CHILD_ACTIVE_STATUSES
        item["can_follow_up"] = item.get("status") in CHILD_STATUSES
        item["can_resume"] = item.get("status") in CHILD_CONTINUABLE_STATUSES
        return item

    @staticmethod
    def _decode_message(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        item = dict(row)
        item["metadata"] = json.loads(item.pop("metadata_json", "{}") or "{}")
        return item

    @staticmethod
    def _decode_event(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        item = dict(row)
        item["data"] = json.loads(item.pop("data_json", "{}") or "{}")
        return item

    @staticmethod
    def _decode_attempt(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        item = dict(row)
        item["report"] = json.loads(item.pop("report_json", "{}") or "{}")
        return item

    @staticmethod
    def _append_event(
        conn: sqlite3.Connection,
        *,
        child_id: str,
        event_type: str,
        status: str,
        event_key: str = "",
        run_id: str = "",
        data: dict[str, Any] | None = None,
    ) -> str:
        event_id = f"child-event-{uuid.uuid4().hex}"
        key = event_key or event_id
        conn.execute(
            """INSERT OR IGNORE INTO child_events(
                   event_id, event_key, child_id, event_type, status,
                   run_id, data_json, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                key,
                child_id,
                event_type,
                status,
                run_id,
                _json(data or {}),
                _now(),
            ),
        )
        return event_id

    def create(
        self,
        *,
        owner_project_id: str,
        brain_id: str,
        parent_session_id: str,
        spec: list[dict[str, Any]],
        idempotency_class: str,
        kind: str = "worker_group",
    ) -> dict[str, Any]:
        owner = str(owner_project_id or "").strip()
        brain = str(brain_id or "").strip()
        if not owner or not brain:
            raise ValueError("Child sessions require owner_project_id and brain_id")
        if not spec:
            raise ValueError("Child sessions require a resumable task specification")
        child_id = f"child-{uuid.uuid4().hex[:16]}"
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO child_sessions(
                       child_id, owner_project_id, brain_id, parent_session_id,
                       kind, status, idempotency_class, spec_json,
                       created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, 'ready', ?, ?, ?, ?)""",
                (
                    child_id,
                    owner,
                    brain,
                    str(parent_session_id or ""),
                    str(kind or "worker_group"),
                    str(idempotency_class or "side_effecting"),
                    _json(spec),
                    now,
                    now,
                ),
            )
            self._append_event(
                conn,
                child_id=child_id,
                event_type="child.created",
                status="ready",
                event_key=f"{child_id}:created",
                data={"parent_session_id": str(parent_session_id or ""), "kind": kind},
            )
        return self.get(child_id, owner_project_id=owner) or {}

    def get(self, child_id: str, *, owner_project_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM child_sessions WHERE child_id=? AND owner_project_id=?",
                (str(child_id), str(owner_project_id)),
            ).fetchone()
        return self._decode_session(row) if row else None

    def list(
        self,
        *,
        owner_project_id: str,
        parent_session_id: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses = ["owner_project_id=?"]
        args: list[Any] = [str(owner_project_id)]
        if parent_session_id:
            clauses.append("parent_session_id=?")
            args.append(str(parent_session_id))
        args.append(max(1, min(int(limit), 500)))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM child_sessions WHERE "
                + " AND ".join(clauses)
                + " ORDER BY updated_at DESC LIMIT ?",
                args,
            ).fetchall()
        return [self._decode_session(row) for row in rows]

    def claim_attempt(self, child_id: str, *, owner_project_id: str, reason: str) -> int:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM child_sessions WHERE child_id=? AND owner_project_id=?",
                (str(child_id), str(owner_project_id)),
            ).fetchone()
            if not row:
                raise KeyError(child_id)
            if row["status"] in CHILD_ACTIVE_STATUSES:
                raise RuntimeError(f"Child session {child_id} already has an active attempt")
            generation = int(row["generation"] or 0) + 1
            conn.execute(
                """UPDATE child_sessions
                   SET status='starting', generation=?, current_run_id='',
                       latest_attempt_id='', updated_at=? WHERE child_id=?""",
                (generation, _now(), child_id),
            )
            self._append_event(
                conn,
                child_id=child_id,
                event_type="child.attempt_claimed",
                status="starting",
                event_key=f"{child_id}:generation:{generation}:claimed",
                data={"generation": generation, "reason": str(reason or "")},
            )
        return generation

    def bind_attempt(
        self,
        child_id: str,
        *,
        owner_project_id: str,
        generation: int,
        run_id: str,
        attempt_id: str,
        reason: str,
    ) -> dict[str, Any]:
        now = _now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT generation, status FROM child_sessions "
                "WHERE child_id=? AND owner_project_id=?",
                (child_id, owner_project_id),
            ).fetchone()
            if not row:
                raise KeyError(child_id)
            if int(row["generation"] or 0) != int(generation):
                raise RuntimeError("Child generation changed while binding attempt")
            conn.execute(
                """INSERT INTO child_attempts(
                       child_id, generation, run_id, attempt_id, status,
                       reason, started_at
                   ) VALUES (?, ?, ?, ?, 'running', ?, ?)""",
                (child_id, int(generation), run_id, attempt_id, str(reason or ""), now),
            )
            conn.execute(
                """UPDATE child_sessions SET status='running', current_run_id=?,
                       latest_attempt_id=?, updated_at=? WHERE child_id=?""",
                (run_id, attempt_id, now, child_id),
            )
            self._append_event(
                conn,
                child_id=child_id,
                event_type="child.attempt_started",
                status="running",
                event_key=f"{child_id}:{run_id}:started",
                run_id=run_id,
                data={"attempt_id": attempt_id, "generation": int(generation), "reason": reason},
            )
        return self.get(child_id, owner_project_id=owner_project_id) or {}

    def mark_interrupt_requested(
        self,
        child_id: str,
        *,
        owner_project_id: str,
        reason: str,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM child_sessions WHERE child_id=? AND owner_project_id=?",
                (child_id, owner_project_id),
            ).fetchone()
            if not row:
                raise KeyError(child_id)
            if row["status"] not in CHILD_ACTIVE_STATUSES:
                return self._decode_session(row)
            conn.execute(
                "UPDATE child_sessions SET status='interrupt_requested', updated_at=? "
                "WHERE child_id=?",
                (_now(), child_id),
            )
            self._append_event(
                conn,
                child_id=child_id,
                event_type="child.interrupt_requested",
                status="interrupt_requested",
                event_key=f"{child_id}:{row['current_run_id']}:interrupt",
                run_id=str(row["current_run_id"] or ""),
                data={"reason": str(reason or "")},
            )
        return self.get(child_id, owner_project_id=owner_project_id) or {}

    def settle_attempt(
        self,
        child_id: str,
        *,
        owner_project_id: str,
        run_id: str,
        status: str,
        report: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        child_status = (
            "ready"
            if status in {"completed", "completed_with_limits"}
            else "interrupted"
            if status in {"cancelled", "interrupted"}
            else "failed"
        )
        now = _now()
        inserted = False
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM child_sessions WHERE child_id=? AND owner_project_id=?",
                (child_id, owner_project_id),
            ).fetchone()
            if not row:
                raise KeyError(child_id)
            existing = conn.execute(
                "SELECT event_id FROM child_events WHERE event_key=?",
                (f"{child_id}:{run_id}:settled",),
            ).fetchone()
            if not existing:
                inserted = True
                conn.execute(
                    """UPDATE child_attempts SET status=?, report_json=?, finished_at=?
                       WHERE child_id=? AND run_id=?""",
                    (str(status), _json(report), now, child_id, run_id),
                )
                conn.execute(
                    """UPDATE child_sessions SET status=?, current_run_id='',
                           latest_report_json=?, updated_at=?, last_settled_at=?
                       WHERE child_id=?""",
                    (child_status, _json(report), now, now, child_id),
                )
                self._append_event(
                    conn,
                    child_id=child_id,
                    event_type="child.settled",
                    status=child_status,
                    event_key=f"{child_id}:{run_id}:settled",
                    run_id=run_id,
                    data={"attempt_status": status, "report": report},
                )
        return self.get(child_id, owner_project_id=owner_project_id) or {}, inserted

    def enqueue_message(
        self,
        child_id: str,
        *,
        owner_project_id: str,
        direction: str,
        kind: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if direction not in MESSAGE_DIRECTIONS:
            raise ValueError(f"Unsupported child message direction: {direction}")
        if kind not in MESSAGE_KINDS:
            raise ValueError(f"Unsupported child message kind: {kind}")
        clean_content = str(content or "").strip()
        if not clean_content:
            raise ValueError("Child message content is required")
        if len(clean_content) > 20_000:
            raise ValueError("Child message exceeds 20000 characters")
        message_id = f"child-message-{uuid.uuid4().hex}"
        now = _now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status FROM child_sessions WHERE child_id=? AND owner_project_id=?",
                (child_id, owner_project_id),
            ).fetchone()
            if not row:
                raise KeyError(child_id)
            conn.execute(
                """INSERT INTO child_inbox(
                       message_id, child_id, direction, kind, content,
                       metadata_json, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    message_id,
                    child_id,
                    direction,
                    kind,
                    clean_content,
                    _json(metadata or {}),
                    now,
                ),
            )
            conn.execute(
                "UPDATE child_sessions SET updated_at=? WHERE child_id=?",
                (now, child_id),
            )
            self._append_event(
                conn,
                child_id=child_id,
                event_type=f"child.message.{kind}",
                status=str(row["status"]),
                event_key=f"{child_id}:{message_id}",
                data={
                    "message_id": message_id,
                    "direction": direction,
                    "kind": kind,
                    "content_chars": len(clean_content),
                },
            )
        return {
            "message_id": message_id,
            "child_id": child_id,
            "direction": direction,
            "kind": kind,
            "content": clean_content,
            "metadata": dict(metadata or {}),
            "created_at": now,
            "consumed_at": "",
        }

    def claim_messages(
        self,
        child_id: str,
        *,
        owner_project_id: str,
        direction: str,
        kinds: Iterable[str] = (),
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if direction not in MESSAGE_DIRECTIONS:
            raise ValueError(f"Unsupported child message direction: {direction}")
        selected_kinds = sorted({str(kind) for kind in kinds})
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            owner = conn.execute(
                "SELECT 1 FROM child_sessions WHERE child_id=? AND owner_project_id=?",
                (child_id, owner_project_id),
            ).fetchone()
            if not owner:
                raise KeyError(child_id)
            clauses = ["child_id=?", "direction=?", "consumed_at='' "]
            args: list[Any] = [child_id, direction]
            if selected_kinds:
                clauses.append("kind IN (" + ",".join("?" for _ in selected_kinds) + ")")
                args.extend(selected_kinds)
            args.append(max(1, min(int(limit), 100)))
            rows = conn.execute(
                "SELECT * FROM child_inbox WHERE "
                + " AND ".join(clauses)
                + " ORDER BY created_at LIMIT ?",
                args,
            ).fetchall()
            consumed_at = _now()
            if rows:
                conn.executemany(
                    "UPDATE child_inbox SET consumed_at=? WHERE message_id=?",
                    [(consumed_at, row["message_id"]) for row in rows],
                )
        messages = [self._decode_message(row) for row in rows]
        for message in messages:
            message["consumed_at"] = consumed_at
        return messages

    def release_messages(
        self,
        child_id: str,
        *,
        owner_project_id: str,
        message_ids: Iterable[str],
    ) -> int:
        """Return claimed messages to the inbox if an attempt could not start."""
        ids = sorted({str(message_id) for message_id in message_ids if message_id})
        if not ids:
            return 0
        with self._connect() as conn:
            owner = conn.execute(
                "SELECT 1 FROM child_sessions WHERE child_id=? AND owner_project_id=?",
                (child_id, owner_project_id),
            ).fetchone()
            if not owner:
                raise KeyError(child_id)
            cursor = conn.execute(
                "UPDATE child_inbox SET consumed_at='' WHERE child_id=? AND message_id IN ("
                + ",".join("?" for _ in ids)
                + ")",
                [child_id, *ids],
            )
        return int(cursor.rowcount or 0)

    def report(
        self,
        child_id: str,
        *,
        owner_project_id: str,
        event_limit: int = 100,
        message_limit: int = 100,
    ) -> dict[str, Any]:
        child = self.get(child_id, owner_project_id=owner_project_id)
        if not child:
            raise KeyError(child_id)
        with self._connect() as conn:
            attempts = conn.execute(
                "SELECT * FROM child_attempts WHERE child_id=? ORDER BY generation",
                (child_id,),
            ).fetchall()
            events = conn.execute(
                "SELECT * FROM child_events WHERE child_id=? "
                "ORDER BY sequence DESC LIMIT ?",
                (child_id, max(1, min(int(event_limit), 500))),
            ).fetchall()
            messages = conn.execute(
                "SELECT * FROM child_inbox WHERE child_id=? "
                "ORDER BY created_at DESC LIMIT ?",
                (child_id, max(1, min(int(message_limit), 500))),
            ).fetchall()
        return {
            "child": child,
            "attempts": [self._decode_attempt(row) for row in attempts],
            "events": [self._decode_event(row) for row in reversed(events)],
            "messages": [self._decode_message(row) for row in reversed(messages)],
        }


def get_child_session_store() -> ChildSessionStore:
    """Return a store rooted in the current, possibly test-overridden DATA_DIR."""
    return ChildSessionStore()
