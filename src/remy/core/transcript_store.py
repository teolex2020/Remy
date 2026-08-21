"""Exact, durable conversation transcript with SQLite FTS5 search."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from remy.config.settings import settings

_INIT_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TranscriptStore:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else settings.DATA_DIR / "transcripts.sqlite3"
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _ensure_schema(self) -> None:
        with _INIT_LOCK, self._connect() as conn:
            # Journal mode is persistent database state. Setting it on every
            # transcript read needlessly acquires SQLite locks, so configure it
            # once when the store instance initializes.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS transcript_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    owner_project_id TEXT NOT NULL DEFAULT '',
                    brain_id TEXT NOT NULL DEFAULT '',
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    message_type TEXT NOT NULL DEFAULT 'text',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_transcript_session
                    ON transcript_messages(session_id, id);
                CREATE VIRTUAL TABLE IF NOT EXISTS transcript_fts USING fts5(
                    content, session_id UNINDEXED, role UNINDEXED,
                    content='transcript_messages', content_rowid='id'
                );
                CREATE TRIGGER IF NOT EXISTS transcript_ai AFTER INSERT ON transcript_messages BEGIN
                    INSERT INTO transcript_fts(rowid, content, session_id, role)
                    VALUES (new.id, new.content, new.session_id, new.role);
                END;
                CREATE TRIGGER IF NOT EXISTS transcript_ad AFTER DELETE ON transcript_messages BEGIN
                    INSERT INTO transcript_fts(transcript_fts, rowid, content, session_id, role)
                    VALUES ('delete', old.id, old.content, old.session_id, old.role);
                END;
                CREATE TRIGGER IF NOT EXISTS transcript_au AFTER UPDATE ON transcript_messages BEGIN
                    INSERT INTO transcript_fts(transcript_fts, rowid, content, session_id, role)
                    VALUES ('delete', old.id, old.content, old.session_id, old.role);
                    INSERT INTO transcript_fts(rowid, content, session_id, role)
                    VALUES (new.id, new.content, new.session_id, new.role);
                END;
                """
            )
            columns = {
                str(row["name"])
                for row in conn.execute(
                    "PRAGMA table_info(transcript_messages)"
                ).fetchall()
            }
            if "owner_project_id" not in columns:
                conn.execute(
                    "ALTER TABLE transcript_messages "
                    "ADD COLUMN owner_project_id TEXT NOT NULL DEFAULT ''"
                )
            if "brain_id" not in columns:
                conn.execute(
                    "ALTER TABLE transcript_messages "
                    "ADD COLUMN brain_id TEXT NOT NULL DEFAULT ''"
                )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_transcript_project_session "
                "ON transcript_messages(owner_project_id, session_id, id)"
            )

    def append(
        self,
        *,
        session_id: str,
        owner_project_id: str = "",
        brain_id: str = "",
        role: str,
        content: str,
        message_type: str = "text",
        metadata: dict[str, Any] | None = None,
        message_id: str | None = None,
    ) -> str:
        owner_project_id = str(owner_project_id or "").strip()
        brain_id = str(brain_id or "").strip()
        if not owner_project_id or not brain_id:
            raise ValueError(
                "Transcript messages require owner_project_id and brain_id"
            )
        message_id = message_id or f"message-{uuid.uuid4().hex}"
        with self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO transcript_messages(
                    message_id, session_id, owner_project_id, brain_id,
                    role, content, message_type,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    message_id,
                    session_id,
                    owner_project_id,
                    brain_id,
                    role,
                    content,
                    message_type,
                    json.dumps(metadata or {}, ensure_ascii=False), _now(),
                ),
            )
        return message_id

    def list_session(
        self,
        session_id: str,
        *,
        owner_project_id: str,
        include_legacy_unscoped: bool = False,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        owner_project_id = str(owner_project_id or "").strip()
        if not owner_project_id:
            raise ValueError("Transcript history requires owner_project_id")
        project_clause = (
            "owner_project_id IN (?, '')"
            if include_legacy_unscoped
            else "owner_project_id=?"
        )
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ("
                "SELECT * FROM transcript_messages WHERE session_id=? AND "
                + project_clause
                + " ORDER BY id DESC LIMIT ?"
                ") ORDER BY id",
                (
                    session_id,
                    owner_project_id,
                    max(1, min(int(limit), 2000)),
                ),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json", "{}") or "{}")
            result.append(item)
        if not result:
            try:
                from remy.core.session_event_store import (
                    TranscriptProjection,
                    get_session_event_store,
                )

                recovered = get_session_event_store().project(
                    TranscriptProjection(),
                    project_id=owner_project_id,
                    session_id=session_id,
                )
                result = list(recovered)[-max(1, min(int(limit), 2000)):]
            except (OSError, sqlite3.Error):
                # Legacy installations may not have initialized the unified
                # event ledger yet. The normal empty transcript contract stays
                # intact until the writer is available.
                pass
        return result

    def search(
        self,
        query: str,
        *,
        session_id: str = "",
        owner_project_id: str = "",
        include_legacy_unscoped: bool = False,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        owner_project_id = str(owner_project_id or "").strip()
        if not owner_project_id:
            raise ValueError("Transcript search requires owner_project_id")
        terms = [part.replace('"', '""') for part in query.split() if part.strip()]
        if not terms:
            return []
        match_query = " AND ".join(f'"{term}"' for term in terms)
        sql = (
            "SELECT m.*, bm25(transcript_fts) AS rank, "
            "snippet(transcript_fts, 0, '[', ']', ' … ', 24) AS snippet "
            "FROM transcript_fts JOIN transcript_messages m ON m.id=transcript_fts.rowid "
            "WHERE transcript_fts MATCH ?"
        )
        args: list[Any] = [match_query]
        if session_id:
            sql += " AND m.session_id=?"
            args.append(session_id)
        if owner_project_id:
            sql += (
                " AND m.owner_project_id IN (?, '')"
                if include_legacy_unscoped
                else " AND m.owner_project_id=?"
            )
            args.append(owner_project_id)
        sql += " ORDER BY rank, m.id DESC LIMIT ?"
        args.append(max(1, min(int(limit), 100)))
        with self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json", "{}") or "{}")
            result.append(item)
        return result


@lru_cache(maxsize=8)
def _cached_transcript_store(path: str) -> TranscriptStore:
    return TranscriptStore(Path(path))


def get_transcript_store() -> TranscriptStore:
    """Reuse one initialized store per configured data path.

    Keying the cache by path keeps tests and project-specific runtime settings
    isolated while avoiding schema checks for every transcript request.
    """
    return _cached_transcript_store(str(settings.DATA_DIR / "transcripts.sqlite3"))
