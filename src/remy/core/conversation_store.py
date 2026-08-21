"""Project-scoped conversation catalog.

Conversation IDs are also LangGraph thread IDs. Catalogs live inside the
project filesystem boundary; archiving a conversation never deletes its
transcript or checkpoints.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from remy.core.file_utils import atomic_write
from remy.core.project_store import get_project_store, project_data_root

CATALOG_VERSION = 1
_CATALOG_LOCK = threading.RLock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ConversationRecord:
    conversation_id: str
    project_id: str
    brain_id: str
    title: str
    created_at: str
    updated_at: str
    archived_at: str = ""
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConversationStore:
    def __init__(self, project_id: str):
        self.project = get_project_store().require_project(project_id)
        self.root = project_data_root(project_id) / "conversations"
        self.catalog_path = self.root / "index.json"
        self.active_path = self.root / "active.json"
        self._lock = _CATALOG_LOCK
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.catalog_path.exists():
            self._write([])
        else:
            self._read()

    def _read(self) -> list[ConversationRecord]:
        try:
            payload = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Conversation catalog is unreadable") from exc
        if int(payload.get("version", 0)) != CATALOG_VERSION:
            raise RuntimeError("Unsupported conversation catalog version")
        records = []
        seen: set[str] = set()
        for item in payload.get("conversations", []):
            record = ConversationRecord(**item)
            uuid.UUID(record.conversation_id)
            if record.project_id != self.project.project_id:
                raise RuntimeError("Conversation escapes its project boundary")
            if record.brain_id != self.project.brain_id:
                raise RuntimeError("Conversation brain identity is invalid")
            if record.conversation_id in seen:
                raise RuntimeError("Duplicate conversation ID")
            seen.add(record.conversation_id)
            records.append(record)
        return records

    def _write(self, records: list[ConversationRecord]) -> None:
        atomic_write(
            self.catalog_path,
            json.dumps(
                {
                    "version": CATALOG_VERSION,
                    "updated_at": _now(),
                    "conversations": [record.to_dict() for record in records],
                },
                ensure_ascii=False,
                indent=2,
            ),
        )

    def list(self, *, include_archived: bool = False) -> list[ConversationRecord]:
        with self._lock:
            records = self._read()
        if not include_archived:
            records = [record for record in records if not record.archived_at]
        return sorted(records, key=lambda record: record.updated_at, reverse=True)

    def get(
        self,
        conversation_id: str,
        *,
        include_archived: bool = True,
    ) -> ConversationRecord | None:
        for record in self.list(include_archived=include_archived):
            if record.conversation_id == conversation_id:
                return record
        return None

    def require(self, conversation_id: str) -> ConversationRecord:
        record = self.get(conversation_id, include_archived=False)
        if record is None:
            raise KeyError(conversation_id)
        return record

    def create(
        self,
        title: str = "New conversation",
        *,
        metadata: dict[str, Any] | None = None,
        activate: bool = True,
    ) -> ConversationRecord:
        now = _now()
        record = ConversationRecord(
            conversation_id=str(uuid.uuid4()),
            project_id=self.project.project_id,
            brain_id=self.project.brain_id,
            title=(str(title or "").strip() or "New conversation")[:120],
            created_at=now,
            updated_at=now,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            records = self._read()
            records.append(record)
            self._write(records)
            if activate:
                self.set_active(record.conversation_id)
        return record

    def register_existing(
        self,
        conversation_id: str,
        *,
        title: str = "Current conversation",
    ) -> ConversationRecord:
        uuid.UUID(conversation_id)
        existing = self.get(conversation_id)
        if existing is not None:
            return existing
        now = _now()
        record = ConversationRecord(
            conversation_id=conversation_id,
            project_id=self.project.project_id,
            brain_id=self.project.brain_id,
            title=title,
            created_at=now,
            updated_at=now,
            metadata={"migration_source": "active-web-session"},
        )
        with self._lock:
            records = self._read()
            records.append(record)
            self._write(records)
        return record

    def update(self, conversation_id: str, *, title: str | None = None) -> ConversationRecord:
        with self._lock:
            records = self._read()
            for record in records:
                if record.conversation_id != conversation_id:
                    continue
                if record.archived_at:
                    raise ValueError("Archived conversation cannot be edited")
                if title is not None:
                    clean = str(title).strip()
                    if not clean:
                        raise ValueError("Conversation title cannot be empty")
                    record.title = clean[:120]
                record.updated_at = _now()
                self._write(records)
                return record
        raise KeyError(conversation_id)

    def touch_from_user_message(self, conversation_id: str, text: str) -> None:
        record = self.require(conversation_id)
        title = None
        if record.title in {"New conversation", "Current conversation"}:
            title = " ".join(str(text or "").split())[:72] or record.title
        self.update(conversation_id, title=title)

    def archive(self, conversation_id: str) -> ConversationRecord:
        with self._lock:
            records = self._read()
            for record in records:
                if record.conversation_id != conversation_id:
                    continue
                record.archived_at = record.archived_at or _now()
                record.updated_at = _now()
                self._write(records)
                if self.get_active_id() == conversation_id:
                    self.active_path.unlink(missing_ok=True)
                return record
        raise KeyError(conversation_id)

    def get_active_id(self) -> str:
        try:
            payload = json.loads(self.active_path.read_text(encoding="utf-8"))
            conversation_id = str(payload.get("conversation_id") or "")
            return conversation_id if self.get(conversation_id, include_archived=False) else ""
        except (OSError, json.JSONDecodeError):
            return ""

    def set_active(self, conversation_id: str) -> ConversationRecord:
        record = self.require(conversation_id)
        atomic_write(
            self.active_path,
            json.dumps(
                {
                    "conversation_id": record.conversation_id,
                    "project_id": record.project_id,
                    "brain_id": record.brain_id,
                    "updated_at": _now(),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        return record


def get_conversation_store(project_id: str) -> ConversationStore:
    return ConversationStore(project_id)
