"""Staged post-turn learning proposals.

Only explicit user corrections/preferences are proposed.  Nothing is written to
long-term memory until a separate approval action is recorded.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from remy.config.settings import settings
from remy.core.file_utils import atomic_write

_LOCK = threading.Lock()
_MAX_REVIEWS = 500
_CORRECTION_MARKERS = re.compile(
    r"\b(неправильно|це не так|не вигадуй|не вигадувати|помилк|виправ|wrong|incorrect|don't invent|do not invent)\b",
    re.IGNORECASE,
)
_PREFERENCE_MARKERS = re.compile(
    r"\b(запам['’]?ятай|завжди|ніколи|я хочу щоб|надаю перевагу|remember|always|never|i prefer)\b",
    re.IGNORECASE,
)


def _path() -> Path:
    return settings.DATA_DIR / "learning_reviews.json"


def _load() -> list[dict[str, Any]]:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def _save(items: list[dict[str, Any]]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(items[-_MAX_REVIEWS:], ensure_ascii=False, indent=2) + "\n")


def stage_learning_review(
    *,
    session_id: str,
    user_text: str,
    assistant_text: str = "",
) -> dict[str, Any] | None:
    text = str(user_text or "").strip()
    if len(text) < 8:
        return None
    if _CORRECTION_MARKERS.search(text):
        category = "correction"
    elif _PREFERENCE_MARKERS.search(text):
        category = "preference"
    else:
        return None
    review_id = "learning-" + hashlib.sha256(
        f"{session_id}\0{text}".encode("utf-8")
    ).hexdigest()[:20]
    with _LOCK:
        items = _load()
        existing = next((item for item in items if item.get("review_id") == review_id), None)
        if existing:
            return existing
        item = {
            "review_id": review_id,
            "status": "pending",
            "category": category,
            "candidate_text": text,
            "session_id": session_id,
            "assistant_context": str(assistant_text or "")[:1000],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "decided_at": "",
            "memory_record_id": "",
        }
        items.append(item)
        _save(items)
        return item


def list_learning_reviews(*, status: str | None = "pending", limit: int = 100) -> list[dict[str, Any]]:
    with _LOCK:
        items = _load()
    if status:
        items = [item for item in items if item.get("status") == status]
    return list(reversed(items[-max(1, min(int(limit), 500)):]))


def decide_learning_review(
    review_id: str,
    *,
    status: str,
    memory_record_id: str = "",
) -> dict[str, Any] | None:
    if status not in {"approved", "rejected"}:
        raise ValueError("status must be approved or rejected")
    with _LOCK:
        items = _load()
        selected = None
        for item in items:
            if item.get("review_id") != review_id:
                continue
            if item.get("status") != "pending":
                return item
            item["status"] = status
            item["decided_at"] = datetime.now(timezone.utc).isoformat()
            item["memory_record_id"] = memory_record_id
            selected = item
            break
        if selected:
            _save(items)
        return selected
