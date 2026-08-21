"""Project document visibility metadata.

Documents are user-owned project artifacts.  They are private by default and
only become readable by agent filesystem tools after an explicit opt-in.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from remy.core.file_utils import atomic_write

_MANIFEST_NAME = ".document_access.json"
_LOCK = threading.RLock()


def document_access_manifest(documents_dir: Path) -> Path:
    return documents_dir.resolve() / _MANIFEST_NAME


def _load(documents_dir: Path) -> dict[str, Any]:
    path = document_access_manifest(documents_dir)
    if not path.exists():
        return {"version": 1, "documents": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {"version": 1, "documents": {}}
    documents = payload.get("documents")
    if not isinstance(documents, dict):
        documents = {}
    return {"version": 1, "documents": documents}


def is_agent_accessible(documents_dir: Path, filename: str) -> bool:
    """Return whether a Markdown project document is explicitly shared."""
    name = Path(str(filename or "")).name
    if not name.lower().endswith(".md"):
        return False
    with _LOCK:
        entry = _load(documents_dir)["documents"].get(name, {})
    return bool(entry.get("agent_access")) if isinstance(entry, dict) else False


def set_agent_access(documents_dir: Path, filename: str, allowed: bool) -> None:
    """Persist explicit agent visibility for one project document."""
    name = Path(str(filename or "")).name
    if not name.lower().endswith(".md"):
        raise ValueError("Only Markdown project documents can be shared with the agent")
    documents_dir = documents_dir.resolve()
    documents_dir.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        payload = _load(documents_dir)
        payload["documents"][name] = {"agent_access": bool(allowed)}
        atomic_write(
            document_access_manifest(documents_dir),
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        )


def remove_document_access(documents_dir: Path, filename: str) -> None:
    """Remove stale visibility metadata after a document is deleted."""
    name = Path(str(filename or "")).name
    with _LOCK:
        payload = _load(documents_dir)
        if payload["documents"].pop(name, None) is None:
            return
        atomic_write(
            document_access_manifest(documents_dir),
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        )
