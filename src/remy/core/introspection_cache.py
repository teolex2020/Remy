"""Introspection cache — TTL-stamped record of recent aura_cognitive_ops calls.

Purpose: when the agent later quotes internal metrics (stability %, temperature,
belief count, etc.), the response auditor checks whether a fresh introspection
call backs the claim. "Fresh" = within _TTL_SEC of the current turn.

This is the Phase 3 seam. The aura_cognitive_ops handler stamps on every
successful call; the resolver reads on audit.

Thread-safety: access is expected to happen on the brain-locked path (stamp
from tool handler) and on the response path (read from auditor). For now use
a module-level dict guarded by a threading.Lock.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any


_TTL_SEC = 60.0
_MAX_ENTRIES_PER_SESSION = 32

_lock = threading.Lock()
_cache: dict[str, list["_Entry"]] = {}


@dataclass(frozen=True)
class _Entry:
    op: str
    result: Any
    ts: float


def stamp(session_id: str | None, op: str, result: Any) -> None:
    """Record that op was called successfully in this session at now()."""
    if not session_id or not op:
        return
    entry = _Entry(op=op, result=result, ts=time.time())
    with _lock:
        entries = _cache.setdefault(session_id, [])
        entries.append(entry)
        if len(entries) > _MAX_ENTRIES_PER_SESSION:
            del entries[: len(entries) - _MAX_ENTRIES_PER_SESSION]


def get_fresh(
    session_id: str | None, op: str | None = None, *, ttl_sec: float = _TTL_SEC
) -> _Entry | None:
    """Return the newest entry for this session (optionally filtered by op) if within TTL."""
    if not session_id:
        return None
    cutoff = time.time() - ttl_sec
    with _lock:
        entries = _cache.get(session_id) or []
        for e in reversed(entries):
            if e.ts < cutoff:
                return None
            if op is None or e.op == op:
                return e
    return None


def has_any_fresh(session_id: str | None, *, ttl_sec: float = _TTL_SEC) -> bool:
    return get_fresh(session_id, op=None, ttl_sec=ttl_sec) is not None


def reset(session_id: str | None = None) -> None:
    """Clear cache entries. session_id=None clears everything (test helper)."""
    with _lock:
        if session_id is None:
            _cache.clear()
        else:
            _cache.pop(session_id, None)
