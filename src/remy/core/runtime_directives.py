"""
Runtime Directives (AUTON-2) — dynamic system instruction modification.

Allows the agent to add, remove, and manage runtime directives that modify
its system instruction during a session or across sessions.

Two tiers:
- Session directives: in-memory, expire when session ends or TTL reached
- Persistent directives: stored in brain, survive across sessions

Priority: user-set > persistent > session > default instruction
"""

import json
import logging
import threading
import time
from datetime import datetime

from remy.core.agent_tools import brain
from remy.core.event_bus import event_bus

logger = logging.getLogger("Autonomy.Directives")

DIRECTIVE_TAGS = ["runtime-directive"]

# In-memory session directives: {session_id: [{text, ttl_seconds, created_at, source}]}
_session_directives: dict[str, list[dict]] = {}
_directives_lock = threading.Lock()


def add_session_directive(
    text: str,
    session_id: str,
    ttl_seconds: int | None = None,
    source: str = "agent",
) -> str:
    """Add a temporary directive for the current session.

    Args:
        text: The directive instruction text.
        session_id: Current session ID.
        ttl_seconds: Time-to-live in seconds. None = until session ends.
        source: Who set this ("agent", "user", "system").

    Returns:
        Directive ID (index-based).
    """
    directive = {
        "text": text,
        "ttl_seconds": ttl_seconds,
        "created_at": time.time(),
        "source": source,
    }

    with _directives_lock:
        if session_id not in _session_directives:
            _session_directives[session_id] = []
        _session_directives[session_id].append(directive)
        idx = len(_session_directives[session_id]) - 1

    logger.info(
        "Added session directive #%d (ttl=%s): %s",
        idx,
        ttl_seconds or "session",
        text[:80],
    )
    event_bus.emit(
        "directive_added",
        {
            "type": "session",
            "index": idx,
            "text": text[:200],
            "source": source,
        },
    )

    return f"session-{idx}"


def add_persistent_directive(text: str, source: str = "agent") -> str | None:
    """Add a persistent directive that survives across sessions.

    Stored as a brain record with tags ["runtime-directive"].
    Returns record ID if stored successfully.
    """
    try:
        from remy.core.agent_tools import Level

        record = brain.store(
            content=text,
            level=Level.DOMAIN,
            tags=DIRECTIVE_TAGS,
            metadata={
                "type": "persistent_directive",
                "source": source,
                "created_at": datetime.now().isoformat(),
                "active": True,
            },
        )

        record_id = record.id if hasattr(record, "id") else str(record)
        logger.info("Added persistent directive %s: %s", record_id[:8], text[:80])
        event_bus.emit(
            "directive_added",
            {
                "type": "persistent",
                "record_id": record_id,
                "text": text[:200],
                "source": source,
            },
        )
        return record_id

    except Exception as e:
        logger.warning("Failed to store persistent directive: %s", e)
        return None


def get_active_directives(session_id: str) -> list[dict]:
    """Get all active directives for the current session.

    Returns list of {"text", "source", "type"} dicts, sorted by priority:
    1. user-set directives first
    2. persistent directives
    3. session directives
    """
    directives = []
    now = time.time()

    # 1. Session directives (check TTL)
    with _directives_lock:
        session_dirs = list(_session_directives.get(session_id, []))

    for d in session_dirs:
        ttl = d.get("ttl_seconds")
        if ttl is not None and (now - d["created_at"]) > ttl:
            continue  # Expired
        directives.append(
            {
                "text": d["text"],
                "source": d.get("source", "agent"),
                "type": "session",
            }
        )

    # 2. Persistent directives from brain
    try:
        from remy.core.agent_tools import brain_lock

        with brain_lock:
            records = brain.search(query="", tags=DIRECTIVE_TAGS, limit=20)

        for r in records:
            meta = getattr(r, "metadata", None) or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}

            if not meta.get("active", True):
                continue

            directives.append(
                {
                    "text": r.content,
                    "source": meta.get("source", "agent"),
                    "type": "persistent",
                    "record_id": r.id if hasattr(r, "id") else str(r),
                }
            )

    except Exception as e:
        logger.warning("Failed to load persistent directives: %s", e)

    # Sort: user > persistent > session
    priority = {"user": 0, "system": 1, "agent": 2}
    directives.sort(key=lambda d: (priority.get(d["source"], 3), d["type"] != "persistent"))

    return directives


def format_directives_for_instruction(session_id: str) -> str:
    """Format active directives as text to append to system instruction.

    Returns empty string if no directives are active.
    """
    directives = get_active_directives(session_id)
    if not directives:
        return ""

    lines = []
    for d in directives:
        prefix = f"[{d['source'].upper()}]" if d["source"] != "agent" else ""
        lines.append(f"- {prefix} {d['text']}".strip())

    return (
        "\n\n=== RUNTIME DIRECTIVES (follow these instructions) ===\n"
        + "\n".join(lines)
        + "\n=== END DIRECTIVES ===\n"
    )


def remove_session_directive(session_id: str, index: int) -> bool:
    """Remove a session directive by index."""
    with _directives_lock:
        dirs = _session_directives.get(session_id, [])
        if 0 <= index < len(dirs):
            dirs.pop(index)
            return True
    return False


def deactivate_persistent_directive(record_id: str) -> bool:
    """Deactivate a persistent directive (mark as inactive in brain)."""
    try:
        from remy.core.agent_tools import brain_lock

        with brain_lock:
            record = brain.get(record_id)
        if not record:
            return False

        meta = getattr(record, "metadata", None) or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}

        meta["active"] = False
        meta["deactivated_at"] = datetime.now().isoformat()

        with brain_lock:
            brain.update(record_id, metadata=meta)

        logger.info("Deactivated persistent directive %s", record_id[:8])
        return True

    except Exception as e:
        logger.warning("Failed to deactivate directive %s: %s", record_id[:8], e)
        return False


def clear_session_directives(session_id: str) -> int:
    """Clear all session directives for a session. Returns count cleared."""
    with _directives_lock:
        dirs = _session_directives.pop(session_id, [])
    return len(dirs)


def cleanup_expired(session_id: str) -> int:
    """Remove expired TTL directives from a session. Returns count removed."""
    now = time.time()
    removed = 0

    with _directives_lock:
        dirs = _session_directives.get(session_id, [])
        active = []
        for d in dirs:
            ttl = d.get("ttl_seconds")
            if ttl is not None and (now - d["created_at"]) > ttl:
                removed += 1
            else:
                active.append(d)
        if removed:
            _session_directives[session_id] = active

    return removed
