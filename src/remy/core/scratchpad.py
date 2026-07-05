"""
Working Memory Ring / Scratchpad.

Gives the agent a short-term notepad for intermediate results that would
otherwise be lost when compact_history trims old tool messages.

- Notes are stored at Level.WORKING (fast decay: hours, not days).
- Tagged ``scratchpad`` so they can be recalled and cleaned specifically.
- ``get_scratchpad_context()`` returns current notes for injection into
  the LLM context, so the agent always sees its own notes.
- Ring buffer behavior: oldest notes auto-evict when the count exceeds
  ``MAX_SCRATCHPAD_NOTES`` (default 20).
- Older notes can be summarized into one ``scratchpad-summary`` record
  to preserve signal without keeping all raw detail.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any

logger = logging.getLogger("Scratchpad")

# ============== Configuration ==============

MAX_SCRATCHPAD_NOTES = 20
SUMMARY_TRIGGER_NOTES = 12
SUMMARY_BATCH_SIZE = 8
FILTER_MIN_SCORE = 0.18
_SCRATCHPAD_TAG = "scratchpad"
_SUMMARY_TAG = "scratchpad-summary"
_seq_counter = 0  # Monotonic counter for stable ordering
_session_metrics: dict[str, dict[str, Any]] = {}


# ============== Core Operations ==============


def write_note(content: str, *, session_id: str = "", channel: str = "") -> dict:
    """Write a scratchpad note. Returns ``{id, content_preview}``."""
    from remy.core.agent_tools import Level, brain, brain_lock
    from remy.core.provenance import _stamp_provenance

    content = content.strip()
    if not content:
        return {"error": "Empty note"}

    _maybe_auto_summarize(session_id=session_id, channel=channel)

    global _seq_counter

    with brain_lock:
        _evict_overflow_locked()

        _seq_counter += 1
        meta = _stamp_provenance(
            {
                "scratchpad": True,
                "written_at": time.time(),
                "seq": _seq_counter,
                "session_id": session_id or "",
                "note_kind": "note",
            },
            channel,
            tags=[_SCRATCHPAD_TAG],
        )
        rec = brain.store(
            content=content,
            level=Level.WORKING,
            tags=[_SCRATCHPAD_TAG],
            metadata=meta,
            channel=channel,
        )

    logger.info("Scratchpad note written: %s (id=%s)", content[:60], rec.id)
    return {"id": rec.id, "content_preview": content[:120]}


def read_notes() -> list[dict]:
    """Return all current scratchpad notes (newest first)."""
    from remy.core.agent_tools import brain, brain_lock

    with brain_lock:
        records = brain.search(tags=[_SCRATCHPAD_TAG], limit=MAX_SCRATCHPAD_NOTES + 5) or []

    notes = []
    for r in records:
        meta = r.metadata or {}
        notes.append(
            {
                "id": r.id,
                "content": r.content,
                "written_at": meta.get("written_at", 0),
                "seq": meta.get("seq", 0),
                "score": meta.get("_scratchpad_score"),
                "summary": bool(meta.get("is_summary")),
                "tags": list(getattr(r, "tags", []) or []),
            }
        )

    notes.sort(key=lambda n: n["seq"], reverse=True)
    return notes[:MAX_SCRATCHPAD_NOTES]


def clear_notes() -> int:
    """Delete all scratchpad notes. Returns count deleted."""
    from remy.core.agent_tools import brain, brain_lock

    with brain_lock:
        records = brain.search(tags=[_SCRATCHPAD_TAG], limit=100) or []
        count = 0
        for r in records:
            if brain.delete(r.id):
                count += 1

    logger.info("Scratchpad cleared: %d notes deleted", count)
    return count


def summarize_notes(
    *,
    session_id: str = "",
    channel: str = "",
    force: bool = False,
) -> dict:
    """Compress older scratchpad notes into one summary record."""
    from remy.core.agent_tools import Level, brain, brain_lock
    from remy.core.provenance import _stamp_provenance

    with brain_lock:
        records = _sorted_scratchpad_records_locked(brain)
        source_records = [r for r in records if _SUMMARY_TAG not in set(r.tags or [])]
        if len(source_records) < SUMMARY_TRIGGER_NOTES and not force:
            return {
                "summarized": False,
                "reason": f"Only {len(source_records)} raw notes; threshold is {SUMMARY_TRIGGER_NOTES}",
            }
        if len(source_records) < 2:
            return {"summarized": False, "reason": "Not enough notes to summarize"}

        batch = _select_summary_batch(source_records, session_id=session_id)
        if len(batch) < 2:
            return {"summarized": False, "reason": "No coherent note batch found"}
        batch_payload = [_record_to_note(rec) for rec in batch]

    summary = _generate_summary_text(batch_payload)
    if not summary:
        return {"summarized": False, "reason": "Summary generation failed"}

    with brain_lock:
        refreshed = {r.id: r for r in _sorted_scratchpad_records_locked(brain)}
        batch = [refreshed[item["id"]] for item in batch_payload if item["id"] in refreshed]
        if len(batch) < 2:
            return {"summarized": False, "reason": "Source notes changed during summarization"}

        global _seq_counter
        _seq_counter += 1
        original_chars = sum(len((r.content or "")) for r in batch)
        ratio = round(len(summary) / max(original_chars, 1), 3)
        session_value = session_id or _majority_session_id(batch)
        meta = _stamp_provenance(
            {
                "scratchpad": True,
                "written_at": time.time(),
                "seq": _seq_counter,
                "session_id": session_value,
                "note_kind": "summary",
                "is_summary": True,
                "source_note_count": len(batch),
                "source_ids": [r.id for r in batch],
                "original_char_count": original_chars,
                "summary_char_count": len(summary),
                "compression_ratio": ratio,
                "summarized_at": time.time(),
            },
            channel or _majority_channel(batch),
            tags=[_SCRATCHPAD_TAG, _SUMMARY_TAG],
        )
        rec = brain.store(
            content=summary,
            level=Level.WORKING,
            tags=[_SCRATCHPAD_TAG, _SUMMARY_TAG],
            metadata=meta,
            channel=channel,
        )
        deleted_count = 0
        for item in batch:
            if brain.delete(item.id):
                deleted_count += 1

    _update_session_metrics(
        session_value,
        scratchpad_compression_ratio=ratio,
        scratchpad_last_summary={
            "summary_record_id": rec.id,
            "deleted_count": deleted_count,
            "source_count": len(batch),
            "compression_ratio": ratio,
        },
    )
    logger.info(
        "Scratchpad summarized %d notes into %s (ratio=%.3f)",
        len(batch), rec.id, ratio,
    )
    return {
        "summarized": True,
        "summary_record_id": rec.id,
        "source_count": len(batch),
        "deleted_count": deleted_count,
        "compression_ratio": ratio,
        "content_preview": summary[:200],
    }


def filter_working_memory(
    query: str,
    *,
    session_id: str = "",
    min_score: float = FILTER_MIN_SCORE,
    delete_irrelevant: bool = False,
) -> dict:
    """Keep only query-relevant scratchpad-managed WORKING records active."""
    from remy.core.agent_tools import Level, brain, brain_lock

    query = (query or "").strip()
    if len(query) < 3:
        return {"filtered": False, "reason": "query too short"}

    with brain_lock:
        working_records = brain.search(tags=[_SCRATCHPAD_TAG], limit=200) or []
        targets = [r for r in working_records if _is_scratchpad_record(r)]
        working_ids = {r.id for r in targets}
        if not targets:
            _update_session_metrics(
                session_id,
                scratchpad_working_total=0,
                scratchpad_working_active=0,
                scratchpad_working_bloat=0,
            )
            return {"filtered": True, "total": 0, "active": 0, "demoted": 0, "deleted": 0}

        recalled = brain.recall_structured(query, top_k=25, min_strength=0.05, session_id=session_id or None) or []
        relevant_scores = {
            item.get("id"): float(item.get("score", 0.0))
            for item in recalled
            if isinstance(item, dict) and item.get("id") in working_ids
        }
        relevant_ids = {rid for rid, score in relevant_scores.items() if score >= min_score}

        demoted = 0
        deleted = 0
        for rec in targets:
            meta = dict(rec.metadata or {})
            meta["_scratchpad_score"] = round(relevant_scores.get(rec.id, 0.0), 3)
            meta["last_filtered_at"] = time.time()
            meta["last_filter_query"] = query[:200]
            if rec.id in relevant_ids:
                brain.update(rec.id, metadata=meta, strength=max(float(getattr(rec, "strength", 0.5) or 0.5), 0.35))
                continue
            if delete_irrelevant and _SCRATCHPAD_TAG in set(rec.tags or []):
                if brain.delete(rec.id):
                    deleted += 1
                continue
            lowered = min(float(getattr(rec, "strength", 0.5) or 0.5), 0.12)
            brain.update(rec.id, metadata=meta, strength=lowered)
            demoted += 1

    active = len(relevant_ids & {r.id for r in targets})
    total = len(targets)
    _update_session_metrics(
        session_id,
        scratchpad_working_total=total,
        scratchpad_working_active=active,
        scratchpad_working_bloat=max(total - active, 0),
        working_memory_total=total,
        working_memory_active=active,
        working_memory_bloat=max(total - active, 0),
        last_filter={"query": query[:200], "active": active, "total": total},
    )
    return {
        "filtered": True,
        "total": total,
        "active": active,
        "demoted": demoted,
        "deleted": deleted,
        "min_score": min_score,
    }


def get_scratchpad_context(
    query: str | None = None,
    *,
    auto_filter: bool = False,
    session_id: str = "",
) -> str | None:
    """Build scratchpad context string for LLM injection."""
    if auto_filter and query:
        try:
            filter_working_memory(query, session_id=session_id)
        except Exception as e:
            logger.debug("Scratchpad auto-filter failed: %s", e)

    notes = read_notes()
    if not notes:
        return None

    lines = [f"[SCRATCHPAD - {len(notes)} working notes]"]
    for i, note in enumerate(notes, 1):
        prefix = "[summary] " if note.get("summary") else ""
        lines.append(f"  {i}. {prefix}{note['content'][:200]}")

    return "\n".join(lines)


def get_scratchpad_metrics(session_id: str = "") -> dict[str, Any]:
    """Return latest scratchpad telemetry for the given session."""
    return dict(_session_metrics.get(session_id or "", {}))


# ============== Internal Helpers ==============


def _evict_overflow_locked() -> None:
    """Evict oldest notes if count exceeds MAX_SCRATCHPAD_NOTES.

    Must be called under brain_lock.
    """
    from remy.core.agent_tools import brain

    records = _sorted_scratchpad_records_locked(brain)
    if len(records) < MAX_SCRATCHPAD_NOTES:
        return

    to_delete = records[: len(records) - MAX_SCRATCHPAD_NOTES + 1]
    for r in to_delete:
        brain.delete(r.id)
        logger.debug("Scratchpad evicted oldest note: %s", r.id)


def _sorted_scratchpad_records_locked(brain) -> list:
    records = brain.search(tags=[_SCRATCHPAD_TAG], limit=MAX_SCRATCHPAD_NOTES + 20) or []
    return sorted((r for r in records if _is_scratchpad_record(r)), key=lambda r: (r.metadata or {}).get("seq", 0))


def _is_scratchpad_record(record) -> bool:
    tags = set(getattr(record, "tags", []) or [])
    metadata = dict(getattr(record, "metadata", {}) or {})
    if _SCRATCHPAD_TAG not in tags:
        return False
    return bool(
        metadata.get("scratchpad")
        or metadata.get("note_kind") in {"note", "summary"}
        or _SUMMARY_TAG in tags
    )


def _record_to_note(rec) -> dict[str, Any]:
    meta = rec.metadata or {}
    return {
        "id": rec.id,
        "content": rec.content or "",
        "seq": meta.get("seq", 0),
        "written_at": meta.get("written_at", 0),
        "session_id": meta.get("session_id", ""),
        "channel": meta.get("channel", ""),
    }


def _select_summary_batch(records: list, *, session_id: str = "") -> list:
    if not records:
        return []
    if session_id:
        session_records = [r for r in records if (r.metadata or {}).get("session_id") == session_id]
        if len(session_records) >= 2:
            return session_records[:SUMMARY_BATCH_SIZE]
    bucketed: dict[str, list] = {}
    for rec in records:
        key = (rec.metadata or {}).get("session_id", "") or "default"
        bucketed.setdefault(key, []).append(rec)
    best_bucket = max(bucketed.values(), key=len, default=[])
    if len(best_bucket) >= 2:
        return best_bucket[:SUMMARY_BATCH_SIZE]
    return records[:SUMMARY_BATCH_SIZE]


def _generate_summary_text(notes: list[dict[str, Any]]) -> str | None:
    if len(notes) < 2:
        return None
    lines = [f"{idx}. {note['content'][:300]}" for idx, note in enumerate(notes, 1)]
    prompt = (
        "Compress these scratchpad notes into 3-5 short bullet points.\n"
        "Preserve concrete facts, decisions, blockers, and next steps.\n"
        "Do not add new information. Keep the original language.\n\n"
        "Notes:\n"
        + "\n".join(lines)
    )
    try:
        from remy.core.llm import call_llm

        result = call_llm(prompt, purpose="scratchpad_summary")
        content = getattr(result, "content", "")
        if isinstance(content, list):
            content = " ".join(str(part) for part in content)
        summary = str(content).strip()
        return summary if len(summary) >= 12 else None
    except Exception as e:
        logger.warning("Scratchpad summary generation failed: %s", e)
        return None


def _majority_session_id(records: list) -> str:
    counter = Counter((r.metadata or {}).get("session_id", "") for r in records)
    return counter.most_common(1)[0][0] if counter else ""


def _majority_channel(records: list) -> str:
    counter = Counter((r.metadata or {}).get("channel", "") for r in records)
    return counter.most_common(1)[0][0] if counter else ""


def _update_session_metrics(session_id: str, **updates) -> None:
    key = session_id or ""
    if key not in _session_metrics:
        _session_metrics[key] = {}
    _session_metrics[key].update({k: v for k, v in updates.items() if v is not None})


def _maybe_auto_summarize(*, session_id: str = "", channel: str = "") -> None:
    from remy.core.agent_tools import brain, brain_lock

    with brain_lock:
        records = _sorted_scratchpad_records_locked(brain)
        raw_notes = [r for r in records if _SUMMARY_TAG not in set(r.tags or [])]
        should_summarize = len(records) >= MAX_SCRATCHPAD_NOTES and len(raw_notes) >= SUMMARY_TRIGGER_NOTES
        target_session = session_id or _majority_session_id(raw_notes)
        target_channel = channel or _majority_channel(raw_notes)

    if not should_summarize:
        return

    try:
        summarize_notes(session_id=target_session, channel=target_channel)
    except Exception as e:
        logger.warning("Scratchpad overflow summary failed: %s", e)
