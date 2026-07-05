"""
Phase 3 Step 4 — Promotion Audit Trail.

Every promotion surface that drops a record (recall primary surface,
gated_connect) emits one structured event describing *why* it was blocked.
Without this trail, Step 2/3's gates are silent — you can see *that*
something didn't become belief, but not *why*, and that makes drift on
admission/promotion policy invisible.

Design:

  * Single module, no hash-chain, no sensitive-value sanitization — these
    events describe structural decisions, not user/tool IO. The heavier
    `audit_trail.AuditLogger` is deliberately not reused.
  * `block_reason(item)` is the single enumerator. Ordering mirrors
    `_is_factual_forbidden` so the *first* matching signal is named, not
    a union. Callers who already know the record is forbidden can ask
    "why?" without re-running the whole rule-set.
  * `PromotionAuditLog` is a process-wide singleton with both a ring buffer
    (for tests + `/diagnostics`) and an optional JSONL sink. Emission is
    best-effort: a logging failure must never break the gate.
  * Surfaces pass structured `extra` fields (partner endpoint id on connect,
    query hint on recall) so the event is enough to reconstruct the decision
    offline.

Out of scope: tamper-evidence (use audit_trail for that), Rust-side events,
promotion *success* events (we only record blocks).
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ── Block-reason enumeration ─────────────────────────────────────────────────


# Reason codes — stable strings, one per distinct signal. Ordering matters:
# `block_reason()` returns the *first* matching signal so downstream analysis
# can count distinct causes rather than overlapping ones.
REASON_FORBIDDEN_TAG = "forbidden_tag"
REASON_REQUIRES_PROMOTION = "requires_promotion_unpromoted"
REASON_SUPERSEDED_BY = "superseded_by"
REASON_UNRESOLVED_CONFLICT_FLAG = "unresolved_conflict_flag"
REASON_TRUTH_STATUS_STALE_HARD = "truth_status_stale_hard"
REASON_TRUTH_STATUS_CONFLICT_UNRESOLVED = "truth_status_conflict_unresolved"
REASON_TRUTH_STATUS_SUPERSEDED = "truth_status_superseded"
REASON_FORBIDDEN_ADMISSION_CLASS = "forbidden_admission_class"

# Surface-level reasons (not record-intrinsic)
REASON_MISSING_ENDPOINT = "missing_endpoint"
REASON_SDK_FAILURE = "sdk_failure"


_TRUTH_STATUS_REASON_MAP = {
    "stale_hard": REASON_TRUTH_STATUS_STALE_HARD,
    "conflict_unresolved": REASON_TRUTH_STATUS_CONFLICT_UNRESOLVED,
    "superseded": REASON_TRUTH_STATUS_SUPERSEDED,
}


def block_reason(item: dict | None) -> str | None:
    """Return the first signal that would cause `_is_factual_forbidden` to
    block this item, or None if nothing blocks.

    Ordering must stay in sync with hybrid_search._is_factual_forbidden.
    If it drifts, Step 2/3's decisions diverge from the reason codes emitted
    here — callers would see a block happen and get back None as "no reason".
    """
    if not item:
        return None

    tags = item.get("tags") or []
    meta = item.get("metadata") or {}

    # 1) forbidden tag (fast path, matches _is_factual_forbidden ordering)
    try:
        from remy.core.hybrid_search import _FACTUAL_FORBIDDEN_TAGS
        if _FACTUAL_FORBIDDEN_TAGS.intersection(set(tags)):
            return REASON_FORBIDDEN_TAG
    except Exception:
        pass

    # 2-4) metadata flags (order: promotion stamp, supersession, conflict)
    if meta.get("requires_promotion") and not meta.get("promoted"):
        return REASON_REQUIRES_PROMOTION
    if meta.get("superseded_by"):
        return REASON_SUPERSEDED_BY
    if meta.get("unresolved_conflict"):
        return REASON_UNRESOLVED_CONFLICT_FLAG

    # 5) truth_status lifecycle (stale_hard / conflict_unresolved / superseded)
    try:
        from remy.core.retrieval.freshness import truth_status
        status = truth_status(meta)
        mapped = _TRUTH_STATUS_REASON_MAP.get(status)
        if mapped:
            return mapped
    except Exception:
        pass

    # 6) admission_class fallback (derives when explicit class missing)
    try:
        from remy.core.memory_policy import (
            FACTUAL_FORBIDDEN_ADMISSION_CLASSES,
            derive_admission_class,
        )
        cls = derive_admission_class(meta, tags)
        if cls in FACTUAL_FORBIDDEN_ADMISSION_CLASSES:
            return REASON_FORBIDDEN_ADMISSION_CLASS
    except Exception:
        pass

    return None


# ── Audit log (ring buffer + optional JSONL sink) ────────────────────────────


class PromotionAuditLog:
    """In-memory ring buffer of blocked-promotion events, plus optional JSONL.

    Use cases:
      * Tests: `get_recent()` returns the ring contents for assertions.
      * Diagnostics: `/promotion-audit` endpoint (future) reads the ring.
      * Offline analysis: the JSONL sink gives a durable record of *why*
        records didn't promote over time.

    Not thread-contended: emission is rare relative to normal brain ops, so
    a single lock is enough.
    """

    DEFAULT_CAPACITY = 1024

    def __init__(
        self,
        sink_path: Path | None = None,
        capacity: int = DEFAULT_CAPACITY,
    ):
        self._buffer: deque[dict] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._sink_path = sink_path
        if sink_path is not None:
            try:
                sink_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.warning(
                    "PromotionAuditLog: could not prepare sink dir %s: %s",
                    sink_path.parent,
                    e,
                )
                self._sink_path = None

    def record(
        self,
        *,
        surface: str,
        record_id: str | None,
        reason: str,
        extra: dict | None = None,
    ) -> dict:
        """Append a block event. Best-effort — emission failure never raises."""
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "surface": surface,
            "record_id": record_id,
            "reason": reason,
        }
        if extra:
            entry["extra"] = extra

        with self._lock:
            self._buffer.append(entry)
            if self._sink_path is not None:
                try:
                    with open(self._sink_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                except OSError as e:
                    # Degrade gracefully: ring buffer still has the event.
                    logger.warning(
                        "PromotionAuditLog: sink write failed: %s", e,
                    )
        return entry

    def get_recent(
        self,
        n: int = 50,
        *,
        surface: str | None = None,
        reason: str | None = None,
    ) -> list[dict]:
        """Return up to n most recent events, newest first, optionally filtered."""
        with self._lock:
            snapshot = list(self._buffer)
        snapshot.reverse()  # newest first
        if surface is not None:
            snapshot = [e for e in snapshot if e.get("surface") == surface]
        if reason is not None:
            snapshot = [e for e in snapshot if e.get("reason") == reason]
        return snapshot[:n]

    def clear(self) -> None:
        """Reset the ring buffer. Used by tests; does not truncate the JSONL sink."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


# ── Module-level singleton ──────────────────────────────────────────────────


_log_singleton: PromotionAuditLog | None = None
_singleton_lock = threading.Lock()


def _resolve_sink_path() -> Path | None:
    """Pick a sink path from settings, falling back to data/promotion_audit.jsonl.
    Returns None if everything fails — ring buffer still works."""
    try:
        from remy.config.settings import settings
        base = getattr(settings, "AUDIT_LOG_DIR", None)
        if base is not None:
            return Path(base) / "promotion_audit.jsonl"
    except Exception:
        pass
    try:
        return Path("data/audit_logs/promotion_audit.jsonl")
    except Exception:
        return None


def get_promotion_audit_log() -> PromotionAuditLog:
    """Lazy singleton. First caller decides the sink path."""
    global _log_singleton
    if _log_singleton is None:
        with _singleton_lock:
            if _log_singleton is None:
                _log_singleton = PromotionAuditLog(sink_path=_resolve_sink_path())
    return _log_singleton


def reset_promotion_audit_log() -> None:
    """Reset the module-level singleton. Used by tests."""
    global _log_singleton
    with _singleton_lock:
        _log_singleton = None


def record_block(
    surface: str,
    record_id: str | None,
    reason: str,
    extra: dict | None = None,
) -> None:
    """Top-level convenience — never raises."""
    try:
        get_promotion_audit_log().record(
            surface=surface,
            record_id=record_id,
            reason=reason,
            extra=extra,
        )
    except Exception:
        # Emission failure must not break the gate.
        logger.debug("promotion_audit: record_block failed", exc_info=True)


# ── Surface constants (stable strings for grep / diagnostics) ────────────────


SURFACE_RECALL = "recall_primary"
SURFACE_CONNECT = "brain_connect"


def iter_reasons_for_items(items: Iterable[dict]) -> Iterable[tuple[dict, str]]:
    """Yield `(item, reason)` for every blocked item in the iterable.
    Convenience for call sites that filter a list and want to emit per-item
    events in one pass."""
    for item in items or []:
        reason = block_reason(item)
        if reason is not None:
            yield item, reason
