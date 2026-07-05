"""Freshness, volatility tiers, and conflict detection.

Phase 4 of the brain-native retrieval roadmap:
    D:\\AuraSDK-verify\\private\\roadmaps\\BRAIN_NATIVE_RETRIEVAL_ROADMAP_2026-04-13.md

Problem this solves:
  - "langchain latest stable version" answered in 2024 is wrong in 2026
  - "python 3.13 release notes" is eternal, but "current python version" is not
  - When new research on the same topic disagrees with stored research,
    silently overwriting the prior belief destroys the conflict signal

Volatility tiers (coarse, deterministic):
  - LOW:    facts that rarely change (historical events, published papers,
            definitions, identifiers).   Default TTL: none.
  - MEDIUM: facts that evolve slowly (API docs, conventions, standards).
            Default TTL: 90 days.
  - HIGH:   facts that change rapidly (versions, prices, rankings, news).
            Default TTL: 7 days.

Classification is a cheap keyword match on the topic + findings. Caller
can always pass an explicit volatility. Mis-classification errs toward
MEDIUM — never silently marks volatile facts LOW.

Conflict detection is local + structural: compare new findings with any
prior research on the same topic_slug, flag diverging version/date/number
tokens. No LLM involved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal


Volatility = Literal["low", "medium", "high"]


# Default TTLs by tier. None = no expiry.
_TTL_DAYS: dict[Volatility, int | None] = {
    "low": None,
    "medium": 90,
    "high": 7,
}


# Keyword signals. Lowercase substring matches against topic + findings.
_HIGH_VOLATILITY_TOKENS = (
    "latest", "current", "today", "this week", "this month",
    "price", "prices", "stock", "ranking", "leaderboard",
    "breaking", "news", "live",
    "newest version", "latest version", "current version",
)

_MEDIUM_VOLATILITY_TOKENS = (
    "version", "release", "docs", "documentation", "api",
    "roadmap", "status", "supported",
    "best practice", "convention",
)

_LOW_VOLATILITY_TOKENS = (
    "history", "historical", "biography", "published",
    "definition", "theorem", "proof",
    "arxiv", "doi", "isbn",
)


def classify_volatility(topic: str, findings: str = "") -> Volatility:
    """Best-effort tier classification. Defaults to MEDIUM when unsure."""
    blob = f"{topic}\n{findings}".lower()
    # High wins over everything.
    for tok in _HIGH_VOLATILITY_TOKENS:
        if tok in blob:
            return "high"
    # Low signals beat medium only when no version/release language present.
    has_low = any(tok in blob for tok in _LOW_VOLATILITY_TOKENS)
    has_medium = any(tok in blob for tok in _MEDIUM_VOLATILITY_TOKENS)
    if has_medium:
        return "medium"
    if has_low:
        return "low"
    return "medium"


def ttl_days_for(volatility: Volatility) -> int | None:
    return _TTL_DAYS.get(volatility, 90)


def stale_after(cached_at: datetime, volatility: Volatility) -> datetime | None:
    """Return absolute cutoff datetime after which the fact is considered stale."""
    days = ttl_days_for(volatility)
    if days is None:
        return None
    if cached_at.tzinfo is None:
        cached_at = cached_at.replace(tzinfo=timezone.utc)
    return cached_at + timedelta(days=days)


def is_stale(cached_at: datetime, volatility: Volatility, *, now: datetime | None = None) -> bool:
    cutoff = stale_after(cached_at, volatility)
    if cutoff is None:
        return False
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now >= cutoff


# ── Conflict detection ────────────────────────────────────────────────────


_VERSION_RE = re.compile(r"\bv?(\d+\.\d+(?:\.\d+)?(?:[-.][a-z0-9]+)?)\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\b")
_DATE_RE = re.compile(r"\b(20\d{2}(?:-\d{2}(?:-\d{2})?)?)\b")


def _extract_signals(text: str) -> dict[str, set[str]]:
    """Pull version/date/number tokens from text — these are the fields most
    likely to diverge between old and new beliefs on the same topic."""
    text = text or ""
    return {
        "versions": {m.group(1).lower() for m in _VERSION_RE.finditer(text)},
        "dates": {m.group(1) for m in _DATE_RE.finditer(text)},
        "numbers": {m.group(1) for m in _NUMBER_RE.finditer(text) if not _VERSION_RE.match(m.group(0))},
    }


@dataclass
class ConflictReport:
    has_conflict: bool = False
    prior_record_ids: list[str] = field(default_factory=list)
    diverging_signals: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    # Shape: {signal_kind: {"old": [...], "new": [...]}}

    def to_dict(self) -> dict:
        return {
            "has_conflict": self.has_conflict,
            "prior_record_ids": list(self.prior_record_ids),
            "diverging_signals": {
                k: {"old": sorted(v.get("old", [])), "new": sorted(v.get("new", []))}
                for k, v in self.diverging_signals.items()
            },
        }


def detect_conflict(new_findings: str, prior_records: list[dict]) -> ConflictReport:
    """Compare new findings against prior stored research on same topic.

    Parameters
    ----------
    new_findings:  text of the new research report
    prior_records: list of dicts with at least {"id", "content"}; typically
                   the result of a brain.search() for the same topic.

    A "conflict" is raised when the new and prior signal sets for the same
    kind (versions, dates, numbers) are both non-empty AND disjoint. This
    is deliberately strict — merely having *different* version tokens in
    new output (e.g. "0.3.0 -> 0.4.0 changelog") is not a conflict if the
    old belief is a subset.
    """
    report = ConflictReport()
    if not new_findings or not prior_records:
        return report

    new_sig = _extract_signals(new_findings)

    for rec in prior_records:
        content = str(rec.get("content") or "")
        if not content:
            continue
        old_sig = _extract_signals(content)

        rec_has_conflict = False
        for kind in ("versions", "dates", "numbers"):
            old = old_sig.get(kind, set())
            new = new_sig.get(kind, set())
            if not old or not new:
                continue
            if old.isdisjoint(new):
                rec_has_conflict = True
                slot = report.diverging_signals.setdefault(kind, {"old": set(), "new": set()})
                slot["old"] |= old
                slot["new"] |= new

        if rec_has_conflict:
            rid = str(rec.get("id") or "")
            if rid and rid not in report.prior_record_ids:
                report.prior_record_ids.append(rid)

    # Normalize set -> list inside dict
    for k, v in report.diverging_signals.items():
        report.diverging_signals[k] = {
            "old": list(v["old"]) if isinstance(v["old"], set) else v["old"],
            "new": list(v["new"]) if isinstance(v["new"], set) else v["new"],
        }

    report.has_conflict = bool(report.prior_record_ids)
    return report


def freshness_metadata(
    volatility: Volatility,
    *,
    now: datetime | None = None,
) -> dict:
    """Emit the metadata stamp to attach to a stored research record."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    cutoff = stale_after(now, volatility)
    return {
        "volatility": volatility,
        "cached_at": now.isoformat(),
        "stale_after": cutoff.isoformat() if cutoff else None,
        "ttl_days": ttl_days_for(volatility),
    }


# ── Status surfacing + revalidation queue ────────────────────────────────────
#
# Phase 4 wants truth-pressure to be a continuous layer rather than a one-shot
# gate. A claim's lifecycle is:
#
#   fresh                — within TTL, no conflict
#   stale_soft           — past TTL but within 2× TTL; suggest revalidation
#   stale_hard           — past 2× TTL; treat as expired until revalidated
#   conflict_unresolved  — someone tried to supersede, pending decision
#   conflict_resolved    — a later write was accepted as supersession
#   superseded           — this record was explicitly replaced
#   no_expiry            — volatility=low, TTL=None
#
# Status is computed from metadata, not stored redundantly. Persistent flags
# like "unresolved_conflict" and "superseded_by" are set by the write path
# when it decides to mark instead of overwrite.


TruthStatus = Literal[
    "fresh",
    "stale_soft",
    "stale_hard",
    "conflict_unresolved",
    "conflict_resolved",
    "superseded",
    "no_expiry",
    "unknown",
]


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def truth_status(meta: dict, *, now: datetime | None = None) -> TruthStatus:
    """Compute lifecycle status from record metadata.

    Explicit flags win over time-based state: a record marked `superseded`
    stays superseded even if it was fresh at the time of supersession.
    """
    meta = meta or {}
    if meta.get("superseded_by"):
        return "superseded"
    if meta.get("unresolved_conflict"):
        return "conflict_unresolved"
    if meta.get("conflict_resolved"):
        return "conflict_resolved"

    volatility: Volatility = meta.get("volatility", "medium")  # type: ignore
    ttl = ttl_days_for(volatility)
    if ttl is None:
        return "no_expiry"

    cached_at = _parse_iso(meta.get("cached_at"))
    if cached_at is None:
        return "unknown"
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    age = (now - cached_at).total_seconds()
    ttl_s = ttl * 86400
    if age < ttl_s:
        return "fresh"
    if age < 2 * ttl_s:
        return "stale_soft"
    return "stale_hard"


def needs_revalidation(meta: dict, *, now: datetime | None = None) -> bool:
    """True when a record should be rechecked against fresh sources."""
    status = truth_status(meta, now=now)
    return status in ("stale_soft", "stale_hard", "conflict_unresolved")


@dataclass
class RevalidationEntry:
    record_id: str
    topic: str
    volatility: Volatility
    status: TruthStatus
    cached_at: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "topic": self.topic,
            "volatility": self.volatility,
            "status": self.status,
            "cached_at": self.cached_at,
            "reason": self.reason,
        }


def build_revalidation_queue(
    records: list[dict],
    *,
    now: datetime | None = None,
) -> list[RevalidationEntry]:
    """Scan a list of record-shaped dicts and return those needing review.

    Expected record shape (minimum):
        {"id": str, "metadata": dict, "topic"?: str}

    Ordering: conflict_unresolved first, then stale_hard, then stale_soft
    (within each bucket, oldest cached_at first so the most-decayed knowledge
    is revalidated first).
    """
    entries: list[RevalidationEntry] = []
    for rec in records or []:
        meta = rec.get("metadata") or {}
        status = truth_status(meta, now=now)
        if status not in ("stale_soft", "stale_hard", "conflict_unresolved"):
            continue
        rid = str(rec.get("id") or "")
        if not rid:
            continue
        topic = str(meta.get("topic") or rec.get("topic") or "")
        volatility: Volatility = meta.get("volatility", "medium")  # type: ignore
        cached_at = str(meta.get("cached_at") or "")
        if status == "conflict_unresolved":
            reason = "conflict flagged; pending resolution"
        elif status == "stale_hard":
            reason = f"age > 2× TTL for volatility={volatility}"
        else:
            reason = f"age > TTL for volatility={volatility}"
        entries.append(RevalidationEntry(
            record_id=rid,
            topic=topic,
            volatility=volatility,
            status=status,
            cached_at=cached_at,
            reason=reason,
        ))

    priority = {"conflict_unresolved": 0, "stale_hard": 1, "stale_soft": 2}
    entries.sort(key=lambda e: (priority.get(e.status, 9), e.cached_at or ""))
    return entries


def conflict_flag_metadata(
    conflict: ConflictReport,
    *,
    now: datetime | None = None,
) -> dict:
    """Metadata stamp to mark a record as carrying an unresolved conflict.

    Applied to the *prior* record(s) when a new write disagrees and the
    caller chose `flag` (not replace/append). Preserves history rather than
    overwriting belief — this is the Phase 4 contract: truth-pressure is
    continuous, not a single gate.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return {
        "unresolved_conflict": True,
        "conflict_flagged_at": now.isoformat(),
        "conflict_diverging": conflict.to_dict().get("diverging_signals") or {},
    }


def supersede_metadata(
    new_record_id: str,
    *,
    now: datetime | None = None,
) -> dict:
    """Metadata stamp applied to a prior record when a new write replaces it."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return {
        "superseded_by": new_record_id,
        "superseded_at": now.isoformat(),
    }
