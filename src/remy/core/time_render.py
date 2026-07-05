"""Relative time rendering for memory records in LLM prompts.

LLMs reason poorly on raw ISO timestamps. This module converts a stored_at
value into a short locale-neutral label like "just now" / "2d ago" /
"2026-03-24" / "2025-03" so recall results carry a sense of "when" into the
prompt without dictating a natural language.

Why English/ISO and not Ukrainian: the output lands inside LLM prompts, not
user-facing UI. Locale-neutral tokens keep the bibrary language-agnostic and
avoid accidentally steering the model's response language.

Policy:
    < 2 min     → "just now"
    < 60 min    → "Nm ago"
    < 24 h      → "Nh ago"
    exactly yesterday (UTC calendar) → "yesterday"
    < 14 days   → "Nd ago"
    < 365 days  → "YYYY-MM-DD"    (ISO 8601)
    >= 365 days → "YYYY-MM"       (ISO 8601 year-month)

Timezone: all arithmetic in UTC. Naive datetimes are treated as UTC, matching
remy.core.retrieval.freshness.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Union


def _to_utc_datetime(ts: Union[float, int, str, datetime, None]) -> datetime | None:
    if ts is None or ts == "":
        return None
    if isinstance(ts, datetime):
        dt = ts
    elif isinstance(ts, (int, float)):
        try:
            dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    elif isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _coerce_now(now: Union[float, int, datetime, None]) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if isinstance(now, (int, float)):
        return datetime.fromtimestamp(float(now), tz=timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def format_age(
    ts: Union[float, int, str, datetime, None],
    now: Union[float, int, datetime, None] = None,
) -> str:
    """Return a short locale-neutral age label, or "" if unknown/future."""
    dt = _to_utc_datetime(ts)
    if dt is None:
        return ""

    now_dt = _coerce_now(now)
    delta_sec = (now_dt - dt).total_seconds()
    if delta_sec < 0:
        return ""

    if delta_sec < 120:
        return "just now"
    if delta_sec < 3600:
        return f"{int(delta_sec // 60)}m ago"
    if delta_sec < 86400:
        return f"{int(delta_sec // 3600)}h ago"

    days = int(delta_sec // 86400)
    if days == 1:
        return "yesterday"
    if days < 14:
        return f"{days}d ago"
    if days < 365:
        return dt.strftime("%Y-%m-%d")
    return dt.strftime("%Y-%m")


def format_age_labeled(
    ts: Union[float, int, str, datetime, None],
    kind: str = "record",
    now: Union[float, int, datetime, None] = None,
) -> str:
    """Return age wrapped with a short verb prefix, or "" if age is empty.

    kind → prefix:
        "message" → "said"
        "update"  → "updated"
        "record"  → "stored"   (default)
    """
    age = format_age(ts, now=now)
    if not age:
        return ""
    verb = {"message": "said", "update": "updated", "record": "stored"}.get(
        kind, "stored"
    )
    return f"{verb} {age}"
