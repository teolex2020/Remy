"""Deterministic temporal validity checks for claim-level web evidence."""

from __future__ import annotations

from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
import re
from typing import Any, Mapping, Sequence

from remy.core.retrieval.freshness import ttl_days_for
from remy.core.search_gateway import canonicalize_url


_HIGH_SIGNALS = (
    "latest",
    "currently",
    "current ",
    "today",
    "right now",
    "live ",
    "price",
    "prices",
    "costs ",
    "stock price",
    "exchange rate",
    "latest version",
    "current version",
    "зараз",
    "сьогодні",
    "поточн",
    "останн",
    "найновіш",
    "ціна",
    "коштує",
    "актуальн",
    "теперішн",
    "сейчас",
    "сегодня",
    "текущ",
    "последн",
    "цена",
    "стоит ",
)

_MEDIUM_SIGNALS = (
    "supported release",
    "supported version",
    "active version",
    "release status",
    "availability",
    "is available",
    "is supported",
    "roadmap status",
    "чинна верс",
    "підтримувана верс",
    "доступний зараз",
    "статус релізу",
    "версія підтримується",
    "поддерживаемая верс",
    "статус релиза",
)

_HISTORICAL_SCOPE_RE = re.compile(
    r"\b(?:as\s+of|in|during|on|станом\s+на|у|в|на)\s+"
    r"(?:20\d{2}(?:[-/.]\d{1,2}(?:[-/.]\d{1,2})?)?)\b",
    re.IGNORECASE,
)
_DATE_PREFIX_RE = re.compile(r"^(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?")
_TEMPORAL_KEYS = (
    "date",
    "published_at",
    "publication_date",
    "date_published",
    "published",
    "modified_at",
    "last_modified",
    "date_modified",
    "datePublished",
    "dateModified",
)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _signal_present(text: str, signal: str) -> bool:
    normalized = signal.strip()
    if normalized.isascii() and re.fullmatch(r"[a-z0-9 ]+", normalized):
        pattern = r"\b" + r"\s+".join(
            re.escape(part) for part in normalized.split()
        ) + r"\b"
        return bool(re.search(pattern, text))
    return normalized in text


def classify_claim_temporality(claim: str) -> dict[str, Any]:
    """Classify whether a claim requires current evidence."""
    text = " ".join(str(claim or "").casefold().split())
    high = [signal.strip() for signal in _HIGH_SIGNALS if _signal_present(text, signal)]
    medium = [
        signal.strip() for signal in _MEDIUM_SIGNALS if _signal_present(text, signal)
    ]
    explicit_historical_scope = bool(_HISTORICAL_SCOPE_RE.search(text))
    # A dated historical assertion remains historical unless it explicitly says
    # it is also current/latest today.
    current_override = any(
        _signal_present(text, signal)
        for signal in ("currently", "current ", "today", "right now", "зараз", "сьогодні")
    )
    if explicit_historical_scope and not current_override:
        return {
            "time_sensitive": False,
            "volatility": "low",
            "ttl_days": None,
            "signals": sorted(set(high + medium)),
            "historically_scoped": True,
        }
    volatility = "high" if high else "medium" if medium else "low"
    sensitive = bool(high or medium)
    return {
        "time_sensitive": sensitive,
        "volatility": volatility,
        "ttl_days": ttl_days_for(volatility) if sensitive else None,
        "signals": sorted(set(high + medium)),
        "historically_scoped": False,
    }


def parse_source_datetime(value: Any) -> datetime | None:
    """Parse common structured publication-date values without dependencies."""
    if isinstance(value, datetime):
        return _utc(value)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp /= 1000
        try:
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        return _utc(datetime.fromisoformat(normalized))
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(text)
        return _utc(parsed)
    except (TypeError, ValueError, OverflowError):
        pass
    match = _DATE_PREFIX_RE.match(text)
    if match:
        try:
            return datetime(
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3) or 1),
                tzinfo=timezone.utc,
            )
        except ValueError:
            return None
    return None


def source_datetime(source: Mapping[str, Any]) -> tuple[datetime | None, str]:
    """Return the best structured publication/update date and its field name."""
    containers = [source]
    for key in ("metadata", "evidence_packet", "result", "source"):
        value = source.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
    for container in containers:
        for key in _TEMPORAL_KEYS:
            parsed = parse_source_datetime(container.get(key))
            if parsed is not None:
                return parsed, key
    return None, ""


def assess_source_temporality(
    claim: str,
    source: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Assess one supporting source against a claim's freshness window."""
    classification = classify_claim_temporality(claim)
    observed_now = _utc(now or datetime.now(timezone.utc))
    published_at, date_field = source_datetime(source)
    base = {
        **classification,
        "source_url": canonicalize_url(str(source.get("url") or source.get("uri") or "")),
        "source_date": published_at.isoformat() if published_at else "",
        "date_field": date_field,
        "reference_time": observed_now.isoformat(),
        "age_days": None,
    }
    if not classification["time_sensitive"]:
        return {**base, "status": "not_applicable", "temporally_valid": True}
    if published_at is None:
        return {**base, "status": "undated", "temporally_valid": False}
    age_days = (observed_now - published_at).total_seconds() / 86_400
    base["age_days"] = round(age_days, 2)
    if age_days < -1:
        return {**base, "status": "future_dated", "temporally_valid": False}
    if age_days <= float(classification["ttl_days"] or 0):
        return {**base, "status": "fresh", "temporally_valid": True}
    return {**base, "status": "stale", "temporally_valid": False}


def evaluate_claim_temporal_evidence(
    claim: str,
    supporting_sources: Sequence[Mapping[str, Any]],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Aggregate temporal validity across the sources supporting one claim."""
    classification = classify_claim_temporality(claim)
    assessments = [
        assess_source_temporality(claim, source, now=now)
        for source in supporting_sources
    ]
    if not classification["time_sensitive"]:
        status = "historical" if classification["historically_scoped"] else "not_applicable"
        ready = True
    elif any(item["status"] == "fresh" for item in assessments):
        status = "fresh"
        ready = True
    elif any(item["status"] == "stale" for item in assessments):
        status = "stale"
        ready = False
    elif any(item["status"] == "future_dated" for item in assessments):
        status = "future_dated"
        ready = False
    else:
        status = "undated"
        ready = False
    result = {
        **classification,
        "status": status,
        "temporal_ready": ready,
        "fresh_source_count": sum(item["status"] == "fresh" for item in assessments),
        "dated_source_count": sum(bool(item["source_date"]) for item in assessments),
        "assessments": assessments,
    }
    if classification["time_sensitive"] and not ready:
        compact = " ".join(str(claim or "").split())[:180]
        result["repair_query"] = (
            f'{compact} current official source published within '
            f'{classification["ttl_days"]} days'
        )
    return result
