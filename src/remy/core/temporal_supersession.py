"""Safe, local resolution of contradictions caused by facts changing over time."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Mapping, Sequence

from remy.core.claim_temporal_evidence import (
    assess_source_temporality,
    classify_claim_temporality,
    source_datetime,
)
from remy.core.search_gateway import canonicalize_url
from remy.core.source_provenance_graph import build_source_provenance_graph


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_MUTABLE_SIGNALS = (
    "price",
    "cost",
    "version",
    "release",
    "status",
    "supported",
    "available",
    "availability",
    "ceo",
    "chief executive",
    "president",
    "prime minister",
    "head coach",
    "ціна",
    "кошту",
    "версі",
    "реліз",
    "статус",
    "підтрим",
    "доступн",
    "директор",
    "президент",
    "прем'єр",
    "цена",
    "стоит",
    "верси",
    "релиз",
    "поддерж",
    "доступ",
)
_VALUE_RE = re.compile(
    r"(?:[$в‚¬ВЈв‚ґ]\s?\d+(?:[.,]\d+)?)|"
    r"(?:\bv?\d+(?:\.\d+){1,3}(?:[-.][a-z0-9]+)?\b)|"
    r"(?:\b\d+(?:[.,]\d+)?%?\b)",
    re.IGNORECASE,
)
_SUBJECT_STOP = {
    "the", "a", "an", "is", "was", "are", "were", "now", "current",
    "currently", "latest", "today", "version", "price", "cost", "status",
    "supported", "release", "as", "of", "in", "on", "to", "for", "and",
}
_RESOLVED_STATUSES = {
    "resolved",
    "superseded",
    "resolved_by_temporal_supersession",
    "dismissed",
}


def _text(contradiction: Mapping[str, Any], side: str) -> str:
    keys = (
        ("claim_a", "finding_a", "content_a", "old_claim")
        if side == "a"
        else ("claim_b", "finding_b", "content_b", "new_claim")
    )
    return next(
        (str(contradiction.get(key) or "").strip() for key in keys if contradiction.get(key)),
        "",
    )


def _url(contradiction: Mapping[str, Any], side: str) -> str:
    keys = (
        ("source_a", "url_a", "old_source_url")
        if side == "a"
        else ("source_b", "url_b", "new_source_url")
    )
    return next(
        (
            canonicalize_url(str(contradiction.get(key) or ""))
            for key in keys
            if contradiction.get(key)
        ),
        "",
    )


def _subject_tokens(text: str) -> set[str]:
    scrubbed = _VALUE_RE.sub(" ", str(text or "").casefold())
    return {
        token
        for token in _TOKEN_RE.findall(scrubbed)
        if len(token) >= 3 and token not in _SUBJECT_STOP
    }


def claim_subject_similarity(left: str, right: str) -> float:
    """Compare the stable subject of two claims after removing changing values."""
    left_tokens = _subject_tokens(left)
    right_tokens = _subject_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return round(len(left_tokens & right_tokens) / len(left_tokens | right_tokens), 4)


def _has_mutable_semantics(left: str, right: str) -> bool:
    text = f"{left}\n{right}".casefold()
    classified = classify_claim_temporality(text)
    return bool(
        classified.get("time_sensitive")
        or any(_signal_present(text, signal) for signal in _MUTABLE_SIGNALS)
    )


def _signal_present(text: str, signal: str) -> bool:
    """Avoid treating ASCII fragments such as ``cost`` in ``costume`` as slots."""
    if signal.isascii() and signal.replace(" ", "").isalpha():
        return bool(re.search(rf"(?<![a-z]){re.escape(signal)}(?![a-z])", text))
    return signal in text


def _value_changed(left: str, right: str) -> bool:
    left_values = {match.group(0).casefold() for match in _VALUE_RE.finditer(left)}
    right_values = {match.group(0).casefold() for match in _VALUE_RE.finditer(right)}
    if left_values and right_values:
        return left_values != right_values
    return " ".join(left.casefold().split()) != " ".join(right.casefold().split())


def _already_resolved(contradiction: Mapping[str, Any]) -> bool:
    return str(
        contradiction.get("resolution_status")
        or contradiction.get("status")
        or "unresolved"
    ).casefold() in _RESOLVED_STATUSES


def evaluate_temporal_supersession(
    contradiction: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    *,
    now: datetime | None = None,
    minimum_subject_similarity: float = 0.45,
    maximum_authority_drop: float = 0.10,
) -> dict[str, Any]:
    """Resolve a mutable contradiction only when the newer evidence is safe."""
    result = dict(contradiction)
    if _already_resolved(contradiction):
        result.setdefault("resolution_status", "already_resolved")
        result.setdefault("resolution_reason", "contradiction_already_resolved")
        return result

    observed_now = now or datetime.now(timezone.utc)
    if observed_now.tzinfo is None:
        observed_now = observed_now.replace(tzinfo=timezone.utc)
    source_list = [dict(source) for source in sources if isinstance(source, Mapping)]
    graph = build_source_provenance_graph(source_list, minimum_content_chars=20)
    nodes = {
        str(node.get("url") or ""): node
        for node in graph.get("nodes") or []
        if isinstance(node, Mapping)
    }
    by_url = {
        canonicalize_url(str(source.get("url") or source.get("uri") or "")): source
        for source in source_list
        if source.get("url") or source.get("uri")
    }
    claim_a, claim_b = _text(contradiction, "a"), _text(contradiction, "b")
    url_a, url_b = _url(contradiction, "a"), _url(contradiction, "b")
    checks: dict[str, Any] = {
        "claims_present": bool(claim_a and claim_b),
        "sources_present": bool(url_a and url_b and url_a in by_url and url_b in by_url),
    }
    if not all(checks.values()):
        return {
            **result,
            "resolution_status": "unresolved",
            "resolution_reason": "missing_claim_or_source_evidence",
            "supersession_checks": checks,
        }

    date_a, field_a = source_datetime(by_url[url_a])
    date_b, field_b = source_datetime(by_url[url_b])
    checks["both_sources_dated"] = bool(date_a and date_b)
    if not date_a or not date_b or date_a == date_b:
        return {
            **result,
            "resolution_status": "unresolved",
            "resolution_reason": "source_dates_missing_or_equal",
            "supersession_checks": checks,
        }

    if date_a > date_b:
        new_claim, old_claim = claim_a, claim_b
        new_url, old_url = url_a, url_b
        new_date, old_date = date_a, date_b
        new_field, old_field = field_a, field_b
    else:
        new_claim, old_claim = claim_b, claim_a
        new_url, old_url = url_b, url_a
        new_date, old_date = date_b, date_a
        new_field, old_field = field_b, field_a

    similarity = claim_subject_similarity(old_claim, new_claim)
    mutable = _has_mutable_semantics(old_claim, new_claim)
    changed = _value_changed(old_claim, new_claim)
    new_node, old_node = nodes.get(new_url, {}), nodes.get(old_url, {})
    new_authority = float(new_node.get("authority_score") or 0.0)
    old_authority = float(old_node.get("authority_score") or 0.0)
    independent_roots = bool(
        new_node.get("evidence_root")
        and old_node.get("evidence_root")
        and new_node.get("evidence_root") != old_node.get("evidence_root")
    )
    temporal = assess_source_temporality(new_claim, by_url[new_url], now=observed_now)
    freshness_required = bool(
        classify_claim_temporality(f"{old_claim}\n{new_claim}").get("time_sensitive")
    )
    fresh_enough = bool(
        temporal.get("temporally_valid") if freshness_required else new_date <= observed_now
    )
    authority_ok = new_authority + maximum_authority_drop >= old_authority
    checks.update(
        {
            "newer_source": True,
            "mutable_semantics": mutable,
            "value_changed": changed,
            "subject_similarity": similarity,
            "subject_match": similarity >= minimum_subject_similarity,
            "independent_roots": independent_roots,
            "new_source_fresh": fresh_enough,
            "authority_not_weaker": authority_ok,
        }
    )
    safe = all(
        bool(checks[key])
        for key in (
            "mutable_semantics",
            "value_changed",
            "subject_match",
            "independent_roots",
            "new_source_fresh",
            "authority_not_weaker",
        )
    )
    if not safe:
        failed = [key for key, value in checks.items() if value is False]
        return {
            **result,
            "resolution_status": "unresolved",
            "resolution_reason": "supersession_safety_gate_failed",
            "failed_checks": failed,
            "supersession_checks": checks,
        }

    confidence = min(
        1.0,
        0.45 * similarity
        + 0.20 * max(0.0, min(new_authority, 1.0))
        + 0.20
        + 0.15 * min((new_date - old_date).days / 90, 1.0),
    )
    return {
        **result,
        "status": "resolved",
        "resolution_status": "resolved_by_temporal_supersession",
        "resolution_reason": "newer_independent_authoritative_evidence",
        "resolved_at": observed_now.isoformat(),
        "supersession_checks": checks,
        "supersession": {
            "old_claim": old_claim,
            "new_claim": new_claim,
            "old_source_url": old_url,
            "new_source_url": new_url,
            "old_source_date": old_date.isoformat(),
            "new_source_date": new_date.isoformat(),
            "old_date_field": old_field,
            "new_date_field": new_field,
            "old_evidence_root": str(old_node.get("evidence_root") or old_url),
            "new_evidence_root": str(new_node.get("evidence_root") or new_url),
            "old_authority_score": old_authority,
            "new_authority_score": new_authority,
            "effective_at": new_date.isoformat(),
            "confidence": round(confidence, 3),
            "history_preserved": True,
        },
    }


def evaluate_temporal_supersessions(
    contradictions: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    return [
        evaluate_temporal_supersession(item, sources, now=now)
        for item in contradictions
        if isinstance(item, Mapping)
    ]
