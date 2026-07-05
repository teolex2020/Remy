"""Detect quoted live telemetry vs memory-grounded stable knowledge.

SAFETY NET ONLY. Do not treat this detector as primary defense against
self-metric hallucination. Keyword hints cannot cover the space of ways an
LLM can phrase a number, and expanding the hint list is a maintenance trap.

The architectural defense is structured metric injection: the agent never
writes self-metric numbers in prose, it references metric ids that the
renderer substitutes with values pulled from a snapshot. This module stays
as best-effort logging for whatever slips through that contract.

Live telemetry = numbers about the agent's own runtime state. Memory-grounded
knowledge = facts about the world that happen to contain numbers
(e.g. "65% людей стресують") — these are skipped via _WORLD_HINTS.
"""

from __future__ import annotations

import re

from remy.core.retrieval.claim_spans import ClaimSpan, EvidenceRequirement


# Words that, within a small radius of a number, mark it as live telemetry.
_LIVE_HINTS = (
    "стабільн", "волатильн", "температур", "темпер",
    "стрес", "цікавіст", "цікавост",
    "коefіцієнт", "coefficient",
    "thermal", "coherence", "entropy", "volatility",
    "stability", "curiosity",
)

# Words that mark the number as a record/belief count about the agent itself.
_RECORD_HINTS = ("записів", "записи", "record", "у пам'ят", "у памят", "в пам'ят")
_BELIEF_HINTS = ("переконан", "belief", "гіпотез")
_CLUSTER_HINTS = ("конфлікт", "contradict", "cluster", "кластер")

# Hints that mark the number as memory-grounded fact (skip — not live).
_WORLD_HINTS = (
    "людей", "користувач", "респондент", "загалом", "населенн",
    "країн", "компаній", "ринк", "ринок",
    "people", "users", "respondents", "population",
)

_NUM_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*%?")
_TEMP_RE = re.compile(r"\b0?[.,]\d{2,3}\b")


def detect_live_telemetry(text: str) -> list[ClaimSpan]:
    spans: list[ClaimSpan] = []

    for m in _NUM_RE.finditer(text):
        window = _window(text, m.start(), m.end()).lower()
        if _has_any(window, _WORLD_HINTS):
            continue

        num_str = m.group(1).replace(",", ".")
        try:
            value = float(num_str)
        except ValueError:
            continue

        is_percent = m.group(0).rstrip().endswith("%")

        if _has_any(window, _RECORD_HINTS) and _looks_count(value, is_percent):
            spans.append(_make_span(
                m, text, "record_count",
                EvidenceRequirement.TOOL_CALL_IN_TURN, value,
            ))
            continue

        if _has_any(window, _BELIEF_HINTS) and _looks_count(value, is_percent):
            spans.append(_make_span(
                m, text, "belief_count",
                EvidenceRequirement.TOOL_CALL_IN_TURN, value,
            ))
            continue

        if _has_any(window, _CLUSTER_HINTS) and _looks_count(value, is_percent):
            spans.append(_make_span(
                m, text, "belief_count",
                EvidenceRequirement.TOOL_CALL_IN_TURN, value,
            ))
            continue

        if _has_any(window, _LIVE_HINTS):
            spans.append(_make_span(
                m, text, "live_metric",
                EvidenceRequirement.FRESH_INTROSPECTION, value,
            ))
            continue

    for m in _TEMP_RE.finditer(text):
        window = _window(text, m.start(), m.end()).lower()
        if _has_any(window, _LIVE_HINTS):
            try:
                value = float(m.group(0).replace(",", "."))
            except ValueError:
                continue
            spans.append(_make_span(
                m, text, "live_metric",
                EvidenceRequirement.FRESH_INTROSPECTION, value,
            ))

    return _dedupe(spans)


def _make_span(m, text: str, claim_type, req, value: float) -> ClaimSpan:
    return ClaimSpan(
        text=m.group(0),
        span=m.span(),
        claim_type=claim_type,
        requires_evidence=req,
        detector="live_telemetry",
        numeric_value=value,
        context_window=_window(text, m.start(), m.end()),
    )


def _window(text: str, start: int, end: int, radius: int = 50) -> str:
    a = max(0, start - radius)
    b = min(len(text), end + radius)
    return text[a:b]


def _has_any(haystack: str, needles) -> bool:
    return any(n in haystack for n in needles)


def _looks_count(value: float, is_percent: bool) -> bool:
    # A "count" is an integer >= 10 and not a percent.
    if is_percent:
        return False
    return value >= 10 and float(value).is_integer()


def _dedupe(spans: list[ClaimSpan]) -> list[ClaimSpan]:
    """Keep first span covering each offset — avoids double-flagging the same number."""
    out = []
    seen_starts: set[int] = set()
    for s in spans:
        if s.span[0] in seen_starts:
            continue
        seen_starts.add(s.span[0])
        out.append(s)
    return out
