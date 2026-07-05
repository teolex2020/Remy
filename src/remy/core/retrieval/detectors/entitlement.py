"""Detect entitlement claims: "я знайшла / я виділила / я виявила" + factual target.

Fires only when the agent speaks in first person about an active discovery
(a research result, a list, a source) — not about feelings or states.
"я почуваюся бадьоро" must NOT trip.
"""

from __future__ import annotations

import re

from remy.core.retrieval.claim_spans import ClaimSpan, EvidenceRequirement


# First-person discovery verbs (Ukrainian fem/masc + English).
_VERB_RE = re.compile(
    r"\bя\s+("
    r"знайш(?:ла|ов)|"
    r"виділ(?:ила|ив)|"
    r"вияв(?:ила|ив)|"
    r"помітила|помітив|"
    r"дослідила|дослідив|"
    r"перевірила|перевірив|"
    r"проаналізувала|проаналізував"
    r")\b",
    re.I,
)

# Words that signal a factual target (vs. feeling/state).
# Keep stems specific enough to avoid accidental substring matches
# (e.g. "роб" used to match "проблему").
_FACTUAL_TARGETS = (
    "робіт", "роботи", "роботу",
    "статт", "папер", "paper", "article",
    "дослідженн", "джерел", "source",
    "лід", "лідів", "lead", "клієнт", "client", "контакт",
    "компані", "проєкт", "project",
    "url", "посиланн", "link",
    "arxiv", "doi",
    "ключов",  # "ключових робіт", "ключові статті"
)

# Words that signal non-factual targets (skip).
_STATE_WORDS = (
    "почуваюс", "почуваєш",
    "настр", "стрес", "радіс", "сум",
    "проблем",  # "я виявила проблему" — internal, not factual lookup
)


def detect_entitlement(text: str) -> list[ClaimSpan]:
    spans: list[ClaimSpan] = []
    for m in _VERB_RE.finditer(text):
        window = _window(text, m.start(), m.end(), radius=80).lower()
        if _has_any(window, _STATE_WORDS) and not _has_any(window, _FACTUAL_TARGETS):
            continue
        if not _has_any(window, _FACTUAL_TARGETS):
            continue

        spans.append(
            ClaimSpan(
                text=m.group(0),
                span=m.span(),
                claim_type="entitlement",
                requires_evidence=EvidenceRequirement.TOOL_CALL_ANY,
                detector="entitlement",
                context_window=_window(text, m.start(), m.end(), radius=80),
            )
        )
    return spans


def _window(text: str, start: int, end: int, radius: int) -> str:
    a = max(0, start - radius)
    b = min(len(text), end + radius)
    return text[a:b]


def _has_any(haystack: str, needles) -> bool:
    return any(n in haystack for n in needles)
