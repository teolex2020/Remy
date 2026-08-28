"""Deterministic query-to-result relevance scoring without an external API."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


_STOP_WORDS = {
    "about",
    "and",
    "best",
    "для",
    "from",
    "how",
    "into",
    "latest",
    "про",
    "the",
    "this",
    "with",
    "або",
    "как",
    "що",
    "это",
    "який",
    "яка",
    "яке",
}
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_QUERY_OPERATOR_RE = re.compile(r"\b(?:site|filetype|inurl|intitle):\S+", re.IGNORECASE)


def query_terms(query: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", query or "").casefold()
    normalized = _QUERY_OPERATOR_RE.sub(" ", normalized)
    terms: list[str] = []
    for token in _TOKEN_RE.findall(normalized):
        if len(token) < 3 or token in _STOP_WORDS or token.isdigit() and len(token) < 4:
            continue
        if token not in terms:
            terms.append(token)
    return terms[:24]


def _term_matches(term: str, tokens: set[str]) -> bool:
    if term in tokens:
        return True
    if len(term) < 6:
        return False
    prefix = term[:5]
    return any(len(token) >= 6 and token.startswith(prefix) for token in tokens)


def assess_query_relevance(
    query: str,
    *,
    title: str = "",
    snippet: str = "",
    content: str = "",
    url: str = "",
) -> dict[str, Any]:
    """Return an explainable lexical relevance score in the range [0, 1]."""
    terms = query_terms(query)
    if not terms:
        return {
            "score": 0.5,
            "coverage": 1.0,
            "matched_terms": [],
            "missing_terms": [],
            "relevant": True,
            "reason": "query_has_no_distinctive_terms",
        }

    fields = {
        "title": unicodedata.normalize("NFKC", title or "").casefold(),
        "snippet": unicodedata.normalize("NFKC", snippet or "").casefold(),
        "content": unicodedata.normalize("NFKC", (content or "")[:50_000]).casefold(),
        "url": unicodedata.normalize("NFKC", url or "").casefold(),
    }
    field_tokens = {name: set(_TOKEN_RE.findall(text)) for name, text in fields.items()}
    weights = {"title": 1.0, "snippet": 0.65, "content": 0.45, "url": 0.25}

    matched: list[str] = []
    weighted_hits = 0.0
    for term in terms:
        hit_weight = max(
            (
                weight
                for field, weight in weights.items()
                if _term_matches(term, field_tokens[field])
            ),
            default=0.0,
        )
        if hit_weight:
            matched.append(term)
            weighted_hits += hit_weight

    coverage = len(matched) / len(terms)
    score = (weighted_hits / len(terms)) * 0.7 + coverage * 0.3
    normalized_query = " ".join(terms)
    if len(terms) >= 2 and normalized_query in fields["title"]:
        score += 0.12
    elif len(terms) >= 2 and normalized_query in fields["snippet"]:
        score += 0.06
    score = round(max(0.0, min(score, 1.0)), 3)

    minimum_matches = 1 if len(terms) <= 3 else 2
    relevant = len(matched) >= minimum_matches and (coverage >= 0.25 or score >= 0.35)
    return {
        "score": score,
        "coverage": round(coverage, 3),
        "matched_terms": matched,
        "missing_terms": [term for term in terms if term not in matched],
        "relevant": relevant,
        "reason": "lexical_coverage" if relevant else "insufficient_query_overlap",
    }
