"""Local evidence selection and sufficiency checks for web research.

The implementation is intentionally deterministic and dependency-free.  It
combines query coverage, source diversity, content redundancy, and source
quality so callers can explain why a set of pages is (or is not) sufficient.
"""

from __future__ import annotations

import ipaddress
import re
from itertools import combinations
from typing import Any, Sequence
from urllib.parse import urlsplit

from remy.core.search_relevance import assess_query_relevance, query_terms


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_COMMON_SECOND_LEVEL_SUFFIXES = {
    "co.in",
    "co.jp",
    "co.uk",
    "com.au",
    "com.br",
    "com.cn",
    "com.mx",
    "com.tr",
    "org.uk",
}


def _domain(value: str) -> str:
    try:
        host = (urlsplit(value or "").hostname or "").lower().removeprefix("www.")
    except ValueError:
        return ""
    if not host:
        return ""
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    suffix = ".".join(labels[-2:])
    width = 3 if suffix in _COMMON_SECOND_LEVEL_SUFFIXES else 2
    return ".".join(labels[-width:])


def _content_tokens(value: str) -> set[str]:
    tokens = _TOKEN_RE.findall((value or "")[:40_000].casefold())
    return {token for token in tokens if len(token) >= 4}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _extract_fields(item: dict[str, Any]) -> tuple[str, str, str, str]:
    result = item.get("result") if isinstance(item.get("result"), dict) else {}
    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    url = str(result.get("url") or item.get("url") or source.get("uri") or "")
    title = str(result.get("title") or source.get("title") or "")
    snippet = str(source.get("snippet") or item.get("snippet") or "")
    content = str(result.get("content") or item.get("content") or "")
    return url, title, snippet, content


def select_diverse_evidence(
    query: str,
    candidates: Sequence[dict[str, Any]],
    *,
    limit: int = 3,
    minimum_sources: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select evidence with lexical marginal relevance and report sufficiency."""
    target = max(1, int(minimum_sources))
    selection_limit = max(1, int(limit))
    terms = set(query_terms(query))
    prepared: list[dict[str, Any]] = []

    for index, original in enumerate(candidates):
        item = dict(original)
        url, title, snippet, content = _extract_fields(item)
        relevance = item.get("query_relevance")
        if not isinstance(relevance, dict):
            relevance = assess_query_relevance(
                query,
                title=title,
                snippet=snippet,
                content=content,
                url=url,
            )
        prepared.append(
            {
                "index": index,
                "item": item,
                "url": url,
                "domain": _domain(url),
                "tokens": _content_tokens(content),
                "matched": set(relevance.get("matched_terms") or []),
                "relevance": relevance,
            }
        )

    selected: list[dict[str, Any]] = []
    covered_terms: set[str] = set()
    selected_domains: set[str] = set()

    while prepared and len(selected) < selection_limit:
        best: dict[str, Any] | None = None
        best_key: tuple[float, float, int] | None = None
        for candidate in prepared:
            relevance_score = float(candidate["relevance"].get("score") or 0.0)
            new_terms = candidate["matched"] - covered_terms
            marginal_coverage = len(new_terms) / max(1, len(terms))
            redundancy = max(
                (
                    _jaccard(candidate["tokens"], chosen["tokens"])
                    for chosen in selected
                ),
                default=0.0,
            )
            new_domain = bool(
                candidate["domain"] and candidate["domain"] not in selected_domains
            )
            source = candidate["item"].get("source") or {}
            trust_score = max(-50.0, min(float(source.get("trust_score") or 0), 150.0))
            trust_normalized = (trust_score + 50.0) / 200.0
            marginal_score = (
                relevance_score * 0.55
                + marginal_coverage * 0.25
                + trust_normalized * 0.10
                + (0.10 if new_domain else 0.0)
                - redundancy * 0.30
            )
            key = (round(marginal_score, 6), relevance_score, -candidate["index"])
            if best_key is None or key > best_key:
                best = candidate
                best_key = key

        if best is None:
            break
        prepared.remove(best)
        new_terms = best["matched"] - covered_terms
        selected.append(best)
        covered_terms.update(best["matched"])
        if best["domain"]:
            selected_domains.add(best["domain"])
        best["item"]["evidence_selection"] = {
            "position": len(selected),
            "marginal_score": best_key[0] if best_key else 0.0,
            "new_query_terms": sorted(new_terms),
            "domain": best["domain"],
        }

    relevant_count = sum(
        1 for item in selected if item["relevance"].get("relevant") is True
    )
    query_coverage = len(covered_terms) / max(1, len(terms)) if terms else 1.0
    similarities = [
        _jaccard(left["tokens"], right["tokens"])
        for left, right in combinations(selected, 2)
    ]
    distinct_domains = len(selected_domains)
    reasons: list[str] = []
    if len(selected) < target:
        reasons.append("not_enough_readable_sources")
    if relevant_count < target:
        reasons.append("not_enough_relevant_sources")
    if distinct_domains < target:
        reasons.append("not_enough_independent_domains")
    if terms and query_coverage < 0.5:
        reasons.append("insufficient_query_coverage")

    metrics = {
        "sufficient": not reasons,
        "source_count": len(selected),
        "source_target": target,
        "relevant_source_count": relevant_count,
        "distinct_domains": distinct_domains,
        "query_coverage": round(query_coverage, 3),
        "average_relevance": round(
            sum(float(item["relevance"].get("score") or 0.0) for item in selected)
            / max(1, len(selected)),
            3,
        ),
        "maximum_content_redundancy": round(max(similarities, default=0.0), 3),
        "reasons": reasons,
        "method": "local_statistical_marginal_gain",
    }
    return [item["item"] for item in selected], metrics
