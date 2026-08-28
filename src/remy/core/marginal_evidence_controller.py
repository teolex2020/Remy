"""Local marginal-value controls for research discovery and fetched evidence."""

from __future__ import annotations

import copy
import ipaddress
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from remy.core.search_gateway import canonicalize_url
from remy.core.search_relevance import assess_query_relevance


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_COMMON_SECOND_LEVEL_SUFFIXES = {
    "co.in", "co.jp", "co.uk", "com.au", "com.br", "com.cn", "com.mx",
    "com.tr", "org.uk",
}


def _domain(value: str) -> str:
    try:
        host = (urlsplit(value or "").hostname or "").casefold().removeprefix("www.")
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


def _fields(item: Mapping[str, Any]) -> tuple[str, str, str, str]:
    result = item.get("result") if isinstance(item.get("result"), Mapping) else {}
    source = item.get("source") if isinstance(item.get("source"), Mapping) else {}
    url = canonicalize_url(
        str(
            item.get("url") or item.get("uri") or item.get("href")
            or result.get("url") or source.get("uri") or ""
        )
    )
    title = str(item.get("title") or result.get("title") or source.get("title") or "")
    snippet = str(
        item.get("snippet") or item.get("body") or result.get("snippet")
        or source.get("snippet") or ""
    )
    content = str(item.get("content") or result.get("content") or "")
    return url, title, snippet, content


def _tokens(value: str) -> set[str]:
    return {
        token for token in _TOKEN_RE.findall((value or "")[:50_000].casefold())
        if len(token) >= 4
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def prioritize_fetch_candidates(
    topic: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    existing_sources: Sequence[Mapping[str, Any]] = (),
    limit: int = 4,
    per_domain_limit: int = 1,
    duplicate_threshold: float = 0.82,
) -> dict[str, Any]:
    """Create a fetch allow-list before paying the cost of page extraction."""
    existing_urls: set[str] = set()
    existing_tokens: list[set[str]] = []
    domain_counts: dict[str, int] = {}
    for item in existing_sources:
        url, _title, snippet, content = _fields(item)
        if url:
            existing_urls.add(url)
            domain = _domain(url)
            if domain:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        material_tokens = _tokens(content or snippet)
        if material_tokens:
            existing_tokens.append(material_tokens)

    prepared: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for index, original in enumerate(candidates):
        url, title, snippet, content = _fields(original)
        if not url:
            rejected.append({"url": "", "reason": "invalid_url"})
            continue
        if url in existing_urls or url in seen_urls:
            rejected.append({"url": url, "reason": "duplicate_url"})
            continue
        seen_urls.add(url)
        relevance = assess_query_relevance(
            topic, title=title, snippet=snippet, content=content, url=url
        )
        tokens = _tokens(content or snippet or title)
        redundancy = max((_jaccard(tokens, prior) for prior in existing_tokens), default=0.0)
        if not relevance["relevant"]:
            rejected.append({"url": url, "reason": "irrelevant", "relevance": relevance})
            continue
        if redundancy >= duplicate_threshold:
            rejected.append(
                {"url": url, "reason": "content_duplicate", "redundancy": round(redundancy, 3)}
            )
            continue
        prepared.append(
            {
                "index": index,
                "item": dict(original),
                "url": url,
                "domain": _domain(url),
                "tokens": tokens,
                "relevance": relevance,
                "base_score": float(relevance["score"]),
            }
        )

    selected: list[dict[str, Any]] = []
    selected_tokens = list(existing_tokens)
    while prepared and len(selected) < max(1, int(limit)):
        best = None
        best_key = None
        for candidate in prepared:
            domain = candidate["domain"]
            if domain and domain_counts.get(domain, 0) >= max(1, int(per_domain_limit)):
                continue
            redundancy = max(
                (_jaccard(candidate["tokens"], prior) for prior in selected_tokens),
                default=0.0,
            )
            novelty = 1.0 - redundancy
            new_domain = bool(domain and not domain_counts.get(domain, 0))
            score = candidate["base_score"] * 0.65 + novelty * 0.25 + (0.10 if new_domain else 0.0)
            key = (round(score, 6), -candidate["index"])
            if best_key is None or key > best_key:
                best, best_key = candidate, key
        if best is None:
            break
        prepared.remove(best)
        selected_tokens.append(best["tokens"])
        if best["domain"]:
            domain_counts[best["domain"]] = domain_counts.get(best["domain"], 0) + 1
        item = dict(best["item"])
        item["url"] = best["url"]
        item["prefetch_decision"] = {
            "score": best_key[0],
            "domain": best["domain"],
            "reason": "highest_local_marginal_value",
        }
        selected.append(item)

    for candidate in prepared:
        rejected.append(
            {
                "url": candidate["url"],
                "reason": (
                    "same_publisher_budget"
                    if candidate["domain"] and domain_counts.get(candidate["domain"], 0)
                    else "fetch_budget"
                ),
            }
        )
    return {
        "version": 1,
        "selected": selected,
        "selected_urls": [str(item.get("url") or "") for item in selected],
        "selected_count": len(selected),
        "rejected": rejected,
        "rejected_count": len(rejected),
        "duplicate_suppressed": sum(
            item.get("reason") in {"duplicate_url", "content_duplicate"} for item in rejected
        ),
        "publisher_suppressed": sum(
            item.get("reason") == "same_publisher_budget" for item in rejected
        ),
        "method": "local_prefetch_marginal_value",
    }


def evaluate_marginal_evidence(
    topic: str,
    sources: Sequence[Mapping[str, Any]],
    *,
    minimum_sources: int = 3,
    minimum_domains: int = 3,
    minimum_gain: float = 0.22,
    duplicate_threshold: float = 0.82,
) -> dict[str, Any]:
    """Measure the incremental value of each successfully fetched page."""
    rows: list[dict[str, Any]] = []
    accepted_tokens: list[set[str]] = []
    accepted_urls: set[str] = set()
    accepted_domains: set[str] = set()
    seen_urls: set[str] = set()

    for item in sources:
        url, title, snippet, content = _fields(item)
        if not url or url in seen_urls or len(re.sub(r"\s+", " ", content).strip()) < 120:
            continue
        seen_urls.add(url)
        domain = _domain(url)
        tokens = _tokens(content)
        relevance = assess_query_relevance(
            topic, title=title, snippet=snippet, content=content, url=url
        )
        redundancy = max((_jaccard(tokens, prior) for prior in accepted_tokens), default=0.0)
        novelty = 1.0 - redundancy
        new_domain = bool(domain and domain not in accepted_domains)
        quality = min(len(content) / 2_000, 1.0)
        gain = round(
            float(relevance["score"]) * 0.45
            + novelty * 0.35
            + (0.15 if new_domain else 0.0)
            + quality * 0.05,
            4,
        )
        if not relevance["relevant"]:
            accepted = False
            reason = "irrelevant"
        elif redundancy >= duplicate_threshold:
            accepted = False
            reason = "content_duplicate"
        elif gain < float(minimum_gain):
            accepted = False
            reason = "low_marginal_gain"
        else:
            accepted = True
            reason = "marginal_value_accepted"
            accepted_urls.add(url)
            accepted_tokens.append(tokens)
            if domain:
                accepted_domains.add(domain)
        rows.append(
            {
                "url": url,
                "domain": domain,
                "accepted": accepted,
                "reason": reason,
                "marginal_gain": gain,
                "relevance": relevance,
                "novelty": round(novelty, 4),
                "redundancy": round(redundancy, 4),
                "new_domain": new_domain,
            }
        )

    source_target = max(1, int(minimum_sources))
    domain_target = max(1, int(minimum_domains))
    enough_sources = len(accepted_urls) >= source_target
    enough_domains = len(accepted_domains) >= domain_target
    sufficient = enough_sources and enough_domains
    rejected_tail = [row["reason"] for row in rows[-2:] if not row["accepted"]]
    saturated = len(rejected_tail) >= 2 and all(
        reason in {"content_duplicate", "low_marginal_gain"} for reason in rejected_tail
    )
    if sufficient:
        decision = "stop_sufficient"
    elif saturated:
        decision = "stop_saturated_incomplete"
    elif not enough_domains and accepted_urls:
        decision = "continue_diversify"
    else:
        decision = "continue_fetch"
    reasons = []
    if not enough_sources:
        reasons.append("not_enough_high_gain_sources")
    if not enough_domains:
        reasons.append("not_enough_high_gain_domains")
    if saturated:
        reasons.append("marginal_gain_saturated")
    repair_queries = []
    if not sufficient:
        repair_queries.append(f"{topic} independent source with novel evidence different publisher")
    return {
        "version": 1,
        "method": "local_postfetch_marginal_gain",
        "sufficient": sufficient,
        "stop_recommended": sufficient or saturated,
        "saturated": saturated,
        "decision": decision,
        "accepted_source_count": len(accepted_urls),
        "source_target": source_target,
        "accepted_domain_count": len(accepted_domains),
        "domain_target": domain_target,
        "accepted_urls": sorted(accepted_urls),
        "accepted_domains": sorted(accepted_domains),
        "average_marginal_gain": round(
            sum(row["marginal_gain"] for row in rows if row["accepted"])
            / max(1, sum(row["accepted"] for row in rows)),
            4,
        ),
        "duplicate_rejected": sum(row["reason"] == "content_duplicate" for row in rows),
        "low_gain_rejected": sum(row["reason"] == "low_marginal_gain" for row in rows),
        "reasons": reasons,
        "repair_queries": repair_queries,
        "rows": rows,
    }


def apply_marginal_evidence_to_schedule(
    schedule: Mapping[str, Any], assessment: Mapping[str, Any]
) -> dict[str, Any]:
    """Make scheduler completion conditional on high-value, non-duplicate evidence."""
    updated = copy.deepcopy(dict(schedule))
    updated["marginal_evidence"] = dict(assessment)
    if assessment.get("sufficient"):
        return updated
    updated["sufficient"] = False
    updated["status"] = "incomplete"
    reasons = list(updated.get("reasons") or [])
    for reason in assessment.get("reasons") or []:
        marker = f"marginal:{reason}"
        if marker not in reasons:
            reasons.append(marker)
    updated["reasons"] = reasons
    updated["next_action"] = (
        "stop_saturated_incomplete"
        if assessment.get("saturated")
        else "diversify_sources"
    )
    updated["stop_reason"] = (
        "marginal_gain_saturated" if assessment.get("saturated") else "continue"
    )
    updated["repair_queries"] = list(
        dict.fromkeys(
            [
                *list(updated.get("repair_queries") or []),
                *list(assessment.get("repair_queries") or []),
            ]
        )
    )[:8]
    return updated
