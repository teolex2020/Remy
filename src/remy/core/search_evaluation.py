"""Deterministic quality metrics and regression checks for search ranking.

The live web is not reproducible: engines change their result sets, networks
fail, and pages disappear.  This module therefore contains only pure metric
code.  Snapshot and live runners can share the same calculations without
putting network behaviour into CI.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any
from urllib.parse import urlsplit

from remy.core.search_gateway import canonicalize_url


DEFAULT_REGRESSION_TOLERANCES: dict[str, float] = {
    "recall_at_10": 0.03,
    "ndcg_at_10": 0.03,
    "mrr": 0.05,
    "target_domain_hit_rate": 0.03,
    "independent_domain_rate": 0.05,
    "duplicate_rate": 0.02,
}


def _domain(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower().removeprefix("www.")
    except ValueError:
        return ""


def _matches_domain(url: str, expected: str) -> bool:
    host = _domain(url)
    wanted = expected.lower().removeprefix("www.")
    return bool(host and wanted and (host == wanted or host.endswith("." + wanted)))


def _dcg(grades: Sequence[int]) -> float:
    return sum(
        (2 ** max(0, int(grade)) - 1) / math.log2(position + 2)
        for position, grade in enumerate(grades)
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * max(0.0, min(1.0, percentile))
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def normalize_judgments(judgments: Mapping[str, Any] | None) -> dict[str, int]:
    """Canonicalize URL keys and clamp relevance grades to ``0..3``."""
    normalized: dict[str, int] = {}
    for url, raw_grade in (judgments or {}).items():
        canonical = canonicalize_url(str(url))
        if not canonical:
            continue
        try:
            grade = int(raw_grade)
        except (TypeError, ValueError):
            grade = 0
        normalized[canonical] = max(0, min(3, grade))
    return normalized


def evaluate_ranking(
    candidates: Sequence[Mapping[str, Any]],
    judgments: Mapping[str, Any] | None,
    *,
    k: int = 10,
    expected_domains: Sequence[str] = (),
    min_independent_domains: int = 1,
    raw_candidate_count: int | None = None,
    raw_unique_candidate_count: int | None = None,
) -> dict[str, Any]:
    """Evaluate one ranked result list against graded URL judgments."""
    limit = max(1, int(k))
    normalized = normalize_judgments(judgments)
    urls = [
        canonicalize_url(str(item.get("uri") or item.get("url") or ""))
        for item in candidates[:limit]
    ]
    urls = [url for url in urls if url]
    grades = [normalized.get(url, 0) for url in urls]
    relevant_total = sum(1 for grade in normalized.values() if grade > 0)
    relevant_retrieved = sum(1 for grade in grades if grade > 0)
    recall = relevant_retrieved / relevant_total if relevant_total else 1.0
    reciprocal_rank = next(
        (1.0 / position for position, grade in enumerate(grades, start=1) if grade > 0),
        0.0,
    )
    ideal_grades = sorted(normalized.values(), reverse=True)[:limit]
    ideal_dcg = _dcg(ideal_grades)
    ndcg = _dcg(grades) / ideal_dcg if ideal_dcg else 1.0

    unique_urls = len(set(urls))
    output_duplicates = max(0, len(urls) - unique_urls)
    duplicate_rate = output_duplicates / len(urls) if urls else 0.0
    raw_count = max(len(urls), int(raw_candidate_count or len(urls)))
    if raw_unique_candidate_count is None:
        raw_unique_count = unique_urls
    else:
        raw_unique_count = max(0, min(raw_count, int(raw_unique_candidate_count)))
    duplicate_suppressed = max(0, raw_count - raw_unique_count)
    duplicate_suppression_rate = duplicate_suppressed / raw_count if raw_count else 0.0

    domains = {_domain(url) for url in urls if _domain(url)}
    desired_domain_count = max(1, int(min_independent_domains))
    independent_domain_rate = min(1.0, len(domains) / desired_domain_count)
    expected = [str(domain) for domain in expected_domains if str(domain).strip()]
    target_hit = (
        any(_matches_domain(url, domain) for url in urls for domain in expected)
        if expected
        else True
    )

    return {
        "recall_at_10": round(recall, 4),
        "ndcg_at_10": round(ndcg, 4),
        "mrr": round(reciprocal_rank, 4),
        "relevant_retrieved": relevant_retrieved,
        "relevant_total": relevant_total,
        "result_count": len(urls),
        "independent_domain_count": len(domains),
        "independent_domain_rate": round(independent_domain_rate, 4),
        "target_domain_hit": target_hit,
        "duplicate_rate": round(duplicate_rate, 4),
        "duplicate_suppressed": duplicate_suppressed,
        "duplicate_suppression_rate": round(duplicate_suppression_rate, 4),
    }


def aggregate_case_metrics(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Produce stable macro metrics from per-case benchmark results."""
    if not cases:
        return {
            "case_count": 0,
            "recall_at_10": 0.0,
            "ndcg_at_10": 0.0,
            "mrr": 0.0,
            "target_domain_hit_rate": 0.0,
            "independent_domain_rate": 0.0,
            "duplicate_rate": 0.0,
            "duplicate_suppression_rate": 0.0,
            "local_cache_hit_rate": 0.0,
            "external_search_rate": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
        }

    def average(name: str) -> float:
        return mean(float(case.get("metrics", {}).get(name, 0.0)) for case in cases)

    latencies = [float(case.get("duration_ms") or 0.0) for case in cases]
    external_rate = mean(bool(case.get("external_search_used", True)) for case in cases)
    return {
        "case_count": len(cases),
        "recall_at_10": round(average("recall_at_10"), 4),
        "ndcg_at_10": round(average("ndcg_at_10"), 4),
        "mrr": round(average("mrr"), 4),
        "target_domain_hit_rate": round(
            mean(bool(case.get("metrics", {}).get("target_domain_hit")) for case in cases),
            4,
        ),
        "independent_domain_rate": round(average("independent_domain_rate"), 4),
        "duplicate_rate": round(average("duplicate_rate"), 4),
        "duplicate_suppression_rate": round(average("duplicate_suppression_rate"), 4),
        "local_cache_hit_rate": round(1.0 - external_rate, 4),
        "external_search_rate": round(external_rate, 4),
        "latency_p50_ms": round(_percentile(latencies, 0.50), 2),
        "latency_p95_ms": round(_percentile(latencies, 0.95), 2),
    }


def compare_reports(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    tolerances: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Compare summaries and report only material quality regressions.

    Latency is intentionally not a snapshot gate because sub-millisecond local
    timings are machine-dependent. Live runs still report p50/p95 for humans.
    """
    allowed = dict(DEFAULT_REGRESSION_TOLERANCES)
    allowed.update({str(key): float(value) for key, value in (tolerances or {}).items()})
    current_summary = current.get("summary", current)
    baseline_summary = baseline.get("summary", baseline)
    regressions: list[dict[str, Any]] = []
    deltas: dict[str, float] = {}

    for metric, tolerance in allowed.items():
        if metric not in current_summary or metric not in baseline_summary:
            continue
        now = float(current_summary[metric])
        before = float(baseline_summary[metric])
        delta = now - before
        deltas[metric] = round(delta, 4)
        lower_is_better = metric == "duplicate_rate"
        regressed = delta > tolerance if lower_is_better else delta < -tolerance
        if regressed:
            regressions.append(
                {
                    "metric": metric,
                    "baseline": before,
                    "current": now,
                    "delta": round(delta, 4),
                    "tolerance": tolerance,
                }
            )

    return {
        "passed": not regressions,
        "regressions": regressions,
        "deltas": deltas,
        "tolerances": allowed,
    }
