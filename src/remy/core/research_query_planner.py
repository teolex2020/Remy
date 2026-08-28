"""Deterministic, evidence-aware query planning for research runs.

The planner is deliberately local: it needs no model, account, API key, or
download.  It turns one broad topic into complementary search lanes instead of
issuing several cosmetic rewrites of the same query.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha1
import re
from typing import Any, Mapping, Sequence


_MODE_BUDGETS = {"speed": 2, "balanced": 3, "deep": 7}
_TEMPORAL_TERMS = {
    "latest", "today", "current", "recent", "new", "update", "2025", "2026",
    "останн", "сьогодні", "актуальн", "новин", "оновлен",
}
_RESEARCH_TERMS = {
    "research", "study", "paper", "survey", "benchmark", "evidence", "clinical",
    "дослідж", "статт", "науков", "доказ",
}


def _compact(value: Any, limit: int = 220) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" .")[:limit]


def _normalized_key(text: str) -> str:
    return " ".join(re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE))


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)
        if len(token) > 2
    }


def _near_duplicate(left: str, right: str) -> bool:
    left_key, right_key = _normalized_key(left), _normalized_key(right)
    if not left_key or not right_key:
        return True
    if left_key == right_key:
        return True
    a, b = _tokens(left_key), _tokens(right_key)
    if not a or not b:
        return False
    return len(a & b) / max(1, min(len(a), len(b))) >= 0.88


def _is_temporal(topic: str) -> bool:
    lowered = topic.casefold()
    return any(term in lowered for term in _TEMPORAL_TERMS)


def _looks_like_research(topic: str, source_scope: str) -> bool:
    lowered = topic.casefold()
    return source_scope == "papers" or any(term in lowered for term in _RESEARCH_TERMS)


def _primary_suffix(topic: str, source_scope: str) -> tuple[str, list[str]]:
    if _looks_like_research(topic, source_scope):
        return "original study paper dataset methodology", ["paper", "dataset", "methodology"]
    if any(term in topic.casefold() for term in ("software", "api", "library", "framework", "agent")):
        return "official documentation repository release notes", ["official_docs", "repository"]
    return "official primary source report data", ["official", "primary_source", "data"]


def _repair_intent(query: str) -> str:
    lowered = query.casefold()
    if any(term in lowered for term in ("conflict", "contradict", "resolve", "супереч")):
        return "contradiction"
    if any(term in lowered for term in ("independent", "corrobor", "підтверд")):
        return "corroboration"
    return "gap_repair"


def _candidate(
    text: str,
    intent: str,
    priority: int,
    rationale: str,
    expected_source_types: Sequence[str],
    *,
    origin: str = "generated",
) -> dict[str, Any]:
    # Short focused queries are easier for both local indexes and public search
    # engines to interpret than instruction-shaped paragraphs.
    compact = _compact(text, 118)
    return {
        "text": compact,
        "intent": intent,
        "priority": priority,
        "rationale": rationale,
        "expected_source_types": list(expected_source_types),
        "origin": origin,
    }


def build_research_query_plan(
    topic: str,
    *,
    mode: str = "balanced",
    source_scope: str = "web",
    source_domains: Sequence[str] = (),
    seed_queries: Sequence[str] = (),
    repair_queries: Sequence[str] = (),
    max_queries: int | None = None,
) -> dict[str, Any]:
    """Build a budgeted plan with complementary evidence-seeking intents."""
    subject = _compact(topic)
    if not subject:
        return {
            "version": 1,
            "strategy": "evidence_lanes",
            "topic": "",
            "mode": mode,
            "budget": 0,
            "query_count": 0,
            "lane_counts": {},
            "queries": [],
        }

    budget = max_queries if max_queries is not None else _MODE_BUDGETS.get(mode, 3)
    budget = max(1, min(int(budget), 12))
    candidates: list[dict[str, Any]] = []

    for query in repair_queries:
        text = _compact(query)
        if text:
            intent = _repair_intent(text)
            candidates.append(
                _candidate(
                    text,
                    intent,
                    0,
                    "Close a concrete unsupported, partial, or conflicting claim.",
                    ["primary_source", "independent_source"],
                    origin="repair",
                )
            )

    for query in seed_queries:
        text = _compact(query)
        if text:
            candidates.append(
                _candidate(
                    text,
                    "user_seed",
                    10,
                    "Preserve an explicit operator-provided search query.",
                    ["operator_requested"],
                    origin="user",
                )
            )

    domains = [
        re.sub(r"^https?://", "", _compact(domain)).strip("/").casefold()
        for domain in source_domains
        if _compact(domain)
    ]
    if source_scope == "domain":
        for domain in dict.fromkeys(domains):
            candidates.append(
                _candidate(
                    f"site:{domain} {subject}",
                    "scoped_primary",
                    15,
                    "Search the operator-approved domain boundary first.",
                    ["allowed_domain"],
                )
            )

    primary_suffix, primary_types = _primary_suffix(subject, source_scope)
    candidates.append(
        _candidate(
            f"{subject} {primary_suffix}",
            "primary",
            20,
            "Find first-party material or the original underlying evidence.",
            primary_types,
        )
    )

    if source_scope == "papers":
        corroboration_suffix = "systematic review replication independent study"
        corroboration_types = ["systematic_review", "independent_study"]
    elif source_scope == "discussions":
        corroboration_suffix = "independent user experience forum discussion"
        corroboration_types = ["discussion", "user_report"]
    else:
        corroboration_suffix = "independent analysis evidence comparison"
        corroboration_types = ["independent_analysis", "secondary_source"]
    candidates.append(
        _candidate(
            f"{subject} {corroboration_suffix}",
            "corroboration",
            30,
            "Seek a different publisher that can corroborate the primary evidence.",
            corroboration_types,
        )
    )
    candidates.append(
        _candidate(
            f"{subject} criticism limitations contradictory evidence",
            "counterevidence",
            40,
            "Actively search for disconfirming evidence and known limitations.",
            ["critical_analysis", "contradictory_source"],
        )
    )

    if _is_temporal(subject):
        candidates.append(
            _candidate(
                f"{subject} official latest date release announcement",
                "freshness",
                25,
                "Verify time-sensitive claims against a dated first-party source.",
                ["dated_primary_source", "release_notes"],
            )
        )

    selected: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["priority"]):
        if not candidate["text"]:
            continue
        if any(
            candidate["intent"] == item["intent"]
            and _near_duplicate(candidate["text"], item["text"])
            for item in selected
        ):
            continue
        candidate = dict(candidate)
        candidate["query_id"] = "q-" + sha1(
            f'{candidate["intent"]}|{candidate["text"]}'.encode("utf-8")
        ).hexdigest()[:10]
        selected.append(candidate)
        if len(selected) >= budget:
            break

    lane_counts = Counter(item["intent"] for item in selected)
    return {
        "version": 1,
        "strategy": "evidence_lanes",
        "topic": subject,
        "mode": mode,
        "source_scope": source_scope,
        "budget": budget,
        "query_count": len(selected),
        "lane_counts": dict(sorted(lane_counts.items())),
        "has_primary_lane": any(
            item["intent"] in {"primary", "scoped_primary", "user_seed"}
            for item in selected
        ),
        "has_corroboration_lane": any(
            item["intent"] in {"corroboration", "gap_repair"} for item in selected
        ),
        "has_counterevidence_lane": any(
            item["intent"] in {"counterevidence", "contradiction"} for item in selected
        ),
        "queries": selected,
    }


def query_texts(plan: Mapping[str, Any]) -> list[str]:
    """Return the execution-compatible string query list."""
    return [
        str(item.get("text") or "").strip()
        for item in (plan.get("queries") or [])
        if isinstance(item, Mapping) and str(item.get("text") or "").strip()
    ]


def format_query_plan(plan: Mapping[str, Any]) -> str:
    """Render a compact worker instruction with intent labels and rationale."""
    lines = []
    for item in plan.get("queries") or []:
        if not isinstance(item, Mapping):
            continue
        lines.append(
            f'- [{item.get("intent", "discovery")}] {item.get("text", "")} '
            f'— {item.get("rationale", "")}'
        )
    return "\n".join(lines)
