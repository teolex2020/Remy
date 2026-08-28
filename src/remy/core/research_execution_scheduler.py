"""Deterministic execution accounting for evidence-aware research plans.

The LLM remains the tool-using executor, while this state machine decides
whether planned search lanes were actually searched and fetched.  It prevents
an eloquent response from being mistaken for completed evidence collection.
"""

from __future__ import annotations

from hashlib import sha1
import ipaddress
import json
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit


_FETCH_TOOLS = {"extract_content", "http_get", "browse_page"}
_COMMON_SECOND_LEVEL_SUFFIXES = {
    "co.in", "co.jp", "co.uk", "com.au", "com.br", "com.cn", "com.mx",
    "com.tr", "org.uk",
}


def _normalize(text: Any) -> str:
    return " ".join(re.findall(r"[\w]+", str(text or "").casefold(), flags=re.UNICODE))


def _tokens(text: Any) -> set[str]:
    return {token for token in _normalize(text).split() if len(token) > 2}


def _query_similarity(left: str, right: str) -> float:
    if _normalize(left) == _normalize(right):
        return 1.0
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


def _url(value: Any) -> str:
    text = str(value or "").strip().rstrip(".,;)")
    try:
        split = urlsplit(text)
    except ValueError:
        return ""
    return text if split.scheme in {"http", "https"} and split.hostname else ""


def _domain(value: str) -> str:
    try:
        host = (urlsplit(value).hostname or "").casefold().removeprefix("www.")
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


def _urls_from(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        direct = [_url(value.get(key)) for key in ("url", "uri", "href", "final_url")]
        nested: list[str] = []
        for key in ("results", "candidates", "sources"):
            items = value.get(key) or []
            if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
                for item in items:
                    nested.extend(_urls_from(item))
        return list(dict.fromkeys([item for item in [*direct, *nested] if item]))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(dict.fromkeys(url for item in value for url in _urls_from(item)))
    return list(
        dict.fromkeys(
            _url(match)
            for match in re.findall(r"https?://[^\s<>\]\[\"']+", str(value or ""))
            if _url(match)
        )
    )


def _result_material(result: Any) -> str:
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return result
        return _result_material(parsed)
    if isinstance(result, Mapping):
        return " ".join(
            str(result.get(key) or "")
            for key in ("content", "text", "body", "snippet", "result")
        ).strip()
    return ""


def reconcile_execution_schedule(
    query_plan: Mapping[str, Any],
    session_log: Sequence[Mapping[str, Any]],
    *,
    minimum_sources: int = 3,
    minimum_domains: int = 3,
) -> dict[str, Any]:
    """Reconcile planned lanes with actual search/fetch tool calls."""
    states = []
    for item in query_plan.get("queries") or []:
        if not isinstance(item, Mapping):
            continue
        states.append(
            {
                "query_id": str(item.get("query_id") or ""),
                "query": str(item.get("text") or ""),
                "intent": str(item.get("intent") or "discovery"),
                "search_calls": 0,
                "discovered_urls": [],
                "fetched_urls": [],
                "readable_urls": [],
                "status": "planned",
            }
        )

    active_index: int | None = None
    url_owner: dict[str, int] = {}
    unmatched_searches: list[str] = []
    fetch_failures: list[dict[str, str]] = []

    for entry in session_log or []:
        if entry.get("type") != "tool_call":
            continue
        tool = str(entry.get("tool") or "")
        args = entry.get("args") if isinstance(entry.get("args"), Mapping) else {}
        result = entry.get("result_full") or entry.get("result") or {}

        if tool == "web_search":
            executed = str(args.get("query") or args.get("q") or "").strip()
            ranked = sorted(
                (
                    (_query_similarity(executed, state["query"]), index)
                    for index, state in enumerate(states)
                ),
                reverse=True,
            )
            if ranked and ranked[0][0] >= 0.62:
                active_index = ranked[0][1]
                state = states[active_index]
                state["search_calls"] += 1
                discovered = _urls_from(result)
                state["discovered_urls"] = list(
                    dict.fromkeys([*state["discovered_urls"], *discovered])
                )
                state["status"] = "discovered" if discovered else "searched_empty"
                for discovered_url in discovered:
                    url_owner[discovered_url] = active_index
            else:
                active_index = None
                if executed:
                    unmatched_searches.append(executed)
            continue

        if tool not in _FETCH_TOOLS:
            continue
        fetched_url = _url(args.get("url") or args.get("uri") or args.get("href"))
        owner = url_owner.get(fetched_url, active_index)
        if owner is None or owner >= len(states):
            continue
        state = states[owner]
        if fetched_url:
            state["fetched_urls"] = list(
                dict.fromkeys([*state["fetched_urls"], fetched_url])
            )
        material = _result_material(result)
        if fetched_url and len(re.sub(r"\s+", " ", material).strip()) >= 120:
            state["readable_urls"] = list(
                dict.fromkeys([*state["readable_urls"], fetched_url])
            )
            state["status"] = "fetched"
        else:
            state["status"] = "fetch_failed"
            fetch_failures.append(
                {"query_id": state["query_id"], "url": fetched_url, "reason": "unreadable"}
            )

    readable_urls = list(
        dict.fromkeys(url for state in states for url in state["readable_urls"])
    )
    domains = sorted({_domain(url) for url in readable_urls if _domain(url)})
    executed_lanes = sum(state["search_calls"] > 0 for state in states)
    fetched_lanes = sum(bool(state["readable_urls"]) for state in states)
    required_lanes = len(states)
    target_sources = max(1, int(minimum_sources))
    target_domains = max(1, int(minimum_domains))
    reasons: list[str] = []
    if executed_lanes < required_lanes:
        reasons.append("planned_lanes_not_executed")
    if fetched_lanes < required_lanes:
        reasons.append("planned_lanes_not_fetched")
    if len(readable_urls) < target_sources:
        reasons.append("not_enough_readable_sources")
    if len(domains) < target_domains:
        reasons.append("not_enough_independent_domains")

    repair_queries: list[str] = []
    for state in states:
        if state["search_calls"] == 0:
            repair_queries.append(state["query"])
        elif not state["readable_urls"]:
            repair_queries.append(f'{state["query"]} readable full text source')
    if len(domains) < target_domains and states:
        repair_queries.append(
            f'{query_plan.get("topic", states[0]["query"])} independent source different publisher'
        )
    repair_queries = list(dict.fromkeys(query for query in repair_queries if query))[:8]

    if not reasons:
        next_action = "complete"
        stop_reason = "evidence_targets_satisfied"
    elif "planned_lanes_not_executed" in reasons:
        next_action = "execute_missing_lane"
        stop_reason = "continue"
    elif "planned_lanes_not_fetched" in reasons:
        next_action = "fetch_discovered_source"
        stop_reason = "continue"
    else:
        next_action = "diversify_sources"
        stop_reason = "continue"

    return {
        "version": 1,
        "schedule_id": "rs-" + sha1(
            "|".join(state["query_id"] for state in states).encode("utf-8")
        ).hexdigest()[:12],
        "status": "sufficient" if not reasons else "incomplete",
        "sufficient": not reasons,
        "stop_reason": stop_reason,
        "next_action": next_action,
        "required_lanes": required_lanes,
        "executed_lanes": executed_lanes,
        "fetched_lanes": fetched_lanes,
        "readable_source_count": len(readable_urls),
        "source_target": target_sources,
        "distinct_domains": len(domains),
        "domain_target": target_domains,
        "domains": domains,
        "reasons": reasons,
        "repair_queries": repair_queries,
        "unmatched_searches": unmatched_searches[:8],
        "fetch_failures": fetch_failures[:8],
        "lanes": states,
    }


def build_same_run_recovery(
    schedule: Mapping[str, Any],
    *,
    attempt: int = 0,
    max_attempts: int = 1,
    worker_status: str = "success",
    enabled: bool = True,
    preferred_fetch_urls: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build one bounded recovery pass from an incomplete execution schedule."""
    if not enabled:
        return {
            "should_retry": False,
            "reason": "same_run_recovery_disabled",
            "attempt": attempt,
            "max_attempts": max_attempts,
            "actions": [],
        }
    if bool(schedule.get("sufficient")):
        return {
            "should_retry": False,
            "reason": "evidence_already_sufficient",
            "attempt": attempt,
            "max_attempts": max_attempts,
            "actions": [],
        }
    if schedule.get("stop_reason") == "marginal_gain_saturated":
        return {
            "should_retry": False,
            "reason": "marginal_gain_saturated",
            "attempt": attempt,
            "max_attempts": max_attempts,
            "actions": [],
        }
    if attempt >= max(0, int(max_attempts)):
        return {
            "should_retry": False,
            "reason": "recovery_limit_reached",
            "attempt": attempt,
            "max_attempts": max_attempts,
            "actions": [],
        }
    if worker_status in {"error", "timeout", "cancelled", "saturated"}:
        return {
            "should_retry": False,
            "reason": f"unsafe_terminal_status:{worker_status}",
            "attempt": attempt,
            "max_attempts": max_attempts,
            "actions": [],
        }

    actions: list[dict[str, Any]] = []
    for lane in schedule.get("lanes") or []:
        if not isinstance(lane, Mapping):
            continue
        if not int(lane.get("search_calls") or 0):
            actions.append(
                {
                    "action": "search",
                    "query_id": lane.get("query_id", ""),
                    "intent": lane.get("intent", ""),
                    "query": lane.get("query", ""),
                    "reason": "planned lane was not searched",
                }
            )
            continue
        if not lane.get("readable_urls"):
            discovered = list(lane.get("discovered_urls") or [])
            preferred = (
                discovered
                if preferred_fetch_urls is None
                else [url for url in discovered if url in set(preferred_fetch_urls)]
            )
            if preferred:
                actions.append(
                    {
                        "action": "fetch",
                        "query_id": lane.get("query_id", ""),
                        "intent": lane.get("intent", ""),
                        "url": preferred[0],
                        "fallback_query": f'{lane.get("query", "")} readable full text source',
                        "reason": "search succeeded but no readable evidence was fetched",
                    }
                )
            else:
                actions.append(
                    {
                        "action": "search",
                        "query_id": lane.get("query_id", ""),
                        "intent": lane.get("intent", ""),
                        "query": f'{lane.get("query", "")} readable full text source',
                        "reason": "search returned no fetchable source",
                    }
                )

    if int(schedule.get("distinct_domains") or 0) < int(
        schedule.get("domain_target") or 3
    ):
        diversity_query = next(
            (
                str(query)
                for query in reversed(list(schedule.get("repair_queries") or []))
                if "different publisher" in str(query)
            ),
            "",
        )
        if diversity_query:
            actions.append(
                {
                    "action": "search",
                    "query_id": "diversity",
                    "intent": "corroboration",
                    "query": diversity_query,
                    "reason": "minimum independent publisher target was not met",
                }
            )

    unique_actions: list[dict[str, Any]] = []
    seen = set()
    for action in actions:
        key = (
            action.get("action"),
            action.get("query_id"),
            action.get("url"),
            action.get("query"),
        )
        if key not in seen:
            seen.add(key)
            unique_actions.append(action)
    # A recovery pass is intentionally small even when the original deep plan
    # was large. Remaining gaps persist to the next durable research cycle.
    unique_actions = unique_actions[:4]
    lines = []
    for action in unique_actions:
        if action["action"] == "fetch":
            lines.append(
                f'- FETCH {action["url"]} for [{action["intent"]}]. '
                f'If unreadable, search: {action["fallback_query"]}'
            )
        else:
            lines.append(
                f'- SEARCH [{action["intent"]}] {action["query"]}, then fetch one readable page.'
            )
    return {
        "should_retry": bool(unique_actions),
        "reason": "bounded_evidence_recovery" if unique_actions else "no_recoverable_action",
        "attempt": attempt + 1,
        "max_attempts": max_attempts,
        "step_budget": max(2, min(4, len(unique_actions) + 1)),
        "timeout_sec": 30,
        "actions": unique_actions,
        "instruction": "\n".join(lines),
    }
