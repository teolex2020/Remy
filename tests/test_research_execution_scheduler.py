from remy.core.research_execution_scheduler import (
    build_same_run_recovery,
    reconcile_execution_schedule,
)
from remy.core.research_query_planner import build_research_query_plan


def _search(query, url):
    return {
        "type": "tool_call",
        "tool": "web_search",
        "args": {"query": query},
        "result": {"results": [{"url": url, "title": "Evidence"}]},
    }


def _fetch(url, content="Evidence " * 30):
    return {
        "type": "tool_call",
        "tool": "extract_content",
        "args": {"url": url},
        "result": {"url": url, "content": content},
    }


def _complete_log(plan):
    log = []
    for index, query in enumerate(plan["queries"], start=1):
        url = f"https://source{index}.example/evidence"
        log.extend((_search(query["text"], url), _fetch(url)))
    return log


def test_complete_three_lane_schedule_is_sufficient():
    plan = build_research_query_plan("AI agent memory architecture")
    schedule = reconcile_execution_schedule(plan, _complete_log(plan))

    assert schedule["sufficient"] is True
    assert schedule["executed_lanes"] == 3
    assert schedule["fetched_lanes"] == 3
    assert schedule["distinct_domains"] == 3
    assert schedule["next_action"] == "complete"


def test_pretty_answer_without_tools_cannot_complete():
    plan = build_research_query_plan("AI agent memory architecture")
    schedule = reconcile_execution_schedule(plan, [])

    assert schedule["sufficient"] is False
    assert "planned_lanes_not_executed" in schedule["reasons"]
    assert schedule["next_action"] == "execute_missing_lane"
    assert len(schedule["repair_queries"]) >= 3


def test_search_without_fetch_requires_fetch_action():
    plan = build_research_query_plan("AI agent memory architecture")
    log = [
        _search(query["text"], f"https://source{i}.example/page")
        for i, query in enumerate(plan["queries"], start=1)
    ]
    schedule = reconcile_execution_schedule(plan, log)

    assert schedule["executed_lanes"] == 3
    assert schedule["fetched_lanes"] == 0
    assert schedule["next_action"] == "fetch_discovered_source"
    assert all("readable full text source" in q for q in schedule["repair_queries"][:3])


def test_short_fetch_is_not_counted_as_readable_evidence():
    plan = build_research_query_plan("database replication", mode="speed")
    query = plan["queries"][0]["text"]
    url = "https://docs.example/replication"
    schedule = reconcile_execution_schedule(plan, [_search(query, url), _fetch(url, "short")])

    assert schedule["readable_source_count"] == 0
    assert schedule["fetch_failures"][0]["reason"] == "unreadable"


def test_three_pages_on_one_domain_fail_independence_gate():
    plan = build_research_query_plan("AI agent memory architecture")
    log = []
    for index, query in enumerate(plan["queries"], start=1):
        url = f"https://same.example/evidence-{index}"
        log.extend((_search(query["text"], url), _fetch(url)))
    schedule = reconcile_execution_schedule(plan, log)

    assert schedule["readable_source_count"] == 3
    assert schedule["distinct_domains"] == 1
    assert schedule["sufficient"] is False
    assert "not_enough_independent_domains" in schedule["reasons"]
    assert schedule["repair_queries"][-1].endswith("independent source different publisher")


def test_subdomains_of_one_publisher_are_not_independent():
    plan = build_research_query_plan("AI agent memory architecture")
    log = []
    for index, query in enumerate(plan["queries"], start=1):
        url = f"https://section{index}.publisher.co.uk/evidence"
        log.extend((_search(query["text"], url), _fetch(url)))
    schedule = reconcile_execution_schedule(plan, log)

    assert schedule["domains"] == ["publisher.co.uk"]
    assert schedule["sufficient"] is False


def test_unplanned_search_does_not_satisfy_a_lane():
    plan = build_research_query_plan("AI agent memory architecture")
    schedule = reconcile_execution_schedule(
        plan,
        [_search("weather in Kyiv", "https://weather.example/kyiv")],
    )

    assert schedule["executed_lanes"] == 0
    assert schedule["unmatched_searches"] == ["weather in Kyiv"]


def test_query_variation_can_match_planned_lane():
    plan = build_research_query_plan("WebAuthn passkey security", mode="speed")
    planned = plan["queries"][0]["text"]
    varied = planned.replace("official documentation", "official docs")
    url = "https://w3.org/webauthn"
    schedule = reconcile_execution_schedule(plan, [_search(varied, url), _fetch(url)])

    assert schedule["lanes"][0]["search_calls"] == 1


def test_stable_schedule_id_for_same_plan():
    plan = build_research_query_plan("SQLite WAL concurrency")
    assert reconcile_execution_schedule(plan, [])["schedule_id"] == reconcile_execution_schedule(plan, [])["schedule_id"]


def test_recovery_targets_missing_search_lanes():
    plan = build_research_query_plan("AI agent memory architecture")
    schedule = reconcile_execution_schedule(plan, [])
    recovery = build_same_run_recovery(schedule)

    assert recovery["should_retry"] is True
    assert recovery["attempt"] == 1
    assert recovery["step_budget"] <= 4
    assert recovery["timeout_sec"] == 30
    assert recovery["actions"][0]["action"] == "search"


def test_recovery_prefers_fetch_for_discovered_url():
    plan = build_research_query_plan("database replication", mode="speed")
    first = plan["queries"][0]
    url = "https://docs.example/replication"
    schedule = reconcile_execution_schedule(plan, [_search(first["text"], url)])
    recovery = build_same_run_recovery(schedule)

    assert recovery["actions"][0]["action"] == "fetch"
    assert recovery["actions"][0]["url"] == url


def test_recovery_does_not_repeat_after_limit():
    plan = build_research_query_plan("AI agent memory architecture")
    schedule = reconcile_execution_schedule(plan, [])
    recovery = build_same_run_recovery(schedule, attempt=1, max_attempts=1)

    assert recovery["should_retry"] is False
    assert recovery["reason"] == "recovery_limit_reached"


def test_recovery_skips_timeout_and_error_workers():
    plan = build_research_query_plan("AI agent memory architecture")
    schedule = reconcile_execution_schedule(plan, [])

    assert not build_same_run_recovery(schedule, worker_status="timeout")["should_retry"]
    assert not build_same_run_recovery(schedule, worker_status="error")["should_retry"]


def test_recovery_skips_already_sufficient_schedule():
    plan = build_research_query_plan("AI agent memory architecture")
    schedule = reconcile_execution_schedule(plan, _complete_log(plan))

    recovery = build_same_run_recovery(schedule)
    assert recovery["should_retry"] is False
    assert recovery["reason"] == "evidence_already_sufficient"


def test_recovery_can_be_disabled_for_non_citation_runs():
    plan = build_research_query_plan("AI agent memory architecture")
    schedule = reconcile_execution_schedule(plan, [])

    recovery = build_same_run_recovery(schedule, enabled=False)
    assert recovery["should_retry"] is False
    assert recovery["reason"] == "same_run_recovery_disabled"


def test_recovery_uses_prefetch_allow_list():
    plan = build_research_query_plan("database replication", mode="speed")
    first = plan["queries"][0]
    low_value = "https://same.example/duplicate"
    preferred = "https://independent.example/evidence"
    log = [
        {
            "type": "tool_call",
            "tool": "web_search",
            "args": {"query": first["text"]},
            "result": {"results": [{"url": low_value}, {"url": preferred}]},
        }
    ]
    schedule = reconcile_execution_schedule(plan, log)
    recovery = build_same_run_recovery(
        schedule, preferred_fetch_urls=[preferred]
    )

    assert recovery["actions"][0]["url"] == preferred


def test_recovery_stops_when_marginal_gain_is_saturated():
    recovery = build_same_run_recovery(
        {
            "sufficient": False,
            "stop_reason": "marginal_gain_saturated",
            "lanes": [],
        }
    )

    assert recovery["should_retry"] is False
    assert recovery["reason"] == "marginal_gain_saturated"
