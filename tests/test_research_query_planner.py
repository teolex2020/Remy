from remy.core.research_query_planner import (
    build_research_query_plan,
    format_query_plan,
    query_texts,
)


def _intents(plan):
    return [item["intent"] for item in plan["queries"]]


def test_balanced_plan_uses_three_complementary_evidence_lanes():
    plan = build_research_query_plan("AI agent memory architecture")

    assert plan["query_count"] == 3
    assert _intents(plan) == ["primary", "corroboration", "counterevidence"]
    assert plan["has_primary_lane"] is True
    assert plan["has_corroboration_lane"] is True
    assert plan["has_counterevidence_lane"] is True


def test_speed_mode_respects_two_query_budget():
    plan = build_research_query_plan("database replication", mode="speed")

    assert plan["budget"] == 2
    assert plan["query_count"] == 2
    assert _intents(plan) == ["primary", "corroboration"]


def test_deep_temporal_plan_adds_freshness_lane():
    plan = build_research_query_plan("latest PostgreSQL update 2026", mode="deep")

    assert "freshness" in _intents(plan)
    freshness = next(item for item in plan["queries"] if item["intent"] == "freshness")
    assert "official latest date" in freshness["text"]


def test_paper_scope_seeks_original_and_replication_evidence():
    plan = build_research_query_plan(
        "retrieval augmented generation benchmark",
        source_scope="papers",
    )

    assert "original study paper dataset methodology" in plan["queries"][0]["text"]
    corroboration = next(
        item for item in plan["queries"] if item["intent"] == "corroboration"
    )
    assert "systematic review replication" in corroboration["text"]


def test_domain_scope_stays_inside_operator_boundary_first():
    plan = build_research_query_plan(
        "Python pathlib",
        source_scope="domain",
        source_domains=["https://docs.python.org/"],
    )

    assert plan["queries"][0]["intent"] == "scoped_primary"
    assert plan["queries"][0]["text"].startswith("site:docs.python.org Python pathlib")


def test_repair_query_has_priority_over_generic_discovery():
    plan = build_research_query_plan(
        "PostgreSQL backup",
        repair_queries=["PostgreSQL WAL restore official primary source evidence"],
    )

    assert plan["queries"][0]["origin"] == "repair"
    assert plan["queries"][0]["intent"] == "gap_repair"


def test_conflict_repair_is_classified_as_contradiction_lane():
    plan = build_research_query_plan(
        "product pricing",
        repair_queries=["product costs $20 resolve conflicting evidence"],
    )

    assert plan["queries"][0]["intent"] == "contradiction"
    assert plan["has_counterevidence_lane"] is True


def test_duplicate_seed_queries_are_collapsed_locally():
    plan = build_research_query_plan(
        "SQLite concurrency",
        seed_queries=["SQLite WAL concurrency", "sqlite wal concurrency"],
        mode="deep",
    )

    texts = query_texts(plan)
    assert sum(text.casefold() == "sqlite wal concurrency" for text in texts) == 1


def test_query_ids_are_stable_for_same_input():
    left = build_research_query_plan("WebAuthn passkey security")
    right = build_research_query_plan("WebAuthn passkey security")

    assert [q["query_id"] for q in left["queries"]] == [
        q["query_id"] for q in right["queries"]
    ]


def test_formatter_exposes_intent_and_rationale():
    text = format_query_plan(build_research_query_plan("FastAPI dependency injection"))

    assert "[primary]" in text
    assert "[corroboration]" in text
    assert "first-party material" in text


def test_empty_topic_returns_empty_plan():
    plan = build_research_query_plan("  ")

    assert plan["query_count"] == 0
    assert query_texts(plan) == []
