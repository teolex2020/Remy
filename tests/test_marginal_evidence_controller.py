from remy.core.marginal_evidence_controller import (
    apply_marginal_evidence_to_schedule,
    evaluate_marginal_evidence,
    prioritize_fetch_candidates,
)


def _candidate(url, snippet, title="Database consistency evidence"):
    return {"url": url, "title": title, "snippet": snippet}


def _source(url, unique):
    content = (
        "Database consistency replication evidence explains quorum and transactions. "
        f"{unique} " * 18
    )
    return {"url": url, "title": "Database consistency", "content": content}


def test_prefetch_prioritizes_relevant_independent_publishers():
    result = prioritize_fetch_candidates(
        "database consistency",
        [
            _candidate("https://one.example/a", "database consistency quorum"),
            _candidate("https://one.example/b", "database consistency transactions"),
            _candidate("https://two.example/a", "database consistency consensus"),
            _candidate("https://noise.example/ad", "buy cheap hosting advertisement", "Sale"),
        ],
        limit=3,
    )

    assert result["selected_count"] == 2
    assert {item["prefetch_decision"]["domain"] for item in result["selected"]} == {
        "one.example",
        "two.example",
    }
    assert result["publisher_suppressed"] == 1
    assert any(item["reason"] == "irrelevant" for item in result["rejected"])


def test_prefetch_suppresses_existing_and_duplicate_urls():
    existing = [_source("https://one.example/a", "alpha")]
    result = prioritize_fetch_candidates(
        "database consistency",
        [
            _candidate("https://one.example/a", "database consistency alpha"),
            _candidate("https://one.example/a?utm_source=x", "database consistency alpha"),
            _candidate("https://two.example/b", "database consistency beta"),
        ],
        existing_sources=existing,
    )

    assert result["selected_urls"] == ["https://two.example/b"]
    assert result["duplicate_suppressed"] == 2


def test_three_high_gain_sources_are_sufficient():
    assessment = evaluate_marginal_evidence(
        "database consistency",
        [
            _source("https://one.example/a", "quorum majority failure tolerance"),
            _source("https://two.example/b", "serializable isolation anomaly prevention"),
            _source("https://three.example/c", "raft consensus leader election log"),
        ],
    )

    assert assessment["sufficient"] is True
    assert assessment["decision"] == "stop_sufficient"
    assert assessment["accepted_source_count"] == 3
    assert assessment["accepted_domain_count"] == 3


def test_duplicate_content_does_not_count_as_new_evidence():
    duplicate = _source("https://one.example/a", "same repeated material")["content"]
    assessment = evaluate_marginal_evidence(
        "database consistency",
        [
            {"url": "https://one.example/a", "content": duplicate},
            {"url": "https://two.example/b", "content": duplicate},
            {"url": "https://three.example/c", "content": duplicate},
        ],
    )

    assert assessment["accepted_source_count"] == 1
    assert assessment["duplicate_rejected"] == 2
    assert assessment["saturated"] is True
    assert assessment["decision"] == "stop_saturated_incomplete"


def test_irrelevant_long_page_is_rejected():
    assessment = evaluate_marginal_evidence(
        "database consistency",
        [{"url": "https://noise.example/a", "content": "cooking recipe ingredients " * 30}],
    )

    assert assessment["accepted_source_count"] == 0
    assert assessment["rows"][0]["reason"] == "irrelevant"


def test_subdomains_do_not_fake_independent_publishers():
    assessment = evaluate_marginal_evidence(
        "database consistency",
        [
            _source("https://docs.publisher.co.uk/a", "quorum majority"),
            _source("https://blog.publisher.co.uk/b", "isolation anomalies"),
            _source("https://news.publisher.co.uk/c", "consensus logs"),
        ],
    )

    assert assessment["accepted_domain_count"] == 1
    assert assessment["sufficient"] is False


def test_assessment_can_reopen_superficially_sufficient_schedule():
    schedule = {
        "status": "sufficient",
        "sufficient": True,
        "reasons": [],
        "repair_queries": [],
    }
    assessment = {
        "sufficient": False,
        "saturated": False,
        "reasons": ["not_enough_high_gain_sources"],
        "repair_queries": ["database consistency independent novel evidence"],
    }
    updated = apply_marginal_evidence_to_schedule(schedule, assessment)

    assert updated["sufficient"] is False
    assert updated["next_action"] == "diversify_sources"
    assert "marginal:not_enough_high_gain_sources" in updated["reasons"]
    assert updated["repair_queries"]


def test_saturated_assessment_recommends_no_more_same_run_fetching():
    updated = apply_marginal_evidence_to_schedule(
        {"sufficient": True, "reasons": [], "repair_queries": []},
        {
            "sufficient": False,
            "saturated": True,
            "reasons": ["marginal_gain_saturated"],
            "repair_queries": ["different publisher"],
        },
    )

    assert updated["next_action"] == "stop_saturated_incomplete"
    assert updated["stop_reason"] == "marginal_gain_saturated"
