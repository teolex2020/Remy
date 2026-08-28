from remy.core.evidence_sufficiency import select_diverse_evidence


def _candidate(domain: str, content: str, *, score: float = 0.8, relevant: bool = True):
    return {
        "url": f"https://{domain}/article",
        "source": {"uri": f"https://{domain}/article", "trust_score": 60},
        "result": {"content": content},
        "query_relevance": {
            "score": score,
            "relevant": relevant,
            "matched_terms": ["database", "consistency"] if relevant else [],
        },
    }


def test_three_relevant_domains_meet_source_target():
    candidates = [
        _candidate("one.example", "database consistency replication quorum"),
        _candidate("two.example", "database consistency transaction isolation"),
        _candidate("three.example", "database consistency distributed consensus"),
    ]

    selected, metrics = select_diverse_evidence(
        "database consistency", candidates, minimum_sources=3
    )

    assert len(selected) == 3
    assert metrics["sufficient"] is True
    assert metrics["distinct_domains"] == 3
    assert metrics["query_coverage"] == 1.0


def test_duplicate_domain_is_deferred_for_independent_source():
    candidates = [
        _candidate("docs.one.example", "database consistency alpha", score=0.95),
        _candidate("blog.one.example", "database consistency beta", score=0.94),
        _candidate("two.example", "database consistency gamma", score=0.8),
        _candidate("three.example", "database consistency delta", score=0.79),
    ]

    selected, metrics = select_diverse_evidence(
        "database consistency", candidates, limit=3, minimum_sources=3
    )

    domains = {item["evidence_selection"]["domain"] for item in selected}
    assert domains == {"one.example", "two.example", "three.example"}
    assert metrics["sufficient"] is True


def test_insufficient_evidence_explains_every_failed_gate():
    candidates = [
        _candidate("one.example", "unrelated promotional landing page", relevant=False),
        _candidate("one.example", "another unrelated page", relevant=False),
    ]

    _selected, metrics = select_diverse_evidence(
        "database consistency", candidates, minimum_sources=3
    )

    assert metrics["sufficient"] is False
    assert "not_enough_readable_sources" in metrics["reasons"]
    assert "not_enough_relevant_sources" in metrics["reasons"]
    assert "not_enough_independent_domains" in metrics["reasons"]
