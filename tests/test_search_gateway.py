from __future__ import annotations

from dataclasses import dataclass

from remy.core.search_gateway import (
    LocalIndexSearchProvider,
    SearchGateway,
    SearchRequest,
    canonicalize_url,
    requires_live_discovery,
)


@dataclass
class FakeProvider:
    name: str
    results: list[dict] | None = None
    error: Exception | None = None

    def search(self, _request: SearchRequest) -> list[dict]:
        if self.error is not None:
            raise self.error
        return list(self.results or [])


def test_canonicalize_url_removes_tracking_fragment_and_default_port():
    value = "HTTPS://Example.COM:443/docs/?utm_source=x&b=2&a=1#install"

    assert canonicalize_url(value) == "https://example.com/docs?a=1&b=2"


def test_gateway_fuses_duplicate_urls_and_rewards_provider_agreement():
    gateway = SearchGateway(
        providers=[
            FakeProvider(
                "engine-a",
                [{"title": "Python docs", "href": "https://docs.python.org/3/?utm_source=a"}],
            ),
            FakeProvider(
                "engine-b",
                [
                    {
                        "title": "Official Python documentation",
                        "url": "https://docs.python.org/3#top",
                        "description": "The official language documentation.",
                    }
                ],
            ),
        ]
    )

    response = gateway.search(SearchRequest("python documentation", max_results=5))

    assert len(response.candidates) == 1
    candidate = response.candidates[0]
    assert candidate["uri"] == "https://docs.python.org/3"
    assert candidate["providers"] == ["engine-a", "engine-b"]
    assert candidate["provider_count"] == 2
    assert candidate["agreement_bonus"] == 1
    assert candidate["title"] == "Official Python documentation"


def test_local_multilingual_reranker_recovers_early_typo_and_can_be_disabled():
    results = [
        {
            "title": "Transaction isolation levels",
            "href": "https://generic.test/transaction-isolation",
            "body": "General transaction isolation levels.",
        },
        {
            "title": "Database transaction isolation levels",
            "href": "https://postgresql.org/about/transaction-isolation",
            "body": "Official database transaction isolation reference.",
        },
    ]
    enabled = SearchGateway(providers=[FakeProvider("engine", results)])
    disabled = SearchGateway(
        providers=[FakeProvider("engine", results)],
        enable_local_reranker=False,
    )

    enabled_response = enabled.search(
        SearchRequest("daatbase transaction isolation levels")
    )
    disabled_response = disabled.search(
        SearchRequest("daatbase transaction isolation levels")
    )

    assert enabled_response.candidates[0]["uri"].startswith("https://postgresql.org/")
    assert enabled_response.candidates[0]["local_rerank"]["matched_fuzzy"] == {
        "daatbase": "database"
    }
    assert disabled_response.candidates[0]["uri"].startswith("https://generic.test/")
    assert "local_rerank" not in disabled_response.candidates[0]


def test_gateway_keeps_partial_results_when_another_provider_fails():
    gateway = SearchGateway(
        providers=[
            FakeProvider("broken", error=TimeoutError("too slow")),
            FakeProvider(
                "healthy",
                [{"title": "Result", "href": "https://example.test/item", "body": "text"}],
            ),
        ]
    )

    response = gateway.search(SearchRequest("resilient search"))

    assert [candidate["uri"] for candidate in response.candidates] == [
        "https://example.test/item"
    ]
    assert response.had_provider_error is True
    diagnostics = response.diagnostics()
    assert diagnostics["attempts"][0]["status"] == "error"
    assert diagnostics["attempts"][1]["status"] == "ok"


def test_gateway_enforces_site_constraint_after_provider_search():
    gateway = SearchGateway(
        providers=[
            FakeProvider(
                "engine",
                [
                    {"title": "Allowed", "href": "https://docs.python.org/3/library/"},
                    {"title": "Spillover", "href": "https://example.com/python"},
                ],
            )
        ]
    )

    response = gateway.search(SearchRequest("site:docs.python.org pathlib"))

    assert response.site_constraint == "docs.python.org"
    assert [candidate["uri"] for candidate in response.candidates] == [
        "https://docs.python.org/3/library"
    ]


def test_gateway_uses_independent_recovery_providers_after_empty_primary():
    gateway = SearchGateway(
        providers=[FakeProvider("primary")],
        recovery_providers=[
            FakeProvider("fallback-empty"),
            FakeProvider(
                "fallback-ok",
                [{"title": "Recovered", "href": "https://example.test/recovered"}],
            ),
        ],
    )

    response = gateway.search(SearchRequest("recover me"))

    assert response.candidates[0]["uri"] == "https://example.test/recovered"
    assert [attempt.status for attempt in response.attempts] == ["empty", "empty", "ok"]


def test_three_fresh_local_results_skip_external_discovery():
    remote = FakeProvider(
        "remote",
        [{"title": "Should not run", "href": "https://remote.test/result"}],
    )
    local = FakeProvider(
        "local:web-index",
        [
            {
                "title": f"Database consistency source {number}",
                "href": f"https://source{number}.test/page",
                "body": "Database consistency evidence",
                "local_index": True,
                "local_fresh": True,
                "local_quality_score": 0.8,
            }
            for number in range(3)
        ],
    )
    gateway = SearchGateway(providers=[remote], local_provider=local)

    response = gateway.search(SearchRequest("database consistency"))

    assert response.external_search_used is False
    assert response.fresh_local_result_count == 3
    assert [attempt.provider for attempt in response.attempts] == ["local:web-index"]
    assert all(candidate["local_index"] for candidate in response.candidates)


def test_insufficient_local_results_are_supplemented_from_web():
    local = FakeProvider(
        "local:web-index",
        [
            {
                "title": "Database consistency local evidence",
                "href": "https://local.test/page",
                "local_index": True,
                "local_fresh": True,
            }
        ],
    )
    remote = FakeProvider(
        "remote",
        [
            {
                "title": "Database consistency current evidence",
                "href": "https://remote.test/result",
            }
        ],
    )
    gateway = SearchGateway(providers=[remote], local_provider=local)

    response = gateway.search(SearchRequest("database consistency"))

    assert response.external_search_used is True
    assert [attempt.provider for attempt in response.attempts] == [
        "local:web-index",
        "remote",
    ]
    assert {candidate["uri"] for candidate in response.candidates} == {
        "https://local.test/page",
        "https://remote.test/result",
    }


def test_live_query_refreshes_even_with_three_cached_results():
    local = FakeProvider(
        "local:web-index",
        [
            {
                "title": f"Latest database news {number}",
                "href": f"https://cached{number}.test/page",
                "body": "Latest database news",
                "local_index": True,
                "local_fresh": True,
            }
            for number in range(3)
        ],
    )
    remote = FakeProvider(
        "remote",
        [{"title": "Live result", "href": "https://remote.test/live"}],
    )
    gateway = SearchGateway(providers=[remote], local_provider=local)

    response = gateway.search(SearchRequest("latest database news"))

    assert requires_live_discovery("останні новини database") is True
    assert response.external_search_used is True
    assert response.attempts[-1].provider == "remote"


def test_live_query_ranks_new_discovery_ahead_of_cached_archive():
    local = FakeProvider(
        "local:web-index",
        [
            {
                "title": "PostgreSQL cached news archive",
                "href": "https://cache.test/postgresql-news",
                "body": "PostgreSQL news archive",
                "local_index": True,
                "local_fresh": True,
                "local_quality_score": 1.0,
            }
        ],
    )
    remote = FakeProvider(
        "remote",
        [
            {
                "title": "PostgreSQL News",
                "href": "https://postgresql.org/about/news/",
                "body": "Latest PostgreSQL project news today",
            }
        ],
    )
    gateway = SearchGateway(providers=[remote], local_provider=local)

    response = gateway.search(SearchRequest("latest PostgreSQL news today"))

    assert response.candidates[0]["uri"] == "https://postgresql.org/about/news"
    assert response.candidates[0]["live_discovery_bonus"] == 2.0
    assert response.candidates[1]["local_cache_bonus"] == 0.0


def test_persisted_corpus_can_close_search_without_remote_provider(tmp_path):
    from remy.core.local_web_index import LocalWebIndex

    index = LocalWebIndex(tmp_path / "web.sqlite3")
    evidence = [
        "Database consistency quorum replication transaction evidence. ",
        "Database consistency consensus leader election recovery evidence. ",
        "Database consistency serializable isolation locking evidence. ",
    ]
    for number in range(3):
        index.upsert(
            url=f"https://independent{number}.test/database",
            title=f"Database consistency reference {number}",
            content=evidence[number] * 12,
            quality_score=0.8,
        )
    remote = FakeProvider("remote", error=AssertionError("remote search should be skipped"))
    gateway = SearchGateway(
        providers=[remote],
        local_provider=LocalIndexSearchProvider(index, fresh_hours=24),
    )

    response = gateway.search(SearchRequest("database consistency"))

    assert response.external_search_used is False
    assert response.fresh_local_result_count == 3
    assert [attempt.provider for attempt in response.attempts] == ["local:web-index"]


def test_local_query_correction_is_exposed_and_can_avoid_remote_search(tmp_path):
    from remy.core.local_web_index import LocalWebIndex

    index = LocalWebIndex(tmp_path / "web.sqlite3")
    index.upsert(
        url="https://example.test/database",
        title="Database architecture",
        content="Database consistency and replication reference. " * 15,
        quality_score=0.8,
    )
    remote = FakeProvider("remote", error=AssertionError("remote search should be skipped"))
    gateway = SearchGateway(
        providers=[remote],
        local_provider=LocalIndexSearchProvider(index, fresh_hours=24),
    )

    response = gateway.search(SearchRequest("databse", max_results=1))

    assert response.external_search_used is False
    assert response.local_query_corrections == {"databse": "database"}
    assert response.diagnostics()["local_query_corrections"] == {
        "databse": "database"
    }


def test_local_query_correction_repairs_supplemental_web_query(tmp_path):
    from remy.core.local_web_index import LocalWebIndex

    seen_queries = []

    class RecordingProvider:
        name = "remote"

        def search(self, request):
            seen_queries.append(request.query)
            return []

    index = LocalWebIndex(tmp_path / "web.sqlite3")
    index.upsert(
        url="https://example.test/database",
        title="Database architecture",
        content="Database consistency and replication reference. " * 15,
    )
    gateway = SearchGateway(
        providers=[RecordingProvider()],
        local_provider=LocalIndexSearchProvider(index, fresh_hours=24),
    )

    response = gateway.search(SearchRequest("databse reliability", max_results=3))

    assert response.external_search_used is True
    assert seen_queries == ["database reliability"]
    assert response.diagnostics()["effective_query"] == "database reliability"
