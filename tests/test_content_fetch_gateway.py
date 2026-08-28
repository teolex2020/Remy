from __future__ import annotations

import json

import pytest

from remy.core.content_fetch_gateway import (
    ContentFetchGateway,
    ContentFetchRequest,
    ContentFetchResult,
    assess_content_quality,
)


def _article_html(*, repeat: int = 35) -> str:
    paragraph = (
        "This is a factual paragraph with enough sentence structure for reliable extraction. "
        "It describes the tested system, its behavior, and the observable result. "
    )
    return (
        "<html><head><title>Evidence page</title></head><body><article><p>"
        f"{paragraph * repeat}</p></article></body></html>"
    )


def test_quality_score_distinguishes_evidence_from_tiny_placeholder():
    weak_score, weak_flags = assess_content_quality("Enable JavaScript")
    good_score, good_flags = assess_content_quality("Evidence sentence. " * 180)

    assert weak_score < good_score
    assert "too_short" in weak_flags
    assert "too_short" not in good_flags


def test_static_html_is_extracted_without_starting_browser(monkeypatch):
    browser_calls = []

    def http_fetcher(_url: str, _max_bytes: int):
        return _article_html(), "https://example.test/article", 200, "text/html"

    def browser_fetcher(*_args):
        browser_calls.append(True)
        raise AssertionError("browser should not be needed")

    monkeypatch.setattr("remy.core.content_fetch_gateway._check_ssrf", lambda _url: None)
    gateway = ContentFetchGateway(
        http_fetcher=http_fetcher,
        browser_fetcher=browser_fetcher,
        cache_ttl_seconds=0,
    )

    result = gateway.fetch(ContentFetchRequest("https://example.test/article"))

    assert result.ok is True
    assert result.title == "Evidence page"
    assert result.extraction_method.startswith("http+")
    assert result.quality_score > 0.5
    assert [attempt.stage for attempt in result.attempts] == ["http"]
    assert browser_calls == []


def test_weak_static_page_uses_browser_fallback(monkeypatch):
    def http_fetcher(_url: str, _max_bytes: int):
        return (
            "<html><body>Enable JavaScript</body></html>",
            "https://example.test/app",
            200,
            "text/html",
        )

    def browser_fetcher(_url: str, _timeout_ms: int):
        return (
            "Rendered evidence sentence with useful details. " * 80,
            "Rendered application",
            "https://example.test/app",
        )

    monkeypatch.setattr("remy.core.content_fetch_gateway._check_ssrf", lambda _url: None)
    gateway = ContentFetchGateway(
        http_fetcher=http_fetcher,
        browser_fetcher=browser_fetcher,
        cache_ttl_seconds=0,
    )

    result = gateway.fetch(ContentFetchRequest("https://example.test/app"))

    assert result.ok is True
    assert result.title == "Rendered application"
    assert result.extraction_method == "chromium+visible_text"
    assert [attempt.stage for attempt in result.attempts] == ["http", "browser"]


def test_successful_fetch_is_served_from_short_lived_cache(monkeypatch):
    calls = []

    def http_fetcher(_url: str, _max_bytes: int):
        calls.append(True)
        return _article_html(), "https://example.test/cached", 200, "text/html"

    monkeypatch.setattr("remy.core.content_fetch_gateway._check_ssrf", lambda _url: None)
    gateway = ContentFetchGateway(http_fetcher=http_fetcher, cache_ttl_seconds=60)
    request = ContentFetchRequest("https://example.test/cached", allow_browser=False)

    first = gateway.fetch(request)
    second = gateway.fetch(request)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.attempts[0].stage == "cache"
    assert len(calls) == 1


def test_successful_fetch_is_persisted_in_local_web_index(monkeypatch):
    stored = []

    class FakeIndex:
        def get(self, _url, *, max_age_seconds):
            assert max_age_seconds == 600
            return None

        def upsert(self, **document):
            stored.append(document)
            return True

    monkeypatch.setattr("remy.core.content_fetch_gateway._check_ssrf", lambda _url: None)
    gateway = ContentFetchGateway(
        http_fetcher=lambda _url, _max_bytes: (
            _article_html(),
            "https://example.test/persisted",
            200,
            "text/html",
        ),
        cache_ttl_seconds=0,
        document_index=FakeIndex(),
        persistent_cache_max_age_seconds=600,
    )

    result = gateway.fetch(ContentFetchRequest("https://example.test/persisted"))

    assert result.ok is True
    assert stored[0]["url"] == "https://example.test/persisted"
    assert stored[0]["quality_score"] > 0.5
    assert "factual paragraph" in stored[0]["content"]


def test_fresh_local_document_skips_network_fetch(monkeypatch):
    network_calls = []

    class FakeIndex:
        def get(self, url, *, max_age_seconds):
            assert url == "https://example.test/local"
            assert max_age_seconds == 600
            return {
                "url": url,
                "title": "Local evidence",
                "content": "Locally cached factual evidence. " * 20,
                "quality_score": 0.84,
            }

    monkeypatch.setattr("remy.core.content_fetch_gateway._check_ssrf", lambda _url: None)
    gateway = ContentFetchGateway(
        http_fetcher=lambda *_args: network_calls.append(True),
        cache_ttl_seconds=0,
        document_index=FakeIndex(),
        persistent_cache_max_age_seconds=600,
    )

    result = gateway.fetch(ContentFetchRequest("https://example.test/local"))

    assert result.ok is True
    assert result.cache_hit is True
    assert result.extraction_method == "local_index"
    assert result.attempts[0].stage == "local_index"
    assert network_calls == []


def test_force_refresh_bypasses_local_document(monkeypatch):
    network_calls = []

    class FakeIndex:
        def get(self, *_args, **_kwargs):
            raise AssertionError("force refresh must bypass local lookup")

        def upsert(self, **_document):
            return True

    def fetcher(_url, _max_bytes):
        network_calls.append(True)
        return _article_html(), "https://example.test/live", 200, "text/html"

    monkeypatch.setattr("remy.core.content_fetch_gateway._check_ssrf", lambda _url: None)
    gateway = ContentFetchGateway(
        http_fetcher=fetcher,
        cache_ttl_seconds=0,
        document_index=FakeIndex(),
    )

    result = gateway.fetch(
        ContentFetchRequest("https://example.test/live", force_refresh=True)
    )

    assert result.ok is True
    assert result.extraction_method.startswith("http+")
    assert network_calls == [True]


def test_gateway_honors_content_limit_and_reports_original_size(monkeypatch):
    monkeypatch.setattr("remy.core.content_fetch_gateway._check_ssrf", lambda _url: None)
    gateway = ContentFetchGateway(
        http_fetcher=lambda _url, _max_bytes: (
            _article_html(repeat=80),
            "https://example.test/long",
            200,
            "text/html",
        ),
        cache_ttl_seconds=0,
    )

    result = gateway.fetch(
        ContentFetchRequest("https://example.test/long", max_chars=500, allow_browser=False)
    )
    payload = result.to_payload()

    assert len(result.content) == 500
    assert payload["truncated"] is True
    assert payload["total_chars"] > 500


def test_private_address_is_rejected_before_fetcher_runs():
    calls = []
    gateway = ContentFetchGateway(
        http_fetcher=lambda *_args: calls.append(True),
        cache_ttl_seconds=0,
    )

    result = gateway.fetch(ContentFetchRequest("http://127.0.0.1/admin"))

    assert result.ok is False
    assert "private" in result.error.lower() or "internal" in result.error.lower()
    assert calls == []


@pytest.mark.asyncio
async def test_pipeline_page_fetch_uses_shared_gateway(monkeypatch):
    from remy.core import pipeline_runner

    class FakeGateway:
        def fetch(self, request):
            assert request.url == "https://example.test/page"
            return ContentFetchResult(
                requested_url=request.url,
                final_url=request.url,
                content="shared gateway evidence",
                extraction_method="http+trafilatura",
                quality_score=0.8,
            )

    monkeypatch.setattr(
        "remy.core.content_fetch_gateway.get_content_fetch_gateway",
        lambda: FakeGateway(),
    )

    text = await pipeline_runner._fetch_page_text("https://example.test/page")

    assert text == "shared gateway evidence"


@pytest.mark.asyncio
async def test_pipeline_search_surfaces_fetch_method_and_quality(monkeypatch):
    from types import SimpleNamespace

    from remy.core import pipeline_runner

    class FakeSearchGateway:
        def search(self, _request):
            return SimpleNamespace(
                candidates=[
                    {
                        "title": "Result",
                        "uri": "https://example.test/page",
                        "snippet": "Candidate snippet",
                    }
                ],
                attempts=[SimpleNamespace(status="ok")],
            )

    class FakeContentGateway:
        def fetch(self, request):
            return ContentFetchResult(
                requested_url=request.url,
                final_url=request.url,
                content="Fetched pipeline evidence",
                extraction_method="http+trafilatura",
                quality_score=0.731,
            )

    monkeypatch.setattr(
        "remy.core.search_gateway.get_search_gateway",
        lambda: FakeSearchGateway(),
    )
    monkeypatch.setattr(
        "remy.core.content_fetch_gateway.get_content_fetch_gateway",
        lambda: FakeContentGateway(),
    )

    output = await pipeline_runner._run_web_search(
        {"query": "observable fetch", "num_results": 1, "fetch_content": True}
    )

    assert "Fetch method: http+trafilatura" in output
    assert "Content quality: 0.731" in output


def test_research_source_uses_shared_gateway(monkeypatch):
    from remy.core import research_supervisor

    class FakeGateway:
        def fetch(self, request):
            return ContentFetchResult(
                requested_url=request.url,
                final_url="https://example.test/final",
                content="Extracted factual evidence. " * 20,
                title="Resolved title",
                extraction_method="http+trafilatura",
                quality_score=0.8,
            )

    monkeypatch.setattr(
        "remy.core.content_fetch_gateway.get_content_fetch_gateway",
        lambda: FakeGateway(),
    )

    url, title, evidence = research_supervisor._fetch_source(
        {"uri": "https://example.test/start", "title": "Candidate"}
    )

    assert url == "https://example.test/final"
    assert title == "Resolved title"
    assert "Extracted page evidence" in evidence


def test_extract_content_tool_preserves_evidence_contract(monkeypatch):
    from remy.core import tool_dispatch
    from remy.core import tool_utils

    fetched = ContentFetchResult(
        requested_url="https://example.test/start",
        final_url="https://example.test/final",
        content="Grounded page evidence with a concrete factual sentence. " * 10,
        title="Expected resource",
        site="Example",
        extraction_method="http+trafilatura",
        quality_score=0.82,
    )

    class FakeGateway:
        def fetch(self, _request):
            return fetched

    monkeypatch.setattr(tool_utils, "_check_ssrf", lambda _url: None)
    monkeypatch.setattr(
        "remy.core.content_fetch_gateway.get_content_fetch_gateway",
        lambda: FakeGateway(),
    )

    raw = tool_dispatch._execute_tool_inner(
        "extract_content",
        {
            "url": "https://example.test/start",
            "expected_title": "Expected resource",
        },
        session_id="content-gateway-contract",
    )
    payload = json.loads(raw)

    assert payload["url"] == "https://example.test/final"
    assert payload["content_quality"]["score"] == 0.82
    assert payload["fetch_diagnostics"]["attempts"] == []
    assert "evidence_packet" in payload
