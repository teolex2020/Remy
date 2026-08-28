"""Shared, API-key-free search discovery for Remy.

The gateway keeps search-engine quirks out of chat, pipelines, and research
workers.  It normalizes every provider response, records partial failures,
deduplicates URLs, and applies Remy's deterministic source-quality rules.

The local FTS5 index is queried first. ``ddgs`` supplies missing or live
discovery results and requires neither an account nor an API key. The provider
interface stays small so local SearXNG can be added without changing callers.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from remy.core.retrieval.source_filter import (
    annotate,
    enforce_site_constraint,
    extract_site_constraint,
)
from remy.core.search_relevance import assess_query_relevance


DEFAULT_DDGS_BACKENDS = ("duckduckgo", "brave", "google", "mojeek", "yahoo")
_LIVE_DISCOVERY_RE = re.compile(
    r"\b(?:latest|today|current|breaking|news|now|recent|"
    r"сьогодні|зараз|останні|остання|останній|новини|актуальн\w*|"
    r"сегодня|сейчас|последн\w*|новости|актуальн\w*)\b",
    re.IGNORECASE,
)
_TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "dclid",
    "msclkid",
    "mc_cid",
    "mc_eid",
    "ref_src",
}


@dataclass(frozen=True, slots=True)
class SearchRequest:
    query: str
    max_results: int = 10
    drop_classes: tuple[str, ...] = ("seo",)


@dataclass(slots=True)
class ProviderAttempt:
    provider: str
    status: str
    duration_ms: int
    result_count: int = 0
    error: str = ""


@dataclass(slots=True)
class SearchResponse:
    query: str
    candidates: list[dict[str, Any]]
    attempts: list[ProviderAttempt]
    duration_ms: int
    site_constraint: str | None = None
    local_result_count: int = 0
    fresh_local_result_count: int = 0
    local_duplicate_suppressed: int = 0
    local_query_corrections: dict[str, str] | None = None
    effective_query: str = ""
    external_search_used: bool = True

    @property
    def had_provider_error(self) -> bool:
        return any(attempt.status == "error" for attempt in self.attempts)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "duration_ms": self.duration_ms,
            "site_constraint": self.site_constraint,
            "local_result_count": self.local_result_count,
            "fresh_local_result_count": self.fresh_local_result_count,
            "local_duplicate_suppressed": self.local_duplicate_suppressed,
            "local_query_corrections": dict(self.local_query_corrections or {}),
            "effective_query": self.effective_query or self.query,
            "external_search_used": self.external_search_used,
            "attempts": [asdict(attempt) for attempt in self.attempts],
        }


class SearchProvider(Protocol):
    name: str

    def search(self, request: SearchRequest) -> Sequence[dict[str, Any]]: ...


@dataclass(slots=True)
class DDGSSearchProvider:
    """Thin adapter around ddgs with a bounded network timeout."""

    backend: str
    name: str
    timeout_seconds: int = 15

    def search(self, request: SearchRequest) -> Sequence[dict[str, Any]]:
        from ddgs import DDGS

        try:
            client = DDGS(timeout=self.timeout_seconds)
        except TypeError:
            # Compatibility with older ddgs releases and lightweight test doubles.
            client = DDGS()

        kwargs: dict[str, Any] = {"max_results": request.max_results}
        if self.backend:
            kwargs["backend"] = self.backend
        try:
            return list(client.text(request.query, **kwargs) or [])
        except TypeError as exc:
            # Some older ddgs versions did not expose the backend keyword.
            message = str(exc).lower()
            if "backend" not in message and "unexpected keyword" not in message:
                raise
            kwargs.pop("backend", None)
            return list(client.text(request.query, **kwargs) or [])


@dataclass(slots=True)
class LocalIndexSearchProvider:
    """Search Remy's own previously fetched corpus before internet discovery."""

    index: Any
    fresh_hours: int = 168
    name: str = "local:web-index"

    def search(self, request: SearchRequest) -> Sequence[dict[str, Any]]:
        documents = self.index.search(
            request.query,
            limit=max(3, request.max_results),
            max_age_seconds=90 * 24 * 60 * 60,
        )
        fresh_seconds = max(1, int(self.fresh_hours)) * 60 * 60
        return [
            {
                "title": document.get("title") or document.get("url") or "Local evidence",
                "url": document.get("url") or "",
                "body": document.get("snippet") or str(document.get("content") or "")[:500],
                "local_index": True,
                "local_fresh": int(document.get("age_seconds") or 0) <= fresh_seconds,
                "local_age_seconds": int(document.get("age_seconds") or 0),
                "local_quality_score": float(document.get("quality_score") or 0.0),
                "local_fetched_at": str(document.get("fetched_at_iso") or ""),
                "local_duplicate_count": int(document.get("duplicate_count") or 0),
                "local_query_corrections": dict(
                    document.get("query_corrections") or {}
                ),
            }
            for document in documents
            if document.get("url")
        ]


def requires_live_discovery(query: str) -> bool:
    """Return True when cached discovery must not replace a live web search."""
    return bool(_LIVE_DISCOVERY_RE.search(query or ""))


def _apply_query_corrections(query: str, corrections: dict[str, str]) -> str:
    corrected = query
    for original, replacement in corrections.items():
        corrected = re.sub(
            rf"\b{re.escape(original)}\b",
            replacement,
            corrected,
            flags=re.IGNORECASE,
        )
    return corrected


def canonicalize_url(value: str) -> str:
    """Return a stable HTTP(S) URL and remove click-tracking parameters."""
    try:
        split = urlsplit((value or "").strip())
    except ValueError:
        return ""
    scheme = split.scheme.lower()
    if scheme not in {"http", "https"} or not split.hostname:
        return ""

    host = split.hostname.lower()
    try:
        port = split.port
    except ValueError:
        return ""
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"

    path = split.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(split.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_QUERY_KEYS
        ),
        doseq=True,
    )
    return urlunsplit((scheme, host, path, query, ""))


def _normalize_candidate(
    raw: dict[str, Any], provider: str, position: int
) -> dict[str, Any] | None:
    uri = canonicalize_url(str(raw.get("uri") or raw.get("href") or raw.get("url") or ""))
    if not uri:
        return None
    candidate = {
        "title": str(raw.get("title") or "").strip(),
        "uri": uri,
        "snippet": str(
            raw.get("snippet") or raw.get("body") or raw.get("description") or ""
        ).strip(),
        "providers": [provider],
        "provider_count": 1,
        "provider_positions": {provider: position},
        "raw_rank": position,
    }
    for key in (
        "local_index",
        "local_fresh",
        "local_age_seconds",
        "local_quality_score",
        "local_fetched_at",
        "local_duplicate_count",
        "local_query_corrections",
    ):
        if key in raw:
            candidate[key] = raw[key]
    return candidate


def _fuse_candidates(
    provider_results: Sequence[tuple[str, Sequence[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    for provider, raw_results in provider_results:
        for position, raw in enumerate(raw_results, start=1):
            if not isinstance(raw, dict):
                continue
            candidate = _normalize_candidate(raw, provider, position)
            if candidate is None:
                continue
            uri = candidate["uri"]
            existing = fused.get(uri)
            if existing is None:
                fused[uri] = candidate
                continue
            if provider not in existing["providers"]:
                existing["providers"].append(provider)
            existing["provider_count"] = len(existing["providers"])
            existing["provider_positions"][provider] = position
            existing["raw_rank"] = min(existing["raw_rank"], position)
            if len(candidate["title"]) > len(existing["title"]):
                existing["title"] = candidate["title"]
            if len(candidate["snippet"]) > len(existing["snippet"]):
                existing["snippet"] = candidate["snippet"]
            for key in (
                "local_index",
                "local_fresh",
                "local_age_seconds",
                "local_quality_score",
                "local_fetched_at",
                "local_duplicate_count",
                "local_query_corrections",
            ):
                if key in candidate:
                    existing[key] = candidate[key]
    return list(fused.values())


class SearchGateway:
    """Run discovery with graceful fallback and deterministic result fusion."""

    def __init__(
        self,
        providers: Sequence[SearchProvider] | None = None,
        recovery_providers: Sequence[SearchProvider] | None = None,
        local_provider: SearchProvider | None = None,
        reranker: Any | None = None,
        enable_local_reranker: bool | None = None,
    ) -> None:
        from remy.config.settings import settings

        if providers is None:
            joined = ",".join(DEFAULT_DDGS_BACKENDS)
            providers = [DDGSSearchProvider(joined, "ddgs:metasearch")]
            if local_provider is None and settings.LOCAL_WEB_INDEX_ENABLED:
                from remy.core.local_web_index import get_local_web_index

                local_provider = LocalIndexSearchProvider(
                    get_local_web_index(),
                    fresh_hours=settings.LOCAL_WEB_INDEX_FRESH_HOURS,
                )
            if recovery_providers is None:
                recovery_providers = [
                    DDGSSearchProvider(backend, f"ddgs:{backend}")
                    for backend in DEFAULT_DDGS_BACKENDS
                ]
        self.providers = list(providers)
        self.recovery_providers = list(recovery_providers or [])
        self.local_provider = local_provider
        if enable_local_reranker is None:
            enable_local_reranker = settings.LOCAL_SEARCH_RERANKER_ENABLED
        if enable_local_reranker:
            if reranker is None:
                from remy.core.local_search_reranker import (
                    get_local_multilingual_reranker,
                )

                reranker = get_local_multilingual_reranker()
            self.reranker = reranker
        else:
            self.reranker = None

    @staticmethod
    def _attempt(
        provider: SearchProvider, request: SearchRequest
    ) -> tuple[ProviderAttempt, Sequence[dict[str, Any]]]:
        started = time.perf_counter()
        try:
            results = list(provider.search(request) or [])
            attempt = ProviderAttempt(
                provider=provider.name,
                status="ok" if results else "empty",
                duration_ms=round((time.perf_counter() - started) * 1000),
                result_count=len(results),
            )
            return attempt, results
        except Exception as exc:
            return (
                ProviderAttempt(
                    provider=provider.name,
                    status="error",
                    duration_ms=round((time.perf_counter() - started) * 1000),
                    error=f"{type(exc).__name__}: {exc}"[:500],
                ),
                [],
            )

    def search(self, request: SearchRequest) -> SearchResponse:
        started = time.perf_counter()
        query = request.query.strip()
        limit = max(1, min(int(request.max_results), 50))
        bounded = SearchRequest(query=query, max_results=limit, drop_classes=request.drop_classes)
        attempts: list[ProviderAttempt] = []
        provider_results: list[tuple[str, Sequence[dict[str, Any]]]] = []
        local_result_count = 0
        fresh_local_result_count = 0
        local_duplicate_suppressed = 0
        local_query_corrections: dict[str, str] = {}
        site_constraint = extract_site_constraint(query)

        if self.local_provider is not None:
            local_attempt, local_results = self._attempt(self.local_provider, bounded)
            attempts.append(local_attempt)
            if local_results:
                provider_results.append((self.local_provider.name, local_results))
                normalized_local = enforce_site_constraint(
                    _fuse_candidates([(self.local_provider.name, local_results)]),
                    site_constraint,
                )
                normalized_local = annotate(normalized_local)
                normalized_local = [
                    candidate
                    for candidate in normalized_local
                    if candidate.get("source_class") not in set(request.drop_classes)
                ]
                local_result_count = len(normalized_local)
                local_duplicate_suppressed = sum(
                    int(candidate.get("local_duplicate_count") or 0)
                    for candidate in normalized_local
                )
                for candidate in normalized_local:
                    local_query_corrections.update(
                        candidate.get("local_query_corrections") or {}
                    )
                relevance_query = " ".join(
                    [query, *local_query_corrections.values()]
                ).strip()
                fresh_local_result_count = sum(
                    1
                    for candidate in normalized_local
                    if candidate.get("local_fresh")
                    and assess_query_relevance(
                        relevance_query,
                        title=str(candidate.get("title") or ""),
                        snippet=str(candidate.get("snippet") or ""),
                        url=str(candidate.get("uri") or ""),
                    ).get("relevant")
                )

        local_target = min(3, limit)
        effective_query = _apply_query_corrections(query, local_query_corrections)
        remote_bounded = SearchRequest(
            query=effective_query,
            max_results=limit,
            drop_classes=request.drop_classes,
        )
        live_discovery_required = requires_live_discovery(query)
        external_search_used = (
            live_discovery_required
            or fresh_local_result_count < local_target
        )
        remote_provider_results: list[tuple[str, Sequence[dict[str, Any]]]] = []

        if external_search_used:
            for provider in self.providers:
                attempt, results = self._attempt(provider, remote_bounded)
                attempts.append(attempt)
                if results:
                    remote_provider_results.append((provider.name, results))
                    provider_results.append((provider.name, results))

        # The metasearch provider is normally enough.  If it fully fails or
        # returns nothing, query its engines independently so one bad backend
        # cannot erase every result.  These attempts are parallel and isolated.
        if external_search_used and not remote_provider_results and self.recovery_providers:
            ordered: dict[str, tuple[ProviderAttempt, Sequence[dict[str, Any]]]] = {}
            workers = min(3, len(self.recovery_providers))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="remy-search") as pool:
                future_to_provider = {
                    pool.submit(self._attempt, provider, remote_bounded): provider
                    for provider in self.recovery_providers
                }
                for future in as_completed(future_to_provider):
                    provider = future_to_provider[future]
                    ordered[provider.name] = future.result()
            for provider in self.recovery_providers:
                attempt, results = ordered[provider.name]
                attempts.append(attempt)
                if results:
                    provider_results.append((provider.name, results))

        candidates = _fuse_candidates(provider_results)
        candidates = enforce_site_constraint(candidates, site_constraint)
        candidates = annotate(candidates)
        drop_classes = set(request.drop_classes)
        candidates = [c for c in candidates if c.get("source_class") not in drop_classes]
        relevance_query = " ".join(
            [query, *local_query_corrections.values()]
        ).strip()
        rerank_scores = (
            self.reranker.score_candidates(relevance_query, candidates)
            if self.reranker is not None
            else [None] * len(candidates)
        )
        for candidate, local_rerank in zip(candidates, rerank_scores):
            agreement_bonus = min(max(candidate["provider_count"] - 1, 0), 2)
            query_relevance = assess_query_relevance(
                relevance_query,
                title=str(candidate.get("title") or ""),
                snippet=str(candidate.get("snippet") or ""),
                url=str(candidate.get("uri") or ""),
            )
            candidate["query_relevance"] = query_relevance
            if local_rerank is not None:
                candidate["local_rerank"] = local_rerank
            candidate["agreement_bonus"] = agreement_bonus
            candidate["force_refresh_recommended"] = live_discovery_required
            local_bonus = (
                0.35 + float(candidate.get("local_quality_score") or 0.0) * 0.5
                if candidate.get("local_fresh") and not live_discovery_required
                else 0.0
            )
            # A cache entry can be fresh by TTL while still being the wrong
            # evidence for "today/latest/current". Once a live provider has
            # answered such a query, prefer its discovery candidates over the
            # local archive. Content verification later in the pipeline still
            # decides whether the page actually supports the final claim.
            live_discovery_bonus = (
                2.0
                if live_discovery_required and not candidate.get("local_index")
                else 0.0
            )
            candidate["local_cache_bonus"] = round(local_bonus, 3)
            candidate["live_discovery_bonus"] = round(live_discovery_bonus, 3)
            candidate["retrieval_score"] = round(
                (candidate.get("source_score") or 0)
                + agreement_bonus
                + query_relevance["score"]
                + (float((local_rerank or {}).get("score") or 0.0) * 2.5)
                + local_bonus
                + live_discovery_bonus,
                3,
            )
        candidates.sort(
            key=lambda candidate: (
                -candidate["retrieval_score"],
                -candidate["provider_count"],
                candidate["raw_rank"],
                candidate["uri"],
            )
        )

        return SearchResponse(
            query=query,
            candidates=candidates[:limit],
            attempts=attempts,
            duration_ms=round((time.perf_counter() - started) * 1000),
            site_constraint=site_constraint,
            local_result_count=local_result_count,
            fresh_local_result_count=fresh_local_result_count,
            local_duplicate_suppressed=local_duplicate_suppressed,
            local_query_corrections=local_query_corrections,
            effective_query=effective_query,
            external_search_used=external_search_used,
        )


_default_gateway: SearchGateway | None = None


def get_search_gateway() -> SearchGateway:
    global _default_gateway
    if _default_gateway is None:
        _default_gateway = SearchGateway()
    return _default_gateway
