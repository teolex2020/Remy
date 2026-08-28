"""Shared web-page fetching and evidence extraction.

The gateway is local and API-key-free.  It reuses HTTP connections, bounds
response sizes and concurrency, validates every redirect against SSRF, and
falls back to one queued, reusable Chromium worker only when static HTML does
not contain enough readable evidence.
"""

from __future__ import annotations

import atexit
import copy
import logging
import queue
import re
import threading
import time
from concurrent.futures import Future
from dataclasses import asdict, dataclass, field
from typing import Any, Callable
from urllib.parse import urljoin

from remy.core.tool_utils import _check_ssrf
from remy.core.web_content import extract_visible_text


_USER_AGENT = "Mozilla/5.0 (compatible; Remy-Agent/0.9; +local-assistant)"
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FetchAttempt:
    stage: str
    status: str
    duration_ms: int
    status_code: int | None = None
    content_chars: int = 0
    error: str = ""


@dataclass(slots=True)
class ContentFetchRequest:
    url: str
    max_chars: int = 16_000
    include_links: bool = False
    include_tables: bool = True
    allow_browser: bool = True
    force_refresh: bool = False


@dataclass(slots=True)
class ContentFetchResult:
    requested_url: str
    final_url: str
    content: str = ""
    title: str = ""
    author: str = ""
    date: str = ""
    site: str = ""
    extraction_method: str = ""
    quality_score: float = 0.0
    quality_flags: list[str] = field(default_factory=list)
    attempts: list[FetchAttempt] = field(default_factory=list)
    duration_ms: int = 0
    total_chars: int = 0
    cache_hit: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.content.strip()) and not self.error

    def diagnostics(self) -> dict[str, Any]:
        return {
            "duration_ms": self.duration_ms,
            "cache_hit": self.cache_hit,
            "quality_score": self.quality_score,
            "quality_flags": list(self.quality_flags),
            "attempts": [asdict(attempt) for attempt in self.attempts],
        }

    def to_payload(self, *, max_chars: int | None = None) -> dict[str, Any]:
        limit = max_chars if max_chars is not None else len(self.content)
        content = self.content[: max(0, limit)]
        payload: dict[str, Any] = {
            "url": self.final_url or self.requested_url,
            "content": content,
            "extraction_method": self.extraction_method,
            "content_quality": {
                "score": self.quality_score,
                "flags": list(self.quality_flags),
            },
            "fetch_diagnostics": self.diagnostics(),
        }
        for key in ("title", "author", "date", "site"):
            value = getattr(self, key)
            if value:
                payload[key] = value
        total_chars = max(self.total_chars, len(self.content))
        if total_chars > len(content):
            payload["truncated"] = True
            payload["total_chars"] = total_chars
        if self.error:
            payload["error"] = self.error
        return payload


def assess_content_quality(text: str) -> tuple[float, list[str]]:
    """Score extracted evidence without an LLM or an external service."""
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return 0.0, ["empty"]

    flags: list[str] = []
    words = re.findall(r"\b\w+\b", clean, flags=re.UNICODE)
    alpha = sum(character.isalpha() for character in clean)
    alpha_ratio = alpha / max(len(clean), 1)
    sentence_marks = sum(clean.count(mark) for mark in (".", "!", "?", ":"))

    score = min(len(clean) / 2_000, 1.0) * 0.55
    score += min(len(words) / 300, 1.0) * 0.20
    score += min(sentence_marks / 12, 1.0) * 0.10
    score += min(alpha_ratio / 0.65, 1.0) * 0.15

    if len(clean) < 120:
        flags.append("too_short")
        score *= 0.35
    elif len(clean) < 500:
        flags.append("short")

    boilerplate_hits = sum(
        clean.casefold().count(phrase)
        for phrase in ("cookie", "privacy policy", "sign in", "subscribe", "accept all")
    )
    if boilerplate_hits >= 4:
        flags.append("boilerplate_heavy")
        score -= 0.20
    if alpha_ratio < 0.35:
        flags.append("low_text_density")
        score -= 0.15
    if len(set(word.casefold() for word in words)) < min(20, max(1, len(words) // 3)):
        flags.append("repetitive")
        score -= 0.10

    return round(max(0.0, min(score, 1.0)), 3), flags


class _QueuedBrowserReader:
    """Own Playwright on one thread and reuse a single Chromium process."""

    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[str, int, Future] | None] = queue.Queue(maxsize=16)
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()

    def _ensure_started(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        with self._start_lock:
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run,
                name="remy-content-browser",
                daemon=True,
            )
            self._thread.start()

    def fetch(self, url: str, timeout_ms: int = 30_000) -> tuple[str, str, str]:
        self._ensure_started()
        future: Future = Future()
        self._queue.put((url, timeout_ms, future), timeout=2)
        return future.result(timeout=(timeout_ms / 1_000) + 15)

    def _run(self) -> None:
        playwright = None
        browser = None
        try:
            while True:
                task = self._queue.get()
                if task is None:
                    break
                url, timeout_ms, future = task
                page = None
                try:
                    if browser is None:
                        from playwright.sync_api import sync_playwright

                        playwright = sync_playwright().start()
                        browser = playwright.chromium.launch(headless=True)
                    page = browser.new_page()
                    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                    try:
                        page.wait_for_load_state("networkidle", timeout=5_000)
                    except Exception:
                        pass
                    future.set_result(
                        (
                            page.locator("body").inner_text(timeout=10_000),
                            page.title(),
                            page.url,
                        )
                    )
                except Exception as exc:
                    future.set_exception(exc)
                finally:
                    if page is not None:
                        try:
                            page.close()
                        except Exception:
                            pass
        finally:
            if browser is not None:
                try:
                    browser.close()
                except Exception:
                    pass
            if playwright is not None:
                try:
                    playwright.stop()
                except Exception:
                    pass

    def close(self) -> None:
        if self._thread and self._thread.is_alive():
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass


HttpFetcher = Callable[[str, int], tuple[str, str, int, str]]
BrowserFetcher = Callable[[str, int], tuple[str, str, str]]


class ContentFetchGateway:
    def __init__(
        self,
        *,
        http_fetcher: HttpFetcher | None = None,
        browser_fetcher: BrowserFetcher | None = None,
        cache_ttl_seconds: int = 300,
        max_cache_entries: int = 64,
        max_concurrency: int = 6,
        document_index: Any | None = None,
        persistent_cache_max_age_seconds: int = 7 * 24 * 60 * 60,
    ) -> None:
        self._http_fetcher = http_fetcher
        self._browser_fetcher = browser_fetcher
        self._cache_ttl = max(0, cache_ttl_seconds)
        self._max_cache_entries = max(1, max_cache_entries)
        self._cache: dict[tuple[Any, ...], tuple[float, ContentFetchResult]] = {}
        self._cache_lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(max(1, max_concurrency))
        self._client = None
        self._client_lock = threading.Lock()
        self._document_index = document_index
        self._persistent_cache_max_age = max(0, persistent_cache_max_age_seconds)

    def _client_instance(self):
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                import httpx

                self._client = httpx.Client(
                    timeout=20,
                    follow_redirects=False,
                    headers={
                        "User-Agent": _USER_AGENT,
                        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.4",
                        "Accept-Language": "uk,en;q=0.8",
                    },
                    limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
                )
        return self._client

    def _fetch_http(self, url: str, max_bytes: int) -> tuple[str, str, int, str]:
        client = self._client_instance()
        current = url
        for _redirect in range(6):
            ssrf_error = _check_ssrf(current)
            if ssrf_error:
                raise ValueError(ssrf_error)
            with client.stream("GET", current) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("Redirect response has no Location header")
                    current = urljoin(current, location)
                    continue
                response.raise_for_status()
                content_type = str(response.headers.get("content-type") or "")
                if "html" not in content_type.casefold() and "text/" not in content_type.casefold():
                    raise ValueError(
                        f"Expected an HTML/text response, got {content_type or 'unknown type'}"
                    )
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"Response exceeds {max_bytes} byte safety limit")
                    chunks.append(chunk)
                encoding = response.encoding or "utf-8"
                return (
                    b"".join(chunks).decode(encoding, errors="replace"),
                    current,
                    response.status_code,
                    content_type,
                )
        raise ValueError("Too many redirects")

    @staticmethod
    def _extract_html(
        html: str, *, include_links: bool, include_tables: bool
    ) -> tuple[str, dict[str, str], str]:
        import trafilatura

        article_text = trafilatura.extract(
            html,
            include_comments=False,
            include_links=include_links,
            include_tables=include_tables,
            favor_recall=True,
            no_fallback=False,
        ) or ""
        visible_text, fallback_title = extract_visible_text(html)
        article_score, _ = assess_content_quality(article_text)
        visible_score, _ = assess_content_quality(visible_text)
        if visible_score > article_score + 0.08:
            text = visible_text
            method = "visible_html"
        else:
            text = article_text or visible_text
            method = "trafilatura" if article_text else "visible_html"

        metadata: dict[str, str] = {}
        try:
            extracted = trafilatura.extract_metadata(html)
            if extracted:
                for source_key, target_key in (
                    ("title", "title"),
                    ("author", "author"),
                    ("date", "date"),
                    ("sitename", "site"),
                ):
                    value = getattr(extracted, source_key, None)
                    if value:
                        metadata[target_key] = str(value)
        except Exception:
            pass
        if fallback_title and not metadata.get("title"):
            metadata["title"] = fallback_title
        return text, metadata, method

    def _cache_key(self, request: ContentFetchRequest) -> tuple[Any, ...]:
        return (
            request.url.strip(),
            request.max_chars,
            request.include_links,
            request.include_tables,
            request.allow_browser,
            request.force_refresh,
        )

    def _get_cached(self, key: tuple[Any, ...]) -> ContentFetchResult | None:
        if not self._cache_ttl:
            return None
        with self._cache_lock:
            item = self._cache.get(key)
            if not item:
                return None
            stored_at, result = item
            if time.monotonic() - stored_at > self._cache_ttl:
                self._cache.pop(key, None)
                return None
            cached = copy.deepcopy(result)
            cached.cache_hit = True
            cached.attempts = [FetchAttempt("cache", "hit", 0)]
            cached.duration_ms = 0
            return cached

    def _store_cached(self, key: tuple[Any, ...], result: ContentFetchResult) -> None:
        if not self._cache_ttl or not result.ok:
            return
        with self._cache_lock:
            if len(self._cache) >= self._max_cache_entries:
                oldest = min(self._cache, key=lambda item: self._cache[item][0])
                self._cache.pop(oldest, None)
            self._cache[key] = (time.monotonic(), copy.deepcopy(result))

    def fetch(self, request: ContentFetchRequest) -> ContentFetchResult:
        started = time.perf_counter()
        url = request.url.strip()
        result = ContentFetchResult(requested_url=url, final_url=url)
        ssrf_error = _check_ssrf(url)
        if ssrf_error:
            result.error = ssrf_error
            return result

        key = self._cache_key(request)
        cached = self._get_cached(key)
        if cached is not None:
            return cached

        if (
            not request.force_refresh
            and self._document_index is not None
            and self._persistent_cache_max_age > 0
        ):
            try:
                document = self._document_index.get(
                    url,
                    max_age_seconds=self._persistent_cache_max_age,
                )
            except Exception as exc:
                logger.debug("Local web cache lookup failed for %s: %s", url, exc)
                document = None
            if document:
                content_limit = max(80, min(int(request.max_chars), 100_000))
                stored_content = str(document.get("content") or "")
                content = stored_content[:content_limit]
                result = ContentFetchResult(
                    requested_url=url,
                    final_url=str(document.get("url") or url),
                    content=content,
                    title=str(document.get("title") or ""),
                    author=str(document.get("author") or ""),
                    date=str(document.get("published_date") or ""),
                    site=str(document.get("site") or ""),
                    extraction_method="local_index",
                    quality_score=float(document.get("quality_score") or 0.0),
                    quality_flags=["persistent_local_cache"],
                    attempts=[
                        FetchAttempt(
                            stage="local_index",
                            status="hit",
                            duration_ms=round((time.perf_counter() - started) * 1_000),
                            content_chars=len(content),
                        )
                    ],
                    duration_ms=round((time.perf_counter() - started) * 1_000),
                    total_chars=len(stored_content),
                    cache_hit=True,
                )
                return result

        acquired = self._slots.acquire(timeout=30)
        if not acquired:
            result.error = "Content fetch queue is busy; retry later"
            return result
        try:
            fetcher = self._http_fetcher or self._fetch_http
            http_started = time.perf_counter()
            try:
                html, final_url, status_code, _content_type = fetcher(url, 4 * 1024 * 1024)
                result.final_url = final_url
                text, metadata, method = self._extract_html(
                    html,
                    include_links=request.include_links,
                    include_tables=request.include_tables,
                )
                result.content = text
                result.extraction_method = f"http+{method}"
                for metadata_key, value in metadata.items():
                    setattr(result, metadata_key, value)
                result.quality_score, result.quality_flags = assess_content_quality(text)
                result.attempts.append(
                    FetchAttempt(
                        stage="http",
                        status="ok",
                        duration_ms=round((time.perf_counter() - http_started) * 1_000),
                        status_code=status_code,
                        content_chars=len(text),
                    )
                )
            except Exception as exc:
                result.attempts.append(
                    FetchAttempt(
                        stage="http",
                        status="error",
                        duration_ms=round((time.perf_counter() - http_started) * 1_000),
                        error=f"{type(exc).__name__}: {exc}"[:500],
                    )
                )

            needs_browser = len(result.content.strip()) < 120 or result.quality_score < 0.22
            if request.allow_browser and needs_browser:
                browser_started = time.perf_counter()
                try:
                    browser_fetcher = self._browser_fetcher or _browser_reader.fetch
                    browser_text, browser_title, browser_url = browser_fetcher(url, 30_000)
                    redirect_error = _check_ssrf(browser_url or url)
                    if redirect_error:
                        raise ValueError(f"Blocked browser redirect: {redirect_error}")
                    browser_score, browser_flags = assess_content_quality(browser_text)
                    if browser_score >= result.quality_score:
                        result.content = browser_text
                        result.title = browser_title or result.title
                        result.final_url = browser_url or result.final_url
                        result.quality_score = browser_score
                        result.quality_flags = browser_flags
                        result.extraction_method = "chromium+visible_text"
                    result.attempts.append(
                        FetchAttempt(
                            stage="browser",
                            status="ok",
                            duration_ms=round((time.perf_counter() - browser_started) * 1_000),
                            content_chars=len(browser_text),
                        )
                    )
                except Exception as exc:
                    result.attempts.append(
                        FetchAttempt(
                            stage="browser",
                            status="error",
                            duration_ms=round((time.perf_counter() - browser_started) * 1_000),
                            error=f"{type(exc).__name__}: {exc}"[:500],
                        )
                    )

            if len(result.content.strip()) < 80:
                errors = [attempt.error for attempt in result.attempts if attempt.error]
                result.error = errors[-1] if errors else "Page returned too little readable text"
            result.duration_ms = round((time.perf_counter() - started) * 1_000)
            if result.ok:
                full_content = result.content
                result.total_chars = len(full_content)
                if self._document_index is not None:
                    try:
                        self._document_index.upsert(
                            url=result.final_url or url,
                            title=result.title,
                            content=full_content,
                            site=result.site,
                            author=result.author,
                            published_date=result.date,
                            quality_score=result.quality_score,
                            extraction_method=result.extraction_method,
                        )
                    except Exception as exc:
                        logger.debug("Local web indexing failed for %s: %s", url, exc)
                content_limit = max(80, min(int(request.max_chars), 100_000))
                if len(result.content) > content_limit:
                    result.content = result.content[:content_limit]
                    result.quality_flags.append("gateway_truncated")
                self._store_cached(key, result)
            return result
        finally:
            self._slots.release()

    def close(self) -> None:
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None


_browser_reader = _QueuedBrowserReader()
_default_gateway: ContentFetchGateway | None = None
_default_gateway_lock = threading.Lock()


def get_content_fetch_gateway() -> ContentFetchGateway:
    global _default_gateway
    if _default_gateway is None:
        with _default_gateway_lock:
            if _default_gateway is None:
                from remy.config.settings import settings

                document_index = None
                if settings.LOCAL_WEB_INDEX_ENABLED:
                    from remy.core.local_web_index import get_local_web_index

                    document_index = get_local_web_index()
                _default_gateway = ContentFetchGateway(
                    document_index=document_index,
                    persistent_cache_max_age_seconds=(
                        settings.LOCAL_WEB_INDEX_FRESH_HOURS * 60 * 60
                    ),
                )
    return _default_gateway


def _close_default_resources() -> None:
    if _default_gateway is not None:
        _default_gateway.close()
    _browser_reader.close()


atexit.register(_close_default_resources)
