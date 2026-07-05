"""
Pipeline runner — executes pipeline steps sequentially, streaming progress events.

Each step receives the previous step's output (or original input for step 1).
Variables like {{input}}, {{s1.output}}, {{s2.output}} are substituted before execution.
"""

from __future__ import annotations

import asyncio
import ast
import json
from html import unescape
from html.parser import HTMLParser
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator
from urllib.parse import urlparse

from remy.core.file_utils import atomic_write

logger = logging.getLogger("PipelineRunner")
STOP_TOKEN = "__REMY_WORKFLOW_STOP__"
ERROR_PREFIXES = (
    "[Unknown block type:",
    "[Search error:",
    "[Memory search error:",
    "[Save error:",
    "[HTTP error:",
    "[Scrape error:",
    "[JSON parse error:",
    "[Regex error:",
    "[File read error:",
    "[File write error:",
)

_SEARCH_STOP_WORDS = {
    "the", "and", "for", "with", "про", "для", "що", "как", "или", "або", "та", "і",
}
_SEARCH_NOISE_DOMAINS = (
    "music.apple.com",
    "open.spotify.com",
    "soundcloud.com",
    "genius.com",
    "lyrics",
    "reverso.net",
    "context.reverso.net",
    "translate.google.",
    "deepl.com",
    "linguee.",
)
_CURRENCY_HINTS = {
    "курс", "валют", "долар", "доллар", "usd", "eur", "євро", "евро", "uah", "грн",
    "exchange", "currency", "cash", "bank", "nbu", "finance", "minfin",
}

# ── Variable substitution ─────────────────────────────────────────────────────

def _substitute(template: str, ctx: dict[str, Any]) -> str:
    """Replace {{var}} and {{stepId.output}} with values from context."""
    def _replace(m: re.Match) -> str:
        key = m.group(1).strip()
        val = ctx.get(key, m.group(0))
        return str(val) if val is not None else m.group(0)
    return re.sub(r"\{\{([^}]+)\}\}", _replace, template)


def _resolve_config(config: dict, ctx: dict[str, Any]) -> dict:
    resolved = {}
    for k, v in config.items():
        if isinstance(v, str):
            resolved[k] = _substitute(v, ctx)
        else:
            resolved[k] = v
    return resolved


def _resolve_local_secret(secret_key: str) -> str:
    key = (secret_key or "").strip()
    if not key:
        return ""
    secret_map = {
        "gemini_api_key": "GEMINI_API_KEY",
        "openrouter_api_key": "OPENROUTER_API_KEY",
        "telegram_bot_token": "TELEGRAM_BOT_TOKEN",
        "smtp_password": "SMTP_PASSWORD",
    }
    setting_name = secret_map.get(key)
    if not setting_name:
        return ""
    from remy.config.settings import settings

    value = getattr(settings, setting_name, None)
    return str(value or "")


# ── Step executors ────────────────────────────────────────────────────────────

async def _run_llm_call(config: dict) -> str:
    prompt = config.get("prompt", "")
    model = config.get("model", "") or None

    from remy.core.llm import get_llm
    from remy.config.settings import settings

    llm = get_llm(model or settings.SUMMARY_MODEL)
    result = await asyncio.to_thread(llm.invoke, prompt)
    content = result.content if hasattr(result, "content") else str(result)
    if isinstance(content, list):
        content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content or ""


async def _fetch_page_text(url: str, max_chars: int = 3000) -> str:
    """Fetch a URL and extract readable text via trafilatura."""
    try:
        import httpx
        import trafilatura
        async with httpx.AsyncClient(timeout=10, follow_redirects=True,
                                     headers={"User-Agent": "Mozilla/5.0"}) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        text = await asyncio.to_thread(
            trafilatura.extract, resp.text,
            include_comments=False, include_tables=True, no_fallback=False,
        )
        if text:
            return text[:max_chars]
    except Exception as exc:
        logger.debug("Page fetch failed for %s: %s", url, exc)
    return ""


class _ReadablePageParser(HTMLParser):
    """Small dependency-free HTML reader for workflow page scraping."""

    _SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas"}
    _BLOCK_TAGS = {"p", "div", "section", "article", "li", "tr", "br", "h1", "h2", "h3", "h4"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._stack: list[str] = []
        self._skip_depth = 0
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        self._stack.append(tag)
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        if tag == "a":
            href = dict(attrs).get("href")
            if href and href not in self.links:
                self.links.append(href)
        if tag in self._BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        for index in range(len(self._stack) - 1, -1, -1):
            if self._stack[index] == tag:
                del self._stack[index:]
                break
        if tag in self._BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = unescape((data or "").strip())
        if not text:
            return
        if "title" in self._stack:
            self.title_parts.append(text)
        else:
            self.text_parts.append(text)

    @property
    def title(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.title_parts)).strip()

    @property
    def text(self) -> str:
        return _normalize_scraped_text("".join(self.text_parts))


def _normalize_scraped_text(raw: str) -> str:
    lines: list[str] = []
    blank_pending = False
    for raw_line in (raw or "").splitlines():
        line = re.sub(r"[ \t\r\f\v]+", " ", raw_line).strip()
        if not line:
            if lines:
                blank_pending = True
            continue
        if blank_pending and lines and lines[-1] != "":
            lines.append("")
        lines.append(line)
        blank_pending = False
    return "\n".join(lines).strip()


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(minimum, min(parsed, maximum))


def _friendly_http_error(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status:
        if status in {401, 403}:
            return (
                f"[HTTP error: Request returned HTTP {status}. Check the selected Authorization secret, "
                "auth scheme, and endpoint permissions. Try Test Connection before running the workflow.]"
            )
        if status == 404:
            return (
                "[HTTP error: Endpoint was not found (HTTP 404). Check the URL path and method. "
                "Try Test Connection before running the workflow.]"
            )
        if 400 <= int(status) < 500:
            return (
                f"[HTTP error: Request was rejected by the server (HTTP {status}). Check URL, method, "
                "headers, and request body.]"
            )
        if int(status) >= 500:
            return (
                f"[HTTP error: The server returned HTTP {status}. The endpoint may be temporarily down; "
                "retry later or enable block retry.]"
            )
    detail = str(exc).strip()
    if "timed out" in detail.lower() or "timeout" in detail.lower():
        return "[HTTP error: Request timed out. Check the URL or reduce the endpoint work before retrying.]"
    return (
        "[HTTP error: Request could not be completed. Check the URL, network connection, and HTTP method. "
        f"Detail: {detail or exc.__class__.__name__}]"
    )


def _friendly_scrape_error(exc: Exception | None = None, *, reason: str = "") -> str:
    detail = (reason or (str(exc).strip() if exc else "")).strip()
    low = detail.lower()
    if "timed out" in low or "timeout" in low:
        return "[Scrape error: Page loading timed out. Try Test Scrape, then reduce Max characters or retry later.]"
    response = getattr(exc, "response", None) if exc else None
    status = getattr(response, "status_code", None)
    if status in {401, 403}:
        return "[Scrape error: Page blocked access. This page may require login or deny automated reading.]"
    if status == 404:
        return "[Scrape error: Page was not found. Check the URL and try Test Scrape again.]"
    if status and int(status) >= 500:
        return "[Scrape error: Website returned a server error. Retry later or enable block retry.]"
    if "no readable" in low:
        return (
            "[Scrape error: Page could not be read as clean text. Try Extract: title or links, "
            "or use HTTP Request if this is an API/JSON page.]"
        )
    if "valid http" in low:
        return "[Scrape error: Enter a valid http(s) page URL before running the scraper.]"
    return (
        "[Scrape error: Page could not be read. Check the URL, try Test Scrape, or reduce Max characters. "
        f"Detail: {detail or (exc.__class__.__name__ if exc else 'unknown error')}]"
    )


async def _run_page_scrape(config: dict) -> str:
    import httpx

    url = str(config.get("url", "") or "").strip()
    mode = str(config.get("mode", "text") or "text").strip().lower()
    max_chars = _clamp_int(config.get("max_chars"), 12000, 500, 50000)
    max_bytes = _clamp_int(config.get("max_bytes"), 1_000_000, 10000, 2_000_000)

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return _friendly_scrape_error(reason="valid http URL required")

    try:
        async with httpx.AsyncClient(
            timeout=20,
            follow_redirects=True,
            headers={"User-Agent": "Remy local workflow scraper/1.0"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        raw_html = resp.text[:max_bytes]
        parser = _ReadablePageParser()
        parser.feed(raw_html)
        parser.close()
        if mode == "title":
            return parser.title or "[No page title found]"
        if mode == "links":
            return "\n".join(parser.links[:200]) or "[No links found]"
        title_line = f"Title: {parser.title}\n" if parser.title else ""
        page_text = parser.text[:max_chars]
        if not page_text:
            return _friendly_scrape_error(reason="No readable page text found")
        return f"PAGE SCRAPE\nURL: {url}\n{title_line}\n{page_text}"
    except Exception as exc:
        return _friendly_scrape_error(exc)


def _search_tokens(query: str) -> list[str]:
    tokens = re.findall(r"[\w]+", (query or "").lower(), flags=re.UNICODE)
    return [t for t in tokens if len(t) > 2 and t not in _SEARCH_STOP_WORDS]


def _search_text(result: dict) -> str:
    return " ".join(
        str(result.get(key, "") or "")
        for key in ("title", "body", "href")
    ).lower()


def _result_domain(result: dict) -> str:
    try:
        return urlparse(result.get("href", "") or "").netloc.lower()
    except Exception:
        return ""


def _query_has_currency_intent(query: str) -> bool:
    q = (query or "").lower()
    return any(hint in q for hint in _CURRENCY_HINTS)


def _query_allows_noise_domain(query: str, domain: str) -> bool:
    q = (query or "").lower()
    if "music" in domain or "spotify" in domain or "soundcloud" in domain:
        return any(word in q for word in ("music", "song", "пісн", "песн", "трек"))
    if "reverso" in domain or "translate" in domain or "linguee" in domain or "deepl" in domain:
        return any(word in q for word in ("translate", "translation", "переклад", "перевод"))
    return True


def _score_search_result(query: str, result: dict) -> int:
    domain = _result_domain(result)
    text = _search_text(result)
    if any(noise in domain for noise in _SEARCH_NOISE_DOMAINS) and not _query_allows_noise_domain(query, domain):
        return -100

    tokens = _search_tokens(query)
    score = sum(2 for token in tokens if token in text)
    if _query_has_currency_intent(query):
        score += sum(1 for hint in _CURRENCY_HINTS if hint in text)
        if any(hint in domain for hint in ("finance", "bank", "nbu", "minfin", "liga")):
            score += 4
    return score


def _rank_search_results(query: str, results: list[dict], limit: int) -> list[dict]:
    scored = [
        (index, _score_search_result(query, result), result)
        for index, result in enumerate(results)
    ]
    relevant = [(i, score, result) for i, score, result in scored if score >= 0]
    pool = relevant or scored
    pool.sort(key=lambda item: (item[1], -item[0]), reverse=True)
    return [result for _i, _score, result in pool[:limit]]


async def _run_web_search(config: dict) -> str:
    query = config.get("query", "")
    num_results = _clamp_int(config.get("num_results"), 5, 1, 10)
    fetch_content = config.get("fetch_content", True)  # read page text by default
    if not query:
        return ""
    try:
        from ddgs import DDGS
        raw_limit = min(max(num_results * 3, num_results), 20)
        results = await asyncio.to_thread(
            lambda: list(DDGS().text(query, max_results=raw_limit))
        )
        if not results:
            return "[No search results found]"
        selected = _rank_search_results(query, results, num_results)

        parts = []
        # Fetch full content for all results concurrently (respects num_results)
        if fetch_content:
            urls = [r.get("href", "") for r in selected if r.get("href")]
            page_texts = await asyncio.gather(*[_fetch_page_text(u) for u in urls])
            url_to_text = dict(zip(urls, page_texts))
        else:
            url_to_text = {}

        parts.append(
            "WEB SEARCH RESULTS\n"
            f"Query: {query}\n"
            f"Selected sources: {len(selected)} of {len(results)} raw results\n"
            "Use only the source text below. If sources are mixed or weak, say so explicitly."
        )
        for index, r in enumerate(selected, start=1):
            title   = r.get("title", "")
            snippet = r.get("body", "")
            url     = r.get("href", "")
            full    = url_to_text.get(url, "")
            if full:
                parts.append(
                    f"[{index}] {title}\n"
                    f"URL: {url}\n"
                    "Fetched content: yes\n"
                    f"Snippet: {snippet}\n\n"
                    f"Content:\n{full}"
                )
            else:
                parts.append(
                    f"[{index}] {title}\n"
                    f"URL: {url}\n"
                    "Fetched content: no\n"
                    f"Snippet: {snippet}"
                )

        return "\n\n---\n\n".join(parts)
    except Exception as exc:
        logger.warning("Web search failed: %s", exc)
        return f"[Search error: {exc}]"


async def _run_memory_search(config: dict) -> str:
    query = config.get("query", "")
    limit = _clamp_int(config.get("limit"), 5, 1, 20)
    skip_empty = bool(config.get("skip_empty_result"))
    if not query:
        return ""
    try:
        from remy.core.agent_tools import brain, brain_lock
        results = await asyncio.to_thread(lambda: brain.search(query, top_k=limit))
        if not results:
            if skip_empty:
                return ""
            return "[Nothing found in memory]"
        parts = []
        for r in results:
            text = (
                getattr(r, "content", None)
                or getattr(r, "text", None)
                or (r.get("content") or r.get("text") if isinstance(r, dict) else None)
                or str(r)
            )
            parts.append(text or "")
        return "\n\n---\n\n".join(p for p in parts if p)
    except Exception as exc:
        logger.warning("Memory search failed: %s", exc)
        return f"[Memory search error: {exc}]"


async def _run_memory_save(config: dict) -> str:
    text = config.get("text") or config.get("input_source") or config.get("_input") or ""
    tags = config.get("tags", "pipeline")
    dedup_guard = bool(config.get("dedup_guard") or config.get("deduplicate"))
    if not text:
        return "[Empty text — nothing saved]"
    try:
        from remy.core.agent_tools import brain, brain_lock
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        if dedup_guard:
            existing = await asyncio.to_thread(lambda: brain.search(text, top_k=3))
            normalized_text = _normalize_memory_text(text)
            for record in existing or []:
                existing_text = (
                    getattr(record, "content", None)
                    or getattr(record, "text", None)
                    or (record.get("content") or record.get("text") if isinstance(record, dict) else None)
                    or str(record)
                )
                if _normalize_memory_text(existing_text or "") == normalized_text:
                    return "Skipped duplicate memory save"
        await asyncio.to_thread(
            lambda: brain.store(text, tags=tag_list, metadata={"source": "pipeline"})
        )
        return f"Saved to memory ({len(text)} characters)"
    except Exception as exc:
        logger.warning("Memory save failed: %s", exc)
        return f"[Save error: {exc}]"


def _normalize_memory_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


async def _run_http_request(config: dict) -> str:
    import httpx
    url = config.get("url", "")
    method = config.get("method", "GET").upper()
    body_text = config.get("body", "")
    headers_raw = config.get("headers", "")

    if not url:
        return "[HTTP error: Enter a valid http(s) URL before running the request.]"
    if method not in {"GET", "POST"}:
        return "[HTTP error: method must be GET or POST]"

    headers: dict[str, str] = {}
    for line in headers_raw.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            headers[k.strip()] = v.strip()
    secret_value = _resolve_local_secret(str(config.get("auth_secret_key", "") or ""))
    if secret_value:
        scheme = str(config.get("auth_scheme", "Bearer") or "Bearer").strip()
        headers["Authorization"] = f"{scheme} {secret_value}".strip() if scheme else secret_value

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            if method == "GET":
                resp = await client.get(url, headers=headers)
            else:
                resp = await client.post(url, content=body_text.encode(), headers=headers)
            resp.raise_for_status()
            text = resp.text
            return text[:4000] if len(text) > 4000 else text
    except Exception as exc:
        return _friendly_http_error(exc)


async def _run_template(config: dict) -> str:
    return config.get("text", "")


async def _run_merge(config: dict) -> str:
    raw_inputs = config.get("_merge_inputs") or []
    inputs = [str(item or "") for item in raw_inputs] if isinstance(raw_inputs, list) else [str(raw_inputs or "")]
    mode = str(config.get("mode", "combine_text") or "combine_text").strip().lower()
    separator = str(config.get("separator", "\n\n---\n\n") or "")

    if mode == "first_non_empty":
        return next((item for item in inputs if item.strip()), "")
    if mode == "json_array":
        return json.dumps(inputs, ensure_ascii=False)
    return separator.join(inputs)


async def _run_delay(config: dict) -> str:
    seconds = max(0.0, min(float(config.get("seconds") or 0), 300.0))
    if seconds:
        await asyncio.sleep(seconds)
    return str(config.get("_input", "") or config.get("input", "") or "")


async def _run_filter(config: dict) -> str:
    data_ref = str(config.get("_data_ref", "") or config.get("input", "") or "")
    route = {
        "operator": str(config.get("operator", "contains") or "contains"),
        "value": str(config.get("value", "") or ""),
        "condition": str(config.get("condition", "") or ""),
    }
    matched = _route_condition_simple_match(route, data_ref)
    if matched is None:
        matched = await _route_condition_matches(route, data_ref)
    return data_ref if matched else STOP_TOKEN


async def _run_set_variable(config: dict) -> str:
    return str(config.get("value", "") or "")


def _json_path_value(data: Any, path: str) -> Any:
    if not path or path == "$":
        return data
    current = data
    parts = path[2:].split(".") if path.startswith("$.") else path.split(".")
    for part in [p for p in parts if p != ""]:
        match = re.fullmatch(r"([^\[]+)(?:\[(\d+)\])?", part)
        if not match:
            raise KeyError(part)
        key, index = match.groups()
        if isinstance(current, dict):
            current = current[key]
        else:
            raise KeyError(key)
        if index is not None:
            current = current[int(index)]
    return current


async def _run_parse_json(config: dict) -> str:
    raw = str(config.get("text", "") or "")
    path = str(config.get("path", "$") or "$").strip()
    if not raw.strip():
        return ""
    try:
        value = _json_path_value(json.loads(raw), path)
    except Exception as exc:
        return f"[JSON parse error: {exc}]"
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)


async def _run_transform(config: dict) -> str:
    text = str(config.get("text", "") or "")
    mode = str(config.get("mode", "trim") or "trim").strip().lower()
    if mode == "trim":
        return text.strip()
    if mode == "lower":
        return text.lower()
    if mode == "upper":
        return text.upper()
    if mode == "replace":
        return text.replace(str(config.get("find", "") or ""), str(config.get("replace", "") or ""))
    if mode == "regex_replace":
        try:
            return re.sub(str(config.get("pattern", "") or ""), str(config.get("replace", "") or ""), text)
        except re.error as exc:
            return f"[Regex error: {exc}]"
    if mode == "extract_regex":
        try:
            match = re.search(str(config.get("pattern", "") or ""), text, flags=re.MULTILINE)
        except re.error as exc:
            return f"[Regex error: {exc}]"
        if not match:
            return ""
        return match.group(1) if match.groups() else match.group(0)
    if mode == "truncate":
        limit = max(0, min(int(config.get("limit") or 1000), 50000))
        return text[:limit]
    if mode == "join_lines":
        separator = str(config.get("separator", " ") or " ")
        return separator.join(line.strip() for line in text.splitlines() if line.strip())
    return text


async def _run_notification(config: dict) -> str:
    title = str(config.get("title", "Notification") or "Notification")
    message = str(config.get("message", "") or "")
    content = f"{title}: {message}" if message else title
    logger.info("Workflow notification: %s - %s", title, message)
    try:
        from remy.core.agent_tools import brain, brain_lock

        def _store() -> None:
            with brain_lock:
                brain.store(
                    content,
                    tags=["workflow-notification", "notification"],
                    metadata={"source": "workflow", "title": title},
                )

        await asyncio.to_thread(_store)
    except Exception as exc:
        logger.debug("Workflow notification memory store skipped: %s", exc)
    return content


def _workflow_files_dir() -> Path:
    from remy.config.settings import settings

    directory = settings.DATA_DIR / "workflow_files"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _safe_workflow_file(name: str) -> Path:
    filename = Path(str(name or "")).name
    if not filename or filename in {".", ".."}:
        raise ValueError("File name is required")
    return _workflow_files_dir() / filename


async def _run_file_read(config: dict) -> str:
    try:
        path = _safe_workflow_file(str(config.get("filename", "") or ""))
        if not path.exists():
            return f"[File not found: {path.name}]"
        limit = max(1, min(int(config.get("max_chars") or 20000), 100000))
        return path.read_text(encoding="utf-8")[:limit]
    except Exception as exc:
        return f"[File read error: {exc}]"


async def _run_file_write(config: dict) -> str:
    try:
        path = _safe_workflow_file(str(config.get("filename", "") or ""))
        text = str(config.get("text", "") or "")
        mode = str(config.get("mode", "overwrite") or "overwrite")
        if mode == "append":
            with path.open("a", encoding="utf-8") as fh:
                fh.write(text)
        elif mode == "overwrite":
            atomic_write(path, text)
        else:
            return "[File write error: mode must be overwrite or append]"
        return f"Saved file: {path.name} ({len(text)} characters)"
    except Exception as exc:
        return f"[File write error: {exc}]"


_SAFE_CODE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "sorted": sorted,
    "str": str,
    "sum": sum,
}
_SAFE_CODE_MODULES = {
    "json": json,
    "re": re,
}
_SAFE_CODE_NAMES = set(_SAFE_CODE_BUILTINS) | set(_SAFE_CODE_MODULES) | {"input", "prev", "vars"}


def _validate_safe_expression(expr: str) -> None:
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Lambda, ast.FunctionDef, ast.ClassDef, ast.Global, ast.Nonlocal)):
            raise ValueError("imports, functions, classes, and lambdas are not allowed in safe expression mode")
        if isinstance(node, ast.Name) and node.id not in _SAFE_CODE_NAMES:
            raise ValueError(f"name '{node.id}' is not available in safe expression mode")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("private attributes are not allowed in safe expression mode")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id not in _SAFE_CODE_BUILTINS:
                raise ValueError(f"function '{func.id}' is not allowed in safe expression mode")
            if isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Name) and func.value.id in _SAFE_CODE_MODULES:
                    continue
                if isinstance(func.value, ast.Name) and func.value.id in {"input", "prev"}:
                    continue
                if isinstance(func.value, ast.Call):
                    continue
                raise ValueError("only json.*, re.*, input.*, and prev.* calls are allowed in safe expression mode")


def _clip_code_output(value: Any, limit: int) -> str:
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = "" if value is None else str(value)
    limit = max(1, min(int(limit or 12000), 50000))
    return text[:limit]


def _run_safe_python_expression(config: dict, ctx_input: str, output_limit: int) -> str:
    expr = str(config.get("code", "") or "").strip()
    if not expr:
        return ""
    try:
        _validate_safe_expression(expr)
        value = eval(  # noqa: S307 - AST-validated and builtins are explicitly constrained.
            compile(ast.parse(expr, mode="eval"), "<workflow-safe-expression>", "eval"),
            {"__builtins__": {}, **_SAFE_CODE_BUILTINS, **_SAFE_CODE_MODULES},
            {
                "input": ctx_input,
                "prev": ctx_input,
                "vars": config.get("_vars", {}) if isinstance(config.get("_vars"), dict) else {},
            },
        )
        return _clip_code_output(value, output_limit)
    except Exception as exc:
        return f"[Code error: {exc}]"


def _run_local_python_script(config: dict, ctx_input: str, timeout_s: float, output_limit: int) -> str:
    code = str(config.get("code", "") or "")
    if not code.strip():
        return ""

    payload = {
        "input": ctx_input,
        "prev": ctx_input,
        "vars": config.get("_vars", {}) if isinstance(config.get("_vars"), dict) else {},
    }
    wrapper = (
        "import json, sys\n"
        "_payload = json.loads(sys.stdin.read() or '{}')\n"
        "input = _payload.get('input', '')\n"
        "prev = _payload.get('prev', '')\n"
        "vars = _payload.get('vars', {})\n"
        "result = None\n"
        f"{code}\n"
        "if result is not None:\n"
        "    print(json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list, tuple)) else str(result))\n"
    )
    work_dir = _workflow_files_dir()
    with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False, dir=work_dir) as fh:
        fh.write(wrapper)
        script_path = Path(fh.name)
    try:
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            cwd=work_dir,
            timeout=timeout_s,
            env={},
            check=False,
        )
        output = completed.stdout.strip()
        if completed.returncode != 0:
            err = (completed.stderr or output or f"exit code {completed.returncode}").strip()
            return f"[Code error: {err[:output_limit]}]"
        return _clip_code_output(output, output_limit)
    except subprocess.TimeoutExpired:
        return f"[Code error: timed out after {timeout_s:g}s]"
    except Exception as exc:
        return f"[Code error: {exc}]"
    finally:
        try:
            script_path.unlink(missing_ok=True)
        except Exception:
            pass


def _run_local_javascript_script(config: dict, ctx_input: str, timeout_s: float, output_limit: int) -> str:
    node = shutil.which("node")
    if not node:
        return "[Code error: Node.js is not available in this runtime. Use Python or Safe Expression.]"
    code = str(config.get("code", "") or "")
    if not code.strip():
        return ""
    payload = {
        "input": ctx_input,
        "prev": ctx_input,
        "vars": config.get("_vars", {}) if isinstance(config.get("_vars"), dict) else {},
    }
    wrapper = (
        "const fs = require('fs');\n"
        "const payload = JSON.parse(fs.readFileSync(0, 'utf8') || '{}');\n"
        "const input = payload.input || '';\n"
        "const prev = payload.prev || '';\n"
        "const vars = payload.vars || {};\n"
        "let result;\n"
        "(async () => {\n"
        f"{code}\n"
        "if (result !== undefined) console.log(typeof result === 'object' ? JSON.stringify(result) : String(result));\n"
        "})().catch(err => { console.error(err && err.stack ? err.stack : String(err)); process.exit(1); });\n"
    )
    work_dir = _workflow_files_dir()
    with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8", delete=False, dir=work_dir) as fh:
        fh.write(wrapper)
        script_path = Path(fh.name)
    try:
        completed = subprocess.run(
            [node, str(script_path)],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            cwd=work_dir,
            timeout=timeout_s,
            env={},
            check=False,
        )
        output = completed.stdout.strip()
        if completed.returncode != 0:
            err = (completed.stderr or output or f"exit code {completed.returncode}").strip()
            return f"[Code error: {err[:output_limit]}]"
        return _clip_code_output(output, output_limit)
    except subprocess.TimeoutExpired:
        return f"[Code error: timed out after {timeout_s:g}s]"
    except Exception as exc:
        return f"[Code error: {exc}]"
    finally:
        try:
            script_path.unlink(missing_ok=True)
        except Exception:
            pass


async def _run_code(config: dict) -> str:
    mode = str(config.get("mode", "safe_expression") or "safe_expression").strip().lower()
    language = str(config.get("language", "python") or "python").strip().lower()
    ctx_input = str(config.get("_input", "") or config.get("input", "") or "")
    timeout_s = max(0.1, min(float(config.get("timeout_seconds") or 3), 30.0))
    output_limit = max(1, min(int(config.get("max_output_chars") or 12000), 50000))

    if mode in {"safe", "safe_expression", "expression"}:
        return await asyncio.to_thread(_run_safe_python_expression, config, ctx_input, output_limit)

    if mode in {"local", "local_script"}:
        if not bool(config.get("allow_local_execution")):
            return "[Code blocked: enable local execution in this block to run scripts on this computer.]"
        if language in {"python", "py"}:
            return await asyncio.to_thread(_run_local_python_script, config, ctx_input, timeout_s, output_limit)
        if language in {"javascript", "js", "node"}:
            return await asyncio.to_thread(_run_local_javascript_script, config, ctx_input, timeout_s, output_limit)
        return f"[Code error: unsupported language '{language}']"

    return f"[Code error: unsupported code mode '{mode}']"


async def _run_error_handler(config: dict) -> str:
    fallback = str(config.get("fallback_text", "") or "").strip()
    if fallback:
        return fallback
    error = str(config.get("_error", "") or config.get("error", "") or "").strip()
    current = str(config.get("_input", "") or config.get("input", "") or "").strip()
    return error or current


async def _run_currency(config: dict) -> str:
    """Fetch exchange rates from NBU (National Bank of Ukraine) official API."""
    import httpx, json as _json
    currencies = [c.strip().upper() for c in config.get("currencies", "USD,EUR").split(",") if c.strip()]
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange?json")
            resp.raise_for_status()
            data = resp.json()
        rates = {r["cc"]: r for r in data}
        lines = [f"Курс НБУ на сьогодні:"]
        for cc in currencies:
            r = rates.get(cc)
            if r:
                lines.append(f"  {cc}: {r['rate']:.4f} грн  ({r['txt']})")
            else:
                lines.append(f"  {cc}: не знайдено")
        return "\n".join(lines)
    except Exception as exc:
        return f"[Помилка отримання курсу: {exc}]"


async def _run_condition(config: dict) -> str:
    """Evaluate a condition using LLM and return 'true' or 'false'."""
    condition = config.get("condition", "")
    data_ref = config.get("_data_ref", "")  # already substituted by _resolve_config
    if not condition:
        return "true"
    from remy.core.llm import get_llm
    from remy.config.settings import settings
    llm = get_llm(settings.SUMMARY_MODEL)
    data_section = f"\n\nДані для перевірки:\n{data_ref}" if data_ref else ""
    prompt = (
        f"Evaluate this condition and respond with ONLY 'true' or 'false' (no other text):\n{condition}{data_section}"
    )
    result = await asyncio.to_thread(llm.invoke, prompt)
    content = result.content if hasattr(result, "content") else str(result)
    if isinstance(content, list):
        content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    verdict = content.strip().lower()
    return "true" if "true" in verdict else "false"


def _router_routes(config: dict) -> list[dict[str, str]]:
    routes: list[dict[str, str]] = []
    raw_routes = config.get("routes")
    if isinstance(raw_routes, list):
        for idx, raw in enumerate(raw_routes, start=1):
            if not isinstance(raw, dict):
                continue
            label = str(raw.get("label", "") or "").strip()
            condition = str(raw.get("condition", "") or "").strip()
            operator = str(raw.get("operator", "") or "").strip()
            value = str(raw.get("value", "") or "").strip()
            if not operator and condition:
                operator, value = _split_legacy_router_condition(condition)
            if label or condition or operator or value:
                routes.append({
                    "index": str(idx),
                    "label": label or f"Route {idx}",
                    "condition": condition,
                    "operator": operator or ("fallback" if not condition and not value else "contains"),
                    "value": value or condition,
                })

    if not routes:
        for idx in range(1, 25):
            label = str(config.get(f"route_{idx}_label", "") or "").strip()
            condition = str(config.get(f"route_{idx}_condition", "") or "").strip()
            if label or condition:
                operator, value = _split_legacy_router_condition(condition)
                routes.append({
                    "index": str(idx),
                    "label": label or f"Route {idx}",
                    "condition": condition,
                    "operator": operator,
                    "value": value,
                })
    return routes or [{"index": "1", "label": "Route 1", "condition": "", "operator": "fallback", "value": ""}]


def _split_legacy_router_condition(condition: str) -> tuple[str, str]:
    condition = (condition or "").strip()
    cond_l = condition.lower()
    if not condition:
        return "fallback", ""
    if cond_l in {"always", "true", "*"}:
        return "always", ""
    if cond_l in {"never", "false"}:
        return "never", ""
    for operator, prefixes in (
        ("contains", ("contains:", "contains ")),
        ("not_contains", ("not contains:", "not contains ")),
        ("equals", ("equals:", "equals ")),
    ):
        for prefix in prefixes:
            if cond_l.startswith(prefix):
                return operator, condition[len(prefix):].strip()
    return "contains" if len(cond_l) <= 120 and "\n" not in cond_l else "ai", condition


def _route_condition_simple_match(route: dict[str, str], data_ref: str) -> bool | None:
    operator = (route.get("operator") or "").strip().lower()
    value = (route.get("value") or "").strip()
    condition = (route.get("condition") or "").strip()
    if not operator and condition:
        operator, value = _split_legacy_router_condition(condition)
    if operator in {"", "fallback"}:
        return None
    value_l = value.lower()
    data_l = (data_ref or "").lower()
    if operator in {"always", "true"}:
        return True
    if operator in {"never", "false"}:
        return False
    if operator == "contains":
        return bool(value_l) and value_l in data_l
    if operator == "not_contains":
        return bool(value_l) and value_l not in data_l
    if operator == "equals":
        return data_l.strip() == value_l
    if operator == "not_equals":
        return data_l.strip() != value_l
    if operator == "starts_with":
        return bool(value_l) and data_l.strip().startswith(value_l)
    if operator == "ends_with":
        return bool(value_l) and data_l.strip().endswith(value_l)
    if operator == "is_empty":
        return not data_l.strip()
    if operator == "is_not_empty":
        return bool(data_l.strip())
    if operator == "regex":
        try:
            return re.search(value, data_ref or "", flags=re.IGNORECASE) is not None
        except re.error:
            return False
    return None


async def _route_condition_matches(route: dict[str, str], data_ref: str) -> bool:
    simple = _route_condition_simple_match(route, data_ref)
    if simple is not None:
        return simple

    from remy.core.llm import get_llm
    from remy.config.settings import settings

    prompt = (
        "Evaluate whether this router condition allows the route to run. "
        "Respond ONLY with 'true' or 'false'.\n\n"
        f"Input:\n{data_ref}\n\nCondition:\n{route.get('value') or route.get('condition') or ''}"
    )
    result = await asyncio.to_thread(get_llm(settings.SUMMARY_MODEL).invoke, prompt)
    content = result.content if hasattr(result, "content") else str(result)
    if isinstance(content, list):
        content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    verdict = (content or "").strip().lower()
    return "true" in verdict and "false" not in verdict


async def _run_router(config: dict) -> str:
    """Return comma-separated 1-based route indexes selected by router mode."""
    routes = _router_routes(config)
    mode = str(config.get("mode", "all_matching") or "all_matching").strip().lower()
    data_ref = str(config.get("_data_ref", "") or config.get("input", "") or "")

    if mode in {"all", "pass_all", "all_routes"}:
        return ",".join(route["index"] for route in routes)

    matched: list[str] = []
    fallbacks: list[str] = []
    for route in routes:
        operator = route.get("operator", "").strip().lower()
        value = route.get("value", "").strip()
        if operator in {"", "fallback"} or (not value and operator not in {"always", "never", "is_empty", "is_not_empty"}):
            fallbacks.append(route["index"])
            continue
        if await _route_condition_matches(route, data_ref):
            matched.append(route["index"])
            if mode in {"first", "first_match", "first_matching"}:
                return route["index"]

    if matched:
        return ",".join(matched)
    if fallbacks:
        return fallbacks[0] if mode in {"first", "first_match", "first_matching"} else ",".join(fallbacks)
    return routes[0]["index"]


async def _run_loop(config: dict) -> str:
    """Run inner steps in a loop until stop_condition is met or max_iterations reached."""
    max_iter = int(config.get("max_iterations", 3))
    stop_condition = config.get("stop_condition", "")
    inner_steps = config.get("steps", [])
    last_output = config.get("_input", "")

    if not inner_steps:
        return last_output or "[Empty loop]"

    from remy.core.llm import get_llm
    from remy.config.settings import settings

    for i in range(max_iter):
        inner_ctx: dict[str, Any] = {"input": last_output}
        for step in inner_steps:
            last_output = await _execute_step(step, inner_ctx)
            inner_ctx[step.get("id", "s1")] = {"output": last_output}

        if stop_condition:
            llm = get_llm(settings.SUMMARY_MODEL)
            check_prompt = (
                f"Evaluate if this stop condition is met: '{stop_condition}'\n"
                f"Current output: {last_output}\n"
                f"Respond ONLY 'yes' or 'no'."
            )
            check = await asyncio.to_thread(llm.invoke, check_prompt)
            check_content = check.content if hasattr(check, "content") else str(check)
            if isinstance(check_content, list):
                check_content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in check_content)
            if "yes" in check_content.strip().lower():
                break

    return last_output


# ── Dispatcher ────────────────────────────────────────────────────────────────

_RUNNERS = {
    "llm_call": _run_llm_call,
    "web_search": _run_web_search,
    "memory_search": _run_memory_search,
    "memory_save": _run_memory_save,
    "http_request": _run_http_request,
    "page_scrape": _run_page_scrape,
    "template": _run_template,
    "merge": _run_merge,
    "delay": _run_delay,
    "filter": _run_filter,
    "set_variable": _run_set_variable,
    "parse_json": _run_parse_json,
    "transform": _run_transform,
    "notification": _run_notification,
    "file_read": _run_file_read,
    "file_write": _run_file_write,
    "code": _run_code,
    "error_handler": _run_error_handler,
    "router": _run_router,
    "condition": _run_condition,
    "loop": _run_loop,
    "currency": _run_currency,
}


async def run_single_step(step: dict, input_text: str) -> str:
    ctx = {"input": input_text}
    return await _execute_step(step, ctx)


async def _execute_step(step: dict, ctx: dict) -> str:
    step_type = step.get("type", "")
    config = _resolve_config(step.get("config", {}), ctx)
    config.setdefault("_input", ctx.get("prev", ctx.get("input", "")))
    config.setdefault("_error", ctx.get("error", ""))
    if config.get("_pinned_enabled") and str(config.get("_pinned_output", "") or ""):
        return str(config.get("_pinned_output", "") or "")

    runner = _RUNNERS.get(step_type)
    if runner is None:
        return f"[Unknown block type: {step_type}]"

    return await runner(config)


# ── Main pipeline runner ──────────────────────────────────────────────────────

def _pipeline_has_graph(steps: list[dict]) -> bool:
    return any(step.get("_connections") for step in steps)


def _step_sort_key(step: dict) -> int:
    try:
        return int(step.get("_df_id") or str(step.get("id", "")).lstrip("s") or 0)
    except Exception:
        return 0


def _next_step_targets_from_connection(step: dict, steps_by_id: dict[str, dict], output_name: str) -> list[tuple[dict, str]]:
    connections = (
        (step.get("_connections") or {})
        .get(output_name, {})
        .get("connections", [])
    )
    next_steps: list[tuple[dict, str]] = []
    for conn in connections:
        node_id = str(conn.get("node", "") or "")
        next_step = steps_by_id.get(f"s{node_id}")
        if next_step:
            next_steps.append((next_step, str(conn.get("output", "input_1") or "input_1")))
    return next_steps


def _next_steps_from_connection(step: dict, steps_by_id: dict[str, dict], output_name: str) -> list[dict]:
    return [next_step for next_step, _input_name in _next_step_targets_from_connection(step, steps_by_id, output_name)]


def _merge_required_inputs(step: dict) -> list[str]:
    inputs = step.get("_inputs") or {}
    required = [
        name
        for name, input_meta in sorted(inputs.items())
        if (input_meta or {}).get("connections")
    ]
    if required:
        return required
    try:
        count = int((step.get("config") or {}).get("input_count") or 2)
    except Exception:
        count = 2
    count = max(2, min(count, 10))
    return [f"input_{idx}" for idx in range(1, count + 1)]


def _merge_step_with_inputs(step: dict, inputs: list[str]) -> dict:
    return {
        **step,
        "config": {
            **(step.get("config") or {}),
            "_merge_inputs": inputs,
        },
    }


def _router_output_names(output: str) -> list[str]:
    indexes = re.findall(r"\d+", output or "")
    if not indexes:
        indexes = ["1"]
    seen: set[str] = set()
    names: list[str] = []
    for idx in indexes:
        name = f"output_{idx}"
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _is_error_output(output: str) -> bool:
    return any((output or "").startswith(prefix) for prefix in ERROR_PREFIXES)


def _first_pipeline_step(steps: list[dict]) -> dict | None:
    if not steps:
        return None
    target_ids: set[str] = set()
    step_ids = {str(step.get("id")) for step in steps}
    for step in steps:
        for output in (step.get("_connections") or {}).values():
            for conn in output.get("connections", []):
                target = f"s{conn.get('node', '')}"
                if target in step_ids:
                    target_ids.add(target)
    roots = [step for step in steps if str(step.get("id")) not in target_ids]
    return sorted(roots or steps, key=_step_sort_key)[0]


async def run_pipeline_steps(
    steps: list[dict],
    input_text: str,
) -> AsyncGenerator[dict, None]:
    """
    Execute steps sequentially, yielding SSE-ready event dicts.

    Events:
      {type: "start",    total: N}
      {type: "step_start",  index: i, id: str, step_type: str, label: str}
      {type: "step_done",   index: i, id: str, step_type: str, label: str, output: str}
      {type: "step_error",  index: i, id: str, step_type: str, label: str, error: str}
      {type: "done",     output: str}
    """
    ctx: dict[str, Any] = {"input": input_text}
    last_output = input_text

    yield {"type": "start", "total": len(steps)}

    if _pipeline_has_graph(steps):
        steps_by_id = {str(step.get("id")): step for step in steps}
        first_step = _first_pipeline_step(steps)
        queue: list[tuple[dict, str]] = [(first_step, input_text)] if first_step else []
        final_outputs: list[str] = []
        merge_buffers: dict[str, dict[str, str]] = {}
        seen: set[tuple[str, str]] = set()
        i = 0

        while queue and i < max(len(steps) * 8, 32):
            step, inherited_output = queue.pop(0)
            step_id = step.get("id", f"s{i+1}")
            visit_key = (str(step_id), inherited_output[:500])
            if visit_key in seen:
                continue
            seen.add(visit_key)

            label = step.get("label", f"Step {i+1}")
            step_type = step.get("type", "step")
            ctx["prev"] = inherited_output
            ctx[f"s{i}.output"] = inherited_output

            yield {"type": "step_start", "index": i, "id": step_id, "step_type": step_type, "label": label}

            try:
                output = await _execute_step(step, ctx)
                error_output = _is_error_output(output) and step_type != "router"
                if step_type == "set_variable":
                    variable_name = str((step.get("config") or {}).get("name", "") or "").strip()
                    if variable_name:
                        ctx[variable_name] = output
                if error_output:
                    ctx["error"] = output
                    ctx["failed_step"] = label
                ctx[step_id] = {"output": output}
                ctx[f"s{i+1}"] = {"output": output}
                ctx[f"{step_id}.output"] = output
                ctx[f"s{i+1}.output"] = output

                if error_output:
                    route_outputs = ["output_2"]
                    trace_output = output
                    branch_input = output
                elif step_type == "router":
                    route_outputs = _router_output_names(output)
                    trace_output = f"Selected routes: {', '.join(route_outputs)}"
                    branch_input = inherited_output
                elif output == STOP_TOKEN:
                    route_outputs = []
                    trace_output = "Stopped by filter"
                    branch_input = ""
                else:
                    last_output = output
                    trace_output = output
                    route_outputs = ["output_1"]
                    branch_input = output

                event_type = "step_error" if error_output else "step_done"
                event = {"type": event_type, "index": i, "id": step_id, "step_type": step_type, "label": label}
                if error_output:
                    event["error"] = trace_output
                else:
                    event["output"] = trace_output
                yield event

                enqueued = False
                for output_name in route_outputs:
                    for next_step, input_name in _next_step_targets_from_connection(step, steps_by_id, output_name):
                        if next_step.get("type") == "merge":
                            merge_id = str(next_step.get("id"))
                            buffer = merge_buffers.setdefault(merge_id, {})
                            buffer[input_name] = branch_input
                            required_inputs = _merge_required_inputs(next_step)
                            if all(name in buffer for name in required_inputs):
                                ordered_inputs = [buffer.get(name, "") for name in required_inputs]
                                queue.append((_merge_step_with_inputs(next_step, ordered_inputs), "\n\n---\n\n".join(ordered_inputs)))
                                merge_buffers.pop(merge_id, None)
                            enqueued = True
                            continue
                        queue.append((next_step, branch_input))
                        enqueued = True
                if not enqueued and step_type != "router" and output != STOP_TOKEN and not error_output:
                    final_outputs.append(branch_input)
            except Exception as exc:
                error_text = str(exc) or exc.__class__.__name__
                logger.error("Step %d (%s) failed: %s", i, label, exc)
                yield {"type": "step_error", "index": i, "id": step_id, "step_type": step_type, "label": label, "error": error_text}
                ctx[step_id] = {"output": ""}
                ctx[f"s{i+1}"] = {"output": ""}
                if step_type != "router":
                    ctx["error"] = error_text
                    ctx["failed_step"] = label
                    for next_step, input_name in _next_step_targets_from_connection(step, steps_by_id, "output_2"):
                        if next_step.get("type") == "merge":
                            merge_id = str(next_step.get("id"))
                            buffer = merge_buffers.setdefault(merge_id, {})
                            buffer[input_name] = error_text
                            required_inputs = _merge_required_inputs(next_step)
                            if all(name in buffer for name in required_inputs):
                                ordered_inputs = [buffer.get(name, "") for name in required_inputs]
                                queue.append((_merge_step_with_inputs(next_step, ordered_inputs), "\n\n---\n\n".join(ordered_inputs)))
                                merge_buffers.pop(merge_id, None)
                            continue
                        queue.append((next_step, error_text))
            i += 1

        if final_outputs:
            last_output = "\n\n---\n\n".join(final_outputs)
    else:
        for i, step in enumerate(steps):
            label = step.get("label", f"Step {i+1}")
            step_id = step.get("id", f"s{i+1}")
            step_type = step.get("type", "step")

            yield {"type": "step_start", "index": i, "id": step_id, "step_type": step_type, "label": label}

            try:
                ctx["prev"] = last_output
                output = await _execute_step(step, ctx)
                error_output = _is_error_output(output) and step_type != "router"
                if step_type == "set_variable":
                    variable_name = str((step.get("config") or {}).get("name", "") or "").strip()
                    if variable_name:
                        ctx[variable_name] = output
                if error_output:
                    ctx["error"] = output
                    ctx["failed_step"] = label
                ctx[step_id] = {"output": output}
                ctx[f"s{i+1}"] = {"output": output}
                ctx[f"{step_id}.output"] = output
                ctx[f"s{i+1}.output"] = output
                if error_output:
                    yield {"type": "step_error", "index": i, "id": step_id, "step_type": step_type, "label": label, "error": output}
                    last_output = output
                    continue
                if step_type == "router":
                    trace_output = f"Selected routes: {', '.join(_router_output_names(output))}"
                elif output == STOP_TOKEN:
                    trace_output = "Stopped by filter"
                    yield {"type": "step_done", "index": i, "id": step_id, "step_type": step_type, "label": label, "output": trace_output}
                    break
                else:
                    last_output = output
                    trace_output = output
                yield {"type": "step_done", "index": i, "id": step_id, "step_type": step_type, "label": label, "output": trace_output}
            except Exception as exc:
                logger.error("Step %d (%s) failed: %s", i, label, exc)
                yield {"type": "step_error", "index": i, "id": step_id, "step_type": step_type, "label": label, "error": str(exc)}
                ctx[step_id] = {"output": ""}
                ctx[f"s{i+1}"] = {"output": ""}

    yield {"type": "done", "output": last_output}
