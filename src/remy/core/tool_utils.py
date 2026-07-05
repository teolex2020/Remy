"""
Tool Utilities — parsing, caching, SSRF protection, tag sanitization.

Stateless helpers (parse_llm_json, estimate_tokens, _check_ssrf) plus
two caching layers (recall cache, web search cache) and tag utilities.
"""

import json
import logging
import random
import re
import time


def _get_brain():
    """Lazy accessor — reads brain from brain_tools (supports test patching)."""
    import remy.core.brain_tools as _bt

    return _bt.brain


def _get_brain_lock():
    """Lazy accessor — reads brain_lock from brain_tools (supports test patching)."""
    import remy.core.brain_tools as _bt

    return _bt.brain_lock


logger = logging.getLogger("BrainTools")

# ============== SHARED UTILITIES ==============


def parse_llm_json(raw: str):
    """Parse JSON from LLM output, tolerating markdown fences, single quotes, trailing commas."""
    text = raw.strip()

    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fix common LLM issues: single quotes, trailing commas
    fixed = text.replace("'", '"')
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Fix unquoted keys: word immediately before colon -> add quotes
    fixed2 = re.sub(r"(?<=[{,\s])(\w+)\s*:", r'"\1":', text)
    fixed2 = fixed2.replace("'", '"')
    fixed2 = re.sub(r",\s*([}\]])", r"\1", fixed2)
    try:
        return json.loads(fixed2)
    except json.JSONDecodeError:
        pass

    # Last resort: find first { or [ and extract to matching close
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        idx = text.find(start_char)
        if idx >= 0:
            depth = 0
            for i in range(idx, len(text)):
                if text[i] == start_char:
                    depth += 1
                elif text[i] == end_char:
                    depth -= 1
                    if depth == 0:
                        candidate = text[idx : i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            pass
                        candidate = re.sub(r"(?<=[{,\s])(\w+)\s*:", r'"\1":', candidate)
                        candidate = candidate.replace("'", '"')
                        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break

    raise json.JSONDecodeError("No valid JSON found in LLM output", text, 0)


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length. Cheap heuristic (~4 chars/token)."""
    return max(1, len(text) // 4)


def _sleep_with_jitter(base_delay: float):
    """Sleep for base_delay +/- 30% random jitter to avoid thundering herd."""
    jitter = base_delay * 0.3 * (2 * random.random() - 1)  # +/-30%
    # Use brain_tools.time.sleep so tests can patch remy.core.brain_tools.time.sleep
    import remy.core.brain_tools as _bt

    _bt.time.sleep(max(0.1, base_delay + jitter))


# ============== SSRF PROTECTION ==============


def _check_ssrf(url: str) -> str | None:
    """Validate URL against SSRF attacks. Returns error string or None if safe."""
    import ipaddress
    import socket
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
    except Exception:
        return "Invalid URL"

    # Block non-HTTP schemes
    if parsed.scheme not in ("http", "https"):
        return f"Blocked scheme: {parsed.scheme}. Only http/https allowed."

    hostname = parsed.hostname
    if not hostname:
        return "Missing hostname"

    # Resolve hostname to IP and check against private ranges
    try:
        addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for family, _, _, _, sockaddr in addr_info:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return f"Blocked: {hostname} resolves to private/internal IP ({ip})"
    except (socket.gaierror, ValueError):
        # DNS resolution failed — let urllib handle it downstream
        pass

    return None


# ============== RECALL CACHE ==============
# In-memory cache for recall results. Avoids repeated 6-query recall pipeline
# for identical queries within the same session. TTL-based expiry.
# Thread-safe: all access is serialized under brain_lock via _execute_tool_locked().

_recall_cache: dict[str, tuple[float, str]] = {}  # query_key -> (timestamp, result)
_RECALL_CACHE_TTL_SEC = 300  # 5 minutes
_RECALL_CACHE_MAX_SIZE = 50  # prevent unbounded growth


def _get_cached_recall(query: str) -> str | None:
    """Check in-memory recall cache. Returns cached result or None."""
    key = query.lower().strip()
    entry = _recall_cache.get(key)
    if entry is None:
        return None
    ts, result = entry
    if time.time() - ts < _RECALL_CACHE_TTL_SEC:
        logger.info("Recall cache HIT for: %s", query[:60])
        return result
    # Expired
    _recall_cache.pop(key, None)
    return None


def _cache_recall_result(query: str, result: str) -> None:
    """Store recall result in in-memory cache."""
    key = query.lower().strip()
    if len(_recall_cache) >= _RECALL_CACHE_MAX_SIZE:
        oldest_key = min(_recall_cache, key=lambda k: _recall_cache[k][0])
        del _recall_cache[oldest_key]
    _recall_cache[key] = (time.time(), result)


def clear_recall_cache(new_content: str = "") -> None:
    """Invalidate recall cache entries related to *new_content*.

    If *new_content* is empty (or not provided), falls back to full clear
    for safety (delete / connect / persona-update paths).

    Selective invalidation (v2.3, Rec 14.4): extracts the first 10 keywords
    from *new_content* and removes only cache entries whose query key shares
    at least one keyword.  This preserves unrelated cache entries during
    batch operations (extract_facts, research findings, tracked metrics).
    """
    if not new_content or not _recall_cache:
        _recall_cache.clear()
        return

    keywords = set(new_content.lower().split()[:10])
    if not keywords:
        _recall_cache.clear()
        return

    to_remove = [key for key in _recall_cache if keywords & set(key.split())]

    if len(to_remove) >= len(_recall_cache):
        # All entries affected — just clear
        _recall_cache.clear()
    else:
        for key in to_remove:
            del _recall_cache[key]


# ============== WEB SEARCH CACHE ==============

_SEARCH_CACHE_TAG = "web-search-cache"
_SEARCH_CACHE_TTL_HOURS = 24
_SEARCH_CACHE_BACKEND = "ddgs-v3-pinned"


def _get_cached_search(query: str) -> dict | None:
    """Check if a similar web search was run recently. Returns cached result or None."""
    try:
        with _get_brain_lock():
            cached = _get_brain().search(query=query, tags=[_SEARCH_CACHE_TAG], limit=3)
        if not cached:
            return None

        from datetime import datetime, timedelta

        cutoff = (datetime.now() - timedelta(hours=_SEARCH_CACHE_TTL_HOURS)).isoformat()

        for rec in cached:
            meta = rec.metadata or {}
            cached_at = meta.get("cached_at", "")
            if cached_at > cutoff:
                if meta.get("backend") != _SEARCH_CACHE_BACKEND:
                    continue
                cached_query = meta.get("query", "")
                # Exact or near-exact match (lowercase comparison)
                if cached_query.lower().strip() == query.lower().strip():
                    logger.info("Web search cache HIT for: %s", query[:60])
                    return {
                        "answer": meta.get("answer", rec.content),
                        "sources": meta.get("sources", []),
                        "cached": True,
                        "cached_at": cached_at,
                    }
        return None
    except Exception:
        return None


def _cache_search_result(query: str, answer: str, sources: list[dict]):
    """Cache a web search result for future deduplication."""
    try:
        from datetime import datetime

        from remy.core.agent_tools import Level

        # Summarize answer for storage (save space)
        cached_answer = answer[:500] if len(answer) > 500 else answer

        with _get_brain_lock():
            _get_brain().store(
                content=f"Web search: {query}\n{cached_answer}",
                level=Level.WORKING,
                tags=[_SEARCH_CACHE_TAG],
                metadata={
                    "type": "web_search_cache",
                    "query": query,
                    "answer": cached_answer,
                    "sources": sources,
                    "cached_at": datetime.now().isoformat(),
                    "backend": _SEARCH_CACHE_BACKEND,
                },
                deduplicate=False,
            )
    except Exception as e:
        logger.debug("Failed to cache search result: %s", e)


# ============== TAG UTILITIES ==============

# Cyrillic -> Latin transliteration for tag sanitization
_CYRILLIC_MAP = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "ґ": "g",
    "д": "d",
    "е": "e",
    "ж": "z",
    "з": "z",
    "и": "y",
    "і": "i",
    "ї": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "c",
    "ч": "c",
    "ш": "s",
    "щ": "s",
    "ь": "",
    "ю": "u",
    "я": "a",
    "є": "e",
    "ё": "o",
}


def _clean_tag(tag: str) -> str:
    """Ensure tag only contains valid chars: alphanumeric, underscore, hyphen, colon, period.

    Replaces spaces with hyphens, strips other invalid chars.
    Preserves case and Unicode alphanumeric (Cyrillic, etc.).
    """
    tag = tag.strip().replace(" ", "-")
    tag = re.sub(r"[^a-zA-Z0-9_\-:.\u0400-\u04FF]", "", tag)
    return tag or "unknown"


def _sanitize_tag(text: str) -> str:
    """Convert arbitrary text into a valid tag (ASCII alphanumeric + hyphen)."""
    lowered = text.lower()
    ascii_text = "".join(_CYRILLIC_MAP.get(ch, ch) for ch in lowered)
    ascii_text = re.sub(r"[^a-z0-9-]", "", ascii_text)
    return ascii_text or "unknown"


def _check_duplicates(query: str, tags: list[str] | None = None, limit: int = 3) -> list[dict]:
    """Search brain for existing similar records before storing."""
    results = []
    try:
        brain_hits = _get_brain().search(query=query, tags=tags, limit=limit)
        if brain_hits:
            results = [
                {
                    "id": r.id,
                    "content": r.content[:150],
                    "tags": list(r.tags) if r.tags else [],
                }
                for r in brain_hits
            ]
    except Exception:
        pass

    return results


# Tools that depend on network/infrastructure — only these trip the circuit breaker
_NETWORK_TOOLS = frozenset({"web_search", "http_get", "code_execution"})
