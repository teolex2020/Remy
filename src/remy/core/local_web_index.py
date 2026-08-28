"""Durable local full-text index for web evidence already read by Remy."""

from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from remy.config.settings import settings
from remy.core.search_relevance import query_terms


_SCHEMA_LOCK = threading.Lock()
_WORD_RE = re.compile(r"[\w]+", re.UNICODE)
_SENSITIVE_QUERY_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "key",
    "password",
    "session",
    "signature",
    "token",
}


def _now_iso(timestamp: float | None = None) -> str:
    return datetime.fromtimestamp(timestamp or time.time(), timezone.utc).isoformat()


def _safe_url(value: str) -> str:
    try:
        parsed = urlsplit((value or "").strip())
    except ValueError:
        return ""
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return ""
    if parsed.username or parsed.password:
        return ""
    host = parsed.hostname.lower()
    try:
        port = parsed.port
    except ValueError:
        return ""
    if port:
        default_port = (parsed.scheme.lower() == "http" and port == 80) or (
            parsed.scheme.lower() == "https" and port == 443
        )
        if not default_port:
            host = f"{host}:{port}"
    query = urlencode(
        [
            (key, item)
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            if key.casefold() not in _SENSITIVE_QUERY_KEYS
            and not key.casefold().startswith("utm_")
        ],
        doseq=True,
    )
    return urlunsplit(
        (parsed.scheme.lower(), host, parsed.path or "/", query, "")
    )


def _simhash(value: str) -> str:
    """Return a stable 64-bit near-duplicate fingerprint for readable text."""
    words = _WORD_RE.findall((value or "")[:100_000].casefold())[:20_000]
    if not words:
        return ""
    features = (
        (" ".join(words[index:index + 4]) for index in range(len(words) - 3))
        if len(words) >= 4
        else iter(words)
    )
    vector = [0] * 64
    for feature in features:
        digest = int.from_bytes(
            hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        for bit in range(64):
            vector[bit] += 1 if digest & (1 << bit) else -1
    fingerprint = sum(1 << bit for bit, weight in enumerate(vector) if weight >= 0)
    return f"{fingerprint:016x}"


def _hamming_distance(left: str, right: str) -> int:
    if len(left) != 16 or len(right) != 16:
        return 64
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError:
        return 64


def _edit_distance(left: str, right: str, *, cutoff: int) -> int:
    """Bounded Unicode Levenshtein distance; values above cutoff collapse."""
    if abs(len(left) - len(right)) > cutoff:
        return cutoff + 1
    previous = list(range(len(right) + 1))
    for row_index, left_char in enumerate(left, start=1):
        current = [row_index]
        row_minimum = row_index
        for column_index, right_char in enumerate(right, start=1):
            value = min(
                current[-1] + 1,
                previous[column_index] + 1,
                previous[column_index - 1] + (left_char != right_char),
            )
            current.append(value)
            row_minimum = min(row_minimum, value)
        if row_minimum > cutoff:
            return cutoff + 1
        previous = current
    return previous[-1]


class LocalWebIndex:
    """SQLite FTS5 corpus with a LIKE fallback for unusual SQLite builds."""

    def __init__(self, path: Path | str | None = None, *, max_documents: int = 5_000):
        self.path = Path(path) if path else settings.DATA_DIR / "web_index.sqlite3"
        self.max_documents = max(100, int(max_documents))
        self._fts_enabled = False
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _ensure_schema(self) -> None:
        with _SCHEMA_LOCK, self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS web_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL,
                    site TEXT NOT NULL DEFAULT '',
                    author TEXT NOT NULL DEFAULT '',
                    published_date TEXT NOT NULL DEFAULT '',
                    quality_score REAL NOT NULL DEFAULT 0,
                    extraction_method TEXT NOT NULL DEFAULT '',
                    content_hash TEXT NOT NULL,
                    content_fingerprint TEXT NOT NULL DEFAULT '',
                    fetched_at REAL NOT NULL,
                    last_accessed_at REAL NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_web_documents_fetched
                    ON web_documents(fetched_at DESC);
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(web_documents)"
                ).fetchall()
            }
            if "content_fingerprint" not in columns:
                connection.execute(
                    "ALTER TABLE web_documents ADD COLUMN "
                    "content_fingerprint TEXT NOT NULL DEFAULT ''"
                )
            try:
                connection.executescript(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS web_documents_fts USING fts5(
                        title, content, url UNINDEXED,
                        content='web_documents', content_rowid='id'
                    );
                    CREATE TRIGGER IF NOT EXISTS web_documents_ai
                    AFTER INSERT ON web_documents BEGIN
                        INSERT INTO web_documents_fts(rowid, title, content, url)
                        VALUES (new.id, new.title, new.content, new.url);
                    END;
                    CREATE TRIGGER IF NOT EXISTS web_documents_ad
                    AFTER DELETE ON web_documents BEGIN
                        INSERT INTO web_documents_fts(
                            web_documents_fts, rowid, title, content, url
                        ) VALUES ('delete', old.id, old.title, old.content, old.url);
                    END;
                    CREATE TRIGGER IF NOT EXISTS web_documents_au
                    AFTER UPDATE ON web_documents BEGIN
                        INSERT INTO web_documents_fts(
                            web_documents_fts, rowid, title, content, url
                        ) VALUES ('delete', old.id, old.title, old.content, old.url);
                        INSERT INTO web_documents_fts(rowid, title, content, url)
                        VALUES (new.id, new.title, new.content, new.url);
                    END;
                    CREATE VIRTUAL TABLE IF NOT EXISTS web_documents_vocab
                    USING fts5vocab(web_documents_fts, 'row');
                    """
                )
                self._fts_enabled = True
            except sqlite3.OperationalError:
                self._fts_enabled = False

    def upsert(
        self,
        *,
        url: str,
        content: str,
        title: str = "",
        site: str = "",
        author: str = "",
        published_date: str = "",
        quality_score: float = 0.0,
        extraction_method: str = "",
    ) -> bool:
        canonical_url = _safe_url(url)
        clean_content = (content or "").strip()[:100_000]
        if not canonical_url or len(clean_content) < 80:
            return False
        timestamp = time.time()
        digest = hashlib.sha256(clean_content.encode("utf-8")).hexdigest()
        fingerprint = _simhash(clean_content)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO web_documents(
                    url, title, content, site, author, published_date,
                    quality_score, extraction_method, content_hash,
                    content_fingerprint, fetched_at, last_accessed_at, access_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(url) DO UPDATE SET
                    title=excluded.title,
                    content=excluded.content,
                    site=excluded.site,
                    author=excluded.author,
                    published_date=excluded.published_date,
                    quality_score=excluded.quality_score,
                    extraction_method=excluded.extraction_method,
                    content_hash=excluded.content_hash,
                    content_fingerprint=excluded.content_fingerprint,
                    fetched_at=excluded.fetched_at,
                    last_accessed_at=excluded.last_accessed_at,
                    access_count=web_documents.access_count + 1
                """,
                (
                    canonical_url,
                    (title or "").strip()[:1_000],
                    clean_content,
                    (site or "").strip()[:500],
                    (author or "").strip()[:500],
                    (published_date or "").strip()[:100],
                    max(0.0, min(float(quality_score or 0.0), 1.0)),
                    (extraction_method or "").strip()[:100],
                    digest,
                    fingerprint,
                    timestamp,
                    timestamp,
                ),
            )
            count = int(
                connection.execute("SELECT COUNT(*) FROM web_documents").fetchone()[0]
            )
            excess = count - self.max_documents
            if excess > 0:
                connection.execute(
                    """
                    DELETE FROM web_documents WHERE id IN (
                        SELECT id FROM web_documents
                        ORDER BY last_accessed_at ASC LIMIT ?
                    )
                    """,
                    (excess,),
                )
        return True

    def get(self, url: str, *, max_age_seconds: int) -> dict[str, Any] | None:
        canonical_url = _safe_url(url)
        if not canonical_url:
            return None
        cutoff = time.time() - max(0, int(max_age_seconds))
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM web_documents WHERE url=? AND fetched_at>=?",
                (canonical_url, cutoff),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE web_documents
                SET last_accessed_at=?, access_count=access_count + 1
                WHERE id=?
                """,
                (time.time(), row["id"]),
            )
        return self._decode(row)

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        max_age_seconds: int = 90 * 24 * 60 * 60,
    ) -> list[dict[str, Any]]:
        terms = query_terms(query)
        if not terms:
            return []
        bounded_limit = max(1, min(int(limit), 50))
        cutoff = time.time() - max(0, int(max_age_seconds))
        corrections: dict[str, str] = {}
        with self._connect() as connection:
            if self._fts_enabled:
                corrections = self._query_corrections(connection, terms)
                expanded_terms = list(
                    dict.fromkeys([*terms, *corrections.values()])
                )
                match_query = " OR ".join(f'"{term}"*' for term in expanded_terms)
                raw_limit = min(max(bounded_limit * 4, bounded_limit), 200)
                rows = connection.execute(
                    """
                    SELECT d.*, bm25(web_documents_fts, 4.0, 1.0) AS local_rank,
                           snippet(web_documents_fts, 1, '', '', ' … ', 32) AS snippet
                    FROM web_documents_fts
                    JOIN web_documents d ON d.id=web_documents_fts.rowid
                    WHERE web_documents_fts MATCH ? AND d.fetched_at>=?
                    ORDER BY local_rank, d.fetched_at DESC LIMIT ?
                    """,
                    (match_query, cutoff, raw_limit),
                ).fetchall()
            else:
                clauses = " OR ".join("(title LIKE ? OR content LIKE ?)" for _ in terms)
                like_args = [value for term in terms for value in (f"%{term}%", f"%{term}%")]
                rows = connection.execute(
                    f"""
                    SELECT *, 0.0 AS local_rank, substr(content, 1, 500) AS snippet
                    FROM web_documents WHERE ({clauses}) AND fetched_at>=?
                    ORDER BY fetched_at DESC LIMIT ?
                    """,
                    (*like_args, cutoff, bounded_limit),
                ).fetchall()
        decoded = [self._decode(row) for row in rows]
        selected: list[dict[str, Any]] = []
        for candidate in decoded:
            duplicate_index = next(
                (
                    index
                    for index, existing in enumerate(selected)
                    if candidate.get("content_hash") == existing.get("content_hash")
                    or _hamming_distance(
                        str(candidate.get("content_fingerprint") or ""),
                        str(existing.get("content_fingerprint") or ""),
                    ) <= 4
                ),
                None,
            )
            if duplicate_index is None:
                candidate["duplicate_count"] = 0
                selected.append(candidate)
            else:
                existing = selected[duplicate_index]
                duplicate_count = int(existing.get("duplicate_count") or 0) + 1
                candidate_quality = (
                    float(candidate.get("quality_score") or 0.0),
                    float(candidate.get("fetched_at") or 0.0),
                )
                existing_quality = (
                    float(existing.get("quality_score") or 0.0),
                    float(existing.get("fetched_at") or 0.0),
                )
                if candidate_quality > existing_quality:
                    candidate["duplicate_count"] = duplicate_count
                    selected[duplicate_index] = candidate
                else:
                    existing["duplicate_count"] = duplicate_count
        if corrections:
            for candidate in selected:
                candidate["query_corrections"] = dict(corrections)
        return selected[:bounded_limit]

    @staticmethod
    def _query_corrections(
        connection: sqlite3.Connection,
        terms: list[str],
    ) -> dict[str, str]:
        corrections: dict[str, str] = {}
        for term in terms:
            if len(term) < 4:
                continue
            exact = connection.execute(
                "SELECT 1 FROM web_documents_vocab WHERE term=? LIMIT 1",
                (term,),
            ).fetchone()
            if exact:
                continue
            cutoff = 1 if len(term) <= 6 else 2
            lower_bound = term[0]
            upper_bound = chr(ord(term[0]) + 1)
            vocabulary = connection.execute(
                """
                SELECT term, doc FROM web_documents_vocab
                WHERE term>=? AND term<? AND length(term) BETWEEN ? AND ?
                ORDER BY doc DESC LIMIT 500
                """,
                (lower_bound, upper_bound, len(term) - cutoff, len(term) + cutoff),
            ).fetchall()
            ranked = []
            for row in vocabulary:
                candidate = str(row["term"])
                distance = _edit_distance(term, candidate, cutoff=cutoff)
                if distance <= cutoff:
                    ranked.append((distance, -int(row["doc"]), candidate))
            if ranked:
                corrections[term] = min(ranked)[2]
        return corrections

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["fetched_at_iso"] = _now_iso(float(item["fetched_at"]))
        item["age_seconds"] = max(0, round(time.time() - float(item["fetched_at"])))
        return item


@lru_cache(maxsize=8)
def _cached_local_web_index(path: str, max_documents: int) -> LocalWebIndex:
    return LocalWebIndex(Path(path), max_documents=max_documents)


def get_local_web_index() -> LocalWebIndex:
    max_documents = int(getattr(settings, "LOCAL_WEB_INDEX_MAX_DOCUMENTS", 5_000))
    return _cached_local_web_index(
        str(settings.DATA_DIR / "web_index.sqlite3"),
        max_documents,
    )
