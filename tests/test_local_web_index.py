import sqlite3

from remy.core.local_web_index import LocalWebIndex


def test_local_index_round_trip_and_full_text_search(tmp_path):
    index = LocalWebIndex(tmp_path / "web.sqlite3")
    stored = index.upsert(
        url="https://docs.example.test/database?utm_source=test",
        title="Distributed database guide",
        content="Distributed database consistency uses quorum replication. " * 10,
        quality_score=0.82,
        extraction_method="http+trafilatura",
    )

    assert stored is True
    hits = index.search("database consistency")
    assert len(hits) == 1
    assert hits[0]["url"] == "https://docs.example.test/database"
    assert hits[0]["quality_score"] == 0.82

    cached = index.get(
        "https://docs.example.test/database?utm_source=another",
        max_age_seconds=60,
    )
    assert cached is not None
    assert "quorum replication" in cached["content"]


def test_upsert_replaces_content_without_duplicate_search_rows(tmp_path):
    index = LocalWebIndex(tmp_path / "web.sqlite3")
    index.upsert(
        url="https://example.test/page",
        title="Old title",
        content="Old database consistency material. " * 10,
    )
    index.upsert(
        url="https://example.test/page",
        title="New title",
        content="Updated database consistency evidence. " * 10,
    )

    hits = index.search("updated evidence")
    assert len(hits) == 1
    assert hits[0]["title"] == "New title"


def test_sensitive_url_credentials_are_not_persisted(tmp_path):
    index = LocalWebIndex(tmp_path / "web.sqlite3")

    assert index.upsert(
        url="https://user:secret@example.test/private",
        content="Private evidence content. " * 10,
    ) is False

    index.upsert(
        url="https://example.test/page?token=secret&view=full",
        content="Public evidence content. " * 10,
    )
    hits = index.search("public evidence")
    assert hits[0]["url"] == "https://example.test/page?view=full"


def test_document_limit_prunes_least_recently_accessed(tmp_path):
    index = LocalWebIndex(tmp_path / "web.sqlite3", max_documents=100)
    for number in range(105):
        index.upsert(
            url=f"https://example{number}.test/page",
            content=f"Unique searchable evidence document number {number}. " * 4,
        )

    with index._connect() as connection:
        count = connection.execute("SELECT COUNT(*) FROM web_documents").fetchone()[0]
    assert count == 100


def test_exact_content_copies_collapse_to_best_document(tmp_path):
    index = LocalWebIndex(tmp_path / "web.sqlite3")
    shared = "Database consistency quorum replication evidence. " * 20
    index.upsert(
        url="https://weak.example/article",
        title="Weak copy",
        content=shared,
        quality_score=0.4,
    )
    index.upsert(
        url="https://primary.example/report",
        title="Primary report",
        content=shared,
        quality_score=0.9,
    )

    hits = index.search("database consistency")

    assert len(hits) == 1
    assert hits[0]["url"] == "https://primary.example/report"
    assert hits[0]["duplicate_count"] == 1


def test_near_duplicate_content_is_clustered(tmp_path):
    index = LocalWebIndex(tmp_path / "web.sqlite3")
    shared = "Database consistency uses quorum replication and consensus. " * 40
    index.upsert(
        url="https://one.example/report",
        content=shared,
        quality_score=0.8,
    )
    index.upsert(
        url="https://two.example/reprint",
        content="Updated edition. " + shared + " Copyright notice.",
        quality_score=0.7,
    )

    hits = index.search("quorum replication")

    assert len(hits) == 1
    assert hits[0]["duplicate_count"] == 1


def test_local_vocabulary_corrects_typo_without_external_service(tmp_path):
    index = LocalWebIndex(tmp_path / "web.sqlite3")
    index.upsert(
        url="https://example.test/database",
        title="Database architecture",
        content="Database consistency and replication reference. " * 15,
    )

    hits = index.search("databse")

    assert len(hits) == 1
    assert hits[0]["query_corrections"] == {"databse": "database"}


def test_existing_index_schema_is_migrated_for_near_duplicate_fingerprint(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE web_documents (
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
                fetched_at REAL NOT NULL,
                last_accessed_at REAL NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 1
            );
            """
        )

    index = LocalWebIndex(path)

    with index._connect() as connection:
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(web_documents)")
        }
    assert "content_fingerprint" in columns
