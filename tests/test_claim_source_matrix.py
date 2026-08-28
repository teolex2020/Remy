from datetime import datetime, timezone

from remy.core.claim_source_matrix import (
    build_claim_source_matrix,
    evaluate_claim_source_matrix,
    matrix_to_markdown,
    sources_from_session_log,
)
from remy.core.agent import _build_claim_source_matrix_event


def _source(url: str, content: str, title: str = "") -> dict:
    return {"url": url, "title": title, "content": content}


def test_explicit_source_with_matching_fetched_content_supports_claim():
    matrix = build_claim_source_matrix(
        [
            {
                "summary": "PostgreSQL uses MVCC for transaction isolation.",
                "source_url": "https://postgresql.org/about/mvcc",
            }
        ],
        [
            _source(
                "https://postgresql.org/about/mvcc",
                "PostgreSQL uses multiversion concurrency control (MVCC) "
                "to implement transaction isolation.",
            )
        ],
    )

    assert matrix["claim_coverage_rate"] == 1.0
    assert matrix["publication_ready"] is True
    assert matrix["rows"][0]["status"] == "supported"
    assert matrix["rows"][0]["relations"][0]["relation"] == "supports"


def test_url_without_readable_material_is_partial_not_supported():
    matrix = build_claim_source_matrix(
        [{"summary": "The product costs $20.", "source_url": "https://vendor.test/pricing"}],
        [{"url": "https://vendor.test/pricing", "title": "Pricing", "content": ""}],
    )

    assert matrix["partial_claims"] == 1
    assert matrix["claim_coverage_rate"] == 0.0
    assert matrix["evidence_coverage_rate"] == 1.0
    assert matrix["publication_ready"] is False
    assert matrix["repair_queries"]


def test_claim_without_source_or_matching_evidence_is_unsupported():
    matrix = build_claim_source_matrix(
        ["Quantum batteries currently dominate the consumer market."],
        [_source("https://example.test/pasta", "A recipe for tomato pasta.")],
    )

    assert matrix["unsupported_claims"] == 1
    assert matrix["evidence_coverage_rate"] == 0.0
    assert matrix["rows"][0]["relations"] == []


def test_explicit_contradiction_overrides_support_status():
    claim = {
        "summary": "The supported release is version 17.",
        "source_url": "https://vendor.test/releases",
    }
    matrix = build_claim_source_matrix(
        [claim],
        [_source("https://vendor.test/releases", "The supported release is version 17.")],
        contradictions=[
            {
                "id": "c1",
                "source_a": "https://vendor.test/releases",
                "source_b": "https://mirror.test/releases",
                "claim_a": "The supported release is version 17.",
                "claim_b": "The supported release is version 16.",
            }
        ],
    )

    assert matrix["conflicting_claims"] == 1
    assert matrix["rows"][0]["status"] == "conflict"
    assert "resolve conflicting evidence" in matrix["rows"][0]["repair_query"]


def test_two_independent_supporting_domains_mark_claim_corroborated():
    claim = "WebAuthn passkeys use public-key cryptography for authentication."
    matrix = build_claim_source_matrix(
        [claim],
        [
            _source(
                "https://w3.org/webauthn",
                "WebAuthn passkeys use public key cryptography for authentication. "
                "The specification defines relying parties, authenticators, "
                "credential creation, challenge verification, and signature counters.",
            ),
            _source(
                "https://developer.mozilla.org/passkeys",
                "WebAuthn passkeys use public-key cryptography for authentication. "
                "The browser guide explains conditional mediation, user activation, "
                "credential discovery, account recovery, and interface compatibility.",
            ),
        ],
    )

    assert matrix["rows"][0]["corroborated"] is True
    assert matrix["rows"][0]["independent_support_roots"] == 2
    assert matrix["rows"][0]["false_corroboration"] is False
    assert matrix["corroboration_rate"] == 1.0


def test_cross_domain_derivatives_do_not_create_claim_corroboration():
    claim = "The database protocol tolerates one replica failure."
    origin = "https://origin.test/protocol-study"
    matrix = build_claim_source_matrix(
        [claim],
        [
            {
                "url": "https://publisher-one.test/report",
                "original_url": origin,
                "content": (
                    "The database protocol tolerates one replica failure. "
                    "Quorum intersection safety proof and replicated state machine "
                    "analysis cover the formal model and node assumptions."
                ),
            },
            {
                "url": "https://publisher-two.test/article",
                "original_source_url": origin,
                "content": (
                    "The database protocol tolerates one replica failure. "
                    "Production telemetry, network partition experiments, recovery "
                    "timelines, and operational measurements summarize the result."
                ),
            },
        ],
    )

    row = matrix["rows"][0]
    assert row["status"] == "supported"
    assert row["independent_support_domains"] == 2
    assert row["independent_support_roots"] == 1
    assert row["corroborated"] is False
    assert row["false_corroboration"] is True
    assert row["provenance_status"] == "false_corroboration"
    assert "not derived" in row["repair_query"]
    assert matrix["false_corroborated_claims"] == 1
    assert matrix["provenance_ready"] is False
    assert matrix["publication_ready"] is False


def test_stale_source_cannot_make_current_claim_publication_ready():
    now = datetime(2026, 8, 22, tzinfo=timezone.utc)
    matrix = build_claim_source_matrix(
        [
            {
                "summary": "The current service price is $20 per month.",
                "source_url": "https://vendor.test/pricing",
            }
        ],
        [
            {
                "url": "https://vendor.test/pricing",
                "date": "2026-06-01",
                "content": "The current service price is $20 per month.",
            }
        ],
        now=now,
    )

    row = matrix["rows"][0]
    assert row["status"] == "supported"
    assert row["time_sensitive"] is True
    assert row["temporal_status"] == "stale"
    assert row["temporal_ready"] is False
    assert row["relations"][0]["temporally_valid"] is False
    assert matrix["stale_claims"] == 1
    assert matrix["temporal_ready"] is False
    assert matrix["publication_ready"] is False
    assert "within 7 days" in matrix["repair_queries"][0]
    assert "| stale |" in matrix_to_markdown(matrix)


def test_recent_dated_source_validates_current_claim():
    now = datetime(2026, 8, 22, tzinfo=timezone.utc)
    matrix = build_claim_source_matrix(
        [
            {
                "summary": "The current service price is $20 per month.",
                "source_url": "https://vendor.test/pricing",
            }
        ],
        [
            {
                "url": "https://vendor.test/pricing",
                "published_at": "2026-08-21T09:00:00Z",
                "content": "The current service price is $20 per month.",
            }
        ],
        now=now,
    )

    row = matrix["rows"][0]
    assert row["temporal_status"] == "fresh"
    assert row["fresh_support_count"] == 1
    assert row["temporally_valid_support_roots"] == 1
    assert matrix["temporal_coverage_rate"] == 1.0
    assert matrix["publication_ready"] is True


def test_historical_price_claim_accepts_old_evidence_date():
    matrix = build_claim_source_matrix(
        [
            {
                "summary": "As of 2024, the service price was $20 per month.",
                "source_url": "https://vendor.test/2024-pricing",
            }
        ],
        [
            {
                "url": "https://vendor.test/2024-pricing",
                "date": "2024-06-01",
                "content": "As of 2024, the service price was $20 per month.",
            }
        ],
        now=datetime(2026, 8, 22, tzinfo=timezone.utc),
    )

    row = matrix["rows"][0]
    assert row["historically_scoped"] is True
    assert row["temporal_status"] == "historical"
    assert row["temporal_ready"] is True
    assert matrix["publication_ready"] is True


def test_temporal_supersession_retires_old_claim_and_preserves_history():
    old_claim = "The current service price is $20 per month."
    new_claim = "The current service price is $25 per month."
    matrix = build_claim_source_matrix(
        [
            {"summary": old_claim, "source_url": "https://old.test/pricing"},
            {"summary": new_claim, "source_url": "https://new.test/pricing"},
        ],
        [
            {
                "url": "https://old.test/pricing",
                "published_at": "2026-06-01",
                "content": old_claim + " Archived plans and legacy quota details.",
            },
            {
                "url": "https://new.test/pricing",
                "published_at": "2026-08-21",
                "content": new_claim + " Active checkout terms and updated limits.",
            },
        ],
        contradictions=[
            {
                "id": "price-change",
                "claim_a": old_claim,
                "claim_b": new_claim,
                "source_a": "https://old.test/pricing",
                "source_b": "https://new.test/pricing",
            }
        ],
        now=datetime(2026, 8, 22, tzinfo=timezone.utc),
    )

    old_row, new_row = matrix["rows"]
    assert old_row["status"] == "superseded"
    assert old_row["supersession_role"] == "old"
    assert old_row["superseded_by"]["new_claim"] == new_claim
    assert new_row["status"] == "supported"
    assert new_row["supersession_role"] == "new"
    assert matrix["claim_count"] == 2
    assert matrix["active_claim_count"] == 1
    assert matrix["superseded_claims"] == 1
    assert matrix["conflicting_claims"] == 0
    assert matrix["resolved_temporal_conflicts"] == 1
    assert matrix["unresolved_contradictions"] == 0
    assert matrix["publication_ready"] is True
    assert matrix["contradiction_resolutions"][0]["supersession"][
        "history_preserved"
    ] is True


def test_newer_nonmutable_research_disagreement_remains_conflict():
    old_claim = "The experiment reports accuracy of 71 percent."
    new_claim = "The experiment reports accuracy of 82 percent."
    matrix = build_claim_source_matrix(
        [
            {"summary": old_claim, "source_url": "https://paper-one.test/result"},
            {"summary": new_claim, "source_url": "https://paper-two.test/result"},
        ],
        [
            {
                "url": "https://paper-one.test/result",
                "published_at": "2026-05-01",
                "content": old_claim + " Dataset alpha and evaluation protocol.",
            },
            {
                "url": "https://paper-two.test/result",
                "published_at": "2026-08-20",
                "content": new_claim + " Dataset beta and alternative evaluation protocol.",
            },
        ],
        contradictions=[
            {
                "id": "paper-conflict",
                "claim_a": old_claim,
                "claim_b": new_claim,
                "source_a": "https://paper-one.test/result",
                "source_b": "https://paper-two.test/result",
            }
        ],
        now=datetime(2026, 8, 22, tzinfo=timezone.utc),
    )

    assert matrix["resolved_temporal_conflicts"] == 0
    assert matrix["unresolved_contradictions"] == 1
    assert matrix["conflicting_claims"] == 2
    assert {row["status"] for row in matrix["rows"]} == {"conflict"}
    assert matrix["publication_ready"] is False


def test_identity_mismatch_cannot_support_explicit_claim():
    matrix = build_claim_source_matrix(
        [{"summary": "Claimed paper result", "source_url": "https://arxiv.org/abs/1"}],
        [
            {
                "url": "https://arxiv.org/abs/1",
                "title": "Different paper",
                "content": "Claimed paper result",
                "evidence_packet": {"ok": False, "has_mismatch": True},
            }
        ],
    )

    assert matrix["rows"][0]["status"] == "partial"
    assert matrix["rows"][0]["relations"][0]["relation"] == "identity_mismatch"


def test_sources_from_session_log_reads_full_fetch_payload_and_packet():
    sources = sources_from_session_log(
        [
            {
                "type": "tool_call",
                "tool": "extract_content",
                "args_full": {"url": "https://example.test/page"},
                "result_full": '{"url":"https://example.test/page","title":"Page",'
                '"content":"Fetched evidence body long enough to use.",'
                '"evidence_packet":{"ok":true,"has_mismatch":false}}',
            }
        ]
    )

    assert sources[0]["url"] == "https://example.test/page"
    assert sources[0]["content"].startswith("Fetched evidence")
    assert sources[0]["identity_mismatch"] is False


def test_markdown_renderer_exposes_status_claim_and_source():
    matrix = build_claim_source_matrix(
        [{"summary": "SQLite WAL allows concurrent readers.", "source_url": "https://sqlite.org/wal"}],
        [_source("https://sqlite.org/wal", "SQLite WAL allows concurrent readers and writers.")],
    )

    markdown = matrix_to_markdown(matrix)

    assert "Claim–source matrix" in markdown
    assert "supported" in markdown
    assert "[sqlite.org](https://sqlite.org/wal)" in markdown


def test_agent_diagnostic_event_binds_response_claim_to_turn_fetch():
    report = type(
        "Report",
        (),
        {
            "claim_details": [
                type(
                    "Claim",
                    (),
                    {
                        "text": "SQLite WAL allows concurrent readers.",
                        "claim_class": "observed_fact",
                    },
                )()
            ]
        },
    )()
    event = _build_claim_source_matrix_event(
        report,
        [
            {
                "type": "tool_call",
                "tool": "extract_content",
                "args_full": {"url": "https://sqlite.org/wal"},
                "result_full": {
                    "url": "https://sqlite.org/wal",
                    "title": "Write-Ahead Logging",
                    "content": "SQLite WAL allows concurrent readers and writers.",
                },
            }
        ],
    )

    assert event is not None
    assert event["type"] == "claim_source_matrix"
    assert event["supported_claims"] == 1
    assert event["rows"][0]["claim_class"] == "observed_fact"


def test_matrix_evaluator_measures_status_binding_and_false_supports():
    matrix = build_claim_source_matrix(
        [
            {"summary": "SQLite WAL allows concurrent readers.", "source_url": "https://sqlite.org/wal"},
            "Quantum batteries dominate the consumer market.",
        ],
        [_source("https://sqlite.org/wal", "SQLite WAL allows concurrent readers.")],
    )

    metrics = evaluate_claim_source_matrix(
        matrix,
        [
            {"status": "supported", "supporting_urls": ["https://sqlite.org/wal"]},
            {"status": "unsupported", "supporting_urls": []},
        ],
    )

    assert metrics["status_accuracy"] == 1.0
    assert metrics["support_url_precision"] == 1.0
    assert metrics["support_url_recall"] == 1.0
    assert metrics["unsupported_recall"] == 1.0
    assert metrics["false_support_rate"] == 0.0
