from datetime import datetime, timezone

from remy.core.claim_temporal_evidence import (
    assess_source_temporality,
    classify_claim_temporality,
    evaluate_claim_temporal_evidence,
    parse_source_datetime,
    source_datetime,
)


NOW = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)


def test_current_price_is_high_volatility():
    result = classify_claim_temporality("The current price is $20.")

    assert result["time_sensitive"] is True
    assert result["volatility"] == "high"
    assert result["ttl_days"] == 7


def test_supported_release_is_medium_volatility():
    result = classify_claim_temporality("Version 17 is the supported release.")

    assert result["time_sensitive"] is True
    assert result["volatility"] == "medium"
    assert result["ttl_days"] == 90


def test_concurrent_does_not_trigger_current_signal():
    result = classify_claim_temporality(
        "SQLite WAL allows concurrent readers while a writer appends."
    )

    assert result["time_sensitive"] is False


def test_historical_as_of_claim_does_not_require_current_source():
    result = classify_claim_temporality("As of 2024, the service price was $20.")

    assert result["time_sensitive"] is False
    assert result["historically_scoped"] is True
    assert result["ttl_days"] is None


def test_ukrainian_current_claim_is_detected():
    result = classify_claim_temporality("Поточна ціна сервісу зараз становить $20.")

    assert result["time_sensitive"] is True
    assert result["volatility"] == "high"


def test_iso_rfc_and_timestamp_dates_are_parsed():
    assert parse_source_datetime("2026-08-20T10:30:00Z").day == 20
    assert parse_source_datetime("Thu, 20 Aug 2026 10:30:00 GMT").day == 20
    assert parse_source_datetime(1787221800).tzinfo == timezone.utc


def test_nested_metadata_date_is_extracted():
    parsed, field = source_datetime(
        {"url": "https://example.test", "metadata": {"datePublished": "ignored", "published_at": "2026-08-20"}}
    )

    assert parsed == datetime(2026, 8, 20, tzinfo=timezone.utc)
    assert field == "published_at"


def test_recent_source_validates_current_claim():
    result = assess_source_temporality(
        "The current price is $20.",
        {"url": "https://vendor.test", "published_at": "2026-08-20"},
        now=NOW,
    )

    assert result["status"] == "fresh"
    assert result["temporally_valid"] is True
    assert result["age_days"] == 2.5


def test_old_source_cannot_validate_current_claim():
    result = assess_source_temporality(
        "The latest version is 5.0.",
        {"url": "https://vendor.test", "date": "2026-07-01"},
        now=NOW,
    )

    assert result["status"] == "stale"
    assert result["temporally_valid"] is False


def test_undated_source_cannot_validate_current_claim():
    result = evaluate_claim_temporal_evidence(
        "The current status is operational.",
        [{"url": "https://status.test"}],
        now=NOW,
    )

    assert result["status"] == "undated"
    assert result["temporal_ready"] is False
    assert "within 7 days" in result["repair_query"]


def test_one_fresh_source_is_enough_when_another_is_stale():
    result = evaluate_claim_temporal_evidence(
        "The current price is $20.",
        [
            {"url": "https://old.test", "date": "2025-01-01"},
            {"url": "https://fresh.test", "date": "2026-08-21"},
        ],
        now=NOW,
    )

    assert result["status"] == "fresh"
    assert result["temporal_ready"] is True
    assert result["fresh_source_count"] == 1
