from datetime import datetime, timezone

from remy.core.temporal_supersession import (
    claim_subject_similarity,
    evaluate_temporal_supersession,
    evaluate_temporal_supersessions,
)


NOW = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)


def _source(url, date, content, **extra):
    return {"url": url, "published_at": date, "content": content, **extra}


def _price_contradiction():
    return {
        "id": "price-change",
        "claim_a": "The current service price is $20 per month.",
        "claim_b": "The current service price is $25 per month.",
        "source_a": "https://old.example/pricing",
        "source_b": "https://new.example/pricing",
        "status": "unresolved",
    }


def _price_sources(**new_extra):
    return [
        _source(
            "https://old.example/pricing",
            "2026-06-01",
            "The current service price is $20 per month. Archived billing tiers "
            "describe legacy limits and the former monthly subscription.",
        ),
        _source(
            "https://new.example/pricing",
            "2026-08-21",
            "The current service price is $25 per month. Updated checkout terms "
            "describe new quotas, annual discounts, and active billing conditions.",
            **new_extra,
        ),
    ]


def test_subject_similarity_ignores_changed_price_value():
    assert claim_subject_similarity(
        "The current service price is $20 per month.",
        "The current service price is $25 per month.",
    ) == 1.0


def test_newer_fresh_independent_price_evidence_resolves_conflict():
    result = evaluate_temporal_supersession(
        _price_contradiction(), _price_sources(), now=NOW
    )

    assert result["resolution_status"] == "resolved_by_temporal_supersession"
    assert result["status"] == "resolved"
    assert result["supersession"]["old_claim"].endswith("$20 per month.")
    assert result["supersession"]["new_claim"].endswith("$25 per month.")
    assert result["supersession"]["effective_at"].startswith("2026-08-21")
    assert result["supersession"]["history_preserved"] is True


def test_side_order_does_not_control_which_claim_wins():
    contradiction = _price_contradiction()
    contradiction.update(
        {
            "claim_a": contradiction["claim_b"],
            "claim_b": contradiction["claim_a"],
            "source_a": contradiction["source_b"],
            "source_b": contradiction["source_a"],
        }
    )
    result = evaluate_temporal_supersession(contradiction, _price_sources(), now=NOW)

    assert result["supersession"]["new_source_url"] == "https://new.example/pricing"


def test_missing_or_equal_dates_leave_conflict_unresolved():
    missing = _price_sources()
    missing[1].pop("published_at")
    missing_result = evaluate_temporal_supersession(
        _price_contradiction(), missing, now=NOW
    )
    equal = _price_sources()
    equal[0]["published_at"] = equal[1]["published_at"]
    equal_result = evaluate_temporal_supersession(
        _price_contradiction(), equal, now=NOW
    )

    assert missing_result["resolution_reason"] == "source_dates_missing_or_equal"
    assert equal_result["resolution_reason"] == "source_dates_missing_or_equal"


def test_newer_scientific_claim_does_not_auto_supersede():
    contradiction = {
        "claim_a": "The experiment reports accuracy of 71 percent.",
        "claim_b": "The experiment reports accuracy of 82 percent.",
        "source_a": "https://paper-one.example/result",
        "source_b": "https://paper-two.example/result",
    }
    sources = [
        _source(
            contradiction["source_a"],
            "2026-05-01",
            contradiction["claim_a"] + " Dataset alpha and evaluation protocol details.",
        ),
        _source(
            contradiction["source_b"],
            "2026-08-20",
            contradiction["claim_b"] + " Dataset beta and alternative evaluation details.",
        ),
    ]

    result = evaluate_temporal_supersession(contradiction, sources, now=NOW)

    assert result["resolution_status"] == "unresolved"
    assert "mutable_semantics" in result["failed_checks"]


def test_mutable_signal_does_not_match_inside_unrelated_word():
    contradiction = {
        "claim_a": "The costume exhibit contains 20 pieces.",
        "claim_b": "The costume exhibit contains 25 pieces.",
        "source_a": "https://museum-one.example/exhibit",
        "source_b": "https://museum-two.example/exhibit",
    }
    sources = [
        _source(
            contradiction["source_a"],
            "2026-05-01",
            contradiction["claim_a"] + " Archived exhibition catalogue and curator notes.",
        ),
        _source(
            contradiction["source_b"],
            "2026-08-20",
            contradiction["claim_b"] + " Updated exhibition catalogue and curator notes.",
        ),
    ]

    result = evaluate_temporal_supersession(contradiction, sources, now=NOW)

    assert result["resolution_status"] == "unresolved"
    assert "mutable_semantics" in result["failed_checks"]


def test_newer_weaker_mirror_cannot_supersede_primary_source():
    sources = _price_sources(
        evidence_packet={"source_class": "mirror", "ok": True}
    )
    sources[0]["url"] = "https://docs.python.org/3/reference/pricing.html"
    contradiction = _price_contradiction()
    contradiction["source_a"] = sources[0]["url"]

    result = evaluate_temporal_supersession(contradiction, sources, now=NOW)

    assert result["resolution_status"] == "unresolved"
    assert "authority_not_weaker" in result["failed_checks"]


def test_derivative_with_same_origin_cannot_supersede():
    origin = "https://origin.example/pricing"
    sources = _price_sources()
    sources[0]["original_url"] = origin
    sources[1]["original_url"] = origin

    result = evaluate_temporal_supersession(
        _price_contradiction(), sources, now=NOW
    )

    assert result["resolution_status"] == "unresolved"
    assert "independent_roots" in result["failed_checks"]


def test_newest_source_must_be_fresh_for_current_claim():
    sources = _price_sources()
    sources[0]["published_at"] = "2026-01-01"
    sources[1]["published_at"] = "2026-06-01"

    result = evaluate_temporal_supersession(
        _price_contradiction(), sources, now=NOW
    )

    assert result["resolution_status"] == "unresolved"
    assert "new_source_fresh" in result["failed_checks"]


def test_already_resolved_contradiction_is_not_rewritten():
    contradiction = {**_price_contradiction(), "status": "resolved"}
    result = evaluate_temporal_supersession(contradiction, _price_sources(), now=NOW)

    assert result["resolution_status"] == "already_resolved"
    assert "supersession" not in result


def test_batch_evaluation_preserves_all_contradictions():
    results = evaluate_temporal_supersessions(
        [_price_contradiction(), {"id": "missing"}],
        _price_sources(),
        now=NOW,
    )

    assert len(results) == 2
    assert results[0]["resolution_status"] == "resolved_by_temporal_supersession"
    assert results[1]["resolution_status"] == "unresolved"
