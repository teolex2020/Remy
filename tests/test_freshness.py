"""Unit tests for retrieval.freshness (Phase 4)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from remy.core.retrieval.freshness import (
    RevalidationEntry,
    build_revalidation_queue,
    classify_volatility,
    conflict_flag_metadata,
    detect_conflict,
    freshness_metadata,
    is_stale,
    needs_revalidation,
    stale_after,
    supersede_metadata,
    truth_status,
    ttl_days_for,
)


# ── Classification ────────────────────────────────────────────────────────


def test_classify_version_query_is_high():
    assert classify_volatility("langchain latest stable version") == "high"


def test_classify_prices_is_high():
    assert classify_volatility("current LLM pricing per token") == "high"


def test_classify_arxiv_lookup_is_low():
    assert classify_volatility("arxiv 2312.10997 authors") == "low"


def test_classify_api_docs_is_medium():
    assert classify_volatility("asyncio task group api docs") == "medium"


def test_classify_unknown_topic_defaults_to_medium():
    assert classify_volatility("some random question") == "medium"


def test_high_volatility_wins_over_low_signals():
    # Both "latest" (high) and "definition" (low) present — high must win.
    assert classify_volatility("latest definition of agentic AI") == "high"


# ── TTL ───────────────────────────────────────────────────────────────────


def test_ttl_days_per_tier():
    assert ttl_days_for("low") is None
    assert ttl_days_for("medium") == 90
    assert ttl_days_for("high") == 7


def test_low_volatility_is_never_stale():
    old = datetime.now(timezone.utc) - timedelta(days=3650)
    assert not is_stale(old, "low")
    assert stale_after(old, "low") is None


def test_high_volatility_stale_after_7_days():
    old = datetime.now(timezone.utc) - timedelta(days=8)
    assert is_stale(old, "high")
    fresh = datetime.now(timezone.utc) - timedelta(days=1)
    assert not is_stale(fresh, "high")


def test_medium_volatility_stale_after_90_days():
    old = datetime.now(timezone.utc) - timedelta(days=95)
    assert is_stale(old, "medium")
    fresh = datetime.now(timezone.utc) - timedelta(days=30)
    assert not is_stale(fresh, "medium")


def test_freshness_metadata_stamp():
    now = datetime(2026, 4, 13, 12, 0, 0, tzinfo=timezone.utc)
    meta = freshness_metadata("high", now=now)
    assert meta["volatility"] == "high"
    assert meta["ttl_days"] == 7
    assert meta["cached_at"].startswith("2026-04-13")
    assert meta["stale_after"].startswith("2026-04-20")


def test_freshness_metadata_low_has_no_cutoff():
    meta = freshness_metadata("low")
    assert meta["stale_after"] is None


# ── Conflict detection ────────────────────────────────────────────────────


def test_conflict_none_on_empty_prior():
    rep = detect_conflict("langchain 0.4.0 is current", [])
    assert not rep.has_conflict


def test_conflict_detected_on_different_versions():
    prior = [{"id": "rec1", "content": "langchain 0.1.0 is the stable release"}]
    rep = detect_conflict("langchain 0.4.0 is the current stable release", prior)
    assert rep.has_conflict
    assert "rec1" in rep.prior_record_ids
    assert "versions" in rep.diverging_signals
    assert "0.1.0" in rep.diverging_signals["versions"]["old"]
    assert "0.4.0" in rep.diverging_signals["versions"]["new"]


def test_no_conflict_when_signals_overlap():
    # Old says 0.1.0 and 0.2.0; new says 0.2.0 and 0.3.0 — overlap at 0.2.0 -> not disjoint -> no conflict.
    prior = [{"id": "rec1", "content": "versions 0.1.0 and 0.2.0 were released"}]
    rep = detect_conflict("versions 0.2.0 and 0.3.0 are available", prior)
    assert not rep.has_conflict


def test_no_conflict_when_one_side_empty():
    # Old has versions, new has no version tokens at all.
    prior = [{"id": "rec1", "content": "langchain 0.1.0 released"}]
    rep = detect_conflict("some qualitative finding with no numbers at all", prior)
    assert not rep.has_conflict


def test_conflict_report_to_dict_shape():
    prior = [{"id": "rec1", "content": "python 3.11 is current"}]
    rep = detect_conflict("python 3.13 is current", prior)
    d = rep.to_dict()
    assert d["has_conflict"] is True
    assert d["prior_record_ids"] == ["rec1"]
    assert "versions" in d["diverging_signals"]
    # Sorted lists
    assert isinstance(d["diverging_signals"]["versions"]["old"], list)
    assert isinstance(d["diverging_signals"]["versions"]["new"], list)


def test_conflict_across_multiple_prior_records():
    prior = [
        {"id": "rec1", "content": "langchain 0.1.0"},
        {"id": "rec2", "content": "langchain 0.2.0"},
    ]
    rep = detect_conflict("langchain 0.4.0 is current", prior)
    assert rep.has_conflict
    assert set(rep.prior_record_ids) == {"rec1", "rec2"}


# ── Phase 4: truth_status lifecycle ──────────────────────────────────────────


def _meta(volatility="medium", cached_days_ago=0, **extra):
    cached_at = (datetime.now(timezone.utc) - timedelta(days=cached_days_ago)).isoformat()
    return {"volatility": volatility, "cached_at": cached_at, **extra}


def test_truth_status_fresh_within_ttl():
    assert truth_status(_meta("high", cached_days_ago=1)) == "fresh"


def test_truth_status_stale_soft_past_ttl():
    assert truth_status(_meta("high", cached_days_ago=10)) == "stale_soft"


def test_truth_status_stale_hard_past_double_ttl():
    assert truth_status(_meta("high", cached_days_ago=20)) == "stale_hard"


def test_truth_status_no_expiry_for_low_volatility():
    assert truth_status(_meta("low", cached_days_ago=3650)) == "no_expiry"


def test_truth_status_superseded_wins_over_time():
    meta = _meta("high", cached_days_ago=1, superseded_by="rec-new")
    assert truth_status(meta) == "superseded"


def test_truth_status_conflict_unresolved_wins_over_time():
    meta = _meta("high", cached_days_ago=1, unresolved_conflict=True)
    assert truth_status(meta) == "conflict_unresolved"


def test_truth_status_conflict_resolved_flag():
    meta = _meta("high", cached_days_ago=1, conflict_resolved=True)
    assert truth_status(meta) == "conflict_resolved"


def test_truth_status_unknown_when_no_cached_at():
    assert truth_status({"volatility": "medium"}) == "unknown"


def test_truth_status_medium_boundary_90_days():
    assert truth_status(_meta("medium", cached_days_ago=60)) == "fresh"
    assert truth_status(_meta("medium", cached_days_ago=120)) == "stale_soft"
    assert truth_status(_meta("medium", cached_days_ago=200)) == "stale_hard"


# ── Phase 4: needs_revalidation ──────────────────────────────────────────────


def test_needs_revalidation_fresh_false():
    assert needs_revalidation(_meta("high", cached_days_ago=1)) is False


def test_needs_revalidation_stale_soft_true():
    assert needs_revalidation(_meta("high", cached_days_ago=10)) is True


def test_needs_revalidation_stale_hard_true():
    assert needs_revalidation(_meta("high", cached_days_ago=20)) is True


def test_needs_revalidation_conflict_unresolved_true():
    assert needs_revalidation(_meta("high", cached_days_ago=1, unresolved_conflict=True)) is True


def test_needs_revalidation_superseded_false():
    # Superseded records are done — they don't need revalidation anymore.
    assert needs_revalidation(_meta("high", cached_days_ago=20, superseded_by="x")) is False


# ── Phase 4: build_revalidation_queue ────────────────────────────────────────


def test_revalidation_queue_filters_fresh_records():
    records = [
        {"id": "a", "metadata": _meta("high", cached_days_ago=1)},
        {"id": "b", "metadata": _meta("high", cached_days_ago=10)},
    ]
    q = build_revalidation_queue(records)
    assert [e.record_id for e in q] == ["b"]


def test_revalidation_queue_prioritises_conflict_first():
    records = [
        {"id": "soft", "metadata": _meta("high", cached_days_ago=10)},
        {"id": "hard", "metadata": _meta("high", cached_days_ago=20)},
        {"id": "conflict", "metadata": _meta("high", cached_days_ago=1, unresolved_conflict=True)},
    ]
    q = build_revalidation_queue(records)
    assert [e.record_id for e in q] == ["conflict", "hard", "soft"]


def test_revalidation_queue_entries_have_reason():
    records = [{"id": "r1", "metadata": _meta("high", cached_days_ago=10)}]
    q = build_revalidation_queue(records)
    assert len(q) == 1
    assert q[0].status == "stale_soft"
    assert "TTL" in q[0].reason
    assert q[0].volatility == "high"


def test_revalidation_queue_oldest_first_within_bucket():
    records = [
        {"id": "newer", "metadata": _meta("high", cached_days_ago=10)},
        {"id": "older", "metadata": _meta("high", cached_days_ago=12)},
    ]
    q = build_revalidation_queue(records)
    # Both stale_soft; older cached_at (= smaller ISO) first.
    assert [e.record_id for e in q] == ["older", "newer"]


def test_revalidation_queue_skips_records_missing_id():
    records = [{"id": "", "metadata": _meta("high", cached_days_ago=10)}]
    q = build_revalidation_queue(records)
    assert q == []


def test_revalidation_entry_to_dict_shape():
    entry = RevalidationEntry(
        record_id="r",
        topic="t",
        volatility="high",
        status="stale_soft",
        cached_at="2026-04-01T00:00:00+00:00",
        reason="age > TTL for volatility=high",
    )
    d = entry.to_dict()
    assert d["record_id"] == "r"
    assert d["status"] == "stale_soft"
    assert d["volatility"] == "high"


# ── Phase 4: conflict_flag_metadata ──────────────────────────────────────────


def test_conflict_flag_metadata_shape():
    prior = [{"id": "rec1", "content": "python 3.11 is current"}]
    rep = detect_conflict("python 3.13 is current", prior)
    stamp = conflict_flag_metadata(rep)
    assert stamp["unresolved_conflict"] is True
    assert "conflict_flagged_at" in stamp
    assert "versions" in stamp["conflict_diverging"]


def test_conflict_flag_metadata_preserves_diverging_signals():
    prior = [{"id": "rec1", "content": "langchain 0.1.0"}]
    rep = detect_conflict("langchain 0.4.0", prior)
    stamp = conflict_flag_metadata(rep)
    old = stamp["conflict_diverging"]["versions"]["old"]
    new = stamp["conflict_diverging"]["versions"]["new"]
    assert "0.1.0" in old
    assert "0.4.0" in new


# ── Phase 4: supersede_metadata ──────────────────────────────────────────────


def test_supersede_metadata_shape():
    stamp = supersede_metadata("rec-new-id")
    assert stamp["superseded_by"] == "rec-new-id"
    assert "superseded_at" in stamp


def test_supersede_metadata_drives_status():
    # End-to-end: stamp a prior record, then ask truth_status.
    meta = _meta("high", cached_days_ago=1)
    meta.update(supersede_metadata("rec-new"))
    assert truth_status(meta) == "superseded"


def test_conflict_flag_metadata_drives_status():
    # End-to-end: flag a prior record, then truth_status reports unresolved.
    prior = [{"id": "r", "content": "foo 1.0"}]
    rep = detect_conflict("foo 2.0", prior)
    meta = _meta("high", cached_days_ago=1)
    meta.update(conflict_flag_metadata(rep))
    assert truth_status(meta) == "conflict_unresolved"
    assert needs_revalidation(meta) is True
