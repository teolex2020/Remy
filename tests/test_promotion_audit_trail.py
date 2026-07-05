"""
Phase 3 Step 4 — Promotion Audit Trail.

Step 2 closed recall leak; Step 3 closed connect/graph leak. The gates are
real but silent — you see *that* a record didn't promote, not *why*. Step 4
attaches one structured event per block so drift on the admission/promotion
policy becomes visible.

Test contract:

  A. `block_reason` enumerator — each of 8 distinct signals maps to its own
     stable reason code, and the ordering matches `_is_factual_forbidden`
     (first matching signal wins; downstream counts stay clean).

  B. `PromotionAuditLog` — ring buffer stores events, `get_recent` filters,
     `clear` drops them, JSONL sink is optional, emission never raises.

  C. Recall surface integration — `_apply_factual_recall_filter` emits one
     event per blocked record, with `surface="recall_primary"` and the
     correct reason code.

  D. Connect surface integration — `gated_connect` emits events for each
     blocked endpoint (strictest-gate semantics: either/both endpoints may
     contribute reasons) and records `partner_id` in `extra` so analysis
     can see which edge was attempted.

Behavioral only. No exact-text assertions on reasons beyond the stable
constants.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from remy.core.agent_tools import _apply_factual_recall_filter, gated_connect
from remy.core.promotion_audit import (
    REASON_FORBIDDEN_ADMISSION_CLASS,
    REASON_FORBIDDEN_TAG,
    REASON_MISSING_ENDPOINT,
    REASON_REQUIRES_PROMOTION,
    REASON_SUPERSEDED_BY,
    REASON_TRUTH_STATUS_STALE_HARD,
    REASON_UNRESOLVED_CONFLICT_FLAG,
    SURFACE_CONNECT,
    SURFACE_RECALL,
    PromotionAuditLog,
    block_reason,
    get_promotion_audit_log,
    record_block,
    reset_promotion_audit_log,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _fresh_audit_log(tmp_path, monkeypatch):
    """Give every test a clean singleton with a throwaway sink path."""
    reset_promotion_audit_log()
    # Force sink into tmp_path so tests don't write into real data/ dir.
    from remy.core import promotion_audit as mod

    def _tmp_sink() -> Path:
        return tmp_path / "promotion_audit.jsonl"

    monkeypatch.setattr(mod, "_resolve_sink_path", _tmp_sink)
    yield
    reset_promotion_audit_log()


def _safe_item(**meta_overrides) -> dict:
    meta = {
        "source": "https://arxiv.org/abs/2411.02534",
        "verified": True,
        "admission_class": "grounded_external_fact",
    }
    meta.update(meta_overrides)
    return {
        "id": "rec-1",
        "tags": [],
        "metadata": meta,
        "content": "Factual statement.",
        "score": 0.7,
    }


# ── A. block_reason enumerator ──────────────────────────────────────────────


def test_block_reason_allows_safe_record():
    assert block_reason(_safe_item()) is None


def test_block_reason_requires_promotion_wins_before_truth():
    """requires_promotion check runs before truth_status lookup."""
    item = _safe_item(requires_promotion=True)
    assert block_reason(item) == REASON_REQUIRES_PROMOTION


def test_block_reason_promoted_flag_clears_requires_promotion():
    item = _safe_item(requires_promotion=True, promoted=True)
    assert block_reason(item) is None


def test_block_reason_superseded_by_explicit():
    item = _safe_item(superseded_by="rec-newer")
    assert block_reason(item) == REASON_SUPERSEDED_BY


def test_block_reason_unresolved_conflict_flag():
    item = _safe_item(unresolved_conflict=True)
    assert block_reason(item) == REASON_UNRESOLVED_CONFLICT_FLAG


def test_block_reason_truth_status_stale_hard():
    item = _safe_item(
        volatility="high",
        cached_at=(datetime.now(timezone.utc) - timedelta(days=365)).isoformat(),
    )
    assert block_reason(item) == REASON_TRUTH_STATUS_STALE_HARD


def test_block_reason_forbidden_admission_class():
    item = _safe_item(admission_class="working_state")
    assert block_reason(item) == REASON_FORBIDDEN_ADMISSION_CLASS


def test_block_reason_forbidden_tag():
    item = _safe_item()
    # "research-finding" lives in hybrid_search._FACTUAL_FORBIDDEN_TAGS.
    item["tags"] = ["research-finding"]
    assert block_reason(item) == REASON_FORBIDDEN_TAG


def test_block_reason_none_input_returns_none():
    assert block_reason(None) is None
    assert block_reason({}) is None


def test_block_reason_first_signal_wins_promotion_before_supersession():
    """When multiple signals fire, ordering mirrors _is_factual_forbidden."""
    item = _safe_item(requires_promotion=True, superseded_by="rec-x")
    # requires_promotion is checked before superseded_by in the enumerator.
    assert block_reason(item) == REASON_REQUIRES_PROMOTION


# ── B. PromotionAuditLog (ring + sink) ──────────────────────────────────────


def test_log_records_to_ring_buffer(tmp_path):
    log = PromotionAuditLog(sink_path=tmp_path / "p.jsonl")
    log.record(surface=SURFACE_RECALL, record_id="r1", reason=REASON_SUPERSEDED_BY)
    recent = log.get_recent()
    assert len(recent) == 1
    assert recent[0]["surface"] == SURFACE_RECALL
    assert recent[0]["record_id"] == "r1"
    assert recent[0]["reason"] == REASON_SUPERSEDED_BY
    assert "timestamp" in recent[0]


def test_log_writes_to_sink(tmp_path):
    sink = tmp_path / "p.jsonl"
    log = PromotionAuditLog(sink_path=sink)
    log.record(surface=SURFACE_CONNECT, record_id="r2", reason=REASON_REQUIRES_PROMOTION)
    assert sink.exists()
    content = sink.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1
    import json
    entry = json.loads(content[0])
    assert entry["surface"] == SURFACE_CONNECT
    assert entry["reason"] == REASON_REQUIRES_PROMOTION


def test_log_get_recent_filters_by_surface_and_reason(tmp_path):
    log = PromotionAuditLog(sink_path=tmp_path / "p.jsonl")
    log.record(surface=SURFACE_RECALL, record_id="a", reason=REASON_SUPERSEDED_BY)
    log.record(surface=SURFACE_CONNECT, record_id="b", reason=REASON_REQUIRES_PROMOTION)
    log.record(surface=SURFACE_RECALL, record_id="c", reason=REASON_REQUIRES_PROMOTION)

    by_surface = log.get_recent(surface=SURFACE_RECALL)
    assert {e["record_id"] for e in by_surface} == {"a", "c"}

    by_reason = log.get_recent(reason=REASON_REQUIRES_PROMOTION)
    assert {e["record_id"] for e in by_reason} == {"b", "c"}


def test_log_ring_capacity_drops_oldest(tmp_path):
    log = PromotionAuditLog(sink_path=tmp_path / "p.jsonl", capacity=3)
    for i in range(5):
        log.record(surface=SURFACE_RECALL, record_id=f"r{i}", reason=REASON_SUPERSEDED_BY)
    # Ring keeps last 3; newest-first so the kept IDs are r4, r3, r2.
    ids = [e["record_id"] for e in log.get_recent()]
    assert ids == ["r4", "r3", "r2"]


def test_log_clear_empties_ring(tmp_path):
    log = PromotionAuditLog(sink_path=tmp_path / "p.jsonl")
    log.record(surface=SURFACE_RECALL, record_id="r", reason=REASON_SUPERSEDED_BY)
    assert len(log) == 1
    log.clear()
    assert len(log) == 0


def test_record_block_never_raises_on_infra_failure(monkeypatch):
    """Even if the singleton blows up, record_block must swallow it."""
    from remy.core import promotion_audit as mod

    def _boom():
        raise RuntimeError("simulated")

    monkeypatch.setattr(mod, "get_promotion_audit_log", _boom)
    # Should not raise.
    record_block(SURFACE_RECALL, "r", REASON_SUPERSEDED_BY)


def test_singleton_lazy_and_reset():
    # First call creates; second call returns the same instance.
    log1 = get_promotion_audit_log()
    log2 = get_promotion_audit_log()
    assert log1 is log2
    reset_promotion_audit_log()
    log3 = get_promotion_audit_log()
    assert log3 is not log1


# ── C. Recall surface integration ────────────────────────────────────────────


def test_recall_filter_emits_one_event_per_blocked_item():
    blocked = [
        {"id": "b1", "tags": [], "metadata": {"admission_class": "grounded_external_fact",
                                              "superseded_by": "b1-new"}},
        {"id": "b2", "tags": [], "metadata": {"admission_class": "working_state"}},
    ]
    kept = _safe_item()
    items = [kept, *blocked]

    out = _apply_factual_recall_filter(items)
    # Only the safe item survives.
    assert len(out) == 1
    assert out[0]["id"] == "rec-1"

    events = get_promotion_audit_log().get_recent(surface=SURFACE_RECALL)
    assert {e["record_id"] for e in events} == {"b1", "b2"}
    reasons = {e["record_id"]: e["reason"] for e in events}
    assert reasons["b1"] == REASON_SUPERSEDED_BY
    assert reasons["b2"] == REASON_FORBIDDEN_ADMISSION_CLASS


def test_recall_filter_passes_through_when_nothing_blocked():
    items = [_safe_item()]
    out = _apply_factual_recall_filter(items)
    assert out == items
    assert get_promotion_audit_log().get_recent() == []


def test_recall_filter_handles_empty_and_none():
    assert _apply_factual_recall_filter([]) == []
    assert _apply_factual_recall_filter(None) == []
    assert get_promotion_audit_log().get_recent() == []


# ── D. Connect surface integration ──────────────────────────────────────────


class _FakeRec:
    def __init__(self, rec_id, *, tags=None, metadata=None):
        self.id = rec_id
        self.tags = list(tags or [])
        self.metadata = dict(metadata or {})


class _FakeBrain:
    def __init__(self):
        self.records: dict[str, _FakeRec] = {}
        self.connect_calls: list[tuple] = []

    def add_safe(self, rec_id):
        rec = _FakeRec(
            rec_id,
            metadata={
                "source": "https://arxiv.org/abs/2411.02534",
                "verified": True,
                "admission_class": "grounded_external_fact",
            },
        )
        self.records[rec_id] = rec
        return rec

    def get(self, rec_id):
        return self.records.get(rec_id)

    def connect(self, id_a, id_b, weight=0.0, **_):
        self.connect_calls.append((id_a, id_b, weight))


def test_connect_emits_event_for_blocked_first_endpoint():
    brain = _FakeBrain()
    a = brain.add_safe("a")
    a.metadata["superseded_by"] = "z"
    brain.add_safe("b")
    ok = gated_connect(brain, "a", "b", weight=0.5)
    assert ok is False

    events = get_promotion_audit_log().get_recent(surface=SURFACE_CONNECT)
    assert len(events) == 1
    ev = events[0]
    assert ev["record_id"] == "a"
    assert ev["reason"] == REASON_SUPERSEDED_BY
    assert ev["extra"]["partner_id"] == "b"
    assert ev["extra"]["weight"] == 0.5


def test_connect_emits_event_for_blocked_second_endpoint():
    brain = _FakeBrain()
    brain.add_safe("a")
    b = brain.add_safe("b")
    b.metadata["unresolved_conflict"] = True
    ok = gated_connect(brain, "a", "b", weight=0.7)
    assert ok is False
    events = get_promotion_audit_log().get_recent(surface=SURFACE_CONNECT)
    assert len(events) == 1
    assert events[0]["record_id"] == "b"
    assert events[0]["reason"] == REASON_UNRESOLVED_CONFLICT_FLAG
    assert events[0]["extra"]["partner_id"] == "a"


def test_connect_emits_two_events_when_both_endpoints_blocked():
    brain = _FakeBrain()
    a = brain.add_safe("a")
    a.metadata["requires_promotion"] = True
    b = brain.add_safe("b")
    b.metadata["superseded_by"] = "c"
    ok = gated_connect(brain, "a", "b", weight=0.9)
    assert ok is False
    events = get_promotion_audit_log().get_recent(surface=SURFACE_CONNECT)
    # Strictest-gate semantics: both endpoints contributed reasons → two events.
    assert len(events) == 2
    by_id = {e["record_id"]: e["reason"] for e in events}
    assert by_id == {
        "a": REASON_REQUIRES_PROMOTION,
        "b": REASON_SUPERSEDED_BY,
    }


def test_connect_emits_missing_endpoint_when_record_not_found():
    brain = _FakeBrain()
    brain.add_safe("a")
    # "b" absent
    ok = gated_connect(brain, "a", "b", weight=0.3)
    assert ok is False
    events = get_promotion_audit_log().get_recent(surface=SURFACE_CONNECT)
    assert len(events) == 1
    assert events[0]["reason"] == REASON_MISSING_ENDPOINT


def test_connect_emits_no_event_when_both_allowed():
    brain = _FakeBrain()
    brain.add_safe("a")
    brain.add_safe("b")
    ok = gated_connect(brain, "a", "b", weight=0.4)
    assert ok is True
    assert brain.connect_calls == [("a", "b", 0.4)]
    assert get_promotion_audit_log().get_recent() == []


def test_events_persist_to_jsonl_sink_via_singleton(tmp_path, monkeypatch):
    """Reset singleton pointing sink at tmp_path, trigger a real block, read
    the JSONL back. Confirms the pipeline is durable, not just in-memory."""
    sink = tmp_path / "real_sink.jsonl"
    from remy.core import promotion_audit as mod

    monkeypatch.setattr(mod, "_resolve_sink_path", lambda: sink)
    reset_promotion_audit_log()

    blocked = [{"id": "x", "tags": [], "metadata": {"admission_class": "working_state"}}]
    _apply_factual_recall_filter(blocked)

    assert sink.exists()
    content = sink.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1
    import json
    entry = json.loads(content[0])
    assert entry["record_id"] == "x"
    assert entry["reason"] == REASON_FORBIDDEN_ADMISSION_CLASS
    assert entry["surface"] == SURFACE_RECALL
