"""
Phase 3 Step 3 — Promotion Governance / brain.connect() Gate.

Design rule: a record is *admitted* to memory by the memory_policy layer, but
only becomes part of the promoted graph (concept / causal / policy edges) if
it passes the five promotion signals. `gated_connect()` is the only wrapper
that the codebase's 19 active `brain.connect()` call sites now go through.

Covered here:

  A. `promotion_allowed` predicate — structural: five blocking signals, each
     in isolation, using the same rule-set `_is_factual_forbidden` already
     enforces at the recall surface. Behavioral only, no SDK round-trip.

  B. `gated_connect` behavior — an in-memory brain fake exposes `.get(id)` and
     `.connect(...)`; the wrapper must call `connect` only when *both*
     endpoints are promotion-eligible. Strictest-gate semantics: either
     endpoint blocked → edge blocked, return False. Missing endpoint → also
     blocked.

Out of scope: Rust belief paths, global blocked-promotion audit events,
converting `requires_promotion` stamp beyond recall+connect surfaces.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from remy.core.agent_tools import (
    _record_to_check_item,
    gated_connect,
    promotion_allowed,
)


# ── Fake brain ──────────────────────────────────────────────────────────────


class _FakeRec:
    def __init__(self, rec_id: str, *, tags=None, metadata=None):
        self.id = rec_id
        self.tags = list(tags or [])
        self.metadata = dict(metadata or {})


class _FakeBrain:
    """Minimal stand-in: stores records by id, tracks connect() calls."""

    def __init__(self):
        self.records: dict[str, _FakeRec] = {}
        self.connect_calls: list[tuple] = []
        self.raise_on_get: bool = False

    def add(self, rec_id: str, **kwargs) -> _FakeRec:
        rec = _FakeRec(rec_id, **kwargs)
        self.records[rec_id] = rec
        return rec

    def add_safe(self, rec_id: str) -> _FakeRec:
        """Promotion-eligible baseline: grounded_external_fact, no blocking meta."""
        return self.add(
            rec_id,
            tags=[],
            metadata={
                "source": "https://arxiv.org/abs/2411.02534",
                "verified": True,
                "admission_class": "grounded_external_fact",
            },
        )

    def get(self, rec_id: str):
        if self.raise_on_get:
            raise RuntimeError("simulated SDK failure")
        return self.records.get(rec_id)

    def connect(self, id_a, id_b, weight=0.0, **_ignored):
        self.connect_calls.append((id_a, id_b, weight))


# ── Shape helpers ───────────────────────────────────────────────────────────


def test_record_to_check_item_handles_object():
    rec = _FakeRec("x", tags=["t"], metadata={"k": "v"})
    item = _record_to_check_item(rec)
    assert item["id"] == "x"
    assert item["tags"] == ["t"]
    assert item["metadata"] == {"k": "v"}


def test_record_to_check_item_handles_dict():
    item = _record_to_check_item({"id": "y", "tags": ["u"], "metadata": {"m": 1}})
    assert item["id"] == "y"
    assert item["tags"] == ["u"]
    assert item["metadata"] == {"m": 1}


def test_record_to_check_item_handles_none():
    assert _record_to_check_item(None) == {}


# ── promotion_allowed: base + five blocking signals ─────────────────────────


def _safe_rec(rec_id: str = "r") -> _FakeRec:
    return _FakeRec(
        rec_id,
        tags=[],
        metadata={
            "source": "https://arxiv.org/abs/2411.02534",
            "verified": True,
            "admission_class": "grounded_external_fact",
        },
    )


def test_allowed_baseline_grounded_fact():
    assert promotion_allowed(_safe_rec()) is True


def test_blocked_by_requires_promotion_unpromoted():
    rec = _safe_rec()
    rec.metadata["requires_promotion"] = True
    # promoted flag missing → blocked
    assert promotion_allowed(rec) is False


def test_allowed_when_requires_promotion_and_promoted():
    rec = _safe_rec()
    rec.metadata["requires_promotion"] = True
    rec.metadata["promoted"] = True
    assert promotion_allowed(rec) is True


def test_blocked_by_unresolved_conflict():
    rec = _safe_rec()
    rec.metadata["unresolved_conflict"] = True
    assert promotion_allowed(rec) is False


def test_blocked_by_superseded_by():
    rec = _safe_rec()
    rec.metadata["superseded_by"] = "rec-newer"
    assert promotion_allowed(rec) is False


def test_blocked_by_stale_hard_truth_status():
    rec = _safe_rec()
    # volatility=high TTL is short; cached_at well in the past → stale_hard
    rec.metadata["volatility"] = "high"
    rec.metadata["cached_at"] = (
        datetime.now(timezone.utc) - timedelta(days=365)
    ).isoformat()
    assert promotion_allowed(rec) is False


def test_blocked_by_forbidden_admission_class():
    rec = _safe_rec()
    rec.metadata["admission_class"] = "working_state"
    assert promotion_allowed(rec) is False


def test_allowed_on_none_input():
    # Best-effort: None shouldn't crash, defaults to allowed (infra failure
    # must not block promotion — mirrors _is_factual_forbidden's try/except).
    assert promotion_allowed(None) is True


# ── gated_connect: strictest-gate behavior ───────────────────────────────────


def test_connect_when_both_endpoints_allowed():
    brain = _FakeBrain()
    brain.add_safe("a")
    brain.add_safe("b")
    ok = gated_connect(brain, "a", "b", weight=0.8)
    assert ok is True
    assert brain.connect_calls == [("a", "b", 0.8)]


def test_blocked_when_first_endpoint_forbidden():
    brain = _FakeBrain()
    a = brain.add_safe("a")
    a.metadata["superseded_by"] = "z"
    brain.add_safe("b")
    ok = gated_connect(brain, "a", "b", weight=0.5)
    assert ok is False
    assert brain.connect_calls == []


def test_blocked_when_second_endpoint_forbidden():
    brain = _FakeBrain()
    brain.add_safe("a")
    b = brain.add_safe("b")
    b.metadata["unresolved_conflict"] = True
    ok = gated_connect(brain, "a", "b", weight=0.5)
    assert ok is False
    assert brain.connect_calls == []


def test_blocked_when_either_endpoint_missing():
    brain = _FakeBrain()
    brain.add_safe("a")
    # "b" absent
    assert gated_connect(brain, "a", "b") is False
    assert brain.connect_calls == []

    brain2 = _FakeBrain()
    brain2.add_safe("b")
    assert gated_connect(brain2, "a", "b") is False
    assert brain2.connect_calls == []


def test_blocked_by_five_signals_one_at_a_time():
    """Each of the five promotion signals, in isolation, blocks the edge."""
    signals = [
        {"requires_promotion": True},                 # admitted-not-promoted
        {"unresolved_conflict": True},                # conflict
        {"superseded_by": "rec-new"},                 # supersession
        {                                             # stale_hard truth
            "volatility": "high",
            "cached_at": (
                datetime.now(timezone.utc) - timedelta(days=365)
            ).isoformat(),
        },
        {"admission_class": "working_state"},         # forbidden class
    ]
    for sig in signals:
        brain = _FakeBrain()
        brain.add_safe("a")
        b = brain.add_safe("b")
        b.metadata.update(sig)
        ok = gated_connect(brain, "a", "b", weight=0.7)
        assert ok is False, f"signal {sig} should have blocked"
        assert brain.connect_calls == [], (
            f"signal {sig} allowed a brain.connect() call"
        )


def test_positional_weight_compat_legacy_caller():
    """One caller form historically passed weight positionally
    (brain.connect(a, b, 0.9)). The wrapper must handle that via TypeError
    catch so legacy call sites keep working."""

    class _LegacyBrain(_FakeBrain):
        def connect(self, id_a, id_b, weight):  # no kwarg
            self.connect_calls.append((id_a, id_b, weight))

    brain = _LegacyBrain()
    brain.add_safe("a")
    brain.add_safe("b")
    ok = gated_connect(brain, "a", "b", weight=0.9)
    assert ok is True
    assert brain.connect_calls == [("a", "b", 0.9)]


def test_sdk_get_exception_fails_closed():
    """If brain.get() raises, we must NOT create the edge.
    Fail-closed on infrastructure error is the safe default at the gate."""
    brain = _FakeBrain()
    brain.add_safe("a")
    brain.add_safe("b")
    brain.raise_on_get = True
    ok = gated_connect(brain, "a", "b", weight=0.5)
    assert ok is False
    assert brain.connect_calls == []
