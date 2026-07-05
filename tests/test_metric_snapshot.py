"""Tests for metric_snapshot (collection) and metric_render (substitution).

These cover the contract that primary-defends against self-metric hallucination:
    - Failed extractors do not crash a turn; they produce a partial snapshot.
    - {{metric:id}} tokens substitute ONLY against exact ids from the snapshot.
    - Unknown / stale ids get deterministic placeholders, never guesses.
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace

import pytest

from remy.core.metric_render import (
    STALE_PLACEHOLDER,
    UNKNOWN_PLACEHOLDER,
    build_compact_injection,
    build_full_injection,
    render_metrics,
)
from remy.core.metric_snapshot import (
    METRIC_SCHEMA,
    MetricSnapshot,
    MetricSource,
    MetricValue,
    collect_metric_snapshot,
)


# ---------- fake brain helpers ----------

def _thermal_summary(
    total_records: int = 681,
    hot_zones_count: int = 6,
    volatile: int = 53,
    clusters: int = 8,
):
    hot_zones = [SimpleNamespace(label=f"hz{i}", temperature=1.0) for i in range(hot_zones_count)]
    return SimpleNamespace(
        total_records=total_records,
        hot_zones=hot_zones,
        high_volatility_belief_count=volatile,
        contradiction_cluster_count=clusters,
    )


def _health_digest(recent_corrections: int = 6):
    return SimpleNamespace(recent_correction_count=recent_corrections)


def _make_brain(thermal=None, health=None):
    aura = SimpleNamespace(export_acl_thermal_summary=lambda: thermal or _thermal_summary())
    brain = SimpleNamespace(
        _aura=aura,
        get_memory_health_digest=lambda: health or _health_digest(),
    )
    return brain


# ---------- collect_metric_snapshot ----------

class TestCollectSnapshot:
    def test_all_metrics_present(self):
        snap = collect_metric_snapshot(_make_brain(), session_id="s1", turn_id="t1")
        assert snap.session_id == "s1"
        assert snap.turn_id == "t1"
        assert snap.missing_metric_ids == ()
        assert set(snap.available_ids) == {
            "total_records", "hot_zones", "volatile_beliefs",
            "conflict_clusters", "recent_corrections",
        }

    def test_values_match_source(self):
        snap = collect_metric_snapshot(_make_brain())
        assert snap.get("total_records").value == 681
        assert snap.get("hot_zones").value == 6
        assert snap.get("volatile_beliefs").value == 53
        assert snap.get("conflict_clusters").value == 8
        assert snap.get("recent_corrections").value == 6

    def test_each_metric_carries_source_path(self):
        snap = collect_metric_snapshot(_make_brain())
        for mid in snap.available_ids:
            v = snap.get(mid)
            assert v.source_method
            assert v.source_path
            assert v.collected_at > 0
            assert v.stale_after_sec > 0

    def test_partial_failure_does_not_crash(self):
        # thermal summary raises — other metrics still resolve.
        def raising_thermal():
            raise RuntimeError("export broken")
        brain = SimpleNamespace(
            _aura=SimpleNamespace(export_acl_thermal_summary=raising_thermal),
            get_memory_health_digest=lambda: _health_digest(),
        )
        snap = collect_metric_snapshot(brain)
        assert "recent_corrections" in snap.available_ids
        # The four thermal-derived metrics are missing.
        missing = set(snap.missing_metric_ids)
        assert {"total_records", "hot_zones", "volatile_beliefs", "conflict_clusters"} <= missing

    def test_missing_attribute_is_tolerated(self):
        # thermal summary missing a field → that single metric drops, others keep.
        bad_thermal = SimpleNamespace(
            total_records=500,
            hot_zones=[],
            # high_volatility_belief_count absent on purpose
            contradiction_cluster_count=3,
        )
        brain = _make_brain(thermal=bad_thermal)
        snap = collect_metric_snapshot(brain)
        assert "volatile_beliefs" in snap.missing_metric_ids
        assert snap.get("total_records").value == 500
        assert snap.get("conflict_clusters").value == 3

    def test_all_metrics_fail_yields_empty_snapshot(self):
        def boom():
            raise RuntimeError("x")
        brain = SimpleNamespace(
            _aura=SimpleNamespace(export_acl_thermal_summary=boom),
            get_memory_health_digest=boom,
        )
        snap = collect_metric_snapshot(brain)
        assert snap.values == {}
        assert len(snap.missing_metric_ids) == len(METRIC_SCHEMA)

    def test_missing_optional_thermal_summary_does_not_warn(self, caplog):
        brain = SimpleNamespace(get_memory_health_digest=lambda: _health_digest())

        with caplog.at_level(logging.WARNING, logger="remy.core.metric_snapshot"):
            snap = collect_metric_snapshot(brain)

        assert {"hot_zones", "volatile_beliefs", "conflict_clusters"} <= set(snap.missing_metric_ids)
        assert "metric_snapshot:" not in caplog.text

    def test_real_metric_source_error_still_warns(self, caplog):
        def boom():
            raise RuntimeError("x")

        brain = SimpleNamespace(
            _aura=SimpleNamespace(export_acl_thermal_summary=boom),
            get_memory_health_digest=lambda: _health_digest(),
        )

        with caplog.at_level(logging.WARNING, logger="remy.core.metric_snapshot"):
            collect_metric_snapshot(brain)

        assert "metric_snapshot:" in caplog.text

    def test_custom_schema_honored(self):
        custom = (
            MetricSource(
                id="only_one",
                extract=lambda b: 42,
                source_method="test.only_one",
                source_path="test",
            ),
        )
        snap = collect_metric_snapshot(_make_brain(), schema=custom)
        assert snap.available_ids == ("only_one",)
        assert snap.get("only_one").value == 42


# ---------- render_metrics ----------

def _snapshot(values: dict[str, int | float | str], age_sec: float = 0.0, ttl: float = 60.0) -> MetricSnapshot:
    now = time.time()
    wrapped = {
        k: MetricValue(
            value=v,
            source_method="test",
            source_path="test",
            collected_at=now - age_sec,
            stale_after_sec=ttl,
        )
        for k, v in values.items()
    }
    return MetricSnapshot(values=wrapped, collected_at=now)


class TestRenderMetrics:
    def test_single_token_substituted(self):
        snap = _snapshot({"total_records": 681})
        r = render_metrics("У мене {{metric:total_records}} записів.", snap)
        assert r.text == "У мене 681 записів."
        assert r.used_metric_ids == ("total_records",)

    def test_multiple_tokens_substituted(self):
        snap = _snapshot({"total_records": 681, "hot_zones": 6})
        r = render_metrics("{{metric:total_records}} records / {{metric:hot_zones}} zones", snap)
        assert r.text == "681 records / 6 zones"
        assert sorted(r.used_metric_ids) == ["hot_zones", "total_records"]

    def test_unknown_id_replaced_with_placeholder(self):
        snap = _snapshot({"total_records": 681})
        r = render_metrics("foo {{metric:nonexistent}} bar", snap)
        assert r.text == f"foo {UNKNOWN_PLACEHOLDER} bar"
        assert r.unknown_metric_ids == ("nonexistent",)
        assert r.used_metric_ids == ()

    def test_no_normalization_of_similar_ids(self):
        # "HotZones" is NOT the same as "hot_zones" — must stay unknown.
        snap = _snapshot({"hot_zones": 6})
        r = render_metrics("{{metric:HotZones}}", snap)
        assert r.text == UNKNOWN_PLACEHOLDER
        assert r.unknown_metric_ids == ("HotZones",)

    def test_stale_value_redacted(self):
        snap = _snapshot({"total_records": 681}, age_sec=3600, ttl=60)
        r = render_metrics("{{metric:total_records}}", snap)
        assert r.text == STALE_PLACEHOLDER
        assert r.stale_metric_ids == ("total_records",)
        assert r.used_metric_ids == ()

    def test_empty_text_returns_empty(self):
        snap = _snapshot({"total_records": 681})
        r = render_metrics("", snap)
        assert r.text == ""
        assert r.used_metric_ids == ()

    def test_text_without_tokens_unchanged(self):
        snap = _snapshot({"total_records": 681})
        original = "Просто речення без токенів."
        r = render_metrics(original, snap)
        assert r.text == original
        assert r.used_metric_ids == ()

    def test_mixed_known_unknown(self):
        snap = _snapshot({"hot_zones": 6})
        r = render_metrics("A={{metric:hot_zones}} B={{metric:ghost}}", snap)
        assert r.text == f"A=6 B={UNKNOWN_PLACEHOLDER}"
        assert r.used_metric_ids == ("hot_zones",)
        assert r.unknown_metric_ids == ("ghost",)

    def test_empty_snapshot_everything_unknown(self):
        snap = _snapshot({})
        r = render_metrics("{{metric:any}}", snap)
        assert r.text == UNKNOWN_PLACEHOLDER
        assert r.unknown_metric_ids == ("any",)


# ---------- build_compact_injection ----------

class TestCompactInjection:
    def test_format_is_machine_like(self):
        snap = _snapshot({"total_records": 681, "hot_zones": 6})
        line = build_compact_injection(snap)
        assert line.startswith("[metrics]")
        assert "total_records=681" in line
        assert "hot_zones=6" in line
        # Single line, no prose words.
        assert "\n" not in line

    def test_empty_snapshot_returns_empty_string(self):
        snap = _snapshot({})
        assert build_compact_injection(snap) == ""


# ---------- build_full_injection ----------

class TestFullInjection:
    def test_contains_contract_header(self):
        snap = _snapshot({"total_records": 681})
        block = build_full_injection(snap)
        assert "=== AVAILABLE INTERNAL METRICS ===" in block
        assert "{{metric:ID}}" in block
        assert "total_records" in block
        assert "681" in block
        assert "=== END METRICS ===" in block

    def test_empty_snapshot_returns_empty(self):
        snap = _snapshot({})
        assert build_full_injection(snap) == ""
