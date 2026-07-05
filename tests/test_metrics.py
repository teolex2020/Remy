"""Tests for Prometheus metrics endpoint — MetricsCollector, path normalization, collect_metrics."""

import threading
import time
from unittest.mock import patch, MagicMock

import pytest
from aura import Aura as CognitiveMemory
from aura import Level


# ============== MetricsCollector ==============


class TestMetricsCollector:

    def _make(self):
        from remy.core.metrics import MetricsCollector
        return MetricsCollector()

    def test_initial_snapshot_empty(self):
        """New collector has zero counts."""
        mc = self._make()
        snap = mc.get_snapshot()
        assert snap["http_requests_total"] == {}
        assert snap["active_ws"] == {}
        assert snap["llm_calls_total"] == 0
        assert snap["llm_duration_seconds"] == []

    def test_record_http_increments(self):
        """Recording requests increments the correct bucket."""
        mc = self._make()
        mc.record_http_request("GET", "/api/stats", 200, 0.05)
        mc.record_http_request("GET", "/api/stats", 200, 0.03)
        snap = mc.get_snapshot()
        assert sum(snap["http_requests_total"].values()) == 2

    def test_record_http_different_statuses(self):
        """Different status classes create separate counter entries."""
        mc = self._make()
        mc.record_http_request("GET", "/api/stats", 200, 0.01)
        mc.record_http_request("GET", "/api/stats", 500, 0.01)
        snap = mc.get_snapshot()
        assert len(snap["http_requests_total"]) == 2

    def test_record_http_different_methods(self):
        """Different HTTP methods create separate entries."""
        mc = self._make()
        mc.record_http_request("GET", "/api/stats", 200, 0.01)
        mc.record_http_request("POST", "/api/stats", 200, 0.01)
        snap = mc.get_snapshot()
        assert len(snap["http_requests_total"]) == 2

    def test_ws_connected_disconnected(self):
        """WebSocket connect/disconnect tracking."""
        mc = self._make()
        mc.ws_connected("chat")
        mc.ws_connected("chat")
        assert mc.get_snapshot()["active_ws"]["chat"] == 2
        mc.ws_disconnected("chat")
        assert mc.get_snapshot()["active_ws"]["chat"] == 1

    def test_ws_disconnected_floor_zero(self):
        """Disconnect below zero stays at zero."""
        mc = self._make()
        mc.ws_disconnected("chat")
        assert mc.get_snapshot()["active_ws"]["chat"] == 0

    def test_record_llm_call(self):
        """LLM call recording increments counter and stores duration."""
        mc = self._make()
        mc.record_llm_call(1.5)
        mc.record_llm_call(2.0)
        snap = mc.get_snapshot()
        assert snap["llm_calls_total"] == 2
        assert len(snap["llm_duration_seconds"]) == 2
        assert snap["llm_duration_seconds"] == [1.5, 2.0]

    def test_duration_cap(self):
        """Duration lists are capped at 1000 entries."""
        mc = self._make()
        for i in range(1100):
            mc.record_http_request("GET", "/api/test", 200, 0.01)
        snap = mc.get_snapshot()
        for durations in snap["http_duration_seconds"].values():
            assert len(durations) <= 1000

    def test_llm_duration_cap(self):
        """LLM duration list is capped at 1000."""
        mc = self._make()
        for _ in range(1100):
            mc.record_llm_call(0.1)
        snap = mc.get_snapshot()
        assert len(snap["llm_duration_seconds"]) <= 1000

    def test_thread_safety(self):
        """Multiple threads recording concurrently do not raise."""
        mc = self._make()
        errors = []

        def worker():
            try:
                for _ in range(100):
                    mc.record_http_request("GET", "/api/x", 200, 0.01)
                    mc.ws_connected("chat")
                    mc.ws_disconnected("chat")
                    mc.record_llm_call(0.1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        snap = mc.get_snapshot()
        assert snap["llm_calls_total"] == 400


# ============== Path Normalization ==============


class TestNormalizePath:

    def _norm(self, path):
        from remy.core.metrics import _normalize_path
        return _normalize_path(path)

    def test_simple_api_path(self):
        assert self._norm("/api/stats") == "/api/stats"

    def test_record_id_normalized(self):
        result = self._norm("/api/records/abc123def456")
        assert ":id" in result
        assert "abc123" not in result

    def test_short_path_preserved(self):
        assert self._norm("/api/export") == "/api/export"

    def test_nested_id_path(self):
        """Paths like /api/todos/rec-xyz123/toggle normalize the ID segment."""
        result = self._norm("/api/todos/rec-xyz123abc/toggle")
        assert ":id" in result
        assert "toggle" in result


# ============== collect_metrics() Output ==============


class TestCollectMetrics:

    def test_output_contains_help_type_lines(self, tmp_path):
        """Output has proper Prometheus format markers."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        with patch("remy.core.agent_tools.brain", b), \
             patch("remy.web.api._start_time", time.time() - 60):
            from remy.core.metrics import collect_metrics
            output = collect_metrics()
            assert "# HELP" in output
            assert "# TYPE" in output
            assert "remy_uptime_seconds" in output
        b.close()

    def test_output_contains_brain_records(self, tmp_path):
        """Brain record count appears in output."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        b.store(content="test record", level=Level.Domain, tags=["test"])
        with patch("remy.core.agent_tools.brain", b), \
             patch("remy.web.api._start_time", time.time()):
            from remy.core.metrics import collect_metrics
            output = collect_metrics()
            assert "remy_brain_records_total" in output
        b.close()

    def test_prometheus_format_parseable(self, tmp_path):
        """Every non-empty, non-comment line has metric_name + numeric value."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        with patch("remy.core.agent_tools.brain", b), \
             patch("remy.web.api._start_time", time.time()):
            from remy.core.metrics import collect_metrics
            output = collect_metrics()
            for line in output.strip().split("\n"):
                if not line or line.startswith("#"):
                    continue
                parts = line.rsplit(" ", 1)
                assert len(parts) == 2, f"Invalid metric line: {line}"
                try:
                    float(parts[-1])
                except ValueError:
                    pytest.fail(f"Non-numeric value in metric line: {line}")
        b.close()

    def test_http_metrics_appear_after_recording(self, tmp_path):
        """HTTP metrics show up after recording requests."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        with patch("remy.core.agent_tools.brain", b), \
             patch("remy.web.api._start_time", time.time()):
            from remy.core.metrics import metrics_collector, collect_metrics, MetricsCollector
            # Use a fresh collector to avoid cross-test pollution
            fresh = MetricsCollector()
            fresh.record_http_request("GET", "/api/stats", 200, 0.05)
            with patch("remy.core.metrics.metrics_collector", fresh):
                output = collect_metrics()
                assert "remy_http_requests_total" in output
                assert "remy_http_duration_seconds" in output
        b.close()

    def test_ws_metrics_appear(self, tmp_path):
        """WebSocket metrics always appear (even with 0 connections)."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        with patch("remy.core.agent_tools.brain", b), \
             patch("remy.web.api._start_time", time.time()):
            from remy.core.metrics import collect_metrics
            output = collect_metrics()
            assert "remy_active_websockets" in output
        b.close()

    def test_llm_metrics_appear(self, tmp_path):
        """LLM call counter always appears."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        with patch("remy.core.agent_tools.brain", b), \
             patch("remy.web.api._start_time", time.time()):
            from remy.core.metrics import collect_metrics
            output = collect_metrics()
            assert "remy_llm_calls_total" in output
        b.close()

    def test_goal_metrics_use_shared_goal_snapshot(self, tmp_path):
        """Goal metrics come from the shared operator snapshot seam."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        with patch("remy.core.agent_tools.brain", b), \
             patch("remy.web.api._start_time", time.time()), \
             patch(
                 "remy.core.combined_runner.get_goal_runtime_snapshot",
                 return_value={
                     "total": 7,
                     "active": 3,
                     "blocked": 2,
                     "active_list": [],
                 },
             ):
            from remy.core.metrics import collect_metrics

            output = collect_metrics()

            assert 'remy_goals{status="active"} 3' in output
            assert 'remy_goals{status="blocked"} 2' in output
            assert 'remy_goals{status="completed"} 2' in output
            assert "remy_goal_completion_rate 0.286" in output
        b.close()

    def test_budget_token_metrics_use_shared_budget_snapshot(self, tmp_path):
        """Autonomy token metrics come from the shared budget snapshot seam."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        with patch("remy.core.agent_tools.brain", b), \
             patch("remy.web.api._start_time", time.time()), \
             patch(
                 "remy.core.combined_runner.get_budget_runtime_snapshot",
                 return_value={
                     "llm_tokens_today": 321,
                     "llm_tokens_this_hour": 45,
                     "llm_tokens_lifetime": 6543,
                 },
             ):
            from remy.core.metrics import collect_metrics

            output = collect_metrics()

            assert "remy_autonomy_tokens_today 321" in output
            assert "remy_autonomy_tokens_this_hour 45" in output
            assert "remy_autonomy_tokens_lifetime 6543" in output
        b.close()


