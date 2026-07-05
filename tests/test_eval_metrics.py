"""Tests for F5: Evaluation Metrics — zero-LLM quality tracking."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from remy.core.eval_metrics import (
    ResponseMetrics,
    compute_response_metrics,
    get_metrics_summary,
    store_eval_metrics,
)


class TestComputeResponseMetrics:
    """Tests for compute_response_metrics()."""

    def test_compute_metrics_basic(self):
        metrics = compute_response_metrics(
            session_id="s1",
            channel="desktop",
            messages=[HumanMessage(content="hi"), AIMessage(content="hello there")],
            session_log=[],
            response_text="hello there",
            duration_ms=500,
            context_injected=False,
        )

        assert metrics.session_id == "s1"
        assert metrics.channel == "desktop"
        assert metrics.response_length == len("hello there")
        assert metrics.response_word_count == 2
        assert metrics.tools_called == 0
        assert metrics.total_duration_ms == 500
        assert metrics.llm_calls == 1
        assert metrics.context_injected is False

    def test_compute_with_tools(self):
        session_log = [
            {"type": "tool_call", "tool": "recall", "result": '{"memories": []}'},
            {"type": "tool_call", "tool": "web_search", "result": '{"answer": "found"}'},
            {"type": "tool_call", "tool": "store", "result": "Error: failed to store"},
        ]

        metrics = compute_response_metrics(
            session_id="s2",
            channel="telegram",
            messages=[
                HumanMessage(content="search"),
                AIMessage(content="", tool_calls=[{"name": "recall", "args": {}, "id": "1"}]),
                ToolMessage(content="ok", tool_call_id="1"),
                AIMessage(content="result"),
            ],
            session_log=session_log,
            response_text="Here is what I found",
            duration_ms=2000,
            context_injected=True,
        )

        assert metrics.tools_called == 3
        assert metrics.tools_succeeded == 2
        assert metrics.tools_failed == 1
        assert "recall" in metrics.unique_tools
        assert "web_search" in metrics.unique_tools
        assert metrics.recall_used is True
        assert metrics.store_used is True
        assert metrics.context_injected is True
        assert metrics.llm_calls == 2

    def test_recall_detection_via_recall_knowledge(self):
        session_log = [
            {"type": "tool_call", "tool": "recall_knowledge", "result": "{}"},
        ]

        metrics = compute_response_metrics(
            session_id="s3",
            channel="desktop",
            messages=[],
            session_log=session_log,
            response_text="ok",
        )

        assert metrics.recall_used is True

    def test_empty_session_log(self):
        metrics = compute_response_metrics(
            session_id="s4",
            channel="desktop",
            messages=[],
            session_log=[],
            response_text="hello",
        )

        assert metrics.tools_called == 0
        assert metrics.recall_used is False
        assert metrics.store_used is False


class TestStoreEvalMetrics:
    """Tests for store_eval_metrics() — writes to JSONL file."""

    def test_store_metrics(self, tmp_path):
        mock_settings = MagicMock()
        mock_settings.DATA_DIR = tmp_path

        with patch("remy.core.eval_metrics.settings", mock_settings):
            metrics = ResponseMetrics(
                session_id="s1",
                channel="desktop",
                response_length=100,
                response_word_count=20,
                tools_called=3,
                tools_succeeded=2,
                tools_failed=1,
                recall_used=True,
            )

            store_eval_metrics(metrics)

            metrics_file = tmp_path / "eval_metrics.jsonl"
            assert metrics_file.exists()
            lines = metrics_file.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["session_id"] == "s1"
            assert entry["channel"] == "desktop"
            assert entry["tools_called"] == 3
            assert entry["tools_succeeded"] == 2
            assert entry["tools_failed"] == 1
            assert entry["recall_used"] is True

    def test_store_metrics_appends(self, tmp_path):
        mock_settings = MagicMock()
        mock_settings.DATA_DIR = tmp_path

        with patch("remy.core.eval_metrics.settings", mock_settings):
            for i in range(3):
                metrics = ResponseMetrics(session_id=f"s{i}", channel="desktop")
                store_eval_metrics(metrics)

            metrics_file = tmp_path / "eval_metrics.jsonl"
            lines = metrics_file.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 3

    def test_store_metrics_handles_errors(self):
        # Point to non-existent, read-only path — should not raise
        mock_settings = MagicMock()
        mock_settings.DATA_DIR = Path("/nonexistent/path/that/doesnt/exist")

        with patch("remy.core.eval_metrics.settings", mock_settings):
            metrics = ResponseMetrics(session_id="s1", channel="desktop")
            # Should not raise
            store_eval_metrics(metrics)


class TestGetMetricsSummary:
    """Tests for get_metrics_summary() — reads from JSONL file."""

    def _write_entries(self, tmp_path, entries):
        metrics_file = tmp_path / "eval_metrics.jsonl"
        with open(metrics_file, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def test_summary_aggregation(self, tmp_path):
        entries = [
            {
                "session_id": "s1", "channel": "desktop",
                "tools_called": 4, "tools_succeeded": 3, "tools_failed": 1,
                "response_word_count": 50, "recall_used": True, "store_used": False,
                "context_injected": True, "total_duration_ms": 1000, "llm_calls": 2,
            },
            {
                "session_id": "s2", "channel": "desktop",
                "tools_called": 2, "tools_succeeded": 2, "tools_failed": 0,
                "response_word_count": 30, "recall_used": False, "store_used": True,
                "context_injected": False, "total_duration_ms": 500, "llm_calls": 1,
            },
        ]
        self._write_entries(tmp_path, entries)

        mock_settings = MagicMock()
        mock_settings.DATA_DIR = tmp_path

        with patch("remy.core.eval_metrics.settings", mock_settings):
            result = get_metrics_summary()

        assert result["total_responses"] == 2
        assert result["avg_word_count"] == 40.0
        assert result["avg_tools_per_response"] == 3.0
        assert result["tool_success_rate"] == pytest.approx(83.3, abs=0.1)
        assert result["recall_usage_rate"] == 50.0
        assert result["store_usage_rate"] == 50.0
        assert result["context_injection_rate"] == 50.0
        assert result["avg_duration_ms"] == 750
        assert result["avg_llm_calls"] == 1.5

    def test_summary_empty(self, tmp_path):
        mock_settings = MagicMock()
        mock_settings.DATA_DIR = tmp_path
        # No JSONL file at all

        with patch("remy.core.eval_metrics.settings", mock_settings):
            result = get_metrics_summary()
            assert result["total_responses"] == 0

    def test_summary_with_channel_filter(self, tmp_path):
        entries = [
            {"session_id": "s1", "channel": "desktop", "tools_called": 1,
             "tools_succeeded": 1, "tools_failed": 0, "response_word_count": 10,
             "recall_used": False, "store_used": False, "context_injected": False,
             "total_duration_ms": 100, "llm_calls": 1},
            {"session_id": "s2", "channel": "telegram", "tools_called": 2,
             "tools_succeeded": 2, "tools_failed": 0, "response_word_count": 20,
             "recall_used": True, "store_used": False, "context_injected": False,
             "total_duration_ms": 200, "llm_calls": 1},
        ]
        self._write_entries(tmp_path, entries)

        mock_settings = MagicMock()
        mock_settings.DATA_DIR = tmp_path

        with patch("remy.core.eval_metrics.settings", mock_settings):
            result = get_metrics_summary(channel="telegram")

        assert result["total_responses"] == 1
        assert result["recall_usage_rate"] == 100.0

    def test_summary_handles_corrupt_lines(self, tmp_path):
        metrics_file = tmp_path / "eval_metrics.jsonl"
        with open(metrics_file, "w", encoding="utf-8") as f:
            f.write("not valid json\n")
            f.write(json.dumps({
                "session_id": "s1", "channel": "desktop",
                "tools_called": 1, "tools_succeeded": 1, "tools_failed": 0,
                "response_word_count": 10, "recall_used": False, "store_used": False,
                "context_injected": False, "total_duration_ms": 100, "llm_calls": 1,
            }) + "\n")

        mock_settings = MagicMock()
        mock_settings.DATA_DIR = tmp_path

        with patch("remy.core.eval_metrics.settings", mock_settings):
            result = get_metrics_summary()

        assert result["total_responses"] == 1
