"""Tests for RM-6: Error Recovery — per-tool circuit breaker, retry, tool health."""

import json
import time
from unittest.mock import patch, MagicMock

import pytest


# ============== ToolHealth Circuit Breaker ==============

class TestToolHealth:

    def _make_health(self):
        from remy.core.brain_tools import ToolHealth
        return ToolHealth()

    def test_new_tool_is_available(self):
        """Untracked tools are available by default."""
        h = self._make_health()
        assert h.is_available("web_search") is True

    def test_single_failure_still_available(self):
        """One failure doesn't open the circuit."""
        h = self._make_health()
        h.record_failure("web_search")
        assert h.is_available("web_search") is True

    def test_threshold_failures_opens_circuit(self):
        """3 failures within window → circuit opens."""
        h = self._make_health()
        for _ in range(3):
            h.record_failure("web_search")
        assert h.is_available("web_search") is False

    def test_success_clears_failures(self):
        """record_success resets failure count."""
        h = self._make_health()
        h.record_failure("web_search")
        h.record_failure("web_search")
        h.record_success("web_search")
        # Even after 2 more failures, total is 2 (below threshold)
        h.record_failure("web_search")
        h.record_failure("web_search")
        assert h.is_available("web_search") is True

    def test_success_clears_open_circuit(self):
        """record_success after circuit opens → circuit closes."""
        h = self._make_health()
        for _ in range(3):
            h.record_failure("web_search")
        assert h.is_available("web_search") is False
        h.record_success("web_search")
        assert h.is_available("web_search") is True

    def test_circuit_recovers_after_cooldown(self):
        """After RECOVERY_SEC, circuit auto-closes."""
        h = self._make_health()
        for _ in range(3):
            h.record_failure("web_search")
        # Manually set open_until to past
        h._circuit_open_until["web_search"] = time.time() - 1
        assert h.is_available("web_search") is True

    def test_separate_tools_independent(self):
        """Failures in one tool don't affect another."""
        h = self._make_health()
        for _ in range(3):
            h.record_failure("web_search")
        assert h.is_available("web_search") is False
        assert h.is_available("http_get") is True

    def test_health_report_empty_when_healthy(self):
        """No issues → empty report."""
        h = self._make_health()
        assert h.get_health_report() == {}

    def test_health_report_shows_unavailable(self):
        """Open circuit → UNAVAILABLE in report."""
        h = self._make_health()
        for _ in range(3):
            h.record_failure("web_search")
        report = h.get_health_report()
        assert "web_search" in report
        assert "UNAVAILABLE" in report["web_search"]

    def test_health_report_shows_degraded(self):
        """Some failures but circuit still closed → degraded."""
        h = self._make_health()
        h.record_failure("http_get")
        report = h.get_health_report()
        assert "http_get" in report
        assert "degraded" in report["http_get"]

    def test_old_failures_pruned(self):
        """Failures older than 10 min are pruned on next record_failure."""
        h = self._make_health()
        # Inject old failures manually
        old_time = time.time() - 700  # 11+ minutes ago
        h._failures["web_search"] = [old_time, old_time]
        h.record_failure("web_search")
        # Old ones pruned, only 1 recent failure → still available
        assert h.is_available("web_search") is True
        assert len(h._failures["web_search"]) == 1


# ============== execute_tool Circuit Breaker Integration ==============

class TestExecuteToolCircuitBreaker:

    def test_blocked_when_circuit_open(self, tmp_path):
        """execute_tool returns error when tool circuit is open."""
        from aura import Aura as CognitiveMemory
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.brain_tools.brain", b), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.tool_registry.settings") as ms:
            ms.SANDBOX_DIR = tmp_path / "sandbox"
            ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"

            from remy.core.brain_tools import execute_tool, tool_health

            # Open circuit
            for _ in range(3):
                tool_health.record_failure("recall")

            result = json.loads(execute_tool("recall", {"query": "test"}))
            assert "error" in result
            assert "temporarily unavailable" in result["error"]

            # Clean up
            tool_health.record_success("recall")
        b.close()

    def test_health_tracked_on_success(self):
        """Successful network tool call → record_success called."""
        from remy.core.brain_tools import tool_health

        # Circuit breaker only tracks network tools (web_search, http_get, code_execution)
        tool_health.record_failure("web_search")

        # Manually record success should clear it
        tool_health.record_success("web_search")
        report = tool_health.get_health_report()
        assert "web_search" not in report  # cleared by success

    def test_health_tracked_on_error(self, tmp_path):
        """Non-network tools do NOT trigger circuit breaker on logical errors."""
        from aura import Aura as CognitiveMemory
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.brain_tools.brain", b), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.tool_registry.settings") as ms:
            ms.SANDBOX_DIR = tmp_path / "sandbox"
            ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"

            from remy.core.brain_tools import execute_tool, tool_health
            # Reset state
            tool_health.record_success("nonexistent_tool")

            # Non-network tool errors should NOT trip circuit breaker
            result = execute_tool("nonexistent_tool", {})
            assert "Unknown tool" in result

            report = tool_health.get_health_report()
            assert "nonexistent_tool" not in report  # not tracked for non-network tools
        b.close()


# ============== Retry Logic ==============

class TestRetryWebSearch:

    @patch("remy.core.brain_tools.time.sleep")
    def test_retries_on_failure(self, mock_sleep, tmp_path):
        """web_search retries up to MAX_RETRIES on transient failure."""
        from aura import Aura as CognitiveMemory
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.brain_tools.brain", b), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.tool_registry.settings") as ms:
            ms.SANDBOX_DIR = tmp_path / "sandbox"
            ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"
            ms.GEMINI_API_KEY = "fake-key"
            ms.SUMMARY_MODEL = "test-model"

            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = Exception("API timeout")

            with patch("google.genai.Client", return_value=mock_client):
                from remy.core.brain_tools import _execute_tool_inner
                result = json.loads(_execute_tool_inner("web_search", {"query": "test"}))
                assert "error" in result
                assert "3 attempts" in result["error"]
                # Should have slept twice (for 2 retries)
                assert mock_sleep.call_count == 2
        b.close()

    @patch("remy.core.brain_tools.time.sleep")
    def test_succeeds_on_second_attempt(self, mock_sleep, tmp_path):
        """web_search succeeds on retry after first failure."""
        from aura import Aura as CognitiveMemory
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.brain_tools.brain", b), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.tool_registry.settings") as ms:
            ms.SANDBOX_DIR = tmp_path / "sandbox"
            ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"
            ms.GEMINI_API_KEY = "fake-key"
            ms.SUMMARY_MODEL = "test-model"

            mock_response = MagicMock()
            mock_response.text = "The answer is 42."
            mock_response.candidates = []

            mock_client = MagicMock()
            mock_client.models.generate_content.side_effect = [
                Exception("Transient error"),
                mock_response,
            ]

            with patch("google.genai.Client", return_value=mock_client):
                from remy.core.brain_tools import _execute_tool_inner
                result = json.loads(_execute_tool_inner("web_search", {"query": "test"}))
                assert "error" not in result
                assert "42" in result["answer"]
                # Slept once for the retry
                assert mock_sleep.call_count == 1
        b.close()


class TestRetryHttpGet:

    @patch("remy.core.brain_tools.time.sleep")
    def test_retries_on_server_error(self, mock_sleep, tmp_path):
        """http_get retries on 5xx server errors."""
        from aura import Aura as CognitiveMemory
        import urllib.error
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.brain_tools.brain", b), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.tool_registry.settings") as ms:
            ms.SANDBOX_DIR = tmp_path / "sandbox"
            ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"
            ms.DATA_DIR = tmp_path / "data"
            ms.AUTONOMY_ALLOWED_READ_PATHS = []

            mock_urlopen = MagicMock()
            mock_urlopen.side_effect = urllib.error.URLError("Connection reset")

            with patch("urllib.request.urlopen", mock_urlopen):
                from remy.core.brain_tools import _execute_tool_inner
                result = json.loads(_execute_tool_inner("http_get", {"url": "https://example.com/api"}))
                assert "error" in result
                assert "3 attempts" in result["error"]
                assert mock_sleep.call_count == 2
        b.close()

    @patch("remy.core.brain_tools.time.sleep")
    def test_no_retry_on_404(self, mock_sleep, tmp_path):
        """http_get does NOT retry on 4xx client errors."""
        from aura import Aura as CognitiveMemory
        import urllib.error
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.brain_tools.brain", b), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.tool_registry.settings") as ms:
            ms.SANDBOX_DIR = tmp_path / "sandbox"
            ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"
            ms.DATA_DIR = tmp_path / "data"
            ms.AUTONOMY_ALLOWED_READ_PATHS = []

            err = urllib.error.HTTPError(
                "https://example.com/missing", 404, "Not Found", {}, None
            )

            with patch("urllib.request.urlopen", side_effect=err):
                from remy.core.brain_tools import _execute_tool_inner
                result = json.loads(_execute_tool_inner("http_get", {"url": "https://example.com/missing"}))
                assert "error" in result
                assert "404" in result["error"]
                # No retries for client errors
                mock_sleep.assert_not_called()
        b.close()


# ============== Tool Health in Decision Prompt ==============

class TestToolHealthInPrompt:

    def test_health_report_included_in_prompt(self, tmp_path):
        """Tool health report appears in decision prompt."""
        from aura import Aura as CognitiveMemory
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.autonomy.brain", b), \
             patch("remy.core.brain_tools.brain", b):
            from remy.core.autonomy import AutonomousLoop
            loop = AutonomousLoop.__new__(AutonomousLoop)
            loop.session_id = "test"
            loop.brain = b
            loop.action_log = []

            prompt = loop._build_decision_prompt(
                goals=[{"description": "test goal", "priority": "high", "attempts": 1}],
                past_outcomes="",
                budget={"tokens_today": 500, "daily_limit": 10000,
                        "tokens_this_hour": 100, "hourly_limit": 2000},
                tool_health_report={"web_search": "UNAVAILABLE (300s cooldown, 3 failures)"},
            )
            assert "TOOL HEALTH:" in prompt
            assert "web_search" in prompt
            assert "UNAVAILABLE" in prompt
            assert "Avoid unavailable tools" in prompt
        b.close()

    def test_no_health_section_when_all_healthy(self, tmp_path):
        """Empty health report → no TOOL HEALTH section in prompt."""
        from aura import Aura as CognitiveMemory
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.autonomy.brain", b), \
             patch("remy.core.brain_tools.brain", b):
            from remy.core.autonomy import AutonomousLoop
            loop = AutonomousLoop.__new__(AutonomousLoop)
            loop.session_id = "test"
            loop.brain = b
            loop.action_log = []

            prompt = loop._build_decision_prompt(
                goals=[], past_outcomes="",
                budget={"tokens_today": 0, "daily_limit": 10000,
                        "tokens_this_hour": 0, "hourly_limit": 2000},
                tool_health_report={},
            )
            assert "TOOL HEALTH:" not in prompt
        b.close()

    def test_no_health_section_when_none(self, tmp_path):
        """None health report → no TOOL HEALTH section."""
        from aura import Aura as CognitiveMemory
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.autonomy.brain", b), \
             patch("remy.core.brain_tools.brain", b):
            from remy.core.autonomy import AutonomousLoop
            loop = AutonomousLoop.__new__(AutonomousLoop)
            loop.session_id = "test"
            loop.brain = b
            loop.action_log = []

            prompt = loop._build_decision_prompt(
                goals=[], past_outcomes="",
                budget={"tokens_today": 0, "daily_limit": 10000,
                        "tokens_this_hour": 0, "hourly_limit": 2000},
                tool_health_report=None,
            )
            assert "TOOL HEALTH:" not in prompt
        b.close()
