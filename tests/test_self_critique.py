"""Tests for Self-Critique Loop (AUTON-13) — autonomy_critique.py + integration."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from remy.core.agent_tools import _AuraCompat as Aura


@pytest.fixture
def critique_env(tmp_path):
    """Isolated environment for critique tests."""
    brain = Aura(str(tmp_path / "critique_brain"))

    with patch("remy.core.autonomy.settings") as mock_settings:
        mock_settings.AUTONOMY_DAILY_TOKEN_LIMIT = 100_000
        mock_settings.AUTONOMY_HOURLY_TOKEN_LIMIT = 20_000
        mock_settings.AUTONOMY_SESSION_TOKEN_LIMIT = 500_000
        mock_settings.AUTONOMY_CYCLE_INTERVAL_SEC = 0
        mock_settings.AUTONOMY_TELEGRAM_NOTIFICATIONS = False
        mock_settings.AUTONOMY_SELF_CRITIQUE_ENABLED = True
        mock_settings.SUMMARY_MODEL = "test-model"
        mock_settings.GEMINI_API_KEY = "test-key"
        mock_settings.DATA_DIR = tmp_path / "data"
        (tmp_path / "data" / "logs").mkdir(parents=True, exist_ok=True)

        with (
            patch("remy.core.autonomy.brain", brain),
            patch("remy.core.autonomy_critique.brain", brain),
        ):
            yield {"brain": brain}

    brain.close()


# ============== Unit Tests: should_critique ==============


class TestShouldCritique:
    """Tests for should_critique() decision logic."""

    def test_no_tool_calls_skips(self):
        from remy.core.autonomy_critique import should_critique

        log = [{"type": "text", "content": "hello"}]
        assert should_critique(log) is False

    def test_empty_log_skips(self):
        from remy.core.autonomy_critique import should_critique

        assert should_critique([]) is False

    def test_trivial_tools_skip(self):
        from remy.core.autonomy_critique import should_critique

        log = [
            {"type": "tool_call", "tool": "recall", "result": "data"},
            {"type": "tool_call", "tool": "get_current_datetime", "result": "2026-01-01"},
        ]
        assert should_critique(log) is False

    def test_high_impact_tool_triggers(self):
        from remy.core.autonomy_critique import should_critique

        log = [
            {"type": "tool_call", "tool": "write_file", "result": "ok"},
        ]
        assert should_critique(log) is True

    def test_mixed_tools_trigger(self):
        from remy.core.autonomy_critique import should_critique

        log = [
            {"type": "tool_call", "tool": "recall", "result": "data"},
            {"type": "tool_call", "tool": "web_search", "result": "results"},
        ]
        assert should_critique(log) is True

    def test_always_critique_tools(self):
        """Every tool in ALWAYS_CRITIQUE_TOOLS should trigger critique."""
        from remy.core.autonomy_critique import ALWAYS_CRITIQUE_TOOLS, should_critique

        for tool_name in ALWAYS_CRITIQUE_TOOLS:
            log = [{"type": "tool_call", "tool": tool_name, "result": "ok"}]
            assert should_critique(log) is True, f"{tool_name} should trigger critique"

    def test_skip_critique_tools_alone(self):
        """Every tool in SKIP_CRITIQUE_TOOLS alone should NOT trigger critique."""
        from remy.core.autonomy_critique import SKIP_CRITIQUE_TOOLS, should_critique

        for tool_name in SKIP_CRITIQUE_TOOLS:
            log = [{"type": "tool_call", "tool": tool_name, "result": "ok"}]
            assert should_critique(log) is False, f"{tool_name} alone should skip critique"


# ============== Unit Tests: critique_response ==============


class TestCritiqueResponse:
    """Tests for critique_response() LLM analysis."""

    @pytest.mark.asyncio
    async def test_good_quality_response(self, critique_env):
        from remy.core.autonomy_critique import critique_response

        mock_llm = MagicMock()
        mock_llm.content = json.dumps(
            {
                "quality": 0.9,
                "issues": [],
                "suggestions": [],
                "should_retry": False,
            }
        )

        with patch("remy.core.llm.call_llm", return_value=mock_llm):
            result = await critique_response(
                "Research AI safety",
                "Find information about AI safety",
                "I found 3 papers on AI safety and stored them.",
                [{"type": "tool_call", "tool": "web_search", "result": "3 results"}],
            )

        assert result["quality"] == 0.9
        assert result["issues"] == []
        assert result["should_retry"] is False

    @pytest.mark.asyncio
    async def test_low_quality_with_issues(self, critique_env):
        from remy.core.autonomy_critique import critique_response

        mock_llm = MagicMock()
        mock_llm.content = json.dumps(
            {
                "quality": 0.2,
                "issues": ["Agent claimed to store data but no store tool was called"],
                "suggestions": ["Call the store tool explicitly"],
                "should_retry": True,
            }
        )

        with patch("remy.core.llm.call_llm", return_value=mock_llm):
            result = await critique_response(
                "Store findings",
                "Store the research findings",
                "I stored all findings successfully!",
                [{"type": "tool_call", "tool": "web_search", "result": "results"}],
            )

        assert result["quality"] == 0.2
        assert len(result["issues"]) == 1
        assert result["should_retry"] is True

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self, critique_env):
        from remy.core.autonomy_critique import critique_response

        with patch("remy.core.llm.call_llm", side_effect=Exception("API down")):
            result = await critique_response(
                "goal",
                "prompt",
                "response",
                [{"type": "tool_call", "tool": "store", "result": "ok"}],
            )

        # Should not crash — returns default pass-through
        assert result["quality"] == 0.5
        assert result["should_retry"] is False
        assert "Critique unavailable" in result["critique_text"]

    @pytest.mark.asyncio
    async def test_handles_markdown_fenced_json(self, critique_env):
        from remy.core.autonomy_critique import critique_response

        mock_llm = MagicMock()
        mock_llm.content = '```json\n{"quality": 0.6, "issues": ["minor"], "suggestions": [], "should_retry": false}\n```'

        with patch("remy.core.llm.call_llm", return_value=mock_llm):
            result = await critique_response(
                "goal",
                "prompt",
                "response",
                [{"type": "tool_call", "tool": "browse_page", "result": "ok"}],
            )

        assert result["quality"] == 0.6
        assert result["issues"] == ["minor"]

    @pytest.mark.asyncio
    async def test_quality_clamped_to_range(self, critique_env):
        from remy.core.autonomy_critique import critique_response

        mock_llm = MagicMock()
        mock_llm.content = json.dumps(
            {
                "quality": 1.5,  # Out of range
                "issues": [],
                "suggestions": [],
                "should_retry": False,
            }
        )

        with patch("remy.core.llm.call_llm", return_value=mock_llm):
            result = await critique_response(
                "goal",
                "prompt",
                "response",
                [{"type": "tool_call", "tool": "store", "result": "ok"}],
            )

        assert result["quality"] == 1.0  # Clamped

    @pytest.mark.asyncio
    async def test_issues_capped_at_five(self, critique_env):
        from remy.core.autonomy_critique import critique_response

        mock_llm = MagicMock()
        mock_llm.content = json.dumps(
            {
                "quality": 0.1,
                "issues": [f"issue-{i}" for i in range(10)],
                "suggestions": [f"sug-{i}" for i in range(10)],
                "should_retry": False,
            }
        )

        with patch("remy.core.llm.call_llm", return_value=mock_llm):
            result = await critique_response(
                "goal",
                "prompt",
                "response",
                [{"type": "tool_call", "tool": "store", "result": "ok"}],
            )

        assert len(result["issues"]) == 5
        assert len(result["suggestions"]) == 3

    @pytest.mark.asyncio
    async def test_empty_llm_response(self, critique_env):
        from remy.core.autonomy_critique import critique_response

        mock_llm = MagicMock()
        mock_llm.content = ""

        with patch("remy.core.llm.call_llm", return_value=mock_llm):
            result = await critique_response(
                "goal",
                "prompt",
                "response",
                [{"type": "tool_call", "tool": "store", "result": "ok"}],
            )

        assert result["quality"] == 0.5  # Default
        assert result["should_retry"] is False


# ============== Unit Tests: store_critique ==============


class TestStoreCritique:
    """Tests for store_critique() brain persistence."""

    def test_low_quality_critique_stored(self, critique_env):
        from remy.core.autonomy_critique import store_critique

        critique = {
            "quality": 0.3,
            "issues": ["hallucinated tool call"],
            "suggestions": ["actually call the tool"],
            "should_retry": True,
            "critique_text": "test",
        }

        record_id = store_critique(critique, "test goal", "action-123")
        assert record_id is not None

    def test_high_quality_critique_skipped(self, critique_env):
        from remy.core.autonomy_critique import store_critique

        critique = {
            "quality": 0.9,
            "issues": [],
            "suggestions": [],
            "should_retry": False,
            "critique_text": "all good",
        }

        record_id = store_critique(critique, "test goal", "action-456")
        assert record_id is None  # Not stored — no noise

    def test_borderline_quality_with_issues_stored(self, critique_env):
        from remy.core.autonomy_critique import store_critique

        critique = {
            "quality": 0.65,
            "issues": ["minor formatting issue"],
            "suggestions": [],
            "should_retry": False,
            "critique_text": "mostly ok",
        }

        record_id = store_critique(critique, "test goal", "action-789")
        assert record_id is not None  # Has issues → stored


# ============== Unit Tests: extract_tool_summary ==============


class TestExtractToolSummary:
    """Tests for _extract_tool_summary()."""

    def test_empty_log(self):
        from remy.core.autonomy_critique import _extract_tool_summary

        assert _extract_tool_summary([]) == "(no tool calls)"

    def test_with_tool_calls(self):
        from remy.core.autonomy_critique import _extract_tool_summary

        log = [
            {
                "type": "tool_call",
                "tool": "web_search",
                "args": {"query": "AI"},
                "result": "3 results",
            },
            {"type": "tool_call", "tool": "store", "args": {"content": "data"}, "result": "stored"},
        ]
        summary = _extract_tool_summary(log)
        assert "web_search" in summary
        assert "store" in summary

    def test_caps_at_max_tools(self):
        from remy.core.autonomy_critique import _extract_tool_summary

        log = [{"type": "tool_call", "tool": f"tool_{i}", "result": f"r{i}"} for i in range(20)]
        summary = _extract_tool_summary(log, max_tools=3)
        # Should only contain last 3 tools
        assert "tool_17" in summary
        assert "tool_18" in summary
        assert "tool_19" in summary
        assert "tool_0" not in summary


# ============== Integration Tests: critique in _decide_and_act ==============


class TestCritiqueIntegration:
    """Tests that critique integrates correctly into the autonomous cycle."""

    @pytest.mark.asyncio
    async def test_critique_runs_on_high_impact_tools(self, critique_env):
        """Critique should run when session_log contains high-impact tool calls."""
        from remy.core.autonomy import AutonomousLoop, create_goal

        create_goal("Write a report", priority="medium")
        loop = AutonomousLoop()

        session_log_with_write = [
            {
                "type": "tool_call",
                "tool": "write_file",
                "args": {"path": "report.txt"},
                "result": "ok",
            },
        ]

        with patch("remy.core.agent.invoke_agent", new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = ("Report written successfully!", [], session_log_with_write)

            # Mock evaluation
            eval_result = {
                "success": True,
                "confidence": 0.9,
                "reason": "Report written",
                "goal_completed": False,
            }
            with patch.object(
                loop, "_evaluate_outcome", new_callable=AsyncMock, return_value=eval_result
            ):
                # Mock critique to return good quality
                mock_critique_llm = MagicMock()
                mock_critique_llm.content = json.dumps(
                    {
                        "quality": 0.85,
                        "issues": [],
                        "suggestions": [],
                        "should_retry": False,
                    }
                )

                with patch("remy.core.llm.call_llm", return_value=mock_critique_llm):
                    loop.running = True
                    action = await loop._decide_and_act()

        assert action is not None
        assert action.success is True

    @pytest.mark.asyncio
    async def test_critique_skipped_for_trivial_tools(self, critique_env):
        """Critique should NOT run when only trivial tools were called."""
        from remy.core.autonomy import AutonomousLoop, create_goal

        create_goal("Recall data", priority="medium")
        loop = AutonomousLoop()

        trivial_log = [
            {"type": "tool_call", "tool": "recall", "args": {"query": "test"}, "result": "data"},
        ]

        with patch("remy.core.agent.invoke_agent", new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = ("Recalled some data.", [], trivial_log)

            eval_result = {
                "success": True,
                "confidence": 0.8,
                "reason": "Recalled data",
                "goal_completed": False,
            }
            with patch.object(
                loop, "_evaluate_outcome", new_callable=AsyncMock, return_value=eval_result
            ):
                # Critique LLM should NOT be called
                with patch("remy.core.llm.call_llm") as mock_llm:
                    loop.running = True
                    action = await loop._decide_and_act()

                    # critique_response should not have been called (trivial tools)
                    # The only LLM call would be from _evaluate_outcome which is mocked
                    mock_llm.assert_not_called()

        assert action.success is True

    @pytest.mark.asyncio
    async def test_low_critique_downgrades_evaluation(self, critique_env):
        """Very low critique quality should downgrade a successful evaluation."""
        from remy.core.autonomy import AutonomousLoop, create_goal

        create_goal("Store important data", priority="high")
        loop = AutonomousLoop()

        log_with_store = [
            {"type": "tool_call", "tool": "store", "args": {"content": "data"}, "result": "ok"},
        ]

        with patch("remy.core.agent.invoke_agent", new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = ("Stored data!", [], log_with_store)

            # Evaluation says success
            eval_result = {
                "success": True,
                "confidence": 0.8,
                "reason": "Data stored",
                "goal_completed": False,
            }
            with patch.object(
                loop, "_evaluate_outcome", new_callable=AsyncMock, return_value=eval_result
            ):
                # But critique says very low quality
                mock_critique_llm = MagicMock()
                mock_critique_llm.content = json.dumps(
                    {
                        "quality": 0.2,
                        "issues": ["Agent claims success but stored garbage data"],
                        "suggestions": ["Verify data before storing"],
                        "should_retry": False,
                    }
                )

                with patch("remy.core.llm.call_llm", return_value=mock_critique_llm):
                    loop.running = True
                    action = await loop._decide_and_act()

        # Critique should have downgraded the success
        assert action.success is False

    @pytest.mark.asyncio
    async def test_critique_disabled_by_setting(self, critique_env):
        """When AUTONOMY_SELF_CRITIQUE_ENABLED=False, critique should be skipped entirely."""
        from remy.core.autonomy import AutonomousLoop, create_goal

        create_goal("Write something", priority="medium")
        loop = AutonomousLoop()

        log_with_write = [
            {"type": "tool_call", "tool": "write_file", "args": {}, "result": "ok"},
        ]

        with patch("remy.core.autonomy.settings") as mock_settings:
            mock_settings.AUTONOMY_SELF_CRITIQUE_ENABLED = False
            mock_settings.AUTONOMY_DAILY_TOKEN_LIMIT = 100_000
            mock_settings.AUTONOMY_HOURLY_TOKEN_LIMIT = 20_000
            mock_settings.AUTONOMY_SESSION_TOKEN_LIMIT = 500_000

            with patch("remy.core.agent.invoke_agent", new_callable=AsyncMock) as mock_agent:
                mock_agent.return_value = ("Done!", [], log_with_write)

                eval_result = {
                    "success": True,
                    "confidence": 0.9,
                    "reason": "Done",
                    "goal_completed": False,
                }
                with patch.object(
                    loop, "_evaluate_outcome", new_callable=AsyncMock, return_value=eval_result
                ):
                    with patch("remy.core.llm.call_llm") as mock_llm:
                        loop.running = True
                        action = await loop._decide_and_act()

                        # No critique LLM call
                        mock_llm.assert_not_called()

        assert action.success is True

    @pytest.mark.asyncio
    async def test_critique_retry_loop(self, critique_env):
        """When critique says should_retry with low quality, agent should retry."""
        from remy.core.autonomy import AutonomousLoop, create_goal

        create_goal("Complex task", priority="high")
        loop = AutonomousLoop()

        log_with_browse = [
            {
                "type": "tool_call",
                "tool": "browse_page",
                "args": {"url": "http://example.com"},
                "result": "error",
            },
        ]

        call_count = 0

        def make_critique_response(quality, should_retry):
            mock = MagicMock()
            mock.content = json.dumps(
                {
                    "quality": quality,
                    "issues": ["browser error"] if quality < 0.5 else [],
                    "suggestions": ["try different URL"] if should_retry else [],
                    "should_retry": should_retry,
                }
            )
            return mock

        critique_responses = [
            make_critique_response(0.2, True),  # First: low quality, retry
            make_critique_response(0.7, False),  # Second: good quality, stop
        ]

        async def mock_invoke(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return (f"Attempt {call_count}", [], log_with_browse)

        with patch("remy.core.agent.invoke_agent", new_callable=AsyncMock, side_effect=mock_invoke):
            eval_result = {
                "success": True,
                "confidence": 0.8,
                "reason": "Success",
                "goal_completed": False,
            }
            with patch.object(
                loop, "_evaluate_outcome", new_callable=AsyncMock, return_value=eval_result
            ):
                with patch("remy.core.llm.call_llm", side_effect=critique_responses):
                    loop.running = True
                    action = await loop._decide_and_act()

        # Should have been called twice: initial + 1 retry
        assert call_count == 2
        assert action is not None


# ============== Unit Tests: estimate_critique_tokens ==============


class TestEstimateCritiqueTokens:
    """Tests for token estimation."""

    def test_returns_positive_int(self):
        from remy.core.autonomy_critique import estimate_critique_tokens

        tokens = estimate_critique_tokens(
            "goal",
            "prompt" * 100,
            "response" * 50,
            [{"type": "tool_call", "tool": "store", "result": "ok"}],
        )
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_empty_inputs(self):
        from remy.core.autonomy_critique import estimate_critique_tokens

        tokens = estimate_critique_tokens("", "", "", [])
        assert tokens > 0  # At least overhead
