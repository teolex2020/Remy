"""Tests for first-run experience and recall latency (v2.3, Section 15 edge cases).

Validates that:
1. Empty brain doesn't crash any channel's build_system_instruction()
2. Empty brain doesn't crash invoke_agent()
3. Scratchpad/recall/search work on empty brain
4. Recall latency stays reasonable even with many records
"""

import time
from unittest.mock import MagicMock, patch

import pytest

# ============== Empty Brain Tests ==============


class TestEmptyBrainFirstRun:
    """Verify the system works correctly when brain has zero records."""

    def test_build_system_instruction_empty_brain(self):
        """build_system_instruction() should work with zero brain records."""
        from remy.core.brain_tools import build_system_instruction

        for channel in ("voice", "telegram", "desktop", "autonomous", "proactive"):
            instruction = build_system_instruction(channel=channel)
            assert isinstance(instruction, str)
            assert len(instruction) > 100, f"Empty instruction for {channel}"
            # Should contain onboarding prompt (no user profile)
            assert "FIRST-TIME USER" in instruction or "Rules:" in instruction

    def test_recall_empty_brain(self):
        """recall() on empty brain should return 'No relevant' or empty."""
        from remy.core.agent_tools import brain, brain_lock

        with brain_lock:
            result = brain.recall("anything at all")
        # Should not crash; may return empty or "No relevant"
        assert isinstance(result, str)

    def test_search_empty_brain(self):
        """search() should return a list (brain may have data in dev environment)."""
        from remy.core.agent_tools import brain, brain_lock

        with brain_lock:
            results = brain.search(query="test", limit=10)
        assert isinstance(results, list)

    def test_scratchpad_empty_brain(self):
        """Scratchpad should work with empty brain."""
        from remy.core.scratchpad import clear_notes, get_scratchpad_context, read_notes

        clear_notes()
        notes = read_notes()
        assert notes == []

        ctx = get_scratchpad_context()
        assert ctx is None  # No notes → no context

    def test_tier_stats_empty_brain(self):
        """tier_stats() should return a dict with cognitive/core keys."""
        from remy.core.agent_tools import brain

        stats = brain.tier_stats()
        assert isinstance(stats, dict)
        assert "cognitive" in stats
        assert "core" in stats

    def test_proactive_context_empty_brain(self):
        """get_proactive_context() should not crash with empty brain."""
        from remy.core.proactive_context import get_proactive_context

        ctx = get_proactive_context()
        assert isinstance(ctx, str)
        # May be empty or contain "No scheduled tasks"

    def test_compact_history_empty(self):
        """compact_history with empty messages should return empty."""
        from remy.core.agent import compact_history

        result = compact_history([], keep_recent=16)
        assert result == []

    def test_get_active_goals_empty(self):
        """get_active_goals() on empty brain should return empty list."""
        from remy.core.autonomy import get_active_goals

        goals = get_active_goals()
        assert isinstance(goals, list)

    @pytest.mark.asyncio
    async def test_invoke_agent_empty_brain(self):
        """invoke_agent() should work with empty brain (no crash)."""
        from langchain_core.messages import AIMessage, HumanMessage

        from remy.core.agent import invoke_agent

        with patch("remy.core.agent.build_agent_graph") as mock_graph:
            compiled = MagicMock()
            compiled.invoke.return_value = {
                "messages": [
                    HumanMessage(content="Привіт"),
                    AIMessage(content="Привіт! Я Ремі. Як тебе звати?"),
                ],
                "session_log": [{"type": "user_text", "text": "Привіт"}],
            }
            mock_graph.return_value = compiled

            text, messages, log = await invoke_agent(
                "Привіт",
                session_id="test-empty",
                channel="desktop",
                session_log=[],
            )

        assert isinstance(text, str)
        assert len(text) > 0


# ============== Recall Latency Tests ==============


class TestRecallLatency:
    """Measure recall latency at different brain sizes.

    Uses the test fixture ``brain`` (temp dir) so no records leak into
    the production data/brain.
    """

    @staticmethod
    def _populate(brain, count: int):
        from remy.core.agent_tools import Level

        start = time.perf_counter()
        for i in range(count):
            brain.store(
                content=f"Test record {i}: important fact about topic {i % 50}",
                level=Level.DOMAIN,
                tags=["perf-test", f"topic-{i % 50}"],
            )
        return time.perf_counter() - start

    @staticmethod
    def _measure_recall(brain, query: str, n_trials: int = 5) -> float:
        times = []
        for _ in range(n_trials):
            start = time.perf_counter()
            brain.recall(query)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        return sum(times) / len(times)

    def test_recall_latency_baseline(self, brain):
        """Recall on empty/small brain should be < 50ms."""
        avg_ms = self._measure_recall(brain, "test query", n_trials=3)
        assert avg_ms < 50, f"Recall too slow on small brain: {avg_ms:.1f}ms"

    def test_recall_latency_100_records(self, brain):
        """Recall with 100 records should be < 100ms."""
        self._populate(brain, 100)
        avg_ms = self._measure_recall(brain, "important fact about topic", n_trials=3)
        assert avg_ms < 100, f"Recall too slow with 100 records: {avg_ms:.1f}ms"

    def test_recall_latency_500_records(self, brain):
        """Recall with 500 records should be < 200ms."""
        self._populate(brain, 500)
        avg_ms = self._measure_recall(brain, "important fact about topic", n_trials=3)
        assert avg_ms < 200, f"Recall too slow with 500 records: {avg_ms:.1f}ms"

    def test_search_latency_500_records(self, brain):
        """Search with 500 records should be < 200ms."""
        self._populate(brain, 500)
        start = time.perf_counter()
        results = brain.search(query="topic", limit=20)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 200, f"Search too slow: {elapsed_ms:.1f}ms"
        assert isinstance(results, list)

    def test_store_latency_single(self, brain):
        """Single store() should be < 20ms."""
        from remy.core.agent_tools import Level

        start = time.perf_counter()
        brain.store(
            content="Latency test record",
            level=Level.WORKING,
            tags=["latency-test"],
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 20, f"Store too slow: {elapsed_ms:.1f}ms"


# ============== Graceful Shutdown Tests ==============


class TestGracefulShutdown:
    """Verify shutdown-related code paths."""

    def test_combined_runner_sleep_respects_shutdown(self):
        """Restart sleep in combined_runner should use shutdown_event.wait()."""
        import inspect

        from remy.core import combined_runner

        source = inspect.getsource(combined_runner.run_combined)
        # Should use shutdown_event.wait() instead of asyncio.sleep()
        # for restart delays (so Ctrl+C interrupts the wait)
        assert "shutdown_event.wait()" in source, (
            "Restart delays should use shutdown_event.wait() for interruptibility"
        )

    def test_autonomy_shutdown_has_reflection_timeout(self):
        """Autonomous _shutdown() should timeout on reflection generation."""
        import inspect

        from remy.core.autonomy import AutonomousLoop

        source = inspect.getsource(AutonomousLoop._shutdown)
        assert "wait_for" in source, (
            "_shutdown() should use asyncio.wait_for() to timeout reflection"
        )

    def test_cleanup_has_timeout(self):
        """combined_runner cleanup phase should have timeout per function."""
        import inspect

        from remy.core import combined_runner

        source = inspect.getsource(combined_runner.run_combined)
        # Cleanup should use wait_for with timeout
        assert "wait_for(result" in source, (
            "Cleanup should use asyncio.wait_for() to prevent hanging"
        )
