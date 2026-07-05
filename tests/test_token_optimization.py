"""Tests for RM-10: Token Optimization — web search cache + budget-aware annotations."""

import json
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest
from aura import Aura as CognitiveMemory, Level


# ============== Web Search Cache ==============

class TestGetCachedSearch:

    def test_cache_miss_empty_brain(self, tmp_path):
        """No cached results in empty brain → returns None."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        with patch("remy.core.brain_tools.brain", b):
            from remy.core.brain_tools import _get_cached_search
            result = _get_cached_search("latest news")
            assert result is None
        b.close()

    def test_cache_hit_exact_match(self, tmp_path):
        """Exact query match within TTL → returns cached result."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        with patch("remy.core.brain_tools.brain", b):
            from remy.core.brain_tools import (
                _get_cached_search, _cache_search_result,
            )
            # Store a result
            _cache_search_result(
                "weather forecast",
                "It will be sunny tomorrow.",
                [{"title": "Weather.com", "uri": "https://weather.com"}],
            )
            # Should find it
            result = _get_cached_search("weather forecast")
            assert result is not None
            assert result["cached"] is True
            assert "sunny" in result["answer"]
            assert len(result["sources"]) == 1
        b.close()

    def test_cache_hit_case_insensitive(self, tmp_path):
        """Query matching is case-insensitive."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        with patch("remy.core.brain_tools.brain", b):
            from remy.core.brain_tools import (
                _get_cached_search, _cache_search_result,
            )
            _cache_search_result("Python Tutorials", "Learn Python.", [])
            result = _get_cached_search("python tutorials")
            assert result is not None
            assert result["cached"] is True
        b.close()

    def test_cache_miss_different_query(self, tmp_path):
        """Different query → returns None (exact match only)."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        with patch("remy.core.brain_tools.brain", b):
            from remy.core.brain_tools import (
                _get_cached_search, _cache_search_result,
            )
            _cache_search_result("weather forecast", "Sunny.", [])
            result = _get_cached_search("sports news")
            assert result is None
        b.close()

    def test_cache_expired_ttl(self, tmp_path):
        """Cached result older than TTL → returns None."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        with patch("remy.core.brain_tools.brain", b):
            from remy.core.brain_tools import (
                _get_cached_search, _SEARCH_CACHE_TAG,
            )
            # Manually store a record with old timestamp
            old_time = (datetime.now() - timedelta(hours=25)).isoformat()
            b.store(
                content="Web search: old query\nOld answer",
                level=Level.WORKING,
                tags=[_SEARCH_CACHE_TAG],
                metadata={
                    "type": "web_search_cache",
                    "query": "old query",
                    "answer": "Old answer",
                    "sources": [],
                    "cached_at": old_time,
                },
                deduplicate=False,
            )
            result = _get_cached_search("old query")
            assert result is None
        b.close()

    def test_cache_exception_returns_none(self, tmp_path):
        """Exception during cache check → returns None gracefully."""
        mock_brain = MagicMock()
        mock_brain.search.side_effect = Exception("DB error")
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import _get_cached_search
            result = _get_cached_search("any query")
            assert result is None


class TestCacheSearchResult:

    def test_stores_to_brain(self, tmp_path):
        """Cache result creates a brain record with correct tags and metadata."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        with patch("remy.core.brain_tools.brain", b):
            from remy.core.brain_tools import (
                _cache_search_result, _SEARCH_CACHE_TAG,
            )
            _cache_search_result(
                "test query",
                "Answer text here.",
                [{"title": "Source", "uri": "https://example.com"}],
            )
            records = b.search(query="", tags=[_SEARCH_CACHE_TAG], limit=5)
            assert len(records) >= 1
            meta = records[0].metadata
            assert meta["query"] == "test query"
            assert meta["type"] == "web_search_cache"
            assert "cached_at" in meta
        b.close()

    def test_truncates_long_answer(self, tmp_path):
        """Answers longer than 500 chars are truncated."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        with patch("remy.core.brain_tools.brain", b):
            from remy.core.brain_tools import (
                _cache_search_result, _SEARCH_CACHE_TAG,
            )
            long_answer = "x" * 1000
            _cache_search_result("long", long_answer, [])
            records = b.search(query="", tags=[_SEARCH_CACHE_TAG], limit=5)
            assert len(records) >= 1
            assert len(records[0].metadata["answer"]) == 500
        b.close()

    def test_exception_does_not_propagate(self):
        """Exception during cache store → swallowed, no crash."""
        mock_brain = MagicMock()
        mock_brain.store.side_effect = Exception("Write error")
        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import _cache_search_result
            # Should not raise
            _cache_search_result("q", "a", [])


# ============== Web Search with Cache Integration ==============

class TestWebSearchCacheIntegration:

    def test_web_search_returns_cached_on_hit(self, tmp_path):
        """web_search returns cached result without calling Gemini API."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        with patch("remy.core.brain_tools.brain", b), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.tool_registry.settings") as ms, \
             patch("remy.core.brain_tools.tool_health") as mh:
            ms.SANDBOX_DIR = tmp_path / "sandbox"
            ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"
            mh.is_available.return_value = True

            from remy.core.brain_tools import (
                _cache_search_result, _execute_tool_inner,
            )
            _cache_search_result("cached query", "Cached answer!", [{"title": "S", "uri": "u"}])

            result = json.loads(_execute_tool_inner("web_search", {"query": "cached query"}))
            assert result["cached"] is True
            assert "Cached answer" in result["answer"]


# ============== Consolidation Skips Cache Tags ==============

class TestConsolidationSkipsCache:

    def test_web_search_cache_tag_in_skip_list(self):
        """web-search-cache tag should be in _CONSOLIDATION_SKIP_TAGS."""
        from remy.core.background_brain import _CONSOLIDATION_SKIP_TAGS
        assert "web-search-cache" in _CONSOLIDATION_SKIP_TAGS


# ============== Budget-Aware Tool Annotations ==============

class TestToolCostAnnotations:

    def test_decision_prompt_contains_tool_costs(self, tmp_path):
        """_build_decision_prompt includes TOOL COSTS section."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        with patch("remy.core.autonomy.brain", b), \
             patch("remy.core.brain_tools.brain", b):
            from remy.core.autonomy import AutonomousLoop
            loop = AutonomousLoop.__new__(AutonomousLoop)
            loop.session_id = "test"
            loop.brain = b
            loop.action_log = []

            prompt = loop._build_decision_prompt(
                goals=[{"description": "test", "priority": "medium", "attempts": 0}],
                past_outcomes="",
                budget={"tokens_today": 100, "daily_limit": 10000,
                        "tokens_this_hour": 50, "hourly_limit": 2000},
            )
            assert "TOOL COSTS" in prompt
            assert "web_search (~800)" in prompt
            assert "recall (~50)" in prompt
            assert "EXPENSIVE" in prompt
        b.close()

    def test_decision_prompt_recall_before_search_instruction(self, tmp_path):
        """Prompt instructs to use recall before web_search."""
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
            )
            assert "recall before web_search" in prompt.lower() or "recall before web_search" in prompt
        b.close()
