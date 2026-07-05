"""Tests for Selective Recall Cache Invalidation (v2.3, Rec 14.4)."""


# ============== Unit Tests: selective invalidation ==============


class TestSelectiveInvalidation:
    def setup_method(self):
        from remy.core.tool_utils import _recall_cache

        _recall_cache.clear()

    def test_empty_content_clears_all(self):
        from remy.core.tool_utils import _cache_recall_result, _recall_cache, clear_recall_cache

        _cache_recall_result("ai safety research", "result1")
        _cache_recall_result("blood pressure data", "result2")
        assert len(_recall_cache) == 2

        clear_recall_cache("")  # empty → full clear
        assert len(_recall_cache) == 0

    def test_no_content_clears_all(self):
        from remy.core.tool_utils import _cache_recall_result, _recall_cache, clear_recall_cache

        _cache_recall_result("ai safety", "r1")
        _cache_recall_result("health data", "r2")
        clear_recall_cache()  # no arg → full clear
        assert len(_recall_cache) == 0

    def test_selective_removes_related(self):
        from remy.core.tool_utils import _cache_recall_result, _recall_cache, clear_recall_cache

        _cache_recall_result("ai safety research", "result1")
        _cache_recall_result("blood pressure data", "result2")
        _cache_recall_result("cooking recipes", "result3")

        # Store something about AI → only "ai safety research" should be invalidated
        clear_recall_cache("New AI model discovered with safety implications")
        assert len(_recall_cache) == 2
        assert "ai safety research" not in _recall_cache
        assert "blood pressure data" in _recall_cache
        assert "cooking recipes" in _recall_cache

    def test_selective_preserves_unrelated(self):
        from remy.core.tool_utils import _cache_recall_result, _recall_cache, clear_recall_cache

        _cache_recall_result("python programming", "r1")
        _cache_recall_result("weather forecast", "r2")
        _cache_recall_result("math equations", "r3")

        clear_recall_cache("Today's blood pressure reading is 120/80")
        # None of these should match
        assert len(_recall_cache) == 3

    def test_selective_case_insensitive(self):
        from remy.core.tool_utils import _cache_recall_result, _recall_cache, clear_recall_cache

        _cache_recall_result("health data", "r1")  # key stored lowercase
        clear_recall_cache("New HEALTH metric recorded")  # content has uppercase
        assert "health data" not in _recall_cache

    def test_multiple_keywords_match(self):
        from remy.core.tool_utils import _cache_recall_result, _recall_cache, clear_recall_cache

        _cache_recall_result("recent research papers", "r1")
        _cache_recall_result("old research notes", "r2")
        _cache_recall_result("shopping list", "r3")

        clear_recall_cache("Found important research on quantum computing")
        # Both "research" entries should be invalidated
        assert "recent research papers" not in _recall_cache
        assert "old research notes" not in _recall_cache
        assert "shopping list" in _recall_cache

    def test_only_first_10_keywords(self):
        from remy.core.tool_utils import _cache_recall_result, _recall_cache, clear_recall_cache

        _cache_recall_result("unicorn data", "r1")

        # "unicorn" is the 15th word — should NOT match
        content = "a b c d e f g h i j k l m n unicorn"
        clear_recall_cache(content)
        # "unicorn data" should survive because "unicorn" is beyond first 10 words
        assert "unicorn data" in _recall_cache

    def test_all_entries_affected_clears_all(self):
        from remy.core.tool_utils import _cache_recall_result, _recall_cache, clear_recall_cache

        _cache_recall_result("data analysis", "r1")
        _cache_recall_result("data science", "r2")

        # "data" matches all entries → full clear
        clear_recall_cache("data processing pipeline")
        assert len(_recall_cache) == 0


# ============== Unit Tests: cache still works after selective clear ==============


class TestCacheAfterSelectiveClear:
    def setup_method(self):
        from remy.core.tool_utils import _recall_cache

        _recall_cache.clear()

    def test_cache_hit_after_selective(self):
        from remy.core.tool_utils import (
            _cache_recall_result,
            _get_cached_recall,
            clear_recall_cache,
        )

        _cache_recall_result("weather forecast", "sunny")
        _cache_recall_result("health metrics", "bp 120/80")

        # Invalidate health-related
        clear_recall_cache("new health data stored")

        # Weather should still be cached
        assert _get_cached_recall("weather forecast") == "sunny"
        # Health should be gone
        assert _get_cached_recall("health metrics") is None

    def test_new_cache_entries_after_clear(self):
        from remy.core.tool_utils import (
            _cache_recall_result,
            _get_cached_recall,
            clear_recall_cache,
        )

        _cache_recall_result("old query", "old result")
        clear_recall_cache("old data updated")

        _cache_recall_result("old query", "new result")
        assert _get_cached_recall("old query") == "new result"
