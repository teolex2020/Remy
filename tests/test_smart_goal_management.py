"""Tests for Smart Goal Management (AUTON-8) — smart_goals.py."""

from datetime import datetime, timedelta
from unittest.mock import patch

# ============== Unit Tests: Dependency Graph ==============


class TestDependencyGraph:
    def test_empty_goals(self):
        from remy.core.smart_goals import get_dependency_graph

        assert get_dependency_graph([]) == {}

    def test_no_dependencies(self):
        from remy.core.smart_goals import get_dependency_graph

        goals = [
            {"goal_id": "g1", "description": "A"},
            {"goal_id": "g2", "description": "B"},
        ]
        graph = get_dependency_graph(goals)
        assert graph == {"g1": [], "g2": []}

    def test_with_dependencies(self):
        from remy.core.smart_goals import get_dependency_graph

        goals = [
            {"goal_id": "g1", "description": "A"},
            {"goal_id": "g2", "description": "B", "depends_on": ["g1"]},
        ]
        graph = get_dependency_graph(goals)
        assert graph["g2"] == ["g1"]

    def test_string_depends_on(self):
        from remy.core.smart_goals import get_dependency_graph

        goals = [
            {"goal_id": "g1", "description": "A"},
            {"goal_id": "g2", "description": "B", "depends_on": "g1"},
        ]
        graph = get_dependency_graph(goals)
        assert graph["g2"] == ["g1"]


# ============== Unit Tests: Topological Sort ==============


class TestTopologicalSort:
    def test_empty(self):
        from remy.core.smart_goals import topological_sort_goals

        assert topological_sort_goals([]) == []

    def test_no_deps_preserves_order(self):
        from remy.core.smart_goals import topological_sort_goals

        goals = [
            {"goal_id": "g1", "description": "First"},
            {"goal_id": "g2", "description": "Second"},
        ]
        result = topological_sort_goals(goals)
        assert result[0]["goal_id"] == "g1"

    def test_dependency_order(self):
        from remy.core.smart_goals import topological_sort_goals

        goals = [
            {"goal_id": "g2", "description": "Depends on g1", "depends_on": ["g1"]},
            {"goal_id": "g1", "description": "No deps"},
        ]
        result = topological_sort_goals(goals)
        ids = [g["goal_id"] for g in result]
        assert ids.index("g1") < ids.index("g2")

    def test_chain_dependency(self):
        from remy.core.smart_goals import topological_sort_goals

        goals = [
            {"goal_id": "g3", "description": "C", "depends_on": ["g2"]},
            {"goal_id": "g1", "description": "A"},
            {"goal_id": "g2", "description": "B", "depends_on": ["g1"]},
        ]
        result = topological_sort_goals(goals)
        ids = [g["goal_id"] for g in result]
        assert ids.index("g1") < ids.index("g2") < ids.index("g3")

    def test_cycle_handled_gracefully(self):
        from remy.core.smart_goals import topological_sort_goals

        goals = [
            {"goal_id": "g1", "description": "A", "depends_on": ["g2"]},
            {"goal_id": "g2", "description": "B", "depends_on": ["g1"]},
        ]
        # Should not crash — cycles get appended at end
        result = topological_sort_goals(goals)
        assert len(result) == 2


# ============== Unit Tests: Blocked Goals ==============


class TestBlockedGoals:
    def test_no_blocked(self):
        from remy.core.smart_goals import get_blocked_goals

        goals = [
            {"goal_id": "g1", "description": "A"},
            {"goal_id": "g2", "description": "B"},
        ]
        assert get_blocked_goals(goals) == []

    def test_blocked_by_active_dep(self):
        from remy.core.smart_goals import get_blocked_goals

        goals = [
            {"goal_id": "g1", "description": "A"},
            {"goal_id": "g2", "description": "B", "depends_on": ["g1"]},
        ]
        blocked = get_blocked_goals(goals)
        assert "g2" in blocked

    def test_not_blocked_by_completed_dep(self):
        from remy.core.smart_goals import get_blocked_goals

        # g1 is not in active goals (completed), so g2 is not blocked
        goals = [
            {"goal_id": "g2", "description": "B", "depends_on": ["g1"]},
        ]
        blocked = get_blocked_goals(goals)
        assert "g2" not in blocked


# ============== Unit Tests: Reprioritize ==============


class TestReprioritize:
    def test_empty(self):
        from remy.core.smart_goals import reprioritize_goals

        assert reprioritize_goals([]) == []

    def test_high_before_low(self):
        from remy.core.smart_goals import reprioritize_goals

        goals = [
            {"goal_id": "g1", "description": "Low", "priority": "low", "attempts": 0},
            {"goal_id": "g2", "description": "High", "priority": "high", "attempts": 0},
        ]
        result = reprioritize_goals(goals)
        assert result[0]["goal_id"] == "g2"

    def test_deadline_boosts_priority(self):
        from remy.core.smart_goals import reprioritize_goals

        soon = (datetime.now() + timedelta(hours=3)).isoformat()
        goals = [
            {"goal_id": "g1", "description": "No deadline", "priority": "medium", "attempts": 0},
            {
                "goal_id": "g2",
                "description": "Deadline",
                "priority": "medium",
                "attempts": 0,
                "deadline": soon,
            },
        ]
        result = reprioritize_goals(goals)
        assert result[0]["goal_id"] == "g2"

    def test_blocked_goals_deprioritized(self):
        from remy.core.smart_goals import reprioritize_goals

        goals = [
            {"goal_id": "g1", "description": "Free", "priority": "medium", "attempts": 0},
            {
                "goal_id": "g2",
                "description": "Blocked",
                "priority": "high",
                "attempts": 0,
                "depends_on": ["g1"],
            },
        ]
        result = reprioritize_goals(goals)
        assert result[0]["goal_id"] == "g1"

    def test_many_attempts_penalized(self):
        from remy.core.smart_goals import reprioritize_goals

        goals = [
            {"goal_id": "g1", "description": "Fresh", "priority": "medium", "attempts": 0},
            {"goal_id": "g2", "description": "Stuck", "priority": "medium", "attempts": 6},
        ]
        result = reprioritize_goals(goals)
        assert result[0]["goal_id"] == "g1"

    def test_subgoals_boosted(self):
        from remy.core.smart_goals import reprioritize_goals

        goals = [
            {"goal_id": "g1", "description": "Parent", "priority": "medium", "attempts": 0},
            {
                "goal_id": "g2",
                "description": "Sub",
                "priority": "medium",
                "attempts": 0,
                "parent_goal_id": "g1",
            },
        ]
        result = reprioritize_goals(goals)
        assert result[0]["goal_id"] == "g2"


# ============== Unit Tests: Similar Goals ==============


class TestFindSimilarGoals:
    def test_no_similar(self):
        from remy.core.smart_goals import find_similar_goals

        goals = [
            {"description": "Research AI safety"},
            {"description": "Store user preferences"},
        ]
        pairs = find_similar_goals(goals)
        assert len(pairs) == 0

    def test_similar_found(self):
        from remy.core.smart_goals import find_similar_goals

        goals = [
            {"description": "Research AI safety alignment practices overview"},
            {"description": "Research AI safety alignment practices summary"},
        ]
        # Jaccard: 5 shared / 7 total = 0.71
        pairs = find_similar_goals(goals)
        assert len(pairs) >= 1
        assert pairs[0] == (0, 1)

    def test_identical(self):
        from remy.core.smart_goals import find_similar_goals

        goals = [
            {"description": "Same goal text here"},
            {"description": "Same goal text here"},
        ]
        pairs = find_similar_goals(goals)
        assert len(pairs) == 1


# ============== Unit Tests: Jaccard Similarity ==============


class TestJaccardSimilarity:
    def test_identical(self):
        from remy.core.smart_goals import _jaccard_similarity

        assert _jaccard_similarity("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        from remy.core.smart_goals import _jaccard_similarity

        assert _jaccard_similarity("hello world", "foo bar") == 0.0

    def test_partial_overlap(self):
        from remy.core.smart_goals import _jaccard_similarity

        sim = _jaccard_similarity("hello world foo", "hello world bar")
        assert 0.3 < sim < 0.8

    def test_empty(self):
        from remy.core.smart_goals import _jaccard_similarity

        assert _jaccard_similarity("", "") == 0.0


# ============== Unit Tests: Merge Goals ==============


class TestMergeGoals:
    def test_merge_keeps_higher_priority(self):
        from remy.core.smart_goals import merge_goals

        goals = [
            {"description": "Goal A", "priority": "low", "attempts": 1},
            {"description": "Goal B", "priority": "high", "attempts": 2},
        ]
        merged = merge_goals(goals, 0, 1)
        assert merged["priority"] == "high"

    def test_merge_keeps_longer_description(self):
        from remy.core.smart_goals import merge_goals

        goals = [
            {"description": "Short", "priority": "medium", "attempts": 0},
            {
                "description": "A much longer and more detailed description",
                "priority": "medium",
                "attempts": 0,
            },
        ]
        merged = merge_goals(goals, 0, 1)
        assert "longer" in merged["description"]

    def test_merge_sums_attempts(self):
        from remy.core.smart_goals import merge_goals

        goals = [
            {"description": "A", "priority": "medium", "attempts": 3},
            {"description": "B", "priority": "medium", "attempts": 2},
        ]
        merged = merge_goals(goals, 0, 1)
        assert merged["attempts"] == 5

    def test_merge_keeps_closer_deadline(self):
        from remy.core.smart_goals import merge_goals

        goals = [
            {"description": "A", "priority": "medium", "attempts": 0, "deadline": "2026-03-10"},
            {"description": "B", "priority": "medium", "attempts": 0, "deadline": "2026-03-05"},
        ]
        merged = merge_goals(goals, 0, 1)
        assert merged["deadline"] == "2026-03-05"

    def test_merge_invalid_index(self):
        from remy.core.smart_goals import merge_goals

        goals = [{"description": "A", "priority": "medium", "attempts": 0}]
        assert merge_goals(goals, 0, 5) is None


# ============== Unit Tests: Stale Goals ==============


class TestStaleGoals:
    def test_no_stale(self):
        from remy.core.smart_goals import find_stale_goals

        goals = [
            {
                "description": "Fresh",
                "priority": "medium",
                "attempts": 1,
                "last_attempt": datetime.now().isoformat(),
                "created_at": datetime.now().isoformat(),
            },
        ]
        assert find_stale_goals(goals) == []

    def test_old_no_attempts_is_stale(self):
        from remy.core.smart_goals import find_stale_goals

        old = (datetime.now() - timedelta(hours=72)).isoformat()
        goals = [
            {"description": "Old", "priority": "medium", "attempts": 0, "created_at": old},
        ]
        stale = find_stale_goals(goals)
        assert len(stale) == 1

    def test_high_priority_never_stale(self):
        from remy.core.smart_goals import find_stale_goals

        old = (datetime.now() - timedelta(hours=72)).isoformat()
        goals = [
            {"description": "Important", "priority": "high", "attempts": 0, "created_at": old},
        ]
        assert find_stale_goals(goals) == []

    def test_low_priority_old_attempt_is_stale(self):
        from remy.core.smart_goals import find_stale_goals

        old = (datetime.now() - timedelta(hours=72)).isoformat()
        goals = [
            {
                "description": "Stale",
                "priority": "low",
                "attempts": 2,
                "last_attempt": old,
                "created_at": old,
            },
        ]
        stale = find_stale_goals(goals)
        assert len(stale) == 1


# ============== Unit Tests: Goal Batching ==============


class TestGoalBatching:
    def test_batch_browser_goals(self):
        from remy.core.smart_goals import batch_goals_by_tool

        goals = [
            {"description": "Browse the web for data"},
            {"description": "Fill out web form"},
            {"description": "Research AI safety"},
        ]
        batches = batch_goals_by_tool(goals)
        assert "browser" in batches
        assert len(batches["browser"]) == 2

    def test_batch_research_goals(self):
        from remy.core.smart_goals import batch_goals_by_tool

        goals = [
            {"description": "Research AI safety"},
            {"description": "Investigate climate change"},
        ]
        batches = batch_goals_by_tool(goals)
        assert "research" in batches
        assert len(batches["research"]) == 2

    def test_general_fallback(self):
        from remy.core.smart_goals import batch_goals_by_tool

        goals = [
            {"description": "Do something vague"},
        ]
        batches = batch_goals_by_tool(goals)
        assert "general" in batches

    def test_batch_hint_format(self):
        from remy.core.smart_goals import get_batch_hint

        goals = [
            {"description": "Browse page A"},
            {"description": "Browse page B"},
        ]
        hint = get_batch_hint(goals)
        assert "BATCHING" in hint
        assert "BROWSER" in hint

    def test_no_batch_hint_single_goals(self):
        from remy.core.smart_goals import get_batch_hint

        goals = [
            {"description": "Research AI"},
            {"description": "Write file"},
        ]
        hint = get_batch_hint(goals)
        assert hint == ""


# ============== Unit Tests: Smart Sort ==============


class TestSmartSort:
    def test_smart_sort_empty(self):
        from remy.core.smart_goals import smart_sort_goals

        assert smart_sort_goals([]) == []

    def test_smart_sort_combines_deps_and_priority(self):
        from remy.core.smart_goals import smart_sort_goals

        goals = [
            {
                "goal_id": "g2",
                "description": "Step 2",
                "priority": "high",
                "attempts": 0,
                "depends_on": ["g1"],
            },
            {"goal_id": "g1", "description": "Step 1", "priority": "medium", "attempts": 0},
        ]
        result = smart_sort_goals(goals)
        # g1 should come first despite lower priority (g2 depends on it)
        assert result[0]["goal_id"] == "g1"


# ============== Unit Tests: Format Goal Management Hints ==============


class TestFormatGoalManagementHints:
    def test_empty_goals(self):
        from remy.core.smart_goals import format_goal_management_hints

        assert format_goal_management_hints([]) == ""

    def test_blocked_goals_shown(self):
        from remy.core.smart_goals import format_goal_management_hints

        goals = [
            {"goal_id": "g1", "description": "A", "priority": "medium", "attempts": 0},
            {
                "goal_id": "g2",
                "description": "B",
                "priority": "medium",
                "attempts": 0,
                "depends_on": ["g1"],
            },
        ]
        text = format_goal_management_hints(goals)
        assert "BLOCKED" in text

    def test_no_hints_for_simple_goals(self):
        from remy.core.smart_goals import format_goal_management_hints

        goals = [
            {
                "goal_id": "g1",
                "description": "Simple task",
                "priority": "medium",
                "attempts": 1,
                "created_at": datetime.now().isoformat(),
                "last_attempt": datetime.now().isoformat(),
            },
        ]
        text = format_goal_management_hints(goals)
        assert text == ""


# ============== Integration: depends_on in create_goal ==============


class TestCreateGoalDependencies:
    def test_depends_on_stored_in_metadata(self):
        with patch("remy.core.autonomy.settings") as mock_settings:
            mock_settings.AUTONOMY_DAILY_TOKEN_LIMIT = 100_000
            mock_settings.AUTONOMY_HOURLY_TOKEN_LIMIT = 20_000
            mock_settings.AUTONOMY_SESSION_TOKEN_LIMIT = 500_000
            mock_settings.AUTONOMY_CYCLE_INTERVAL_SEC = 0
            mock_settings.AUTONOMY_TELEGRAM_NOTIFICATIONS = False
            mock_settings.AUTONOMY_SELF_CRITIQUE_ENABLED = False
            mock_settings.SUMMARY_MODEL = "test-model"
            mock_settings.GEMINI_API_KEY = "test-key"
            mock_settings.DATA_DIR = "/tmp"

            from remy.core.autonomy_goals import create_goal, get_active_goals

            # Create two goals with dependency
            rid1 = create_goal("Step 1: gather data")
            goals1 = get_active_goals()
            g1_id = [g for g in goals1 if g["record_id"] == rid1][0]["goal_id"]

            rid2 = create_goal("Step 2: analyze data", depends_on=[g1_id])
            goals2 = get_active_goals()
            g2 = [g for g in goals2 if g["record_id"] == rid2][0]

            assert g2["depends_on"] == [g1_id]


# ============== Integration: smart sort in decision prompt ==============


class TestSmartGoalsInPrompt:
    def test_goal_management_hints_in_prompt(self):
        with patch("remy.core.autonomy.settings") as mock_settings:
            mock_settings.AUTONOMY_DAILY_TOKEN_LIMIT = 100_000
            mock_settings.AUTONOMY_HOURLY_TOKEN_LIMIT = 20_000
            mock_settings.AUTONOMY_SESSION_TOKEN_LIMIT = 500_000
            mock_settings.AUTONOMY_CYCLE_INTERVAL_SEC = 0
            mock_settings.AUTONOMY_TELEGRAM_NOTIFICATIONS = False
            mock_settings.AUTONOMY_SELF_CRITIQUE_ENABLED = False
            mock_settings.SUMMARY_MODEL = "test-model"
            mock_settings.GEMINI_API_KEY = "test-key"
            mock_settings.DATA_DIR = "/tmp"

            from remy.core.autonomy import AutonomousLoop

            loop = AutonomousLoop()
            prompt = loop._build_decision_prompt(
                goals=[
                    {
                        "goal_id": "g1",
                        "record_id": "r1",
                        "description": "Step 1",
                        "priority": "medium",
                        "attempts": 0,
                        "success_criteria": [],
                    },
                    {
                        "goal_id": "g2",
                        "record_id": "r2",
                        "description": "Step 2",
                        "priority": "medium",
                        "attempts": 0,
                        "depends_on": ["g1"],
                        "success_criteria": [],
                    },
                ],
                past_outcomes="",
                budget={
                    "tokens_today": 0,
                    "daily_limit": 100000,
                    "tokens_this_hour": 0,
                    "hourly_limit": 20000,
                },
            )

            assert "BLOCKED" in prompt or "GOAL MANAGEMENT" in prompt
