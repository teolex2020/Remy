"""Tests for Adaptive Role Switching (AUTON-4) — autonomy_roles.py."""

from unittest.mock import patch

import pytest


@pytest.fixture
def roles_env(brain, monkeypatch):
    """Redirect autonomy_roles to a temp brain so tests don't pollute production data."""
    import threading
    monkeypatch.setattr("remy.core.autonomy_roles.brain", brain)
    monkeypatch.setattr("remy.core.agent_tools.brain", brain)
    monkeypatch.setattr("remy.core.agent_tools.brain_lock", threading.RLock())
    yield


# ============== Unit Tests: infer_goal_type ==============


class TestInferGoalType:
    def test_research_keyword(self):
        from remy.core.autonomy_roles import infer_goal_type

        assert infer_goal_type("Research AI safety papers") == "research"

    def test_health_keyword(self):
        from remy.core.autonomy_roles import infer_goal_type

        assert infer_goal_type("Track health metrics") == "health"

    def test_plan_keyword(self):
        from remy.core.autonomy_roles import infer_goal_type

        assert infer_goal_type("Plan weekly schedule") == "plan"

    def test_general_fallback(self):
        from remy.core.autonomy_roles import infer_goal_type

        assert infer_goal_type("Do something random") == "general"

    def test_case_insensitive(self):
        from remy.core.autonomy_roles import infer_goal_type

        assert infer_goal_type("RESEARCH deep learning") == "research"


# ============== Unit Tests: record_role_performance ==============


class TestRecordRolePerformance:
    def test_records_success(self, roles_env):
        from remy.core.autonomy_roles import get_role_stats, record_role_performance

        record_role_performance("researcher", "research", True, tokens_used=500)

        stats = get_role_stats()
        assert "researcher" in stats
        assert stats["researcher"]["research"]["successes"] == 1

    def test_records_failure(self, roles_env):
        from remy.core.autonomy_roles import get_role_stats, record_role_performance

        record_role_performance("executor", "write", False, tokens_used=300)

        stats = get_role_stats()
        assert stats["executor"]["write"]["successes"] == 0
        assert stats["executor"]["write"]["attempts"] == 1


# ============== Unit Tests: get_role_stats ==============


class TestGetRoleStats:
    def test_empty_brain(self, roles_env):
        from remy.core.autonomy_roles import get_role_stats

        stats = get_role_stats()
        assert stats == {}

    def test_computes_rate(self, roles_env):
        from remy.core.autonomy_roles import get_role_stats, record_role_performance

        for _ in range(3):
            record_role_performance("researcher", "research", True)
        record_role_performance("researcher", "research", False)

        stats = get_role_stats()
        assert stats["researcher"]["research"]["rate"] == 0.75
        assert stats["researcher"]["research"]["attempts"] == 4

    def test_multiple_roles(self, roles_env):
        from remy.core.autonomy_roles import get_role_stats, record_role_performance

        record_role_performance("researcher", "research", True)
        record_role_performance("analyst", "health", True)
        record_role_performance("executor", "write", False)

        stats = get_role_stats()
        assert "researcher" in stats
        assert "analyst" in stats
        assert "executor" in stats


# ============== Unit Tests: select_best_role ==============


class TestSelectBestRole:
    def test_fallback_to_keywords_no_data(self, roles_env):
        from remy.core.autonomy_roles import select_best_role

        role = select_best_role("Research AI safety")
        assert role.name == "researcher"

    def test_fallback_to_keywords_for_analyst(self, roles_env):
        from remy.core.autonomy_roles import select_best_role

        role = select_best_role("Analyze health patterns")
        assert role.name == "analyst"

    def test_fallback_to_keywords_for_planner(self, roles_env):
        from remy.core.autonomy_roles import select_best_role

        role = select_best_role("Plan and organize schedule")
        assert role.name == "planner"

    def test_fallback_to_executor_default(self, roles_env):
        from remy.core.autonomy_roles import select_best_role

        role = select_best_role("Do the thing")
        assert role.name == "executor"

    def test_selects_by_performance(self, roles_env):
        from remy.core.autonomy_roles import (
            record_role_performance,
            select_best_role,
        )

        # Analyst is great at "general" goals, executor is bad
        for _ in range(5):
            record_role_performance("analyst", "general", True)
        for _ in range(5):
            record_role_performance("executor", "general", False)

        role = select_best_role("Do something general", goal_type="general")
        assert role.name == "analyst"

    def test_needs_min_samples(self, roles_env):
        from remy.core.autonomy_roles import (
            record_role_performance,
            select_best_role,
        )

        # Only 1 data point — not enough
        record_role_performance("analyst", "research", True)

        # Should fallback to keywords
        role = select_best_role("Research something", goal_type="research")
        assert role.name == "researcher"  # keyword match

    def test_fallback_chain_on_failures(self, roles_env):
        from remy.core.autonomy_roles import select_best_role

        role = select_best_role(
            "Do something",
            goal_type="general",
            current_failures=2,
            current_role_name="researcher",
        )
        # Should pick from fallback chain, not researcher again
        assert role.name != "researcher"

    def test_fallback_prefers_good_performer(self, roles_env):
        from remy.core.autonomy_roles import (
            record_role_performance,
            select_best_role,
        )

        # Build up stats: executor is great at "general", analyst is OK
        for _ in range(5):
            record_role_performance("executor", "general", True)
        for _ in range(5):
            record_role_performance("analyst", "general", False)

        role = select_best_role(
            "Do something general",
            goal_type="general",
            current_failures=2,
            current_role_name="researcher",
        )
        assert role.name == "executor"


# ============== Unit Tests: _select_by_keywords ==============


class TestSelectByKeywords:
    def test_ukrainian_keywords(self):
        from remy.core.autonomy_roles import _select_by_keywords

        assert _select_by_keywords("дослідити нові ліки").name == "researcher"
        assert _select_by_keywords("аналіз здоров'я").name == "analyst"
        assert _select_by_keywords("організувати задачі").name == "planner"

    def test_english_keywords(self):
        from remy.core.autonomy_roles import _select_by_keywords

        assert _select_by_keywords("investigate the problem").name == "researcher"
        assert _select_by_keywords("look up documentation").name == "researcher"
        assert _select_by_keywords("review and analyze data").name == "analyst"
        assert _select_by_keywords("structure the project").name == "planner"


# ============== Unit Tests: format_role_performance_hint ==============


class TestFormatRolePerformanceHint:
    def test_no_stats_returns_empty(self, roles_env):
        from remy.core.autonomy_models import AGENT_ROLES
        from remy.core.autonomy_roles import format_role_performance_hint

        hint = format_role_performance_hint(AGENT_ROLES["researcher"], "research")
        assert hint == ""

    def test_good_performer_hint(self, roles_env):
        from remy.core.autonomy_models import AGENT_ROLES
        from remy.core.autonomy_roles import (
            format_role_performance_hint,
            record_role_performance,
        )

        for _ in range(5):
            record_role_performance("researcher", "research", True)

        hint = format_role_performance_hint(AGENT_ROLES["researcher"], "research")
        assert "100%" in hint
        assert "ROLE NOTE" in hint

    def test_bad_performer_warning(self, roles_env):
        from remy.core.autonomy_models import AGENT_ROLES
        from remy.core.autonomy_roles import (
            format_role_performance_hint,
            record_role_performance,
        )

        for _ in range(4):
            record_role_performance("executor", "research", False)

        hint = format_role_performance_hint(AGENT_ROLES["executor"], "research")
        assert "WARNING" in hint


# ============== Unit Tests: ROLE_FALLBACK_CHAINS ==============


class TestFallbackChains:
    def test_all_roles_have_chains(self):
        from remy.core.autonomy_models import AGENT_ROLES
        from remy.core.autonomy_roles import ROLE_FALLBACK_CHAINS

        for role_name in AGENT_ROLES:
            assert role_name in ROLE_FALLBACK_CHAINS
            assert len(ROLE_FALLBACK_CHAINS[role_name]) >= 2

    def test_chain_does_not_include_self(self):
        from remy.core.autonomy_roles import ROLE_FALLBACK_CHAINS

        for role_name, chain in ROLE_FALLBACK_CHAINS.items():
            assert role_name not in chain


# ============== Integration: role in AutonomousLoop ==============


class TestRoleInAutonomousLoop:
    def test_loop_has_role_tracking_attrs(self):
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
            assert hasattr(loop, "_last_role_name")
            assert hasattr(loop, "_in_role_failures")
            assert loop._last_role_name == ""
            assert loop._in_role_failures == 0
