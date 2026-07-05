"""Tests for Autonomous Agent Mode — budget, goals, outcomes, loop."""

import json
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from aura import Aura as CognitiveMemory, Level


# ============== FIXTURES ==============


@pytest.fixture
def mock_brain(tmp_path):
    b = CognitiveMemory(str(tmp_path / "test_brain"))
    yield b
    b.close()


@pytest.fixture(autouse=True)
def patch_settings(tmp_path):
    """Patch settings for all tests."""
    with patch("remy.core.autonomy.settings") as mock_settings:
        mock_settings.AUTONOMY_DAILY_TOKEN_LIMIT = 100_000
        mock_settings.AUTONOMY_HOURLY_TOKEN_LIMIT = 20_000
        mock_settings.AUTONOMY_SESSION_TOKEN_LIMIT = 500_000
        mock_settings.AUTONOMY_CYCLE_INTERVAL_SEC = 1
        mock_settings.AUTONOMY_AUTO_APPROVE_SANDBOX = False
        mock_settings.AUTONOMY_TELEGRAM_NOTIFICATIONS = False
        mock_settings.AUTONOMY_MAX_ACTIONS_PER_HOUR = 20
        mock_settings.TELEGRAM_BOT_TOKEN = None
        mock_settings.PROACTIVE_CHAT_ID = None
        mock_settings.SUMMARY_MODEL = "test-model"
        mock_settings.AURA_BRAIN_PATH = tmp_path / "brain"
        mock_settings.DATA_DIR = tmp_path / "data"
        yield mock_settings


# ============== RESOURCE BUDGET ==============


class TestResourceBudget:

    def test_fresh_budget_allows_spending(self):
        from remy.core.autonomy import ResourceBudget
        budget = ResourceBudget(daily_limit=100_000, hourly_limit=20_000, session_limit=500_000)
        can, reason = budget.can_spend(1000)
        assert can is True
        assert reason == "ok"

    def test_session_limit_enforced(self):
        from remy.core.autonomy import ResourceBudget
        budget = ResourceBudget(daily_limit=100_000, hourly_limit=20_000, session_limit=5000)
        budget.record_usage(5000)
        can, reason = budget.can_spend(1000)
        assert can is False
        assert "Session limit" in reason

    def test_hourly_limit_enforced(self):
        from remy.core.autonomy import ResourceBudget
        budget = ResourceBudget(daily_limit=100_000, hourly_limit=5000, session_limit=500_000)
        budget.record_usage(5000)
        can, reason = budget.can_spend(1000)
        assert can is False
        assert "Hourly limit" in reason

    def test_daily_limit_enforced(self):
        from remy.core.autonomy import ResourceBudget
        budget = ResourceBudget(daily_limit=5000, hourly_limit=20_000, session_limit=500_000)
        budget.record_usage(5000)
        can, reason = budget.can_spend(1000)
        assert can is False
        assert "Daily limit" in reason

    def test_hourly_reset(self):
        from remy.core.autonomy import ResourceBudget
        budget = ResourceBudget(daily_limit=100_000, hourly_limit=5000, session_limit=500_000)
        budget.record_usage(5000)
        # Simulate hour passed
        budget.last_hour_reset = time.time() - 3601
        can, reason = budget.can_spend(1000)
        assert can is True

    def test_daily_reset(self):
        from remy.core.autonomy import ResourceBudget
        budget = ResourceBudget(daily_limit=5000, hourly_limit=20_000, session_limit=500_000)
        budget.record_usage(5000)
        # Simulate day passed
        budget.last_day_reset = time.time() - 86401
        can, reason = budget.can_spend(1000)
        assert can is True

    def test_record_usage_tracks_all(self):
        from remy.core.autonomy import ResourceBudget
        budget = ResourceBudget(daily_limit=100_000, hourly_limit=20_000, session_limit=500_000)
        budget.record_usage(500)
        budget.record_usage(300)
        assert budget.tokens_today == 800
        assert budget.tokens_this_hour == 800
        assert budget.tokens_this_session == 800
        assert budget.total_tokens_lifetime == 800

    def test_to_dict(self):
        from remy.core.autonomy import ResourceBudget
        budget = ResourceBudget(daily_limit=100_000, hourly_limit=20_000, session_limit=500_000)
        budget.record_usage(1234)
        d = budget.to_dict()
        assert d["daily_limit"] == 100_000
        assert d["tokens_today"] == 1234
        assert d["total_tokens_lifetime"] == 1234


# ============== GOAL SYSTEM ==============


class TestGoalSystem:

    def test_create_goal(self, mock_brain):
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import create_goal
            record_id = create_goal("Learn about climate change", priority="high")

        assert record_id is not None
        goals = mock_brain.search(query="", tags=["autonomous-goal"], limit=10)
        assert len(goals) == 1
        assert "climate change" in goals[0].content.lower()
        assert goals[0].metadata["priority"] == "high"
        assert goals[0].metadata["status"] == "active"

    def test_get_active_goals_sorted(self, mock_brain):
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import create_goal, get_active_goals
            create_goal("Low priority task", priority="low")
            create_goal("Critical task", priority="critical")
            create_goal("Medium task", priority="medium")
            goals = get_active_goals()

        assert len(goals) == 3
        assert goals[0]["priority"] == "critical"
        assert goals[1]["priority"] == "medium"
        assert goals[2]["priority"] == "low"

    def test_update_goal_status(self, mock_brain):
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import create_goal, update_goal_status, get_active_goals
            rid = create_goal("Test goal", priority="medium")
            update_goal_status(rid, "completed", notes="Done successfully")
            active = get_active_goals()

        assert len(active) == 0  # Completed goal is not active
        rec = mock_brain.get(rid)
        assert rec.metadata["status"] == "completed"
        assert rec.metadata["status_notes"] == "Done successfully"

    def test_record_goal_attempt(self, mock_brain):
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import create_goal, record_goal_attempt
            rid = create_goal("Research topic", priority="medium")
            record_goal_attempt(rid)
            record_goal_attempt(rid)

        rec = mock_brain.get(rid)
        assert rec.metadata["attempts"] == 2
        assert rec.metadata["last_attempt"] is not None

    def test_goal_with_deadline(self, mock_brain):
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import create_goal, get_active_goals
            create_goal("Finish project", priority="high", deadline="2026-03-01")
            goals = get_active_goals()

        assert goals[0]["deadline"] == "2026-03-01"
        assert "Deadline" in mock_brain.search(query="", tags=["autonomous-goal"], limit=1)[0].content


# ============== OUTCOME TRACKING ==============


class TestOutcomeTracking:

    def test_record_outcome_stores(self, mock_brain):
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import ActionRecord, record_outcome
            action = ActionRecord(
                action_id="test123",
                timestamp=datetime.now().isoformat(),
                goal_id="g1",
                action_type="agent_invoke",
                description="Researched topic X",
                result="Found 3 articles",
                success=True,
                tokens_used=1500,
                duration_ms=3000,
            )
            outcome_id = record_outcome(action)

        outcomes = mock_brain.search(query="", tags=["autonomous-outcome"], limit=10)
        assert len(outcomes) == 1
        assert "SUCCESS" in outcomes[0].content
        assert outcomes[0].metadata["tokens_used"] == 1500

    def test_failed_outcome_tagged(self, mock_brain):
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import ActionRecord, record_outcome
            action = ActionRecord(
                action_id="fail1",
                timestamp=datetime.now().isoformat(),
                goal_id=None,
                action_type="tool_call",
                description="Web search failed",
                result="Error: timeout",
                success=False,
                tokens_used=500,
                duration_ms=5000,
            )
            record_outcome(action)

        outcomes = mock_brain.search(query="", tags=["outcome-failure"], limit=10)
        assert len(outcomes) == 1
        assert "FAILURE" in outcomes[0].content

    def test_outcome_connected_to_goal(self, mock_brain):
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import create_goal, ActionRecord, record_outcome
            goal_rid = create_goal("Test goal", priority="medium")
            action = ActionRecord(
                action_id="a1", timestamp=datetime.now().isoformat(),
                goal_id="g1", action_type="tool_call",
                description="Called web_search", result="OK",
                success=True, tokens_used=500, duration_ms=1000,
            )
            outcome_id = record_outcome(action, goal_record_id=goal_rid)

        outcome_rec = mock_brain.get(outcome_id)
        assert goal_rid in outcome_rec.connections

    def test_recall_similar_outcomes(self, mock_brain):
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import ActionRecord, record_outcome, recall_similar_outcomes
            action = ActionRecord(
                action_id="a1", timestamp=datetime.now().isoformat(),
                goal_id=None, action_type="research",
                description="Researched climate change impacts",
                result="Found useful data", success=True,
                tokens_used=2000, duration_ms=5000,
            )
            record_outcome(action)
            similar = recall_similar_outcomes("climate change")

        assert len(similar) >= 1
        assert similar[0]["success"] is True


# ============== AUTONOMOUS LOOP ==============


class TestAutonomousLoop:

    def test_loop_init(self, patch_settings):
        from remy.core.autonomy import AutonomousLoop
        loop = AutonomousLoop()
        assert loop.session_id.startswith("auto-")
        assert loop.budget.daily_limit == 100_000
        assert loop.action_log == []
        assert loop.consecutive_failures == 0

    def test_budget_pause_skips_cycle(self, mock_brain, patch_settings):
        """When budget is exhausted, cycle should skip."""
        from remy.core.autonomy import AutonomousLoop

        with patch("remy.core.autonomy.brain", mock_brain):
            loop = AutonomousLoop()
            # Exhaust budget
            loop.budget.record_usage(loop.budget.session_limit)

            can, reason = loop.budget.can_spend(2000)
            assert can is False

    def test_seed_goals_created(self, mock_brain, patch_settings):
        """First-time loop should create seed goals."""
        from remy.core.autonomy import AutonomousLoop

        with patch("remy.core.autonomy.brain", mock_brain):
            loop = AutonomousLoop()
            loop._seed_initial_goals()

        goals = mock_brain.search(query="", tags=["autonomous-goal"], limit=10)
        assert len(goals) >= 2  # 2 defaults if LLM unavailable, 2-3 if LLM generates

    def test_seed_goals_with_profile(self, mock_brain, patch_settings):
        """Seed goals with profile uses LLM; falls back to defaults if unavailable."""
        # Store a user profile first
        mock_brain.store(
            content="User Profile: Name: Alex",
            level=Level.IDENTITY,
            tags=["user-profile"],
            metadata={"name": "Alex"},
        )

        from remy.core.autonomy import AutonomousLoop

        with patch("remy.core.autonomy.brain", mock_brain):
            loop = AutonomousLoop()
            loop._seed_initial_goals()

        goals = mock_brain.search(query="", tags=["autonomous-goal"], limit=10)
        # Goals created (fallback defaults when LLM unavailable in tests)
        assert len(goals) >= 2

    def test_decision_prompt_includes_goals(self, mock_brain, patch_settings):
        """Decision prompt should include active goals and budget."""
        from remy.core.autonomy import AutonomousLoop, create_goal

        with patch("remy.core.autonomy.brain", mock_brain):
            create_goal("Test goal", priority="high")
            loop = AutonomousLoop()
            from remy.core.autonomy import get_active_goals
            goals = get_active_goals()
            prompt = loop._build_decision_prompt(goals, "", loop.budget.to_dict())

        assert "AUTONOMOUS MODE" in prompt
        assert "HIGH" in prompt
        assert "Test goal" in prompt
        assert "100000" in prompt  # daily limit

    def test_recent_outcomes_summary(self, patch_settings):
        """Summary of recent actions should be formatted."""
        from remy.core.autonomy import AutonomousLoop, ActionRecord

        loop = AutonomousLoop()
        loop.action_log.append(ActionRecord(
            action_id="a1", timestamp="2026-01-01", goal_id=None,
            action_type="agent_invoke", description="Did something",
            result="OK", success=True, tokens_used=500, duration_ms=1000,
        ))
        loop.action_log.append(ActionRecord(
            action_id="a2", timestamp="2026-01-01", goal_id=None,
            action_type="agent_invoke", description="Failed thing",
            result="Error", success=False, tokens_used=300, duration_ms=2000,
        ))

        summary = loop._recent_outcomes_summary()
        assert "[OK]" in summary
        assert "[FAIL]" in summary
        assert "500 tokens" in summary


# ============== BUDGET PERSISTENCE ==============


class TestBudgetPersistence:

    def test_save_and_load_budget(self, patch_settings):
        from remy.core.autonomy import ResourceBudget, save_budget, load_budget

        budget = ResourceBudget(daily_limit=100_000, hourly_limit=20_000, session_limit=500_000)
        budget.record_usage(5000)
        budget.total_tokens_lifetime = 42000
        save_budget(budget)

        # Create a fresh budget and load
        budget2 = ResourceBudget(daily_limit=100_000, hourly_limit=20_000, session_limit=500_000)
        load_budget(budget2)

        assert budget2.tokens_today == 5000
        assert budget2.tokens_this_hour == 5000
        assert budget2.total_tokens_lifetime == 42000

    def test_load_resets_expired_daily(self, patch_settings):
        """Daily counter should reset if saved more than 24h ago."""
        from remy.core.autonomy import ResourceBudget, save_budget, load_budget, _get_budget_path

        budget = ResourceBudget(daily_limit=100_000, hourly_limit=20_000, session_limit=500_000)
        budget.record_usage(9999)
        save_budget(budget)

        # Tamper with saved file to simulate old timestamp
        path = _get_budget_path()
        data = json.loads(path.read_text())
        data["last_day_reset"] = time.time() - 90000  # > 24h ago
        path.write_text(json.dumps(data))

        budget2 = ResourceBudget(daily_limit=100_000, hourly_limit=20_000, session_limit=500_000)
        load_budget(budget2)

        assert budget2.tokens_today == 0  # reset
        assert budget2.total_tokens_lifetime == 9999  # lifetime preserved

    def test_load_resets_expired_hourly(self, patch_settings):
        """Hourly counter should reset if saved more than 1h ago."""
        from remy.core.autonomy import ResourceBudget, save_budget, load_budget, _get_budget_path

        budget = ResourceBudget(daily_limit=100_000, hourly_limit=20_000, session_limit=500_000)
        budget.record_usage(3000)
        save_budget(budget)

        path = _get_budget_path()
        data = json.loads(path.read_text())
        data["last_hour_reset"] = time.time() - 4000  # > 1h ago
        path.write_text(json.dumps(data))

        budget2 = ResourceBudget(daily_limit=100_000, hourly_limit=20_000, session_limit=500_000)
        load_budget(budget2)

        assert budget2.tokens_this_hour == 0  # reset
        assert budget2.tokens_today == 3000  # still within day

    def test_load_missing_file(self, patch_settings):
        """Loading when no file exists should be a no-op."""
        from remy.core.autonomy import ResourceBudget, load_budget

        budget = ResourceBudget(daily_limit=100_000, hourly_limit=20_000, session_limit=500_000)
        load_budget(budget)  # should not raise

        assert budget.tokens_today == 0
        assert budget.total_tokens_lifetime == 0

    def test_load_corrupted_file(self, patch_settings):
        """Loading a corrupted file should not crash."""
        from remy.core.autonomy import ResourceBudget, load_budget, _get_budget_path

        path = _get_budget_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not json{{{")

        budget = ResourceBudget(daily_limit=100_000, hourly_limit=20_000, session_limit=500_000)
        load_budget(budget)  # should not raise

        assert budget.tokens_today == 0


# ============== GOAL CLEANUP ==============


class TestGoalCleanup:

    def test_archive_completed_old_goals(self, mock_brain):
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import create_goal, update_goal_status, archive_completed_goals

            rid = create_goal("Old completed goal", priority="medium")
            update_goal_status(rid, "completed", notes="Done")

            # Make it look old (>24h)
            rec = mock_brain.get(rid)
            meta = dict(rec.metadata)
            meta["updated_at"] = (datetime.now() - timedelta(hours=25)).isoformat()
            mock_brain.update(rid, metadata=meta)

            count = archive_completed_goals()

        assert count == 1
        rec = mock_brain.get(rid)
        assert rec.metadata["status"] == "archived"
        assert "archived_at" in rec.metadata

    def test_archive_skips_recent_completed(self, mock_brain):
        """Goals completed less than 24h ago should NOT be archived."""
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import create_goal, update_goal_status, archive_completed_goals

            rid = create_goal("Recent completed", priority="medium")
            update_goal_status(rid, "completed")

            count = archive_completed_goals()

        assert count == 0
        rec = mock_brain.get(rid)
        assert rec.metadata["status"] == "completed"  # not archived

    def test_archive_skips_active_goals(self, mock_brain):
        """Active goals should never be archived."""
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import create_goal, archive_completed_goals

            create_goal("Active goal", priority="high")
            count = archive_completed_goals()

        assert count == 0

    def test_archive_failed_goals(self, mock_brain):
        """Failed goals older than 24h should also be archived."""
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import create_goal, update_goal_status, archive_completed_goals

            rid = create_goal("Failed goal", priority="low")
            update_goal_status(rid, "failed", notes="Couldn't do it")

            rec = mock_brain.get(rid)
            meta = dict(rec.metadata)
            meta["updated_at"] = (datetime.now() - timedelta(hours=30)).isoformat()
            mock_brain.update(rid, metadata=meta)

            count = archive_completed_goals()

        assert count == 1
        assert mock_brain.get(rid).metadata["status"] == "archived"


# ============== AUTONOMY LOGGER ==============


class TestAutonomyLogger:

    def test_setup_creates_log_file(self, patch_settings):
        """_setup_autonomy_logger should create data/logs/ dir."""
        import logging
        from remy.core.autonomy import _setup_autonomy_logger

        _setup_autonomy_logger()

        log_dir = patch_settings.DATA_DIR / "logs"
        assert log_dir.exists()

        # Check that a RotatingFileHandler was added
        auto_logger = logging.getLogger("Autonomy")
        from logging.handlers import RotatingFileHandler
        rfh = [h for h in auto_logger.handlers if isinstance(h, RotatingFileHandler)]
        assert len(rfh) >= 1

        # Cleanup handlers to avoid leaking between tests
        for h in rfh:
            auto_logger.removeHandler(h)
            h.close()
