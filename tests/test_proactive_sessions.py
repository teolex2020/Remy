"""Tests for Proactive Sessions — autonomous agent initiates Telegram conversations."""

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
    with patch("remy.core.autonomy.settings") as mock_settings:
        mock_settings.AUTONOMY_DAILY_TOKEN_LIMIT = 100_000
        mock_settings.AUTONOMY_HOURLY_TOKEN_LIMIT = 20_000
        mock_settings.AUTONOMY_SESSION_TOKEN_LIMIT = 500_000
        mock_settings.AUTONOMY_CYCLE_INTERVAL_SEC = 0
        mock_settings.AUTONOMY_AUTO_APPROVE_SANDBOX = False
        mock_settings.AUTONOMY_TELEGRAM_NOTIFICATIONS = False
        mock_settings.AUTONOMY_MAX_ACTIONS_PER_HOUR = 20
        mock_settings.TELEGRAM_BOT_TOKEN = "test-token"
        mock_settings.PROACTIVE_CHAT_ID = 12345
        mock_settings.SUMMARY_MODEL = "test-model"
        mock_settings.GEMINI_API_KEY = "test-key"
        mock_settings.AURA_BRAIN_PATH = tmp_path / "brain"
        mock_settings.DATA_DIR = tmp_path / "data"
        mock_settings.AUTONOMY_QUIET_HOURS_START = 23
        mock_settings.AUTONOMY_QUIET_HOURS_END = 7
        mock_settings.AUTONOMY_MAX_SESSION_MINUTES = 30
        mock_settings.AUTONOMY_PROACTIVE_SESSIONS_ENABLED = True
        mock_settings.AUTONOMY_PROACTIVE_MAX_PER_DAY = 3
        mock_settings.AUTONOMY_PROACTIVE_MIN_INTERVAL_SEC = 7200
        (tmp_path / "data" / "logs").mkdir(parents=True, exist_ok=True)
        yield mock_settings


# ============== TRIGGER LOGIC ==============


class TestShouldStartProactiveSession:
    """Tests for _should_start_proactive_session() guard and trigger logic."""

    def test_returns_none_when_disabled(self, mock_brain, patch_settings):
        patch_settings.AUTONOMY_PROACTIVE_SESSIONS_ENABLED = False
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import AutonomousLoop
            loop = AutonomousLoop()
            assert loop._should_start_proactive_session() is None

    def test_returns_none_without_telegram_token(self, mock_brain, patch_settings):
        patch_settings.TELEGRAM_BOT_TOKEN = None
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import AutonomousLoop
            loop = AutonomousLoop()
            assert loop._should_start_proactive_session() is None

    def test_returns_none_without_chat_id(self, mock_brain, patch_settings):
        patch_settings.PROACTIVE_CHAT_ID = None
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import AutonomousLoop
            loop = AutonomousLoop()
            assert loop._should_start_proactive_session() is None

    def test_returns_none_during_quiet_hours(self, mock_brain, patch_settings):
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import AutonomousLoop
            loop = AutonomousLoop()
            with patch.object(loop, "_is_quiet_hours", return_value=True):
                assert loop._should_start_proactive_session() is None

    def test_returns_none_max_per_day_reached(self, mock_brain, patch_settings):
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import AutonomousLoop
            loop = AutonomousLoop()
            loop._proactive_sessions_today = 3
            loop._last_proactive_day = datetime.now().date().isoformat()
            assert loop._should_start_proactive_session() is None

    def test_returns_none_min_interval_not_met(self, mock_brain, patch_settings):
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import AutonomousLoop
            loop = AutonomousLoop()
            loop._last_proactive_time = time.time() - 60  # 1 min ago
            assert loop._should_start_proactive_session() is None

    def test_triggers_on_due_task(self, mock_brain, patch_settings):
        """Should trigger when a scheduled task is due today."""
        today = datetime.now().date().isoformat()
        mock_brain.store(
            content=f"Take vitamin D | Due: {today}",
            level=Level.DOMAIN,
            tags=["scheduled-task"],
            metadata={
                "description": "Take vitamin D",
                "due_date": today,
                "status": "active",
            },
        )
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import AutonomousLoop
            loop = AutonomousLoop()
            with patch.object(loop, "_is_quiet_hours", return_value=False):
                trigger = loop._should_start_proactive_session()

        assert trigger is not None
        assert trigger["reason"] == "scheduled_task_due"
        assert "vitamin D" in trigger["context"]
        assert trigger["priority"] == "high"

    def test_skips_already_reminded_task(self, mock_brain, patch_settings):
        """Should not re-trigger for a task already reminded in this session."""
        today = datetime.now().date().isoformat()
        rec = mock_brain.store(
            content=f"Take vitamin D | Due: {today}",
            level=Level.DOMAIN,
            tags=["scheduled-task"],
            metadata={
                "description": "Take vitamin D",
                "due_date": today,
                "status": "active",
            },
        )
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import AutonomousLoop
            loop = AutonomousLoop()
            loop._proactive_reminders_sent.add(rec.id)
            # Set today so daily reset doesn't clear reminders_sent
            loop._last_proactive_day = today
            trigger = loop._should_start_proactive_session()

        # Should be None or a different trigger (no tasks to remind about)
        if trigger is not None:
            assert trigger["reason"] != "scheduled_task_due"

    def test_daily_counter_resets_on_new_day(self, mock_brain, patch_settings):
        """Counter should reset when day changes."""
        today = datetime.now().date().isoformat()
        mock_brain.store(
            content=f"Test task | Due: {today}",
            level=Level.DOMAIN,
            tags=["scheduled-task"],
            metadata={"description": "Test task", "due_date": today, "status": "active"},
        )
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import AutonomousLoop
            loop = AutonomousLoop()
            # Simulate yesterday's exhausted counter
            loop._proactive_sessions_today = 3
            loop._last_proactive_day = "2020-01-01"
            with patch.object(loop, "_is_quiet_hours", return_value=False):
                trigger = loop._should_start_proactive_session()

        assert trigger is not None  # Counter was reset, trigger fires

    def test_inactivity_checkin_after_4_hours(self, mock_brain, patch_settings):
        """Should trigger inactivity check-in if no session in >4 hours."""
        old_time = (datetime.now() - timedelta(hours=5)).isoformat()
        mock_brain.store(
            content="Last session summary",
            level=Level.DOMAIN,
            tags=["session-summary"],
            metadata={"timestamp": old_time},
        )
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import AutonomousLoop
            loop = AutonomousLoop()
            with patch.object(loop, "_is_quiet_hours", return_value=False):
                trigger = loop._should_start_proactive_session()

        assert trigger is not None
        assert trigger["reason"] == "inactivity_checkin"
        assert trigger["priority"] == "low"

    def test_no_trigger_when_user_recently_active(self, mock_brain, patch_settings):
        """Should return None if user was active within 30 minutes."""
        recent_time = (datetime.now() - timedelta(minutes=10)).isoformat()
        mock_brain.store(
            content="Recent session",
            level=Level.DOMAIN,
            tags=["session-summary"],
            metadata={"timestamp": recent_time},
        )
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import AutonomousLoop
            loop = AutonomousLoop()
            assert loop._should_start_proactive_session() is None


# ============== SESSION START ==============


class TestStartProactiveSession:
    """Tests for _start_proactive_session() — message generation and sending."""

    @pytest.mark.asyncio
    async def test_sends_telegram_message(self, mock_brain, patch_settings):
        """Should invoke agent and send message via Telegram."""
        with patch("remy.core.autonomy.brain", mock_brain), \
             patch("remy.core.agent.invoke_agent", new_callable=AsyncMock) as mock_invoke, \
             patch("remy.core.brain_tools.get_proactive_context", return_value=""):

            mock_invoke.return_value = ("Hey! Just checking in about vitamin D.", [], [])

            from remy.core.autonomy import AutonomousLoop
            loop = AutonomousLoop()

            trigger = {
                "reason": "scheduled_task_due",
                "context": "Task due today: Take vitamin D",
                "priority": "high",
                "record_id": "test-id",
            }

            with patch("telegram.Bot") as MockBot:
                mock_bot_instance = AsyncMock()
                MockBot.return_value = mock_bot_instance
                await loop._start_proactive_session(trigger)

            mock_invoke.assert_called_once()
            call_kwargs = mock_invoke.call_args[1]
            assert call_kwargs["channel"] == "proactive"
            mock_bot_instance.send_message.assert_called_once()
            assert loop._proactive_sessions_today == 1
            assert "test-id" in loop._proactive_reminders_sent

    @pytest.mark.asyncio
    async def test_skips_on_budget_exhausted(self, mock_brain, patch_settings):
        """Should not send if budget is exhausted."""
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import AutonomousLoop
            loop = AutonomousLoop()
            loop.budget.record_usage(loop.budget.session_limit)

            trigger = {"reason": "test", "context": "test", "priority": "low"}
            await loop._start_proactive_session(trigger)

            assert loop._proactive_sessions_today == 0

    @pytest.mark.asyncio
    async def test_stores_in_brain(self, mock_brain, patch_settings):
        """Proactive session should be recorded in brain."""
        with patch("remy.core.autonomy.brain", mock_brain), \
             patch("remy.core.agent.invoke_agent", new_callable=AsyncMock) as mock_invoke, \
             patch("remy.core.brain_tools.get_proactive_context", return_value=""):

            mock_invoke.return_value = ("Hello there!", [], [])

            from remy.core.autonomy import AutonomousLoop
            loop = AutonomousLoop()

            trigger = {"reason": "decay_risk", "context": "Fading memory", "priority": "medium"}

            with patch("telegram.Bot") as MockBot:
                MockBot.return_value = AsyncMock()
                await loop._start_proactive_session(trigger)

        records = mock_brain.search(query="", tags=["proactive-session"], limit=5)
        assert len(records) == 1
        assert records[0].metadata["trigger_reason"] == "decay_risk"

    @pytest.mark.asyncio
    async def test_skips_empty_response(self, mock_brain, patch_settings):
        """Should not send if agent returns empty response."""
        with patch("remy.core.autonomy.brain", mock_brain), \
             patch("remy.core.agent.invoke_agent", new_callable=AsyncMock) as mock_invoke, \
             patch("remy.core.brain_tools.get_proactive_context", return_value=""):

            mock_invoke.return_value = ("", [], [])

            from remy.core.autonomy import AutonomousLoop
            loop = AutonomousLoop()

            trigger = {"reason": "test", "context": "test", "priority": "low"}

            with patch("telegram.Bot") as MockBot:
                mock_bot_instance = AsyncMock()
                MockBot.return_value = mock_bot_instance
                await loop._start_proactive_session(trigger)

            mock_bot_instance.send_message.assert_not_called()
            assert loop._proactive_sessions_today == 0


# ============== CYCLE INTEGRATION ==============


class TestProactiveInCycle:
    """Tests for proactive session integration in _cycle()."""

    @pytest.mark.asyncio
    async def test_cycle_calls_proactive_check(self, mock_brain, patch_settings):
        """_cycle should call _should_start_proactive_session."""
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import AutonomousLoop
            loop = AutonomousLoop()

            with patch.object(loop, "_is_quiet_hours", return_value=False), \
                 patch.object(loop, "_should_start_proactive_session", return_value=None) as mock_check, \
                 patch.object(loop, "_decide_and_act", new_callable=AsyncMock, return_value=None), \
                 patch("remy.core.background_brain.run_background", side_effect=ImportError):
                await loop._cycle()

            mock_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_cycle_starts_proactive_when_triggered(self, mock_brain, patch_settings):
        """When trigger fires, _start_proactive_session is called."""
        trigger = {"reason": "test", "context": "test", "priority": "low"}

        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import AutonomousLoop
            loop = AutonomousLoop()

            with patch.object(loop, "_is_quiet_hours", return_value=False), \
                 patch.object(loop, "_should_start_proactive_session", return_value=trigger), \
                 patch.object(loop, "_start_proactive_session", new_callable=AsyncMock) as mock_start, \
                 patch.object(loop, "_decide_and_act", new_callable=AsyncMock, return_value=None), \
                 patch("remy.core.background_brain.run_background", side_effect=ImportError):
                await loop._cycle()

            mock_start.assert_called_once_with(trigger)


# ============== SETTINGS ==============


class TestProactiveSettings:

    def test_defaults(self):
        from remy.config.settings import Settings
        import os
        os.environ.pop("AUTONOMY_PROACTIVE_SESSIONS_ENABLED", None)
        os.environ.pop("AUTONOMY_PROACTIVE_MAX_PER_DAY", None)
        os.environ.pop("AUTONOMY_PROACTIVE_MIN_INTERVAL_SEC", None)
        s = Settings(GEMINI_API_KEY="test", _env_file=None)
        assert s.AUTONOMY_PROACTIVE_SESSIONS_ENABLED is True
        assert s.AUTONOMY_PROACTIVE_MAX_PER_DAY == 3
        assert s.AUTONOMY_PROACTIVE_MIN_INTERVAL_SEC == 7200
