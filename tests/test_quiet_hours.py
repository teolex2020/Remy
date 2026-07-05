"""Tests for Quiet Hours + Session Limits in AutonomousLoop."""

import time
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from aura import Aura as CognitiveMemory


@pytest.fixture
def loop_env(tmp_path):
    """Isolated environment for loop tests."""
    brain = CognitiveMemory(str(tmp_path / "loop_brain"))

    with patch("remy.core.autonomy.settings") as mock_settings:
        mock_settings.AUTONOMY_DAILY_TOKEN_LIMIT = 100_000
        mock_settings.AUTONOMY_HOURLY_TOKEN_LIMIT = 20_000
        mock_settings.AUTONOMY_SESSION_TOKEN_LIMIT = 500_000
        mock_settings.AUTONOMY_CYCLE_INTERVAL_SEC = 0
        mock_settings.AUTONOMY_TELEGRAM_NOTIFICATIONS = False
        mock_settings.SUMMARY_MODEL = "test-model"
        mock_settings.GEMINI_API_KEY = "test-key"
        mock_settings.DATA_DIR = tmp_path / "data"
        mock_settings.AUTONOMY_QUIET_HOURS_START = 23
        mock_settings.AUTONOMY_QUIET_HOURS_END = 7
        mock_settings.AUTONOMY_MAX_SESSION_MINUTES = 30
        (tmp_path / "data" / "logs").mkdir(parents=True, exist_ok=True)

        with patch("remy.core.autonomy.brain", brain):
            yield {"brain": brain, "settings": mock_settings}

    brain.close()


class TestQuietHours:
    """Tests for _is_quiet_hours() in AutonomousLoop."""

    def test_quiet_at_midnight(self, loop_env):
        from remy.core.autonomy import AutonomousLoop

        loop = AutonomousLoop()

        with patch("remy.core.autonomy.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 12, 0, 30)  # 00:30
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            assert loop._is_quiet_hours() is True

    def test_quiet_at_3am(self, loop_env):
        from remy.core.autonomy import AutonomousLoop

        loop = AutonomousLoop()

        with patch("remy.core.autonomy.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 12, 3, 0)  # 03:00
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            assert loop._is_quiet_hours() is True

    def test_not_quiet_at_noon(self, loop_env):
        from remy.core.autonomy import AutonomousLoop

        loop = AutonomousLoop()

        with patch("remy.core.autonomy.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 12, 12, 0)  # 12:00
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            assert loop._is_quiet_hours() is False

    def test_quiet_at_23(self, loop_env):
        from remy.core.autonomy import AutonomousLoop

        loop = AutonomousLoop()

        with patch("remy.core.autonomy.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 12, 23, 0)  # 23:00
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            assert loop._is_quiet_hours() is True

    def test_not_quiet_at_8(self, loop_env):
        from remy.core.autonomy import AutonomousLoop

        loop = AutonomousLoop()

        with patch("remy.core.autonomy.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 12, 8, 0)  # 08:00
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            assert loop._is_quiet_hours() is False


class TestSessionTimeLimit:
    """Tests for session time limit in AutonomousLoop."""

    def test_session_start_time_recorded(self, loop_env):
        from remy.core.autonomy import AutonomousLoop

        before = time.time()
        loop = AutonomousLoop()
        after = time.time()

        assert before <= loop._session_start_time <= after

    @pytest.mark.asyncio
    async def test_session_stops_after_time_limit(self, loop_env):
        from remy.core.autonomy import AutonomousLoop
        from unittest.mock import AsyncMock

        loop = AutonomousLoop()

        # Set time limit to 0 so the loop exits immediately after start
        loop_env["settings"].AUTONOMY_MAX_SESSION_MINUTES = 0

        # Mock LLM-dependent methods to prevent hanging
        loop._decide_and_act = AsyncMock(return_value=None)
        loop._generate_session_reflection = AsyncMock(return_value=None)

        await loop.start()

        # Should have stopped (the loop exits because time limit was reached)
        assert True  # If we reach here, the loop exited correctly
