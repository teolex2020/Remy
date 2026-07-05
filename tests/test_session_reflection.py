"""Tests for Session Reflection (Feature 6)."""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from aura import Aura as CognitiveMemory


@pytest.fixture
def reflect_env(tmp_path):
    brain = CognitiveMemory(str(tmp_path / "reflect_brain"))

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


class TestSessionReflection:
    @pytest.mark.asyncio
    async def test_generates_reflection(self, reflect_env):
        from remy.core.autonomy import AutonomousLoop, ActionRecord

        loop = AutonomousLoop()
        loop.action_log = [
            ActionRecord("a1", "2026-02-12T10:00:00", None, "agent_invoke",
                         "Searched for health data", "Found results", True, 500, 1000),
            ActionRecord("a2", "2026-02-12T10:05:00", None, "agent_invoke",
                         "Stored research findings", "Stored successfully", True, 300, 800),
        ]

        mock_result = MagicMock()
        mock_result.content = "- Search was effective\n- Should store more specific data"

        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value.invoke.return_value = mock_result
            reflection = await loop._generate_session_reflection()

        assert reflection is not None
        assert "Search was effective" in reflection

        # Check it was stored in brain
        stored = reflect_env["brain"].search(query="", tags=["session-reflection"], limit=1)
        assert len(stored) == 1

    @pytest.mark.asyncio
    async def test_reflection_empty_log(self, reflect_env):
        from remy.core.autonomy import AutonomousLoop

        loop = AutonomousLoop()
        loop.action_log = []

        reflection = await loop._generate_session_reflection()
        assert reflection is None

    @pytest.mark.asyncio
    async def test_reflection_fallback(self, reflect_env):
        from remy.core.autonomy import AutonomousLoop, ActionRecord

        loop = AutonomousLoop()
        loop.action_log = [
            ActionRecord("a1", "2026-02-12T10:00:00", None, "agent_invoke",
                         "Did something", "Result", True, 500, 1000),
        ]

        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value.invoke.side_effect = Exception("LLM down")
            reflection = await loop._generate_session_reflection()

        assert reflection is None  # Graceful failure


class TestGetLastReflection:
    def test_retrieves_stored_reflection(self, reflect_env):
        from remy.core.autonomy import AutonomousLoop
        from aura import Level

        reflect_env["brain"].store(
            content="- Lesson 1\n- Lesson 2",
            level=Level.DOMAIN,
            tags=["session-reflection"],
            metadata={"type": "session_reflection"},
        )

        loop = AutonomousLoop()
        reflection = loop._get_last_reflection()

        assert "Lesson 1" in reflection

    def test_returns_empty_if_none(self, reflect_env):
        from remy.core.autonomy import AutonomousLoop

        loop = AutonomousLoop()
        reflection = loop._get_last_reflection()
        assert reflection == ""
