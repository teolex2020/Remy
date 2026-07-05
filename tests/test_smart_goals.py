"""Tests for Smart Goal Generation (Feature 7)."""

import json
import pytest
from unittest.mock import patch, MagicMock
from aura import Aura as CognitiveMemory


@pytest.fixture
def goals_env(tmp_path):
    brain = CognitiveMemory(str(tmp_path / "goals_brain"))

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


class TestSmartGoals:
    def test_generates_goals_from_llm(self, goals_env):
        from remy.core.autonomy import AutonomousLoop, get_active_goals

        loop = AutonomousLoop()

        mock_result = MagicMock()
        mock_result.content = json.dumps([
            {"description": "Organize health records", "priority": "high"},
            {"description": "Research nutrition trends", "priority": "medium"},
        ])

        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value.invoke.return_value = mock_result
            created = loop._generate_smart_goals()

        assert len(created) == 2
        goals = get_active_goals()
        descs = [g["description"] for g in goals]
        assert any("Organize health records" in d for d in descs)

    def test_fallback_goals_on_failure(self, goals_env):
        from remy.core.autonomy import AutonomousLoop, get_active_goals

        loop = AutonomousLoop()

        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value.invoke.side_effect = Exception("LLM down")
            created = loop._generate_smart_goals()

        assert len(created) == 2  # Default goals
        goals = get_active_goals()
        assert len(goals) >= 2

    def test_caps_at_3_goals(self, goals_env):
        from remy.core.autonomy import AutonomousLoop

        loop = AutonomousLoop()

        mock_result = MagicMock()
        mock_result.content = json.dumps([
            {"description": f"Goal {i}", "priority": "low"}
            for i in range(6)
        ])

        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value.invoke.return_value = mock_result
            created = loop._generate_smart_goals()

        assert len(created) == 3  # Capped at 3

    def test_seed_initial_goals_uses_smart(self, goals_env):
        from remy.core.autonomy import AutonomousLoop

        loop = AutonomousLoop()

        with patch.object(loop, "_generate_smart_goals", return_value=["g1"]) as mock:
            loop._seed_initial_goals()
            mock.assert_called_once()
