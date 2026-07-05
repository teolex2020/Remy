"""Tests for Adaptive Strategy (Feature 8)."""

import pytest
from unittest.mock import patch
from aura import Aura as CognitiveMemory, Level


@pytest.fixture
def strategy_env(tmp_path):
    brain = CognitiveMemory(str(tmp_path / "strat_brain"))

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


class TestAdaptiveStrategy:
    def test_empty_outcomes(self, strategy_env):
        from remy.core.autonomy import AutonomousLoop

        loop = AutonomousLoop()
        result = loop._analyze_strategy_effectiveness()
        assert result == ""

    def test_computes_success_rates(self, strategy_env):
        from remy.core.autonomy import AutonomousLoop

        # Seed outcomes
        brain = strategy_env["brain"]
        for i in range(5):
            brain.store(
                content=f"Outcome {i}",
                level=Level.DOMAIN,
                tags=["autonomous-outcome"],
                metadata={
                    "action_type": "recall",
                    "success": True,
                },
            )
        for i in range(3):
            brain.store(
                content=f"Fail {i}",
                level=Level.DOMAIN,
                tags=["autonomous-outcome"],
                metadata={
                    "action_type": "web_search",
                    "success": i == 0,  # 1 success, 2 failures
                },
            )

        loop = AutonomousLoop()
        result = loop._analyze_strategy_effectiveness()

        assert "recall" in result
        assert "100%" in result  # 5/5 success
        assert "web_search" in result
        assert "33%" in result  # 1/3 success

    def test_skips_types_with_few_attempts(self, strategy_env):
        from remy.core.autonomy import AutonomousLoop

        brain = strategy_env["brain"]
        brain.store(
            content="Single outcome",
            level=Level.DOMAIN,
            tags=["autonomous-outcome"],
            metadata={
                "action_type": "rare_action",
                "success": True,
            },
        )

        loop = AutonomousLoop()
        result = loop._analyze_strategy_effectiveness()
        assert "rare_action" not in result  # < 2 attempts = skipped

    def test_strategy_in_prompt(self, strategy_env):
        from remy.core.autonomy import AutonomousLoop

        loop = AutonomousLoop()

        prompt = loop._build_decision_prompt(
            goals=[],
            past_outcomes="",
            budget={"tokens_today": 0, "daily_limit": 100000,
                    "tokens_this_hour": 0, "hourly_limit": 20000},
            strategy_hints="✓ recall: 90% success (10 attempts)",
        )

        assert "STRATEGY INSIGHTS" in prompt
        assert "recall: 90%" in prompt
