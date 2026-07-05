"""Tests for Failure Prediction & Pre-Flight Analysis (AUTON-6) — preflight.py."""

from unittest.mock import patch

# ============== Unit Tests: PreflightResult ==============


class TestPreflightResult:
    def test_defaults(self):
        from remy.core.preflight import PreflightResult

        r = PreflightResult()
        assert r.can_proceed is True
        assert r.predicted_success == 0.5
        assert r.difficulty == "medium"
        assert r.warnings == []
        assert r.suggestion == ""

    def test_custom_values(self):
        from remy.core.preflight import PreflightResult

        r = PreflightResult(
            can_proceed=False,
            predicted_success=0.2,
            difficulty="hard",
            warnings=["Low budget"],
            suggestion="skip",
        )
        assert r.can_proceed is False
        assert r.difficulty == "hard"


# ============== Unit Tests: _check_budget ==============


class TestCheckBudget:
    def test_research_goal_low_budget(self):
        from remy.core.preflight import _check_budget

        w = _check_budget("Research AI safety", 1000)
        assert "research" in w.lower() or "budget" in w.lower()

    def test_research_goal_ok_budget(self):
        from remy.core.preflight import _check_budget

        w = _check_budget("Research AI safety", 10000)
        assert w == ""

    def test_browse_goal_low_budget(self):
        from remy.core.preflight import _check_budget

        w = _check_budget("Browse the web for data", 500)
        assert w != ""

    def test_generic_goal_ok_budget(self):
        from remy.core.preflight import _check_budget

        w = _check_budget("Do something", 5000)
        assert w == ""

    def test_generic_goal_low_budget(self):
        from remy.core.preflight import _check_budget

        w = _check_budget("Do something", 500)
        assert w != ""


# ============== Unit Tests: _check_tool_health ==============


class TestCheckToolHealth:
    def test_no_issues_healthy_tools(self):
        from remy.core.preflight import _check_tool_health

        warnings = _check_tool_health(
            "Research topic", {"web_search": "closed", "recall": "closed"}
        )
        assert len(warnings) == 0

    def test_open_circuit_warning(self):
        from remy.core.preflight import _check_tool_health

        warnings = _check_tool_health("Research topic", {"web_search": "open"})
        assert len(warnings) == 1
        assert "UNAVAILABLE" in warnings[0]

    def test_half_open_warning(self):
        from remy.core.preflight import _check_tool_health

        warnings = _check_tool_health("Research topic", {"web_search": "half_open"})
        assert len(warnings) == 1
        assert "WARNING" in warnings[0]

    def test_unrelated_tool_no_warning(self):
        from remy.core.preflight import _check_tool_health

        warnings = _check_tool_health("Research topic", {"write_file": "open"})
        assert len(warnings) == 0


# ============== Unit Tests: _predict_success ==============


class TestPredictSuccess:
    def test_first_attempt_base_prediction(self):
        from remy.core.preflight import _predict_success

        p = _predict_success("Some goal", attempts=0)
        assert 0.3 < p < 1.0

    def test_many_attempts_lower_prediction(self):
        from remy.core.preflight import _predict_success

        p0 = _predict_success("Some goal", attempts=0)
        p5 = _predict_success("Some goal", attempts=5)
        assert p5 < p0

    def test_prediction_never_below_floor(self):
        from remy.core.preflight import _predict_success

        p = _predict_success("Some goal", attempts=20)
        assert p >= 0.05


# ============== Unit Tests: _estimate_difficulty ==============


class TestEstimateDifficulty:
    def test_easy(self):
        from remy.core.preflight import _estimate_difficulty

        assert _estimate_difficulty("Simple task", 0, 0.8) == "easy"

    def test_hard_low_prediction(self):
        from remy.core.preflight import _estimate_difficulty

        assert _estimate_difficulty("Complex task", 2, 0.2) == "hard"

    def test_hard_many_attempts(self):
        from remy.core.preflight import _estimate_difficulty

        assert _estimate_difficulty("Any task", 5, 0.5) == "hard"

    def test_medium(self):
        from remy.core.preflight import _estimate_difficulty

        assert _estimate_difficulty("Normal task", 2, 0.5) == "medium"


# ============== Unit Tests: run_preflight ==============


class TestRunPreflight:
    def test_healthy_goal_proceeds(self):
        from remy.core.preflight import run_preflight

        result = run_preflight("Store user data", goal_attempts=0, budget_tokens_remaining=10000)
        assert result.can_proceed is True
        assert result.suggestion == "proceed"

    def test_low_budget_blocks(self):
        from remy.core.preflight import run_preflight

        result = run_preflight("Anything", budget_tokens_remaining=100)
        assert result.can_proceed is False
        assert result.suggestion == "skip"

    def test_many_failures_suggests_skip(self):
        from remy.core.preflight import run_preflight

        result = run_preflight("Hard goal", goal_attempts=5, budget_tokens_remaining=10000)
        # With 5 attempts, prediction drops significantly
        assert result.suggestion in ("skip", "decompose")

    def test_tool_unavailable_blocks(self):
        from remy.core.preflight import run_preflight

        result = run_preflight(
            "Research something",
            budget_tokens_remaining=10000,
            tool_health_report={"web_search": "open"},
        )
        assert result.can_proceed is False
        assert any("UNAVAILABLE" in w for w in result.warnings)

    def test_moderate_attempts_suggests_decompose(self):
        from remy.core.preflight import run_preflight

        result = run_preflight("Moderate goal", goal_attempts=3, budget_tokens_remaining=10000)
        # After 3 attempts prediction is ~0.3, should suggest decompose
        assert result.suggestion in ("decompose", "proceed")


# ============== Unit Tests: format_preflight_for_prompt ==============


class TestFormatPreflightForPrompt:
    def test_no_warnings_returns_empty(self):
        from remy.core.preflight import PreflightResult, format_preflight_for_prompt

        r = PreflightResult(suggestion="proceed")
        assert format_preflight_for_prompt(r) == ""

    def test_with_warnings_returns_text(self):
        from remy.core.preflight import PreflightResult, format_preflight_for_prompt

        r = PreflightResult(
            predicted_success=0.3,
            difficulty="hard",
            warnings=["Budget low", "Tool failing"],
            suggestion="decompose",
        )
        text = format_preflight_for_prompt(r)
        assert "PRE-FLIGHT" in text
        assert "30%" in text
        assert "hard" in text
        assert "Decompose" in text

    def test_skip_suggestion(self):
        from remy.core.preflight import PreflightResult, format_preflight_for_prompt

        r = PreflightResult(
            predicted_success=0.1,
            warnings=["Very low prediction"],
            suggestion="skip",
        )
        text = format_preflight_for_prompt(r)
        assert "Skip" in text


# ============== Integration: preflight in decision prompt ==============


class TestPreflightInPrompt:
    def test_preflight_text_in_decision_prompt(self):
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
                        "description": "Hard goal that keeps failing",
                        "priority": "high",
                        "attempts": 5,
                        "success_criteria": [],
                    }
                ],
                past_outcomes="",
                budget={
                    "tokens_today": 0,
                    "daily_limit": 100000,
                    "tokens_this_hour": 0,
                    "hourly_limit": 20000,
                },
                preflight_text="\nPRE-FLIGHT ANALYSIS: predicted success 20%\n",
            )

            assert "PRE-FLIGHT" in prompt
