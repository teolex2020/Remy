"""Tests for Proactive Error Escalation (AUTON-10) — error_escalation.py."""

import time
from unittest.mock import AsyncMock, patch

import pytest

# ============== Unit Tests: DegradationLevel ==============


class TestDegradationLevel:
    def test_green_lowest(self):
        from remy.core.error_escalation import DegradationLevel

        assert DegradationLevel.GREEN < DegradationLevel.YELLOW
        assert DegradationLevel.YELLOW < DegradationLevel.RED

    def test_values(self):
        from remy.core.error_escalation import DegradationLevel

        assert DegradationLevel.GREEN == 0
        assert DegradationLevel.YELLOW == 1
        assert DegradationLevel.RED == 2


# ============== Unit Tests: assess_system_health ==============


class TestAssessSystemHealth:
    def test_all_healthy_is_green(self):
        from remy.core.error_escalation import DegradationLevel, assess_system_health

        health = assess_system_health(
            budget_dict={"daily_limit": 100000, "tokens_today": 0},
            maintenance_only=False,
        )
        assert health.level == DegradationLevel.GREEN
        assert health.llm_available is True
        assert health.budget_pct == 100.0

    def test_maintenance_mode_is_red(self):
        from remy.core.error_escalation import DegradationLevel, assess_system_health

        health = assess_system_health(maintenance_only=True)
        assert health.level == DegradationLevel.RED
        assert health.llm_available is False

    def test_low_budget_is_yellow(self):
        from remy.core.error_escalation import DegradationLevel, assess_system_health

        health = assess_system_health(
            budget_dict={"daily_limit": 100000, "tokens_today": 75000},
        )
        assert health.level == DegradationLevel.YELLOW
        assert health.budget_pct == 25.0

    def test_critical_budget_is_red(self):
        from remy.core.error_escalation import DegradationLevel, assess_system_health

        health = assess_system_health(
            budget_dict={"daily_limit": 100000, "tokens_today": 95000},
        )
        assert health.level == DegradationLevel.RED
        assert abs(health.budget_pct - 5.0) < 0.01

    def test_no_budget_defaults_to_100(self):
        from remy.core.error_escalation import DegradationLevel, assess_system_health

        health = assess_system_health()
        assert health.budget_pct == 100.0
        assert health.level == DegradationLevel.GREEN

    def test_circuit_breakers_affect_level(self):
        from remy.core.error_escalation import DegradationLevel, assess_system_health
        from remy.core.tool_health import tool_health

        # Open 3 circuit breakers
        for tool in ["web_search", "http_get", "browse_page"]:
            tool_health._circuit_open_until[tool] = time.time() + 600
            tool_health._failures[tool] = [time.time()] * 3

        try:
            health = assess_system_health(
                budget_dict={"daily_limit": 100000, "tokens_today": 0},
            )
            assert health.level == DegradationLevel.RED
            assert len(health.circuit_breakers_open) == 3
        finally:
            # Clean up
            for tool in ["web_search", "http_get", "browse_page"]:
                tool_health._circuit_open_until.pop(tool, None)
                tool_health._failures.pop(tool, None)


# ============== Unit Tests: get_recovery_suggestions ==============


class TestRecoverySuggestions:
    def test_llm_suggestions(self):
        from remy.core.error_escalation import get_recovery_suggestions

        suggestions = get_recovery_suggestions("llm_unavailable")
        assert len(suggestions) > 0
        assert any("API" in s for s in suggestions)

    def test_budget_suggestions(self):
        from remy.core.error_escalation import get_recovery_suggestions

        suggestions = get_recovery_suggestions("budget_low")
        assert len(suggestions) > 0

    def test_unknown_issue(self):
        from remy.core.error_escalation import get_recovery_suggestions

        suggestions = get_recovery_suggestions("unknown_issue_xyz")
        assert len(suggestions) > 0  # Returns default


# ============== Unit Tests: build_alert_message ==============


class TestBuildAlertMessage:
    def setup_method(self):
        from remy.core.error_escalation import reset_alert_cooldowns

        reset_alert_cooldowns()

    def test_no_alert_when_healthy(self):
        from remy.core.error_escalation import (
            DegradationLevel,
            SystemHealth,
            build_alert_message,
        )

        health = SystemHealth(level=DegradationLevel.GREEN, llm_available=True, budget_pct=80)
        assert build_alert_message(health) is None

    def test_alert_on_llm_down(self):
        from remy.core.error_escalation import (
            DegradationLevel,
            SystemHealth,
            build_alert_message,
        )

        health = SystemHealth(
            level=DegradationLevel.RED,
            llm_available=False,
            budget_pct=80,
        )
        msg = build_alert_message(health)
        assert msg is not None
        assert "CRITICAL" in msg
        assert "LLM" in msg

    def test_alert_on_low_budget(self):
        from remy.core.error_escalation import (
            DegradationLevel,
            SystemHealth,
            build_alert_message,
        )

        health = SystemHealth(
            level=DegradationLevel.RED,
            llm_available=True,
            budget_pct=5,
        )
        msg = build_alert_message(health)
        assert msg is not None
        assert "budget" in msg.lower() or "Budget" in msg

    def test_alert_on_mass_tool_failure(self):
        from remy.core.error_escalation import (
            DegradationLevel,
            SystemHealth,
            build_alert_message,
        )

        health = SystemHealth(
            level=DegradationLevel.RED,
            llm_available=True,
            budget_pct=80,
            circuit_breakers_open=["web_search", "http_get", "browse_page"],
        )
        msg = build_alert_message(health)
        assert msg is not None
        assert "tools" in msg.lower() or "unavailable" in msg.lower()

    def test_cooldown_prevents_duplicate(self):
        from remy.core.error_escalation import (
            DegradationLevel,
            SystemHealth,
            build_alert_message,
        )

        health = SystemHealth(
            level=DegradationLevel.RED,
            llm_available=False,
            budget_pct=80,
        )
        msg1 = build_alert_message(health)
        msg2 = build_alert_message(health)
        assert msg1 is not None
        assert msg2 is None  # Cooldown active

    def test_combined_alerts(self):
        from remy.core.error_escalation import (
            DegradationLevel,
            SystemHealth,
            build_alert_message,
        )

        health = SystemHealth(
            level=DegradationLevel.RED,
            llm_available=False,
            budget_pct=5,
            circuit_breakers_open=["a", "b", "c"],
        )
        msg = build_alert_message(health)
        assert msg is not None
        assert "LLM" in msg
        assert "budget" in msg.lower() or "Budget" in msg


class TestBuildOperatorWatchMessage:
    def setup_method(self):
        from remy.core.error_escalation import reset_alert_cooldowns

        reset_alert_cooldowns()

    def test_no_message_when_healthy_and_stable(self):
        from remy.core.error_escalation import (
            DegradationLevel,
            SystemHealth,
            build_operator_watch_message,
        )

        health = SystemHealth(level=DegradationLevel.GREEN, llm_available=True, budget_pct=90)
        assert (
            build_operator_watch_message(
                health,
                previous_level=DegradationLevel.GREEN,
                gateway_health="ok",
                previous_gateway_health="ok",
            )
            is None
        )

    def test_message_on_degradation_transition(self):
        from remy.core.error_escalation import (
            DegradationLevel,
            SystemHealth,
            build_operator_watch_message,
        )

        health = SystemHealth(level=DegradationLevel.YELLOW, llm_available=True, budget_pct=24)
        result = build_operator_watch_message(
            health,
            previous_level=DegradationLevel.GREEN,
            gateway_health="degraded",
            previous_gateway_health="ok",
        )
        assert result is not None
        message, level = result
        assert "degraded" in message.lower()
        assert level == "warning"

    def test_message_on_recovery_transition(self):
        from remy.core.error_escalation import (
            DegradationLevel,
            SystemHealth,
            build_operator_watch_message,
        )

        health = SystemHealth(level=DegradationLevel.GREEN, llm_available=True, budget_pct=80)
        result = build_operator_watch_message(
            health,
            previous_level=DegradationLevel.RED,
            gateway_health="ok",
            previous_gateway_health="error",
        )
        assert result is not None
        message, level = result
        assert "recovered" in message.lower()
        assert level == "info"


# ============== Unit Tests: should_skip_cycle ==============


class TestShouldSkipCycle:
    def test_skip_when_red_and_llm_down(self):
        from remy.core.error_escalation import (
            DegradationLevel,
            SystemHealth,
            should_skip_cycle,
        )

        health = SystemHealth(level=DegradationLevel.RED, llm_available=False)
        assert should_skip_cycle(health) is True

    def test_no_skip_when_red_but_llm_up(self):
        from remy.core.error_escalation import (
            DegradationLevel,
            SystemHealth,
            should_skip_cycle,
        )

        health = SystemHealth(level=DegradationLevel.RED, llm_available=True)
        assert should_skip_cycle(health) is False

    def test_no_skip_when_green(self):
        from remy.core.error_escalation import (
            DegradationLevel,
            SystemHealth,
            should_skip_cycle,
        )

        health = SystemHealth(level=DegradationLevel.GREEN, llm_available=True)
        assert should_skip_cycle(health) is False


# ============== Unit Tests: get_strategy_for_level ==============


class TestStrategy:
    def test_green_is_full(self):
        from remy.core.error_escalation import DegradationLevel, get_strategy_for_level

        assert get_strategy_for_level(DegradationLevel.GREEN) == "full"

    def test_yellow_is_conservative(self):
        from remy.core.error_escalation import DegradationLevel, get_strategy_for_level

        assert get_strategy_for_level(DegradationLevel.YELLOW) == "conservative"

    def test_red_is_maintenance(self):
        from remy.core.error_escalation import DegradationLevel, get_strategy_for_level

        assert get_strategy_for_level(DegradationLevel.RED) == "maintenance"


# ============== Unit Tests: send_critical_alert ==============


class TestSendCriticalAlert:
    def setup_method(self):
        from remy.core.error_escalation import reset_alert_cooldowns

        reset_alert_cooldowns()

    @pytest.mark.asyncio
    async def test_no_send_when_healthy(self):
        from remy.core.error_escalation import (
            DegradationLevel,
            SystemHealth,
            send_critical_alert,
        )

        health = SystemHealth(level=DegradationLevel.GREEN, llm_available=True)
        sent = await send_critical_alert(health)
        assert sent is False

    @pytest.mark.asyncio
    async def test_send_when_llm_down_with_telegram(self):
        from remy.core.error_escalation import (
            DegradationLevel,
            SystemHealth,
            send_critical_alert,
        )

        health = SystemHealth(level=DegradationLevel.RED, llm_available=False)

        mock_bot = AsyncMock()
        mock_settings = type(
            "S",
            (),
            {
                "TELEGRAM_BOT_TOKEN": "fake-token",
                "PROACTIVE_CHAT_ID": "12345",
            },
        )()

        with (
            patch("telegram.Bot", return_value=mock_bot),
            patch("remy.config.settings.settings", mock_settings),
        ):
            sent = await send_critical_alert(health)
            assert sent is True
            mock_bot.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_send_without_telegram(self):
        from remy.core.error_escalation import (
            DegradationLevel,
            SystemHealth,
            send_critical_alert,
        )

        health = SystemHealth(level=DegradationLevel.RED, llm_available=False)

        mock_settings = type(
            "S",
            (),
            {
                "TELEGRAM_BOT_TOKEN": "",
                "PROACTIVE_CHAT_ID": "",
            },
        )()
        with patch("remy.config.settings.settings", mock_settings):
            sent = await send_critical_alert(health)
            assert sent is False


# ============== Unit Tests: get_detailed_health ==============


class TestDetailedHealth:
    def test_returns_all_fields(self):
        from remy.core.error_escalation import get_detailed_health

        result = get_detailed_health(
            budget_dict={"daily_limit": 100000, "tokens_today": 50000},
        )
        assert "level" in result
        assert "strategy" in result
        assert "llm_available" in result
        assert "budget_remaining_pct" in result
        assert "tools" in result
        assert result["budget_remaining_pct"] == 50.0

    def test_healthy_system(self):
        from remy.core.error_escalation import get_detailed_health

        result = get_detailed_health(
            budget_dict={"daily_limit": 100000, "tokens_today": 0},
        )
        assert result["level"] == "GREEN"
        assert result["strategy"] == "full"

    def test_maintenance_mode(self):
        from remy.core.error_escalation import get_detailed_health

        result = get_detailed_health(maintenance_only=True)
        assert result["level"] == "RED"
        assert result["strategy"] == "maintenance"
        assert len(result["recovery_suggestions"]) > 0


# ============== Unit Tests: auto-recovery ==============


class TestAutoRecovery:
    @pytest.mark.asyncio
    async def test_recovery_when_llm_back(self):
        from remy.core.error_escalation import (
            DegradationLevel,
            SystemHealth,
            attempt_auto_recovery,
        )

        health = SystemHealth(level=DegradationLevel.RED, llm_available=False)

        with patch("remy.core.error_escalation._test_llm_health", return_value=True):
            result = await attempt_auto_recovery(health)
            assert result is True

    @pytest.mark.asyncio
    async def test_no_recovery_when_llm_still_down(self):
        from remy.core.error_escalation import (
            DegradationLevel,
            SystemHealth,
            attempt_auto_recovery,
        )

        health = SystemHealth(level=DegradationLevel.RED, llm_available=False)

        with patch("remy.core.error_escalation._test_llm_health", return_value=False):
            result = await attempt_auto_recovery(health)
            assert result is False

    @pytest.mark.asyncio
    async def test_no_recovery_when_llm_available(self):
        from remy.core.error_escalation import (
            DegradationLevel,
            SystemHealth,
            attempt_auto_recovery,
        )

        health = SystemHealth(level=DegradationLevel.RED, llm_available=True)
        result = await attempt_auto_recovery(health)
        assert result is False
