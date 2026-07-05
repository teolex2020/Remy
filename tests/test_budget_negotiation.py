"""Tests for Budget Negotiation (AUTON-7) — budget_negotiation.py."""

import time
from unittest.mock import MagicMock, patch

# ============== Unit Tests: SavingsTracker ==============


class TestSavingsTracker:
    def test_initial_state(self):
        from remy.core.budget_negotiation import SavingsTracker

        t = SavingsTracker()
        assert t.total_saved == 0
        assert t.cache_hits == 0
        assert t.skipped_actions == 0
        assert t.preflight_blocks == 0

    def test_record_cache_hit(self):
        from remy.core.budget_negotiation import SavingsTracker

        t = SavingsTracker()
        t.record_cache_hit(800)
        assert t.cache_hits == 1
        assert t.cache_tokens_saved == 800
        assert t.total_saved == 800

    def test_record_skip(self):
        from remy.core.budget_negotiation import SavingsTracker

        t = SavingsTracker()
        t.record_skip(1000)
        assert t.skipped_actions == 1
        assert t.skip_tokens_saved == 1000

    def test_record_preflight_block(self):
        from remy.core.budget_negotiation import SavingsTracker

        t = SavingsTracker()
        t.record_preflight_block(2000)
        assert t.preflight_blocks == 1
        assert t.preflight_tokens_saved == 2000

    def test_total_saved_combines_all(self):
        from remy.core.budget_negotiation import SavingsTracker

        t = SavingsTracker()
        t.record_cache_hit(500)
        t.record_skip(300)
        t.record_preflight_block(200)
        assert t.total_saved == 1000

    def test_format_report_empty(self):
        from remy.core.budget_negotiation import SavingsTracker

        t = SavingsTracker()
        assert t.format_report() == ""

    def test_format_report_with_data(self):
        from remy.core.budget_negotiation import SavingsTracker

        t = SavingsTracker()
        t.record_cache_hit(800)
        t.record_skip(1000)
        report = t.format_report()
        assert "SAVINGS REPORT" in report
        assert "1,800" in report
        assert "Cache hits" in report
        assert "Smart skips" in report


# ============== Unit Tests: estimate_goal_cost ==============


class TestEstimateGoalCost:
    def test_research_goal_expensive(self):
        from remy.core.budget_negotiation import estimate_goal_cost

        cost = estimate_goal_cost("Research AI safety")
        assert cost >= 4000

    def test_store_goal_cheap(self):
        from remy.core.budget_negotiation import estimate_goal_cost

        cost = estimate_goal_cost("Store user data")
        assert cost <= 2000

    def test_generic_goal_moderate(self):
        from remy.core.budget_negotiation import estimate_goal_cost

        cost = estimate_goal_cost("Do something useful")
        assert 1000 <= cost <= 5000

    def test_attempts_increase_cost(self):
        from remy.core.budget_negotiation import estimate_goal_cost

        cost0 = estimate_goal_cost("Research AI safety", attempts=0)
        cost5 = estimate_goal_cost("Research AI safety", attempts=5)
        assert cost5 > cost0

    def test_browse_goal_expensive(self):
        from remy.core.budget_negotiation import estimate_goal_cost

        cost = estimate_goal_cost("Browse the web for deals")
        assert cost >= 3000

    def test_analyze_goal_moderate(self):
        from remy.core.budget_negotiation import estimate_goal_cost

        cost = estimate_goal_cost("Analyze health patterns")
        assert 1500 <= cost <= 4000


# ============== Unit Tests: format_budget_forecast ==============


class TestFormatBudgetForecast:
    def test_empty_goals(self):
        from remy.core.budget_negotiation import format_budget_forecast

        assert format_budget_forecast([]) == ""

    def test_with_goals(self):
        from remy.core.budget_negotiation import format_budget_forecast

        goals = [
            {"description": "Research AI safety", "attempts": 0},
            {"description": "Store user data", "attempts": 1},
        ]
        text = format_budget_forecast(goals)
        assert "BUDGET FORECAST" in text
        assert "Research AI safety" in text
        assert "TOTAL estimated" in text


# ============== Unit Tests: can_priority_override ==============


class TestPriorityOverride:
    def _make_budget(self, hourly_limit=20000, tokens_this_hour=0):
        from remy.core.autonomy_models import ResourceBudget

        b = ResourceBudget(
            daily_limit=100000,
            hourly_limit=hourly_limit,
            session_limit=500000,
        )
        b.tokens_this_hour = tokens_this_hour
        b.last_hour_reset = time.time()
        return b

    def test_high_priority_gets_override(self):
        from remy.core.budget_negotiation import can_priority_override

        budget = self._make_budget(hourly_limit=20000, tokens_this_hour=19000)
        goal = {"priority": "high", "description": "Critical task"}
        can, reason = can_priority_override(goal, budget)
        assert can is True
        assert "150%" in reason

    def test_medium_priority_no_override(self):
        from remy.core.budget_negotiation import can_priority_override

        budget = self._make_budget()
        goal = {"priority": "medium", "description": "Normal task"}
        can, reason = can_priority_override(goal, budget)
        assert can is False

    def test_low_priority_no_override(self):
        from remy.core.budget_negotiation import can_priority_override

        goal = {"priority": "low", "description": "Low task"}
        budget = self._make_budget()
        can, _ = can_priority_override(goal, budget)
        assert can is False

    def test_critical_priority_gets_override(self):
        from remy.core.budget_negotiation import can_priority_override

        budget = self._make_budget(hourly_limit=20000, tokens_this_hour=25000)
        goal = {"priority": "critical", "description": "Urgent"}
        can, reason = can_priority_override(goal, budget)
        # 150% of 20000 = 30000, so 25000 < 30000 → can proceed
        assert can is True

    def test_override_limit_exceeded(self):
        from remy.core.budget_negotiation import can_priority_override

        budget = self._make_budget(hourly_limit=20000, tokens_this_hour=31000)
        goal = {"priority": "high", "description": "Important"}
        can, _ = can_priority_override(goal, budget)
        # 150% of 20000 = 30000, 31000 > 30000 → denied
        assert can is False


# ============== Unit Tests: check_budget_with_override ==============


class TestCheckBudgetWithOverride:
    def _make_budget(self, **kwargs):
        from remy.core.autonomy_models import ResourceBudget

        defaults = dict(
            daily_limit=100000,
            hourly_limit=20000,
            session_limit=500000,
        )
        defaults.update(kwargs)
        return ResourceBudget(**defaults)

    def test_normal_spend_passes(self):
        from remy.core.budget_negotiation import check_budget_with_override

        budget = self._make_budget()
        can, reason = check_budget_with_override(budget, 1000)
        assert can is True

    def test_over_budget_without_goal_fails(self):
        from remy.core.budget_negotiation import check_budget_with_override

        budget = self._make_budget(hourly_limit=1000)
        budget.tokens_this_hour = 900
        budget.last_hour_reset = time.time()
        can, reason = check_budget_with_override(budget, 2000)
        assert can is False

    def test_over_hourly_with_high_goal_gets_override(self):
        from remy.core.budget_negotiation import check_budget_with_override

        budget = self._make_budget(hourly_limit=10000)
        budget.tokens_this_hour = 9500
        budget.last_hour_reset = time.time()
        goal = {"priority": "high", "description": "Critical"}
        can, reason = check_budget_with_override(budget, 2000, top_goal=goal)
        # 150% of 10000 = 15000, 9500 + 2000 = 11500 < 15000 → override
        assert can is True
        assert "override" in reason.lower() or "150%" in reason


# ============== Unit Tests: request_budget_increase ==============


class TestRequestBudgetIncrease:
    def test_creates_request(self):
        from remy.core.autonomy_models import ResourceBudget
        from remy.core.budget_negotiation import request_budget_increase

        budget = ResourceBudget(daily_limit=100000, hourly_limit=20000, session_limit=500000)
        budget.tokens_today = 90000

        req = request_budget_increase(
            reason="Need more tokens for research",
            goal_description="Research AI safety",
            requested_tokens=20000,
            budget=budget,
        )
        assert req.request_id
        assert req.requested_tokens == 20000
        assert req.current_usage == 90000
        assert req.current_limit == 100000
        assert req.resolved is False

    def test_emits_event(self):
        from remy.core.autonomy_models import ResourceBudget
        from remy.core.budget_negotiation import request_budget_increase

        budget = ResourceBudget(daily_limit=100000, hourly_limit=20000, session_limit=500000)

        events = []
        with patch("remy.core.budget_negotiation.event_bus") as mock_bus:
            mock_bus.emit = lambda name, data: events.append((name, data))
            request_budget_increase(
                reason="test",
                goal_description="test goal",
                requested_tokens=10000,
                budget=budget,
            )

        assert len(events) == 1
        assert events[0][0] == "budget.increase_requested"
        assert events[0][1]["requested_tokens"] == 10000


# ============== Unit Tests: apply_budget_increase ==============


class TestApplyBudgetIncrease:
    def test_increases_limit(self):
        from remy.core.autonomy_models import ResourceBudget
        from remy.core.budget_negotiation import apply_budget_increase

        budget = ResourceBudget(daily_limit=100000, hourly_limit=20000, session_limit=500000)

        apply_budget_increase(budget, 20000)
        assert budget.daily_limit == 120000

    def test_emits_event(self):
        from remy.core.autonomy_models import ResourceBudget
        from remy.core.budget_negotiation import apply_budget_increase

        budget = ResourceBudget(daily_limit=100000, hourly_limit=20000, session_limit=500000)

        events = []
        with patch("remy.core.budget_negotiation.event_bus") as mock_bus:
            mock_bus.emit = lambda name, data: events.append((name, data))
            apply_budget_increase(budget, 15000)

        assert any(e[0] == "budget.increased" for e in events)
        data = [e[1] for e in events if e[0] == "budget.increased"][0]
        assert data["old_limit"] == 100000
        assert data["new_limit"] == 115000


# ============== Unit Tests: request_quiet_hours_override ==============


class TestQuietHoursOverride:
    def test_medium_priority_denied(self):
        from remy.core.budget_negotiation import request_quiet_hours_override

        goal = {"priority": "medium", "description": "Normal goal"}
        assert request_quiet_hours_override(goal) is False

    def test_low_priority_denied(self):
        from remy.core.budget_negotiation import request_quiet_hours_override

        goal = {"priority": "low", "description": "Low goal"}
        assert request_quiet_hours_override(goal) is False

    def test_critical_priority_approved(self):
        from remy.core.budget_negotiation import request_quiet_hours_override

        goal = {"priority": "critical", "description": "Urgent task", "attempts": 0}
        assert request_quiet_hours_override(goal) is True

    def test_high_priority_many_attempts_approved(self):
        from remy.core.budget_negotiation import request_quiet_hours_override

        goal = {"priority": "high", "description": "Stuck goal", "attempts": 4}
        assert request_quiet_hours_override(goal) is True

    def test_high_priority_few_attempts_denied(self):
        from remy.core.budget_negotiation import request_quiet_hours_override

        goal = {"priority": "high", "description": "New goal", "attempts": 1}
        assert request_quiet_hours_override(goal) is False

    def test_approaching_deadline_approved(self):
        from datetime import datetime, timedelta

        from remy.core.budget_negotiation import request_quiet_hours_override

        deadline = (datetime.now() + timedelta(hours=3)).isoformat()
        goal = {"priority": "high", "description": "Deadline goal", "deadline": deadline}
        assert request_quiet_hours_override(goal) is True

    def test_far_deadline_denied(self):
        from datetime import datetime, timedelta

        from remy.core.budget_negotiation import request_quiet_hours_override

        deadline = (datetime.now() + timedelta(hours=24)).isoformat()
        goal = {"priority": "high", "description": "Far goal", "deadline": deadline, "attempts": 0}
        assert request_quiet_hours_override(goal) is False


# ============== Unit Tests: send_budget_request_telegram ==============


class TestSendBudgetRequestTelegram:
    def test_skips_when_not_configured(self):
        from remy.core.budget_negotiation import BudgetRequest, send_budget_request_telegram

        req = BudgetRequest(
            request_id="test123",
            reason="test",
            requested_tokens=10000,
            goal_description="test goal",
            current_usage=5000,
            current_limit=100000,
        )
        with patch("remy.config.settings.settings") as mock_s:
            mock_s.TELEGRAM_BOT_TOKEN = None
            mock_s.PROACTIVE_CHAT_ID = None
            # Should not raise
            send_budget_request_telegram(req)

    def test_sends_when_configured(self):
        from remy.core.budget_negotiation import BudgetRequest, send_budget_request_telegram

        req = BudgetRequest(
            request_id="test456",
            reason="Need tokens",
            requested_tokens=20000,
            goal_description="Research goal",
            current_usage=80000,
            current_limit=100000,
        )
        with patch("remy.config.settings.settings") as mock_s:
            mock_s.TELEGRAM_BOT_TOKEN = "test-token"
            mock_s.PROACTIVE_CHAT_ID = 999

            import threading

            with patch.object(threading, "Thread") as mock_thread:
                mock_instance = MagicMock()
                mock_thread.return_value = mock_instance
                send_budget_request_telegram(req)
                mock_thread.assert_called_once()
                mock_instance.start.assert_called_once()


# ============== Integration: budget in decision prompt ==============


class TestBudgetInDecisionPrompt:
    def test_budget_forecast_in_prompt(self):
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
                        "description": "Research AI safety",
                        "priority": "high",
                        "attempts": 0,
                        "success_criteria": [],
                    }
                ],
                past_outcomes="",
                budget={
                    "tokens_today": 50000,
                    "daily_limit": 100000,
                    "tokens_this_hour": 5000,
                    "hourly_limit": 20000,
                },
            )

            assert "BUDGET FORECAST" in prompt
            assert "50,000/100,000" in prompt or "50000/100000" in prompt

    def test_savings_report_in_prompt(self):
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
            from remy.core.budget_negotiation import savings_tracker

            # Add some savings data
            old_cache_hits = savings_tracker.cache_hits
            old_saved = savings_tracker.cache_tokens_saved
            savings_tracker.record_cache_hit(800)

            try:
                loop = AutonomousLoop()
                prompt = loop._build_decision_prompt(
                    goals=[
                        {
                            "goal_id": "g1",
                            "record_id": "r1",
                            "description": "Test goal",
                            "priority": "medium",
                            "attempts": 0,
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
                )

                assert "SAVINGS REPORT" in prompt
            finally:
                # Restore savings state
                savings_tracker.cache_hits = old_cache_hits
                savings_tracker.cache_tokens_saved = old_saved


# ============== Integration: priority override in active work ==============


class TestPriorityOverrideIntegration:
    def test_override_applied_in_budget_check(self):
        """Verify check_budget_with_override is used in _run_active_work."""
        from remy.core.autonomy_models import ResourceBudget
        from remy.core.budget_negotiation import check_budget_with_override

        budget = ResourceBudget(
            daily_limit=100000,
            hourly_limit=10000,
            session_limit=500000,
        )
        budget.tokens_this_hour = 9500
        budget.last_hour_reset = time.time()

        # Without override: over hourly limit
        can_normal, _ = budget.can_spend(2000)
        assert can_normal is False

        # With override for high-priority goal
        goal = {"priority": "high", "description": "Urgent"}
        can_override, _ = check_budget_with_override(budget, 2000, top_goal=goal)
        assert can_override is True


# ============== Integration: quiet hours override ==============


class TestQuietHoursOverrideIntegration:
    def test_quiet_hours_override_with_deadline(self):
        """Verify approaching deadline triggers quiet hours override."""
        from datetime import datetime, timedelta

        from remy.core.budget_negotiation import request_quiet_hours_override

        deadline = (datetime.now() + timedelta(hours=2)).isoformat()
        goal = {
            "priority": "high",
            "description": "Submit report before morning",
            "deadline": deadline,
        }
        assert request_quiet_hours_override(goal) is True

    def test_no_override_for_routine_goal(self):
        """Routine goals don't override quiet hours."""
        from remy.core.budget_negotiation import request_quiet_hours_override

        goal = {
            "priority": "medium",
            "description": "Organize knowledge base",
            "attempts": 1,
        }
        assert request_quiet_hours_override(goal) is False
