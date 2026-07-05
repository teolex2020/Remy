"""Tests for Multi-Step Action Plans (Feature 5)."""

import json
import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from aura import Aura as CognitiveMemory


@pytest.fixture
def plan_env(tmp_path):
    brain = CognitiveMemory(str(tmp_path / "plan_brain"))

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


class TestCreatePlan:
    def test_create_plan_with_llm(self, plan_env):
        from remy.core.autonomy import create_plan_for_goal

        mock_result = MagicMock()
        mock_result.content = json.dumps([
            "Search for relevant data",
            "Analyze the results",
            "Store findings",
        ])

        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value.invoke.return_value = mock_result
            plan = create_plan_for_goal("goal-123", "Research AI safety")

        assert plan is not None
        assert plan.goal_id == "goal-123"
        assert len(plan.steps) == 3
        assert plan.current_step == 0
        assert plan.status == "active"

    def test_create_plan_fallback_on_failure(self, plan_env):
        from remy.core.autonomy import create_plan_for_goal

        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value.invoke.side_effect = Exception("LLM error")
            plan = create_plan_for_goal("goal-123", "Research AI safety")

        assert plan is None  # Graceful failure

    def test_rejects_single_step(self, plan_env):
        from remy.core.autonomy import create_plan_for_goal

        mock_result = MagicMock()
        mock_result.content = json.dumps(["Only one step"])

        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value.invoke.return_value = mock_result
            plan = create_plan_for_goal("goal-123", "Simple task")

        assert plan is None  # Too few steps


class TestPlanRevision:
    def test_revise_plan_includes_consequence_policy_hints(self, plan_env, monkeypatch):
        from remy.core import autonomy

        old_plan = autonomy.ActionPlan(
            plan_id="plan-old",
            goal_id="goal-1",
            goal_description="Find verified patient facts",
            steps=["Repeat unsafe lookup", "Summarize result"],
            failed_step_history=[
                {"step": "Repeat unsafe lookup", "reason": "unsupported source"},
            ],
        )

        class FakeAura:
            def policy_hint(self, situation, action, namespace=None):
                assert situation == "Find verified patient facts"
                assert action == "Repeat unsafe lookup"
                return {
                    "hint": "avoid",
                    "reason": "Prior consequence refuted this lookup.",
                    "verdict": "refutes",
                    "refutes": 1,
                    "supports": 0,
                    "should_block": True,
                }

        captured = {}

        def fake_call_llm(prompt, purpose=""):
            captured["prompt"] = prompt
            captured["purpose"] = purpose
            result = MagicMock()
            result.content = json.dumps([
                "Search a verified source",
                "Summarize supported result",
            ])
            return result

        monkeypatch.setattr(autonomy, "brain", SimpleNamespace(_aura=FakeAura()))
        monkeypatch.setattr("remy.core.llm.call_llm", fake_call_llm)
        monkeypatch.setattr(autonomy, "_save_plan", lambda plan: None)

        revised = autonomy._revise_plan(old_plan)

        assert revised is not None
        assert captured["purpose"] == "revise_plan"
        assert "LONG-TERM CONSEQUENCE MEMORY" in captured["prompt"]
        assert "Repeat unsafe lookup: policy=avoid" in captured["prompt"]
        assert "Prior consequence refuted this lookup." in captured["prompt"]


class TestAdvancePlan:
    def test_advance_plan_records_refuted_step_consequence(self, plan_env, monkeypatch):
        from remy.core import autonomy

        captures = []

        class FakeBrain:
            def capture_consequence(self, **kwargs):
                captures.append(kwargs)

        plan = autonomy.ActionPlan(
            plan_id="plan-scar",
            goal_id="goal-1",
            goal_description="Find verified patient facts",
            steps=["Repeat unsafe lookup", "Summarize result"],
            current_step=0,
            consecutive_step_failures=1,
        )

        monkeypatch.setattr(autonomy, "brain", FakeBrain())
        monkeypatch.setattr(autonomy, "_save_plan", lambda plan: None)
        monkeypatch.setattr(autonomy, "_revise_plan", lambda plan: None)

        next_step = autonomy.advance_plan(plan, success=False)

        assert next_step is None
        assert captures
        assert captures[0]["situation"] == "Find verified patient facts"
        assert captures[0]["action"] == "Repeat unsafe lookup"
        assert captures[0]["consequence"] == "REFUTES"
        assert captures[0]["trust"] == -1
        assert captures[0]["namespace"] == "remy-autonomy"
        assert "autonomous-plan-step" in captures[0]["scope"]

    def test_advance_on_success(self, plan_env):
        from remy.core.autonomy import ActionPlan, advance_plan

        plan = ActionPlan(
            plan_id="plan-test",
            goal_id="goal-1",
            goal_description="Test goal",
            steps=["Step 1", "Step 2", "Step 3"],
            current_step=0,
        )

        next_step = advance_plan(plan, success=True)
        assert next_step == "Step 2"
        assert plan.current_step == 1

    def test_retry_on_failure(self, plan_env):
        from remy.core.autonomy import ActionPlan, advance_plan

        plan = ActionPlan(
            plan_id="plan-test",
            goal_id="goal-1",
            goal_description="Test goal",
            steps=["Step 1", "Step 2"],
            current_step=0,
        )

        next_step = advance_plan(plan, success=False)
        assert next_step == "Step 1"  # Same step
        assert plan.current_step == 0

    def test_complete_plan(self, plan_env):
        from remy.core.autonomy import ActionPlan, advance_plan

        plan = ActionPlan(
            plan_id="plan-test",
            goal_id="goal-1",
            goal_description="Test goal",
            steps=["Step 1", "Step 2"],
            current_step=1,
        )

        next_step = advance_plan(plan, success=True)
        assert next_step is None
        assert plan.status == "completed"


class TestPlanPersistence:
    def test_save_and_load(self, plan_env):
        from remy.core.autonomy import (
            ActionPlan, _save_plan, load_plan_for_goal,
        )

        plan = ActionPlan(
            plan_id="plan-persist",
            goal_id="goal-persist",
            goal_description="Persistence test",
            steps=["Step A", "Step B"],
            current_step=0,
        )
        _save_plan(plan)

        loaded = load_plan_for_goal("goal-persist")
        assert loaded is not None
        assert loaded.plan_id == "plan-persist"
        assert len(loaded.steps) == 2
        assert loaded.current_step == 0

    def test_load_returns_none_for_missing(self, plan_env):
        from remy.core.autonomy import load_plan_for_goal

        loaded = load_plan_for_goal("nonexistent-goal")
        assert loaded is None


class TestPlanInPrompt:
    def test_prompt_includes_plan_step(self, plan_env):
        from remy.core.autonomy import AutonomousLoop, ActionPlan

        loop = AutonomousLoop()
        plan = ActionPlan(
            plan_id="plan-p",
            goal_id="g1",
            goal_description="Test",
            steps=["Do research", "Write report"],
            current_step=0,
        )

        prompt = loop._build_decision_prompt(
            goals=[{
                "priority": "high",
                "description": "Test goal",
                "deadline": None,
                "attempts": 1,
            }],
            past_outcomes="",
            budget={"tokens_today": 0, "daily_limit": 100000,
                    "tokens_this_hour": 0, "hourly_limit": 20000},
            current_plan=plan,
        )

        assert "ACTION PLAN" in prompt
        assert "Do research" in prompt
        assert "step 1/2" in prompt
