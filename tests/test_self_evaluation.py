"""Tests for Self-Evaluation Loop — _evaluate_outcome() in AutonomousLoop."""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from aura import Aura as CognitiveMemory


@pytest.fixture
def eval_env(tmp_path):
    """Isolated environment for evaluation tests."""
    brain = CognitiveMemory(str(tmp_path / "eval_brain"))

    with patch("remy.core.autonomy.settings") as mock_settings:
        mock_settings.AUTONOMY_DAILY_TOKEN_LIMIT = 100_000
        mock_settings.AUTONOMY_HOURLY_TOKEN_LIMIT = 20_000
        mock_settings.AUTONOMY_SESSION_TOKEN_LIMIT = 500_000
        mock_settings.AUTONOMY_CYCLE_INTERVAL_SEC = 0
        mock_settings.AUTONOMY_TELEGRAM_NOTIFICATIONS = False
        mock_settings.SUMMARY_MODEL = "test-model"
        mock_settings.GEMINI_API_KEY = "test-key"
        mock_settings.DATA_DIR = tmp_path / "data"
        (tmp_path / "data" / "logs").mkdir(parents=True, exist_ok=True)

        with patch("remy.core.autonomy.brain", brain):
            yield {"brain": brain}

    brain.close()


class TestEvaluateOutcome:
    """Tests for AutonomousLoop._evaluate_outcome()."""

    @pytest.mark.asyncio
    async def test_success_detected(self, eval_env):
        from remy.core.autonomy import AutonomousLoop

        loop = AutonomousLoop()

        # Mock LLM to return success evaluation
        mock_llm_response = MagicMock()
        mock_llm_response.content = json.dumps({
            "success": True,
            "confidence": 0.9,
            "reason": "Agent successfully recalled relevant data",
            "goal_completed": False,
        })

        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as MockLLM:
            instance = MockLLM.return_value
            instance.invoke.return_value = mock_llm_response

            result = await loop._evaluate_outcome(
                "Organize family records",
                "I found and organized 5 family records related to the Smith family.",
            )

        assert result["success"] is True
        assert result["confidence"] == 0.9
        assert "recalled" in result["reason"]
        assert result["goal_completed"] is False

    @pytest.mark.asyncio
    async def test_failure_detected(self, eval_env):
        from remy.core.autonomy import AutonomousLoop

        loop = AutonomousLoop()

        mock_llm_response = MagicMock()
        mock_llm_response.content = json.dumps({
            "success": False,
            "confidence": 0.85,
            "reason": "Agent could not find the requested information",
            "goal_completed": False,
        })

        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as MockLLM:
            instance = MockLLM.return_value
            instance.invoke.return_value = mock_llm_response

            result = await loop._evaluate_outcome(
                "Find Atlantis",
                "I searched everywhere but could not find Atlantis.",
            )

        assert result["success"] is False
        assert result["confidence"] == 0.85
        assert result["goal_completed"] is False

    @pytest.mark.asyncio
    async def test_goal_completed_detected(self, eval_env):
        from remy.core.autonomy import AutonomousLoop

        loop = AutonomousLoop()

        mock_llm_response = MagicMock()
        mock_llm_response.content = json.dumps({
            "success": True,
            "confidence": 0.95,
            "reason": "Goal fully achieved — all records organized",
            "goal_completed": True,
        })

        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as MockLLM:
            instance = MockLLM.return_value
            instance.invoke.return_value = mock_llm_response

            result = await loop._evaluate_outcome(
                "Organize all records",
                "Done! All 12 records are now properly tagged and connected.",
            )

        assert result["success"] is True
        assert result["goal_completed"] is True

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self, eval_env):
        from remy.core.autonomy import AutonomousLoop

        loop = AutonomousLoop()

        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as MockLLM:
            instance = MockLLM.return_value
            instance.invoke.side_effect = Exception("API quota exceeded")

            result = await loop._evaluate_outcome(
                "Some goal",
                "Some response",
            )

        # Should fallback gracefully (defaults to failure so goal doesn't loop forever)
        assert result["success"] is False
        assert result["confidence"] == 0.3
        assert "Evaluation unavailable" in result["reason"]
        assert result["goal_completed"] is False

    @pytest.mark.asyncio
    async def test_handles_markdown_fenced_json(self, eval_env):
        from remy.core.autonomy import AutonomousLoop

        loop = AutonomousLoop()

        mock_llm_response = MagicMock()
        mock_llm_response.content = '```json\n{"success": false, "confidence": 0.7, "reason": "failed", "goal_completed": false}\n```'

        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as MockLLM:
            instance = MockLLM.return_value
            instance.invoke.return_value = mock_llm_response

            result = await loop._evaluate_outcome("goal", "response")

        assert result["success"] is False
        assert result["confidence"] == 0.7

    @pytest.mark.asyncio
    async def test_confidence_score_in_range(self, eval_env):
        from remy.core.autonomy import AutonomousLoop

        loop = AutonomousLoop()

        # Test with various confidence values
        for conf in [0.0, 0.5, 1.0]:
            mock_llm_response = MagicMock()
            mock_llm_response.content = json.dumps({
                "success": True, "confidence": conf,
                "reason": "test", "goal_completed": False,
            })

            with patch("langchain_google_genai.ChatGoogleGenerativeAI") as MockLLM:
                instance = MockLLM.return_value
                instance.invoke.return_value = mock_llm_response

                result = await loop._evaluate_outcome("goal", "response")
                assert 0.0 <= result["confidence"] <= 1.0


class TestSelfEvaluationIntegration:
    """Test that _decide_and_act uses self-evaluation properly."""

    @pytest.mark.asyncio
    async def test_decide_and_act_uses_evaluation(self, eval_env):
        """Verify _decide_and_act calls _evaluate_outcome instead of hardcoding success."""
        from remy.core.autonomy import AutonomousLoop, create_goal

        create_goal("Test goal", priority="medium")
        loop = AutonomousLoop()

        # Mock invoke_agent
        with patch("remy.core.agent.invoke_agent", new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = ("I failed to do the thing.", [], [])

            # Mock evaluation to return failure
            eval_result = {
                "success": False, "confidence": 0.8,
                "reason": "Agent failed", "goal_completed": False,
            }
            with patch.object(loop, "_evaluate_outcome", new_callable=AsyncMock) as mock_eval:
                mock_eval.return_value = eval_result

                action = await loop._decide_and_act()

        assert action is not None
        assert action.success is False  # Would be True without self-evaluation

    @pytest.mark.asyncio
    async def test_decide_and_act_records_turn_class_from_tools(self, eval_env):
        """Turn class comes from actual tools used, not success/failure alone."""
        from remy.core.autonomy import AutonomousLoop, create_goal

        create_goal("Maintenance-only goal", priority="medium")
        loop = AutonomousLoop()
        session_log = [
            {
                "type": "tool_call",
                "tool": "tool_status",
                "args": {},
                "result": "{\"ok\": true}",
            }
        ]
        captured = {}

        async def fake_dispatch_worker(**kwargs):
            return ("Checked tool status.", [], session_log, None)

        with patch("remy.core.orchestrator.dispatch_worker", new_callable=AsyncMock) as mock_dispatch:
            mock_dispatch.side_effect = fake_dispatch_worker
            with patch.object(loop, "_evaluate_outcome", new_callable=AsyncMock) as mock_eval, \
                 patch("remy.core.orchestrator.check_obvious_failure", return_value=None), \
                 patch("remy.core.orchestrator.check_zero_tool_cycle", return_value=None), \
                 patch("remy.core.execution_log.record_cycle_execution") as mock_record:
                mock_eval.return_value = {
                    "success": True,
                    "confidence": 0.9,
                    "reason": "Maintenance check completed",
                    "goal_completed": False,
                }
                mock_record.side_effect = lambda **kwargs: captured.update(kwargs)

                action = await loop._decide_and_act()

        assert action is not None
        assert action.turn_class == "maintenance"
        assert captured["turn_class"] == "maintenance"

    @pytest.mark.asyncio
    async def test_decide_and_act_injects_preflight_prompt(self, eval_env):
        """Preflight analysis is wired into the runtime decision prompt."""
        from remy.core.autonomy import AutonomousLoop, create_goal

        create_goal("Research a difficult topic", priority="medium")
        loop = AutonomousLoop()
        loop.budget.tokens_today = loop.budget.daily_limit - 100
        captured = {}

        async def fake_dispatch_worker(**kwargs):
            captured["decision_prompt"] = kwargs["decision_prompt"]
            return ("No action yet.", [], [], None)

        with patch("remy.core.orchestrator.dispatch_worker", new_callable=AsyncMock) as mock_dispatch:
            mock_dispatch.side_effect = fake_dispatch_worker
            with patch.object(loop, "_evaluate_outcome", new_callable=AsyncMock) as mock_eval, \
                 patch("remy.core.orchestrator.check_obvious_failure", return_value=None), \
                 patch("remy.core.orchestrator.check_zero_tool_cycle", return_value=None):
                mock_eval.return_value = {
                    "success": False,
                    "confidence": 0.7,
                    "reason": "No progress",
                    "goal_completed": False,
                }

                await loop._decide_and_act()

        assert "PRE-FLIGHT ANALYSIS" in captured["decision_prompt"]
        assert loop.status()["last_preflight"]["suggestion"] == "skip"

    @pytest.mark.asyncio
    async def test_decide_and_act_injects_loop_detection_prompt(self, eval_env):
        """Repeated tool patterns are surfaced to the next autonomy prompt."""
        from remy.core.autonomy import AutonomousLoop, create_goal

        create_goal("Keep checking system status", priority="medium")
        loop = AutonomousLoop()
        session_log = [
            {
                "type": "tool_call",
                "tool": "tool_status",
                "args": {"scope": "health"},
                "result": "{\"ok\": true}",
            }
        ]
        prompts = []

        async def fake_dispatch_worker(**kwargs):
            prompts.append(kwargs["decision_prompt"])
            return ("Checked tool status.", [], session_log, None)

        with patch("remy.core.orchestrator.dispatch_worker", new_callable=AsyncMock) as mock_dispatch:
            mock_dispatch.side_effect = fake_dispatch_worker
            with patch.object(loop, "_evaluate_outcome", new_callable=AsyncMock) as mock_eval, \
                 patch("remy.core.orchestrator.check_obvious_failure", return_value=None), \
                 patch("remy.core.orchestrator.check_zero_tool_cycle", return_value=None), \
                 patch("remy.core.execution_log.record_cycle_execution"):
                mock_eval.return_value = {
                    "success": True,
                    "confidence": 0.9,
                    "reason": "Status checked",
                    "goal_completed": False,
                }

                await loop._decide_and_act()
                await loop._decide_and_act()
                await loop._decide_and_act()

        status = loop.status()
        assert status["last_loop_detection"]["level"] in {"warning", "force_change"}
        assert status["last_loop_detection"]["repetition_count"] >= 2
        assert "LOOP DETECTION" in prompts[2]

    @pytest.mark.asyncio
    async def test_decide_and_act_records_plan_health(self, eval_env):
        """Plan invalidation is wired into the runtime cycle."""
        from remy.core.autonomy import (
            ActionPlan,
            AutonomousLoop,
            _save_plan,
            create_goal,
            get_active_goals,
            load_plan_for_goal,
        )

        create_goal("Install missing dependency before continuing", priority="medium")
        goal_id = get_active_goals()[0]["goal_id"]
        plan = ActionPlan(
            plan_id="plan-health-test",
            goal_id=goal_id,
            goal_description="Install missing dependency before continuing",
            steps=["run dependency-based task", "summarize result"],
        )
        _save_plan(plan)

        loop = AutonomousLoop()

        async def fake_dispatch_worker(**kwargs):
            return (
                "Permission denied: configuration required before running this task.",
                [],
                [],
                None,
            )

        with patch("remy.core.orchestrator.dispatch_worker", new_callable=AsyncMock) as mock_dispatch:
            mock_dispatch.side_effect = fake_dispatch_worker
            with patch.object(loop, "_evaluate_outcome", new_callable=AsyncMock) as mock_eval, \
                 patch("remy.core.orchestrator.check_obvious_failure", return_value=None), \
                 patch("remy.core.orchestrator.check_zero_tool_cycle", return_value=None), \
                 patch("remy.core.execution_log.record_cycle_execution"):
                mock_eval.return_value = {
                    "success": True,
                    "confidence": 0.8,
                    "reason": "Prerequisite discovered",
                    "goal_completed": False,
                }

                await loop._decide_and_act()

        status = loop.status()
        loaded = load_plan_for_goal(goal_id)
        assert status["last_plan_health"]["suggested_action"] == "add_prerequisite"
        assert loaded.steps[0].startswith("Resolve prerequisite before retrying")
        assert loaded.current_step == 0

    @pytest.mark.asyncio
    async def test_decide_and_act_records_confidence_policy_without_enforcing(self, eval_env):
        """Confidence autonomy is observability/policy only for the desktop app."""
        from remy.core.autonomy import AutonomousLoop, create_goal
        from remy.core.confidence_autonomy import reset_domain_stats, reset_user_trust

        reset_domain_stats()
        reset_user_trust()
        create_goal("browse web page", priority="medium")
        loop = AutonomousLoop()

        async def fake_dispatch_worker(**kwargs):
            return ("Could not browse the page.", [], [], None)

        with patch("remy.core.orchestrator.dispatch_worker", new_callable=AsyncMock) as mock_dispatch:
            mock_dispatch.side_effect = fake_dispatch_worker
            with patch.object(loop, "_evaluate_outcome", new_callable=AsyncMock) as mock_eval, \
                 patch("remy.core.orchestrator.check_obvious_failure", return_value=None), \
                 patch("remy.core.orchestrator.check_zero_tool_cycle", return_value=None), \
                 patch("remy.core.execution_log.record_cycle_execution"):
                mock_eval.return_value = {
                    "success": False,
                    "confidence": 0.4,
                    "reason": "Browser action failed",
                    "goal_completed": False,
                }

                action = await loop._decide_and_act()

        policy = loop.status()["last_confidence_policy"]
        assert action is not None
        assert action.success is False
        assert policy["domain"] == "web"
        assert policy["recommended_action"] in {"request_guidance", "skip"}
        assert policy["enforced"] is False

    @pytest.mark.asyncio
    async def test_goal_auto_completed(self, eval_env):
        """Verify goals are auto-completed when evaluation says so."""
        from remy.core.autonomy import (
            AutonomousLoop, create_goal, get_active_goals,
        )

        goal_id = create_goal("Simple test goal", priority="medium")
        loop = AutonomousLoop()

        with patch("remy.core.agent.invoke_agent", new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = ("Goal completed successfully!", [], [])

            eval_result = {
                "success": True, "confidence": 0.95,
                "reason": "Goal fully achieved", "goal_completed": True,
            }
            with patch.object(loop, "_evaluate_outcome", new_callable=AsyncMock) as mock_eval, \
                 patch("remy.core.orchestrator.check_obvious_failure", return_value=None), \
                 patch("remy.core.orchestrator.check_zero_tool_cycle", return_value=None):
                mock_eval.return_value = eval_result

                action = await loop._decide_and_act()

        assert action.success is True

        # Goal should be marked as completed
        active = get_active_goals()
        active_ids = [g["goal_id"] for g in active]
        assert goal_id not in active_ids, "Completed goal should no longer be active"
