"""Tests for Sub-Goal Decomposition — decompose_goal(), create_subgoal/complete_goal tools."""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from aura import Aura as CognitiveMemory


@pytest.fixture
def subgoal_env(tmp_path):
    """Isolated brain + mocked settings for sub-goal tests."""
    brain = CognitiveMemory(str(tmp_path / "subgoal_brain"))

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


class TestDecomposeGoal:
    """Tests for decompose_goal() function."""

    def test_decompose_creates_subgoals(self, subgoal_env):
        from remy.core.autonomy import (
            create_goal, decompose_goal, get_active_goals,
        )

        goal_id = create_goal("Build a house from scratch", priority="high")

        # Mock LLM to return sub-goals
        mock_response = MagicMock()
        mock_response.content = json.dumps([
            "Find a suitable plot of land",
            "Create architectural blueprints",
            "Pour the foundation",
        ])

        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value.invoke.return_value = mock_response
            sub_ids = decompose_goal(goal_id)

        assert len(sub_ids) == 3
        active = get_active_goals()
        descriptions = [g["description"] for g in active]
        assert any("blueprints" in d.lower() for d in descriptions)

    def test_double_decompose_guard(self, subgoal_env):
        from remy.core.autonomy import create_goal, decompose_goal

        goal_id = create_goal("Research family history", priority="medium")

        mock_response = MagicMock()
        mock_response.content = json.dumps(["Step 1", "Step 2"])

        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value.invoke.return_value = mock_response

            # First decompose
            sub_ids_1 = decompose_goal(goal_id)
            assert len(sub_ids_1) == 2

            # Second decompose — should be blocked
            sub_ids_2 = decompose_goal(goal_id)
            assert len(sub_ids_2) == 0, "Should not decompose the same goal twice"

    def test_decompose_marks_parent_as_decomposed(self, subgoal_env):
        from remy.core.autonomy import create_goal, decompose_goal

        brain = subgoal_env["brain"]
        goal_id = create_goal("Complex task", priority="high")

        mock_response = MagicMock()
        mock_response.content = json.dumps(["Sub-task A", "Sub-task B"])

        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value.invoke.return_value = mock_response
            decompose_goal(goal_id)

        rec = brain.get(goal_id)
        assert rec.metadata["status"] == "decomposed"

    def test_decompose_fallback_on_llm_failure(self, subgoal_env):
        from remy.core.autonomy import create_goal, decompose_goal

        goal_id = create_goal("Test goal", priority="low")

        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value.invoke.side_effect = Exception("API error")
            sub_ids = decompose_goal(goal_id)

        assert sub_ids == [], "Should return empty list on LLM failure"


class TestSubGoalPreference:
    """Tests for get_active_goals() sub-goal ordering."""

    def test_subgoals_appear_before_parents(self, subgoal_env):
        from remy.core.autonomy import create_goal, get_active_goals

        # Create parent goal
        parent_id = create_goal("Parent goal", priority="high")
        parent_rec = subgoal_env["brain"].get(parent_id)
        parent_goal_id = parent_rec.metadata["goal_id"]

        # Create sub-goal
        sub_id = create_goal(
            "Sub-goal", priority="high",
            parent_goal_id=parent_goal_id,
        )

        active = get_active_goals()
        assert len(active) >= 2

        # Find positions
        sub_idx = next(i for i, g in enumerate(active) if g["record_id"] == sub_id)
        parent_idx = next(i for i, g in enumerate(active) if g["record_id"] == parent_id)
        assert sub_idx < parent_idx, "Sub-goals should appear before parent goals"


class TestAutoDecompose:
    """Tests for auto-decomposition trigger in _decide_and_act."""

    @pytest.mark.asyncio
    async def test_auto_decompose_after_failures(self, subgoal_env):
        from remy.core.autonomy import (
            AutonomousLoop, create_goal, record_goal_attempt,
        )

        goal_id = create_goal("Very complex goal", priority="critical")

        # Simulate 3 failed attempts
        for _ in range(3):
            record_goal_attempt(goal_id)

        loop = AutonomousLoop()

        mock_response = MagicMock()
        mock_response.content = json.dumps(["Small step 1", "Small step 2"])

        with patch("langchain_google_genai.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.return_value.invoke.return_value = mock_response

            with patch(
                "remy.core.agent.invoke_agent",
                new_callable=AsyncMock,
            ) as mock_agent:
                mock_agent.return_value = ("I did small step 1", [], [])

                with patch.object(
                    loop, "_evaluate_outcome", new_callable=AsyncMock,
                ) as mock_eval:
                    mock_eval.return_value = {
                        "success": True, "confidence": 0.8,
                        "reason": "ok", "goal_completed": False,
                    }
                    action = await loop._decide_and_act()

        assert action is not None
        # The decompose should have been called (it mockes LLM for decompose too)


class TestGoalTools:
    """Tests for create_subgoal and complete_goal tool handlers."""

    def test_create_subgoal_tool(self, subgoal_env):
        from remy.core.autonomy import create_goal, get_active_goals

        # Create parent goal first
        parent_id = create_goal("Parent task", priority="high")
        parent_rec = subgoal_env["brain"].get(parent_id)
        parent_goal_id = parent_rec.metadata["goal_id"]

        # Patch brain in brain_tools module too
        with patch("remy.core.brain_tools.brain", subgoal_env["brain"]):
            from remy.core.brain_tools import execute_tool
            result = execute_tool("create_subgoal", {
                "parent_goal_id": parent_goal_id,
                "description": "First sub-step",
                "priority": "medium",
            })

        data = json.loads(result)
        assert data["created"] is True
        assert data["parent_goal_id"] == parent_goal_id

    def test_complete_goal_tool(self, subgoal_env):
        from remy.core.autonomy import create_goal, get_active_goals

        goal_id = create_goal("Completable task", priority="medium")
        rec = subgoal_env["brain"].get(goal_id)
        gid = rec.metadata["goal_id"]

        with patch("remy.core.brain_tools.brain", subgoal_env["brain"]):
            from remy.core.brain_tools import execute_tool
            result = execute_tool("complete_goal", {
                "goal_id": gid,
                "notes": "All done!",
            })

        data = json.loads(result)
        assert data["completed"] is True

        # Verify goal is no longer active
        active = get_active_goals()
        active_ids = [g["goal_id"] for g in active]
        assert gid not in active_ids
