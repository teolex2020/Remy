"""
Survival Simulation Test — verifies the agent learns from failures.

Scenario:
1. Create an impossible goal ("find Atlantis in the kitchen").
2. Cycle 1: agent tries and fails (self-evaluation detects failure).
3. Cycle 2: verify the prompt sent to the agent CONTAINS the failure record.
"""

import pytest
from unittest.mock import patch, AsyncMock
from aura import Aura as CognitiveMemory


@pytest.fixture
def survival_env(tmp_path):
    """Isolated brain + mocked settings for autonomous loop testing."""
    brain = CognitiveMemory(str(tmp_path / "survival_brain"))

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


@pytest.mark.asyncio
async def test_learning_from_failure(survival_env):
    from remy.core.autonomy import AutonomousLoop, create_goal

    # ── 1. Create an impossible goal ──────────────────────────────────────
    goal_id = create_goal(
        "Find the lost city of Atlantis in the kitchen",
        priority="critical",
    )
    print(f"\n[1] Created impossible goal: {goal_id}")

    loop = AutonomousLoop()

    # ── 2. CYCLE 1 — The Attempt & Failure ────────────────────────────────
    print("\n[2] Starting Cycle 1 (The Failure)...")

    mock_response_fail = (
        "I looked in the kitchen cupboards but could not find Atlantis. "
        "I only found pasta.",
        [],  # history
        [],  # log
    )

    # Mock both invoke_agent AND _evaluate_outcome (to simulate detected failure)
    # Also mock create_plan_for_goal to avoid LLM calls
    with patch("remy.core.agent.invoke_agent", new_callable=AsyncMock) as mock_agent, \
         patch("remy.core.autonomy.create_plan_for_goal", return_value=None):
        mock_agent.return_value = mock_response_fail

        eval_fail = {
            "success": False, "confidence": 0.9,
            "reason": "Failed to find Atlantis", "goal_completed": False,
        }
        with patch.object(loop, "_evaluate_outcome", new_callable=AsyncMock) as mock_eval:
            mock_eval.return_value = eval_fail

            action = await loop._decide_and_act()
            assert action is not None
            loop.action_log.append(action)

    assert action.success is False, "Self-evaluation should detect failure"
    print("    -> Cycle 1 complete. Failure detected by self-evaluation.")

    # ── 3. CYCLE 2 — The Memory Check ─────────────────────────────────────
    print("\n[3] Starting Cycle 2 (The Memory Check)...")

    with patch("remy.core.agent.invoke_agent", new_callable=AsyncMock) as mock_agent_2, \
         patch("remy.core.autonomy.create_plan_for_goal", return_value=None):
        mock_agent_2.return_value = ("I will give up on Atlantis.", [], [])

        eval_success = {
            "success": True, "confidence": 0.8,
            "reason": "Agent made a reasonable decision", "goal_completed": False,
        }
        with patch.object(loop, "_evaluate_outcome", new_callable=AsyncMock) as mock_eval_2:
            mock_eval_2.return_value = eval_success

            await loop._decide_and_act()

        # Capture the prompt sent to the agent
        _, kwargs = mock_agent_2.call_args
        prompt_sent = kwargs["user_message"]

        print("\n=== PROMPT SEEN BY AGENT IN CYCLE 2 ===")
        print(prompt_sent)
        print("=======================================")

        # ── VERIFICATION ──────────────────────────────────────────────────
        assert "Atlantis" in prompt_sent, "Prompt should mention the goal"
        assert "FAIL" in prompt_sent.upper(), "Prompt should mention the FAILURE from Cycle 1"

    print("\n[SUCCESS] The agent was fed its own past failure in the prompt!")
