"""Tests for F1: Multi-Agent Delegation — role-based system prompts."""

from unittest.mock import MagicMock, patch

from remy.core.autonomy import (
    AGENT_ROLES,
    ActionPlan,
    AgentRole,
    AutonomousLoop,
)


def _make_loop():
    """Create an AutonomousLoop with mocked brain."""
    with patch("remy.core.autonomy.brain"):
        loop = AutonomousLoop.__new__(AutonomousLoop)
        # Minimal init for _select_role (doesn't need full __init__)
        loop._is_research_goal = AutonomousLoop._is_research_goal.__get__(loop)
        loop._select_role = AutonomousLoop._select_role.__get__(loop)
        return loop


class TestAgentRoles:
    """Tests for AGENT_ROLES configuration."""

    def test_all_roles_defined(self):
        assert "researcher" in AGENT_ROLES
        assert "planner" in AGENT_ROLES
        assert "executor" in AGENT_ROLES
        assert "analyst" in AGENT_ROLES

    def test_role_has_required_fields(self):
        for name, role in AGENT_ROLES.items():
            assert role.name == name
            assert len(role.description) > 0
            assert len(role.priority_tools) > 0
            assert len(role.instruction_suffix) > 0
            assert role.max_tool_iterations > 0


class TestSelectRole:
    """Tests for AutonomousLoop._select_role()."""

    def test_select_role_no_goal_returns_planner(self):
        loop = _make_loop()
        role = loop._select_role(None, None)
        assert role.name == "planner"

    def test_select_role_research_goal(self):
        loop = _make_loop()
        goal = {"description": "Research the history of Ukrainian cuisine", "attempts": 0}
        role = loop._select_role(goal, None)
        assert role.name == "researcher"

    def test_select_role_analysis_goal(self):
        loop = _make_loop()
        goal = {"description": "Analyze health data patterns for the user", "attempts": 0}
        role = loop._select_role(goal, None)
        assert role.name == "analyst"

    def test_select_role_planning_goal(self):
        loop = _make_loop()
        goal = {"description": "Organize and prioritize knowledge base entries", "attempts": 0}
        role = loop._select_role(goal, None)
        assert role.name == "planner"

    def test_select_role_default_executor(self):
        loop = _make_loop()
        goal = {"description": "Update the user profile with new data", "attempts": 0}
        role = loop._select_role(goal, None)
        assert role.name == "executor"

    def test_select_role_from_plan_step_research(self):
        loop = _make_loop()
        goal = {"description": "Complete project alpha", "attempts": 1}
        plan = ActionPlan(
            plan_id="p1", goal_id="g1", goal_description="Complete project",
            steps=["Search for relevant background information", "Write summary"],
            current_step=0, status="active",
        )
        role = loop._select_role(goal, plan)
        assert role.name == "researcher"

    def test_select_role_from_plan_step_analyze(self):
        loop = _make_loop()
        goal = {"description": "Complete project alpha", "attempts": 1}
        plan = ActionPlan(
            plan_id="p1", goal_id="g1", goal_description="Complete project",
            steps=["Analyze existing data and correlate findings"],
            current_step=0, status="active",
        )
        role = loop._select_role(goal, plan)
        assert role.name == "analyst"

    def test_select_role_from_plan_step_execute(self):
        loop = _make_loop()
        goal = {"description": "Complete project alpha", "attempts": 1}
        plan = ActionPlan(
            plan_id="p1", goal_id="g1", goal_description="Complete project",
            steps=["Write the report to file"],
            current_step=0, status="active",
        )
        role = loop._select_role(goal, plan)
        assert role.name == "executor"

    def test_select_role_plan_not_active_uses_goal(self):
        loop = _make_loop()
        goal = {"description": "Research quantum computing", "attempts": 1}
        plan = ActionPlan(
            plan_id="p1", goal_id="g1", goal_description="Research",
            steps=["Write report"], current_step=0, status="completed",
        )
        role = loop._select_role(goal, plan)
        assert role.name == "researcher"  # Falls through to goal-based selection


class TestBuildDecisionPromptWithRole:
    """Tests for role injection in _build_decision_prompt()."""

    @patch("remy.core.autonomy.brain")
    def test_decision_prompt_includes_role(self, mock_brain):
        mock_brain.search.return_value = []
        loop = _make_loop()
        loop.budget = MagicMock()
        loop.budget.to_dict.return_value = {
            "tokens_today": 100, "daily_limit": 10000,
            "tokens_this_hour": 50, "hourly_limit": 2000,
        }

        role = AGENT_ROLES["researcher"]
        prompt = loop._build_decision_prompt(
            goals=[],
            past_outcomes="",
            budget=loop.budget.to_dict(),
            role=role,
        )

        assert "ROLE: RESEARCHER" in prompt
        assert "PRIORITY TOOLS:" in prompt
        assert "recall" in prompt

    @patch("remy.core.autonomy.brain")
    def test_decision_prompt_works_without_role(self, mock_brain):
        mock_brain.search.return_value = []
        loop = _make_loop()

        prompt = loop._build_decision_prompt(
            goals=[],
            past_outcomes="",
            budget={"tokens_today": 0, "daily_limit": 10000,
                    "tokens_this_hour": 0, "hourly_limit": 2000},
        )

        assert "AUTONOMOUS MODE" in prompt
        assert "ROLE:" not in prompt
