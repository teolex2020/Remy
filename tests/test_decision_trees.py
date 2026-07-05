"""Tests for F2: Decision Trees — branching plans for autonomous mode."""

from unittest.mock import MagicMock, patch

from remy.core.autonomy import (
    ActionPlan,
    DecisionTreePlan,
    PlanNode,
    _advance_decision_tree,
    _create_decision_tree_plan,
    _create_linear_plan,
    _find_node_index,
    _format_plan_text,
    _get_node,
    advance_plan,
    load_plan_for_goal,
)


# ============== PlanNode / DecisionTreePlan basics ==============


class TestPlanNodeDataclass:
    def test_plan_node_defaults(self):
        node = PlanNode(step_id=0, description="Do X")
        assert node.success_next is None
        assert node.failure_next is None
        assert node.max_retries == 2
        assert node.retry_count == 0
        assert node.condition == ""

    def test_plan_node_with_branches(self):
        node = PlanNode(step_id=1, description="Check", success_next=2, failure_next=3)
        assert node.success_next == 2
        assert node.failure_next == 3


class TestDecisionTreePlanDataclass:
    def test_tree_plan_defaults(self):
        nodes = [PlanNode(step_id=0, description="Start")]
        tree = DecisionTreePlan(
            plan_id="tree-abc",
            goal_id="g1",
            goal_description="Test goal",
            nodes=nodes,
        )
        assert tree.current_node == 0
        assert tree.status == "active"
        assert tree.history == []

    def test_tree_plan_with_history(self):
        nodes = [PlanNode(step_id=0, description="Start")]
        tree = DecisionTreePlan(
            plan_id="tree-abc",
            goal_id="g1",
            goal_description="Test",
            nodes=nodes,
            history=[{"step_id": 0, "success": True}],
        )
        assert len(tree.history) == 1


# ============== Helper functions ==============


class TestHelpers:
    def _make_tree(self):
        nodes = [
            PlanNode(step_id=0, description="Step A", success_next=1, failure_next=2),
            PlanNode(step_id=1, description="Step B", success_next=None, failure_next=None),
            PlanNode(step_id=2, description="Step C (fallback)", success_next=None, failure_next=None),
        ]
        return DecisionTreePlan(
            plan_id="tree-test",
            goal_id="g1",
            goal_description="Test",
            nodes=nodes,
        )

    def test_find_node_index(self):
        tree = self._make_tree()
        assert _find_node_index(tree, 0) == 0
        assert _find_node_index(tree, 1) == 1
        assert _find_node_index(tree, 2) == 2
        assert _find_node_index(tree, 99) is None

    def test_get_node(self):
        tree = self._make_tree()
        assert _get_node(tree, 0).description == "Step A"
        assert _get_node(tree, 1).description == "Step B"
        assert _get_node(tree, 99) is None


# ============== Advance Decision Tree ==============


class TestAdvanceDecisionTree:
    def _make_tree(self):
        nodes = [
            PlanNode(step_id=0, description="Search data", success_next=1, failure_next=2, max_retries=1),
            PlanNode(step_id=1, description="Analyze results", success_next=None, failure_next=None),
            PlanNode(step_id=2, description="Try alternative source", success_next=1, failure_next=None, max_retries=1),
        ]
        return DecisionTreePlan(
            plan_id="tree-adv",
            goal_id="g1",
            goal_description="Research topic",
            nodes=nodes,
        )

    @patch("remy.core.autonomy._save_plan")
    def test_success_follows_success_next(self, mock_save):
        tree = self._make_tree()
        result = _advance_decision_tree(tree, success=True)
        assert result == "Analyze results"
        assert tree.current_node == 1
        assert len(tree.history) == 1
        assert tree.history[0]["success"] is True

    @patch("remy.core.autonomy._save_plan")
    def test_success_completes_on_null_next(self, mock_save):
        tree = self._make_tree()
        tree.current_node = 1  # "Analyze results" — success_next=None
        result = _advance_decision_tree(tree, success=True)
        assert result is None
        assert tree.status == "completed"

    @patch("remy.core.autonomy._save_plan")
    def test_failure_retries_within_limit(self, mock_save):
        tree = self._make_tree()
        # Node 0 has max_retries=1, retry_count starts at 0
        result = _advance_decision_tree(tree, success=False)
        assert result == "Search data"  # Same step (retry)
        assert tree.current_node == 0
        assert tree.nodes[0].retry_count == 1

    @patch("remy.core.autonomy._save_plan")
    def test_failure_takes_alternative_after_max_retries(self, mock_save):
        tree = self._make_tree()
        tree.nodes[0].retry_count = 1  # Already at max_retries=1
        result = _advance_decision_tree(tree, success=False)
        assert result == "Try alternative source"
        assert tree.current_node == 2

    @patch("remy.core.autonomy._save_plan")
    def test_failure_abandons_when_no_alternative(self, mock_save):
        tree = self._make_tree()
        tree.current_node = 2  # failure_next=None, max_retries=1
        tree.nodes[2].retry_count = 1  # Exhausted retries
        result = _advance_decision_tree(tree, success=False)
        assert result is None
        assert tree.status == "abandoned"

    @patch("remy.core.autonomy._save_plan")
    def test_advance_records_history(self, mock_save):
        tree = self._make_tree()
        _advance_decision_tree(tree, success=True)
        _advance_decision_tree(tree, success=False)
        assert len(tree.history) == 2
        assert tree.history[0]["step_id"] == 0
        assert tree.history[1]["step_id"] == 1

    @patch("remy.core.autonomy._save_plan")
    def test_advance_invalid_current_node_abandons(self, mock_save):
        tree = self._make_tree()
        tree.current_node = 99  # Doesn't exist
        result = _advance_decision_tree(tree, success=True)
        assert result is None
        assert tree.status == "abandoned"


# ============== advance_plan dispatches correctly ==============


class TestAdvancePlanDispatch:
    @patch("remy.core.autonomy._save_plan")
    def test_advance_plan_dispatches_to_tree(self, mock_save):
        nodes = [
            PlanNode(step_id=0, description="Start", success_next=1),
            PlanNode(step_id=1, description="End", success_next=None),
        ]
        tree = DecisionTreePlan(
            plan_id="tree-d", goal_id="g1",
            goal_description="Test", nodes=nodes,
        )
        result = advance_plan(tree, success=True)
        assert result == "End"
        assert tree.current_node == 1

    @patch("remy.core.autonomy._save_plan")
    def test_advance_plan_dispatches_to_linear(self, mock_save):
        linear = ActionPlan(
            plan_id="plan-l", goal_id="g1",
            goal_description="Test", steps=["A", "B"],
        )
        result = advance_plan(linear, success=True)
        assert result == "B"
        assert linear.current_step == 1


# ============== _create_decision_tree_plan ==============


class TestCreateDecisionTreePlan:
    @patch("remy.core.autonomy._save_plan")
    @patch("remy.core.llm.call_llm")
    def test_create_tree_from_llm(self, mock_llm, mock_save):
        mock_llm.return_value = MagicMock(content='['
            '{"step_id": 0, "description": "Research topic", "success_next": 1, "failure_next": 2, "max_retries": 2},'
            '{"step_id": 1, "description": "Write summary", "success_next": null, "failure_next": null, "max_retries": 1},'
            '{"step_id": 2, "description": "Try alternative query", "success_next": 1, "failure_next": null, "max_retries": 1}'
            ']')

        result = _create_decision_tree_plan("g1", "Learn about AI")

        assert result is not None
        assert isinstance(result, DecisionTreePlan)
        assert len(result.nodes) == 3
        assert result.nodes[0].success_next == 1
        assert result.nodes[0].failure_next == 2
        assert result.plan_id.startswith("tree-")
        mock_save.assert_called_once()

    @patch("remy.core.llm.call_llm")
    def test_create_tree_returns_none_on_bad_json(self, mock_llm):
        mock_llm.return_value = MagicMock(content="not valid json")
        result = _create_decision_tree_plan("g1", "Test")
        assert result is None

    @patch("remy.core.llm.call_llm")
    def test_create_tree_returns_none_on_single_node(self, mock_llm):
        mock_llm.return_value = MagicMock(
            content='[{"step_id": 0, "description": "Only one"}]'
        )
        result = _create_decision_tree_plan("g1", "Test")
        assert result is None

    @patch("remy.core.llm.call_llm")
    def test_create_tree_returns_none_on_invalid_references(self, mock_llm):
        # success_next=99 doesn't exist
        mock_llm.return_value = MagicMock(content='['
            '{"step_id": 0, "description": "A", "success_next": 99},'
            '{"step_id": 1, "description": "B"}'
            ']')
        result = _create_decision_tree_plan("g1", "Test")
        assert result is None

    @patch("remy.core.llm.call_llm")
    def test_create_tree_returns_none_on_duplicate_ids(self, mock_llm):
        mock_llm.return_value = MagicMock(content='['
            '{"step_id": 0, "description": "A"},'
            '{"step_id": 0, "description": "B"}'
            ']')
        result = _create_decision_tree_plan("g1", "Test")
        assert result is None

    @patch("remy.core.llm.call_llm")
    def test_create_tree_handles_llm_exception(self, mock_llm):
        mock_llm.side_effect = RuntimeError("LLM down")
        result = _create_decision_tree_plan("g1", "Test")
        assert result is None

    @patch("remy.core.autonomy._save_plan")
    @patch("remy.core.llm.call_llm")
    def test_create_tree_strips_code_fences(self, mock_llm, mock_save):
        mock_llm.return_value = MagicMock(content='```json\n['
            '{"step_id": 0, "description": "A", "success_next": 1},'
            '{"step_id": 1, "description": "B"}'
            ']\n```')
        result = _create_decision_tree_plan("g1", "Test")
        assert result is not None
        assert len(result.nodes) == 2


# ============== load_plan_for_goal ==============


class TestLoadPlanForGoal:
    @patch("remy.core.autonomy.brain")
    def test_load_decision_tree(self, mock_brain):
        rec = MagicMock()
        rec.content = "Plan: Do research"
        rec.metadata = {
            "goal_id": "g1",
            "status": "active",
            "plan_type": "decision_tree",
            "plan_id": "tree-123",
            "current_node": 0,
            "nodes": [
                {"step_id": 0, "description": "Search", "success_next": 1, "failure_next": None, "max_retries": 2, "retry_count": 0},
                {"step_id": 1, "description": "Summarize", "success_next": None, "failure_next": None, "max_retries": 1, "retry_count": 0},
            ],
            "history": [],
        }
        mock_brain.search.return_value = [rec]

        result = load_plan_for_goal("g1")

        assert isinstance(result, DecisionTreePlan)
        assert result.plan_id == "tree-123"
        assert len(result.nodes) == 2
        assert result.nodes[0].success_next == 1

    @patch("remy.core.autonomy.brain")
    def test_load_linear_plan_backward_compat(self, mock_brain):
        rec = MagicMock()
        rec.content = "Plan: Old goal"
        rec.metadata = {
            "goal_id": "g2",
            "status": "active",
            "plan_id": "plan-456",
            "steps": ["A", "B", "C"],
            "current_step": 1,
        }
        mock_brain.search.return_value = [rec]

        result = load_plan_for_goal("g2")

        assert isinstance(result, ActionPlan)
        assert result.plan_id == "plan-456"
        assert result.steps == ["A", "B", "C"]
        assert result.current_step == 1

    @patch("remy.core.autonomy.brain")
    def test_load_returns_none_when_no_match(self, mock_brain):
        mock_brain.search.return_value = []
        assert load_plan_for_goal("g_missing") is None


# ============== _format_plan_text ==============


class TestFormatPlanText:
    def test_format_tree_plan(self):
        nodes = [
            PlanNode(step_id=0, description="Search data", success_next=1, failure_next=2, max_retries=2),
            PlanNode(step_id=1, description="Analyze results", success_next=None),
            PlanNode(step_id=2, description="Try alternative", success_next=1),
        ]
        tree = DecisionTreePlan(
            plan_id="t1", goal_id="g1",
            goal_description="Test", nodes=nodes,
        )

        text = _format_plan_text(tree)

        assert "decision tree" in text
        assert "Search data" in text
        assert "On success" in text
        assert "Analyze results" in text
        assert "On failure" in text
        assert "Try alternative" in text

    def test_format_tree_plan_complete_on_success(self):
        nodes = [
            PlanNode(step_id=0, description="Final step", success_next=None, failure_next=None, max_retries=1),
        ]
        tree = DecisionTreePlan(
            plan_id="t2", goal_id="g1",
            goal_description="Test", nodes=nodes,
        )

        text = _format_plan_text(tree)
        assert "PLAN COMPLETE" in text
        assert "retry" in text.lower()

    def test_format_linear_plan(self):
        plan = ActionPlan(
            plan_id="p1", goal_id="g1",
            goal_description="Test", steps=["A", "B", "C"],
        )

        text = _format_plan_text(plan)
        assert "step 1/3" in text
        assert "A" in text
        assert "A → B → C" in text

    def test_format_linear_plan_empty_steps(self):
        plan = ActionPlan(
            plan_id="p2", goal_id="g1",
            goal_description="Test", steps=[],
        )
        assert _format_plan_text(plan) == ""


# ============== _create_linear_plan (backward compat) ==============


class TestCreateLinearPlan:
    @patch("remy.core.autonomy._save_plan")
    @patch("remy.core.llm.call_llm")
    def test_create_linear_plan(self, mock_llm, mock_save):
        mock_llm.return_value = MagicMock(content='["Step 1", "Step 2", "Step 3"]')

        result = _create_linear_plan("g1", "Simple goal")

        assert isinstance(result, ActionPlan)
        assert len(result.steps) == 3
        assert result.plan_id.startswith("plan-")

    @patch("remy.core.llm.call_llm")
    def test_create_linear_plan_returns_none_on_failure(self, mock_llm):
        mock_llm.side_effect = RuntimeError("down")
        assert _create_linear_plan("g1", "Test") is None
