"""Tests for Dynamic Plan Invalidation & Re-Planning (AUTON-14) — plan_invalidation.py."""


# ============== Unit Tests: check_plan_validity ==============


class TestCheckPlanValidity:
    def _make_plan(self, steps=None, current=0):
        from remy.core.autonomy_goals import ActionPlan

        return ActionPlan(
            plan_id="test-plan",
            goal_id="test-goal",
            goal_description="test goal",
            steps=steps or ["step 1", "step 2", "step 3"],
            current_step=current,
        )

    def test_success_stays_valid(self):
        from remy.core.plan_invalidation import check_plan_validity

        plan = self._make_plan()
        result = check_plan_validity(plan, "step done", step_success=True)
        assert result.valid is True
        assert result.suggested_action == "continue"

    def test_single_failure_retries(self):
        from remy.core.plan_invalidation import check_plan_validity

        plan = self._make_plan()
        result = check_plan_validity(plan, "failed", step_success=False, consecutive_failures=1)
        assert result.valid is True
        assert result.suggested_action == "continue"

    def test_two_failures_suggests_replan(self):
        from remy.core.plan_invalidation import check_plan_validity

        plan = self._make_plan()
        result = check_plan_validity(
            plan, "failed again", step_success=False, consecutive_failures=2
        )
        assert result.needs_update is True
        assert result.suggested_action == "replan"

    def test_three_failures_abandons(self):
        from remy.core.plan_invalidation import check_plan_validity

        plan = self._make_plan()
        result = check_plan_validity(
            plan, "failed thrice", step_success=False, consecutive_failures=3
        )
        assert result.valid is False
        assert result.abandon is True
        assert result.suggested_action == "abandon"

    def test_prerequisite_detection(self):
        from remy.core.plan_invalidation import check_plan_validity

        plan = self._make_plan()
        result = check_plan_validity(
            plan,
            "Error: need to install library X first",
            step_success=True,
        )
        assert result.needs_update is True
        assert result.suggested_action == "add_prerequisite"

    def test_confidence_decays_with_failures(self):
        from remy.core.plan_invalidation import check_plan_validity

        plan = self._make_plan()
        result = check_plan_validity(plan, "fail", step_success=False, consecutive_failures=2)
        assert result.confidence < 1.0

    def test_progress_increases_confidence(self):
        from remy.core.plan_invalidation import check_plan_validity

        plan = self._make_plan(current=2)  # 2/3 steps done
        result = check_plan_validity(plan, "ok", step_success=True)
        assert result.confidence > 0.8

    def test_runtime_autonomy_plan_is_supported(self):
        from remy.core.autonomy import ActionPlan
        from remy.core.plan_invalidation import check_plan_validity

        plan = ActionPlan(
            plan_id="runtime-plan",
            goal_id="runtime-goal",
            goal_description="runtime goal",
            steps=["step 1", "step 2", "step 3"],
            current_step=1,
        )

        result = check_plan_validity(plan, "step done", step_success=True)

        assert result.valid is True
        assert result.suggested_action == "continue"
        assert result.confidence > 0.6


# ============== Unit Tests: _detect_prerequisite_needed ==============


class TestDetectPrerequisite:
    def test_install_needed(self):
        from remy.core.plan_invalidation import _detect_prerequisite_needed

        assert _detect_prerequisite_needed("Error: need to install pandas") is True

    def test_permission_denied(self):
        from remy.core.plan_invalidation import _detect_prerequisite_needed

        assert _detect_prerequisite_needed("Permission denied accessing /root") is True

    def test_normal_result(self):
        from remy.core.plan_invalidation import _detect_prerequisite_needed

        assert _detect_prerequisite_needed("Task completed successfully") is False

    def test_not_found(self):
        from remy.core.plan_invalidation import _detect_prerequisite_needed

        assert _detect_prerequisite_needed("File not found: config.yaml") is True


# ============== Unit Tests: plan confidence ==============


class TestPlanConfidence:
    def setup_method(self):
        from remy.core.plan_invalidation import clear_plan_confidence

        clear_plan_confidence("test-plan")

    def test_starts_at_1(self):
        from remy.core.plan_invalidation import get_plan_confidence

        assert get_plan_confidence("new-plan") == 1.0

    def test_success_increases(self):
        from remy.core.plan_invalidation import update_plan_confidence

        new = update_plan_confidence("test-plan", step_success=True)
        assert new >= 1.0  # Already at max

    def test_failure_decreases(self):
        from remy.core.plan_invalidation import update_plan_confidence

        new = update_plan_confidence("test-plan", step_success=False)
        assert new < 1.0
        assert new == 0.8  # 1.0 - 0.2

    def test_multiple_failures_decay(self):
        from remy.core.plan_invalidation import update_plan_confidence

        for _ in range(4):
            confidence = update_plan_confidence("test-plan", step_success=False)
        assert confidence < 0.3

    def test_should_replan_at_low_confidence(self):
        from remy.core.plan_invalidation import should_replan, update_plan_confidence

        for _ in range(4):
            update_plan_confidence("test-plan", step_success=False)
        assert should_replan("test-plan") is True

    def test_should_not_replan_at_high_confidence(self):
        from remy.core.plan_invalidation import should_replan

        assert should_replan("test-plan") is False


# ============== Unit Tests: insert_prerequisite ==============


class TestInsertPrerequisite:
    def _make_plan(self):
        from remy.core.autonomy_goals import ActionPlan

        return ActionPlan(
            plan_id="test-plan",
            goal_id="test-goal",
            goal_description="test goal",
            steps=["step 1", "step 2", "step 3"],
            current_step=1,
        )

    def test_inserts_at_current(self):
        from remy.core.plan_invalidation import insert_prerequisite

        plan = self._make_plan()
        ok = insert_prerequisite(plan, "install library X")
        assert ok is True
        assert plan.steps[1] == "install library X"
        assert len(plan.steps) == 4

    def test_preserves_order(self):
        from remy.core.plan_invalidation import insert_prerequisite

        plan = self._make_plan()
        insert_prerequisite(plan, "setup config")
        assert plan.steps == ["step 1", "setup config", "step 2", "step 3"]

    def test_rejects_decision_tree(self):
        from remy.core.autonomy import DecisionTreePlan
        from remy.core.plan_invalidation import insert_prerequisite

        plan = DecisionTreePlan(
            plan_id="t",
            goal_id="g",
            goal_description="d",
            nodes=[],
        )
        assert insert_prerequisite(plan, "prereq") is False


# ============== Unit Tests: abandon_plan ==============


class TestAbandonPlan:
    def test_sets_abandoned(self):
        from remy.core.autonomy_goals import ActionPlan
        from remy.core.plan_invalidation import abandon_plan

        plan = ActionPlan(
            plan_id="p1",
            goal_id="g1",
            goal_description="test",
            steps=["s1", "s2"],
        )
        abandon_plan(plan)
        assert plan.status == "abandoned"


# ============== Unit Tests: build_replan_context ==============


class TestBuildReplanContext:
    def test_linear_plan_context(self):
        from remy.core.autonomy_goals import ActionPlan
        from remy.core.plan_invalidation import build_replan_context

        plan = ActionPlan(
            plan_id="p1",
            goal_id="g1",
            goal_description="Research AI safety",
            steps=["search papers", "read abstracts", "summarize"],
            current_step=1,
        )
        context = build_replan_context(plan, failures=["API timeout"])
        assert "Research AI safety" in context
        assert "COMPLETED" in context
        assert "search papers" in context
        assert "REMAINING" in context
        assert "API timeout" in context

    def test_decision_tree_context(self):
        from remy.core.autonomy_goals import DecisionTreePlan
        from remy.core.plan_invalidation import build_replan_context

        plan = DecisionTreePlan(
            plan_id="p2",
            goal_id="g2",
            goal_description="Complex task",
            nodes=[],
            history=[
                {"step_id": 0, "description": "first step", "success": True},
                {"step_id": 1, "description": "second step", "success": False},
            ],
        )
        context = build_replan_context(plan, failures=[])
        assert "Complex task" in context
        assert "HISTORY" in context

    def test_no_failures(self):
        from remy.core.autonomy_goals import ActionPlan
        from remy.core.plan_invalidation import build_replan_context

        plan = ActionPlan(
            plan_id="p3",
            goal_id="g3",
            goal_description="Simple task",
            steps=["do it"],
            current_step=0,
        )
        context = build_replan_context(plan, failures=[])
        assert "FAILURE" not in context


# ============== Unit Tests: process_step_result ==============


class TestProcessStepResult:
    def setup_method(self):
        from remy.core.plan_invalidation import clear_plan_confidence

        clear_plan_confidence("proc-plan")

    def _make_plan(self):
        from remy.core.autonomy_goals import ActionPlan

        return ActionPlan(
            plan_id="proc-plan",
            goal_id="g1",
            goal_description="test",
            steps=["s1", "s2", "s3"],
        )

    def test_success_continues(self):
        from remy.core.plan_invalidation import process_step_result

        plan = self._make_plan()
        result = process_step_result(plan, "done", step_success=True)
        assert result.suggested_action == "continue"

    def test_repeated_failures_trigger_replan(self):
        from remy.core.plan_invalidation import process_step_result

        plan = self._make_plan()
        # Reduce confidence with failures
        for i in range(4):
            result = process_step_result(
                plan,
                "failed",
                step_success=False,
                consecutive_failures=i + 1,
            )
        # After 4 failures, should abandon
        assert result.abandon or result.needs_update

    def test_confidence_updates(self):
        from remy.core.plan_invalidation import get_plan_confidence, process_step_result

        plan = self._make_plan()
        process_step_result(plan, "failed", step_success=False)
        assert get_plan_confidence("proc-plan") < 1.0
