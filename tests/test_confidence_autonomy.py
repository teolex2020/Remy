"""Tests for Confidence-Based Autonomy Levels (AUTON-15) — confidence_autonomy.py."""


# ============== Unit Tests: get_autonomy_action ==============


class TestGetAutonomyAction:
    def test_high_confidence_silent(self):
        from remy.core.confidence_autonomy import AutonomyAction, get_autonomy_action

        assert get_autonomy_action(0.9) == AutonomyAction.EXECUTE_SILENT

    def test_moderate_confidence_notify(self):
        from remy.core.confidence_autonomy import AutonomyAction, get_autonomy_action

        assert get_autonomy_action(0.6) == AutonomyAction.EXECUTE_NOTIFY

    def test_low_confidence_guidance(self):
        from remy.core.confidence_autonomy import AutonomyAction, get_autonomy_action

        assert get_autonomy_action(0.4) == AutonomyAction.REQUEST_GUIDANCE

    def test_very_low_confidence_skip(self):
        from remy.core.confidence_autonomy import AutonomyAction, get_autonomy_action

        assert get_autonomy_action(0.1) == AutonomyAction.SKIP

    def test_boundary_silent(self):
        from remy.core.confidence_autonomy import AutonomyAction, get_autonomy_action

        assert get_autonomy_action(0.8) == AutonomyAction.EXECUTE_SILENT

    def test_boundary_notify(self):
        from remy.core.confidence_autonomy import AutonomyAction, get_autonomy_action

        assert get_autonomy_action(0.5) == AutonomyAction.EXECUTE_NOTIFY


# ============== Unit Tests: compute_confidence ==============


class TestComputeConfidence:
    def test_all_high(self):
        from remy.core.confidence_autonomy import ConfidenceFactors, compute_confidence

        factors = ConfidenceFactors(
            domain_familiarity=1.0,
            tool_reliability=1.0,
            goal_clarity=1.0,
            budget_health=1.0,
            recent_success_rate=1.0,
        )
        assert compute_confidence(factors) == 1.0

    def test_all_low(self):
        from remy.core.confidence_autonomy import ConfidenceFactors, compute_confidence

        factors = ConfidenceFactors(
            domain_familiarity=0.0,
            tool_reliability=0.0,
            goal_clarity=0.0,
            budget_health=0.0,
            recent_success_rate=0.0,
        )
        assert compute_confidence(factors) == 0.0

    def test_mixed_factors(self):
        from remy.core.confidence_autonomy import ConfidenceFactors, compute_confidence

        factors = ConfidenceFactors(
            domain_familiarity=0.8,
            tool_reliability=1.0,
            goal_clarity=0.5,
            budget_health=0.9,
            recent_success_rate=0.7,
        )
        score = compute_confidence(factors)
        assert 0.5 < score < 1.0

    def test_domain_has_highest_weight(self):
        from remy.core.confidence_autonomy import ConfidenceFactors, compute_confidence

        # High domain, low everything else
        high_domain = ConfidenceFactors(
            domain_familiarity=1.0,
            tool_reliability=0.0,
            goal_clarity=0.0,
            budget_health=0.0,
            recent_success_rate=0.0,
        )
        # Low domain, high everything else
        low_domain = ConfidenceFactors(
            domain_familiarity=0.0,
            tool_reliability=1.0,
            goal_clarity=1.0,
            budget_health=1.0,
            recent_success_rate=1.0,
        )
        assert compute_confidence(high_domain) > 0.3  # Domain alone gives 0.35
        assert compute_confidence(low_domain) < 0.7  # Missing 0.35 weight


# ============== Unit Tests: infer_domain ==============


class TestInferDomain:
    def test_research(self):
        from remy.core.confidence_autonomy import infer_domain

        assert infer_domain("research AI safety trends") == "research"

    def test_web(self):
        from remy.core.confidence_autonomy import infer_domain

        assert infer_domain("browse page at http://example.com") == "web"

    def test_file_ops(self):
        from remy.core.confidence_autonomy import infer_domain

        assert infer_domain("write results to file") == "file_ops"

    def test_memory(self):
        from remy.core.confidence_autonomy import infer_domain

        assert infer_domain("recall and store related records") == "memory"

    def test_unknown_is_general(self):
        from remy.core.confidence_autonomy import infer_domain

        assert infer_domain("do something vague") == "general"


# ============== Unit Tests: domain confidence ==============


class TestDomainConfidence:
    def setup_method(self):
        from remy.core.confidence_autonomy import reset_domain_stats

        reset_domain_stats()

    def test_unknown_domain_neutral(self):
        from remy.core.confidence_autonomy import get_domain_confidence

        assert get_domain_confidence("new_domain") == 0.5

    def test_success_increases_confidence(self):
        from remy.core.confidence_autonomy import get_domain_confidence, record_domain_outcome

        for _ in range(5):
            record_domain_outcome("research", success=True)
        assert get_domain_confidence("research") > 0.8

    def test_failure_decreases_confidence(self):
        from remy.core.confidence_autonomy import get_domain_confidence, record_domain_outcome

        for _ in range(5):
            record_domain_outcome("web", success=False)
        assert get_domain_confidence("web") < 0.2

    def test_mixed_outcomes(self):
        from remy.core.confidence_autonomy import get_domain_confidence, record_domain_outcome

        for _ in range(3):
            record_domain_outcome("memory", success=True)
        record_domain_outcome("memory", success=False)
        conf = get_domain_confidence("memory")
        assert 0.5 < conf < 0.9  # 3/4 = 75%


# ============== Unit Tests: user trust calibration ==============


class TestUserTrustCalibration:
    def setup_method(self):
        from remy.core.confidence_autonomy import reset_user_trust

        reset_user_trust()

    def test_initial_thresholds(self):
        from remy.core.confidence_autonomy import get_calibrated_thresholds

        thresholds = get_calibrated_thresholds()
        assert thresholds["silent"] == 0.8
        assert thresholds["notify"] == 0.5
        assert thresholds["guidance"] == 0.3

    def test_approvals_lower_thresholds(self):
        from remy.core.confidence_autonomy import get_calibrated_thresholds, record_user_decision

        for _ in range(10):
            record_user_decision(approved=True)
        thresholds = get_calibrated_thresholds()
        # More approvals → lower thresholds (more autonomous)
        assert thresholds["silent"] < 0.8

    def test_rejections_raise_thresholds(self):
        from remy.core.confidence_autonomy import get_calibrated_thresholds, record_user_decision

        for _ in range(10):
            record_user_decision(approved=False)
        thresholds = get_calibrated_thresholds()
        # More rejections → higher thresholds (more cautious)
        assert thresholds["silent"] > 0.8

    def test_balanced_no_change(self):
        from remy.core.confidence_autonomy import get_calibrated_thresholds, record_user_decision

        for _ in range(5):
            record_user_decision(approved=True)
        for _ in range(5):
            record_user_decision(approved=False)
        thresholds = get_calibrated_thresholds()
        assert abs(thresholds["silent"] - 0.8) < 0.01


# ============== Unit Tests: assess_action_confidence ==============


class TestAssessActionConfidence:
    def setup_method(self):
        from remy.core.confidence_autonomy import reset_domain_stats, reset_user_trust

        reset_domain_stats()
        reset_user_trust()

    def test_returns_tuple(self):
        from remy.core.confidence_autonomy import assess_action_confidence

        score, action = assess_action_confidence("research AI topics")
        assert isinstance(score, float)
        assert isinstance(action, str)
        assert 0.0 <= score <= 1.0

    def test_high_confidence_with_history(self):
        from remy.core.confidence_autonomy import (
            assess_action_confidence,
            record_domain_outcome,
        )

        # Build domain confidence
        for _ in range(10):
            record_domain_outcome("research", success=True)

        score, action = assess_action_confidence(
            "research new topic",
            budget_pct=90,
            tool_health_issues=0,
            recent_successes=5,
            recent_failures=0,
        )
        assert score > 0.7

    def test_low_confidence_with_failures(self):
        from remy.core.confidence_autonomy import (
            assess_action_confidence,
            record_domain_outcome,
        )

        # Build bad history
        for _ in range(10):
            record_domain_outcome("web", success=False)

        score, action = assess_action_confidence(
            "browse web page",
            budget_pct=5,
            tool_health_issues=3,
            recent_successes=0,
            recent_failures=5,
        )
        assert score < 0.3


# ============== Unit Tests: _assess_goal_clarity ==============


class TestAssessGoalClarity:
    def test_empty_low(self):
        from remy.core.confidence_autonomy import _assess_goal_clarity

        assert _assess_goal_clarity("") == 0.1

    def test_short_vague_low(self):
        from remy.core.confidence_autonomy import _assess_goal_clarity

        score = _assess_goal_clarity("do thing")
        assert score <= 0.5

    def test_specific_high(self):
        from remy.core.confidence_autonomy import _assess_goal_clarity

        score = _assess_goal_clarity(
            "search for recent papers about transformer architecture and create a summary with key findings"
        )
        assert score > 0.5


# ============== Unit Tests: format_confidence_info ==============


class TestFormatConfidenceInfo:
    def test_format_high(self):
        from remy.core.confidence_autonomy import AutonomyAction, format_confidence_info

        text = format_confidence_info(0.9, AutonomyAction.EXECUTE_SILENT, "research")
        assert "90%" in text
        assert "HIGH" in text
        assert "research" in text

    def test_format_low(self):
        from remy.core.confidence_autonomy import AutonomyAction, format_confidence_info

        text = format_confidence_info(0.2, AutonomyAction.SKIP)
        assert "20%" in text
        assert "VERY LOW" in text
