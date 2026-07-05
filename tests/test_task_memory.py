"""Tests for P3.2 task memory — per-goal execution history."""

from unittest.mock import patch

from remy.core.task_memory import (
    FailureReason,
    GoalAttemptRecord,
    classify_failure,
    format_goal_history_for_prompt,
    get_goal_history,
    get_goal_history_summary,
    log_goal_attempt,
)

# ============== Failure Classification ==============


class TestClassifyFailure:
    def test_timeout(self):
        assert classify_failure(None, timeout=True) == FailureReason.TIMEOUT

    def test_blocked_external(self):
        assert classify_failure(None, blocked_external=True) == FailureReason.BLOCKED_EXTERNAL

    def test_zero_tool(self):
        assert classify_failure(None, zero_tool=True) == FailureReason.ZERO_TOOL

    def test_success(self):
        assert classify_failure({"success": True}) == FailureReason.SUCCESS

    def test_no_action(self):
        assert (
            classify_failure({"success": False}, worker_status="no_action")
            == FailureReason.NO_ACTION
        )

    def test_tool_error(self):
        log = [{"type": "tool_call", "tool": "browse_page", "error": "timeout"}]
        assert classify_failure({"success": False}, session_log=log) == FailureReason.TOOL_ERROR

    def test_validation_failed(self):
        assert (
            classify_failure({"success": False, "reason": "Success criteria not met"})
            == FailureReason.VALIDATION_FAILED
        )

    def test_repeated_failure(self):
        assert (
            classify_failure({"success": False, "reason": "Same approach repeated again"})
            == FailureReason.REPEATED_FAILURE
        )

    def test_unknown(self):
        assert (
            classify_failure({"success": False, "reason": "something happened"})
            == FailureReason.UNKNOWN
        )

    def test_none_evaluation(self):
        assert classify_failure(None) == FailureReason.UNKNOWN

    def test_priority_timeout_over_blocked(self):
        assert classify_failure(None, timeout=True, blocked_external=True) == FailureReason.TIMEOUT


# ============== Goal Attempt Recording ==============


class TestLogGoalAttempt:
    def test_record_and_retrieve(self, tmp_path):
        with patch("remy.core.task_memory.settings") as mock:
            mock.DATA_DIR = tmp_path
            log_goal_attempt(
                "goal-123",
                worker="browser_worker",
                status="verified",
                failure_reason=FailureReason.SUCCESS,
                success=True,
                duration_ms=2000,
                tokens_used=100,
            )
            history = get_goal_history("goal-123")

        assert len(history) == 1
        assert history[0]["worker"] == "browser_worker"
        assert history[0]["success"] is True
        assert history[0]["failure_reason"] == "success"

    def test_multiple_attempts(self, tmp_path):
        with patch("remy.core.task_memory.settings") as mock:
            mock.DATA_DIR = tmp_path
            for i in range(5):
                log_goal_attempt(
                    "goal-456",
                    worker="browser_worker",
                    status="failed" if i < 3 else "verified",
                    failure_reason=FailureReason.TOOL_ERROR if i < 3 else FailureReason.SUCCESS,
                    success=i >= 3,
                    duration_ms=1000 + i * 100,
                )
            history = get_goal_history("goal-456")

        assert len(history) == 5
        assert sum(1 for a in history if a["success"]) == 2
        assert sum(1 for a in history if not a["success"]) == 3

    def test_caps_at_max(self, tmp_path):
        with patch("remy.core.task_memory.settings") as mock:
            mock.DATA_DIR = tmp_path
            for i in range(60):
                log_goal_attempt(
                    "goal-cap",
                    worker="",
                    status="failed",
                    failure_reason=FailureReason.UNKNOWN,
                    success=False,
                )
            history = get_goal_history("goal-cap")

        assert len(history) == 50  # _MAX_ATTEMPTS_PER_GOAL

    def test_empty_goal_id_ignored(self, tmp_path):
        with patch("remy.core.task_memory.settings") as mock:
            mock.DATA_DIR = tmp_path
            log_goal_attempt("", worker="", status="", failure_reason="", success=False)
            # Should not create any file
            assert list(tmp_path.glob("goal_history/*.json")) == []

    def test_history_limit(self, tmp_path):
        with patch("remy.core.task_memory.settings") as mock:
            mock.DATA_DIR = tmp_path
            for _ in range(10):
                log_goal_attempt(
                    "goal-lim",
                    worker="",
                    status="failed",
                    failure_reason=FailureReason.UNKNOWN,
                    success=False,
                )
            history = get_goal_history("goal-lim", limit=3)

        assert len(history) == 3

    def test_nonexistent_goal_returns_empty(self, tmp_path):
        with patch("remy.core.task_memory.settings") as mock:
            mock.DATA_DIR = tmp_path
            assert get_goal_history("nonexistent") == []


# ============== Goal History Summary ==============


class TestGoalHistorySummary:
    def test_summary_stats(self, tmp_path):
        with patch("remy.core.task_memory.settings") as mock:
            mock.DATA_DIR = tmp_path
            log_goal_attempt(
                "goal-sum",
                worker="browser_worker",
                status="failed",
                failure_reason=FailureReason.TOOL_ERROR,
                success=False,
                duration_ms=1000,
                tokens_used=50,
            )
            log_goal_attempt(
                "goal-sum",
                worker="browser_worker",
                status="failed",
                failure_reason=FailureReason.BLOCKED_EXTERNAL,
                success=False,
                duration_ms=2000,
                tokens_used=80,
            )
            log_goal_attempt(
                "goal-sum",
                worker="browser_worker",
                status="verified",
                failure_reason=FailureReason.SUCCESS,
                success=True,
                duration_ms=1500,
                tokens_used=60,
            )
            summary = get_goal_history_summary("goal-sum")

        assert summary["total_attempts"] == 3
        assert summary["successes"] == 1
        assert summary["failures"] == 2
        assert summary["completion_rate"] == round(1 / 3, 3)
        assert summary["total_duration_ms"] == 4500
        assert summary["total_tokens"] == 190
        assert summary["last_status"] == "verified"
        assert summary["top_failure"] in ("tool_error", "blocked_external")

    def test_empty_summary(self, tmp_path):
        with patch("remy.core.task_memory.settings") as mock:
            mock.DATA_DIR = tmp_path
            summary = get_goal_history_summary("no-such-goal")

        assert summary["total_attempts"] == 0

    def test_all_successes(self, tmp_path):
        with patch("remy.core.task_memory.settings") as mock:
            mock.DATA_DIR = tmp_path
            for _ in range(3):
                log_goal_attempt(
                    "goal-ok",
                    worker="",
                    status="verified",
                    failure_reason=FailureReason.SUCCESS,
                    success=True,
                )
            summary = get_goal_history_summary("goal-ok")

        assert summary["completion_rate"] == 1.0
        assert summary["top_failure"] == ""


# ============== Prompt Formatting ==============


class TestFormatGoalHistory:
    def test_empty_returns_empty(self, tmp_path):
        with patch("remy.core.task_memory.settings") as mock:
            mock.DATA_DIR = tmp_path
            text = format_goal_history_for_prompt("nonexistent")

        assert text == ""

    def test_formats_recent_attempts(self, tmp_path):
        with patch("remy.core.task_memory.settings") as mock:
            mock.DATA_DIR = tmp_path
            log_goal_attempt(
                "goal-fmt",
                worker="browser_worker",
                status="failed",
                failure_reason=FailureReason.TOOL_ERROR,
                success=False,
                duration_ms=1500,
            )
            log_goal_attempt(
                "goal-fmt",
                worker="browser_worker",
                status="verified",
                failure_reason=FailureReason.SUCCESS,
                success=True,
                duration_ms=2000,
            )
            text = format_goal_history_for_prompt("goal-fmt")

        assert "PREVIOUS ATTEMPTS" in text
        assert "[FAIL]" in text
        assert "[OK]" in text
        assert "browser_worker" in text
        assert "Do NOT repeat" in text

    def test_warns_on_repeated_failure(self, tmp_path):
        with patch("remy.core.task_memory.settings") as mock:
            mock.DATA_DIR = tmp_path
            for _ in range(4):
                log_goal_attempt(
                    "goal-rep",
                    worker="browser_worker",
                    status="failed",
                    failure_reason=FailureReason.TOOL_ERROR,
                    success=False,
                )
            text = format_goal_history_for_prompt("goal-rep")

        assert "WARNING" in text
        assert "tool_error" in text
        assert "repeated 3+" in text

    def test_no_warning_on_varied_failures(self, tmp_path):
        with patch("remy.core.task_memory.settings") as mock:
            mock.DATA_DIR = tmp_path
            log_goal_attempt(
                "goal-var",
                worker="",
                status="failed",
                failure_reason=FailureReason.TOOL_ERROR,
                success=False,
            )
            log_goal_attempt(
                "goal-var",
                worker="",
                status="failed",
                failure_reason=FailureReason.TIMEOUT,
                success=False,
            )
            log_goal_attempt(
                "goal-var",
                worker="",
                status="failed",
                failure_reason=FailureReason.BLOCKED_EXTERNAL,
                success=False,
            )
            text = format_goal_history_for_prompt("goal-var")

        assert "WARNING" not in text

    def test_limits_output(self, tmp_path):
        with patch("remy.core.task_memory.settings") as mock:
            mock.DATA_DIR = tmp_path
            for i in range(10):
                log_goal_attempt(
                    "goal-lim2",
                    worker="",
                    status="failed",
                    failure_reason=FailureReason.UNKNOWN,
                    success=False,
                )
            text = format_goal_history_for_prompt("goal-lim2", limit=3)

        # Should show "3 of 10 total"
        assert "3 of 10" in text

    def test_includes_blocker_info(self, tmp_path):
        with patch("remy.core.task_memory.settings") as mock:
            mock.DATA_DIR = tmp_path
            log_goal_attempt(
                "goal-blk",
                worker="browser_worker",
                status="blocked_external",
                failure_reason=FailureReason.BLOCKED_EXTERNAL,
                success=False,
                blocker="captcha challenge on example.com",
            )
            text = format_goal_history_for_prompt("goal-blk")

        assert "captcha" in text
        assert "blocker=" in text


# ============== GoalAttemptRecord Dataclass ==============


class TestGoalAttemptRecord:
    def test_basic(self):
        r = GoalAttemptRecord(
            timestamp="2026-03-07T12:00:00",
            worker="browser_worker",
            status="verified",
            failure_reason="success",
            success=True,
        )
        assert r.worker == "browser_worker"
        assert r.success is True

    def test_with_all_fields(self):
        r = GoalAttemptRecord(
            timestamp="2026-03-07T12:00:00",
            worker="research_worker",
            status="searching",
            failure_reason="unknown",
            success=False,
            duration_ms=5000,
            tokens_used=200,
            blocker="rate limited",
            evidence_summary="Found 2 of 5 sources",
            goal_template="market_research",
        )
        assert r.goal_template == "market_research"
        assert r.blocker == "rate limited"
