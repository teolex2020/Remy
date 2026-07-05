"""Tests for orchestrator — worker selection, dispatch routing, and post-exec decisions."""

from remy.core.orchestrator import (
    _should_use_browser_worker,
    _should_use_research_worker,
    check_obvious_failure,
    check_zero_tool_cycle,
    detect_external_blocker,
    focus_execution_goals,
    format_execution_report,
    select_worker,
)
from remy.core.workers.contracts import WorkerExecutionResult

# ============== Worker Selection ==============


class TestSelectWorker:
    def test_signup_routes_to_browser(self):
        assert select_worker({"goal_template": "signup_operator"}) == "browser_worker"

    def test_publisher_routes_to_browser(self):
        assert select_worker({"goal_template": "publisher"}) == "browser_worker"

    def test_market_research_routes_to_research(self):
        assert select_worker({"goal_template": "market_research"}) == "research_worker"

    def test_research_keyword_routes_to_research(self):
        assert select_worker({"description": "Research competitor pricing"}) == "research_worker"

    def test_osint_keyword_routes_to_research(self):
        assert select_worker({"description": "Run OSINT analysis on targets"}) == "research_worker"

    def test_competitive_analysis_keyword(self):
        assert (
            select_worker({"description": "Competitive analysis of the market"})
            == "research_worker"
        )

    def test_generic_goal_routes_to_generic(self):
        assert (
            select_worker({"description": "Update user profile", "goal_template": "general"})
            == "generic"
        )

    def test_none_goal_routes_to_generic(self):
        assert select_worker(None) == "generic"

    def test_empty_goal_routes_to_generic(self):
        assert select_worker({}) == "generic"

    def test_browser_takes_priority_over_research_keywords(self):
        # signup_operator template should win even if description has research keywords
        assert (
            select_worker(
                {
                    "goal_template": "signup_operator",
                    "description": "Research and sign up for platform",
                }
            )
            == "browser_worker"
        )


# ============== Backward Compat Helpers ==============


class TestBackwardCompat:
    def test_should_use_browser_worker(self):
        assert _should_use_browser_worker({"goal_template": "signup_operator"}) is True
        assert _should_use_browser_worker({"goal_template": "market_research"}) is False
        assert _should_use_browser_worker(None) is False

    def test_should_use_research_worker(self):
        assert _should_use_research_worker({"goal_template": "market_research"}) is True
        assert _should_use_research_worker({"description": "Research AI trends"}) is True
        assert _should_use_research_worker({"description": "Sign up for service"}) is False
        assert _should_use_research_worker(None) is False


class TestFocusExecutionGoals:
    def test_prefers_runnable_mission_tasks(self):
        goals = [
            {"goal_id": "legacy", "status": "active", "description": "Old legacy goal"},
            {
                "goal_id": "mission-task",
                "mission_id": "m1",
                "mission_task_id": "t1",
                "status": "active",
            },
            {"goal_id": "mission-parent", "mission_id": "m1", "status": "active"},
        ]
        focused = focus_execution_goals(goals)
        assert [g["goal_id"] for g in focused] == ["mission-task", "mission-parent"]

    def test_returns_original_when_no_mission_tasks(self):
        goals = [
            {"goal_id": "g1", "status": "active"},
            {"goal_id": "g2", "status": "active"},
        ]
        assert focus_execution_goals(goals) == goals


# ============== Blocker Detection ==============


class TestDetectExternalBlocker:
    def test_none_for_non_browser_goal(self):
        assert detect_external_blocker({"goal_template": "market_research"}, []) is None

    def test_none_for_none_goal(self):
        assert detect_external_blocker(None, []) is None

    def test_detects_structured_blocker(self):
        session_log = [
            {
                "type": "tool_call",
                "tool": "browse_page",
                "external_blocker_likely": True,
                "blocker_reason": "email verification required",
                "evidence": {"page_url": "https://example.com/verify"},
            }
        ]
        result = detect_external_blocker({"goal_template": "signup_operator"}, session_log)
        assert result is not None
        assert result["reason"] == "email verification required"
        assert "example.com" in result["evidence"]

    def test_detects_marker_in_page_state(self):
        session_log = [
            {
                "type": "tool_call",
                "tool": "browser_act",
                "page_state": "Please verify your email to continue",
                "evidence": {"page_url": "https://example.com/signup"},
            }
        ]
        result = detect_external_blocker({"goal_template": "signup_operator"}, session_log)
        assert result is not None
        assert "email verification" in result["reason"]

    def test_no_blocker_in_clean_session(self):
        session_log = [
            {
                "type": "tool_call",
                "tool": "browse_page",
                "page_state": "dashboard loaded",
                "evidence": {"page_url": "https://example.com/dashboard"},
            }
        ]
        result = detect_external_blocker({"goal_template": "signup_operator"}, session_log)
        assert result is None

    def test_ignores_non_browser_tool_calls(self):
        session_log = [
            {
                "type": "tool_call",
                "tool": "web_search",
                "page_state": "captcha required",
            }
        ]
        result = detect_external_blocker({"goal_template": "signup_operator"}, session_log)
        assert result is None


# ============== Zero-Tool Guard ==============


class TestCheckZeroToolCycle:
    def test_returns_failure_when_no_tools(self):
        result = check_zero_tool_cycle([], "Agent produced some text")
        assert result is not None
        assert result["success"] is False
        assert "No tools called" in result["reason"]

    def test_returns_none_when_tools_used(self):
        log = [{"type": "tool_call", "tool": "recall", "args": {}}]
        result = check_zero_tool_cycle(log, "Agent produced some text")
        assert result is None

    def test_returns_none_when_no_response(self):
        result = check_zero_tool_cycle([], "")
        assert result is None


# ============== Obvious Failure ==============


class TestCheckObviousFailure:
    def test_detects_error_response(self):
        result = check_obvious_failure("error: something went wrong")
        assert result is not None
        assert result["success"] is False

    def test_detects_failed_to(self):
        result = check_obvious_failure("failed to connect to API")
        assert result is not None
        assert result["success"] is False

    def test_ignores_long_responses(self):
        # Long responses with "error" might be legitimate reports
        result = check_obvious_failure("error " + "x" * 300)
        assert result is None

    def test_returns_none_for_normal_response(self):
        result = check_obvious_failure("Successfully completed the research task with 5 findings.")
        assert result is None


# ============== Format Execution Report ==============


class TestFormatExecutionReport:
    def test_returns_fallback_when_no_result(self):
        assert format_execution_report(None, "raw text") == "raw text"

    def test_formats_browser_result(self):
        result = WorkerExecutionResult(
            worker="browser_worker",
            status="verified",
            response_text="Done.",
            evidence={"current_url": "https://example.com/dashboard"},
        )
        text = format_execution_report(result, "fallback")
        assert "Status: verified" in text
        assert "example.com" in text

    def test_formats_research_result(self):
        result = WorkerExecutionResult(
            worker="research_worker",
            status="findings_collected",
            response_text="Found competitors.",
            evidence={"findings_count": 3, "sources": ["https://a.com"]},
        )
        text = format_execution_report(result, "fallback")
        assert "Status: findings_collected" in text
        assert "Findings: 3" in text


# ============== Import from autonomy.py (backward compat) ==============


class TestAutonomyBackwardCompat:
    def test_detect_external_blocker_importable(self):
        from remy.core.autonomy import _detect_external_blocker

        assert callable(_detect_external_blocker)

    def test_invoke_goal_worker_importable(self):
        from remy.core.autonomy import _invoke_goal_worker

        assert callable(_invoke_goal_worker)

    def test_format_execution_report_importable(self):
        from remy.core.autonomy import _format_execution_report

        assert callable(_format_execution_report)

    def test_should_use_browser_worker_importable(self):
        from remy.core.autonomy import _should_use_browser_worker

        assert _should_use_browser_worker({"goal_template": "signup_operator"}) is True

    def test_should_use_research_worker_importable(self):
        from remy.core.autonomy import _should_use_research_worker

        assert _should_use_research_worker({"goal_template": "market_research"}) is True
