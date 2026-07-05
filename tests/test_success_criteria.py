"""Tests for Success Criteria (AUTON-5) — success_criteria.py."""

from unittest.mock import patch

import pytest

# ============== Unit Tests: verify_criterion ==============


class TestVerifyCriterion:
    def test_record_stored_found(self):
        from remy.core.agent_tools import Level, brain
        from remy.core.success_criteria import verify_criterion

        brain.store(content="test metric data", level=Level.WORKING, tags=["metric"])

        met, reason = verify_criterion({"type": "record_stored", "tags": ["metric"]})
        assert met is True
        assert "Found" in reason

    def test_record_stored_not_found(self):
        from remy.core.success_criteria import verify_criterion

        met, reason = verify_criterion({"type": "record_stored", "tags": ["nonexistent-tag-xyz"]})
        assert met is False

    def test_record_stored_no_tags(self):
        from remy.core.success_criteria import verify_criterion

        met, reason = verify_criterion({"type": "record_stored", "tags": []})
        assert met is False

    def test_brain_count_increased_met(self):
        from remy.core.agent_tools import Level, brain
        from remy.core.success_criteria import verify_criterion

        start = brain.count()
        brain.store(content="new record for count test", level=Level.WORKING, tags=["count-test"])

        met, reason = verify_criterion(
            {
                "type": "brain_count_increased",
                "min_delta": 1,
                "start_count": start,
            }
        )
        assert met is True

    def test_brain_count_increased_not_met(self):
        from remy.core.agent_tools import brain
        from remy.core.success_criteria import verify_criterion

        met, reason = verify_criterion(
            {
                "type": "brain_count_increased",
                "min_delta": 100,
                "start_count": brain.count(),
            }
        )
        assert met is False

    def test_brain_count_no_baseline(self):
        from remy.core.success_criteria import verify_criterion

        met, reason = verify_criterion({"type": "brain_count_increased", "min_delta": 1})
        assert met is False
        assert "baseline" in reason.lower()

    def test_file_exists_found(self, tmp_path):
        from remy.core.success_criteria import verify_criterion

        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        met, reason = verify_criterion({"type": "file_exists", "path": str(test_file)})
        assert met is True

    def test_file_exists_not_found(self):
        from remy.core.success_criteria import verify_criterion

        met, reason = verify_criterion({"type": "file_exists", "path": "/nonexistent/path/xyz"})
        assert met is False

    def test_research_complete_not_found(self):
        from remy.core.success_criteria import verify_criterion

        met, reason = verify_criterion({"type": "research_complete", "topic": "quantum physics"})
        assert met is False

    def test_custom_always_not_met(self):
        from remy.core.success_criteria import verify_criterion

        met, reason = verify_criterion({"type": "custom", "description": "Do something"})
        assert met is False
        assert "LLM" in reason

    def test_unknown_type(self):
        from remy.core.success_criteria import verify_criterion

        met, reason = verify_criterion({"type": "unknown_type_xyz"})
        assert met is False
        assert "Unknown" in reason

    def test_tool_result_verified_from_session_log(self):
        from remy.core.success_criteria import verify_criterion

        session_log = [
            {
                "type": "tool_call",
                "tool": "browser_act",
                "verified": True,
                "status": "verified",
                "requested_url": "https://example.com/signup",
                "evidence": {
                    "action": "goto",
                    "page_url": "https://example.com/signup",
                    "screenshot": "shot.png",
                },
            }
        ]

        met, reason = verify_criterion(
            {
                "type": "tool_result",
                "tool": ["browser_act", "browse_page"],
                "verified": True,
                "url_contains": "example.com/signup",
                "evidence_keys": ["page_url", "screenshot"],
            },
            session_log=session_log,
        )
        assert met is True
        assert "runtime result" in reason

    def test_tool_result_requires_session_log(self):
        from remy.core.success_criteria import verify_criterion

        met, reason = verify_criterion(
            {
                "type": "tool_result",
                "tool": "browser_act",
                "verified": True,
            }
        )
        assert met is False
        assert "session log" in reason.lower()

    def test_artifact_created_from_report_result(self):
        from remy.core.success_criteria import verify_criterion

        session_log = [
            {
                "type": "tool_call",
                "tool": "generate_report",
                "generated": True,
                "url": "/api/reports/market.pdf",
                "record_id": "rec-123",
                "filename": "market.pdf",
            }
        ]

        met, reason = verify_criterion(
            {
                "type": "artifact_created",
                "tool": "generate_report",
                "artifact_fields": ["url", "record_id", "filename"],
            },
            session_log=session_log,
        )
        assert met is True
        assert "Artifact field" in reason

    def test_numeric_result_threshold_met(self):
        from remy.core.success_criteria import verify_criterion

        session_log = [
            {
                "type": "tool_call",
                "tool": "add_research_finding",
                "stored": True,
                "findings_count": 12,
            }
        ]

        met, reason = verify_criterion(
            {
                "type": "numeric_result",
                "tool": "add_research_finding",
                "fields": ["findings_count", "count"],
                "min_value": 10,
            },
            session_log=session_log,
        )
        assert met is True
        assert "12" in reason

    def test_signup_completed_detected_from_dashboard_page(self):
        from remy.core.success_criteria import verify_criterion

        session_log = [
            {
                "type": "tool_call",
                "tool": "browser_act",
                "verified": True,
                "status": "verified",
                "page_state": "normal",
                "evidence": {
                    "page_url": "https://app.example.com/dashboard",
                    "page_text_snippet": "Welcome back. Dashboard ready. Sign out",
                },
            }
        ]

        met, reason = verify_criterion(
            {"type": "signup_completed"},
            session_log=session_log,
        )
        assert met is True
        assert "dashboard" in reason.lower()

    def test_signup_not_completed_when_email_verification_pending(self):
        from remy.core.success_criteria import verify_criterion

        session_log = [
            {
                "type": "tool_call",
                "tool": "browser_act",
                "verified": True,
                "status": "verified",
                "page_state": "normal",
                "evidence": {
                    "page_url": "https://app.example.com/welcome",
                    "page_text_snippet": "Check your email to confirm your account",
                },
            }
        ]

        met, reason = verify_criterion(
            {"type": "signup_completed"},
            session_log=session_log,
        )
        assert met is False
        assert "signup evidence" in reason.lower()

    def test_post_published_detected_from_live_url(self):
        from remy.core.success_criteria import verify_criterion

        session_log = [
            {
                "type": "tool_call",
                "tool": "browser_act",
                "verified": True,
                "status": "verified",
                "answer": "Your post is live now",
                "evidence": {
                    "page_url": "https://x.com/test/status/12345",
                    "page_text_snippet": "Your post is live",
                },
            }
        ]

        met, reason = verify_criterion(
            {"type": "post_published"},
            session_log=session_log,
        )
        assert met is True
        assert "live post url" in reason.lower()


# ============== Unit Tests: verify_criteria ==============


class TestVerifyCriteria:
    def test_empty_criteria(self):
        from remy.core.success_criteria import verify_criteria

        met, total, details = verify_criteria([])
        assert met == 0
        assert total == 0
        assert details == []

    def test_mixed_criteria(self):
        from remy.core.agent_tools import Level, brain
        from remy.core.success_criteria import verify_criteria

        brain.store(content="metric record", level=Level.WORKING, tags=["metric"])

        criteria = [
            {"type": "record_stored", "tags": ["metric"]},
            {"type": "custom", "description": "Something else"},
        ]
        met, total, details = verify_criteria(criteria)
        assert met == 1
        assert total == 2
        assert details[0]["met"] is True
        assert details[1]["met"] is False

    def test_all_met(self):
        from remy.core.agent_tools import Level, brain
        from remy.core.success_criteria import verify_criteria

        brain.store(content="tag-a record", level=Level.WORKING, tags=["tag-a"])
        brain.store(content="tag-b record", level=Level.WORKING, tags=["tag-b"])

        criteria = [
            {"type": "record_stored", "tags": ["tag-a"]},
            {"type": "record_stored", "tags": ["tag-b"]},
        ]
        met, total, details = verify_criteria(criteria)
        assert met == 2
        assert total == 2


# ============== Unit Tests: all_criteria_met ==============


class TestAllCriteriaMet:
    def test_returns_false_for_empty(self):
        from remy.core.success_criteria import all_criteria_met

        assert all_criteria_met([]) is False

    def test_returns_true_when_all_met(self):
        from remy.core.agent_tools import Level, brain
        from remy.core.success_criteria import all_criteria_met

        brain.store(content="criteria-met-test", level=Level.WORKING, tags=["criteria-test-tag"])

        assert (
            all_criteria_met(
                [
                    {"type": "record_stored", "tags": ["criteria-test-tag"]},
                ]
            )
            is True
        )

    def test_returns_false_when_partial(self):
        from remy.core.success_criteria import all_criteria_met

        assert (
            all_criteria_met(
                [
                    {"type": "custom", "description": "can't auto-verify"},
                ]
            )
            is False
        )

    def test_returns_true_for_runtime_tool_result(self):
        from remy.core.success_criteria import all_criteria_met

        criteria = [
            {
                "type": "tool_result",
                "tool": "browser_act",
                "verified": True,
                "status_in": ["verified"],
                "url_contains": "example.com/dashboard",
            }
        ]
        session_log = [
            {
                "type": "tool_call",
                "tool": "browser_act",
                "verified": True,
                "status": "verified",
                "evidence": {"page_url": "https://example.com/dashboard"},
            }
        ]

        assert all_criteria_met(criteria, session_log=session_log) is True


# ============== Unit Tests: format_criteria_for_prompt ==============


class TestFormatCriteriaForPrompt:
    def test_empty_returns_empty(self):
        from remy.core.success_criteria import format_criteria_for_prompt

        assert format_criteria_for_prompt([]) == ""

    def test_includes_progress(self):
        from remy.core.agent_tools import Level, brain
        from remy.core.success_criteria import format_criteria_for_prompt

        brain.store(content="fmt-test", level=Level.WORKING, tags=["fmt-test-tag"])

        text = format_criteria_for_prompt(
            [
                {"type": "record_stored", "tags": ["fmt-test-tag"], "description": "Store data"},
                {"type": "custom", "description": "Review results"},
            ]
        )
        assert "SUCCESS CRITERIA" in text
        assert "1/2" in text
        assert "[x]" in text
        assert "[ ]" in text


# ============== Unit Tests: generate_criteria_for_goal ==============


class TestGenerateCriteriaForGoal:
    def test_infer_market_research_template(self):
        from remy.core.success_criteria import infer_goal_template

        template = infer_goal_template(
            "Do market research and competitor analysis for AI memory tools"
        )
        assert template is not None
        assert template["name"] == "market_research"

    def test_infer_signup_template(self):
        from remy.core.success_criteria import infer_goal_template

        template = infer_goal_template("Register at https://example.com/signup")
        assert template is not None
        assert template["name"] == "signup_operator"

    def test_infer_publisher_template(self):
        from remy.core.success_criteria import infer_goal_template

        template = infer_goal_template("Publish a post on https://x.com")
        assert template is not None
        assert template["name"] == "publisher"

    def test_research_goal(self):
        from remy.core.success_criteria import generate_criteria_for_goal

        criteria = generate_criteria_for_goal("Research AI safety techniques")
        assert any(c["type"] == "research_complete" for c in criteria)

    def test_store_goal(self):
        from remy.core.success_criteria import generate_criteria_for_goal

        criteria = generate_criteria_for_goal("Store user preferences")
        assert any(c["type"] == "brain_count_increased" for c in criteria)

    def test_metric_goal(self):
        from remy.core.success_criteria import generate_criteria_for_goal

        criteria = generate_criteria_for_goal("Track project metrics daily")
        assert any(c["type"] == "record_stored" for c in criteria)

    def test_generic_goal_gets_custom(self):
        from remy.core.success_criteria import generate_criteria_for_goal

        criteria = generate_criteria_for_goal("Do something unspecified")
        assert len(criteria) >= 1
        assert criteria[0]["type"] == "custom"

    def test_browser_goal_with_url_gets_tool_result(self):
        from remy.core.success_criteria import generate_criteria_for_goal

        criteria = generate_criteria_for_goal("Register at https://example.com/signup")
        assert any(c["type"] == "tool_result" for c in criteria)

    def test_report_goal_gets_artifact_criterion(self):
        from remy.core.success_criteria import generate_criteria_for_goal

        criteria = generate_criteria_for_goal("Write a market analysis report for AI memory tools")
        assert any(c["type"] == "artifact_created" for c in criteria)

    def test_lead_goal_with_number_gets_numeric_criterion(self):
        from remy.core.success_criteria import generate_criteria_for_goal

        criteria = generate_criteria_for_goal("Collect 25 leads for AI memory startups")
        assert any(c["type"] == "numeric_result" for c in criteria)

    def test_signup_goal_gets_signup_criterion(self):
        from remy.core.success_criteria import generate_criteria_for_goal

        criteria = generate_criteria_for_goal("Register at https://example.com/signup")
        assert any(c["type"] == "signup_completed" for c in criteria)

    def test_publish_goal_gets_post_criterion(self):
        from remy.core.success_criteria import generate_criteria_for_goal

        criteria = generate_criteria_for_goal("Publish a post on https://x.com")
        # post_published or draft_created (agent stops at draft, publishing requires user approval)
        assert any(c["type"] in ("post_published", "draft_created") for c in criteria)

    def test_market_research_template_gets_three_strong_criteria(self):
        from remy.core.success_criteria import generate_criteria_for_goal

        criteria = generate_criteria_for_goal(
            "Do market research and competitor analysis for AI memory tools"
        )
        types = {c["type"] for c in criteria}
        assert "research_complete" in types
        assert "numeric_result" in types
        assert "artifact_created" in types


# ============== Integration: criteria in goal creation ==============


class TestCriteriaInGoalCreation:
    def test_create_goal_includes_criteria(self):
        from remy.core.autonomy_goals import create_goal, get_active_goals

        with patch("remy.core.autonomy.settings") as mock_settings:
            mock_settings.AUTONOMY_DAILY_TOKEN_LIMIT = 100_000
            mock_settings.AUTONOMY_HOURLY_TOKEN_LIMIT = 20_000
            mock_settings.AUTONOMY_SESSION_TOKEN_LIMIT = 500_000
            mock_settings.AUTONOMY_CYCLE_INTERVAL_SEC = 0
            mock_settings.AUTONOMY_TELEGRAM_NOTIFICATIONS = False
            mock_settings.AUTONOMY_SELF_CRITIQUE_ENABLED = False
            mock_settings.SUMMARY_MODEL = "test-model"
            mock_settings.GEMINI_API_KEY = "test-key"
            mock_settings.DATA_DIR = "/tmp"

            record_id = create_goal("Research quantum computing")
            assert record_id is not None

            goals = get_active_goals()
            research_goals = [g for g in goals if "quantum" in g["description"].lower()]
            assert len(research_goals) >= 1
            assert len(research_goals[0].get("success_criteria", [])) >= 1

    def test_create_goal_stores_goal_template_metadata(self):
        from remy.core.agent_tools import brain
        from remy.core.autonomy_goals import create_goal

        with patch("remy.core.autonomy.settings") as mock_settings:
            mock_settings.AUTONOMY_DAILY_TOKEN_LIMIT = 100_000
            mock_settings.AUTONOMY_HOURLY_TOKEN_LIMIT = 20_000
            mock_settings.AUTONOMY_SESSION_TOKEN_LIMIT = 500_000
            mock_settings.AUTONOMY_CYCLE_INTERVAL_SEC = 0
            mock_settings.AUTONOMY_TELEGRAM_NOTIFICATIONS = False
            mock_settings.AUTONOMY_SELF_CRITIQUE_ENABLED = False
            mock_settings.SUMMARY_MODEL = "test-model"
            mock_settings.GEMINI_API_KEY = "test-key"
            mock_settings.DATA_DIR = "/tmp"

            record_id = create_goal("Publish a post on https://x.com", priority="medium")
            rec = brain.get(record_id)

            assert rec.metadata["goal_template"] == "publisher"
            assert rec.metadata["goal_type"] in ("publish", "general")

    def test_create_goal_prefers_explicit_goal_metadata(self):
        from remy.core.agent_tools import brain
        from remy.core.autonomy_goals import create_goal

        with patch("remy.core.autonomy.settings") as mock_settings:
            mock_settings.AUTONOMY_DAILY_TOKEN_LIMIT = 100_000
            mock_settings.AUTONOMY_HOURLY_TOKEN_LIMIT = 20_000
            mock_settings.AUTONOMY_SESSION_TOKEN_LIMIT = 500_000
            mock_settings.AUTONOMY_CYCLE_INTERVAL_SEC = 0
            mock_settings.AUTONOMY_TELEGRAM_NOTIFICATIONS = False
            mock_settings.AUTONOMY_SELF_CRITIQUE_ENABLED = False
            mock_settings.SUMMARY_MODEL = "test-model"
            mock_settings.GEMINI_API_KEY = "test-key"
            mock_settings.DATA_DIR = "/tmp"

            record_id = create_goal(
                "Post-mortem review of launch issues",
                priority="medium",
                goal_type="market",
                goal_template="market_research",
            )
            rec = brain.get(record_id)

            assert rec.metadata["goal_template"] == "market_research"
            assert rec.metadata["goal_type"] == "market"


# ============== Integration: criteria in evaluation ==============


class TestCriteriaInEvaluation:
    @pytest.mark.asyncio
    async def test_auto_complete_when_all_criteria_met(self):
        from remy.core.agent_tools import Level, brain

        brain.store(content="eval-test-data", level=Level.WORKING, tags=["eval-test-tag"])

        criteria = [{"type": "record_stored", "tags": ["eval-test-tag"]}]

        with patch("remy.core.autonomy.settings") as mock_settings:
            mock_settings.AUTONOMY_DAILY_TOKEN_LIMIT = 100_000
            mock_settings.AUTONOMY_HOURLY_TOKEN_LIMIT = 20_000
            mock_settings.AUTONOMY_SESSION_TOKEN_LIMIT = 500_000
            mock_settings.AUTONOMY_CYCLE_INTERVAL_SEC = 0
            mock_settings.AUTONOMY_TELEGRAM_NOTIFICATIONS = False
            mock_settings.AUTONOMY_SELF_CRITIQUE_ENABLED = False
            mock_settings.SUMMARY_MODEL = "test-model"
            mock_settings.GEMINI_API_KEY = "test-key"
            mock_settings.DATA_DIR = "/tmp"

            from remy.core.autonomy import AutonomousLoop

            loop = AutonomousLoop()

            evaluation = await loop._evaluate_outcome(
                "Test goal",
                "Agent did something",
                success_criteria=criteria,
            )

            assert evaluation["success"] is True
            assert evaluation["goal_completed"] is True
            assert evaluation["confidence"] >= 0.9

    @pytest.mark.asyncio
    async def test_auto_complete_when_runtime_criteria_met(self):
        with patch("remy.core.autonomy.settings") as mock_settings:
            mock_settings.AUTONOMY_DAILY_TOKEN_LIMIT = 100_000
            mock_settings.AUTONOMY_HOURLY_TOKEN_LIMIT = 20_000
            mock_settings.AUTONOMY_SESSION_TOKEN_LIMIT = 500_000
            mock_settings.AUTONOMY_CYCLE_INTERVAL_SEC = 0
            mock_settings.AUTONOMY_TELEGRAM_NOTIFICATIONS = False
            mock_settings.AUTONOMY_SELF_CRITIQUE_ENABLED = False
            mock_settings.SUMMARY_MODEL = "test-model"
            mock_settings.GEMINI_API_KEY = "test-key"
            mock_settings.DATA_DIR = "/tmp"

            from remy.core.autonomy import AutonomousLoop

            loop = AutonomousLoop()

            criteria = [
                {
                    "type": "tool_result",
                    "tool": "browser_act",
                    "verified": True,
                    "status_in": ["verified"],
                    "url_contains": "example.com/dashboard",
                }
            ]
            session_log = [
                {
                    "type": "tool_call",
                    "tool": "browser_act",
                    "verified": True,
                    "status": "verified",
                    "evidence": {
                        "action": "goto",
                        "page_url": "https://example.com/dashboard",
                        "screenshot": "dash.png",
                    },
                }
            ]

            evaluation = await loop._evaluate_outcome(
                "Open dashboard",
                "Agent navigated successfully",
                success_criteria=criteria,
                session_log=session_log,
            )

            assert evaluation["success"] is True
            assert evaluation["goal_completed"] is True

    @pytest.mark.asyncio
    async def test_auto_complete_when_artifact_and_count_criteria_met(self):
        with patch("remy.core.autonomy.settings") as mock_settings:
            mock_settings.AUTONOMY_DAILY_TOKEN_LIMIT = 100_000
            mock_settings.AUTONOMY_HOURLY_TOKEN_LIMIT = 20_000
            mock_settings.AUTONOMY_SESSION_TOKEN_LIMIT = 500_000
            mock_settings.AUTONOMY_CYCLE_INTERVAL_SEC = 0
            mock_settings.AUTONOMY_TELEGRAM_NOTIFICATIONS = False
            mock_settings.AUTONOMY_SELF_CRITIQUE_ENABLED = False
            mock_settings.SUMMARY_MODEL = "test-model"
            mock_settings.GEMINI_API_KEY = "test-key"
            mock_settings.DATA_DIR = "/tmp"

            from remy.core.autonomy import AutonomousLoop

            loop = AutonomousLoop()

            criteria = [
                {
                    "type": "artifact_created",
                    "tool": "generate_report",
                    "artifact_fields": ["url", "record_id"],
                },
                {
                    "type": "numeric_result",
                    "tool": "add_research_finding",
                    "fields": ["findings_count"],
                    "min_value": 3,
                },
            ]
            session_log = [
                {
                    "type": "tool_call",
                    "tool": "add_research_finding",
                    "stored": True,
                    "findings_count": 3,
                },
                {
                    "type": "tool_call",
                    "tool": "generate_report",
                    "generated": True,
                    "url": "/api/reports/market.pdf",
                    "record_id": "rec-123",
                },
            ]

            evaluation = await loop._evaluate_outcome(
                "Produce a market analysis report",
                "Agent generated the report",
                success_criteria=criteria,
                session_log=session_log,
            )

            assert evaluation["success"] is True
            assert evaluation["goal_completed"] is True

    @pytest.mark.asyncio
    async def test_auto_complete_when_signup_criteria_met(self):
        with patch("remy.core.autonomy.settings") as mock_settings:
            mock_settings.AUTONOMY_DAILY_TOKEN_LIMIT = 100_000
            mock_settings.AUTONOMY_HOURLY_TOKEN_LIMIT = 20_000
            mock_settings.AUTONOMY_SESSION_TOKEN_LIMIT = 500_000
            mock_settings.AUTONOMY_CYCLE_INTERVAL_SEC = 0
            mock_settings.AUTONOMY_TELEGRAM_NOTIFICATIONS = False
            mock_settings.AUTONOMY_SELF_CRITIQUE_ENABLED = False
            mock_settings.SUMMARY_MODEL = "test-model"
            mock_settings.GEMINI_API_KEY = "test-key"
            mock_settings.DATA_DIR = "/tmp"

            from remy.core.autonomy import AutonomousLoop

            loop = AutonomousLoop()

            evaluation = await loop._evaluate_outcome(
                "Register account",
                "Agent reached the dashboard",
                success_criteria=[{"type": "signup_completed"}],
                session_log=[
                    {
                        "type": "tool_call",
                        "tool": "browser_act",
                        "verified": True,
                        "status": "verified",
                        "evidence": {
                            "page_url": "https://app.example.com/dashboard",
                            "page_text_snippet": "Welcome dashboard sign out",
                        },
                    }
                ],
            )

            assert evaluation["success"] is True
            assert evaluation["goal_completed"] is True
