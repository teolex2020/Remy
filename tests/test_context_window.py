"""Tests for Dynamic Context Window & State Summarization (AUTON-12) — context_window.py."""


# ============== Unit Tests: estimate_complexity ==============


class TestEstimateComplexity:
    def test_empty_goal(self):
        from remy.core.context_window import estimate_complexity

        score = estimate_complexity("")
        assert score == 0.3

    def test_simple_goal(self):
        from remy.core.context_window import estimate_complexity

        score = estimate_complexity("check status of the system")
        assert score < 0.3  # Simple keywords reduce score

    def test_complex_goal(self):
        from remy.core.context_window import estimate_complexity

        score = estimate_complexity("research and analyze the topic, then implement a solution")
        assert score > 0.5

    def test_long_description_adds_complexity(self):
        from remy.core.context_window import estimate_complexity

        short = estimate_complexity("do something")
        long = estimate_complexity("do something " + "with details " * 20)
        assert long > short

    def test_numbered_steps_increase_complexity(self):
        from remy.core.context_window import estimate_complexity

        without = estimate_complexity("implement the feature")
        with_steps = estimate_complexity("1. plan the feature 2. implement it 3. test it")
        assert with_steps > without

    def test_many_attempts_increase_complexity(self):
        from remy.core.context_window import estimate_complexity

        fresh = estimate_complexity("do task", attempts=0)
        retried = estimate_complexity("do task", attempts=5)
        assert retried > fresh

    def test_capped_at_1(self):
        from remy.core.context_window import estimate_complexity

        score = estimate_complexity(
            "research and analyze and investigate and implement and develop and build "
            "1. step 2. step 3. step " * 5,
            attempts=10,
        )
        assert score <= 1.0

    def test_capped_at_0(self):
        from remy.core.context_window import estimate_complexity

        score = estimate_complexity("check get list count verify show read")
        assert score >= 0.0


# ============== Unit Tests: context_size_for_complexity ==============


class TestContextSizeForComplexity:
    def test_low_complexity_small_context(self):
        from remy.core.context_window import context_size_for_complexity

        size = context_size_for_complexity(0.1)
        assert 12 <= size <= 16

    def test_medium_complexity_medium_context(self):
        from remy.core.context_window import context_size_for_complexity

        size = context_size_for_complexity(0.5)
        assert 16 <= size <= 28

    def test_high_complexity_large_context(self):
        from remy.core.context_window import context_size_for_complexity

        size = context_size_for_complexity(0.9)
        assert 22 <= size <= 28  # v2.4: reduced from 28-48

    def test_zero_gives_minimum(self):
        from remy.core.context_window import context_size_for_complexity

        assert context_size_for_complexity(0.0) == 12

    def test_one_gives_maximum(self):
        from remy.core.context_window import context_size_for_complexity

        assert context_size_for_complexity(1.0) == 28  # v2.4: reduced from 48


# ============== Unit Tests: dynamic_keep_recent ==============


class TestDynamicKeepRecent:
    def test_non_autonomous_defaults(self):
        from remy.core.context_window import dynamic_keep_recent

        assert dynamic_keep_recent("telegram", "") == 16
        assert dynamic_keep_recent("desktop", "") == 16

    def test_non_autonomous_research(self):
        from remy.core.context_window import dynamic_keep_recent

        assert dynamic_keep_recent("telegram", "research topic") == 32

    def test_autonomous_simple_goal(self):
        from remy.core.context_window import dynamic_keep_recent

        size = dynamic_keep_recent("autonomous", "check system status")
        assert size < 20

    def test_autonomous_complex_goal(self):
        from remy.core.context_window import dynamic_keep_recent

        size = dynamic_keep_recent(
            "autonomous", "research and analyze AI trends, then implement findings"
        )
        assert size > 20


# ============== Unit Tests: score_message_importance ==============


class TestScoreMessageImportance:
    def test_system_message_highest(self):
        from langchain_core.messages import SystemMessage

        from remy.core.context_window import score_message_importance

        msg = SystemMessage(content="system context")
        assert score_message_importance(msg) == 1.0

    def test_human_message_high(self):
        from langchain_core.messages import HumanMessage

        from remy.core.context_window import score_message_importance

        msg = HumanMessage(content="user question")
        assert score_message_importance(msg) >= 0.7

    def test_tool_error_important(self):
        from langchain_core.messages import ToolMessage

        from remy.core.context_window import score_message_importance

        msg = ToolMessage(content='{"error": "something failed"}', tool_call_id="1")
        assert score_message_importance(msg) >= 0.7

    def test_tool_result_moderate(self):
        from langchain_core.messages import ToolMessage

        from remy.core.context_window import score_message_importance

        msg = ToolMessage(
            content='{"data": "some result value here with useful info"}', tool_call_id="1"
        )
        assert 0.3 <= score_message_importance(msg) <= 0.5


# ============== Unit Tests: select_important_messages ==============


class TestSelectImportantMessages:
    def test_short_list_unchanged(self):
        from remy.core.context_window import select_important_messages

        msgs = [1, 2, 3]
        assert select_important_messages(msgs, budget=5) == [1, 2, 3]

    def test_budget_exceeded_selects_important(self):
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        from remy.core.context_window import select_important_messages

        msgs = [
            SystemMessage(content="system"),  # importance 1.0
            ToolMessage(content="ok", tool_call_id="1"),  # importance 0.3
            HumanMessage(content="question"),  # importance 0.7
            ToolMessage(content='{"error": "fail"}', tool_call_id="2"),  # importance 0.8
            HumanMessage(content="another"),  # importance 0.7 (recent)
        ]
        selected = select_important_messages(msgs, budget=3, always_keep_recent=1)
        assert len(selected) <= 3
        # Recent message always kept
        assert msgs[-1] in selected

    def test_recent_always_kept(self):
        from langchain_core.messages import HumanMessage

        from remy.core.context_window import select_important_messages

        msgs = [HumanMessage(content=f"msg{i}") for i in range(10)]
        selected = select_important_messages(msgs, budget=5, always_keep_recent=3)
        # Last 3 always kept
        assert msgs[-1] in selected
        assert msgs[-2] in selected
        assert msgs[-3] in selected


# ============== Unit Tests: SessionState ==============


class TestSessionState:
    def test_to_text(self):
        from remy.core.context_window import SessionState

        state = SessionState(
            current_goal="research AI",
            progress="50% done",
            key_findings=["finding 1", "finding 2"],
            blockers=["network timeout"],
            actions_taken=15,
        )
        text = state.to_text()
        assert "research AI" in text
        assert "15" in text
        assert "finding 1" in text
        assert "network timeout" in text

    def test_empty_state(self):
        from remy.core.context_window import SessionState

        state = SessionState()
        text = state.to_text()
        assert "actions: 0" in text


# ============== Unit Tests: update_session_state ==============


class TestUpdateSessionState:
    def test_updates_action_count(self):
        from remy.core.context_window import clear_session_state, update_session_state

        sid = "test-session-update"
        try:
            state = update_session_state(sid, goal="test goal", action_result="done", success=True)
            assert state.actions_taken == 1
            state = update_session_state(sid, action_result="done again", success=True)
            assert state.actions_taken == 2
        finally:
            clear_session_state(sid)

    def test_tracks_findings(self):
        from remy.core.context_window import clear_session_state, update_session_state

        sid = "test-session-findings"
        try:
            state = update_session_state(
                sid,
                goal="research",
                action_result="Found that AI models improve with more data. This is well established.",
                success=True,
            )
            assert len(state.key_findings) == 1
        finally:
            clear_session_state(sid)

    def test_tracks_blockers(self):
        from remy.core.context_window import clear_session_state, update_session_state

        sid = "test-session-blockers"
        try:
            state = update_session_state(
                sid,
                action_result="Network timeout connecting to API",
                success=False,
            )
            assert len(state.blockers) == 1
            assert "Network" in state.blockers[0]
        finally:
            clear_session_state(sid)


# ============== Unit Tests: get_state_summary ==============


class TestGetStateSummary:
    def test_no_summary_when_few_actions(self):
        from remy.core.context_window import (
            clear_session_state,
            get_state_summary,
            update_session_state,
        )

        sid = "test-few-actions"
        try:
            update_session_state(sid, goal="test", action_result="ok", success=True)
            assert get_state_summary(sid) is None  # < 10 actions
        finally:
            clear_session_state(sid)

    def test_summary_after_enough_actions(self):
        from remy.core.context_window import (
            clear_session_state,
            get_state_summary,
            update_session_state,
        )

        sid = "test-many-actions"
        try:
            for i in range(11):
                update_session_state(
                    sid, goal="research AI", action_result=f"step {i} done", success=True
                )
            summary = get_state_summary(sid)
            assert summary is not None
            assert "research AI" in summary
        finally:
            clear_session_state(sid)

    def test_unknown_session_returns_none(self):
        from remy.core.context_window import get_state_summary

        assert get_state_summary("nonexistent-session") is None


# ============== Unit Tests: should_inject_state ==============


class TestShouldInjectState:
    def test_inject_at_intervals(self):
        from remy.core.context_window import should_inject_state

        assert should_inject_state("s", 10) is True
        assert should_inject_state("s", 20) is True

    def test_no_inject_between_intervals(self):
        from remy.core.context_window import should_inject_state

        assert should_inject_state("s", 5) is False
        assert should_inject_state("s", 15) is False

    def test_no_inject_at_zero(self):
        from remy.core.context_window import should_inject_state

        assert should_inject_state("s", 0) is False


# ============== Integration: agent _estimate_keep_recent ==============


class TestAgentIntegration:
    def test_autonomous_uses_dynamic(self):
        from langchain_core.messages import HumanMessage

        from remy.core.agent import _estimate_keep_recent

        # Simple autonomous goal → smaller context
        simple = _estimate_keep_recent("autonomous", HumanMessage(content="check status"))
        # Complex autonomous goal → larger context
        complex_ = _estimate_keep_recent(
            "autonomous", HumanMessage(content="research and analyze trends, then implement")
        )
        assert complex_ > simple

    def test_telegram_unchanged(self):
        from remy.core.agent import _estimate_keep_recent

        assert _estimate_keep_recent("telegram", "hello") == 16
        assert _estimate_keep_recent("desktop", "hello") == 16
