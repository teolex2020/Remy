"""Tests for F3: Implicit Feedback Signals — behavioral adaptation."""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from remy.core.brain_tools import (
    FeedbackSignal,
    apply_latest_user_correction_feedback,
    detect_feedback_signals,
    get_recent_feedback_summary,
    store_feedback_signal,
)


class TestDetectFeedbackSignals:
    """Tests for detect_feedback_signals()."""

    def test_detect_verbose_feedback(self):
        # AI gives 200+ word response, user replies with just "ok"
        long_response = "word " * 200
        msgs = [
            HumanMessage(content="Tell me about everything"),
            AIMessage(content=long_response),
            HumanMessage(content="ok"),
        ]

        signals = detect_feedback_signals(msgs, "desktop")

        verbose_signals = [s for s in signals if s.signal_type == "too_verbose"]
        assert len(verbose_signals) >= 1
        assert verbose_signals[0].severity > 0.5

    def test_detect_topic_switch(self):
        # AI talks about cooking, user asks about programming
        msgs = [
            HumanMessage(content="Tell me about cooking"),
            AIMessage(content="Here are some great pasta recipes. You can use tomatoes and basil for a delicious sauce."),
            HumanMessage(content="How do I install Python on my computer?"),
        ]

        signals = detect_feedback_signals(msgs, "desktop")

        switch_signals = [s for s in signals if s.signal_type == "topic_switch"]
        assert len(switch_signals) >= 1
        assert switch_signals[0].severity == 0.6

    def test_detect_repeat_question(self):
        # User asks similar question twice
        msgs = [
            HumanMessage(content="What is the weather like in Kyiv today?"),
            AIMessage(content="It's sunny and warm in Kyiv."),
            HumanMessage(content="Tell me about weather in Kyiv today please"),
        ]

        signals = detect_feedback_signals(msgs, "telegram")

        repeat_signals = [s for s in signals if s.signal_type == "repeat_question"]
        assert len(repeat_signals) >= 1
        assert repeat_signals[0].severity > 0.6

    def test_no_false_positives_short_convo(self):
        msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]

        signals = detect_feedback_signals(msgs, "desktop")
        assert len(signals) == 0

    def test_no_signals_on_empty_messages(self):
        signals = detect_feedback_signals([], "desktop")
        assert signals == []

    def test_no_verbose_when_ai_is_short(self):
        msgs = [
            HumanMessage(content="Hey there"),
            AIMessage(content="Short reply."),
            HumanMessage(content="Thanks"),
        ]

        signals = detect_feedback_signals(msgs, "desktop")

        verbose_signals = [s for s in signals if s.signal_type == "too_verbose"]
        assert len(verbose_signals) == 0

    def test_no_topic_switch_with_overlap(self):
        # Same topic discussed
        msgs = [
            HumanMessage(content="Tell me about Python"),
            AIMessage(content="Python is a great programming language for beginners."),
            HumanMessage(content="What Python libraries should I learn first?"),
        ]

        signals = detect_feedback_signals(msgs, "desktop")

        switch_signals = [s for s in signals if s.signal_type == "topic_switch"]
        assert len(switch_signals) == 0


class TestStoreFeedbackSignal:
    """Tests for store_feedback_signal()."""

    def test_store_signal(self):
        with patch("remy.core.brain_tools.brain") as mock_brain:
            signal = FeedbackSignal(
                signal_type="too_verbose",
                severity=0.8,
                context="AI: 200w -> User: 'ok'",
                channel="desktop",
                timestamp="2026-02-16T10:00:00",
            )

            store_feedback_signal(signal)

            mock_brain.store.assert_called_once()
            call_kwargs = mock_brain.store.call_args.kwargs
            assert "feedback-signal" in call_kwargs["tags"]
            assert "too_verbose" in call_kwargs["tags"]

    def test_store_handles_errors(self):
        with patch("remy.core.brain_tools.brain") as mock_brain:
            mock_brain.store.side_effect = RuntimeError("brain error")

            signal = FeedbackSignal(
                signal_type="topic_switch", severity=0.6,
                context="test", channel="desktop", timestamp="2026-01-01",
            )
            # Should not raise
            store_feedback_signal(signal)


class TestApplyLatestUserCorrectionFeedback:
    def test_prefers_factuality_evidence_record_ids_over_text_guess(self):
        with patch("remy.core.brain_tools.brain") as mock_brain, patch("remy.core.brain_tools.brain_lock"):
            evidence_record = MagicMock()
            evidence_record.id = "rec-evidence"
            evidence_record.content = "Ганна народилася 1940 року."

            distractor = MagicMock()
            distractor.id = "rec-distractor"
            distractor.content = "Марія народилася 1940 року."
            distractor.strength = 0.9

            mock_brain.get.return_value = evidence_record
            mock_brain.search.return_value = [distractor]
            mock_brain.feedback_stats.return_value = (0, 1, -1)

            result = apply_latest_user_correction_feedback(
                [
                    AIMessage(content="Я пам'ятаю, що твою бабусю звати Марія."),
                    HumanMessage(content="Ні, ти помилився. Мою бабусю звати Ганна."),
                ],
                "desktop",
                session_log=[
                    {
                        "type": "factuality_analysis",
                        "evidence_record_ids": ["rec-evidence"],
                        "claims": [
                            {"supporting_record_ids": ["rec-evidence"]},
                        ],
                    }
                ],
            )

            assert len(result) == 1
            mock_brain.feedback.assert_called_once_with("rec-evidence", False)
            mock_brain.search.assert_not_called()
            assert result[0]["record_id"] == "rec-evidence"

    def test_marks_matching_records_negative_on_user_correction(self):
        from langchain_core.messages import AIMessage, HumanMessage

        with patch("remy.core.brain_tools.brain") as mock_brain, patch("remy.core.brain_tools.brain_lock"):
            record = MagicMock()
            record.id = "rec-1"
            record.content = "Бабусю звати Марія і вона народилася 1940 року."
            record.strength = 0.8
            mock_brain.search.return_value = [record]
            mock_brain.feedback_stats.return_value = (0, 1, -1)

            result = apply_latest_user_correction_feedback(
                [
                    AIMessage(content="Я пам'ятаю, що твою бабусю звати Марія."),
                    HumanMessage(content="Ні, ти помилився. Мою бабусю звати Ганна."),
                ],
                "desktop",
            )

            assert len(result) == 1
            mock_brain.feedback.assert_called_once_with("rec-1", False)
            assert result[0]["record_id"] == "rec-1"
            mock_brain.search.assert_called()

    def test_ignores_non_correction_follow_up(self):
        from langchain_core.messages import AIMessage, HumanMessage

        with patch("remy.core.brain_tools.brain") as mock_brain, patch("remy.core.brain_tools.brain_lock"):
            result = apply_latest_user_correction_feedback(
                [
                    AIMessage(content="Твоя бабуся народилася 1940 року."),
                    HumanMessage(content="Дякую, це корисно."),
                ],
                "desktop",
            )

            assert result == []
            mock_brain.feedback.assert_not_called()


class TestGetRecentFeedbackSummary:
    """Tests for get_recent_feedback_summary()."""

    def test_summary_with_verbose_signals(self):
        with patch("remy.core.brain_tools.brain") as mock_brain:
            rec1 = MagicMock()
            rec1.metadata = {"signal_type": "too_verbose"}
            rec2 = MagicMock()
            rec2.metadata = {"signal_type": "too_verbose"}
            mock_brain.search.return_value = [rec1, rec2]

            result = get_recent_feedback_summary()

            assert "shorter responses" in result.lower()

    def test_summary_with_repeat_question(self):
        with patch("remy.core.brain_tools.brain") as mock_brain:
            rec = MagicMock()
            rec.metadata = {"signal_type": "repeat_question"}
            mock_brain.search.return_value = [rec]

            result = get_recent_feedback_summary()

            assert "repeated" in result.lower() or "unclear" in result.lower()

    def test_summary_empty_when_no_signals(self):
        with patch("remy.core.brain_tools.brain") as mock_brain:
            mock_brain.search.return_value = []

            result = get_recent_feedback_summary()
            assert result == ""

    def test_summary_with_topic_switches(self):
        with patch("remy.core.brain_tools.brain") as mock_brain:
            recs = [MagicMock(metadata={"signal_type": "topic_switch"}) for _ in range(3)]
            mock_brain.search.return_value = recs

            result = get_recent_feedback_summary()

            assert "focused" in result.lower() or "brief" in result.lower()

    def test_summary_with_user_correction(self):
        with patch("remy.core.brain_tools.brain") as mock_brain:
            rec = MagicMock()
            rec.metadata = {"signal_type": "user_correction"}
            mock_brain.search.return_value = [rec]

            result = get_recent_feedback_summary()

            assert "corrected" in result.lower() or "careful" in result.lower()
