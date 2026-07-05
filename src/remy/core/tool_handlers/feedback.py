"""
Feedback Signal Handlers — implicit feedback detection and storage.

Detects conversational patterns (verbosity, topic switching, repeated questions)
and stores them as feedback signals for behavioral adaptation.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger("BrainTools")


def _get_brain():
    """Lazy accessor — reads brain from brain_tools (supports test patching)."""
    import remy.core.brain_tools as _bt

    return _bt.brain


FEEDBACK_TAG = "feedback-signal"

_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "i",
        "you",
        "my",
        "your",
        "what",
        "how",
        "to",
        "and",
        "of",
        "in",
        "for",
        "it",
        "that",
        "this",
        "with",
        "on",
        "at",
        "be",
        "do",
        "have",
        "not",
        "but",
        "or",
        "so",
        "if",
        "me",
        "we",
        "he",
        "she",
        "they",
        "can",
        "will",
        "just",
        "я",
        "ти",
        "ви",
        "це",
        "що",
        "як",
        "та",
        "і",
        "не",
        "в",
        "на",
        "з",
    }
)


@dataclass
class FeedbackSignal:
    """An implicit feedback signal detected from conversation patterns."""

    signal_type: str  # "too_verbose" | "topic_switch" | "repeat_question"
    severity: float  # 0.0-1.0
    context: str  # What triggered it (max 200 chars)
    channel: str
    timestamp: str


def detect_feedback_signals(messages: list, channel: str) -> list[FeedbackSignal]:
    """Detect implicit feedback from conversation patterns. Zero LLM calls."""
    from langchain_core.messages import AIMessage, HumanMessage

    signals: list[FeedbackSignal] = []
    if len(messages) < 3:
        return signals

    now = datetime.now().isoformat()

    # Scan (AI response, next user message) pairs
    for i in range(len(messages) - 1):
        ai_msg = messages[i]
        user_msg = messages[i + 1]
        if not isinstance(ai_msg, AIMessage) or not isinstance(user_msg, HumanMessage):
            continue

        ai_text = ai_msg.content if isinstance(ai_msg.content, str) else ""
        user_text = user_msg.content if isinstance(user_msg.content, str) else ""
        if not ai_text or not user_text:
            continue

        ai_words = len(ai_text.split())
        user_words = len(user_text.split())

        # Signal 1: Verbosity — long AI response followed by very short user reply
        if ai_words > 150 and user_words < 5:
            signals.append(
                FeedbackSignal(
                    signal_type="too_verbose",
                    severity=min(1.0, ai_words / 300),
                    context=f"AI: {ai_words}w -> User: '{user_text[:50]}'",
                    channel=channel,
                    timestamp=now,
                )
            )

        # Signal 2: Topic switch — zero content-word overlap
        ai_last = ai_text.split(".")[-2] if ai_text.count(".") >= 2 else ai_text[-200:]
        ai_content = (
            {w.strip(".,!?;:\"'()[]") for w in ai_last.lower().split()} - _STOP_WORDS - {""}
        )
        user_content = (
            {w.strip(".,!?;:\"'()[]") for w in user_text.lower().split()} - _STOP_WORDS - {""}
        )
        if len(ai_content) > 3 and len(user_content) > 2:
            if not (ai_content & user_content):
                signals.append(
                    FeedbackSignal(
                        signal_type="topic_switch",
                        severity=0.6,
                        context=f"AI: '{ai_last[:50]}' -> User: '{user_text[:50]}'",
                        channel=channel,
                        timestamp=now,
                    )
                )

    # Signal 3: Repeat question — user asks similar thing as before
    def _content_words(text: str) -> set[str]:
        """Extract content words: lowercase, strip punctuation, remove stop words."""
        return {w.strip(".,!?;:\"'()[]") for w in text.lower().split()} - _STOP_WORDS - {""}

    from langchain_core.messages import HumanMessage as HM

    user_msgs = [m for m in messages if isinstance(m, HM)]
    if len(user_msgs) >= 2:
        latest = user_msgs[-1].content if isinstance(user_msgs[-1].content, str) else ""
        latest_words = _content_words(latest)
        if len(latest_words) >= 3:
            for earlier in user_msgs[:-1]:
                earlier_text = earlier.content if isinstance(earlier.content, str) else ""
                earlier_words = _content_words(earlier_text)
                if len(earlier_words) >= 3:
                    overlap = latest_words & earlier_words
                    ratio = len(overlap) / min(len(latest_words), len(earlier_words))
                    if ratio > 0.6:
                        signals.append(
                            FeedbackSignal(
                                signal_type="repeat_question",
                                severity=ratio,
                                context=f"Repeated: '{latest[:50]}' ~ '{earlier_text[:50]}'",
                                channel=channel,
                                timestamp=now,
                            )
                        )
                        break  # One repeat signal per turn

    return signals


def store_feedback_signal(signal: FeedbackSignal) -> None:
    """Store a feedback signal in brain for behavioral adaptation."""
    try:
        from remy.core.agent_tools import Level, brain_lock
        from remy.core.provenance import _stamp_provenance

        tags = [FEEDBACK_TAG, signal.signal_type]
        with brain_lock:
            _get_brain().store(
                content=f"Feedback [{signal.signal_type}]: {signal.context}",
                level=Level.WORKING,
                tags=tags,
                metadata=_stamp_provenance(
                    {
                        "type": "feedback_signal",
                        "signal_type": signal.signal_type,
                        "severity": signal.severity,
                        "channel": signal.channel,
                        "timestamp": signal.timestamp,
                    },
                    signal.channel,
                    tags=tags,
                ),
                deduplicate=False,
            )
    except Exception as e:
        logger.warning("Failed to store feedback signal: %s", e)


def get_recent_feedback_summary(limit: int = 10) -> str:
    """Aggregate recent feedback signals into behavioral hints. Zero LLM calls."""
    try:
        records = _get_brain().search(query="", tags=[FEEDBACK_TAG], limit=limit)
        if not records:
            return ""

        counts: dict[str, int] = {}
        for r in records:
            meta = getattr(r, "metadata", None) or {}
            stype = meta.get("signal_type", "unknown")
            counts[stype] = counts.get(stype, 0) + 1

        hints = []
        if counts.get("too_verbose", 0) >= 2:
            hints.append("User prefers shorter responses. Be more concise.")
        if counts.get("topic_switch", 0) >= 2:
            hints.append("User frequently switches topics — stay focused on the current topic and keep replies brief. Don't bring up old topics unless asked.")
        if counts.get("repeat_question", 0) >= 1:
            hints.append("User has repeated questions. Previous answers may have been unclear.")

        return "\n".join(hints)
    except Exception:
        return ""
