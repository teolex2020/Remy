"""
Dynamic Context Window & State Summarization (AUTON-12).

Adaptive context sizing based on goal complexity + importance scoring
for message retention + periodic state snapshots for long sessions.
"""

import logging
import re
import time
from dataclasses import dataclass, field

logger = logging.getLogger("Autonomy.ContextWindow")


# ============== Complexity-Based Context Sizing ==============


# Complexity indicators in goal text
_COMPLEX_KEYWORDS = {
    "research",
    "analyze",
    "compare",
    "investigate",
    "plan",
    "design",
    "implement",
    "build",
    "create",
    "develop",
    "integrate",
    "migrate",
    "refactor",
    "debug",
    "troubleshoot",
    "optimize",
    "multi-step",
    "sequential",
    "parallel",
}

_SIMPLE_KEYWORDS = {
    "check",
    "verify",
    "list",
    "count",
    "read",
    "recall",
    "get",
    "status",
    "show",
    "display",
    "report",
    "summary",
}

# Context size ranges
# v2.4: Reduced MAX from 48→28 — scratchpad + session context provide working memory
_MIN_CONTEXT = 12  # Simple read-only tasks
_DEFAULT_CONTEXT = 16
_MEDIUM_CONTEXT = 22
_MAX_CONTEXT = 28  # Complex multi-step tasks (was 48, scratchpad compensates)


def estimate_complexity(goal_description: str, attempts: int = 0) -> float:
    """Estimate goal complexity on 0.0-1.0 scale.

    Considers: keyword complexity, text length, sub-step mentions, attempt count.
    """
    if not goal_description:
        return 0.3

    text = goal_description.lower()
    words = set(text.split())

    score = 0.3  # baseline

    # Complex keyword bonus
    complex_matches = words & _COMPLEX_KEYWORDS
    score += min(0.3, len(complex_matches) * 0.1)

    # Simple keyword discount
    simple_matches = words & _SIMPLE_KEYWORDS
    score -= min(0.2, len(simple_matches) * 0.1)

    # Long description suggests complexity
    if len(goal_description) > 200:
        score += 0.1
    elif len(goal_description) > 100:
        score += 0.05

    # Numbered steps suggest multi-step
    if re.search(r"\d+\.\s", goal_description):
        score += 0.15

    # Many attempts means it's harder than expected
    if attempts >= 3:
        score += 0.15
    elif attempts >= 1:
        score += 0.05

    return max(0.0, min(1.0, score))


def context_size_for_complexity(complexity: float) -> int:
    """Map complexity score to context window size.

    0.0-0.3: simple → small context (12-16)
    0.3-0.6: moderate → medium context (16-28)
    0.6-1.0: complex → large context (28-48)
    """
    if complexity <= 0.3:
        return _MIN_CONTEXT + int((complexity / 0.3) * (_DEFAULT_CONTEXT - _MIN_CONTEXT))
    elif complexity <= 0.6:
        ratio = (complexity - 0.3) / 0.3
        return _DEFAULT_CONTEXT + int(ratio * (_MEDIUM_CONTEXT - _DEFAULT_CONTEXT))
    else:
        ratio = (complexity - 0.6) / 0.4
        return _MEDIUM_CONTEXT + int(ratio * (_MAX_CONTEXT - _MEDIUM_CONTEXT))


def dynamic_keep_recent(
    channel: str,
    goal_description: str = "",
    goal_attempts: int = 0,
) -> int:
    """Calculate dynamic keep_recent value based on channel + goal complexity.

    Replaces the static _estimate_keep_recent for autonomous channel.
    """
    if channel != "autonomous":
        # Non-autonomous channels keep existing behavior
        if any(
            kw in goal_description.lower()
            for kw in ("research", "browse", "investigate", "analyze")
        ):
            return 32
        return 16

    complexity = estimate_complexity(goal_description, goal_attempts)
    return context_size_for_complexity(complexity)


# ============== Message Importance Scoring ==============


def score_message_importance(msg) -> float:
    """Score a message's importance for context retention (0.0-1.0).

    Higher-importance messages are kept over lower ones during compaction.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    # System messages are always critical
    if isinstance(msg, SystemMessage):
        return 1.0

    # Human messages — user input is important
    if isinstance(msg, HumanMessage):
        content = msg.content if isinstance(msg.content, str) else ""
        if len(content) > 100:
            return 0.9  # Long user messages are more important
        return 0.7

    # AI messages with tool calls — decision points
    if isinstance(msg, AIMessage):
        if getattr(msg, "tool_calls", None):
            return 0.6  # Tool decision matters
        content = msg.content if isinstance(msg.content, str) else ""
        if content.strip():
            return 0.5  # AI text responses
        return 0.2  # Empty AI messages (just tool calls with no text)

    # Tool messages — results vary in importance
    if isinstance(msg, ToolMessage):
        content = msg.content if isinstance(msg.content, str) else ""
        # Error results are important (need to avoid repeating)
        if "error" in content.lower()[:100]:
            return 0.8
        # Short results are less valuable than long ones
        if len(content) < 50:
            return 0.3
        return 0.4

    return 0.3  # Default


def select_important_messages(
    messages: list,
    budget: int,
    always_keep_recent: int = 6,
) -> list:
    """Select the most important messages within a budget.

    Always keeps the most recent `always_keep_recent` messages,
    then fills remaining budget from oldest messages scored by importance.
    """
    if len(messages) <= budget:
        return messages

    # Always keep recent messages
    recent = messages[-always_keep_recent:] if always_keep_recent < len(messages) else messages
    old = messages[:-always_keep_recent] if always_keep_recent < len(messages) else []

    if not old:
        return recent

    remaining_budget = budget - len(recent)
    if remaining_budget <= 0:
        return recent

    # Score and sort old messages by importance
    scored = [(i, score_message_importance(msg)) for i, msg in enumerate(old)]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Select top important messages, but maintain original order
    selected_indices = sorted([idx for idx, _ in scored[:remaining_budget]])
    selected = [old[i] for i in selected_indices]

    return selected + recent


# ============== State Summarization ==============


@dataclass
class SessionState:
    """Compressed state snapshot for long-running sessions."""

    current_goal: str = ""
    progress: str = ""
    key_findings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    actions_taken: int = 0
    timestamp: float = 0.0

    def to_text(self) -> str:
        """Format as compact text for system message injection."""
        lines = [f"SESSION STATE (actions: {self.actions_taken}):"]
        if self.current_goal:
            lines.append(f"  Goal: {self.current_goal}")
        if self.progress:
            lines.append(f"  Progress: {self.progress}")
        if self.key_findings:
            lines.append("  Findings: " + "; ".join(self.key_findings[:5]))
        if self.blockers:
            lines.append("  Blockers: " + "; ".join(self.blockers[:3]))
        return "\n".join(lines)


_session_states: dict[str, SessionState] = {}
_STATE_SNAPSHOT_INTERVAL = 10  # Take snapshot every N actions


def update_session_state(
    session_id: str,
    goal: str = "",
    action_result: str = "",
    success: bool = True,
) -> SessionState:
    """Update session state with latest action result.

    Called after each autonomous action.
    """
    state = _session_states.get(session_id, SessionState())

    if goal:
        state.current_goal = goal[:200]

    state.actions_taken += 1
    state.timestamp = time.time()

    # Extract findings from successful results
    if success and action_result:
        # Keep first sentence as finding
        first_sentence = action_result.split(".")[0][:150]
        if first_sentence and len(first_sentence) > 20:
            state.key_findings.append(first_sentence)
            # Cap findings
            if len(state.key_findings) > 10:
                state.key_findings = state.key_findings[-10:]

    # Track blockers from failures
    if not success and action_result:
        blocker = action_result[:100]
        state.blockers.append(blocker)
        if len(state.blockers) > 5:
            state.blockers = state.blockers[-5:]

    _session_states[session_id] = state
    return state


def get_state_summary(session_id: str) -> str | None:
    """Get state summary if enough actions have accumulated.

    Returns compact text suitable for system message, or None.
    """
    state = _session_states.get(session_id)
    if not state:
        return None

    if state.actions_taken < _STATE_SNAPSHOT_INTERVAL:
        return None

    return state.to_text()


def should_inject_state(session_id: str, action_count: int) -> bool:
    """Whether to inject state summary into context."""
    return action_count > 0 and action_count % _STATE_SNAPSHOT_INTERVAL == 0


def clear_session_state(session_id: str):
    """Clear state for a session (on session end)."""
    _session_states.pop(session_id, None)
