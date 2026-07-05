"""
Failure Prediction & Pre-Flight Analysis (AUTON-6).

Before executing an autonomous cycle, run a quick programmatic check:
1. Are required tools healthy?
2. Is there enough budget for this goal type?
3. How did similar goals fare historically?
4. Should the goal be decomposed or skipped?

All checks are zero-LLM — pure Python heuristics from outcome history.
"""

import logging
from dataclasses import dataclass

from remy.core.agent_tools import brain

logger = logging.getLogger("Autonomy.Preflight")


@dataclass
class PreflightResult:
    """Result of pre-flight analysis."""

    can_proceed: bool = True
    predicted_success: float = 0.5
    difficulty: str = "medium"  # easy / medium / hard
    warnings: list[str] = None
    suggestion: str = ""  # "proceed" / "decompose" / "skip" / "change_role"

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


def run_preflight(
    goal_description: str,
    goal_attempts: int = 0,
    budget_tokens_remaining: int = 100_000,
    tool_health_report: dict[str, str] | None = None,
) -> PreflightResult:
    """Run pre-flight checks on a goal before attempting it.

    Returns PreflightResult with prediction, warnings, and suggestion.
    """
    result = PreflightResult()
    warnings = []

    # 1. Budget check
    budget_warning = _check_budget(goal_description, budget_tokens_remaining)
    if budget_warning:
        warnings.append(budget_warning)

    # 2. Tool health check
    tool_warnings = _check_tool_health(goal_description, tool_health_report or {})
    warnings.extend(tool_warnings)

    # 3. Historical success prediction
    prediction = _predict_success(goal_description, goal_attempts)
    result.predicted_success = prediction

    # 4. Difficulty estimation
    result.difficulty = _estimate_difficulty(goal_description, goal_attempts, prediction)

    # 5. Decide suggestion
    if prediction < 0.2 and goal_attempts >= 3:
        result.suggestion = "skip"
        warnings.append(
            f"Very low success prediction ({prediction:.0%}) after {goal_attempts} attempts. "
            "Consider skipping this goal."
        )
    elif prediction < 0.3:
        result.suggestion = "decompose"
        warnings.append(
            f"Low success prediction ({prediction:.0%}). "
            "Consider decomposing into smaller sub-goals."
        )
    else:
        result.suggestion = "proceed"

    # Budget too low → can't proceed
    if budget_tokens_remaining < 500:
        result.can_proceed = False
        result.suggestion = "skip"
        warnings.append("Insufficient budget for any action.")

    # Unavailable critical tools → can't proceed
    if any("UNAVAILABLE" in w for w in tool_warnings):
        result.can_proceed = False

    result.warnings = warnings
    return result


def _check_budget(goal_description: str, tokens_remaining: int) -> str:
    """Check if budget is sufficient for the expected goal cost."""
    desc_lower = goal_description.lower()

    # Research goals typically cost more
    if any(kw in desc_lower for kw in ("research", "investigate", "study")):
        if tokens_remaining < 3000:
            return "Budget low for research goal (needs ~3000+ tokens)"
    elif any(kw in desc_lower for kw in ("browse", "web", "http")):
        if tokens_remaining < 2000:
            return "Budget low for web browsing goal (needs ~2000+ tokens)"
    else:
        if tokens_remaining < 1000:
            return "Budget running low (<1000 tokens remaining)"

    return ""


def _check_tool_health(
    goal_description: str,
    health_report: dict[str, str],
) -> list[str]:
    """Check if tools needed for this goal are healthy."""
    warnings = []
    desc_lower = goal_description.lower()

    # Map goal keywords to required tools
    tool_needs: dict[str, list[str]] = {
        "research": ["web_search", "recall"],
        "browse": ["browse_page", "browser_act"],
        "write": ["write_file"],
        "read": ["read_file"],
        "store": ["store"],
        "metric": ["metric_summary", "track_metric"],
    }

    for keyword, tools in tool_needs.items():
        if keyword in desc_lower:
            for tool in tools:
                status = health_report.get(tool, "")
                if status == "open":
                    warnings.append(
                        f"UNAVAILABLE: Tool '{tool}' circuit is open (recent failures). "
                        f"Goal may fail."
                    )
                elif status == "half_open":
                    warnings.append(f"WARNING: Tool '{tool}' is recovering from failures.")

    return warnings


def _predict_success(goal_description: str, attempts: int = 0) -> float:
    """Predict success probability from historical outcomes.

    Uses simple heuristics from past outcome records with similar goal types.
    """
    # Attempt penalty: each failed attempt reduces prediction
    base_prediction = 0.6
    attempt_penalty = min(attempts * 0.1, 0.4)

    # Check historical outcomes for similar goal types
    try:
        from remy.core.agent_tools import brain_lock

        with brain_lock:
            outcomes = brain.search(
                query=goal_description[:80],
                tags=["autonomous-outcome"],
                limit=10,
            )

        if not outcomes:
            return max(0.1, base_prediction - attempt_penalty)

        successes = 0
        total = 0
        for r in outcomes:
            tags = getattr(r, "tags", None) or []
            total += 1
            if "outcome-success" in tags:
                successes += 1

        if total >= 3:
            historical_rate = successes / total
            # Blend historical rate with base prediction
            prediction = (historical_rate * 0.7) + (base_prediction * 0.3)
        else:
            prediction = base_prediction

        return max(0.05, prediction - attempt_penalty)

    except Exception:
        return max(0.1, base_prediction - attempt_penalty)


def _estimate_difficulty(
    goal_description: str,
    attempts: int,
    predicted_success: float,
) -> str:
    """Classify goal difficulty as easy/medium/hard."""
    if predicted_success >= 0.7 and attempts <= 1:
        return "easy"
    elif predicted_success < 0.3 or attempts >= 5:
        return "hard"
    return "medium"


def format_preflight_for_prompt(result: PreflightResult) -> str:
    """Format preflight result for injection into the decision prompt."""
    if not result.warnings and result.suggestion == "proceed":
        return ""

    lines = [
        f"\nPRE-FLIGHT ANALYSIS (predicted success: {result.predicted_success:.0%}, "
        f"difficulty: {result.difficulty}):"
    ]

    for w in result.warnings:
        lines.append(f"  ⚠ {w}")

    if result.suggestion == "decompose":
        lines.append("  → RECOMMENDATION: Decompose this goal into smaller steps.")
    elif result.suggestion == "skip":
        lines.append("  → RECOMMENDATION: Skip this goal and move to the next one.")

    return "\n".join(lines) + "\n"
