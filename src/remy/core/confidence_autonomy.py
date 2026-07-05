"""
Confidence-Based Autonomy Levels (AUTON-15).

Gradient autonomy: confidence score 0.0-1.0 determines action behavior.
- > 0.8: execute silently
- 0.5-0.8: execute + notify user
- 0.3-0.5: request guidance (AUTON-3)
- < 0.3: skip or escalate

Domain-specific confidence tracks familiarity from past outcomes.
User trust calibration adjusts thresholds over time.
"""

import logging
import time
from dataclasses import dataclass

from remy.core.event_bus import event_bus

logger = logging.getLogger("Autonomy.Confidence")


# ============== Autonomy Actions ==============


class AutonomyAction:
    """What the agent should do based on confidence level."""

    EXECUTE_SILENT = "execute_silent"  # > 0.8
    EXECUTE_NOTIFY = "execute_notify"  # 0.5-0.8
    REQUEST_GUIDANCE = "request_guidance"  # 0.3-0.5
    SKIP = "skip"  # < 0.3


# Default thresholds — can be calibrated
_THRESHOLDS = {
    "silent": 0.8,
    "notify": 0.5,
    "guidance": 0.3,
}


def get_autonomy_action(confidence: float) -> str:
    """Map confidence score to autonomy action."""
    if confidence >= _THRESHOLDS["silent"]:
        return AutonomyAction.EXECUTE_SILENT
    elif confidence >= _THRESHOLDS["notify"]:
        return AutonomyAction.EXECUTE_NOTIFY
    elif confidence >= _THRESHOLDS["guidance"]:
        return AutonomyAction.REQUEST_GUIDANCE
    else:
        return AutonomyAction.SKIP


# ============== Confidence Scoring ==============


@dataclass
class ConfidenceFactors:
    """Factors that contribute to confidence score."""

    domain_familiarity: float = 0.5  # 0-1 based on past success in this domain
    tool_reliability: float = 1.0  # 0-1 based on tool health
    goal_clarity: float = 0.5  # 0-1 based on goal description quality
    budget_health: float = 1.0  # 0-1 based on remaining budget
    recent_success_rate: float = 0.5  # 0-1 based on recent outcomes


def compute_confidence(factors: ConfidenceFactors) -> float:
    """Compute overall confidence from factors.

    Weighted average with domain familiarity having the highest weight.
    """
    weights = {
        "domain_familiarity": 0.35,
        "tool_reliability": 0.20,
        "goal_clarity": 0.15,
        "budget_health": 0.10,
        "recent_success_rate": 0.20,
    }

    score = (
        factors.domain_familiarity * weights["domain_familiarity"]
        + factors.tool_reliability * weights["tool_reliability"]
        + factors.goal_clarity * weights["goal_clarity"]
        + factors.budget_health * weights["budget_health"]
        + factors.recent_success_rate * weights["recent_success_rate"]
    )

    return max(0.0, min(1.0, score))


# ============== Domain-Specific Confidence ==============


# Domain -> {successes, failures, last_updated}
_domain_stats: dict[str, dict] = {}


_ACTION_DOMAINS = {
    "research": {"research", "investigate", "analyze", "study", "explore", "learn"},
    "file_ops": {"read", "write", "list", "file", "directory", "create"},
    "memory": {"recall", "store", "remember", "record", "search"},
    "web": {"browse", "search", "web", "http", "url", "page", "fetch"},
    "planning": {"plan", "organize", "schedule", "goal", "prioritize"},
    "communication": {"notify", "message", "send", "telegram", "report"},
}


def infer_domain(action_description: str) -> str:
    """Infer action domain from description."""
    text = action_description.lower()
    words = set(text.split())

    best_domain = "general"
    best_score = 0

    for domain, keywords in _ACTION_DOMAINS.items():
        score = len(words & keywords)
        if score > best_score:
            best_score = score
            best_domain = domain

    return best_domain


def get_domain_confidence(domain: str) -> float:
    """Get confidence for a specific domain based on past outcomes."""
    stats = _domain_stats.get(domain)
    if not stats:
        return 0.5  # Unknown domain — neutral confidence

    total = stats.get("successes", 0) + stats.get("failures", 0)
    if total == 0:
        return 0.5

    success_rate = stats["successes"] / total

    # Apply recency weighting — older stats contribute less
    age_hours = (time.time() - stats.get("last_updated", time.time())) / 3600
    recency_factor = max(0.5, 1.0 - (age_hours / 168))  # Decays over a week

    return success_rate * recency_factor


def record_domain_outcome(domain: str, success: bool):
    """Record an outcome for a domain."""
    if domain not in _domain_stats:
        _domain_stats[domain] = {"successes": 0, "failures": 0, "last_updated": time.time()}

    if success:
        _domain_stats[domain]["successes"] += 1
    else:
        _domain_stats[domain]["failures"] += 1

    _domain_stats[domain]["last_updated"] = time.time()


# ============== User Trust Calibration ==============


_user_trust_adjustments: dict = {
    "approvals": 0,
    "rejections": 0,
    "threshold_offset": 0.0,
}


def record_user_decision(approved: bool):
    """Track user approval/rejection to calibrate thresholds."""
    if approved:
        _user_trust_adjustments["approvals"] += 1
    else:
        _user_trust_adjustments["rejections"] += 1

    total = _user_trust_adjustments["approvals"] + _user_trust_adjustments["rejections"]
    if total >= 5:
        approval_rate = _user_trust_adjustments["approvals"] / total
        # More approvals → lower thresholds (more autonomous)
        # More rejections → higher thresholds (more cautious)
        _user_trust_adjustments["threshold_offset"] = (approval_rate - 0.5) * 0.1


def get_calibrated_thresholds() -> dict:
    """Get thresholds adjusted by user trust calibration."""
    offset = _user_trust_adjustments["threshold_offset"]
    return {
        "silent": max(0.5, min(0.95, _THRESHOLDS["silent"] - offset)),
        "notify": max(0.3, min(0.7, _THRESHOLDS["notify"] - offset)),
        "guidance": max(0.1, min(0.5, _THRESHOLDS["guidance"] - offset)),
    }


# ============== Full Confidence Assessment ==============


def assess_action_confidence(
    action_description: str,
    budget_pct: float = 100.0,
    tool_health_issues: int = 0,
    recent_successes: int = 0,
    recent_failures: int = 0,
) -> tuple[float, str]:
    """Full confidence assessment for an action.

    Returns (confidence_score, recommended_action).
    """
    domain = infer_domain(action_description)

    # Build factors
    total_recent = recent_successes + recent_failures
    recent_rate = recent_successes / total_recent if total_recent > 0 else 0.5

    factors = ConfidenceFactors(
        domain_familiarity=get_domain_confidence(domain),
        tool_reliability=max(0.0, 1.0 - (tool_health_issues * 0.3)),
        goal_clarity=_assess_goal_clarity(action_description),
        budget_health=min(1.0, budget_pct / 100.0),
        recent_success_rate=recent_rate,
    )

    confidence = compute_confidence(factors)

    # Apply user trust calibration
    calibrated = get_calibrated_thresholds()
    if confidence >= calibrated["silent"]:
        action = AutonomyAction.EXECUTE_SILENT
    elif confidence >= calibrated["notify"]:
        action = AutonomyAction.EXECUTE_NOTIFY
    elif confidence >= calibrated["guidance"]:
        action = AutonomyAction.REQUEST_GUIDANCE
    else:
        action = AutonomyAction.SKIP

    event_bus.emit(
        "confidence_assessed",
        {
            "domain": domain,
            "confidence": round(confidence, 2),
            "action": action,
        },
    )

    return confidence, action


def _assess_goal_clarity(description: str) -> float:
    """Assess how clear/specific a goal description is."""
    if not description:
        return 0.1

    score = 0.3  # baseline

    # Longer descriptions are clearer
    if len(description) > 100:
        score += 0.2
    elif len(description) > 50:
        score += 0.1

    # Specific words suggest clarity
    specifics = {"use", "with", "from", "to", "about", "for", "using", "via"}
    words = set(description.lower().split())
    if words & specifics:
        score += 0.2

    # Action verbs suggest clear intent
    verbs = {"search", "find", "create", "update", "read", "write", "analyze", "compare"}
    if words & verbs:
        score += 0.2

    return min(1.0, score)


def format_confidence_info(confidence: float, action: str, domain: str = "") -> str:
    """Format confidence assessment for prompt or display."""
    labels = {
        AutonomyAction.EXECUTE_SILENT: "HIGH — executing silently",
        AutonomyAction.EXECUTE_NOTIFY: "MODERATE — execute + notify",
        AutonomyAction.REQUEST_GUIDANCE: "LOW — requesting guidance",
        AutonomyAction.SKIP: "VERY LOW — skipping",
    }
    label = labels.get(action, action)
    domain_text = f" (domain: {domain})" if domain else ""
    return f"Confidence: {confidence:.0%} — {label}{domain_text}"


def reset_domain_stats():
    """Reset domain stats (for testing)."""
    _domain_stats.clear()


def reset_user_trust():
    """Reset user trust calibration (for testing)."""
    _user_trust_adjustments["approvals"] = 0
    _user_trust_adjustments["rejections"] = 0
    _user_trust_adjustments["threshold_offset"] = 0.0
