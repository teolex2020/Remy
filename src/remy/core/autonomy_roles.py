"""
Adaptive Role Switching (AUTON-4) — performance-based role selection.

Tracks role effectiveness per goal type and selects the best role based on
historical success rates instead of simple keyword matching.

Key features:
1. Record role performance after each cycle
2. Select roles by historical success rate for similar goal types
3. Fallback chain when a role underperforms
4. Role switch hint after consecutive in-role failures
"""

import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime

from remy.core.agent_tools import brain
from remy.core.autonomy_models import AGENT_ROLES, AgentRole

logger = logging.getLogger("Autonomy.Roles")

ROLE_PERF_TAGS = ["role-performance"]

# Minimum data points before trusting performance stats
MIN_SAMPLES_FOR_STATS = 3

# Fallback chains: if primary role fails, try these in order
ROLE_FALLBACK_CHAINS: dict[str, list[str]] = {
    "researcher": ["osint", "analyst", "executor", "planner"],
    "planner": ["researcher", "analyst", "executor"],
    "executor": ["planner", "researcher", "analyst"],
    "analyst": ["researcher", "osint", "planner", "executor"],
    "osint": ["researcher", "analyst", "executor", "planner"],
}


# Failure reasons that indicate environment problems (not role incompetence).
# These should NOT count against a role's success rate.
_ENVIRONMENTAL_FAILURES = frozenset(
    {
        "timeout",
        "network_error",
        "rate_limit",
        "connection_error",
        "dns_error",
        "service_unavailable",
        "auth_error",
    }
)


def record_role_performance(
    role_name: str,
    goal_type: str,
    success: bool,
    tokens_used: int = 0,
    duration_ms: int = 0,
    failure_reason: str = "",
) -> None:
    """Record a role's performance for a goal type.

    Stored as brain records for long-term learning. Lightweight — no LLM call.

    If failure_reason matches an environmental error (timeout, network, etc.),
    the record is stored but marked as `env_failure=True` so it doesn't
    pollute role success rate calculations.
    """
    is_env_failure = (
        not success
        and failure_reason
        and _classify_failure_reason(failure_reason) in _ENVIRONMENTAL_FAILURES
    )

    try:
        from remy.core.agent_tools import Level

        uid = uuid.uuid4().hex[:8]
        content = (
            f"Role '{role_name}' on '{goal_type}': "
            f"{'success' if success else 'failure'}"
            f"{' [env]' if is_env_failure else ''} "
            f"({tokens_used} tokens, {duration_ms}ms) [{uid}]"
        )

        brain.store(
            content=content,
            level=Level.WORKING,
            tags=ROLE_PERF_TAGS,
            metadata={
                "role": role_name,
                "goal_type": goal_type,
                "success": success,
                "tokens_used": tokens_used,
                "duration_ms": duration_ms,
                "env_failure": is_env_failure,
                "failure_reason": failure_reason[:100] if failure_reason else "",
                "timestamp": datetime.now().isoformat(),
            },
        )
    except Exception as e:
        logger.debug("Failed to record role performance: %s", e)


def _classify_failure_reason(reason: str) -> str:
    """Classify a failure reason into a canonical category."""
    r = reason.lower()
    if any(kw in r for kw in ("timeout", "timed out", "deadline")):
        return "timeout"
    if any(kw in r for kw in ("network", "connection", "unreachable", "socket")):
        return "network_error"
    if any(kw in r for kw in ("rate limit", "429", "too many requests", "quota")):
        return "rate_limit"
    if any(kw in r for kw in ("dns",)):
        return "dns_error"
    if any(kw in r for kw in ("503", "502", "504", "service unavailable")):
        return "service_unavailable"
    if any(kw in r for kw in ("401", "403", "unauthorized", "forbidden")):
        return "auth_error"
    return reason[:50]


def get_role_stats(limit: int = 100) -> dict[str, dict[str, dict]]:
    """Get aggregated role performance stats.

    Returns:
        {
            "researcher": {
                "research": {"attempts": 5, "successes": 4, "rate": 0.8, "avg_tokens": 500},
                ...
            },
            ...
        }
    """
    try:
        from remy.core.agent_tools import brain_lock

        with brain_lock:
            records = brain.search(query="", tags=ROLE_PERF_TAGS, limit=limit)
    except Exception:
        return {}

    stats: dict[str, dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"attempts": 0, "successes": 0, "total_tokens": 0})
    )

    for r in records:
        meta = getattr(r, "metadata", None) or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                continue

        role = meta.get("role", "")
        goal_type = meta.get("goal_type", "unknown")
        if not role:
            continue

        # Skip environmental failures — they don't reflect role competence
        _env = meta.get("env_failure", False)
        if _env is True or str(_env).lower() == "true":
            continue

        entry = stats[role][goal_type]
        entry["attempts"] += 1
        _succ = meta.get("success", False)
        if _succ is True or str(_succ).lower() == "true":
            entry["successes"] += 1
        entry["total_tokens"] += int(meta.get("tokens_used", 0))

    # Compute rates
    result = {}
    for role_name, goal_types in stats.items():
        result[role_name] = {}
        for goal_type, data in goal_types.items():
            attempts = data["attempts"]
            result[role_name][goal_type] = {
                "attempts": attempts,
                "successes": data["successes"],
                "rate": data["successes"] / attempts if attempts > 0 else 0.0,
                "avg_tokens": data["total_tokens"] // attempts if attempts > 0 else 0,
            }

    return result


def select_best_role(
    goal_description: str,
    goal_type: str = "",
    current_failures: int = 0,
    current_role_name: str = "",
) -> AgentRole:
    """Select the best role for a goal, using performance history + keyword fallback.

    Strategy:
    1. If we have enough performance data for this goal_type, pick the
       highest success rate role.
    2. If current role has failed consecutively, try fallback chain.
    3. Fall back to keyword matching if no data.
    """
    # If the current role keeps failing, try fallback chain
    if current_role_name and current_failures >= 2:
        fallback = _get_fallback_role(current_role_name, goal_type)
        if fallback:
            logger.info(
                "Role fallback: %s → %s (after %d failures)",
                current_role_name,
                fallback.name,
                current_failures,
            )
            return fallback

    # Try performance-based selection
    stats = get_role_stats()
    if stats and goal_type:
        best_role = _select_by_performance(stats, goal_type)
        if best_role:
            return best_role

    # Fall back to keyword matching (existing logic)
    return _select_by_keywords(goal_description)


def _select_by_performance(
    stats: dict[str, dict[str, dict]],
    goal_type: str,
) -> AgentRole | None:
    """Pick the role with the highest success rate for a goal type.

    Only considers roles with MIN_SAMPLES_FOR_STATS+ attempts.
    """
    candidates: list[tuple[str, float, int]] = []

    for role_name, goal_types in stats.items():
        if role_name not in AGENT_ROLES:
            continue
        data = goal_types.get(goal_type)
        if not data or data["attempts"] < MIN_SAMPLES_FOR_STATS:
            continue
        candidates.append((role_name, data["rate"], data["avg_tokens"]))

    if not candidates:
        return None

    # Sort by success rate (descending), then by avg_tokens (ascending = cheaper)
    candidates.sort(key=lambda c: (-c[1], c[2]))

    best_name = candidates[0][0]
    best_rate = candidates[0][1]

    if best_rate > 0:
        logger.info(
            "Performance-based role: %s (%.0f%% success for '%s')",
            best_name,
            best_rate * 100,
            goal_type,
        )
        return AGENT_ROLES[best_name]

    return None


def _get_fallback_role(
    current_role: str,
    goal_type: str = "",
) -> AgentRole | None:
    """Get next role from fallback chain, preferring roles with good stats."""
    chain = ROLE_FALLBACK_CHAINS.get(current_role, [])
    if not chain:
        return None

    # If we have stats, pick the fallback with best performance
    stats = get_role_stats()
    if stats and goal_type:
        best_fallback = None
        best_rate = -1.0
        for role_name in chain:
            if role_name not in AGENT_ROLES:
                continue
            data = stats.get(role_name, {}).get(goal_type)
            if data and data["attempts"] >= MIN_SAMPLES_FOR_STATS:
                if data["rate"] > best_rate:
                    best_rate = data["rate"]
                    best_fallback = role_name

        if best_fallback:
            return AGENT_ROLES[best_fallback]

    # No stats — just pick first in chain
    for role_name in chain:
        if role_name in AGENT_ROLES and role_name != current_role:
            return AGENT_ROLES[role_name]

    return None


def _select_by_keywords(goal_description: str) -> AgentRole:
    """Original keyword-based role selection as fallback."""
    desc_lower = goal_description.lower()

    # OSINT — more specific than generic research, must be checked first
    osint_keywords = (
        "osint",
        "competitive",
        "market research",
        "competitors",
        "lead discovery",
        "promotion",
        "positioning",
        "monitor community",
        "конкурент",
        "ринок",
        "просування",
        "моніторинг",
    )
    if any(kw in desc_lower for kw in osint_keywords):
        return AGENT_ROLES["osint"]

    research_keywords = (
        "research",
        "search",
        "find out",
        "investigate",
        "learn",
        "discover",
        "explore",
        "study",
        "query",
        "look up",
        "дослідити",
        "знайти",
        "вивчити",
        "пошук",
    )
    if any(kw in desc_lower for kw in research_keywords):
        return AGENT_ROLES["researcher"]

    if any(
        kw in desc_lower
        for kw in (
            "analyze",
            "health",
            "correlate",
            "pattern",
            "insight",
            "review",
            "аналіз",
            "здоров",
        )
    ):
        return AGENT_ROLES["analyst"]

    if any(
        kw in desc_lower
        for kw in (
            "organize",
            "plan",
            "schedule",
            "prioritize",
            "break down",
            "structure",
            "план",
            "організ",
        )
    ):
        return AGENT_ROLES["planner"]

    return AGENT_ROLES["executor"]


def infer_goal_type(goal_description: str) -> str:
    """Infer a simple goal type from description for performance tracking."""
    desc_lower = goal_description.lower()
    if any(
        kw in desc_lower
        for kw in ("register", "sign up", "signup", "create account", "log in", "login")
    ):
        return "signup"
    if any(kw in desc_lower for kw in ("publish", "post", "tweet", "submit post", "share post")):
        return "publish"
    for kw in (
        "osint",
        "competitive",
        "market",
        "research",
        "write",
        "organize",
        "analyze",
        "learn",
        "health",
        "plan",
        "schedule",
        "build",
        "fix",
        "improve",
    ):
        if kw in desc_lower:
            return kw
    return "general"


def format_role_performance_hint(role: AgentRole, goal_type: str) -> str:
    """Generate a hint about role effectiveness for the decision prompt."""
    stats = get_role_stats()
    if not stats:
        return ""

    data = stats.get(role.name, {}).get(goal_type)
    if not data or data["attempts"] < MIN_SAMPLES_FOR_STATS:
        return ""

    rate = data["rate"]
    if rate >= 0.7:
        return f"\nROLE NOTE: '{role.name}' has {rate:.0%} success rate on '{goal_type}' goals ({data['attempts']} attempts).\n"
    elif rate < 0.4:
        return (
            f"\nROLE WARNING: '{role.name}' has only {rate:.0%} success rate on '{goal_type}' goals. "
            f"Consider requesting a role switch if this attempt fails.\n"
        )
    return ""
