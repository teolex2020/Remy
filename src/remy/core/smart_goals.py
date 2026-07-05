"""
Smart Goal Management (AUTON-8) — dependencies, reprioritization, merge, cleanup, batching.

Enhances the goal system with:
- Goal dependencies (depends_on metadata) with topological sorting
- Auto-reprioritization based on deadline proximity, dependency chain, success prediction
- Duplicate/similar goal detection and merging
- Stale goal cleanup (no attempts for 48h + low priority)
- Goal batching (group goals needing the same tool type)
"""

import logging
from datetime import datetime, timedelta

from remy.core.event_bus import event_bus

logger = logging.getLogger("Autonomy.SmartGoals")


# ============== Goal Dependencies ==============


def get_dependency_graph(goals: list[dict]) -> dict[str, list[str]]:
    """Build a dependency adjacency list from goal metadata.

    Returns: {goal_id: [depends_on_goal_id, ...]}
    """
    graph: dict[str, list[str]] = {}
    for g in goals:
        gid = g.get("goal_id", "")
        deps = g.get("depends_on", [])
        if isinstance(deps, str):
            deps = [deps]
        graph[gid] = deps
    return graph


def topological_sort_goals(goals: list[dict]) -> list[dict]:
    """Sort goals respecting dependencies: dependencies first.

    Falls back to original order for goals without dependencies.
    Uses Kahn's algorithm for topological sort.
    """
    if not goals:
        return goals

    graph = get_dependency_graph(goals)
    goal_map = {g.get("goal_id", ""): g for g in goals}
    all_ids = set(goal_map.keys())

    # Filter deps to only include active goal IDs
    in_degree: dict[str, int] = {gid: 0 for gid in all_ids}
    adj: dict[str, list[str]] = {gid: [] for gid in all_ids}

    for gid, deps in graph.items():
        for dep in deps:
            if dep in all_ids:
                adj[dep].append(gid)  # dep → gid (dep must come first)
                in_degree[gid] = in_degree.get(gid, 0) + 1

    # Kahn's algorithm
    queue = [gid for gid in all_ids if in_degree.get(gid, 0) == 0]
    sorted_ids: list[str] = []

    while queue:
        # Stable: prefer original order among equal candidates
        queue.sort(
            key=lambda gid: next((i for i, g in enumerate(goals) if g.get("goal_id") == gid), 999)
        )
        node = queue.pop(0)
        sorted_ids.append(node)
        for neighbor in adj.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Any remaining (cycles) get appended at the end
    for gid in all_ids:
        if gid not in sorted_ids:
            sorted_ids.append(gid)

    return [goal_map[gid] for gid in sorted_ids if gid in goal_map]


def get_blocked_goals(goals: list[dict]) -> list[str]:
    """Return goal_ids that are blocked by unfinished dependencies."""
    graph = get_dependency_graph(goals)
    active_ids = {g.get("goal_id", "") for g in goals}

    blocked = []
    for gid, deps in graph.items():
        # If any dependency is still active (not completed), this goal is blocked
        active_deps = [d for d in deps if d in active_ids]
        if active_deps:
            blocked.append(gid)

    return blocked


# ============== Auto-Reprioritize ==============


def reprioritize_goals(goals: list[dict]) -> list[dict]:
    """Re-sort goals based on dynamic factors.

    Scoring factors:
    1. Deadline proximity (closer = higher priority)
    2. Dependency readiness (no blockers = higher)
    3. Success prediction (lower prediction for many attempts)
    4. Current priority level
    """
    if not goals:
        return goals

    blocked_ids = set(get_blocked_goals(goals))
    now = datetime.now()
    has_live_mission_tasks = any(g.get("mission_task_id") for g in goals)

    scored: list[tuple[float, dict]] = []
    for g in goals:
        score = _compute_priority_score(g, now, blocked_ids, has_live_mission_tasks)
        scored.append((score, g))

    # Sort by score descending (higher = should run first)
    scored.sort(key=lambda x: x[0], reverse=True)
    return [g for _, g in scored]


def _compute_priority_score(
    goal: dict,
    now: datetime,
    blocked_ids: set[str],
    has_live_mission_tasks: bool = False,
) -> float:
    """Compute a numeric priority score for a goal.

    Higher score = should be worked on first.
    """
    score = 0.0
    gid = goal.get("goal_id", "")

    # Base priority
    priority_scores = {"critical": 100, "high": 70, "medium": 40, "low": 10}
    score += priority_scores.get(goal.get("priority", "medium"), 40)

    # Blocked goals get heavily penalized
    if gid in blocked_ids:
        score -= 80

    # Sub-goals get a boost (more actionable)
    if goal.get("parent_goal_id"):
        score += 15

    # When a live mission batch exists, keep atomic mission tasks front and center.
    if goal.get("mission_task_id"):
        score += 45
        if goal.get("goal_template") in ("signup_operator", "publisher", "market_research"):
            score += 10
    elif goal.get("mission_id"):
        score += 5
    elif has_live_mission_tasks:
        score -= 25

    # Deadline proximity: closer deadline = much higher score
    deadline = goal.get("deadline")
    if deadline:
        try:
            deadline_dt = datetime.fromisoformat(deadline)
            hours_until = (deadline_dt - now).total_seconds() / 3600
            if hours_until < 0:
                score += 50  # Overdue — urgent!
            elif hours_until < 6:
                score += 40  # Very close
            elif hours_until < 24:
                score += 25  # Today
            elif hours_until < 72:
                score += 10  # Soon
        except (ValueError, TypeError):
            pass

    # Many attempts = lower score (diminishing returns)
    attempts = goal.get("attempts", 0)
    if attempts >= 5:
        score -= 20
    elif attempts >= 3:
        score -= 10

    # Legacy account/login style goals should not steal focus from live mission tasks.
    if has_live_mission_tasks and not goal.get("mission_task_id"):
        desc_lower = str(goal.get("description", "") or "").lower()
        legacy_identity_markers = (
            "account.mail.ru",
            "proton.me",
            "protonmail",
            "gmail",
            "github.com/signup",
            "twitter",
            "x.com",
            "register own email",
            "log into your github",
            "create account",
            "sign up",
            "login",
        )
        if any(marker in desc_lower for marker in legacy_identity_markers):
            score -= 30

    return score


# ============== Goal Merge ==============


def find_similar_goals(goals: list[dict], threshold: float = 0.7) -> list[tuple[int, int]]:
    """Find pairs of goals with similar descriptions.

    Uses simple word overlap (Jaccard similarity) — zero LLM.
    Returns list of (index_i, index_j) pairs.
    """
    pairs = []

    for i in range(len(goals)):
        for j in range(i + 1, len(goals)):
            desc_i = goals[i].get("description", "").lower()
            desc_j = goals[j].get("description", "").lower()
            sim = _jaccard_similarity(desc_i, desc_j)
            if sim >= threshold:
                pairs.append((i, j))

    return pairs


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between two texts (word-level)."""
    words_a = set(text_a.split())
    words_b = set(text_b.split())

    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b
    union = words_a | words_b

    return len(intersection) / len(union) if union else 0.0


def merge_goals(goals: list[dict], idx_keep: int, idx_remove: int) -> dict | None:
    """Merge two similar goals — keep the one with more progress.

    Returns the merged goal dict, or None if merge isn't possible.
    """
    if idx_keep >= len(goals) or idx_remove >= len(goals):
        return None

    keep = goals[idx_keep]
    remove = goals[idx_remove]

    # Merge description (keep the longer/more detailed one)
    if len(remove.get("description", "")) > len(keep.get("description", "")):
        keep["description"] = remove["description"]

    # Keep the higher priority
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    keep_pri = priority_order.get(keep.get("priority", "medium"), 2)
    remove_pri = priority_order.get(remove.get("priority", "medium"), 2)
    if remove_pri < keep_pri:
        keep["priority"] = remove["priority"]

    # Keep the closer deadline
    if remove.get("deadline") and not keep.get("deadline"):
        keep["deadline"] = remove["deadline"]
    elif remove.get("deadline") and keep.get("deadline"):
        if remove["deadline"] < keep["deadline"]:
            keep["deadline"] = remove["deadline"]

    # Sum attempts
    keep["attempts"] = keep.get("attempts", 0) + remove.get("attempts", 0)

    logger.info(
        "Goals merged: kept '%s', removed '%s'",
        keep.get("description", "?")[:40],
        remove.get("description", "?")[:40],
    )

    return keep


# ============== Goal Cleanup ==============


def find_stale_goals(goals: list[dict], stale_hours: int = 48) -> list[dict]:
    """Find goals that haven't been attempted in `stale_hours` and are low priority.

    These are candidates for automatic cleanup.
    """
    now = datetime.now()
    cutoff = (now - timedelta(hours=stale_hours)).isoformat()
    stale = []

    for g in goals:
        priority = g.get("priority", "medium")
        if priority in ("high", "critical"):
            continue  # Never auto-clean high/critical

        last_attempt = g.get("last_attempt")
        created_at = g.get("created_at", "")

        # No attempts ever and created long ago
        if g.get("attempts", 0) == 0 and created_at and created_at < cutoff:
            stale.append(g)
            continue

        # Has been attempted but not recently
        if last_attempt and last_attempt < cutoff and priority == "low":
            stale.append(g)

    return stale


def cleanup_stale_goals(goals: list[dict], stale_hours: int = 48) -> int:
    """Auto-archive stale goals. Returns count of goals cleaned up."""
    from remy.core.autonomy_goals import update_goal_status

    stale = find_stale_goals(goals, stale_hours)
    cleaned = 0

    for g in stale:
        record_id = g.get("record_id")
        if not record_id:
            continue

        try:
            update_goal_status(
                record_id,
                "failed",
                notes=f"Auto-cleaned: stale ({stale_hours}h no activity)",
            )
            cleaned += 1
            event_bus.emit(
                "goal_cleaned",
                {
                    "goal_id": g.get("goal_id", ""),
                    "description": g.get("description", "")[:100],
                    "reason": "stale",
                },
            )
        except Exception as e:
            logger.debug("Goal cleanup failed: %s", e)

    if cleaned:
        logger.info("Cleaned up %d stale goals", cleaned)

    return cleaned


# ============== Goal Batching ==============


# Tool keywords → required tool category
_TOOL_CATEGORIES = {
    "browser": ["browse", "web page", "form", "click", "download page", "screenshot"],
    "research": ["research", "investigate", "find out", "study", "learn about"],
    "file": ["file", "write", "read", "save to disk", "create file"],
    "analysis": ["analyze", "correlate", "summarize", "pattern", "health"],
}


def batch_goals_by_tool(goals: list[dict]) -> dict[str, list[dict]]:
    """Group goals by their primary tool requirement.

    Batching allows running related goals together for efficiency
    (e.g., all browser goals while browser is open).
    """
    batches: dict[str, list[dict]] = {}

    for g in goals:
        category = _infer_tool_category(g.get("description", ""))
        if category not in batches:
            batches[category] = []
        batches[category].append(g)

    return batches


def _infer_tool_category(description: str) -> str:
    """Infer which tool category a goal primarily needs."""
    desc = description.lower()

    for category, keywords in _TOOL_CATEGORIES.items():
        if any(kw in desc for kw in keywords):
            return category

    return "general"


def get_batch_hint(goals: list[dict]) -> str:
    """Generate a batching hint for the decision prompt.

    If multiple goals share a tool category, suggest batching.
    """
    batches = batch_goals_by_tool(goals)

    hints = []
    for category, batch_goals in batches.items():
        if len(batch_goals) >= 2:
            descriptions = [g.get("description", "")[:40] for g in batch_goals[:3]]
            hints.append(
                f"  - {category.upper()}: {len(batch_goals)} goals could be batched — "
                f"{', '.join(descriptions)}"
            )

    if not hints:
        return ""

    return "\nBATCHING OPPORTUNITY:\n" + "\n".join(hints) + "\n"


# ============== Integrated Smart Sort ==============


def smart_sort_goals(goals: list[dict]) -> list[dict]:
    """Apply full smart sorting: dependencies → reprioritization.

    This replaces the simple priority sort in get_active_goals().
    """
    if not goals:
        return goals

    # Step 1: Topological sort (respect dependencies)
    sorted_goals = topological_sort_goals(goals)

    # Step 2: Within dependency-safe ordering, reprioritize
    sorted_goals = reprioritize_goals(sorted_goals)

    return sorted_goals


def format_goal_management_hints(goals: list[dict]) -> str:
    """Generate management hints for the decision prompt.

    Includes: blocked goals warning, batching hints, stale goal alerts.
    """
    if not goals:
        return ""

    parts = []

    # Blocked goals
    blocked = get_blocked_goals(goals)
    if blocked:
        parts.append(
            f"BLOCKED GOALS: {len(blocked)} goal(s) waiting on dependencies. "
            f"Complete their prerequisites first."
        )

    # Batching opportunities
    batch_hint = get_batch_hint(goals)
    if batch_hint:
        parts.append(batch_hint)

    # Stale goals
    stale = find_stale_goals(goals)
    if stale:
        names = [g.get("description", "?")[:40] for g in stale[:3]]
        parts.append(f"STALE GOALS ({len(stale)}): Consider archiving — {', '.join(names)}")

    if not parts:
        return ""

    return "\nGOAL MANAGEMENT:\n" + "\n".join(parts) + "\n"
