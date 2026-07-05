"""
Budget Negotiation (AUTON-7) — agent can request more resources, forecast costs,
track savings, and override limits for critical goals.

Instead of a hard stop when the budget runs out, the agent can:
- Request a budget increase with justification (via Telegram/GUI)
- Override hourly limits by 50% for critical-priority goals
- Estimate future goal costs from historical data
- Report savings from cache hits and skipped actions
- Request quiet hours override for urgent tasks
"""

import logging
import time
from dataclasses import dataclass, field

from remy.core.event_bus import event_bus

logger = logging.getLogger("Autonomy.Budget")


# ============== Budget Request ==============


@dataclass
class BudgetRequest:
    """A pending request for budget increase."""

    request_id: str
    reason: str
    requested_tokens: int
    goal_description: str
    current_usage: int
    current_limit: int
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    approved: bool = False


# ============== Savings Tracker ==============


class SavingsTracker:
    """Tracks tokens saved through caching, skips, and efficient routing."""

    def __init__(self):
        self.cache_hits: int = 0
        self.cache_tokens_saved: int = 0
        self.skipped_actions: int = 0
        self.skip_tokens_saved: int = 0
        self.preflight_blocks: int = 0
        self.preflight_tokens_saved: int = 0
        self._session_start: float = time.time()

    def record_cache_hit(self, estimated_cost: int = 800):
        """Record a cache hit (e.g., recall found cached web_search result)."""
        self.cache_hits += 1
        self.cache_tokens_saved += estimated_cost

    def record_skip(self, estimated_cost: int = 1000):
        """Record a skipped action (e.g., preflight blocked, low confidence skip)."""
        self.skipped_actions += 1
        self.skip_tokens_saved += estimated_cost

    def record_preflight_block(self, estimated_cost: int = 2000):
        """Record a cycle blocked by preflight analysis."""
        self.preflight_blocks += 1
        self.preflight_tokens_saved += estimated_cost

    @property
    def total_saved(self) -> int:
        return self.cache_tokens_saved + self.skip_tokens_saved + self.preflight_tokens_saved

    def format_report(self) -> str:
        """Format savings as a text report for the decision prompt."""
        if self.total_saved == 0:
            return ""

        lines = [f"\nSAVINGS REPORT ({self.total_saved:,} tokens saved this session):"]
        if self.cache_tokens_saved > 0:
            lines.append(
                f"  - Cache hits: {self.cache_hits} ({self.cache_tokens_saved:,} tokens saved)"
            )
        if self.skip_tokens_saved > 0:
            lines.append(
                f"  - Smart skips: {self.skipped_actions} ({self.skip_tokens_saved:,} tokens saved)"
            )
        if self.preflight_tokens_saved > 0:
            lines.append(
                f"  - Pre-flight blocks: {self.preflight_blocks} ({self.preflight_tokens_saved:,} tokens saved)"
            )
        return "\n".join(lines) + "\n"


# Module-level singleton
savings_tracker = SavingsTracker()


# ============== Cost Estimation ==============


def estimate_goal_cost(goal_description: str, attempts: int = 0) -> int:
    """Estimate token cost for a goal based on historical data.

    Uses outcome records from brain to compute average cost per goal type,
    with fallback to heuristic estimates.
    """
    from remy.core.agent_tools import brain, brain_lock

    # Try historical data first
    try:
        with brain_lock:
            outcomes = brain.search(
                query=goal_description[:80],
                tags=["autonomous-outcome"],
                limit=10,
            )

        if outcomes:
            token_costs = []
            for r in outcomes:
                meta = getattr(r, "metadata", None) or {}
                tokens = meta.get("tokens_used", 0)
                if tokens > 0:
                    token_costs.append(tokens)

            if token_costs:
                avg_cost = sum(token_costs) // len(token_costs)
                # Add 20% buffer for uncertainty
                return int(avg_cost * 1.2)
    except Exception as e:
        logger.debug("Historical cost lookup failed: %s", e)

    # Heuristic fallback based on goal keywords
    desc = goal_description.lower()

    if any(kw in desc for kw in ("research", "investigate", "study", "find out")):
        base = 5000  # Research goals are expensive
    elif any(kw in desc for kw in ("browse", "web", "download", "scrape")):
        base = 4000  # Browser actions are expensive
    elif any(kw in desc for kw in ("analyze", "correlate", "summarize")):
        base = 2000  # Analysis goals
    elif any(kw in desc for kw in ("store", "save", "record", "update")):
        base = 1000  # Storage goals are cheap
    else:
        base = 2500  # Default

    # More attempts = likely need more tokens (retries, decomposition)
    attempt_multiplier = 1.0 + (attempts * 0.15)
    return int(base * attempt_multiplier)


def format_budget_forecast(goals: list[dict]) -> str:
    """Format cost estimates for active goals."""
    if not goals:
        return ""

    lines = ["\nBUDGET FORECAST (estimated tokens per goal):"]
    total_estimated = 0
    for g in goals[:5]:
        cost = estimate_goal_cost(g["description"], g.get("attempts", 0))
        total_estimated += cost
        lines.append(f"  - {g['description'][:60]}: ~{cost:,} tokens")

    lines.append(f"  TOTAL estimated: ~{total_estimated:,} tokens")
    return "\n".join(lines) + "\n"


# ============== Priority Override ==============


def can_priority_override(goal: dict, budget) -> tuple[bool, str]:
    """Check if a critical goal can override the hourly budget limit.

    Critical goals can use up to 150% of the hourly limit.
    Returns (can_proceed, reason).
    """
    priority = goal.get("priority", "medium").lower()
    if priority != "high" and priority != "critical":
        return False, "Only high/critical priority goals qualify for override"

    # Check if we're within the override threshold (150% of hourly)
    override_limit = int(budget.hourly_limit * 1.5)
    now = time.time()

    # Reset hourly counter if needed
    if now - budget.last_hour_reset > 3600:
        budget.tokens_this_hour = 0
        budget.last_hour_reset = now

    if budget.tokens_this_hour < override_limit:
        remaining = override_limit - budget.tokens_this_hour
        logger.info(
            "Priority override active for '%s': %d/%d tokens available (150%% hourly)",
            goal.get("description", "?")[:40],
            remaining,
            override_limit,
        )
        return True, f"Priority override: {remaining:,} tokens available (150% hourly limit)"

    return (
        False,
        f"Even with override, hourly limit exceeded ({budget.tokens_this_hour}/{override_limit})",
    )


# ============== Budget Increase Request ==============


def request_budget_increase(
    reason: str,
    goal_description: str,
    requested_tokens: int,
    budget,
) -> BudgetRequest:
    """Create a budget increase request and notify via event bus.

    The request is sent to Telegram/GUI for human approval.
    """
    import uuid

    req = BudgetRequest(
        request_id=uuid.uuid4().hex[:12],
        reason=reason,
        requested_tokens=requested_tokens,
        goal_description=goal_description[:200],
        current_usage=budget.tokens_today,
        current_limit=budget.daily_limit,
    )

    event_bus.emit(
        "budget.increase_requested",
        {
            "request_id": req.request_id,
            "reason": req.reason,
            "requested_tokens": req.requested_tokens,
            "goal": req.goal_description,
            "current_usage": req.current_usage,
            "current_limit": req.current_limit,
        },
    )

    logger.info(
        "Budget increase requested: +%d tokens for '%s' (reason: %s)",
        requested_tokens,
        goal_description[:40],
        reason[:60],
    )

    return req


def apply_budget_increase(budget, additional_tokens: int) -> None:
    """Apply an approved budget increase to the daily limit."""
    old_limit = budget.daily_limit
    budget.daily_limit += additional_tokens

    logger.info(
        "Budget increased: %d → %d (+%d tokens)",
        old_limit,
        budget.daily_limit,
        additional_tokens,
    )

    event_bus.emit(
        "budget.increased",
        {
            "old_limit": old_limit,
            "new_limit": budget.daily_limit,
            "added": additional_tokens,
        },
    )


# ============== Dynamic Quiet Hours ==============

# Per-goal override limit during a single quiet hours window.
# Prevents a buggy critical goal from waking the agent endlessly.
MAX_OVERRIDES_PER_GOAL = 2
MAX_OVERRIDES_PER_SESSION = 4

# Tracks overrides: goal_id → count.  Reset when quiet hours window changes.
_override_counts: dict[str, int] = {}
_override_total: int = 0
_override_window: str = ""  # "YYYY-MM-DD" of current quiet hours night


def _get_quiet_window_key() -> str:
    """Return a key representing the current quiet hours window (one per night)."""
    from datetime import datetime

    now = datetime.now()
    # Quiet hours span midnight, so use the date of the start (before midnight = today)
    return now.strftime("%Y-%m-%d")


def _reset_if_new_window() -> None:
    """Reset override counters if we've entered a new quiet hours window."""
    global _override_counts, _override_total, _override_window
    key = _get_quiet_window_key()
    if key != _override_window:
        _override_counts = {}
        _override_total = 0
        _override_window = key


def request_quiet_hours_override(goal: dict) -> bool:
    """Request to temporarily disable quiet hours for an urgent goal.

    Only allowed for high-priority goals with approaching deadlines.
    Returns True if override is justified.

    SAFETY: Max 2 overrides per goal and 4 total per quiet hours window.
    This prevents a stuck critical goal from burning the entire API budget overnight.
    """
    global _override_total

    priority = goal.get("priority", "medium").lower()
    if priority not in ("high", "critical"):
        return False

    _reset_if_new_window()

    # Check session-level cap
    if _override_total >= MAX_OVERRIDES_PER_SESSION:
        logger.warning(
            "Quiet hours override DENIED: session cap reached (%d/%d)",
            _override_total,
            MAX_OVERRIDES_PER_SESSION,
        )
        return False

    # Check per-goal cap
    goal_id = goal.get("id", goal.get("description", "?")[:60])
    goal_count = _override_counts.get(goal_id, 0)
    if goal_count >= MAX_OVERRIDES_PER_GOAL:
        logger.warning(
            "Quiet hours override DENIED for goal '%s': per-goal cap reached (%d/%d)",
            goal_id[:30],
            goal_count,
            MAX_OVERRIDES_PER_GOAL,
        )
        return False

    # Check deadline proximity
    deadline = goal.get("deadline")
    if deadline:
        from datetime import datetime

        try:
            deadline_dt = datetime.fromisoformat(deadline)
            hours_until = (deadline_dt - datetime.now()).total_seconds() / 3600
            if hours_until < 6:  # Less than 6 hours to deadline
                _override_counts[goal_id] = goal_count + 1
                _override_total += 1
                logger.info(
                    "Quiet hours override granted (%d/%d): goal '%s' deadline in %.1f hours",
                    _override_total,
                    MAX_OVERRIDES_PER_SESSION,
                    goal.get("description", "?")[:40],
                    hours_until,
                )
                event_bus.emit(
                    "quiet_hours.override",
                    {
                        "goal": goal.get("description", "")[:100],
                        "hours_until_deadline": round(hours_until, 1),
                        "override_count": _override_total,
                    },
                )
                return True
        except (ValueError, TypeError):
            pass

    # High priority + many attempts = urgent
    attempts = goal.get("attempts", 0)
    if priority == "critical" or (priority == "high" and attempts >= 3):
        _override_counts[goal_id] = goal_count + 1
        _override_total += 1
        logger.info(
            "Quiet hours override granted (%d/%d): critical goal with %d attempts",
            _override_total,
            MAX_OVERRIDES_PER_SESSION,
            attempts,
        )
        event_bus.emit(
            "quiet_hours.override",
            {
                "goal": goal.get("description", "")[:100],
                "reason": "critical_priority",
                "override_count": _override_total,
            },
        )
        return True

    return False


# ============== Budget Notification via Telegram ==============


def send_budget_request_telegram(request: BudgetRequest) -> None:
    """Send budget increase request via Telegram if configured."""
    try:
        from remy.config.settings import settings

        if not settings.TELEGRAM_BOT_TOKEN or not settings.PROACTIVE_CHAT_ID:
            return
        try:
            from remy.core.notification_router import should_notify_telegram

            if not should_notify_telegram():
                logger.debug("Budget request suppressed for Telegram because web runtime is active")
                return
        except Exception as e:
            logger.debug("Could not evaluate Telegram suppression for budget request: %s", e)

        import threading

        def _send():
            try:
                import httpx

                text = (
                    f"💰 *Budget Increase Request*\n\n"
                    f"*Goal:* {request.goal_description[:100]}\n"
                    f"*Reason:* {request.reason[:100]}\n"
                    f"*Requested:* +{request.requested_tokens:,} tokens\n"
                    f"*Current:* {request.current_usage:,}/{request.current_limit:,}\n\n"
                    f"Reply `approve {request.request_id[:8]}` to grant "
                    f"or `deny {request.request_id[:8]}` to reject."
                )

                url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
                httpx.post(
                    url,
                    json={
                        "chat_id": settings.PROACTIVE_CHAT_ID,
                        "text": text,
                        "parse_mode": "Markdown",
                    },
                    timeout=10,
                )
            except Exception as e:
                logger.debug("Telegram budget request failed: %s", e)

        t = threading.Thread(target=_send, daemon=True)
        t.start()

    except Exception as e:
        logger.debug("Budget telegram notification skipped: %s", e)


# ============== Integration helpers ==============


def check_budget_with_override(
    budget,
    estimated_tokens: int,
    top_goal: dict | None = None,
) -> tuple[bool, str]:
    """Check budget, applying priority override if applicable.

    Returns (can_proceed, reason).
    """
    can_spend, reason = budget.can_spend(estimated_tokens)
    if can_spend:
        return True, reason

    # Try priority override for hourly limit
    if top_goal and "Hourly limit" in reason:
        can_override, override_reason = can_priority_override(top_goal, budget)
        if can_override:
            return True, override_reason

    return False, reason
