"""Daily digest — proactive summary sent to user once per day.

Collects: completed goals, tasks, actions count, LLM spend, balance.
Sends via notification_router (Telegram if user not on web, otherwise event_bus).
Triggered from autonomy _cycle() — checks if digest hour reached and not yet sent today.
"""

import logging
from datetime import datetime, date

logger = logging.getLogger(__name__)

# Hour of day (local time) to send digest (09:00)
DIGEST_HOUR = 9

# In-memory guard: date string of last sent digest (resets on restart — acceptable)
_last_digest_date: str = ""


def should_send_digest() -> bool:
    """Return True if it's time to send the daily digest and we haven't sent it today."""
    global _last_digest_date
    now = datetime.now()
    today = date.today().isoformat()
    if _last_digest_date == today:
        return False
    return now.hour >= DIGEST_HOUR


def _mark_digest_sent() -> None:
    """Mark digest as sent for today — called by morning_digest proactive trigger."""
    global _last_digest_date
    _last_digest_date = date.today().isoformat()


def _collect_completed_today(brain, brain_lock) -> list[str]:
    """Collect goals/tasks completed today from brain."""
    today = date.today().isoformat()
    completed = []
    try:
        with brain_lock:
            records = brain.search(query="", tags=["goal"], limit=100)
        for r in records:
            meta = r.metadata or {}
            if meta.get("status") != "completed":
                continue
            updated = meta.get("updated_at", "") or meta.get("completed_at", "")
            if updated.startswith(today):
                label = (r.content or "")[:80].strip()
                if label:
                    completed.append(label)
    except Exception as e:
        logger.debug("Could not collect completed goals: %s", e)
    return completed[:10]


def _collect_actions_today() -> tuple[int, int]:
    """Return (total_actions, successful_actions) from execution_log for today."""
    today = date.today().isoformat()
    total, success = 0, 0
    try:
        from remy.core.execution_log import execution_log
        entries = execution_log.get_recent(limit=200)
        for e in entries:
            ts = e.get("timestamp", "") or e.get("started_at", "")
            if not ts.startswith(today):
                continue
            total += 1
            if e.get("status") in ("success", "partial_progress"):
                success += 1
    except Exception as e:
        logger.debug("Could not collect action stats: %s", e)
    return total, success


def _collect_budget() -> tuple[float, float]:
    """Return (balance_usd, llm_cost_today_usd)."""
    try:
        from remy.core.combined_runner import get_budget_runtime_snapshot
        budget = get_budget_runtime_snapshot()
        balance = float(budget.get("balance_usd") or budget.get("total_usd") or 0.0)
        llm_cost = float(budget.get("cost_today_usd") or budget.get("llm_cost_today") or 0.0)
        return round(balance, 4), round(llm_cost, 4)
    except Exception as e:
        logger.debug("Could not collect budget: %s", e)
        return 0.0, 0.0


def _collect_active_goals() -> list[str]:
    """Return top active goals (max 3)."""
    goals = []
    try:
        from remy.core.autonomy_goals import get_active_goals
        active = get_active_goals()
        for g in active[:3]:
            desc = (g.get("description") or "")[:70].strip()
            if desc:
                goals.append(desc)
    except Exception as e:
        logger.debug("Could not collect active goals: %s", e)
    return goals


def build_digest(brain, brain_lock) -> str:
    """Build the daily digest message text."""
    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    completed = _collect_completed_today(brain, brain_lock)
    total_actions, success_actions = _collect_actions_today()
    balance, llm_cost = _collect_budget()
    active_goals = _collect_active_goals()

    lines = [f"Daily report — {now}\n"]

    # Completed
    if completed:
        lines.append(f"Completed today ({len(completed)}):")
        for c in completed:
            lines.append(f"  • {c}")
    else:
        lines.append("No goals completed today yet.")

    # Actions
    if total_actions:
        rate = round(success_actions / total_actions * 100) if total_actions else 0
        lines.append(f"\nActions: {total_actions} total, {success_actions} successful ({rate}%)")

    # Active goals
    if active_goals:
        lines.append("\nCurrently working on:")
        for g in active_goals:
            lines.append(f"  • {g}")

    # Budget
    lines.append(f"\nBudget: ${balance} balance | ${llm_cost} spent on LLM today")

    return "\n".join(lines)


def maybe_send_digest(brain=None, brain_lock=None) -> bool:
    """Check if digest should be sent; if yes — build and send it. Returns True if sent."""
    global _last_digest_date

    if not should_send_digest():
        return False

    if brain is None or brain_lock is None:
        try:
            from remy.core.agent_tools import brain as _b, brain_lock as _bl
            brain, brain_lock = _b, _bl
        except Exception as e:
            logger.warning("Daily digest: could not get brain: %s", e)
            return False

    try:
        message = build_digest(brain, brain_lock)
    except Exception as e:
        logger.warning("Daily digest: build failed: %s", e)
        return False

    try:
        from remy.core.notification_router import notify
        notify(message, level="info", event_type="daily.digest", parse_mode="")
        _last_digest_date = date.today().isoformat()
        logger.info("Daily digest sent for %s", _last_digest_date)
        return True
    except Exception as e:
        logger.warning("Daily digest: send failed: %s", e)
        return False
