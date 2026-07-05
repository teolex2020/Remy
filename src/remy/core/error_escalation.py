"""
Proactive Error Escalation (AUTON-10).

Immediate notification of critical problems with recovery suggestions.
Degradation levels: GREEN / YELLOW / RED → different strategies.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum

from remy.core.event_bus import event_bus

logger = logging.getLogger("Autonomy.ErrorEscalation")


# ============== Degradation Levels ==============


class DegradationLevel(IntEnum):
    GREEN = 0  # Everything healthy
    YELLOW = 1  # Degraded — conservative strategy
    RED = 2  # Critical — maintenance-only


@dataclass
class SystemHealth:
    """Snapshot of current system health."""

    level: DegradationLevel = DegradationLevel.GREEN
    llm_available: bool = True
    budget_pct: float = 100.0
    tools_failed: list[str] = field(default_factory=list)
    circuit_breakers_open: list[str] = field(default_factory=list)
    last_error: str = ""
    last_check: float = 0.0


# ============== Recovery Suggestions ==============


_RECOVERY_SUGGESTIONS: dict[str, list[str]] = {
    "llm_unavailable": [
        "Check GEMINI_API_KEY validity",
        "Check API quota at console.cloud.google.com",
        "Check network connectivity",
        "Try a different model (SUMMARY_MODEL setting)",
    ],
    "budget_low": [
        "Increase AUTONOMY_DAILY_TOKEN_LIMIT in settings",
        "Reduce AUTONOMY_CYCLE_INTERVAL_SEC to space out cycles",
        "Focus on low-cost goals (recall, file ops)",
    ],
    "tools_failing": [
        "Check network connectivity for web_search/http_get",
        "Check browser installation for browse_page",
        "Review tool error logs in data/logs/",
    ],
    "session_error": [
        "Check data/ directory permissions",
        "Verify brain database integrity",
        "Restart the application",
    ],
}


def get_recovery_suggestions(issue: str) -> list[str]:
    """Get recovery suggestions for a specific issue type."""
    return _RECOVERY_SUGGESTIONS.get(issue, ["Check logs for details"])


# ============== Health Assessment ==============


def assess_system_health(
    budget_dict: dict | None = None,
    maintenance_only: bool = False,
) -> SystemHealth:
    """Assess current system health across all components.

    Returns SystemHealth with degradation level.
    """
    health = SystemHealth(last_check=time.time())

    # 1. LLM availability
    health.llm_available = not maintenance_only

    # 2. Tool health
    try:
        from remy.core.tool_health import tool_health

        report = tool_health.get_health_report()
        for tool_name, status in report.items():
            if "UNAVAILABLE" in status:
                health.circuit_breakers_open.append(tool_name)
            elif "degraded" in status:
                health.tools_failed.append(tool_name)
    except Exception:
        pass

    # 3. Budget
    if budget_dict:
        daily_limit = budget_dict.get("daily_limit", 1)
        tokens_today = budget_dict.get("tokens_today", 0)
        if daily_limit > 0:
            health.budget_pct = max(0, (1 - tokens_today / daily_limit)) * 100

    # Determine degradation level
    if maintenance_only or not health.llm_available:
        health.level = DegradationLevel.RED
    elif health.budget_pct < 10 or len(health.circuit_breakers_open) >= 3:
        health.level = DegradationLevel.RED
    elif (
        health.budget_pct < 30
        or len(health.circuit_breakers_open) >= 1
        or len(health.tools_failed) >= 2
    ):
        health.level = DegradationLevel.YELLOW
    else:
        health.level = DegradationLevel.GREEN

    return health


def should_skip_cycle(health: SystemHealth) -> bool:
    """Whether the current cycle should be skipped based on health."""
    return health.level == DegradationLevel.RED and not health.llm_available


def get_strategy_for_level(level: DegradationLevel) -> str:
    """Return strategy description for a degradation level."""
    strategies = {
        DegradationLevel.GREEN: "full",  # All tools, all goals
        DegradationLevel.YELLOW: "conservative",  # Skip risky tools, prioritize safe goals
        DegradationLevel.RED: "maintenance",  # Background tasks only
    }
    return strategies.get(level, "full")


# ============== Critical Alerts ==============


_ALERT_COOLDOWNS: dict[str, float] = {}  # alert_key -> last_sent timestamp
_ALERT_COOLDOWN_SEC = 600  # 10 min between duplicate alerts


def _should_alert(key: str) -> bool:
    """Check if we should send an alert (respects cooldown)."""
    now = time.time()
    last = _ALERT_COOLDOWNS.get(key, 0)
    if now - last < _ALERT_COOLDOWN_SEC:
        return False
    _ALERT_COOLDOWNS[key] = now
    return True


def build_alert_message(health: SystemHealth) -> str | None:
    """Build alert message for critical issues. Returns None if no alert needed."""
    alerts = []

    if not health.llm_available and _should_alert("llm_down"):
        suggestions = get_recovery_suggestions("llm_unavailable")
        alerts.append(
            "[CRITICAL] LLM unavailable — maintenance-only mode.\n"
            "Possible fixes:\n" + "\n".join(f"  - {s}" for s in suggestions)
        )

    if health.budget_pct < 10 and _should_alert("budget_critical"):
        suggestions = get_recovery_suggestions("budget_low")
        alerts.append(
            f"[WARNING] Budget critically low: {health.budget_pct:.0f}% remaining.\n"
            "Suggestions:\n" + "\n".join(f"  - {s}" for s in suggestions)
        )

    if len(health.circuit_breakers_open) >= 3 and _should_alert("tools_mass_failure"):
        tools = ", ".join(health.circuit_breakers_open[:5])
        suggestions = get_recovery_suggestions("tools_failing")
        alerts.append(
            f"[WARNING] {len(health.circuit_breakers_open)} tools unavailable: {tools}\n"
            "Suggestions:\n" + "\n".join(f"  - {s}" for s in suggestions)
        )

    if not alerts:
        return None

    return "\n\n".join(alerts)


def build_operator_watch_message(
    health: SystemHealth,
    *,
    previous_level: DegradationLevel | None = None,
    gateway_health: str = "ok",
    previous_gateway_health: str | None = None,
) -> tuple[str, str] | None:
    """Build proactive operator alerts for runtime degradation and recovery."""
    gateway_health = (gateway_health or "ok").lower()
    previous_gateway_health = (previous_gateway_health or "").lower() or None

    if previous_level is None and previous_gateway_health is None:
        if health.level == DegradationLevel.GREEN and gateway_health == "ok":
            return None
        return (
            (
                "*Remy operator alert*\n"
                f"Runtime health: `{health.level.name}`\n"
                f"Gateway: `{gateway_health}`\n"
                f"Budget remaining: `{health.budget_pct:.0f}%`"
            ),
            "critical" if health.level == DegradationLevel.RED or gateway_health == "error" else "warning",
        )

    if previous_level is not None and health.level > previous_level:
        return (
            (
                "*Remy operator alert*\n"
                f"System health degraded: `{previous_level.name}` -> `{health.level.name}`\n"
                f"Gateway: `{gateway_health}`\n"
                f"Budget remaining: `{health.budget_pct:.0f}%`"
            ),
            "critical" if health.level == DegradationLevel.RED else "warning",
        )

    if previous_gateway_health and gateway_health != previous_gateway_health and gateway_health in {"degraded", "error"}:
        return (
            (
                "*Remy operator alert*\n"
                f"Gateway state changed: `{previous_gateway_health}` -> `{gateway_health}`\n"
                f"System health: `{health.level.name}`\n"
                f"Budget remaining: `{health.budget_pct:.0f}%`"
            ),
            "critical" if gateway_health == "error" else "warning",
        )

    if (
        previous_level is not None
        and previous_gateway_health
        and previous_level != DegradationLevel.GREEN
        and health.level == DegradationLevel.GREEN
        and previous_gateway_health != "ok"
        and gateway_health == "ok"
    ):
        return (
            (
                "*Remy operator update*\n"
                f"Runtime recovered: `{previous_level.name}` -> `GREEN`\n"
                "Gateway is back to `ok`."
            ),
            "info",
        )

    if previous_level is not None and previous_level != DegradationLevel.GREEN and health.level == DegradationLevel.GREEN:
        return (
            (
                "*Remy operator update*\n"
                f"System health recovered: `{previous_level.name}` -> `GREEN`\n"
                f"Gateway: `{gateway_health}`"
            ),
            "info",
        )

    if previous_gateway_health and previous_gateway_health != "ok" and gateway_health == "ok":
        return (
            (
                "*Remy operator update*\n"
                f"Gateway recovered: `{previous_gateway_health}` -> `ok`\n"
                f"System health: `{health.level.name}`"
            ),
            "info",
        )

    if 10 <= health.budget_pct < 30 and _should_alert("budget_warning"):
        return (
            (
                "*Remy operator alert*\n"
                f"Budget pressure detected: `{health.budget_pct:.0f}%` remaining.\n"
                "Autonomy should favor lower-cost work until budget recovers."
            ),
            "warning",
        )

    return None


async def send_critical_alert(health: SystemHealth) -> bool:
    """Send critical alert via Telegram if configured. Returns True if sent."""
    message = build_alert_message(health)
    if not message:
        return False

    logger.warning("Critical alert: %s", message[:200])
    event_bus.emit(
        "critical_alert",
        {
            "level": health.level.name,
            "message": message[:500],
        },
    )

    try:
        from remy.config.settings import settings

        if not settings.TELEGRAM_BOT_TOKEN or not settings.PROACTIVE_CHAT_ID:
            return False
        try:
            from remy.core.notification_router import should_notify_telegram

            if not should_notify_telegram():
                logger.debug("Health alert suppressed for Telegram because web runtime is active")
                return False
        except Exception as e:
            logger.debug("Could not evaluate Telegram suppression for health alert: %s", e)

        from telegram import Bot

        bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
        await bot.send_message(
            chat_id=settings.PROACTIVE_CHAT_ID,
            text=f"[Remy Health Alert]\n\n{message}",
        )
        return True
    except Exception as e:
        logger.debug("Alert send failed: %s", e)
        return False


# ============== Auto-Recovery ==============


async def attempt_auto_recovery(health: SystemHealth) -> bool:
    """Try automatic recovery steps before escalating. Returns True if recovered."""
    if not health.llm_available:
        # Try a lightweight LLM call to check if it recovered
        recovered = await _test_llm_health()
        if recovered:
            logger.info("LLM auto-recovered")
            event_bus.emit("auto_recovery", {"component": "llm", "success": True})
            return True
        event_bus.emit("auto_recovery", {"component": "llm", "success": False})

    return False


async def _test_llm_health() -> bool:
    """Quick LLM health check — single cheap call."""
    try:
        import google.genai as genai

        from remy.config.settings import settings

        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        response = client.models.generate_content(
            model=settings.SUMMARY_MODEL,
            contents="Reply with OK",
            config={"max_output_tokens": 5},
        )
        return bool(response.text)
    except Exception:
        return False


# ============== Detailed Health API ==============


def get_detailed_health(budget_dict: dict | None = None, maintenance_only: bool = False) -> dict:
    """Full health report for API endpoint."""
    health = assess_system_health(budget_dict, maintenance_only)

    # Tool health details
    tool_details = {}
    try:
        from remy.core.tool_health import tool_health

        report = tool_health.get_health_report()
        tool_details = report
    except Exception:
        pass

    return {
        "level": health.level.name,
        "strategy": get_strategy_for_level(health.level),
        "llm_available": health.llm_available,
        "budget_remaining_pct": round(health.budget_pct, 1),
        "tools": {
            "circuit_breakers_open": health.circuit_breakers_open,
            "degraded": health.tools_failed,
            "details": tool_details,
        },
        "last_check": health.last_check,
        "recovery_suggestions": (
            get_recovery_suggestions("llm_unavailable")
            if not health.llm_available
            else get_recovery_suggestions("budget_low")
            if health.budget_pct < 10
            else []
        ),
    }


def reset_alert_cooldowns():
    """Reset alert cooldowns (for testing)."""
    _ALERT_COOLDOWNS.clear()
