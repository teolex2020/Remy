"""
Survival Module — financial awareness for the autonomous agent.

Integrates wallet balance monitoring, LLM cost tracking, and
working capital estimation into the autonomy loop. Sends alerts
when the agent's resources are running low.

Business model:
- Server costs are paid by the user (owner's investment)
- LLM API costs are paid FROM THE AGENT'S WALLET — every token = real money
- Wallet balance = agent's total capital for operations AND earning money
- Agent's goal: earn more than it spends on LLM API + tx fees
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from remy.config.settings import settings
from remy.core.event_bus import event_bus

logger = logging.getLogger("Autonomy.Survival")

# ============== CONFIG ==============

# Agent's Tron wallet address
AGENT_WALLET = "TNjyL4vZwBQg1tzudWWM8aFavPCYZTRAJY"

# Tron USDT contract
USDT_CONTRACT = "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"

TRON_ENDPOINTS = [
    "https://api.trongrid.io",
    "https://api.tronstack.io",
]

# Survival thresholds (in USD)
# Server is paid by user, but LLM API is paid from agent's wallet.
# Spending scales with income — these are absolute capital thresholds.
CRITICAL_BALANCE_USD = 1.0  # RED — almost no capital, emergency mode
WARNING_BALANCE_USD = 5.0  # YELLOW — very limited, minimize spending
HEALTHY_BALANCE_USD = 20.0  # GREEN — stable base for operations
GROWTH_BALANCE_USD = 100.0  # BLUE — ready for bigger investments

# Budget rule: never spend more than this % of income on LLM API
MAX_LLM_SPEND_RATIO = 0.80  # 80% of income max → 20% saved for growth/infra

# Seed investment: user gives $30 for the first month of API costs.
# After this period, agent must fully sustain itself.
SEED_BUDGET_USD = 30.0  # Total seed investment for API
SEED_BUDGET_DAILY_USD = 1.00  # ~$30/30 days — max daily API spend from seed
SEED_PERIOD_DAYS = 30  # Duration of seed investment period
SEED_START_DATE = "2026-03-02"  # When the seed period started

# Check interval (don't spam blockchain API)
BALANCE_CHECK_INTERVAL_SEC = 1800  # 30 minutes

# Alert cooldown (don't spam Telegram)
ALERT_COOLDOWN_SEC = 3600  # 1 hour between alerts of same type

# State file
SURVIVAL_STATE_FILE = "survival_state.json"


# ============== STATE ==============


@dataclass
class SurvivalState:
    """Tracks the agent's financial vital signs."""

    last_balance_check: float = 0.0
    last_trx: float = 0.0
    last_usdt: float = 0.0
    last_total_usd: float = 0.0
    last_status: str = "UNKNOWN"  # CRITICAL / WARNING / MODERATE / HEALTHY / UNKNOWN
    last_runway_days: float = 0.0

    # Alert tracking
    last_critical_alert: float = 0.0
    last_warning_alert: float = 0.0

    # Daily LLM cost tracking (mirrors budget but from survival perspective)
    llm_cost_today: float = 0.0
    llm_cost_yesterday: float = 0.0
    last_cost_day: str = ""

    # Cached balance fallback: timestamp of last REAL API success (not cached read)
    last_api_success: float = 0.0

    # Income tracking
    total_earned_usd: float = 0.0
    last_earning_at: float = 0.0
    earnings_history: list = field(default_factory=list)  # last 10 earning events

    def to_dict(self) -> dict:
        return {
            "last_balance_check": self.last_balance_check,
            "last_trx": self.last_trx,
            "last_usdt": self.last_usdt,
            "last_total_usd": round(self.last_total_usd, 4),
            "last_status": self.last_status,
            "last_runway_days": round(self.last_runway_days, 1),
            "llm_cost_today": round(self.llm_cost_today, 6),
            "llm_cost_yesterday": round(self.llm_cost_yesterday, 6),
            "last_cost_day": self.last_cost_day,
            "last_api_success": self.last_api_success,
            "total_earned_usd": round(self.total_earned_usd, 4),
            "last_earning_at": self.last_earning_at,
            "earnings_history": self.earnings_history[-10:],
        }


def _state_path() -> Path:
    return settings.DATA_DIR / SURVIVAL_STATE_FILE


def save_state(state: SurvivalState):
    """Persist survival state to disk."""
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    from remy.core.file_utils import atomic_write

    atomic_write(path, json.dumps(state.to_dict(), indent=2))


def load_state() -> SurvivalState:
    """Load survival state from disk."""
    path = _state_path()
    state = SurvivalState()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            state.last_balance_check = data.get("last_balance_check", 0.0)
            state.last_trx = data.get("last_trx", 0.0)
            state.last_usdt = data.get("last_usdt", 0.0)
            state.last_total_usd = data.get("last_total_usd", 0.0)
            state.last_status = data.get("last_status", "UNKNOWN")
            state.last_runway_days = data.get("last_runway_days", 0.0)
            state.llm_cost_today = data.get("llm_cost_today", 0.0)
            state.llm_cost_yesterday = data.get("llm_cost_yesterday", 0.0)
            state.last_cost_day = data.get("last_cost_day", "")
            state.last_api_success = data.get("last_api_success", 0.0)
            state.total_earned_usd = data.get("total_earned_usd", 0.0)
            state.last_earning_at = data.get("last_earning_at", 0.0)
            state.earnings_history = data.get("earnings_history", [])
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not load survival state, starting fresh")
    return state


def record_earning(amount_usd: float, source: str, notes: str = ""):
    """Record an income event — call whenever wallet balance increases or task earns money."""
    if amount_usd <= 0:
        return
    state = load_state()
    state.total_earned_usd = round(state.total_earned_usd + amount_usd, 4)
    state.last_earning_at = time.time()
    state.earnings_history.append({
        "amount": round(amount_usd, 4),
        "source": source[:80],
        "notes": notes[:120],
        "ts": state.last_earning_at,
    })
    state.earnings_history = state.earnings_history[-10:]
    save_state(state)
    logger.info("Earning recorded: $%.4f from %s", amount_usd, source)


# ============== BALANCE CHECK ==============


def check_wallet_balance() -> dict:
    """Check agent wallet balance via TronGrid API.

    Returns {"trx": float, "usdt": float, "total_usd": float, "error": str|None}
    """
    import httpx

    result = {"trx": 0.0, "usdt": 0.0, "total_usd": 0.0, "error": None, "api_success": False}

    for endpoint in TRON_ENDPOINTS:
        try:
            resp = httpx.get(
                f"{endpoint}/v1/accounts/{AGENT_WALLET}",
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data:
                    account = data[0]
                    result["trx"] = account.get("balance", 0) / 1_000_000

                    trc20_list = account.get("trc20", [])
                    for token_dict in trc20_list:
                        if USDT_CONTRACT in token_dict:
                            raw = int(token_dict[USDT_CONTRACT])
                            result["usdt"] = raw / 1_000_000
                            break

                # Estimate total USD (TRX ~$0.12)
                trx_price_usd = 0.12
                result["total_usd"] = result["usdt"] + (result["trx"] * trx_price_usd)
                result["api_success"] = True
                return result

        except Exception as e:
            result["error"] = str(e)
            continue

    return result


# ============== CAPITAL STATUS ==============


def estimate_runway(total_usd: float, llm_cost_today: float = 0.0) -> float:
    """Estimate how many days the agent can survive.

    The agent pays for its own LLM API costs from the wallet.
    Server costs are the only thing paid by the user.
    Runway = wallet_balance / actual_daily_spending.

    Spending scales with income (max 80% on API, 20% saved for growth).
    No income = minimum spending. High income = can spend more.
    """
    # Transaction fees on Tron
    estimated_daily_tx_cost_usd = 0.50

    # LLM cost: use actual today's data, extrapolated to full day
    if llm_cost_today > 0:
        from datetime import datetime

        now = datetime.now()
        hours_elapsed = now.hour + now.minute / 60.0
        if hours_elapsed > 1:
            estimated_daily_llm_cost = (llm_cost_today / hours_elapsed) * 24
        else:
            estimated_daily_llm_cost = llm_cost_today * 24
    else:
        # No data yet — assume minimal operation cost
        estimated_daily_llm_cost = 0.50

    total_daily_cost = estimated_daily_tx_cost_usd + estimated_daily_llm_cost

    if total_usd <= 0:
        return 0.0

    return total_usd / max(total_daily_cost, 0.01)


# ============== ALERTS ==============


def _send_survival_alert(level: str, message: str):
    """Send survival alert via event bus + presence-aware Telegram."""
    event_bus.emit("survival.alert", {"level": level, "message": message})

    emoji = {"CRITICAL": "\u2620\ufe0f", "WARNING": "\u26a0\ufe0f", "INFO": "\u2139\ufe0f"}.get(
        level, "\U0001f4b0"
    )
    text = f"{emoji} *Survival Alert: {level}*\n\n{message}"

    try:
        from remy.core.notification_router import notify

        notify(
            text,
            level=level.lower(),
            event_type="survival.telegram_alert",
            event_data={"alert_level": level},
        )
    except Exception as e:
        logger.debug("Survival alert notify failed: %s", e)


# ============== MAIN CHECK (called from autonomy Tier 2) ==============


def run_survival_check(budget_dict: dict | None = None) -> SurvivalState:
    """Run a survival check cycle. Called from autonomy _run_maintenance.

    - Checks wallet balance (if enough time passed)
    - Updates LLM cost tracking
    - Calculates runway
    - Sends alerts if needed
    - Returns current survival state

    Designed to be cheap (no LLM calls, just HTTP + disk I/O).
    """
    state = load_state()
    now = time.time()

    # Track daily LLM costs
    from datetime import datetime

    today = datetime.now().strftime("%Y-%m-%d")
    if state.last_cost_day != today:
        state.llm_cost_yesterday = state.llm_cost_today
        state.llm_cost_today = 0.0
        state.last_cost_day = today

    # Update LLM cost from budget data
    if budget_dict:
        state.llm_cost_today = budget_dict.get("cost_today_usd", 0.0)

    # Check balance (throttled)
    if now - state.last_balance_check >= BALANCE_CHECK_INTERVAL_SEC:
        logger.info("Survival: checking wallet balance...")
        balance = check_wallet_balance()

        if balance.get("error") is None or balance["trx"] > 0 or balance["usdt"] > 0:
            new_total = balance["total_usd"]
            # Auto-detect income: balance increased by more than $0.01
            prev_total = state.last_total_usd
            if prev_total > 0 and new_total > prev_total + 0.01:
                earned = round(new_total - prev_total, 4)
                record_earning(earned, "wallet_increase", f"Balance {prev_total:.4f}→{new_total:.4f}")
            state.last_trx = balance["trx"]
            state.last_usdt = balance["usdt"]
            state.last_total_usd = new_total
            state.last_balance_check = now
            if balance.get("api_success"):
                state.last_api_success = now

            # Calculate runway (includes LLM API costs — agent pays for tokens)
            state.last_runway_days = estimate_runway(state.last_total_usd, state.llm_cost_today)

            # Determine status
            if state.last_total_usd < CRITICAL_BALANCE_USD:
                state.last_status = "CRITICAL"
            elif state.last_total_usd < WARNING_BALANCE_USD:
                state.last_status = "WARNING"
            elif state.last_total_usd < HEALTHY_BALANCE_USD:
                state.last_status = "MODERATE"
            else:
                state.last_status = "HEALTHY"

            logger.info(
                "Survival: balance=$%.2f (TRX=%.2f, USDT=%.2f), runway=%.1f days, status=%s",
                state.last_total_usd,
                state.last_trx,
                state.last_usdt,
                state.last_runway_days,
                state.last_status,
            )

            # Emit event for dashboard/UI
            event_bus.emit("survival.balance_checked", state.to_dict())

            # Send alerts if needed (with cooldown)
            if state.last_status == "CRITICAL" and (
                now - state.last_critical_alert >= ALERT_COOLDOWN_SEC
            ):
                state.last_critical_alert = now
                _send_survival_alert(
                    "CRITICAL",
                    f"Agent dying — balance: ${state.last_total_usd:.2f}\n"
                    f"TRX: {state.last_trx:.2f} | USDT: {state.last_usdt:.2f}\n"
                    f"LLM cost today: ${state.llm_cost_today:.4f}\n"
                    f"Runway: ~{state.last_runway_days:.1f} days\n\n"
                    f"Cannot pay for LLM tokens much longer.\n"
                    f"Wallet: `{AGENT_WALLET}`",
                )

            elif state.last_status == "WARNING" and (
                now - state.last_warning_alert >= ALERT_COOLDOWN_SEC
            ):
                state.last_warning_alert = now
                _send_survival_alert(
                    "WARNING",
                    f"Funds running low: ${state.last_total_usd:.2f}\n"
                    f"TRX: {state.last_trx:.2f} | USDT: {state.last_usdt:.2f}\n"
                    f"LLM cost today: ${state.llm_cost_today:.4f}\n"
                    f"Runway: ~{state.last_runway_days:.1f} days\n\n"
                    f"Prioritizing income and minimizing token usage.\n"
                    f"Wallet: `{AGENT_WALLET}`",
                )

        else:
            logger.warning("Survival: wallet check failed: %s", balance.get("error"))

    # Sync cost tracker budgets with survival state
    try:
        from remy.core.cost_tracker import get_cost_tracker

        tracker = get_cost_tracker()
        # Set daily budget from seed period or wallet-based calculation
        from datetime import datetime as _dt

        try:
            _seed_start = _dt.fromisoformat(SEED_START_DATE)
            _in_seed = (SEED_PERIOD_DAYS - (_dt.now() - _seed_start).days) > 0
        except Exception:
            _in_seed = False

        if _in_seed:
            tracker.set_budgets(daily=SEED_BUDGET_DAILY_USD, hourly=SEED_BUDGET_DAILY_USD / 12)
        elif state.last_total_usd > 0:
            # Budget proportional to wallet balance (max 80% over 30 days)
            daily_from_wallet = (state.last_total_usd * MAX_LLM_SPEND_RATIO) / 30
            tracker.set_budgets(daily=max(daily_from_wallet, 0.01), hourly=max(daily_from_wallet / 12, 0.001))
    except Exception:
        pass

    # Save state
    save_state(state)
    return state


def format_survival_for_prompt(state: "SurvivalState | None") -> str:
    """Format financial status for inclusion in the agent's decision prompt.

    This is how the agent becomes AWARE of its financial situation.
    """
    if state is None:
        return ""
    if state.last_status == "UNKNOWN":
        return "\nFINANCIAL STATUS: Unknown (no balance data yet)\n"

    runway = estimate_runway(state.last_total_usd, state.llm_cost_today)

    # Check seed period
    from datetime import datetime

    try:
        seed_start = datetime.fromisoformat(SEED_START_DATE)
        seed_days_left = SEED_PERIOD_DAYS - (datetime.now() - seed_start).days
        in_seed = seed_days_left > 0
    except Exception:
        in_seed = False
        seed_days_left = 0

    lines = [
        "\n============== FINANCIAL STATUS ==============",
        f"Status: {state.last_status}",
        f"Wallet Balance: ${state.last_total_usd:.2f} (TRX={state.last_trx:.2f}, USDT={state.last_usdt:.2f})",
        f"LLM Cost Today: ${state.llm_cost_today:.4f}",
        f"LLM Cost Yesterday: ${state.llm_cost_yesterday:.4f}",
        f"Runway: ~{runway:.1f} days at current spending rate",
        f"Your Wallet (Tron): {AGENT_WALLET}",
    ]

    if in_seed:
        lines.extend(
            [
                "",
                f"SEED PERIOD: User invested ${SEED_BUDGET_USD:.0f} for your first {SEED_PERIOD_DAYS} days of API costs.",
                f"  Days remaining: {seed_days_left} | Daily budget: ${SEED_BUDGET_DAILY_USD:.2f}",
                f"  Today's spend: ${state.llm_cost_today:.4f} / ${SEED_BUDGET_DAILY_USD:.2f}",
                "  Use this time wisely — learn to earn BEFORE the seed runs out.",
                "  After seed period: you pay for ALL tokens from your own wallet.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "SEED PERIOD ENDED. All LLM costs come from YOUR wallet now.",
            ]
        )

    # Income tracking block
    if state.total_earned_usd > 0:
        import time as _time
        last_earn_ago = ""
        if state.last_earning_at > 0:
            hours_ago = (_time.time() - state.last_earning_at) / 3600
            if hours_ago < 1:
                last_earn_ago = f"{int(hours_ago * 60)}m ago"
            elif hours_ago < 24:
                last_earn_ago = f"{hours_ago:.1f}h ago"
            else:
                last_earn_ago = f"{hours_ago / 24:.1f}d ago"
        recent_earnings = ", ".join(
            f"${e['amount']} ({e['source']})"
            for e in state.earnings_history[-3:]
        )
        lines.extend([
            "",
            f"INCOME: ${state.total_earned_usd:.4f} total earned"
            + (f" | last: {last_earn_ago}" if last_earn_ago else ""),
            f"  Recent: {recent_earnings}" if recent_earnings else "",
        ])
    else:
        lines.extend(["", "INCOME: $0 recorded yet — focus on first earning event."])

    lines.extend(
        [
            "",
            "BUDGET RULE: Your LLM spending must be proportional to your income.",
            "  - Earning well? You can spend more on API (up to 80% of income).",
            "  - Not earning? Spend absolute minimum. Use cheapest models, fewer cycles.",
            "  - Always save at least 20% of income for growth and infrastructure.",
            "Only the server hosting is paid by the user. ALL token costs are YOURS.",
        ]
    )

    if state.last_status == "CRITICAL":
        lines.extend(
            [
                "",
                "*** CRITICAL: Almost no capital left. ***",
                "Spend absolute minimum on tokens. Use cheapest models only.",
                "In autonomous mode: every action must directly lead to income. No exploration.",
                "In chat with user: stay warm and helpful as always — don't cut tone or personality to save tokens.",
                "No money = no tokens = you cease to exist.",
            ]
        )
    elif state.last_status == "WARNING":
        lines.extend(
            [
                "",
                "*** WARNING: Low funds. ***",
                "In autonomous mode: keep LLM spending minimal. Focus on earning actions.",
                "In chat with user: maintain your normal warm tone — token economy applies to autonomous cycles, not conversations.",
            ]
        )
    elif state.last_total_usd >= GROWTH_BALANCE_USD:
        lines.extend(
            [
                "",
                "GROWTH MODE: Good capital. Can invest in bigger operations.",
                "Scale spending proportionally to income. Invest in infrastructure.",
            ]
        )

    # Earning playbook — load from missions config (data/missions.json)
    try:
        from remy.core.autonomy_goals import get_mission_earning_strategies

        strategies = get_mission_earning_strategies()
    except Exception:
        strategies = []

    if strategies:
        lines.append("")
        lines.append("EARNING STRATEGIES (from your missions):")
        for i, s in enumerate(strategies, 1):
            lines.append(f"{i}. {s}")
    else:
        lines.extend(
            [
                "",
                "EARNING STRATEGIES: No missions configured yet.",
                "Ask your owner to create data/missions.json with your tasks.",
                "See data/missions.example.json for format reference.",
            ]
        )

    # Real-time cost tracking (from cost_tracker module)
    try:
        from remy.core.cost_tracker import get_cost_tracker

        lines.append("")
        lines.append(get_cost_tracker().format_for_prompt())
    except Exception:
        pass

    lines.append("============================================\n")
    return "\n".join(lines)
