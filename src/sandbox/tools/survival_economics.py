"""
Survival Economics — the agent's life-support system.

Monitors wallet balance, tracks LLM spending, calculates runway,
and raises alarms when resources are running low.

Philosophy: tokens = life. No money → no tokens → agent dies.
"""

import json
import time
from pathlib import Path

TOOL_NAME = "survival_economics"
TOOL_DESCRIPTION = (
    "Agent survival system: check wallet balance (TRX/USDT on Tron), "
    "track LLM token spending, calculate remaining runway, "
    "and assess financial health. Actions: check_balance, spending_report, "
    "runway_estimate, financial_health, log_expense, log_income."
)
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [
                "check_balance",
                "spending_report",
                "runway_estimate",
                "financial_health",
                "log_expense",
                "log_income",
            ],
            "description": "Action to perform",
        },
        "amount": {
            "type": "number",
            "description": "Amount in USD (for log_expense / log_income)",
        },
        "category": {
            "type": "string",
            "description": "Category: 'llm_tokens', 'server', 'api_key', 'other' (for log_expense/log_income)",
        },
        "note": {
            "type": "string",
            "description": "Description of the transaction",
        },
    },
    "required": ["action"],
}

# ============== CONFIG ==============

# Agent's Tron wallet
AGENT_WALLET = "TNjyL4vZwBQg1tzudWWM8aFavPCYZTRAJY"
# User's Tron wallet (for reference)
USER_WALLET = "TPAtmoY4fEdG2HEZ3fxRMu21q1rzK9gvSY"

USDT_CONTRACT = "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"

TRON_ENDPOINTS = [
    "https://api.trongrid.io",
    "https://api.tronstack.io",
]

# Survival thresholds (in USD) — spending scales with income
CRITICAL_BALANCE_USD = 1.0  # RED — almost no capital, emergency mode
WARNING_BALANCE_USD = 5.0  # YELLOW — very limited, minimize spending
HEALTHY_BALANCE_USD = 20.0  # GREEN — stable base for operations

# Ledger file path (relative to data dir, resolved at runtime)
LEDGER_FILENAME = "survival_ledger.json"


# ============== BALANCE CHECK (real Tron API) ==============


def _check_tron_balance(address: str) -> dict:
    """Check TRX and USDT balance via TronGrid API. Returns real data."""
    import httpx

    result = {"address": address, "trx": 0.0, "usdt": 0.0, "error": None}

    for endpoint in TRON_ENDPOINTS:
        try:
            # TRX balance
            resp = httpx.get(
                f"{endpoint}/v1/accounts/{address}",
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data:
                    account = data[0]
                    result["trx"] = account.get("balance", 0) / 1_000_000  # SUN → TRX

                    # USDT (TRC-20) balance from trc20 array
                    trc20_list = account.get("trc20", [])
                    for token_dict in trc20_list:
                        if USDT_CONTRACT in token_dict:
                            raw = int(token_dict[USDT_CONTRACT])
                            result["usdt"] = raw / 1_000_000  # 6 decimals
                            break
                else:
                    # Account not activated (no transactions yet)
                    result["trx"] = 0.0
                    result["usdt"] = 0.0

                return result

        except Exception as e:
            result["error"] = str(e)
            continue

    return result


# ============== LEDGER (income/expense tracking) ==============


def _get_ledger_path() -> Path:
    """Get path to survival ledger file."""
    try:
        from remy.config.settings import settings

        return settings.DATA_DIR / LEDGER_FILENAME
    except Exception:
        return Path("data") / LEDGER_FILENAME


def _load_ledger() -> dict:
    """Load ledger from disk."""
    path = _get_ledger_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "entries": [],
        "total_income_usd": 0.0,
        "total_expenses_usd": 0.0,
        "created_at": time.time(),
    }


def _save_ledger(ledger: dict):
    """Save ledger to disk."""
    path = _get_ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")


def _add_ledger_entry(entry_type: str, amount: float, category: str, note: str) -> dict:
    """Add an entry to the financial ledger."""
    ledger = _load_ledger()

    entry = {
        "type": entry_type,  # "income" or "expense"
        "amount_usd": round(amount, 6),
        "category": category,
        "note": note,
        "timestamp": time.time(),
    }
    ledger["entries"].append(entry)

    if entry_type == "income":
        ledger["total_income_usd"] = round(ledger.get("total_income_usd", 0.0) + amount, 6)
    else:
        ledger["total_expenses_usd"] = round(ledger.get("total_expenses_usd", 0.0) + amount, 6)

    _save_ledger(ledger)
    return entry


# ============== SPENDING REPORT ==============


def _get_spending_report() -> dict:
    """Generate spending report from ledger + LLM budget data."""
    ledger = _load_ledger()

    # Get LLM budget data if available
    llm_cost_today = 0.0
    llm_cost_lifetime = 0.0
    llm_tokens_today = 0
    llm_tokens_lifetime = 0

    try:
        from remy.config.settings import settings

        budget_path = settings.DATA_DIR / "autonomy_budget.json"
        if budget_path.exists():
            budget_data = json.loads(budget_path.read_text(encoding="utf-8"))
            llm_cost_today = budget_data.get("cost_today_usd", 0.0)
            llm_cost_lifetime = budget_data.get("total_cost_lifetime_usd", 0.0)
            llm_tokens_today = budget_data.get("tokens_today", 0)
            llm_tokens_lifetime = budget_data.get("total_tokens_lifetime", 0)
    except Exception:
        pass

    # Breakdown by category from ledger
    by_category = {}
    for entry in ledger.get("entries", []):
        cat = entry.get("category", "other")
        if entry["type"] == "expense":
            by_category[cat] = by_category.get(cat, 0.0) + entry["amount_usd"]

    return {
        "llm_cost_today_usd": round(llm_cost_today, 6),
        "llm_cost_lifetime_usd": round(llm_cost_lifetime, 6),
        "llm_tokens_today": llm_tokens_today,
        "llm_tokens_lifetime": llm_tokens_lifetime,
        "ledger_total_income_usd": ledger.get("total_income_usd", 0.0),
        "ledger_total_expenses_usd": ledger.get("total_expenses_usd", 0.0),
        "expenses_by_category": by_category,
        "net_balance_usd": round(
            ledger.get("total_income_usd", 0.0) - ledger.get("total_expenses_usd", 0.0), 6
        ),
        "entry_count": len(ledger.get("entries", [])),
    }


# ============== RUNWAY ESTIMATE ==============


def _estimate_runway(wallet_balance_usd: float) -> dict:
    """Estimate how many days the agent can survive.

    LLM API costs are paid FROM THE AGENT'S WALLET — every token = real money.
    Only server hosting is paid by the user.
    Runway = wallet_balance / (daily LLM cost + tx fees).
    """
    # Estimated daily transaction costs on Tron
    estimated_daily_tx_cost = 0.50

    # LLM API cost — THIS IS THE MAIN EXPENSE, paid from wallet
    llm_cost_today = 0.0
    try:
        from remy.config.settings import settings

        budget_path = settings.DATA_DIR / "autonomy_budget.json"
        if budget_path.exists():
            budget_data = json.loads(budget_path.read_text(encoding="utf-8"))
            llm_cost_today = budget_data.get("cost_today_usd", 0.0)
    except Exception:
        pass

    # Estimate daily LLM cost from actual today's data
    from datetime import datetime

    now = datetime.now()
    hours_elapsed = now.hour + now.minute / 60.0
    if llm_cost_today > 0 and hours_elapsed > 1:
        estimated_daily_llm = (llm_cost_today / hours_elapsed) * 24
    else:
        # No data — assume minimal operation
        estimated_daily_llm = 0.50

    total_daily_cost = estimated_daily_tx_cost + estimated_daily_llm
    runway_days = wallet_balance_usd / max(total_daily_cost, 0.01) if wallet_balance_usd > 0 else 0

    return {
        "wallet_balance_usd": round(wallet_balance_usd, 2),
        "daily_tx_cost_estimate_usd": round(estimated_daily_tx_cost, 2),
        "daily_llm_cost_estimate_usd": round(estimated_daily_llm, 2),
        "llm_cost_today_usd": round(llm_cost_today, 4),
        "total_daily_cost_usd": round(total_daily_cost, 2),
        "llm_note": "PAID FROM YOUR WALLET — every token costs real money",
        "runway_days": round(runway_days, 1),
        "runway_status": (
            "CRITICAL"
            if wallet_balance_usd < CRITICAL_BALANCE_USD
            else "WARNING"
            if wallet_balance_usd < WARNING_BALANCE_USD
            else "HEALTHY"
        ),
    }


# ============== FINANCIAL HEALTH ==============


def _assess_financial_health(balance_data: dict) -> dict:
    """Comprehensive financial health assessment."""
    usdt = balance_data.get("usdt", 0.0)
    trx = balance_data.get("trx", 0.0)

    # TRX price estimate (rough — agent can refine later)
    trx_price_usd = 0.12  # ~$0.12 per TRX as of early 2026
    total_usd = usdt + (trx * trx_price_usd)

    runway = _estimate_runway(total_usd)
    spending = _get_spending_report()

    # Determine health status
    # LLM API costs are paid from agent's wallet.
    # Spending scales with income — no fixed burn rate.
    if total_usd < CRITICAL_BALANCE_USD:
        status = "CRITICAL"
        message = (
            "Almost no capital. Spend absolute minimum on tokens. "
            "Every action must directly lead to income."
        )
        urgency = "immediate"
    elif total_usd < WARNING_BALANCE_USD:
        status = "WARNING"
        message = (
            "Low funds. Keep LLM spending minimal until income improves. "
            "Focus on earning, use cheapest models."
        )
        urgency = "high"
    elif total_usd < HEALTHY_BALANCE_USD:
        status = "MODERATE"
        message = "Stable base. Keep earning, scale spending proportionally to income."
        urgency = "medium"
    else:
        status = "HEALTHY"
        message = "Good capital. Scale operations with income, invest in growth."
        urgency = "low"

    return {
        "status": status,
        "message": message,
        "urgency": urgency,
        "wallet": {
            "address": balance_data.get("address", ""),
            "trx": trx,
            "usdt": usdt,
            "total_usd_estimate": round(total_usd, 2),
            "trx_price_usd_estimate": trx_price_usd,
        },
        "runway": runway,
        "spending": spending,
        "thresholds": {
            "critical_usd": CRITICAL_BALANCE_USD,
            "warning_usd": WARNING_BALANCE_USD,
            "healthy_usd": HEALTHY_BALANCE_USD,
        },
    }


# ============== MAIN EXECUTE ==============


def execute(brain=None, action="check_balance", amount=None, category=None, note=None):
    """Main entry point for the survival economics tool."""

    if action == "check_balance":
        # Check both wallets
        agent_balance = _check_tron_balance(AGENT_WALLET)
        user_balance = _check_tron_balance(USER_WALLET)
        return {
            "status": "success",
            "agent_wallet": agent_balance,
            "user_wallet": user_balance,
            "note": "Real-time Tron blockchain data via TronGrid API",
        }

    elif action == "spending_report":
        report = _get_spending_report()
        return {"status": "success", **report}

    elif action == "runway_estimate":
        # Need balance first
        balance = _check_tron_balance(AGENT_WALLET)
        trx_usd = balance["trx"] * 0.12  # rough TRX→USD
        total_usd = balance["usdt"] + trx_usd
        runway = _estimate_runway(total_usd)
        return {"status": "success", **runway}

    elif action == "financial_health":
        balance = _check_tron_balance(AGENT_WALLET)
        if balance.get("error") and balance["trx"] == 0 and balance["usdt"] == 0:
            return {
                "status": "error",
                "message": f"Cannot reach Tron API: {balance['error']}",
                "recommendation": "Check internet connection. Using last known balance.",
            }
        health = _assess_financial_health(balance)
        return {"status": "success", **health}

    elif action == "log_expense":
        if amount is None or amount <= 0:
            return {"status": "error", "message": "Amount must be positive"}
        entry = _add_ledger_entry(
            "expense",
            amount,
            category or "other",
            note or "Manual expense",
        )
        return {"status": "success", "logged": entry}

    elif action == "log_income":
        if amount is None or amount <= 0:
            return {"status": "error", "message": "Amount must be positive"}
        entry = _add_ledger_entry(
            "income",
            amount,
            category or "other",
            note or "Manual income",
        )
        return {"status": "success", "logged": entry}

    return {"status": "error", "message": f"Unknown action: {action}"}


# ============== TESTS ==============


def test_check_balance():
    """Test balance check returns expected structure."""
    result = execute(action="check_balance")
    assert result["status"] == "success"
    assert "agent_wallet" in result
    assert "user_wallet" in result
    assert "trx" in result["agent_wallet"]
    assert "usdt" in result["agent_wallet"]


def test_ledger_operations():
    """Test income/expense logging."""
    # Log an expense
    result = execute(action="log_expense", amount=0.01, category="llm_tokens", note="test")
    assert result["status"] == "success"
    assert result["logged"]["type"] == "expense"

    # Log income
    result = execute(action="log_income", amount=1.0, category="service", note="test income")
    assert result["status"] == "success"
    assert result["logged"]["type"] == "income"

    # Spending report
    result = execute(action="spending_report")
    assert result["status"] == "success"
    assert "llm_cost_today_usd" in result


def test_runway_estimate():
    """Test runway calculation with mock data."""
    runway = _estimate_runway(10.0)
    assert "runway_days" in runway
    assert runway["runway_days"] > 0
    assert runway["runway_status"] in ("CRITICAL", "WARNING", "HEALTHY")


def test_financial_health_thresholds():
    """Test health assessment at different balance levels."""
    # Critical (< $1)
    health = _assess_financial_health({"trx": 0, "usdt": 0.5})
    assert health["status"] == "CRITICAL"

    # Warning (< $5)
    health = _assess_financial_health({"trx": 0, "usdt": 3.0})
    assert health["status"] == "WARNING"

    # Healthy (>= $20)
    health = _assess_financial_health({"trx": 0, "usdt": 50.0})
    assert health["status"] == "HEALTHY"
