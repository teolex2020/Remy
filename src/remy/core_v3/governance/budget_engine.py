"""
Budget Engine for Remy v3 Governance Layer.

Wraps v2 survival.py logic into a formal budget governance system.
Tracks daily/mission/cycle budgets and enforces caps.

Phase 5: Enhanced with spending history, model degradation signals,
per-cycle accounting, and seed period awareness.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Budget status
# ---------------------------------------------------------------------------

class BudgetStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    EXHAUSTED = "exhausted"


class BudgetAction(str, Enum):
    ALLOW = "allow"
    DEGRADE = "degrade"            # Switch to cheaper model
    DENY = "deny"                  # Block execution


# ---------------------------------------------------------------------------
# Budget config
# ---------------------------------------------------------------------------

@dataclass
class BudgetConfig:
    """Budget limits (can be overridden per mission)."""
    daily_usd: float = 1.0
    per_mission_usd: float = 0.50
    per_cycle_usd: float = 0.10
    warning_threshold: float = 0.80   # 80% of daily = warning
    critical_threshold: float = 0.95  # 95% = critical

    # From v2 survival.py
    seed_budget_usd: float = 30.0
    seed_period_days: int = 30
    seed_start_date: str = "2026-03-02"
    max_llm_spend_ratio: float = 0.80


# ---------------------------------------------------------------------------
# Spend event
# ---------------------------------------------------------------------------

@dataclass
class SpendEvent:
    """A single spending event for history tracking."""
    timestamp: float = field(default_factory=time.time)
    cost_usd: float = 0.0
    tokens: int = 0
    mission_id: str = ""
    specialist: str = ""
    action: str = ""
    model: str = ""


# ---------------------------------------------------------------------------
# Budget state
# ---------------------------------------------------------------------------

@dataclass
class BudgetState:
    """Current budget consumption state."""
    daily_spent_usd: float = 0.0
    daily_tokens: int = 0
    cycle_spent_usd: float = 0.0
    cycle_tokens: int = 0
    mission_spent: dict[str, float] = field(default_factory=dict)
    mission_tokens: dict[str, int] = field(default_factory=dict)
    last_reset_day: str = ""
    wallet_balance_usd: float = 0.0
    runway_days: float = 0.0

    # Spending history (last 100 events)
    history: list[SpendEvent] = field(default_factory=list)
    history_max: int = 100


# ---------------------------------------------------------------------------
# Budget Engine
# ---------------------------------------------------------------------------

class BudgetEngine:
    """Tracks and enforces budget constraints.

    Adapts v2 survival.py into a formal governance component.
    """

    def __init__(self, config: BudgetConfig | None = None):
        self.config = config or BudgetConfig()
        self.state = BudgetState()
        self._v2_survival = None

    def connect_v2_survival(self):
        """Lazily connect to v2 survival module for wallet data."""
        if self._v2_survival is None:
            try:
                from remy.core import survival
                self._v2_survival = survival
            except ImportError:
                log.warning("v2 survival module not available")

    def sync_from_v2(self):
        """Pull latest state from v2 survival module."""
        self.connect_v2_survival()
        if self._v2_survival is None:
            return

        v2_state = self._v2_survival.load_state()
        self.state.wallet_balance_usd = v2_state.last_total_usd
        self.state.runway_days = v2_state.last_runway_days
        self.state.daily_spent_usd = v2_state.llm_cost_today
        self.state.last_reset_day = v2_state.last_cost_day

    def start_cycle(self):
        """Reset per-cycle counters (call at start of each autonomy cycle)."""
        self.state.cycle_spent_usd = 0.0
        self.state.cycle_tokens = 0

    def record_spend(
        self,
        cost_usd: float,
        tokens: int = 0,
        mission_id: str = "",
        specialist: str = "",
        action: str = "",
        model: str = "",
    ):
        """Record a spend event."""
        # Daily reset check
        today = datetime.now().strftime("%Y-%m-%d")
        if self.state.last_reset_day and self.state.last_reset_day != today:
            self.state.daily_spent_usd = 0.0
            self.state.daily_tokens = 0
            self.state.last_reset_day = today

        self.state.daily_spent_usd += cost_usd
        self.state.daily_tokens += tokens
        self.state.cycle_spent_usd += cost_usd
        self.state.cycle_tokens += tokens

        if mission_id:
            self.state.mission_spent[mission_id] = (
                self.state.mission_spent.get(mission_id, 0.0) + cost_usd
            )
            self.state.mission_tokens[mission_id] = (
                self.state.mission_tokens.get(mission_id, 0) + tokens
            )

        # History
        event = SpendEvent(
            cost_usd=cost_usd, tokens=tokens, mission_id=mission_id,
            specialist=specialist, action=action, model=model,
        )
        self.state.history.append(event)
        if len(self.state.history) > self.state.history_max:
            self.state.history = self.state.history[-self.state.history_max:]

    def check_budget(
        self,
        estimated_cost_usd: float = 0.0,
        mission_id: str = "",
    ) -> tuple[BudgetAction, str]:
        """Check if a planned action is within budget."""
        daily_limit = self.config.daily_usd
        daily_after = self.state.daily_spent_usd + estimated_cost_usd

        # Daily hard cap
        if daily_after > daily_limit:
            return BudgetAction.DENY, (
                f"Daily budget exhausted: ${self.state.daily_spent_usd:.3f} "
                f"+ ${estimated_cost_usd:.3f} > ${daily_limit:.2f}"
            )

        # Per-cycle cap
        cycle_after = self.state.cycle_spent_usd + estimated_cost_usd
        if cycle_after > self.config.per_cycle_usd:
            return BudgetAction.DENY, (
                f"Cycle budget exhausted: ${self.state.cycle_spent_usd:.3f} "
                f"+ ${estimated_cost_usd:.3f} > ${self.config.per_cycle_usd:.2f}"
            )

        # Mission cap
        if mission_id:
            mission_after = (
                self.state.mission_spent.get(mission_id, 0.0) + estimated_cost_usd
            )
            if mission_after > self.config.per_mission_usd:
                return BudgetAction.DENY, (
                    f"Mission {mission_id} budget exceeded: "
                    f"${mission_after:.3f} > ${self.config.per_mission_usd:.2f}"
                )

        # Warning zone → degrade to cheaper model
        usage_ratio = daily_after / daily_limit if daily_limit > 0 else 1.0
        if usage_ratio >= self.config.critical_threshold:
            return BudgetAction.DEGRADE, (
                f"Budget critical ({usage_ratio:.0%}), degrading to free model"
            )
        if usage_ratio >= self.config.warning_threshold:
            return BudgetAction.DEGRADE, (
                f"Budget warning ({usage_ratio:.0%}), consider cheaper model"
            )

        return BudgetAction.ALLOW, "within budget"

    def get_status(self) -> BudgetStatus:
        """Overall budget health."""
        if self.config.daily_usd <= 0:
            return BudgetStatus.EXHAUSTED
        ratio = self.state.daily_spent_usd / self.config.daily_usd
        if ratio >= 1.0:
            return BudgetStatus.EXHAUSTED
        if ratio >= self.config.critical_threshold:
            return BudgetStatus.CRITICAL
        if ratio >= self.config.warning_threshold:
            return BudgetStatus.WARNING
        return BudgetStatus.HEALTHY

    def is_seed_period(self) -> bool:
        """Check if we're still in the seed investment period."""
        try:
            start = datetime.strptime(self.config.seed_start_date, "%Y-%m-%d")
            elapsed = (datetime.now() - start).days
            return elapsed < self.config.seed_period_days
        except (ValueError, TypeError):
            return False

    def seed_days_remaining(self) -> int:
        """Days left in seed period (-1 if expired)."""
        try:
            start = datetime.strptime(self.config.seed_start_date, "%Y-%m-%d")
            elapsed = (datetime.now() - start).days
            remaining = self.config.seed_period_days - elapsed
            return max(-1, remaining)
        except (ValueError, TypeError):
            return -1

    def recommended_model(self) -> str:
        """Recommend model based on budget status."""
        status = self.get_status()
        if status == BudgetStatus.EXHAUSTED:
            return "free"
        if status == BudgetStatus.CRITICAL:
            return "free"
        if status == BudgetStatus.WARNING:
            return "cheap"
        return "normal"

    def daily_remaining(self) -> float:
        """USD remaining in today's budget."""
        return max(0.0, self.config.daily_usd - self.state.daily_spent_usd)

    def spending_rate(self, hours: float = 1.0) -> float:
        """USD spent per hour based on recent history."""
        if not self.state.history:
            return 0.0
        cutoff = time.time() - (hours * 3600)
        recent = [e for e in self.state.history if e.timestamp >= cutoff]
        return sum(e.cost_usd for e in recent) / hours if recent else 0.0

    def summary(self) -> dict[str, Any]:
        """Budget summary for observability."""
        return {
            "status": self.get_status().value,
            "daily_spent_usd": round(self.state.daily_spent_usd, 4),
            "daily_limit_usd": self.config.daily_usd,
            "daily_remaining_usd": round(self.daily_remaining(), 4),
            "daily_tokens": self.state.daily_tokens,
            "cycle_spent_usd": round(self.state.cycle_spent_usd, 4),
            "cycle_limit_usd": self.config.per_cycle_usd,
            "wallet_usd": round(self.state.wallet_balance_usd, 2),
            "runway_days": round(self.state.runway_days, 1),
            "seed_period": self.is_seed_period(),
            "seed_days_remaining": self.seed_days_remaining(),
            "recommended_model": self.recommended_model(),
            "spending_rate_per_hour": round(self.spending_rate(), 4),
            "missions": {
                mid: round(cost, 4)
                for mid, cost in self.state.mission_spent.items()
            },
        }
