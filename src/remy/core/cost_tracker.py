"""
Cost Tracker — real-time per-request cost tracking with GCRA rate limiting.

Tracks every LLM call's token cost in real-time (not post-factum).
Uses a Generic Cell Rate Algorithm (GCRA) style budget enforcer:
  - Per-request cost is computed immediately from token counts + pricing
  - Running totals updated atomically (thread-safe)
  - Budget checks happen BEFORE each call, not after

Integrates with:
  - pricing.py: model → $/M token rates
  - survival.py: daily budget from seed period or wallet
  - autonomy.py: budget reporting in decision prompt
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("CostTracker")


# ── Configuration ─────────────────────────────────────────────────────────

# Budget periods
HOURLY_BUDGET_USD = 0.20  # Default hourly cap ($0.20/hr ≈ $4.80/day)
DAILY_BUDGET_USD = 1.00  # Default daily cap (seed period)

# Burst allowance: allow short bursts up to 3x hourly rate
BURST_MULTIPLIER = 3.0

# Minimum interval between cost flushes to disk (seconds)
FLUSH_INTERVAL_SEC = 60


# ── Data Structures ───────────────────────────────────────────────────────


@dataclass
class CostEntry:
    """A single LLM call cost record."""

    timestamp: float
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    purpose: str  # "chat", "autonomy", "eval", "reflect", "review"
    channel: str  # "desktop", "telegram", "autonomous", etc.


@dataclass
class CostBucket:
    """Rolling cost accumulator for a time period."""

    total_usd: float = 0.0
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    period_start: float = field(default_factory=time.time)

    def add(self, cost: float, input_tokens: int, output_tokens: int) -> None:
        self.total_usd += cost
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def reset(self) -> None:
        self.total_usd = 0.0
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.period_start = time.time()

    def to_dict(self) -> dict:
        return {
            "total_usd": round(self.total_usd, 8),
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "period_start": self.period_start,
        }


class CostTracker:
    """Thread-safe real-time cost tracker with GCRA-style budget enforcement.

    Usage::

        tracker = get_cost_tracker()

        # Before calling LLM:
        allowed, reason = tracker.check_budget(model, estimated_tokens)
        if not allowed:
            # switch to free model or skip

        # After LLM call:
        tracker.record(model, input_tokens, output_tokens, purpose, channel)

        # In decision prompt:
        status = tracker.format_for_prompt()
    """

    def __init__(self, data_dir: Path | None = None):
        self._lock = threading.Lock()
        self._data_dir = data_dir
        self._hourly = CostBucket()
        self._daily = CostBucket()
        self._session = CostBucket()  # since last restart
        self._lifetime_usd: float = 0.0
        self._recent: list[CostEntry] = []  # last 50 calls for debugging
        self._last_flush: float = 0.0
        self._hourly_budget = HOURLY_BUDGET_USD
        self._daily_budget = DAILY_BUDGET_USD

        # Per-model cost breakdown (model → total_usd today)
        self._model_costs: dict[str, float] = {}

        self._load_state()

    # ── Budget configuration ──────────────────────────────────────────

    def set_budgets(self, daily: float | None = None, hourly: float | None = None) -> None:
        """Update budget limits. Called from survival module when budget changes."""
        with self._lock:
            if daily is not None:
                self._daily_budget = daily
            if hourly is not None:
                self._hourly_budget = hourly

    # ── GCRA Budget Check ─────────────────────────────────────────────

    def check_budget(
        self,
        model: str = "",
        estimated_output_tokens: int = 1000,
    ) -> tuple[bool, str]:
        """Check if budget allows a call. Returns (allowed, reason).

        GCRA approach: compares current spend rate against allowed rate.
        Allows bursts up to BURST_MULTIPLIER of hourly budget.
        """
        with self._lock:
            self._roll_periods()

            # Estimate cost of this call
            estimated_cost = self._estimate_cost(model, estimated_output_tokens)

            # Daily hard cap
            if self._daily.total_usd + estimated_cost > self._daily_budget:
                return False, (
                    f"Daily budget exhausted: ${self._daily.total_usd:.4f} / "
                    f"${self._daily_budget:.2f} "
                    f"(+${estimated_cost:.4f} would exceed)"
                )

            # Hourly soft cap with burst allowance
            hourly_burst_limit = self._hourly_budget * BURST_MULTIPLIER
            if self._hourly.total_usd + estimated_cost > hourly_burst_limit:
                return False, (
                    f"Hourly burst limit hit: ${self._hourly.total_usd:.4f} / "
                    f"${hourly_burst_limit:.4f} "
                    f"(burst {BURST_MULTIPLIER}x of ${self._hourly_budget:.2f}/hr)"
                )

            return True, "ok"

    # ── Record ────────────────────────────────────────────────────────

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        purpose: str = "general",
        channel: str = "unknown",
    ) -> float:
        """Record a completed LLM call's cost. Returns the cost in USD."""
        cost = self._calculate_cost(model, input_tokens, output_tokens)

        with self._lock:
            self._roll_periods()

            self._hourly.add(cost, input_tokens, output_tokens)
            self._daily.add(cost, input_tokens, output_tokens)
            self._session.add(cost, input_tokens, output_tokens)
            self._lifetime_usd += cost

            # Per-model breakdown
            self._model_costs[model] = self._model_costs.get(model, 0.0) + cost

            # Recent log
            entry = CostEntry(
                timestamp=time.time(),
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                purpose=purpose,
                channel=channel,
            )
            self._recent.append(entry)
            if len(self._recent) > 50:
                self._recent = self._recent[-50:]

            # Periodic flush
            now = time.time()
            if now - self._last_flush >= FLUSH_INTERVAL_SEC:
                self._flush_state()
                self._last_flush = now

        if cost > 0:
            logger.debug(
                "Cost: $%.6f (%s, %d in / %d out, %s)",
                cost, model, input_tokens, output_tokens, purpose,
            )

        return cost

    # ── Reporting ─────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Current cost tracking status for APIs/dashboard."""
        with self._lock:
            self._roll_periods()
            return {
                "hourly": self._hourly.to_dict(),
                "daily": self._daily.to_dict(),
                "session": self._session.to_dict(),
                "lifetime_usd": round(self._lifetime_usd, 6),
                "hourly_budget": self._hourly_budget,
                "daily_budget": self._daily_budget,
                "hourly_remaining": round(max(0, self._hourly_budget - self._hourly.total_usd), 6),
                "daily_remaining": round(max(0, self._daily_budget - self._daily.total_usd), 6),
                "model_costs_today": {k: round(v, 6) for k, v in self._model_costs.items()},
            }

    def format_for_prompt(self) -> str:
        """Format cost status for agent decision prompt."""
        status = self.get_status()
        daily = status["daily"]
        hourly = status["hourly"]

        lines = [
            "COST TRACKING (real-time):",
            f"  Today: ${daily['total_usd']:.4f} / ${status['daily_budget']:.2f} "
            f"({daily['total_calls']} calls, "
            f"{daily['total_input_tokens']}+{daily['total_output_tokens']} tokens)",
            f"  This hour: ${hourly['total_usd']:.4f} / ${status['hourly_budget']:.2f}",
            f"  Remaining today: ${status['daily_remaining']:.4f}",
        ]

        # Top spending models
        top_models = sorted(
            status["model_costs_today"].items(),
            key=lambda x: x[1], reverse=True,
        )[:3]
        if top_models:
            model_parts = [f"{m}: ${c:.4f}" for m, c in top_models]
            lines.append(f"  Top models: {', '.join(model_parts)}")

        return "\n".join(lines)

    # ── Internal ──────────────────────────────────────────────────────

    def _roll_periods(self) -> None:
        """Reset hourly/daily buckets when periods expire. Must hold _lock."""
        now = time.time()

        # Hourly reset
        if now - self._hourly.period_start >= 3600:
            self._hourly.reset()

        # Daily reset
        today = datetime.now().strftime("%Y-%m-%d")
        daily_date = datetime.fromtimestamp(self._daily.period_start).strftime("%Y-%m-%d")
        if today != daily_date:
            self._daily.reset()
            self._model_costs.clear()

    def _estimate_cost(self, model: str, estimated_output_tokens: int) -> float:
        """Estimate cost of an upcoming call."""
        try:
            from remy.core.pricing import pricing_registry
            input_rate, output_rate = pricing_registry.get_price(model)
        except Exception:
            input_rate, output_rate = 0.0, 0.0

        # Assume ~500 input tokens for a typical prompt
        estimated_input = 500
        return (
            (estimated_input / 1_000_000) * input_rate
            + (estimated_output_tokens / 1_000_000) * output_rate
        )

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate actual cost from token counts."""
        try:
            from remy.core.pricing import pricing_registry
            return pricing_registry.calculate_cost(model, input_tokens, output_tokens)
        except Exception:
            return 0.0

    # ── Persistence ───────────────────────────────────────────────────

    def _state_path(self) -> Path:
        if self._data_dir:
            return self._data_dir / "cost_tracker.json"
        try:
            from remy.config.settings import settings
            return settings.DATA_DIR / "cost_tracker.json"
        except Exception:
            return Path("data/cost_tracker.json")

    def _load_state(self) -> None:
        """Load persisted daily totals."""
        path = self._state_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._lifetime_usd = data.get("lifetime_usd", 0.0)

            # Restore daily bucket if same day
            today = datetime.now().strftime("%Y-%m-%d")
            if data.get("daily_date") == today:
                d = data.get("daily", {})
                self._daily.total_usd = d.get("total_usd", 0.0)
                self._daily.total_calls = d.get("total_calls", 0)
                self._daily.total_input_tokens = d.get("total_input_tokens", 0)
                self._daily.total_output_tokens = d.get("total_output_tokens", 0)
                self._model_costs = data.get("model_costs", {})
        except Exception as e:
            logger.warning("Failed to load cost tracker state: %s", e)

    def _flush_state(self) -> None:
        """Persist current state to disk. Must hold _lock."""
        path = self._state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "lifetime_usd": round(self._lifetime_usd, 8),
            "daily_date": datetime.now().strftime("%Y-%m-%d"),
            "daily": self._daily.to_dict(),
            "model_costs": {k: round(v, 8) for k, v in self._model_costs.items()},
            "last_flush": datetime.now().isoformat(),
        }
        try:
            from remy.core.file_utils import atomic_write
            atomic_write(path, json.dumps(data, indent=2))
        except Exception as e:
            logger.warning("Failed to flush cost tracker: %s", e)


# ── Module singleton ──────────────────────────────────────────────────────

_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    """Get the module-level CostTracker singleton."""
    global _tracker
    if _tracker is None:
        _tracker = CostTracker()
    return _tracker


def reset_cost_tracker() -> None:
    """Reset singleton (for testing)."""
    global _tracker
    _tracker = None
