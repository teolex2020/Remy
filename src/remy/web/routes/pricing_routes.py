"""
Pricing routes — model pricing, USD cost breakdown, daily limit.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("WebAPI")

router = APIRouter()


class PricingUpdatePayload(BaseModel):
    model: str
    input_cost_per_1m_tokens: float
    output_cost_per_1m_tokens: float


class DailyCostLimitPayload(BaseModel):
    daily_cost_limit_usd: float


@router.get("/pricing")
async def get_pricing():
    """Return all model prices and cost summary."""
    from remy.core.pricing import pricing_registry
    from remy.core.usage_stats import usage_tracker

    prices = pricing_registry.get_all_prices()
    stats = usage_tracker.get_stats()

    return {
        "models": prices,
        "cost_summary": {
            "user_cost_usd": stats.get("user_cost_usd", 0.0),
            "autonomy_cost_usd": stats.get("autonomy_cost_usd", 0.0),
            "total_cost_usd": stats.get("user_cost_usd", 0.0) + stats.get("autonomy_cost_usd", 0.0),
            "user_tokens": stats.get("user_tokens", 0),
            "autonomy_tokens": stats.get("autonomy_tokens", 0),
        },
    }


@router.put("/pricing")
async def update_pricing(payload: PricingUpdatePayload):
    """Update pricing for a model (saves to data/pricing.json)."""
    from remy.core.pricing import pricing_registry

    if payload.input_cost_per_1m_tokens < 0 or payload.output_cost_per_1m_tokens < 0:
        raise HTTPException(status_code=400, detail="Costs cannot be negative")

    pricing_registry.update_price(
        payload.model,
        payload.input_cost_per_1m_tokens,
        payload.output_cost_per_1m_tokens,
    )
    return {
        "updated": payload.model,
        "input_cost_per_1m_tokens": payload.input_cost_per_1m_tokens,
        "output_cost_per_1m_tokens": payload.output_cost_per_1m_tokens,
    }


@router.delete("/pricing/{model_name:path}")
async def delete_pricing(model_name: str):
    """Remove a user override for a model."""
    from remy.core.pricing import pricing_registry

    if pricing_registry.delete_price(model_name):
        return {"deleted": model_name}
    raise HTTPException(status_code=404, detail=f"No user override for model '{model_name}'")


@router.get("/usage-cost")
async def get_usage_cost():
    """Detailed cost breakdown."""
    from remy.core.combined_runner import get_budget_runtime_snapshot
    from remy.core.usage_stats import usage_tracker

    stats = usage_tracker.get_stats()
    budget_snapshot = get_budget_runtime_snapshot(goal_limit=5, approval_limit=10)
    budget_info = {"daily_cost_limit_usd": budget_snapshot.get("daily_cost_limit_usd")}
    if budget_snapshot:
        if budget_snapshot.get("cost_today_usd") is not None:
            budget_info["cost_today_usd"] = budget_snapshot.get("cost_today_usd")
        if budget_snapshot.get("llm_cost_lifetime_usd") is not None:
            budget_info["total_cost_lifetime_usd"] = budget_snapshot.get("llm_cost_lifetime_usd")

    return {
        "totals": {
            "user_cost_usd": stats.get("user_cost_usd", 0.0),
            "autonomy_cost_usd": stats.get("autonomy_cost_usd", 0.0),
            "total_cost_usd": stats.get("user_cost_usd", 0.0) + stats.get("autonomy_cost_usd", 0.0),
            "user_tokens": stats.get("user_tokens", 0),
            "autonomy_tokens": stats.get("autonomy_tokens", 0),
        },
        "autonomy_budget": budget_info,
    }


@router.put("/usage-cost/daily-limit")
async def set_daily_cost_limit(payload: DailyCostLimitPayload):
    """Update daily cost limit for autonomous mode."""
    from remy.config.settings import set_runtime_setting, settings

    if payload.daily_cost_limit_usd < 0:
        raise HTTPException(status_code=400, detail="Limit cannot be negative")

    set_runtime_setting(
        "AUTONOMY_DAILY_COST_LIMIT_USD",
        payload.daily_cost_limit_usd,
        target=settings,
    )
    return {
        "daily_cost_limit_usd": payload.daily_cost_limit_usd,
        "note": "Changes apply immediately and are saved to data/runtime_settings.json.",
    }
