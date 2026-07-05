from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_client():
    from remy.web.routes.pricing_routes import router

    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def test_usage_cost_uses_shared_operator_snapshot(monkeypatch):
    client = _make_client()
    monkeypatch.setattr(
        "remy.core.combined_runner.get_budget_runtime_snapshot",
        lambda goal_limit=5, approval_limit=10: {
            "daily_cost_limit_usd": 3.5,
            "cost_today_usd": 0.82,
            "llm_cost_lifetime_usd": 12.34,
        },
    )
    monkeypatch.setattr(
        "remy.core.usage_stats.usage_tracker.get_stats",
        lambda: {
            "user_cost_usd": 1.0,
            "autonomy_cost_usd": 2.5,
            "user_tokens": 100,
            "autonomy_tokens": 250,
        },
    )

    res = client.get("/api/usage-cost")

    assert res.status_code == 200
    data = res.json()
    assert data["totals"]["user_cost_usd"] == 1.0
    assert data["totals"]["autonomy_cost_usd"] == 2.5
    assert data["autonomy_budget"] == {
        "cost_today_usd": 0.82,
        "total_cost_lifetime_usd": 12.34,
        "daily_cost_limit_usd": 3.5,
    }


def test_usage_cost_keeps_daily_limit_without_budget_snapshot(monkeypatch):
    client = _make_client()
    monkeypatch.setattr(
        "remy.core.combined_runner.get_budget_runtime_snapshot",
        lambda goal_limit=5, approval_limit=10: {"daily_cost_limit_usd": 4.25},
    )
    monkeypatch.setattr("remy.core.usage_stats.usage_tracker.get_stats", lambda: {})

    res = client.get("/api/usage-cost")

    assert res.status_code == 200
    assert res.json()["autonomy_budget"] == {"daily_cost_limit_usd": 4.25}
