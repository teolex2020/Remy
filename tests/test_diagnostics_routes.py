from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_client():
    from remy.web.routes.diagnostics import router

    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def test_get_task_metrics_uses_shared_goal_snapshot(monkeypatch):
    client = _make_client()

    with monkeypatch.context() as m:
        m.setattr(
            "remy.core.task_metrics.task_metrics.get_all",
            lambda: {
                "totals": {
                    "runs": 12,
                },
                "families": {},
            },
        )
        m.setattr(
            "remy.core.combined_runner.get_goal_runtime_snapshot",
            lambda goal_limit=5, approval_limit=10: {
                "total": 7,
                "active": 3,
                "blocked": 2,
                "active_list": [],
            },
        )
        response = client.get("/api/task-metrics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["totals"]["runs"] == 12
    assert payload["totals"]["active_goals"] == 3
    assert payload["totals"]["blocked_goals"] == 2
    assert payload["totals"]["total_goals"] == 7


def test_get_harness_eval_history(monkeypatch):
    client = _make_client()

    with monkeypatch.context() as m:
        m.setattr(
            "remy.core.harness_eval_history.get_harness_eval_history_summary",
            lambda limit=20: {
                "total_runs": 6,
                "scenario_counts": {"verify_gate_ablation": 2, "recovery_replay_ablation": 1},
                "status_counts": {"completed": 5, "not_enough_data": 1},
                "latest_entries": [
                    {"id": "verify_gate_ablation", "status": "completed", "summary": "Verify gate prevented false success.", "executed_at": 1}
                ],
            },
        )
        response = client.get("/api/harness-eval-history")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_runs"] == 6
    assert payload["scenario_counts"]["verify_gate_ablation"] == 2
    assert payload["status_counts"]["completed"] == 5
    assert payload["latest_entries"][0]["status"] == "completed"
