from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from remy.web.routes.todos_routes import router


def _client_with_goals(goal_records):
    app = FastAPI()
    app.include_router(router, prefix="/api")
    fake_api = SimpleNamespace(
        brain_lock=nullcontext(),
        brain=SimpleNamespace(
            search=lambda query, tags, limit=50: goal_records if "autonomous-goal" in tags else []
        ),
    )
    return app, fake_api


def test_list_todos_includes_blocked_external_goal():
    rec = SimpleNamespace(
        id="goal-1",
        content="Goal [HIGH]: Register at example.com",
        metadata={
            "type": "autonomous_goal",
            "status": "blocked_external",
            "priority": "high",
            "goal_type": "signup",
            "goal_template": "signup_operator",
            "blocked_reason": "email verification required",
            "resume_context": "Continue from dashboard after verification",
            "attempts": 2,
            "created_at": "2026-03-07T10:00:00",
        },
    )
    app, fake_api = _client_with_goals([rec])
    with patch("remy.web.routes.todos_routes._get_api", return_value=fake_api):
        client = TestClient(app)
        resp = client.get("/api/todos?status=active&category=agent")

    data = resp.json()
    assert resp.status_code == 200
    assert data["total"] == 1
    assert data["todos"][0]["raw_status"] == "blocked_external"
    assert data["todos"][0]["status"] == "in_progress"
    assert data["todos"][0]["blocked_reason"] == "email verification required"


def test_list_todos_includes_scheduled_reminder():
    reminder = SimpleNamespace(
        id="rem-1",
        content="Scheduled: Monitor social media | Due: 2026-03-11T10:00:00Z | Cron: 0 10 * * *",
        metadata={
            "type": "scheduled_task",
            "description": "Monitor social media",
            "status": "active",
            "due_date": "2026-03-11T10:00:00Z",
            "cron": "0 10 * * *",
            "repeat": "daily",
            "source": "agent-interactive",
            "timestamp": "2026-03-10T12:16:28.913187+00:00",
        },
    )
    app = FastAPI()
    app.include_router(router, prefix="/api")
    fake_api = SimpleNamespace(
        brain_lock=nullcontext(),
        brain=SimpleNamespace(
            search=lambda query, tags, limit=50: [reminder] if "scheduled-task" in tags else []
        ),
    )
    with patch("remy.web.routes.todos_routes._get_api", return_value=fake_api):
        client = TestClient(app)
        resp = client.get("/api/todos?status=active")

    data = resp.json()
    assert resp.status_code == 200
    assert data["total"] == 1
    assert data["todos"][0]["source"] == "reminder"
    assert data["todos"][0]["cron"] == "0 10 * * *"
    assert data["todos"][0]["repeat"] == "daily"
