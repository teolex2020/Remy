from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from remy.config.settings import settings
from remy.core.project_store import (
    LEGACY_BRAIN_ID,
    LEGACY_PROJECT_ID,
    reset_project_store_for_tests,
)
from remy.core.run_envelope import request_run_stop
from remy.core.worker import WorkerResult, WorkerTask
from remy.core.worker_tasks import (
    get_worker_task,
    launch_worker_tasks,
    resume_worker_tasks,
    shutdown_worker_task_runtime,
)


@pytest.fixture(autouse=True)
def isolated_worker_runtime(tmp_path, monkeypatch):
    shutdown_worker_task_runtime()
    reset_project_store_for_tests()
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(settings, "AURA_BRAIN_PATH", tmp_path / "data" / "brain")
    monkeypatch.setattr(settings, "WORKER_MAX_PARALLEL", 3)
    monkeypatch.setattr(settings, "WORKER_MAX_TOOL_ITERATIONS", 5)
    yield
    shutdown_worker_task_runtime()
    reset_project_store_for_tests()


def _wait_for_terminal(task_id: str, timeout: float = 4.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        handle = get_worker_task(task_id, owner_project_id=LEGACY_PROJECT_ID)
        if handle and handle["status"] in {
            "completed",
            "completed_with_limits",
            "failed",
            "cancelled",
            "interrupted",
            "blocked",
        }:
            return handle
        time.sleep(0.02)
    raise AssertionError(f"Task {task_id} did not reach a terminal state")


def test_background_worker_handle_completes_and_delivers_result(monkeypatch):
    async def fake_execute(tasks, session_id, channel):
        return [
            WorkerResult(
                role=tasks[0].role,
                status="success",
                output="Verified result",
                tool_calls=2,
                elapsed_sec=0.2,
            )
        ]

    monkeypatch.setattr("remy.core.worker.execute_workers", fake_execute)
    handle = launch_worker_tasks(
        [WorkerTask(role="researcher", instruction="Verify the source")],
        session_id="conversation-1",
        channel="web",
    )
    terminal = _wait_for_terminal(handle["task_id"])

    assert terminal["status"] == "completed"
    assert terminal["can_cancel"] is False
    assert terminal["can_resume"] is True
    assert terminal["usage"]["active_workers"] == 0
    assert terminal["usage"]["peak_workers"] == 1
    assert terminal["results"][0]["output"] == "Verified result"

    from remy.core.execution_ledger import get_execution_ledger

    messages = get_execution_ledger().consume_continuations(
        "conversation-1",
        owner_project_id=LEGACY_PROJECT_ID,
        brain_id=LEGACY_BRAIN_ID,
    )
    assert len(messages) == 1
    assert handle["task_id"] in messages[0]["content"]
    assert "Verified result" in messages[0]["content"]


def test_background_worker_handle_can_be_cancelled(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    async def slow_execute(tasks, session_id, channel):
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.01)
        return [WorkerResult(role=tasks[0].role, status="success", output="late")]

    monkeypatch.setattr("remy.core.worker.execute_workers", slow_execute)
    handle = launch_worker_tasks(
        [WorkerTask(role="researcher", instruction="Wait")],
        session_id="conversation-2",
        channel="web",
    )
    assert started.wait(2.0)
    stopping = request_run_stop(
        handle["task_id"],
        owner_project_id=LEGACY_PROJECT_ID,
        reason="Operator cancelled",
    )
    assert stopping["status"] == "stopping"
    release.set()

    terminal = _wait_for_terminal(handle["task_id"])
    assert terminal["status"] == "cancelled"
    assert terminal["stop_reason"] == "user_stopped"
    assert terminal["usage"]["active_workers"] == 0


def test_resume_links_new_handle_and_guards_executor_side_effects(monkeypatch):
    async def fake_execute(tasks, session_id, channel):
        return [WorkerResult(role=tasks[0].role, status="success", output="done")]

    monkeypatch.setattr("remy.core.worker.execute_workers", fake_execute)
    read_only = launch_worker_tasks(
        [WorkerTask(role="analyst", instruction="Analyse")],
        session_id="conversation-3",
        channel="web",
    )
    _wait_for_terminal(read_only["task_id"])
    resumed = resume_worker_tasks(
        read_only["task_id"],
        owner_project_id=LEGACY_PROJECT_ID,
    )
    assert resumed["task_id"] != read_only["task_id"]
    assert resumed["resumed_from"] == read_only["task_id"]
    _wait_for_terminal(resumed["task_id"])

    executor = launch_worker_tasks(
        [WorkerTask(role="executor", instruction="Write a file")],
        session_id="conversation-4",
        channel="web",
    )
    _wait_for_terminal(executor["task_id"])
    with pytest.raises(PermissionError, match="explicit confirmation"):
        resume_worker_tasks(
            executor["task_id"],
            owner_project_id=LEGACY_PROJECT_ID,
        )


def test_delegate_task_background_returns_handle(monkeypatch):
    from remy.core.tool_handlers.delegate import _handle_delegate_task

    monkeypatch.setattr(
        "remy.core.worker_tasks.launch_worker_tasks",
        lambda tasks, **kwargs: {"task_id": "run-worker-1", "status": "running"},
    )
    payload = json.loads(
        _handle_delegate_task(
            {
                "background": True,
                "tasks": [{"role": "researcher", "instruction": "Investigate"}],
            },
            "conversation-5",
            "desktop",
        )
    )
    assert payload["background"] is True
    assert payload["task"]["task_id"] == "run-worker-1"


def test_worker_task_api_launches_a_project_scoped_handle(monkeypatch):
    from remy.web.routes import run_routes

    captured = {}

    def fake_launch(tasks, **kwargs):
        captured["tasks"] = tasks
        captured.update(kwargs)
        return {"task_id": "run-api-1", "status": "running"}

    monkeypatch.setattr(run_routes, "current_project_id", lambda: LEGACY_PROJECT_ID)
    monkeypatch.setattr("remy.core.worker_tasks.launch_worker_tasks", fake_launch)
    app = FastAPI()
    app.include_router(run_routes.router, prefix="/api")
    response = TestClient(app).post(
        "/api/tasks/workers",
        json={
            "conversation_id": "conversation-api",
            "tasks": [
                {
                    "role": "analyst",
                    "instruction": "Inspect metrics",
                    "context": "Project context",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["task"]["task_id"] == "run-api-1"
    assert captured["owner_project_id"] == LEGACY_PROJECT_ID
    assert captured["session_id"] == "conversation-api"
    assert captured["tasks"][0].role == "analyst"
