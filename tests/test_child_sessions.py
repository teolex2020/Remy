from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from remy.config.settings import settings
from remy.core.child_sessions import ChildSessionStore, get_child_session_store
from remy.core.project_store import (
    LEGACY_BRAIN_ID,
    LEGACY_PROJECT_ID,
    reset_project_store_for_tests,
)
from remy.core.worker import WorkerResult, WorkerTask
from remy.core.worker_tasks import (
    follow_up_child_session,
    get_child_report,
    get_child_session,
    get_worker_task,
    interrupt_child_session,
    launch_worker_tasks,
    recover_interrupted_child_sessions,
    resume_child_session,
    shutdown_worker_task_runtime,
)


@pytest.fixture(autouse=True)
def isolated_child_runtime(tmp_path, monkeypatch):
    shutdown_worker_task_runtime()
    reset_project_store_for_tests()
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(settings, "AURA_BRAIN_PATH", tmp_path / "data" / "brain")
    monkeypatch.setattr(settings, "WORKER_MAX_PARALLEL", 3)
    monkeypatch.setattr(settings, "WORKER_MAX_TOOL_ITERATIONS", 5)
    yield
    shutdown_worker_task_runtime()
    reset_project_store_for_tests()


def _wait_task(task_id: str, timeout: float = 5.0) -> dict:
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
    raise AssertionError(f"Task {task_id} did not settle")


def _wait_child(child_id: str, generation: int, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        child = get_child_session(child_id, owner_project_id=LEGACY_PROJECT_ID)
        if (
            child
            and int(child["generation"]) >= generation
            and child["status"] in {"ready", "failed", "interrupted"}
        ):
            return child
        time.sleep(0.02)
    raise AssertionError(f"Child {child_id} generation {generation} did not settle")


def test_store_has_stable_identity_inbox_attempt_history_and_idempotent_settlement(tmp_path):
    store = ChildSessionStore(tmp_path / "children.sqlite3")
    child = store.create(
        owner_project_id="project-1",
        brain_id="brain-1",
        parent_session_id="parent-1",
        spec=[{"role": "researcher", "instruction": "Investigate"}],
        idempotency_class="read_only",
    )
    child_id = child["child_id"]
    message = store.enqueue_message(
        child_id,
        owner_project_id="project-1",
        direction="parent_to_child",
        kind="follow_up",
        content="Check the primary source",
    )
    claimed = store.claim_messages(
        child_id,
        owner_project_id="project-1",
        direction="parent_to_child",
        kinds={"follow_up"},
    )
    assert [item["message_id"] for item in claimed] == [message["message_id"]]
    assert store.release_messages(
        child_id,
        owner_project_id="project-1",
        message_ids=[message["message_id"]],
    ) == 1

    generation = store.claim_attempt(
        child_id,
        owner_project_id="project-1",
        reason="initial",
    )
    store.bind_attempt(
        child_id,
        owner_project_id="project-1",
        generation=generation,
        run_id="run-1",
        attempt_id="attempt-1",
        reason="initial",
    )
    settled, inserted = store.settle_attempt(
        child_id,
        owner_project_id="project-1",
        run_id="run-1",
        status="completed",
        report={"summary": "Done"},
    )
    _, inserted_again = store.settle_attempt(
        child_id,
        owner_project_id="project-1",
        run_id="run-1",
        status="completed",
        report={"summary": "Duplicate"},
    )

    report = store.report(child_id, owner_project_id="project-1")
    assert settled["child_id"] == child_id
    assert settled["status"] == "ready"
    assert inserted is True
    assert inserted_again is False
    assert len(report["attempts"]) == 1
    assert report["attempts"][0]["report"]["summary"] == "Done"
    assert [event["event_type"] for event in report["events"]].count("child.settled") == 1


def test_attempt_claim_is_atomic_and_project_scoped(tmp_path):
    store = ChildSessionStore(tmp_path / "children-atomic.sqlite3")
    child = store.create(
        owner_project_id="project-a",
        brain_id="brain-a",
        parent_session_id="parent-a",
        spec=[{"role": "analyst", "instruction": "Analyse"}],
        idempotency_class="read_only",
    )
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def claim() -> None:
        barrier.wait()
        try:
            store.claim_attempt(
                child["child_id"],
                owner_project_id="project-a",
                reason="race",
            )
            outcomes.append("claimed")
        except RuntimeError:
            outcomes.append("rejected")

    threads = [threading.Thread(target=claim) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)

    assert sorted(outcomes) == ["claimed", "rejected"]
    assert store.get(child["child_id"], owner_project_id="project-b") is None


def test_follow_up_reuses_child_identity_and_adds_generation(monkeypatch):
    instructions: list[str] = []

    async def fake_execute(tasks, session_id, channel):
        instructions.append(tasks[0].instruction)
        return [
            WorkerResult(
                role=tasks[0].role,
                status="success",
                output=f"result-{len(instructions)}",
            )
        ]

    monkeypatch.setattr("remy.core.worker.execute_workers", fake_execute)
    first = launch_worker_tasks(
        [WorkerTask(role="researcher", instruction="Investigate")],
        session_id="parent-follow-up",
        channel="web",
    )
    _wait_task(first["task_id"])

    continuation = follow_up_child_session(
        first["child_id"],
        "Now compare the two sources",
        owner_project_id=LEGACY_PROJECT_ID,
    )
    assert continuation["task"] is not None
    second = _wait_task(continuation["task"]["task_id"])
    report = get_child_report(first["child_id"], owner_project_id=LEGACY_PROJECT_ID)

    assert second["child_id"] == first["child_id"]
    assert second["child_generation"] == 2
    assert len(report["attempts"]) == 2
    assert "Now compare the two sources" in instructions[1]
    assert report["messages"][-1]["kind"] == "report"


def test_follow_up_queued_while_running_continues_at_safe_boundary(monkeypatch):
    first_started = threading.Event()
    release_first = threading.Event()
    calls = 0

    async def controlled_execute(tasks, session_id, channel):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_started.set()
            while not release_first.is_set():
                await asyncio.sleep(0.01)
        return [WorkerResult(role=tasks[0].role, status="success", output=f"run-{calls}")]

    monkeypatch.setattr("remy.core.worker.execute_workers", controlled_execute)
    first = launch_worker_tasks(
        [WorkerTask(role="analyst", instruction="Analyse")],
        session_id="parent-queued",
        channel="web",
    )
    assert first_started.wait(2.0)
    queued = follow_up_child_session(
        first["child_id"],
        "Inspect the anomaly too",
        owner_project_id=LEGACY_PROJECT_ID,
    )
    assert queued["queued"] is True
    assert queued["task"] is None

    release_first.set()
    child = _wait_child(first["child_id"], generation=2)
    report = get_child_report(first["child_id"], owner_project_id=LEGACY_PROJECT_ID)
    assert child["generation"] == 2
    assert calls == 2
    assert len(report["attempts"]) == 2


def test_interrupt_then_cold_resume_preserves_child_identity(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    async def controlled_execute(tasks, session_id, channel):
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.01)
        return [WorkerResult(role=tasks[0].role, status="success", output="done")]

    monkeypatch.setattr("remy.core.worker.execute_workers", controlled_execute)
    first = launch_worker_tasks(
        [WorkerTask(role="researcher", instruction="Long investigation")],
        session_id="parent-interrupt",
        channel="web",
    )
    assert started.wait(2.0)
    interrupted = interrupt_child_session(
        first["child_id"],
        owner_project_id=LEGACY_PROJECT_ID,
        reason="Pause for operator review",
    )
    assert interrupted["interrupt_requested"] is True
    release.set()
    terminal = _wait_task(first["task_id"])
    assert terminal["status"] == "cancelled"
    assert get_child_session(
        first["child_id"], owner_project_id=LEGACY_PROJECT_ID
    )["status"] == "interrupted"

    resumed = resume_child_session(
        first["child_id"],
        owner_project_id=LEGACY_PROJECT_ID,
    )
    resumed_terminal = _wait_task(resumed["task_id"])
    assert resumed_terminal["child_id"] == first["child_id"]
    assert resumed_terminal["child_generation"] == 2


def test_side_effecting_child_requires_confirmation_for_follow_up(monkeypatch):
    async def fake_execute(tasks, session_id, channel):
        return [WorkerResult(role=tasks[0].role, status="success", output="written")]

    monkeypatch.setattr("remy.core.worker.execute_workers", fake_execute)
    task = launch_worker_tasks(
        [WorkerTask(role="executor", instruction="Write a draft")],
        session_id="parent-executor",
        channel="web",
    )
    _wait_task(task["task_id"])
    with pytest.raises(PermissionError, match="explicit confirmation"):
        follow_up_child_session(
            task["child_id"],
            "Write another draft",
            owner_project_id=LEGACY_PROJECT_ID,
        )


def test_recovery_settles_orphan_without_replaying_work():
    store = get_child_session_store()
    child = store.create(
        owner_project_id=LEGACY_PROJECT_ID,
        brain_id=LEGACY_BRAIN_ID,
        parent_session_id="parent-recovery",
        spec=[{"role": "researcher", "instruction": "Recover me"}],
        idempotency_class="read_only",
    )
    generation = store.claim_attempt(
        child["child_id"],
        owner_project_id=LEGACY_PROJECT_ID,
        reason="initial",
    )
    store.bind_attempt(
        child["child_id"],
        owner_project_id=LEGACY_PROJECT_ID,
        generation=generation,
        run_id="run-orphan",
        attempt_id="attempt-orphan",
        reason="initial",
    )
    recovered = recover_interrupted_child_sessions(
        [
            {
                "kind": "worker_group",
                "run_id": "run-orphan",
                "attempt_id": "attempt-orphan",
                "owner_project_id": LEGACY_PROJECT_ID,
                "brain_id": LEGACY_BRAIN_ID,
                "conversation_id": "parent-recovery",
                "status": "interrupted",
                "stop_reason": "process_restarted",
                "error": "process restarted",
                "artifacts": [],
                "usage": {},
                "metadata": {
                    "child_id": child["child_id"],
                    "child_generation": generation,
                },
            }
        ]
    )
    restored = store.get(child["child_id"], owner_project_id=LEGACY_PROJECT_ID)
    assert recovered == 1
    assert restored["status"] == "interrupted"
    assert restored["can_resume"] is True
    assert len(store.report(child["child_id"], owner_project_id=LEGACY_PROJECT_ID)["attempts"]) == 1


def test_settlement_is_visible_in_parent_trajectory(tmp_path):
    from remy.core.trajectory_store import TrajectoryStore

    trajectory = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    event_id = trajectory.record_child_lifecycle(
        session_id="parent-trajectory",
        project_id="project-trajectory",
        child_id="child-trajectory",
        event_type="child.settled",
        status="completed",
        run_id="run-trajectory",
        attempt_id="attempt-trajectory",
        generation=2,
        summary="Child reported successfully",
    )
    events = trajectory.list_events(
        project_id="project-trajectory",
        session_id="parent-trajectory",
        limit=20,
    )
    event = next(item for item in events if item["event_id"] == event_id)
    assert event["kind"] == "SETTLEMENT"
    assert event["details"]["child_id"] == "child-trajectory"
    assert event["details"]["generation"] == 2


def test_child_lifecycle_tool_handler_exposes_report(monkeypatch):
    from remy.core.tool_handlers.delegate import _handle_child_session_tool

    async def fake_execute(tasks, session_id, channel):
        return [WorkerResult(role=tasks[0].role, status="success", output="reported")]

    monkeypatch.setattr("remy.core.worker.execute_workers", fake_execute)
    task = launch_worker_tasks(
        [WorkerTask(role="analyst", instruction="Inspect")],
        session_id="parent-tool",
        channel="web",
    )
    _wait_task(task["task_id"])
    result = json.loads(
        _handle_child_session_tool(
            "get_child_report",
            {"child_id": task["child_id"]},
            "parent-tool",
            "desktop",
        )
    )
    assert result["child"]["child_id"] == task["child_id"]
    assert result["attempts"][0]["status"] == "completed"


def test_child_session_api_supports_list_report_and_follow_up(monkeypatch):
    from remy.web.routes import run_routes

    async def fake_execute(tasks, session_id, channel):
        return [WorkerResult(role=tasks[0].role, status="success", output="api-result")]

    monkeypatch.setattr("remy.core.worker.execute_workers", fake_execute)
    first = launch_worker_tasks(
        [WorkerTask(role="researcher", instruction="API investigation")],
        session_id="parent-api-child",
        channel="web",
    )
    _wait_task(first["task_id"])
    app = FastAPI()
    app.include_router(run_routes.router, prefix="/api")
    client = TestClient(app)

    listed = client.get("/api/children", params={"parent_session_id": "parent-api-child"})
    report = client.get(f"/api/children/{first['child_id']}/report")
    follow_up = client.post(
        f"/api/children/{first['child_id']}/follow-up",
        json={"message": "Check one more source"},
    )

    assert listed.status_code == 200
    assert listed.json()["children"][0]["child_id"] == first["child_id"]
    assert report.status_code == 200
    assert report.json()["attempts"][0]["status"] == "completed"
    assert follow_up.status_code == 200
    assert follow_up.json()["task"]["child_id"] == first["child_id"]
    _wait_task(follow_up.json()["task"]["task_id"])
