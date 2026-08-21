import json
import time
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from remy.config.settings import settings
from remy.core.ptc_pilot import (
    HARD_MAX_CALLS,
    PTC_TOOLS,
    execute_ptc_program,
    list_ptc_tools,
    validate_ptc_program,
)
from remy.core.trajectory_store import TrajectoryStore


def _program(*steps):
    return {"version": 1, "steps": list(steps)}


def _step(step_id, tool, args=None):
    return {"id": step_id, "tool": tool, "args": args or {}}


def _patch_runtime(monkeypatch, tmp_path):
    project = SimpleNamespace(project_id="project-ptc", brain_id="brain-ptc")
    store = SimpleNamespace(require_project=lambda project_id: project)
    monkeypatch.setattr("remy.core.project_store.get_project_store", lambda: store)
    monkeypatch.setattr("remy.core.ptc_pilot._record_ptc_trajectory", lambda **kwargs: None)
    original = settings.DATA_DIR
    settings.DATA_DIR = tmp_path
    return original


def test_ptc_catalog_is_explicit_read_only_and_matches_agent_declarations():
    from remy.core.brain_tools import BRAIN_TOOLS

    declared = {tool.name for tool in BRAIN_TOOLS}
    catalog = list_ptc_tools()
    dangerous = {
        "browser_act",
        "web_search",
        "http_get",
        "shell_exec",
        "code_execution",
        "write_file",
        "fs_write",
        "update_todo",
        "execute_trade",
    }

    assert catalog
    assert {item["name"] for item in catalog} == set(PTC_TOOLS)
    assert set(PTC_TOOLS) <= declared
    assert not (set(PTC_TOOLS) & dangerous)
    assert all(item["read_only"] is True for item in catalog)


def test_ptc_validation_rejects_unsafe_tools_and_non_prior_references():
    unsafe = validate_ptc_program(_program(_step("mutate", "write_file", {"path": "x"})))
    future_ref = validate_ptc_program(
        _program(
            _step("first", "read_file", {"path": {"$ref": "later.path"}}),
            _step("later", "list_local_workspaces"),
        )
    )

    assert unsafe["valid"] is False
    assert "read-only PTC allowlist" in " ".join(unsafe["errors"])
    assert future_ref["valid"] is False
    assert "not a prior step" in " ".join(future_ref["errors"])


def test_ptc_validation_is_strict_about_schema_types_and_limits():
    result = validate_ptc_program(
        _program(
            {"id": "clock", "tool": "get_current_datetime", "args": {"extra": True}},
            _step("bad", "metric_summary", {"period": "decade"}),
        ),
        {"max_calls": HARD_MAX_CALLS + 1},
    )

    errors = " ".join(result["errors"])
    assert result["valid"] is False
    assert "unknown arguments" in errors
    assert "week, month, year" in errors
    assert "max_calls" in errors


def test_ptc_validation_accepts_typed_chain_and_is_deterministic():
    program = _program(
        _step("workspaces", "list_local_workspaces"),
        _step("read", "read_file", {"path": {"$ref": "workspaces.items.0.path"}}),
    )
    first = validate_ptc_program(program)
    second = validate_ptc_program(program)

    assert first["valid"] is True
    assert first["program_hash"] == second["program_hash"]
    assert first["read_only"] is True


def test_ptc_execution_resolves_prior_results_and_persists_receipts(monkeypatch, tmp_path):
    original = _patch_runtime(monkeypatch, tmp_path)
    calls = []

    def invoke(name, args, session_id, channel):
        calls.append((name, args, session_id, channel))
        if name == "list_local_workspaces":
            return json.dumps({"items": [{"path": "Documents/report.md"}]}), {"decision": "allow"}
        return "report body", {"decision": "allow", "stages": ["policy"]}

    try:
        receipt = execute_ptc_program(
            _program(
                _step("workspaces", "list_local_workspaces"),
                _step("read", "read_file", {"path": {"$ref": "workspaces.items.0.path"}}),
            ),
            owner_project_id="project-ptc",
            session_id="conversation-ptc",
            tool_invoker=invoke,
        )
    finally:
        settings.DATA_DIR = original

    assert receipt["status"] == "completed"
    assert receipt["usage"]["calls"] == 2
    assert calls[1][1] == {"path": "Documents/report.md"}
    assert receipt["steps"][1]["pipeline"]["decision"] == "allow"
    assert receipt["steps"][1]["result_sha256"]
    assert receipt["run_status"] == "completed"


def test_ptc_execution_enforces_call_and_output_budgets(monkeypatch, tmp_path):
    original = _patch_runtime(monkeypatch, tmp_path)
    try:
        call_limited = execute_ptc_program(
            _program(
                _step("one", "get_current_datetime"),
                _step("two", "get_current_datetime"),
            ),
            owner_project_id="project-ptc",
            session_id="conversation-ptc",
            limits={"max_calls": 1, "output_chars": 256, "time_budget_ms": 1_000},
            tool_invoker=lambda *args: ("ok", None),
        )
        output_limited = execute_ptc_program(
            _program(_step("one", "get_current_datetime")),
            owner_project_id="project-ptc",
            session_id="conversation-ptc",
            limits={"max_calls": 1, "output_chars": 256, "time_budget_ms": 1_000},
            tool_invoker=lambda *args: ("x" * 300, None),
        )
    finally:
        settings.DATA_DIR = original

    assert (call_limited["status"], call_limited["stop_reason"]) == (
        "completed_with_limits",
        "call_budget",
    )
    assert call_limited["usage"]["calls"] == 1
    assert (output_limited["status"], output_limited["stop_reason"]) == (
        "completed_with_limits",
        "output_budget",
    )
    assert len(output_limited["steps"][0]["result"]) == 256
    assert output_limited["steps"][0]["truncated"] is True


def test_ptc_execution_enforces_wall_clock_budget(monkeypatch, tmp_path):
    original = _patch_runtime(monkeypatch, tmp_path)

    def slow(*args):
        time.sleep(0.15)
        return "late", None

    started = time.monotonic()
    try:
        receipt = execute_ptc_program(
            _program(_step("slow", "get_current_datetime")),
            owner_project_id="project-ptc",
            session_id="conversation-ptc",
            limits={"max_calls": 1, "output_chars": 256, "time_budget_ms": 50},
            tool_invoker=slow,
        )
    finally:
        settings.DATA_DIR = original

    assert receipt["status"] == "completed_with_limits"
    assert receipt["stop_reason"] == "time_budget"
    assert receipt["steps"][0]["status"] == "timed_out"
    assert receipt["usage"]["elapsed_ms"] < 140
    assert time.monotonic() - started < 0.5


def test_ptc_run_is_visible_in_trajectory(tmp_path):
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    event_id = store.record_ptc_run(
        project_id="project-1",
        session_id="conversation-1",
        run_id="run-1",
        program_hash="abc123",
        receipt={
            "status": "completed_with_limits",
            "stop_reason": "call_budget",
            "limits": {"max_calls": 1},
            "usage": {"calls": 1, "elapsed_ms": 12, "output_chars": 20},
            "steps": [{"id": "clock", "tool": "get_current_datetime"}],
        },
    )
    events = store.list_events(project_id="project-1", session_id="conversation-1")
    event = next(item for item in events if item["event_id"] == event_id)

    assert event["kind"] == "PTC"
    assert event["status"] == "completed_with_limits"
    assert event["details"]["read_only"] is True
    assert event["details"]["stop_reason"] == "call_budget"


def test_ptc_api_exposes_catalog_validation_and_run(monkeypatch):
    from remy.core import ptc_pilot
    from remy.web.routes import run_routes

    monkeypatch.setattr(run_routes, "current_project_id", lambda: "project-api")
    monkeypatch.setattr(
        ptc_pilot,
        "execute_ptc_program",
        lambda program, **kwargs: {
            "status": "completed",
            "project_id": kwargs["owner_project_id"],
            "session_id": kwargs["session_id"],
        },
    )
    app = FastAPI()
    app.include_router(run_routes.router, prefix="/api")
    client = TestClient(app)
    program = _program(_step("clock", "get_current_datetime"))

    tools = client.get("/api/ptc/tools")
    validation = client.post("/api/ptc/validate", json={"program": program})
    run = client.post(
        "/api/ptc/run",
        json={"program": program, "session_id": "conversation-api"},
    )

    assert tools.status_code == 200
    assert tools.json()["read_only"] is True
    assert validation.json()["valid"] is True
    assert run.json() == {
        "status": "completed",
        "project_id": "project-api",
        "session_id": "conversation-api",
    }
