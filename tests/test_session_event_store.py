from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from remy.core.session_event_store import (
    AuditProjection,
    CheckpointProjection,
    ExecutionLedgerProjection,
    MetricsProjection,
    ModelHistoryProjection,
    RecoveryProjection,
    SessionEventStore,
    ToolRunProjection,
    TranscriptProjection,
)
from remy.core.trajectory_store import TrajectoryStore
from remy.core.transcript_store import TranscriptStore


def test_append_only_session_log_versions_and_replays_runtime_projections(tmp_path):
    path = tmp_path / "trajectory.sqlite3"
    trajectory = TrajectoryStore(path)
    trajectory.begin_turn(
        session_id="session-1",
        project_id="project-1",
        content="Investigate the failure",
    )
    request_id = trajectory.begin_request(
        session_id="session-1",
        messages=[HumanMessage("Investigate the failure")],
        tools=[],
        routing={"preferred_model": "provider/candidate"},
    )
    trajectory.complete_request(
        session_id="session-1",
        response=AIMessage(
            "Recovered answer",
            response_metadata={"model": "provider/candidate"},
            usage_metadata={
                "input_tokens": 30,
                "output_tokens": 12,
                "total_tokens": 42,
            },
        ),
    )
    tool_event_id = trajectory.begin_tool(
        session_id="session-1",
        call_id="call-1",
        name="inspect_status",
        payload={"scope": "local"},
    )
    trajectory.complete_tool(event_id=tool_event_id, result={"ok": True})
    trajectory.finish_turn(session_id="session-1", evaluate_regressions=False)

    event_store = SessionEventStore(path)
    events = event_store.list_events(project_id="project-1", session_id="session-1")
    request_versions = [
        event for event in events if event["subject_event_id"] == request_id
    ]

    assert [event["event_version"] for event in request_versions] == [1, 2]
    assert [event["payload"]["status"] for event in request_versions] == [
        "running", "completed",
    ]
    assert all(event["integrity_ok"] for event in events)
    assert all(event["payload"]["contract_version"] == 1 for event in events)

    transcript = event_store.project(
        TranscriptProjection(), project_id="project-1", session_id="session-1"
    )
    assert [(row["role"], row["content"]) for row in transcript] == [
        ("user", "Investigate the failure"),
        ("assistant", "Recovered answer"),
    ]
    tools = event_store.project(
        ToolRunProjection(), project_id="project-1", session_id="session-1"
    )
    assert tools[0]["name"] == "inspect_status"
    assert tools[0]["status"] == "completed"
    metrics = event_store.project(
        MetricsProjection(), project_id="project-1", session_id="session-1"
    )
    assert metrics["requests"] == 1
    assert metrics["tool_calls"] == 1
    assert metrics["total_tokens"] == 42
    models = event_store.project(
        ModelHistoryProjection(), project_id="project-1", session_id="session-1"
    )
    assert models[0]["model"] == "provider/candidate"
    assert event_store.verify_materialized_projection(
        project_id="project-1", session_id="session-1"
    )["ok"] is True


def test_restart_recovery_is_an_append_only_event_and_transcript_fallback(
    tmp_path, monkeypatch
):
    path = tmp_path / "trajectory.sqlite3"
    trajectory = TrajectoryStore(path)
    trajectory.begin_turn(
        session_id="session-2",
        project_id="project-2",
        content="Resume me",
    )
    request_id = trajectory.begin_request(
        session_id="session-2",
        messages=[HumanMessage("Resume me")],
        tools=[],
    )

    TrajectoryStore(path)
    event_store = SessionEventStore(path)
    recovery = event_store.project(
        RecoveryProjection(), project_id="project-2", session_id="session-2"
    )
    assert recovery[-1]["event_id"] == request_id
    assert recovery[-1]["event_type"] == "event.recovered"
    assert "process restart" in recovery[-1]["error"]

    from remy.core import session_event_store as session_event_module

    monkeypatch.setattr(session_event_module, "_STORE", event_store)
    transcript = TranscriptStore(tmp_path / "transcript.sqlite3").list_session(
        "session-2",
        owner_project_id="project-2",
    )
    assert [(row["role"], row["content"]) for row in transcript] == [
        ("user", "Resume me"),
    ]


def test_projection_verifier_detects_materialized_row_drift(tmp_path):
    path = tmp_path / "trajectory.sqlite3"
    trajectory = TrajectoryStore(path)
    trajectory.begin_turn(
        session_id="session-3",
        project_id="project-3",
        content="Immutable source",
    )
    event_store = SessionEventStore(path)

    with trajectory._connect() as conn:
        conn.execute(
            """UPDATE trajectory_events SET output_json='\"tampered\"'
               WHERE project_id='project-3' AND session_id='session-3'"""
        )

    verification = event_store.verify_materialized_projection(
        project_id="project-3", session_id="session-3"
    )
    assert verification["ok"] is False
    assert verification["drifted"][0]["reason"] == "projection_drift"


def test_generic_runtime_projections_share_hash_verified_session_journal(tmp_path):
    path = tmp_path / "trajectory.sqlite3"
    TrajectoryStore(path)
    event_store = SessionEventStore(path)

    event_store.append_event(
        subject_event_id="checkpoint:remy:desktop:session-4",
        project_id="project-4",
        session_id="session-4",
        event_type="checkpoint.observed",
        kind="CHECKPOINT",
        status="resumable",
        payload={
            "details": {
                "thread_id": "remy:desktop:session-4",
                "channel": "desktop",
                "next": ["approval"],
                "pending_tasks": [{"name": "approval", "has_error": False}],
                "checkpoint_created_at": "2026-08-21T12:00:00+00:00",
            }
        },
    )
    event_store.append_event(
        subject_event_id="audit:one",
        project_id="project-4",
        session_id="session-4",
        event_type="audit.recorded",
        kind="AUDIT",
        status="success",
        payload={
            "entry": {
                "timestamp": "2026-08-21T12:01:00+00:00",
                "tool_name": "http_request",
                "status": "success",
                "checksum": "abc123",
            }
        },
    )

    checkpoint = event_store.project(
        CheckpointProjection(), project_id="project-4", session_id="session-4"
    )
    audits = event_store.project(
        AuditProjection(), project_id="project-4", session_id="session-4"
    )
    events = event_store.list_events(project_id="project-4", session_id="session-4")

    assert checkpoint["status"] == "resumable"
    assert checkpoint["next"] == ["approval"]
    assert audits[0]["tool_name"] == "http_request"
    assert all(event["integrity_ok"] for event in events)
    assert event_store.verify_materialized_projection(
        project_id="project-4", session_id="session-4"
    )["ok"] is True


def test_execution_ledger_projection_is_primary_after_dual_write(tmp_path):
    from remy.core.execution_ledger import ExecutionLedger

    event_path = tmp_path / "trajectory.sqlite3"
    TrajectoryStore(event_path)
    event_store = SessionEventStore(event_path)
    ledger = ExecutionLedger(tmp_path / "execution.sqlite3", event_store=event_store)

    attempt = ledger.claim(
        kind="pipeline",
        job_id="pipeline-1",
        idempotency_class="read_only",
        owner_project_id="project-5",
        brain_id="brain-5",
        session_id="session-5",
    )
    ledger.mark_running(attempt["attempt_id"])
    ledger.finish(attempt["attempt_id"], "completed", output_ref="artifact://result")

    projected = event_store.project(
        ExecutionLedgerProjection(),
        project_id="project-5",
        session_id="session-5",
    )
    assert projected[0]["state"] == "completed"
    assert [receipt["event"] for receipt in projected[0]["receipts"]] == [
        "claimed",
        "running",
        "completed",
    ]

    with ledger._connect() as conn:
        conn.execute(
            "UPDATE execution_attempts SET job_id='legacy-drift' WHERE attempt_id=?",
            (attempt["attempt_id"],),
        )

    assert ledger.get(attempt["attempt_id"])["job_id"] == "pipeline-1"
    listed = ledger.list_attempts(owner_project_id="project-5", state="completed")
    assert listed[0]["attempt_id"] == attempt["attempt_id"]


def test_scoped_audit_consumer_reads_common_projection(tmp_path):
    from remy.core.audit_trail import AuditLogger

    event_path = tmp_path / "trajectory.sqlite3"
    TrajectoryStore(event_path)
    event_store = SessionEventStore(event_path)
    audit = AuditLogger(tmp_path / "audit", event_store=event_store)
    audit.log_action(
        tool_name="http_request",
        tool_input={"url": "https://example.test", "api_key": "secret"},
        raw_output="ok",
        status="success",
        execution_time_ms=12,
        project_id="project-6",
        session_id="session-6",
    )

    scoped = audit.get_recent_logs(
        project_id="project-6",
        session_id="session-6",
    )
    assert scoped[0]["tool_input"]["api_key"] == "***REDACTED***"
    assert audit.get_summary(
        project_id="project-6",
        session_id="session-6",
    )["success"] == 1
    integrity = audit.verify_integrity(
        project_id="project-6",
        session_id="session-6",
    )
    assert integrity["integrity"] == "OK"
    assert integrity["event_log_entries"] == 1


@pytest.mark.asyncio
async def test_trajectory_api_reports_unified_event_log_state(tmp_path, monkeypatch):
    from remy.web.routes import trajectory_routes

    trajectory = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    trajectory.begin_turn(
        session_id="conversation-1",
        project_id="project-1",
        content="Show the journal",
    )
    trajectory.finish_turn(session_id="conversation-1", evaluate_regressions=False)
    conversation = SimpleNamespace(
        conversation_id="conversation-1",
        project_id="project-1",
    )
    monkeypatch.setattr(trajectory_routes, "current_project_id", lambda: "project-1")
    monkeypatch.setattr(
        trajectory_routes,
        "get_conversation_store",
        lambda _: SimpleNamespace(require=lambda __: conversation),
    )
    monkeypatch.setattr(trajectory_routes, "get_trajectory_store", lambda: trajectory)
    monkeypatch.setattr(trajectory_routes, "_legacy_projection", lambda *_: [])

    payload = await trajectory_routes.get_conversation_trajectory(
        "conversation-1", limit=100
    )

    assert payload["event_log"]["status"] == "available"
    assert payload["event_log"]["append_only"] is True
    assert payload["event_log"]["contract_version"] == 1
    assert payload["event_log"]["event_versions"] == 1
    assert payload["event_log"]["subjects"] == 1
    assert payload["event_log"]["integrity"]["ok"] is True
    assert payload["event_log"]["projections"] == [
        "transcript",
        "tool_runs",
        "metrics",
        "recovery",
        "model_history",
        "checkpoint",
        "execution_ledger",
        "audit",
    ]
