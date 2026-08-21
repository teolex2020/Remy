import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch


def test_recovers_legacy_and_interrupted_projects():
    from remy.core import research_supervisor as supervisor

    records = [
        SimpleNamespace(metadata={"project_id": "legacy", "status": "researching", "queries_done": 1}),
        SimpleNamespace(metadata={"project_id": "running", "status": "researching", "job_state": "running"}),
        SimpleNamespace(metadata={"project_id": "done", "status": "complete", "job_state": "completed"}),
    ]
    receipts = []
    fake_ledger = SimpleNamespace(recover_orphans=lambda **_: [], get=lambda _attempt_id: None)
    with patch.object(supervisor, "get_execution_ledger", return_value=fake_ledger), patch.object(
        supervisor, "_project_records", return_value=records
    ), patch.object(
        supervisor, "_append_receipt", side_effect=lambda *args, **kwargs: receipts.append((args, kwargs))
    ), patch.object(
        supervisor, "_checkpoint"
    ):
        assert supervisor._recover_interrupted_projects() == 2
    assert {item[0][0] for item in receipts} == {"legacy", "running"}
    assert all(item[1]["job_state"] == "queued" for item in receipts)


def test_query_tick_persists_grounded_progress():
    from remy.core import research_supervisor as supervisor

    updates = []
    receipts = []
    meta = {
        "query_plan": ["example query"],
        "next_query_index": 0,
        "session_id": "research:test",
        "findings_count": 0,
    }
    fake_ledger = SimpleNamespace(heartbeat=lambda *args, **kwargs: None, append_receipt=lambda *args, **kwargs: None)
    with patch.object(
        supervisor,
        "_owner_identity",
        return_value=("project-test", "brain-test"),
    ), patch(
        "remy.core.microbrain.bind_project",
        return_value=nullcontext(),
    ), patch.object(supervisor, "_ensure_execution_attempt", return_value="attempt-test"), patch.object(
        supervisor, "get_execution_ledger", return_value=fake_ledger
    ), patch.object(supervisor, "_discover", return_value=[{"uri": "https://example.com"}]), patch.object(
        supervisor, "_fetch_source", return_value=("https://example.com", "Example", "Grounded page evidence")
    ), patch.object(supervisor, "_update_project", side_effect=lambda *args, **kwargs: updates.append(kwargs)), patch.object(
        supervisor, "_append_receipt", side_effect=lambda *args, **kwargs: receipts.append(kwargs)
    ), patch.object(supervisor, "_checkpoint"), patch.object(
        supervisor, "_find_project", return_value=None
    ), patch("remy.core.brain_tools._add_research_finding", return_value='{"stored": true, "finding_id": "f-1"}'), patch(
        "remy.core.claim_provenance.record_turn_fetch_evidence"
    ) as evidence:
        asyncio.run(supervisor._run_query("rp-test", meta))
    evidence.assert_called_once()
    assert updates[0]["job_state"] == "running"
    assert receipts[-1]["job_state"] == "queued"
    assert receipts[-1]["queries_done"] == 1
    assert receipts[-1]["next_query_index"] == 1


def test_pause_and_resume_use_committed_query_boundary():
    from remy.core import research_supervisor as supervisor

    record = SimpleNamespace(metadata={
        "project_id": "rp-pause",
        "status": "researching",
        "job_state": "running",
        "next_query_index": 2,
        "queries_done": 2,
        "findings_count": 2,
    })
    updates = []

    def update(_project_id, **changes):
        record.metadata.update(changes)
        updates.append(changes)
        return dict(record.metadata)

    with patch.object(supervisor, "_find_project", return_value=record), patch.object(
        supervisor, "_update_project", side_effect=update
    ), patch.object(supervisor, "wake_research_supervisor", return_value=True):
        paused = supervisor.pause_research_project("rp-pause")
        assert paused["job_state"] == "pausing"
        assert paused["pause_requested"] is True
        assert paused["durable_checkpoint"]["query_index"] == 2

        resumed = supervisor.resume_research_project("rp-pause")
        assert resumed["job_state"] == "queued"
        assert resumed["pause_requested"] is False
        assert resumed["durable_checkpoint"]["node"] == "query_queue"

    assert len(updates) == 2


def test_completion_uses_durable_notification_outbox():
    from remy.core import research_supervisor as supervisor

    receipts = []
    result = '{"completed": true, "project_id": "rp-test", "topic": "Topic", "report_id": "r-1", "markdown": "Report"}'
    with patch.object(supervisor, "_append_receipt", side_effect=lambda *args, **kwargs: receipts.append(kwargs)), patch.object(
        supervisor, "_checkpoint"
    ), patch(
        "remy.core.notification_router.notify"
    ) as notify:
        asyncio.run(supervisor._finish_project("rp-test", {"topic": "Topic", "findings_count": 1}, lambda *_: result))
    assert receipts[-1]["job_state"] == "completed"
    assert receipts[-1]["notification_sent"] is True
    assert notify.call_args.kwargs["event_type"] == "research.complete"


def test_execution_and_continuation_keep_microbrain_owner():
    from remy.core import research_supervisor as supervisor

    claimed = {}
    queued = {}

    class FakeLedger:
        @staticmethod
        def get(_attempt_id):
            return None

        @staticmethod
        def claim(**kwargs):
            claimed.update(kwargs)
            return {"attempt_id": "attempt-owned"}

        @staticmethod
        def mark_running(_attempt_id):
            return None

        @staticmethod
        def enqueue_continuation(**kwargs):
            queued.update(kwargs)
            return "continuation-owned"

    meta = {
        "project_id": "rp-owned",
        "owner_project_id": "project-owner",
        "brain_id": "brain-owner",
        "session_id": "session-owner",
        "channel": "web",
    }
    with patch.object(
        supervisor,
        "_owner_identity",
        return_value=("project-owner", "brain-owner"),
    ), patch.object(supervisor, "get_execution_ledger", return_value=FakeLedger()), patch.object(
        supervisor, "_update_project"
    ):
        assert supervisor._ensure_execution_attempt("rp-owned", meta) == "attempt-owned"
        assert supervisor._queue_continuation(
            meta, "Owned result", status="completed"
        ) == "continuation-owned"

    assert claimed["owner_project_id"] == "project-owner"
    assert claimed["brain_id"] == "brain-owner"
    assert queued["owner_project_id"] == "project-owner"
    assert queued["brain_id"] == "brain-owner"


def test_discovery_omits_removed_startpage_backend(monkeypatch):
    from remy.core import research_supervisor as supervisor

    captured = {}

    class FakeDDGS:
        def __init__(self, **_kwargs):
            pass

        def text(self, _query, **kwargs):
            captured.update(kwargs)
            return []

    monkeypatch.setattr("ddgs.DDGS", FakeDDGS)

    assert supervisor._discover("backend compatibility") == []
    assert "startpage" not in captured["backend"].split(",")


def test_research_control_routes_delegate_to_durable_supervisor():
    from remy.web.routes import knowledge_routes

    with patch(
        "remy.core.research_supervisor.pause_research_project",
        return_value={"job_state": "pausing"},
    ), patch(
        "remy.core.research_supervisor.resume_research_project",
        return_value={"job_state": "queued"},
    ):
        paused = asyncio.run(knowledge_routes.pause_research("rp-route"))
        resumed = asyncio.run(knowledge_routes.resume_research("rp-route"))

    assert paused == {"paused": True, "project_id": "rp-route", "job_state": "pausing"}
    assert resumed == {"resumed": True, "project_id": "rp-route", "job_state": "queued"}
