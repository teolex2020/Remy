import sqlite3

import pytest

from remy.core.execution_ledger import ExecutionLedger


OWNER_PROJECT_ID = "project-test"
OWNER_BRAIN_ID = "brain-test"


def test_attempt_lifecycle_has_receipts_and_immutable_terminal_state(tmp_path):
    ledger = ExecutionLedger(tmp_path / "ledger.sqlite3")
    attempt = ledger.claim(
        kind="pipeline",
        job_id="pipe-1:run-1",
        idempotency_class="side_effecting",
        session_id="session-1",
        owner_project_id=OWNER_PROJECT_ID,
        brain_id=OWNER_BRAIN_ID,
    )
    ledger.mark_running(attempt["attempt_id"])
    ledger.heartbeat(attempt["attempt_id"], {"step": "fetch"})
    finished = ledger.finish(attempt["attempt_id"], "completed", output_ref="artifact:1")

    assert finished["state"] == "completed"
    assert finished["output_ref"] == "artifact:1"
    assert [item["event"] for item in finished["receipts"]] == [
        "claimed", "running", "heartbeat", "completed"
    ]
    with pytest.raises(RuntimeError):
        ledger.finish(attempt["attempt_id"], "failed")


def test_orphan_becomes_unknown_and_is_never_blindly_retried(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    ledger = ExecutionLedger(path)
    attempt = ledger.claim(
        kind="automation",
        job_id="send-mail",
        idempotency_class="side_effecting",
        owner_project_id=OWNER_PROJECT_ID,
        brain_id=OWNER_BRAIN_ID,
    )
    ledger.mark_running(attempt["attempt_id"])
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE execution_attempts SET owner_pid=-1, owner_started_at='dead' WHERE attempt_id=?",
            (attempt["attempt_id"],),
        )

    recovered = ledger.recover_orphans(kind="automation")

    assert recovered[0]["attempt_id"] == attempt["attempt_id"]
    assert ledger.get(attempt["attempt_id"])["state"] == "unknown"
    # Recovery records uncertainty only. A new attempt requires an explicit claim.
    replacement = ledger.claim(
        kind="automation",
        job_id="send-mail",
        idempotency_class="side_effecting",
        owner_project_id=OWNER_PROJECT_ID,
        brain_id=OWNER_BRAIN_ID,
    )
    assert replacement["attempt_id"] != attempt["attempt_id"]


def test_continuation_survives_session_reconnect_and_delivers_once(tmp_path):
    ledger = ExecutionLedger(tmp_path / "ledger.sqlite3")
    continuation_id = ledger.enqueue_continuation(
        session_id="old-session",
        kind="research_result",
        source_id="rp-1",
        content="Grounded report",
        metadata={"delivery_target": "web"},
        owner_project_id=OWNER_PROJECT_ID,
        brain_id=OWNER_BRAIN_ID,
    )

    delivered = ledger.consume_continuations(
        "new-session",
        include_unmatched_sessions=True,
        delivery_targets={"web", "desktop"},
        owner_project_id=OWNER_PROJECT_ID,
        brain_id=OWNER_BRAIN_ID,
    )

    assert [item["continuation_id"] for item in delivered] == [continuation_id]
    assert delivered[0]["content"] == "Grounded report"
    assert ledger.consume_continuations(
        "new-session",
        include_unmatched_sessions=True,
        delivery_targets={"web"},
        owner_project_id=OWNER_PROJECT_ID,
        brain_id=OWNER_BRAIN_ID,
    ) == []


def test_existing_ledger_is_migrated_without_losing_rows(tmp_path):
    path = tmp_path / "legacy-ledger.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE execution_attempts (
                attempt_id TEXT PRIMARY KEY, kind TEXT NOT NULL, job_id TEXT NOT NULL,
                idempotency_class TEXT NOT NULL, state TEXT NOT NULL,
                owner_pid INTEGER NOT NULL, owner_started_at TEXT NOT NULL,
                session_id TEXT NOT NULL DEFAULT '', channel TEXT NOT NULL DEFAULT '',
                claimed_at TEXT NOT NULL, started_at TEXT NOT NULL DEFAULT '',
                heartbeat_at TEXT NOT NULL DEFAULT '', finished_at TEXT NOT NULL DEFAULT '',
                output_ref TEXT NOT NULL DEFAULT '', error TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE execution_receipts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, attempt_id TEXT NOT NULL,
                event TEXT NOT NULL, created_at TEXT NOT NULL,
                data_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE continuation_inbox (
                continuation_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                kind TEXT NOT NULL, source_id TEXT NOT NULL, content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
                delivered_at TEXT NOT NULL DEFAULT ''
            );
            INSERT INTO continuation_inbox(
                continuation_id, session_id, kind, source_id, content, created_at
            ) VALUES ('old-1', 'session-1', 'research_result', 'rp-old', 'Old result', '2026-01-01');
            """
        )

    ledger = ExecutionLedger(path)

    with sqlite3.connect(path) as conn:
        attempt_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(execution_attempts)")
        }
        continuation_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(continuation_inbox)")
        }
        assert conn.execute(
            "SELECT content FROM continuation_inbox WHERE continuation_id='old-1'"
        ).fetchone()[0] == "Old result"
    assert {"owner_project_id", "brain_id"} <= attempt_columns
    assert {"owner_project_id", "brain_id"} <= continuation_columns


def test_continuations_never_cross_project_boundary(tmp_path):
    ledger = ExecutionLedger(tmp_path / "ledger.sqlite3")
    first = ledger.enqueue_continuation(
        session_id="shared-session",
        owner_project_id="project-a",
        brain_id="brain-a",
        kind="research_result",
        source_id="rp-a",
        content="Project A result",
    )
    ledger.enqueue_continuation(
        session_id="shared-session",
        owner_project_id="project-b",
        brain_id="brain-b",
        kind="research_result",
        source_id="rp-b",
        content="Project B result",
    )

    delivered = ledger.consume_continuations(
        "new-session",
        include_unmatched_sessions=True,
        owner_project_id="project-a",
        brain_id="brain-a",
    )

    assert [item["continuation_id"] for item in delivered] == [first]
    assert delivered[0]["owner_project_id"] == "project-a"
    remaining = ledger.consume_continuations(
        "new-session",
        include_unmatched_sessions=True,
        owner_project_id="project-b",
        brain_id="brain-b",
    )
    assert [item["content"] for item in remaining] == ["Project B result"]
