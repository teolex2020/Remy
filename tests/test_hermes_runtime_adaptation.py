import asyncio
import json
import time

import pytest

from remy.core.cancellation import (
    CancellationToken,
    OperationCancelled,
    bind_cancellation_token,
    check_cancelled,
)
from remy.core.execution_ledger import ExecutionLedger
from remy.core.memory_write_queue import MemoryWriteQueue
from remy.core.transcript_store import TranscriptStore


def test_cancellation_token_crosses_to_thread():
    async def scenario():
        token = CancellationToken()
        with bind_cancellation_token(token):
            token.cancel("stop")
            with pytest.raises(OperationCancelled, match="stop"):
                await asyncio.to_thread(check_cancelled)

    asyncio.run(scenario())


def test_exact_transcript_fts_search(tmp_path):
    store = TranscriptStore(tmp_path / "transcripts.sqlite3")
    exact = "The launch codename is Blue Orchard, including every detail."
    owner = {"owner_project_id": "project-test", "brain_id": "brain-test"}
    store.append(session_id="s1", role="user", content=exact, **owner)
    store.append(
        session_id="s2",
        role="user",
        content="Unrelated orchard notes",
        **owner,
    )

    matches = store.search(
        "Blue Orchard",
        session_id="s1",
        owner_project_id=owner["owner_project_id"],
    )

    assert len(matches) == 1
    assert matches[0]["content"] == exact
    assert matches[0]["session_id"] == "s1"


def test_execution_ledger_lists_filtered_attempts(tmp_path):
    ledger = ExecutionLedger(tmp_path / "ledger.sqlite3")
    attempt = ledger.claim(
        kind="worker",
        job_id="job-1",
        idempotency_class="read_only",
        owner_project_id="project-test",
        brain_id="brain-test",
    )
    ledger.mark_running(attempt["attempt_id"])
    ledger.finish(attempt["attempt_id"], "completed")

    assert ledger.list_attempts(state="completed", kind="worker")[0]["job_id"] == "job-1"
    assert ledger.list_attempts(state="failed") == []


def test_memory_write_queue_is_serial_and_drains():
    queue = MemoryWriteQueue()
    order = []

    def write(value):
        time.sleep(0.01)
        order.append(value)
        return value

    first = queue.submit(write, 1)
    second = queue.submit(write, 2)
    assert first.result(timeout=1) == 1
    assert second.result(timeout=1) == 2
    assert queue.close(timeout=1)
    assert order == [1, 2]


def test_progressive_skill_enables_real_tools():
    from remy.core.tool_dispatch import execute_tool

    result = json.loads(execute_tool("enable_skill", {"skill_name": "memory_audit"}))
    assert result["skill"] == "memory_audit"
    assert "search_transcript_history" in result["enabled"]
