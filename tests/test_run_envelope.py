import sqlite3

import pytest

from remy.config.settings import settings
from remy.core.run_envelope import (
    RunCoordinator,
    RunLimitExceeded,
    RunLimits,
    finish_run,
    get_run,
    list_runs,
    recover_interrupted_runs,
    register_run_stop,
    request_run_stop,
    start_run,
    unregister_run_stop,
)


def _start(**kwargs):
    return start_run(
        kind=kwargs.pop("kind", "chat"),
        source_id=kwargs.pop("source_id", "conversation-1"),
        goal=kwargs.pop("goal", "Analyse the project"),
        owner_project_id=kwargs.pop("owner_project_id", "project-a"),
        brain_id=kwargs.pop("brain_id", "brain-a"),
        **kwargs,
    )


def test_run_envelope_lifecycle_is_project_scoped_and_observable(tmp_path):
    original = settings.DATA_DIR
    settings.DATA_DIR = tmp_path
    try:
        run = _start(
            limits=RunLimits(max_turns=5, token_budget=10_000, max_parallel_workers=2)
        )
        coordinator = RunCoordinator(run["attempt_id"])
        coordinator.step("Read files", signature="tool:files")
        coordinator.consume_tokens(input_tokens=120, output_tokens=30)
        coordinator.worker_started("reviewer")
        coordinator.worker_finished()
        final = finish_run(
            run["attempt_id"],
            status="completed",
            output_ref="conversation:conversation-1",
        )

        assert final["status"] == "completed"
        assert final["ledger_state"] == "completed"
        assert final["usage"]["turns"] == 1
        assert final["usage"]["total_tokens"] == 150
        assert final["usage"]["peak_workers"] == 1
        assert get_run(run["run_id"], owner_project_id="project-a")["run_id"] == run["run_id"]
        assert get_run(run["run_id"], owner_project_id="project-b") is None
        assert [item["run_id"] for item in list_runs(owner_project_id="project-a")] == [run["run_id"]]
    finally:
        settings.DATA_DIR = original


def test_coordinator_stops_repeated_loops_and_records_reason(tmp_path):
    original = settings.DATA_DIR
    settings.DATA_DIR = tmp_path
    try:
        run = _start(
            limits=RunLimits(
                max_turns=20,
                token_budget=10_000,
                max_parallel_workers=1,
                loop_repeat_limit=3,
            )
        )
        coordinator = RunCoordinator(run["attempt_id"])
        coordinator.step("Search", signature="same-query")
        coordinator.step("Search", signature="same-query")
        with pytest.raises(RunLimitExceeded) as error:
            coordinator.step("Search", signature="same-query")

        assert error.value.reason == "loop_detected"
        stopped = get_run(run["run_id"], owner_project_id="project-a")
        assert stopped["status"] == "stopping"
        assert stopped["stop_reason"] == "loop_detected"
        finish_run(
            run["attempt_id"],
            status="completed_with_limits",
            stop_reason="loop_detected",
        )
    finally:
        settings.DATA_DIR = original


def test_stop_request_invokes_registered_runtime_control(tmp_path):
    original = settings.DATA_DIR
    settings.DATA_DIR = tmp_path
    calls = []
    try:
        run = _start()
        register_run_stop(run["run_id"], calls.append)

        stopping = request_run_stop(
            run["run_id"], owner_project_id="project-a", reason="Operator stop"
        )

        assert calls == ["Operator stop"]
        assert stopping["status"] == "stopping"
        assert stopping["stop_reason"] == "user_stopped"
    finally:
        unregister_run_stop(run["run_id"])
        settings.DATA_DIR = original


def test_dead_process_run_recovers_as_interrupted(tmp_path):
    original = settings.DATA_DIR
    settings.DATA_DIR = tmp_path
    try:
        run = _start()
        with sqlite3.connect(tmp_path / "execution_ledger.sqlite3") as conn:
            conn.execute(
                "UPDATE execution_attempts SET owner_pid=-1, owner_started_at='dead' "
                "WHERE attempt_id=?",
                (run["attempt_id"],),
            )

        recovered = recover_interrupted_runs()

        assert [item["run_id"] for item in recovered] == [run["run_id"]]
        assert recovered[0]["status"] == "interrupted"
        assert recovered[0]["stop_reason"] == "process_restarted"
        assert recovered[0]["ledger_state"] == "unknown"
    finally:
        settings.DATA_DIR = original
