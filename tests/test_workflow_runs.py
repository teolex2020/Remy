from remy.config.settings import settings
from remy.core.file_utils import atomic_write as real_atomic_write
from remy.core import workflow_runs
from remy.core.workflow_memory_evaluator import evaluate_workflow_memory


def test_workflow_memory_evaluator_flags_empty_search_and_write_before_search():
    report = evaluate_workflow_memory([
        {"index": 1, "id": "s1", "type": "memory_save", "label": "Save", "output": "Saved to memory (10 characters)"},
        {"index": 2, "id": "s2", "type": "memory_search", "label": "Search", "output": "[Nothing found in memory]"},
    ])

    assert report["memory_search_count"] == 1
    assert report["memory_save_count"] == 1
    assert report["empty_search_count"] == 1
    assert report["missed_search_before_save"] is True
    assert report["score"] < 100


def test_workflow_run_roundtrip(tmp_path):
    original = settings.DATA_DIR
    settings.DATA_DIR = tmp_path
    try:
        record = workflow_runs.start_workflow_run(
            kind="pipeline",
            workflow_id="pipe-1",
            workflow_name="Research",
            input_text="hello",
            trigger="manual",
        )
        finished = workflow_runs.finish_workflow_run(
            record,
            status="ok",
            output="done",
            trace=[{"id": "s1", "status": "ok", "output": "done"}],
            steps_run=1,
        )

        loaded = workflow_runs.get_workflow_run("pipeline", "pipe-1", record["run_id"])
        runs = workflow_runs.list_workflow_runs("pipeline", "pipe-1")

        assert loaded["run_id"] == record["run_id"]
        assert loaded["execution_attempt_id"].startswith("attempt-")
        assert loaded["status"] == "ok"
        assert loaded["output"] == "done"
        assert loaded["trace"][0]["id"] == "s1"
        assert loaded["memory_evaluation"]["score"] < 100
        assert loaded["memory_evaluation"]["recommendations"]
        assert finished["duration_ms"] is not None
        assert runs[0]["run_id"] == record["run_id"]
        assert "memory_evaluation" in runs[0]
        assert runs[0]["output_preview"] == "done"
    finally:
        settings.DATA_DIR = original


def test_workflow_run_persistence_uses_atomic_write(tmp_path, monkeypatch):
    original = settings.DATA_DIR
    settings.DATA_DIR = tmp_path
    calls = []

    def _spy_atomic_write(path, content, encoding="utf-8"):
        calls.append(path)
        return real_atomic_write(path, content, encoding)

    monkeypatch.setattr(workflow_runs, "atomic_write", _spy_atomic_write)
    try:
        record = workflow_runs.start_workflow_run(kind="pipeline", workflow_id="pipe-atomic")
        workflow_runs.finish_workflow_run(record, status="ok", output="done")
    finally:
        settings.DATA_DIR = original

    assert len(calls) >= 2
    assert all(str(path).endswith(".json") for path in calls)


def test_workflow_memory_summary_aggregates_recent_runs(tmp_path):
    original = settings.DATA_DIR
    settings.DATA_DIR = tmp_path
    try:
        first = workflow_runs.start_workflow_run(
            kind="pipeline",
            workflow_id="pipe-1",
            workflow_name="Research",
        )
        workflow_runs.finish_workflow_run(
            first,
            status="ok",
            trace=[
                {"id": "s1", "type": "memory_save", "label": "Save", "output": "Saved to memory"},
                {"id": "s2", "type": "memory_search", "label": "Search", "output": "[Nothing found in memory]"},
            ],
        )
        second = workflow_runs.start_workflow_run(
            kind="pipeline",
            workflow_id="pipe-1",
            workflow_name="Research",
        )
        workflow_runs.finish_workflow_run(
            second,
            status="ok",
            trace=[
                {"id": "s1", "type": "memory_search", "label": "Search", "output": "Found 2 memories"},
                {"id": "s2", "type": "memory_save", "label": "Save", "output": "Saved to memory"},
            ],
        )

        summary = workflow_runs.summarize_workflow_memory("pipeline", "pipe-1")

        assert summary["run_count"] == 2
        assert summary["evaluated_run_count"] == 2
        assert summary["average_score"] < 100
        assert summary["status"] in {"needs_attention", "watch"}
        assert summary["totals"]["memory_search_count"] == 2
        assert summary["totals"]["memory_save_count"] == 2
        assert summary["totals"]["empty_search_count"] == 1
        assert summary["totals"]["missed_search_before_save_count"] == 1
        assert summary["top_recommendations"]
        assert summary["trend"]["direction"] == "improved"
        assert summary["trend"]["latest_score"] > summary["trend"]["previous_score"]
        assert summary["trend"]["delta"] > 0
    finally:
        settings.DATA_DIR = original


def test_workflow_runs_prune_old_records(tmp_path, monkeypatch):
    original = settings.DATA_DIR
    settings.DATA_DIR = tmp_path
    monkeypatch.setattr(workflow_runs, "MAX_RUN_RECORDS_PER_WORKFLOW", 3)
    try:
        for index in range(5):
            record = workflow_runs.start_workflow_run(
                kind="automation",
                workflow_id="auto-1",
                workflow_name="Automation",
                input_text=str(index),
            )
            workflow_runs.finish_workflow_run(record, status="ok", output=str(index))

        runs = workflow_runs.list_workflow_runs("automation", "auto-1", limit=10)

        assert len(runs) == 3
    finally:
        settings.DATA_DIR = original


def test_workflow_run_list_limit_is_clamped(tmp_path, monkeypatch):
    original = settings.DATA_DIR
    settings.DATA_DIR = tmp_path
    monkeypatch.setattr(workflow_runs, "MAX_RUN_RECORDS_PER_WORKFLOW", 5)
    try:
        for index in range(5):
            record = workflow_runs.start_workflow_run(
                kind="pipeline",
                workflow_id="pipe-limit",
                workflow_name="Pipeline",
                input_text=str(index),
            )
            workflow_runs.finish_workflow_run(record, status="ok", output=str(index))

        assert len(workflow_runs.list_workflow_runs("pipeline", "pipe-limit", limit=2)) == 2
        assert len(workflow_runs.list_workflow_runs("pipeline", "pipe-limit", limit=0)) == 5
        assert len(workflow_runs.list_workflow_runs("pipeline", "pipe-limit", limit=-10)) == 5

        monkeypatch.setattr(workflow_runs, "MAX_RUN_LIST_LIMIT", 3)
        assert len(workflow_runs.list_workflow_runs("pipeline", "pipe-limit", limit=999)) == 3
    finally:
        settings.DATA_DIR = original
