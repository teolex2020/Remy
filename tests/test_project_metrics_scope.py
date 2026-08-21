from __future__ import annotations

import time
from pathlib import Path

import pytest

from remy.core.project_store import (
    get_project_store,
    project_state_path,
    reset_project_store_for_tests,
)


@pytest.fixture
def project_metrics(tmp_path, monkeypatch):
    from remy.config.settings import settings

    data_dir = tmp_path / "data"
    monkeypatch.setattr(settings, "DATA_DIR", data_dir)
    monkeypatch.setattr(settings, "AURA_BRAIN_PATH", data_dir / "brain")
    reset_project_store_for_tests()
    store = get_project_store()
    first = store.create_project("Metrics One")
    second = store.create_project("Metrics Two")
    yield store, first, second
    reset_project_store_for_tests()


def test_project_metric_paths_are_physically_isolated(project_metrics):
    _, first, second = project_metrics

    first_path = project_state_path("metrics", "sample.json", first.project_id)
    second_path = project_state_path("metrics", "sample.json", second.project_id)

    assert first_path == Path(first.brain_path).parent / ".meta" / "metrics" / "sample.json"
    assert second_path == Path(second.brain_path).parent / ".meta" / "metrics" / "sample.json"
    assert first_path != second_path
    with pytest.raises(ValueError, match="plain filename"):
        project_state_path("metrics", "../escape.json", first.project_id)


def test_task_metrics_follow_the_active_microbrain(project_metrics):
    from remy.core.task_metrics import CycleOutcome, TaskMetricsTracker

    store, first, second = project_metrics
    store.set_active_project(first.project_id)
    tracker = TaskMetricsTracker()
    tracker.record(CycleOutcome(family="general", success=True))

    store.set_active_project(second.project_id)
    assert tracker.get_all()["totals"]["total_cycles"] == 0
    tracker.record(CycleOutcome(family="general", success=False))

    store.set_active_project(first.project_id)
    first_metrics = tracker.get_all()["totals"]
    store.set_active_project(second.project_id)
    second_metrics = tracker.get_all()["totals"]

    assert first_metrics["total_cycles"] == 1
    assert first_metrics["successes"] == 1
    assert second_metrics["total_cycles"] == 1
    assert second_metrics["failures"] == 1


def test_execution_log_follows_the_active_microbrain(project_metrics):
    from remy.core.execution_log import ExecutionEntry, ExecutionLog

    store, first, second = project_metrics
    store.set_active_project(first.project_id)
    log = ExecutionLog()
    log.record(
        ExecutionEntry(
            timestamp=time.time(),
            goal_id="first-goal",
            pack_id="general",
            status="success",
        )
    )

    store.set_active_project(second.project_id)
    assert log.get_recent() == []
    log.record(
        ExecutionEntry(
            timestamp=time.time(),
            goal_id="second-goal",
            pack_id="general",
            status="failure",
        )
    )

    store.set_active_project(first.project_id)
    assert [item["goal_id"] for item in log.get_recent()] == ["first-goal"]
    store.set_active_project(second.project_id)
    assert [item["goal_id"] for item in log.get_recent()] == ["second-goal"]


def test_eval_metrics_follow_the_active_microbrain(project_metrics):
    from remy.core.eval_metrics import (
        ResponseMetrics,
        get_metrics_summary,
        store_eval_metrics,
    )

    store, first, second = project_metrics
    store.set_active_project(first.project_id)
    store_eval_metrics(ResponseMetrics(session_id="first", channel="desktop"))

    store.set_active_project(second.project_id)
    assert get_metrics_summary()["total_responses"] == 0
    store_eval_metrics(ResponseMetrics(session_id="second", channel="desktop"))

    store.set_active_project(first.project_id)
    assert get_metrics_summary()["total_responses"] == 1
    store.set_active_project(second.project_id)
    assert get_metrics_summary()["total_responses"] == 1


@pytest.mark.asyncio
async def test_diagnostics_report_the_active_project_scope(project_metrics, monkeypatch):
    from remy.web.routes import diagnostics

    store, first, second = project_metrics
    monkeypatch.setattr(
        "remy.core.combined_runner.get_goal_runtime_snapshot",
        lambda goal_limit=5, approval_limit=10: {
            "total": 0,
            "active": 0,
            "blocked": 0,
        },
    )

    store.set_active_project(first.project_id)
    first_payload = await diagnostics.get_task_metrics()
    store.set_active_project(second.project_id)
    second_payload = await diagnostics.get_execution_log_summary()

    assert first_payload["scope"] == {
        "kind": "project",
        "project_id": first.project_id,
        "brain_id": first.brain_id,
        "project_name": first.name,
    }
    assert second_payload["scope"]["project_id"] == second.project_id
    assert second_payload["scope"]["brain_id"] == second.brain_id
