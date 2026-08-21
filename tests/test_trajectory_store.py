from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from remy.core.trajectory_store import TrajectoryStore


class SystemMessage:
    def __init__(self, content):
        self.content = content


class HumanMessage:
    def __init__(self, content):
        self.content = content


class AIMessage:
    def __init__(self, content, *, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.usage_metadata = {
            "input_tokens": 12,
            "output_tokens": 4,
            "total_tokens": 16,
        }
        self.response_metadata = {
            "provider": "test-provider",
            "model_name": "test-model",
        }


def test_trajectory_records_causal_request_assistant_and_tool(tmp_path):
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    turn_id = store.begin_turn(
        session_id="conversation-1",
        project_id="project-1",
        content="Research this",
    )
    request_id = store.begin_request(
        session_id="conversation-1",
        messages=[
            SystemMessage("Always cite evidence"),
            SystemMessage("Relevant project context"),
            HumanMessage("Research this"),
        ],
        tools=[SimpleNamespace(name="web_search", description="Search", args_schema=None)],
        routing={"preferred_model": "test-model"},
        context_sources=[{
            "kind": "memory",
            "name": "relevant-memory-context",
            "trust_tier": "stored-memory",
        }],
    )
    assistant_id = store.complete_request(
        session_id="conversation-1",
        response=AIMessage(
            "I will search",
            tool_calls=[{"id": "call-1", "name": "web_search", "args": {"q": "x"}}],
        ),
    )
    tool_id = store.begin_tool(
        session_id="conversation-1",
        call_id="call-1",
        name="web_search",
        payload={"q": "x"},
        schema=SimpleNamespace(name="web_search", description="Search", args_schema=None),
    )
    store.complete_tool(event_id=tool_id, result={"sources": ["https://example.test"]})
    store.finish_turn(session_id="conversation-1")

    records = store.list_events(project_id="project-1", session_id="conversation-1")
    kinds = [record["kind"] for record in records]

    assert turn_id
    assert kinds == ["USER", "REQUEST", "SYSTEM", "CONTEXT", "ASSISTANT", "TOOL"]
    request = next(record for record in records if record["event_id"] == request_id)
    assistant = next(record for record in records if record["event_id"] == assistant_id)
    tool = next(record for record in records if record["event_id"] == tool_id)
    context = next(record for record in records if record["kind"] == "CONTEXT")
    assert request["status"] == "completed"
    assert request["details"]["model"] == "test-model"
    assert request["schema"][0]["name"] == "web_search"
    assert assistant["request_id"] == request_id
    assert assistant["details"]["usage"]["total_tokens"] == 16
    assert tool["request_id"] == request_id
    assert tool["call_id"] == "call-1"
    assert tool["output"]["sources"] == ["https://example.test"]
    assert tool["schema"]["name"] == "web_search"
    assert context["source"]["name"] == "relevant-memory-context"


def test_trajectory_records_system_diff_only_when_header_changes(tmp_path):
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")

    for prompt in ("First prompt", "First prompt", "Changed prompt"):
        store.begin_turn(
            session_id="conversation-1",
            project_id="project-1",
            content="next",
        )
        store.begin_request(
            session_id="conversation-1",
            messages=[SystemMessage(prompt), HumanMessage("next")],
            tools=[],
        )
        store.complete_request(
            session_id="conversation-1",
            response=AIMessage("done"),
        )
        store.finish_turn(session_id="conversation-1")

    records = store.list_events(project_id="project-1", session_id="conversation-1")
    systems = [record for record in records if record["kind"] == "SYSTEM"]
    assert len(systems) == 2
    assert systems[0]["details"]["change_kind"] == "initial"
    assert systems[1]["details"]["change_kind"] == "system"
    assert systems[1]["details"]["previous_prompt"] == "First prompt"


def test_failed_request_remains_visible(tmp_path):
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    store.begin_turn(
        session_id="conversation-1",
        project_id="project-1",
        content="fail",
    )
    request_id = store.begin_request(
        session_id="conversation-1",
        messages=[SystemMessage("prompt"), HumanMessage("fail")],
        tools=[],
    )
    store.fail_request(session_id="conversation-1", error="provider timeout")
    store.finish_turn(session_id="conversation-1")

    request = next(
        record
        for record in store.list_events(project_id="project-1", session_id="conversation-1")
        if record["event_id"] == request_id
    )
    assert request["status"] == "failed"
    assert request["error"] == "provider timeout"
    assert request["completed_at"] is not None


def test_compaction_diagnostics_are_visible_without_prompt_content(tmp_path):
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    store.begin_turn(
        session_id="conversation-compaction",
        project_id="project-1",
        content="continue",
    )
    store.begin_request(
        session_id="conversation-compaction",
        messages=[HumanMessage("continue")],
        tools=[],
    )
    store.record_diagnostics(
        session_id="conversation-compaction",
        entries=[{
            "type": "compaction",
            "reason": "provider_overflow",
            "model": "bounded-model",
            "tokens_before": 12_400,
            "tokens_after": 4_100,
            "context_window_tokens": 16_000,
            "overflow_retry": True,
        }],
    )
    store.finish_turn(session_id="conversation-compaction")

    event = next(
        record
        for record in store.list_events(
            project_id="project-1",
            session_id="conversation-compaction",
        )
        if record["kind"] == "COMPACTED"
    )
    assert event["status"] == "completed"
    assert event["input"]["tokens_before"] == 12_400
    assert event["input"]["tokens_after"] == 4_100
    assert event["input"]["overflow_retry"] is True
    assert "prompt" not in str(event["input"]).lower()


def test_self_modification_event_is_project_scoped_and_prompt_redacted(tmp_path):
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    event_id = store.record_self_modification_event(
        project_id="project-1",
        proposal_id="self-mod-1",
        event_type="SELF_MOD_CANARY",
        status="completed",
        payload={
            "candidate_hash": "a" * 64,
            "baseline_hash": "b" * 64,
            "candidate_text": "private experimental guidance",
            "canary_percent": 10,
            "decision": "canary",
        },
    )

    event = next(
        item
        for item in store.list_events(
            project_id="project-1",
            session_id="self-mod:self-mod-1",
        )
        if item["event_id"] == event_id
    )
    assert event["kind"] == "SELF_MOD_CANARY"
    assert event["output"]["canary_percent"] == 10
    assert "candidate_text" not in event["output"]
    assert "private experimental guidance" not in str(event)


def test_pipeline_run_is_a_redacted_causal_trajectory(tmp_path):
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    turn_id = store.begin_turn(
        session_id="conversation-pipeline",
        project_id="project-1",
        content="Prepare a brief",
        metadata={"pipeline_id": "brief"},
    )
    run_event_id = store.begin_pipeline_run(
        project_id="project-1",
        session_id="conversation-pipeline",
        pipeline_id="brief",
        pipeline_name="Daily brief",
        run_id="run-1",
        attempt_id="attempt-1",
        input_value="Authorization: Bearer private-key",
        definition_hash="abc123",
        steps=[{"id": "s1", "type": "router", "label": "Choose"}],
        trigger="manual-chat",
    )
    step_event_id = store.begin_pipeline_step(
        parent_event_id=run_event_id,
        step_id="s1",
        step_type="router",
        label="Choose",
        index=0,
        input_value={"password": "hidden", "topic": "release"},
    )
    store.complete_pipeline_step(
        event_id=step_event_id,
        output="token=very-secret selected output_2",
        route_outputs=["output_2"],
    )
    route_event_id = store.record_pipeline_route(
        step_event_id=step_event_id,
        selected_outputs=["output_2"],
    )
    result_event_id = store.complete_pipeline_run(
        event_id=run_event_id,
        status="completed",
        output="Brief ready",
        steps_run=1,
    )
    store.finish_turn(session_id="conversation-pipeline")

    records = store.list_events(
        project_id="project-1", session_id="conversation-pipeline"
    )
    assert [record["kind"] for record in records] == [
        "USER", "PIPELINE_RUN", "PIPELINE_STEP", "PIPELINE_ROUTE", "PIPELINE_RESULT"
    ]
    run = next(record for record in records if record["event_id"] == run_event_id)
    step = next(record for record in records if record["event_id"] == step_event_id)
    route = next(record for record in records if record["event_id"] == route_event_id)
    result = next(record for record in records if record["event_id"] == result_event_id)

    assert run["turn_id"] == turn_id
    assert run["status"] == "completed"
    assert "private-key" not in str(run["input"])
    assert run["details"]["definition_hash"] == "abc123"
    assert run["source"]["run_id"] == "run-1"
    assert run["source"]["attempt_id"] == "attempt-1"
    assert step["parent_id"] == run_event_id
    assert step["input"]["password"] == "***REDACTED***"
    assert "very-secret" not in str(step["output"])
    assert step["details"]["route_outputs"] == ["output_2"]
    assert step["source"]["definition_hash"] == "abc123"
    assert route["parent_id"] == step_event_id
    assert route["output"]["selected_outputs"] == ["output_2"]
    assert result["parent_id"] == run_event_id
    assert result["output"] == "Brief ready"


@pytest.mark.parametrize(
    ("scope", "event_kind"),
    [("experiment", "EXPERIMENT_MODEL"), ("automation", "AUTOMATION_STEP")],
)
def test_execution_run_is_project_scoped_redacted_and_causal(tmp_path, scope, event_kind):
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    run_event_id = store.begin_execution_run(
        scope=scope,
        project_id="project-1",
        source_id="source-1",
        source_name="Observed run",
        run_id="run-1",
        attempt_id="attempt-1",
        goal={"token": "secret-value", "topic": "release"},
        definition_hash="hash-1",
        schema={"password": "hidden", "steps": ["one"]},
    )
    child_event_id = store.record_execution_event(
        parent_event_id=run_event_id,
        event_kind=event_kind,
        name="Observed stage",
        input_value={"authorization": "Bearer private-key"},
        output_value={"result": "ready"},
        details={"step_id": "stage-1"},
    )
    result_event_id = store.complete_execution_run(
        event_id=run_event_id,
        status="completed",
        output={"answer": "done"},
        details={"steps_run": 1},
    )

    records = store.list_events(
        project_id="project-1",
        session_id=f"{scope}:source-1:run-1",
    )
    assert [record["kind"] for record in records] == [
        f"{scope.upper()}_RUN", event_kind, f"{scope.upper()}_RESULT"
    ]
    run = next(record for record in records if record["event_id"] == run_event_id)
    child = next(record for record in records if record["event_id"] == child_event_id)
    result = next(record for record in records if record["event_id"] == result_event_id)
    assert run["status"] == "completed"
    assert run["input"]["token"] == "***REDACTED***"
    assert run["schema"]["password"] == "***REDACTED***"
    assert child["parent_id"] == run_event_id
    assert "private-key" not in str(child["input"])
    assert result["parent_id"] == run_event_id
    assert result["output"] == {"answer": "done"}


def test_turn_completion_marks_unfinished_tool_interrupted(tmp_path):
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    store.begin_turn(
        session_id="conversation-1",
        project_id="project-1",
        content="run tool",
    )
    tool_id = store.begin_tool(
        session_id="conversation-1",
        call_id="call-running",
        name="slow_tool",
        payload={},
    )
    store.finish_turn(session_id="conversation-1", error="cancelled by user")

    tool = next(
        record
        for record in store.list_events(project_id="project-1", session_id="conversation-1")
        if record["event_id"] == tool_id
    )
    assert tool["status"] == "failed"
    assert tool["error"] == "cancelled by user"


def test_stream_first_output_and_cumulative_usage_are_projected(tmp_path):
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    request_ids = []
    first_output = None

    for index in range(2):
        store.begin_turn(
            session_id="conversation-1",
            project_id="project-1",
            content=f"turn {index}",
        )
        request_id = store.begin_request(
            session_id="conversation-1",
            messages=[SystemMessage("prompt"), HumanMessage(f"turn {index}")],
            tools=[],
        )
        request_ids.append(request_id)
        if index == 0:
            request_before = next(
                record
                for record in store.list_events(
                    project_id="project-1", session_id="conversation-1"
                )
                if record["event_id"] == request_id
            )
            first_output = float(request_before["started_at"]) + 0.001
            time.sleep(0.002)
            store.mark_first_output(
                session_id="conversation-1",
                timestamp=first_output,
            )
        store.complete_request(
            session_id="conversation-1",
            response=AIMessage("done"),
        )
        store.finish_turn(session_id="conversation-1")

    requests = [
        record
        for record in store.list_events(project_id="project-1", session_id="conversation-1")
        if record["kind"] == "REQUEST"
    ]
    assert requests[0]["first_output_at"] == first_output
    assert requests[0]["details"]["session_cumulative_usage"]["total_tokens"] == 16
    assert requests[1]["details"]["session_cumulative_usage"]["total_tokens"] == 32


def test_provider_attempts_capture_retry_and_enrich_parent_request(tmp_path):
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    store.begin_turn(
        session_id="conversation-1",
        project_id="project-1",
        content="retry this",
    )
    request_id = store.begin_request(
        session_id="conversation-1",
        messages=[SystemMessage("prompt"), HumanMessage("retry this")],
        tools=[],
    )
    first = store.begin_model_attempt(
        session_id="conversation-1",
        model="primary-model",
        provider="provider-a",
        model_index=0,
        attempt=1,
        purpose="agent",
        tools_enabled=False,
    )
    store.complete_model_attempt(
        event_id=first,
        success=False,
        duration_ms=18,
        error="timeout",
        retry_action="fallback-model",
    )
    second = store.begin_model_attempt(
        session_id="conversation-1",
        model="backup-model",
        provider="provider-b",
        model_index=1,
        attempt=1,
        purpose="agent",
        tools_enabled=False,
    )
    store.complete_model_attempt(
        event_id=second,
        success=True,
        duration_ms=12,
        output={"served_by": "backup-model", "has_output": True},
    )
    response = AIMessage("recovered")
    response.response_metadata["_served_by"] = "backup-model"
    response.response_metadata["_fallback_used"] = True
    store.complete_request(session_id="conversation-1", response=response)
    store.finish_turn(session_id="conversation-1")

    records = store.list_events(project_id="project-1", session_id="conversation-1")
    attempts = [record for record in records if record["kind"] == "ATTEMPT"]
    request = next(record for record in records if record["event_id"] == request_id)

    assert [record["status"] for record in attempts] == ["failed", "completed"]
    assert attempts[0]["details"]["retry_action"] == "fallback-model"
    assert attempts[1]["parent_id"] == request_id
    assert request["details"]["attempt_count"] == 2
    assert request["details"]["retry_count"] == 1
    assert request["details"]["attempt_models"] == ["primary-model", "backup-model"]


def test_trajectory_store_counts_full_session_outside_loaded_window(tmp_path):
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    for index in range(3):
        store.begin_turn(
            session_id="conversation-1",
            project_id="project-1",
            content=f"turn {index}",
        )
        store.finish_turn(session_id="conversation-1")

    records = store.list_events(
        project_id="project-1", session_id="conversation-1", limit=2
    )

    assert len(records) == 2
    assert store.count_events(
        project_id="project-1", session_id="conversation-1"
    ) == 3


def test_trajectory_store_lists_recent_project_window_without_cross_project_rows(tmp_path):
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    for project_id, session_id in (
        ("project-1", "conversation-1"),
        ("project-1", "conversation-2"),
        ("project-2", "conversation-3"),
    ):
        store.begin_turn(
            session_id=session_id,
            project_id=project_id,
            content=f"private {project_id}",
        )
        store.finish_turn(session_id=session_id)

    project_rows = store.list_project_events(project_id="project-1", limit=10)

    assert {row["session_id"] for row in project_rows} == {
        "conversation-1", "conversation-2"
    }
    assert store.count_project_events(project_id="project-1") == 2
    assert all(row["project_id"] == "project-1" for row in project_rows)


def test_trajectory_baseline_and_alert_lifecycle_is_durable_and_deduplicated(
    tmp_path, monkeypatch
):
    notifications = []
    monkeypatch.setattr(
        "remy.core.notification_router.notify",
        lambda message, **kwargs: notifications.append({"message": message, **kwargs}),
    )
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    baseline = store.create_analytics_baseline(
        project_id="project-1",
        name="Stable release",
        days=30,
        metrics={
            "failure_rate": 0,
            "avg_request_ms": 500,
            "tokens_per_request": 100,
            "raw_prompt": "must not persist",
        },
        summary={"sessions": 12, "requests": 25, "raw_output": "must not persist"},
        source_event_count=250,
    )
    regression = [{
        "conversation_id": "conversation-1",
        "event_id": "request-1",
        "severity": "error",
        "reasons": [{
            "metric": "latency", "observed": 3_000,
            "baseline": 500, "delta": 2_500,
        }],
    }]

    store.sync_regression_alerts(
        project_id="project-1",
        baseline_id=baseline["baseline_id"],
        session_id="conversation-1",
        regressions=regression,
    )
    store.sync_regression_alerts(
        project_id="project-1",
        baseline_id=baseline["baseline_id"],
        session_id="conversation-1",
        regressions=regression,
    )
    alerts = store.list_regression_alerts(project_id="project-1")

    assert len(alerts) == 1
    assert alerts[0]["status"] == "open"
    assert alerts[0]["delta"] == 2_500
    assert store.get_active_analytics_baseline(project_id="project-1") == baseline
    assert "raw_prompt" not in str(baseline)
    assert "raw_output" not in str(baseline)
    acknowledged = store.update_regression_alert(
        project_id="project-1", alert_id=alerts[0]["alert_id"], status="acknowledged"
    )
    assert acknowledged["status"] == "acknowledged"

    store.sync_regression_alerts(
        project_id="project-1",
        baseline_id=baseline["baseline_id"],
        session_id="conversation-1",
        regressions=[],
    )
    assert store.list_regression_alerts(project_id="project-1")[0]["status"] == "resolved"
    history = store.list_alert_history(project_id="project-1")
    assert [row["action"] for row in reversed(history)] == [
        "triggered", "acknowledged", "auto-resolved"
    ]
    assert all(row["project_id"] == "project-1" for row in history)
    assert "must not persist" not in str(history)
    assert len(notifications) == 2
    assert notifications[0]["event_type"] == "operator_alert"
    assert notifications[0]["event_data"]["source"] == "trajectory"
    assert notifications[0]["event_data"]["dedupe_key"].startswith("trajectory:alert-")
    assert notifications[1]["event_data"]["resolves"] == [
        notifications[0]["event_data"]["dedupe_key"]
    ]


def test_finish_turn_automatically_evaluates_active_regression_baseline(
    tmp_path, monkeypatch
):
    notifications = []
    monkeypatch.setattr(
        "remy.core.notification_router.notify",
        lambda message, **kwargs: notifications.append({"message": message, **kwargs}),
    )
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    baseline = store.create_analytics_baseline(
        project_id="project-1",
        name="Healthy baseline",
        days=30,
        metrics={"failure_rate": 0, "avg_request_ms": 60_000, "tokens_per_request": 0},
        summary={"sessions": 3, "requests": 8},
        source_event_count=40,
    )
    store.begin_turn(
        session_id="conversation-1",
        project_id="project-1",
        content="fail privately",
    )
    store.begin_request(
        session_id="conversation-1",
        messages=[HumanMessage("fail privately")],
        tools=[],
    )

    store.finish_turn(session_id="conversation-1", error="SECRET provider failure")

    alerts = store.list_regression_alerts(project_id="project-1", status="open")
    assert len(alerts) == 1
    assert alerts[0]["baseline_id"] == baseline["baseline_id"]
    assert alerts[0]["metric"] == "failure_rate"
    assert "SECRET" not in str(alerts)
    assert len(notifications) == 1
    assert "SECRET" not in str(notifications)


def test_alert_policy_is_validated_persisted_and_project_scoped(tmp_path):
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    policy = store.create_alert_policy(
        project_id="project-1",
        name="Search reliability",
        scope_type="tool",
        scope_value="web_search",
        thresholds={
            "failure_rate_warning": 0.1,
            "failure_rate_critical": 0.4,
            "latency_warning_ms": 5_000,
            "latency_critical_ms": 20_000,
        },
    )

    assert store.list_alert_policies(project_id="project-1") == [policy]
    assert store.list_alert_policies(project_id="project-2") == []
    updated = store.update_alert_policy(
        project_id="project-1",
        policy_id=policy["policy_id"],
        name="Search latency",
        scope_type="tool",
        scope_value="web_search",
        thresholds={"latency_warning_ms": 7_000, "latency_critical_ms": 25_000},
        enabled=False,
    )
    assert updated["enabled"] is False
    assert updated["thresholds"]["latency_warning_ms"] == 7_000
    assert store.delete_alert_policy(
        project_id="project-1", policy_id=policy["policy_id"]
    ) is True
    assert store.list_alert_policies(project_id="project-1") == []


def test_finish_turn_automatically_evaluates_tool_alert_policy_without_baseline(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("remy.core.notification_router.notify", lambda *args, **kwargs: None)
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    policy = store.create_alert_policy(
        project_id="project-1",
        name="Tool failure guard",
        scope_type="tool",
        scope_value="web_search",
        thresholds={"failure_rate_warning": 0.2, "failure_rate_critical": 1.0},
    )
    store.begin_turn(
        session_id="conversation-1",
        project_id="project-1",
        content="search",
    )
    tool_id = store.begin_tool(
        session_id="conversation-1",
        call_id="call-1",
        name="web_search",
        payload={"query": "private"},
    )
    store.complete_tool(event_id=tool_id, result=None, error="SECRET tool failure")

    store.finish_turn(session_id="conversation-1")

    alerts = store.list_regression_alerts(project_id="project-1", status="open")
    assert len(alerts) == 1
    assert alerts[0]["baseline_id"] == policy["policy_id"]
    assert alerts[0]["source_type"] == "policy"
    assert alerts[0]["metric"] == "tool.failure_rate"
    assert alerts[0]["severity"] == "error"
    assert "SECRET" not in str(alerts)
    store.create_analytics_baseline(
        project_id="project-1",
        name="New baseline",
        days=30,
        metrics={"failure_rate": 0, "avg_request_ms": 100, "tokens_per_request": 0},
        summary={"sessions": 1, "requests": 1},
        source_event_count=4,
    )
    still_open = store.list_regression_alerts(project_id="project-1", status="open")
    assert [row["alert_id"] for row in still_open] == [alerts[0]["alert_id"]]


def test_trajectory_slo_config_is_durable_validated_and_project_scoped(tmp_path):
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")

    default = store.get_slo_config(project_id="project-1")
    updated = store.update_slo_config(
        project_id="project-1",
        target_success_rate=0.995,
        window_days=14,
        min_operations=25,
    )

    assert default["target_success_rate"] == 0.99
    assert updated["target_success_rate"] == 0.995
    assert updated["window_days"] == 14
    assert updated["min_operations"] == 25
    assert store.get_slo_config(project_id="project-2")["target_success_rate"] == 0.99


def test_slo_incident_is_coalesced_acknowledged_and_auto_resolved(tmp_path, monkeypatch):
    notifications = []
    monkeypatch.setattr(
        "remy.core.notification_router.notify",
        lambda message, **kwargs: notifications.append({"message": message, **kwargs}),
    )
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    failing = {
        "status": "critical",
        "alert": {"reason": "fast-burn", "windows": ["1h", "6h"]},
        "windows": [
            {"label": "1h", "burn_rate": 25.0},
            {"label": "6h", "burn_rate": 12.0},
        ],
        "operations": 10,
        "failures": 2,
    }

    first = store.sync_slo_incidents(
        project_id="project-1",
        conversation_id="conversation-1",
        event_id="tool-1",
        slo=failing,
    )
    second = store.sync_slo_incidents(
        project_id="project-1",
        conversation_id="conversation-2",
        event_id="tool-2",
        slo=failing,
    )

    assert len(first) == len(second) == 1
    assert first[0]["incident_id"] == second[0]["incident_id"]
    assert second[0]["conversation_id"] == "conversation-2"
    assert len(notifications) == 1
    acknowledged = store.update_slo_incident(
        project_id="project-1",
        incident_id=first[0]["incident_id"],
        status="acknowledged",
    )
    assert acknowledged["status"] == "acknowledged"

    store.sync_slo_incidents(
        project_id="project-1",
        conversation_id="conversation-2",
        event_id="request-3",
        slo={
            "status": "healthy",
            "alert": {"reason": "within-budget", "windows": []},
            "windows": [],
            "operations": 20,
            "failures": 0,
        },
    )

    resolved = store.list_slo_incidents(project_id="project-1")[0]
    assert resolved["status"] == "resolved"
    assert len(notifications) == 2
    assert notifications[1]["event_data"]["resolves"] == [
        notifications[0]["event_data"]["dedupe_key"]
    ]
    history = store.list_alert_history(project_id="project-1")
    assert [row["action"] for row in reversed(history)] == [
        "triggered", "acknowledged", "auto-resolved"
    ]
    assert all((row["details"] or {}).get("source_type") == "slo" for row in history)


def test_finish_turn_automatically_evaluates_project_slo(tmp_path, monkeypatch):
    monkeypatch.setattr("remy.core.notification_router.notify", lambda *args, **kwargs: None)
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    store.begin_turn(
        session_id="conversation-1",
        project_id="project-1",
        content="run tools",
    )
    for index in range(5):
        event_id = store.begin_tool(
            session_id="conversation-1",
            call_id=f"call-{index}",
            name="unstable_tool",
            payload={"private": index},
        )
        store.complete_tool(event_id=event_id, result=None, error="SECRET failure")

    store.finish_turn(session_id="conversation-1")

    incidents = store.list_slo_incidents(project_id="project-1", status="open")
    assert len(incidents) == 1
    assert incidents[0]["reason"] == "fast-burn"
    assert incidents[0]["severity"] == "critical"
    assert incidents[0]["operations"] == 5
    assert "SECRET" not in str(incidents)


def test_trajectory_eval_case_and_runs_are_durable_and_project_scoped(tmp_path):
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    case = store.create_eval_case(
        project_id="project-1",
        incident_id="alert-1",
        name="Search regression",
        source_conversation_id="conversation-1",
        criteria={
            "max_failure_rate": 0.2,
            "max_error_count": 0,
            "blocked_fingerprints": ["fingerprint-1"],
        },
        baseline_snapshot={
            "status": "failed", "score": 25,
            "metrics": {
                "failure_rate": 1, "fingerprints": ["fingerprint-1"],
                "raw_output": "must not persist",
            },
            "raw_output": "must not persist",
        },
    )
    run = store.record_eval_run(
        project_id="project-1",
        case_id=case["case_id"],
        candidate_conversation_id="conversation-2",
        evaluation={
            "status": "passed", "score": 100,
            "checks": [{
                "check_id": "max_error_count", "passed": True,
                "raw_error": "must not persist",
            }],
            "metrics": {
                "failure_rate": 0, "fingerprints": [],
                "raw_output": "must not persist",
            },
            "comparison": {
                "removed_fingerprints": ["fingerprint-1"],
                "raw_error": "must not persist",
            },
            "raw_output": "must not persist",
        },
        mode="sandbox-replay",
        replay={
            "sandboxed": True, "status": "completed", "tool_calls": 1,
            "fixture_hits": 1, "blocked_calls": 0,
            "decisions": [{
                "tool": "web_search", "action": "fixture-hit",
                "args_fingerprint": "a" * 64,
                "fixture_event_id": "tool-1", "source_status": "completed",
                "raw_output": "must not persist",
            }],
            "raw_output": "must not persist",
        },
    )

    loaded = store.get_eval_case(project_id="project-1", case_id=case["case_id"])
    assert loaded["latest_run"]["run_id"] == run["run_id"]
    assert loaded["latest_run"]["status"] == "passed"
    assert loaded["latest_run"]["mode"] == "sandbox-replay"
    assert loaded["latest_run"]["replay"]["side_effects_executed"] == 0
    assert store.list_eval_cases(project_id="project-2") == []
    assert "must not persist" not in str(loaded)
    assert store.delete_eval_case(project_id="project-1", case_id=case["case_id"])
    assert store.list_eval_runs(
        project_id="project-1", case_id=case["case_id"]
    ) == []


def test_trajectory_eval_matrix_is_a_durable_project_scoped_release_gate(tmp_path):
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    matrix = store.create_eval_matrix(
        project_id="project-1",
        name="Release 0.9 gate",
        preferred_model="provider/model-v2",
        agent_version="0.9.0",
        case_count=2,
    )
    store.record_eval_matrix_entry(
        project_id="project-1",
        matrix_id=matrix["matrix_id"],
        case_id="case-1",
        run_id="run-1",
        candidate_conversation_id="conversation-1",
        status="passed",
        score=100,
    )
    store.record_eval_matrix_entry(
        project_id="project-1",
        matrix_id=matrix["matrix_id"],
        case_id="case-2",
        status="error",
        error_type="SECRET raw provider failure that must not persist" * 10,
    )

    completed = store.complete_eval_matrix(
        project_id="project-1", matrix_id=matrix["matrix_id"]
    )

    assert completed["status"] == "failed"
    assert completed["gate_passed"] is False
    assert completed["passed_count"] == 1
    assert completed["error_count"] == 1
    assert completed["preferred_model"] == "provider/model-v2"
    assert len(completed["entries"]) == 2
    assert "SECRET" not in completed["entries"][1]["error_type"]
    assert store.list_eval_matrices(project_id="project-2") == []

    passing = store.create_eval_matrix(
        project_id="project-1", name="Passing gate",
        preferred_model="provider/model-v2", agent_version="0.9.0", case_count=1,
    )
    store.record_eval_matrix_entry(
        project_id="project-1", matrix_id=passing["matrix_id"],
        case_id="case-1", status="passed", score=100,
    )
    passing = store.complete_eval_matrix(
        project_id="project-1", matrix_id=passing["matrix_id"]
    )
    assert passing["gate_passed"] is True

    interrupted = store.create_eval_matrix(
        project_id="project-1", name="Interrupted gate",
        preferred_model="provider/model-v3", agent_version="0.9.1", case_count=3,
    )
    reopened = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    recovered = next(
        row for row in reopened.list_eval_matrices(project_id="project-1")
        if row["matrix_id"] == interrupted["matrix_id"]
    )
    assert recovered["status"] == "failed"
    assert recovered["error_count"] == 3


def test_trajectory_model_comparison_is_durable_ranked_and_project_scoped(tmp_path):
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    comparison = store.create_eval_comparison(
        project_id="project-1",
        name="Candidate models",
        agent_version="0.10.0",
        model_count=3,
        case_count=4,
    )
    matrices = [
        {
            "matrix_id": "matrix-fast", "preferred_model": "provider/fast",
            "status": "failed", "gate_passed": False,
            "passed_count": 3, "failed_count": 1, "error_count": 0,
            "avg_score": 94, "avg_failure_rate": 0.01,
            "avg_request_ms": 400, "avg_tokens_per_request": 800,
        },
        {
            "matrix_id": "matrix-safe", "preferred_model": "provider/safe",
            "status": "passed", "gate_passed": True,
            "passed_count": 4, "failed_count": 0, "error_count": 0,
            "avg_score": 91, "avg_failure_rate": 0,
            "avg_request_ms": 900, "avg_tokens_per_request": 900,
        },
        {
            "matrix_id": "matrix-error", "preferred_model": "provider/error",
            "status": "failed", "gate_passed": False,
            "passed_count": 0, "failed_count": 0, "error_count": 4,
            "avg_score": 0, "avg_failure_rate": 0,
            "avg_request_ms": 0, "avg_tokens_per_request": 0,
        },
    ]
    for matrix in matrices:
        store.record_eval_comparison_model(
            project_id="project-1",
            comparison_id=comparison["comparison_id"],
            matrix=matrix,
        )

    completed = store.complete_eval_comparison(
        project_id="project-1", comparison_id=comparison["comparison_id"]
    )

    assert completed["status"] == "completed"
    assert completed["winner_model"] == "provider/safe"
    assert [row["preferred_model"] for row in completed["models"]] == [
        "provider/safe", "provider/fast", "provider/error",
    ]
    assert [row["rank"] for row in completed["models"]] == [1, 2, 3]
    assert store.list_eval_comparisons(project_id="project-2") == []

    interrupted = store.create_eval_comparison(
        project_id="project-1", name="Interrupted", agent_version="0.10.1",
        model_count=2, case_count=4,
    )
    reopened = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    recovered = next(
        row for row in reopened.list_eval_comparisons(project_id="project-1")
        if row["comparison_id"] == interrupted["comparison_id"]
    )
    assert recovered["status"] == "failed"
    assert recovered["winner_model"] == ""


def test_trajectory_model_promotion_canary_is_durable_ready_and_rollback_safe(tmp_path):
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    promotion = store.create_model_promotion(
        project_id="project-1",
        previous_model="provider/current",
        candidate_model="provider/candidate",
        canary_percent=80,
        evidence_comparison_ids=["comparison-1", "comparison-2", "comparison-3"],
    )
    assert promotion["status"] == "canary"
    assert promotion["canary_percent"] == 10
    assert promotion["ramp_stage"] == 0
    assert promotion["ramp_complete"] is False
    assert promotion["max_requests_per_arm"] == 200
    assert promotion["familywise_alpha"] == 0.05
    assert promotion["max_canary_hours"] == 24
    assert store.get_active_model_promotion(project_id="project-2") is None

    def comparison(comparison_id, *, healthy=True):
        return {
            "comparison_id": comparison_id,
            "status": "completed",
            "winner_model": "provider/candidate" if healthy else "provider/current",
            "models": [
                {
                    "preferred_model": "provider/candidate",
                    "rank": 1 if healthy else 2,
                    "gate_passed": healthy,
                    "error_count": 0,
                },
                {
                    "preferred_model": "provider/current",
                    "rank": 2 if healthy else 1,
                    "gate_passed": True,
                    "error_count": 0,
                },
            ],
        }

    observed = store.observe_model_promotion_comparison(
        project_id="project-1", comparison=comparison("comparison-4")
    )
    assert observed["status"] == "canary"
    assert observed["healthy_comparisons"] == 1
    ready = store.observe_model_promotion_comparison(
        project_id="project-1", comparison=comparison("comparison-5")
    )
    assert ready["status"] == "ready"
    assert ready["healthy_comparisons"] == 2

    healthy_telemetry = lambda requests: {
        "status": "healthy",
        "candidate": {"requests": requests},
        "control": {"requests": requests},
    }
    ramp_windows = [
        (100, "2026-08-20T10:00:00+00:00"),
        (101, "2026-08-20T10:01:00+00:00"),
        (101, "2026-08-20T10:05:00+00:00"),
        (102, "2026-08-20T10:05:00+00:00"),
        (100, "2026-08-20T10:10:00+00:00"),
        (101, "2026-08-20T10:15:00+00:00"),
        (100, "2026-08-20T10:20:00+00:00"),
        (101, "2026-08-20T10:25:00+00:00"),
    ]
    ramp = ready
    for index, (requests, observed_at) in enumerate(ramp_windows):
        ramp = store.observe_model_promotion_telemetry(
            project_id="project-1",
            promotion_id=promotion["promotion_id"],
            telemetry=healthy_telemetry(requests),
            observed_at=observed_at,
        )
        if index == 1:
            assert ramp["ramp_stage"] == 0
            assert ramp["healthy_windows"] == 1
        if index == 2:
            assert ramp["ramp_stage"] == 0
            assert ramp["healthy_windows"] == 1
    assert ramp["canary_percent"] == 50
    assert ramp["ramp_stage"] == 2
    assert ramp["ramp_complete"] is True
    assert [row["percent"] for row in ramp["ramp_history"]] == [10, 25, 50, 50]

    promoted = store.mark_model_promotion_promoted(
        project_id="project-1", promotion_id=promotion["promotion_id"]
    )
    assert promoted["status"] == "promoted"
    rollback_pending = store.observe_model_promotion_comparison(
        project_id="project-1",
        comparison=comparison("comparison-6", healthy=False),
    )
    assert rollback_pending["status"] == "rollback_pending"
    assert rollback_pending["rollback_reason"] == "candidate_gate_failed"
    rolled_back = store.complete_model_promotion_auto_rollback(
        project_id="project-1", promotion_id=promotion["promotion_id"]
    )
    assert rolled_back["status"] == "rolled_back"
    assert rolled_back["rollback_reason"] == "candidate_gate_failed"
    assert store.get_active_model_promotion(project_id="project-1") is None

    reopened = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    history = reopened.list_model_promotions(project_id="project-1")
    assert history[0]["status"] == "rolled_back"
    assert history[0]["evidence_comparison_ids"] == [
        "comparison-1", "comparison-2", "comparison-3",
    ]


def test_trajectory_lists_only_requests_assigned_to_specific_canary(tmp_path):
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    for index, promotion_id in enumerate(("promotion-1", "promotion-other")):
        session_id = f"conversation-{index}"
        store.begin_turn(
            session_id=session_id,
            project_id="project-1",
            content="test canary telemetry",
        )
        store.begin_request(
            session_id=session_id,
            messages=[HumanMessage("test")],
            tools=[],
            routing={
                "promotion_id": promotion_id,
                "canary_applied": index == 0,
                "preferred_model": "provider/candidate",
            },
        )
        store.complete_request(session_id=session_id, response=AIMessage("done"))
        store.finish_turn(session_id=session_id)

    requests = store.list_model_promotion_requests(
        project_id="project-1", promotion_id="promotion-1"
    )

    assert len(requests) == 1
    assert requests[0]["kind"] == "REQUEST"
    assert requests[0]["details"]["options"]["promotion_id"] == "promotion-1"
    assert requests[0]["details"]["usage"]["total_tokens"] == 16


def test_trajectory_annotations_are_persisted_enriched_and_deletable(tmp_path):
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    store.begin_turn(
        session_id="conversation-1",
        project_id="project-1",
        content="inspect this",
    )
    store.finish_turn(session_id="conversation-1")
    event_id = store.list_events(
        project_id="project-1", session_id="conversation-1"
    )[0]["event_id"]

    annotation = store.set_annotation(
        project_id="project-1",
        session_id="conversation-1",
        event_id=event_id,
        label="investigate",
        note="Provider latency increased after retry.",
        bookmarked=True,
    )
    record = store.list_events(
        project_id="project-1", session_id="conversation-1"
    )[0]

    assert annotation["label"] == "investigate"
    assert annotation["bookmarked"] is True
    assert record["annotation"] == annotation
    assert store.delete_annotation(
        project_id="project-1",
        session_id="conversation-1",
        event_id=event_id,
    ) is True
    assert store.list_events(
        project_id="project-1", session_id="conversation-1"
    )[0]["annotation"] is None


def test_trajectory_mutations_publish_scoped_live_hints(tmp_path, monkeypatch):
    from remy.core.event_bus import event_bus

    emitted = []
    monkeypatch.setattr(event_bus, "emit", lambda name, payload: emitted.append((name, payload)))
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")

    store.begin_turn(
        session_id="conversation-1",
        project_id="project-1",
        content="watch this live",
    )
    store.begin_request(
        session_id="conversation-1",
        messages=[HumanMessage("watch this live")],
        tools=[],
    )
    store.complete_request(
        session_id="conversation-1",
        response=AIMessage("live result"),
    )
    event_id = store.list_events(
        project_id="project-1", session_id="conversation-1"
    )[0]["event_id"]
    store.finish_turn(session_id="conversation-1")
    store.set_annotation(
        project_id="project-1",
        session_id="conversation-1",
        event_id=event_id,
        label="investigate",
        note="Live incident",
        bookmarked=True,
    )

    assert all(name == "trajectory.changed" for name, _ in emitted)
    appended = emitted[0][1]
    assert appended["event_domain"] == "trajectory"
    assert appended["owner_project_id"] == "project-1"
    assert appended["payload"] == {
        "conversation_id": "conversation-1",
        "project_id": "project-1",
        "event_id": event_id,
        "change": "append",
        "kind": "USER",
        "status": "completed",
        "sequence": 1,
        "fields": [],
    }
    assert any(payload["change"] == "update" for _, payload in emitted)
    annotation_hint = next(
        payload for _, payload in emitted if payload["change"] == "annotation"
    )
    assert "note" not in annotation_hint
    assert "Live incident" not in str(annotation_hint)


def test_trajectory_records_non_executing_fork_provenance(tmp_path):
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")

    event_id = store.record_fork(
        session_id="fork-conversation",
        project_id="project-1",
        source_session_id="source-conversation",
        boundary_event_id="event-42",
        boundary_sequence=42,
        boundary_kind="TOOL",
        copied_messages=6,
        preferred_model="test-model",
    )
    record = store.list_events(
        project_id="project-1", session_id="fork-conversation"
    )[0]

    assert record["event_id"] == event_id
    assert record["kind"] == "FORK"
    assert record["source"]["kind"] == "session-fork"
    assert record["details"]["boundary_event_id"] == "event-42"
    assert record["details"]["copied_messages"] == 6
    assert record["details"]["auto_executed"] is False


def test_trajectory_frontend_contract_is_wired():
    from pathlib import Path

    html = Path("src/remy/web/static/index.html").read_text(encoding="utf-8")
    javascript = Path("src/remy/web/static/js/trajectory.js").read_text(encoding="utf-8")
    chat_javascript = Path("src/remy/web/static/js/chat.js").read_text(encoding="utf-8")
    api_client = Path("src/remy/web/static/js/api-client.js").read_text(encoding="utf-8")
    app_javascript = Path("src/remy/web/static/js/app.js").read_text(encoding="utf-8")
    stylesheet = Path("src/remy/web/static/css/main.css").read_text(encoding="utf-8")

    assert 'id="chat-surface-trajectory"' in html
    assert 'id="trajectory-inspector"' in html
    assert '/js/trajectory.js?v=4.0' in html
    assert '/js/api-client.js?v=1.49' in html
    assert '/js/chat.js?v=1.36' in html
    assert "getConversationTrajectory" in javascript
    assert "tabsFor" in javascript
    assert "renderDiagnostics" in javascript
    assert "Alpha-spending repeated-look guardrail" in javascript
    assert "trajectory-confidence-intervals" in javascript
    assert "Adaptive sample plan" in javascript
    assert "trajectory-sample-metric-plans" in javascript
    assert 'id="trajectory-problems-only"' in html
    assert 'id="trajectory-diagnostics"' in html
    assert 'id="trajectory-turn-filter"' in html
    assert 'id="trajectory-replay-play"' in html
    assert 'id="trajectory-compare-panel"' in html
    assert 'id="trajectory-chain-clear"' in html
    assert 'id="trajectory-report"' in html
    assert 'id="trajectory-root-causes"' in html
    assert 'id="trajectory-recovery-paths"' in html
    assert 'id="trajectory-error-clusters"' in html
    assert 'id="trajectory-timeline-modes"' in html
    assert 'id="trajectory-timing-breakdown"' in html
    assert 'id="trajectory-integrity-health"' in html
    assert 'id="trajectory-integrity"' in html
    assert 'id="trajectory-window-status"' in html
    assert 'id="trajectory-load-older"' in html
    assert "toggleReplay" in javascript
    assert "showReplayRecord" in javascript
    assert "renderComparison" in javascript
    assert "focusCausalChain" in javascript
    assert "buildDebugReport" in javascript
    assert "downloadDebugReport" in javascript
    assert "renderTimingBreakdown" in javascript
    assert 'timelineMode === "calls"' in javascript
    assert "diagnostics.integrity" in javascript
    assert "loadOlderTrajectory" in javascript
    assert "trajectoryLimit = 750" in javascript
    assert "LEDGER_VIRTUALIZATION_THRESHOLD = 100" in javascript
    assert "LEDGER_VIRTUAL_OVERSCAN = 12" in javascript
    assert "function ledgerVirtualRange" in javascript
    assert "function revealLedgerRecord" in javascript
    assert 'className = `trajectory-ledger-spacer trajectory-ledger-spacer-${position}`' in javascript
    assert ".trajectory-ledger.virtualized .trajectory-record" in stylesheet
    assert "trajectory-ledger-window-control" not in javascript
    assert "TIMELINE_BAR_LIMIT = 1200" in javascript
    assert "ensureLedgerIndex" in javascript
    assert "issues preserved" in javascript
    assert 'id="trajectory-bookmarks-only"' in html
    assert "wireAnnotationEditor" in javascript
    assert "updateTrajectoryAnnotation" in api_client
    assert "deleteTrajectoryAnnotation" in api_client
    assert "error_clusters" in javascript
    assert "Error fingerprints" in javascript
    assert 'id="trajectory-live"' in html
    assert 'event?.type !== "trajectory.changed"' in javascript
    assert "onRuntimeStatus" in javascript
    assert "liveDraftElement" in javascript
    assert "_runtimeStatusHandlers" in api_client
    assert '<option value="FORK">Forks</option>' in html
    assert '<option value="PIPELINE">Pipelines</option>' in html
    assert '<option value="EXPERIMENT">Experiments</option>' in html
    assert '<option value="AUTOMATION">Automations</option>' in html
    assert 'PIPELINE_RUN: "pipeline"' in javascript
    assert 'kind === "PIPELINE_RUN"' in javascript
    assert 'evt.type === "trajectory_link"' in chat_javascript
    assert 'conversation_id: _conversationId' in chat_javascript
    assert 'trajectory-open-alert' in chat_javascript
    assert ".pipeline-trajectory-link" in stylesheet
    assert 'EXPERIMENT_RUN: "experiment"' in javascript
    assert 'AUTOMATION_RUN: "automation"' in javascript
    assert "trajectory-open-execution" in javascript
    assert "execution-trajectory-open" in app_javascript
    assert ".execution-trajectory-link" in stylesheet
    assert 'Fork: () => forkContent(record)' in javascript
    assert "wireForkEditor" in javascript
    assert "forkConversationTrajectory" in api_client
    assert "fork-replay-ready" in javascript
    assert 'id="trajectory-analytics"' in html
    assert 'id="trajectory-analytics-open"' in html
    assert "loadProjectAnalytics" in javascript
    assert "renderProjectAnalytics" in javascript
    assert "openAnalyticsRecord" in javascript
    assert "getTrajectoryAnalytics" in api_client
    assert ".trajectory-analytics-grid" in stylesheet
    assert 'id="trajectory-baseline-select"' in html
    assert 'id="trajectory-baseline-create"' in html
    assert 'id="trajectory-alert-count"' in html
    assert "createTrajectoryBaseline" in api_client
    assert "activateTrajectoryBaseline" in api_client
    assert "updateTrajectoryAlert" in api_client
    assert "createTrajectoryAlertPolicy" in api_client
    assert "updateTrajectoryAlertPolicy" in api_client
    assert "deleteTrajectoryAlertPolicy" in api_client
    assert "getTrajectoryAlertHistory" in api_client
    assert "updateTrajectorySlo" in api_client
    assert "getTrajectorySloIncidents" in api_client
    assert "updateTrajectorySloIncident" in api_client
    assert "getTrajectoryIncidentDossier" in api_client
    assert "createTrajectoryEvalCase" in api_client
    assert "runTrajectoryEvalCase" in api_client
    assert "sandboxReplayTrajectoryEvalCase" in api_client
    assert "runTrajectoryEvalMatrix" in api_client
    assert "getTrajectoryEvalMatrices" in api_client
    assert "runTrajectoryEvalComparison" in api_client
    assert "getTrajectoryEvalComparisons" in api_client
    assert "deleteTrajectoryEvalCase" in api_client
    assert "trajectory-policy-form" in javascript
    assert "wirePolicyEditor" in javascript
    assert "trajectory-slo-form" in javascript
    assert "wireSloEditor" in javascript
    assert "Alert lifecycle" in javascript
    assert "trajectory-slo-incidents" in javascript
    assert "showIncidentDossier" in javascript
    assert "Incident dossier" in javascript
    assert "Download Markdown" in javascript
    assert "Create regression eval" in javascript
    assert "Regression evals" in javascript
    assert "data-eval-run" in javascript
    assert "data-eval-replay" in javascript
    assert "Sandbox replay" in javascript
    assert "Run release gate" in javascript
    assert "Release allowed" in javascript
    assert "Compare models" in javascript
    assert "Winner ·" in javascript
    assert "Promotion candidate ·" in javascript
    assert "Recommendation only" in javascript
    assert "Start canary" in javascript
    assert "Promote model" in javascript
    assert "Roll back" in javascript
    assert "Production canary health" in javascript
    assert "traffic ramp" in javascript
    assert "startTrajectoryModelPromotion" in api_client
    assert "finalizeTrajectoryModelPromotion" in api_client
    assert "rollbackTrajectoryModelPromotion" in api_client
    assert ".trajectory-matrix-list" in stylesheet
    assert ".trajectory-comparison-ranking" in stylesheet
    assert ".trajectory-promotion-metrics" in stylesheet
    assert ".trajectory-promotion-history" in stylesheet
    assert ".trajectory-canary-arms" in stylesheet
    assert ".trajectory-ramp-stages" in stylesheet
    assert 'id="btn-operator-alerts"' in html
    assert 'id="operator-alert-count"' in html
    assert "acknowledgeOperatorAlert" in api_client
    assert "getOperatorAlerts" in api_client
    assert "_initOperatorAlertCenter" in app_javascript
    assert "trajectory-open-alert" in app_javascript
    assert "trajectory-open-alert" in javascript
    assert ".operator-alert-center" in stylesheet
    assert ".trajectory-policy-form" in stylesheet
    assert ".trajectory-slo-card" in stylesheet
    assert ".trajectory-alert-history" in stylesheet
    assert ".trajectory-incident-dossier" in stylesheet
    assert ".trajectory-eval-list" in stylesheet
    assert 'event?.type === "trajectory.analytics.changed"' in javascript
    assert ".trajectory-analytics-alerts" in stylesheet
    assert ".trajectory-inspector" in stylesheet
