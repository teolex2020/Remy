from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from remy.core.trajectory_replay import (
    ReplaySandbox,
    activate_replay_sandbox,
    build_replay_spec,
    current_replay_sandbox,
    run_trajectory_sandbox_replay,
)
from remy.core.trajectory_diagnostics import evaluate_trajectory_regression
from remy.core.trajectory_store import TrajectoryStore


def _source_records():
    return [
        {
            "sequence": 1, "event_id": "user-0", "turn_id": "turn-0",
            "kind": "USER", "status": "completed", "input": "Earlier question",
            "output": "Earlier question", "details": {},
        },
        {
            "sequence": 2, "event_id": "assistant-0", "turn_id": "turn-0",
            "kind": "ASSISTANT", "status": "completed",
            "output": {"content": "Earlier answer"}, "details": {},
        },
        {
            "sequence": 3, "event_id": "user-1", "turn_id": "turn-1",
            "kind": "USER", "status": "completed", "input": "Write the report",
            "output": "Write the report", "details": {},
        },
        {
            "sequence": 4, "event_id": "tool-1", "turn_id": "turn-1",
            "kind": "TOOL", "status": "completed",
            "input": {"path": "report.md"}, "output": "recorded fixture",
            "source": {"name": "write_file"}, "details": {"name": "write_file"},
        },
    ]


def test_replay_spec_and_sandbox_require_exact_fixture_match():
    spec = build_replay_spec(
        _source_records(), selected_event={"event_id": "tool-1", "turn_id": "turn-1"}
    )
    sandbox = ReplaySandbox(spec["fixtures"])

    fixture_result, fixture_policy = sandbox.resolve(
        "write_file", {"path": "report.md"}
    )
    blocked_result, blocked_policy = sandbox.resolve(
        "write_file", {"path": "different.md"}
    )

    assert spec["prompt"] == "Write the report"
    assert spec["history"] == [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
    ]
    assert fixture_result == "recorded fixture"
    assert fixture_policy["action"] == "fixture-hit"
    assert blocked_result["sandbox_replay"]["status"] == "blocked"
    assert blocked_policy["action"] == "blocked"
    assert sandbox.summary()["side_effects_executed"] == 0


def test_agent_tool_node_never_invokes_real_tool_during_replay(monkeypatch):
    from remy.core import agent

    calls = []

    class FakeTool:
        name = "write_file"

        def invoke(self, args):
            calls.append(args)
            return "REAL SIDE EFFECT"

    fake_trajectory = SimpleNamespace(
        begin_tool=lambda **kwargs: "",
        complete_tool=lambda **kwargs: None,
    )
    monkeypatch.setattr(agent, "get_all_tools", lambda: [FakeTool()])
    monkeypatch.setattr(
        "remy.core.trajectory_store.get_trajectory_store", lambda: fake_trajectory
    )
    spec = build_replay_spec(
        _source_records(), selected_event={"event_id": "tool-1", "turn_id": "turn-1"}
    )
    sandbox = ReplaySandbox(spec["fixtures"])
    state = {
        "messages": [AIMessage(content="", tool_calls=[{
            "name": "write_file", "args": {"path": "report.md"}, "id": "call-1"
        }])],
        "session_id": "sandbox-session",
        "channel": "trajectory-replay",
        "session_log": [],
        "enabled_tools": set(),
    }

    with activate_replay_sandbox(sandbox):
        result = agent.call_tools(state)

    assert calls == []
    assert result["messages"][0].content == "recorded fixture"
    assert result["session_log"][0]["consequence_gate"]["policy_hint"]["sandbox_replay"]


def test_sandbox_block_is_a_deterministic_regression_failure():
    records = [{
        "event_id": "tool-new", "turn_id": "turn-1", "kind": "TOOL",
        "status": "completed", "error": "", "started_at": 1.0,
        "completed_at": 1.1, "duration_ms": 100,
        "details": {"policy": {"sandbox_replay": True, "action": "blocked"}},
    }]
    evaluation = evaluate_trajectory_regression(
        records,
        criteria={
            "max_error_count": 0, "max_sandbox_blocked": 0,
            "min_trace_coverage": 0,
        },
    )

    blocked = next(
        check for check in evaluation["checks"]
        if check["check_id"] == "max_sandbox_blocked"
    )
    assert evaluation["status"] == "failed"
    assert evaluation["metrics"]["sandbox_blocked_calls"] == 1
    assert blocked["passed"] is False


@pytest.mark.asyncio
async def test_sandbox_replay_is_durable_but_excluded_from_project_analytics(tmp_path):
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    created = SimpleNamespace(
        conversation_id="51d42e51-8e0e-4cb3-9532-691759b9ff19",
        to_dict=lambda: {"conversation_id": "51d42e51-8e0e-4cb3-9532-691759b9ff19"},
    )
    conversations = SimpleNamespace(create=lambda *args, **kwargs: created)

    async def fake_invoke(*, prompt, history, session_id, preferred_model=""):
        assert preferred_model == "model-under-test"
        assert current_replay_sandbox() is not None
        result, policy = current_replay_sandbox().resolve(
            "write_file", {"path": "report.md"}
        )
        assert result == "recorded fixture"
        assert policy["side_effect_executed"] is False
        store.begin_request(
            session_id=session_id,
            messages=[HumanMessage(content=prompt)],
            tools=[],
        )
        response = AIMessage(content="The report would be written in a real run.")
        store.complete_request(session_id=session_id, response=response)
        return response.content, [response], []

    result = await run_trajectory_sandbox_replay(
        project_id="project-1",
        case={
            "case_id": "case-1", "name": "Report regression",
            "source_conversation_id": "conversation-1",
        },
        source_records=_source_records(),
        selected_event={"event_id": "tool-1", "turn_id": "turn-1"},
        trajectory_store=store,
        conversation_store=conversations,
        preferred_model="model-under-test",
        invoke=fake_invoke,
    )

    assert result["replay"]["fixture_hits"] == 1
    assert result["replay"]["blocked_calls"] == 0
    assert result["replay"]["side_effects_executed"] == 0
    assert store.is_replay_session(
        project_id="project-1", session_id=created.conversation_id
    )
    assert store.list_project_events(project_id="project-1") == []
    assert any(row["kind"] == "REQUEST" for row in result["records"])
