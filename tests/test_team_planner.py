from pathlib import Path

import pytest

from remy.config.settings import settings
from remy.core.team_planner import (
    TEAM_ROLE_TOOL_CEILINGS,
    TeamLimits,
    normalize_team_mode,
    plan_agent_team,
    run_agent_team,
    validate_team_plan,
)
from remy.core.trajectory_store import TrajectoryStore
from remy.core.worker import WorkerResult, get_worker_tools


def _team_plan():
    return {
        "team_required": True,
        "reason": "Independent research and analysis can run in parallel",
        "members": [
            {
                "id": "research",
                "role": "researcher",
                "instruction": "Collect and verify relevant evidence for the task.",
            },
            {
                "id": "analysis",
                "role": "analyst",
                "instruction": "Analyze available project evidence and identify conflicts.",
            },
        ],
    }


def test_team_mode_defaults_to_off_and_off_never_calls_planner():
    called = []
    result = plan_agent_team(
        "Simple question",
        mode="invalid-from-client",
        planner=lambda *args: called.append(args),
    )

    assert normalize_team_mode(None) == "off"
    assert normalize_team_mode("ADAPTIVE") == "adaptive"
    assert result["mode"] == "off"
    assert result["team_required"] is False
    assert called == []


def test_adaptive_mode_can_choose_one_agent_without_spawning_a_team():
    result = plan_agent_team(
        "What is two plus two?",
        mode="adaptive",
        planner=lambda *args: {
            "team_required": False,
            "reason": "A direct answer is sufficient",
            "members": [],
        },
    )

    assert result["valid"] is True
    assert result["team_required"] is False
    assert result["members"] == []


def test_force_mode_requires_a_real_bounded_team():
    no_team = validate_team_plan(
        {"team_required": False, "reason": "No team", "members": []},
        mode="force",
    )
    oversized = validate_team_plan(
        {
            "team_required": True,
            "reason": "Too many",
            "members": [
                {
                    "id": f"member_{index}",
                    "role": "researcher",
                    "instruction": "Investigate one independent evidence source.",
                }
                for index in range(4)
            ],
        },
        mode="force",
    )

    assert no_team["valid"] is False
    assert "force mode" in " ".join(no_team["errors"])
    assert oversized["valid"] is False
    assert "2-3 members" in " ".join(oversized["errors"])


def test_team_gate_rejects_executor_and_model_supplied_capabilities():
    raw = _team_plan()
    raw["members"][0] = {
        "id": "writer",
        "role": "executor",
        "instruction": "Modify project files and deploy the result.",
        "allowed_tools": ["write_file", "shell_exec"],
    }
    result = validate_team_plan(raw, mode="adaptive")

    errors = " ".join(result["errors"])
    assert result["valid"] is False
    assert "unknown fields" in errors
    assert "executor" in errors
    assert result["team_required"] is False


def test_team_plan_rejects_model_outside_allowed_catalog():
    raw = _team_plan()
    raw["members"][0]["model"] = "invented-model"
    raw["members"][1]["model"] = "model-a"
    result = validate_team_plan(
        raw,
        mode="adaptive",
        available_models=[{"name": "model-a", "provider": "test"}],
    )

    assert result["valid"] is False
    assert "not in the allowed model catalog" in " ".join(result["errors"])


def test_team_role_ceilings_are_read_only_and_enforced_by_worker_runtime():
    from remy.core.autonomy import AGENT_ROLES

    forbidden = {
        "delegate_task",
        "browser_act",
        "write_file",
        "store",
        "store_knowledge",
        "update_record",
        "execute_trade",
        "shell_exec",
        "code_execution",
        "sandbox_create_tool",
    }
    for role_name, ceiling in TEAM_ROLE_TOOL_CEILINGS.items():
        assert not (set(ceiling) & forbidden)
        actual = {
            tool.name
            for tool in get_worker_tools(AGENT_ROLES[role_name], ceiling)
        }
        assert actual <= set(ceiling)
        assert not (actual & forbidden)


@pytest.mark.asyncio
async def test_team_execution_compiles_ceilings_and_fans_results_into_context(
    monkeypatch,
    tmp_path,
):
    original = settings.DATA_DIR
    settings.DATA_DIR = tmp_path
    captured = {}
    events = []

    async def fake_runner(tasks, session_id, channel, **kwargs):
        captured["tasks"] = tasks
        captured["session_id"] = session_id
        captured["channel"] = channel
        captured.update(kwargs)
        return [
            WorkerResult("researcher", "success", "Evidence A", tool_calls=2),
            WorkerResult("analyst", "success", "Analysis B", tool_calls=1),
        ]

    monkeypatch.setattr(
        "remy.core.team_planner._record_team_event",
        lambda **kwargs: events.append(kwargs),
    )
    try:
        receipt = await run_agent_team(
            "Compare two independent approaches",
            mode="adaptive",
            project_id="project-team",
            brain_id="brain-team",
            session_id="conversation-team",
            planner=lambda *args: _team_plan(),
            worker_runner=fake_runner,
        )
    finally:
        settings.DATA_DIR = original

    assert receipt["status"] == "completed"
    assert receipt["team_required"] is True
    assert "Evidence A" in receipt["context"]
    assert "Analysis B" in receipt["context"]
    assert receipt["usage"]["tool_calls"] == 3
    assert captured["step_budget"] == TeamLimits().step_budget_per_member
    assert captured["tasks"][0].delegation_depth == 0
    assert captured["tasks"][0].allowed_tools == TEAM_ROLE_TOOL_CEILINGS["researcher"]
    assert [item["event_type"] for item in events] == [
        "team_plan",
        "team_gate",
        "team_result",
    ]


@pytest.mark.asyncio
async def test_adaptive_single_agent_path_does_not_call_worker_runner():
    called = []
    receipt = await run_agent_team(
        "Give a short answer",
        mode="adaptive",
        project_id="project-team",
        brain_id="brain-team",
        session_id="conversation-team",
        planner=lambda *args: {
            "team_required": False,
            "reason": "Simple response",
            "members": [],
        },
        worker_runner=lambda *args, **kwargs: called.append((args, kwargs)),
    )

    assert receipt["status"] == "single_agent"
    assert receipt["context"] == ""
    assert called == []


@pytest.mark.asyncio
async def test_runtime_profile_can_only_narrow_worker_tools(monkeypatch, tmp_path):
    original = settings.DATA_DIR
    settings.DATA_DIR = tmp_path
    captured = []

    async def fake_runner(tasks, *args, **kwargs):
        captured.extend(tasks)
        return [
            WorkerResult(task.role, "success", f"local result {index}")
            for index, task in enumerate(tasks)
        ]

    monkeypatch.setattr("remy.core.team_planner._record_team_event", lambda **kwargs: None)
    try:
        receipt = await run_agent_team(
            "Analyze two local-only aspects",
            mode="force",
            project_id="project-lab",
            brain_id="brain-lab",
            session_id="lab-run",
            planner=lambda *args: _team_plan(),
            worker_runner=fake_runner,
            role_tool_ceilings={role: ("recall",) for role in TEAM_ROLE_TOOL_CEILINGS},
            capability_profile="agent_lab_local_read_only",
        )
    finally:
        settings.DATA_DIR = original

    assert all(task.allowed_tools == ("recall",) for task in captured)
    assert all(member["allowed_tools"] == ["recall"] for member in receipt["plan"]["members"])
    assert receipt["plan"]["capability_profile"] == "agent_lab_local_read_only"
    assert receipt["plan"]["runtime_plan_hash"]


@pytest.mark.asyncio
async def test_runtime_profile_cannot_reintroduce_forbidden_tools(monkeypatch, tmp_path):
    original = settings.DATA_DIR
    settings.DATA_DIR = tmp_path
    captured = []

    async def fake_runner(tasks, *args, **kwargs):
        captured.extend(tasks)
        return [WorkerResult(task.role, "success", "no external access") for task in tasks]

    monkeypatch.setattr("remy.core.team_planner._record_team_event", lambda **kwargs: None)
    try:
        receipt = await run_agent_team(
            "Analyze two bounded aspects",
            mode="force",
            project_id="project-lab",
            brain_id="brain-lab",
            session_id="lab-run-no-tools",
            planner=lambda *args: _team_plan(),
            worker_runner=fake_runner,
            role_tool_ceilings={
                role: ("shell_exec", "write_file") for role in TEAM_ROLE_TOOL_CEILINGS
            },
            capability_profile="agent_lab_local_read_only",
        )
    finally:
        settings.DATA_DIR = original

    assert all(task.allowed_tools == ("__no_tools__",) for task in captured)
    assert all(member["allowed_tools"] == [] for member in receipt["plan"]["members"])


@pytest.mark.asyncio
async def test_team_runtime_propagates_exact_worker_model_assignments(monkeypatch, tmp_path):
    original = settings.DATA_DIR
    settings.DATA_DIR = tmp_path
    captured = []
    raw = _team_plan()
    raw["members"][0]["model"] = "model-research"
    raw["members"][1]["model"] = "model-analysis"

    async def fake_runner(tasks, *args, **kwargs):
        captured.extend(tasks)
        return [
            WorkerResult(
                task.role,
                "success",
                "assigned result",
                assigned_model=task.model,
                served_by=task.model,
            )
            for task in tasks
        ]

    monkeypatch.setattr("remy.core.team_planner._record_team_event", lambda **kwargs: None)
    try:
        receipt = await run_agent_team(
            "Use specialist models",
            mode="force",
            project_id="project-models",
            brain_id="brain-models",
            session_id="team-models",
            planner=lambda *args: raw,
            worker_runner=fake_runner,
            available_models=[
                {"name": "model-research", "provider": "test"},
                {"name": "model-analysis", "provider": "test"},
            ],
        )
    finally:
        settings.DATA_DIR = original

    assert [task.model for task in captured] == ["model-research", "model-analysis"]
    assert all(task.allow_model_fallback is False for task in captured)
    assert [item["served_by"] for item in receipt["results"]] == [
        "model-research", "model-analysis"
    ]


def test_team_events_are_directly_visible_in_trajectory(tmp_path):
    store = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    event_id = store.record_team_event(
        project_id="project-1",
        session_id="conversation-1",
        event_type="team_gate",
        status="completed",
        run_id="run-team-1",
        payload={
            "mode": "adaptive",
            "plan_hash": "hash-1",
            "read_only": True,
            "members": [
                {"id": "research", "role": "researcher"},
                {"id": "analysis", "role": "analyst"},
            ],
        },
    )
    event = next(
        item
        for item in store.list_events(
            project_id="project-1",
            session_id="conversation-1",
        )
        if item["event_id"] == event_id
    )

    assert event["kind"] == "TEAM_GATE"
    assert event["details"]["member_count"] == 2
    assert event["details"]["read_only"] is True
    assert event["parent_id"] == "run-team-1"


def test_chat_ui_sends_explicit_team_mode_and_force_is_one_shot():
    root = Path(__file__).resolve().parents[1]
    html = (root / "src/remy/web/static/index.html").read_text(encoding="utf-8")
    chat = (root / "src/remy/web/static/js/chat.js").read_text(encoding="utf-8")
    client = (root / "src/remy/web/static/js/api-client.js").read_text(encoding="utf-8")

    assert 'id="chat-team-mode"' in html
    assert 'value="off"' in html
    assert 'value="adaptive"' in html
    assert 'value="force"' in html
    assert "function consumeTeamMode()" in chat
    assert 'if (mode === "force"' in chat
    assert 'team_mode: options.teamMode || "off"' in client
