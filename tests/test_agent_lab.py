import json

import pytest

from remy.core.agent_lab import AgentLabStore, _recover_agent_lab_store


def test_agent_lab_prepares_agent_owned_team_plan_and_workspace(tmp_path):
    store = AgentLabStore(tmp_path, owner_project_id="project-a", brain_id="brain-a")
    record = store.create(
        title="Build a report",
        goal="Research the evidence and create a verified presentation",
        policy={"max_agents": 5, "network_access": True, "hardware_access": True},
    )

    prepared = store.prepare(record["run_id"])

    assert prepared["status"] == "prepared"
    assert prepared["phase"] == "ready_for_execution"
    assert prepared["workspace"]["ready"] is True
    assert {agent["role"] for agent in prepared["team"]} >= {
        "coordinator", "researcher", "builder", "verifier"
    }
    assert [step["step_id"] for step in prepared["plan"]] == [
        "scope", "evidence", "build", "verify", "handoff"
    ]
    assert prepared["workflow_plan"]["version"] == 2
    assert prepared["workflow_plan"]["nodes"][2]["workspace_mode"] == "private_snapshot"
    assert prepared["task_ledger"]["node_states"]["scope"]["status"] == "pending"
    assert len(prepared["progress_ledger"]) == 1
    # API callers cannot enable dangerous capabilities in the foundation stage.
    assert prepared["policy"]["network_access"] is False
    assert prepared["policy"]["hardware_access"] is False
    manifest = tmp_path / "agent-lab" / record["run_id"] / "workspace" / "lab-manifest.json"
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["owner_project_id"] == "project-a"
    assert manifest_data["workflow_plan"]["version"] == 2
    assert (manifest.parent / "artifacts").is_dir()
    assert (manifest.parent / "tests").is_dir()


def test_agent_lab_state_machine_rejects_unsafe_skips(tmp_path):
    store = AgentLabStore(tmp_path)
    record = store.create(goal="Build and test a local tool")

    with pytest.raises(ValueError, match="draft to running"):
        store.transition(record["run_id"], "running")

    store.prepare(record["run_id"])
    running = store.transition(record["run_id"], "running")
    paused = store.transition(record["run_id"], "paused")
    cancelled = store.transition(record["run_id"], "cancelled")

    assert running["started_at"]
    assert paused["status"] == "paused"
    assert cancelled["status"] == "cancelled"
    assert cancelled["completed_at"]
    with pytest.raises(ValueError, match="cancelled to running"):
        store.transition(record["run_id"], "running")


def test_agent_lab_runs_are_project_directory_isolated(tmp_path):
    first = AgentLabStore(tmp_path / "one", owner_project_id="one")
    second = AgentLabStore(tmp_path / "two", owner_project_id="two")
    first_run = first.create(goal="First project")
    second_run = second.create(goal="Second project")

    assert first.require(first_run["run_id"])["owner_project_id"] == "one"
    assert second.require(second_run["run_id"])["owner_project_id"] == "two"
    assert first.get(second_run["run_id"]) is None
    assert second.get(first_run["run_id"]) is None


def test_agent_lab_rejects_path_like_run_ids(tmp_path):
    store = AgentLabStore(tmp_path)
    with pytest.raises(ValueError, match="Invalid Agent Lab run id"):
        store.get("../outside")


def test_agent_lab_recovers_orphaned_running_coordinator(tmp_path):
    store = AgentLabStore(tmp_path)
    record = store.create(goal="Recover this autonomous workspace")
    store.prepare(record["run_id"])
    store.transition(record["run_id"], "running")

    assert _recover_agent_lab_store(store) == 1
    recovered = store.require(record["run_id"])
    assert recovered["status"] == "paused"
    assert recovered["phase"] == "interrupted"
    assert "preserved" in recovered["error"]


def test_agent_lab_archive_hides_history_without_destroying_evidence(tmp_path):
    store = AgentLabStore(tmp_path)
    record = store.create(goal="Start over without losing the previous evidence")

    archived = store.archive(record["run_id"])

    assert archived["archived_at"]
    assert store.list() == []
    preserved = store.require(record["run_id"])
    assert preserved["events"][-1]["type"] == "archived"
    assert preserved["goal"] == record["goal"]


def test_agent_lab_archive_requires_running_task_to_stop_first(tmp_path):
    store = AgentLabStore(tmp_path)
    record = store.create(goal="Keep active work protected")
    store.prepare(record["run_id"])
    store.transition(record["run_id"], "running")

    with pytest.raises(ValueError, match="stopped before removal"):
        store.archive(record["run_id"])
