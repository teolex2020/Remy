import copy

import pytest

from remy.core.agent_lab import AgentLabStore
from remy.core.agent_lab_workflow import (
    build_default_lab_workflow_plan,
    validate_lab_workflow_plan,
)


def _team():
    return [
        {"agent_id": "agent-1", "role": "coordinator"},
        {"agent_id": "agent-2", "role": "researcher"},
        {"agent_id": "agent-3", "role": "builder"},
        {"agent_id": "agent-4", "role": "verifier"},
    ]


def _policy(max_agents=4):
    return {"max_agents": max_agents}


def test_default_lab_workflow_is_a_valid_policy_bounded_dag():
    plan = build_default_lab_workflow_plan("Build a verified artifact", _team(), _policy())

    assert plan["version"] == 2
    assert plan["max_parallel"] == 3
    assert plan["max_total_agents"] == 4
    assert [node["node_id"] for node in plan["nodes"]] == [
        "scope",
        "evidence",
        "build",
        "verify",
        "handoff",
    ]
    assert next(node for node in plan["nodes"] if node["node_id"] == "build")[
        "workspace_mode"
    ] == "private_snapshot"
    assert {tuple(edge.values()) for edge in plan["edges"]} == {
        ("scope", "evidence"),
        ("evidence", "build"),
        ("build", "verify"),
        ("verify", "handoff"),
    }


def test_lab_workflow_rejects_cycles_caps_models_and_tools():
    plan = build_default_lab_workflow_plan("Build", _team(), _policy())

    cyclic = copy.deepcopy(plan)
    cyclic["nodes"][0]["depends_on"] = ["handoff"]
    cyclic["edges"] = []
    with pytest.raises(ValueError, match="acyclic"):
        validate_lab_workflow_plan(cyclic, policy=_policy())

    excessive = copy.deepcopy(plan)
    excessive["max_parallel"] = 5
    with pytest.raises(ValueError, match="max_parallel"):
        validate_lab_workflow_plan(excessive, policy=_policy())

    unavailable = copy.deepcopy(plan)
    unavailable["nodes"][0]["model"] = "unconnected-model"
    with pytest.raises(ValueError, match="unavailable model"):
        validate_lab_workflow_plan(
            unavailable,
            policy=_policy(),
            available_models={"connected-model"},
        )

    elevated = copy.deepcopy(plan)
    elevated["nodes"][0]["allowed_tools"] = ["shell"]
    with pytest.raises(ValueError, match="disallowed tool"):
        validate_lab_workflow_plan(
            elevated,
            policy=_policy(),
            allowed_tools={"recall"},
        )


def test_task_and_progress_ledgers_follow_node_state_and_artifact_hashes(tmp_path):
    store = AgentLabStore(tmp_path)
    run = store.create(goal="Build a verified local result")
    prepared = store.prepare(run["run_id"])

    assert prepared["version"] == 2
    assert prepared["task_ledger"]["plan_id"] == "initial-plan"
    assert prepared["progress_ledger"][0]["progress_made"] is True
    assert prepared["task_ledger"]["node_states"]["build"]["status"] == "pending"

    updated = store.update_node_statuses(
        run["run_id"],
        {"scope": "completed", "evidence": "in_progress"},
        reason="Scope accepted",
    )
    assert updated["workflow_plan"]["nodes"][0]["status"] == "completed"
    assert updated["plan"][0]["status"] == "completed"
    assert updated["task_ledger"]["node_states"]["evidence"]["status"] == "in_progress"
    assert updated["progress_ledger"][-1]["active_nodes"] == ["evidence"]
    assert updated["progress_ledger"][-1]["state_fingerprint"]

    with_artifacts = store.set_artifacts(
        run["run_id"],
        [{"artifact_id": "artifact-1", "path": "artifacts/result.txt", "sha256": "abc123"}],
    )
    assert with_artifacts["task_ledger"]["artifact_hashes"] == {
        "artifacts/result.txt": "abc123"
    }

    blocked = store.record_blocker(
        run["run_id"], node_id="build", kind="verification", message="Test failed"
    )
    assert blocked["task_ledger"]["blockers"][-1]["message"] == "Test failed"
    assert blocked["progress_ledger"][-1]["progress_made"] is False


def test_existing_prepared_run_is_upgraded_without_losing_checkpoint(tmp_path):
    store = AgentLabStore(tmp_path)
    run = store.create(goal="Preserve an old checkpoint")
    prepared = store.prepare(run["run_id"])

    def downgrade(record):
        record["version"] = 1
        record.pop("workflow_plan", None)
        record.pop("task_ledger", None)
        record.pop("progress_ledger", None)
        record["plan"][2]["status"] = "needs_repair"

    store.mutate(prepared["run_id"], downgrade)
    upgraded = store.ensure_workflow_state(prepared["run_id"])

    assert upgraded["version"] == 2
    assert upgraded["task_ledger"]["node_states"]["build"]["status"] == "needs_repair"
    assert upgraded["progress_ledger"][0]["reason"].startswith("Existing Agent Lab")
    assert any(event["type"] == "workflow_migrated" for event in upgraded["events"])
