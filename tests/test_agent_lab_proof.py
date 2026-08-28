import json

from remy.core.agent_lab import AgentLabStore
from remy.core.agent_lab_executor import AgentLabExecutor
from remy.core.agent_lab_proof import (
    build_agent_lab_proof_pack,
    write_agent_lab_proof_pack,
)


def prepared(tmp_path):
    store = AgentLabStore(tmp_path, owner_project_id="project-a")
    run = store.create(goal="Produce a verifiable local artifact")
    store.prepare(run["run_id"])
    return store, run["run_id"]


def test_proof_pack_contains_receipts_hashes_and_no_hidden_prompts(tmp_path):
    store, run_id = prepared(tmp_path)
    store.mutate(
        run_id,
        lambda item: item.update({
            "autonomous": {"model": "builder-model", "served_by": "builder-model"},
            "verifier": {
                "assigned_model": "verifier-model",
                "served_by": "verifier-model",
                "independence_level": "cross_model",
                "rationale": "hidden verifier rationale",
                "model_catalog": [{"name": "secret-catalog"}],
            },
            "verification": [{
                "execution_id": "verify-1",
                "entrypoint": "tests/verify.py",
                "status": "passed",
                "world_fact": "supports",
                "exit_code": 0,
                "read_only": True,
                "stdout": "verified",
                "stderr": "",
            }],
        }),
    )
    artifact = store.workspace_path(run_id) / "artifacts" / "result.txt"
    artifact.write_text("result", encoding="utf-8")
    executor = AgentLabExecutor(store)
    store.set_artifacts(run_id, executor.inventory_artifacts(run_id))

    pack = build_agent_lab_proof_pack(
        store.require(run_id), decision="accepted", reason="Verification passed"
    )

    assert pack["plan"]["sha256"]
    assert pack["artifact_evidence"]["tree_hash"]
    assert pack["final_decision"]["verifier_independence"] == "cross_model"
    assert pack["runtime_evidence"]["verification"][0]["read_only"] is True
    serialized = json.dumps(pack)
    assert "hidden verifier rationale" not in serialized
    assert "secret-catalog" not in serialized
    assert "prompt" not in pack
    assert len(pack["proof_pack_sha256"]) == 64


def test_written_proof_pack_is_exported_and_inventory_addressable(tmp_path):
    store, run_id = prepared(tmp_path)
    receipt = write_agent_lab_proof_pack(
        store,
        run_id,
        decision="inconclusive",
        reason="No independent verification receipt",
    )

    json_path = store.workspace_path(run_id) / receipt["json_path"]
    markdown_path = store.workspace_path(run_id) / receipt["markdown_path"]
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload["proof_pack_sha256"] == receipt["sha256"]
    assert markdown_path.read_text(encoding="utf-8").startswith("# Agent Lab Proof Pack")
    assert store.require(run_id)["proof_pack"]["decision"] == "inconclusive"
    names = {
        item["name"] for item in AgentLabExecutor(store).inventory_artifacts(run_id)
    }
    assert {"proof-pack.json", "proof-pack.md"} <= names
