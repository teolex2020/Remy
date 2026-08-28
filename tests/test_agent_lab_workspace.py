import pytest

from remy.core.agent_lab import AgentLabStore
from remy.core.agent_lab_executor import AgentLabExecutor
from remy.core.agent_lab_workspace import (
    AgentLabWorkspaceManager,
    WorkspaceMergeConflict,
)


def prepared(tmp_path):
    store = AgentLabStore(tmp_path)
    run = store.create(goal="Build in an isolated private workspace")
    store.prepare(run["run_id"])
    return store, run["run_id"]


def test_private_builder_file_is_invisible_until_merge_gate_accepts_it(tmp_path):
    store, run_id = prepared(tmp_path)
    manager = AgentLabWorkspaceManager(store)
    snapshot = manager.create_snapshot(run_id, node_id="build")

    staged = manager.stage_file(
        run_id,
        snapshot["workspace_id"],
        path="src/main.py",
        content="print('private')\n",
    )

    canonical = store.workspace_path(run_id) / "src" / "main.py"
    assert staged["sha256"]
    assert not canonical.exists()

    receipt = manager.merge(run_id, snapshot["workspace_id"])

    assert receipt["status"] == "merged"
    assert receipt["applied_files"] == ["src/main.py"]
    assert receipt["baseline_root_hash"]
    assert receipt["canonical_before_hash"] != receipt["canonical_after_hash"]
    assert canonical.read_text(encoding="utf-8") == "print('private')\n"
    record = store.require(run_id)
    assert record["workspace_branches"][-1]["status"] == "merged"
    assert record["merge_receipts"][-1]["merge_id"] == receipt["merge_id"]


def test_merge_gate_rejects_stale_snapshot_and_preserves_canonical_file(tmp_path):
    store, run_id = prepared(tmp_path)
    executor = AgentLabExecutor(store)
    executor.write_file(run_id, path="src/main.py", content="print('baseline')\n")
    manager = AgentLabWorkspaceManager(store)
    snapshot = manager.create_snapshot(run_id, node_id="build")
    manager.stage_file(
        run_id,
        snapshot["workspace_id"],
        path="src/main.py",
        content="print('candidate')\n",
    )

    executor.write_file(run_id, path="src/main.py", content="print('concurrent')\n")

    with pytest.raises(WorkspaceMergeConflict) as exc:
        manager.merge(run_id, snapshot["workspace_id"])

    assert exc.value.receipt["status"] == "conflict"
    assert exc.value.receipt["conflicts"][0]["reason"] == "canonical_changed_since_snapshot"
    assert (store.workspace_path(run_id) / "src" / "main.py").read_text(
        encoding="utf-8"
    ) == "print('concurrent')\n"
    record = store.require(run_id)
    assert record["workspace_branches"][-1]["status"] == "conflict"
    assert record["task_ledger"]["blockers"][-1]["kind"] == "merge_conflict"


def test_private_workspace_cannot_expand_node_or_path_authority(tmp_path):
    store, run_id = prepared(tmp_path)
    manager = AgentLabWorkspaceManager(store)

    with pytest.raises(ValueError, match="not authorized"):
        manager.create_snapshot(run_id, node_id="evidence")

    snapshot = manager.create_snapshot(run_id, node_id="build")
    with pytest.raises(ValueError, match="src/ or tests"):
        manager.stage_file(
            run_id,
            snapshot["workspace_id"],
            path="artifacts/unreviewed.txt",
            content="not allowed",
        )
    with pytest.raises(ValueError, match="src/ or tests"):
        manager.stage_file(
            run_id,
            snapshot["workspace_id"],
            path="../host.py",
            content="print('escape')\n",
        )


def test_merge_revalidates_source_and_closes_snapshot_after_success(tmp_path):
    store, run_id = prepared(tmp_path)
    manager = AgentLabWorkspaceManager(store)
    snapshot = manager.create_snapshot(run_id, node_id="build")
    manager.stage_file(
        run_id,
        snapshot["workspace_id"],
        path="tests/verify.py",
        content="assert True\n",
    )

    manager.merge(run_id, snapshot["workspace_id"])

    with pytest.raises(ValueError, match="already closed"):
        manager.merge(run_id, snapshot["workspace_id"])


def test_private_workspace_allows_imports_from_its_snapshot_src_namespace(tmp_path):
    store, run_id = prepared(tmp_path)
    manager = AgentLabWorkspaceManager(store)
    build = manager.create_snapshot(run_id, node_id="build")
    manager.stage_file(
        run_id,
        build["workspace_id"],
        path="src/main.py",
        content="def answer():\n    return 42\n",
    )
    manager.merge(run_id, build["workspace_id"])

    verifier = manager.create_snapshot(run_id, node_id="verify")
    manager.stage_file(
        run_id,
        verifier["workspace_id"],
        path="tests/verify.py",
        content="import src.main\nassert src.main.answer() == 42\n",
    )
    receipt = manager.merge(run_id, verifier["workspace_id"])

    assert receipt["status"] == "merged"
    assert receipt["applied_files"] == ["tests/verify.py"]


def test_retention_cleanup_removes_merged_copy_but_preserves_canonical_and_receipts(tmp_path):
    store, run_id = prepared(tmp_path)
    manager = AgentLabWorkspaceManager(store)
    snapshot = manager.create_snapshot(run_id, node_id="build")
    manager.stage_file(
        run_id,
        snapshot["workspace_id"],
        path="src/main.py",
        content="print('retained canonical')\n",
    )
    merge = manager.merge(run_id, snapshot["workspace_id"])

    before = manager.retention_status(run_id)
    assert before["snapshot_count"] == 1
    assert before["cleanup_eligible_count"] == 1
    assert before["used_bytes"] > 0

    result = manager.cleanup_snapshots(run_id)

    assert result["receipt"]["status"] == "completed"
    assert result["receipt"]["removed_workspace_ids"] == [snapshot["workspace_id"]]
    assert result["receipt"]["recovered_bytes"] > 0
    assert result["retention"]["snapshot_count"] == 0
    assert (store.workspace_path(run_id) / "src" / "main.py").read_text(
        encoding="utf-8"
    ) == "print('retained canonical')\n"
    record = store.require(run_id)
    assert record["merge_receipts"][-1]["merge_id"] == merge["merge_id"]
    assert record["workspace_branches"][-1]["retained"] is False
    assert record["snapshot_cleanup_receipts"][-1]["cleanup_id"].startswith("cleanup-")


def test_retention_protects_open_and_conflict_snapshots_without_explicit_authority(tmp_path):
    store, run_id = prepared(tmp_path)
    executor = AgentLabExecutor(store)
    executor.write_file(run_id, path="src/main.py", content="print('baseline')\n")
    manager = AgentLabWorkspaceManager(store)
    open_snapshot = manager.create_snapshot(run_id, node_id="build")

    with pytest.raises(ValueError, match="Active private snapshot"):
        manager.cleanup_snapshots(
            run_id, workspace_ids=[open_snapshot["workspace_id"]]
        )

    manager.stage_file(
        run_id,
        open_snapshot["workspace_id"],
        path="src/main.py",
        content="print('candidate')\n",
    )
    executor.write_file(run_id, path="src/main.py", content="print('concurrent')\n")
    with pytest.raises(WorkspaceMergeConflict):
        manager.merge(run_id, open_snapshot["workspace_id"])

    with pytest.raises(ValueError, match="explicit confirmation"):
        manager.cleanup_snapshots(
            run_id, workspace_ids=[open_snapshot["workspace_id"]]
        )
    result = manager.cleanup_snapshots(
        run_id,
        workspace_ids=[open_snapshot["workspace_id"]],
        include_conflicts=True,
    )
    assert result["retention"]["snapshot_count"] == 0
    assert store.require(run_id)["merge_receipts"][-1]["status"] == "conflict"


def test_retention_cleanup_is_locked_while_run_is_active(tmp_path):
    store, run_id = prepared(tmp_path)
    manager = AgentLabWorkspaceManager(store)
    snapshot = manager.create_snapshot(run_id, node_id="build")
    manager.stage_file(
        run_id, snapshot["workspace_id"], path="src/main.py", content="print('x')\n"
    )
    manager.merge(run_id, snapshot["workspace_id"])
    store.transition(run_id, "running")

    with pytest.raises(ValueError, match="locked while Agent Lab is running"):
        manager.cleanup_snapshots(run_id)
    assert manager.retention_status(run_id)["snapshot_count"] == 1
