import concurrent.futures
import time

import pytest

from remy.core.agent_lab import AgentLabStore
from remy.core.agent_lab_executor import AgentLabExecutor, validate_python_source


def prepared_store(tmp_path):
    store = AgentLabStore(tmp_path, owner_project_id="project-a")
    record = store.create(goal="Build and verify a local artifact")
    store.prepare(record["run_id"])
    return store, record["run_id"]


def test_executor_runs_python_in_workspace_and_inventories_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_LAB_TEST_SECRET", "must-not-leak")
    store, run_id = prepared_store(tmp_path)
    executor = AgentLabExecutor(store)
    executor.write_file(
        run_id,
        path="src/main.py",
        content=(
            'from pathlib import Path\n'
            'Path("artifacts/result.txt").write_text("bounded", encoding="utf-8")\n'
            'print("artifact ready")\n'
        ),
    )

    receipt = executor.execute(run_id, entrypoint="src/main.py", timeout_seconds=5)
    artifacts = executor.inventory_artifacts(run_id)

    assert receipt["status"] == "passed"
    assert receipt["exit_code"] == 0
    assert "artifact ready" in receipt["stdout"]
    assert "must-not-leak" not in str(receipt)
    assert artifacts[0]["path"] == "artifacts/result.txt"
    assert len(artifacts[0]["sha256"]) == 64


def test_container_required_executor_fails_closed_before_process_spawn(tmp_path, monkeypatch):
    store, run_id = prepared_store(tmp_path)
    executor = AgentLabExecutor(store)
    executor.write_file(run_id, path="src/main.py", content="print('never started')\n")
    monkeypatch.setattr(
        "remy.core.agent_lab_executor.probe_agent_lab_container_runtime",
        lambda: {
            "available": False,
            "reason": "local image missing",
            "engine": "docker",
            "image": "remy-agent-lab-runtime:py312",
        },
    )
    monkeypatch.setattr(
        "remy.core.agent_lab_executor.subprocess.Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not spawn")),
    )

    with pytest.raises(ValueError, match="Container-required.*local image missing"):
        executor.execute(
            run_id,
            entrypoint="src/main.py",
            isolation_mode="container_required",
        )
    assert not (store.workspace_path(run_id) / "artifacts" / "result.txt").exists()


def test_container_executor_uses_immutable_image_and_records_boundary(tmp_path, monkeypatch):
    store, run_id = prepared_store(tmp_path)
    executor = AgentLabExecutor(store)
    executor.write_file(run_id, path="src/main.py", content="print('container')\n")
    runtime = {
        "available": True,
        "engine": "docker",
        "executable": "docker",
        "image": "remy-agent-lab-runtime:py312",
        "image_id": "sha256:" + "c" * 64,
        "reason": "",
    }
    commands, teardowns = [], []

    class FakeProcess:
        pid = 12345
        returncode = 0

        def __init__(self, command, **kwargs):
            commands.append(command)

        def communicate(self, timeout=None):
            return "container\n", ""

    monkeypatch.setattr(
        "remy.core.agent_lab_executor.probe_agent_lab_container_runtime",
        lambda: runtime,
    )
    monkeypatch.setattr("remy.core.agent_lab_executor.subprocess.Popen", FakeProcess)
    monkeypatch.setattr(
        "remy.core.agent_lab_executor._resource_watchdog",
        lambda *args, **kwargs: [1.0],
    )
    monkeypatch.setattr(
        "remy.core.agent_lab_executor.remove_agent_lab_container",
        lambda runtime_receipt, name: teardowns.append((runtime_receipt, name)),
    )

    receipt = executor.execute(
        run_id,
        entrypoint="src/main.py",
        isolation_mode="container_required",
    )

    assert receipt["status"] == "passed"
    assert receipt["isolation_mode"] == "container_required"
    assert receipt["isolation_engine"] == "docker"
    assert receipt["container_image_id"] == "sha256:" + "c" * 64
    assert "sha256:" + "c" * 64 in commands[0]
    assert "--pull=never" in commands[0]
    assert teardowns and teardowns[0][1].startswith("remy-")


def test_read_only_verifier_cannot_modify_observed_artifacts(tmp_path):
    store, run_id = prepared_store(tmp_path)
    executor = AgentLabExecutor(store)
    executor.write_file(
        run_id,
        path="src/main.py",
        content='from pathlib import Path\nPath("artifacts/result.txt").write_text("original", encoding="utf-8")\n',
    )
    executor.write_file(
        run_id,
        path="tests/verify.py",
        content='from pathlib import Path\nPath("artifacts/result.txt").write_text("forged", encoding="utf-8")\n',
    )
    store.transition(run_id, "running")
    assert executor.execute(run_id, entrypoint="src/main.py")["status"] == "passed"

    receipt = executor.execute(
        run_id,
        entrypoint="tests/verify.py",
        read_only=True,
    )

    assert receipt["status"] == "failed"
    assert receipt["read_only"] is True
    assert "read-only" in receipt["stderr"]
    assert (store.workspace_path(run_id) / "artifacts" / "result.txt").read_text(
        encoding="utf-8"
    ) == "original"


@pytest.mark.parametrize(
    "mutation",
    [
        'Path("artifacts/result.txt").unlink()',
        'Path("artifacts/link.txt").symlink_to("result.txt")',
    ],
)
def test_read_only_verifier_blocks_path_mutation_methods(tmp_path, mutation):
    store, run_id = prepared_store(tmp_path)
    executor = AgentLabExecutor(store)
    artifact = store.workspace_path(run_id) / "artifacts" / "result.txt"
    artifact.write_text("original", encoding="utf-8")
    executor.write_file(
        run_id,
        path="tests/verify.py",
        content=f"from pathlib import Path\n{mutation}\n",
    )

    receipt = executor.execute(
        run_id,
        entrypoint="tests/verify.py",
        read_only=True,
    )

    assert receipt["status"] == "failed"
    assert "read-only" in receipt["stderr"]
    assert artifact.read_text(encoding="utf-8") == "original"
    assert not (artifact.parent / "link.txt").exists()


@pytest.mark.parametrize(
    "source",
    [
        "import os\nprint(os.getcwd())\n",
        "import subprocess\nsubprocess.run(['whoami'])\n",
        "from pathlib import Path\nprint(Path.__subclasses__())\n",
        "from pathlib import Path\nprint(Path.os)\n",
        "eval('1 + 1')\n",
        "from pathlib import Path\nprint(getattr(Path, 'os'))\n",
    ],
)
def test_source_policy_blocks_system_and_dynamic_code(source):
    with pytest.raises(ValueError, match="violates Agent Lab policy"):
        validate_python_source(source)


def test_source_policy_allows_only_explicitly_validated_local_import_roots():
    validate_python_source(
        "from __future__ import annotations\nfrom dataclasses import dataclass\nfrom enum import Enum\n@dataclass\nclass Result:\n    value: int | Enum\n",
    )
    validate_python_source(
        "from domain import result\nprint(result())\n",
        allowed_local_imports={"domain"},
    )
    with pytest.raises(ValueError, match="from requests import"):
        validate_python_source(
            "from requests import get\n",
            allowed_local_imports={"domain"},
        )


def test_executor_blocks_workspace_path_escape_at_stage_time(tmp_path):
    store, run_id = prepared_store(tmp_path)
    executor = AgentLabExecutor(store)
    with pytest.raises(ValueError, match="stay inside"):
        executor.write_file(run_id, path="../outside.py", content="print('no')")
    assert not (tmp_path / "outside.py").exists()


def test_executor_runtime_blocks_read_outside_workspace(tmp_path):
    store, run_id = prepared_store(tmp_path)
    executor = AgentLabExecutor(store)
    secret = tmp_path / "host-secret.txt"
    secret.write_text("hidden", encoding="utf-8")
    executor.write_file(
        run_id,
        path="src/main.py",
        content=f'print(open({str(secret)!r}, encoding="utf-8").read())\n',
    )

    receipt = executor.execute(run_id, entrypoint="src/main.py", timeout_seconds=5)

    assert receipt["status"] == "failed"
    assert "filesystem boundary denied" in receipt["stderr"]
    assert "hidden" not in receipt["stdout"]


def test_executor_enforces_timeout_and_stops_process(tmp_path):
    store, run_id = prepared_store(tmp_path)
    executor = AgentLabExecutor(store)
    executor.write_file(run_id, path="src/main.py", content="while True:\n    pass\n")

    receipt = executor.execute(run_id, entrypoint="src/main.py", timeout_seconds=1)

    assert receipt["status"] == "timeout"
    assert receipt["duration_ms"] < 5000


def test_executor_bounds_stdout(tmp_path):
    store, run_id = prepared_store(tmp_path)
    store.mutate(run_id, lambda item: item["policy"].update({"max_output_bytes": 200}))
    executor = AgentLabExecutor(store)
    executor.write_file(run_id, path="src/main.py", content='print("x" * 10000)\n')

    receipt = executor.execute(run_id, entrypoint="src/main.py", timeout_seconds=5)

    assert receipt["status"] == "passed"
    assert len(receipt["stdout"]) <= 240
    assert "output truncated" in receipt["stdout"]


def test_executor_cancel_stops_active_process(tmp_path):
    store, run_id = prepared_store(tmp_path)
    executor = AgentLabExecutor(store)
    executor.write_file(run_id, path="src/main.py", content="while True:\n    pass\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(executor.execute, run_id, entrypoint="src/main.py", timeout_seconds=20)
        deadline = time.monotonic() + 3
        cancelled = False
        while time.monotonic() < deadline and not cancelled:
            cancelled = executor.cancel(run_id)
            if not cancelled:
                time.sleep(0.02)
        receipt = future.result(timeout=5)

    assert cancelled is True
    assert receipt["status"] == "cancelled"
    assert receipt["duration_ms"] < 5000


def test_artifact_inventory_enforces_total_size_limit(tmp_path):
    store, run_id = prepared_store(tmp_path)
    store.mutate(run_id, lambda item: item["policy"].update({"max_artifact_bytes": 8}))
    executor = AgentLabExecutor(store)
    artifact = store.workspace_path(run_id) / "artifacts" / "too-large.txt"
    artifact.write_text("0123456789", encoding="utf-8")

    with pytest.raises(ValueError, match="exceed"):
        executor.inventory_artifacts(run_id)


def test_executor_enforces_workspace_disk_limit_and_removes_overflow(tmp_path):
    store, run_id = prepared_store(tmp_path)
    executor = AgentLabExecutor(store)
    executor.write_file(
        run_id,
        path="src/main.py",
        content='from pathlib import Path\nPath("artifacts/flood.txt").write_text("x" * 20000, encoding="utf-8")\n',
    )
    baseline = sum(path.stat().st_size for path in store.workspace_path(run_id).rglob("*") if path.is_file())
    store.mutate(
        run_id,
        lambda item: item["policy"].update({"max_workspace_bytes": baseline + 1024}),
    )

    receipt = executor.execute(run_id, entrypoint="src/main.py", timeout_seconds=5)

    assert receipt["status"] == "disk_limit"
    assert not (store.workspace_path(run_id) / "artifacts" / "flood.txt").exists()
