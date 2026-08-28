import os
import subprocess

import pytest

from remy.core.agent_lab_container import (
    BOUNDED_PROCESS,
    CONTAINER_REQUIRED,
    agent_lab_runtime_build_context,
    build_agent_lab_container_command,
    normalize_isolation_mode,
    prepare_agent_lab_container_runtime,
    probe_agent_lab_container_runtime,
)
from remy.core.agent_lab import AgentLabStore
from remy.core.agent_lab_executor import AgentLabExecutor


def completed(command, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def test_isolation_mode_contract_is_strict():
    assert normalize_isolation_mode("") == BOUNDED_PROCESS
    assert normalize_isolation_mode(CONTAINER_REQUIRED) == CONTAINER_REQUIRED
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_isolation_mode("best_effort_container")


def test_container_probe_is_local_only_and_never_pulls_images(monkeypatch):
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        if command[1:3] == ["context", "inspect"]:
            return completed(command, stdout="npipe:////./pipe/docker_engine\n")
        if command[1] == "version":
            return completed(command, stdout="27.1.0\n")
        if command[1:3] == ["image", "inspect"]:
            return completed(command, stdout="sha256:" + "a" * 64)
        raise AssertionError(command)

    monkeypatch.delenv("DOCKER_HOST", raising=False)
    receipt = probe_agent_lab_container_runtime(
        which=lambda name: "C:/Docker/docker.exe" if name == "docker" else None,
        runner=runner,
    )

    assert receipt["available"] is True
    assert receipt["reason_code"] == "ready"
    assert receipt["engine"] == "docker"
    assert receipt["image_present"] is True
    assert all("pull" not in command for command, _ in calls)
    assert all(kwargs["env"].get("DOCKER_CONTEXT") == "default" for _, kwargs in calls)
    assert all("DOCKER_HOST" not in kwargs["env"] for _, kwargs in calls)


def test_container_probe_rejects_remote_docker_context_without_runtime_call():
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        return completed(command, stdout="tcp://remote.example:2376\n")

    receipt = probe_agent_lab_container_runtime(
        which=lambda name: "docker" if name == "docker" else None,
        runner=runner,
    )
    assert receipt["available"] is False
    assert receipt["reason_code"] == "remote_context"
    assert "non-local Docker" in receipt["reason"]
    assert len(calls) == 1


def test_container_probe_explains_when_cli_is_missing():
    receipt = probe_agent_lab_container_runtime(which=lambda _name: None)

    assert receipt["available"] is False
    assert receipt["reason_code"] == "cli_missing"
    assert receipt["image_present"] is False


def test_bundled_runtime_context_pins_the_base_image():
    dockerfile = agent_lab_runtime_build_context() / "Dockerfile"
    first_line = dockerfile.read_text(encoding="utf-8").splitlines()[0]

    assert "python:3.12-alpine@sha256:" in first_line


def test_explicit_runtime_preparation_builds_then_revalidates(tmp_path):
    calls = []
    image_inspections = 0
    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.12-alpine@sha256:" + "d" * 64 + "\n",
        encoding="utf-8",
    )

    def runner(command, **kwargs):
        nonlocal image_inspections
        calls.append((command, kwargs))
        if command[1:3] == ["context", "inspect"]:
            return completed(command, stdout="npipe:////./pipe/docker_engine\n")
        if command[1] == "version":
            return completed(command, stdout="29.0.0\n")
        if command[1:3] == ["image", "inspect"]:
            image_inspections += 1
            if image_inspections == 1:
                return completed(command, returncode=1, stderr="No such image")
            return completed(command, stdout="sha256:" + "e" * 64)
        if command[1] == "build":
            return completed(command, stdout="built")
        raise AssertionError(command)

    receipt = prepare_agent_lab_container_runtime(
        which=lambda name: "C:/Docker/docker.exe" if name == "docker" else None,
        runner=runner,
        context=tmp_path,
    )

    build_calls = [(command, kwargs) for command, kwargs in calls if command[1] == "build"]
    assert receipt["prepared"] is True
    assert receipt["reason_code"] == "ready"
    assert receipt["image_id"] == "sha256:" + "e" * 64
    assert len(build_calls) == 1
    command, kwargs = build_calls[0]
    assert command[1:5] == ["build", "--pull=false", "--tag", "remy-agent-lab-runtime:py312"]
    assert kwargs["timeout"] == 600
    assert kwargs["env"]["DOCKER_CONTEXT"] == "default"
    assert "shell" not in kwargs


def test_container_command_has_hard_boundary_and_read_only_verifier_mount(tmp_path):
    runtime = {
        "available": True,
        "engine": "docker",
        "executable": "docker",
        "image": "remy-agent-lab-runtime:py312",
        "image_id": "sha256:" + "b" * 64,
    }
    command = build_agent_lab_container_command(
        runtime,
        workspace=tmp_path,
        container_name="remy-lab-test",
        wrapper="print('ok')",
        max_memory_mb=256,
        read_only=True,
    )
    joined = " ".join(command)
    assert "--pull=never" in command
    assert "--network none" in joined
    assert "--read-only" in command
    assert "--cap-drop ALL" in joined
    assert "no-new-privileges=true" in command
    assert "--pids-limit 64" in joined
    assert "--memory 256m" in joined
    assert "target=/workspace,readonly" in joined
    assert "sha256:" + "b" * 64 in command
    assert all("secret" not in value.lower() for value in command)


@pytest.mark.skipif(
    os.environ.get("REMY_AGENT_LAB_REAL_CONTAINER_TEST") != "1",
    reason="requires an explicitly prepared local Docker/Podman runtime image",
)
def test_real_container_build_and_read_only_verifier(tmp_path):
    store = AgentLabStore(tmp_path, owner_project_id="container-smoke")
    run = store.create(goal="Real container boundary smoke test")
    run_id = run["run_id"]
    store.prepare(run_id)
    executor = AgentLabExecutor(store)
    executor.write_file(
        run_id,
        path="src/main.py",
        content=(
            "from pathlib import Path\n"
            'Path("artifacts/container.txt").write_text("isolated", encoding="utf-8")\n'
            'print("container build passed")\n'
        ),
    )
    executor.write_file(
        run_id,
        path="tests/verify.py",
        content=(
            "from pathlib import Path\n"
            'assert Path("artifacts/container.txt").read_text(encoding="utf-8") == "isolated"\n'
            'print("read-only verification passed")\n'
        ),
    )

    build = executor.execute(
        run_id,
        entrypoint="src/main.py",
        isolation_mode=CONTAINER_REQUIRED,
        timeout_seconds=20,
    )
    verification = executor.execute(
        run_id,
        entrypoint="tests/verify.py",
        isolation_mode=CONTAINER_REQUIRED,
        read_only=True,
        timeout_seconds=20,
    )

    assert build["status"] == "passed", build["stderr"]
    assert verification["status"] == "passed", verification["stderr"]
    assert build["container_image_id"].startswith("sha256:")
    assert verification["container_security"][-3:] == [
        "workspace_read_only", "no_image_pull", "clean_environment"
    ]
    assert (store.workspace_path(run_id) / "artifacts" / "container.txt").read_text(
        encoding="utf-8"
    ) == "isolated"
