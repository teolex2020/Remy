from pathlib import Path

import pytest

from remy.core.agent_lab_backends import (
    AgentLabBackendRequest,
    BoundedProcessBackend,
    ContainerRequiredBackend,
    create_agent_lab_execution_backend,
)


class FakeProcess:
    pid = 123
    returncode = 0


def request(tmp_path: Path) -> AgentLabBackendRequest:
    return AgentLabBackendRequest(
        run_id="lab-123",
        execution_id="exec-456",
        workspace=tmp_path,
        environment={"SAFE": "1"},
        local_wrapper="print('local')",
        container_wrapper="print('container')",
        max_memory_mb=128,
    )


def test_backend_factory_resolves_explicit_modes_without_downgrade():
    assert isinstance(create_agent_lab_execution_backend("bounded_process"), BoundedProcessBackend)
    assert isinstance(create_agent_lab_execution_backend("container_required"), ContainerRequiredBackend)
    with pytest.raises(ValueError, match="Unsupported"):
        create_agent_lab_execution_backend("best_effort")


def test_bounded_backend_uses_common_handle_contract(tmp_path):
    calls = []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return FakeProcess()

    backend = BoundedProcessBackend(popen=popen, python_executable="python-safe")
    handle = backend.launch(request(tmp_path))

    assert handle.process.pid == 123
    assert handle.isolation_mode == "bounded_process"
    assert handle.isolation_engine == "python-process"
    assert calls[0][0][:4] == ["python-safe", "-I", "-S", "-c"]
    assert calls[0][1]["env"] == {"SAFE": "1"}
    assert backend.teardown(handle) is None


def test_container_backend_fails_closed_before_spawn(tmp_path):
    spawned = []
    backend = ContainerRequiredBackend(
        popen=lambda *args, **kwargs: spawned.append((args, kwargs)),
        probe_runtime=lambda: {"available": False, "reason": "image missing"},
    )

    with pytest.raises(ValueError, match="image missing"):
        backend.launch(request(tmp_path))
    assert spawned == []


def test_container_backend_launch_and_teardown_share_one_handle(tmp_path):
    commands, removed = [], []
    runtime = {
        "available": True,
        "engine": "docker",
        "executable": "docker",
        "image": "remy-agent-lab-runtime:py312",
        "image_id": "sha256:" + "a" * 64,
    }

    def popen(command, **kwargs):
        commands.append(command)
        return FakeProcess()

    backend = ContainerRequiredBackend(
        popen=popen,
        probe_runtime=lambda: runtime,
        remove_container=lambda receipt, name: removed.append((receipt, name)),
    )
    handle = backend.launch(request(tmp_path))
    backend.teardown(handle)

    assert handle.isolation_mode == "container_required"
    assert handle.isolation_engine == "docker"
    assert handle.container_image_id == runtime["image_id"]
    assert runtime["image_id"] in commands[0]
    assert removed == [(runtime, handle.container_name)]
