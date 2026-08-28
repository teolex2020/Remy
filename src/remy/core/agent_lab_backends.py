"""Interchangeable execution backends for Agent Lab.

Backends own only the isolation-specific launch and teardown boundary. The
executor remains responsible for source validation, budgets, monitoring,
receipts, and artifact accounting so every backend follows the same lifecycle.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from remy.core.agent_lab_container import (
    BOUNDED_PROCESS,
    CONTAINER_REQUIRED,
    build_agent_lab_container_command,
    container_security_contract,
    normalize_isolation_mode,
    probe_agent_lab_container_runtime,
    remove_agent_lab_container,
)


@dataclass(frozen=True)
class AgentLabBackendRequest:
    """Validated inputs an isolation backend needs to start one execution."""

    run_id: str
    execution_id: str
    workspace: Path
    environment: dict[str, str]
    local_wrapper: str
    container_wrapper: str
    max_memory_mb: int
    read_only: bool = False


@dataclass
class AgentLabBackendHandle:
    """Opaque running handle plus privacy-safe isolation provenance."""

    process: subprocess.Popen
    isolation_mode: str
    isolation_engine: str
    security_contract: list[str]
    runtime: dict[str, Any] = field(default_factory=dict)
    container_name: str = ""
    container_image: str = ""
    container_image_id: str = ""


class AgentLabExecutionBackend(Protocol):
    """Stable lifecycle implemented by every Agent Lab isolation backend."""

    mode: str

    def launch(self, request: AgentLabBackendRequest) -> AgentLabBackendHandle:
        """Start a validated execution or fail before returning a handle."""

    def teardown(self, handle: AgentLabBackendHandle) -> None:
        """Idempotently release backend-specific resources."""


PopenFactory = Callable[..., subprocess.Popen]
ProbeRuntime = Callable[[], dict[str, Any]]
RemoveContainer = Callable[[dict[str, Any], str], None]


def _spawn(
    popen: PopenFactory,
    command: list[str],
    request: AgentLabBackendRequest,
) -> subprocess.Popen:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    return popen(
        command,
        cwd=str(request.workspace),
        env=request.environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=flags,
    )


class BoundedProcessBackend:
    mode = BOUNDED_PROCESS

    def __init__(
        self,
        *,
        popen: PopenFactory = subprocess.Popen,
        python_executable: str = sys.executable,
    ) -> None:
        self._popen = popen
        self._python_executable = python_executable

    def launch(self, request: AgentLabBackendRequest) -> AgentLabBackendHandle:
        process = _spawn(
            self._popen,
            [self._python_executable, "-I", "-S", "-c", request.local_wrapper],
            request,
        )
        return AgentLabBackendHandle(
            process=process,
            isolation_mode=self.mode,
            isolation_engine="python-process",
            security_contract=[
                "runtime_guards",
                "clean_environment",
                "process_tree_teardown",
            ],
        )

    def teardown(self, handle: AgentLabBackendHandle) -> None:
        return None


class ContainerRequiredBackend:
    mode = CONTAINER_REQUIRED

    def __init__(
        self,
        *,
        popen: PopenFactory = subprocess.Popen,
        probe_runtime: ProbeRuntime = probe_agent_lab_container_runtime,
        remove_container: RemoveContainer = remove_agent_lab_container,
    ) -> None:
        self._popen = popen
        self._probe_runtime = probe_runtime
        self._remove_container = remove_container

    def launch(self, request: AgentLabBackendRequest) -> AgentLabBackendHandle:
        runtime = self._probe_runtime()
        if not runtime.get("available"):
            raise ValueError(
                "Container-required Agent Lab execution cannot start: "
                + str(runtime.get("reason") or "local runtime unavailable")
            )
        container_name = (
            f"remy-{request.run_id[-16:]}-{request.execution_id[-12:]}"
            .lower()
            .replace("_", "-")
        )[:63]
        command = build_agent_lab_container_command(
            runtime,
            workspace=request.workspace,
            container_name=container_name,
            wrapper=request.container_wrapper,
            max_memory_mb=request.max_memory_mb,
            read_only=request.read_only,
        )
        process = _spawn(self._popen, command, request)
        return AgentLabBackendHandle(
            process=process,
            isolation_mode=self.mode,
            isolation_engine=str(runtime.get("engine") or "container"),
            security_contract=container_security_contract(read_only=request.read_only),
            runtime=runtime,
            container_name=container_name,
            container_image=str(runtime.get("image") or ""),
            container_image_id=str(runtime.get("image_id") or ""),
        )

    def teardown(self, handle: AgentLabBackendHandle) -> None:
        if handle.container_name:
            self._remove_container(handle.runtime, handle.container_name)


def create_agent_lab_execution_backend(
    mode: str,
    *,
    popen: PopenFactory = subprocess.Popen,
    probe_runtime: ProbeRuntime = probe_agent_lab_container_runtime,
    remove_container: RemoveContainer = remove_agent_lab_container,
    python_executable: str = sys.executable,
) -> AgentLabExecutionBackend:
    """Resolve a backend without weakening or silently changing isolation."""
    isolation = normalize_isolation_mode(mode)
    if isolation == CONTAINER_REQUIRED:
        return ContainerRequiredBackend(
            popen=popen,
            probe_runtime=probe_runtime,
            remove_container=remove_container,
        )
    return BoundedProcessBackend(
        popen=popen,
        python_executable=python_executable,
    )
