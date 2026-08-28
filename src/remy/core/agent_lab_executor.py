"""Bounded local code execution for Agent Lab workspaces.

This is a constrained Python runner, not a general shell. It combines static
source validation, a clean child environment, runtime filesystem/network
guards, process-tree teardown, output bounds, timeouts, and artifact receipts.
"""

from __future__ import annotations

import ast
import hashlib
import json
import mimetypes
import os
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from remy.core.agent_lab import AgentLabStore
from remy.core.agent_lab_backends import (
    AgentLabBackendHandle,
    AgentLabBackendRequest,
    AgentLabExecutionBackend,
)
from remy.core.agent_lab_backend_registry import (
    AUTOMATIC_BACKEND,
    get_agent_lab_backend_registry,
)
from remy.core.agent_lab_container import (
    BOUNDED_PROCESS,
    CONTAINER_REQUIRED,
    probe_agent_lab_container_runtime,
    remove_agent_lab_container,
)
from remy.core.file_utils import atomic_write


SAFE_IMPORT_ROOTS = {
    "__future__", "collections", "csv", "dataclasses", "datetime", "decimal", "enum",
    "fractions", "functools",
    "hashlib", "html", "itertools", "json", "math", "pathlib", "random",
    "re", "statistics", "string", "textwrap", "time", "typing", "uuid",
}
BLOCKED_CALLS = {
    "eval", "exec", "compile", "__import__", "breakpoint", "input",
    "getattr", "setattr", "delattr", "globals", "locals", "vars",
}
BLOCKED_ATTRIBUTES = {"os", "sys", "socket", "subprocess", "ctypes", "importlib", "__builtins__"}
ALLOWED_FILE_ROOTS = {"inputs", "src", "tests"}
_active: dict[str, subprocess.Popen] = {}
_active_backends: dict[str, tuple[AgentLabExecutionBackend, AgentLabBackendHandle]] = {}
_cancelled: set[str] = set()
_active_lock = threading.RLock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative_path(value: str, *, allowed_roots: set[str]) -> PurePosixPath:
    clean = str(value or "").strip().replace("\\", "/")
    path = PurePosixPath(clean)
    if not clean or path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Path must stay inside the Agent Lab workspace")
    if path.parts[0] not in allowed_roots:
        raise ValueError(f"Path must start with one of: {', '.join(sorted(allowed_roots))}")
    return path


def validate_python_source(
    source: str, *, allowed_local_imports: set[str] | None = None
) -> None:
    local_imports = set(allowed_local_imports or set())
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"Python syntax error: {exc}") from exc
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in SAFE_IMPORT_ROOTS and root not in local_imports:
                    violations.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = str(node.module or "").split(".", 1)[0]
            if (
                not root
                or (root not in SAFE_IMPORT_ROOTS and root not in local_imports)
                or node.level
            ):
                violations.append(f"from {node.module or '.'} import")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_CALLS:
                violations.append(f"call {node.func.id}()")
        elif isinstance(node, ast.Attribute) and (
            node.attr.startswith("__") or node.attr in BLOCKED_ATTRIBUTES
        ):
            violations.append(f"attribute {node.attr}")
    if violations:
        unique = list(dict.fromkeys(violations))
        raise ValueError("Source violates Agent Lab policy: " + "; ".join(unique[:8]))


def _runtime_wrapper(
    workspace: Path | PurePosixPath,
    entrypoint: Path | PurePosixPath,
    argv: list[str],
    output_limit: int,
    max_memory_mb: int,
    read_only: bool = False,
) -> str:
    # All values are encoded as JSON literals before interpolation.
    return f'''import builtins, io, json, os, pathlib, runpy, socket, sys
ROOT = os.path.realpath({json.dumps(str(workspace))})
ENTRYPOINT = os.path.realpath({json.dumps(str(entrypoint))})
OUTPUT_LIMIT = {int(output_limit)}
READ_ONLY = {bool(read_only)!r}

if os.name != "nt":
    import resource
    memory_bytes = {int(max_memory_mb)} * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

def inside(path):
    candidate = os.path.normcase(os.path.abspath(os.fspath(path)))
    boundary = os.path.normcase(ROOT)
    return candidate == boundary or candidate.startswith(boundary + os.sep)

original_open = builtins.open
def guarded_open(file, mode="r", *args, **kwargs):
    if not inside(file):
        raise PermissionError("Agent Lab filesystem boundary denied access")
    if READ_ONLY and any(flag in str(mode) for flag in ("w", "a", "x", "+")):
        raise PermissionError("Agent Lab verifier filesystem is read-only")
    return original_open(file, mode, *args, **kwargs)
builtins.open = guarded_open
io.open = guarded_open
sys.path[:0] = [os.path.dirname(ENTRYPOINT), os.path.join(ROOT, "src")]

original_os_open = os.open
def guarded_os_open(path, flags, *args, **kwargs):
    if not inside(path):
        raise PermissionError("Agent Lab filesystem boundary denied access")
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    if READ_ONLY and flags & write_flags:
        raise PermissionError("Agent Lab verifier filesystem is read-only")
    return original_os_open(path, flags, *args, **kwargs)
os.open = guarded_os_open

def guard_single_path(function):
    def guarded(path, *args, **kwargs):
        if not inside(path):
            raise PermissionError("Agent Lab filesystem boundary denied access")
        return function(path, *args, **kwargs)
    return guarded

def guard_two_paths(function):
    def guarded(source, target, *args, **kwargs):
        if not inside(source) or not inside(target):
            raise PermissionError("Agent Lab filesystem boundary denied access")
        return function(source, target, *args, **kwargs)
    return guarded

for name in (
    "listdir", "scandir", "stat", "lstat", "readlink", "remove", "unlink",
    "mkdir", "rmdir", "chmod", "chown", "lchown", "utime", "truncate",
    "mkfifo", "mknod",
):
    if hasattr(os, name):
        original = getattr(os, name)
        if READ_ONLY and name in (
            "remove", "unlink", "mkdir", "rmdir", "chmod", "chown", "lchown",
            "utime", "truncate", "mkfifo", "mknod",
        ):
            setattr(os, name, lambda *a, **k: (_ for _ in ()).throw(PermissionError("Agent Lab verifier filesystem is read-only")))
        else:
            setattr(os, name, guard_single_path(original))
for name in ("rename", "replace", "link", "symlink"):
    if hasattr(os, name):
        if READ_ONLY:
            setattr(os, name, lambda *a, **k: (_ for _ in ()).throw(PermissionError("Agent Lab verifier filesystem is read-only")))
        else:
            setattr(os, name, guard_two_paths(getattr(os, name)))

class BlockedSocket:
    def __init__(self, *args, **kwargs):
        raise PermissionError("Agent Lab network access is disabled")
socket.socket = BlockedSocket
socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(PermissionError("Agent Lab network access is disabled"))

class BoundedText(io.TextIOBase):
    def __init__(self, target): self.target, self.used, self.truncated = target, 0, False
    def write(self, value):
        text = str(value)
        remaining = max(0, OUTPUT_LIMIT - self.used)
        chunk = text[:remaining]
        if chunk: self.target.write(chunk); self.target.flush(); self.used += len(chunk)
        if len(text) > remaining and not self.truncated:
            self.target.write("\\n[Agent Lab output truncated]\\n"); self.target.flush(); self.truncated = True
        return len(text)
    def flush(self): self.target.flush()
sys.stdout = BoundedText(sys.__stdout__)
sys.stderr = BoundedText(sys.__stderr__)
sys.argv = [ENTRYPOINT] + {json.dumps(argv)}
os.chdir(ROOT)
runpy.run_path(ENTRYPOINT, run_name="__main__")
'''


def _clean_environment(workspace: Path) -> dict[str, str]:
    allowed = ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "LANG")
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "NO_PROXY": "*",
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
        "AGENT_LAB_WORKSPACE": str(workspace),
        "DOCKER_CONTEXT": "default",
    })
    return env


def _stop_process_tree(process: subprocess.Popen) -> None:
    try:
        import psutil

        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        parent.terminate()
        _, alive = psutil.wait_procs([*children, parent], timeout=1.5)
        for item in alive:
            item.kill()
    except Exception:
        try:
            process.terminate()
            process.wait(timeout=1)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


def _workspace_sizes(root: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root):
                sizes[path.relative_to(root).as_posix()] = path.stat().st_size
        except OSError:
            continue
    return sizes


def _restore_workspace_size_snapshot(root: Path, before: dict[str, int]) -> None:
    """Remove new overflow files and shrink grown files to their prior size."""
    for path in list(root.rglob("*")):
        try:
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
                continue
            relative = path.relative_to(root).as_posix()
            if relative not in before:
                path.unlink()
            elif path.stat().st_size > before[relative]:
                with path.open("r+b") as handle:
                    handle.truncate(before[relative])
        except OSError:
            continue


def _resource_watchdog(
    process: subprocess.Popen,
    done: threading.Event,
    workspace: Path,
    memory_limit_mb: int,
    workspace_limit_bytes: int,
    memory_exceeded: threading.Event,
    disk_exceeded: threading.Event,
) -> list[float]:
    peak = [0.0]
    try:
        import psutil

        root = psutil.Process(process.pid)
        while not done.is_set():
            try:
                rss = root.memory_info().rss + sum(child.memory_info().rss for child in root.children(recursive=True))
                peak[0] = max(peak[0], rss / (1024 * 1024))
                if peak[0] > memory_limit_mb:
                    memory_exceeded.set()
                    _stop_process_tree(process)
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            if sum(_workspace_sizes(workspace).values()) > workspace_limit_bytes:
                disk_exceeded.set()
                _stop_process_tree(process)
                break
            if done.wait(0.02):
                break
    except Exception:
        pass
    return peak


class AgentLabExecutor:
    def __init__(self, store: AgentLabStore):
        self.store = store

    def write_file(
        self,
        run_id: str,
        *,
        path: str,
        content: str,
        allow_running: bool = False,
    ) -> dict[str, Any]:
        record = self.store.require(run_id)
        allowed_states = {"prepared", "paused", "running"} if allow_running else {"prepared", "paused"}
        if record.get("status") not in allowed_states:
            raise ValueError("Files can be staged only for a prepared or paused Agent Lab run")
        relative = _relative_path(path, allowed_roots=ALLOWED_FILE_ROOTS)
        raw = str(content or "").encode("utf-8")
        if len(raw) > int(record["policy"]["max_source_bytes"]):
            raise ValueError("Agent Lab source file exceeds the configured size limit")
        if relative.suffix.lower() == ".py":
            workspace = self.store.workspace_path(run_id)
            local_imports = {
                path.relative_to(workspace / "src").parts[0].removesuffix(".py")
                for path in (workspace / "src").rglob("*.py")
                if path.is_file()
            } if (workspace / "src").exists() else set()
            if (workspace / "src").is_dir():
                local_imports.add("src")
            validate_python_source(content, allowed_local_imports=local_imports)
        with self.store.workspace_lock:
            target = (self.store.workspace_path(run_id) / Path(*relative.parts)).resolve()
            if not target.is_relative_to(self.store.workspace_path(run_id)):
                raise ValueError("Source path escapes the Agent Lab workspace")
            current_sizes = _workspace_sizes(self.store.workspace_path(run_id))
            prior_size = current_sizes.get(relative.as_posix(), 0)
            projected = sum(current_sizes.values()) - prior_size + len(raw)
            if projected > int(record["policy"]["max_workspace_bytes"]):
                raise ValueError("Agent Lab workspace exceeds the configured size limit")
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(target, content)
        return {
            "path": relative.as_posix(),
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "written_at": _now(),
        }

    def cancel(self, run_id: str) -> bool:
        with _active_lock:
            process = _active.get(run_id)
            backend_state = _active_backends.get(run_id)
        if not process:
            return False
        with _active_lock:
            _cancelled.add(run_id)
        _stop_process_tree(process)
        if backend_state:
            backend, handle = backend_state
            backend.teardown(handle)
        return True

    def execute(
        self,
        run_id: str,
        *,
        entrypoint: str,
        arguments: list[str] | None = None,
        timeout_seconds: int = 30,
        read_only: bool = False,
        isolation_mode: str = BOUNDED_PROCESS,
        execution_requirements: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = self.store.require(run_id)
        if record.get("status") not in {"prepared", "paused", "running"}:
            raise ValueError("Agent Lab execution requires a prepared or paused run")
        relative = _relative_path(entrypoint, allowed_roots={"src", "tests"})
        if relative.suffix.lower() != ".py":
            raise ValueError("Agent Lab currently executes Python entrypoints only")
        workspace = self.store.workspace_path(run_id)
        target = (workspace / Path(*relative.parts)).resolve()
        if not target.is_relative_to(workspace) or not target.is_file():
            raise FileNotFoundError(relative.as_posix())
        local_imports = {
            path.relative_to(workspace / "src").parts[0].removesuffix(".py")
            for path in (workspace / "src").rglob("*.py")
            if path.is_file() and not path.is_symlink()
        } if (workspace / "src").exists() else set()
        if (workspace / "src").is_dir():
            local_imports.add("src")
        for workspace_path in workspace.rglob("*"):
            if workspace_path.is_symlink():
                raise ValueError("Agent Lab workspace cannot contain symlinks")
        for source_path in sorted(workspace.rglob("*.py")):
            if source_path.is_symlink() or not source_path.resolve().is_relative_to(workspace):
                raise ValueError("Agent Lab source tree contains an unsafe path")
            validate_python_source(
                source_path.read_text(encoding="utf-8"),
                allowed_local_imports=local_imports,
            )
        budget = min(max(1, int(timeout_seconds)), int(record["policy"]["time_budget_seconds"]), 300)
        argv = [str(item)[:500] for item in (arguments or [])[:20]]
        registry = get_agent_lab_backend_registry()
        requested_isolation = str(isolation_mode or BOUNDED_PROCESS).strip().lower()
        backend_selection: dict[str, Any] | None = None
        if requested_isolation == AUTOMATIC_BACKEND:
            backend_selection = registry.select(
                requested_isolation,
                execution_requirements,
                probe_runtime=probe_agent_lab_container_runtime,
            )
            isolation = backend_selection["resolved_mode"]
        else:
            isolation = registry.descriptor(requested_isolation).mode
        execution_id = f"exec-{uuid.uuid4().hex[:12]}"
        local_wrapper = _runtime_wrapper(
            workspace,
            target,
            argv,
            int(record["policy"]["max_output_bytes"]),
            int(record["policy"]["max_memory_mb"]),
            read_only=read_only,
        )
        container_workspace = PurePosixPath("/workspace")
        container_wrapper = _runtime_wrapper(
            container_workspace,
            container_workspace / PurePosixPath(*relative.parts),
            argv,
            int(record["policy"]["max_output_bytes"]),
            int(record["policy"]["max_memory_mb"]),
            read_only=read_only,
        )
        backend = registry.create(
            isolation,
            popen=subprocess.Popen,
            probe_runtime=probe_agent_lab_container_runtime,
            remove_container=remove_agent_lab_container,
        )
        request = AgentLabBackendRequest(
            run_id=run_id,
            execution_id=execution_id,
            workspace=workspace,
            environment=_clean_environment(workspace),
            local_wrapper=local_wrapper,
            container_wrapper=container_wrapper,
            max_memory_mb=int(record["policy"]["max_memory_mb"]),
            read_only=read_only,
        )
        started_at = _now()
        started = time.monotonic()
        size_snapshot = _workspace_sizes(workspace)
        handle = backend.launch(request)
        if backend_selection is None:
            backend_selection = {
                "schema": "agent-lab-backend-selection/v1",
                "requested_mode": requested_isolation,
                "resolved_mode": isolation,
                "automatic": False,
                "selected_reason": "explicit_mode_satisfied",
            }
        process = handle.process
        with _active_lock:
            if run_id in _active:
                _stop_process_tree(process)
                backend.teardown(handle)
                raise ValueError("An Agent Lab process is already active for this run")
            _cancelled.discard(run_id)
            _active[run_id] = process
            _active_backends[run_id] = (backend, handle)
        done = threading.Event()
        exceeded = threading.Event()
        disk_exceeded = threading.Event()
        peak = [0.0]
        def monitor_target():
            measured = _resource_watchdog(
                process,
                done,
                workspace,
                int(record["policy"]["max_memory_mb"]),
                int(record["policy"]["max_workspace_bytes"]),
                exceeded,
                disk_exceeded,
            )
            peak[0] = measured[0]
        monitor = threading.Thread(target=monitor_target, daemon=True)
        monitor.start()
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=budget)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop_process_tree(process)
            stdout, stderr = process.communicate()
        finally:
            done.set()
            monitor.join(timeout=1)
            backend.teardown(handle)
            with _active_lock:
                _active.pop(run_id, None)
                _active_backends.pop(run_id, None)
        if sum(_workspace_sizes(workspace).values()) > int(record["policy"]["max_workspace_bytes"]):
            disk_exceeded.set()
        if disk_exceeded.is_set():
            _restore_workspace_size_snapshot(workspace, size_snapshot)
        output_limit = int(record["policy"]["max_output_bytes"])
        with _active_lock:
            was_cancelled = run_id in _cancelled
            _cancelled.discard(run_id)
        status = (
            "cancelled" if was_cancelled
            else "memory_limit" if exceeded.is_set()
            else "disk_limit" if disk_exceeded.is_set()
            else "timeout" if timed_out
            else "memory_limit" if handle.isolation_mode == CONTAINER_REQUIRED and process.returncode == 137
            else "passed" if process.returncode == 0
            else "failed"
        )
        truncation_marker = "[Agent Lab output truncated]"
        captured_limit = output_limit + len(truncation_marker) + 4
        receipt = {
            "execution_id": execution_id,
            "entrypoint": relative.as_posix(),
            "arguments": argv,
            "status": status,
            "exit_code": process.returncode,
            "stdout": (stdout or "")[:captured_limit],
            "stderr": (stderr or "")[:captured_limit],
            "stdout_truncated": truncation_marker in (stdout or "") or len(stdout or "") > captured_limit,
            "stderr_truncated": truncation_marker in (stderr or "") or len(stderr or "") > captured_limit,
            "duration_ms": max(0, int((time.monotonic() - started) * 1000)),
            "peak_memory_mb": round(peak[0], 2),
            "timeout_seconds": budget,
            "read_only": bool(read_only),
            "isolation_mode": handle.isolation_mode,
            "isolation_engine": handle.isolation_engine,
            "container_image": handle.container_image,
            "container_image_id": handle.container_image_id,
            "container_security": handle.security_contract,
            "backend_selection": backend_selection,
            "started_at": started_at,
            "completed_at": _now(),
        }
        return receipt

    def inventory_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        record = self.store.require(run_id)
        root = (self.store.workspace_path(run_id) / "artifacts").resolve()
        total = 0
        artifacts = []
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                continue
            size = path.stat().st_size
            total += size
            if total > int(record["policy"]["max_artifact_bytes"]):
                raise ValueError("Agent Lab artifacts exceed the configured total size limit")
            relative = path.relative_to(root).as_posix()
            artifacts.append({
                "artifact_id": "artifact-" + hashlib.sha256(relative.encode()).hexdigest()[:12],
                "name": path.name,
                "path": f"artifacts/{relative}",
                "size": size,
                "mime_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
            if len(artifacts) >= 200:
                break
        return artifacts


_executors: dict[int, AgentLabExecutor] = {}
_executors_lock = threading.RLock()


def get_agent_lab_executor(store: AgentLabStore) -> AgentLabExecutor:
    key = id(store)
    with _executors_lock:
        executor = _executors.get(key)
        if executor is None:
            executor = AgentLabExecutor(store)
            _executors[key] = executor
        return executor
