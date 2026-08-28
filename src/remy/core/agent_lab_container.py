"""Optional, fail-closed container boundary for Agent Lab execution.

Remy never pulls an image during execution or silently downgrades a
container-required run. A separate user-initiated preparation step may build
the bundled runtime from its pinned base image.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable


BOUNDED_PROCESS = "bounded_process"
CONTAINER_REQUIRED = "container_required"
ISOLATION_MODES = {BOUNDED_PROCESS, CONTAINER_REQUIRED}
DEFAULT_CONTAINER_IMAGE = "remy-agent-lab-runtime:py312"
_IMAGE_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._/:@-]{0,239}$")
_CONTAINER_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,62}$")
_PREPARATION_LOCK = threading.Lock()


def _local_cli_environment() -> dict[str, str]:
    allowed = ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "PATH")
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    env["DOCKER_CONTEXT"] = "default"
    return env


def normalize_isolation_mode(value: Any) -> str:
    mode = str(value or BOUNDED_PROCESS).strip().lower()
    if mode not in ISOLATION_MODES:
        raise ValueError("Unsupported Agent Lab isolation mode")
    return mode


def configured_container_image() -> str:
    image = str(os.environ.get("REMY_AGENT_LAB_CONTAINER_IMAGE") or DEFAULT_CONTAINER_IMAGE).strip()
    if not _IMAGE_RE.fullmatch(image) or image.startswith(("http://", "https://")):
        raise ValueError("Agent Lab container image reference is invalid")
    return image


def agent_lab_runtime_build_context() -> Path:
    """Resolve the read-only build context in source and frozen desktop builds."""
    if getattr(sys, "frozen", False):
        root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
        context = root / "agent-lab-runtime"
    else:
        context = Path(__file__).resolve().parents[3] / "packaging" / "agent-lab-runtime"
    dockerfile = context / "Dockerfile"
    if not context.is_dir() or context.is_symlink() or not dockerfile.is_file() or dockerfile.is_symlink():
        raise ValueError("The bundled Remy Agent Lab runtime is unavailable; update or reinstall Remy")
    first_line = dockerfile.read_text(encoding="utf-8").splitlines()[0].strip()
    if "python:3.12-alpine@sha256:" not in first_line:
        raise ValueError("The bundled Remy Agent Lab runtime does not pin its base image")
    return context


def probe_agent_lab_container_runtime(
    *,
    which: Callable[[str], str | None] = shutil.which,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict[str, Any]:
    """Inspect local state only. The probe never pulls images or contacts registries."""
    image = configured_container_image()
    executable = which("docker") or which("podman")
    if not executable:
        return {
            "available": False,
            "reason_code": "cli_missing",
            "engine": "",
            "image": image,
            "image_present": False,
            "reason": "Docker or Podman CLI is not installed",
            "security_contract": container_security_contract(read_only=False),
        }
    engine = Path(executable).stem.lower()
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    cli_env = _local_cli_environment()
    if engine == "docker":
        try:
            context = runner(
                [executable, "context", "inspect", "default", "--format", "{{(index .Endpoints \"docker\").Host}}"],
                capture_output=True,
                text=True,
                timeout=4,
                creationflags=flags,
                env=cli_env,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {
                "available": False,
                "reason_code": "context_unavailable",
                "engine": engine,
                "image": image,
                "image_present": False,
                "reason": f"Could not validate local Docker context: {str(exc)[:240]}",
                "security_contract": container_security_contract(read_only=False),
            }
        endpoint = (context.stdout or "").strip().lower()
        if context.returncode != 0 or not endpoint.startswith(("unix://", "npipe://")):
            return {
                "available": False,
                "reason_code": "remote_context",
                "engine": engine,
                "image": image,
                "image_present": False,
                "reason": "Agent Lab rejects non-local Docker contexts",
                "security_contract": container_security_contract(read_only=False),
            }
    try:
        version = runner(
            [executable, "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=4,
            creationflags=flags,
            env=cli_env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "reason_code": "runtime_unavailable",
            "engine": engine,
            "image": image,
            "image_present": False,
            "reason": f"{engine} runtime is unavailable: {str(exc)[:240]}",
            "security_contract": container_security_contract(read_only=False),
        }
    if version.returncode != 0:
        reason = (version.stderr or version.stdout or "runtime did not respond").strip()[:240]
        return {
            "available": False,
            "reason_code": "runtime_unavailable",
            "engine": engine,
            "image": image,
            "image_present": False,
            "reason": f"{engine} runtime is unavailable: {reason}",
            "security_contract": container_security_contract(read_only=False),
        }
    try:
        inspected = runner(
            [executable, "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=flags,
            env=cli_env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        inspected = None
        inspect_reason = str(exc)[:240]
    else:
        inspect_reason = (inspected.stderr or "").strip()[:240]
    image_present = bool(inspected and inspected.returncode == 0)
    image_id = (inspected.stdout or "").strip()[:160] if image_present else ""
    if image_present and not image_id.startswith("sha256:"):
        image_present = False
        inspect_reason = "runtime returned an invalid image identity"
    return {
        "available": image_present,
        "reason_code": "ready" if image_present else "image_missing",
        "engine": engine,
        "executable": executable,
        "version": (version.stdout or "").strip()[:80],
        "image": image,
        "image_id": image_id if image_present else "",
        "image_present": image_present,
        "reason": "" if image_present else (
            f"Local image {image} is missing. Use the explicit Prepare secure runtime action; Agent Lab execution will never pull it automatically"
            + (f": {inspect_reason}" if inspect_reason else "")
        ),
        "security_contract": container_security_contract(read_only=False),
    }


def prepare_agent_lab_container_runtime(
    *,
    which: Callable[[str], str | None] = shutil.which,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    context: Path | None = None,
) -> dict[str, Any]:
    """Build the pinned local runtime after an explicit user action.

    This operation may fetch the pinned base image when Docker does not already
    have it. It never uses a shell, credentials, a remote Docker context, or an
    unpinned execution image.
    """
    if not _PREPARATION_LOCK.acquire(blocking=False):
        raise ValueError("Agent Lab runtime preparation is already in progress")
    try:
        before = probe_agent_lab_container_runtime(which=which, runner=runner)
        if before.get("available"):
            return {**before, "prepared": False, "already_ready": True}
        if before.get("reason_code") != "image_missing":
            raise ValueError(str(before.get("reason") or "Local container runtime is unavailable"))
        build_context = Path(context) if context is not None else agent_lab_runtime_build_context()
        dockerfile = build_context / "Dockerfile"
        if not build_context.is_dir() or build_context.is_symlink() or not dockerfile.is_file() or dockerfile.is_symlink():
            raise ValueError("The bundled Remy Agent Lab runtime is unavailable; update or reinstall Remy")
        executable = str(before.get("executable") or "")
        image = configured_container_image()
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            build = runner(
                [executable, "build", "--pull=false", "--tag", image, str(build_context.resolve())],
                capture_output=True,
                text=True,
                timeout=600,
                creationflags=flags,
                env=_local_cli_environment(),
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("Agent Lab runtime preparation timed out after 10 minutes") from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError(f"Agent Lab runtime preparation could not start: {str(exc)[:240]}") from exc
        if build.returncode != 0:
            detail = (build.stderr or build.stdout or "Docker build failed").strip().splitlines()
            raise ValueError(f"Agent Lab runtime preparation failed: {(detail[-1] if detail else 'Docker build failed')[:300]}")
        after = probe_agent_lab_container_runtime(which=which, runner=runner)
        if not after.get("available"):
            raise ValueError(
                "Agent Lab runtime build completed but validation failed: "
                + str(after.get("reason") or "local image unavailable")
            )
        return {**after, "prepared": True, "already_ready": False}
    finally:
        _PREPARATION_LOCK.release()


def container_security_contract(*, read_only: bool) -> list[str]:
    return [
        "network_none",
        "capabilities_dropped",
        "no_new_privileges",
        "read_only_rootfs",
        "pids_limited",
        "memory_limited",
        "cpu_limited",
        "workspace_read_only" if read_only else "workspace_read_write",
        "no_image_pull",
        "clean_environment",
    ]


def build_agent_lab_container_command(
    runtime: dict[str, Any],
    *,
    workspace: Path,
    container_name: str,
    wrapper: str,
    max_memory_mb: int,
    read_only: bool,
) -> list[str]:
    if not runtime.get("available"):
        raise ValueError(str(runtime.get("reason") or "Agent Lab container runtime is unavailable"))
    executable = str(runtime.get("executable") or "")
    image = str(runtime.get("image") or "")
    image_id = str(runtime.get("image_id") or "")
    if (
        not executable
        or not _IMAGE_RE.fullmatch(image)
        or not re.fullmatch(r"sha256:[a-fA-F0-9]{64}", image_id)
    ):
        raise ValueError("Agent Lab container runtime receipt is invalid")
    if not _CONTAINER_NAME_RE.fullmatch(container_name):
        raise ValueError("Agent Lab container name is invalid")
    root = workspace.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Agent Lab container workspace is unavailable")
    mount = f"type=bind,source={root},target=/workspace"
    if read_only:
        mount += ",readonly"
    uid = str(os.getuid()) if hasattr(os, "getuid") else "65532"
    gid = str(os.getgid()) if hasattr(os, "getgid") else "65532"
    return [
        executable,
        "run",
        "--rm",
        "--pull=never",
        "--name", container_name,
        "--network", "none",
        "--read-only",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges=true",
        "--pids-limit", "64",
        "--memory", f"{max(32, int(max_memory_mb))}m",
        "--cpus", "1.0",
        "--user", f"{uid}:{gid}",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
        "--mount", mount,
        "--workdir", "/workspace",
        "--env", "PYTHONIOENCODING=utf-8",
        "--env", "PYTHONUTF8=1",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        image_id,
        "python",
        "-I",
        "-S",
        "-B",
        "-c",
        wrapper,
    ]


def remove_agent_lab_container(runtime: dict[str, Any], container_name: str) -> None:
    """Best-effort idempotent teardown for timeout/cancellation and normal exit."""
    executable = str(runtime.get("executable") or "")
    if not executable or not _CONTAINER_NAME_RE.fullmatch(container_name):
        return
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        subprocess.run(
            [executable, "rm", "-f", container_name],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=flags,
            env=_local_cli_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return
