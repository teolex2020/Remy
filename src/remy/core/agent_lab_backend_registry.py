"""Registry and unified preflight surface for Agent Lab execution backends."""

from __future__ import annotations

import re
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from typing import Any, Callable

from remy.core.agent_lab_backends import (
    AgentLabExecutionBackend,
    BoundedProcessBackend,
    ContainerRequiredBackend,
)
from remy.core.agent_lab_container import (
    BOUNDED_PROCESS,
    CONTAINER_REQUIRED,
    prepare_agent_lab_container_runtime,
    probe_agent_lab_container_runtime,
    remove_agent_lab_container,
)


_MODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
AUTOMATIC_BACKEND = "automatic"
BackendFactory = Callable[..., AgentLabExecutionBackend]
BackendProbe = Callable[..., dict[str, Any]]
BackendPrepare = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class AgentLabBackendCapabilities:
    languages: tuple[str, ...] = ("python",)
    artifact_kinds: tuple[str, ...] = ("file", "text", "json", "csv", "html", "svg")
    network_access: bool = False
    gpu_access: bool = False
    read_only_verification: bool = True
    isolation_rank: int = 10


@dataclass(frozen=True)
class AgentLabBackendDescriptor:
    mode: str
    label: str
    description: str
    security_tier: str
    preparation_supported: bool = False
    capabilities: AgentLabBackendCapabilities = AgentLabBackendCapabilities()


@dataclass(frozen=True)
class AgentLabExecutionRequirements:
    language: str = "python"
    artifact_kinds: tuple[str, ...] = ("file",)
    network_required: bool = False
    gpu_required: bool = False
    read_only_verification_required: bool = True
    minimum_isolation_rank: int = 10


@dataclass(frozen=True)
class AgentLabBackendRegistration:
    descriptor: AgentLabBackendDescriptor
    factory: BackendFactory
    probe: BackendProbe
    prepare: BackendPrepare | None = None


class AgentLabBackendRegistry:
    """Thread-safe catalog used by execution, coordination, API, and UI."""

    def __init__(self) -> None:
        self._registrations: dict[str, AgentLabBackendRegistration] = {}
        self._lock = threading.RLock()

    def register(
        self,
        registration: AgentLabBackendRegistration,
        *,
        replace: bool = False,
    ) -> None:
        mode = registration.descriptor.mode
        if not _MODE_RE.fullmatch(mode):
            raise ValueError("Agent Lab backend mode is invalid")
        if registration.descriptor.preparation_supported != bool(registration.prepare):
            raise ValueError("Agent Lab backend preparation contract is inconsistent")
        with self._lock:
            if mode in self._registrations and not replace:
                raise ValueError(f"Agent Lab backend is already registered: {mode}")
            self._registrations[mode] = registration

    def modes(self) -> list[str]:
        with self._lock:
            return list(self._registrations)

    def descriptor(self, mode: str) -> AgentLabBackendDescriptor:
        with self._lock:
            registration = self._registrations.get(str(mode or "").strip().lower())
        if registration is None:
            raise ValueError("Unsupported Agent Lab isolation mode")
        return registration.descriptor

    def create(self, mode: str, **dependencies: Any) -> AgentLabExecutionBackend:
        normalized = self.descriptor(mode).mode
        with self._lock:
            registration = self._registrations[normalized]
        return registration.factory(**dependencies)

    def preflight(self, mode: str, **dependencies: Any) -> dict[str, Any]:
        descriptor = self.descriptor(mode)
        with self._lock:
            registration = self._registrations[descriptor.mode]
        try:
            receipt = dict(registration.probe(**dependencies) or {})
        except Exception as exc:
            receipt = {
                "available": False,
                "reason_code": "probe_failed",
                "reason": f"Backend preflight failed: {str(exc)[:240]}",
            }
        return {
            **receipt,
            "mode": descriptor.mode,
            "label": descriptor.label,
            "description": descriptor.description,
            "security_tier": descriptor.security_tier,
            "capabilities": asdict(descriptor.capabilities),
            "preparation_supported": descriptor.preparation_supported,
            "available": bool(receipt.get("available")),
        }

    def prepare(self, mode: str, **dependencies: Any) -> dict[str, Any]:
        descriptor = self.descriptor(mode)
        with self._lock:
            registration = self._registrations[descriptor.mode]
        if registration.prepare is None:
            raise ValueError(f"Agent Lab backend does not require preparation: {descriptor.label}")
        receipt = dict(registration.prepare(**dependencies) or {})
        return {
            **receipt,
            "mode": descriptor.mode,
            "label": descriptor.label,
            "description": descriptor.description,
            "security_tier": descriptor.security_tier,
            "capabilities": asdict(descriptor.capabilities),
            "preparation_supported": True,
            "available": bool(receipt.get("available")),
        }

    def preflight_all(self, **dependencies: Any) -> list[dict[str, Any]]:
        return [self.preflight(mode, **dependencies) for mode in self.modes()]

    def select(
        self,
        requested_mode: str,
        requirements: AgentLabExecutionRequirements | dict[str, Any] | None = None,
        **dependencies: Any,
    ) -> dict[str, Any]:
        """Select the least-privileged matching backend with an auditable receipt.

        Explicit modes evaluate exactly one backend and never fall back. Automatic
        mode may compare registered backends, but still fails closed when none meet
        both the declared capabilities and live preflight.
        """
        required = normalize_execution_requirements(requirements)
        requested = str(requested_mode or BOUNDED_PROCESS).strip().lower()
        modes = self.modes() if requested == AUTOMATIC_BACKEND else [self.descriptor(requested).mode]
        candidates: list[dict[str, Any]] = []
        eligible: list[tuple[int, int, dict[str, Any]]] = []
        for order, mode in enumerate(modes):
            descriptor = self.descriptor(mode)
            mismatch = _capability_mismatches(descriptor.capabilities, required)
            status = self.preflight(mode, **dependencies) if not mismatch else {
                "mode": mode,
                "available": False,
                "reason_code": "capability_mismatch",
                "reason": "; ".join(mismatch),
            }
            candidate = {
                "mode": mode,
                "label": descriptor.label,
                "capability_match": not mismatch,
                "available": bool(status.get("available")),
                "reason_code": str(status.get("reason_code") or ""),
                "reason": str(status.get("reason") or ""),
                "isolation_rank": descriptor.capabilities.isolation_rank,
            }
            candidates.append(candidate)
            if not mismatch and candidate["available"]:
                eligible.append((descriptor.capabilities.isolation_rank, order, candidate))
        if not eligible:
            detail = "; ".join(
                f"{item['label']}: {item['reason'] or item['reason_code'] or 'unavailable'}"
                for item in candidates
            )
            if requested == AUTOMATIC_BACKEND:
                raise ValueError("No registered Agent Lab backend satisfies the execution requirements: " + detail)
            raise ValueError(f"{candidates[0]['label']} Agent Lab run cannot start: {detail}")
        # The smallest sufficient isolation rank is least privilege. Registration
        # order is a deterministic tie-breaker, never a hidden fallback policy.
        selected = min(eligible, key=lambda item: (item[0], item[1]))[2]
        return {
            "schema": "agent-lab-backend-selection/v1",
            "requested_mode": requested,
            "resolved_mode": selected["mode"],
            "automatic": requested == AUTOMATIC_BACKEND,
            "requirements": asdict(required),
            "selected_reason": (
                "least_privileged_available_match"
                if requested == AUTOMATIC_BACKEND
                else "explicit_mode_satisfied"
            ),
            "candidates": candidates,
        }


def normalize_execution_requirements(
    value: AgentLabExecutionRequirements | dict[str, Any] | None,
) -> AgentLabExecutionRequirements:
    if isinstance(value, AgentLabExecutionRequirements):
        return value
    raw = dict(value or {})
    language = str(raw.get("language") or "python").strip().lower()[:32]
    kinds_raw = raw.get("artifact_kinds") or ("file",)
    if not isinstance(kinds_raw, (list, tuple)):
        raise ValueError("Agent Lab artifact requirements must be a list")
    kinds = tuple(dict.fromkeys(
        str(item).strip().lower()[:32]
        for item in kinds_raw[:8]
        if str(item).strip()
    )) or ("file",)
    minimum = raw.get("minimum_isolation_rank", 10)
    if isinstance(minimum, str):
        minimum = {"guarded_process": 10, "hardened_container": 20}.get(minimum.strip().lower(), -1)
    minimum = int(minimum)
    if not language or not _MODE_RE.fullmatch(language.replace("-", "_")):
        raise ValueError("Agent Lab execution language is invalid")
    if minimum < 0 or minimum > 100:
        raise ValueError("Agent Lab minimum isolation requirement is invalid")
    for flag in ("network_required", "gpu_required", "read_only_verification_required"):
        if flag in raw and not isinstance(raw[flag], bool):
            raise ValueError(f"Agent Lab {flag} requirement must be a boolean")
    return AgentLabExecutionRequirements(
        language=language,
        artifact_kinds=kinds,
        network_required=bool(raw.get("network_required", False)),
        gpu_required=bool(raw.get("gpu_required", False)),
        read_only_verification_required=bool(raw.get("read_only_verification_required", True)),
        minimum_isolation_rank=minimum,
    )


def _capability_mismatches(
    capabilities: AgentLabBackendCapabilities,
    requirements: AgentLabExecutionRequirements,
) -> list[str]:
    reasons: list[str] = []
    if requirements.language not in capabilities.languages:
        reasons.append(f"language {requirements.language} is unsupported")
    unsupported_artifacts = sorted(set(requirements.artifact_kinds) - set(capabilities.artifact_kinds))
    if unsupported_artifacts:
        reasons.append("unsupported artifact kinds: " + ", ".join(unsupported_artifacts))
    if requirements.network_required and not capabilities.network_access:
        reasons.append("network access is unavailable")
    if requirements.gpu_required and not capabilities.gpu_access:
        reasons.append("GPU access is unavailable")
    if requirements.read_only_verification_required and not capabilities.read_only_verification:
        reasons.append("read-only verification is unavailable")
    if capabilities.isolation_rank < requirements.minimum_isolation_rank:
        reasons.append(
            f"isolation rank {capabilities.isolation_rank} is below required {requirements.minimum_isolation_rank}"
        )
    return reasons


def _bounded_factory(**dependencies: Any) -> AgentLabExecutionBackend:
    return BoundedProcessBackend(
        popen=dependencies.get("popen", subprocess.Popen),
        python_executable=dependencies.get("python_executable", sys.executable),
    )


def _bounded_probe(**_dependencies: Any) -> dict[str, Any]:
    return {
        "available": True,
        "reason_code": "ready",
        "reason": "",
        "engine": "python-process",
        "security_contract": [
            "runtime_guards",
            "clean_environment",
            "process_tree_teardown",
        ],
    }


def _container_factory(**dependencies: Any) -> AgentLabExecutionBackend:
    return ContainerRequiredBackend(
        popen=dependencies.get("popen", subprocess.Popen),
        probe_runtime=dependencies.get("probe_runtime", probe_agent_lab_container_runtime),
        remove_container=dependencies.get("remove_container", remove_agent_lab_container),
    )


def _container_probe(**dependencies: Any) -> dict[str, Any]:
    probe = dependencies.get("probe_runtime", probe_agent_lab_container_runtime)
    return probe()


def _container_prepare(**dependencies: Any) -> dict[str, Any]:
    prepare = dependencies.get("prepare_runtime", prepare_agent_lab_container_runtime)
    return prepare()


def _build_default_registry() -> AgentLabBackendRegistry:
    registry = AgentLabBackendRegistry()
    registry.register(AgentLabBackendRegistration(
        descriptor=AgentLabBackendDescriptor(
            mode=BOUNDED_PROCESS,
            label="Bounded process",
            description="Compatible local Python process with runtime guards.",
            security_tier="guarded_process",
        ),
        factory=_bounded_factory,
        probe=_bounded_probe,
    ))
    registry.register(AgentLabBackendRegistration(
        descriptor=AgentLabBackendDescriptor(
            mode=CONTAINER_REQUIRED,
            label="Secure container",
            description="Fail-closed local Docker or Podman isolation.",
            security_tier="hardened_container",
            preparation_supported=True,
            capabilities=AgentLabBackendCapabilities(isolation_rank=20),
        ),
        factory=_container_factory,
        probe=_container_probe,
        prepare=_container_prepare,
    ))
    return registry


_registry = _build_default_registry()


def get_agent_lab_backend_registry() -> AgentLabBackendRegistry:
    return _registry


def public_backend_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """Remove host implementation details before returning status to the UI."""
    blocked = {"command", "env", "environment", "executable"}

    def sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: sanitize(item)
                for key, item in value.items()
                if str(key).lower() not in blocked
            }
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        return value

    return sanitize(receipt)
