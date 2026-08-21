"""Ownership-scoped resources for integration plugins."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class PluginResourceKind(str, Enum):
    TOOL = "tool"
    LISTENER = "listener"
    SERVICE = "service"
    PROMPT_SECTION = "prompt_section"
    CUSTOM = "custom"


class PluginResourceConflict(RuntimeError):
    """Raised when a plugin tries to replace a resource it does not own."""


class PluginContextDisposed(RuntimeError):
    """Raised when a disposed or execution-only context attempts registration."""


@dataclass(frozen=True, slots=True)
class PluginResourceRecord:
    kind: PluginResourceKind
    name: str
    plugin_id: str
    value: Any = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class DisposeFailure:
    kind: str
    name: str
    error: str


@dataclass(frozen=True, slots=True)
class DisposeReport:
    plugin_id: str
    disposed_count: int
    order: tuple[str, ...]
    failures: tuple[DisposeFailure, ...] = ()
    already_disposed: bool = False

    @property
    def ok(self) -> bool:
        return not self.failures


class PluginResourceRegistry:
    """Shared host registry with strict per-resource ownership."""

    def __init__(self):
        self._resources: dict[PluginResourceKind, dict[str, PluginResourceRecord]] = {
            kind: {} for kind in PluginResourceKind if kind != PluginResourceKind.CUSTOM
        }
        self._lock = threading.RLock()

    def register(
        self,
        kind: PluginResourceKind,
        name: str,
        value: Any,
        *,
        plugin_id: str,
    ) -> PluginResourceRecord:
        normalized = str(name or "").strip()
        owner = str(plugin_id or "").strip()
        if not normalized or not owner:
            raise ValueError("plugin_id and resource name are required")
        if kind == PluginResourceKind.CUSTOM:
            raise ValueError("custom cleanup callbacks are not host resources")
        with self._lock:
            existing = self._resources[kind].get(normalized)
            if existing is not None:
                raise PluginResourceConflict(
                    f"{kind.value} '{normalized}' is already owned by plugin "
                    f"'{existing.plugin_id}'"
                )
            record = PluginResourceRecord(kind, normalized, owner, value)
            self._resources[kind][normalized] = record
            return record

    def unregister(
        self,
        kind: PluginResourceKind,
        name: str,
        *,
        plugin_id: str,
    ) -> bool:
        with self._lock:
            existing = self._resources[kind].get(name)
            if existing is None:
                return False
            if existing.plugin_id != plugin_id:
                raise PluginResourceConflict(
                    f"plugin '{plugin_id}' cannot remove {kind.value} '{name}' "
                    f"owned by '{existing.plugin_id}'"
                )
            del self._resources[kind][name]
            return True

    def get(
        self,
        kind: PluginResourceKind,
        name: str,
        *,
        plugin_id: str | None = None,
    ) -> Any:
        with self._lock:
            record = self._resources[kind].get(str(name or ""))
            if record is None or (plugin_id and record.plugin_id != plugin_id):
                return None
            return record.value

    def list(
        self,
        kind: PluginResourceKind,
        *,
        plugin_id: str | None = None,
    ) -> list[PluginResourceRecord]:
        with self._lock:
            records = list(self._resources[kind].values())
        if plugin_id:
            records = [record for record in records if record.plugin_id == plugin_id]
        return records

    def snapshot(self) -> dict[str, list[dict[str, str]]]:
        with self._lock:
            return {
                kind.value: [
                    {"name": record.name, "plugin_id": record.plugin_id}
                    for record in resources.values()
                ]
                for kind, resources in self._resources.items()
            }


@dataclass
class PluginContext:
    """Execution metadata plus an optional plugin-owned lifecycle scope."""

    mission_id: str = ""
    actor: str = "chief"
    specialist: str = ""
    budget_remaining_usd: float = 0.0
    use_cheap_model: bool = False
    evidence_required: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    plugin_id: str = ""
    _resource_registry: PluginResourceRegistry | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _registration_enabled: bool = field(default=False, repr=False, compare=False)
    _registrations: list[tuple[PluginResourceKind, str, Callable[[], None]]] = field(
        default_factory=list,
        repr=False,
        compare=False,
    )
    _state: str = field(default="execution", repr=False, compare=False)
    _lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
        compare=False,
    )

    @classmethod
    def scoped(
        cls,
        plugin_id: str,
        resources: PluginResourceRegistry,
    ) -> "PluginContext":
        owner = str(plugin_id or "").strip()
        if not owner:
            raise ValueError("plugin_id is required")
        return cls(
            plugin_id=owner,
            _resource_registry=resources,
            _registration_enabled=True,
            _state="active",
        )

    def bind_resource_view(
        self,
        plugin_id: str,
        resources: PluginResourceRegistry,
    ) -> "PluginContext":
        """Attach read-only plugin resources to a per-execution context."""

        with self._lock:
            if self._registration_enabled:
                raise RuntimeError("a lifecycle scope cannot be rebound for execution")
            self.plugin_id = str(plugin_id or "")
            self._resource_registry = resources
            self._state = "execution"
        return self

    def _require_registration_scope(self) -> PluginResourceRegistry:
        if (
            not self._registration_enabled
            or self._resource_registry is None
            or self._state != "active"
        ):
            raise PluginContextDisposed(
                "plugin resources can only be registered during an active setup scope"
            )
        return self._resource_registry

    @staticmethod
    def _infer_disposer(value: Any) -> Callable[[], None] | None:
        for attr in ("dispose", "close", "stop", "shutdown"):
            callback = getattr(value, attr, None)
            if callable(callback):
                return callback
        return None

    def _register(
        self,
        kind: PluginResourceKind,
        name: str,
        value: Any,
        dispose: Callable[[], None] | None,
    ) -> Any:
        with self._lock:
            resources = self._require_registration_scope()
            record = resources.register(kind, name, value, plugin_id=self.plugin_id)

            def cleanup() -> None:
                # Hide the capability first even when its external cleanup fails.
                resources.unregister(kind, record.name, plugin_id=self.plugin_id)
                if dispose is not None:
                    dispose()

            self._registrations.append((kind, record.name, cleanup))
        return value

    def register_tool(
        self,
        name: str,
        handler: Any,
        *,
        dispose: Callable[[], None] | None = None,
    ) -> Any:
        return self._register(PluginResourceKind.TOOL, name, handler, dispose)

    def register_listener(
        self,
        name: str,
        listener: Any,
        *,
        dispose: Callable[[], None] | None = None,
    ) -> Any:
        return self._register(PluginResourceKind.LISTENER, name, listener, dispose)

    def register_service(
        self,
        name: str,
        service: Any,
        *,
        dispose: Callable[[], None] | None = None,
    ) -> Any:
        return self._register(
            PluginResourceKind.SERVICE,
            name,
            service,
            dispose or self._infer_disposer(service),
        )

    def register_prompt_section(
        self,
        name: str,
        section: str | Callable[..., str],
        *,
        dispose: Callable[[], None] | None = None,
    ) -> str | Callable[..., str]:
        return self._register(PluginResourceKind.PROMPT_SECTION, name, section, dispose)

    def subscribe_event_bus(self, name: str, event_bus: Any) -> Any:
        queue = event_bus.subscribe()
        return self.register_listener(
            name,
            queue,
            dispose=lambda: event_bus.unsubscribe(queue),
        )

    def defer(self, name: str, callback: Callable[[], None]) -> None:
        with self._lock:
            self._require_registration_scope()
            if not callable(callback):
                raise TypeError("cleanup callback must be callable")
            self._registrations.append(
                (PluginResourceKind.CUSTOM, str(name or "cleanup"), callback)
            )

    def resource(self, kind: PluginResourceKind, name: str) -> Any:
        if self._resource_registry is None or not self.plugin_id:
            return None
        return self._resource_registry.get(kind, name, plugin_id=self.plugin_id)

    def tool(self, name: str) -> Any:
        return self.resource(PluginResourceKind.TOOL, name)

    def listener(self, name: str) -> Any:
        return self.resource(PluginResourceKind.LISTENER, name)

    def service(self, name: str) -> Any:
        return self.resource(PluginResourceKind.SERVICE, name)

    def prompt_section(self, name: str) -> Any:
        return self.resource(PluginResourceKind.PROMPT_SECTION, name)

    @property
    def disposed(self) -> bool:
        with self._lock:
            return self._state == "disposed"

    def dispose(self) -> DisposeReport:
        """Dispose every owned registration in strict reverse order."""

        with self._lock:
            if self._state in {"disposing", "disposed"}:
                return DisposeReport(
                    plugin_id=self.plugin_id,
                    disposed_count=0,
                    order=(),
                    already_disposed=True,
                )
            self._state = "disposing"
            registrations = list(reversed(self._registrations))
            self._registrations.clear()

        failures: list[DisposeFailure] = []
        order: list[str] = []
        for kind, name, cleanup in registrations:
            order.append(f"{kind.value}:{name}")
            try:
                cleanup()
            except Exception as exc:
                failures.append(
                    DisposeFailure(kind.value, name, f"{type(exc).__name__}: {exc}")
                )

        with self._lock:
            self._registration_enabled = False
            self._state = "disposed"
        return DisposeReport(
            plugin_id=self.plugin_id,
            disposed_count=len(registrations),
            order=tuple(order),
            failures=tuple(failures),
        )

    def __enter__(self) -> "PluginContext":
        self._require_registration_scope()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.dispose()
        return False
