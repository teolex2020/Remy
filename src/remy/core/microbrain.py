"""Runtime project binding and lazy MicroBrain instance registry."""

from __future__ import annotations

import contextvars
import logging
import threading
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from typing import Callable, Iterator, Protocol, TypeVar

from remy.core.project_store import (
    LOCAL_BRAIN_PROVIDER,
    ProjectRecord,
    ProjectStore,
    get_project_store,
)


class ClosableBrain(Protocol):
    def close(self) -> None: ...


BrainT = TypeVar("BrainT", bound=ClosableBrain)
logger = logging.getLogger(__name__)
_bound_project_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "remy_project_id",
    default="",
)
_scope_lock = threading.RLock()
_active_scopes: dict[str, int] = {}
_registries: weakref.WeakSet[object] = weakref.WeakSet()


def _scope_count(project_id: str) -> int:
    with _scope_lock:
        return int(_active_scopes.get(project_id, 0))


def _change_scope_count(project_id: str, delta: int) -> None:
    with _scope_lock:
        count = int(_active_scopes.get(project_id, 0)) + delta
        if count <= 0:
            _active_scopes.pop(project_id, None)
        else:
            _active_scopes[project_id] = count


def _trim_registered_hosts() -> None:
    for registry in list(_registries):
        try:
            errors = registry.trim_idle()
        except Exception:  # noqa: BLE001 - one host must not break scope cleanup
            logger.exception("MicroBrain host trim failed")
            continue
        for project_id, exc in errors:
            logger.warning(
                "Could not evict idle MicroBrain for project %s: %s",
                project_id,
                exc,
            )


def current_project_id() -> str:
    """Return the explicitly bound project, or the locally active project."""
    bound = _bound_project_id.get().strip()
    if bound:
        return bound
    return get_project_store().get_active_project().project_id


@contextmanager
def bind_project(project_id: str) -> Iterator[ProjectRecord]:
    """Bind all nested memory access to one project for this async context."""
    project = get_project_store().require_project(project_id, include_archived=False)
    _change_scope_count(project.project_id, 1)
    token = _bound_project_id.set(project.project_id)
    try:
        yield project
    finally:
        _bound_project_id.reset(token)
        _change_scope_count(project.project_id, -1)
        _trim_registered_hosts()


class MicroBrainRegistry:
    """Open at most one Aura instance per project and close all on shutdown."""

    def __init__(
        self,
        opener: Callable[[ProjectRecord], BrainT] | None = None,
        *,
        providers: dict[str, Callable[[ProjectRecord], BrainT]] | None = None,
        project_store: ProjectStore | None = None,
        max_open: int = 8,
    ):
        self._providers = {
            str(name or "").strip().lower(): provider
            for name, provider in dict(providers or {}).items()
        }
        if opener is not None:
            self._providers.setdefault(LOCAL_BRAIN_PROVIDER, opener)
        self._project_store = project_store or get_project_store()
        if int(max_open) < 1:
            raise ValueError("MicroBrain max_open must be at least 1")
        self._max_open = int(max_open)
        self._instances: OrderedDict[str, BrainT] = OrderedDict()
        self._project_by_brain: dict[str, str] = {}
        self._eviction_errors: list[tuple[str, str]] = []
        self._lock = threading.RLock()
        self._closed = False
        _registries.add(self)

    def register_provider(
        self,
        name: str,
        opener: Callable[[ProjectRecord], BrainT],
        *,
        replace: bool = False,
    ) -> None:
        """Register an in-process provider without coupling projects to its SDK."""
        provider_id = str(name or "").strip().lower()
        if not provider_id:
            raise ValueError("MicroBrain provider name is required")
        with self._lock:
            if self._closed:
                raise RuntimeError("MicroBrain registry has already been closed.")
            if provider_id in self._providers and not replace:
                raise ValueError(
                    f"MicroBrain provider is already registered: {provider_id}"
                )
            self._providers[provider_id] = opener

    def available_providers(self) -> list[str]:
        with self._lock:
            return sorted(self._providers)

    def get(self, project_id: str | None = None) -> BrainT:
        if self._closed:
            raise RuntimeError("MicroBrain registry has already been closed.")
        resolved_id = str(project_id or "").strip() or current_project_id()
        project = self._project_store.require_project(resolved_id, include_archived=False)
        with self._lock:
            instance = self._instances.get(project.brain_id)
            if instance is None:
                opener = self._providers.get(project.brain_provider)
                if opener is None:
                    raise RuntimeError(
                        "MicroBrain provider is not configured: "
                        f"{project.brain_provider!r}"
                    )
                instance = opener(project)
                self._instances[project.brain_id] = instance
                self._project_by_brain[project.brain_id] = project.project_id
            self._instances.move_to_end(project.brain_id)
            errors = self._trim_idle_locked(protect={project.project_id})
            self._remember_eviction_errors(errors)
            return instance

    def _active_project_id(self) -> str | None:
        try:
            return self._project_store.get_active_project().project_id
        except Exception:
            # An unreadable active marker is already a hard project-store
            # error. Never evict memory while ownership is uncertain.
            return None

    def _trim_idle_locked(
        self,
        *,
        protect: set[str] | None = None,
    ) -> list[tuple[str, Exception]]:
        protected = set(protect or ())
        active_project_id = self._active_project_id()
        if active_project_id is None:
            return []
        if active_project_id:
            protected.add(active_project_id)
        errors: list[tuple[str, Exception]] = []

        while len(self._instances) > self._max_open:
            candidate = None
            for brain_id, instance in self._instances.items():
                project_id = self._project_by_brain.get(brain_id, brain_id)
                if project_id in protected or _scope_count(project_id) > 0:
                    continue
                candidate = (brain_id, project_id, instance)
                break
            if candidate is None:
                break

            brain_id, project_id, instance = candidate
            try:
                instance.close()
            except Exception as exc:  # noqa: BLE001 - retain failed instance
                errors.append((project_id, exc))
                protected.add(project_id)
                self._instances.move_to_end(brain_id)
                continue
            self._instances.pop(brain_id, None)
            self._project_by_brain.pop(brain_id, None)
        return errors

    def _remember_eviction_errors(
        self,
        errors: list[tuple[str, Exception]],
    ) -> None:
        for project_id, exc in errors:
            self._eviction_errors.append((project_id, str(exc)))
        if len(self._eviction_errors) > 20:
            self._eviction_errors = self._eviction_errors[-20:]

    def trim_idle(self) -> list[tuple[str, Exception]]:
        """Evict least-recently-used brains that are not active or leased."""
        with self._lock:
            if self._closed:
                return []
            errors = self._trim_idle_locked()
            self._remember_eviction_errors(errors)
            return errors

    def host_status(self) -> dict:
        """Return safe lifecycle diagnostics for operators and tests."""
        with self._lock:
            active_project_id = self._active_project_id()
            mounted = [
                self._project_by_brain.get(brain_id, brain_id)
                for brain_id in self._instances
            ]
            pinned = sorted(
                project_id
                for project_id in mounted
                if _scope_count(project_id) > 0
                or project_id == active_project_id
            )
            return {
                "capacity": self._max_open,
                "mounted_count": len(mounted),
                "mounted_projects": mounted,
                "pinned_projects": pinned,
                "over_capacity": len(mounted) > self._max_open,
                "ownership_known": active_project_id is not None,
                "eviction_errors": [
                    {"project_id": project_id, "error": error}
                    for project_id, error in self._eviction_errors
                ],
            }

    def is_initialized(self, project_id: str | None = None) -> bool:
        resolved_id = str(project_id or "").strip() or current_project_id()
        project = self._project_store.require_project(resolved_id, include_archived=True)
        with self._lock:
            return project.brain_id in self._instances

    def initialized_projects(self) -> list[str]:
        with self._lock:
            return sorted(self._project_by_brain.values())

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def owns_instance(self, instance: object) -> bool:
        with self._lock:
            return any(item is instance for item in self._instances.values())

    def close_project(self, project_id: str) -> bool:
        project = self._project_store.require_project(project_id, include_archived=True)
        if _scope_count(project.project_id) > 0:
            raise RuntimeError(
                f"MicroBrain is still in use by project {project.project_id}"
            )
        with self._lock:
            instance = self._instances.get(project.brain_id)
            if instance is None:
                return False
            instance.close()
            self._instances.pop(project.brain_id, None)
            self._project_by_brain.pop(project.brain_id, None)
            return True

    def close_provider(self, provider_id: str) -> list[tuple[str, Exception]]:
        """Close mounted brains from one provider so configuration can refresh."""
        normalized = str(provider_id or "").strip().lower()
        with self._lock:
            project_ids = [
                project_id
                for project_id in self._project_by_brain.values()
                if self._project_store.require_project(
                    project_id,
                    include_archived=True,
                ).brain_provider == normalized
            ]
        errors: list[tuple[str, Exception]] = []
        for project_id in project_ids:
            try:
                self.close_project(project_id)
            except Exception as exc:  # noqa: BLE001 - refresh remaining brains
                errors.append((project_id, exc))
        return errors

    def close_all(self) -> list[tuple[str, Exception]]:
        with self._lock:
            if self._closed:
                return []
            self._closed = True
            items = [
                (self._project_by_brain.get(brain_id, brain_id), instance)
                for brain_id, instance in self._instances.items()
            ]
            self._instances.clear()
            self._project_by_brain.clear()
            _registries.discard(self)

        errors: list[tuple[str, Exception]] = []
        for project_id, instance in items:
            try:
                instance.close()
            except Exception as exc:  # noqa: BLE001 - collect every close failure
                errors.append((project_id, exc))
        return errors
