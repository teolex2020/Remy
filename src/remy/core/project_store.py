"""Persistent project catalog and isolated MicroBrain locations.

Projects are the top-level workspace boundary in Remy.  Every project owns
exactly one Aura store (a MicroBrain).  The legacy ``data/brain`` store is
registered as a normal project without moving or copying any user data.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from remy.config.settings import settings
from remy.core.file_utils import atomic_write

PROJECT_CATALOG_VERSION = 1
LEGACY_PROJECT_ID = "legacy-workspace"
LEGACY_BRAIN_ID = "brain-legacy-workspace"
LOCAL_BRAIN_PROVIDER = "local-aura"
SERVER_BRAIN_PROVIDER = "aura-server"
PROJECT_ARTIFACT_DIRS = frozenset(
    {
        "documents",
        "reports",
        "presentations",
        "generated_images",
        "browser_screenshots",
        "history",
        "artifacts",
    }
)
PROJECT_STATE_DIRS = frozenset({"metrics"})
_PROJECT_ID_RE = re.compile(r"^project-[a-f0-9]{32}$")
_BRAIN_ID_RE = re.compile(r"^brain-[a-f0-9]{32}$")
_PROVIDER_ID_RE = re.compile(r"^[a-z][a-z0-9-]{1,47}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


@dataclass(slots=True)
class ProjectRecord:
    project_id: str
    brain_id: str
    name: str
    brain_path: str
    created_at: str
    updated_at: str
    domain: str = ""
    description: str = ""
    brain_provider: str = LOCAL_BRAIN_PROVIDER
    brain_locator: str = ""
    workspace_id: str = ""
    archived_at: str = ""
    legacy: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProjectRecord":
        allowed = {
            "project_id",
            "brain_id",
            "name",
            "brain_path",
            "created_at",
            "updated_at",
            "domain",
            "description",
            "brain_provider",
            "brain_locator",
            "workspace_id",
            "archived_at",
            "legacy",
            "metadata",
        }
        values = {key: value for key, value in payload.items() if key in allowed}
        values.setdefault("brain_provider", LOCAL_BRAIN_PROVIDER)
        values.setdefault("domain", "")
        values.setdefault("description", "")
        values.setdefault("brain_locator", "")
        values.setdefault("workspace_id", "")
        values.setdefault("archived_at", "")
        values.setdefault("legacy", False)
        values.setdefault("metadata", {})
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def brain_uri(self) -> str:
        """Stable public identity that does not expose a path or server locator."""
        return f"microbrain://{self.brain_id}/"


class ProjectStore:
    """Atomic JSON catalog for projects and their MicroBrain paths."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        legacy_brain_path: str | Path | None = None,
    ):
        self.data_dir = _canonical(data_dir or settings.DATA_DIR)
        self.projects_root = self.data_dir / "projects"
        self.catalog_path = self.projects_root / "index.json"
        self.active_path = self.data_dir / "runtime" / "active_project.json"
        self.legacy_brain_path = _canonical(legacy_brain_path or settings.AURA_BRAIN_PATH)
        self._lock = threading.RLock()
        self.projects_root.mkdir(parents=True, exist_ok=True)
        self._ensure_catalog()

    def _ensure_catalog(self) -> None:
        with self._lock:
            if not self.catalog_path.exists():
                now = _now()
                legacy = ProjectRecord(
                    project_id=LEGACY_PROJECT_ID,
                    brain_id=LEGACY_BRAIN_ID,
                    name="Legacy Workspace",
                    brain_path=str(self.legacy_brain_path),
                    brain_provider=LOCAL_BRAIN_PROVIDER,
                    brain_locator=str(self.legacy_brain_path),
                    created_at=now,
                    updated_at=now,
                    legacy=True,
                    metadata={"migration_source": "legacy-global-brain"},
                )
                self._write_catalog([legacy])
            else:
                # Parse and validate eagerly.  A corrupted catalog must never
                # silently redirect an Aura store to an arbitrary path.
                self._read_catalog()

            if not self.active_path.exists():
                self.set_active_project(LEGACY_PROJECT_ID)
            else:
                active_id = self.get_active_project_id()
                if not self.get_project(active_id, include_archived=False):
                    raise RuntimeError(
                        f"Active project marker references an unavailable project: {active_id!r}"
                    )

    def _read_catalog(self) -> list[ProjectRecord]:
        try:
            payload = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Project catalog is unreadable: {self.catalog_path}") from exc

        if int(payload.get("version", 0)) != PROJECT_CATALOG_VERSION:
            raise RuntimeError(
                f"Unsupported project catalog version: {payload.get('version')!r}"
            )

        projects = [ProjectRecord.from_dict(item) for item in payload.get("projects", [])]
        seen_projects: set[str] = set()
        seen_brains: set[str] = set()
        for project in projects:
            self._validate_record(project)
            if project.project_id in seen_projects:
                raise RuntimeError(f"Duplicate project_id in catalog: {project.project_id}")
            if project.brain_id in seen_brains:
                raise RuntimeError(f"Duplicate brain_id in catalog: {project.brain_id}")
            seen_projects.add(project.project_id)
            seen_brains.add(project.brain_id)
        if LEGACY_PROJECT_ID not in seen_projects:
            raise RuntimeError("Project catalog is missing the Legacy Workspace")
        return projects

    def _write_catalog(self, projects: list[ProjectRecord]) -> None:
        payload = {
            "version": PROJECT_CATALOG_VERSION,
            "updated_at": _now(),
            "projects": [project.to_dict() for project in projects],
        }
        atomic_write(
            self.catalog_path,
            json.dumps(payload, ensure_ascii=False, indent=2),
        )

    def _validate_record(self, project: ProjectRecord) -> None:
        is_legacy = project.project_id == LEGACY_PROJECT_ID
        if is_legacy:
            if project.brain_id != LEGACY_BRAIN_ID or not project.legacy:
                raise RuntimeError("Legacy Workspace identity is invalid")
        else:
            if not _PROJECT_ID_RE.fullmatch(project.project_id):
                raise RuntimeError(f"Invalid project_id: {project.project_id!r}")
            if not _BRAIN_ID_RE.fullmatch(project.brain_id):
                raise RuntimeError(f"Invalid brain_id: {project.brain_id!r}")

        name = str(project.name or "").strip()
        if not name or len(name) > 120:
            raise RuntimeError(f"Invalid project name for {project.project_id}")
        domain = " ".join(str(project.domain or "").split())
        if len(domain) > 80:
            raise RuntimeError(f"Invalid project domain for {project.project_id}")
        description = str(project.description or "").strip()
        if len(description) > 2000:
            raise RuntimeError(f"Invalid project description for {project.project_id}")
        project.domain = domain
        project.description = description

        provider = str(project.brain_provider or "").strip().lower()
        if not _PROVIDER_ID_RE.fullmatch(provider):
            raise RuntimeError(
                f"Invalid MicroBrain provider for {project.project_id}: {provider!r}"
            )
        project.brain_provider = provider

        locator = str(project.brain_locator or "").strip()
        if provider == LOCAL_BRAIN_PROVIDER:
            if not str(project.brain_path or "").strip():
                raise RuntimeError(
                    f"Local MicroBrain has no path: {project.project_id}"
                )
            brain_path = _canonical(project.brain_path)
            if is_legacy:
                if brain_path != self.legacy_brain_path:
                    raise RuntimeError(
                        "Legacy Workspace brain path does not match AURA_BRAIN_PATH"
                    )
            else:
                expected_root = self.project_root(project.project_id)
                if not _is_within(brain_path, expected_root):
                    raise RuntimeError(
                        f"MicroBrain path escapes its project boundary: {project.project_id}"
                    )
            if locator and _canonical(locator) != brain_path:
                raise RuntimeError(
                    f"Local MicroBrain locator does not match its path: {project.project_id}"
                )
            project.brain_path = str(brain_path)
            project.brain_locator = str(brain_path)
        else:
            if is_legacy:
                raise RuntimeError("Legacy Workspace must use the local-aura provider")
            if not locator:
                raise RuntimeError(
                    f"Remote MicroBrain has no locator: {project.project_id}"
                )
            if len(locator) > 512:
                raise RuntimeError(
                    f"Remote MicroBrain locator is too long: {project.project_id}"
                )
            project.brain_path = ""
            project.brain_locator = locator

    def project_root(self, project_id: str) -> Path:
        """Return the local artifact boundary independent of memory provider."""
        project_id = str(project_id or "").strip()
        if project_id == LEGACY_PROJECT_ID:
            return self.data_dir
        if not _PROJECT_ID_RE.fullmatch(project_id):
            raise ValueError(f"Invalid project_id: {project_id!r}")
        root = _canonical(self.projects_root / project_id)
        if not _is_within(root, self.projects_root):
            raise RuntimeError("Project root escapes the projects boundary")
        return root

    def list_projects(self, *, include_archived: bool = False) -> list[ProjectRecord]:
        with self._lock:
            projects = self._read_catalog()
        if not include_archived:
            projects = [project for project in projects if not project.archived_at]
        return sorted(projects, key=lambda project: (project.created_at, project.project_id))

    def get_project(
        self,
        project_id: str,
        *,
        include_archived: bool = True,
    ) -> ProjectRecord | None:
        project_id = str(project_id or "").strip()
        for project in self.list_projects(include_archived=include_archived):
            if project.project_id == project_id:
                return project
        return None

    def require_project(
        self,
        project_id: str,
        *,
        include_archived: bool = False,
    ) -> ProjectRecord:
        project = self.get_project(project_id, include_archived=include_archived)
        if project is None:
            raise KeyError(project_id)
        return project

    def create_project(
        self,
        name: str,
        *,
        domain: str = "",
        description: str = "",
        workspace_id: str = "",
        metadata: dict[str, Any] | None = None,
        brain_provider: str = LOCAL_BRAIN_PROVIDER,
        brain_locator: str = "",
    ) -> ProjectRecord:
        name = str(name or "").strip()
        if not name or len(name) > 120:
            raise ValueError("Project name must contain 1-120 characters")
        domain = " ".join(str(domain or "").split())
        if len(domain) > 80:
            raise ValueError("Project area must contain at most 80 characters")
        description = str(description or "").strip()
        if len(description) > 2000:
            raise ValueError("Project purpose must contain at most 2000 characters")

        with self._lock:
            projects = self._read_catalog()
            project_id = f"project-{uuid.uuid4().hex}"
            brain_id = f"brain-{uuid.uuid4().hex}"
            provider = str(brain_provider or "").strip().lower()
            project_root = self.project_root(project_id)
            project_root.mkdir(parents=True, exist_ok=False)
            if provider == LOCAL_BRAIN_PROVIDER:
                brain_path = _canonical(project_root / "brain")
                brain_path.mkdir(parents=True, exist_ok=False)
                resolved_locator = str(brain_path)
            else:
                brain_path = ""
                resolved_locator = str(brain_locator or "").strip()
            now = _now()
            project = ProjectRecord(
                project_id=project_id,
                brain_id=brain_id,
                name=name,
                brain_path=str(brain_path),
                domain=domain,
                description=description,
                brain_provider=provider,
                brain_locator=resolved_locator,
                workspace_id=str(workspace_id or "").strip(),
                created_at=now,
                updated_at=now,
                metadata=dict(metadata or {}),
            )
            self._validate_record(project)
            projects.append(project)
            try:
                self._write_catalog(projects)
            except BaseException:
                # The empty directory is harmless and deliberately retained.
                # Avoid destructive rollback if another process touched it.
                raise
        return project

    def update_project(
        self,
        project_id: str,
        *,
        name: str | None = None,
        domain: str | None = None,
        description: str | None = None,
        workspace_id: str | None = None,
    ) -> ProjectRecord:
        with self._lock:
            projects = self._read_catalog()
            for project in projects:
                if project.project_id != project_id:
                    continue
                if name is not None:
                    clean_name = str(name).strip()
                    if not clean_name or len(clean_name) > 120:
                        raise ValueError("Project name must contain 1-120 characters")
                    project.name = clean_name
                if domain is not None:
                    clean_domain = " ".join(str(domain).split())
                    if len(clean_domain) > 80:
                        raise ValueError("Project area must contain at most 80 characters")
                    project.domain = clean_domain
                if description is not None:
                    clean_description = str(description).strip()
                    if len(clean_description) > 2000:
                        raise ValueError(
                            "Project purpose must contain at most 2000 characters"
                        )
                    project.description = clean_description
                if workspace_id is not None:
                    project.workspace_id = str(workspace_id or "").strip()
                project.updated_at = _now()
                self._validate_record(project)
                self._write_catalog(projects)
                return project
        raise KeyError(project_id)

    def archive_project(self, project_id: str) -> ProjectRecord:
        if project_id == LEGACY_PROJECT_ID:
            raise ValueError("Legacy Workspace cannot be archived")
        with self._lock:
            projects = self._read_catalog()
            for project in projects:
                if project.project_id != project_id:
                    continue
                project.archived_at = project.archived_at or _now()
                project.updated_at = _now()
                self._write_catalog(projects)
                if self.get_active_project_id() == project_id:
                    self.set_active_project(LEGACY_PROJECT_ID)
                return project
        raise KeyError(project_id)

    def restore_project(self, project_id: str) -> ProjectRecord:
        """Restore an archived project without changing its MicroBrain identity."""
        with self._lock:
            projects = self._read_catalog()
            for project in projects:
                if project.project_id != project_id:
                    continue
                if project.archived_at:
                    project.archived_at = ""
                    project.updated_at = _now()
                    self._validate_record(project)
                    self._write_catalog(projects)
                return project
        raise KeyError(project_id)

    def get_active_project_id(self) -> str:
        if not self.active_path.exists():
            raise RuntimeError(f"Active project marker is missing: {self.active_path}")
        try:
            payload = json.loads(self.active_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Active project marker is unreadable: {self.active_path}"
            ) from exc
        project_id = str(payload.get("project_id") or "").strip()
        if not project_id:
            raise RuntimeError("Active project marker has no project_id")
        return project_id

    def get_active_project(self) -> ProjectRecord:
        project_id = self.get_active_project_id()
        project = self.get_project(project_id, include_archived=False)
        if project is None:
            raise RuntimeError(
                f"Active project is unavailable or archived: {project_id!r}"
            )
        return project

    def set_active_project(self, project_id: str) -> ProjectRecord:
        project = self.require_project(project_id, include_archived=False)
        atomic_write(
            self.active_path,
            json.dumps(
                {
                    "project_id": project.project_id,
                    "brain_id": project.brain_id,
                    "brain_provider": project.brain_provider,
                    "updated_at": _now(),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        return project


_project_store: ProjectStore | None = None
_project_store_key: tuple[str, str] | None = None
_project_store_lock = threading.Lock()


def get_project_store() -> ProjectStore:
    """Return a singleton that follows runtime path overrides used by tests."""
    global _project_store, _project_store_key
    key = (
        str(_canonical(settings.DATA_DIR)),
        str(_canonical(settings.AURA_BRAIN_PATH)),
    )
    if _project_store is not None and _project_store_key == key:
        return _project_store
    with _project_store_lock:
        if _project_store is None or _project_store_key != key:
            _project_store = ProjectStore()
            _project_store_key = key
    return _project_store


def project_data_root(project_id: str | None = None) -> Path:
    """Return the filesystem boundary for one project's non-Aura artifacts."""
    from remy.core.microbrain import current_project_id

    project = get_project_store().require_project(
        str(project_id or "").strip() or current_project_id()
    )
    if project.project_id == LEGACY_PROJECT_ID:
        return _canonical(settings.DATA_DIR)
    return get_project_store().project_root(project.project_id)


def local_brain_path(project_id: str | None = None) -> Path:
    """Resolve a local Aura path or fail explicitly for a remote provider."""
    from remy.core.microbrain import current_project_id

    project = get_project_store().require_project(
        str(project_id or "").strip() or current_project_id()
    )
    if project.brain_provider != LOCAL_BRAIN_PROVIDER:
        raise RuntimeError(
            f"MicroBrain provider {project.brain_provider!r} has no local brain path"
        )
    return _canonical(project.brain_locator)


def brain_display_location(project_id: str | None = None) -> str:
    """Return a safe operator-facing location without exposing remote locators."""
    from remy.core.microbrain import current_project_id

    project = get_project_store().require_project(
        str(project_id or "").strip() or current_project_id()
    )
    if project.brain_provider == LOCAL_BRAIN_PROVIDER:
        return str(local_brain_path(project.project_id))
    return project.brain_uri


def project_artifact_dir(
    kind: str,
    project_id: str | None = None,
    *,
    legacy_data_dir: str | Path | None = None,
    create: bool = False,
) -> Path:
    """Return one trusted artifact directory inside a project's boundary."""
    normalized_kind = str(kind or "").strip()
    if normalized_kind not in PROJECT_ARTIFACT_DIRS:
        raise ValueError(f"Unsupported project artifact directory: {normalized_kind}")

    from remy.core.microbrain import current_project_id

    resolved_project_id = str(project_id or "").strip() or current_project_id()
    project = get_project_store().require_project(resolved_project_id)
    if project.project_id == LEGACY_PROJECT_ID and legacy_data_dir is not None:
        root = _canonical(legacy_data_dir)
    else:
        root = project_data_root(project.project_id)
    path = _canonical(root / normalized_kind)
    if not _is_within(path, root):
        raise RuntimeError("Artifact directory escapes its project boundary")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def project_state_path(
    kind: str,
    filename: str,
    project_id: str | None = None,
    *,
    legacy_data_root: bool = False,
    create_parent: bool = False,
) -> Path:
    """Return a trusted project-owned runtime state path.

    New projects keep operational state below ``<project>/.meta``. The
    Legacy Workspace preserves its historical locations so upgrades do not
    make existing metrics disappear.
    """
    normalized_kind = str(kind or "").strip()
    if normalized_kind not in PROJECT_STATE_DIRS:
        raise ValueError(f"Unsupported project state directory: {normalized_kind}")

    clean_filename = str(filename or "").strip()
    if not clean_filename or Path(clean_filename).name != clean_filename:
        raise ValueError("Project state filename must be a plain filename")

    from remy.core.microbrain import current_project_id

    resolved_project_id = str(project_id or "").strip() or current_project_id()
    project = get_project_store().require_project(resolved_project_id)
    if project.project_id == LEGACY_PROJECT_ID:
        if legacy_data_root:
            path = _canonical(settings.DATA_DIR / clean_filename)
        else:
            from remy.core.meta_store import resolve_path

            path = _canonical(resolve_path(clean_filename, normalized_kind))
    else:
        root = project_data_root(project.project_id)
        state_root = _canonical(root / ".meta" / normalized_kind)
        if not _is_within(state_root, root):
            raise RuntimeError("Project state directory escapes its project boundary")
        path = _canonical(state_root / clean_filename)
        if not _is_within(path, state_root):
            raise RuntimeError("Project state file escapes its project boundary")

    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def reset_project_store_for_tests() -> None:
    global _project_store, _project_store_key
    with _project_store_lock:
        _project_store = None
        _project_store_key = None
