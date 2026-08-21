"""Capability-based access to user-approved local workspaces.

Every local file and shell tool must pass through this module.  A workspace
grant is an explicit allow-list entry with independent read/write/execute
capabilities.  Paths are resolved before the boundary check, which also blocks
``..`` and symlink escapes.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from remy.config.settings import settings
from remy.core.file_utils import atomic_write

logger = logging.getLogger(__name__)

_MAX_READ_SIZE = 10 * 1024 * 1024
_MAX_WRITE_SIZE = 1_000_000
_SKIP_PARTS = {".git", "__pycache__", "node_modules"}
_CODE_SKIP_PARTS = _SKIP_PARTS | {
    ".idea", ".vscode", ".venv", "venv", "dist", "build", "coverage",
    ".next", ".nuxt", "target", "vendor",
}
_CODE_MANIFESTS = (
    "README.md", "README.rst", "pyproject.toml", "package.json", "Cargo.toml",
    "go.mod", "pom.xml", "build.gradle", "requirements.txt", "Dockerfile",
    "docker-compose.yml", "Makefile", "tsconfig.json",
)
_WRITE_BLOCKED_PARTS = {".git", "brain"}
_BUILTIN_AGENT_DIRS = {"documents", "sandbox"}
_SHELL_BLOCKED_RE = re.compile(
    r"rm\s+(-[rRf]+\s+)?[/~]|mkfs\.|dd\s+.*of=/dev/|"
    r"shutdown|reboot|halt|poweroff|:\(\)\{\s*:\|:&\s*\};:|"
    r"chmod\s+(-R\s+)?777\s+/|>\s*/dev/sd|"
    r"(?:curl|wget).*\|\s*(?:bash|sh)|"
    r"taskkill\s+/f\s+/im\s+(?:python|remy|node)|"
    r"net\s+stop|reg\s+delete|format\s+[a-zA-Z]:",
    re.IGNORECASE,
)
_LOCK = threading.RLock()


class WorkspaceAccessError(PermissionError):
    """Raised when a path is outside an approved capability boundary."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _is_filesystem_root(path: Path) -> bool:
    return path == Path(path.anchor) or path.parent == path


class WorkspacePermissionManager:
    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = _canonical(data_dir or settings.DATA_DIR)
        self.store_path = self.data_dir / "workspace_permissions.json"
        self.audit_path = self.data_dir / "audit_logs" / "workspace_access.jsonl"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_store()

    def _builtin(self) -> dict[str, Any]:
        return {
            "workspace_id": "data",
            "name": "Remy data",
            "root_path": str(self.data_dir),
            "permissions": ["read", "write"],
            "source": "builtin",
            "created_at": "",
            "updated_at": "",
            "revoked_at": "",
            "uri": "workspace://data/",
        }

    def _ensure_store(self) -> None:
        with _LOCK:
            if self.store_path.exists():
                return
            grants: list[dict[str, Any]] = []
            for raw in getattr(settings, "AUTONOMY_ALLOWED_READ_PATHS", []) or []:
                try:
                    root = _canonical(raw)
                    if root.exists() and root.is_dir() and not _is_filesystem_root(root):
                        grants.append(self._make_grant(root, root.name, {"read"}, "legacy-migration"))
                except Exception:
                    logger.warning("Could not migrate legacy read path: %s", raw)
            self._save(grants)

    def _load(self) -> list[dict[str, Any]]:
        try:
            payload = json.loads(self.store_path.read_text(encoding="utf-8"))
            return list(payload.get("workspaces", []))
        except (OSError, ValueError, TypeError):
            return []

    def _save(self, grants: list[dict[str, Any]]) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(
            self.store_path,
            json.dumps({"version": 1, "workspaces": grants}, ensure_ascii=False, indent=2) + "\n",
        )

    @staticmethod
    def _make_grant(root: Path, name: str, permissions: set[str], source: str) -> dict[str, Any]:
        workspace_id = "ws-" + hashlib.sha256(os.path.normcase(str(root)).encode()).hexdigest()[:12]
        now = _now()
        return {
            "workspace_id": workspace_id,
            "name": name or root.name or str(root),
            "root_path": str(root),
            "permissions": sorted(permissions),
            "source": source,
            "created_at": now,
            "updated_at": now,
            "revoked_at": "",
            "uri": f"workspace://{workspace_id}/",
        }

    def list_grants(self, include_revoked: bool = False) -> list[dict[str, Any]]:
        grants = [self._builtin(), *self._load()]
        return grants if include_revoked else [g for g in grants if not g.get("revoked_at")]

    def get_grant(self, workspace_id: str) -> dict[str, Any] | None:
        return next(
            (grant for grant in self.list_grants() if grant.get("workspace_id") == workspace_id),
            None,
        )

    def add_grant(
        self,
        root_path: str,
        *,
        name: str = "",
        permissions: set[str] | None = None,
        source: str = "user",
    ) -> dict[str, Any]:
        root = _canonical(root_path)
        if not root.exists() or not root.is_dir():
            raise ValueError("The selected folder does not exist or is not a directory.")
        if _is_filesystem_root(root):
            raise ValueError("Granting an entire filesystem or drive is not allowed. Select a narrower folder.")
        perms = set(permissions or {"read"})
        if not perms or not perms <= {"read", "write", "execute"}:
            raise ValueError("Permissions must contain read, write, or execute.")
        grant = self._make_grant(root, name or root.name, perms, source)
        with _LOCK:
            grants = self._load()
            existing = next((g for g in grants if g.get("workspace_id") == grant["workspace_id"]), None)
            if existing:
                existing.update({
                    "name": grant["name"],
                    "root_path": grant["root_path"],
                    "permissions": grant["permissions"],
                    "updated_at": _now(),
                    "revoked_at": "",
                    "uri": grant["uri"],
                })
                grant = existing
            else:
                grants.append(grant)
            self._save(grants)
        self.audit("grant", root, grant, True, "workspace granted", "settings")
        return grant

    def update_grant(self, workspace_id: str, *, name: str | None = None, permissions: set[str] | None = None) -> dict[str, Any]:
        if workspace_id == "data":
            raise ValueError("The built-in Remy data workspace cannot be changed.")
        if permissions is not None and (not permissions or not permissions <= {"read", "write", "execute"}):
            raise ValueError("Permissions must contain read, write, or execute.")
        with _LOCK:
            grants = self._load()
            grant = next((g for g in grants if g.get("workspace_id") == workspace_id and not g.get("revoked_at")), None)
            if not grant:
                raise KeyError(workspace_id)
            if name is not None:
                grant["name"] = name.strip() or Path(grant["root_path"]).name
            if permissions is not None:
                grant["permissions"] = sorted(permissions)
            grant["updated_at"] = _now()
            self._save(grants)
        self.audit("update_grant", Path(grant["root_path"]), grant, True, "permissions updated", "settings")
        return grant

    def revoke_grant(self, workspace_id: str) -> dict[str, Any]:
        if workspace_id == "data":
            raise ValueError("The built-in Remy data workspace cannot be revoked.")
        with _LOCK:
            grants = self._load()
            grant = next((g for g in grants if g.get("workspace_id") == workspace_id and not g.get("revoked_at")), None)
            if not grant:
                raise KeyError(workspace_id)
            grant["revoked_at"] = _now()
            grant["updated_at"] = grant["revoked_at"]
            self._save(grants)
        self.audit("revoke", Path(grant["root_path"]), grant, True, "workspace revoked", "settings")
        return grant

    def resolve(self, raw_path: str, permission: str, *, must_exist: bool = False) -> tuple[Path, dict[str, Any]]:
        raw = str(raw_path or "").strip()
        if not raw:
            raw = "workspace://data/"
        grants = self.list_grants()
        grant: dict[str, Any] | None = None
        if raw.startswith("workspace://"):
            rest = raw[len("workspace://"):]
            workspace_id, _, relative = rest.partition("/")
            grant = next((g for g in grants if g["workspace_id"] == workspace_id), None)
            if not grant:
                raise WorkspaceAccessError(f"Unknown or revoked workspace: {workspace_id}")
            target = _canonical(Path(grant["root_path"]) / relative)
        else:
            candidate = Path(raw)
            target = _canonical(candidate if candidate.is_absolute() else self.data_dir / candidate)
            eligible = []
            for item in grants:
                root = _canonical(item["root_path"])
                if target == root or target.is_relative_to(root):
                    eligible.append((len(root.parts), item))
            if eligible:
                grant = max(eligible, key=lambda pair: pair[0])[1]
        if not grant:
            self.audit("resolve", target, None, False, "outside approved workspaces", permission)
            raise WorkspaceAccessError("Access denied: path is outside approved workspaces. Add the folder in Settings → Local Workspaces.")
        root = _canonical(grant["root_path"])
        if target != root and not target.is_relative_to(root):
            self.audit("resolve", target, grant, False, "path or symlink escapes workspace", permission)
            raise WorkspaceAccessError("Access denied: path escapes the approved workspace.")
        if permission not in grant.get("permissions", []):
            self.audit("resolve", target, grant, False, f"missing {permission} permission", permission)
            raise WorkspaceAccessError(f"Access denied: workspace '{grant['name']}' has no {permission} permission.")
        if grant.get("source") == "builtin":
            self._enforce_builtin_boundary(target, root, permission)
        if must_exist and not target.exists():
            raise FileNotFoundError(str(target))
        return target, grant

    def _enforce_builtin_boundary(
        self,
        target: Path,
        root: Path,
        permission: str,
    ) -> None:
        """Expose a small project workspace, never Remy's internal state."""
        relative = target.relative_to(root)
        if not relative.parts:
            if permission == "read":
                return
            raise WorkspaceAccessError(
                "Access denied: write inside Documents or Sandbox, not the project root."
            )
        top = relative.parts[0].lower()
        if top not in _BUILTIN_AGENT_DIRS:
            raise WorkspaceAccessError(
                "Access denied: Remy internal project data is not available to the agent. "
                "Use workspace://data/documents/ for explicitly shared documents or "
                "workspace://data/sandbox/ for working files."
            )
        if top == "documents" and permission == "read" and len(relative.parts) > 1:
            from remy.core.project_documents import (
                document_access_manifest,
                is_agent_accessible,
            )

            documents_dir = root / "documents"
            if target == document_access_manifest(documents_dir):
                raise WorkspaceAccessError("Access denied: document visibility metadata is internal.")
            if target.is_file() and not is_agent_accessible(documents_dir, target.name):
                raise WorkspaceAccessError(
                    "Access denied: this document is private. "
                    "The user must enable Agent access in Documents."
                )

    def audit(self, action: str, target: Path, grant: dict[str, Any] | None, allowed: bool, reason: str, tool: str) -> None:
        record = {
            "timestamp": _now(), "action": action, "tool": tool,
            "workspace_id": grant.get("workspace_id") if grant else None,
            "path": str(target), "allowed": allowed, "reason": reason,
        }
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with _LOCK, self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            logger.exception("Could not append workspace access audit")

    def read_audit(self, limit: int = 100) -> list[dict[str, Any]]:
        if not self.audit_path.exists():
            return []
        lines = self.audit_path.read_text(encoding="utf-8", errors="replace").splitlines()[-max(1, min(limit, 1000)):]
        records = []
        for line in reversed(lines):
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
        return records


def get_workspace_manager() -> WorkspacePermissionManager:
    # The built-in workspace follows the bound project.  User-granted external
    # workspaces remain explicit capabilities stored inside that project.
    try:
        from remy.core.project_store import project_data_root

        configured_root = _canonical(settings.DATA_DIR)
        candidate = _canonical(project_data_root())
        data_dir = (
            candidate
            if candidate == configured_root or candidate.is_relative_to(configured_root)
            else configured_root
        )
    except Exception:
        # Startup and isolated unit tests may run before the project catalog is
        # available. Falling back preserves a closed local boundary.
        data_dir = settings.DATA_DIR
    return WorkspacePermissionManager(data_dir)


def build_code_workspace_context(workspace_id: str) -> str:
    """Build a compact, deterministic repository brief for the agent.

    The brief intentionally contains structure and capabilities, not whole file
    contents. The model must use fs_search/fs_read for evidence before making
    claims about the code.
    """
    manager = get_workspace_manager()
    grant = manager.get_grant(str(workspace_id or "").strip())
    if not grant or grant.get("source") == "builtin":
        return ""
    if "read" not in grant.get("permissions", []):
        return ""

    root = _canonical(grant["root_path"])
    top_level: list[str] = []
    manifests: list[str] = []
    extension_counts: dict[str, int] = {}
    file_count = 0
    truncated = False

    try:
        for entry in sorted(root.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            if entry.name in _CODE_SKIP_PARTS:
                continue
            suffix = "/" if entry.is_dir() else ""
            top_level.append(entry.name + suffix)
            if entry.name in _CODE_MANIFESTS:
                manifests.append(entry.name)
            if len(top_level) >= 80:
                break

        for current_root, dir_names, file_names in os.walk(root):
            dir_names[:] = [name for name in dir_names if name not in _CODE_SKIP_PARTS]
            for name in file_names:
                file_count += 1
                suffix = Path(name).suffix.lower() or "[no extension]"
                extension_counts[suffix] = extension_counts.get(suffix, 0) + 1
                if name in _CODE_MANIFESTS and name not in manifests:
                    relative = str((Path(current_root) / name).relative_to(root))
                    manifests.append(relative)
                if file_count >= 5000:
                    truncated = True
                    break
            if truncated:
                break
    except OSError:
        pass

    languages = ", ".join(
        f"{suffix}:{count}"
        for suffix, count in sorted(extension_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
    ) or "unknown"
    capabilities = ", ".join(grant.get("permissions", []))
    return (
        "=== ACTIVE CODE WORKSPACE ===\n"
        f"Name: {grant['name']}\n"
        f"Workspace URI: {grant['uri']}\n"
        f"Local root: {root}\n"
        f"Capabilities: {capabilities}\n"
        f"Files sampled: {file_count}{'+' if truncated else ''}\n"
        f"Main extensions: {languages}\n"
        f"Detected manifests: {', '.join(manifests[:20]) or 'none'}\n"
        f"Top level: {', '.join(top_level) or '(empty)'}\n"
        "Use fs_search and fs_read with the Workspace URI to inspect code. "
        "Do not claim the repository was analyzed until those tools return evidence in this turn. "
        "Read access does not authorize edits or command execution."
    )


def preflight_workspace_access(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Resolve capability-scoped I/O before execution without performing I/O.

    Handlers intentionally resolve the capability again immediately before the
    operation.  The second check protects against grant revocation or path
    changes between policy evaluation and execution.
    """

    name = str(tool_name or "")
    specs = {
        "read_file": ("path", "read", True, True),
        "fs_read": ("path", "read", True, True),
        "write_file": ("path", "write", False, True),
        "fs_write": ("path", "write", False, True),
        "list_directory": ("path", "read", True, False),
        "fs_search": ("path", "read", True, False),
        "shell_exec": ("working_dir", "execute", True, True),
    }
    spec = specs.get(name)
    if spec is None:
        return {"applies": False, "allowed": True}

    key, permission, must_exist, required = spec
    raw = str(args.get(key) or "").strip()
    if not raw and not required:
        raw = "workspace://data/"
    try:
        if not raw:
            label = "working_dir" if key == "working_dir" else "path"
            raise ValueError(f"{label} is required")
        if name == "shell_exec":
            command = str(args.get("command") or "").strip()
            if not command:
                raise ValueError("command is required")
            if _SHELL_BLOCKED_RE.search(command):
                raise WorkspaceAccessError(
                    "Command blocked because it matches a catastrophic or irreversible pattern."
                )
        manager = get_workspace_manager()
        target, grant = manager.resolve(raw, permission, must_exist=must_exist)
        if name in {"write_file", "fs_write"} and any(
            part.lower() in _WRITE_BLOCKED_PARTS for part in target.parts
        ):
            raise WorkspaceAccessError("Write denied: protected internal directory.")
        return {
            "applies": True,
            "allowed": True,
            "permission": permission,
            "workspace_id": str(grant.get("workspace_id") or ""),
            "defense_in_depth": True,
        }
    except Exception as exc:
        reason = f"File not found: {exc}" if isinstance(exc, FileNotFoundError) else str(exc)
        return {
            "applies": True,
            "allowed": False,
            "permission": permission,
            "reason": reason,
            "error_type": type(exc).__name__,
            "defense_in_depth": True,
        }


def workspace_read(args: dict[str, Any], *, tool: str = "workspace_read") -> str:
    manager = get_workspace_manager()
    raw = str(args.get("path") or "").strip()
    try:
        target, grant = manager.resolve(raw, "read", must_exist=True)
        if not target.is_file():
            raise IsADirectoryError(str(target))
        stat = target.stat()
        if stat.st_size > _MAX_READ_SIZE:
            raise ValueError(f"File too large: {stat.st_size} bytes (max {_MAX_READ_SIZE})")
        offset = max(0, int(args.get("offset") or 0))
        limit = min(2000, max(1, int(args.get("limit") or 500)))
        raw_bytes = target.read_bytes()
        if b"\x00" in raw_bytes[:512]:
            result = {"path": str(target), "workspace_id": grant["workspace_id"], "size": stat.st_size,
                      "binary": True, "content_base64": base64.b64encode(raw_bytes[:75000]).decode("ascii"),
                      "truncated": len(raw_bytes) > 75000}
        else:
            lines = raw_bytes.decode(str(args.get("encoding") or "utf-8"), errors="replace").splitlines()
            selected = lines[offset:offset + limit]
            result = {"path": str(target), "workspace_id": grant["workspace_id"], "size": stat.st_size,
                      "total_lines": len(lines), "offset": offset, "lines_returned": len(selected),
                      "content": "\n".join(selected)}
        manager.audit("read", target, grant, True, "read completed", tool)
        return json.dumps(result, ensure_ascii=False)
    except FileNotFoundError as exc:
        return json.dumps({"error": f"File not found: {exc}"}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


def workspace_write(args: dict[str, Any], *, tool: str = "workspace_write") -> str:
    manager = get_workspace_manager()
    raw = str(args.get("path") or "").strip()
    content = str(args.get("content") or "")
    try:
        if not raw:
            raise ValueError("path is required")
        if len(content) > _MAX_WRITE_SIZE:
            raise ValueError(f"Content too large: {len(content)} chars (max {_MAX_WRITE_SIZE})")
        target, grant = manager.resolve(raw, "write")
        if any(part.lower() in _WRITE_BLOCKED_PARTS for part in target.parts):
            raise WorkspaceAccessError("Write denied: protected internal directory.")
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = str(args.get("mode") or "write").lower()
        if mode == "append":
            with target.open("a", encoding="utf-8") as handle:
                handle.write(content)
        elif mode == "write":
            atomic_write(target, content)
        else:
            raise ValueError("mode must be 'write' or 'append'")
        if (
            grant.get("source") == "builtin"
            and target.parent == _canonical(grant["root_path"]) / "documents"
            and target.suffix.lower() == ".md"
        ):
            # A document authored by the agent must remain readable to the same
            # agent. User-created and legacy documents remain private by default.
            from remy.core.project_documents import set_agent_access

            set_agent_access(target.parent, target.name, True)
        manager.audit("write", target, grant, True, mode, tool)
        return json.dumps({"written": True, "path": str(target), "workspace_id": grant["workspace_id"], "size": len(content), "mode": mode})
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


def workspace_list(args: dict[str, Any], *, tool: str = "workspace_list") -> str:
    manager = get_workspace_manager()
    raw = str(args.get("path") or "workspace://data/")
    try:
        target, grant = manager.resolve(raw, "read", must_exist=True)
        if not target.is_dir():
            raise NotADirectoryError(str(target))
        entries = []
        if (
            grant.get("source") == "builtin"
            and target == _canonical(grant["root_path"])
        ):
            candidates = [
                target / name
                for name in sorted(_BUILTIN_AGENT_DIRS)
                if (target / name).exists()
            ]
        else:
            candidates = list(target.iterdir())
        for entry in sorted(candidates, key=lambda p: (not p.is_dir(), p.name.lower())):
            if entry.name in _SKIP_PARTS:
                continue
            if grant.get("source") == "builtin":
                root = _canonical(grant["root_path"])
                relative = entry.resolve().relative_to(root)
                if len(relative.parts) == 1 and relative.parts[0].lower() not in _BUILTIN_AGENT_DIRS:
                    continue
                if relative.parts and relative.parts[0].lower() == "documents" and entry.is_file():
                    from remy.core.project_documents import is_agent_accessible

                    if not is_agent_accessible(root / "documents", entry.name):
                        continue
            entries.append({"name": entry.name, "type": "dir" if entry.is_dir() else "file",
                            "size": entry.stat().st_size if entry.is_file() else None})
            if len(entries) >= 500:
                break
        manager.audit("list", target, grant, True, "directory listed", tool)
        return json.dumps({"path": str(target), "workspace_id": grant["workspace_id"], "entries": entries, "count": len(entries)})
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


def workspace_search(args: dict[str, Any], *, tool: str = "workspace_search") -> str:
    manager = get_workspace_manager()
    raw = str(args.get("path") or "workspace://data/")
    pattern = str(args.get("pattern") or "").strip()
    mode = str(args.get("mode") or "glob").lower()
    max_results = min(200, max(1, int(args.get("max_results") or 50)))
    try:
        if not pattern:
            raise ValueError("pattern is required")
        target, grant = manager.resolve(raw, "read", must_exist=True)
        if not target.is_dir():
            raise NotADirectoryError(str(target))
        results: list[dict[str, Any]] = []
        builtin_root = (
            _canonical(grant["root_path"])
            if grant.get("source") == "builtin"
            else None
        )

        def visible_to_agent(path: Path) -> bool:
            if builtin_root is None:
                return True
            relative = path.relative_to(builtin_root)
            if not relative.parts or relative.parts[0].lower() not in _BUILTIN_AGENT_DIRS:
                return False
            if relative.parts[0].lower() != "documents" or path.is_dir():
                return True
            from remy.core.project_documents import is_agent_accessible

            return is_agent_accessible(builtin_root / "documents", path.name)

        search_targets: list[tuple[Path, str]] = [(target, pattern)]
        if builtin_root is not None and target == builtin_root:
            search_targets = []
            normalized_pattern = pattern.replace("\\", "/")
            for allowed_name in sorted(_BUILTIN_AGENT_DIRS):
                allowed_root = builtin_root / allowed_name
                if not allowed_root.exists():
                    continue
                if mode == "grep":
                    search_targets.append((allowed_root, "*"))
                    continue
                prefix = f"{allowed_name}/"
                if normalized_pattern.startswith(prefix):
                    scoped_pattern = normalized_pattern[len(prefix):] or "*"
                elif "/" in normalized_pattern and not normalized_pattern.startswith("**/"):
                    continue
                else:
                    scoped_pattern = normalized_pattern
                search_targets.append((allowed_root, scoped_pattern))

        if mode == "glob":
            for search_root, scoped_pattern in search_targets:
                for path in sorted(search_root.glob(scoped_pattern)):
                    if len(results) >= max_results:
                        break
                    resolved = path.resolve()
                    if resolved != search_root and not resolved.is_relative_to(search_root):
                        continue
                    if any(part in _SKIP_PARTS for part in resolved.parts):
                        continue
                    if not visible_to_agent(resolved):
                        continue
                    results.append({"path": str(resolved), "type": "dir" if resolved.is_dir() else "file",
                                    "size": resolved.stat().st_size if resolved.is_file() else None})
                if len(results) >= max_results:
                    break
        elif mode == "grep":
            regex = re.compile(pattern, re.IGNORECASE)
            for search_root, _ in search_targets:
                for path in search_root.rglob("*"):
                    if len(results) >= max_results:
                        break
                    resolved = path.resolve()
                    if not resolved.is_file() or any(part in _SKIP_PARTS for part in resolved.parts):
                        continue
                    if resolved != search_root and not resolved.is_relative_to(search_root):
                        continue
                    if not visible_to_agent(resolved):
                        continue
                    if resolved.stat().st_size > 500_000:
                        continue
                    matches = []
                    for number, line in enumerate(resolved.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                        if regex.search(line):
                            matches.append({"line": number, "text": line.strip()[:200]})
                            if len(matches) >= 5:
                                break
                    if matches:
                        results.append({"path": str(resolved), "matches": matches})
                if len(results) >= max_results:
                    break
        else:
            raise ValueError("mode must be 'glob' or 'grep'")
        manager.audit("search", target, grant, True, f"{mode}:{pattern}", tool)
        return json.dumps({"mode": mode, "pattern": pattern, "search_dir": str(target),
                           "workspace_id": grant["workspace_id"], "results": results,
                           "count": len(results), "truncated": len(results) >= max_results}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


def workspace_shell(args: dict[str, Any], *, tool: str = "shell_exec") -> str:
    manager = get_workspace_manager()
    command = str(args.get("command") or "").strip()
    raw_wd = str(args.get("working_dir") or "").strip()
    timeout = min(120, max(1, int(args.get("timeout") or 30)))
    try:
        if not command:
            raise ValueError("command is required")
        if not raw_wd:
            raise ValueError("working_dir is required and must belong to a workspace with Execute permission")
        if _SHELL_BLOCKED_RE.search(command):
            raise WorkspaceAccessError("Command blocked because it matches a catastrophic or irreversible pattern.")
        working_dir, grant = manager.resolve(raw_wd, "execute", must_exist=True)
        if not working_dir.is_dir():
            raise NotADirectoryError(str(working_dir))

        def run() -> str:
            completed = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout, cwd=str(working_dir))
            manager.audit("execute", working_dir, grant, True, command[:300], tool)
            return json.dumps({"exit_code": completed.returncode, "stdout": (completed.stdout or "")[:50000],
                               "stderr": (completed.stderr or "")[:10000], "command": command,
                               "workspace_id": grant["workspace_id"],
                               "truncated_stdout": len(completed.stdout or "") > 50000,
                               "truncated_stderr": len(completed.stderr or "") > 10000})

        from remy.core.approval_queue import approval_queue
        description = (f"Run a local command in workspace '{grant['name']}'?\n\n"
                       f"Directory: {working_dir}\nCommand: {command}\n\n"
                       "Execute runs with the current Windows user privileges and is not an OS sandbox.")
        return approval_queue.request_approval_sync(description, run, tool_name=tool, tool_args=args)
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"Command timed out after {timeout}s", "command": command})
    except Exception as exc:
        return json.dumps({"error": str(exc), "command": command}, ensure_ascii=False)
