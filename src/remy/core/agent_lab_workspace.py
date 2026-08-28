"""Private builder snapshots and deterministic merge gates for Agent Lab."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from remy.core.agent_lab import AgentLabStore
from remy.core.agent_lab_executor import validate_python_source
from remy.core.file_utils import atomic_write


SNAPSHOT_ROOTS = {"inputs", "src", "tests", "artifacts"}
MERGE_ROOTS = {"src", "tests"}
MAX_MERGE_FILES = 100
MAX_RETAINED_SNAPSHOTS = 100
MAX_CLEANUP_SELECTION = 100


class WorkspaceMergeConflict(ValueError):
    """Raised after a rejected merge receipt has been durably recorded."""

    def __init__(self, receipt: dict[str, Any]):
        super().__init__("Private workspace merge rejected by conflict gate")
        self.receipt = receipt


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(root: Path, *, roots: set[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for root_name in sorted(roots):
        directory = (root / root_name).resolve()
        if not directory.is_relative_to(root) or not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError("Agent Lab private workspaces cannot contain symlinks")
            if not path.is_file():
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError("Agent Lab private workspace path escaped its boundary")
            relative = resolved.relative_to(root).as_posix()
            result[relative] = {"sha256": _sha256(resolved), "size": resolved.stat().st_size}
    return result


def _root_hash(manifest: dict[str, dict[str, Any]]) -> str:
    payload = [
        [path, value.get("sha256", ""), int(value.get("size") or 0)]
        for path, value in sorted(manifest.items())
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _directory_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return total
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root):
                total += path.stat().st_size
        except OSError:
            continue
    return total


def _relative_source_path(value: str) -> PurePosixPath:
    clean = str(value or "").strip().replace("\\", "/")
    relative = PurePosixPath(clean)
    if (
        not clean
        or relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[0] not in MERGE_ROOTS
    ):
        raise ValueError("Private builder files must stay inside src/ or tests/")
    return relative


def _claimed_local_imports(record: dict[str, Any]) -> set[str]:
    roots = set()
    for claim in record.get("file_claims", []):
        if claim.get("status") not in {"active", "merged"}:
            continue
        path = PurePosixPath(str(claim.get("path") or ""))
        if len(path.parts) >= 2 and path.parts[0] == "src" and path.suffix == ".py":
            roots.add(path.parts[1].removesuffix(".py"))
    return roots


def _workspace_local_imports(workspace: Path) -> set[str]:
    """Discover import roots that exist inside one private workspace snapshot."""
    source_root = (workspace / "src").resolve()
    if not source_root.is_dir() or not source_root.is_relative_to(workspace):
        return set()
    roots = {
        path.relative_to(source_root).parts[0].removesuffix(".py")
        for path in source_root.rglob("*.py")
        if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(source_root)
    }
    if roots:
        roots.add("src")
    return roots


class AgentLabWorkspaceManager:
    """Own run-local snapshots; no caller receives authority outside the run."""

    def __init__(self, store: AgentLabStore):
        self.store = store
        self._lock = threading.RLock()

    def _snapshot_dir(self, run_id: str, workspace_id: str) -> Path:
        if not workspace_id.startswith("ws-") or len(workspace_id) > 80:
            raise ValueError("Invalid Agent Lab private workspace id")
        root = self.store.private_workspaces_path(run_id)
        target = (root / workspace_id).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Private workspace escapes its run boundary")
        return target

    def _metadata_path(self, run_id: str, workspace_id: str) -> Path:
        return self._snapshot_dir(run_id, workspace_id) / "snapshot.json"

    def _load_metadata(self, run_id: str, workspace_id: str) -> dict[str, Any]:
        path = self._metadata_path(run_id, workspace_id)
        if not path.is_file():
            raise KeyError(workspace_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("Private workspace metadata is unavailable") from exc
        if not isinstance(value, dict) or value.get("run_id") != run_id:
            raise ValueError("Private workspace metadata does not belong to this run")
        return value

    def _save_metadata(
        self, run_id: str, workspace_id: str, metadata: dict[str, Any]
    ) -> None:
        atomic_write(
            self._metadata_path(run_id, workspace_id),
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        )

    def retention_status(self, run_id: str) -> dict[str, Any]:
        """Return bounded disk evidence without exposing private source content."""
        record = self.store.require(run_id)
        root = self.store.private_workspaces_path(run_id)
        active_claim_workspaces = {
            str(claim.get("workspace_id") or "")
            for claim in record.get("file_claims", [])
            if claim.get("status") == "active"
        }
        items = []
        with self._lock, self.store.workspace_lock:
            for path in sorted(root.iterdir(), key=lambda item: item.name):
                if not path.is_dir() or path.is_symlink():
                    continue
                workspace_id = path.name
                size = _directory_bytes(path)
                try:
                    metadata = self._load_metadata(run_id, workspace_id)
                    status = str(metadata.get("status") or "unknown")
                    created_at = str(metadata.get("created_at") or "")
                    node_id = str(metadata.get("node_id") or "")
                except (KeyError, ValueError):
                    status, created_at, node_id = "unsafe_metadata", "", ""
                active_claim = workspace_id in active_claim_workspaces
                eligible = status == "merged" and not active_claim
                items.append({
                    "workspace_id": workspace_id,
                    "node_id": node_id,
                    "status": status,
                    "size_bytes": size,
                    "created_at": created_at,
                    "active_claim": active_claim,
                    "cleanup_eligible": eligible,
                    "conflict_cleanup_eligible": status == "conflict" and not active_claim,
                })
        used = sum(int(item["size_bytes"]) for item in items)
        budget = int(record["policy"]["max_workspace_bytes"]) * max(
            1, int(record["policy"].get("max_agents") or 1)
        )
        return {
            "run_id": run_id,
            "snapshot_count": len(items),
            "used_bytes": used,
            "budget_bytes": budget,
            "usage_ratio": round(used / budget, 6) if budget else 0,
            "retention_limit": MAX_RETAINED_SNAPSHOTS,
            "cleanup_eligible_count": sum(bool(item["cleanup_eligible"]) for item in items),
            "conflict_count": sum(item["status"] == "conflict" for item in items),
            "protected_count": sum(
                item["status"] in {"open", "unsafe_metadata"} or item["active_claim"]
                for item in items
            ),
            "items": sorted(items, key=lambda item: (item["created_at"], item["workspace_id"]), reverse=True),
        }

    def cleanup_snapshots(
        self,
        run_id: str,
        *,
        workspace_ids: list[str] | None = None,
        include_conflicts: bool = False,
    ) -> dict[str, Any]:
        """Delete only closed run-owned snapshots after an all-target preflight."""
        record = self.store.require(run_id)
        if record.get("status") == "running":
            raise ValueError("Snapshot cleanup is locked while Agent Lab is running")
        requested = list(dict.fromkeys(str(value or "") for value in (workspace_ids or [])))
        if len(requested) > MAX_CLEANUP_SELECTION:
            raise ValueError("Snapshot cleanup selection exceeds its bounded limit")
        root = self.store.private_workspaces_path(run_id)
        active_claim_workspaces = {
            str(claim.get("workspace_id") or "")
            for claim in record.get("file_claims", [])
            if claim.get("status") == "active"
        }
        targets: list[tuple[str, Path, dict[str, Any], int]] = []
        with self._lock, self.store.workspace_lock:
            available = {
                path.name: path
                for path in root.iterdir()
                if path.is_dir() and not path.is_symlink()
            }
            candidate_ids = requested or sorted(available)
            for workspace_id in candidate_ids:
                # Re-run the canonical id/boundary validator even for discovered names.
                target = self._snapshot_dir(run_id, workspace_id)
                if workspace_id not in available or not target.exists():
                    if requested:
                        raise ValueError(f"Private snapshot is unavailable: {workspace_id}")
                    continue
                metadata = self._load_metadata(run_id, workspace_id)
                status = str(metadata.get("status") or "")
                if workspace_id in active_claim_workspaces or status == "open":
                    if requested:
                        raise ValueError(f"Active private snapshot cannot be cleaned: {workspace_id}")
                    continue
                allowed = status == "merged" or (include_conflicts and status == "conflict")
                if not allowed:
                    if requested:
                        if status == "conflict":
                            raise ValueError("Conflict snapshot cleanup requires explicit confirmation")
                        raise ValueError(f"Private snapshot is protected: {workspace_id}")
                    continue
                if target.is_symlink() or not target.resolve().is_relative_to(root):
                    raise ValueError("Private snapshot cleanup target escaped its run boundary")
                targets.append((workspace_id, target, metadata, _directory_bytes(target)))

            cleanup_id = f"cleanup-{uuid.uuid4().hex[:12]}"
            removed, errors = [], []
            recovered = 0
            for workspace_id, target, _metadata, size in targets:
                try:
                    shutil.rmtree(target)
                    removed.append(workspace_id)
                    recovered += size
                except OSError as exc:
                    errors.append({"workspace_id": workspace_id, "error": str(exc)[:500]})
            receipt = {
                "cleanup_id": cleanup_id,
                "status": "partial" if errors else "completed",
                "requested_workspace_ids": requested,
                "include_conflicts": bool(include_conflicts),
                "removed_workspace_ids": removed,
                "removed_count": len(removed),
                "recovered_bytes": recovered,
                "errors": errors,
                "created_at": _now(),
            }
        self.store.append_snapshot_cleanup_receipt(run_id, receipt)
        return {"receipt": receipt, "retention": self.retention_status(run_id)}

    def create_snapshot(self, run_id: str, *, node_id: str = "build") -> dict[str, Any]:
        record = self.store.require(run_id)
        if record.get("status") not in {"prepared", "paused", "running"}:
            raise ValueError("Private workspace requires a prepared Agent Lab run")
        node = next(
            (
                item
                for item in (record.get("workflow_plan") or {}).get("nodes", [])
                if item.get("node_id") == node_id
            ),
            None,
        )
        if not node or node.get("workspace_mode") != "private_snapshot":
            raise ValueError("Workflow node is not authorized for a private workspace")
        workspace_id = f"ws-{uuid.uuid4().hex[:12]}"
        snapshots_root = self.store.private_workspaces_path(run_id)
        snapshot_dir = self._snapshot_dir(run_id, workspace_id)
        private_root = snapshot_dir / "workspace"
        canonical = self.store.workspace_path(run_id)
        with self._lock, self.store.workspace_lock:
            existing = [path for path in snapshots_root.iterdir() if path.is_dir()]
            if len(existing) >= MAX_RETAINED_SNAPSHOTS:
                raise ValueError("Agent Lab private workspace retention limit reached")
            existing_bytes = _directory_bytes(snapshots_root)
            aggregate_budget = int(record["policy"]["max_workspace_bytes"]) * max(
                1, int(record["policy"].get("max_agents") or 1)
            )
            snapshot_dir.mkdir(parents=True, exist_ok=False)
            try:
                total = 0
                for root_name in sorted(SNAPSHOT_ROOTS):
                    source_root = canonical / root_name
                    target_root = private_root / root_name
                    target_root.mkdir(parents=True, exist_ok=True)
                    if not source_root.exists():
                        continue
                    for source in sorted(source_root.rglob("*")):
                        if source.is_symlink():
                            raise ValueError("Agent Lab canonical workspace contains a symlink")
                        if not source.is_file():
                            continue
                        resolved = source.resolve()
                        if not resolved.is_relative_to(canonical):
                            raise ValueError("Canonical workspace path escaped its boundary")
                        relative = resolved.relative_to(canonical)
                        total += resolved.stat().st_size
                        if total > int(record["policy"]["max_workspace_bytes"]):
                            raise ValueError("Private workspace snapshot exceeds the size policy")
                        if existing_bytes + total > aggregate_budget:
                            raise ValueError("Agent Lab private workspaces exceed their aggregate disk budget")
                        destination = (private_root / relative).resolve()
                        if not destination.is_relative_to(private_root):
                            raise ValueError("Private workspace copy escaped its boundary")
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(resolved, destination)
                baseline = _manifest(canonical, roots=MERGE_ROOTS)
                metadata = {
                    "workspace_id": workspace_id,
                    "run_id": run_id,
                    "node_id": node_id,
                    "status": "open",
                    "relative_path": snapshot_dir.relative_to(self.store._dir(run_id)).as_posix(),
                    "baseline": baseline,
                    "baseline_root_hash": _root_hash(baseline),
                    "staged_paths": [],
                    "created_at": _now(),
                    "updated_at": _now(),
                    "merged_at": "",
                }
                self._save_metadata(run_id, workspace_id, metadata)
            except Exception:
                shutil.rmtree(snapshot_dir, ignore_errors=True)
                raise

        summary = {key: value for key, value in metadata.items() if key != "baseline"}
        self.store.append_workspace_branch(run_id, summary)
        return summary

    def stage_file(
        self,
        run_id: str,
        workspace_id: str,
        *,
        path: str,
        content: str,
        allowed_local_imports: set[str] | None = None,
    ) -> dict[str, Any]:
        record = self.store.require(run_id)
        relative = _relative_source_path(path)
        raw = str(content or "").encode("utf-8")
        if len(raw) > int(record["policy"]["max_source_bytes"]):
            raise ValueError("Private workspace source exceeds the configured size limit")
        with self._lock:
            metadata = self._load_metadata(run_id, workspace_id)
            if metadata.get("status") != "open":
                raise ValueError("Private workspace is already closed")
            private_root = (self._snapshot_dir(run_id, workspace_id) / "workspace").resolve()
            if relative.suffix.lower() == ".py":
                validate_python_source(
                    content,
                    allowed_local_imports=(
                        _claimed_local_imports(record)
                        | _workspace_local_imports(private_root)
                        | set(allowed_local_imports or set())
                    ),
                )
            target = (private_root / Path(*relative.parts)).resolve()
            if not target.is_relative_to(private_root):
                raise ValueError("Private builder path escaped its workspace")
            current = _manifest(private_root, roots=SNAPSHOT_ROOTS)
            previous = int((current.get(relative.as_posix()) or {}).get("size") or 0)
            projected = sum(int(item.get("size") or 0) for item in current.values()) - previous + len(raw)
            if projected > int(record["policy"]["max_workspace_bytes"]):
                raise ValueError("Private workspace exceeds the configured size limit")
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(target, content)
            staged = metadata.setdefault("staged_paths", [])
            if relative.as_posix() not in staged:
                staged.append(relative.as_posix())
            metadata["updated_at"] = _now()
            self._save_metadata(run_id, workspace_id, metadata)
        return {
            "workspace_id": workspace_id,
            "path": relative.as_posix(),
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "written_at": _now(),
        }

    def merge(self, run_id: str, workspace_id: str) -> dict[str, Any]:
        """Merge explicit staged files only when their baseline is still current."""
        record = self.store.require(run_id)
        canonical = self.store.workspace_path(run_id)
        with self._lock, self.store.workspace_lock:
            metadata = self._load_metadata(run_id, workspace_id)
            if metadata.get("status") != "open":
                raise ValueError("Private workspace is already closed")
            private_root = (self._snapshot_dir(run_id, workspace_id) / "workspace").resolve()
            staged_paths = list(dict.fromkeys(metadata.get("staged_paths") or []))
            if len(staged_paths) > MAX_MERGE_FILES:
                raise ValueError("Private workspace exceeds the merge file limit")
            baseline = dict(metadata.get("baseline") or {})
            current = _manifest(canonical, roots=MERGE_ROOTS)
            candidate = _manifest(private_root, roots=MERGE_ROOTS)
            changes = []
            conflicts = []
            for text in staged_paths:
                relative = _relative_source_path(text)
                path = relative.as_posix()
                private_file = (private_root / Path(*relative.parts)).resolve()
                if (
                    not private_file.is_relative_to(private_root)
                    or not private_file.is_file()
                    or private_file.is_symlink()
                ):
                    conflicts.append({"path": path, "reason": "candidate_missing_or_unsafe"})
                    continue
                if relative.suffix.lower() == ".py":
                    validate_python_source(
                        private_file.read_text(encoding="utf-8"),
                        allowed_local_imports=(
                            _claimed_local_imports(record)
                            | _workspace_local_imports(private_root)
                        ),
                    )
                baseline_hash = str((baseline.get(path) or {}).get("sha256") or "")
                current_hash = str((current.get(path) or {}).get("sha256") or "")
                candidate_hash = str((candidate.get(path) or {}).get("sha256") or "")
                if current_hash != baseline_hash and candidate_hash != current_hash:
                    conflicts.append({
                        "path": path,
                        "reason": "canonical_changed_since_snapshot",
                        "baseline_sha256": baseline_hash,
                        "current_sha256": current_hash,
                        "candidate_sha256": candidate_hash,
                    })
                    continue
                if candidate_hash == current_hash:
                    continue
                changes.append({
                    "path": path,
                    "action": "modify" if current_hash else "add",
                    "baseline_sha256": baseline_hash,
                    "current_sha256": current_hash,
                    "candidate_sha256": candidate_hash,
                    "size": private_file.stat().st_size,
                })

            projected = sum(int(value.get("size") or 0) for value in current.values())
            for change in changes:
                projected -= int((current.get(change["path"]) or {}).get("size") or 0)
                projected += int(change["size"])
            if projected > int(record["policy"]["max_workspace_bytes"]):
                conflicts.append({"path": "", "reason": "workspace_size_policy"})

            receipt = {
                "merge_id": f"merge-{uuid.uuid4().hex[:12]}",
                "workspace_id": workspace_id,
                "node_id": str(metadata.get("node_id") or ""),
                "status": "conflict" if conflicts else "no_changes" if not changes else "merged",
                "baseline_root_hash": str(metadata.get("baseline_root_hash") or ""),
                "canonical_before_hash": _root_hash(current),
                "candidate_root_hash": _root_hash(candidate),
                "canonical_after_hash": "",
                "changes": changes,
                "conflicts": conflicts,
                "applied_files": [],
                "created_at": _now(),
            }
            if conflicts:
                metadata["status"] = "conflict"
                metadata["updated_at"] = _now()
                self._save_metadata(run_id, workspace_id, metadata)
                self.store.append_merge_receipt(run_id, receipt)
                self.store.record_blocker(
                    run_id,
                    node_id=str(metadata.get("node_id") or "build"),
                    kind="merge_conflict",
                    message=f"Private workspace merge rejected with {len(conflicts)} conflict(s)",
                )
                raise WorkspaceMergeConflict(receipt)

            backups: dict[str, bytes | None] = {}
            try:
                for change in changes:
                    relative = _relative_source_path(change["path"])
                    source = (private_root / Path(*relative.parts)).resolve()
                    target = (canonical / Path(*relative.parts)).resolve()
                    if not source.is_relative_to(private_root) or not target.is_relative_to(canonical):
                        raise ValueError("Merge target escaped Agent Lab workspace")
                    backups[change["path"]] = target.read_bytes() if target.exists() else None
                    target.parent.mkdir(parents=True, exist_ok=True)
                    atomic_write(target, source.read_text(encoding="utf-8"))
                    receipt["applied_files"].append(change["path"])
            except Exception:
                for path, backup in backups.items():
                    target = canonical / Path(*PurePosixPath(path).parts)
                    if backup is None:
                        if target.exists():
                            target.unlink()
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(backup)
                raise

            after = _manifest(canonical, roots=MERGE_ROOTS)
            receipt["canonical_after_hash"] = _root_hash(after)
            metadata["status"] = "merged"
            metadata["merged_at"] = _now()
            metadata["updated_at"] = metadata["merged_at"]
            self._save_metadata(run_id, workspace_id, metadata)
            self.store.append_merge_receipt(run_id, receipt)
            return receipt


_managers: dict[int, AgentLabWorkspaceManager] = {}
_managers_lock = threading.RLock()


def get_agent_lab_workspace_manager(store: AgentLabStore) -> AgentLabWorkspaceManager:
    key = id(store)
    with _managers_lock:
        manager = _managers.get(key)
        if manager is None:
            manager = AgentLabWorkspaceManager(store)
            _managers[key] = manager
        return manager
