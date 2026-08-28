"""Project-scoped foundation for Remy's autonomous Agent Lab.

The Agent Lab is deliberately separate from the operator-authored Experiment
Lab. A user supplies a goal and hard boundaries; Remy owns the derived plan
and team. Generated Python is executed only through the separate bounded
executor contract, never imported into the Remy process.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from remy.core.agent_lab_workflow import (
    MAX_PROGRESS_SNAPSHOTS,
    build_default_lab_workflow_plan,
    build_progress_snapshot,
    create_task_ledger,
    legacy_plan_projection,
)
from remy.core.file_utils import atomic_write


TERMINAL_STATES = {"completed", "failed", "cancelled"}
ALLOWED_TRANSITIONS = {
    "draft": {"prepared", "cancelled"},
    "prepared": {"running", "cancelled"},
    "running": {"paused", "completed", "failed", "cancelled"},
    "paused": {"running", "cancelled"},
    "completed": set(),
    "failed": set(),
    "cancelled": set(),
}
DEFAULT_POLICY = {
    "network_access": False,
    "hardware_access": False,
    "external_side_effects": False,
    "self_modification": False,
    "workspace_write": True,
    "max_agents": 4,
    "max_models": 4,
    "time_budget_seconds": 900,
    "max_artifact_bytes": 50 * 1024 * 1024,
    "max_source_bytes": 1_000_000,
    "max_output_bytes": 100_000,
    "max_memory_mb": 256,
    "max_workspace_bytes": 64 * 1024 * 1024,
}
_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,80}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bounded_policy(value: dict[str, Any] | None = None) -> dict[str, Any]:
    supplied = dict(value or {})
    # Stage one is intentionally local-only. These switches cannot be enabled
    # merely by crafting an API request.
    policy = dict(DEFAULT_POLICY)
    policy["max_agents"] = max(1, min(int(supplied.get("max_agents", 4)), 10))
    policy["max_models"] = max(1, min(int(supplied.get("max_models", 4)), 10))
    policy["time_budget_seconds"] = max(
        60, min(int(supplied.get("time_budget_seconds", 900)), 86_400)
    )
    return policy


def _derive_team(goal: str, max_agents: int) -> list[dict[str, str]]:
    text = goal.lower()
    roles: list[tuple[str, str, str]] = [
        ("coordinator", "Coordinator", "Own the plan, delegation, checkpoints, and final handoff."),
        ("builder", "Builder", "Create the bounded implementation inside the isolated workspace."),
        ("verifier", "Verifier", "Test claims and artifacts independently before completion."),
    ]
    if any(word in text for word in ("research", "аналіз", "дослід", "пошук", "compare")):
        roles.insert(1, ("researcher", "Researcher", "Collect and structure evidence needed by the builders."))
    if any(word in text for word in ("site", "website", "document", "presentation", "сайт", "документ", "презентац")):
        roles.insert(-1, ("artifact_designer", "Artifact designer", "Turn verified work into a usable, reviewable artifact."))
    if any(word in text for word in ("security", "drone", "hardware", "безпек", "дрон", "систем")):
        roles.insert(-1, ("safety_reviewer", "Safety reviewer", "Challenge permissions, failure modes, and unsafe assumptions."))
    return [
        {"agent_id": f"agent-{index + 1}", "role": role, "name": name, "responsibility": responsibility}
        for index, (role, name, responsibility) in enumerate(roles[:max_agents])
    ]


class AgentLabStore:
    """Durable store whose run directories never escape one project root."""

    def __init__(self, data_dir: str | Path, *, owner_project_id: str = "", brain_id: str = ""):
        self.project_root = Path(data_dir).resolve()
        self.root = (self.project_root / "agent-lab").resolve()
        if not self.root.is_relative_to(self.project_root):
            raise RuntimeError("Agent Lab root escapes the project boundary")
        self.root.mkdir(parents=True, exist_ok=True)
        self.owner_project_id = str(owner_project_id or "")
        self.brain_id = str(brain_id or "")
        self._lock = threading.RLock()
        self.workspace_lock = threading.RLock()

    def _dir(self, run_id: str) -> Path:
        if not _SAFE_ID_RE.fullmatch(str(run_id or "")):
            raise ValueError("Invalid Agent Lab run id")
        target = (self.root / run_id).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError("Invalid Agent Lab run path")
        return target

    def _path(self, run_id: str) -> Path:
        return self._dir(run_id) / "run.json"

    def _save(self, record: dict[str, Any]) -> dict[str, Any]:
        record["updated_at"] = _now()
        path = self._path(str(record["run_id"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        return record

    def create(self, *, goal: str, title: str = "", policy: dict[str, Any] | None = None) -> dict[str, Any]:
        clean_goal = str(goal or "").strip()
        if not clean_goal:
            raise ValueError("Agent Lab requires a goal")
        run_id = f"lab-{uuid.uuid4().hex[:12]}"
        now = _now()
        record = {
            "version": 2,
            "run_id": run_id,
            "owner_project_id": self.owner_project_id,
            "brain_id": self.brain_id,
            "title": str(title or "").strip()[:200] or clean_goal[:80],
            "goal": clean_goal[:20_000],
            "messages": [
                {
                    "at": now,
                    "role": "user",
                    "kind": "goal",
                    "content": clean_goal[:20_000],
                }
            ],
            "status": "draft",
            "phase": "intake",
            "policy": _bounded_policy(policy),
            "team": [],
            "plan": [],
            "workflow_plan": {},
            "task_ledger": {},
            "progress_ledger": [],
            "delegation": {},
            "artifacts": [],
            "verification": [],
            "verifier": {},
            "proof_pack": {},
            "builder_fanout": {},
            "file_claims": [],
            "executions": [],
            "workspace": {"relative_path": f"agent-lab/{run_id}/workspace", "ready": False},
            "workspace_branches": [],
            "merge_receipts": [],
            "snapshot_cleanup_receipts": [],
            "events": [{"at": now, "type": "created", "message": "Agent-owned lab draft created"}],
            "trajectory_session_id": "",
            "trajectory_run_event_id": "",
            "trajectory_result_event_id": "",
            "created_at": now,
            "updated_at": now,
            "started_at": "",
            "completed_at": "",
            "error": "",
        }
        with self._lock:
            self._dir(run_id).mkdir(parents=True, exist_ok=False)
            return self._save(record)

    def get(self, run_id: str) -> dict[str, Any] | None:
        path = self._path(run_id)
        if not path.exists():
            return None
        with self._lock:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
        return value if isinstance(value, dict) else None

    def require(self, run_id: str) -> dict[str, Any]:
        record = self.get(run_id)
        if not record:
            raise KeyError(run_id)
        return record

    def list(self) -> list[dict[str, Any]]:
        records = []
        for path in self.root.glob("lab-*/run.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict) and not value.get("archived_at"):
                    records.append(value)
            except (OSError, ValueError):
                continue
        return sorted(records, key=lambda item: str(item.get("updated_at") or ""), reverse=True)

    def archive(self, run_id: str) -> dict[str, Any]:
        """Hide a closed run from normal history without destroying its evidence."""
        record = self.require(run_id)
        if record.get("status") == "running":
            raise ValueError("A running Agent Lab task must be stopped before removal")

        def apply(item: dict[str, Any]) -> None:
            if item.get("archived_at"):
                raise ValueError("Agent Lab task is already removed from history")
            now = _now()
            item["archived_at"] = now
            item.setdefault("events", []).append({
                "at": now,
                "type": "archived",
                "message": "Task removed from laboratory history; evidence preserved locally",
            })

        return self.mutate(run_id, apply)

    def mutate(self, run_id: str, update) -> dict[str, Any]:
        with self._lock:
            record = self.require(run_id)
            update(record)
            return self._save(record)

    def transition(self, run_id: str, target: str, *, message: str = "") -> dict[str, Any]:
        normalized = str(target or "").strip().lower()

        def apply(record: dict[str, Any]) -> None:
            current = str(record.get("status") or "draft")
            if normalized not in ALLOWED_TRANSITIONS.get(current, set()):
                raise ValueError(f"Agent Lab cannot transition from {current} to {normalized}")
            record["status"] = normalized
            if normalized == "running" and not record.get("started_at"):
                record["started_at"] = _now()
            if normalized in TERMINAL_STATES:
                record["completed_at"] = _now()
            record.setdefault("events", []).append({
                "at": _now(), "type": normalized, "message": message or f"Run moved to {normalized}"
            })

        return self.mutate(run_id, apply)

    def prepare(self, run_id: str) -> dict[str, Any]:
        record = self.require(run_id)
        if record.get("status") != "draft":
            raise ValueError("Only a draft Agent Lab run can be prepared")
        team = _derive_team(record["goal"], int(record["policy"]["max_agents"]))
        workflow_plan = build_default_lab_workflow_plan(
            record["goal"], team, record["policy"]
        )
        plan = legacy_plan_projection(workflow_plan)
        task_ledger = create_task_ledger(workflow_plan)
        progress_ledger = [
            build_progress_snapshot(
                task_ledger,
                sequence=1,
                phase="ready_for_execution",
                progress_made=True,
                reason="Validated LabWorkflowPlan v2 prepared",
            )
        ]
        workspace = self._dir(run_id) / "workspace"
        for name in ("inputs", "work", "src", "tests", "artifacts", "logs"):
            (workspace / name).mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": run_id,
            "owner_project_id": self.owner_project_id,
            "goal": record["goal"],
            "policy": record["policy"],
            "team": team,
            "plan": plan,
            "workflow_plan": workflow_plan,
            "task_ledger": task_ledger,
            "progress_ledger": progress_ledger,
            "prepared_at": _now(),
        }
        atomic_write(workspace / "lab-manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

        def apply(item: dict[str, Any]) -> None:
            item.update({
                "version": 2,
                "status": "prepared",
                "phase": "ready_for_execution",
                "team": team,
                "plan": plan,
                "workflow_plan": workflow_plan,
                "task_ledger": task_ledger,
                "progress_ledger": progress_ledger,
            })
            item["workspace"]["ready"] = True
            item["events"].append({"at": _now(), "type": "prepared", "message": f"Agent selected {len(team)} roles and {len(plan)} steps"})

        return self.mutate(run_id, apply)

    def workspace_path(self, run_id: str) -> Path:
        """Return the canonical, run-owned workspace boundary."""
        workspace = (self._dir(run_id) / "workspace").resolve()
        if not workspace.is_relative_to(self._dir(run_id)):
            raise RuntimeError("Agent Lab workspace escapes the run boundary")
        return workspace

    def private_workspaces_path(self, run_id: str) -> Path:
        """Return the run-owned root for disposable builder snapshots."""
        root = (self._dir(run_id) / "private-workspaces").resolve()
        if not root.is_relative_to(self._dir(run_id)):
            raise RuntimeError("Agent Lab private workspace root escapes the run boundary")
        root.mkdir(parents=True, exist_ok=True)
        return root

    def append_execution(self, run_id: str, receipt: dict[str, Any]) -> dict[str, Any]:
        def apply(item: dict[str, Any]) -> None:
            item.setdefault("executions", []).append(dict(receipt))
            item["executions"] = item["executions"][-100:]
            item.setdefault("events", []).append({
                "at": _now(),
                "type": "execution",
                "message": f"{receipt.get('entrypoint', 'program')} finished with {receipt.get('status', 'unknown')}",
            })

        return self.mutate(run_id, apply)

    def append_verification(self, run_id: str, receipt: dict[str, Any]) -> dict[str, Any]:
        def apply(item: dict[str, Any]) -> None:
            item.setdefault("verification", []).append(dict(receipt))
            item["verification"] = item["verification"][-100:]
            item.setdefault("events", []).append({
                "at": _now(),
                "type": "verification",
                "message": f"Verification {receipt.get('world_fact', 'inconclusive')}",
            })

        return self.mutate(run_id, apply)

    def set_artifacts(self, run_id: str, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
        def apply(item: dict[str, Any]) -> None:
            item["artifacts"] = list(artifacts)
            ledger = item.get("task_ledger")
            if isinstance(ledger, dict):
                ledger["artifact_hashes"] = {
                    str(artifact.get("path") or artifact.get("artifact_id") or ""): str(
                        artifact.get("sha256") or ""
                    )
                    for artifact in artifacts
                    if artifact.get("path") or artifact.get("artifact_id")
                }
                ledger["updated_at"] = _now()

        return self.mutate(run_id, apply)

    def append_workspace_branch(
        self, run_id: str, branch: dict[str, Any]
    ) -> dict[str, Any]:
        def apply(item: dict[str, Any]) -> None:
            branches = item.setdefault("workspace_branches", [])
            branches.append(dict(branch))
            item["workspace_branches"] = branches[-100:]
            item.setdefault("events", []).append({
                "at": _now(),
                "type": "workspace_snapshot",
                "message": f"Private workspace {branch.get('workspace_id', '')} created for {branch.get('node_id', 'build')}",
            })

        return self.mutate(run_id, apply)

    def append_merge_receipt(
        self, run_id: str, receipt: dict[str, Any]
    ) -> dict[str, Any]:
        def apply(item: dict[str, Any]) -> None:
            receipts = item.setdefault("merge_receipts", [])
            receipts.append(dict(receipt))
            item["merge_receipts"] = receipts[-100:]
            workspace_id = str(receipt.get("workspace_id") or "")
            for branch in item.setdefault("workspace_branches", []):
                if branch.get("workspace_id") == workspace_id:
                    branch["status"] = str(receipt.get("status") or branch.get("status") or "")
                    branch["merge_id"] = str(receipt.get("merge_id") or "")
                    branch["merged_at"] = str(receipt.get("created_at") or "")
            item.setdefault("events", []).append({
                "at": _now(),
                "type": "workspace_merge",
                "message": (
                    f"Merge {receipt.get('merge_id', '')} {receipt.get('status', 'unknown')} "
                    f"with {len(receipt.get('applied_files') or [])} applied file(s)"
                ),
            })

        return self.mutate(run_id, apply)

    def append_snapshot_cleanup_receipt(
        self, run_id: str, receipt: dict[str, Any]
    ) -> dict[str, Any]:
        """Persist snapshot deletion evidence without removing historical branches."""
        def apply(item: dict[str, Any]) -> None:
            receipts = item.setdefault("snapshot_cleanup_receipts", [])
            receipts.append(dict(receipt))
            item["snapshot_cleanup_receipts"] = receipts[-200:]
            removed = set(receipt.get("removed_workspace_ids") or [])
            for branch in item.setdefault("workspace_branches", []):
                if branch.get("workspace_id") in removed:
                    branch["retained"] = False
                    branch["cleaned_at"] = str(receipt.get("created_at") or "")
                    branch["cleanup_id"] = str(receipt.get("cleanup_id") or "")
            item.setdefault("events", []).append({
                "at": _now(),
                "type": "snapshot_cleanup",
                "message": (
                    f"Removed {len(removed)} private snapshot(s) and recovered "
                    f"{int(receipt.get('recovered_bytes') or 0)} byte(s)"
                ),
            })

        return self.mutate(run_id, apply)

    def register_file_claims(
        self, run_id: str, assignments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Atomically reserve exact source paths before parallel builders run."""
        created: list[dict[str, Any]] = []

        def apply(item: dict[str, Any]) -> None:
            existing = item.setdefault("file_claims", [])
            active = {
                str(claim.get("path") or ""): str(claim.get("builder_id") or "")
                for claim in existing
                if claim.get("status") == "active"
            }
            pending: dict[str, str] = {}
            for assignment in assignments:
                builder_id = str(assignment.get("builder_id") or "")
                workspace_id = str(assignment.get("workspace_id") or "")
                for raw_path in assignment.get("file_claims") or []:
                    path = str(raw_path or "").strip().replace("\\", "/")
                    owner = active.get(path) or pending.get(path)
                    if owner:
                        raise ValueError(
                            f"Active Agent Lab file claim overlap: {path} is owned by {owner}"
                        )
                    if not builder_id or not workspace_id or not path.startswith("src/"):
                        raise ValueError("Invalid Agent Lab file claim")
                    pending[path] = builder_id
            now = _now()
            for assignment in assignments:
                for raw_path in assignment.get("file_claims") or []:
                    claim = {
                        "claim_id": f"claim-{uuid.uuid4().hex[:12]}",
                        "builder_id": str(assignment.get("builder_id") or ""),
                        "model": str(assignment.get("model") or ""),
                        "workspace_id": str(assignment.get("workspace_id") or ""),
                        "path": str(raw_path).replace("\\", "/"),
                        "status": "active",
                        "merge_id": "",
                        "error": "",
                        "created_at": now,
                        "resolved_at": "",
                    }
                    existing.append(claim)
                    created.append(dict(claim))
            item["file_claims"] = existing[-500:]
            item.setdefault("events", []).append({
                "at": now,
                "type": "file_claims_registered",
                "message": f"Reserved {len(created)} file claim(s) for {len(assignments)} builder(s)",
            })

        self.mutate(run_id, apply)
        return created

    def resolve_file_claims(
        self,
        run_id: str,
        *,
        builder_id: str,
        status: str,
        merge_id: str = "",
        error: str = "",
    ) -> dict[str, Any]:
        allowed = {"merged", "conflict", "failed", "cancelled"}
        if status not in allowed:
            raise ValueError("Invalid Agent Lab file claim resolution")

        def apply(item: dict[str, Any]) -> None:
            changed = 0
            now = _now()
            for claim in item.setdefault("file_claims", []):
                if claim.get("builder_id") == builder_id and claim.get("status") == "active":
                    claim.update({
                        "status": status,
                        "merge_id": str(merge_id or ""),
                        "error": str(error or "")[:2_000],
                        "resolved_at": now,
                    })
                    changed += 1
            item.setdefault("events", []).append({
                "at": now,
                "type": "file_claims_resolved",
                "message": f"Resolved {changed} claim(s) for {builder_id} as {status}",
            })

        return self.mutate(run_id, apply)

    def ensure_workflow_state(self, run_id: str) -> dict[str, Any]:
        """Lazily upgrade a prepared v1 run without discarding its checkpoint."""
        current = self.require(run_id)
        if current.get("workflow_plan") and current.get("task_ledger"):
            return current
        if current.get("status") == "draft":
            return current
        team = list(current.get("team") or []) or _derive_team(
            str(current.get("goal") or ""), int((current.get("policy") or {}).get("max_agents") or 1)
        )
        workflow_plan = build_default_lab_workflow_plan(
            str(current.get("goal") or ""), team, current.get("policy") or DEFAULT_POLICY
        )
        legacy_by_id = {
            str(step.get("step_id") or ""): step for step in current.get("plan", [])
        }
        for node in workflow_plan["nodes"]:
            legacy = legacy_by_id.get(node["node_id"], {})
            if legacy.get("status"):
                status = str(legacy["status"])
                node["status"] = "in_progress" if status == "running" else (
                    status if status in {
                        "pending", "in_progress", "completed", "needs_repair", "blocked", "skipped"
                    } else "pending"
                )
            if legacy.get("owner"):
                node["owner"] = str(legacy["owner"])
        task_ledger = create_task_ledger(workflow_plan)
        progress_ledger = [
            build_progress_snapshot(
                task_ledger,
                sequence=1,
                phase=str(current.get("phase") or "migrated"),
                progress_made=True,
                reason="Existing Agent Lab checkpoint upgraded to LabWorkflowPlan v2",
            )
        ]

        def apply(item: dict[str, Any]) -> None:
            item["version"] = 2
            item["team"] = team
            item["workflow_plan"] = workflow_plan
            item["plan"] = legacy_plan_projection(workflow_plan)
            item["task_ledger"] = task_ledger
            item["progress_ledger"] = progress_ledger
            item.setdefault("workspace_branches", [])
            item.setdefault("merge_receipts", [])
            item.setdefault("snapshot_cleanup_receipts", [])
            item.setdefault("builder_fanout", {})
            item.setdefault("file_claims", [])
            item.setdefault("events", []).append({
                "at": _now(),
                "type": "workflow_migrated",
                "message": "Checkpoint upgraded to validated LabWorkflowPlan v2",
            })

        upgraded = self.mutate(run_id, apply)
        manifest = self.workspace_path(run_id) / "lab-manifest.json"
        if manifest.exists():
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                payload = {"run_id": run_id}
            payload.update({
                "workflow_plan": workflow_plan,
                "task_ledger": task_ledger,
                "progress_ledger": progress_ledger,
                "upgraded_at": _now(),
            })
            atomic_write(manifest, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return upgraded

    def update_node_statuses(
        self,
        run_id: str,
        statuses: dict[str, str],
        *,
        reason: str = "",
        replan_reason: str = "",
    ) -> dict[str, Any]:
        """Atomically update both plan projections and append a progress receipt."""
        from remy.core.agent_lab_workflow import ALLOWED_NODE_STATUSES

        normalized = {str(key): str(value) for key, value in statuses.items()}
        invalid = set(normalized.values()) - ALLOWED_NODE_STATUSES
        if invalid:
            raise ValueError("Unsupported Agent Lab node status: " + ", ".join(sorted(invalid)))

        def apply(item: dict[str, Any]) -> None:
            workflow = item.get("workflow_plan") or {}
            workflow_nodes = workflow.get("nodes") or []
            known = {str(node.get("node_id") or "") for node in workflow_nodes}
            unknown = set(normalized) - known if known else set()
            if unknown:
                raise ValueError("Unknown Agent Lab workflow node: " + ", ".join(sorted(unknown)))
            changed = False
            for node in workflow_nodes:
                node_id = str(node.get("node_id") or "")
                if node_id in normalized and node.get("status") != normalized[node_id]:
                    node["status"] = normalized[node_id]
                    changed = True
            for step in item.get("plan", []):
                step_id = str(step.get("step_id") or "")
                if step_id in normalized and step.get("status") != normalized[step_id]:
                    step["status"] = normalized[step_id]
                    changed = True

            ledger = item.get("task_ledger")
            if not isinstance(ledger, dict) or not ledger:
                return
            now = _now()
            node_states = ledger.setdefault("node_states", {})
            for node_id, status in normalized.items():
                state = node_states.setdefault(node_id, {})
                if state.get("status") != status:
                    state["status"] = status
                    state["updated_at"] = now
                    changed = True
            for criterion in ledger.get("success_criteria", []):
                node_id = str(criterion.get("node_id") or "")
                if node_id in normalized:
                    criterion["status"] = (
                        "satisfied" if normalized[node_id] == "completed" else normalized[node_id]
                    )
            ledger["updated_at"] = now
            progress = item.setdefault("progress_ledger", [])
            progress.append(
                build_progress_snapshot(
                    ledger,
                    sequence=len(progress) + 1,
                    phase=str(item.get("phase") or ""),
                    progress_made=changed,
                    reason=reason,
                    replan_reason=replan_reason,
                )
            )
            item["progress_ledger"] = progress[-MAX_PROGRESS_SNAPSHOTS:]

        return self.mutate(run_id, apply)

    def record_blocker(
        self,
        run_id: str,
        *,
        node_id: str,
        message: str,
        kind: str = "runtime",
    ) -> dict[str, Any]:
        """Persist an observed blocker without replacing earlier evidence."""
        def apply(item: dict[str, Any]) -> None:
            ledger = item.get("task_ledger")
            if not isinstance(ledger, dict) or not ledger:
                return
            now = _now()
            ledger.setdefault("blockers", []).append({
                "at": now,
                "node_id": str(node_id or "")[:80],
                "kind": str(kind or "runtime")[:80],
                "message": str(message or "")[:2_000],
                "resolved": False,
            })
            ledger["blockers"] = ledger["blockers"][-200:]
            ledger["updated_at"] = now
            progress = item.setdefault("progress_ledger", [])
            progress.append(
                build_progress_snapshot(
                    ledger,
                    sequence=len(progress) + 1,
                    phase=str(item.get("phase") or ""),
                    progress_made=False,
                    reason=str(message or "")[:1_000],
                )
            )
            item["progress_ledger"] = progress[-MAX_PROGRESS_SNAPSHOTS:]

        return self.mutate(run_id, apply)


_stores: dict[tuple[str, str], AgentLabStore] = {}
_stores_lock = threading.RLock()


def get_agent_lab_store(project_id: str | None = None) -> AgentLabStore:
    from remy.core.microbrain import current_project_id
    from remy.core.project_store import get_project_store, project_data_root

    owner = get_project_store().require_project(str(project_id or "").strip() or current_project_id())
    root = project_data_root(owner.project_id)
    key = (str(root), owner.project_id)
    with _stores_lock:
        store = _stores.get(key)
        if store is None:
            store = AgentLabStore(root, owner_project_id=owner.project_id, brain_id=owner.brain_id)
            _stores[key] = store
        return store


def reset_agent_lab_stores_for_tests() -> None:
    with _stores_lock:
        _stores.clear()


def _recover_agent_lab_store(store: AgentLabStore) -> int:
    recovered = 0
    for record in store.list():
        if record.get("status") != "running":
            continue
        run_id = str(record.get("run_id") or "")
        store.transition(
            run_id,
            "paused",
            message="Remy restarted; autonomous workspace recovered at a safe checkpoint",
        )
        store.mutate(run_id, lambda item: item.update({
            "phase": "interrupted",
            "error": "Remy restarted during Agent Lab execution. Files and receipts were preserved; retry autonomously to continue.",
        }))
        recovered += 1
    return recovered


def recover_interrupted_agent_labs() -> int:
    """Recover orphaned background coordinator runs after process restart."""
    from remy.core.project_store import get_project_store

    return sum(
        _recover_agent_lab_store(get_agent_lab_store(project.project_id))
        for project in get_project_store().list_projects(include_archived=False)
    )
