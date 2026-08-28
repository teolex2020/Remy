"""Deterministic, exportable evidence packet for an Agent Lab decision."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from remy.core.agent_lab import AgentLabStore
from remy.core.file_utils import atomic_write


PROOF_SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _bounded_execution(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        key: receipt.get(key)
        for key in (
            "execution_id",
            "entrypoint",
            "status",
            "exit_code",
            "duration_ms",
            "peak_memory_mb",
            "read_only",
            "isolation_mode",
            "isolation_engine",
            "container_image",
            "container_image_id",
            "container_security",
            "backend_selection",
            "started_at",
            "completed_at",
            "world_fact",
            "verifier_model",
            "served_by",
            "independence_level",
            "verifier_workspace_id",
            "verifier_merge_id",
        )
        if key in receipt
    } | {
        "stdout": str(receipt.get("stdout") or "")[:4_000],
        "stderr": str(receipt.get("stderr") or "")[:4_000],
        "stdout_truncated": bool(receipt.get("stdout_truncated")),
        "stderr_truncated": bool(receipt.get("stderr_truncated")),
    }


def build_agent_lab_proof_pack(
    record: dict[str, Any],
    *,
    decision: str,
    reason: str,
) -> dict[str, Any]:
    """Build a proof from receipts only; never include prompts or hidden reasoning."""
    workflow = record.get("workflow_plan") or {}
    artifacts = [
        {
            "artifact_id": item.get("artifact_id", ""),
            "path": item.get("path", ""),
            "size": int(item.get("size") or 0),
            "mime_type": item.get("mime_type", ""),
            "sha256": item.get("sha256", ""),
        }
        for item in record.get("artifacts", [])
        if Path(str(item.get("path") or "")).name
        not in {"proof-pack.json", "proof-pack.md"}
    ]
    artifact_tree_hash = _hash_json(
        [[item["path"], item["sha256"], item["size"]] for item in artifacts]
    )
    merges = [
        {
            "merge_id": item.get("merge_id", ""),
            "workspace_id": item.get("workspace_id", ""),
            "node_id": item.get("node_id", ""),
            "status": item.get("status", ""),
            "baseline_root_hash": item.get("baseline_root_hash", ""),
            "canonical_before_hash": item.get("canonical_before_hash", ""),
            "candidate_root_hash": item.get("candidate_root_hash", ""),
            "canonical_after_hash": item.get("canonical_after_hash", ""),
            "changes": list(item.get("changes") or []),
            "conflicts": list(item.get("conflicts") or []),
            "applied_files": list(item.get("applied_files") or []),
            "created_at": item.get("created_at", ""),
        }
        for item in record.get("merge_receipts", [])
    ]
    verifier = dict(record.get("verifier") or {})
    verifier.pop("model_catalog", None)
    verifier.pop("rationale", None)
    fanout = record.get("builder_fanout") or {}
    builder_receipt = {
        "status": fanout.get("status", ""),
        "team_required": bool(fanout.get("team_required")),
        "planner_model": fanout.get("planner_model", ""),
        "planner_served_by": fanout.get("planner_served_by", ""),
        "merge_order": list(fanout.get("merge_order") or []),
        "members": [
            {
                "builder_id": item.get("builder_id") or item.get("id", ""),
                "assigned_model": item.get("model", ""),
                "served_by": item.get("served_by", ""),
                "file_claims": list(item.get("file_claims") or []),
                "workspace_id": item.get("workspace_id", ""),
                "status": item.get("status", ""),
                "merge_id": item.get("merge_id", ""),
            }
            for item in fanout.get("members", [])
        ],
    }
    pack = {
        "schema_version": PROOF_SCHEMA_VERSION,
        "run_id": record.get("run_id", ""),
        "project_id": record.get("owner_project_id", ""),
        "goal": record.get("goal", ""),
        "generated_at": _now(),
        "plan": {
            "plan_id": workflow.get("plan_id", ""),
            "version": workflow.get("version", 0),
            "sha256": _hash_json(workflow),
            "node_states": (record.get("task_ledger") or {}).get("node_states", {}),
            "success_criteria": (record.get("task_ledger") or {}).get(
                "success_criteria", []
            ),
            "blockers": (record.get("task_ledger") or {}).get("blockers", []),
        },
        "model_assignments": {
            "coordinator": (record.get("autonomous") or {}).get("served_by")
            or (record.get("autonomous") or {}).get("model", ""),
            "team": [
                {
                    "agent_id": item.get("agent_id", ""),
                    "role": item.get("role", ""),
                    "assigned_model": item.get("model", ""),
                    "served_by": item.get("served_by", ""),
                }
                for item in record.get("team", [])
            ],
            "builder_fanout": builder_receipt,
            "verifier": verifier,
        },
        "workspace_evidence": {
            "merges": merges,
            "snapshot_cleanups": [
                {
                    key: item.get(key, "")
                    for key in (
                        "cleanup_id", "status", "requested_workspace_ids",
                        "include_conflicts", "removed_workspace_ids", "removed_count",
                        "recovered_bytes", "errors", "created_at"
                    )
                }
                for item in record.get("snapshot_cleanup_receipts", [])
            ],
            "file_claims": [
                {
                    key: item.get(key, "")
                    for key in (
                        "claim_id", "builder_id", "model", "workspace_id", "path",
                        "status", "merge_id", "error", "created_at", "resolved_at"
                    )
                }
                for item in record.get("file_claims", [])
            ],
            "source_tree_hash": next(
                (
                    item.get("canonical_after_hash", "")
                    for item in reversed(merges)
                    if item.get("canonical_after_hash")
                ),
                "",
            ),
        },
        "runtime_evidence": {
            "executions": [
                _bounded_execution(item) for item in record.get("executions", [])
            ],
            "verification": [
                _bounded_execution(item) for item in record.get("verification", [])
            ],
        },
        "artifact_evidence": {
            "artifacts": artifacts,
            "tree_hash": artifact_tree_hash,
        },
        "final_decision": {
            "decision": str(decision or "inconclusive"),
            "reason": str(reason or "")[:2_000],
            "verification_world_fact": (
                (record.get("verification") or [{}])[-1].get("world_fact", "")
                if record.get("verification")
                else ""
            ),
            "verifier_independence": verifier.get("independence_level", ""),
        },
    }
    pack["proof_pack_sha256"] = _hash_json(pack)
    return pack


def _render_markdown(pack: dict[str, Any]) -> str:
    decision = pack["final_decision"]
    artifacts = pack["artifact_evidence"]["artifacts"]
    merges = pack["workspace_evidence"]["merges"]
    verification = pack["runtime_evidence"]["verification"]
    lines = [
        "# Agent Lab Proof Pack",
        "",
        f"- Run: `{pack['run_id']}`",
        f"- Decision: **{decision['decision']}**",
        f"- Reason: {decision['reason']}",
        f"- Verifier independence: `{decision['verifier_independence'] or 'not-recorded'}`",
        f"- Plan hash: `{pack['plan']['sha256']}`",
        f"- Artifact tree hash: `{pack['artifact_evidence']['tree_hash']}`",
        f"- Proof hash: `{pack['proof_pack_sha256']}`",
        "",
        "## Merge receipts",
        "",
    ]
    lines.extend(
        f"- `{item['merge_id']}` — {item['status']}; {len(item['applied_files'])} file(s)"
        for item in merges
    )
    if not merges:
        lines.append("- None")
    lines.extend(["", "## Verification", ""])
    lines.extend(
        f"- `{item.get('execution_id', '')}` — {item.get('world_fact', item.get('status', ''))}; "
        f"exit `{item.get('exit_code', '')}`; read-only `{item.get('read_only', False)}`"
        for item in verification
    )
    if not verification:
        lines.append("- No verification receipt")
    lines.extend(["", "## Artifacts", ""])
    lines.extend(
        f"- `{item['path']}` — {item['size']} bytes — `{item['sha256']}`"
        for item in artifacts
    )
    if not artifacts:
        lines.append("- No produced artifacts")
    lines.append("")
    return "\n".join(lines)


def write_agent_lab_proof_pack(
    store: AgentLabStore,
    run_id: str,
    *,
    decision: str,
    reason: str,
) -> dict[str, Any]:
    record = store.require(run_id)
    pack = build_agent_lab_proof_pack(record, decision=decision, reason=reason)
    json_text = json.dumps(pack, ensure_ascii=False, indent=2) + "\n"
    markdown = _render_markdown(pack)
    root = (store.workspace_path(run_id) / "artifacts").resolve()
    if not root.is_relative_to(store.workspace_path(run_id)):
        raise ValueError("Proof Pack artifact path escaped Agent Lab workspace")
    existing_size = sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.name not in {"proof-pack.json", "proof-pack.md"}
    )
    if existing_size + len(json_text.encode("utf-8")) + len(markdown.encode("utf-8")) > int(
        record["policy"]["max_artifact_bytes"]
    ):
        raise ValueError("Agent Lab Proof Pack exceeds the artifact size policy")
    with store.workspace_lock:
        root.mkdir(parents=True, exist_ok=True)
        atomic_write(root / "proof-pack.json", json_text)
        atomic_write(root / "proof-pack.md", markdown)
    receipt = {
        "schema_version": PROOF_SCHEMA_VERSION,
        "decision": decision,
        "json_path": "artifacts/proof-pack.json",
        "markdown_path": "artifacts/proof-pack.md",
        "sha256": pack["proof_pack_sha256"],
        "artifact_tree_hash": pack["artifact_evidence"]["tree_hash"],
        "plan_sha256": pack["plan"]["sha256"],
        "generated_at": pack["generated_at"],
    }
    store.mutate(run_id, lambda item: item.update({"proof_pack": receipt}))
    return receipt
