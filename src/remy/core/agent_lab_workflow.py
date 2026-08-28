"""Validated workflow and durable cognitive ledgers for Agent Lab.

The model may propose orchestration data, but this module compiles it into a
small policy-bounded DAG.  It deliberately contains no executable workflow
language: nodes describe authority and evidence contracts, never code.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any


WORKFLOW_VERSION = 2
MAX_WORKFLOW_NODES = 32
MAX_PROGRESS_SNAPSHOTS = 500
ALLOWED_NODE_STATUSES = {
    "pending",
    "in_progress",
    "completed",
    "needs_repair",
    "blocked",
    "skipped",
}
ALLOWED_WORKSPACE_MODES = {"readonly", "private_snapshot"}
ALLOWED_GATE_KINDS = {
    "node_output",
    "execution_passed",
    "verification_supports",
    "handoff_complete",
}
_SAFE_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,79}$")
_SAFE_TOOL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_.:-]{0,99}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bounded_text(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _validate_json_schema(value: Any, *, node_id: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Workflow node {node_id} output_schema must be an object")
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Workflow node {node_id} output_schema is not JSON data") from exc
    if len(encoded) > 16_000:
        raise ValueError(f"Workflow node {node_id} output_schema is too large")
    if value.get("type") not in {None, "object", "array", "string", "number", "boolean"}:
        raise ValueError(f"Workflow node {node_id} output_schema has an unsupported root type")
    return json.loads(encoded)


def _normalize_gate(value: Any, *, node_id: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Workflow node {node_id} success_gate must be an object")
    unknown = set(value) - {"kind", "required", "artifact_patterns", "exit_code"}
    if unknown:
        raise ValueError(
            f"Workflow node {node_id} success_gate has unknown fields: "
            + ", ".join(sorted(unknown))
        )
    kind = _bounded_text(value.get("kind"), 80)
    if kind not in ALLOWED_GATE_KINDS:
        raise ValueError(f"Workflow node {node_id} has unsupported success gate {kind!r}")
    patterns = value.get("artifact_patterns") or []
    if not isinstance(patterns, list) or len(patterns) > 20:
        raise ValueError(f"Workflow node {node_id} artifact_patterns must be a bounded list")
    normalized_patterns = []
    for pattern in patterns:
        text = _bounded_text(pattern, 200).replace("\\", "/")
        if not text or text.startswith("/") or ".." in text.split("/"):
            raise ValueError(f"Workflow node {node_id} has an unsafe artifact pattern")
        normalized_patterns.append(text)
    gate = {"kind": kind, "required": bool(value.get("required", True))}
    if normalized_patterns:
        gate["artifact_patterns"] = normalized_patterns
    if "exit_code" in value:
        gate["exit_code"] = int(value["exit_code"])
    return gate


def validate_lab_workflow_plan(
    raw: Any,
    *,
    policy: dict[str, Any],
    available_models: set[str] | None = None,
    allowed_tools: set[str] | None = None,
) -> dict[str, Any]:
    """Validate and normalize a model- or system-authored Agent Lab DAG."""
    if not isinstance(raw, dict):
        raise ValueError("LabWorkflowPlan must be an object")
    unknown = set(raw) - {
        "version",
        "plan_id",
        "goal",
        "nodes",
        "edges",
        "max_parallel",
        "max_total_agents",
        "created_at",
    }
    if unknown:
        raise ValueError("Unknown LabWorkflowPlan fields: " + ", ".join(sorted(unknown)))
    if int(raw.get("version") or 0) != WORKFLOW_VERSION:
        raise ValueError(f"LabWorkflowPlan version must be {WORKFLOW_VERSION}")

    plan_id = _bounded_text(raw.get("plan_id"), 80)
    if not _SAFE_ID_RE.fullmatch(plan_id):
        raise ValueError("LabWorkflowPlan plan_id is invalid")
    goal = _bounded_text(raw.get("goal"), 20_000)
    if not goal:
        raise ValueError("LabWorkflowPlan requires a goal")
    nodes = raw.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("LabWorkflowPlan requires at least one node")
    if len(nodes) > MAX_WORKFLOW_NODES:
        raise ValueError(f"LabWorkflowPlan exceeds the {MAX_WORKFLOW_NODES}-node limit")

    policy_agents = max(1, int(policy.get("max_agents") or 1))
    max_parallel = int(raw.get("max_parallel") or 1)
    max_total_agents = int(raw.get("max_total_agents") or policy_agents)
    if max_parallel < 1 or max_parallel > policy_agents:
        raise ValueError("LabWorkflowPlan max_parallel exceeds the Agent Lab policy")
    if max_total_agents < 1 or max_total_agents > policy_agents:
        raise ValueError("LabWorkflowPlan max_total_agents exceeds the Agent Lab policy")

    normalized_nodes: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"Workflow nodes[{index}] must be an object")
        extra = set(node) - {
            "node_id",
            "title",
            "owner",
            "role",
            "model",
            "depends_on",
            "workspace_mode",
            "input_artifacts",
            "output_schema",
            "success_gate",
            "retry_owner",
            "allowed_tools",
            "status",
        }
        if extra:
            raise ValueError(
                f"Workflow nodes[{index}] has unknown fields: " + ", ".join(sorted(extra))
            )
        node_id = _bounded_text(node.get("node_id"), 80)
        if not _SAFE_ID_RE.fullmatch(node_id) or node_id in ids:
            raise ValueError(f"Workflow node id {node_id!r} is invalid or duplicated")
        ids.add(node_id)
        role = _bounded_text(node.get("role"), 80)
        if not _SAFE_ID_RE.fullmatch(role):
            raise ValueError(f"Workflow node {node_id} role is invalid")
        owner = _bounded_text(node.get("owner") or role, 100)
        if not _SAFE_ID_RE.fullmatch(owner):
            raise ValueError(f"Workflow node {node_id} owner is invalid")
        model = _bounded_text(node.get("model") or "automatic", 200)
        if available_models is not None and model != "automatic" and model not in available_models:
            raise ValueError(f"Workflow node {node_id} requests an unavailable model")
        workspace_mode = _bounded_text(node.get("workspace_mode"), 40)
        if workspace_mode not in ALLOWED_WORKSPACE_MODES:
            raise ValueError(f"Workflow node {node_id} has an unsupported workspace mode")
        status = _bounded_text(node.get("status") or "pending", 40)
        if status not in ALLOWED_NODE_STATUSES:
            raise ValueError(f"Workflow node {node_id} has an unsupported status")

        dependencies = node.get("depends_on") or []
        inputs = node.get("input_artifacts") or []
        tools = node.get("allowed_tools") or []
        if not isinstance(dependencies, list) or len(dependencies) > MAX_WORKFLOW_NODES:
            raise ValueError(f"Workflow node {node_id} depends_on must be a bounded list")
        if not isinstance(inputs, list) or len(inputs) > 50:
            raise ValueError(f"Workflow node {node_id} input_artifacts must be a bounded list")
        if not isinstance(tools, list) or len(tools) > 50:
            raise ValueError(f"Workflow node {node_id} allowed_tools must be a bounded list")
        normalized_tools = []
        for tool in tools:
            name = _bounded_text(tool, 100)
            if not _SAFE_TOOL_RE.fullmatch(name):
                raise ValueError(f"Workflow node {node_id} contains an invalid tool name")
            if allowed_tools is not None and name not in allowed_tools:
                raise ValueError(f"Workflow node {node_id} requests disallowed tool {name}")
            if name not in normalized_tools:
                normalized_tools.append(name)

        retry_owner = _bounded_text(node.get("retry_owner"), 80)
        normalized_inputs = []
        for value in inputs:
            artifact = _bounded_text(value, 300).replace("\\", "/")
            if (
                not artifact
                or artifact.startswith("/")
                or ".." in artifact.split("/")
            ):
                raise ValueError(f"Workflow node {node_id} has an unsafe input artifact")
            normalized_inputs.append(artifact)

        normalized_nodes.append({
            "node_id": node_id,
            "title": _bounded_text(node.get("title"), 300) or node_id.replace("_", " ").title(),
            "owner": owner,
            "role": role,
            "model": model,
            "depends_on": [_bounded_text(value, 80) for value in dependencies],
            "workspace_mode": workspace_mode,
            "input_artifacts": normalized_inputs,
            "output_schema": _validate_json_schema(node.get("output_schema") or {"type": "object"}, node_id=node_id),
            "success_gate": _normalize_gate(node.get("success_gate") or {"kind": "node_output"}, node_id=node_id),
            "retry_owner": retry_owner,
            "allowed_tools": normalized_tools,
            "status": status,
        })

    by_id = {node["node_id"]: node for node in normalized_nodes}
    assigned_models = {
        node["model"] for node in normalized_nodes if node["model"] != "automatic"
    }
    if len(assigned_models) > max(1, int(policy.get("max_models") or 1)):
        raise ValueError("LabWorkflowPlan uses more models than the Agent Lab policy allows")
    for node in normalized_nodes:
        node_id = node["node_id"]
        dependencies = node["depends_on"]
        if len(dependencies) != len(set(dependencies)):
            raise ValueError(f"Workflow node {node_id} repeats a dependency")
        for dependency in dependencies:
            if dependency not in by_id or dependency == node_id:
                raise ValueError(f"Workflow node {node_id} has an invalid dependency")
        if node["retry_owner"] and node["retry_owner"] not in by_id:
            raise ValueError(f"Workflow node {node_id} has an invalid retry_owner")

    edges = raw.get("edges") or []
    if not isinstance(edges, list) or len(edges) > MAX_WORKFLOW_NODES * MAX_WORKFLOW_NODES:
        raise ValueError("LabWorkflowPlan edges must be a bounded list")
    expected_edges = {
        (dependency, node["node_id"])
        for node in normalized_nodes
        for dependency in node["depends_on"]
    }
    supplied_edges: set[tuple[str, str]] = set()
    for edge in edges:
        if not isinstance(edge, dict) or set(edge) != {"source", "target"}:
            raise ValueError("Each workflow edge must contain only source and target")
        pair = (_bounded_text(edge.get("source"), 80), _bounded_text(edge.get("target"), 80))
        if pair[0] not in by_id or pair[1] not in by_id:
            raise ValueError("Workflow edge references an unknown node")
        supplied_edges.add(pair)
    if supplied_edges and supplied_edges != expected_edges:
        raise ValueError("Workflow edges must exactly match node dependencies")

    incoming = {node_id: 0 for node_id in by_id}
    outgoing = {node_id: [] for node_id in by_id}
    for source, target in expected_edges:
        incoming[target] += 1
        outgoing[source].append(target)
    ready = sorted(node_id for node_id, count in incoming.items() if count == 0)
    visited = []
    while ready:
        node_id = ready.pop(0)
        visited.append(node_id)
        for target in sorted(outgoing[node_id]):
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
                ready.sort()
    if len(visited) != len(by_id):
        raise ValueError("LabWorkflowPlan must be acyclic")

    return {
        "version": WORKFLOW_VERSION,
        "plan_id": plan_id,
        "goal": goal,
        "nodes": normalized_nodes,
        "edges": [
            {"source": source, "target": target}
            for source, target in sorted(expected_edges)
        ],
        "max_parallel": max_parallel,
        "max_total_agents": max_total_agents,
        "created_at": _bounded_text(raw.get("created_at"), 100) or _now(),
    }


def build_default_lab_workflow_plan(
    goal: str,
    team: list[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Create the safe v2 DAG used before any model-authored re-planning."""
    role_agents = {str(item.get("role")): str(item.get("agent_id")) for item in team}
    coordinator = role_agents.get("coordinator", "agent-1")
    builder = role_agents.get("builder", coordinator)
    researcher = role_agents.get("researcher", builder)
    verifier = role_agents.get("verifier", coordinator)

    def node(
        node_id: str,
        title: str,
        owner: str,
        role: str,
        depends_on: list[str],
        workspace_mode: str,
        gate: str,
        *,
        retry_owner: str = "",
        allowed_tools: list[str] | None = None,
        output_properties: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "node_id": node_id,
            "title": title,
            "owner": owner,
            "role": role,
            "model": "automatic",
            "depends_on": depends_on,
            "workspace_mode": workspace_mode,
            "input_artifacts": [],
            "output_schema": {"type": "object", "properties": output_properties or {}},
            "success_gate": {"kind": gate, "required": True},
            "retry_owner": retry_owner,
            "allowed_tools": list(allowed_tools or []),
            "status": "pending",
        }

    nodes = [
        node(
            "scope",
            "Bound the goal and success checks",
            coordinator,
            "coordinator",
            [],
            "readonly",
            "node_output",
            output_properties={"success_criteria": {"type": "array"}},
        ),
        node(
            "evidence",
            "Collect inputs and record assumptions",
            researcher,
            "researcher" if "researcher" in role_agents else "builder",
            ["scope"],
            "readonly",
            "node_output",
            allowed_tools=["recall", "search", "recall_knowledge", "get_current_datetime"],
            output_properties={"facts": {"type": "array"}, "assumptions": {"type": "array"}},
        ),
        node(
            "build",
            "Build inside a private workspace snapshot",
            builder,
            "builder",
            ["evidence"],
            "private_snapshot",
            "execution_passed",
            retry_owner="build",
            output_properties={"artifact_manifest": {"type": "array"}},
        ),
        node(
            "verify",
            "Run independent verification",
            verifier,
            "verifier",
            ["build"],
            "private_snapshot",
            "verification_supports",
            retry_owner="build",
            output_properties={"world_fact": {"type": "string"}},
        ),
        node(
            "handoff",
            "Package artifacts, evidence, and limitations",
            coordinator,
            "coordinator",
            ["verify"],
            "readonly",
            "handoff_complete",
            output_properties={"artifacts": {"type": "array"}, "limitations": {"type": "array"}},
        ),
    ]
    raw = {
        "version": WORKFLOW_VERSION,
        "plan_id": "initial-plan",
        "goal": goal,
        "nodes": nodes,
        "edges": [],
        "max_parallel": min(max(1, int(policy.get("max_agents") or 1)), 3),
        "max_total_agents": max(1, int(policy.get("max_agents") or 1)),
    }
    return validate_lab_workflow_plan(
        raw,
        policy=policy,
        allowed_tools={"recall", "search", "recall_knowledge", "get_current_datetime"},
    )


def legacy_plan_projection(workflow_plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep the original UI/API plan contract while v2 rolls out."""
    return [
        {
            "step_id": node["node_id"],
            "title": node["title"],
            "owner": node["owner"],
            "status": node["status"],
        }
        for node in workflow_plan.get("nodes", [])
    ]


def create_task_ledger(workflow_plan: dict[str, Any]) -> dict[str, Any]:
    now = _now()
    return {
        "version": 1,
        "plan_id": workflow_plan["plan_id"],
        "plan_revision": 1,
        "goal": workflow_plan.get("goal", ""),
        "facts": [],
        "assumptions": [],
        "success_criteria": [
            {
                "node_id": node["node_id"],
                "description": node["title"],
                "gate": node["success_gate"],
                "status": "pending",
            }
            for node in workflow_plan.get("nodes", [])
            if node.get("success_gate", {}).get("required", True)
        ],
        "blockers": [],
        "node_states": {
            node["node_id"]: {
                "status": node["status"],
                "owner": node["owner"],
                "model": node["model"],
                "workspace_mode": node["workspace_mode"],
                "updated_at": now,
            }
            for node in workflow_plan.get("nodes", [])
        },
        "artifact_hashes": {},
        "created_at": now,
        "updated_at": now,
    }


def build_progress_snapshot(
    task_ledger: dict[str, Any],
    *,
    sequence: int,
    phase: str,
    progress_made: bool,
    reason: str = "",
    replan_reason: str = "",
) -> dict[str, Any]:
    states = task_ledger.get("node_states") or {}
    completed = sorted(node_id for node_id, value in states.items() if value.get("status") == "completed")
    active = sorted(node_id for node_id, value in states.items() if value.get("status") == "in_progress")
    blocked = sorted(
        node_id
        for node_id, value in states.items()
        if value.get("status") in {"blocked", "needs_repair"}
    )
    pending = sorted(node_id for node_id, value in states.items() if value.get("status") == "pending")
    fingerprint_payload = {
        node_id: value.get("status") for node_id, value in sorted(states.items())
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "sequence": int(sequence),
        "at": _now(),
        "phase": _bounded_text(phase, 100),
        "progress_made": bool(progress_made),
        "reason": _bounded_text(reason, 1_000),
        "completed_nodes": completed,
        "active_nodes": active,
        "blocked_nodes": blocked,
        "next_nodes": pending,
        "state_fingerprint": fingerprint,
        "replan_reason": _bounded_text(replan_reason, 1_000),
    }
