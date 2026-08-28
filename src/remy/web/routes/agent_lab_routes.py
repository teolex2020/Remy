"""REST surface for the project-scoped autonomous Agent Lab foundation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from remy.core.agent_lab import get_agent_lab_store
from remy.core.agent_lab_backend_registry import (
    AUTOMATIC_BACKEND,
    get_agent_lab_backend_registry,
    public_backend_receipt,
)
from remy.core.agent_lab_coordinator import get_agent_lab_coordinator
from remy.core.agent_lab_container import (
    BOUNDED_PROCESS,
    prepare_agent_lab_container_runtime,
    probe_agent_lab_container_runtime,
)
from remy.core.agent_lab_executor import get_agent_lab_executor
from remy.core.agent_lab_proof import write_agent_lab_proof_pack
from remy.core.agent_lab_workspace import get_agent_lab_workspace_manager
from remy.core.microbrain import current_project_id
from remy.core.trajectory_store import get_trajectory_store


router = APIRouter()
AGENT_LAB_EXECUTOR_READY = True


class AgentLabPolicyPayload(BaseModel):
    max_agents: int = Field(default=4, ge=1, le=10)
    max_models: int = Field(default=4, ge=1, le=10)
    time_budget_seconds: int = Field(default=900, ge=60, le=86_400)


class AgentLabCreatePayload(BaseModel):
    goal: str = Field(min_length=1, max_length=20_000)
    title: str = Field(default="", max_length=200)
    interactive: bool = False
    policy: AgentLabPolicyPayload = Field(default_factory=AgentLabPolicyPayload)


class AgentLabFilePayload(BaseModel):
    path: str = Field(min_length=1, max_length=500)
    content: str = Field(max_length=1_000_000)


class AgentLabExecutePayload(BaseModel):
    entrypoint: str = Field(default="src/main.py", min_length=1, max_length=500)
    arguments: list[str] = Field(default_factory=list, max_length=20)
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    isolation_mode: str = Field(default=BOUNDED_PROCESS, min_length=1, max_length=64)


class AgentLabVerifyPayload(BaseModel):
    entrypoint: str = Field(default="tests/verify.py", min_length=1, max_length=500)
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    isolation_mode: str = Field(default=BOUNDED_PROCESS, min_length=1, max_length=64)


class AgentLabAutonomousPayload(BaseModel):
    model: str = Field(default="", max_length=240)
    verifier_model: str = Field(default="", max_length=240)
    max_repair_rounds: int = Field(default=2, ge=0, le=2)
    isolation_mode: str = Field(default=BOUNDED_PROCESS, min_length=1, max_length=64)


class AgentLabClarificationPayload(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)


class AgentLabSnapshotCleanupPayload(BaseModel):
    workspace_ids: list[str] = Field(default_factory=list, max_length=100)
    include_conflicts: bool = False


def _store():
    return get_agent_lab_store(current_project_id())


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": record["run_id"],
        "title": record["title"],
        "goal": record["goal"][:400],
        "status": record["status"],
        "phase": record.get("phase", ""),
        "team_size": len(record.get("team", [])),
        "step_count": len(record.get("plan", [])),
        "artifact_count": len(record.get("artifacts", [])),
        "workspace_ready": bool((record.get("workspace") or {}).get("ready")),
        "created_at": record.get("created_at", ""),
        "updated_at": record.get("updated_at", ""),
        "trajectory_run_event_id": record.get("trajectory_run_event_id", ""),
    }


def _record_event(record: dict[str, Any], event_kind: str, *, name: str, output: Any = None) -> None:
    parent = str(record.get("trajectory_run_event_id") or "")
    if not parent:
        return
    get_trajectory_store().record_execution_event(
        parent_event_id=parent,
        event_kind=event_kind,
        name=name,
        output_value=output,
        details={"phase": record.get("phase", ""), "status": record.get("status", "")},
        source_kind="agent-lab-coordinator",
    )


def _executor():
    return get_agent_lab_executor(_store())


def _coordinator():
    return get_agent_lab_coordinator(_store())


def _workspace_manager():
    return get_agent_lab_workspace_manager(_store())


def _set_phase(run_id: str, phase: str) -> dict[str, Any]:
    return _store().mutate(run_id, lambda item: item.update({"phase": phase}))


def _mark_plan_steps(run_id: str, step_ids: set[str], status: str) -> dict[str, Any]:
    store = _store()
    store.ensure_workflow_state(run_id)
    normalized = "in_progress" if status == "running" else status
    return store.update_node_statuses(
        run_id,
        {step_id: normalized for step_id in step_ids},
        reason="Manual Agent Lab runtime updated workflow node state",
    )


@router.get("/agent-lab/runs")
async def list_agent_lab_runs():
    return {"project_id": current_project_id(), "runs": [_summary(item) for item in _store().list()]}


@router.get("/agent-lab/models")
async def list_agent_lab_models():
    from remy.config.settings import settings
    from remy.core.model_registry import list_registered_models

    models, seen = [], set()
    for item in list_registered_models():
        name = str(item.get("name") or "")
        if name and name not in seen and (item.get("has_key") or name.startswith("llamacpp:")):
            seen.add(name)
            models.append({"name": name, "provider": item.get("provider", ""), "connected": True})
    default = str(settings.SUMMARY_MODEL or "")
    if default and default not in seen:
        models.insert(0, {"name": default, "provider": "default", "connected": True})
    return {"models": models, "default": default}


@router.get("/agent-lab/isolation")
async def get_agent_lab_isolation_status():
    registry = get_agent_lab_backend_registry()
    receipts = await asyncio.to_thread(
        registry.preflight_all,
        probe_runtime=probe_agent_lab_container_runtime,
    )
    backends = [public_backend_receipt(item) for item in receipts]
    by_mode = {item["mode"]: item for item in backends}
    return {
        "default": BOUNDED_PROCESS,
        "modes": registry.modes(),
        "selection_modes": [AUTOMATIC_BACKEND, *registry.modes()],
        "automatic_supported": True,
        "backends": backends,
        "container": by_mode.get("container_required", {}),
    }


@router.post("/agent-lab/isolation/prepare")
async def prepare_agent_lab_isolation_runtime(mode: str = "container_required"):
    try:
        runtime = await asyncio.to_thread(
            get_agent_lab_backend_registry().prepare,
            mode,
            prepare_runtime=prepare_agent_lab_container_runtime,
        )
        public = public_backend_receipt(runtime)
        response = {"backend": public}
        if public.get("mode") == "container_required":
            response["container"] = public
        return response
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/agent-lab/runs")
async def create_agent_lab_run(payload: AgentLabCreatePayload):
    try:
        record = _store().create(
            goal=payload.goal,
            title=payload.title,
            policy=payload.policy.model_dump(),
        )
        if payload.interactive:
            record = _store().mutate(record["run_id"], lambda item: item.update({
                "interactive": {
                    "enabled": True,
                    "awaiting_input": False,
                    "question": "",
                    "missing_fields": [],
                    "answers": [],
                },
            }))
        event_id = get_trajectory_store().begin_execution_run(
            scope="agent_lab",
            project_id=current_project_id(),
            source_id=record["run_id"],
            source_name=record["title"],
            run_id=record["run_id"],
            goal=record["goal"],
            schema={"policy": record["policy"], "mode": "agent-owned"},
            metadata={"phase": "intake", "safety": "local-only"},
        )
        record = _store().mutate(record["run_id"], lambda item: item.update({
            "trajectory_session_id": f"agent_lab:{item['run_id']}:{item['run_id']}",
            "trajectory_run_event_id": event_id,
        }))
        _record_event(record, "AGENT_LAB_PHASE", name="Goal accepted", output={"phase": "intake"})
        return {"run": record, "created": True}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/agent-lab/runs/{run_id}")
async def get_agent_lab_run(run_id: str):
    try:
        return {"run": _store().require(run_id), "active": _coordinator().is_active(run_id)}
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Agent Lab run not found") from exc


@router.post("/agent-lab/runs/{run_id}/archive")
async def archive_agent_lab_run(run_id: str):
    try:
        current = _store().require(run_id)
        if _coordinator().is_active(run_id) or current.get("status") == "running":
            raise ValueError("Stop the laboratory task before removing it from history")
        archived = _store().archive(run_id)
        _record_event(
            archived,
            "AGENT_LAB_PHASE",
            name="Task removed from laboratory history",
            output={"archived": True, "evidence_preserved": True},
        )
        return {"archived": True, "run_id": run_id, "evidence_preserved": True}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Agent Lab run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/agent-lab/runs/{run_id}/snapshots/retention")
async def get_agent_lab_snapshot_retention(run_id: str):
    try:
        retention = await asyncio.to_thread(
            _workspace_manager().retention_status,
            run_id,
        )
        return {"retention": retention}
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Agent Lab run not found") from exc


@router.post("/agent-lab/runs/{run_id}/snapshots/cleanup")
async def cleanup_agent_lab_snapshots(
    run_id: str,
    payload: AgentLabSnapshotCleanupPayload | None = None,
):
    request = payload or AgentLabSnapshotCleanupPayload()
    try:
        result = await asyncio.to_thread(
            _workspace_manager().cleanup_snapshots,
            run_id,
            workspace_ids=request.workspace_ids,
            include_conflicts=request.include_conflicts,
        )
        record = _store().require(run_id)
        _record_event(
            record,
            "AGENT_LAB_RETENTION",
            name="Operator cleaned private snapshots",
            output=result["receipt"],
        )
        return {**result, "run": record, "cleaned": True}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Agent Lab run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/agent-lab/runs/{run_id}/prepare")
async def prepare_agent_lab_run(run_id: str):
    try:
        record = _store().prepare(run_id)
        _record_event(record, "AGENT_LAB_TEAM", name="Agent selected the team", output=record["team"])
        _record_event(
            record,
            "AGENT_LAB_DECISION",
            name="Agent created validated LabWorkflowPlan v2",
            output=record["workflow_plan"],
        )
        _record_event(record, "AGENT_LAB_PHASE", name="Workspace prepared", output=record["workspace"])
        return {"run": record, "prepared": True}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Agent Lab run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/agent-lab/runs/{run_id}/files")
async def stage_agent_lab_file(run_id: str, payload: AgentLabFilePayload):
    try:
        file_receipt = await asyncio.to_thread(
            _executor().write_file,
            run_id,
            path=payload.path,
            content=payload.content,
        )
        record = _store().require(run_id)
        _record_event(record, "AGENT_LAB_ARTIFACT", name="Workspace file staged", output=file_receipt)
        return {"run": record, "file": file_receipt, "staged": True}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Agent Lab run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/agent-lab/runs/{run_id}/autonomous")
async def start_autonomous_agent_lab_run(
    run_id: str,
    payload: AgentLabAutonomousPayload | None = None,
):
    request = payload or AgentLabAutonomousPayload()
    connected = {item["name"] for item in (await list_agent_lab_models())["models"]}
    if request.model and request.model not in connected:
        raise HTTPException(status_code=422, detail="Selected Agent Lab model is not connected")
    if request.verifier_model and request.verifier_model not in connected:
        raise HTTPException(status_code=422, detail="Selected verifier model is not connected")
    try:
        record = _coordinator().start(
            run_id,
            model=request.model,
            verifier_model=request.verifier_model,
            max_repair_rounds=request.max_repair_rounds,
            isolation_mode=request.isolation_mode,
        )
        await asyncio.sleep(0)
        return {"run": record, "started": True, "autonomous": True}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Agent Lab run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/agent-lab/runs/{run_id}/clarify")
async def clarify_agent_lab_run(run_id: str, payload: AgentLabClarificationPayload):
    """Attach operator context to a paused run and continue automatically."""
    try:
        current = _store().require(run_id)
        if current.get("status") != "paused":
            raise ValueError("Only a paused Agent Lab task can accept clarification")
        if _coordinator().is_active(run_id):
            raise ValueError("Agent Lab is already working")

        message = payload.message.strip()
        previous = current.get("autonomous") or {}

        def apply(record: dict[str, Any]) -> None:
            now = datetime.now(timezone.utc).isoformat()
            clarifications = record.setdefault("user_clarifications", [])
            clarifications.append({"at": now, "message": message})
            record["user_clarifications"] = clarifications[-20:]
            messages = record.setdefault("messages", [])
            messages.append({
                "at": now,
                "role": "user",
                "kind": "clarification",
                "content": message,
            })
            record["messages"] = messages[-200:]
            suffix = f"\n\nAdditional user clarification:\n{message}"
            original = str(record.get("goal") or "").rstrip()
            record["goal"] = original[:max(0, 20_000 - len(suffix))] + suffix
            record["error"] = ""
            interaction = record.setdefault("interactive", {})
            interaction["enabled"] = True
            interaction["awaiting_input"] = False
            interaction["question"] = ""
            interaction.setdefault("answers", []).append({"at": now, "message": message})
            interaction["answers"] = interaction["answers"][-20:]
            blockers = (record.get("task_ledger") or {}).get("blockers") or []
            if blockers:
                blockers[-1]["resolved_at"] = now
                blockers[-1]["resolution"] = "User supplied additional information"
            record.setdefault("events", []).append({
                "at": now,
                "type": "clarification",
                "message": "User supplied additional information",
            })

        record = _store().mutate(run_id, apply)
        _record_event(
            record,
            "AGENT_LAB_DECISION",
            name="User supplied clarification",
            output={"message": message},
        )
        resumed = _coordinator().start(
            run_id,
            model=str(previous.get("model") or ""),
            verifier_model=str(previous.get("verifier_model") or ""),
            max_repair_rounds=int(previous.get("max_repair_rounds", 2)),
            isolation_mode=str(previous.get("isolation_mode") or AUTOMATIC_BACKEND),
        )
        await asyncio.sleep(0)
        return {"run": resumed, "continued": True}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Agent Lab run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _transition(run_id: str, target: str, phase: str, message: str) -> dict[str, Any]:
    record = _store().transition(run_id, target, message=message)
    record = _store().mutate(run_id, lambda item: item.update({"phase": phase}))
    _record_event(record, "AGENT_LAB_PHASE", name=message, output={"phase": phase})
    return record


@router.post("/agent-lab/runs/{run_id}/start")
async def start_agent_lab_run(run_id: str, payload: AgentLabExecutePayload | None = None):
    if not AGENT_LAB_EXECUTOR_READY:
        raise HTTPException(
            status_code=409,
            detail="Sandboxed Agent Lab execution is not connected yet. The prepared plan and workspace were preserved.",
        )
    execution = payload or AgentLabExecutePayload()
    try:
        if _coordinator().is_active(run_id):
            raise ValueError("Autonomous coordinator is active; manual execution is locked")
        current = _store().require(run_id)
        if current.get("status") not in {"prepared", "paused"}:
            raise ValueError("Agent Lab execution requires a prepared or paused run")
        record = _transition(run_id, "running", "execution", "Bounded Agent Lab executor started")
        _mark_plan_steps(run_id, {"scope", "evidence", "build"}, "running")
        receipt = await asyncio.to_thread(
            _executor().execute,
            run_id,
            entrypoint=execution.entrypoint,
            arguments=execution.arguments,
            timeout_seconds=execution.timeout_seconds,
            isolation_mode=execution.isolation_mode,
        )
        record = _store().append_execution(run_id, receipt)
        artifacts = await asyncio.to_thread(_executor().inventory_artifacts, run_id)
        record = _store().set_artifacts(run_id, artifacts)
        _record_event(record, "AGENT_LAB_ARTIFACT", name="Artifact inventory updated", output=artifacts)
        _record_event(
            record,
            "AGENT_LAB_PHASE",
            name="Bounded program execution finished",
            output={
                "execution_id": receipt["execution_id"],
                "status": receipt["status"],
                "exit_code": receipt["exit_code"],
                "duration_ms": receipt["duration_ms"],
                "peak_memory_mb": receipt["peak_memory_mb"],
                "isolation_mode": receipt["isolation_mode"],
                "isolation_engine": receipt["isolation_engine"],
                "container_image": receipt["container_image"],
                "container_image_id": receipt["container_image_id"],
                "backend_selection": receipt["backend_selection"],
            },
        )
        latest = _store().require(run_id)
        if latest.get("status") != "running":
            return {"run": latest, "execution": receipt, "started": True}
        phase = "awaiting_verification" if receipt["status"] == "passed" else "repair_needed"
        record = _transition(run_id, "paused", phase, f"Execution {receipt['status']}; checkpoint preserved")
        _mark_plan_steps(run_id, {"scope", "evidence", "build"}, "completed" if receipt["status"] == "passed" else "needs_repair")
        return {"run": _store().require(run_id), "execution": receipt, "started": True}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Agent Lab run not found") from exc
    except FileNotFoundError as exc:
        current = _store().get(run_id)
        if current and current.get("status") == "running":
            _transition(run_id, "paused", "source_required", "Entrypoint is missing; workspace preserved")
        raise HTTPException(status_code=422, detail=f"Agent Lab entrypoint not found: {exc}") from exc
    except ValueError as exc:
        current = _store().get(run_id)
        if current and current.get("status") == "running":
            _transition(run_id, "paused", "repair_needed", str(exc)[:300])
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/agent-lab/runs/{run_id}/verify")
async def verify_agent_lab_run(run_id: str, payload: AgentLabVerifyPayload | None = None):
    verification = payload or AgentLabVerifyPayload()
    try:
        current = _store().require(run_id)
        if current.get("status") != "paused":
            raise ValueError("Verification requires a paused execution checkpoint")
        _transition(run_id, "running", "verification", "Independent verification started")
        receipt = await asyncio.to_thread(
            _executor().execute,
            run_id,
            entrypoint=verification.entrypoint,
            arguments=[],
            timeout_seconds=verification.timeout_seconds,
            read_only=True,
            isolation_mode=verification.isolation_mode,
        )
        receipt["world_fact"] = (
            "supports" if receipt["status"] == "passed"
            else "inconclusive" if receipt["status"] in {"timeout", "memory_limit"}
            else "refutes"
        )
        record = _store().append_verification(run_id, receipt)
        _record_event(record, "AGENT_LAB_VERIFICATION", name="Independent verification finished", output={
            "execution_id": receipt["execution_id"],
            "world_fact": receipt["world_fact"],
            "exit_code": receipt["exit_code"],
            "duration_ms": receipt["duration_ms"],
            "isolation_mode": receipt["isolation_mode"],
            "isolation_engine": receipt["isolation_engine"],
            "container_image": receipt["container_image"],
            "container_image_id": receipt["container_image_id"],
            "backend_selection": receipt["backend_selection"],
        })
        latest = _store().require(run_id)
        if latest.get("status") != "running":
            return {
                "run": latest,
                "verification": receipt,
                "verified": False,
            }
        if receipt["world_fact"] == "supports":
            _mark_plan_steps(run_id, {"verify", "handoff"}, "completed")
            proof = await asyncio.to_thread(
                write_agent_lab_proof_pack,
                _store(),
                run_id,
                decision="accepted",
                reason="Manual independent read-only verification passed",
            )
            artifacts = await asyncio.to_thread(_executor().inventory_artifacts, run_id)
            _store().set_artifacts(run_id, artifacts)
            _record_event(
                _store().require(run_id),
                "AGENT_LAB_PROOF",
                name="Accepted Proof Pack generated",
                output=proof,
            )
            record = _transition(run_id, "completed", "completed", "Verification passed; artifacts are ready")
            result_id = get_trajectory_store().complete_execution_run(
                event_id=record.get("trajectory_run_event_id", ""),
                status="completed",
                output={"artifacts": record.get("artifacts", []), "verification": receipt},
            )
            record = _store().mutate(run_id, lambda item: item.update({"trajectory_result_event_id": result_id}))
        else:
            record = _transition(run_id, "paused", "repair_needed", f"Verification {receipt['world_fact']}; repair required")
            _mark_plan_steps(run_id, {"verify"}, "needs_repair")
            proof = await asyncio.to_thread(
                write_agent_lab_proof_pack,
                _store(),
                run_id,
                decision=("rejected" if receipt["world_fact"] == "refutes" else "inconclusive"),
                reason=f"Manual verification {receipt['world_fact']}",
            )
            artifacts = await asyncio.to_thread(_executor().inventory_artifacts, run_id)
            _store().set_artifacts(run_id, artifacts)
            _record_event(
                _store().require(run_id),
                "AGENT_LAB_PROOF",
                name="Non-acceptance Proof Pack generated",
                output=proof,
            )
        return {"run": _store().require(run_id), "verification": receipt, "verified": receipt["world_fact"] == "supports"}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Agent Lab run not found") from exc
    except FileNotFoundError as exc:
        current = _store().get(run_id)
        if current and current.get("status") == "running":
            _transition(run_id, "paused", "verification_required", "Verification entrypoint is missing")
        raise HTTPException(status_code=422, detail=f"Verification entrypoint not found: {exc}") from exc
    except ValueError as exc:
        current = _store().get(run_id)
        if current and current.get("status") == "running":
            _transition(run_id, "paused", "repair_needed", str(exc)[:300])
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/agent-lab/runs/{run_id}/pause")
async def pause_agent_lab_run(run_id: str):
    try:
        _coordinator().cancel(run_id)
        await asyncio.to_thread(_executor().cancel, run_id)
        current = _store().require(run_id)
        if current.get("status") == "running":
            current = _transition(run_id, "paused", "paused", "Agent Lab paused at a safe checkpoint")
        return {"run": current, "paused": True}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Agent Lab run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/agent-lab/runs/{run_id}/resume")
async def resume_agent_lab_run(run_id: str):
    raise HTTPException(
        status_code=409,
        detail="Resume requires a new bounded execute or verify request; no process is resumed implicitly.",
    )


@router.post("/agent-lab/runs/{run_id}/cancel")
async def cancel_agent_lab_run(run_id: str):
    try:
        _coordinator().cancel(run_id)
        await asyncio.to_thread(_executor().cancel, run_id)
        record = _transition(run_id, "cancelled", "cancelled", "Agent Lab cancelled by the operator")
        result_id = get_trajectory_store().complete_execution_run(
            event_id=record.get("trajectory_run_event_id", ""),
            status="cancelled",
            output={"phase": "cancelled", "artifacts": record.get("artifacts", [])},
        )
        record = _store().mutate(run_id, lambda item: item.update({"trajectory_result_event_id": result_id}))
        return {"run": record, "cancelled": True}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Agent Lab run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/agent-lab/runs/{run_id}/artifacts/{artifact_id}")
async def download_agent_lab_artifact(run_id: str, artifact_id: str):
    try:
        record = _store().require(run_id)
        artifact = next(
            (item for item in record.get("artifacts", []) if item.get("artifact_id") == artifact_id),
            None,
        )
        if not artifact:
            raise KeyError(artifact_id)
        root = (_store().workspace_path(run_id) / "artifacts").resolve()
        relative = str(artifact.get("path") or "").removeprefix("artifacts/")
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or not target.is_file() or target.is_symlink():
            raise KeyError(artifact_id)
        return FileResponse(
            target,
            filename=str(artifact.get("name") or target.name),
            media_type=str(artifact.get("mime_type") or "application/octet-stream"),
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Agent Lab artifact not found") from exc
