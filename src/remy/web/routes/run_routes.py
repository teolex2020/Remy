"""Operator-facing API for unified Remy run envelopes."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from remy.core.microbrain import current_project_id
from remy.core.run_envelope import get_run, list_runs, request_run_stop

router = APIRouter()


class StopRunRequest(BaseModel):
    reason: str = Field(default="Stopped by user", max_length=500)


class WorkerTaskRequest(BaseModel):
    role: str = Field(default="researcher", max_length=80)
    instruction: str = Field(min_length=1, max_length=20_000)
    context: str = Field(default="", max_length=20_000)
    approval_mode: str = Field(default="none", max_length=40)


class LaunchWorkerTasksRequest(BaseModel):
    tasks: list[WorkerTaskRequest] = Field(min_length=1, max_length=16)
    conversation_id: str = Field(default="", max_length=200)
    channel: str = Field(default="web", max_length=40)


class ResumeWorkerTaskRequest(BaseModel):
    confirm_side_effects: bool = False


class ChildFollowUpRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    confirm_side_effects: bool = False


class ResumeChildRequest(BaseModel):
    confirm_side_effects: bool = False


class PTCProgramRequest(BaseModel):
    program: dict
    limits: dict = Field(default_factory=dict)
    session_id: str = Field(default="", max_length=200)
    channel: str = Field(default="ptc", max_length=40)


@router.get("/ptc/tools")
async def get_ptc_tools():
    from remy.core.ptc_pilot import PTC_SCHEMA, PTC_VERSION, list_ptc_tools

    return {
        "schema": PTC_SCHEMA,
        "version": PTC_VERSION,
        "read_only": True,
        "tools": list_ptc_tools(),
    }


@router.post("/ptc/validate")
async def validate_project_ptc(payload: PTCProgramRequest):
    from remy.core.ptc_pilot import validate_ptc_program

    return validate_ptc_program(payload.program, payload.limits)


@router.post("/ptc/run")
async def run_project_ptc(payload: PTCProgramRequest):
    from remy.core.ptc_pilot import execute_ptc_program

    return await asyncio.to_thread(
        execute_ptc_program,
        payload.program,
        owner_project_id=current_project_id(),
        session_id=payload.session_id,
        channel=payload.channel or "ptc",
        limits=payload.limits,
    )


@router.get("/runs")
async def get_project_runs(status: str = "", kind: str = "", limit: int = 100):
    project_id = current_project_id()
    return {
        "project_id": project_id,
        "runs": list_runs(
            owner_project_id=project_id,
            status=str(status or "").strip(),
            kind=str(kind or "").strip(),
            limit=max(1, min(int(limit), 500)),
        ),
    }


@router.get("/runs/{run_id}")
async def get_project_run(run_id: str):
    run = get_run(run_id, owner_project_id=current_project_id())
    if not run:
        raise HTTPException(status_code=404, detail="Run not found in this project.")
    return {"run": run}


@router.post("/runs/{run_id}/stop")
async def stop_project_run(run_id: str, payload: StopRunRequest):
    try:
        run = request_run_stop(
            run_id,
            owner_project_id=current_project_id(),
            reason=payload.reason,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found in this project.") from exc
    return {"run": run, "stop_requested": True}


@router.post("/tasks/workers")
async def launch_project_worker_tasks(payload: LaunchWorkerTasksRequest):
    from remy.core.worker import WorkerTask
    from remy.core.worker_tasks import launch_worker_tasks

    try:
        handle = launch_worker_tasks(
            [
                WorkerTask(
                    role=item.role,
                    instruction=item.instruction,
                    context=item.context,
                    approval_mode=item.approval_mode,
                )
                for item in payload.tasks
            ],
            session_id=payload.conversation_id,
            channel=payload.channel or "web",
            owner_project_id=current_project_id(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"task": handle}


@router.get("/tasks")
async def get_project_worker_tasks(status: str = "", limit: int = 100):
    from remy.core.worker_tasks import public_task_handle

    runs = list_runs(
        owner_project_id=current_project_id(),
        status=str(status or "").strip(),
        kind="worker_group",
        limit=max(1, min(int(limit), 500)),
    )
    return {"tasks": [public_task_handle(run) for run in runs]}


@router.get("/tasks/{task_id}")
async def get_project_worker_task(task_id: str):
    from remy.core.worker_tasks import get_worker_task

    handle = get_worker_task(task_id, owner_project_id=current_project_id())
    if not handle:
        raise HTTPException(status_code=404, detail="Task not found in this project.")
    return {"task": handle}


@router.post("/tasks/{task_id}/cancel")
async def cancel_project_worker_task(task_id: str, payload: StopRunRequest):
    try:
        run = request_run_stop(
            task_id,
            owner_project_id=current_project_id(),
            reason=payload.reason,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Task not found in this project.") from exc
    return {"task_id": task_id, "run": run, "cancel_requested": True}


@router.post("/tasks/{task_id}/resume")
async def resume_project_worker_task(task_id: str, payload: ResumeWorkerTaskRequest):
    from remy.core.worker_tasks import resume_worker_tasks

    try:
        handle = resume_worker_tasks(
            task_id,
            owner_project_id=current_project_id(),
            confirm_side_effects=payload.confirm_side_effects,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Task not found in this project.") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"task": handle, "resumed_from": task_id}


@router.get("/children")
async def get_project_child_sessions(
    parent_session_id: str = "",
    limit: int = 100,
):
    from remy.core.worker_tasks import list_child_sessions

    return {
        "children": list_child_sessions(
            owner_project_id=current_project_id(),
            parent_session_id=str(parent_session_id or ""),
            limit=max(1, min(int(limit), 500)),
        )
    }


@router.get("/children/{child_id}")
async def get_project_child_session(child_id: str):
    from remy.core.worker_tasks import get_child_session

    child = get_child_session(child_id, owner_project_id=current_project_id())
    if not child:
        raise HTTPException(status_code=404, detail="Child session not found in this project.")
    return {"child": child}


@router.get("/children/{child_id}/report")
async def get_project_child_report(child_id: str):
    from remy.core.worker_tasks import get_child_report

    try:
        report = get_child_report(child_id, owner_project_id=current_project_id())
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Child session not found in this project.",
        ) from exc
    return report


@router.post("/children/{child_id}/follow-up")
async def follow_up_project_child(child_id: str, payload: ChildFollowUpRequest):
    from remy.core.worker_tasks import follow_up_child_session

    try:
        return follow_up_child_session(
            child_id,
            payload.message,
            owner_project_id=current_project_id(),
            confirm_side_effects=payload.confirm_side_effects,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Child session not found in this project.",
        ) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/children/{child_id}/interrupt")
async def interrupt_project_child(child_id: str, payload: StopRunRequest):
    from remy.core.worker_tasks import interrupt_child_session

    try:
        return interrupt_child_session(
            child_id,
            owner_project_id=current_project_id(),
            reason=payload.reason,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Child session not found in this project.",
        ) from exc


@router.post("/children/{child_id}/resume")
async def resume_project_child(child_id: str, payload: ResumeChildRequest):
    from remy.core.worker_tasks import resume_child_session

    try:
        task = resume_child_session(
            child_id,
            owner_project_id=current_project_id(),
            confirm_side_effects=payload.confirm_side_effects,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Child session not found in this project.",
        ) from exc
    except (PermissionError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"child_id": child_id, "task": task}
