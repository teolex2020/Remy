"""
Delegate Task Handler — multi-agent worker orchestration.

Handles the delegate_task tool, spawning parallel worker sub-agents.
"""

import asyncio
import json
import logging

from remy.config.settings import settings

logger = logging.getLogger("BrainTools")


def _handle_delegate_task(args: dict, session_id: str | None, channel: str | None) -> str:
    """Handle delegate_task tool — runs outside brain_lock to avoid deadlock."""
    from remy.core.worker import WorkerTask, execute_workers

    raw_tasks = args.get("tasks", [])
    if not raw_tasks:
        return json.dumps({"error": "No tasks provided"})

    if not isinstance(raw_tasks, list):
        return json.dumps({"error": "tasks must be a list"})

    max_parallel = settings.WORKER_MAX_PARALLEL
    tasks = []
    for t in raw_tasks[:max_parallel]:
        if not isinstance(t, dict):
            continue
        role = t.get("role", "researcher")
        instruction = t.get("instruction", "")
        if not instruction:
            continue
        tasks.append(
            WorkerTask(
                role=role,
                instruction=instruction,
                context=t.get("context", ""),
                approval_mode=t.get("approval_mode", "none"),
                delegation_depth=int(t.get("_delegation_depth", 0)),
            )
        )

    if not tasks:
        return json.dumps({"error": "No valid tasks provided"})

    if bool(args.get("background", False)):
        from remy.core.worker_tasks import launch_worker_tasks

        try:
            handle = launch_worker_tasks(
                tasks,
                session_id=session_id or "",
                channel=channel or "desktop",
            )
        except Exception as exc:
            logger.exception("Could not launch background delegated task")
            return json.dumps({"error": str(exc)}, ensure_ascii=False)
        return json.dumps(
            {
                "delegated": len(tasks),
                "background": True,
                "task": handle,
                "message": (
                    "Workers are running in the background. Use child_id for follow-up, "
                    "interrupt, cold resume, and reports; task_id identifies this attempt. "
                    "The settlement report will be delivered to this conversation."
                ),
            },
            ensure_ascii=False,
        )

    # Run workers — handle both sync and async contexts
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                asyncio.run, execute_workers(tasks, session_id or "", channel or "")
            )
            total_timeout = settings.WORKER_TIMEOUT_SEC * len(tasks) + 10
            results = future.result(timeout=total_timeout)
    else:
        results = asyncio.run(execute_workers(tasks, session_id or "", channel or ""))

    return json.dumps(
        {
            "delegated": len(tasks),
            "results": [
                {
                    "role": r.role,
                    "status": r.status,
                    "output": r.output[:1500],
                    "tool_calls": r.tool_calls,
                    "elapsed_sec": round(r.elapsed_sec, 1),
                }
                for r in results
            ],
        },
        ensure_ascii=False,
    )


def _handle_child_session_tool(
    name: str,
    args: dict,
    session_id: str | None,
    channel: str | None,
) -> str:
    """Handle continuable child lifecycle tools outside the brain lock."""
    from remy.core.microbrain import current_project_id
    from remy.core.worker_tasks import (
        follow_up_child_session,
        get_child_report,
        interrupt_child_session,
        list_child_sessions,
        resume_child_session,
    )

    owner_project_id = current_project_id()
    child_id = str(args.get("child_id") or "").strip()
    try:
        if name == "list_child_sessions":
            return json.dumps(
                {
                    "children": list_child_sessions(
                        owner_project_id=owner_project_id,
                        parent_session_id=str(args.get("parent_session_id") or "").strip(),
                        limit=max(1, min(int(args.get("limit") or 50), 100)),
                    )
                },
                ensure_ascii=False,
            )
        if not child_id:
            return json.dumps({"error": "child_id is required"})
        if name == "get_child_report":
            return json.dumps(
                get_child_report(child_id, owner_project_id=owner_project_id),
                ensure_ascii=False,
            )
        if name == "follow_up_child_session":
            message = str(args.get("message") or "").strip()
            if not message:
                return json.dumps({"error": "message is required"})
            result = follow_up_child_session(
                child_id,
                message,
                owner_project_id=owner_project_id,
                confirm_side_effects=bool(args.get("confirm_side_effects", False)),
            )
        elif name == "interrupt_child_session":
            result = interrupt_child_session(
                child_id,
                owner_project_id=owner_project_id,
                reason=str(args.get("reason") or "Interrupted by parent"),
            )
        elif name == "resume_child_session":
            result = {
                "child_id": child_id,
                "task": resume_child_session(
                    child_id,
                    owner_project_id=owner_project_id,
                    confirm_side_effects=bool(args.get("confirm_side_effects", False)),
                ),
            }
        else:
            return json.dumps({"error": f"Unknown child session tool: {name}"})
        return json.dumps(result, ensure_ascii=False)
    except KeyError:
        return json.dumps(
            {"error": "Child session not found in this project", "child_id": child_id},
            ensure_ascii=False,
        )
    except (PermissionError, RuntimeError, ValueError) as exc:
        return json.dumps(
            {"error": str(exc), "child_id": child_id},
            ensure_ascii=False,
        )
