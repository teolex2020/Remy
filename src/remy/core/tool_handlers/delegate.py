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
