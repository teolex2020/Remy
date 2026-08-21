"""Durable task handles for background delegated worker groups."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from dataclasses import asdict
from typing import Any

from remy.config.settings import settings
from remy.core.cancellation import (
    CancellationToken,
    OperationCancelled,
    bind_cancellation_token,
)
from remy.core.run_envelope import (
    ACTIVE_RUN_STATUSES,
    TERMINAL_RUN_STATUSES,
    RunCoordinator,
    RunLimits,
    finish_run,
    get_run,
    register_run_stop,
    start_run,
    unregister_run_stop,
)
from remy.core.worker import WorkerResult, WorkerTask

logger = logging.getLogger(__name__)

_RUNTIME_LOCK = threading.RLock()
_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None
_FUTURES: dict[str, concurrent.futures.Future] = {}
_TOKENS: dict[str, CancellationToken] = {}


def _runtime_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _EXECUTOR
    with _RUNTIME_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                max_workers=max(2, min(int(settings.WORKER_MAX_PARALLEL), 8)),
                thread_name_prefix="remy-worker-task",
            )
        return _EXECUTOR


def _task_payload(task: WorkerTask) -> dict[str, Any]:
    payload = asdict(task)
    payload["instruction"] = str(payload.get("instruction") or "")[:20_000]
    payload["context"] = str(payload.get("context") or "")[:20_000]
    return payload


def _task_from_payload(payload: dict[str, Any]) -> WorkerTask:
    return WorkerTask(
        role=str(payload.get("role") or "researcher"),
        instruction=str(payload.get("instruction") or ""),
        context=str(payload.get("context") or ""),
        approval_mode=str(payload.get("approval_mode") or "none"),
        delegation_depth=max(0, int(payload.get("delegation_depth") or 0)),
        allowed_tools=tuple(str(name) for name in payload.get("allowed_tools") or ()),
    )


def public_task_handle(run: dict[str, Any]) -> dict[str, Any]:
    """Return one stable operator-facing task contract."""
    status = str(run.get("status") or "")
    metadata = dict(run.get("metadata") or {})
    return {
        "task_id": str(run.get("run_id") or ""),
        "run_id": str(run.get("run_id") or ""),
        "attempt_id": str(run.get("attempt_id") or ""),
        "kind": str(run.get("kind") or ""),
        "status": status,
        "phase": str(run.get("phase") or ""),
        "current_step": str(run.get("current_step") or ""),
        "goal": str(run.get("goal") or ""),
        "owner_project_id": str(run.get("owner_project_id") or ""),
        "conversation_id": str(run.get("conversation_id") or ""),
        "usage": dict(run.get("usage") or {}),
        "limits": dict(run.get("limits") or {}),
        "results": list(run.get("artifacts") or []),
        "error": str(run.get("error") or ""),
        "stop_reason": str(run.get("stop_reason") or ""),
        "started_at": str(run.get("started_at") or ""),
        "updated_at": str(run.get("updated_at") or ""),
        "finished_at": str(run.get("finished_at") or ""),
        "resumed_from": str(metadata.get("resumed_from") or ""),
        "child_id": str(metadata.get("child_id") or ""),
        "child_generation": int(metadata.get("child_generation") or 0),
        "continuation_reason": str(metadata.get("continuation_reason") or ""),
        "can_cancel": status in ACTIVE_RUN_STATUSES,
        "can_resume": status in TERMINAL_RUN_STATUSES and bool(metadata.get("tasks")),
    }


def get_worker_task(task_id: str, *, owner_project_id: str) -> dict[str, Any] | None:
    run = get_run(task_id, owner_project_id=owner_project_id)
    if not run or run.get("kind") != "worker_group":
        return None
    handle = public_task_handle(run)
    child_id = str(handle.get("child_id") or "")
    if child_id and handle.get("status") in TERMINAL_RUN_STATUSES:
        try:
            from remy.core.child_sessions import CHILD_ACTIVE_STATUSES, get_child_session_store

            with _RUNTIME_LOCK:
                future = _FUTURES.get(task_id)
            child = get_child_session_store().get(
                child_id,
                owner_project_id=owner_project_id,
            )
            if (
                (future is not None and not future.done())
                or (
                    child
                    and child.get("status") in CHILD_ACTIVE_STATUSES
                    and child.get("current_run_id") == task_id
                )
            ):
                handle["status"] = "settling"
                handle["phase"] = "settling"
                handle["can_cancel"] = False
                handle["can_resume"] = False
        except Exception:
            pass
    return handle


def _result_artifacts(results: list[WorkerResult]) -> list[dict[str, Any]]:
    return [
        {
            "kind": "worker_result",
            "role": result.role,
            "status": result.status,
            "output": result.output[:8_000],
            "tool_calls": int(result.tool_calls),
            "elapsed_sec": float(result.elapsed_sec),
        }
        for result in results
    ]


def _continuation_text(
    run_id: str,
    results: list[WorkerResult],
    *,
    status: str = "completed",
    error: str = "",
) -> str:
    lines = [f"Background delegated task {run_id} finished with status {status}."]
    for result in results:
        lines.append(
            f"[{result.role} · {result.status}] {result.output[:2_000]}"
        )
    if error:
        lines.append(f"Error: {str(error)[:2_000]}")
    return "\n\n".join(lines)


def _record_child_trajectory(
    child: dict[str, Any],
    *,
    event_type: str,
    status: str,
    run_id: str = "",
    attempt_id: str = "",
    generation: int = 0,
    summary: str = "",
    details: dict[str, Any] | None = None,
) -> None:
    try:
        from remy.core.trajectory_store import get_trajectory_store

        get_trajectory_store().record_child_lifecycle(
            session_id=str(child.get("parent_session_id") or f"child:{child['child_id']}"),
            project_id=str(child.get("owner_project_id") or ""),
            child_id=str(child.get("child_id") or ""),
            event_type=event_type,
            status=status,
            run_id=run_id,
            attempt_id=attempt_id,
            generation=generation,
            summary=summary,
            details=details,
        )
    except Exception as exc:
        logger.debug("Child Trajectory event skipped: %s", exc)


def _side_effect_confirmation_required(child: dict[str, Any], confirmed: bool) -> None:
    if child.get("idempotency_class") == "side_effecting" and not confirmed:
        raise PermissionError(
            "Continuing an executor child may repeat side effects; "
            "explicit confirmation is required"
        )


def _continuation_tasks(child: dict[str, Any], message: str = "") -> list[WorkerTask]:
    previous_report = dict(child.get("latest_report") or {})
    previous_summary = str(previous_report.get("summary") or "")[:8_000]
    tasks = []
    for item in list(child.get("spec") or []):
        if not isinstance(item, dict):
            continue
        instruction = str(item.get("instruction") or "").strip()
        if message:
            instruction += f"\n\nParent follow-up:\n{message[:12_000]}"
        context = str(item.get("context") or "")
        if previous_summary:
            context = (
                f"{context}\n\nPrevious child report:\n{previous_summary}"
                if context
                else f"Previous child report:\n{previous_summary}"
            )
        tasks.append(
            WorkerTask(
                role=str(item.get("role") or "researcher"),
                instruction=instruction,
                context=context,
                approval_mode=str(item.get("approval_mode") or "none"),
                delegation_depth=max(0, int(item.get("delegation_depth") or 0)),
                allowed_tools=tuple(
                    str(name) for name in item.get("allowed_tools") or ()
                ),
            )
        )
    return tasks


def _settle_worker_child(
    *,
    run: dict[str, Any],
    final_run: dict[str, Any],
    results: list[WorkerResult],
) -> None:
    child_id = str((run.get("metadata") or {}).get("child_id") or "")
    if not child_id:
        return
    from remy.core.child_sessions import get_child_session_store
    from remy.core.execution_ledger import get_execution_ledger

    owner_project_id = str(run.get("owner_project_id") or "")
    status = str(final_run.get("status") or "failed")
    error = str(final_run.get("error") or "")
    artifacts = list(final_run.get("artifacts") or _result_artifacts(results))
    summary = _continuation_text(run["run_id"], results, status=status, error=error)
    report = {
        "child_id": child_id,
        "run_id": str(run.get("run_id") or ""),
        "attempt_id": str(run.get("attempt_id") or ""),
        "generation": int((run.get("metadata") or {}).get("child_generation") or 0),
        "status": status,
        "summary": summary,
        "error": error,
        "results": artifacts,
        "usage": dict(final_run.get("usage") or {}),
        "stop_reason": str(final_run.get("stop_reason") or ""),
    }
    store = get_child_session_store()
    child, inserted = store.settle_attempt(
        child_id,
        owner_project_id=owner_project_id,
        run_id=str(run.get("run_id") or ""),
        status=status,
        report=report,
    )
    if not inserted:
        return
    store.enqueue_message(
        child_id,
        owner_project_id=owner_project_id,
        direction="child_to_parent",
        kind="report",
        content=summary,
        metadata={"run_id": run["run_id"], "status": status},
    )
    get_execution_ledger().enqueue_continuation(
        session_id=str(run.get("conversation_id") or ""),
        owner_project_id=owner_project_id,
        brain_id=str(run.get("brain_id") or ""),
        kind="child_session",
        source_id=child_id,
        content=summary,
        metadata={
            "delivery_target": "web",
            "task_id": str(run.get("run_id") or ""),
            "child_id": child_id,
            "status": status,
        },
    )
    _record_child_trajectory(
        child,
        event_type="child.settled",
        status=status,
        run_id=str(run.get("run_id") or ""),
        attempt_id=str(run.get("attempt_id") or ""),
        generation=report["generation"],
        summary=summary,
        details={"stop_reason": report["stop_reason"], "result_count": len(artifacts)},
    )


def _launch_pending_child_followups(
    child_id: str,
    *,
    owner_project_id: str,
) -> dict[str, Any] | None:
    from remy.core.child_sessions import get_child_session_store

    store = get_child_session_store()
    child = store.get(child_id, owner_project_id=owner_project_id)
    if not child or child.get("status") not in {"ready", "interrupted", "failed"}:
        return None
    messages = store.claim_messages(
        child_id,
        owner_project_id=owner_project_id,
        direction="parent_to_child",
        kinds={"follow_up"},
        limit=20,
    )
    if not messages:
        return None
    combined = "\n\n".join(str(message.get("content") or "") for message in messages)
    try:
        return launch_worker_tasks(
            _continuation_tasks(child, combined),
            session_id=str(child.get("parent_session_id") or ""),
            channel="web",
            owner_project_id=owner_project_id,
            brain_id=str(child.get("brain_id") or ""),
            child_id=child_id,
            continuation_reason="follow_up",
        )
    except Exception:
        store.release_messages(
            child_id,
            owner_project_id=owner_project_id,
            message_ids=[str(message.get("message_id") or "") for message in messages],
        )
        raise


async def _execute_worker_group(
    *,
    run: dict[str, Any],
    tasks: list[WorkerTask],
    session_id: str,
    channel: str,
    token: CancellationToken,
) -> None:
    from remy.core.microbrain import bind_project
    from remy.core.worker import execute_workers

    run_id = run["run_id"]
    attempt_id = run["attempt_id"]
    owner_project_id = run["owner_project_id"]
    coordinator = RunCoordinator(attempt_id)
    started_workers = 0
    results: list[WorkerResult] = []
    final_run: dict[str, Any] | None = None

    def close_worker_slots() -> None:
        nonlocal started_workers
        while started_workers > 0:
            coordinator.worker_finished()
            started_workers -= 1

    try:
        with bind_project(owner_project_id), bind_cancellation_token(token):
            coordinator.step(
                f"Delegating {len(tasks)} worker(s)",
                signature="roles:" + ",".join(task.role for task in tasks),
            )
            for task in tasks:
                coordinator.worker_started(task.role)
                started_workers += 1
            results = await execute_workers(tasks, session_id, channel)
            token.raise_if_cancelled()
            close_worker_slots()
            artifacts = _result_artifacts(results)
            failed = [item for item in results if item.status not in {"success"}]
            succeeded = [item for item in results if item.status == "success"]
            status = (
                "completed"
                if not failed
                else "completed_with_limits"
                if succeeded
                else "failed"
            )
            stop_reason = (
                ""
                if not failed
                else "worker_partial_failure"
                if succeeded
                else "worker_failure"
            )
            final_run = finish_run(
                attempt_id,
                status=status,
                stop_reason=stop_reason,
                error=("All delegated workers failed" if status == "failed" else ""),
                output_ref=f"worker-task:{run_id}",
                artifacts=artifacts,
            )
    except (OperationCancelled, asyncio.CancelledError) as exc:
        current = get_run(run_id, owner_project_id=owner_project_id)
        if current and current.get("status") not in TERMINAL_RUN_STATUSES:
            close_worker_slots()
            final_run = finish_run(
                attempt_id,
                status="cancelled",
                stop_reason="user_stopped",
                error=str(exc),
            )
    except Exception as exc:  # noqa: BLE001 - terminal receipt is mandatory.
        logger.exception("Background worker task %s failed", run_id)
        current = get_run(run_id, owner_project_id=owner_project_id)
        if current and current.get("status") not in TERMINAL_RUN_STATUSES:
            close_worker_slots()
            final_run = finish_run(attempt_id, status="failed", error=str(exc))
    finally:
        current = get_run(run_id, owner_project_id=owner_project_id)
        if current and current.get("status") in ACTIVE_RUN_STATUSES:
            try:
                close_worker_slots()
            except Exception:
                pass
        unregister_run_stop(run_id)
        final_run = final_run or get_run(run_id, owner_project_id=owner_project_id)
        if final_run and final_run.get("status") in TERMINAL_RUN_STATUSES:
            try:
                _settle_worker_child(run=run, final_run=final_run, results=results)
            except Exception:
                logger.exception("Could not settle child session for run %s", run_id)
            child_id = str((run.get("metadata") or {}).get("child_id") or "")
            if child_id:
                try:
                    _launch_pending_child_followups(
                        child_id,
                        owner_project_id=owner_project_id,
                    )
                except Exception:
                    logger.exception("Could not launch queued follow-up for %s", child_id)


def launch_worker_tasks(
    tasks: list[WorkerTask],
    *,
    session_id: str,
    channel: str,
    owner_project_id: str = "",
    brain_id: str = "",
    resumed_from: str = "",
    child_id: str = "",
    continuation_reason: str = "initial",
) -> dict[str, Any]:
    """Launch a worker group and return immediately with its durable handle."""
    from remy.core.microbrain import current_project_id
    from remy.core.project_store import get_project_store
    from remy.core.child_sessions import get_child_session_store

    valid = [task for task in tasks if task.instruction.strip()]
    if not valid:
        raise ValueError("At least one worker task is required")
    valid = valid[: max(1, int(settings.WORKER_MAX_PARALLEL))]
    owner = get_project_store().require_project(owner_project_id or current_project_id())
    if brain_id and brain_id != owner.brain_id:
        raise ValueError("Worker task brain does not belong to the selected project")
    payload = [_task_payload(task) for task in valid]
    goal = "; ".join(task.instruction.strip() for task in valid)[:4_000]
    idempotency_class = (
        "side_effecting" if any(task.role == "executor" for task in valid) else "read_only"
    )
    child_store = get_child_session_store()
    created_child = False
    if child_id:
        child = child_store.get(child_id, owner_project_id=owner.project_id)
        if not child:
            raise KeyError(child_id)
        if child.get("brain_id") != owner.brain_id:
            raise ValueError("Child session brain does not belong to the selected project")
    else:
        child = child_store.create(
            owner_project_id=owner.project_id,
            brain_id=owner.brain_id,
            parent_session_id=session_id,
            spec=payload,
            idempotency_class=idempotency_class,
        )
        child_id = str(child["child_id"])
        created_child = True
        child_store.enqueue_message(
            child_id,
            owner_project_id=owner.project_id,
            direction="parent_to_child",
            kind="initial",
            content=goal,
            metadata={"task_count": len(valid)},
        )
        child_store.claim_messages(
            child_id,
            owner_project_id=owner.project_id,
            direction="parent_to_child",
            kinds={"initial"},
        )
        _record_child_trajectory(
            child,
            event_type="child.created",
            status="ready",
            summary=goal,
            details={"task_count": len(valid), "idempotency_class": idempotency_class},
        )

    generation = child_store.claim_attempt(
        child_id,
        owner_project_id=owner.project_id,
        reason=str(continuation_reason or "initial"),
    )
    try:
        run = start_run(
            kind="worker_group",
            source_id=session_id or "delegation",
            goal=goal,
            owner_project_id=owner.project_id,
            brain_id=owner.brain_id,
            conversation_id=session_id,
            channel=channel,
            limits=RunLimits(
                max_turns=max(10, len(valid) * int(settings.WORKER_MAX_TOOL_ITERATIONS) + 5),
                token_budget=max(32_000, len(valid) * 64_000),
                max_parallel_workers=len(valid),
            ),
            idempotency_class=idempotency_class,
            metadata={
                "tasks": payload,
                "resumed_from": str(resumed_from or ""),
                "child_id": child_id,
                "child_generation": generation,
                "continuation_reason": str(continuation_reason or "initial"),
            },
        )
    except Exception as exc:
        child_store.settle_attempt(
            child_id,
            owner_project_id=owner.project_id,
            run_id=f"unbound-generation-{generation}",
            status="failed",
            report={"status": "failed", "error": str(exc), "summary": str(exc)},
        )
        raise
    child = child_store.bind_attempt(
        child_id,
        owner_project_id=owner.project_id,
        generation=generation,
        run_id=str(run["run_id"]),
        attempt_id=str(run["attempt_id"]),
        reason=str(continuation_reason or "initial"),
    )
    _record_child_trajectory(
        child,
        event_type="child.attempt_started",
        status="running",
        run_id=str(run["run_id"]),
        attempt_id=str(run["attempt_id"]),
        generation=generation,
        summary=goal,
        details={"created_child": created_child, "continuation_reason": continuation_reason},
    )
    token = CancellationToken()
    register_run_stop(run["run_id"], token.cancel)

    def runner() -> None:
        asyncio.run(
            _execute_worker_group(
                run=run,
                tasks=valid,
                session_id=session_id,
                channel=channel,
                token=token,
            )
        )

    try:
        future = _runtime_executor().submit(runner)
    except Exception as exc:
        unregister_run_stop(run["run_id"])
        final_run = finish_run(run["attempt_id"], status="failed", error=str(exc))
        _settle_worker_child(run=run, final_run=final_run, results=[])
        raise
    with _RUNTIME_LOCK:
        _TOKENS[run["run_id"]] = token
        _FUTURES[run["run_id"]] = future

    def cleanup(_future: concurrent.futures.Future) -> None:
        with _RUNTIME_LOCK:
            _TOKENS.pop(run["run_id"], None)
            _FUTURES.pop(run["run_id"], None)

    future.add_done_callback(cleanup)
    return public_task_handle(run)


def resume_worker_tasks(
    task_id: str,
    *,
    owner_project_id: str,
    confirm_side_effects: bool = False,
) -> dict[str, Any]:
    """Start a linked retry from a terminal handle's original task spec."""
    run = get_run(task_id, owner_project_id=owner_project_id)
    if not run or run.get("kind") != "worker_group":
        raise KeyError(task_id)
    if run.get("status") not in TERMINAL_RUN_STATUSES:
        raise RuntimeError("Only a terminal worker task can be resumed")
    metadata = dict(run.get("metadata") or {})
    child_id = str(metadata.get("child_id") or "")
    child = None
    if child_id:
        from remy.core.child_sessions import get_child_session_store

        child = get_child_session_store().get(
            child_id,
            owner_project_id=owner_project_id,
        )
    payloads = list(metadata.get("tasks") or [])
    tasks = (
        _continuation_tasks(child)
        if child
        else [_task_from_payload(item) for item in payloads if isinstance(item, dict)]
    )
    if not tasks:
        raise RuntimeError("The task has no resumable worker specification")
    if child:
        _side_effect_confirmation_required(child, confirm_side_effects)
    elif any(task.role == "executor" for task in tasks) and not confirm_side_effects:
        raise PermissionError(
            "Resuming an executor may repeat side effects; explicit confirmation is required"
        )
    return launch_worker_tasks(
        tasks,
        session_id=str(run.get("conversation_id") or ""),
        channel="web",
        owner_project_id=owner_project_id,
        brain_id=str(run.get("brain_id") or ""),
        resumed_from=task_id,
        child_id=child_id,
        continuation_reason="cold_resume",
    )


def get_child_session(child_id: str, *, owner_project_id: str) -> dict[str, Any] | None:
    from remy.core.child_sessions import get_child_session_store

    return get_child_session_store().get(child_id, owner_project_id=owner_project_id)


def list_child_sessions(
    *,
    owner_project_id: str,
    parent_session_id: str = "",
    limit: int = 100,
) -> list[dict[str, Any]]:
    from remy.core.child_sessions import get_child_session_store

    return get_child_session_store().list(
        owner_project_id=owner_project_id,
        parent_session_id=parent_session_id,
        limit=limit,
    )


def get_child_report(child_id: str, *, owner_project_id: str) -> dict[str, Any]:
    from remy.core.child_sessions import get_child_session_store

    return get_child_session_store().report(
        child_id,
        owner_project_id=owner_project_id,
    )


def follow_up_child_session(
    child_id: str,
    message: str,
    *,
    owner_project_id: str,
    confirm_side_effects: bool = False,
) -> dict[str, Any]:
    from remy.core.child_sessions import CHILD_ACTIVE_STATUSES, get_child_session_store

    store = get_child_session_store()
    child = store.get(child_id, owner_project_id=owner_project_id)
    if not child:
        raise KeyError(child_id)
    _side_effect_confirmation_required(child, confirm_side_effects)
    queued = store.enqueue_message(
        child_id,
        owner_project_id=owner_project_id,
        direction="parent_to_child",
        kind="follow_up",
        content=message,
        metadata={"confirm_side_effects": bool(confirm_side_effects)},
    )
    _record_child_trajectory(
        child,
        event_type="child.follow_up_queued",
        status=str(child.get("status") or "ready"),
        run_id=str(child.get("current_run_id") or ""),
        generation=int(child.get("generation") or 0),
        summary=str(message)[:4_000],
        details={"message_id": queued["message_id"]},
    )
    task = None
    if child.get("status") not in CHILD_ACTIVE_STATUSES:
        task = _launch_pending_child_followups(
            child_id,
            owner_project_id=owner_project_id,
        )
    return {
        "child": store.get(child_id, owner_project_id=owner_project_id),
        "message": queued,
        "task": task,
        "queued": task is None,
    }


def interrupt_child_session(
    child_id: str,
    *,
    owner_project_id: str,
    reason: str = "Interrupted by parent",
) -> dict[str, Any]:
    from remy.core.child_sessions import CHILD_ACTIVE_STATUSES, get_child_session_store
    from remy.core.run_envelope import request_run_stop

    store = get_child_session_store()
    child = store.get(child_id, owner_project_id=owner_project_id)
    if not child:
        raise KeyError(child_id)
    if child.get("status") not in CHILD_ACTIVE_STATUSES:
        return {"child": child, "interrupt_requested": False, "run": None}
    store.enqueue_message(
        child_id,
        owner_project_id=owner_project_id,
        direction="parent_to_child",
        kind="interrupt",
        content=reason,
    )
    child = store.mark_interrupt_requested(
        child_id,
        owner_project_id=owner_project_id,
        reason=reason,
    )
    run = request_run_stop(
        str(child.get("current_run_id") or ""),
        owner_project_id=owner_project_id,
        reason=reason,
    )
    _record_child_trajectory(
        child,
        event_type="child.interrupt_requested",
        status="interrupt_requested",
        run_id=str(child.get("current_run_id") or ""),
        generation=int(child.get("generation") or 0),
        summary=reason,
    )
    return {"child": child, "interrupt_requested": True, "run": run}


def resume_child_session(
    child_id: str,
    *,
    owner_project_id: str,
    confirm_side_effects: bool = False,
) -> dict[str, Any]:
    from remy.core.child_sessions import CHILD_ACTIVE_STATUSES, get_child_session_store

    store = get_child_session_store()
    child = store.get(child_id, owner_project_id=owner_project_id)
    if not child:
        raise KeyError(child_id)
    if child.get("status") in CHILD_ACTIVE_STATUSES:
        raise RuntimeError("Child session already has an active attempt")
    _side_effect_confirmation_required(child, confirm_side_effects)
    pending = _launch_pending_child_followups(
        child_id,
        owner_project_id=owner_project_id,
    )
    if pending:
        return pending
    tasks = _continuation_tasks(child)
    if not tasks:
        raise RuntimeError("The child session has no resumable task specification")
    return launch_worker_tasks(
        tasks,
        session_id=str(child.get("parent_session_id") or ""),
        channel="web",
        owner_project_id=owner_project_id,
        brain_id=str(child.get("brain_id") or ""),
        resumed_from=str((child.get("latest_report") or {}).get("run_id") or ""),
        child_id=child_id,
        continuation_reason="cold_resume",
    )


def recover_interrupted_child_sessions(recovered_runs: list[dict[str, Any]]) -> int:
    """Settle orphaned child attempts without replaying their side effects."""
    recovered = 0
    for run in recovered_runs:
        metadata = dict(run.get("metadata") or {})
        if run.get("kind") != "worker_group" or not metadata.get("child_id"):
            continue
        try:
            _settle_worker_child(run=run, final_run=run, results=[])
            recovered += 1
        except Exception:
            logger.exception("Could not recover child session for run %s", run.get("run_id"))
    return recovered


def shutdown_worker_task_runtime() -> None:
    """Cooperatively cancel background handles during server shutdown."""
    global _EXECUTOR
    with _RUNTIME_LOCK:
        tokens = list(_TOKENS.values())
        executor = _EXECUTOR
        _EXECUTOR = None
    for token in tokens:
        token.cancel("Remy is shutting down")
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)
