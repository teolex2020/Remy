"""
Diagnostics routes — health probes, diagnostics, audit, metrics, eval, end-session.
"""

import asyncio
import logging
import os
import platform
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from remy.web.routes._helpers import _get_api, run_lambda_in_thread, _TIMEOUT_FAST, run_in_thread

logger = logging.getLogger("WebAPI")

router = APIRouter()


def _project_scope() -> dict:
    """Describe the MicroBrain that owns project-facing diagnostics."""
    from remy.core.microbrain import current_project_id
    from remy.core.project_store import get_project_store

    project = get_project_store().require_project(current_project_id())
    return {
        "kind": "project",
        "project_id": project.project_id,
        "brain_id": project.brain_id,
        "project_name": project.name,
    }


@router.get("/health")
async def health_check():
    """Lightweight liveness probe."""
    api = _get_api()
    return {"status": "ok", "uptime_sec": int(time.time() - api._start_time)}


@router.get("/ready")
async def readiness_check():
    """Readiness probe — confirms the service can handle requests."""
    api = _get_api()
    checks = {}

    try:
        await run_lambda_in_thread(lambda: api.brain.count())
        checks["brain"] = "ok"
    except Exception as e:
        checks["brain"] = f"error: {e}"

    try:
        api.get_session_manager()
        checks["session_manager"] = "ok"
    except RuntimeError:
        checks["session_manager"] = "not initialized"

    api_key = api.settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
    checks["api_key"] = "ok" if api_key else "missing"

    all_ok = all(v == "ok" for v in checks.values())
    status_code = 200 if all_ok else 503

    return JSONResponse(
        content={"status": "ready" if all_ok else "not ready", "checks": checks},
        status_code=status_code,
    )


@router.get("/ping")
async def ping():
    """Heartbeat endpoint to keep the web session alive."""
    api = _get_api()
    try:
        mgr = api.get_session_manager()
        if mgr:
            # get_or_create_session automatically updates last_activity
            sess = mgr.get_or_create_session()
            return {"status": "ok", "session": sess.session_id}
    except Exception as e:
        logger.debug("Ping error: %s", e)
    return {"status": "ok", "session": None}


@router.get("/agent/checkpoint")
async def get_agent_checkpoint():
    """Return a compact, non-sensitive durable-execution status."""
    api = _get_api()
    manager = api.get_session_manager()
    session = manager.get_or_create_session()

    from remy.core.agent import get_durable_agent_graph
    from remy.core.durable_graph_runtime import (
        durable_config,
        durable_thread_id,
        get_durable_graph_runtime,
    )
    from remy.core.microbrain import current_project_id
    from remy.core.session_event_store import CheckpointProjection, get_session_event_store

    graph = await get_durable_agent_graph("desktop")
    config = durable_config(session.session_id, "desktop")
    snapshot = await graph.aget_state(config)
    tasks = list(snapshot.tasks or ())
    interrupt_count = sum(len(task.interrupts or ()) for task in tasks)
    if interrupt_count:
        status = "waiting_for_input"
    elif snapshot.next:
        status = "resumable"
    elif snapshot.values:
        status = "completed"
    else:
        status = "empty"
    direct = {
        "status": status,
        "thread_id": durable_thread_id(session.session_id, "desktop"),
        "next": list(snapshot.next or ()),
        "pending_tasks": [
            {
                "name": task.name,
                "has_error": bool(task.error),
                "interrupts": len(task.interrupts or ()),
            }
            for task in tasks
        ],
        "checkpoint_created_at": snapshot.created_at,
    }
    await get_durable_graph_runtime().record_checkpoint_status(
        session_id=session.session_id,
        channel="desktop",
        project_id=current_project_id(),
        status=status,
        next_nodes=direct["next"],
        pending_tasks=direct["pending_tasks"],
        checkpoint_created_at=snapshot.created_at,
    )
    try:
        projected = get_session_event_store().project(
            CheckpointProjection(),
            project_id=current_project_id(),
            session_id=session.session_id,
        )
        return projected or direct
    except Exception:
        return direct


@router.get("/chat/brain-voice")
async def get_brain_voice(locale: str = "en", limit: int = 10):
    """Return pending proactive brain-voice events for the chat view.

    The desktop frontend polls this endpoint opportunistically. Returning an
    empty event list keeps the UI healthy even when no proactive inbox is wired.
    """
    return {"events": [], "locale": locale, "limit": max(0, min(int(limit or 10), 50))}


@router.post("/chat/brain-voice/{event_id}/ack")
async def ack_brain_voice(event_id: str):
    """Acknowledge a proactive brain-voice event."""
    return {"ok": True, "event_id": event_id}


@router.get("/metrics")
async def get_metrics():
    """Prometheus text exposition format metrics."""
    from fastapi.responses import PlainTextResponse

    from remy.core.metrics import collect_metrics

    return PlainTextResponse(
        content=collect_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get("/task-metrics")
async def get_task_metrics():
    """Per-family task execution metrics (completion rate, blocked rate, etc)."""
    try:
        from remy.core.combined_runner import get_goal_runtime_snapshot
        from remy.core.microbrain import bind_project, current_project_id
        from remy.core.task_metrics import task_metrics

        project_id = current_project_id()
        with bind_project(project_id):
            metrics = task_metrics.get_all()
            goals = await run_in_thread(
                lambda: get_goal_runtime_snapshot(goal_limit=5, approval_limit=10)
            )
            active_goals = int(goals.get("active", 0) or 0)
            blocked_goals = int(goals.get("blocked", 0) or 0)
            total_goals = int(goals.get("total", 0) or 0)

            if "totals" in metrics:
                metrics["totals"]["active_goals"] = active_goals
                metrics["totals"]["blocked_goals"] = blocked_goals
                metrics["totals"]["total_goals"] = total_goals
            metrics["scope"] = _project_scope()
        return metrics
    except Exception as e:
        return {"error": str(e)}


@router.get("/task-metrics/{family}")
async def get_task_metrics_family(family: str):
    """Task metrics for a specific family."""
    try:
        from remy.core.task_metrics import task_metrics

        return {
            **task_metrics.get_family(family),
            "scope": _project_scope(),
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/execution-log")
async def get_execution_log(limit: int = 50, pack: str | None = None):
    """Structured per-cycle execution log."""
    try:
        from remy.core.execution_log import execution_log

        if pack:
            return {
                "scope": _project_scope(),
                "entries": execution_log.get_by_pack(pack, limit=limit),
            }
        return {
            "scope": _project_scope(),
            "entries": execution_log.get_recent(limit=limit),
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/execution-log/summary")
async def get_execution_log_summary():
    """Pack-level execution summary from run log."""
    try:
        from remy.core.execution_log import execution_log

        return {
            "scope": _project_scope(),
            "packs": execution_log.get_pack_summary(),
            "step_efficiency": execution_log.get_step_efficiency(),
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/goal-history/{goal_id}")
async def get_goal_history_endpoint(goal_id: str):
    """Per-goal execution attempt history."""
    try:
        from remy.core.task_memory import get_goal_history, get_goal_history_summary

        return {
            "summary": get_goal_history_summary(goal_id),
            "attempts": get_goal_history(goal_id),
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/end-session")
async def end_session():
    """Close active session."""
    api = _get_api()
    try:
        manager = api.get_session_manager()
        await manager.suspend_session()
        return {"ok": True}
    except Exception as e:
        logger.warning(f"end-session failed: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/health/detailed")
async def detailed_health():
    """Detailed system health — tool health, circuit breakers, budget, LLM status."""
    from remy.core.error_escalation import get_detailed_health

    return get_detailed_health()


@router.get("/diagnostics")
async def get_diagnostics():
    """System diagnostics and health check."""
    api = _get_api()
    from remy.core.brain_tools import get_registry
    from remy.core.browser_failure_memory import (
        get_browser_failure_report,
        get_browser_success_report,
    )

    api_key = api.settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")

    try:
        record_count = await run_lambda_in_thread(lambda: api.brain.count())
        brain_status = "ok"
    except Exception as e:
        record_count = 0
        brain_status = f"error: {e}"

    try:
        registry = get_registry()
        tool_count = registry.tool_count
    except Exception:
        tool_count = 0

    uptime_sec = int(time.time() - api._start_time)
    hours, remainder = divmod(uptime_sec, 3600)
    minutes, seconds = divmod(remainder, 60)
    from remy.core.project_store import (
        LOCAL_BRAIN_PROVIDER,
        get_project_store,
        local_brain_path,
    )

    active_project = get_project_store().get_active_project()
    brain_location = (
        str(local_brain_path(active_project.project_id))
        if active_project.brain_provider == LOCAL_BRAIN_PROVIDER
        else active_project.brain_uri
    )

    return {
        "status": "ok" if api_key and brain_status == "ok" else "degraded",
        "uptime": f"{hours}h {minutes}m {seconds}s",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "api_key_configured": bool(api_key),
        "model": api.settings.SUMMARY_MODEL,
        "brain": {
            "status": brain_status,
            "records": record_count,
            "path": brain_location,
            "provider": active_project.brain_provider,
            "brain_uri": active_project.brain_uri,
        },
        "tools": tool_count,
        "browser_failures": get_browser_failure_report(limit=5),
        "browser_successes": get_browser_success_report(limit=5),
        "telegram_configured": bool(api.settings.TELEGRAM_BOT_TOKEN),
        "sandbox_dir": str(api.settings.SANDBOX_DIR),
    }


# ============== AUDIT TRAIL ==============


@router.get("/audit/logs")
async def get_audit_logs(n: int = 20, tool: str | None = None):
    """Recent audit log entries."""
    from remy.core.audit_trail import get_audit_logger
    from remy.core.microbrain import current_project_id

    session = _get_api().get_session_manager().get_or_create_session()
    return {"logs": get_audit_logger().get_recent_logs(
        n=n,
        tool_name=tool,
        project_id=current_project_id(),
        session_id=session.session_id,
    )}


@router.get("/audit/integrity")
async def get_audit_integrity():
    """Check audit log integrity."""
    from remy.core.audit_trail import get_audit_logger
    from remy.core.microbrain import current_project_id

    session = _get_api().get_session_manager().get_or_create_session()
    return get_audit_logger().verify_integrity(
        project_id=current_project_id(),
        session_id=session.session_id,
    )


@router.get("/audit/summary")
async def get_audit_summary():
    """Aggregate audit stats."""
    from remy.core.audit_trail import get_audit_logger
    from remy.core.microbrain import current_project_id

    session = _get_api().get_session_manager().get_or_create_session()
    return get_audit_logger().get_summary(
        project_id=current_project_id(),
        session_id=session.session_id,
    )


# ============== EVALUATION METRICS ==============


@router.get("/eval-metrics")
async def get_eval_metrics(channel: str | None = None, limit: int = 50):
    """Aggregated evaluation metrics for agent responses."""
    from remy.core.eval_metrics import get_metrics_summary

    return {
        **get_metrics_summary(channel=channel, limit=limit),
        "scope": _project_scope(),
    }


@router.get("/harness-eval-history")
async def get_harness_eval_history(limit: int = 20):
    """Recent harness eval runs and scenario trend counts."""
    from remy.core.harness_eval_history import get_harness_eval_history_summary

    return get_harness_eval_history_summary(limit=limit)
