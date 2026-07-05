"""
Combined Runner — runs Autonomous Loop + Telegram Bot + Web GUI in parallel.

Coordinates multiple channels in a single asyncio event loop so the user can
interact with the agent (via Telegram or Web) while the autonomous loop runs
in the background.

Usage:
    remy --autonomous --telegram --web
    remy --autonomous --telegram
    remy --autonomous --web
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time

import uvicorn

from remy.config.settings import settings
from remy.core.agent_tools import brain, brain_lock, get_brain_startup_status
from remy.core.gateway import channel_degraded, channel_running, channel_starting, channel_stopped
from remy.core.notification_router import set_web_runtime_enabled
from remy.core.pinchtab_service import ensure_pinchtab_running, shutdown_pinchtab

logger = logging.getLogger("CombinedRunner")

# ============== RUNTIME AUTO-LOOP REGISTRY ==============
# Allows the web toggle endpoint to start/stop autonomy at runtime.

_auto_loop = None
_auto_task: asyncio.Task | None = None
_operator_watch_task: asyncio.Task | None = None
_autonomy_runtime = None
_autonomy_version_override: str | None = None
_shutdown_event: asyncio.Event | None = None
_autonomy_restarting: bool = False  # True during session-restart gap


def _configured_autonomy_version() -> str:
    """Return the configured autonomy engine version."""
    if _autonomy_version_override in {"v2", "v3"}:
        return _autonomy_version_override
    return "v3" if getattr(settings, "AUTONOMY_V3", False) is True else "v2"


def get_registry():
    """Lazy registry accessor so banner printing does not force eager tool loading at import time."""
    from remy.core.brain_tools import get_registry as _get_registry

    return _get_registry()


def get_auto_loop():
    """Return the current AutonomousLoop instance (or None)."""
    return _auto_loop


def get_autonomy_runtime():
    """Return the current autonomy runtime container/dict when available."""
    return _autonomy_runtime


def get_autonomy_runtime_component(name: str, default=None):
    """Resolve a runtime component across v2 loop and v3 runtime shapes."""
    runtime = _autonomy_runtime
    if runtime is not None:
        if isinstance(runtime, dict) and name in runtime:
            return runtime[name]
        if hasattr(runtime, name):
            return getattr(runtime, name)

    loop = _auto_loop
    if loop is not None and hasattr(loop, name):
        return getattr(loop, name)

    chief = getattr(loop, "chief", None)
    if chief is not None and hasattr(chief, name):
        return getattr(chief, name)

    return default


def is_autonomy_running() -> bool:
    """Return whether the current autonomy loop is running across v2/v3 shapes."""
    if _autonomy_restarting:
        return True  # session ended, restart in progress — still logically "running"
    task = _auto_task
    if task is not None and not task.done():
        return True  # startup task is alive — treat as control-plane running
    loop = _auto_loop
    if loop is None:
        return False
    return bool(getattr(loop, "running", None) or getattr(loop, "_running", False))


def get_autonomy_runtime_version() -> str:
    """Return the currently active autonomy runtime version label."""
    runtime = _autonomy_runtime
    if isinstance(runtime, dict) and runtime.get("chief") is not None:
        return "v3"
    loop = _auto_loop
    if loop is not None and hasattr(loop, "chief"):
        return "v3"
    return _configured_autonomy_version()


def get_autonomy_control_state() -> dict:
    """Return normalized control-plane state for the autonomy runtime."""
    status = get_autonomy_status_snapshot()
    return {
        "running": bool(status.get("running", False)),
        "session_id": status.get("session_id"),
        "active_version": status.get("version", get_autonomy_runtime_version()),
        "configured_version": _configured_autonomy_version(),
        "runtime_loaded": _autonomy_runtime is not None,
        "maintenance_only": bool(
            status.get("maintenance_only", False)
            or status.get("_maintenance_only", False)
        ),
    }


def get_autonomy_status_snapshot() -> dict:
    """Best-effort normalized status snapshot for the active autonomy runtime."""
    loop = _auto_loop
    if loop is None:
        return {
            "running": False,
            "session_id": None,
            "version": get_autonomy_runtime_version(),
        }

    status_fn = getattr(loop, "status", None)
    if callable(status_fn):
        data = dict(status_fn() or {})
    else:
        data = {}

    # Override running with authoritative check (handles restart gap)
    data["running"] = is_autonomy_running()
    data.setdefault("session_id", getattr(loop, "session_id", None))
    data.setdefault("version", get_autonomy_runtime_version())
    return data


def _instantiate_autonomy_runtime(version_override: str | None = None):
    """Create a fresh autonomy runtime honoring the configured engine version."""
    version = version_override if version_override in {"v2", "v3"} else _configured_autonomy_version()
    if version == "v3":
        from remy.core_v3.runtime.bootstrap import create_v3_runtime, load_v2_state

        runtime = create_v3_runtime()
        load_v2_state(runtime)
        return runtime, runtime["loop"], "v3"

    from remy.core.autonomy import AutonomousLoop

    loop = AutonomousLoop()
    return {"loop": loop}, loop, "v2"


def _register_autonomy_runtime(*, runtime, loop, task: asyncio.Task | None = None) -> None:
    """Register the active autonomy runtime/loop/task for operator surfaces."""
    global _autonomy_runtime, _auto_loop, _auto_task
    _autonomy_runtime = runtime
    _auto_loop = loop
    if task is not None:
        _auto_task = task


def _clear_autonomy_runtime() -> None:
    """Clear the active autonomy runtime registry."""
    global _autonomy_runtime, _auto_loop, _auto_task, _autonomy_restarting
    _autonomy_runtime = None
    _auto_loop = None
    _auto_task = None
    _autonomy_restarting = False


def request_graceful_shutdown() -> bool:
    """Request the shared combined-runner shutdown path if active."""
    global _shutdown_event
    event = _shutdown_event
    if event is None:
        return False
    event.set()
    return True


def _launch_autonomy_task(
    task_name: str = "autonomous",
    *,
    version_override: str | None = None,
) -> tuple[object, object, asyncio.Task, str]:
    """Instantiate, register, and start a fresh autonomy runtime task."""
    runtime, loop, version = _instantiate_autonomy_runtime(version_override=version_override)
    task = asyncio.create_task(loop.start(), name=task_name)
    _register_autonomy_runtime(runtime=runtime, loop=loop, task=task)
    return runtime, loop, task, version


def get_runtime_transport_snapshot() -> dict:
    """Return shared runtime transport state derived from the event bus."""
    from remy.core.event_bus import event_bus

    subscribers = int(event_bus.subscriber_count)
    return {
        "subscribers": subscribers,
        "connected": bool(subscribers > 0),
    }


def is_runtime_transport_connected() -> bool:
    """Return whether any runtime event-bus subscriber is currently attached."""
    return bool(get_runtime_transport_snapshot().get("connected", False))


def get_operator_runtime_snapshot(*, goal_limit: int = 5, approval_limit: int = 10) -> dict:
    """Return a shared operator-facing snapshot for web/Telegram surfaces."""
    snapshot = {
        "autonomy": get_autonomy_status_snapshot(),
        "approvals": {
            "pending_count": 0,
            "pending": [],
            "recent": [],
        },
        "goals": {
            "total": 0,
            "active": 0,
            "blocked": 0,
            "active_list": [],
        },
        "budget": {},
        "evaluation": {},
        "factuality": {},
        "quality_debt_by_specialist": [],
        "evidence_debt_queue": [],
        "scheduler_decisions_recent": [],
        "routing_pressure": {},
    }

    approval_items: list[dict] = []
    approval_error = None

    try:
        from remy.core.approval_queue import approval_queue

        pending = approval_queue.snapshot_pending(limit=approval_limit)
        approval_items.extend(
            {
                **item,
                "action_id": item.get("action_id") or item.get("id") or "",
                "description": str(item.get("description") or "")[:160],
                "source": item.get("source") or "legacy_queue",
                "routing_pressure": bool(item.get("routing_pressure")),
            }
            for item in pending[:approval_limit]
        )
    except Exception as e:
        approval_error = str(e)

    try:
        with brain_lock:
            goals = brain.search(query="", tags=["autonomous-goal"], limit=200) or []
        active = [g for g in goals if (g.metadata or {}).get("status") == "active"]
        blocked = [g for g in goals if (g.metadata or {}).get("status", "").startswith("blocked")]
        snapshot["goals"] = {
            "total": len(goals),
            "active": len(active),
            "blocked": len(blocked),
            "active_list": [
                {
                    "id": goal.id,
                    "content": goal.content[:80],
                    "priority": (goal.metadata or {}).get("priority", "medium"),
                }
                for goal in active[:goal_limit]
            ],
        }
    except Exception as e:
        snapshot["goals"] = {"total": 0, "active": 0, "blocked": 0, "active_list": [], "error": str(e)}

    try:
        from remy.core.survival import load_state

        state = load_state()
        llm_data = {}
        try:
            budget_path = settings.DATA_DIR / "autonomy_budget.json"
            if budget_path.exists():
                llm_data = json.loads(budget_path.read_text(encoding="utf-8"))
        except Exception:
            llm_data = {}
        snapshot["budget"] = {
            "balance_usd": getattr(state, "last_total_usd", None),
            "usdt": getattr(state, "last_usdt", None),
            "trx": getattr(state, "last_trx", None),
            "runway_days": getattr(state, "last_runway_days", None),
            "alert_level": getattr(state, "last_status", "unknown"),
            "daily_limit": llm_data.get("daily_limit"),
            "hourly_limit": llm_data.get("hourly_limit"),
            "tokens_today": llm_data.get("tokens_today"),
            "tokens_this_hour": llm_data.get("tokens_this_hour"),
            "cost_today_usd": llm_data.get("cost_today_usd", getattr(state, "llm_cost_today", None)),
            "llm_cost_today": llm_data.get("cost_today_usd", getattr(state, "llm_cost_today", None)),
            "llm_cost_lifetime_usd": llm_data.get("total_cost_lifetime_usd"),
            "llm_tokens_today": llm_data.get("tokens_today"),
            "llm_tokens_this_hour": llm_data.get("tokens_this_hour"),
            "llm_tokens_lifetime": llm_data.get("total_tokens_lifetime"),
            "last_check": getattr(state, "last_balance_check", None),
        }
    except Exception as e:
        snapshot["budget"] = {"error": str(e)}

    try:
        evaluator = get_autonomy_runtime_component("evaluator")
        if evaluator is not None and hasattr(evaluator, "summary"):
            snapshot["evaluation"] = evaluator.summary() or {}
    except Exception as e:
        snapshot["evaluation"] = {"error": str(e)}

    ops_query_runtime = None
    try:
        ops_query_runtime = get_autonomy_runtime_component("ops_query_runtime")
        if ops_query_runtime is not None:
            if hasattr(ops_query_runtime, "pending_approval_items"):
                approval_items.extend(
                    {
                        **item,
                        "action_id": item.get("action_id") or item.get("id") or "",
                        "description": str(item.get("description") or item.get("action") or "")[:160],
                        "source": item.get("source") or "v3_governance",
                        "routing_pressure": bool(
                            item.get("routing_pressure")
                            or (item.get("context") or {}).get("quality_debt") is not None
                            or "routing pressure" in str(item.get("description") or "").lower()
                        ),
                    }
                    for item in (ops_query_runtime.pending_approval_items(limit=approval_limit) or [])[:approval_limit]
                )
            if hasattr(ops_query_runtime, "recent_approvals"):
                snapshot["approvals"]["recent"] = [
                    {
                        **item,
                        "description": str(item.get("description") or item.get("action") or "")[:160],
                        "routing_pressure": bool(
                            item.get("routing_pressure")
                            or (item.get("context") or {}).get("quality_debt") is not None
                        ),
                    }
                    for item in (ops_query_runtime.recent_approvals(limit=5) or [])[:5]
                ]
            if hasattr(ops_query_runtime, "factuality_summary"):
                snapshot["factuality"] = ops_query_runtime.factuality_summary() or {}
            if hasattr(ops_query_runtime, "quality_debt_by_specialist"):
                snapshot["quality_debt_by_specialist"] = ops_query_runtime.quality_debt_by_specialist() or []
            if hasattr(ops_query_runtime, "evidence_debt_queue"):
                snapshot["evidence_debt_queue"] = ops_query_runtime.evidence_debt_queue(10) or []
            if hasattr(ops_query_runtime, "scheduler_decisions_recent"):
                snapshot["scheduler_decisions_recent"] = ops_query_runtime.scheduler_decisions_recent(5) or []
            if hasattr(ops_query_runtime, "routing_pressure_summary"):
                snapshot["routing_pressure"] = ops_query_runtime.routing_pressure_summary() or {}
    except Exception as e:
        snapshot["factuality"] = {"error": str(e)}

    deduped_approvals = []
    seen_approval_ids = set()
    for item in sorted(
        approval_items,
        key=lambda entry: float(entry.get("created_at") or 0),
        reverse=True,
    ):
        approval_id = item.get("action_id") or item.get("id") or ""
        if approval_id in seen_approval_ids:
            continue
        seen_approval_ids.add(approval_id)
        deduped_approvals.append(item)
        if len(deduped_approvals) >= approval_limit:
            break
    snapshot["approvals"] = {
        "pending_count": len(deduped_approvals),
        "pending": deduped_approvals,
        "recent": list(snapshot.get("approvals", {}).get("recent", []) or []),
    }
    if approval_error:
        snapshot["approvals"]["error"] = approval_error

    return snapshot


def get_autonomy_operator_snapshot(*, goal_limit: int = 3, approval_limit: int = 10) -> dict:
    """Return a merged autonomy/operator snapshot for activity-facing surfaces."""
    snapshot = {
        "running": False,
        "version": get_autonomy_runtime_version(),
        "session_id": None,
        "budget": None,
        "current_goal": None,
        "current_mission": None,
        "current_task": None,
        "current_step": None,
        "last_cycle_result": None,
        "current_role": "",
        "last_agent_response": None,
        "last_research_activity": None,
        "research_session": None,
        "pending_approvals": 0,
        "approval_queue": [],
        "scheduler_reason": "",
        "scheduler_selection": {},
        "scheduler_decisions_recent": [],
        "stuck_missions_count": 0,
        "stuck_missions": [],
        "specialist_resolution": {},
        "quality_debt_by_specialist": [],
        "evidence_debt_queue": [],
        "routing_pressure": {},
        "evaluation": {},
        "factuality": {},
        "decision_flow": [],
        "active_harness_module": {},
    }

    status = get_autonomy_status_snapshot() or {}
    snapshot.update(status)

    operator_snapshot = get_operator_runtime_snapshot(goal_limit=goal_limit, approval_limit=approval_limit)
    approvals = operator_snapshot.get("approvals", {})
    if not snapshot.get("budget"):
        snapshot["budget"] = operator_snapshot.get("budget")
    snapshot["pending_approvals"] = approvals.get("pending_count", snapshot.get("pending_approvals", 0))
    snapshot["approval_queue"] = approvals.get("pending", snapshot.get("approval_queue", []))
    snapshot["evaluation"] = operator_snapshot.get("evaluation", snapshot.get("evaluation", {}))
    snapshot["factuality"] = operator_snapshot.get("factuality", snapshot.get("factuality", {}))
    if not snapshot.get("quality_debt_by_specialist"):
        snapshot["quality_debt_by_specialist"] = operator_snapshot.get("quality_debt_by_specialist", [])
    if not snapshot.get("evidence_debt_queue"):
        snapshot["evidence_debt_queue"] = operator_snapshot.get("evidence_debt_queue", [])
    if not snapshot.get("scheduler_decisions_recent"):
        snapshot["scheduler_decisions_recent"] = operator_snapshot.get("scheduler_decisions_recent", [])
    snapshot["scheduler_selection"] = operator_snapshot.get("scheduler_selection", snapshot.get("scheduler_selection", {}))
    snapshot["specialist_resolution"] = operator_snapshot.get("specialist_resolution", snapshot.get("specialist_resolution", {}))
    snapshot["routing_pressure"] = operator_snapshot.get("routing_pressure", snapshot.get("routing_pressure", {}))
    current_goal = snapshot.get("current_goal") if isinstance(snapshot.get("current_goal"), dict) else {}
    goal_id = str(current_goal.get("id") or current_goal.get("goal_id") or "").strip()
    if goal_id:
        try:
            from remy.core.research_sessions import get_research_session_trace

            snapshot["research_session"] = get_research_session_trace(goal_id)
        except Exception as e:
            snapshot["research_session"] = {"error": str(e), "goal_id": goal_id}

    decision_flow = []
    current_goal = snapshot.get("current_goal") if isinstance(snapshot.get("current_goal"), dict) else {}
    current_mission = snapshot.get("current_mission") if isinstance(snapshot.get("current_mission"), dict) else {}
    scheduler_selection = snapshot.get("scheduler_selection") if isinstance(snapshot.get("scheduler_selection"), dict) else {}
    specialist_resolution = snapshot.get("specialist_resolution") if isinstance(snapshot.get("specialist_resolution"), dict) else {}
    approval_queue = snapshot.get("approval_queue") if isinstance(snapshot.get("approval_queue"), list) else []
    current_task = snapshot.get("current_task") if isinstance(snapshot.get("current_task"), dict) else {}
    current_step = snapshot.get("current_step") if isinstance(snapshot.get("current_step"), dict) else {}
    last_cycle_result = snapshot.get("last_cycle_result") if isinstance(snapshot.get("last_cycle_result"), dict) else {}

    if scheduler_selection.get("mission_id") or scheduler_selection.get("score") is not None:
        details = scheduler_selection.get("details") if isinstance(scheduler_selection.get("details"), dict) else {}
        score = scheduler_selection.get("score")
        score_text = ""
        if score is not None:
            try:
                score_text = f"score {float(score):.2f}"
            except Exception:
                score_text = f"score {score}"
        summary_parts = [part for part in [
            scheduler_selection.get("mission_id"),
            score_text,
            details.get("routing_reason"),
        ] if part]
        decision_flow.append({
            "stage": "selection",
            "title": "Mission selected",
            "summary": " · ".join(str(part) for part in summary_parts) or "Mission selected",
        })
    elif current_mission.get("id") or current_mission.get("description"):
        decision_flow.append({
            "stage": "selection",
            "title": "Mission active",
            "summary": str(current_mission.get("description") or current_mission.get("id") or ""),
        })

    if specialist_resolution.get("specialist_id"):
        quality = specialist_resolution.get("quality_factor")
        quality_text = ""
        if quality is not None:
            try:
                quality_text = f"quality {float(quality):.2f}"
            except Exception:
                quality_text = f"quality {quality}"
        resolution_parts = [part for part in [
            specialist_resolution.get("specialist_id"),
            specialist_resolution.get("reason"),
            quality_text,
        ] if part]
        decision_flow.append({
            "stage": "routing",
            "title": "Specialist chosen",
            "summary": " · ".join(str(part) for part in resolution_parts),
        })

    if approval_queue:
        pending = approval_queue[0] if isinstance(approval_queue[0], dict) else {}
        gate_parts = [part for part in [
            pending.get("description") or pending.get("action"),
            f"risk {pending.get('risk_category')}" if pending.get("risk_category") else "",
            "routing pressure" if pending.get("routing_pressure") else "",
        ] if part]
        decision_flow.append({
            "stage": "risk",
            "title": "Approval gate",
            "summary": " · ".join(str(part) for part in gate_parts) or "Awaiting approval",
        })
    elif last_cycle_result.get("decision"):
        decision_flow.append({
            "stage": "risk",
            "title": "Cycle decision",
            "summary": str(last_cycle_result.get("decision") or ""),
        })

    if current_task.get("action") or current_step.get("instruction") or snapshot.get("current_role"):
        exec_parts = [part for part in [
            snapshot.get("current_role"),
            current_task.get("action"),
            current_step.get("instruction"),
        ] if part]
        decision_flow.append({
            "stage": "execution",
            "title": "Current execution",
            "summary": " · ".join(str(part) for part in exec_parts),
        })

    snapshot["decision_flow"] = decision_flow[:4]
    try:
        from remy.core.harness_modules import derive_active_harness_module

        snapshot["active_harness_module"] = derive_active_harness_module(runtime_snapshot=snapshot)
    except Exception:
        snapshot["active_harness_module"] = {}
    return snapshot


def get_activity_runtime_snapshot(
    *,
    goal_limit: int = 3,
    approval_limit: int = 10,
    transport_connected: bool = False,
) -> dict:
    """Return a shared activity-facing runtime snapshot."""
    snapshot = get_autonomy_operator_snapshot(goal_limit=goal_limit, approval_limit=approval_limit)
    snapshot["transport_connected"] = bool(transport_connected)
    return snapshot


def get_system_runtime_snapshot(*, goal_limit: int = 5, approval_limit: int = 10) -> dict:
    """Return a shared runtime snapshot for system/operator surfaces."""
    operator_snapshot = get_operator_runtime_snapshot(goal_limit=goal_limit, approval_limit=approval_limit)
    control_state = get_autonomy_control_state()

    snapshot = {
        "control": control_state,
        "autonomy": operator_snapshot.get("autonomy", {"running": False, "session_id": None}),
        "approvals": operator_snapshot.get("approvals", {"pending_count": 0, "pending": []}),
        "budget": operator_snapshot.get("budget", {}),
        "evaluation": operator_snapshot.get("evaluation", {}),
        "factuality": {
            **(operator_snapshot.get("factuality", {}) or {}),
            "quality_debt_by_specialist": operator_snapshot.get("quality_debt_by_specialist", []),
            "scheduler_decisions_recent": operator_snapshot.get("scheduler_decisions_recent", []),
        },
        "improvement": {
            "learning": {},
            "playbooks": {},
            "reviewable_insights": [],
            "top_playbooks": [],
        },
    }
    snapshot["autonomy"]["goals"] = operator_snapshot.get("goals", {})
    snapshot["autonomy"]["research_session"] = operator_snapshot.get("research_session")
    snapshot["autonomy"]["decision_flow"] = operator_snapshot.get("decision_flow", [])
    snapshot["autonomy"]["active_harness_module"] = operator_snapshot.get("active_harness_module", {})
    snapshot["evaluation"]["routing_pressure"] = operator_snapshot.get("routing_pressure", {})

    dashboard_runtime = get_autonomy_runtime_component("dashboard_runtime")
    if dashboard_runtime is not None and hasattr(dashboard_runtime, "improvement_summary"):
        snapshot["improvement"] = dashboard_runtime.improvement_summary()

    return snapshot


def get_channel_status_snapshot() -> dict:
    """Return a shared control-plane view of channel and gateway status."""
    from remy.core.gateway import get_registry as get_channel_registry
    from remy.core.notification_router import is_web_runtime_enabled

    registry = get_channel_registry()
    control_state = get_autonomy_control_state()
    channel_health = registry.all()
    registry_summary = registry.summary()
    telegram_health = channel_health.get("telegram")
    telegram_active = bool(telegram_health and telegram_health.get("status") in {"starting", "running", "degraded", "error"})
    web_health = channel_health.get("web")
    web_active = bool(web_health and web_health.get("status") in {"starting", "running", "degraded"})
    primary_remote_surface = settings.PRIMARY_REMOTE_SURFACE
    if primary_remote_surface == "telegram" and not telegram_active and web_active:
        primary_remote_surface = "web"

    return {
        "channels": {
            "web": {
                "enabled": is_web_runtime_enabled(),
                "url": f"http://{settings.WEB_HOST}:{settings.WEB_PORT}",
                "health": channel_health.get("web"),
            },
            "telegram": {
                "enabled": telegram_active,
                "configured": bool(settings.TELEGRAM_BOT_TOKEN),
                "secure": bool(settings.TELEGRAM_ALLOWED_CHAT_IDS),
                "allowed_ids": list(settings.TELEGRAM_ALLOWED_CHAT_IDS),
                "authorization_hint": (
                    "Set TELEGRAM_ALLOWED_CHAT_IDS=<your_chat_id> to move Telegram out of open mode."
                    if telegram_active and settings.TELEGRAM_BOT_TOKEN and not settings.TELEGRAM_ALLOWED_CHAT_IDS
                    else ""
                ),
                "health": telegram_health,
            },
            "autonomy": {
                "enabled": settings.AUTONOMY_ENABLED,
                "version": control_state.get("active_version", control_state.get("configured_version", "v2")),
                "configured_version": control_state.get("configured_version", "v2"),
                "maintenance_only": control_state.get("maintenance_only", False),
                "cycle_sec": settings.AUTONOMY_CYCLE_INTERVAL_SEC,
                "health": channel_health.get("autonomy"),
            },
            "registry_summary": registry_summary,
        },
        "gateway": {
            "name": "Remy Gateway",
            "primary_remote_surface": primary_remote_surface,
            "status": registry_summary.get("health", "unknown"),
        },
        "control": control_state,
    }


def get_operator_console_snapshot(*, goal_limit: int = 5, approval_limit: int = 10) -> dict:
    """Return a merged facade for operator console surfaces."""
    channel_snapshot = get_channel_status_snapshot()
    runtime_snapshot = get_system_runtime_snapshot(goal_limit=goal_limit, approval_limit=approval_limit)
    return {
        "channels": channel_snapshot.get("channels", {}),
        "gateway": channel_snapshot.get("gateway", {}),
        "control": channel_snapshot.get("control", runtime_snapshot.get("control", {})),
        "autonomy": runtime_snapshot.get("autonomy", {"running": False, "session_id": None}),
        "approvals": runtime_snapshot.get("approvals", {"pending_count": 0, "pending": []}),
        "budget": runtime_snapshot.get("budget", {}),
        "evaluation": runtime_snapshot.get("evaluation", {}),
        "factuality": runtime_snapshot.get("factuality", {}),
        "improvement": runtime_snapshot.get(
            "improvement",
            {"learning": {}, "playbooks": {}, "reviewable_insights": [], "top_playbooks": []},
        ),
    }


def get_activity_feed_snapshot(
    brain,
    brain_lock,
    *,
    goal_limit: int = 50,
    outcome_limit: int = 100,
    reflection_limit: int = 10,
    proactive_limit: int = 20,
) -> dict:
    """Return the shared aggregate activity payload for operator-facing views."""
    from remy.web.routes._activity_serialization import build_activity_payload

    with brain_lock:
        goals = brain.search(query="", tags=["autonomous-goal"], limit=goal_limit)
        outcomes = brain.search(query="", tags=["autonomous-outcome"], limit=outcome_limit)
        reflections = brain.search(query="", tags=["session-reflection"], limit=reflection_limit)
        proactive = brain.search(query="", tags=["proactive-session"], limit=proactive_limit)
    return build_activity_payload(goals, outcomes, reflections, proactive)


def get_budget_runtime_snapshot(*, goal_limit: int = 5, approval_limit: int = 10) -> dict:
    """Return a normalized budget snapshot for operator-facing surfaces."""
    budget = dict(get_operator_runtime_snapshot(goal_limit=goal_limit, approval_limit=approval_limit).get("budget", {}) or {})
    budget.setdefault("daily_cost_limit_usd", settings.AUTONOMY_DAILY_COST_LIMIT_USD)
    if "cost_today_usd" not in budget and budget.get("llm_cost_today") is not None:
        budget["cost_today_usd"] = budget.get("llm_cost_today")
    return budget


def get_goal_runtime_snapshot(*, goal_limit: int = 5, approval_limit: int = 10) -> dict:
    """Return the normalized goal summary from the operator snapshot."""
    return dict(get_operator_runtime_snapshot(goal_limit=goal_limit, approval_limit=approval_limit).get("goals", {}) or {})


def get_approval_runtime_snapshot(*, goal_limit: int = 3, approval_limit: int = 10) -> dict:
    """Return the normalized approval summary from the operator snapshot."""
    return dict(get_operator_runtime_snapshot(goal_limit=goal_limit, approval_limit=approval_limit).get("approvals", {}) or {})


def get_guidance_runtime_snapshot(*, limit: int = 10) -> dict:
    """Return the normalized pending guidance summary for operator-facing surfaces."""
    try:
        from remy.core.guidance_queue import guidance_queue

        pending = guidance_queue.snapshot_pending(limit=limit)
        return {
            "pending_count": len(pending),
            "pending": list(pending),
        }
    except Exception as e:
        return {
            "pending_count": 0,
            "pending": [],
            "error": str(e),
        }


def resolve_operator_approval_reply(text: str, *, decided_by: str = "operator-reply") -> dict:
    """Resolve the oldest pending approval from a yes/no style operator reply."""
    from remy.core.approval_queue import _CONFIRM_PHRASES, _REJECT_PHRASES

    text_lower = str(text or "").strip().lower()
    if not text_lower:
        return {"consumed": False, "reason": "empty"}

    is_confirm = any(phrase in text_lower for phrase in _CONFIRM_PHRASES)
    is_reject = any(phrase in text_lower for phrase in _REJECT_PHRASES)
    if not (is_confirm or is_reject):
        return {"consumed": False, "reason": "unrecognized"}

    approvals = get_approval_runtime_snapshot(goal_limit=3, approval_limit=50)
    pending = list(approvals.get("pending", []) or [])
    if not pending:
        return {"consumed": False, "reason": "no_pending"}

    pending.sort(key=lambda item: float(item.get("created_at") or 0))
    oldest = pending[0]
    action_id = str(oldest.get("action_id") or oldest.get("id") or "")
    if not action_id:
        return {"consumed": False, "reason": "missing_action_id"}

    result = resolve_operator_approval(
        action_id,
        approved=bool(is_confirm),
        decided_by=decided_by,
    )
    return {
        "consumed": True,
        "approved": bool(is_confirm),
        "action_id": action_id,
        "result": result,
    }


def resolve_operator_approval(action_id: str, *, approved: bool, decided_by: str = "operator", reason: str = "") -> dict:
    """Resolve a pending approval across v3 governance and legacy approval queue."""
    approval_id = str(action_id or "").strip()
    if not approval_id:
        return {"ok": False, "action_id": action_id, "source": None, "error": "missing_action_id"}

    approval_runtime = get_autonomy_runtime_component("approval")
    if approval_runtime is not None and hasattr(approval_runtime, "pending"):
        try:
            pending = list(approval_runtime.pending() or [])
            matched = next((req for req in pending if str(getattr(req, "id", "")).startswith(approval_id)), None)
            if matched is not None:
                resolved_id = str(getattr(matched, "id", "") or approval_id)
                if approved:
                    ok = bool(approval_runtime.approve(resolved_id, decided_by))
                else:
                    ok = bool(approval_runtime.deny(resolved_id, decided_by, reason=reason))
                return {"ok": ok, "action_id": resolved_id, "source": "v3_governance"}
        except Exception as e:
            logger.debug("v3 approval resolution failed for %s: %s", approval_id, e)

    try:
        from remy.core.approval_queue import approval_queue

        ok = bool(approval_queue.resolve_by_id(approval_id, approved=approved))
        return {"ok": ok, "action_id": approval_id, "source": "legacy_queue"}
    except Exception as e:
        return {"ok": False, "action_id": approval_id, "source": "legacy_queue", "error": str(e)}


def resolve_operator_guidance(request_id: str, answer: str) -> dict:
    """Resolve a pending guidance request through the canonical runtime seam."""
    target_id = str(request_id or "").strip()
    response = str(answer or "").strip()
    if not target_id:
        return {"ok": False, "request_id": request_id, "error": "missing_request_id"}
    if not response:
        return {"ok": False, "request_id": target_id, "error": "missing_answer"}

    try:
        from remy.core.guidance_queue import guidance_queue

        ok = bool(guidance_queue.resolve_by_id(target_id, response))
        return {"ok": ok, "request_id": target_id}
    except Exception as e:
        return {"ok": False, "request_id": target_id, "error": str(e)}


def resolve_operator_guidance_reply(text: str) -> dict:
    """Resolve the oldest pending guidance request from a free-text operator reply."""
    response = str(text or "").strip()
    if not response:
        return {"consumed": False, "reason": "empty"}

    guidance = get_guidance_runtime_snapshot(limit=50)
    pending = list(guidance.get("pending", []) or [])
    if not pending:
        return {"consumed": False, "reason": "no_pending"}

    pending.sort(key=lambda item: float(item.get("created_at") or 0))
    oldest = pending[0]
    request_id = str(oldest.get("request_id") or "")
    if not request_id:
        return {"consumed": False, "reason": "missing_request_id"}

    result = resolve_operator_guidance(request_id, response)
    return {
        "consumed": bool(result.get("ok")),
        "request_id": request_id,
        "result": result,
    }


async def start_autonomy():
    """Start a new autonomous loop in the background (called from web toggle)."""
    if is_autonomy_running():
        return  # Already running

    channel_starting("autonomy")
    try:
        _runtime, loop, _task, version = _launch_autonomy_task(
            "autonomous",
            version_override=_configured_autonomy_version(),
        )
        channel_running("autonomy")
        logger.info("Autonomous %s loop started via web toggle (session: %s)", version, getattr(loop, "session_id", None))
    except Exception as e:
        channel_stopped("autonomy", error=str(e))
        raise


async def stop_autonomy():
    """Stop the running autonomous loop (called from web toggle)."""
    loop = _auto_loop
    task = _auto_task

    if loop and (getattr(loop, "running", None) or getattr(loop, "_running", False)):
        await loop.stop()
        logger.info("Autonomous loop stopped via web toggle")

    if task and not task.done():
        task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
            pass

    _clear_autonomy_runtime()
    channel_stopped("autonomy")


async def run_combined(
    autonomous: bool = True,
    telegram: bool = False,
    web: bool = False,
    autonomy_version_override: str | None = None,
):
    """Run selected channels in parallel within one asyncio event loop."""
    global _autonomy_version_override
    tasks: list[asyncio.Task] = []
    cleanup_fns: list = []
    previous_version_override = _autonomy_version_override
    if autonomy_version_override in {"v2", "v3"}:
        _autonomy_version_override = autonomy_version_override

    _print_combined_banner(autonomous, telegram, web)
    # Start Playwright in background — do NOT block channel startup.
    # Web and Telegram are ready in <1s; Playwright takes 2-5s.
    asyncio.create_task(ensure_pinchtab_running(), name="pinchtab-init")

    # --- 1. Autonomous Loop (native async) ---
    global _operator_watch_task
    if autonomous:
        channel_starting("autonomy")
        _runtime, loop, auto_task, version = _launch_autonomy_task(
            "autonomous",
            version_override=_configured_autonomy_version(),
        )
        tasks.append(auto_task)
        cleanup_fns.append(stop_autonomy)
        channel_running("autonomy")
        logger.info("Autonomous %s loop queued (session: %s)", version, getattr(loop, "session_id", None))

    # --- 2. Telegram Bot (async API, not run_polling) ---
    tg_app = None
    if telegram:
        channel_starting("telegram")
        if not settings.TELEGRAM_ALLOWED_CHAT_IDS:
            logger.warning(
                "SECURITY: Telegram is in OPEN MODE — any user can send messages to this bot. "
                "Set TELEGRAM_ALLOWED_CHAT_IDS in .env to restrict access. "
                "Run `remy --doctor` for details."
            )
            channel_degraded("telegram", "open mode — no allowlist")
        try:
            tg_app = await _start_telegram_async()
            cleanup_fns.append(lambda: _stop_telegram_async(tg_app))
            channel_running("telegram")
            logger.info("Telegram bot started (async polling)")
        except Exception as e:
            channel_stopped("telegram", error=str(e))
            raise

    # --- 3. Web GUI (async uvicorn) ---
    uvicorn_server = None
    if web:
        channel_starting("web")
        set_web_runtime_enabled(True)
        uvicorn_server = _create_uvicorn_server()
        tasks.append(asyncio.create_task(
            _run_uvicorn_safe(uvicorn_server), name="uvicorn",
        ))
        cleanup_fns.append(lambda: _stop_uvicorn(uvicorn_server))
        cleanup_fns.append(lambda: set_web_runtime_enabled(False))
        channel_running("web")
        logger.info(
            "Web GUI started on http://%s:%d", settings.WEB_HOST, settings.WEB_PORT
        )

    if not tasks and tg_app is None:
        logger.error("No channels enabled. Nothing to run.")
        return

    operator_watch_stop = asyncio.Event()
    _operator_watch_task = asyncio.create_task(
        _operator_watch_loop(operator_watch_stop), name="operator-watch"
    )
    tasks.append(_operator_watch_task)
    cleanup_fns.append(operator_watch_stop.set)

    # --- Signal handling for graceful shutdown ---
    global _shutdown_event
    shutdown_event = asyncio.Event()
    _shutdown_event = shutdown_event
    fast_shutdown_requested = False

    def _signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    _is_windows = False

    # Suppress noisy Windows ProactorEventLoop errors from subprocess transports.
    # "_call_connection_lost" fires when Playwright/subprocess pipes are GC'd —
    # harmless but spams the console with ERROR-level asyncio exceptions.
    _orig_exception_handler = loop.get_exception_handler()

    def _quiet_exception_handler(loop, context):
        msg = context.get("message", "")
        if "_call_connection_lost" in msg or "Event loop is closed" in str(context.get("exception", "")):
            return  # Suppress — harmless transport cleanup noise
        if _orig_exception_handler:
            _orig_exception_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(_quiet_exception_handler)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows does not support add_signal_handler
            _is_windows = True

    if _is_windows:
        # Windows: register a classic signal handler.
        # First Ctrl+C → graceful shutdown. Second Ctrl+C → fast shutdown mode.
        # IMPORTANT: Do NOT raise KeyboardInterrupt — it interrupts active
        # WebSocket handlers (session summary, brain.end_session) causing errors.
        # Also avoid os._exit(): it bypasses finally/brain.close() and can corrupt the Aura store.
        # Instead, set shutdown_event and wake the event loop via call_soon_threadsafe.
        _ctrl_c_count = 0

        def _windows_sigint(signum, frame):
            nonlocal _ctrl_c_count, fast_shutdown_requested
            _ctrl_c_count += 1
            if _ctrl_c_count == 1:
                logger.info("Ctrl+C received — shutting down gracefully...")
                print("\nShutting down gracefully... Do not press Ctrl+C again or close the terminal until shutdown finishes.")
                loop.call_soon_threadsafe(shutdown_event.set)
            else:
                fast_shutdown_requested = True
                logger.warning("Second Ctrl+C received — switching to fast shutdown mode")
                print("\nFast shutdown requested. Still waiting for memory to close safely — do not interrupt again.")
                loop.call_soon_threadsafe(shutdown_event.set)

        signal.signal(signal.SIGINT, _windows_sigint)

    try:
        if tasks:
            shutdown_task = asyncio.create_task(
                shutdown_event.wait(), name="shutdown_waiter"
            )
            # On Windows, use a periodic timeout so KeyboardInterrupt can propagate
            _wait_timeout = 2.0 if _is_windows else None
            while not shutdown_event.is_set():
                channel_tasks = [
                    t for t in tasks
                    if t.get_name() != "operator-watch" and not t.done()
                ]
                all_tasks = channel_tasks + [shutdown_task]
                if not channel_tasks:
                    # Only shutdown helpers/watchers left; all real channels finished.
                    break
                done, _pending = await asyncio.wait(
                    all_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=_wait_timeout,
                )
                if not done:
                    # Timeout on Windows — loop back to check shutdown_event
                    continue
                for task in done:
                    if task.get_name() == "shutdown_waiter":
                        # User pressed Ctrl+C or signal received
                        break
                    elif task.get_name() == "autonomous" and not shutdown_event.is_set():
                        if task.cancelled():
                            shutdown_event.set()
                            break
                        # Autonomous loop ended (session limit) — restart it
                        global _autonomy_restarting
                        _autonomy_restarting = True
                        exc = task.exception() if not task.cancelled() else None
                        if exc:
                            channel_stopped("autonomy", error=str(exc))
                            logger.error("Autonomous loop crashed: %s", exc)
                            logger.info("Restarting autonomous loop in 30s...")
                            await asyncio.sleep(30)
                        else:
                            channel_stopped("autonomy")
                            logger.info("Autonomous session ended. Restarting in 10s...")
                            await asyncio.sleep(10)
                        channel_starting("autonomy")
                        _runtime, restarted_loop, new_task, version = _launch_autonomy_task(
                            "autonomous",
                            version_override=_configured_autonomy_version(),
                        )
                        _autonomy_restarting = False
                        # Replace old task in list
                        tasks = [t for t in tasks if t.get_name() != "autonomous"]
                        tasks.append(new_task)
                        channel_running("autonomy")
                        logger.info("Autonomous %s loop restarted (session: %s)", version, getattr(restarted_loop, "session_id", None))
                    elif task.exception():
                        logger.error(
                            "Task '%s' crashed: %s", task.get_name(), task.exception()
                        )
                else:
                    continue
                break  # break outer while if inner for hit break
        else:
            # Only telegram (no async tasks), wait for shutdown signal
            await shutdown_event.wait()

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Interrupted, shutting down...")

    finally:
        logger.info("Shutting down all channels...")
        if autonomous:
            channel_stopped("autonomy")
        if telegram:
            channel_stopped("telegram")
        if web:
            channel_stopped("web")

        for fn in cleanup_fns:
            try:
                result = fn()
                if asyncio.iscoroutine(result):
                    await asyncio.wait_for(result, timeout=10.0)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning("Cleanup error: %s", e)

        try:
            await shutdown_pinchtab()
        except Exception as e:
            logger.warning("PinchTab shutdown error: %s", e)

        for task in tasks:
            if not task.done():
                if task.get_name() == "uvicorn":
                    try:
                        cancel_timeout = 1.5 if fast_shutdown_requested else 5.0
                        await asyncio.wait_for(asyncio.shield(task), timeout=cancel_timeout)
                        continue
                    except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                        pass
                task.cancel()
                try:
                    cancel_timeout = 1.5 if fast_shutdown_requested else 5.0
                    await asyncio.wait_for(asyncio.shield(task), timeout=cancel_timeout)
                except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                    pass

        logger.info("All channels stopped.")

        # Shutdown the default executor so asyncio.run() cleanup doesn't
        # block for 300 s waiting on threads stuck behind brain_lock.
        loop = asyncio.get_running_loop()
        default_executor = getattr(loop, '_default_executor', None)
        if default_executor is not None:
            default_executor.shutdown(wait=False, cancel_futures=True)

        _clear_autonomy_runtime()
        _autonomy_version_override = previous_version_override
        _shutdown_event = None


async def run_autonomy_standalone(*, version_override: str | None = None) -> None:
    """Run autonomy-only mode through the canonical combined-runner control seam."""
    await run_combined(
        autonomous=True,
        telegram=False,
        web=False,
        autonomy_version_override=version_override,
    )


# ============== TELEGRAM ASYNC START/STOP ==============


async def _start_telegram_async():
    """Start Telegram bot using python-telegram-bot's async API.

    Uses app.initialize() / app.start() / app.updater.start_polling()
    instead of the blocking app.run_polling() so it can coexist with
    other async tasks in the same event loop.
    """
    from telegram.ext import ApplicationBuilder
    from remy.core.telegram_bot import TelegramBot

    bot = TelegramBot()
    app = ApplicationBuilder().token(bot.token).build()
    bot.register_handlers(app)
    telegram_conflict_disabled = False

    async def _disable_telegram_polling(reason: str) -> None:
        nonlocal telegram_conflict_disabled
        if telegram_conflict_disabled:
            return
        telegram_conflict_disabled = True
        logger.error("Telegram polling disabled: %s", reason)
        try:
            await app.updater.stop()
        except Exception as e:
            logger.warning("Telegram updater stop error after conflict: %s", e)

    async def _log_telegram_error(_update, context):
        nonlocal telegram_conflict_disabled
        error = context.error
        text = str(error)
        if "Conflict" in text:
            if not telegram_conflict_disabled:
                logger.error("Telegram polling error: %s", text)
                logger.error("Another Telegram bot session is already polling with this token.")
                asyncio.create_task(_disable_telegram_polling("another getUpdates session is active"))
            return
        if "InvalidToken" in text or "Unauthorized" in text:
            logger.error("Telegram polling error: %s", text)
            logger.error("Check TELEGRAM_BOT_TOKEN.")
            return
        if "NetworkError" in text or "TimedOut" in text:
            logger.error("Telegram polling error: %s", text)
            return
        logger.error("Telegram polling error: %s", text)

    app.add_error_handler(_log_telegram_error)

    await app.initialize()
    await bot.configure_app(app)
    await app.start()
    def _polling_error_callback(err):
        nonlocal telegram_conflict_disabled
        text = str(err)
        if "Conflict" in text:
            if not telegram_conflict_disabled:
                logger.error("Telegram polling error: %s", text)
                logger.error("Another Telegram bot session is already polling with this token.")
                asyncio.create_task(_disable_telegram_polling("another getUpdates session is active"))
            return
        logger.error("Telegram polling error: %s", err)

    await app.updater.start_polling(error_callback=_polling_error_callback)

    return app


async def _stop_telegram_async(app):
    """Gracefully stop the Telegram bot."""
    try:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
    except Exception as e:
        logger.warning("Telegram shutdown error: %s", e)


# ============== WEB GUI ASYNC SETUP ==============


async def _run_uvicorn_safe(server: uvicorn.Server):
    """Run uvicorn.serve() but catch SystemExit from port-bind failures."""
    try:
        await server.serve()
    except SystemExit as e:
        logger.error(
            "Web GUI failed to start (port %s:%s likely in use): exit code %s",
            server.config.host, server.config.port, e.code,
        )
    except asyncio.CancelledError:
        logger.info("Web GUI task cancelled during shutdown")
        return


async def _stop_uvicorn(server: uvicorn.Server):
    """Gracefully stop uvicorn, guarding against missing .servers attribute."""
    try:
        server.should_exit = True
        if not hasattr(server, "servers") or server.servers is None:
            server.servers = []
        if getattr(server, "started", False):
            await server.shutdown()
    except Exception as e:
        logger.warning("Uvicorn shutdown error: %s", e)


def _current_maintenance_only() -> bool:
    """Best-effort view of whether the autonomy runtime is in maintenance mode."""
    return bool(get_autonomy_control_state().get("maintenance_only", False))


async def _operator_watch_loop(stop_event: asyncio.Event, interval_sec: int = 60) -> None:
    """Emit proactive runtime alerts for remote operators."""
    from remy.core.error_escalation import (
        DegradationLevel,
        assess_system_health,
        build_operator_watch_message,
        send_critical_alert,
    )
    from remy.core.gateway import get_registry as get_channel_registry
    from remy.core.notification_router import notify

    previous_level: DegradationLevel | None = None
    previous_gateway_health: str | None = None

    while not stop_event.is_set():
        try:
            snapshot = await asyncio.to_thread(get_operator_runtime_snapshot, goal_limit=3, approval_limit=5)
            budget_dict = snapshot.get("budget") or None
            health = assess_system_health(
                budget_dict=budget_dict,
                maintenance_only=_current_maintenance_only(),
            )
            gateway_health = get_channel_registry().summary().get("health", "ok")

            operator_message = build_operator_watch_message(
                health,
                previous_level=previous_level,
                gateway_health=gateway_health,
                previous_gateway_health=previous_gateway_health,
            )
            if operator_message:
                message, level = operator_message
                event_data = {
                    "health_level": health.level.name,
                    "gateway_health": gateway_health,
                    "budget_remaining_pct": round(health.budget_pct, 1),
                }
                if level in {"warning", "critical"}:
                    if gateway_health in {"degraded", "error"} and gateway_health != (previous_gateway_health or ""):
                        event_data["dedupe_key"] = f"gateway:{gateway_health}"
                    elif health.level != DegradationLevel.GREEN and previous_level is not None and health.level >= previous_level:
                        event_data["dedupe_key"] = f"system-health:{health.level.name}"
                    elif 10 <= health.budget_pct < 30:
                        event_data["dedupe_key"] = "budget-pressure"
                else:
                    resolves = []
                    if previous_gateway_health and previous_gateway_health != "ok" and gateway_health == "ok":
                        resolves.append(f"gateway:{previous_gateway_health}")
                    if previous_level is not None and previous_level != DegradationLevel.GREEN and health.level == DegradationLevel.GREEN:
                        resolves.append(f"system-health:{previous_level.name}")
                    if health.budget_pct >= 30:
                        resolves.append("budget-pressure")
                    if resolves:
                        event_data["resolves"] = resolves
                        event_data["resolved"] = True
                        event_data["dedupe_key"] = f"recovery:{health.level.name}:{gateway_health}"
                notify(
                    message,
                    level=level,
                    event_type="operator_alert",
                    event_data=event_data,
                )

            await send_critical_alert(health)
            previous_level = health.level
            previous_gateway_health = gateway_health
        except Exception as e:
            logger.debug("Operator watch loop error: %s", e)

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_sec)
        except asyncio.TimeoutError:
            continue


def _create_uvicorn_server() -> uvicorn.Server:
    """Create an async uvicorn server (non-blocking).

    Uses uvicorn.Server(Config(...)).serve() instead of the blocking
    uvicorn.run() so it can coexist with other async tasks.
    """
    from remy.core.desktop_gui import create_app
    from remy.web.api import set_session_manager
    from remy.web.session import WebSessionManager

    manager = WebSessionManager()
    set_session_manager(manager)
    app = create_app()

    config = uvicorn.Config(
        app=app,
        host=settings.WEB_HOST,
        port=settings.WEB_PORT,
        log_level="warning",
        ws_max_size=30 * 1024 * 1024,
    )
    server = uvicorn.Server(config)
    server.config.install_signal_handlers = False  # We handle signals ourselves
    return server


# ============== BANNER ==============


def _console_text(text: object) -> str:
    """Return text that can be encoded by the active console stream."""
    value = str(text)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return value.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _console_print(text: object = "", **kwargs):
    print(_console_text(text), **kwargs)


def _overwrite_console_line(text: object, previous_text: object = ""):
    """Overwrite an in-place progress line and clear any leftover characters."""
    value = _console_text(text)
    previous = _console_text(previous_text)
    padding = " " * max(0, len(previous) - len(value))
    _console_print(f"\r{value}{padding}")


def _console_can_encode(text: str) -> bool:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
        return True
    except UnicodeEncodeError:
        return False


def _print_combined_banner(autonomous: bool, telegram: bool, web: bool):
    """Print startup banner with live stage events."""
    _G = "\033[92m"
    _Y = "\033[93m"
    _R = "\033[91m"
    _B = "\033[1m"
    _RST = "\033[0m"
    _OK = "✓" if _console_can_encode("✓") else "OK"
    _ERR = "✗" if _console_can_encode("✗") else "X"
    _WARN = "⚠" if _console_can_encode("⚠") else "!"

    channels = []
    if web:
        channels.append("Web")
    if telegram:
        channels.append("Telegram")
    if autonomous:
        channels.append("Autonomy")

    mode = " + ".join(channels) if channels else "No channels"

    _console_print()
    _console_print(f"{_B}{'=' * 56}{_RST}")
    _console_print(f"{_B}  Remy — {mode}{_RST}")
    _console_print(f"{_B}{'=' * 56}{_RST}")

    eager_banner = os.environ.get("REMY_EAGER_STARTUP_BANNER", "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    startup_status = get_brain_startup_status()

    # Stage 1: Memory
    memory_progress = f"  {_B}[1/4]{_RST} Loading memory..."
    _console_print(memory_progress, end="", flush=True)
    if eager_banner:
        try:
            with brain_lock:
                brain_count = brain.count()
            _overwrite_console_line(
                f"  {_G}{_OK}{_RST}     Memory loaded — {brain_count} records",
                memory_progress,
            )
        except Exception as e:
            _overwrite_console_line(f"  {_R}{_ERR}{_RST}     Memory error: {e}", memory_progress)
    else:
        memory_note = "Memory ready"
        if startup_status.get("quarantined_at_startup"):
            recovery = dict(startup_status.get("recovery") or {})
            recovery_status = str(recovery.get("status") or "").strip()
            if recovery_status:
                memory_note = f"Memory ready — {recovery_status.replace('_', ' ')}"
            else:
                memory_note = "Memory ready after quarantine recovery"
        _overwrite_console_line(f"  {_G}{_OK}{_RST}     {memory_note}", memory_progress)

    # Stage 2: Tools
    tools_progress = f"  {_B}[2/4]{_RST} Loading tools..."
    _console_print(tools_progress, end="", flush=True)
    if eager_banner:
        try:
            registry = get_registry()
            tool_count = len(registry.get_all_declarations())
            _overwrite_console_line(
                f"  {_G}{_OK}{_RST}     Tools loaded — {tool_count} tools",
                tools_progress,
            )
        except Exception as e:
            _overwrite_console_line(f"  {_R}{_ERR}{_RST}     Tools error: {e}", tools_progress)
    else:
        _overwrite_console_line(
            f"  {_G}{_OK}{_RST}     Tool registry deferred until first use",
            tools_progress,
        )

    # Stage 3: Web
    if web:
        web_progress = f"  {_B}[3/4]{_RST} Starting web server..."
        _console_print(web_progress, end="", flush=True)
        _overwrite_console_line(
            f"  {_G}{_OK}{_RST}     Web server — http://{settings.WEB_HOST}:{settings.WEB_PORT}",
            web_progress,
        )
    else:
        _console_print(f"  {_B}[3/4]{_RST} {_Y}Web server — disabled{_RST}")

    # Stage 4: Channels
    _console_print(f"  {_B}[4/4]{_RST} Channels:")
    if telegram:
        allowed = settings.TELEGRAM_ALLOWED_CHAT_IDS
        if allowed:
            _console_print(f"          {_G}{_OK}{_RST} Telegram — allowlist mode ({len(allowed)} IDs)")
        else:
            _console_print(f"          {_Y}{_WARN}{_RST} Telegram — OPEN MODE (any user can message)")
    else:
        _console_print(f"          — Telegram disabled")

    if autonomous:
        v = "v3" if settings.AUTONOMY_V3 else "v2"
        _console_print(f"          {_G}{_OK}{_RST} Autonomy {v} — cycle {settings.AUTONOMY_CYCLE_INTERVAL_SEC}s, limit {settings.AUTONOMY_DAILY_TOKEN_LIMIT} tokens/day")
    else:
        _console_print(f"          — Autonomy disabled")

    _console_print()
    _console_print(f"  Model: {settings.SUMMARY_MODEL}")
    _console_print(f"  Press Ctrl+C to stop.")
    _console_print(f"{_B}{'=' * 56}{_RST}")
    _console_print()
