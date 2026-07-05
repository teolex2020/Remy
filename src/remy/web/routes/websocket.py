"""
WebSocket routes — chat, live voice, approvals, activity stream.
"""

import asyncio
import base64
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from remy.web.routes._helpers import _get_api, run_in_thread, run_lambda_in_thread, _TIMEOUT_FAST

logger = logging.getLogger("WebAPI")

router = APIRouter()


def _runtime_subscriber_count() -> int:
    from remy.core.combined_runner import get_runtime_transport_snapshot

    return int(get_runtime_transport_snapshot().get("subscribers", 0))


async def _send_approval_snapshot(websocket: WebSocket) -> None:
    from remy.core.combined_runner import get_approval_runtime_snapshot
    from remy.core.runtime_event_contract import build_runtime_event

    approvals = await run_in_thread(get_approval_runtime_snapshot, goal_limit=3, approval_limit=50)
    for action in approvals.get("pending", []):
        await websocket.send_json(
            build_runtime_event(
                "approval.pending",
                event_domain="approval",
                payload={
                    "action_id": action.get("action_id") or action.get("id", ""),
                    "description": action.get("description", ""),
                    "timeout_sec": action.get("timeout_sec"),
                    "created_at": action.get("created_at"),
                },
                legacy_fields={
                    "action_id": action.get("action_id") or action.get("id", ""),
                    "description": action.get("description", ""),
                    "timeout_sec": action.get("timeout_sec"),
                    "created_at": action.get("created_at"),
                },
            )
        )


async def _send_guidance_snapshot(websocket: WebSocket) -> None:
    from remy.core.combined_runner import get_guidance_runtime_snapshot
    from remy.core.runtime_event_contract import build_runtime_event

    guidance = await run_in_thread(get_guidance_runtime_snapshot, limit=50)
    for req in guidance.get("pending", []):
        await websocket.send_json(
            build_runtime_event(
                "guidance.pending",
                event_domain="guidance",
                payload={
                    "request_id": req.get("request_id"),
                    "question": req.get("question", ""),
                    "context": req.get("context", ""),
                    "timeout_sec": req.get("timeout_sec"),
                    "created_at": req.get("created_at"),
                },
                legacy_fields={
                    "request_id": req.get("request_id"),
                    "question": req.get("question", ""),
                    "context": req.get("context", ""),
                    "timeout_sec": req.get("timeout_sec"),
                    "created_at": req.get("created_at"),
                },
            )
        )


async def _send_budget_init(websocket: WebSocket, api) -> None:
    from remy.core.combined_runner import get_budget_runtime_snapshot
    from remy.core.runtime_event_contract import build_runtime_event

    budget = await run_in_thread(get_budget_runtime_snapshot, goal_limit=3, approval_limit=5)
    await websocket.send_json(
        build_runtime_event(
            "budget_init",
            event_domain="budget",
            payload={"budget": budget},
            legacy_fields={"budget": budget},
        )
    )


async def _send_system_snapshot(websocket: WebSocket) -> None:
    from remy.core.runtime_event_contract import build_runtime_event
    from remy.web.routes.system_routes import build_system_status_payload

    snapshot = await build_system_status_payload(include_packs=True)
    await websocket.send_json(
        build_runtime_event(
            "system.snapshot",
            event_domain="system",
            payload=snapshot,
            legacy_fields={"snapshot": snapshot},
        )
    )


async def _send_activity_snapshot(websocket: WebSocket) -> None:
    from remy.core.runtime_event_contract import build_runtime_event
    from remy.core.combined_runner import (
        get_activity_runtime_snapshot,
        is_runtime_transport_connected,
    )

    snapshot = await run_in_thread(
        get_activity_runtime_snapshot,
        goal_limit=3,
        approval_limit=10,
        transport_connected=is_runtime_transport_connected(),
    )
    await websocket.send_json(
        build_runtime_event(
            "activity.snapshot",
            event_domain="activity",
            payload=snapshot,
            legacy_fields={"snapshot": snapshot},
        )
    )


def _build_activity_mission_state(mission_id: str) -> dict | None:
    from remy.core.activity_state import build_mission_activity_state

    return build_mission_activity_state(mission_id)


def _get_runtime_transport_state() -> dict:
    from remy.core.combined_runner import get_autonomy_control_state

    return get_autonomy_control_state() or {}


def _is_websocket_lifecycle_runtime_error(exc: RuntimeError) -> bool:
    message = str(exc)
    return (
        "WebSocket is not connected" in message
        or "Need to call \"accept\" first." in message
        or "Cannot call \"receive\"" in message
    )


async def _listen_websocket_client(websocket: WebSocket) -> None:
    while True:
        try:
            await websocket.receive_text()
        except WebSocketDisconnect:
            return
        except RuntimeError as exc:
            if _is_websocket_lifecycle_runtime_error(exc):
                return
            raise


async def _wait_for_websocket_tasks(*tasks: asyncio.Task) -> None:
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    for task in done:
        exc = task.exception()
        if exc is None:
            continue
        if isinstance(exc, WebSocketDisconnect):
            return
        raise exc


def _build_research_session_state(goal_id: str) -> dict | None:
    if not goal_id:
        return None
    try:
        from remy.core.research_sessions import get_research_session_trace

        return get_research_session_trace(goal_id)
    except Exception:
        return None


def _build_activity_delta_event(event: dict | None) -> dict | None:
    from remy.core.runtime_event_contract import build_runtime_event

    if not isinstance(event, dict):
        return None

    event_name = str(event.get("event_name") or event.get("type") or "")
    event_payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    payload: dict | None = None
    research_tools = {
        "web_search",
        "extract_content",
        "http_get",
        "add_research_finding",
        "store_research",
        "store_knowledge",
        "extract_facts",
        "start_research",
        "complete_research",
    }

    if event_name == "approval.pending":
        action_id = event.get("action_id") or event_payload.get("action_id")
        if action_id:
            payload = {
                "approval_queue": {
                    "upsert": {
                        "id": str(action_id),
                        "description": str(event.get("description") or event_payload.get("description") or "")[:100],
                    }
                }
            }
    elif event_name == "approval.resolved":
        action_id = event.get("action_id") or event_payload.get("action_id")
        if action_id:
            payload = {
                "approval_queue": {
                    "remove_id": str(action_id),
                }
            }
    elif event_name == "goal_selected":
        control_state = _get_runtime_transport_state()
        goal_id = str(event.get("goal_id") or event_payload.get("goal_id") or "")
        mission_id = str(event.get("mission_id") or event_payload.get("mission_id") or "")
        payload = {
            "running": bool(control_state.get("running", True)),
            "scheduler_reason": "active",
            "current_goal": {
                "id": goal_id,
                "description": str(event.get("description") or event_payload.get("description") or "")[:200],
                "priority": str(event.get("priority") or event_payload.get("priority") or "medium"),
            },
            "current_step": None,
            "last_cycle_result": None,
        }
        if goal_id:
            payload["research_session"] = _build_research_session_state(goal_id)
        if mission_id:
            payload["current_mission"] = _build_activity_mission_state(mission_id)
    elif event_name in {
        "goal_archived",
        "goal_blocked",
        "goal_unblocked",
        "goal_resumed",
        "goal_failed",
        "mission.task_active",
        "mission.task_completed",
        "mission.task_failed",
    }:
        mission_id = str(event.get("mission_id") or event_payload.get("mission_id") or "")
        if mission_id:
            payload = {
                "current_mission": _build_activity_mission_state(mission_id),
            }
    elif event_name == "plan_step":
        payload = {
            "current_step": {
                "instruction": str(event.get("step_description") or event_payload.get("step_description") or "")[:200],
                "step_num": int(event.get("step_num") or event_payload.get("step_num") or 0),
                "total_steps": int(event.get("total_steps") or event_payload.get("total_steps") or 0),
                "plan_type": str(event.get("plan_type") or event_payload.get("plan_type") or "linear"),
            }
        }
    elif event_name == "role_selected":
        payload = {
            "current_role": str(event.get("role") or event_payload.get("role") or ""),
        }
    elif event_name == "worker_role_resolution":
        requested_role = str(event.get("requested_role") or event_payload.get("requested_role") or "")
        resolved_role = str(event.get("resolved_role") or event_payload.get("resolved_role") or "")
        status = str(event.get("status") or event_payload.get("status") or "")
        worker_channel = str(event.get("worker_channel") or event_payload.get("worker_channel") or "")
        payload = {
            "current_role": resolved_role or requested_role,
            "specialist_resolution": {
                "requested_role": requested_role,
                "resolved_role": resolved_role,
                "status": status,
                "worker_channel": worker_channel,
                "timeout_sec": event.get("timeout_sec") or event_payload.get("timeout_sec"),
                "step_budget": event.get("step_budget") or event_payload.get("step_budget"),
            },
        }
    elif event_name == "worker_started":
        worker_channel = str(event.get("worker_channel") or event_payload.get("worker_channel") or "")
        role = str(event.get("role") or event_payload.get("role") or "")
        timeout_sec = event.get("timeout_sec") or event_payload.get("timeout_sec")
        step_budget = event.get("step_budget") or event_payload.get("step_budget")
        payload = {
            "current_task": {
                "action": f"Start {worker_channel or role}".strip(),
                "tool": "worker_start",
                "args_summary": f"timeout={timeout_sec}s, steps={step_budget}",
            },
            "current_role": role,
        }
    elif event_name == "tool_call":
        tool_name = str(event.get("tool") or event_payload.get("tool") or "").strip()
        args_summary = str(event.get("args_summary") or event_payload.get("args_summary") or "").strip()
        if tool_name:
            action = f"{tool_name}({args_summary})" if args_summary else tool_name
            payload = {
                "current_task": {
                    "action": action[:200],
                    "tool": tool_name,
                    "args_summary": args_summary[:500],
                }
            }
            if tool_name in research_tools:
                payload["last_research_activity"] = {
                    "tool": tool_name,
                    "summary": f"Research step: {tool_name}",
                }
                goal_id = str(
                    event.get("goal_id")
                    or event_payload.get("goal_id")
                    or (
                        (event_payload.get("goal") or {}).get("goal_id")
                        if isinstance(event_payload.get("goal"), dict)
                        else ""
                    )
                    or ""
                ).strip()
                if goal_id:
                    payload["research_session"] = _build_research_session_state(goal_id)
    elif event_name == "agent_response":
        payload = {
            "last_agent_response": {
                "response": str(event.get("response") or event_payload.get("response") or "")[:500],
                "duration_ms": int(event.get("duration_ms") or event_payload.get("duration_ms") or 0),
                "tokens_estimated": int(event.get("tokens_estimated") or event_payload.get("tokens_estimated") or 0),
            }
        }
    elif event_name == "tool_result":
        payload = {
            "current_task": None,
        }
    elif event_name == "evaluation":
        success = bool(event.get("success") if event.get("success") is not None else event_payload.get("success"))
        goal_completed = bool(
            event.get("goal_completed") if event.get("goal_completed") is not None else event_payload.get("goal_completed")
        )
        payload = {
            "current_task": None,
            "last_cycle_result": {
                "success": success,
                "confidence": float(event.get("confidence") or event_payload.get("confidence") or 0.0),
                "reason": str(event.get("reason") or event_payload.get("reason") or "")[:200],
                "goal_completed": goal_completed,
                "decision": "completed" if goal_completed else ("success" if success else "failed"),
            }
        }
    elif event_name in ("cycle_start", "cycle_end"):
        control_state = _get_runtime_transport_state()
        payload = {
            "running": bool(control_state.get("running", True)),
            "transport_connected": True,
            "session_id": control_state.get("session_id") or event.get("session_id") or event_payload.get("session_id"),
            "budget": event.get("budget") or event_payload.get("budget"),
        }
        try:
            from remy.core.combined_runner import get_activity_runtime_snapshot

            runtime_snapshot = get_activity_runtime_snapshot(
                goal_limit=3,
                approval_limit=10,
                transport_connected=True,
            )
            payload["specialist_resolution"] = runtime_snapshot.get("specialist_resolution") or {}
            payload["scheduler_selection"] = runtime_snapshot.get("scheduler_selection") or {}
            payload["research_session"] = runtime_snapshot.get("research_session")
        except Exception:
            payload["specialist_resolution"] = {}
            payload["scheduler_selection"] = {}
            payload["research_session"] = None
        if event_name == "cycle_end":
            payload["current_task"] = None
    elif event_name == "budget_warning":
        payload = {
            "scheduler_reason": f"Budget pause: {str(event.get('reason') or event_payload.get('reason') or '')}".strip(),
            "current_task": None,
        }
    elif event_name == "llm_health":
        from remy.core.combined_runner import get_autonomy_control_state

        control_state = get_autonomy_control_state()
        if control_state.get("maintenance_only"):
            payload = {
                "scheduler_reason": "LLM unavailable - maintenance-only mode",
                "running": bool(control_state.get("running", False)),
                "current_task": None,
            }
        elif str(event.get("status") or event_payload.get("status") or "") == "recovered":
            payload = {
                "running": bool(control_state.get("running", False)),
                "scheduler_reason": "active",
            }

    if not payload:
        return None

    return build_runtime_event(
        "activity.delta",
        event_domain="activity",
        payload=payload,
        legacy_fields={"delta": payload},
    )


def _build_system_delta_event(event: dict | None) -> dict | None:
    from remy.core.runtime_event_contract import build_runtime_event

    if not isinstance(event, dict):
        return None

    event_name = str(event.get("event_name") or event.get("type") or "")
    payload: dict | None = None

    if event_name == "approval.pending":
        action_id = event.get("action_id") or event.get("payload", {}).get("action_id")
        if action_id:
            payload = {
                "approvals": {
                    "upsert_pending": {
                        "id": action_id,
                        "description": str(event.get("description") or event.get("payload", {}).get("description") or "")[:100],
                        "age_sec": 0,
                    }
                }
            }
    elif event_name == "approval.resolved":
        action_id = event.get("action_id") or event.get("payload", {}).get("action_id")
        if action_id:
            payload = {
                "approvals": {
                    "remove_pending_id": action_id,
                }
            }
    elif event_name == "operator_alert":
        alert_id = event.get("id") or event.get("payload", {}).get("id")
        if alert_id:
            payload = {
                "operator_alerts": {
                    "upsert": {
                        "id": str(alert_id),
                        "type": str(event.get("type") or "operator_alert"),
                        "level": str(event.get("level") or event.get("payload", {}).get("level") or "info"),
                        "message": str(event.get("message") or event.get("payload", {}).get("message") or "")[:280],
                        "timestamp": event.get("timestamp") or event.get("payload", {}).get("timestamp"),
                        "acknowledged": bool(event.get("acknowledged") or event.get("payload", {}).get("acknowledged")),
                        "resolved": bool(event.get("resolved") or event.get("payload", {}).get("resolved")),
                        "resolved_at": event.get("resolved_at") or event.get("payload", {}).get("resolved_at"),
                        "repeat_count": int(event.get("repeat_count") or event.get("payload", {}).get("repeat_count") or 1),
                        "gateway_health": str(event.get("gateway_health") or event.get("payload", {}).get("gateway_health") or ""),
                        "health_level": str(event.get("health_level") or event.get("payload", {}).get("health_level") or ""),
                        "source": str(event.get("source") or event.get("payload", {}).get("source") or ""),
                        "scenario_id": str(event.get("scenario_id") or event.get("payload", {}).get("scenario_id") or ""),
                        "action_target": str(event.get("action_target") or event.get("payload", {}).get("action_target") or ""),
                        "artifact_ids": list(event.get("artifact_ids") or event.get("payload", {}).get("artifact_ids") or []),
                        "failure_code": str(event.get("failure_code") or event.get("payload", {}).get("failure_code") or ""),
                        "verification_status": str(event.get("verification_status") or event.get("payload", {}).get("verification_status") or ""),
                        "verification_reason": str(event.get("verification_reason") or event.get("payload", {}).get("verification_reason") or ""),
                        "eval_status": str(event.get("eval_status") or event.get("payload", {}).get("eval_status") or ""),
                        "requested": event.get("requested") if event.get("requested") is not None else event.get("payload", {}).get("requested"),
                        "applied": event.get("applied") if event.get("applied") is not None else event.get("payload", {}).get("applied"),
                        "skipped": event.get("skipped") if event.get("skipped") is not None else event.get("payload", {}).get("skipped"),
                    }
                }
            }
    elif event_name == "goal_selected":
        control_state = _get_runtime_transport_state()
        payload = {
            "autonomy": {
                "running": bool(control_state.get("running", True)),
                "session_id": control_state.get("session_id"),
                "goals": {
                    "upsert_active": {
                        "id": str(event.get("goal_id") or event.get("payload", {}).get("goal_id") or ""),
                        "content": str(event.get("description") or event.get("payload", {}).get("description") or "")[:80],
                        "priority": str(event.get("priority") or event.get("payload", {}).get("priority") or "medium"),
                    }
                },
            }
        }
    elif event_name == "goal_failed":
        goal_id = event.get("goal_id") or event.get("payload", {}).get("goal_id")
        if goal_id:
            payload = {
                "autonomy": {
                    "goals": {
                        "remove_active_id": str(goal_id),
                        "increment_blocked": 1,
                    }
                }
            }
    elif event_name in ("cycle_start", "cycle_end"):
        control_state = _get_runtime_transport_state()
        payload = {
            "autonomy": {
                "running": bool(control_state.get("running", True)),
                "session_id": control_state.get("session_id") or event.get("session_id") or event.get("payload", {}).get("session_id"),
            }
        }
    elif event_name == "budget_warning":
        payload = {
            "budget": {
                "alert_level": "warning",
                "warning_reason": str(event.get("reason") or event.get("payload", {}).get("reason") or ""),
            }
        }
    elif event_name == "llm_health":
        from remy.core.combined_runner import get_autonomy_control_state
        from remy.core.gateway import get_registry as get_gateway_registry

        control_state = get_autonomy_control_state()
        llm_status = str(event.get("status") or event.get("payload", {}).get("status") or "unknown")
        gateway_status = str(get_gateway_registry().summary().get("health") or "").lower()
        if not gateway_status:
            gateway_status = "degraded" if llm_status == "maintenance_only" else "ok" if llm_status == "recovered" else "unknown"

        if control_state.get("maintenance_only"):
            autonomy_health_status = "degraded"
        elif control_state.get("running"):
            autonomy_health_status = "running"
        elif llm_status == "recovered":
            autonomy_health_status = "starting"
        else:
            autonomy_health_status = "stopped"
        payload = {
            "gateway": {
                "status": gateway_status,
            },
            "channels": {
                "registry_summary": {
                    "health": gateway_status,
                },
                "autonomy": {
                    "maintenance_only": bool(control_state.get("maintenance_only", False)),
                    "health": {
                        "status": autonomy_health_status,
                    },
                },
            },
        }

    if not payload:
        return None

    return build_runtime_event(
        "system.delta",
        event_domain="system",
        payload=payload,
        legacy_fields={"delta": payload},
    )


def _classify_error(error_text: str) -> dict:
    """Classify error for user-friendly message + recovery estimation."""
    e = error_text.lower()
    if "quota" in e or "429" in e or "resource_exhausted" in e:
        return {
            "message": "API rate limit reached. Please wait a moment and try again.",
            "retryable": True,
            "error_class": "rate_limit",
        }
    if "api key" in e or "401" in e or "403" in e or "permission" in e:
        return {
            "message": "API authentication error. Check your API key in Settings.",
            "retryable": False,
            "error_class": "auth",
        }
    if "timeout" in e or "deadline" in e:
        return {
            "message": "Request timed out. Try again or simplify your message.",
            "retryable": True,
            "error_class": "timeout",
        }
    if "connect" in e or "network" in e or "unreachable" in e or "getaddrinfo" in e:
        return {
            "message": "Network error. Check your internet connection.",
            "retryable": True,
            "error_class": "network",
        }
    if "subscriptable" in e:
        return {
            "message": "API response parsing error. Retrying automatically...",
            "retryable": True,
            "error_class": "transient",
        }
    return {
        "message": "Something went wrong. Try again.",
        "retryable": True,
        "error_class": "unknown",
    }


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """Real-time chat via WebSocket."""
    api = _get_api()
    await websocket.accept()
    api.metrics_collector.ws_connected("chat")
    manager = api.get_session_manager()
    logger.info("WebSocket chat connected")

    _active_task: asyncio.Task | None = None
    inbox: asyncio.Queue = asyncio.Queue()

    async def _receive_loop():
        try:
            while True:
                data = await websocket.receive_json()
                await inbox.put(data)
        except WebSocketDisconnect:
            await inbox.put(None)
        except Exception:
            await inbox.put(None)

    receiver = asyncio.create_task(_receive_loop())

    async def _cancel_active():
        nonlocal _active_task
        if _active_task and not _active_task.done():
            _active_task.cancel()
            try:
                await _active_task
            except (asyncio.CancelledError, Exception):
                pass
            _active_task = None

    async def _do_generation(user_text: str):
        streamed_any = False
        partial_text = ""
        final_answer_text = ""
        generation_ok = False
        try:
            async for event in manager.gemini_respond_stream(user_text):
                if event["type"] == "token":
                    await websocket.send_json({"type": "token", "content": event["content"]})
                    partial_text += event["content"]
                    streamed_any = True
                elif event["type"] == "tool_start":
                    await websocket.send_json({
                        "type": "tool_start",
                        "content": event["tool"],
                        "args": event.get("args", ""),
                    })
                elif event["type"] == "tool_end":
                    await websocket.send_json({
                        "type": "tool_end",
                        "content": event["tool"],
                        "result": event.get("result", ""),
                    })
                elif event["type"] == "thinking":
                    await websocket.send_json({
                        "type": "thinking",
                        "content": event.get("content", "Thinking..."),
                    })
                elif event["type"] == "final":
                    final_ev_text = event.get("text", "")
                    logger.debug(f"Final event: streamed_any={streamed_any}, text_len={len(final_ev_text)}")
                    final_answer_text = final_ev_text or partial_text
                    generation_ok = True
                    if not streamed_any and final_ev_text:
                        await websocket.send_json({"type": "text", "content": final_ev_text})
                    if event.get("factuality"):
                        await websocket.send_json(
                            {
                                "type": "factuality",
                                "factuality": event["factuality"],
                            }
                        )
        except asyncio.CancelledError:
            logger.info("Generation cancelled by user")
            try:
                await websocket.send_json(
                    {
                        "type": "stopped",
                        "content": partial_text,
                    }
                )
            except Exception:
                pass
            return
        except Exception as e:
            logger.error(f"Gemini respond error: {e}")
            err = _classify_error(str(e))
            try:
                await websocket.send_json(
                    {
                        "type": "error",
                        "content": err["message"],
                        "retryable": err["retryable"],
                        "error_class": err["error_class"],
                    }
                )
            except Exception:
                pass
        finally:
            # Shadow session-state fold: background, fire-and-forget, only for
            # successfully completed exchanges (cancelled/failed are skipped so
            # the frozen state is never poisoned). Never affects the answer.
            if generation_ok and final_answer_text.strip():
                try:
                    from remy.core.session_state_shadow import schedule_shadow_fold

                    session = manager.get_or_create_session()
                    schedule_shadow_fold(
                        session.session_id,
                        user_text,
                        final_answer_text,
                        session_log=list(session.session_log),
                    )
                except Exception as shadow_exc:  # noqa: BLE001
                    logger.debug(f"shadow fold scheduling skipped: {shadow_exc}")
            try:
                await websocket.send_json({"type": "done"})
            except Exception:
                pass

    try:
        while True:
            data = await inbox.get()
            if data is None:
                break

            msg_type = data.get("type")

            if msg_type == "cancel":
                await _cancel_active()
                continue

            if msg_type == "message":
                user_text = data.get("text", "").strip()
                if not user_text:
                    continue

                # LLM Optimization Lab: A/B compare raw vs reduced context with
                # live cost priced against the model chosen in the lab window.
                if data.get("context_reducer_compare"):
                    await _cancel_active()
                    await websocket.send_json({"type": "typing"})
                    try:
                        from remy.core.context_reducer import (
                            compare_context_reducer,
                            make_gemini_llm_func,
                        )

                        session = manager.get_or_create_session()
                        lab_model = str(data.get("model") or "").strip() or None
                        lab_llm_func = None
                        if lab_model and lab_model.lower().startswith("gemini"):
                            lab_llm_func = make_gemini_llm_func(lab_model)
                        report = await compare_context_reducer(
                            user_text=user_text,
                            session_log=session.session_log,
                            history=session.history,
                            session_id=session.session_id,
                            model=lab_model,
                            llm_func=lab_llm_func,
                        )
                        await websocket.send_json(
                            {"type": "context_reducer_compare", "report": report}
                        )
                    except Exception as e:
                        logger.error(f"ContextReducer compare error: {e}")
                        await websocket.send_json({"type": "error", "content": str(e)[:300]})
                    await websocket.send_json({"type": "done"})
                    continue

                await _cancel_active()
                await websocket.send_json({"type": "typing"})
                _active_task = asyncio.create_task(_do_generation(user_text))

            elif msg_type == "voice":
                audio_b64 = data.get("audio", "")
                mime_type = data.get("mime_type", "audio/webm")

                if not audio_b64:
                    await websocket.send_json(
                        {"type": "error", "content": "No audio data received."}
                    )
                    continue

                try:
                    audio_bytes = base64.b64decode(audio_b64)
                except Exception:
                    await websocket.send_json(
                        {"type": "error", "content": "Invalid audio encoding."}
                    )
                    continue

                await _cancel_active()
                await websocket.send_json({"type": "typing"})

                try:
                    result = await manager.gemini_respond_multimodal(
                        attachments=[{"mime_type": mime_type, "data": audio_bytes}],
                        is_voice=True,
                    )
                    await websocket.send_json(
                        {
                            "type": "text",
                            "content": result["response"],
                            "speak": True,
                        }
                    )
                except Exception as e:
                    logger.error(f"Voice respond error: {e}")
                    err = _classify_error(str(e))
                    await websocket.send_json(
                        {
                            "type": "error",
                            "content": err["message"],
                            "retryable": err["retryable"],
                            "error_class": err["error_class"],
                        }
                    )

                await websocket.send_json({"type": "done"})

            elif msg_type == "file":
                file_b64 = data.get("data", "")
                mime_type = data.get("mime_type", "application/octet-stream")
                file_name = data.get("name", "unknown")
                accompanying_text = data.get("text", "")

                if not file_b64:
                    await websocket.send_json(
                        {"type": "error", "content": "No file data received."}
                    )
                    continue

                try:
                    file_bytes = base64.b64decode(file_b64)
                except Exception:
                    await websocket.send_json(
                        {"type": "error", "content": "Invalid file encoding."}
                    )
                    continue

                await _cancel_active()
                await websocket.send_json({"type": "typing"})

                prompt = (
                    accompanying_text
                    or f"The user uploaded a file named '{file_name}'. Analyze it and respond."
                )

                try:
                    result = await manager.gemini_respond_multimodal(
                        text=prompt,
                        attachments=[{"mime_type": mime_type, "data": file_bytes}],
                    )
                    await websocket.send_json({"type": "text", "content": result["response"]})
                except Exception as e:
                    logger.error(f"File respond error: {e}")
                    err = _classify_error(str(e))
                    await websocket.send_json(
                        {
                            "type": "error",
                            "content": err["message"],
                            "retryable": err["retryable"],
                            "error_class": err["error_class"],
                        }
                    )

                await websocket.send_json({"type": "done"})

            elif msg_type == "files":
                files_list = data.get("files", [])
                accompanying_text = data.get("text", "")

                if not files_list:
                    await websocket.send_json({"type": "error", "content": "No files received."})
                    continue

                attachments = []
                file_names = []
                for f in files_list:
                    f_b64 = f.get("data", "")
                    f_mime = f.get("mime_type", "application/octet-stream")
                    f_name = f.get("name", "unknown")
                    if not f_b64:
                        continue
                    try:
                        attachments.append({"mime_type": f_mime, "data": base64.b64decode(f_b64)})
                        file_names.append(f_name)
                    except Exception:
                        logger.warning("Skipping file with invalid encoding: %s", f_name)

                if not attachments:
                    await websocket.send_json(
                        {"type": "error", "content": "No valid file data received."}
                    )
                    continue

                await _cancel_active()
                await websocket.send_json({"type": "typing"})

                names_str = ", ".join(file_names)
                prompt = (
                    accompanying_text
                    or f"The user uploaded {len(attachments)} file(s): {names_str}. Analyze them and respond."
                )

                try:
                    result = await manager.gemini_respond_multimodal(
                        text=prompt,
                        attachments=attachments,
                    )
                    await websocket.send_json({"type": "text", "content": result["response"]})
                except Exception as e:
                    logger.error(f"Multi-file respond error: {e}")
                    err = _classify_error(str(e))
                    await websocket.send_json(
                        {
                            "type": "error",
                            "content": err["message"],
                            "retryable": err["retryable"],
                            "error_class": err["error_class"],
                        }
                    )

                await websocket.send_json({"type": "done"})

            elif msg_type == "new_session":
                await _cancel_active()
                await manager.close_session()
                manager.get_or_create_session()
                await websocket.send_json({"type": "session_reset"})

    except WebSocketDisconnect:
        logger.info("WebSocket chat disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        receiver.cancel()
        if _active_task and not _active_task.done():
            _active_task.cancel()
        try:
            await asyncio.wait_for(manager.close_session(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Session close timed out (10s) — skipping summary")
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("Session close interrupted by shutdown")
        except Exception as e:
            logger.warning(f"Session close on disconnect failed: {e}")
        api.metrics_collector.ws_disconnected("chat")


@router.websocket("/ws/compare")
async def websocket_compare(websocket: WebSocket):
    """Multi-model comparison WebSocket — runs same prompt against multiple models in parallel."""
    api = _get_api()
    await websocket.accept()
    manager = api.get_session_manager()
    logger.info("WebSocket compare connected")

    try:
        data = await websocket.receive_json()
        user_text = (data.get("text") or "").strip()
        models = data.get("models") or []

        if not user_text or not models:
            await websocket.send_json({"type": "error", "content": "Missing text or models."})
            return

        async def _stream_model(model: str):
            try:
                async for event in manager.gemini_respond_stream(user_text, model_override=model):
                    if event["type"] == "token":
                        await websocket.send_json({"type": "token", "model": model, "content": event["content"]})
                    elif event["type"] in ("final", "error"):
                        pass
                await websocket.send_json({"type": "done", "model": model})
            except Exception as e:
                logger.error(f"Compare stream error for {model}: {e}")
                await websocket.send_json({"type": "error", "model": model, "content": str(e)})

        await asyncio.gather(*[_stream_model(m) for m in models])
        await websocket.send_json({"type": "all_done"})

    except WebSocketDisconnect:
        logger.info("WebSocket compare disconnected")
    except Exception as e:
        logger.error(f"WebSocket compare error: {e}")


@router.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """Real-time Voice-to-Voice WebSocket proxy for Gemini Live API."""
    api = _get_api()
    await websocket.accept()
    api.metrics_collector.ws_connected("live")
    manager = api.get_session_manager()
    logger.info("WebSocket live connected")

    if manager.readonly or not manager.client:
        await websocket.send_json({"type": "error", "content": "No API key configured."})
        await websocket.close()
        return

    import traceback

    from google.genai import types

    from remy.core.brain_tools import build_system_instruction, execute_tool, get_registry

    registry = get_registry()
    tools_config = registry.get_tools_config()
    session_id = manager.get_or_create_session().session_id

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        media_resolution="MEDIA_RESOLUTION_MEDIUM",
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=api.settings.GEMINI_VOICE
                )
            )
        ),
        system_instruction=types.Content(
            parts=[types.Part(text=build_system_instruction(channel="voice"))]
        ),
        tools=tools_config,
        context_window_compression=types.ContextWindowCompressionConfig(
            trigger_tokens=25600,
            sliding_window=types.SlidingWindow(target_tokens=12800),
        ),
    )

    try:
        async with manager.client.aio.live.connect(
            model=api.settings.GEMINI_MODEL, config=config
        ) as session:

            async def receive_from_browser():
                try:
                    while True:
                        msg = await websocket.receive()
                        if "bytes" in msg:
                            await session.send(
                                input={"mime_type": "audio/pcm", "data": msg["bytes"]}
                            )
                        elif "text" in msg:
                            try:
                                data = json.loads(msg["text"])
                                if data.get("type") == "message":
                                    await session.send(
                                        input=data.get("text") or ".", end_of_turn=True
                                    )
                            except Exception:
                                pass
                except WebSocketDisconnect:
                    logger.info("Browser disconnected from Live WS")
                except Exception as e:
                    if "disconnect" in str(e).lower():
                        logger.debug("Browser WS already disconnected: %s", e)
                    else:
                        logger.error(f"Error receiving from browser: {e}")

            async def receive_from_gemini():
                try:
                    while True:
                        turn = session.receive()
                        async for response in turn:
                            if data := response.data:
                                await websocket.send_bytes(data)
                            if text := response.text:
                                await websocket.send_json({"type": "text", "content": text})
                            if response.tool_call:
                                for fc in response.tool_call.function_calls:
                                    logger.info(f"Live Tool call: {fc.name}({fc.args})")
                                    fc_args = dict(fc.args)
                                    result = await run_in_thread(
                                        execute_tool, fc.name, fc_args, session_id
                                    )
                                    logger.info(f"Live Tool result: {result[:200]}")
                                    await session.send_tool_response(
                                        function_responses=types.FunctionResponse(
                                            name=fc.name,
                                            response={"result": result},
                                            id=fc.id,
                                        )
                                    )
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error receiving from Gemini: {e}")

            browser_task = asyncio.create_task(receive_from_browser())
            gemini_task = asyncio.create_task(receive_from_gemini())

            done, pending = await asyncio.wait(
                [browser_task, gemini_task], return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()

    except Exception as e:
        logger.error(f"Gemini Live session error: {e}")
        traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "content": f"Live API error: {e}"})
        except Exception:
            logger.debug("Failed to send error to WS client")
    finally:
        api.metrics_collector.ws_disconnected("live")
        try:
            await websocket.close()
        except Exception:
            logger.debug("WS already closed")
        logger.info("WebSocket live disconnected")


# ============== APPROVAL REST + WEBSOCKET ==============


@router.get("/approvals")
async def get_pending_approvals():
    """List all currently pending approval actions."""
    from remy.core.combined_runner import get_approval_runtime_snapshot

    approvals = await run_in_thread(get_approval_runtime_snapshot, goal_limit=3, approval_limit=100)
    return {"pending": approvals.get("pending", [])}


@router.post("/approvals/{action_id}/approve")
async def approve_action(action_id: str):
    """Approve a pending action by ID (full UUID or first-8 prefix)."""
    from remy.core.combined_runner import resolve_operator_approval

    return resolve_operator_approval(action_id, approved=True, decided_by="web")


@router.post("/approvals/{action_id}/reject")
async def reject_action(action_id: str):
    """Reject a pending action by ID (full UUID or first-8 prefix)."""
    from remy.core.combined_runner import resolve_operator_approval

    return resolve_operator_approval(action_id, approved=False, decided_by="web")


@router.websocket("/ws/approvals")
async def websocket_approvals(websocket: WebSocket):
    """Push approval.pending / approval.resolved events to the Web GUI in real-time."""
    api = _get_api()
    await websocket.accept()
    queue = api.event_bus.subscribe()
    logger.info("Approvals WebSocket connected (%d subscribers)", _runtime_subscriber_count())

    try:
        await _send_approval_snapshot(websocket)
    except Exception as e:
        logger.debug("Could not send approval snapshot: %s", e)

    try:

        async def _forward_events():
            while True:
                event = await queue.get()
                if event.get("type") in ("approval.pending", "approval.resolved"):
                    await websocket.send_json(event)

        await _wait_for_websocket_tasks(
            asyncio.create_task(_forward_events()),
            asyncio.create_task(_listen_websocket_client(websocket)),
        )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("Approvals WebSocket error: %s", e)
    finally:
        api.event_bus.unsubscribe(queue)
        logger.info(
            "Approvals WebSocket disconnected (%d subscribers)", _runtime_subscriber_count()
        )


# ============== GUIDANCE REST + WEBSOCKET ==============


@router.get("/guidance")
async def get_pending_guidance():
    """List all currently pending guidance requests."""
    from remy.core.combined_runner import get_guidance_runtime_snapshot

    guidance = get_guidance_runtime_snapshot(limit=50)
    return {"pending": guidance.get("pending", [])}


@router.post("/guidance/{request_id}/answer")
async def answer_guidance(request_id: str, body: dict):
    """Submit an answer to a pending guidance request."""
    from remy.core.combined_runner import resolve_operator_guidance

    answer = body.get("answer", "").strip()
    if not answer:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail="answer is required")
    return resolve_operator_guidance(request_id, answer)


@router.websocket("/ws/guidance")
async def websocket_guidance(websocket: WebSocket):
    """Push guidance.pending / guidance.resolved events to Web GUI in real-time."""
    api = _get_api()
    await websocket.accept()
    queue = api.event_bus.subscribe()
    logger.info("Guidance WebSocket connected (%d subscribers)", _runtime_subscriber_count())

    try:
        await _send_guidance_snapshot(websocket)
    except Exception as e:
        logger.debug("Could not send guidance snapshot: %s", e)

    try:

        async def _forward_events():
            while True:
                event = await queue.get()
                if event.get("type") in ("guidance.pending", "guidance.resolved"):
                    await websocket.send_json(event)

        await _wait_for_websocket_tasks(
            asyncio.create_task(_forward_events()),
            asyncio.create_task(_listen_websocket_client(websocket)),
        )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("Guidance WebSocket error: %s", e)
    finally:
        api.event_bus.unsubscribe(queue)
        logger.info(
            "Guidance WebSocket disconnected (%d subscribers)", _runtime_subscriber_count()
        )


@router.websocket("/ws/human-loop")
async def websocket_human_loop(websocket: WebSocket):
    """Push approval + guidance events over a shared human-loop stream."""
    api = _get_api()
    await websocket.accept()
    queue = api.event_bus.subscribe()
    logger.info("Human-loop WebSocket connected (%d subscribers)", _runtime_subscriber_count())

    try:
        await _send_approval_snapshot(websocket)
        await _send_guidance_snapshot(websocket)
    except Exception as e:
        logger.debug("Could not send human-loop snapshot: %s", e)

    try:

        async def _forward_events():
            while True:
                event = await queue.get()
                event_type = event.get("type", "")
                event_domain = event.get("event_domain", "")
                if event_domain in ("approval", "guidance") or event_type.startswith("approval.") or event_type.startswith("guidance."):
                    await websocket.send_json(event)

        await _wait_for_websocket_tasks(
            asyncio.create_task(_forward_events()),
            asyncio.create_task(_listen_websocket_client(websocket)),
        )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("Human-loop WebSocket error: %s", e)
    finally:
        api.event_bus.unsubscribe(queue)
        logger.info(
            "Human-loop WebSocket disconnected (%d subscribers)", _runtime_subscriber_count()
        )


@router.websocket("/ws/runtime")
async def websocket_runtime(websocket: WebSocket):
    """Unified runtime stream for activity, approvals, guidance, and budget events."""
    api = _get_api()
    await websocket.accept()
    api.metrics_collector.ws_connected("activity")
    queue = api.event_bus.subscribe()
    logger.info("Runtime WebSocket connected (%d subscribers)", _runtime_subscriber_count())

    try:
        await _send_system_snapshot(websocket)
        await _send_activity_snapshot(websocket)
        await _send_budget_init(websocket, api)
        await _send_approval_snapshot(websocket)
        await _send_guidance_snapshot(websocket)
    except Exception as e:
        logger.debug("Could not send runtime snapshot: %s", e)

    try:

        async def forward_events():
            while True:
                event = await queue.get()
                await websocket.send_json(event)
                activity_delta = _build_activity_delta_event(event)
                if activity_delta:
                    await websocket.send_json(activity_delta)
                system_delta = _build_system_delta_event(event)
                if system_delta:
                    await websocket.send_json(system_delta)

        await _wait_for_websocket_tasks(
            asyncio.create_task(forward_events()),
            asyncio.create_task(_listen_websocket_client(websocket)),
        )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        if "close message has been sent" in str(e):
            pass  # Client disconnected mid-send — normal
        else:
            logger.error("Runtime WebSocket error: %s", e)
    finally:
        api.event_bus.unsubscribe(queue)
        logger.info("Runtime WebSocket disconnected (%d subscribers)", _runtime_subscriber_count())


# ============== ACTIVITY WEBSOCKET ==============


@router.websocket("/ws/activity")
async def websocket_activity(websocket: WebSocket):
    """Real-time autonomous thought stream via WebSocket."""
    api = _get_api()
    await websocket.accept()
    api.metrics_collector.ws_connected("activity")
    queue = api.event_bus.subscribe()
    logger.info("Activity WebSocket connected (%d subscribers)", _runtime_subscriber_count())

    try:
        await _send_budget_init(websocket, api)
    except Exception as e:
        logger.debug("Could not send budget_init: %s", e)

    try:

        async def forward_events():
            while True:
                event = await queue.get()
                await websocket.send_json(event)

        await _wait_for_websocket_tasks(
            asyncio.create_task(forward_events()),
            asyncio.create_task(_listen_websocket_client(websocket)),
        )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("Activity WebSocket error: %s", e)
    finally:
        api.metrics_collector.ws_disconnected("activity")
        api.event_bus.unsubscribe(queue)
        logger.info(
            "Activity WebSocket disconnected (%d subscribers)", _runtime_subscriber_count()
        )
