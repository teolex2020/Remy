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
    from remy.core.error_classification import classify_llm_error

    return classify_llm_error(error_text)


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """Real-time chat via WebSocket."""
    api = _get_api()
    await websocket.accept()
    api.metrics_collector.ws_connected("chat")
    manager = api.get_session_manager()
    logger.info("WebSocket chat connected")

    _active_task: asyncio.Task | None = None
    _active_token = None
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
        nonlocal _active_task, _active_token
        if _active_task and not _active_task.done():
            if _active_token is not None:
                _active_token.cancel("Cancelled from the web interface")
            _active_task.cancel()
            try:
                await _active_task
            except (asyncio.CancelledError, Exception):
                pass
            _active_task = None
            _active_token = None

    async def _do_generation(
        user_text: str,
        *,
        model_routing_enabled: bool = False,
        workspace_id: str | None = None,
        team_mode: str = "off",
    ):
        streamed_any = False
        partial_text = ""
        final_answer_text = ""
        generation_ok = False
        terminal_status = "failed"
        terminal_reason = ""
        terminal_error = ""
        token_usage: dict = {}
        run = None
        coordinator = None
        from remy.core.cancellation import bind_cancellation_token
        from remy.core.run_envelope import (
            RunCoordinator,
            RunLimitExceeded,
            finish_run,
            register_run_stop,
            start_run,
            unregister_run_stop,
        )

        try:
            session = manager.get_or_create_session()
            from remy.core.project_store import get_project_store

            owner = get_project_store().require_project(session.project_id)
            run = start_run(
                kind="chat",
                source_id=session.session_id,
                goal=user_text,
                owner_project_id=owner.project_id,
                brain_id=owner.brain_id,
                conversation_id=session.session_id,
                channel="web",
                idempotency_class="read_only",
                metadata={
                    "workspace_id": workspace_id or "",
                    "team_mode": team_mode,
                },
            )
            coordinator = RunCoordinator(run["attempt_id"])
            register_run_stop(run["run_id"], _active_token.cancel)
            await websocket.send_json({"type": "run_state", "run": run})
        except Exception as run_exc:
            logger.warning("Chat run envelope could not start: %s", run_exc)
            run = None
            coordinator = None

        try:
            with bind_cancellation_token(_active_token):
                async for event in manager.gemini_respond_stream(
                    user_text,
                    model_routing_enabled=model_routing_enabled,
                    workspace_id=workspace_id,
                    team_mode=team_mode,
                ):
                    if event["type"] == "token":
                        await websocket.send_json({"type": "token", "content": event["content"]})
                        partial_text += event["content"]
                        streamed_any = True
                    elif event["type"] == "provisional_token":
                        content = event.get("content", "")
                        await websocket.send_json({"type": "provisional_token", "content": content})
                        partial_text += content
                        streamed_any = True
                    elif event["type"] == "provisional_reset":
                        await websocket.send_json({"type": "provisional_reset"})
                        partial_text = ""
                        streamed_any = False
                    elif event["type"] == "provisional_commit":
                        await websocket.send_json({"type": "provisional_commit"})
                    elif event["type"] == "provider_status":
                        if coordinator is not None:
                            coordinator.heartbeat(
                                phase=str(event.get("phase") or "working"),
                                step=str(event.get("message") or "Model is working"),
                            )
                        await websocket.send_json({
                            "type": "provider_status",
                            "phase": event.get("phase", "working"),
                            "model": event.get("model", ""),
                            "provider": event.get("provider", ""),
                            "message": event.get("message", ""),
                        })
                    elif event["type"] == "tool_start":
                        if coordinator is not None:
                            coordinator.step(
                                f"Tool {event['tool']}",
                                signature=f"tool:{event['tool']}:{event.get('args', '')}",
                            )
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
                    elif event["type"] == "team_status":
                        await websocket.send_json(event)
                    elif event["type"] == "final":
                        final_ev_text = event.get("text", "")
                        logger.debug(f"Final event: streamed_any={streamed_any}, text_len={len(final_ev_text)}")
                        final_answer_text = final_ev_text or partial_text
                        generation_ok = True
                        terminal_status = "completed"
                        token_usage = dict(event.get("token_usage") or {})
                        if coordinator is not None:
                            coordinator.consume_tokens(
                                input_tokens=int(token_usage.get("input_tokens") or 0),
                                output_tokens=int(token_usage.get("output_tokens") or 0),
                            )
                        if not streamed_any and final_ev_text:
                            await websocket.send_json({"type": "text", "content": final_ev_text})
                        if event.get("factuality"):
                            await websocket.send_json(
                                {
                                    "type": "factuality",
                                    "factuality": event["factuality"],
                                }
                            )
        except RunLimitExceeded as exc:
            terminal_status = "completed_with_limits"
            terminal_reason = exc.reason
            terminal_error = str(exc)
            logger.warning("Chat run stopped by %s: %s", exc.reason, exc)
            try:
                await websocket.send_json({
                    "type": "error",
                    "content": f"Run stopped safely: {exc}",
                    "retryable": False,
                    "error_class": "run_limit",
                })
            except Exception:
                pass
        except asyncio.CancelledError:
            terminal_status = "cancelled"
            terminal_reason = "user_stopped"
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
            terminal_status = "failed"
            terminal_error = str(e)
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
                    from remy.core.learning_review import stage_learning_review

                    session = manager.get_or_create_session()
                    await asyncio.to_thread(
                        stage_learning_review,
                        session_id=session.session_id,
                        user_text=user_text,
                        assistant_text=final_answer_text,
                    )
                except Exception as learning_exc:  # noqa: BLE001
                    logger.debug("learning review staging skipped: %s", learning_exc)
                try:
                    from remy.core.pipeline_evolution import observe_successful_turn

                    session = manager.get_or_create_session()
                    candidate = await asyncio.to_thread(
                        observe_successful_turn,
                        session_id=session.session_id,
                        user_text=user_text,
                        session_log=list(session.session_log),
                    )
                    if candidate and candidate.get("status") == "draft":
                        await websocket.send_json({
                            "type": "pipeline_candidate",
                            "candidate": {
                                "candidate_id": candidate["candidate_id"],
                                "title": candidate["title"],
                                "occurrence_count": candidate["occurrence_count"],
                                "risk": candidate["risk"],
                            },
                        })
                except Exception as pipeline_exc:  # noqa: BLE001
                    logger.debug("pipeline candidate staging skipped: %s", pipeline_exc)
            try:
                final_run = {}
                if run is not None:
                    try:
                        final_run = finish_run(
                            run["attempt_id"],
                            status=terminal_status,
                            stop_reason=terminal_reason,
                            error=terminal_error,
                            output_ref=(
                                f"conversation:{run.get('conversation_id', '')}"
                                if generation_ok else ""
                            ),
                        )
                    except (KeyError, RuntimeError):
                        final_run = run
                    unregister_run_stop(run["run_id"])
                    await websocket.send_json({"type": "run_state", "run": final_run})
                await websocket.send_json({"type": "done", "token_usage": token_usage, "run": final_run if run is not None else {}})
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

                if data.get("context_reducer_apply"):
                    await _cancel_active()
                    await websocket.send_json({"type": "typing"})
                    try:
                        from remy.core.context_reducer import apply_context_reducer

                        session = manager.get_or_create_session()
                        session.session_log.append({"type": "user_text", "text": user_text[:200]})
                        result = await apply_context_reducer(
                            user_text=user_text,
                            session_log=session.session_log,
                            history=session.history,
                            session_id=session.session_id,
                        )
                        answer = str(result.get("answer") or "")
                        report = result.get("report") or {}
                        session.session_log.append({
                            "type": "model_response",
                            "text": answer[:200],
                            "full_text": answer,
                            "source": "context_reducer_apply",
                        })
                        await websocket.send_json({"type": "token", "content": answer})
                        await websocket.send_json({"type": "llm_optimization_apply", "report": report})
                    except Exception as e:
                        logger.error("ContextReducer apply error: %s", e)
                        err = _classify_error(str(e))
                        await websocket.send_json({
                            "type": "error",
                            "content": err["message"],
                            "retryable": err["retryable"],
                            "error_class": err["error_class"],
                        })
                    await websocket.send_json({"type": "done"})
                    continue

                await _cancel_active()
                await websocket.send_json({"type": "typing"})
                from remy.core.cancellation import CancellationToken
                _active_token = CancellationToken()
                workspace_id = str(data.get("workspace_id") or "").strip() or None
                from remy.core.team_planner import normalize_team_mode

                team_mode = normalize_team_mode(data.get("team_mode"))
                _active_task = asyncio.create_task(
                    _do_generation(
                        user_text,
                        workspace_id=workspace_id,
                        team_mode=team_mode,
                    )
                )

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
                from remy.core.conversation_store import get_conversation_store
                from remy.core.microbrain import current_project_id

                project_id = current_project_id()
                conversation = get_conversation_store(project_id).create()
                await manager.switch_conversation(
                    project_id,
                    conversation.conversation_id,
                )
                await websocket.send_json({
                    "type": "session_reset",
                    "conversation": conversation.to_dict(),
                })

    except WebSocketDisconnect:
        logger.info("WebSocket chat disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        receiver.cancel()
        if _active_task and not _active_task.done():
            _active_task.cancel()
        try:
            await manager.suspend_session()
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("Session suspend interrupted by shutdown")
        except Exception as e:
            logger.warning(f"Session suspend on disconnect failed: {e}")
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
        raw_models = data.get("models") or []
        models = list(dict.fromkeys(
            str(model or "").strip() for model in raw_models if str(model or "").strip()
        ))[:4]

        if not user_text or not models:
            await websocket.send_json({"type": "error", "content": "Missing text or models."})
            return

        send_lock = asyncio.Lock()

        async def _send(payload: dict):
            async with send_lock:
                await websocket.send_json(payload)

        async def _stream_model(model: str):
            streamed_any = False
            final_text = ""
            try:
                stream = (
                    manager.compare_model_stream(user_text, model)
                    if hasattr(manager, "compare_model_stream")
                    else manager.gemini_respond_stream(user_text, model_override=model)
                )
                async for event in stream:
                    if event["type"] == "token":
                        content = str(event.get("content") or "")
                        if content:
                            streamed_any = True
                            await _send({
                                "type": "token",
                                "model": model,
                                "content": content,
                            })
                    elif event["type"] == "final":
                        final_text = str(event.get("text") or "")
                    elif event["type"] == "error":
                        raise RuntimeError(str(event.get("content") or "Model failed"))
                if not streamed_any and final_text:
                    await _send({
                        "type": "token",
                        "model": model,
                        "content": final_text,
                    })
                if not streamed_any and not final_text:
                    raise RuntimeError("Model completed without response text")
                await _send({"type": "done", "model": model})
            except Exception as e:
                logger.error(f"Compare stream error for {model}: {e}")
                await _send({
                    "type": "error",
                    "model": model,
                    "content": str(e),
                })

        await asyncio.gather(*[_stream_model(m) for m in models])
        await _send({"type": "all_done"})

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
                owner_project_id = str(event.get("owner_project_id") or "")
                if owner_project_id:
                    from remy.core.microbrain import current_project_id

                    if owner_project_id != current_project_id():
                        continue
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
