import asyncio
from unittest.mock import AsyncMock

import pytest

from remy.web.routes.websocket import (
    _build_activity_delta_event,
    _build_system_delta_event,
    _is_websocket_lifecycle_runtime_error,
    _listen_websocket_client,
    _send_approval_snapshot,
    _send_activity_snapshot,
    _send_budget_init,
    _send_guidance_snapshot,
    _wait_for_websocket_tasks,
    answer_guidance,
    get_pending_guidance,
    get_pending_approvals,
)


def test_activity_delta_includes_plan_step_state():
    event = _build_activity_delta_event({
        "type": "plan_step",
        "step_num": 2,
        "total_steps": 5,
        "step_description": "Collect the primary source",
        "plan_type": "linear",
    })

    assert event is not None
    assert event["event_name"] == "activity.delta"
    assert event["event_domain"] == "activity"
    assert event["payload"]["current_step"] == {
        "instruction": "Collect the primary source",
        "step_num": 2,
        "total_steps": 5,
        "plan_type": "linear",
    }


def test_activity_delta_clears_current_task_on_tool_result():
    event = _build_activity_delta_event({
        "type": "tool_result",
        "tool": "browse_page",
        "result": "ok",
    })

    assert event is not None
    assert event["payload"]["current_task"] is None


def test_activity_stream_can_render_reason_aware_approval_resolution():
    event = {
        "type": "approval.resolved",
        "action_id": "appr-1",
        "decision": "approved",
        "description": "Routing pressure approval: specialist 'researcher' is degraded",
        "routing_pressure": True,
    }

    assert event["type"] == "approval.resolved"
    assert event["decision"] == "approved"
    assert event["description"].startswith("Routing pressure approval")
    assert event["routing_pressure"] is True


def test_activity_delta_includes_last_cycle_result_from_evaluation():
    event = _build_activity_delta_event({
        "type": "evaluation",
        "success": False,
        "confidence": 0.42,
        "reason": "The action timed out",
        "goal_completed": False,
    })

    assert event is not None
    assert event["payload"]["current_task"] is None
    assert event["payload"]["last_cycle_result"] == {
        "success": False,
        "confidence": 0.42,
        "reason": "The action timed out",
        "goal_completed": False,
        "decision": "failed",
    }


def test_activity_delta_carries_role_and_agent_response_state():
    role_event = _build_activity_delta_event({
        "type": "role_selected",
        "role": "researcher",
    })
    response_event = _build_activity_delta_event({
        "type": "agent_response",
        "response": "Collected the primary sources and prepared a summary.",
        "duration_ms": 3210,
        "tokens_estimated": 876,
    })

    assert role_event is not None
    assert role_event["payload"]["current_role"] == "researcher"

    assert response_event is not None
    assert response_event["payload"]["last_agent_response"] == {
        "response": "Collected the primary sources and prepared a summary.",
        "duration_ms": 3210,
        "tokens_estimated": 876,
    }


def test_activity_delta_marks_research_progress_for_research_tools():
    event = _build_activity_delta_event({
        "type": "tool_call",
        "tool": "web_search",
        "args_summary": "latest SEC filing",
    })

    assert event is not None
    assert event["payload"]["current_task"]["tool"] == "web_search"
    assert event["payload"]["last_research_activity"] == {
        "tool": "web_search",
        "summary": "Research step: web_search",
    }


def test_activity_delta_adds_research_session_trace_for_research_tool(monkeypatch):
    monkeypatch.setattr(
        "remy.web.routes.websocket._build_research_session_state",
        lambda goal_id: {
            "session_id": f"rs-{goal_id}",
            "topic": "VAT reporting",
            "recent_queries": ["vat deadlines 2026"],
            "knowledge_gaps": ["Promote fetched sources into accepted evidence."],
        },
    )

    event = _build_activity_delta_event({
        "type": "tool_call",
        "tool": "web_search",
        "args_summary": "vat deadlines 2026",
        "goal_id": "goal-vat",
    })

    assert event is not None
    assert event["payload"]["research_session"]["session_id"] == "rs-goal-vat"
    assert event["payload"]["research_session"]["knowledge_gaps"][0].startswith("Promote fetched")


def test_activity_delta_maps_goal_selected_to_current_goal_not_mission(monkeypatch):
    monkeypatch.setattr(
        "remy.web.routes.websocket._get_runtime_transport_state",
        lambda: {"running": True, "session_id": "sess-alpha"},
    )
    monkeypatch.setattr(
        "remy.web.routes.websocket._build_research_session_state",
        lambda goal_id: {
            "session_id": f"rs-{goal_id}",
            "topic": "Primary source trail",
            "recent_queries": ["primary source trail sec filing"],
            "knowledge_gaps": ["Improve citation coverage for current findings."],
        },
    )
    monkeypatch.setattr(
        "remy.core.autonomy_goals._load_missions",
        lambda: [
            {
                "id": "mission-alpha",
                "description": "Alpha mission",
                "tasks": [
                    {"action": "Collect filings"},
                    {"action": "Compare source trail"},
                    {"action": "Draft summary"},
                ],
            }
        ],
    )
    event = _build_activity_delta_event({
        "type": "goal_selected",
        "goal_id": "goal-123",
        "description": "Investigate primary source trail",
        "priority": "high",
        "mission_id": "mission-alpha",
    })

    assert event is not None
    assert event["payload"]["current_goal"] == {
        "id": "goal-123",
        "description": "Investigate primary source trail",
        "priority": "high",
    }
    assert event["payload"]["research_session"]["session_id"] == "rs-goal-123"
    assert event["payload"]["research_session"]["topic"] == "Primary source trail"
    assert event["payload"]["running"] is True
    mission = event["payload"]["current_mission"]
    assert mission["id"] == "mission-alpha"
    assert mission["description"] == "Alpha mission"
    assert mission["active_tasks"] == 0
    assert mission["pending_tasks"] == 0
    assert mission["completed_tasks"] == 0
    assert mission["focus_stale_cycles"] == 0
    assert mission["total_tasks"] == 3
    assert mission["pending_task_labels"] == [
        "Collect filings",
        "Compare source trail",
        "Draft summary",
    ]


def test_activity_delta_refreshes_mission_queue_on_goal_transition(monkeypatch):
    monkeypatch.setattr(
        "remy.web.routes.websocket._build_activity_mission_state",
        lambda mission_id: {
            "id": mission_id,
            "description": "Alpha mission",
            "pending_tasks": 1,
            "pending_task_labels": ["Draft summary"],
            "completed_tasks": 2,
            "total_tasks": 3,
        },
    )
    event = _build_activity_delta_event({
        "type": "goal_archived",
        "mission_id": "mission-alpha",
    })

    assert event is not None
    assert event["payload"]["current_mission"] == {
        "id": "mission-alpha",
        "description": "Alpha mission",
        "pending_tasks": 1,
        "pending_task_labels": ["Draft summary"],
        "completed_tasks": 2,
        "total_tasks": 3,
    }


def test_activity_delta_refreshes_mission_queue_on_task_transition(monkeypatch):
    monkeypatch.setattr(
        "remy.web.routes.websocket._build_activity_mission_state",
        lambda mission_id: {
            "id": mission_id,
            "description": "Alpha mission",
            "pending_tasks": 0,
            "pending_task_labels": [],
            "completed_tasks": 3,
            "total_tasks": 3,
        },
    )
    event = _build_activity_delta_event({
        "type": "mission.task_completed",
        "mission_id": "mission-alpha",
        "mission_task_id": "task-3",
    })

    assert event is not None
    assert event["payload"]["current_mission"] == {
        "id": "mission-alpha",
        "description": "Alpha mission",
        "pending_tasks": 0,
        "pending_task_labels": [],
        "completed_tasks": 3,
        "total_tasks": 3,
    }


def test_system_delta_maps_llm_health_to_gateway_and_channel_health(monkeypatch):
    monkeypatch.setattr(
        "remy.core.combined_runner.get_autonomy_control_state",
        lambda: {
            "running": True,
            "maintenance_only": True,
            "active_version": "v3",
            "configured_version": "v3",
            "runtime_loaded": True,
            "session_id": "sess-1",
        },
    )
    monkeypatch.setattr(
        "remy.core.gateway.get_registry",
        lambda: type("Registry", (), {"summary": lambda self: {"health": "degraded"}})(),
    )
    event = _build_system_delta_event({
        "type": "llm_health",
        "status": "maintenance_only",
        "failures": 5,
    })

    assert event is not None
    assert event["event_name"] == "system.delta"
    assert event["payload"]["gateway"]["status"] == "degraded"
    assert event["payload"]["channels"]["registry_summary"]["health"] == "degraded"
    assert event["payload"]["channels"]["autonomy"]["maintenance_only"] is True
    assert event["payload"]["channels"]["autonomy"]["health"]["status"] == "degraded"


def test_system_delta_cycle_start_uses_shared_control_state(monkeypatch):
    monkeypatch.setattr(
        "remy.web.routes.websocket._get_runtime_transport_state",
        lambda: {"running": True, "session_id": "sess-cycle"},
    )

    event = _build_system_delta_event({
        "type": "cycle_start",
        "session_id": "stale-session-id",
    })

    assert event is not None
    assert event["payload"]["autonomy"]["running"] is True
    assert event["payload"]["autonomy"]["session_id"] == "sess-cycle"


def test_system_delta_operator_alert_carries_rich_wire_shape():
    event = _build_system_delta_event({
        "type": "operator_alert",
        "id": "alert-1",
        "level": "warning",
        "message": "Verification failed",
        "timestamp": 123.0,
        "source": "reconstruct_missing_memory",
        "scenario_id": "verify_gate_ablation",
        "action_target": "open_missing_memory_review",
        "artifact_ids": ["artifact-1", "artifact-2"],
        "failure_code": "verification_failed",
        "verification_status": "repair_required",
        "verification_reason": "Needs repair.",
        "eval_status": "completed",
        "requested": 2,
        "applied": 1,
        "skipped": 1,
    })

    assert event is not None
    upsert = event["payload"]["operator_alerts"]["upsert"]
    assert upsert["id"] == "alert-1"
    assert upsert["source"] == "reconstruct_missing_memory"
    assert upsert["scenario_id"] == "verify_gate_ablation"
    assert upsert["action_target"] == "open_missing_memory_review"
    assert upsert["artifact_ids"] == ["artifact-1", "artifact-2"]
    assert upsert["failure_code"] == "verification_failed"
    assert upsert["verification_status"] == "repair_required"
    assert upsert["verification_reason"] == "Needs repair."
    assert upsert["eval_status"] == "completed"
    assert upsert["requested"] == 2
    assert upsert["applied"] == 1
    assert upsert["skipped"] == 1


def test_activity_delta_cycle_start_uses_shared_control_state(monkeypatch):
    monkeypatch.setattr(
        "remy.web.routes.websocket._get_runtime_transport_state",
        lambda: {"running": True, "session_id": "sess-activity"},
    )
    monkeypatch.setattr(
        "remy.core.combined_runner.get_activity_runtime_snapshot",
        lambda goal_limit=3, approval_limit=10, transport_connected=False: {
            "scheduler_selection": {
                "mission_id": "mission-alpha",
                "score": 3.25,
                "reason": "runnable_task,routing_prefer=researcher:0.20",
                "details": {"routing_reason": "routing_prefer=researcher"},
            },
            "specialist_resolution": {
                "specialist_id": "researcher",
                "reason": "routing_pressure_override:task_specialist:executor->researcher",
                "quality_factor": 0.85,
            }
        },
    )

    event = _build_activity_delta_event({
        "type": "cycle_start",
        "session_id": "stale-session-id",
        "budget": {"llm_cost_today": 0.82},
    })

    assert event is not None
    assert event["payload"]["running"] is True
    assert event["payload"]["session_id"] == "sess-activity"
    assert event["payload"]["budget"] == {"llm_cost_today": 0.82}
    assert event["payload"]["scheduler_selection"]["mission_id"] == "mission-alpha"
    assert event["payload"]["specialist_resolution"]["specialist_id"] == "researcher"


@pytest.mark.asyncio
async def test_send_budget_init_uses_shared_budget_runtime_snapshot(monkeypatch):
    websocket = AsyncMock()
    monkeypatch.setattr(
        "remy.core.combined_runner.get_budget_runtime_snapshot",
        lambda goal_limit=3, approval_limit=5: {
            "alert_level": "yellow",
            "llm_cost_today": 0.82,
            "runway_days": 14,
        },
    )

    await _send_budget_init(websocket, api=None)

    sent = websocket.send_json.await_args.args[0]
    assert sent["event_name"] == "budget_init"
    assert sent["payload"]["budget"] == {
        "alert_level": "yellow",
        "llm_cost_today": 0.82,
        "runway_days": 14,
    }


@pytest.mark.asyncio
async def test_send_activity_snapshot_uses_shared_activity_runtime_snapshot(monkeypatch):
    from remy.core.event_bus import event_bus

    websocket = AsyncMock()

    monkeypatch.setattr(
        "remy.core.combined_runner.get_activity_runtime_snapshot",
        lambda goal_limit=3, approval_limit=10, transport_connected=False: {
            "running": True,
            "session_id": "sess-123",
            "version": "v3",
            "transport_connected": transport_connected,
            "current_goal": {"id": "goal-1"},
        },
    )
    monkeypatch.setattr(type(event_bus), "subscriber_count", property(lambda self: 2))

    await _send_activity_snapshot(websocket)

    sent = websocket.send_json.await_args.args[0]
    assert sent["event_name"] == "activity.snapshot"
    assert sent["payload"]["running"] is True
    assert sent["payload"]["session_id"] == "sess-123"
    assert sent["payload"]["version"] == "v3"
    assert sent["payload"]["transport_connected"] is True


@pytest.mark.asyncio
async def test_send_approval_snapshot_uses_shared_approval_runtime_snapshot(monkeypatch):
    websocket = AsyncMock()
    monkeypatch.setattr(
        "remy.core.combined_runner.get_approval_runtime_snapshot",
        lambda goal_limit=3, approval_limit=50: {
            "pending": [
                {
                    "id": "appr-1",
                    "action_id": "appr-1",
                    "description": "Review payout request",
                    "timeout_sec": 60,
                    "created_at": 100.0,
                    "expires_at": 160.0,
                    "age_sec": 12,
                }
            ]
        },
    )

    await _send_approval_snapshot(websocket)

    sent = websocket.send_json.await_args.args[0]
    assert sent["event_name"] == "approval.pending"
    assert sent["payload"] == {
        "action_id": "appr-1",
        "description": "Review payout request",
        "timeout_sec": 60,
        "created_at": 100.0,
    }


@pytest.mark.asyncio
async def test_get_pending_approvals_uses_shared_approval_runtime_snapshot(monkeypatch):
    monkeypatch.setattr(
        "remy.core.combined_runner.get_approval_runtime_snapshot",
        lambda goal_limit=3, approval_limit=100: {
            "pending": [
                {
                    "id": "appr-1",
                    "action_id": "appr-1",
                    "description": "Review payout request",
                    "timeout_sec": 60,
                    "created_at": 100.0,
                    "expires_at": 160.0,
                    "age_sec": 12,
                }
            ]
        },
    )

    payload = await get_pending_approvals()

    assert payload == {
        "pending": [
            {
                "id": "appr-1",
                "action_id": "appr-1",
                "description": "Review payout request",
                "timeout_sec": 60,
                "created_at": 100.0,
                "expires_at": 160.0,
                "age_sec": 12,
            }
        ]
    }


@pytest.mark.asyncio
async def test_send_guidance_snapshot_uses_guidance_queue_snapshot(monkeypatch):
    websocket = AsyncMock()

    monkeypatch.setattr(
        "remy.core.combined_runner.get_guidance_runtime_snapshot",
        lambda limit=10: {
            "pending": [
                {
                    "request_id": "guide-1",
                    "question": "Need operator decision",
                    "context": "Task blocked on ambiguity",
                    "timeout_sec": 120,
                    "created_at": 100.0,
                    "expires_at": 220.0,
                }
            ]
        },
    )

    await _send_guidance_snapshot(websocket)

    sent = websocket.send_json.await_args.args[0]
    assert sent["event_name"] == "guidance.pending"
    assert sent["payload"] == {
        "request_id": "guide-1",
        "question": "Need operator decision",
        "context": "Task blocked on ambiguity",
        "timeout_sec": 120,
        "created_at": 100.0,
    }


@pytest.mark.asyncio
async def test_get_pending_guidance_uses_guidance_queue_snapshot(monkeypatch):
    monkeypatch.setattr(
        "remy.core.combined_runner.get_guidance_runtime_snapshot",
        lambda limit=10: {
            "pending": [
                {
                    "request_id": "guide-1",
                    "question": "Need operator decision",
                    "context": "Task blocked on ambiguity",
                    "timeout_sec": 120,
                    "created_at": 100.0,
                    "expires_at": 220.0,
                }
            ]
        },
    )

    payload = await get_pending_guidance()

    assert payload == {
        "pending": [
            {
                "request_id": "guide-1",
                "question": "Need operator decision",
                "context": "Task blocked on ambiguity",
                "timeout_sec": 120,
                "created_at": 100.0,
                "expires_at": 220.0,
            }
        ]
    }


@pytest.mark.asyncio
async def test_answer_guidance_uses_combined_runner_resolver(monkeypatch):
    monkeypatch.setattr(
        "remy.core.combined_runner.resolve_operator_guidance",
        lambda request_id, answer: {"ok": True, "request_id": request_id, "answer": answer},
    )

    payload = await answer_guidance("guide-1", {"answer": "Use the cached report"})

    assert payload == {
        "ok": True,
        "request_id": "guide-1",
        "answer": "Use the cached report",
    }


def test_websocket_lifecycle_runtime_error_detector_matches_starlette_messages():
    assert _is_websocket_lifecycle_runtime_error(
        RuntimeError('WebSocket is not connected. Need to call "accept" first.')
    )
    assert _is_websocket_lifecycle_runtime_error(
        RuntimeError('Cannot call "receive" once a disconnect message has been received.')
    )
    assert not _is_websocket_lifecycle_runtime_error(RuntimeError("Different runtime failure"))


@pytest.mark.asyncio
async def test_listen_websocket_client_exits_cleanly_on_lifecycle_runtime_error():
    websocket = AsyncMock()
    websocket.receive_text.side_effect = RuntimeError(
        'WebSocket is not connected. Need to call "accept" first.'
    )

    await _listen_websocket_client(websocket)

    websocket.receive_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_wait_for_websocket_tasks_retrieves_completed_exceptions():
    async def fail_fast():
        raise RuntimeError("boom")

    async def idle():
        await asyncio.sleep(60)

    with pytest.raises(RuntimeError, match="boom"):
        await _wait_for_websocket_tasks(
            asyncio.create_task(fail_fast()),
            asyncio.create_task(idle()),
        )
