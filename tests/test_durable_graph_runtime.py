from __future__ import annotations

from typing import TypedDict

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from remy.core.durable_graph_runtime import (
    DurableGraphRuntime,
    durable_config,
    durable_thread_id,
)


class ApprovalState(TypedDict, total=False):
    request: str
    approved: bool


def _approval_graph(checkpointer):
    def approval_node(state: ApprovalState):
        approved = interrupt(
            {
                "kind": "approval",
                "request": state.get("request", ""),
            }
        )
        return {"approved": bool(approved)}

    builder = StateGraph(ApprovalState)
    builder.add_node("approval", approval_node)
    builder.add_edge(START, "approval")
    builder.add_edge("approval", END)
    return builder.compile(checkpointer=checkpointer)


def test_durable_thread_id_is_stable_and_namespaced():
    assert durable_thread_id("abc-123", "desktop") == "remy:desktop:abc-123"
    assert durable_config("abc-123", "desktop", recursion_limit=70) == {
        "configurable": {
            "thread_id": "remy:desktop:abc-123",
            "checkpoint_ns": "",
        },
        "recursion_limit": 70,
    }


@pytest.mark.asyncio
async def test_completed_desktop_turn_checkpoint_is_cleared(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from remy.core import agent
    from remy.core import durable_graph_runtime

    runtime = MagicMock()
    runtime.delete_thread = AsyncMock()
    monkeypatch.setattr(
        durable_graph_runtime,
        "get_durable_graph_runtime",
        lambda: runtime,
    )

    await agent._clear_completed_chat_checkpoint("conversation-1", "desktop")

    runtime.delete_thread.assert_awaited_once_with("conversation-1", "desktop")


@pytest.mark.asyncio
async def test_checkpoint_status_is_mirrored_to_session_projection(tmp_path, monkeypatch):
    from remy.core import session_event_store as session_event_module
    from remy.core.session_event_store import CheckpointProjection, SessionEventStore
    from remy.core.trajectory_store import TrajectoryStore

    trajectory_path = tmp_path / "trajectory.sqlite3"
    TrajectoryStore(trajectory_path)
    event_store = SessionEventStore(trajectory_path)
    monkeypatch.setattr(session_event_module, "_STORE", event_store)
    runtime = DurableGraphRuntime(tmp_path / "checkpoints.sqlite3")

    await runtime.record_checkpoint_status(
        session_id="conversation-checkpoint",
        channel="desktop",
        project_id="project-checkpoint",
        status="waiting_for_input",
        next_nodes=["approval"],
        pending_tasks=[{"name": "approval", "has_error": False, "interrupts": 1}],
        checkpoint_created_at="2026-08-21T12:00:00+00:00",
    )

    projected = event_store.project(
        CheckpointProjection(),
        project_id="project-checkpoint",
        session_id="conversation-checkpoint",
    )
    assert projected["status"] == "waiting_for_input"
    assert projected["next"] == ["approval"]
    assert projected["pending_tasks"][0]["interrupts"] == 1


@pytest.mark.asyncio
async def test_interrupt_survives_runtime_restart_and_resumes(tmp_path):
    db_path = tmp_path / "checkpoints.sqlite3"
    config = durable_config("session-1", "desktop")

    first_runtime = DurableGraphRuntime(db_path)
    first_graph = _approval_graph(await first_runtime.get_saver())
    interrupted = await first_graph.ainvoke(
        {"request": "Run a protected action"},
        config,
    )
    assert interrupted["request"] == "Run a protected action"
    first_snapshot = await first_graph.aget_state(config)
    assert first_snapshot.next == ("approval",)
    assert first_snapshot.tasks[0].interrupts[0].value["kind"] == "approval"
    await first_runtime.close()

    second_runtime = DurableGraphRuntime(db_path)
    second_graph = _approval_graph(await second_runtime.get_saver())
    restored_snapshot = await second_graph.aget_state(config)
    assert restored_snapshot.next == ("approval",)

    completed = await second_graph.ainvoke(Command(resume=True), config)
    assert completed["approved"] is True
    final_snapshot = await second_graph.aget_state(config)
    assert final_snapshot.next == ()
    await second_runtime.close()


@pytest.mark.asyncio
async def test_remy_agent_state_continues_without_duplicate_history(
    tmp_path, monkeypatch
):
    from remy.core import agent

    monkeypatch.setattr(
        agent,
        "call_model",
        lambda state: {
            "messages": [AIMessage(content=f"reply-{len(state['messages'])}")]
        },
    )
    runtime = DurableGraphRuntime(tmp_path / "agent-checkpoints.sqlite3")
    graph = agent.build_agent_graph(
        "durable-test",
        checkpointer=await runtime.get_saver(),
    )
    config = durable_config("session-2", "durable-test")
    base_state = {
        "session_id": "session-2",
        "channel": "durable-test",
        "session_log": [],
        "enabled_tools": set(),
        "_cached_session_ctx": "",
        "_cached_scratchpad": "",
        "model_routing_enabled": False,
    }

    first = await graph.ainvoke(
        {**base_state, "messages": [HumanMessage(content="first")]},
        config,
    )
    second = await graph.ainvoke(
        {**base_state, "messages": [HumanMessage(content="second")]},
        config,
    )

    assert [message.content for message in first["messages"]] == [
        "first",
        "reply-1",
    ]
    assert [message.content for message in second["messages"]] == [
        "first",
        "reply-1",
        "second",
        "reply-3",
    ]
    await runtime.close()
    agent.invalidate_graph_cache()
