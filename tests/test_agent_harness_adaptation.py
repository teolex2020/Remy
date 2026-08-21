from __future__ import annotations

from typing import Annotated, TypedDict
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from remy.core.durable_graph_runtime import DurableGraphRuntime, durable_config
from remy.core.message_state import MESSAGE_CHANNEL
from remy.core.model_capabilities import (
    clear_runtime_capability_overrides,
    get_model_capabilities,
    mark_model_capability,
)
from remy.core.turn_middleware import (
    IncompleteToolCallMiddleware,
    ToolResultArtifactMiddleware,
    TurnContext,
    run_before_model_middleware,
)


def test_model_capability_profile_learns_local_tool_incompatibility():
    clear_runtime_capability_overrides()
    initial = get_model_capabilities("llamacpp:gemma-local", provider="llamacpp")
    assert initial.native_tool_calling is None
    assert initial.local is True

    learned = mark_model_capability(
        "llamacpp:gemma-local",
        native_tool_calling=False,
        parallel_tool_calls=False,
    )
    assert learned.native_tool_calling is False
    assert learned.source == "runtime-observation"
    clear_runtime_capability_overrides()


def test_call_llm_does_not_bind_tools_for_incompatible_profile():
    from remy.core.llm import call_llm

    clear_runtime_capability_overrides()
    mark_model_capability("llamacpp:plain-local", native_tool_calling=False)
    local_llm = MagicMock()
    local_llm.invoke.return_value = AIMessage(content="Local answer")

    try:
        with patch("remy.core.llm.get_llm", return_value=local_llm), patch(
            "remy.core.llm.settings"
        ) as mocked_settings:
            mocked_settings.SUMMARY_MODEL = "llamacpp:plain-local"
            mocked_settings.FALLBACK_MODELS = []
            result = call_llm("Hello", tools=[MagicMock()])
    finally:
        clear_runtime_capability_overrides()

    assert result.content == "Local answer"
    local_llm.bind_tools.assert_not_called()
    local_llm.invoke.assert_called_once_with("Hello")
    assert result.response_metadata["_model_capabilities"]["native_tool_calling"] is False


def test_incomplete_tool_call_gets_stable_cancellation_receipt():
    from remy.core.agent import _fix_gemini_turns

    pending = AIMessage(
        content="Working",
        tool_calls=[{"id": "call-17", "name": "web_search", "args": {"q": "x"}}],
    )
    first = run_before_model_middleware(
        [HumanMessage(content="Research"), pending],
        TurnContext(session_id="session-1"),
        middleware=[IncompleteToolCallMiddleware()],
    )

    receipt = first.state_updates[0]
    assert isinstance(receipt, ToolMessage)
    assert receipt.tool_call_id == "call-17"
    assert receipt.status == "error"
    assert receipt.additional_kwargs["remy_receipt"] == "cancelled"
    assert "No successful result exists" in receipt.content
    repaired = _fix_gemini_turns(first.messages)
    assert any(
        isinstance(item, ToolMessage) and item.tool_call_id == "call-17"
        for item in repaired
    )

    second = run_before_model_middleware(
        first.messages,
        TurnContext(session_id="session-1"),
        middleware=[IncompleteToolCallMiddleware()],
    )
    assert second.state_updates == []
    assert sum(isinstance(item, ToolMessage) for item in second.messages) == 1


def test_large_tool_result_is_saved_as_project_artifact(tmp_path, monkeypatch):
    from remy.core import microbrain, project_store
    from remy.core.agent import compact_history

    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(microbrain, "current_project_id", lambda: "project-test")
    monkeypatch.setattr(
        project_store,
        "project_artifact_dir",
        lambda kind, project_id, create=False: artifact_root,
    )
    monkeypatch.setattr(
        project_store,
        "project_data_root",
        lambda project_id: tmp_path,
    )
    content = "result-line\n" * 800
    original = ToolMessage(
        content=content,
        tool_call_id="call-large",
        name="web_search",
        id="tool-message-large",
    )

    result = run_before_model_middleware(
        [original],
        TurnContext(session_id="session-large", project_id="project-test"),
        middleware=[ToolResultArtifactMiddleware(threshold=1000, preview_chars=300)],
    )

    replacement = result.messages[0]
    assert isinstance(replacement, ToolMessage)
    assert replacement.id == original.id
    assert replacement.artifact["kind"] == "tool_result"
    assert "project://artifacts/tool-results/session-large/" in replacement.content
    saved = tmp_path / replacement.artifact["path"]
    assert saved.read_text(encoding="utf-8") == content
    # Artifact-backed results keep their reference during normal compaction.
    assert "Full result: project://" in compact_history([replacement])[0].content


def test_project_store_accepts_internal_artifact_directory():
    from remy.core.project_store import PROJECT_ARTIFACT_DIRS

    assert "artifacts" in PROJECT_ARTIFACT_DIRS


class _LegacyMessages(TypedDict):
    messages: Annotated[list, add_messages]


class _DeltaMessages(TypedDict):
    messages: Annotated[list, MESSAGE_CHANNEL]


def _message_graph(state_schema, label: str, checkpointer):
    def answer(state):
        return {"messages": [AIMessage(content=f"{label}-{len(state['messages'])}")]}

    builder = StateGraph(state_schema)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile(checkpointer=checkpointer)


@pytest.mark.asyncio
async def test_delta_message_channel_reads_legacy_checkpoint(tmp_path):
    runtime = DurableGraphRuntime(tmp_path / "compat.sqlite3")
    saver = await runtime.get_saver()
    config = durable_config("delta-compat", "test")

    legacy = _message_graph(_LegacyMessages, "legacy", saver)
    first = await legacy.ainvoke(
        {"messages": [HumanMessage(content="first")]},
        config,
    )
    assert [message.content for message in first["messages"]] == ["first", "legacy-1"]

    upgraded = _message_graph(_DeltaMessages, "delta", saver)
    second = await upgraded.ainvoke(
        {"messages": [HumanMessage(content="second")]},
        config,
    )
    assert [message.content for message in second["messages"]] == [
        "first",
        "legacy-1",
        "second",
        "delta-3",
    ]
    await runtime.close()
