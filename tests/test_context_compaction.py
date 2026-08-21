from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from remy.core.context_compaction import (
    compact_messages_for_model,
    estimate_messages_tokens,
    is_context_overflow_error,
    resolve_compaction_policy,
)


def _long_dialogue(turns: int = 20, chars: int = 1_000):
    messages = []
    for index in range(turns):
        messages.extend([
            HumanMessage(content=f"question-{index} " + "q" * chars),
            AIMessage(content=f"answer-{index} " + "a" * chars),
        ])
    return messages


def test_short_prompt_stays_intact_below_model_threshold():
    messages = [HumanMessage(content="hello"), AIMessage(content="hi")]

    result = compact_messages_for_model(
        messages,
        model="test-model",
        context_window_tokens=4_096,
    )

    assert result.compacted is False
    assert result.reason == "below_threshold"
    assert result.messages == messages
    assert result.tokens_after == result.tokens_before


def test_bound_tool_schema_reservation_reduces_available_prompt_budget():
    messages = [
        HumanMessage(content="q" * 1_000),
        AIMessage(content="a" * 1_000),
        HumanMessage(content="continue"),
    ]

    result = compact_messages_for_model(
        messages,
        model="test-model",
        context_window_tokens=4_096,
        reserved_input_tokens=1_400,
    )

    assert result.reserved_input_tokens == 1_400
    assert result.compacted is True
    assert result.threshold_tokens < 300


def test_token_budget_compacts_and_preserves_recent_suffix():
    messages = _long_dialogue()

    result = compact_messages_for_model(
        messages,
        model="test-model",
        context_window_tokens=4_096,
        max_recent_messages=12,
    )

    assert result.compacted is True
    assert result.tokens_after < result.tokens_before
    assert result.tokens_after <= result.target_tokens
    assert result.summarized_messages > 0
    assert isinstance(result.messages[0], SystemMessage)
    assert "question-0" in result.messages[0].content
    assert "answer-19" in result.messages[-1].content


def test_tool_sequence_and_artifact_reference_survive_compaction():
    messages = _long_dialogue(10)
    messages.extend([
        HumanMessage(content="inspect"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {}, "id": "call-1"}],
        ),
        ToolMessage(
            content="Full result: project://artifacts/tool-results/result.txt",
            tool_call_id="call-1",
            artifact={"path": "artifacts/tool-results/result.txt"},
        ),
    ])

    result = compact_messages_for_model(
        messages,
        model="test-model",
        context_window_tokens=4_096,
        max_recent_messages=8,
    )
    tool_index = next(
        index for index, message in enumerate(result.messages) if isinstance(message, ToolMessage)
    )

    assert isinstance(result.messages[tool_index - 1], AIMessage)
    assert result.messages[tool_index].tool_call_id == "call-1"
    assert "project://artifacts" in result.messages[tool_index].content


def test_tight_budget_never_keeps_orphan_tool_results():
    messages = [HumanMessage(content="run tools")]
    for index in range(12):
        messages.extend([
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "search",
                    "args": {"query": "x" * 400},
                    "id": f"call-{index}",
                }],
            ),
            ToolMessage(content="result " + "r" * 1_200, tool_call_id=f"call-{index}"),
        ])

    result = compact_messages_for_model(
        messages,
        model="test-model",
        context_window_tokens=4_096,
        max_recent_messages=12,
    )

    for index, message in enumerate(result.messages):
        if not isinstance(message, ToolMessage):
            continue
        assert index > 0
        previous = result.messages[index - 1]
        assert isinstance(previous, AIMessage)
        assert message.tool_call_id in {call["id"] for call in previous.tool_calls}


def test_large_unbacked_tool_result_is_pruned_head_and_tail():
    content = "HEAD" + "x" * 9_000 + "TAIL"
    message = ToolMessage(content=content, tool_call_id="call-large")

    result = compact_messages_for_model(
        [message],
        model="test-model",
        context_window_tokens=32_768,
    )

    assert result.compacted is True
    assert result.reason == "tool_result_pruning"
    assert result.pruned_tool_results == 1
    assert result.messages[0].content.startswith("HEAD")
    assert result.messages[0].content.endswith("TAIL")
    assert "characters omitted" in result.messages[0].content


def test_policy_prefers_explicit_runtime_capability():
    from remy.core import model_capabilities

    model_capabilities.clear_runtime_capability_overrides()
    try:
        model_capabilities.mark_model_capability("bounded-model", context_window=24_000)
        policy = resolve_compaction_policy("bounded-model")
    finally:
        model_capabilities.clear_runtime_capability_overrides()

    assert policy.context_window_tokens == 24_000


def test_context_overflow_detection_reads_nested_provider_errors():
    cause = RuntimeError("maximum context length exceeded")
    outer = RuntimeError("provider rejected request")
    outer.__cause__ = cause

    assert is_context_overflow_error(outer) is True
    assert is_context_overflow_error(RuntimeError("invalid API key")) is False


def test_trajectory_payload_contains_metrics_but_not_prompt_text():
    result = compact_messages_for_model(
        _long_dialogue(),
        model="test-model",
        context_window_tokens=4_096,
    )

    event = result.trajectory_entry(purpose="agent")

    assert event["type"] == "compaction"
    assert event["tokens_before"] > event["tokens_after"]
    assert "messages" not in event
    assert "question-0" not in str(event)


def test_call_llm_retries_context_overflow_with_stricter_compaction(monkeypatch):
    from remy.core import llm as llm_module

    prompt = _long_dialogue()
    response = AIMessage(content="recovered")
    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = [
        RuntimeError("context_length_exceeded: prompt is too long"),
        response,
    ]
    monkeypatch.setattr(llm_module.settings, "SUMMARY_MODEL", "overflow-model")
    monkeypatch.setattr(llm_module.settings, "FALLBACK_MODELS", [])
    monkeypatch.setattr(
        llm_module.settings,
        "MODEL_CONTEXT_WINDOWS",
        {"overflow-model": 4_096},
    )

    with patch("remy.core.llm.get_llm", return_value=fake_llm):
        result = llm_module.call_llm(prompt, purpose="agent")

    first_prompt = fake_llm.invoke.call_args_list[0].args[0]
    retry_prompt = fake_llm.invoke.call_args_list[1].args[0]
    assert result.content == "recovered"
    assert estimate_messages_tokens(retry_prompt) < estimate_messages_tokens(first_prompt)
    assert fake_llm.invoke.call_count == 2


def test_unrecoverable_context_overflow_advances_to_fallback(monkeypatch):
    from remy.core import llm as llm_module

    primary = MagicMock()
    primary.invoke.side_effect = RuntimeError("maximum context length exceeded")
    fallback = MagicMock()
    fallback.invoke.return_value = AIMessage(content="fallback recovered")
    monkeypatch.setattr(llm_module.settings, "SUMMARY_MODEL", "small-primary")
    monkeypatch.setattr(llm_module.settings, "FALLBACK_MODELS", ["larger-fallback"])

    with patch("remy.core.llm.get_llm", side_effect=[primary, fallback]) as get_llm:
        result = llm_module.call_llm("short prompt", purpose="agent")

    assert result.content == "fallback recovered"
    assert result.response_metadata["_served_by"] == "larger-fallback"
    assert get_llm.call_count == 2
