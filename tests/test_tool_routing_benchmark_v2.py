from types import SimpleNamespace

from remy.core.tool_contracts import ROUTING_BENCHMARK_CASES
from remy.core.tool_routing_benchmark import (
    run_live_tool_routing_benchmark,
    select_tool_with_model,
)


def test_live_selector_binds_tools_but_never_executes_them():
    case = ROUTING_BENCHMARK_CASES[0]
    tool = SimpleNamespace(name=case.expected_tool)
    calls = []

    def fake_model(messages, **kwargs):
        calls.append((messages, kwargs))
        return SimpleNamespace(tool_calls=[{"name": case.expected_tool, "args": {}}])

    selected = select_tool_with_model(
        case,
        model="test-model",
        call_model=fake_model,
        tools=[tool],
    )

    assert selected == case.expected_tool
    assert calls[0][1]["tools"] == [tool]
    assert calls[0][1]["preferred_model"] == "test-model"
    assert calls[0][1]["allow_fallback"] is False
    assert calls[0][1]["purpose"] == "tool-routing-benchmark"


def test_live_benchmark_scores_synthetic_model_selections_without_execution():
    cases = ROUTING_BENCHMARK_CASES[:3]
    expected_by_prompt = {case.prompt: case.expected_tool for case in cases}

    def fake_model(messages, **_kwargs):
        prompt = messages[-1].content
        return SimpleNamespace(tool_calls=[{"name": expected_by_prompt[prompt], "args": {}}])

    report = run_live_tool_routing_benchmark(
        model="test-model",
        cases=cases,
        call_model=fake_model,
        tools=[SimpleNamespace(name=case.expected_tool) for case in cases],
    )

    assert report["passed"] is True
    assert report["accuracy"] == 1.0
    assert report["tools_executed"] is False
    assert report["synthetic_prompts_only"] is True


def test_selector_treats_text_or_multiple_calls_as_routing_failure():
    case = ROUTING_BENCHMARK_CASES[0]

    assert select_tool_with_model(
        case,
        call_model=lambda *_args, **_kwargs: SimpleNamespace(tool_calls=[]),
        tools=[],
    ) == ""
    assert select_tool_with_model(
        case,
        call_model=lambda *_args, **_kwargs: SimpleNamespace(
            tool_calls=[{"name": "recall"}, {"name": "search"}]
        ),
        tools=[],
    ) == ""
