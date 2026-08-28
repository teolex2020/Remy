"""Opt-in live evaluation for model tool selection.

The model is allowed to select a tool but the selected tool is never executed.
All prompts are synthetic and contain no project or user data.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from langchain_core.messages import HumanMessage, SystemMessage

from remy.core.tool_contracts import (
    ROUTING_BENCHMARK_CASES,
    TOOL_CONTRACTS,
    RoutingCase,
    run_tool_routing_benchmark,
)


ROUTING_BENCHMARK_SYSTEM_PROMPT = """You are evaluating tool routing only.
For the synthetic user request, select exactly one best tool and issue one tool
call with minimal valid arguments. Do not solve the task, execute anything, or
write a natural-language answer. Tool descriptions are authoritative."""


def _contracted_runtime_tools() -> list[Any]:
    from remy.core.langgraph_tools import get_all_tools

    names = set(TOOL_CONTRACTS)
    return [tool for tool in get_all_tools() if tool.name in names]


def select_tool_with_model(
    case: RoutingCase,
    *,
    model: str = "",
    call_model: Callable[..., Any] | None = None,
    tools: list[Any] | None = None,
) -> str:
    """Ask a model to select one tool without entering the execution loop."""
    if call_model is None:
        from remy.core.llm import call_llm

        call_model = call_llm
    available_tools = tools if tools is not None else _contracted_runtime_tools()
    response = call_model(
        [
            SystemMessage(content=ROUTING_BENCHMARK_SYSTEM_PROMPT),
            HumanMessage(content=case.prompt),
        ],
        tools=available_tools,
        purpose="tool-routing-benchmark",
        preferred_model=str(model or "").strip() or None,
        allow_fallback=not bool(str(model or "").strip()),
    )
    calls = list(getattr(response, "tool_calls", None) or [])
    if len(calls) != 1:
        return ""
    call = calls[0]
    return str(call.get("name") if isinstance(call, dict) else getattr(call, "name", "") or "")


def run_live_tool_routing_benchmark(
    *,
    model: str = "",
    cases: Iterable[RoutingCase] = ROUTING_BENCHMARK_CASES,
    call_model: Callable[..., Any] | None = None,
    tools: list[Any] | None = None,
) -> dict[str, Any]:
    """Run the synthetic suite against one connected model, without tool execution."""
    selected_cases = tuple(cases)
    runtime_tools = tools if tools is not None else _contracted_runtime_tools()
    report = run_tool_routing_benchmark(
        lambda case: select_tool_with_model(
            case,
            model=model,
            call_model=call_model,
            tools=runtime_tools,
        ),
        selected_cases,
    )
    report.update(
        {
            "mode": "live-model-selection",
            "model": str(model or "automatic"),
            "tools_executed": False,
            "synthetic_prompts_only": True,
        }
    )
    return report
