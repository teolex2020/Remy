"""Deterministic topology selection for collaborative experiments."""

from __future__ import annotations

from typing import Any

PARALLEL_SIGNALS = (
    "research", "investigate", "audit", "compare", "survey", "scan", "review",
    "alternatives", "hypotheses", "sources", "literature", "benchmark",
    "дослід", "аудит", "порівн", "огляд", "гіпотез", "джерел", "літератур",
    "варіант", "перспектив",
)
SEQUENTIAL_SIGNALS = (
    "step by step", "step-by-step", "then", "after", "depends on", "workflow",
    "pipeline", "implement", "refactor", "migrate", "deploy", "debug", "fix",
    "navigate", "edit code", "послідов", "після", "залежить", "пайплайн",
    "реаліз", "рефактор", "міграц", "розгор", "виправ", "налагод",
)


def _signal_score(text: str, signals: tuple[str, ...]) -> int:
    return sum(1 for signal in signals if signal in text)


def select_experiment_topology(record: dict[str, Any]) -> dict[str, Any]:
    """Choose a safe execution shape without spending an LLM call."""
    participants = list(record.get("participants") or [])
    plan = record.get("experiment_plan") or {}
    text = " ".join(
        str(value or "").lower()
        for value in (
            record.get("title"), record.get("problem"), record.get("success_criteria"),
            record.get("follow_up_question"),
            plan.get("global_context"), plan.get("global_prompt"),
        )
    )
    parallel_score = _signal_score(text, PARALLEL_SIGNALS)
    sequential_score = _signal_score(text, SEQUENTIAL_SIGNALS)
    web_searches = len(plan.get("web_searches") or [])
    tool_density = web_searches
    if record.get("domain") in {"engineering", "software", "coding", "code"}:
        sequential_score += 1

    if len(participants) <= 1:
        mode = "single"
        reasons = ["Only one model role is connected, so coordination would add no value."]
    elif sequential_score > parallel_score:
        mode = "centralized_sequential"
        reasons = [
            "The task contains dependent or implementation-oriented steps.",
            "A central coordinator keeps one committed board and controls the merge.",
        ]
    elif parallel_score > 0:
        mode = "centralized_parallel"
        reasons = [
            "The task can be explored from independent research or audit perspectives.",
            "The first round is isolated to prevent early answers from anchoring the team.",
            "Only the central coordinator commits and synthesizes model outputs.",
        ]
    else:
        mode = "centralized_sequential"
        reasons = [
            "No strong independent-decomposition signal was detected.",
            "The safer default is a centrally managed sequence with one committed board.",
        ]

    if tool_density:
        reasons.append(
            f"{tool_density} shared web-search source{'s' if tool_density != 1 else ''} "
            "will be collected once by the coordinator."
        )
    return {
        "version": 1,
        "mode": mode,
        "coordinator": True,
        "blind_first_round": mode == "centralized_parallel",
        "parallel_score": parallel_score,
        "sequential_score": sequential_score,
        "tool_density": tool_density,
        "participant_count": len(participants),
        "max_parallel_workers": (
            min(3, max(1, len(participants)))
            if mode == "centralized_parallel"
            else 1
        ),
        "reasons": reasons,
    }
