"""Deterministic memory-use evaluator for workflow run traces.

This is an AutoMem-inspired scaffold signal, not an LLM judge. It evaluates
whether a pipeline/automation used memory deliberately: search before write,
avoid empty searches, and avoid repeated memory saves in one run.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


EMPTY_MEMORY_MARKERS = (
    "[Nothing found in memory]",
    "[Memory search error:",
)


def _norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _fingerprint(value: str) -> str:
    return hashlib.sha256(_norm_text(value).encode("utf-8")).hexdigest()[:16]


def evaluate_workflow_memory(trace: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Return memory-discipline metrics and recommendations for a run trace."""
    steps = list(trace or [])
    step_positions = {id(step): _step_position(step, position) for position, step in enumerate(steps, start=1)}
    memory_searches = [s for s in steps if s.get("type") == "memory_search"]
    memory_saves = [s for s in steps if s.get("type") == "memory_save"]
    empty_searches = [
        s for s in memory_searches
        if not str(s.get("output") or "").strip()
        or any(str(s.get("output") or "").startswith(marker) for marker in EMPTY_MEMORY_MARKERS)
    ]

    duplicate_save_candidates: list[dict[str, Any]] = []
    seen_saves: dict[str, dict[str, Any]] = {}
    for step in memory_saves:
        output = str(step.get("output") or "")
        fp = _fingerprint(output)
        if not fp:
            continue
        if fp in seen_saves:
            duplicate_save_candidates.append({
                "id": step.get("id", ""),
                "label": step.get("label", ""),
                "matches": seen_saves[fp].get("id", ""),
            })
        else:
            seen_saves[fp] = step

    first_save_index = min((step_positions.get(id(s), 0) for s in memory_saves), default=0)
    has_search_before_first_save = any(
        step_positions.get(id(s), 0) < first_save_index
        for s in memory_searches
    ) if first_save_index else False
    missed_search_before_save = bool(memory_saves and not has_search_before_first_save)

    recommendations: list[str] = []
    if empty_searches:
        recommendations.append("Memory Search returned no useful context; tighten the query or skip it for this workflow.")
    if duplicate_save_candidates:
        recommendations.append("Repeated Memory Save output detected; add a guard, dedup key, or save only the final summary.")
    if missed_search_before_save:
        recommendations.append("Memory Save ran before any Memory Search; consider retrieval-before-write discipline.")
    if not memory_searches and not memory_saves:
        recommendations.append("No memory actions used; add Memory Search before AI steps when local context should matter.")

    penalty = len(empty_searches) * 20 + len(duplicate_save_candidates) * 25
    if missed_search_before_save:
        penalty += 15
    if not memory_searches and not memory_saves:
        penalty += 10

    return {
        "memory_search_count": len(memory_searches),
        "memory_save_count": len(memory_saves),
        "empty_search_count": len(empty_searches),
        "duplicate_save_candidate_count": len(duplicate_save_candidates),
        "missed_search_before_save": missed_search_before_save,
        "score": max(0, 100 - penalty),
        "duplicate_save_candidates": duplicate_save_candidates,
        "recommendations": recommendations,
    }


def _step_position(step: dict[str, Any], fallback: int) -> int:
    try:
        return int(step.get("index") or fallback)
    except Exception:
        return fallback
