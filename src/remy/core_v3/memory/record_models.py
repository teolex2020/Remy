"""
Structured record type helpers for v3 memory.

Provides typed constructors for the record types defined in ROADMAP.md:
mission, goal, plan, task, finding, outcome, failure, hypothesis,
playbook, improvement-suggestion, tool-health, budget-event, approval-event.
"""

from __future__ import annotations

import time
from typing import Any

from .memory_api import MemoryClass


def _base(
    record_type: str,
    content: str,
    memory_class: MemoryClass,
    *,
    mission_id: str = "",
    task_id: str = "",
    confidence: float = 1.0,
    provenance: str = "",
    trust: float = 1.0,
    extra_tags: list[str] | None = None,
    extra_meta: dict[str, Any] | None = None,
    semantic_type: str | None = None,
) -> tuple[str, list[str], dict[str, Any], MemoryClass]:
    """Build (content, tags, metadata, memory_class) tuple for store()."""
    tags = [record_type] + (extra_tags or [])
    meta: dict[str, Any] = {
        "record_type": record_type,
        "confidence": confidence,
        "provenance": provenance,
        "trust": trust,
        "created_at": str(time.time()),
    }
    if semantic_type:
        meta["semantic_type"] = semantic_type
    if mission_id:
        meta["mission_id"] = mission_id
    if task_id:
        meta["task_id"] = task_id
    if extra_meta:
        meta.update(extra_meta)
    return content, tags, meta, memory_class


# --- Constructors (return args ready for memory.store()) ---

def mission_record(content: str, *, mission_id: str, **kw):
    return _base("mission", content, MemoryClass.TASK, mission_id=mission_id, semantic_type="decision", **kw)


def goal_record(content: str, *, mission_id: str = "", **kw):
    return _base("goal", content, MemoryClass.TASK, mission_id=mission_id, semantic_type="decision", **kw)


def plan_record(content: str, *, mission_id: str = "", **kw):
    return _base("plan", content, MemoryClass.TASK, mission_id=mission_id, semantic_type="decision", **kw)


def task_record(content: str, *, mission_id: str = "", task_id: str = "", **kw):
    return _base("task", content, MemoryClass.TASK, mission_id=mission_id, task_id=task_id, semantic_type="decision", **kw)


def finding_record(content: str, *, mission_id: str = "", confidence: float = 0.8, **kw):
    return _base("finding", content, MemoryClass.OUTCOME, mission_id=mission_id, confidence=confidence, semantic_type="fact", **kw)


def outcome_record(content: str, *, mission_id: str = "", **kw):
    return _base("outcome", content, MemoryClass.OUTCOME, mission_id=mission_id, semantic_type="fact", **kw)


def failure_record(content: str, *, mission_id: str = "", **kw):
    return _base("failure", content, MemoryClass.OUTCOME, mission_id=mission_id, semantic_type="contradiction", **kw)


def hypothesis_record(content: str, *, confidence: float = 0.5, **kw):
    return _base("hypothesis", content, MemoryClass.WORKING, confidence=confidence, semantic_type="trend", **kw)


def playbook_record(content: str, **kw):
    return _base("playbook", content, MemoryClass.STRATEGIC, semantic_type="decision", **kw)


def improvement_record(content: str, **kw):
    return _base("improvement-suggestion", content, MemoryClass.STRATEGIC, semantic_type="decision", **kw)


def tool_health_record(content: str, **kw):
    return _base("tool-health", content, MemoryClass.OUTCOME, semantic_type="fact", **kw)


def budget_event_record(content: str, *, mission_id: str = "", **kw):
    return _base("budget-event", content, MemoryClass.OUTCOME, mission_id=mission_id, semantic_type="decision", **kw)


def approval_event_record(content: str, *, mission_id: str = "", **kw):
    return _base("approval-event", content, MemoryClass.TASK, mission_id=mission_id, semantic_type="decision", **kw)
