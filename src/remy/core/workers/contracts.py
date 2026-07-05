"""Structured contracts for orchestrator/worker execution."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorkerExecutionResult:
    """Structured result returned by a specialized worker wrapper."""

    worker: str
    status: str
    response_text: str
    history: list = field(default_factory=list)
    session_log: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    tool_calls: int = 0
