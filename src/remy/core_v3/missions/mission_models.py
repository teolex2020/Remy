"""
Mission, Goal, and Task models for Remy v3.

These are the core data structures that drive the autonomous mission engine.
They replace the implicit dict-based goal system from v2.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MissionMode(str, Enum):
    """How the Chief Agent should approach this mission."""
    QUICK_TACTICAL = "quick_tactical"
    DEEP_RESEARCH = "deep_research"
    CAMPAIGN = "campaign"
    CONTINUOUS_MONITORING = "continuous_monitoring"
    SELF_IMPROVEMENT = "self_improvement"


class MissionStatus(str, Enum):
    INTAKE = "intake"
    PLANNING = "planning"
    ACTIVE = "active"
    PAUSED = "paused"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"


class GoalStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    ARCHIVED = "archived"


class TaskStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    BLOCKED_EXTERNAL = "blocked_external"
    BLOCKED_APPROVAL = "blocked_approval"
    ABORTED = "aborted"
    SKIPPED = "skipped"


class TaskRepeat(str, Enum):
    ONCE = "once"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Success Criterion
# ---------------------------------------------------------------------------

@dataclass
class SuccessCriterion:
    """Programmatic check for goal/task completion.

    Maps directly to v2 success_criteria.py criterion types.
    """
    type: str                       # record_stored, file_exists, draft_created, ...
    description: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"type": self.type, "description": self.description, **self.config}

    @classmethod
    def from_dict(cls, d: dict) -> SuccessCriterion:
        cfg = {k: v for k, v in d.items() if k not in ("type", "description")}
        return cls(type=d["type"], description=d.get("description", ""), config=cfg)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """Atomic unit of work within a mission.

    A task is a concrete, executable action — the agent does not need to
    plan further, it just executes the action described.
    """
    id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    action: str = ""                # Exact instruction to execute
    done_when: str = ""             # Human description of completion
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 5              # 1 = highest
    repeat: TaskRepeat = TaskRepeat.ONCE
    depends_on: list[str] = field(default_factory=list)

    # Execution tracking
    attempts: int = 0
    last_attempt_at: float = 0.0
    completed_at: float = 0.0
    next_run_after: float = 0.0   # epoch timestamp — don't run before this (repeat tasks)
    error: str = ""

    # Links
    goal_id: str = ""              # Parent goal
    mission_id: str = ""           # Parent mission
    record_id: str = ""            # Aura record ID (if stored in memory)

    # Success criteria (programmatic)
    success_criteria: list[SuccessCriterion] = field(default_factory=list)

    # Block / waiting metadata
    blocker_reason: str = ""
    waiting_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_runnable(self) -> bool:
        import time as _time
        if self.status not in (TaskStatus.PENDING, TaskStatus.ACTIVE):
            return False
        if self.next_run_after and _time.time() < self.next_run_after:
            return False
        return True

    def is_terminal(self) -> bool:
        return self.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.ABORTED,
            TaskStatus.SKIPPED,
        )

    def is_blocked(self) -> bool:
        return self.status in (
            TaskStatus.BLOCKED,
            TaskStatus.BLOCKED_EXTERNAL,
            TaskStatus.BLOCKED_APPROVAL,
        )

    def is_waiting(self) -> bool:
        return self.status == TaskStatus.WAITING


# ---------------------------------------------------------------------------
# Goal
# ---------------------------------------------------------------------------

@dataclass
class Goal:
    """A high-level objective within a mission.

    Goals are decomposed into tasks. A goal can also be a direct
    migration of a v2 autonomy goal record.
    """
    id: str = field(default_factory=lambda: f"goal_{uuid.uuid4().hex[:12]}")
    description: str = ""
    status: GoalStatus = GoalStatus.PENDING
    priority: int = 5
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Hierarchy
    mission_id: str = ""
    parent_goal_id: str = ""       # For sub-goals
    task_ids: list[str] = field(default_factory=list)

    # Execution
    pack_template: str = ""        # Capability pack / specialist hint
    attempts: int = 0
    max_attempts: int = 5
    immortal: bool = False         # Never archived

    # Success criteria
    success_criteria: list[SuccessCriterion] = field(default_factory=list)

    # Timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: float = 0.0

    # Aura link
    record_id: str = ""            # Brain record ID

    def is_runnable(self) -> bool:
        return self.status in (GoalStatus.PENDING, GoalStatus.ACTIVE)

    def is_terminal(self) -> bool:
        return self.status in (GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.ARCHIVED)


# ---------------------------------------------------------------------------
# Budget Estimate
# ---------------------------------------------------------------------------

@dataclass
class BudgetEstimate:
    """Estimated resource cost for a mission or plan step."""
    tokens: int = 0
    cost_usd: float = 0.0
    time_sec: int = 0
    tool_calls: int = 0
    risk: RiskLevel = RiskLevel.LOW

    @property
    def is_expensive(self) -> bool:
        return self.cost_usd > 0.50 or self.tokens > 50_000


# ---------------------------------------------------------------------------
# Mission
# ---------------------------------------------------------------------------

@dataclass
class Mission:
    """Top-level mission — the unit of autonomous work.

    A mission flows through the lifecycle:
    intake → planning → active → (paused|blocked|escalated) → completed|failed
    """
    id: str = field(default_factory=lambda: f"mission_{uuid.uuid4().hex[:12]}")
    description: str = ""
    objective: str = ""            # Concrete, measurable goal
    status: MissionStatus = MissionStatus.INTAKE
    mode: MissionMode = MissionMode.QUICK_TACTICAL
    priority: int = 5
    tags: list[str] = field(default_factory=list)

    # Hierarchy
    goal_ids: list[str] = field(default_factory=list)
    plan_id: str = ""              # Active plan

    # Budget & risk
    budget: BudgetEstimate = field(default_factory=BudgetEstimate)
    risk: RiskLevel = RiskLevel.LOW
    requires_approval: bool = False

    # Lifecycle
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: float = 0.0

    # Execution stats
    cycles_run: int = 0
    total_cost_usd: float = 0.0
    total_tokens: int = 0

    # Source
    source: str = ""               # "missions.json", "user_chat", "self_improvement"
    immortal: bool = False         # Never archived (survival mission)

    # Evidence & outcomes
    outcomes: list[dict[str, Any]] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)

    # Aura link
    record_id: str = ""

    def is_active(self) -> bool:
        return self.status in (MissionStatus.ACTIVE, MissionStatus.PLANNING)

    def is_terminal(self) -> bool:
        return self.status in (MissionStatus.COMPLETED, MissionStatus.FAILED)

    def add_cost(self, tokens: int, cost_usd: float):
        self.total_tokens += tokens
        self.total_cost_usd += cost_usd
        self.cycles_run += 1
        self.updated_at = time.time()


# ---------------------------------------------------------------------------
# V2 Adapter Helpers
# ---------------------------------------------------------------------------

_V2_STATUS_MAP = {
    "active": GoalStatus.ACTIVE,
    "pending": GoalStatus.PENDING,
    "completed": GoalStatus.COMPLETED,
    "failed": GoalStatus.FAILED,
    "archived": GoalStatus.ARCHIVED,
    "blocked": GoalStatus.BLOCKED,
    "blocked_by_user": GoalStatus.BLOCKED,
    "blocked_external": GoalStatus.BLOCKED,
    "decomposed": GoalStatus.ARCHIVED,  # decomposed goals are effectively done
}


def _parse_goal_status(raw: str) -> GoalStatus:
    """Convert v2 goal status string to v3 GoalStatus enum."""
    return _V2_STATUS_MAP.get(raw, GoalStatus.PENDING)


def _parse_timestamp(value: Any) -> float:
    """Accept Unix timestamps, numeric strings, or ISO datetime strings."""
    if value in (None, "", 0, "0"):
        return time.time()
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    try:
        return float(text)
    except (TypeError, ValueError):
        pass

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return time.time()


def goal_from_v2_record(record: dict) -> Goal:
    """Convert a v2 brain goal record (dict) into a v3 Goal."""
    meta = record.get("metadata", {})
    return Goal(
        id=meta.get("mission_task_id", "") or record.get("id", ""),
        description=record.get("content", ""),
        status=_parse_goal_status(meta.get("status", "pending")),
        priority=_parse_priority(meta.get("priority", 5)),
        tags=record.get("tags", []),
        metadata=meta,
        mission_id=meta.get("mission_id", ""),
        parent_goal_id=meta.get("parent_goal_id", ""),
        pack_template=meta.get("goal_template", ""),
        attempts=int(meta.get("attempts", 0)),
        immortal=str(meta.get("immortal", "")).lower() == "true",
        success_criteria=[
            SuccessCriterion.from_dict(c)
            for c in (meta.get("success_criteria") or [])
        ],
        created_at=_parse_timestamp(meta.get("created_at", 0)),
        updated_at=_parse_timestamp(meta.get("updated_at", 0)),
        record_id=record.get("id", ""),
    )


_PRIORITY_MAP = {"critical": 1, "high": 2, "medium": 5, "low": 8}


def _parse_priority(value) -> int:
    """Convert v2 priority (str or int) to v3 int priority."""
    if isinstance(value, int):
        return value
    return _PRIORITY_MAP.get(str(value).lower(), 5)


def mission_from_json(entry: dict) -> Mission:
    """Convert a missions.json entry into a v3 Mission."""
    return Mission(
        id=entry.get("id", f"mission_{uuid.uuid4().hex[:12]}"),
        description=entry.get("description", ""),
        objective=entry.get("description", ""),
        status=MissionStatus.INTAKE,
        priority=_parse_priority(entry.get("priority", 5)),
        tags=entry.get("tags", []),
        immortal=entry.get("immortal", False),
        source="missions.json",
    )
