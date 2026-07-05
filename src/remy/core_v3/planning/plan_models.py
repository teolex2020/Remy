"""
Plan models for Remy v3.

Plans are explicit, typed, persistent structures that drive execution.
They replace the implicit prompt-loop planning from v2.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PlanType(str, Enum):
    LINEAR = "linear"
    BRANCHING = "branching"
    CONTINGENT = "contingent"
    CAMPAIGN = "campaign"
    CONTINUOUS = "continuous"


class StepStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class ApprovalGate(str, Enum):
    """Points where human approval is required."""
    NONE = "none"
    BEFORE_EXECUTION = "before_execution"
    BEFORE_PUBLISH = "before_publish"
    BEFORE_FINANCIAL = "before_financial"
    ALWAYS = "always"


# ---------------------------------------------------------------------------
# Plan Step
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    """Single step within a plan.

    Each step is concrete enough to be delegated to a specialist agent.
    """
    id: str = field(default_factory=lambda: f"step_{uuid.uuid4().hex[:8]}")
    description: str = ""
    instruction: str = ""          # Exact instruction for the specialist

    # Execution config
    specialist: str = ""           # researcher, executor, analyst, ...
    tools_needed: list[str] = field(default_factory=list)
    expected_evidence: str = ""    # What output we expect
    approval_gate: ApprovalGate = ApprovalGate.NONE

    # Dependencies
    depends_on: list[str] = field(default_factory=list)  # step IDs
    blocks: list[str] = field(default_factory=list)       # step IDs this blocks

    # Failure handling
    retry_limit: int = 2
    fallback_step_id: str = ""     # Alternative step if this fails
    failure_action: str = "replan" # replan | skip | abort | escalate

    # Budget
    cost_estimate_usd: float = 0.0
    token_estimate: int = 0
    timeout_sec: int = 120

    # Completion
    completion_criteria: str = ""  # How to verify this step is done
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    result: dict[str, Any] = field(default_factory=dict)

    # Timestamps
    started_at: float = 0.0
    completed_at: float = 0.0

    def is_ready(self, completed_steps: set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_steps for dep in self.depends_on)

    def is_terminal(self) -> bool:
        return self.status in (StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED)


# ---------------------------------------------------------------------------
# Plan Branch
# ---------------------------------------------------------------------------

@dataclass
class PlanBranch:
    """Conditional branch within a plan.

    Used for branching/contingent plan types.
    """
    id: str = field(default_factory=lambda: f"branch_{uuid.uuid4().hex[:8]}")
    condition: str = ""            # When to take this branch
    step_ids: list[str] = field(default_factory=list)
    is_fallback: bool = False      # Default path if no condition matches
    priority: int = 0              # Higher = evaluated first


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

@dataclass
class Plan:
    """Execution plan for a mission.

    A plan is a structured graph of steps with dependencies,
    branches, and budget constraints.
    """
    id: str = field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:12]}")
    mission_id: str = ""
    plan_type: PlanType = PlanType.LINEAR

    # Steps
    steps: list[PlanStep] = field(default_factory=list)
    branches: list[PlanBranch] = field(default_factory=list)

    # Constraints
    budget_ceiling_usd: float = 1.0
    token_ceiling: int = 100_000
    time_ceiling_sec: int = 600
    risk_level: str = "low"

    # Stop conditions
    stop_on_failure_count: int = 5
    stop_on_budget_exceeded: bool = True
    stop_conditions: list[str] = field(default_factory=list)

    # Success
    success_criteria: str = ""     # Overall plan success description
    fallback_strategy: str = ""    # What to do if plan fails entirely

    # State
    current_step_index: int = 0    # For linear plans
    completed_step_ids: set[str] = field(default_factory=set)
    failed_step_ids: set[str] = field(default_factory=set)

    # Timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Costs
    total_cost_usd: float = 0.0
    total_tokens: int = 0

    # -------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------

    def next_step(self) -> PlanStep | None:
        """Get next runnable step (linear mode)."""
        if self.plan_type == PlanType.LINEAR:
            for step in self.steps:
                if step.status == StepStatus.PENDING:
                    return step
            return None
        # For non-linear, find any ready step
        for step in self.steps:
            if step.status == StepStatus.PENDING and step.is_ready(self.completed_step_ids):
                return step
        return None

    def ready_steps(self) -> list[PlanStep]:
        """Get all steps whose dependencies are satisfied (for parallel execution)."""
        return [
            s for s in self.steps
            if s.status == StepStatus.PENDING
            and s.is_ready(self.completed_step_ids)
        ]

    def advance(self, step_id: str, success: bool, result: dict | None = None):
        """Mark a step as completed or failed and update plan state."""
        for step in self.steps:
            if step.id == step_id:
                if success:
                    step.status = StepStatus.COMPLETED
                    step.completed_at = time.time()
                    step.result = result or {}
                    self.completed_step_ids.add(step_id)
                else:
                    step.attempts += 1
                    if step.attempts >= step.retry_limit:
                        step.status = StepStatus.FAILED
                        self.failed_step_ids.add(step_id)
                    else:
                        step.status = StepStatus.PENDING  # Retry
                self.updated_at = time.time()
                return

    @property
    def progress(self) -> float:
        """0.0 to 1.0 completion ratio."""
        if not self.steps:
            return 0.0
        done = sum(1 for s in self.steps if s.is_terminal())
        return done / len(self.steps)

    @property
    def is_complete(self) -> bool:
        return all(s.is_terminal() for s in self.steps)

    @property
    def is_failed(self) -> bool:
        return len(self.failed_step_ids) >= self.stop_on_failure_count

    @property
    def is_over_budget(self) -> bool:
        return self.total_cost_usd >= self.budget_ceiling_usd

    def should_stop(self) -> tuple[bool, str]:
        """Check if the plan should stop execution."""
        if self.is_complete:
            return True, "all_steps_complete"
        if self.is_failed:
            return True, "failure_limit_reached"
        if self.is_over_budget and self.stop_on_budget_exceeded:
            return True, "budget_exceeded"
        return False, ""


# ---------------------------------------------------------------------------
# V2 Adapter Helpers
# ---------------------------------------------------------------------------

def plan_from_v2_action_plan(ap) -> Plan:
    """Convert a v2 ActionPlan into a v3 Plan."""
    steps = []
    for i, instruction in enumerate(getattr(ap, "steps", [])):
        steps.append(PlanStep(
            id=f"step_{i:03d}",
            description=instruction if isinstance(instruction, str) else str(instruction),
            instruction=instruction if isinstance(instruction, str) else str(instruction),
        ))
    return Plan(
        mission_id=getattr(ap, "goal_id", ""),
        plan_type=PlanType.LINEAR,
        steps=steps,
        current_step_index=getattr(ap, "current_step", 0),
    )


def plan_from_v2_decision_tree(dt) -> Plan:
    """Convert a v2 DecisionTreePlan into a v3 Plan (branching)."""
    steps = []
    branches = []

    def _walk(node, depth=0):
        step = PlanStep(
            id=f"step_dt_{depth}_{uuid.uuid4().hex[:6]}",
            description=getattr(node, "instruction", ""),
            instruction=getattr(node, "instruction", ""),
        )
        steps.append(step)
        for child in getattr(node, "children", []):
            child_step = _walk(child, depth + 1)
            branches.append(PlanBranch(
                condition=getattr(child, "condition", ""),
                step_ids=[child_step.id],
            ))

        return step

    root = getattr(dt, "root", None)
    if root:
        _walk(root)

    return Plan(
        mission_id=getattr(dt, "goal_id", ""),
        plan_type=PlanType.BRANCHING,
        steps=steps,
        branches=branches,
    )
