"""
Dynamic Plan Invalidation & Re-Planning (AUTON-14).

Plan health checks after each step, adaptive re-planning when steps fail,
prerequisite discovery, and confidence decay on repeated failures.
"""

import logging
from dataclasses import dataclass

from remy.core.event_bus import event_bus

logger = logging.getLogger("Autonomy.PlanInvalidation")


# ============== Plan Validity ==============


@dataclass
class PlanHealthCheck:
    """Result of checking whether a plan is still valid."""

    valid: bool = True
    needs_update: bool = False
    abandon: bool = False
    confidence: float = 1.0
    reason: str = ""
    suggested_action: str = ""  # "continue", "replan", "abandon", "add_prerequisite"


def check_plan_validity(
    plan,
    step_result: str,
    step_success: bool,
    consecutive_failures: int = 0,
) -> PlanHealthCheck:
    """Check plan health after a step execution.

    Returns PlanHealthCheck with recommended action.
    """
    check = PlanHealthCheck()

    # 1. Step failed — assess severity
    if not step_success:
        check.confidence = max(0.0, 1.0 - (consecutive_failures * 0.25))

        if consecutive_failures >= 3:
            check.valid = False
            check.abandon = True
            check.reason = f"Plan failing: {consecutive_failures} consecutive failures"
            check.suggested_action = "abandon"
            return check

        if consecutive_failures >= 2:
            check.needs_update = True
            check.reason = f"{consecutive_failures} failures — plan may need revision"
            check.suggested_action = "replan"
            return check

        check.suggested_action = "continue"
        check.reason = "Single failure — retrying"
        return check

    # 2. Check for prerequisite discovery (new info in result)
    if step_result and _detect_prerequisite_needed(step_result):
        check.needs_update = True
        check.reason = "Step result suggests prerequisite needed"
        check.suggested_action = "add_prerequisite"
        return check

    # 3. Plan nearing completion — high confidence
    if _is_linear_plan(plan):
        progress = (plan.current_step + 1) / max(len(plan.steps), 1)
        check.confidence = 0.5 + (progress * 0.5)  # 50-100% as plan progresses
    elif _is_decision_tree_plan(plan):
        check.confidence = max(0.3, 1.0 - (len(plan.history) * 0.05))

    check.suggested_action = "continue"
    return check


_PREREQUISITE_KEYWORDS = {
    "requires",
    "need to install",
    "missing dependency",
    "permission denied",
    "not found",
    "not available",
    "first need to",
    "must first",
    "prerequisite",
    "configuration required",
    "setup needed",
}


def _detect_prerequisite_needed(step_result: str) -> bool:
    """Detect if step result indicates a missing prerequisite."""
    result_lower = step_result.lower()
    return any(kw in result_lower for kw in _PREREQUISITE_KEYWORDS)


def _is_linear_plan(plan) -> bool:
    """Structural check for linear plans from autonomy or autonomy_goals."""
    return hasattr(plan, "steps") and hasattr(plan, "current_step")


def _is_decision_tree_plan(plan) -> bool:
    """Structural check for decision-tree plans from autonomy or autonomy_goals."""
    return hasattr(plan, "nodes") and hasattr(plan, "current_node")


# ============== Confidence Decay ==============


_plan_confidence: dict[str, float] = {}  # plan_id -> confidence


def get_plan_confidence(plan_id: str) -> float:
    """Get current confidence for a plan."""
    return _plan_confidence.get(plan_id, 1.0)


def update_plan_confidence(plan_id: str, step_success: bool) -> float:
    """Update plan confidence after a step.

    Success increases confidence slightly, failure decreases it.
    """
    current = _plan_confidence.get(plan_id, 1.0)

    if step_success:
        new = min(1.0, current + 0.05)
    else:
        new = max(0.0, current - 0.2)

    _plan_confidence[plan_id] = new

    if new < 0.3:
        event_bus.emit(
            "plan_confidence_low",
            {
                "plan_id": plan_id,
                "confidence": new,
            },
        )

    return new


def should_replan(plan_id: str) -> bool:
    """Whether a plan should be regenerated based on confidence."""
    return get_plan_confidence(plan_id) < 0.3


def clear_plan_confidence(plan_id: str):
    """Clear confidence tracking for a plan."""
    _plan_confidence.pop(plan_id, None)


# ============== Prerequisite Insertion ==============


def insert_prerequisite(plan, prerequisite_step: str) -> bool:
    """Insert a prerequisite step before the current step.

    Works for linear ActionPlan only. Returns True if inserted.
    """
    if not _is_linear_plan(plan):
        return False

    if plan.current_step >= len(plan.steps):
        return False

    plan.steps.insert(plan.current_step, prerequisite_step)
    logger.info(
        "Inserted prerequisite step '%s' at position %d in plan %s",
        prerequisite_step[:80],
        plan.current_step,
        plan.plan_id,
    )

    event_bus.emit(
        "plan_prerequisite_added",
        {
            "plan_id": plan.plan_id,
            "step": prerequisite_step[:200],
            "position": plan.current_step,
        },
    )
    return True


# ============== Plan Abandonment ==============


def abandon_plan(plan) -> None:
    """Mark plan as abandoned and clean up."""
    plan.status = "abandoned"
    clear_plan_confidence(getattr(plan, "plan_id", ""))

    event_bus.emit(
        "plan_abandoned",
        {
            "plan_id": getattr(plan, "plan_id", ""),
            "goal_id": getattr(plan, "goal_id", ""),
        },
    )

    logger.info("Plan %s abandoned", getattr(plan, "plan_id", "?"))


# ============== Re-Planning Prompt ==============


def build_replan_context(plan, failures: list[str]) -> str:
    """Build context for re-planning prompt.

    Includes: original goal, completed steps, failed steps, failure details.
    """
    lines = [f"GOAL: {plan.goal_description}"]

    if _is_linear_plan(plan):
        if plan.current_step > 0:
            completed = plan.steps[: plan.current_step]
            lines.append(f"COMPLETED STEPS ({len(completed)}):")
            for i, s in enumerate(completed, 1):
                lines.append(f"  {i}. {s[:100]}")

        remaining = plan.steps[plan.current_step :]
        if remaining:
            lines.append(f"REMAINING STEPS ({len(remaining)}):")
            for i, s in enumerate(remaining, plan.current_step + 1):
                lines.append(f"  {i}. {s[:100]}")

    elif _is_decision_tree_plan(plan):
        if plan.history:
            lines.append(f"HISTORY ({len(plan.history)} steps):")
            for h in plan.history[-5:]:
                status = "OK" if h.get("success") else "FAILED"
                lines.append(f"  [{status}] {h.get('description', '')[:80]}")

    if failures:
        lines.append(f"RECENT FAILURES ({len(failures)}):")
        for f in failures[-3:]:
            lines.append(f"  - {f[:150]}")

    return "\n".join(lines)


# ============== Integration Helper ==============


def process_step_result(
    plan,
    step_result: str,
    step_success: bool,
    consecutive_failures: int = 0,
) -> PlanHealthCheck:
    """Process a step result and update plan health.

    Combines health check + confidence update. Returns action recommendation.
    """
    plan_id = getattr(plan, "plan_id", "")

    # Update confidence
    update_plan_confidence(plan_id, step_success)

    # Health check
    health = check_plan_validity(plan, step_result, step_success, consecutive_failures)

    # Override with confidence-based replan
    if should_replan(plan_id) and not health.abandon:
        health.needs_update = True
        health.suggested_action = "replan"
        health.reason = f"Low confidence ({get_plan_confidence(plan_id):.1f})"

    return health
