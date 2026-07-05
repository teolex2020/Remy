"""
Replan Engine for Remy v3.

Handles failure recovery: decides when to retry, skip, replan,
or escalate based on failure patterns.

Phase 6: Enhanced with blocker-aware decisions, adaptive backoff,
goal decomposition, and evaluation result integration.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from .plan_models import Plan, PlanStep, StepStatus, PlanType

log = logging.getLogger(__name__)


class ReplanAction:
    RETRY = "retry"
    SKIP = "skip"
    FALLBACK = "fallback"
    REPLAN = "replan"
    ESCALATE = "escalate"
    ABORT = "abort"
    WAIT = "wait"           # Wait and retry later (rate limits, etc.)
    DECOMPOSE = "decompose"  # Break goal into sub-goals


@dataclass
class ReplanDecision:
    action: str = ReplanAction.RETRY
    reason: str = ""
    new_plan: Plan | None = None
    modified_step: PlanStep | None = None
    wait_seconds: int = 0    # For WAIT action
    decompose_into: list[str] | None = None  # For DECOMPOSE action


# Blockers that can be retried after waiting
_TRANSIENT_BLOCKERS = {"rate_limit", "timeout", "network"}
# Blockers that need human intervention
_HUMAN_BLOCKERS = {"captcha", "email_verification", "payment_required", "kyc"}
# Blockers that need a different approach
_STRATEGY_BLOCKERS = {"auth_required", "not_found"}


class ReplanEngine:
    """Decides how to handle step failures and plan-level issues."""

    def __init__(self, max_retries_per_step: int = 2, max_plan_failures: int = 3):
        self.max_retries = max_retries_per_step
        self.max_plan_failures = max_plan_failures

    def decide(
        self,
        plan: Plan,
        failed_step: PlanStep,
        eval_result=None,
    ) -> ReplanDecision:
        """Decide what to do after a step failure.

        Priority: wait (transient) → retry → fallback → skip → replan → escalate → abort
        """
        # 0. Check blocker type from eval_result
        if eval_result:
            blocker = getattr(eval_result, "blocker_type", None)
            if blocker:
                blocker_value = blocker.value if hasattr(blocker, "value") else str(blocker)
                decision = self._handle_blocker(blocker_value, failed_step)
                if decision:
                    return decision

        # 1. Can we retry?
        if failed_step.attempts < failed_step.retry_limit:
            return ReplanDecision(
                action=ReplanAction.RETRY,
                reason=f"Retry {failed_step.attempts}/{failed_step.retry_limit}",
                modified_step=failed_step,
            )

        # 2. Is there a fallback step?
        if failed_step.fallback_step_id:
            fallback = self._find_step(plan, failed_step.fallback_step_id)
            if fallback and fallback.status == StepStatus.PENDING:
                return ReplanDecision(
                    action=ReplanAction.FALLBACK,
                    reason=f"Falling back to {fallback.id}",
                    modified_step=fallback,
                )

        # 3. Check step's failure_action preference
        fa = failed_step.failure_action
        if fa == "skip":
            return ReplanDecision(
                action=ReplanAction.SKIP,
                reason="Step skipped per failure_action policy",
            )

        if fa == "abort":
            return ReplanDecision(
                action=ReplanAction.ABORT,
                reason="Step failure_action=abort",
            )

        if fa == "escalate":
            return ReplanDecision(
                action=ReplanAction.ESCALATE,
                reason="Step requires human intervention",
            )

        # 4. Check if this is a repeated failure → suggest decomposition
        if eval_result and getattr(eval_result, "is_repeated_failure", False):
            return ReplanDecision(
                action=ReplanAction.DECOMPOSE,
                reason="Repeated failure — goal may need decomposition",
            )

        # 5. Check plan-level failure threshold assuming current failure counts
        projected_failures = len(plan.failed_step_ids) + 1
        if projected_failures >= self.max_plan_failures:
            return ReplanDecision(
                action=ReplanAction.ESCALATE,
                reason=f"Plan has {projected_failures} failed steps "
                       f"(limit: {self.max_plan_failures})",
            )

        # 7. Check if other steps are still available
        remaining = [
            s for s in plan.steps
            if s.status == StepStatus.PENDING
            and s.is_ready(plan.completed_step_ids)
        ]
        if remaining:
            return ReplanDecision(
                action=ReplanAction.SKIP,
                reason=f"Skipping failed step, {len(remaining)} steps remain",
            )

        # 8. No steps left → replan
        return ReplanDecision(
            action=ReplanAction.REPLAN,
            reason="No executable steps remaining after failures",
        )

    def _handle_blocker(
        self, blocker: str, step: PlanStep,
    ) -> ReplanDecision | None:
        """Handle specific blocker types."""
        if blocker in _TRANSIENT_BLOCKERS:
            # Exponential backoff: 30s, 60s, 120s
            wait = min(30 * (2 ** step.attempts), 300)
            return ReplanDecision(
                action=ReplanAction.WAIT,
                reason=f"Transient blocker ({blocker}), waiting {wait}s",
                wait_seconds=wait,
            )

        if blocker in _HUMAN_BLOCKERS:
            return ReplanDecision(
                action=ReplanAction.ESCALATE,
                reason=f"Blocker requires human: {blocker}",
            )

        if blocker in _STRATEGY_BLOCKERS:
            return ReplanDecision(
                action=ReplanAction.REPLAN,
                reason=f"Need different approach: {blocker}",
            )

        return None  # Unknown blocker — fall through to normal logic

    def should_decompose_goal(self, attempts: int, threshold: int = 3) -> bool:
        """Check if a goal should be decomposed into subtasks.

        Mirrors v2 logic: if top_goal.attempts >= 3, decompose.
        """
        return attempts >= threshold

    def _find_step(self, plan: Plan, step_id: str) -> PlanStep | None:
        for s in plan.steps:
            if s.id == step_id:
                return s
        return None
