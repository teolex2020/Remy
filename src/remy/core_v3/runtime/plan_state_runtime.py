"""
Plan state runtime for Remy v3.

Owns deterministic step lifecycle mutations after evaluation and replanning so
plan/step state does not rely on model side effects.
"""

from __future__ import annotations

import time

from ..planning.plan_models import StepStatus


class PlanStateRuntime:
    """State-policy helper for post-execution plan/step transitions."""

    def add_execution_cost(self, *, plan, exec_result):
        plan.total_cost_usd += exec_result.cost_usd
        plan.total_tokens += exec_result.tokens_used

    def complete_step(self, *, plan, step, result=None):
        step.status = StepStatus.COMPLETED
        step.completed_at = time.time()
        step.result = result or {}
        plan.completed_step_ids.add(step.id)
        plan.updated_at = time.time()

    def reset_step_for_retry(self, *, plan, step):
        step.status = StepStatus.PENDING
        plan.updated_at = time.time()

    def fail_step(self, *, plan, step):
        step.status = StepStatus.FAILED
        plan.failed_step_ids.add(step.id)
        plan.updated_at = time.time()

    def skip_step(self, *, plan, step):
        step.status = StepStatus.SKIPPED
        plan.updated_at = time.time()

    def activate_fallback(self, *, plan, failed_step, fallback_step):
        self.fail_step(plan=plan, step=failed_step)
        fallback_step.status = StepStatus.PENDING
        plan.updated_at = time.time()
