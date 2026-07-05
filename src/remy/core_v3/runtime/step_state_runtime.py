"""
Step state runtime for Remy v3.

Owns deterministic plan-step state transitions for execution-gate paths so
step mutation is not duplicated across runtime layers.
"""

from __future__ import annotations

import time

from ..planning.plan_models import StepStatus


class StepStateRuntime:
    """State-policy helper for pre-execution step transitions."""

    def mark_running(self, *, step):
        step.status = StepStatus.RUNNING
        step.started_at = time.time()
        step.attempts += 1

    def mark_skipped(self, *, step):
        step.status = StepStatus.SKIPPED

    def mark_blocked(self, *, step):
        step.status = StepStatus.BLOCKED
