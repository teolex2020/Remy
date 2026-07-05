"""
Observability / Telemetry for Remy v3.

Phase 8: Full operator console — aggregates all v3 subsystems
into a unified dashboard with mission, budget, specialist,
evaluation, improvement, and research summaries.
"""

from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)


class Telemetry:
    """Central observability hub.

    Aggregates data from all v3 components into operator-facing summaries.
    """

    def __init__(self, chief=None):
        self._chief = chief
        self._learner = None
        self._playbooks = None

    def bind(self, chief):
        """Bind to a Chief Agent instance (deferred init)."""
        self._chief = chief

    def bind_improvement(self, learner=None, playbooks=None):
        """Bind self-improvement components."""
        self._learner = learner
        self._playbooks = playbooks
        runtime = getattr(self._chief, "dashboard_runtime", None)
        if runtime is not None:
            runtime.bind_improvement(learner=learner, playbooks=playbooks)

    # -------------------------------------------------------------------
    # Main dashboard
    # -------------------------------------------------------------------

    def dashboard(self) -> dict[str, Any]:
        """Full dashboard snapshot for operator console."""
        if self._chief is None:
            return {"timestamp": time.time(), "error": "Chief Agent not bound"}
        runtime = getattr(self._chief, "dashboard_runtime", None)
        if runtime is not None:
            return runtime.dashboard()
        return {
            "timestamp": time.time(),
            "missions": [],
            "active_task": None,
            "active_plan_step": None,
            "last_verdict": "",
            "last_decision": "",
            "total_missions": len(self._chief.all_missions()) if hasattr(self._chief, "all_missions") else 0,
            "active_missions": len(self._chief.active_missions()) if hasattr(self._chief, "active_missions") else 0,
            "budget": self._chief.budget.summary() if hasattr(self._chief, "budget") else {},
            "specialists": self._chief.registry.summary() if hasattr(self._chief, "registry") else {},
            "governance": {
                "rules": len(self._chief.policy.get_rules()) if hasattr(self._chief, "policy") else 0,
                "pending_approvals": len(self._chief.approval.pending()) if hasattr(self._chief, "approval") else 0,
                "approval_stats": self._chief.approval.summary() if hasattr(self._chief, "approval") else {},
            },
            "audit": self._chief.audit.summary() if hasattr(self._chief, "audit") else {},
            "audit_recent": [],
            "evaluation": self._chief.evaluator.summary() if hasattr(self._chief, "evaluator") else {},
            "execution": self._chief.recorder.stats() if hasattr(self._chief, "recorder") else {},
            "recent_outcomes": self._chief.recorder.recent_outcomes_summary(5) if hasattr(self._chief, "recorder") else "",
        }

    # -------------------------------------------------------------------
    # Detail views
    # -------------------------------------------------------------------

    def mission_detail(self, mission_id: str) -> dict[str, Any]:
        """Detailed view of a single mission."""
        if self._chief is None:
            return {}
        runtime = getattr(self._chief, "dashboard_runtime", None)
        if runtime is not None:
            return runtime.mission_detail(mission_id)
        return {}

    def budget_detail(self) -> dict[str, Any]:
        """Detailed budget breakdown."""
        if self._chief is None:
            return {}
        runtime = getattr(self._chief, "dashboard_runtime", None)
        if runtime is not None:
            return runtime.budget_detail()
        return {}

    def specialist_detail(self, specialist_id: str) -> dict[str, Any]:
        """Detail for a specific specialist."""
        if self._chief is None:
            return {}
        runtime = getattr(self._chief, "dashboard_runtime", None)
        if runtime is not None:
            return runtime.specialist_detail(specialist_id)
        return {}

    # -------------------------------------------------------------------
    # Health check
    # -------------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """Quick system health summary."""
        if self._chief is None:
            return {"status": "unbound"}
        runtime = getattr(self._chief, "dashboard_runtime", None)
        if runtime is not None:
            return runtime.health_check()
        budget = self._chief.budget.get_status().value if hasattr(self._chief, "budget") else "unknown"
        return {
            "status": "critical" if budget == "exhausted" else "healthy",
            "budget": budget,
            "pending_approvals": len(self._chief.approval.pending()) if hasattr(self._chief, "approval") else 0,
            "recent_errors": len(self._chief.audit.errors(20)) if hasattr(self._chief, "audit") else 0,
            "active_missions": len(self._chief.active_missions()) if hasattr(self._chief, "active_missions") else 0,
            "uptime_cycles": len(self._chief.recorder._records) if hasattr(self._chief, "recorder") else 0,
        }
