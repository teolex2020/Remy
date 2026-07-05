"""
Dashboard runtime for Remy v3.

Builds operator-facing dashboard and detail views from read-side runtimes so
telemetry acts only as a thin adapter.
"""

from __future__ import annotations

import time
from typing import Any


class DashboardRuntime:
    """Composes dashboard-oriented read models for the operator console."""

    def __init__(
        self,
        *,
        mission_query_runtime,
        projection_runtime,
        ops_query_runtime,
        registry,
        policy,
        evaluator,
        recorder,
    ):
        self.mission_query_runtime = mission_query_runtime
        self.projection_runtime = projection_runtime
        self.ops_query_runtime = ops_query_runtime
        self.registry = registry
        self.policy = policy
        self.evaluator = evaluator
        self.recorder = recorder
        self._learner = None
        self._playbooks = None

    def bind_improvement(self, learner=None, playbooks=None):
        self._learner = learner
        self._playbooks = playbooks

    def dashboard(self) -> dict[str, Any]:
        missions = self.mission_query_runtime.active_missions()
        data: dict[str, Any] = {
            "timestamp": time.time(),
            "missions": [
                self.projection_runtime.mission_summary(
                    mission,
                    plan=self.mission_query_runtime.get_plan(mission.id),
                )
                for mission in missions
            ],
            "total_missions": len(self.mission_query_runtime.all_missions()),
            "active_missions": len(missions),
            "budget": self.ops_query_runtime.budget_summary(),
            "specialists": self.registry.summary(),
            "governance": {
                "rules": len(self.policy.get_rules()),
                **self.ops_query_runtime.governance_summary(),
            },
            "audit": self.ops_query_runtime.audit_summary(),
            "audit_recent": self.ops_query_runtime.audit_recent(10),
            "evaluation": self.evaluator.summary(),
            "execution": self.ops_query_runtime.execution_stats(),
            "factuality": self.ops_query_runtime.factuality_summary(),
            "quality_debt_by_specialist": self.ops_query_runtime.quality_debt_by_specialist(),
            "evidence_debt_queue": self.ops_query_runtime.evidence_debt_queue(10),
            "scheduler_decisions_recent": self.ops_query_runtime.scheduler_decisions_recent(10),
            "stuck_missions": self.ops_query_runtime.stuck_missions(10),
            "recent_outcomes": self.ops_query_runtime.recent_outcomes(5),
        }
        active_detail = next((mission for mission in data["missions"] if mission.get("current_task")), None)
        data["active_task"] = active_detail.get("current_task") if active_detail else None
        data["active_plan_step"] = active_detail.get("current_plan_step") if active_detail else None
        data["last_verdict"] = active_detail.get("last_verdict", "") if active_detail else ""
        data["last_decision"] = active_detail.get("last_decision", "") if active_detail else ""
        if self._learner:
            data["learning"] = self._learner.summary()
        if self._playbooks:
            data["playbooks"] = self._playbooks.summary()
        return data

    def improvement_summary(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "learning": {},
            "playbooks": {},
            "reviewable_insights": [],
            "top_playbooks": [],
        }
        if self._learner is not None:
            data["learning"] = self._learner.summary()
            if hasattr(self._learner, "reviewable_insights"):
                data["reviewable_insights"] = self._learner.reviewable_insights(5)
        if self._playbooks is not None:
            data["playbooks"] = self._playbooks.summary()
            if hasattr(self._playbooks, "top_playbooks"):
                data["top_playbooks"] = [
                    playbook.to_dict()
                    for playbook in self._playbooks.top_playbooks(5)
                ]
        return data

    def mission_detail(self, mission_id: str) -> dict[str, Any]:
        mission = self.mission_query_runtime.get_mission(mission_id)
        if mission is None:
            return {}
        summary = self.projection_runtime.mission_summary(
            mission,
            plan=self.mission_query_runtime.get_plan(mission_id),
        )
        summary["audit_trail"] = self.ops_query_runtime.audit_trail(mission_id, 20)
        summary["total_cost_usd"] = self.ops_query_runtime.mission_cost(mission_id)
        return summary

    def budget_detail(self) -> dict[str, Any]:
        return {
            **self.ops_query_runtime.budget_summary(),
            "spending_history": self.ops_query_runtime.spending_history(20),
        }

    def specialist_detail(self, specialist_id: str) -> dict[str, Any]:
        profile = self.registry.get(specialist_id)
        quality = self.ops_query_runtime.specialist_quality(specialist_id)
        return {
            "id": specialist_id,
            "profile": {
                "tools": profile.tools if profile else [],
                "domains": profile.domains if profile else [],
            } if profile else None,
            "success_rate": round(quality["success_rate"], 2),
            "quality_adjusted_success_rate": round(quality["quality_adjusted_success_rate"], 2),
            "unsupported_observed_claims": quality["unsupported_claims"],
            "recent_events": self.ops_query_runtime.specialist_recent_events(specialist_id, 10),
        }

    def health_check(self) -> dict[str, Any]:
        snapshot = self.ops_query_runtime.health_snapshot()
        return {
            "status": snapshot["status"],
            "budget": snapshot["budget"],
            "pending_approvals": snapshot["pending_approvals"],
            "recent_errors": snapshot["recent_errors"],
            "stuck_missions_count": snapshot.get("stuck_missions_count", 0),
            "active_missions": len(self.mission_query_runtime.active_missions()),
            "uptime_cycles": len(self.recorder._records),
        }
