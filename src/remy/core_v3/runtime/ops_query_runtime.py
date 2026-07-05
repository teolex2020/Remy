"""
Ops query runtime for Remy v3.

Read-side facade over budget, audit, approval, and recorder subsystems so
telemetry/dashboard code does not query each service directly.
"""

from __future__ import annotations


class OpsQueryRuntime:
    """Centralized operational read model access."""

    def __init__(self, *, budget, audit, approval, recorder, evaluator=None, evidence_debt_runtime=None):
        self._budget = budget
        self._audit = audit
        self._approval = approval
        self._recorder = recorder
        self._evaluator = evaluator
        self._evidence_debt_runtime = evidence_debt_runtime
        self._loop_runtime = None
        self._scheduler_runtime = None
        self._mission_query_runtime = None
        self._projection_runtime = None

    def bind_evidence_debt_runtime(self, evidence_debt_runtime):
        self._evidence_debt_runtime = evidence_debt_runtime

    def bind_autonomy(
        self,
        *,
        loop_runtime=None,
        scheduler_runtime=None,
        mission_query_runtime=None,
        projection_runtime=None,
    ):
        self._loop_runtime = loop_runtime
        self._scheduler_runtime = scheduler_runtime
        self._mission_query_runtime = mission_query_runtime
        self._projection_runtime = projection_runtime

    def budget_summary(self):
        return self._budget.summary()

    def budget_status(self):
        return self._budget.get_status().value

    def spending_history(self, limit: int = 20):
        return [
            {
                "cost": event.cost_usd,
                "mission": event.mission_id,
                "specialist": event.specialist,
                "model": event.model,
                "time": event.timestamp,
            }
            for event in self._budget.state.history[-limit:]
        ]

    def approval_summary(self):
        return self._approval.summary()

    def pending_approvals(self) -> int:
        return len(self._approval.pending())

    def pending_approval_items(self, limit: int = 5) -> list[dict]:
        items = []
        for req in self._approval.pending()[:limit]:
            items.append({
                "id": req.id,
                "action_id": req.id,
                "action": req.action,
                "description": req.description,
                "mission_id": req.mission_id,
                "specialist": req.specialist,
                "risk_category": req.risk_category,
                "estimated_cost_usd": req.estimated_cost_usd,
                "created_at": req.created_at,
                "expires_at": req.expires_at,
                "context": dict(req.context or {}),
                "source": "v3_governance",
            })
        return items

    def evidence_debt_queue(self, limit: int = 10) -> list[dict]:
        if self._evidence_debt_runtime is None:
            return []
        if not hasattr(self._evidence_debt_runtime, "open_debts"):
            return []
        return self._evidence_debt_runtime.open_debts(limit)

    def governance_summary(self):
        return {
            "pending_approvals": self.pending_approvals(),
            "approval_stats": self.approval_summary(),
            "approval_items": self.pending_approval_items(),
        }

    def recent_approvals(self, limit: int = 10):
        return [
            {
                **item,
                "source": item.get("source") or "v3_governance",
            }
            for item in self._approval.recent_decisions(limit)
        ]

    def audit_summary(self):
        summary = self._audit.summary()
        if "events_24h" not in summary and "event_counts_24h" in summary:
            summary = dict(summary)
            summary["events_24h"] = summary["event_counts_24h"]
        return summary

    def audit_recent(self, limit: int = 10):
        return [
            {
                "event": event.event_type,
                "action": event.action[:80],
                "actor": event.actor,
                "mission": event.mission_id,
                "cost": event.cost_usd,
                "time": event.timestamp,
            }
            for event in self._audit.recent(limit)
        ]

    def audit_trail(self, mission_id: str, limit: int = 20):
        return [
            {
                "event": event.event_type,
                "action": event.action,
                "time": event.timestamp,
                "cost": event.cost_usd,
            }
            for event in self._audit.recent_by_mission(mission_id, limit=limit)
        ]

    def specialist_recent_events(self, specialist_id: str, limit: int = 10):
        return [
            {
                "event": event.event_type,
                "action": event.action[:80],
                "time": event.timestamp,
            }
            for event in self._audit.recent_by_actor(specialist_id, limit=limit)
        ]

    def mission_cost(self, mission_id: str) -> float:
        return round(self._audit.mission_cost(mission_id), 4)

    def recent_errors(self, hours: float = 20):
        return len(self._audit.errors(hours))

    def execution_stats(self):
        return self._recorder.stats()

    def factuality_summary(self):
        stats = self.execution_stats()
        specialist_scores = {}
        if self._evaluator is not None:
            specialist_scores = self._evaluator.summary().get("specialist_scores", {})
        per_specialist = [
            {
                "id": specialist_id,
                "unsupported_claims": details.get("unsupported_claims", 0),
                "success_rate": details.get("success_rate", 0),
                "quality_adjusted_success_rate": details.get(
                    "quality_adjusted_success_rate",
                    details.get("success_rate", 0),
                ),
            }
            for specialist_id, details in specialist_scores.items()
        ]
        per_specialist.sort(key=lambda item: (-item["unsupported_claims"], item["quality_adjusted_success_rate"]))
        return {
            "unsupported_observed_claims_total": stats.get("unsupported_observed_claims_total", 0),
            "per_specialist": per_specialist,
            "top_offenders": per_specialist[:3],
            "best_specialists": sorted(
                per_specialist,
                key=lambda item: item["quality_adjusted_success_rate"],
                reverse=True,
            )[:3],
            "worst_specialists": sorted(
                per_specialist,
                key=lambda item: item["quality_adjusted_success_rate"],
            )[:3],
        }

    def quality_debt_by_specialist(self) -> list[dict]:
        if self._evaluator is None:
            return []
        specialist_scores = self._evaluator.summary().get("specialist_scores", {})
        debt = []
        for specialist_id, details in specialist_scores.items():
            raw = float(details.get("success_rate", 0.5))
            adjusted = float(details.get("quality_adjusted_success_rate", raw))
            debt.append({
                "id": specialist_id,
                "quality_debt": round(max(0.0, raw - adjusted), 3),
                "unsupported_claims": int(details.get("unsupported_claims", 0) or 0),
                "success_rate": raw,
                "quality_adjusted_success_rate": adjusted,
            })
        debt.sort(key=lambda item: (-item["quality_debt"], -item["unsupported_claims"], item["id"]))
        return debt

    def routing_pressure_summary(self) -> dict:
        """Summarize routing preference vs degradation pressure by specialist."""
        if self._evaluator is None:
            return {
                "preferred": [],
                "degraded": [],
                "top_candidate": None,
                "highest_pressure": None,
            }

        specialist_scores = self._evaluator.summary().get("specialist_scores", {})
        items = []
        for specialist_id, details in specialist_scores.items():
            raw = float(details.get("success_rate", 0.5))
            adjusted = float(details.get("quality_adjusted_success_rate", raw))
            unsupported = int(details.get("unsupported_claims", 0) or 0)
            debt = round(max(0.0, raw - adjusted), 3)
            items.append(
                {
                    "id": specialist_id,
                    "success_rate": raw,
                    "quality_adjusted_success_rate": adjusted,
                    "quality_debt": debt,
                    "unsupported_claims": unsupported,
                    "degraded": debt >= 0.15 or unsupported >= 2,
                }
            )

        preferred = sorted(
            items,
            key=lambda item: (-item["quality_adjusted_success_rate"], item["quality_debt"], item["id"]),
        )[:3]
        degraded = [
            item for item in sorted(
                items,
                key=lambda item: (-item["quality_debt"], -item["unsupported_claims"], item["id"]),
            )
            if item["degraded"]
        ][:3]

        return {
            "preferred": preferred,
            "degraded": degraded,
            "top_candidate": preferred[0] if preferred else None,
            "highest_pressure": degraded[0] if degraded else None,
        }

    def specialist_quality(self, specialist_id: str):
        if self._evaluator is None:
            return {
                "success_rate": 0.5,
                "quality_adjusted_success_rate": 0.5,
                "unsupported_claims": 0,
                "factuality_penalty": 0.0,
            }
        details = self._evaluator.summary().get("specialist_scores", {}).get(specialist_id, {})
        return {
            "success_rate": details.get("success_rate", 0.5),
            "quality_adjusted_success_rate": details.get(
                "quality_adjusted_success_rate",
                details.get("success_rate", 0.5),
            ),
            "unsupported_claims": details.get("unsupported_claims", 0),
            "factuality_penalty": details.get("factuality_penalty", 0.0),
        }

    def recent_outcomes(self, limit: int = 5):
        return self._recorder.recent_outcomes_summary(limit)

    def scheduler_decisions_recent(self, limit: int = 10) -> list[dict]:
        if self._scheduler_runtime is None:
            return []
        return self._scheduler_runtime.recent_decisions(limit)

    def stuck_missions(self, limit: int = 10) -> list[dict]:
        if self._loop_runtime is None or self._mission_query_runtime is None or self._projection_runtime is None:
            return []
        items = []
        for stuck in self._loop_runtime.stuck_missions(limit):
            mission = self._mission_query_runtime.get_mission(stuck["mission_id"])
            if mission is None:
                continue
            summary = self._projection_runtime.mission_summary(
                mission,
                plan=self._mission_query_runtime.get_plan(mission.id),
            )
            items.append({
                **stuck,
                "mission": {
                    "id": mission.id,
                    "description": mission.description,
                    "status": getattr(mission.status, "value", str(mission.status)),
                    "current_task": summary.get("current_task"),
                    "current_plan_step": summary.get("current_plan_step"),
                },
            })
        return items

    def health_snapshot(self):
        budget_status = self.budget_status()
        pending = self.pending_approvals()
        errors = self.recent_errors(20)
        stuck_count = len(self.stuck_missions(20))
        if budget_status == "exhausted":
            overall = "critical"
        elif budget_status == "critical" or errors > 5:
            overall = "degraded"
        elif stuck_count > 0:
            overall = "attention"
        elif pending > 10:
            overall = "attention"
        else:
            overall = "healthy"
        return {
            "status": overall,
            "budget": budget_status,
            "pending_approvals": pending,
            "recent_errors": errors,
            "stuck_missions_count": stuck_count,
        }
