"""
Projection runtime for Remy v3.

Builds operator-facing read models from runtime state so dashboard-oriented
projection logic does not live inside ChiefAgent.
"""

from __future__ import annotations


class ProjectionRuntime:
    """Constructs mission summaries and other read models."""

    def __init__(self, *, mission_runtime, mission_query_runtime, plan_query_runtime, goal_query_runtime, goal_tracker, recorder):
        self.mission_runtime = mission_runtime
        self.mission_query_runtime = mission_query_runtime
        self.plan_query_runtime = plan_query_runtime
        self.goal_query_runtime = goal_query_runtime
        self.goal_tracker = goal_tracker
        self.recorder = recorder

    def mission_summary(self, mission, plan=None) -> dict:
        goals = self.goal_query_runtime.summary(mission.id)
        current_task = self.mission_runtime.select_task_for_mission(mission)
        current_step = self.plan_query_runtime.current_step(plan, current_task)
        last_record = None
        for record in reversed(getattr(self.recorder, "_records", [])):
            if getattr(record, "mission_id", "") == mission.id:
                last_record = record
                break

        return {
            "id": mission.id,
            "description": mission.description[:100],
            "status": mission.status.value,
            "mode": mission.mode.value,
            "risk": mission.risk.value,
            "cycles": mission.cycles_run,
            "cost_usd": round(mission.total_cost_usd, 4),
            "plan_progress": self.plan_query_runtime.plan_progress(plan),
            "plan_steps": self.plan_query_runtime.plan_steps(plan),
            "current_task": {
                "id": current_task.id,
                "action": current_task.action[:120],
                "status": current_task.status.value,
                "blocker_reason": current_task.blocker_reason,
                "waiting_reason": current_task.waiting_reason,
            } if current_task else None,
            "current_plan_step": {
                "id": current_step.id,
                "instruction": current_step.instruction[:120],
                "status": current_step.status.value,
            } if current_step else None,
            "last_verdict": getattr(last_record, "verdict", ""),
            "last_decision": getattr(last_record, "decision", ""),
            "goals": goals,
        }
