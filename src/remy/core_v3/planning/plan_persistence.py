"""
Plan Persistence and Advancement for Remy v3.

Saves/loads plans to disk and handles step advancement logic.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from .plan_models import (
    Plan, PlanStep, PlanBranch, PlanType, StepStatus, ApprovalGate,
)

log = logging.getLogger(__name__)

_PLANS_DIR = "v3_plans"


class PlanPersistence:
    """Saves and loads plans to disk."""

    def __init__(self, data_dir: str = ""):
        if not data_dir:
            try:
                from remy.config import settings
                data_dir = str(settings.DATA_DIR)
            except ImportError:
                data_dir = "data"
        self._dir = os.path.join(data_dir, _PLANS_DIR)

    def save(self, plan: Plan):
        """Save plan to disk."""
        try:
            os.makedirs(self._dir, exist_ok=True)
            path = os.path.join(self._dir, f"{plan.id}.json")
            data = self._serialize(plan)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp, path)
        except Exception as e:
            log.error("Failed to save plan %s: %s", plan.id, e)

    def load(self, plan_id: str) -> Plan | None:
        """Load plan from disk."""
        path = os.path.join(self._dir, f"{plan_id}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self._deserialize(data)
        except Exception as e:
            log.error("Failed to load plan %s: %s", plan_id, e)
            return None

    def load_by_mission(self, mission_id: str) -> Plan | None:
        """Load the active plan for a mission."""
        if not os.path.exists(self._dir):
            return None
        for fname in os.listdir(self._dir):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self._dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("mission_id") == mission_id:
                    return self._deserialize(data)
            except Exception:
                continue
        return None

    def delete(self, plan_id: str):
        path = os.path.join(self._dir, f"{plan_id}.json")
        if os.path.exists(path):
            os.remove(path)

    def list_plans(self) -> list[str]:
        """List all stored plan IDs."""
        if not os.path.exists(self._dir):
            return []
        return [
            f[:-5] for f in os.listdir(self._dir) if f.endswith(".json")
        ]

    # -------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------

    def _serialize(self, plan: Plan) -> dict:
        return {
            "id": plan.id,
            "mission_id": plan.mission_id,
            "plan_type": plan.plan_type.value,
            "steps": [self._serialize_step(s) for s in plan.steps],
            "branches": [self._serialize_branch(b) for b in plan.branches],
            "budget_ceiling_usd": plan.budget_ceiling_usd,
            "token_ceiling": plan.token_ceiling,
            "time_ceiling_sec": plan.time_ceiling_sec,
            "risk_level": plan.risk_level,
            "stop_on_failure_count": plan.stop_on_failure_count,
            "stop_on_budget_exceeded": plan.stop_on_budget_exceeded,
            "stop_conditions": plan.stop_conditions,
            "success_criteria": plan.success_criteria,
            "fallback_strategy": plan.fallback_strategy,
            "current_step_index": plan.current_step_index,
            "completed_step_ids": list(plan.completed_step_ids),
            "failed_step_ids": list(plan.failed_step_ids),
            "created_at": plan.created_at,
            "updated_at": plan.updated_at,
            "total_cost_usd": plan.total_cost_usd,
            "total_tokens": plan.total_tokens,
        }

    def _serialize_step(self, s: PlanStep) -> dict:
        return {
            "id": s.id,
            "description": s.description,
            "instruction": s.instruction,
            "specialist": s.specialist,
            "tools_needed": s.tools_needed,
            "expected_evidence": s.expected_evidence,
            "approval_gate": s.approval_gate.value,
            "depends_on": s.depends_on,
            "blocks": s.blocks,
            "retry_limit": s.retry_limit,
            "fallback_step_id": s.fallback_step_id,
            "failure_action": s.failure_action,
            "cost_estimate_usd": s.cost_estimate_usd,
            "token_estimate": s.token_estimate,
            "timeout_sec": s.timeout_sec,
            "completion_criteria": s.completion_criteria,
            "status": s.status.value,
            "attempts": s.attempts,
            "result": s.result,
            "started_at": s.started_at,
            "completed_at": s.completed_at,
        }

    def _serialize_branch(self, b: PlanBranch) -> dict:
        return {
            "id": b.id,
            "condition": b.condition,
            "step_ids": b.step_ids,
            "is_fallback": b.is_fallback,
            "priority": b.priority,
        }

    def _deserialize(self, d: dict) -> Plan:
        plan = Plan(
            id=d["id"],
            mission_id=d.get("mission_id", ""),
            plan_type=PlanType(d.get("plan_type", "linear")),
            steps=[self._deserialize_step(s) for s in d.get("steps", [])],
            branches=[self._deserialize_branch(b) for b in d.get("branches", [])],
            budget_ceiling_usd=d.get("budget_ceiling_usd", 1.0),
            token_ceiling=d.get("token_ceiling", 100_000),
            time_ceiling_sec=d.get("time_ceiling_sec", 600),
            risk_level=d.get("risk_level", "low"),
            stop_on_failure_count=d.get("stop_on_failure_count", 3),
            stop_on_budget_exceeded=d.get("stop_on_budget_exceeded", True),
            stop_conditions=d.get("stop_conditions", []),
            success_criteria=d.get("success_criteria", ""),
            fallback_strategy=d.get("fallback_strategy", ""),
            current_step_index=d.get("current_step_index", 0),
            completed_step_ids=set(d.get("completed_step_ids", [])),
            # On startup reset failed_step_ids — failures from a previous session
            # are stale (the process died mid-execution).  Carrying them over causes
            # premature escalation on the very first cycle of a new session.
            failed_step_ids=set(),
            created_at=d.get("created_at", 0.0),
            updated_at=d.get("updated_at", 0.0),
            total_cost_usd=d.get("total_cost_usd", 0.0),
            total_tokens=d.get("total_tokens", 0),
        )
        return plan

    def _deserialize_step(self, d: dict) -> PlanStep:
        return PlanStep(
            id=d["id"],
            description=d.get("description", ""),
            instruction=d.get("instruction", ""),
            specialist=d.get("specialist", ""),
            tools_needed=d.get("tools_needed", []),
            expected_evidence=d.get("expected_evidence", ""),
            approval_gate=ApprovalGate(d.get("approval_gate", "none")),
            depends_on=d.get("depends_on", []),
            blocks=d.get("blocks", []),
            retry_limit=d.get("retry_limit", 2),
            fallback_step_id=d.get("fallback_step_id", ""),
            failure_action=d.get("failure_action", "replan"),
            cost_estimate_usd=d.get("cost_estimate_usd", 0.0),
            token_estimate=d.get("token_estimate", 0),
            timeout_sec=d.get("timeout_sec", 120),
            completion_criteria=d.get("completion_criteria", ""),
            status=StepStatus(d.get("status", "pending")),
            attempts=d.get("attempts", 0),
            result=d.get("result", {}),
            started_at=d.get("started_at", 0.0),
            completed_at=d.get("completed_at", 0.0),
        )

    def _deserialize_branch(self, d: dict) -> PlanBranch:
        return PlanBranch(
            id=d.get("id", ""),
            condition=d.get("condition", ""),
            step_ids=d.get("step_ids", []),
            is_fallback=d.get("is_fallback", False),
            priority=d.get("priority", 0),
        )
