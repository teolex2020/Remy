"""
Evaluation Runtime for Remy v3.

Builds evaluation inputs and delegates to the EvaluationEngine so ChiefAgent
does not assemble success criteria inline.
"""

from __future__ import annotations

import re


class EvaluationRuntime:
    """Normalize evaluation inputs before calling the evaluator."""

    _URL_RE = re.compile(r"https?://[^\s)>\"]+")

    def __init__(self, evaluator, approval=None, budget=None):
        self.evaluator = evaluator
        self.approval = approval
        self.budget = budget

    @staticmethod
    def criteria_dicts(*, goal=None, task=None):
        criteria_source = (
            task.success_criteria
            if task and task.success_criteria
            else (goal.success_criteria if goal and goal.success_criteria else [])
        )
        return [criterion.to_dict() for criterion in criteria_source] if criteria_source else None

    def _source_link_completeness(self, exec_result) -> float:
        response = getattr(exec_result, "response", "") or ""
        evidence = getattr(exec_result, "evidence", {}) or {}
        evidence_text = " ".join(str(value) for value in evidence.values() if isinstance(value, (str, list, dict)))
        links = self._URL_RE.findall(f"{response} {evidence_text}")
        if getattr(exec_result, "had_external_evidence", False):
            return 1.0 if links else 0.0
        return 1.0

    def _budget_pressure_snapshot(self) -> dict | None:
        if self.budget is None:
            return None
        summary = self.budget.summary()
        return {
            "status": self.budget.get_status().value,
            "daily_remaining_usd": summary.get("daily_remaining_usd", 0.0),
            "recommended_model": summary.get("recommended_model", ""),
        }

    def evaluate(
        self,
        *,
        exec_result,
        goal=None,
        task=None,
        specialist_id: str = "",
    ):
        kwargs = {
            "success_criteria": self.criteria_dicts(goal=goal, task=task),
            "session_log": exec_result.session_log,
            "goal_id": goal.id if goal else "",
            "specialist": specialist_id,
            "unsupported_observed_claims": getattr(exec_result, "unsupported_observed_claims", 0),
            "blocker_history_summary": {
                "recent_failures": self.evaluator.failure_count_for_goal(goal.id if goal else ""),
                "task_status": getattr(task, "status", None).value if getattr(task, "status", None) is not None else "",
                "blocker_reason": getattr(task, "blocker_reason", "") if task else "",
            },
            "approval_state": {
                "pending_approvals": len(self.approval.pending()) if self.approval is not None else 0,
                "task_requires_approval": bool(getattr(task, "status", None) and getattr(task.status, "value", "") == "blocked_approval"),
            },
            "source_link_completeness": self._source_link_completeness(exec_result),
            "budget_pressure_snapshot": self._budget_pressure_snapshot(),
        }
        try:
            return self.evaluator.evaluate(exec_result, **kwargs)
        except TypeError:
            for key in (
                "unsupported_observed_claims",
                "blocker_history_summary",
                "approval_state",
                "source_link_completeness",
                "budget_pressure_snapshot",
            ):
                kwargs.pop(key, None)
            return self.evaluator.evaluate(exec_result, **kwargs)
