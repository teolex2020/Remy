"""Evidence-debt task creation for consequence-memory abstain paths."""

from __future__ import annotations

from ..missions.mission_models import Task


class EvidenceDebtRuntime:
    """Turns verify-first policy hints into concrete follow-up tasks."""

    _WAITING_REASON = "Opened from consequence memory EvidenceDebt"

    def __init__(self, tasks: dict[str, Task]):
        self._tasks = tasks

    def open_debt(
        self,
        *,
        mission,
        goal,
        current_task,
        policy_hint: dict,
    ) -> Task | None:
        hint = str((policy_hint or {}).get("hint") or "")
        if hint not in {"verify_first", "requires_evidence"}:
            return None

        action = str((policy_hint or {}).get("action") or "").strip()
        situation = str((policy_hint or {}).get("situation") or "").strip()
        reason = str((policy_hint or {}).get("reason") or "").strip()
        if not action:
            return None

        debt_action = self._debt_action(action, reason)
        for task in self._tasks.values():
            if (
                task.mission_id == getattr(mission, "id", "")
                and task.action == debt_action
                and not task.is_terminal()
            ):
                return task

        debt = Task(
            action=debt_action,
            done_when="Evidence was collected or the uncertainty was explicitly resolved.",
            mission_id=getattr(mission, "id", ""),
            goal_id=getattr(goal, "id", "") if goal is not None else "",
            priority=1,
            metadata={
                "type": "evidence_debt",
                "source_situation": situation,
                "source_action": action,
                "policy_hint": hint,
                "policy_reason": reason,
                "opened_from_task_id": getattr(current_task, "id", "") if current_task is not None else "",
            },
        )
        if current_task is not None:
            debt.depends_on = list(getattr(current_task, "depends_on", []) or [])
        debt.waiting_reason = self._WAITING_REASON
        self._tasks[debt.id] = debt
        return debt

    def _debt_action(self, action: str, reason: str) -> str:
        suffix = f" Reason: {reason}" if reason else ""
        return f"Verify evidence before relying on action: {action}.{suffix}"[:500]

    def is_debt_task(self, task: Task) -> bool:
        meta = dict(getattr(task, "metadata", {}) or {})
        return (
            meta.get("type") == "evidence_debt"
            or str(getattr(task, "waiting_reason", "") or "") == self._WAITING_REASON
        )

    def open_debts(self, limit: int = 10) -> list[dict]:
        """Return active evidence-debt tasks for operator/runtime status."""
        debts = [
            task for task in self._tasks.values()
            if self.is_debt_task(task) and not task.is_terminal()
        ]
        debts.sort(key=lambda task: (int(getattr(task, "priority", 5) or 5), getattr(task, "id", "")))
        return [
            {
                "id": task.id,
                "action": task.action,
                "mission_id": task.mission_id,
                "goal_id": task.goal_id,
                "status": getattr(task.status, "value", str(task.status)),
                "priority": task.priority,
                "waiting_reason": task.waiting_reason,
                "depends_on": list(task.depends_on or []),
                "source_situation": str((getattr(task, "metadata", {}) or {}).get("source_situation") or ""),
                "source_action": str((getattr(task, "metadata", {}) or {}).get("source_action") or ""),
                "policy_hint": str((getattr(task, "metadata", {}) or {}).get("policy_hint") or ""),
                "policy_reason": str((getattr(task, "metadata", {}) or {}).get("policy_reason") or ""),
            }
            for task in debts[: max(0, int(limit or 0))]
        ]

    def resolve_after_evaluation(
        self,
        *,
        task: Task | None,
        mission,
        goal,
        step,
        specialist,
        exec_result,
        eval_result,
    ) -> str:
        """Persist the consequence learned by executing an evidence-debt task."""
        if task is None or not self.is_debt_task(task):
            return ""

        meta = dict(getattr(task, "metadata", {}) or {})
        situation = str(meta.get("source_situation") or "").strip()
        action = str(meta.get("source_action") or "").strip()
        if not situation or not action:
            return ""

        consequence, trust = self._consequence_from_eval(eval_result)
        try:
            from ..memory.memory_api import get_memory

            memory = get_memory()
            return memory.capture_consequence(
                situation=situation,
                action=action,
                consequence=consequence,
                trust=trust,
                scope=[
                    "evidence-debt-resolution",
                    f"policy:{meta.get('policy_hint') or 'verify_first'}",
                    f"mission:{getattr(mission, 'id', '')}",
                    f"goal:{getattr(goal, 'id', '')}" if goal is not None else "goal:",
                    f"task:{getattr(task, 'id', '')}",
                    f"step:{getattr(step, 'id', '')}" if step is not None else "step:",
                    f"specialist:{getattr(specialist, 'id', '')}" if specialist is not None else "specialist:",
                    f"verdict:{self._verdict_value(eval_result)}",
                ],
                provenance=[
                    "remy:evidence_debt_runtime",
                    f"debt_task:{getattr(task, 'id', '')}",
                    f"reason:{meta.get('policy_reason') or ''}",
                ],
                links={
                    "mission": getattr(mission, "id", ""),
                    "goal": getattr(goal, "id", "") if goal is not None else "",
                    "task": getattr(task, "id", ""),
                    "step": getattr(step, "id", "") if step is not None else "",
                    "source_task": str(meta.get("opened_from_task_id") or ""),
                },
                namespace="remy",
            )
        except Exception:
            return ""

    def _consequence_from_eval(self, eval_result) -> tuple[str, int]:
        verdict = self._verdict_value(eval_result)
        if verdict == "success":
            return "SUPPORTS", 1
        if verdict in {"failure", "blocked", "needs_replan"}:
            return "REFUTES", -1
        return "INCONCLUSIVE", 0

    def _verdict_value(self, eval_result) -> str:
        verdict = getattr(eval_result, "verdict", "")
        return str(getattr(verdict, "value", verdict) or "").lower()
