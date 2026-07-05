"""
Recording runtime for Remy v3.

Builds normalized cycle records and emits matching audit events so the chief
does not handcraft execution event payloads inline.
"""

from __future__ import annotations

from ..evaluation.evaluation_engine import EvalVerdict
from ..execution.cycle_recorder import CycleRecord


class RecordingRuntime:
    """Normalizes cycle recording and audit emission."""

    def __init__(self, recorder, audit):
        self.recorder = recorder
        self.audit = audit

    def record_cycle(
        self,
        *,
        cycle_num: int,
        mission,
        goal,
        task,
        plan,
        step,
        specialist,
        exec_result,
        eval_result,
        decision: str,
        memory_assisted: bool,
        duration_ms: int,
    ) -> CycleRecord:
        record = CycleRecord(
            cycle_num=cycle_num,
            mission_id=mission.id,
            goal_id=goal.id if goal else "",
            goal_description=(goal.description[:120] if goal else mission.description[:120]),
            plan_id=plan.id,
            step_id=step.id,
            specialist=specialist.id,
            status=exec_result.status.value,
            verdict=eval_result.verdict.value,
            confidence=eval_result.confidence,
            reason=eval_result.reason[:120],
            duration_ms=duration_ms,
            tokens_used=exec_result.tokens_used,
            cost_usd=exec_result.cost_usd,
            tool_calls=exec_result.tool_calls,
            model=getattr(exec_result, "model", ""),
            fallback_used=bool(getattr(exec_result, "fallback_used", False)),
            verified=(
                eval_result.verdict == EvalVerdict.SUCCESS
                and not eval_result.should_continue
            ),
            repeated_failure=(
                bool(goal)
                and goal.attempts >= 3
                and eval_result.verdict not in (EvalVerdict.SUCCESS, EvalVerdict.PARTIAL)
            ),
            memory_assisted=memory_assisted,
            decision=decision,
            planned_action=task.action if task is not None else step.instruction,
            unsupported_observed_claims=getattr(exec_result, "unsupported_observed_claims", 0),
            factuality_report=getattr(exec_result, "factuality_report", None),
        )
        self.recorder.record(record)
        self.audit.log_event(
            "cycle_completed",
            f"Step {step.id}: {eval_result.verdict.value} -> {decision}",
            actor="chief",
            mission_id=mission.id,
            goal_id=goal.id if goal else "",
            cost_usd=exec_result.cost_usd,
            details={
                "task_id": task.id if task else "",
                "specialist": specialist.id,
                "verdict": eval_result.verdict.value,
                "decision": decision,
                "memory_assisted": memory_assisted,
                "unsupported_observed_claims": getattr(exec_result, "unsupported_observed_claims", 0),
            },
        )
        return record
