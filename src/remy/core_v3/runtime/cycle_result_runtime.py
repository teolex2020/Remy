"""
Cycle Result Runtime for Remy v3.

Builds and updates the operator-facing cycle result contract so ChiefAgent
does not handcraft `CycleResult` instances inline.
"""

from __future__ import annotations


class CycleResultRuntime:
    """Factory and updater for normalized cycle result payloads."""

    def __init__(self, result_factory):
        self.result_factory = result_factory

    def initial(self, *, mission_id: str):
        return self.result_factory(mission_id=mission_id)

    @staticmethod
    def apply_cycle_prep(result, *, cycle_prep) -> None:
        result.decision = cycle_prep.decision
        result.reason = cycle_prep.reason
        if cycle_prep.goal:
            result.goal_id = cycle_prep.goal.id

    @staticmethod
    def apply_context(result, *, goal, memory_context) -> None:
        result.goal_id = goal.id if goal else ""
        result.memory_context_used = bool(memory_context)

    @staticmethod
    def apply_gate(result, *, gate) -> None:
        result.decision = gate.decision
        result.reason = gate.reason

    @staticmethod
    def apply_specialist(result, *, specialist) -> None:
        result.specialist_used = specialist.id

    @staticmethod
    def apply_execution(result, *, exec_result) -> None:
        result.cost_usd = exec_result.cost_usd
        result.tokens_used = exec_result.tokens_used
        result.unsupported_observed_claims = getattr(exec_result, "unsupported_observed_claims", 0)

    @staticmethod
    def apply_evaluation(result, *, eval_result, step_id: str) -> None:
        result.eval_verdict = eval_result.verdict.value
        result.step_executed = step_id

    @staticmethod
    def apply_outcome(result, *, outcome) -> None:
        result.decision = outcome.decision
        result.reason = outcome.reason
        result.next_action = outcome.next_action
