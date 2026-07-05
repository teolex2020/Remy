"""
Execution Gate Runtime for Remy v3.

Owns deterministic pre-execution checks:
- policy
- approval
- specialist resolution
- agent context construction
- step/task promotion into running state
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from ..governance.policy_engine import PolicyDecision, PolicyEngine
from ..governance.approval_engine import ApprovalEngine
from ..missions.goal_tracker import GoalTracker
from ..missions.mission_models import Goal, Mission, Task, TaskStatus
from ..planning.plan_models import Plan, PlanStep, StepStatus
from .state_machine import transition_task


@dataclass
class ExecutionGateResult:
    proceed: bool = False
    decision: str = "pause"
    reason: str = ""
    specialist: object | None = None
    agent_ctx: object | None = None


class ExecutionGateRuntime:
    """Deterministic gate before delegation."""

    def __init__(
        self,
        *,
        policy: PolicyEngine,
        approval: ApprovalEngine,
        ops_query_runtime,
        goal_tracker: GoalTracker,
        recorder,
        budget,
        context_runtime,
        task_state_runtime,
        task_decision_runtime,
        evidence_debt_runtime=None,
        specialist_resolver: Callable[[PlanStep, Task | None, Mission, Goal | None], object],
    ):
        self.policy = policy
        self.approval = approval
        self.ops_query_runtime = ops_query_runtime
        self.goal_tracker = goal_tracker
        self.recorder = recorder
        self.budget = budget
        self.context_runtime = context_runtime
        self.task_state_runtime = task_state_runtime
        self.task_decision_runtime = task_decision_runtime
        self.evidence_debt_runtime = evidence_debt_runtime
        self._resolve_specialist = specialist_resolver

    def _approval_risk_category(self, *, mission: Mission, plan: Plan) -> str:
        mission_risk = getattr(mission.risk, "value", "") or ""
        plan_risk = (plan.risk_level or "").lower()
        if mission_risk in {"critical", "high", "medium"}:
            return mission_risk
        if plan_risk in {"critical", "high", "medium"}:
            return plan_risk
        if mission_risk == "safe" or plan_risk == "safe":
            return "safe"
        return mission_risk or plan_risk or "low"

    def _specialist_pressure_gate(
        self,
        *,
        mission: Mission,
        goal: Goal | None,
        task: Task | None,
        plan: Plan,
        step: PlanStep,
        specialist,
    ) -> tuple[bool, str, str, dict]:
        if self.ops_query_runtime is None or specialist is None:
            return False, "", "", {}

        specialist_id = getattr(specialist, "id", "") or ""
        if not specialist_id:
            return False, "", "", {}

        quality = self.ops_query_runtime.specialist_quality(specialist_id) or {}
        success_rate = float(quality.get("success_rate", 0.5) or 0.5)
        adjusted = float(quality.get("quality_adjusted_success_rate", success_rate) or success_rate)
        unsupported = int(quality.get("unsupported_claims", 0) or 0)
        debt = max(0.0, success_rate - adjusted)
        degraded = adjusted <= 0.55 or unsupported >= 2 or debt >= 0.15
        if not degraded:
            return False, "", "", quality

        risk_category = self._approval_risk_category(mission=mission, plan=plan)
        if risk_category not in {"medium", "high", "critical"}:
            return False, "", risk_category, quality

        target = task.action if task is not None else step.instruction
        reason = (
            f"Routing pressure approval: specialist '{specialist_id}' is degraded "
            f"(quality={adjusted:.2f}, unsupported_claims={unsupported}) for {risk_category}-risk work"
        )
        return True, reason, risk_category, {
            **quality,
            "quality_debt": round(debt, 3),
            "target": target,
            "goal_id": goal.id if goal is not None else "",
            "task_id": task.id if task is not None else "",
            "step_id": step.id,
        }

    def _planned_action(self, *, task: Task | None, step: PlanStep) -> str:
        return (task.action if task is not None else step.instruction) or ""

    def _situation(self, *, mission: Mission, goal: Goal | None) -> str:
        return (goal.description if goal is not None else mission.description or "")[:240]

    @staticmethod
    def _classify_causal_edge(supports: int, refutes: int) -> str:
        """Label an (action → failure) consequence pair as a typed causal edge.

        Uses the SDK's typed causal grammar to distinguish a genuine causal scar
        (`refutes`: the action counterfactually leads to a bad outcome) from a
        mere correlation (`precedes`: the failure occurs about as often WITHOUT
        the action). For example, this distinguishes a tool action that causes
        a failure from an action that merely happened before the failure.

        Mapping to the classifier's (cause → effect) frame, where the effect is
        FAILURE: support for the (action→failure) pattern is the number of
        REFUTES (failures that followed the action), and counterevidence is the
        number of SUPPORTS (times the action was taken WITHOUT failure). The
        effect polarity is negative for refutes, positive for supports.

        Fail-soft: on older AuraSDK wheels without the classifier, falls back to a
        conservative binary label.
        """
        try:
            from aura import classify_causal_edge

            lift = 2.0 if refutes > supports else 0.0
            return classify_causal_edge(
                support_count=int(refutes),
                counterevidence=int(supports),
                transition_lift=lift,
                positive_effect_signals=int(supports),
                negative_effect_signals=int(refutes),
            )
        except Exception:
            # Conservative fallback: a refutation with no support is a scar.
            if refutes > 0 and supports == 0:
                return "refutes"
            return "precedes"

    def _consequence_scar_gate(
        self,
        *,
        mission: Mission,
        goal: Goal | None,
        task: Task | None,
        step: PlanStep,
    ) -> tuple[bool, str]:
        """Block a planned action that lived consequence memory has refuted.

        Only a genuine causal scar hard-blocks. A merely correlational refutation
        (`precedes` — the failure also occurs without this action) does NOT
        hard-block; it is left to the softer policy-hint / evidence path. This
        prevents the loop from blocking an action just because failure once
        followed it by coincidence.
        """
        if self.recorder is None or not hasattr(self.recorder, "scar_check"):
            return False, ""

        action = self._planned_action(task=task, step=step).strip()
        situation = self._situation(mission=mission, goal=goal).strip()
        if not action or not situation:
            return False, ""

        try:
            verdict = self.recorder.scar_check(situation, action)
        except Exception:
            return False, ""

        if getattr(verdict, "is_refuted", False):
            refutes = int(getattr(verdict, "refutes", 0) or 0)
            supports = int(getattr(verdict, "supports", 0) or 0)
            is_scar = bool(getattr(verdict, "scar", False))

            # The gaslight guard always wins: a hardened scar (a refutation that
            # already SURVIVED later supporting frequency) blocks regardless of
            # the causal-edge label. Supporting frequency must never bury a lived
            # refutation — that is the founding scar-protection principle.
            if not is_scar:
                # Not (yet) a hardened scar: use the typed causal grammar to
                # avoid hard-blocking on mere correlation. Only a genuine causal
                # `refutes` edge blocks; `precedes` (correlation) or `enables`
                # (ambiguous) is deferred to the softer policy-hint / evidence
                # path so the loop isn't stopped by coincidence.
                edge = self._classify_causal_edge(supports, refutes)
                if edge != "refutes":
                    return False, ""
            else:
                edge = "refutes"

            scar_note = ""
            if is_scar:
                scar_note = f"; survived {supports} later support(s)"
            return True, (
                f"Consequence scar blocks execution: action '{action}' in this situation "
                f"was REFUTED by lived memory {refutes} time(s) "
                f"(causal edge: {edge}){scar_note}."
            )
        return False, ""

    def _policy_hint_for_action(
        self,
        *,
        mission: Mission,
        goal: Goal | None,
        task: Task | None,
        step: PlanStep,
    ) -> dict | None:
        if self.recorder is None or not hasattr(self.recorder, "policy_hint"):
            return None

        action = self._planned_action(task=task, step=step).strip()
        situation = self._situation(mission=mission, goal=goal).strip()
        if not action or not situation:
            return None

        try:
            hint = self.recorder.policy_hint(situation, action)
        except Exception:
            return None
        if hint is None:
            return None
        if hasattr(hint, "to_context"):
            context = hint.to_context()
            context.setdefault("situation", situation)
            context.setdefault("action", action)
            return context
        if isinstance(hint, dict):
            context = dict(hint)
            context.setdefault("situation", situation)
            context.setdefault("action", action)
            return context
        return None

    def prepare(
        self,
        *,
        mission: Mission,
        goal: Goal | None,
        task: Task | None,
        plan: Plan,
        step: PlanStep,
        memory_context: list[dict],
    ) -> ExecutionGateResult:
        policy_decision, policy_reason = self.policy.evaluate(
            action=step.instruction[:50],
            cost_usd=step.cost_estimate_usd,
            specialist=step.specialist,
        )
        if policy_decision == PolicyDecision.DENY:
            self.task_decision_runtime.deny_execution(
                step=step,
                task=task,
                reason=policy_reason,
            )
            return ExecutionGateResult(
                proceed=False,
                decision="abort",
                reason=f"Execution denied by policy: {policy_reason}",
            )

        scar_blocked, scar_reason = self._consequence_scar_gate(
            mission=mission,
            goal=goal,
            task=task,
            step=step,
        )
        if scar_blocked:
            self.task_decision_runtime.block_for_consequence_scar(
                step=step,
                task=task,
                reason=scar_reason,
            )
            return ExecutionGateResult(
                proceed=False,
                decision="pause",
                reason=scar_reason,
            )

        policy_hint = self._policy_hint_for_action(
            mission=mission,
            goal=goal,
            task=task,
            step=step,
        )
        if policy_hint and policy_hint.get("should_block"):
            reason = str(policy_hint.get("reason") or "Consequence policy hint blocks execution.")
            self.task_decision_runtime.block_for_consequence_scar(
                step=step,
                task=task,
                reason=reason,
            )
            return ExecutionGateResult(
                proceed=False,
                decision="pause",
                reason=reason,
            )

        if (
            policy_hint
            and policy_hint.get("requires_evidence")
            and self.evidence_debt_runtime is not None
        ):
            debt_task = self.evidence_debt_runtime.open_debt(
                mission=mission,
                goal=goal,
                current_task=task,
                policy_hint=policy_hint,
            )
            if debt_task is not None:
                policy_hint = {
                    **policy_hint,
                    "evidence_debt_task_id": debt_task.id,
                    "evidence_debt_action": debt_task.action,
                }

        if policy_decision == PolicyDecision.APPROVE:
            approval_req = self.approval.request_approval(
                action=step.instruction[:100],
                mission_id=mission.id,
                specialist=step.specialist,
                risk_category=getattr(mission.risk, "value", "") or plan.risk_level,
                cost_usd=step.cost_estimate_usd,
            )
            if not self.approval.is_approved(approval_req.id):
                self.task_decision_runtime.block_for_approval(
                    task=task,
                    approval_id=approval_req.id,
                )
                return ExecutionGateResult(
                    proceed=False,
                    decision="pause",
                    reason=f"Awaiting approval: {approval_req.id}",
                )

        specialist = self._resolve_specialist(step, task, mission, goal)
        needs_pressure_approval, pressure_reason, risk_category, pressure_ctx = self._specialist_pressure_gate(
            mission=mission,
            goal=goal,
            task=task,
            plan=plan,
            step=step,
            specialist=specialist,
        )
        if needs_pressure_approval:
            approval_req = self.approval.request_approval(
                action=f"routing_pressure:{getattr(specialist, 'id', '')}:{step.instruction[:80]}",
                description=pressure_reason,
                mission_id=mission.id,
                specialist=getattr(specialist, "id", "") or step.specialist,
                risk_category=risk_category,
                cost_usd=step.cost_estimate_usd,
                context=pressure_ctx,
            )
            if not self.approval.is_approved(approval_req.id):
                self.task_decision_runtime.block_for_approval(
                    task=task,
                    approval_id=approval_req.id,
                )
                return ExecutionGateResult(
                    proceed=False,
                    decision="pause",
                    reason=pressure_reason,
                    specialist=specialist,
                )

        self.task_state_runtime.promote_for_execution(step=step, task=task)

        agent_ctx = self.context_runtime.build(
            mission=mission,
            goal=goal,
            step=step,
            specialist=specialist,
            memory_context=memory_context,
            policy_hints=[policy_hint] if policy_hint else [],
        )
        return ExecutionGateResult(
            proceed=True,
            decision="execute_step",
            reason="ready_for_delegation",
            specialist=specialist,
            agent_ctx=agent_ctx,
        )
