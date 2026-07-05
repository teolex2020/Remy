"""
Chief Agent for Remy v3.

The strategic orchestrator accepts missions, selects the next concrete unit of
work, delegates to specialists, evaluates the result, and persists runtime
truth. The v3 control path is task-first: if a mission has atomic tasks, they
drive execution ahead of prompt-style goal improvisation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..agents.specialist_registry import SpecialistRegistry
from ..evaluation.evaluation_engine import EvaluationEngine
from ..execution.cycle_recorder import CycleRecorder
from ..execution.delegation_engine import DelegationEngine
from ..governance.approval_engine import ApprovalEngine
from ..governance.audit_engine import AuditEngine
from ..governance.budget_engine import BudgetEngine
from ..governance.policy_engine import PolicyEngine
from ..missions.goal_tracker import GoalTracker
from ..missions.mission_models import (
    Goal,
    Mission,
    MissionStatus,
    Task,
)
from ..missions.mission_runtime import MissionRuntime
from ..missions.mission_persistence import MissionPersistence
from ..planning.plan_models import Plan, PlanStep, PlanType
from ..planning.plan_persistence import PlanPersistence
from ..planning.replan_engine import ReplanEngine
from ..runtime.cycle_runtime import CycleRuntime
from ..runtime.cycle_result_runtime import CycleResultRuntime
from ..runtime.cycle_execution_runtime import CycleExecutionRuntime
from ..runtime.checkpoint_runtime import CheckpointRuntime
from ..runtime.cost_runtime import CostRuntime
from ..runtime.evaluation_runtime import EvaluationRuntime
from ..runtime.agent_output_runtime import AgentOutputRuntime
from ..runtime.context_runtime import ContextRuntime
from ..runtime.completion_runtime import CompletionRuntime
from ..runtime.decision_runtime import DecisionRuntime
from ..runtime.dashboard_runtime import DashboardRuntime
from ..runtime.evidence_debt_runtime import EvidenceDebtRuntime
from ..runtime.factuality_runtime import FactualityRuntime
from ..runtime.execution_runtime import ExecutionRuntime
from ..runtime.execution_gate import ExecutionGateRuntime
from ..runtime.goal_query_runtime import GoalQueryRuntime
from ..runtime.intake_runtime import IntakeRuntime
from ..runtime.memory_runtime import MemoryRuntime
from ..runtime.mission_query_runtime import MissionQueryRuntime
from ..runtime.mission_outcome_runtime import MissionOutcomeRuntime
from ..runtime.mission_state_runtime import MissionStateRuntime
from ..runtime.ops_query_runtime import OpsQueryRuntime
from ..runtime.outcome_runtime import OutcomeRuntime
from ..runtime.post_cycle_runtime import PostCycleRuntime
from ..runtime.goal_outcome_runtime import GoalOutcomeRuntime
from ..runtime.plan_state_runtime import PlanStateRuntime
from ..runtime.plan_query_runtime import PlanQueryRuntime
from ..runtime.projection_runtime import ProjectionRuntime
from ..runtime.recovery_runtime import RecoveryRuntime
from ..runtime.recording_runtime import RecordingRuntime
from ..runtime.result_runtime import ResultRuntime
from ..runtime.specialist_runtime import SpecialistRuntime
from ..runtime.specialist_inference_runtime import SpecialistInferenceRuntime
from ..runtime.goal_context_runtime import GoalContextRuntime
from ..runtime.plan_builder_runtime import PlanBuilderRuntime
from ..runtime.task_state_runtime import TaskStateRuntime
from ..runtime.task_decision_runtime import TaskDecisionRuntime
from ..runtime.task_outcome_runtime import TaskOutcomeRuntime
from ..runtime.step_state_runtime import StepStateRuntime
from ..runtime.timing_runtime import TimingRuntime
from ..runtime.state_machine import can_transition, transition

log = logging.getLogger(__name__)


def _emit_v3_approval_event(event_name: str, req) -> None:
    try:
        from remy.core.event_bus import event_bus
        from remy.core.runtime_event_contract import build_runtime_event

        decision = getattr(req, "status", None)
        decision_value = getattr(decision, "value", str(decision or ""))
        payload = {
            "action_id": req.id,
            "description": req.description or req.action,
            "specialist": req.specialist,
            "risk_category": req.risk_category,
            "created_at": req.created_at,
            "expires_at": req.expires_at,
            "context": dict(req.context or {}),
            "routing_pressure": bool(
                (req.context or {}).get("quality_debt") is not None
                or "routing pressure" in (req.description or "").lower()
            ),
        }
        if event_name == "approval.pending":
            payload["timeout_sec"] = int(max(0, round((req.expires_at or 0) - (req.created_at or 0))))
        else:
            payload["decision"] = decision_value or ("approved" if getattr(req, "approved", False) else "resolved")
            payload["approved"] = decision_value in {"approved", "auto_approved"}
            payload["decided_at"] = getattr(req, "decided_at", 0.0)
            payload["decided_by"] = getattr(req, "decided_by", "")
            payload["denial_reason"] = getattr(req, "denial_reason", "")

        event_bus.emit(
            event_name,
            build_runtime_event(
                event_name,
                event_domain="approval",
                payload=payload,
                legacy_fields=payload,
            ),
        )
    except Exception:
        pass


class ChiefDecision:
    EXECUTE_STEP = "execute_step"
    REPLAN = "replan"
    ESCALATE = "escalate"
    PAUSE = "pause"
    COMPLETE = "complete"
    ABORT = "abort"


@dataclass
class CycleResult:
    decision: str = ChiefDecision.PAUSE
    mission_id: str = ""
    goal_id: str = ""
    step_executed: str = ""
    specialist_used: str = ""
    eval_verdict: str = ""
    cost_usd: float = 0.0
    tokens_used: int = 0
    reason: str = ""
    next_action: str = ""
    memory_context_used: bool = False
    unsupported_observed_claims: int = 0


class ChiefAgent:
    """Strategic orchestrator for Remy v3."""

    def __init__(
        self,
        policy: PolicyEngine | None = None,
        budget: BudgetEngine | None = None,
        approval: ApprovalEngine | None = None,
        audit: AuditEngine | None = None,
        evaluator: EvaluationEngine | None = None,
        registry: SpecialistRegistry | None = None,
        goal_tracker: GoalTracker | None = None,
        cycle_recorder: CycleRecorder | None = None,
        replan_engine: ReplanEngine | None = None,
        mission_persistence: MissionPersistence | None = None,
        plan_persistence: PlanPersistence | None = None,
        delegation_engine: DelegationEngine | None = None,
        bootstrap_defaults: bool = True,
    ):
        self.policy = policy or PolicyEngine()
        self.budget = budget or BudgetEngine()
        self.approval = approval or ApprovalEngine()
        self.approval.set_notify_callback(lambda req: _emit_v3_approval_event("approval.pending", req))
        self.approval.set_decision_callback(lambda req: _emit_v3_approval_event("approval.resolved", req))
        self.audit = audit or AuditEngine()
        self.evaluator = evaluator or EvaluationEngine()
        self.registry = registry or SpecialistRegistry()
        self.goal_tracker = goal_tracker or GoalTracker()
        self.recorder = cycle_recorder or CycleRecorder()
        self.replanner = replan_engine or ReplanEngine()
        self.mission_persistence = mission_persistence or MissionPersistence()
        self.plan_persistence = plan_persistence or PlanPersistence()
        self.delegation = delegation_engine or DelegationEngine(
            registry=self.registry,
            budget=self.budget,
            audit=self.audit,
        )

        self._missions: dict[str, Mission] = {}
        self._goals: dict[str, Goal] = {}
        self._tasks: dict[str, Task] = {}
        self._plans: dict[str, Plan] = {}
        self._cycle_count = 0
        self.cycle_result_runtime = None
        self.checkpoint_runtime = None
        self.cost_runtime = None
        self.timing_runtime = None
        self.specialist_inference_runtime = None
        self.goal_context_runtime = None
        self.plan_builder_runtime = None
        self.evaluation_runtime = None
        self.agent_output_runtime = None
        self.execution_runtime = None
        self.intake_runtime = None
        self.memory_runtime = None
        self.goal_query_runtime = None
        self.ops_query_runtime = None
        self.mission_query_runtime = None
        self.plan_query_runtime = None
        self.mission_state_runtime = None
        self.mission_runtime = None
        self.specialist_runtime = None
        self.goal_outcome_runtime = None
        self.mission_outcome_runtime = None
        self.decision_runtime = None
        self.result_runtime = None
        self.task_outcome_runtime = None
        self.completion_runtime = None
        self.recovery_runtime = None
        self.outcome_runtime = None
        self.plan_state_runtime = None
        self.cycle_runtime = None
        self.context_runtime = None
        self.step_state_runtime = None
        self.task_state_runtime = None
        self.task_decision_runtime = None
        self.execution_gate = None
        self.evidence_debt_runtime = None
        self.projection_runtime = None
        self.dashboard_runtime = None
        self.learning_runtime = None
        self.factuality_runtime = None
        self.persistence_runtime = None
        self.recording_runtime = None
        self.post_cycle_runtime = None
        self.cycle_execution_runtime = None

        if bootstrap_defaults:
            self._init_default_runtimes()

    def _init_default_runtimes(self):
        self.cycle_result_runtime = CycleResultRuntime(result_factory=CycleResult)
        self.checkpoint_runtime = CheckpointRuntime()
        self.cost_runtime = CostRuntime()
        self.timing_runtime = TimingRuntime()
        self.specialist_inference_runtime = SpecialistInferenceRuntime()
        self.goal_context_runtime = GoalContextRuntime()
        self.plan_builder_runtime = PlanBuilderRuntime(self.specialist_inference_runtime)
        self.evaluation_runtime = EvaluationRuntime(
            evaluator=self.evaluator,
            approval=self.approval,
            budget=self.budget,
        )
        self.agent_output_runtime = AgentOutputRuntime()
        self.execution_runtime = ExecutionRuntime(
            delegation_engine=self.delegation,
            agent_output_runtime=self.agent_output_runtime,
        )
        self.intake_runtime = IntakeRuntime(audit=self.audit)
        self.memory_runtime = MemoryRuntime()
        self.goal_query_runtime = GoalQueryRuntime(
            goals=self._goals,
            tasks=self._tasks,
        )
        self.ops_query_runtime = OpsQueryRuntime(
            budget=self.budget,
            audit=self.audit,
            approval=self.approval,
            recorder=self.recorder,
            evaluator=self.evaluator,
        )
        self.mission_query_runtime = MissionQueryRuntime(
            missions=self._missions,
            goals=self._goals,
            tasks=self._tasks,
            plans=self._plans,
        )
        self.plan_query_runtime = PlanQueryRuntime(plans=self._plans)
        self.mission_state_runtime = MissionStateRuntime()
        self.mission_runtime = MissionRuntime(
            goals=self._goals,
            tasks=self._tasks,
            plans=self._plans,
            goal_tracker=self.goal_tracker,
            plan_loader=self.plan_persistence.load_by_mission,
            plan_creator=self.create_plan,
            plan_builder_runtime=self.plan_builder_runtime,
        )
        self.specialist_runtime = SpecialistRuntime(
            registry=self.registry,
            mission_runtime=self.mission_runtime,
            specialist_inference_runtime=self.specialist_inference_runtime,
            goal_context_runtime=self.goal_context_runtime,
            evaluator=self.evaluator,
            ops_query_runtime=self.ops_query_runtime,
            recorder=self.recorder,
        )
        self.goal_outcome_runtime = GoalOutcomeRuntime(
            goal_tracker=self.goal_tracker,
            tasks=self.mission_runtime.tasks,
        )
        self.mission_outcome_runtime = MissionOutcomeRuntime(
            mission_runtime=self.mission_runtime,
        )
        self.decision_runtime = DecisionRuntime(
            replanner=self.replanner,
            mission_outcome_runtime=self.mission_outcome_runtime,
        )
        self.result_runtime = ResultRuntime()
        self.task_outcome_runtime = TaskOutcomeRuntime(
            goal_tracker=self.goal_tracker,
            tasks=self.mission_runtime.tasks,
        )
        self.completion_runtime = CompletionRuntime(
            mission_runtime=self.mission_runtime,
            plan_state_runtime=PlanStateRuntime(),
            task_outcome_runtime=self.task_outcome_runtime,
            goal_outcome_runtime=self.goal_outcome_runtime,
            mission_outcome_runtime=self.mission_outcome_runtime,
            result_runtime=self.result_runtime,
        )
        self.recovery_runtime = RecoveryRuntime(
            plan_state_runtime=self.completion_runtime.plan_state_runtime,
            task_outcome_runtime=self.task_outcome_runtime,
            goal_outcome_runtime=self.goal_outcome_runtime,
            mission_outcome_runtime=self.mission_outcome_runtime,
            result_runtime=self.result_runtime,
        )
        self.outcome_runtime = OutcomeRuntime(
            goal_tracker=self.goal_tracker,
            mission_runtime=self.mission_runtime,
            replanner=self.replanner,
            decision_runtime=self.decision_runtime,
            completion_runtime=self.completion_runtime,
            recovery_runtime=self.recovery_runtime,
            plan_state_runtime=self.completion_runtime.plan_state_runtime,
            task_outcome_runtime=self.task_outcome_runtime,
            goal_outcome_runtime=self.goal_outcome_runtime,
            mission_outcome_runtime=self.mission_outcome_runtime,
        )
        self.plan_state_runtime = self.outcome_runtime.plan_state_runtime
        self.cycle_runtime = CycleRuntime(
            budget=self.budget,
            goal_tracker=self.goal_tracker,
            mission_runtime=self.mission_runtime,
            memory_runtime=self.memory_runtime,
            activate_mission=self.activate_mission,
        )
        self.context_runtime = ContextRuntime(
            recorder=self.recorder,
            budget=self.budget,
            goal_context_runtime=self.goal_context_runtime,
        )
        self.step_state_runtime = StepStateRuntime()
        self.task_state_runtime = TaskStateRuntime(step_state_runtime=self.step_state_runtime)
        self.task_decision_runtime = TaskDecisionRuntime(
            goal_tracker=self.goal_tracker,
            step_state_runtime=self.step_state_runtime,
        )
        self.evidence_debt_runtime = EvidenceDebtRuntime(self._tasks)
        self.ops_query_runtime.bind_evidence_debt_runtime(self.evidence_debt_runtime)
        self.execution_gate = ExecutionGateRuntime(
            policy=self.policy,
            approval=self.approval,
            ops_query_runtime=self.ops_query_runtime,
            goal_tracker=self.goal_tracker,
            recorder=self.recorder,
            budget=self.budget,
            context_runtime=self.context_runtime,
            task_state_runtime=self.task_state_runtime,
            task_decision_runtime=self.task_decision_runtime,
            evidence_debt_runtime=self.evidence_debt_runtime,
            specialist_resolver=lambda step, task, mission, goal: self.specialist_runtime.resolve(
                step=step,
                task=task,
                mission=mission,
                goal=goal,
            ),
        )
        self.projection_runtime = ProjectionRuntime(
            mission_runtime=self.mission_runtime,
            mission_query_runtime=self.mission_query_runtime,
            plan_query_runtime=self.plan_query_runtime,
            goal_query_runtime=self.goal_query_runtime,
            goal_tracker=self.goal_tracker,
            recorder=self.recorder,
        )
        self.dashboard_runtime = DashboardRuntime(
            mission_query_runtime=self.mission_query_runtime,
            projection_runtime=self.projection_runtime,
            ops_query_runtime=self.ops_query_runtime,
            registry=self.registry,
            policy=self.policy,
            evaluator=self.evaluator,
            recorder=self.recorder,
        )
        self.learning_runtime = None
        self.persistence_runtime = None
        self.recording_runtime = RecordingRuntime(recorder=self.recorder, audit=self.audit)
        self.post_cycle_runtime = PostCycleRuntime(
            recording_runtime=self.recording_runtime,
            checkpoint_runtime=self.checkpoint_runtime,
            learning_runtime=self.learning_runtime,
        )
        self.cycle_execution_runtime = CycleExecutionRuntime()
        self.factuality_runtime = FactualityRuntime()

    def bind_runtimes(self, **runtime_bindings):
        for name, value in runtime_bindings.items():
            setattr(self, name, value)

    def save_state(self):
        if self.persistence_runtime is not None:
            self.persistence_runtime.save_runtime_state(
                self._missions,
                self._goals,
                self._tasks,
                self._plans,
            )
            return
        self.mission_persistence.save(self._missions, self._goals, self._tasks)
        for plan in self._plans.values():
            self.plan_persistence.save(plan)

    def load_state(self):
        missions, goals, tasks = self.mission_persistence.load()
        self._missions.update(missions)
        self._goals.update(goals)
        self._tasks.update(tasks)
        self.goal_tracker.bind_goals(self._goals)

        for mission in missions.values():
            if mission.plan_id:
                plan = self.plan_persistence.load(mission.plan_id)
                if plan is None:
                    plan = self.plan_persistence.load_by_mission(mission.id)
                if plan is not None:
                    self._plans[mission.id] = plan

        log.info(
            "Loaded state: %d missions, %d goals, %d tasks, %d plans",
            len(missions),
            len(goals),
            len(tasks),
            len(self._plans),
        )

    def accept_mission(self, mission: Mission) -> Mission:
        mission = self.intake_runtime.accept(mission)
        self._missions[mission.id] = mission
        return mission

    def activate_mission(self, mission: Mission) -> Mission:
        return self.mission_state_runtime.activate_for_execution(mission)

    def add_goal(self, goal: Goal):
        self._goals[goal.id] = goal
        if goal.mission_id and goal.mission_id in self._missions:
            mission = self._missions[goal.mission_id]
            if goal.id not in mission.goal_ids:
                mission.goal_ids.append(goal.id)
        self.goal_tracker.bind_goals(self._goals)

    def add_task(self, task: Task):
        self._tasks[task.id] = task

    def create_plan(self, mission: Mission, steps: list[PlanStep] | None = None) -> Plan:
        plan = Plan(
            mission_id=mission.id,
            plan_type=PlanType.LINEAR,
            steps=steps or [],
            budget_ceiling_usd=mission.budget.cost_usd or 0.50,
            token_ceiling=mission.budget.tokens or 50_000,
        )
        mission.plan_id = plan.id
        if mission.status == MissionStatus.INTAKE:
            transition(mission, MissionStatus.PLANNING, "Plan created")

        self._plans[mission.id] = plan
        if self.persistence_runtime is not None:
            self.persistence_runtime.save_plan(plan)
        else:
            self.plan_persistence.save(plan)
        self.audit.log_event(
            "plan_created",
            f"Plan {plan.id} ({len(plan.steps)} steps) for {mission.id}",
            actor="chief",
            mission_id=mission.id,
        )
        return plan

    def create_plan_from_tasks(self, mission: Mission) -> Plan:
        tasks = self.mission_runtime.mission_tasks(mission.id)
        return self.create_plan(mission, self.plan_builder_runtime.steps_from_tasks(tasks))

    async def run_cycle(self, mission: Mission) -> CycleResult:
        self._cycle_count += 1
        return await self.cycle_execution_runtime.run(self, mission=mission, cycle_num=self._cycle_count)

    def get_mission(self, mission_id: str) -> Mission | None:
        return self._missions.get(mission_id)

    def active_missions(self) -> list[Mission]:
        return [mission for mission in self._missions.values() if mission.is_active()]

    def all_missions(self) -> list[Mission]:
        return list(self._missions.values())

    def mission_summary(self, mission_id: str) -> dict[str, Any]:
        mission = self._missions.get(mission_id)
        if mission is None:
            return {}
        return self.projection_runtime.mission_summary(
            mission,
            plan=self._plans.get(mission_id),
        )
