"""
Runtime container for Remy v3.

Builds and owns the v3 service graph, while allowing existing callers to keep
using a dict-shaped runtime via `as_dict()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RuntimeContainer:
    chief: Any
    checkpoint_runtime: Any
    cycle_execution_runtime: Any
    scheduler_runtime: Any
    policy: Any
    budget: Any
    approval: Any
    audit: Any
    evaluator: Any
    cost_runtime: Any
    evaluation_runtime: Any
    factuality_runtime: Any
    specialist_inference_runtime: Any
    goal_context_runtime: Any
    plan_builder_runtime: Any
    registry: Any
    ops_query_runtime: Any
    goal_tracker: Any
    goal_query_runtime: Any
    mission_runtime: Any
    mission_query_runtime: Any
    plan_query_runtime: Any
    plan_state_runtime: Any
    mission_state_runtime: Any
    intake_runtime: Any
    context_runtime: Any
    step_state_runtime: Any
    task_state_runtime: Any
    task_decision_runtime: Any
    evidence_debt_runtime: Any
    task_outcome_runtime: Any
    goal_outcome_runtime: Any
    mission_outcome_runtime: Any
    decision_runtime: Any
    completion_runtime: Any
    recovery_runtime: Any
    result_runtime: Any
    memory_runtime: Any
    persistence_runtime: Any
    cycle_runtime: Any
    cycle_result_runtime: Any
    post_cycle_runtime: Any
    timing_runtime: Any
    loop_runtime: Any
    maintenance_runtime: Any
    guard_runtime: Any
    lifecycle_runtime: Any
    error_runtime: Any
    execution_gate: Any
    agent_output_runtime: Any
    execution_runtime: Any
    outcome_runtime: Any
    recorder: Any
    replanner: Any
    delegation: Any
    learner: Any
    playbooks: Any
    learning_runtime: Any
    recording_runtime: Any
    projection_runtime: Any
    dashboard_runtime: Any
    specialist_runtime: Any
    store: Any
    telemetry: Any
    loop: Any
    integrations: Any
    integration_registry: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "chief": self.chief,
            "checkpoint_runtime": self.checkpoint_runtime,
            "cycle_execution_runtime": self.cycle_execution_runtime,
            "scheduler_runtime": self.scheduler_runtime,
            "policy": self.policy,
            "budget": self.budget,
            "approval": self.approval,
            "audit": self.audit,
            "evaluator": self.evaluator,
            "cost_runtime": self.cost_runtime,
            "evaluation_runtime": self.evaluation_runtime,
            "factuality_runtime": self.factuality_runtime,
            "specialist_inference_runtime": self.specialist_inference_runtime,
            "goal_context_runtime": self.goal_context_runtime,
            "plan_builder_runtime": self.plan_builder_runtime,
            "registry": self.registry,
            "ops_query_runtime": self.ops_query_runtime,
            "goal_tracker": self.goal_tracker,
            "goal_query_runtime": self.goal_query_runtime,
            "mission_runtime": self.mission_runtime,
            "mission_query_runtime": self.mission_query_runtime,
            "plan_query_runtime": self.plan_query_runtime,
            "plan_state_runtime": self.plan_state_runtime,
            "mission_state_runtime": self.mission_state_runtime,
            "intake_runtime": self.intake_runtime,
            "context_runtime": self.context_runtime,
            "step_state_runtime": self.step_state_runtime,
            "task_state_runtime": self.task_state_runtime,
            "task_decision_runtime": self.task_decision_runtime,
            "evidence_debt_runtime": self.evidence_debt_runtime,
            "task_outcome_runtime": self.task_outcome_runtime,
            "goal_outcome_runtime": self.goal_outcome_runtime,
            "mission_outcome_runtime": self.mission_outcome_runtime,
            "decision_runtime": self.decision_runtime,
            "completion_runtime": self.completion_runtime,
            "recovery_runtime": self.recovery_runtime,
            "result_runtime": self.result_runtime,
            "memory_runtime": self.memory_runtime,
            "persistence_runtime": self.persistence_runtime,
            "cycle_runtime": self.cycle_runtime,
            "cycle_result_runtime": self.cycle_result_runtime,
            "post_cycle_runtime": self.post_cycle_runtime,
            "timing_runtime": self.timing_runtime,
            "loop_runtime": self.loop_runtime,
            "maintenance_runtime": self.maintenance_runtime,
            "guard_runtime": self.guard_runtime,
            "lifecycle_runtime": self.lifecycle_runtime,
            "error_runtime": self.error_runtime,
            "execution_gate": self.execution_gate,
            "agent_output_runtime": self.agent_output_runtime,
            "execution_runtime": self.execution_runtime,
            "outcome_runtime": self.outcome_runtime,
            "recorder": self.recorder,
            "replanner": self.replanner,
            "delegation": self.delegation,
            "learner": self.learner,
            "playbooks": self.playbooks,
            "learning_runtime": self.learning_runtime,
            "recording_runtime": self.recording_runtime,
            "projection_runtime": self.projection_runtime,
            "dashboard_runtime": self.dashboard_runtime,
            "specialist_runtime": self.specialist_runtime,
            "store": self.store,
            "telemetry": self.telemetry,
            "loop": self.loop,
            "integrations": self.integrations,
            "integration_registry": self.integration_registry,
        }

    @classmethod
    def build(cls):
        from ..governance.policy_engine import PolicyEngine
        from ..governance.budget_engine import BudgetEngine
        from ..governance.approval_engine import ApprovalEngine
        from ..governance.audit_engine import AuditEngine
        from ..evaluation.evaluation_engine import EvaluationEngine
        from ..agents.specialist_registry import SpecialistRegistry
        from ..agents.chief_agent import ChiefAgent, CycleResult, _emit_v3_approval_event
        from ..missions.mission_store import MissionStore
        from ..missions.goal_tracker import GoalTracker
        from ..missions.mission_runtime import MissionRuntime
        from ..missions.mission_persistence import MissionPersistence
        from ..planning.plan_persistence import PlanPersistence
        from ..planning.replan_engine import ReplanEngine
        from ..execution.cycle_recorder import CycleRecorder
        from ..execution.delegation_engine import DelegationEngine
        from ..observability.telemetry import Telemetry
        from ..improvement.outcome_learner import OutcomeLearner
        from ..improvement.playbook_engine import get_playbook_engine
        from ..integrations import AuthStore, IntegrationGateway, IntegrationRegistry
        from ..integrations.plugins.browser import BrowserPlugin
        from ..integrations.plugins.email_inbox import EmailInboxPlugin
        from ..integrations.plugins.email_send import EmailSendPlugin
        from ..integrations.plugins.github import GitHubPlugin
        from ..integrations.plugins.telegram import TelegramPlugin
        from .learning_runtime import LearningRuntime
        from .agent_output_runtime import AgentOutputRuntime
        from .checkpoint_runtime import CheckpointRuntime
        from .cycle_execution_runtime import CycleExecutionRuntime
        from .cost_runtime import CostRuntime
        from .cycle_result_runtime import CycleResultRuntime
        from .cycle_runtime import CycleRuntime
        from .execution_runtime import ExecutionRuntime
        from .evaluation_runtime import EvaluationRuntime
        from .factuality_runtime import FactualityRuntime
        from .completion_runtime import CompletionRuntime
        from .context_runtime import ContextRuntime
        from .decision_runtime import DecisionRuntime
        from .dashboard_runtime import DashboardRuntime
        from .evidence_debt_runtime import EvidenceDebtRuntime
        from .guard_runtime import GuardRuntime
        from .intake_runtime import IntakeRuntime
        from .persistence_runtime import PersistenceRuntime
        from .plan_query_runtime import PlanQueryRuntime
        from .plan_state_runtime import PlanStateRuntime
        from .memory_runtime import MemoryRuntime
        from .mission_outcome_runtime import MissionOutcomeRuntime
        from .mission_state_runtime import MissionStateRuntime
        from .goal_outcome_runtime import GoalOutcomeRuntime
        from .goal_context_runtime import GoalContextRuntime
        from .recovery_runtime import RecoveryRuntime
        from .result_runtime import ResultRuntime
        from .projection_runtime import ProjectionRuntime
        from .post_cycle_runtime import PostCycleRuntime
        from .plan_builder_runtime import PlanBuilderRuntime
        from .specialist_inference_runtime import SpecialistInferenceRuntime
        from .timing_runtime import TimingRuntime
        from .recording_runtime import RecordingRuntime
        from .specialist_runtime import SpecialistRuntime
        from .task_state_runtime import TaskStateRuntime
        from .task_decision_runtime import TaskDecisionRuntime
        from .task_outcome_runtime import TaskOutcomeRuntime
        from .step_state_runtime import StepStateRuntime
        from .loop_runtime import LoopRuntime
        from .maintenance_runtime import MaintenanceRuntime
        from .lifecycle_runtime import LifecycleRuntime
        from .error_runtime import ErrorRuntime
        from .goal_query_runtime import GoalQueryRuntime
        from .mission_query_runtime import MissionQueryRuntime
        from .ops_query_runtime import OpsQueryRuntime
        from .scheduler_runtime import SchedulerRuntime
        from .execution_gate import ExecutionGateRuntime
        from .outcome_runtime import OutcomeRuntime
        from .autonomy_loop import AutonomyLoop

        policy = PolicyEngine()
        budget = BudgetEngine()
        approval = ApprovalEngine()
        approval.set_notify_callback(lambda req: _emit_v3_approval_event("approval.pending", req))
        approval.set_decision_callback(lambda req: _emit_v3_approval_event("approval.resolved", req))
        audit = AuditEngine()
        evaluator = EvaluationEngine()

        registry = SpecialistRegistry()
        integration_registry = IntegrationRegistry()
        integration_registry.register(EmailInboxPlugin())
        integration_registry.register(EmailSendPlugin())
        integration_registry.register(GitHubPlugin())
        integration_registry.register(TelegramPlugin())
        integration_registry.register(BrowserPlugin())
        integrations = IntegrationGateway(
            registry=integration_registry,
            policy=policy,
            budget=budget,
            approval=approval,
            audit=audit,
            auth_store=AuthStore(),
        )

        goal_tracker = GoalTracker()
        cycle_recorder = CycleRecorder()
        replan_engine = ReplanEngine()
        mission_persistence = MissionPersistence()
        plan_persistence = PlanPersistence()
        delegation = DelegationEngine(registry=registry, budget=budget, audit=audit)

        learner = OutcomeLearner()
        playbooks = get_playbook_engine()
        learning_runtime = LearningRuntime(learner=learner, playbooks=playbooks)
        intake_runtime = IntakeRuntime(audit=audit)
        memory_runtime = MemoryRuntime(playbooks=playbooks)
        mission_state_runtime = MissionStateRuntime()
        persistence_runtime = PersistenceRuntime(
            mission_persistence=mission_persistence,
            plan_persistence=plan_persistence,
        )
        recording_runtime = RecordingRuntime(recorder=cycle_recorder, audit=audit)
        context_runtime = None

        chief = ChiefAgent(
            policy=policy,
            budget=budget,
            approval=approval,
            audit=audit,
            evaluator=evaluator,
            registry=registry,
            goal_tracker=goal_tracker,
            cycle_recorder=cycle_recorder,
            replan_engine=replan_engine,
            mission_persistence=mission_persistence,
            plan_persistence=plan_persistence,
            delegation_engine=delegation,
            bootstrap_defaults=False,
        )
        checkpoint_runtime = CheckpointRuntime()
        cycle_execution_runtime = CycleExecutionRuntime()
        timing_runtime = TimingRuntime()
        specialist_inference_runtime = SpecialistInferenceRuntime()
        goal_context_runtime = GoalContextRuntime()
        plan_builder_runtime = PlanBuilderRuntime(specialist_inference_runtime)
        cycle_result_runtime = CycleResultRuntime(result_factory=CycleResult)
        cost_runtime = CostRuntime()
        evaluation_runtime = EvaluationRuntime(
            evaluator=evaluator,
            approval=approval,
            budget=budget,
        )
        factuality_runtime = FactualityRuntime()
        agent_output_runtime = AgentOutputRuntime()
        execution_runtime = ExecutionRuntime(
            delegation_engine=delegation,
            agent_output_runtime=agent_output_runtime,
        )
        goal_query_runtime = GoalQueryRuntime(
            goals=chief._goals,
            tasks=chief._tasks,
        )
        ops_query_runtime = OpsQueryRuntime(
            budget=budget,
            audit=audit,
            approval=approval,
            recorder=cycle_recorder,
            evaluator=evaluator,
        )
        plan_query_runtime = PlanQueryRuntime(plans=chief._plans)
        plan_state_runtime = PlanStateRuntime()
        post_cycle_runtime = PostCycleRuntime(
            recording_runtime=recording_runtime,
            checkpoint_runtime=checkpoint_runtime,
            learning_runtime=learning_runtime,
        )
        mission_query_runtime = MissionQueryRuntime(
            missions=chief._missions,
            goals=chief._goals,
            tasks=chief._tasks,
            plans=chief._plans,
        )
        mission_runtime = MissionRuntime(
            goals=chief._goals,
            tasks=chief._tasks,
            plans=chief._plans,
            goal_tracker=goal_tracker,
            plan_loader=plan_persistence.load_by_mission,
            plan_creator=chief.create_plan,
            plan_builder_runtime=plan_builder_runtime,
        )
        context_runtime = ContextRuntime(
            recorder=cycle_recorder,
            budget=budget,
            goal_context_runtime=goal_context_runtime,
        )
        step_state_runtime = StepStateRuntime()
        task_state_runtime = TaskStateRuntime(step_state_runtime=step_state_runtime)
        task_decision_runtime = TaskDecisionRuntime(
            goal_tracker=goal_tracker,
            step_state_runtime=step_state_runtime,
        )
        evidence_debt_runtime = EvidenceDebtRuntime(chief._tasks)
        ops_query_runtime.bind_evidence_debt_runtime(evidence_debt_runtime)
        task_outcome_runtime = TaskOutcomeRuntime(
            goal_tracker=goal_tracker,
            tasks=mission_runtime.tasks,
        )
        goal_outcome_runtime = GoalOutcomeRuntime(
            goal_tracker=goal_tracker,
            tasks=mission_runtime.tasks,
            persistence_runtime=persistence_runtime,
        )
        mission_outcome_runtime = MissionOutcomeRuntime(
            mission_runtime=mission_runtime,
        )
        decision_runtime = DecisionRuntime(
            replanner=replan_engine,
            mission_outcome_runtime=mission_outcome_runtime,
        )
        result_runtime = ResultRuntime()
        completion_runtime = CompletionRuntime(
            mission_runtime=mission_runtime,
            plan_state_runtime=plan_state_runtime,
            task_outcome_runtime=task_outcome_runtime,
            goal_outcome_runtime=goal_outcome_runtime,
            mission_outcome_runtime=mission_outcome_runtime,
            result_runtime=result_runtime,
        )
        recovery_runtime = RecoveryRuntime(
            plan_state_runtime=plan_state_runtime,
            task_outcome_runtime=task_outcome_runtime,
            goal_outcome_runtime=goal_outcome_runtime,
            mission_outcome_runtime=mission_outcome_runtime,
            result_runtime=result_runtime,
        )
        outcome_runtime = OutcomeRuntime(
            goal_tracker=goal_tracker,
            mission_runtime=mission_runtime,
            replanner=replan_engine,
            decision_runtime=decision_runtime,
            completion_runtime=completion_runtime,
            recovery_runtime=recovery_runtime,
            plan_state_runtime=plan_state_runtime,
            task_outcome_runtime=task_outcome_runtime,
            goal_outcome_runtime=goal_outcome_runtime,
            mission_outcome_runtime=mission_outcome_runtime,
            persistence_runtime=persistence_runtime,
        )
        cycle_runtime = CycleRuntime(
            budget=budget,
            goal_tracker=goal_tracker,
            mission_runtime=mission_runtime,
            memory_runtime=memory_runtime,
            activate_mission=chief.activate_mission,
        )
        projection_runtime = ProjectionRuntime(
            mission_runtime=mission_runtime,
            mission_query_runtime=mission_query_runtime,
            plan_query_runtime=plan_query_runtime,
            goal_query_runtime=goal_query_runtime,
            goal_tracker=goal_tracker,
            recorder=cycle_recorder,
        )
        dashboard_runtime = DashboardRuntime(
            mission_query_runtime=mission_query_runtime,
            projection_runtime=projection_runtime,
            ops_query_runtime=ops_query_runtime,
            registry=registry,
            policy=policy,
            evaluator=evaluator,
            recorder=cycle_recorder,
        )

        specialist_runtime = SpecialistRuntime(
            registry=registry,
            mission_runtime=mission_runtime,
            specialist_inference_runtime=specialist_inference_runtime,
            goal_context_runtime=goal_context_runtime,
            evaluator=evaluator,
            ops_query_runtime=ops_query_runtime,
            recorder=cycle_recorder,
        )

        scheduler_runtime = SchedulerRuntime(
            mission_query_runtime,
            projection_runtime,
            mission_state_runtime=mission_state_runtime,
            loop_runtime=None,
            evaluator=evaluator,
            ops_query_runtime=ops_query_runtime,
        )

        execution_gate = ExecutionGateRuntime(
            policy=policy,
            approval=approval,
            ops_query_runtime=ops_query_runtime,
            goal_tracker=goal_tracker,
            recorder=cycle_recorder,
            budget=budget,
            context_runtime=context_runtime,
            task_state_runtime=task_state_runtime,
            task_decision_runtime=task_decision_runtime,
            evidence_debt_runtime=evidence_debt_runtime,
            specialist_resolver=lambda step, task, mission, goal: chief.specialist_runtime.resolve(
                step=step,
                task=task,
                mission=mission,
                goal=goal,
            ),
        )
        chief.bind_runtimes(
            persistence_runtime=persistence_runtime,
            checkpoint_runtime=checkpoint_runtime,
            cycle_execution_runtime=cycle_execution_runtime,
            timing_runtime=timing_runtime,
            specialist_inference_runtime=specialist_inference_runtime,
            goal_context_runtime=goal_context_runtime,
            plan_builder_runtime=plan_builder_runtime,
            intake_runtime=intake_runtime,
            memory_runtime=memory_runtime,
            cycle_result_runtime=cycle_result_runtime,
            cost_runtime=cost_runtime,
            evaluation_runtime=evaluation_runtime,
            factuality_runtime=factuality_runtime,
            agent_output_runtime=agent_output_runtime,
            execution_runtime=execution_runtime,
            goal_query_runtime=goal_query_runtime,
            ops_query_runtime=ops_query_runtime,
            mission_query_runtime=mission_query_runtime,
            mission_state_runtime=mission_state_runtime,
            mission_runtime=mission_runtime,
            plan_query_runtime=plan_query_runtime,
            plan_state_runtime=plan_state_runtime,
            learning_runtime=learning_runtime,
            recording_runtime=recording_runtime,
            post_cycle_runtime=post_cycle_runtime,
            context_runtime=context_runtime,
            step_state_runtime=step_state_runtime,
            task_state_runtime=task_state_runtime,
            task_decision_runtime=task_decision_runtime,
            evidence_debt_runtime=evidence_debt_runtime,
            task_outcome_runtime=task_outcome_runtime,
            goal_outcome_runtime=goal_outcome_runtime,
            mission_outcome_runtime=mission_outcome_runtime,
            decision_runtime=decision_runtime,
            result_runtime=result_runtime,
            completion_runtime=completion_runtime,
            recovery_runtime=recovery_runtime,
            outcome_runtime=outcome_runtime,
            cycle_runtime=cycle_runtime,
            projection_runtime=projection_runtime,
            dashboard_runtime=dashboard_runtime,
            specialist_runtime=specialist_runtime,
            execution_gate=execution_gate,
        )

        loop_runtime = LoopRuntime(chief)
        maintenance_runtime = MaintenanceRuntime(chief)
        guard_runtime = GuardRuntime(chief)
        lifecycle_runtime = LifecycleRuntime(chief)
        error_runtime = ErrorRuntime()
        for agent in delegation._agents.values():
            setattr(agent, "persistence_runtime", persistence_runtime)

        store = MissionStore()
        telemetry = Telemetry(chief)
        telemetry.bind_improvement(learner=learner, playbooks=playbooks)
        dashboard_runtime.bind_improvement(learner=learner, playbooks=playbooks)

        loop = AutonomyLoop(chief=chief)
        loop.scheduler_runtime = scheduler_runtime
        loop.loop_runtime = loop_runtime
        loop.scheduler_runtime.loop_runtime = loop_runtime
        loop.maintenance_runtime = maintenance_runtime
        loop.guard_runtime = guard_runtime
        loop.lifecycle_runtime = lifecycle_runtime
        loop.error_runtime = error_runtime
        ops_query_runtime.bind_autonomy(
            loop_runtime=loop_runtime,
            scheduler_runtime=scheduler_runtime,
            mission_query_runtime=mission_query_runtime,
            projection_runtime=projection_runtime,
        )

        return cls(
            chief=chief,
            checkpoint_runtime=chief.checkpoint_runtime,
            cycle_execution_runtime=chief.cycle_execution_runtime,
            scheduler_runtime=scheduler_runtime,
            policy=policy,
            budget=budget,
            approval=approval,
            audit=audit,
            evaluator=evaluator,
            cost_runtime=chief.cost_runtime,
            evaluation_runtime=chief.evaluation_runtime,
            factuality_runtime=chief.factuality_runtime,
            specialist_inference_runtime=chief.specialist_inference_runtime,
            goal_context_runtime=chief.goal_context_runtime,
            plan_builder_runtime=chief.plan_builder_runtime,
            registry=registry,
            ops_query_runtime=chief.ops_query_runtime,
            goal_tracker=goal_tracker,
            goal_query_runtime=chief.goal_query_runtime,
            mission_runtime=chief.mission_runtime,
            mission_query_runtime=chief.mission_query_runtime,
            plan_query_runtime=chief.plan_query_runtime,
            plan_state_runtime=chief.plan_state_runtime,
            mission_state_runtime=mission_state_runtime,
            intake_runtime=intake_runtime,
            context_runtime=context_runtime,
            step_state_runtime=step_state_runtime,
            task_state_runtime=task_state_runtime,
            task_decision_runtime=task_decision_runtime,
            evidence_debt_runtime=evidence_debt_runtime,
            task_outcome_runtime=task_outcome_runtime,
            goal_outcome_runtime=goal_outcome_runtime,
            mission_outcome_runtime=mission_outcome_runtime,
            decision_runtime=decision_runtime,
            completion_runtime=completion_runtime,
            recovery_runtime=recovery_runtime,
            result_runtime=result_runtime,
            memory_runtime=memory_runtime,
            persistence_runtime=persistence_runtime,
            cycle_runtime=cycle_runtime,
            cycle_result_runtime=cycle_result_runtime,
            post_cycle_runtime=chief.post_cycle_runtime,
            timing_runtime=chief.timing_runtime,
            loop_runtime=loop_runtime,
            maintenance_runtime=maintenance_runtime,
            guard_runtime=guard_runtime,
            lifecycle_runtime=lifecycle_runtime,
            error_runtime=error_runtime,
            execution_gate=execution_gate,
            agent_output_runtime=chief.agent_output_runtime,
            execution_runtime=chief.execution_runtime,
            outcome_runtime=outcome_runtime,
            recorder=cycle_recorder,
            replanner=replan_engine,
            delegation=delegation,
            learner=learner,
            playbooks=playbooks,
            learning_runtime=learning_runtime,
            recording_runtime=recording_runtime,
            projection_runtime=projection_runtime,
            dashboard_runtime=dashboard_runtime,
            specialist_runtime=specialist_runtime,
            store=store,
            telemetry=telemetry,
            loop=loop,
            integrations=integrations,
            integration_registry=integration_registry,
        )
