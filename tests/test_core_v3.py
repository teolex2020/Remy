"""
Tests for Remy v3 core components.

Covers all 8 phases: models, governance, evaluation, research,
agents, improvement, telemetry, and cross-component integration.
"""

import asyncio
import os
import tempfile
import time
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Phase 1: Models
# ---------------------------------------------------------------------------

class TestMissionModels:
    def test_mission_lifecycle(self):
        from remy.core_v3.missions.mission_models import (
            Mission, MissionStatus, Goal, GoalStatus, Task, TaskStatus,
        )
        m = Mission(objective="Test mission")
        assert m.status == MissionStatus.INTAKE
        # INTAKE is not yet active (active = ACTIVE or PLANNING)
        assert not m.is_active()
        m.status = MissionStatus.ACTIVE
        assert m.is_active()

        g = Goal(description="Test goal", mission_id=m.id)
        assert g.status == GoalStatus.PENDING

        t = Task(action="Test task", goal_id=g.id)
        assert t.status == TaskStatus.PENDING
        t.status = TaskStatus.WAITING
        assert t.is_waiting()
        t.status = TaskStatus.BLOCKED_APPROVAL
        assert t.is_blocked()

    def test_task_state_machine(self):
        from remy.core_v3.missions.mission_models import Task, TaskStatus
        from remy.core_v3.runtime.state_machine import transition_task

        task = Task(action="Test")
        transition_task(task, TaskStatus.WAITING, "dependency")
        assert task.status == TaskStatus.WAITING
        assert task.waiting_reason == "dependency"
        transition_task(task, TaskStatus.PENDING, "deps ready")
        assert task.status == TaskStatus.PENDING

    def test_mission_runtime_selects_dependency_ready_task(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Mission, Task, Goal

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        goal = Goal(description="Research market shifts", mission_id=mission.id)
        chief.add_goal(goal)

        t1 = Task(id="task_a", action="Collect source list", mission_id=mission.id, goal_id=goal.id)
        t2 = Task(
            id="task_b",
            action="Analyze the collected sources",
            mission_id=mission.id,
            goal_id=goal.id,
            depends_on=["task_a"],
        )
        chief.add_task(t1)
        chief.add_task(t2)

        selected = chief.mission_runtime.select_task_for_mission(mission)
        assert selected.id == "task_a"
        assert t2.is_waiting()

    def test_plan_models(self):
        from remy.core_v3.planning.plan_models import (
            Plan, PlanStep, StepStatus, PlanType,
        )
        s1 = PlanStep(id="s1", description="Step 1")
        s2 = PlanStep(id="s2", description="Step 2", depends_on=["s1"])
        plan = Plan(steps=[s1, s2], plan_type=PlanType.LINEAR)

        assert plan.progress == 0.0
        assert plan.next_step() == s1
        assert not s2.is_ready(set())
        assert s2.is_ready({"s1"})

    def test_memory_classes(self):
        from remy.core_v3.memory.memory_api import MemoryClass
        assert MemoryClass.IDENTITY.value == "identity"
        assert MemoryClass.STRATEGIC.value == "strategic"

    def test_record_models(self):
        from remy.core_v3.memory.record_models import (
            mission_record, finding_record,
        )
        # record_models return (content, tags, metadata, memory_class) tuples
        _, tags, meta, _ = mission_record("Test mission", mission_id="m1")
        assert tags == ["mission"]
        assert meta["mission_id"] == "m1"

        _, tags, _, _ = finding_record("Some finding", mission_id="m1")
        assert "finding" in tags

    def test_goal_adapter_accepts_iso_timestamps(self):
        from remy.core_v3.missions.mission_models import goal_from_v2_record

        goal = goal_from_v2_record({
            "id": "g1",
            "content": "Test goal",
            "tags": ["autonomous-goal"],
            "metadata": {
                "status": "active",
                "created_at": "2026-03-04T20:17:17.899114",
                "updated_at": "2026-03-04T20:18:17.899114",
            },
        })
        assert goal.created_at > 0
        assert goal.updated_at >= goal.created_at


class TestIntakeRuntime:
    def test_accept_classifies_mode_budget_and_risk(self):
        import tempfile
        from remy.core_v3.governance.audit_engine import AuditEngine
        from remy.core_v3.missions.mission_models import Mission, MissionMode, RiskLevel
        from remy.core_v3.runtime.intake_runtime import IntakeRuntime

        intake = IntakeRuntime(audit=AuditEngine(log_path=tempfile.mktemp(suffix=".jsonl")))
        research = intake.accept(Mission(description="Research market shifts"))
        assert research.mode == MissionMode.DEEP_RESEARCH
        assert research.budget.tokens == 50_000
        assert research.risk == RiskLevel.LOW
        assert not research.requires_approval

        funds = intake.accept(Mission(description="Transfer funds to vendor"))
        assert funds.risk == RiskLevel.CRITICAL
        assert funds.requires_approval


# ---------------------------------------------------------------------------
# Phase 4: Research Engine
# ---------------------------------------------------------------------------

class TestResearchModels:
    def test_research_project(self):
        from remy.core_v3.research.research_models import (
            ResearchProject, ResearchStatus, Source, Finding,
            FindingConfidence, ResearchQuestion,
        )
        p = ResearchProject(objective="Test research")
        assert p.status == ResearchStatus.PLANNING
        assert p.progress == 0.0
        assert p.source_coverage == 0.0

        q = ResearchQuestion(question="What is X?")
        p.questions.append(q)

        src = Source(url="https://example.com", title="Test")
        p.add_source(src)
        assert len(p.sources) == 1

        f = Finding(content="Found X", source_ids=[src.id])
        p.add_finding(f)
        assert len(p.findings) == 1

    def test_source_ranking(self):
        from remy.core_v3.research.source_ranking import SourceRanker
        from remy.core_v3.research.research_models import Source, SourceCredibility

        ranker = SourceRanker()
        s1 = Source(url="https://github.com/test", search_rank=1)
        s2 = Source(url="https://random.blogspot.com/x", search_rank=2)
        ranked = ranker.rank([s1, s2])

        assert ranked[0].credibility == SourceCredibility.HIGH
        assert ranked[-1].credibility == SourceCredibility.LOW

    def test_contradiction_detection(self):
        from remy.core_v3.research.synthesis import SynthesisEngine
        from remy.core_v3.research.research_models import Finding

        se = SynthesisEngine()
        fa = Finding(content="A", category="pricing",
                     structured_data={"price": 10}, source_ids=["s1"])
        fb = Finding(content="B", category="pricing",
                     structured_data={"price": 20}, source_ids=["s2"])

        contras = se.detect_contradictions([fa, fb])
        assert len(contras) == 1
        assert "ratio" in contras[0].description

    def test_synthesis(self):
        from remy.core_v3.research.synthesis import SynthesisEngine
        from remy.core_v3.research.research_models import (
            ResearchProject, Finding, FindingConfidence, Source,
        )

        se = SynthesisEngine(min_corroboration=2)
        p = ResearchProject(objective="Test")
        p.sources = [Source(url="https://example.com", fetched=True)]
        p.findings = [
            Finding(content="Multi source", source_ids=["s1", "s2", "s3"]),
            Finding(content="Single source", source_ids=["s1"]),
        ]
        result = se.synthesize(p)
        assert result.finding_count == 2
        assert 0.0 < result.confidence <= 1.0

    def test_research_runtime_planning(self):
        from remy.core_v3.research.research_runtime import ResearchRuntime
        from remy.core_v3.research.research_models import (
            ResearchProject, ResearchStatus,
        )

        rt = ResearchRuntime()
        proj = ResearchProject(objective="Test objective")

        async def run():
            await rt._phase_plan(proj)
            return proj

        proj = asyncio.run(run())
        assert proj.status == ResearchStatus.COLLECTING
        assert len(proj.questions) == 3
        assert all(q.search_queries for q in proj.questions)

    def test_research_policy_thresholds(self):
        from remy.core_v3.research.research_policy import assess_evidence

        success = assess_evidence(
            usable_sources=3,
            finding_count=4,
            confidence=0.6,
            contradictions_checked=True,
        )
        assert success.verdict == "success"

        partial = assess_evidence(
            usable_sources=1,
            finding_count=2,
            confidence=0.3,
            contradictions_checked=True,
        )
        assert partial.verdict == "partial"

    def test_research_context_hydrates_project_from_memory_and_playbook(self):
        from remy.core_v3.improvement.playbook_engine import PlaybookEngine
        from remy.core_v3.research.research_context import ResearchContextRuntime
        from remy.core_v3.research.research_models import ResearchProject

        class FakeRecord:
            def __init__(self, content):
                self.content = content

        class FakeMemory:
            def __init__(self):
                self.stored = []

            def recall(self, query, tags=None, limit=8):
                return [FakeRecord("Prior finding about competitor positioning")]

            def store(self, content, tags, metadata=None, memory_class=None):
                self.stored.append((content, tags, metadata, memory_class))
                return "rec1"

        memory = FakeMemory()
        playbooks = PlaybookEngine()
        playbooks.create_from_execution(
            name="Research competitors",
            goal_description="Research competitor positioning",
            domain="research",
            steps=[{"action": "competitor positioning latest updates", "specialist": "researcher"}],
        )
        ctx = ResearchContextRuntime(memory=memory, playbooks=playbooks)
        project = ResearchProject(objective="Research competitor positioning")

        ctx.hydrate_project(project)
        assert project.prior_context
        assert project.strategy_hints
        assert project.reused_playbook_id

    def test_research_context_stores_session_summary(self):
        from remy.core_v3.research.research_context import ResearchContextRuntime
        from remy.core_v3.research.research_models import ResearchProject, Synthesis

        class FakeMemory:
            def __init__(self):
                self.stored = []

            def recall(self, query, tags=None, limit=8):
                return []

            def store(self, content, tags, metadata=None, memory_class=None):
                self.stored.append((content, tags, metadata, memory_class))
                return "rec1"

        memory = FakeMemory()
        ctx = ResearchContextRuntime(memory=memory)
        project = ResearchProject(objective="Research market shifts", mission_id="m1", goal_id="g1")
        project.synthesis = Synthesis(summary="Summary", confidence=0.7, source_count=3, finding_count=4)
        project.reused_playbook_id = "pb_123"

        ctx.store_project_summary(project)
        assert memory.stored
        assert "research-session" in memory.stored[0][1]


# ---------------------------------------------------------------------------
# Phase 5: Governance
# ---------------------------------------------------------------------------

class TestPolicyEngine:
    def test_safe_action_allowed(self):
        from remy.core_v3.governance.policy_engine import (
            PolicyEngine, PolicyDecision,
        )
        pe = PolicyEngine()
        dec, _ = pe.evaluate("research_competitors")
        assert dec == PolicyDecision.ALLOW

    def test_financial_action_requires_approval(self):
        from remy.core_v3.governance.policy_engine import (
            PolicyEngine, PolicyDecision,
        )
        pe = PolicyEngine()
        dec, _ = pe.evaluate("financial_transfer")
        assert dec == PolicyDecision.APPROVE

    def test_blocked_tools_denied(self):
        from remy.core_v3.governance.policy_engine import (
            PolicyEngine, PolicyDecision,
        )
        pe = PolicyEngine()
        dec, _ = pe.evaluate("any_action", tools=["delete_all_memory"])
        assert dec == PolicyDecision.DENY

    def test_approval_tools(self):
        from remy.core_v3.governance.policy_engine import (
            PolicyEngine, PolicyDecision,
        )
        pe = PolicyEngine()
        dec, _ = pe.evaluate("any_action", tools=["send_transaction"])
        assert dec == PolicyDecision.APPROVE

    def test_tool_level_check(self):
        from remy.core_v3.governance.policy_engine import (
            PolicyEngine, PolicyDecision,
        )
        pe = PolicyEngine()
        dec, _ = pe.evaluate_tool("factory_reset")
        assert dec == PolicyDecision.DENY
        dec, _ = pe.evaluate_tool("web_search")
        assert dec == PolicyDecision.ALLOW

    def test_high_cost_escalation(self):
        from remy.core_v3.governance.policy_engine import (
            PolicyEngine, PolicyDecision,
        )
        pe = PolicyEngine()
        dec, _ = pe.evaluate("custom_action", cost_usd=0.15)
        assert dec == PolicyDecision.APPROVE

    def test_rule_management(self):
        from remy.core_v3.governance.policy_engine import (
            PolicyEngine, PolicyDecision, PolicyRule,
        )
        pe = PolicyEngine()
        pe.add_rule(PolicyRule(
            id="test", action_pattern="test_*",
            decision=PolicyDecision.DENY, enabled=True,
        ))
        dec, _ = pe.evaluate("test_x")
        assert dec == PolicyDecision.DENY

        pe.disable_rule("test")
        dec, _ = pe.evaluate("test_x")
        assert dec != PolicyDecision.DENY

        pe.remove_rule("test")


class TestBudgetEngine:
    def test_within_budget(self):
        from remy.core_v3.governance.budget_engine import (
            BudgetEngine, BudgetAction, BudgetConfig,
        )
        be = BudgetEngine(config=BudgetConfig(daily_usd=1.0, per_cycle_usd=0.10))
        action, _ = be.check_budget(0.05)
        assert action == BudgetAction.ALLOW

    def test_record_spend(self):
        from remy.core_v3.governance.budget_engine import (
            BudgetEngine, BudgetConfig,
        )
        be = BudgetEngine(config=BudgetConfig(daily_usd=1.0, per_cycle_usd=10.0))
        be.record_spend(0.03, tokens=100, mission_id="m1", specialist="researcher")
        assert be.state.daily_spent_usd == 0.03
        assert be.state.mission_spent["m1"] == 0.03
        assert len(be.state.history) == 1

    def test_cycle_cap(self):
        from remy.core_v3.governance.budget_engine import (
            BudgetEngine, BudgetAction, BudgetConfig,
        )
        be = BudgetEngine(config=BudgetConfig(daily_usd=1.0, per_cycle_usd=0.10))
        be.record_spend(0.03)
        action, _ = be.check_budget(0.08)  # 0.03 + 0.08 > 0.10
        assert action == BudgetAction.DENY

        be.start_cycle()
        action, _ = be.check_budget(0.05)
        assert action == BudgetAction.ALLOW

    def test_warning_threshold(self):
        from remy.core_v3.governance.budget_engine import (
            BudgetEngine, BudgetAction, BudgetStatus, BudgetConfig,
        )
        be = BudgetEngine(config=BudgetConfig(daily_usd=1.0, per_cycle_usd=10.0))
        be.record_spend(0.85)
        action, _ = be.check_budget(0.0)
        assert action == BudgetAction.DEGRADE
        assert be.get_status() == BudgetStatus.WARNING

    def test_exhausted(self):
        from remy.core_v3.governance.budget_engine import (
            BudgetEngine, BudgetStatus, BudgetConfig,
        )
        be = BudgetEngine(config=BudgetConfig(daily_usd=1.0, per_cycle_usd=10.0))
        be.record_spend(1.01)
        assert be.get_status() == BudgetStatus.EXHAUSTED

    def test_summary_fields(self):
        from remy.core_v3.governance.budget_engine import BudgetEngine
        be = BudgetEngine()
        s = be.summary()
        assert "daily_remaining_usd" in s
        assert "recommended_model" in s
        assert "spending_rate_per_hour" in s


class TestApprovalEngine:
    def test_auto_approve_safe(self):
        from remy.core_v3.governance.approval_engine import (
            ApprovalEngine, ApprovalStatus,
        )
        ae = ApprovalEngine()
        req = ae.request_approval("read", risk_category="safe")
        assert req.status == ApprovalStatus.AUTO_APPROVED

    def test_auto_approve_low_cost(self):
        from remy.core_v3.governance.approval_engine import (
            ApprovalEngine, ApprovalStatus,
        )
        ae = ApprovalEngine()
        req = ae.request_approval("small", risk_category="low", cost_usd=0.005)
        assert req.status == ApprovalStatus.AUTO_APPROVED

    def test_approve_deny(self):
        from remy.core_v3.governance.approval_engine import (
            ApprovalEngine, ApprovalStatus,
        )
        ae = ApprovalEngine()
        req = ae.request_approval("publish", risk_category="high", cost_usd=0.1)
        assert req.status == ApprovalStatus.PENDING

        ae.approve(req.id, "user")
        assert ae.is_approved(req.id)

        req2 = ae.request_approval("risky", risk_category="medium")
        ae.deny(req2.id, reason="nope")
        assert not ae.is_approved(req2.id)

    def test_batch_approve(self):
        from remy.core_v3.governance.approval_engine import ApprovalEngine
        ae = ApprovalEngine()
        ae.request_approval("a", risk_category="medium")
        ae.request_approval("b", risk_category="medium")
        count = ae.approve_all("batch")
        assert count == 2
        assert len(ae.pending()) == 0

    def test_notification_callback(self):
        from remy.core_v3.governance.approval_engine import ApprovalEngine
        notified = []
        ae = ApprovalEngine(notify_callback=lambda r: notified.append(r.id))
        ae.request_approval("pub", risk_category="high")
        assert len(notified) == 1

    def test_decision_callback_receives_resolved_request_context(self):
        from remy.core_v3.governance.approval_engine import ApprovalEngine

        decided = []
        ae = ApprovalEngine(decision_callback=lambda r: decided.append(r))
        req = ae.request_approval(
            "publish",
            description="Routing pressure approval: specialist 'researcher' is degraded",
            specialist="researcher",
            risk_category="medium",
            context={"quality_debt": 0.23, "target": "Research counterparty profile"},
        )

        assert ae.approve(req.id, "telegram") is True
        assert len(decided) == 1
        assert decided[0].id == req.id
        assert decided[0].decided_by == "telegram"
        assert decided[0].context["target"] == "Research counterparty profile"

    def test_v3_pending_event_bridge_emits_live_payload_shape(self):
        from remy.core_v3.agents.chief_agent import _emit_v3_approval_event
        from remy.core_v3.governance.approval_engine import ApprovalRequest

        req = ApprovalRequest(
            id="approval-v3-1",
            action="routing_pressure:researcher:Research counterparty profile",
            description="Routing pressure approval: specialist 'researcher' is degraded",
            mission_id="mission-1",
            specialist="researcher",
            risk_category="medium",
            created_at=100.0,
            expires_at=220.0,
            context={"quality_debt": 0.23, "target": "Research counterparty profile"},
        )
        emitted = []

        with patch("remy.core.event_bus.event_bus.emit", side_effect=lambda name, event: emitted.append((name, event))):
            _emit_v3_approval_event("approval.pending", req)

        assert len(emitted) == 1
        name, event = emitted[0]
        assert name == "approval.pending"
        assert event["event_name"] == "approval.pending"
        assert event["event_domain"] == "approval"
        assert event["payload"]["action_id"] == "approval-v3-1"
        assert event["payload"]["specialist"] == "researcher"
        assert event["payload"]["routing_pressure"] is True
        assert event["payload"]["timeout_sec"] == 120
        assert event["payload"]["context"]["target"] == "Research counterparty profile"

    def test_v3_resolved_event_bridge_emits_decision_context(self):
        from remy.core_v3.agents.chief_agent import _emit_v3_approval_event
        from remy.core_v3.governance.approval_engine import ApprovalRequest, ApprovalStatus

        req = ApprovalRequest(
            id="approval-v3-1",
            action="routing_pressure:researcher:Research counterparty profile",
            description="Routing pressure approval: specialist 'researcher' is degraded",
            mission_id="mission-1",
            specialist="researcher",
            risk_category="medium",
            created_at=100.0,
            decided_at=112.0,
            decided_by="telegram",
            context={"quality_debt": 0.23, "target": "Research counterparty profile"},
            status=ApprovalStatus.APPROVED,
        )
        emitted = []

        with patch("remy.core.event_bus.event_bus.emit", side_effect=lambda name, event: emitted.append((name, event))):
            _emit_v3_approval_event("approval.resolved", req)

        assert len(emitted) == 1
        name, event = emitted[0]
        assert name == "approval.resolved"
        assert event["event_name"] == "approval.resolved"
        assert event["payload"]["decision"] == "approved"
        assert event["payload"]["approved"] is True
        assert event["payload"]["decided_by"] == "telegram"
        assert event["payload"]["routing_pressure"] is True
        assert event["payload"]["context"]["target"] == "Research counterparty profile"


class TestAuditEngine:
    def test_log_and_query(self):
        from remy.core_v3.governance.audit_engine import AuditEngine, EventType

        tmp = tempfile.mktemp(suffix=".jsonl")
        try:
            ae = AuditEngine(log_path=tmp)
            ae.log_event(EventType.MISSION_STARTED, "start", mission_id="m1", actor="chief")
            ae.log_event(EventType.ERROR, "fail", mission_id="m1", actor="exec")
            ae.log_event(EventType.BUDGET_SPEND, "llm", mission_id="m2", cost_usd=0.05)

            assert len(ae.recent(10)) == 3
            assert len(ae.recent_by_mission("m1")) == 2
            assert len(ae.errors()) == 1
            assert ae.total_cost(24.0) == 0.05
            assert ae.mission_cost("m2") == 0.05

            stats = ae.actor_stats(24.0)
            assert "chief" in stats
            assert stats["exec"]["errors"] == 1

            with open(tmp) as f:
                assert len(f.readlines()) == 3
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


# ---------------------------------------------------------------------------
# Phase 6: Evaluation & Replanning
# ---------------------------------------------------------------------------

class TestEvaluationEngine:
    def test_evidence_backed_success(self):
        from remy.core_v3.evaluation.evaluation_engine import (
            EvaluationEngine, EvalVerdict,
        )
        from remy.core_v3.execution.execution_runtime import (
            ExecutionResult, ExecutionStatus,
        )

        ee = EvaluationEngine()
        result = ee.evaluate(ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            tool_calls=3,
            evidence={
                "synthesis": {"source_count": 3, "finding_count": 4},
                "findings": [{"content": "a"}, {"content": "b"}, {"content": "c"}],
                "contradictions_checked": True,
            },
        ))
        assert result.verdict == EvalVerdict.SUCCESS

    def test_research_evidence_partial_uses_shared_policy(self):
        from remy.core_v3.evaluation.evaluation_engine import (
            EvaluationEngine, EvalVerdict,
        )
        from remy.core_v3.execution.execution_runtime import (
            ExecutionResult, ExecutionStatus,
        )

        ee = EvaluationEngine()
        result = ee.evaluate(ExecutionResult(
            status=ExecutionStatus.PARTIAL,
            tool_calls=2,
            evidence={
                "synthesis": {"source_count": 1, "finding_count": 2, "confidence": 0.3},
                "findings": [{"content": "a"}, {"content": "b"}],
                "artifacts": [{"type": "source"}],
                "contradictions_checked": True,
            },
        ))
        assert result.verdict == EvalVerdict.PARTIAL

    def test_blocker_classification(self):
        from remy.core_v3.evaluation.evaluation_engine import (
            EvaluationEngine, BlockerType,
        )
        ee = EvaluationEngine()
        assert ee._classify_blocker("captcha required") == BlockerType.CAPTCHA
        assert ee._classify_blocker("429 rate limit") == BlockerType.RATE_LIMIT
        assert ee._classify_blocker("normal text") is None

    def test_repeated_failure_detection(self):
        from remy.core_v3.evaluation.evaluation_engine import EvaluationEngine
        ee = EvaluationEngine()
        ee._failure_history = [
            {"goal_id": "g1", "specialist": "r", "verdict": "failure", "blocker": None},
            {"goal_id": "g1", "specialist": "r", "verdict": "failure", "blocker": None},
        ]
        assert ee._check_repeated_failure("g1") is True
        assert ee._check_repeated_failure("g2") is False

    def test_specialist_scoring(self):
        from remy.core_v3.evaluation.evaluation_engine import (
            EvaluationEngine, EvalResult, EvalVerdict,
        )
        ee = EvaluationEngine()
        ee._record_outcome("g1", "res", EvalResult(verdict=EvalVerdict.SUCCESS))
        ee._record_outcome("g2", "res", EvalResult(verdict=EvalVerdict.FAILURE))
        ee._record_outcome("g3", "res", EvalResult(verdict=EvalVerdict.SUCCESS))
        assert ee.specialist_success_rate("res") > 0.5
        assert ee.specialist_success_rate("unknown") == 0.5

    def test_factuality_penalty_reduces_confidence_and_specialist_score(self):
        from remy.core_v3.evaluation.evaluation_engine import EvaluationEngine
        from remy.core_v3.execution.execution_runtime import ExecutionResult, ExecutionStatus

        ee = EvaluationEngine()
        result = ee.evaluate(
            ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                tool_calls=1,
                response="I checked the repo and it looks good.",
                unsupported_observed_claims=2,
            ),
            specialist="researcher",
            unsupported_observed_claims=2,
        )
        assert result.unsupported_observed_claims == 2
        assert result.factuality_penalty > 0
        assert result.confidence < 0.7
        assert ee.specialist_success_rate("researcher") < 0.8


class TestReplanEngine:
    def test_retry(self):
        from remy.core_v3.planning.replan_engine import ReplanEngine, ReplanAction
        from remy.core_v3.planning.plan_models import Plan, PlanStep

        re = ReplanEngine()
        step = PlanStep(id="s1", description="test", attempts=1, retry_limit=3)
        plan = Plan(steps=[step])
        dec = re.decide(plan, step)
        assert dec.action == ReplanAction.RETRY

    def test_transient_blocker_wait(self):
        from remy.core_v3.planning.replan_engine import ReplanEngine, ReplanAction
        from remy.core_v3.planning.plan_models import Plan, PlanStep

        re = ReplanEngine()
        step = PlanStep(id="s1", description="x", attempts=0)
        plan = Plan(steps=[step])

        class MockEval:
            class blocker_type:
                value = "rate_limit"
            is_repeated_failure = False

        dec = re.decide(plan, step, eval_result=MockEval())
        assert dec.action == ReplanAction.WAIT
        assert dec.wait_seconds > 0

    def test_human_blocker_escalate(self):
        from remy.core_v3.planning.replan_engine import ReplanEngine, ReplanAction
        from remy.core_v3.planning.plan_models import Plan, PlanStep

        re = ReplanEngine()
        step = PlanStep(id="s1", description="x", attempts=0)
        plan = Plan(steps=[step])

        class MockEval:
            class blocker_type:
                value = "captcha"
            is_repeated_failure = False

        dec = re.decide(plan, step, eval_result=MockEval())
        assert dec.action == ReplanAction.ESCALATE

    def test_repeated_failure_decompose(self):
        from remy.core_v3.planning.replan_engine import ReplanEngine, ReplanAction
        from remy.core_v3.planning.plan_models import Plan, PlanStep

        re = ReplanEngine()
        step = PlanStep(id="s1", description="x", attempts=3, retry_limit=2)
        plan = Plan(steps=[step])

        class MockEval:
            blocker_type = None
            is_repeated_failure = True

        dec = re.decide(plan, step, eval_result=MockEval())
        assert dec.action == ReplanAction.DECOMPOSE

    def test_fallback(self):
        from remy.core_v3.planning.replan_engine import ReplanEngine, ReplanAction
        from remy.core_v3.planning.plan_models import Plan, PlanStep

        re = ReplanEngine()
        s_main = PlanStep(id="s1", description="main", attempts=3,
                          retry_limit=2, fallback_step_id="s2")
        s_fb = PlanStep(id="s2", description="fallback")
        plan = Plan(steps=[s_main, s_fb])
        dec = re.decide(plan, s_main)
        assert dec.action == ReplanAction.FALLBACK


# ---------------------------------------------------------------------------
# Phase 7: Self-Improvement
# ---------------------------------------------------------------------------

class TestOutcomeLearner:
    def test_observe_and_analyze(self):
        from remy.core_v3.improvement.outcome_learner import OutcomeLearner

        ol = OutcomeLearner(min_evidence=2)
        for _ in range(3):
            ol.observe_outcome("g1", "research", "researcher", "success")
        for _ in range(3):
            ol.observe_outcome("g2", "signup", "executor", "failure", blocker="captcha")

        insights = ol.analyze()
        assert len(insights) > 0

        fit = ol.get_insights("specialist_fit")
        assert any("researcher" in i.description for i in fit)

        fail = ol.get_insights("failure_pattern")
        assert any("captcha" in i.description for i in fail)

    def test_analyze_factuality_pattern(self):
        from remy.core_v3.improvement.outcome_learner import OutcomeLearner

        ol = OutcomeLearner(min_evidence=2)
        ol.observe_outcome("g1", "research", "researcher", "success", unsupported_observed_claims=1)
        ol.observe_outcome("g2", "research", "researcher", "partial", unsupported_observed_claims=1)

        insights = ol.analyze()
        assert any(i.category == "quality_pattern" for i in insights)
        assert ol.summary()["unsupported_observed_claims_total"] == 2

    def test_store_insights_to_memory_writes_policy_consequences(self, monkeypatch):
        from remy.core_v3.improvement.outcome_learner import OutcomeLearner
        from remy.core_v3.memory import memory_api

        class StubMemory:
            def __init__(self):
                self.records = []
                self.units = []

            def store(self, **kwargs):
                self.records.append(kwargs)
                return f"record-{len(self.records)}"

            def capture_consequence(self, **kwargs):
                self.units.append(kwargs)
                return f"unit-{len(self.units)}"

        memory = StubMemory()
        monkeypatch.setattr(memory_api, "get_memory", lambda: memory)

        ol = OutcomeLearner(min_evidence=2)
        ol.observe_outcome("g1", "research evidence", "researcher", "success", unsupported_observed_claims=1)
        ol.observe_outcome("g2", "research evidence", "researcher", "success", unsupported_observed_claims=1)
        ol.analyze()

        ol.store_insights_to_memory()

        assert any(record["metadata"]["category"] == "specialist_fit" for record in memory.records)
        assert any(
            unit["namespace"] == "remy-routing"
            and unit["action"] == "route_to:researcher"
            and unit["consequence"] == "SUPPORTS"
            and "policy:prefer" in unit["scope"]
            for unit in memory.units
        )
        assert any(
            unit["namespace"] == "remy-factuality"
            and unit["action"] == "answer_without_evidence"
            and unit["consequence"] == "REFUTES"
            and "policy:requires_evidence" in unit["scope"]
            for unit in memory.units
        )

        unit_count = len(memory.units)
        ol.store_insights_to_memory()
        assert len(memory.units) == unit_count


class TestPlaybookEngine:
    def test_create_and_match(self):
        from remy.core_v3.improvement.playbook_engine import PlaybookEngine

        pe = PlaybookEngine()
        pb = pe.create_from_execution(
            name="Research",
            goal_description="Research AI SDK competitors",
            domain="research",
            steps=[{"action": "web_search", "specialist": "researcher"}],
        )
        assert pb.success_rate == 1.0

        matched = pe.match("Research competitors in memory SDK", domain="research")
        assert matched is not None
        assert matched.id == pb.id

        assert pe.match("Buy groceries") is None

    def test_record_usage(self):
        from remy.core_v3.improvement.playbook_engine import PlaybookEngine

        pe = PlaybookEngine()
        pb = pe.create_from_execution(
            name="Test", goal_description="Test goal",
            domain="test", steps=[{"action": "x"}],
        )
        pe.record_usage(pb.id, success=True)
        pe.record_usage(pb.id, success=False)
        assert pb.times_used == 3
        assert pb.times_succeeded == 2


class TestLearningRuntime:
    def test_observe_cycle_creates_generic_playbook(self):
        from remy.core_v3.agents.specialist_registry import SpecialistRegistry
        from remy.core_v3.evaluation.evaluation_engine import EvalResult, EvalVerdict
        from remy.core_v3.execution.execution_runtime import ExecutionResult, ExecutionStatus
        from remy.core_v3.improvement.outcome_learner import OutcomeLearner
        from remy.core_v3.improvement.playbook_engine import PlaybookEngine
        from remy.core_v3.missions.mission_models import Mission, Goal, Task
        from remy.core_v3.planning.plan_models import PlanStep
        from remy.core_v3.runtime.learning_runtime import LearningRuntime

        runtime = LearningRuntime(learner=OutcomeLearner(), playbooks=PlaybookEngine())
        specialist = SpecialistRegistry().get("executor")
        mission = Mission(id="m1", description="Browse and verify signup flow")
        goal = Goal(id="g1", description="Verify signup")
        task = Task(id="t1", action="Open the signup page")
        step = PlanStep(id="s1", instruction="Open the signup page")
        exec_result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            session_log=[{"tool": "browse_page"}],
            duration_ms=120,
            cost_usd=0.01,
        )
        eval_result = EvalResult(
            verdict=EvalVerdict.SUCCESS,
            confidence=0.8,
            reason="Verified page load",
        )

        runtime.observe_cycle(
            mission=mission,
            goal=goal,
            task=task,
            step=step,
            specialist=specialist,
            exec_result=exec_result,
            eval_result=eval_result,
            decision="execute_step",
        )

        assert runtime.learner.summary()["outcomes_observed"] == 1
        playbooks = runtime.playbooks.list_playbooks(domain="execution")
        assert len(playbooks) == 1
        assert playbooks[0].steps[0].specialist == "executor"

    def test_observe_cycle_learns_failure_patterns(self):
        from remy.core_v3.agents.specialist_registry import SpecialistRegistry
        from remy.core_v3.evaluation.evaluation_engine import BlockerType, EvalResult, EvalVerdict
        from remy.core_v3.execution.execution_runtime import ExecutionResult, ExecutionStatus
        from remy.core_v3.improvement.outcome_learner import OutcomeLearner
        from remy.core_v3.improvement.playbook_engine import PlaybookEngine
        from remy.core_v3.missions.mission_models import Mission
        from remy.core_v3.planning.plan_models import PlanStep
        from remy.core_v3.runtime.learning_runtime import LearningRuntime

        learner = OutcomeLearner(min_evidence=2)
        learner.store_insights_to_memory = lambda: None
        runtime = LearningRuntime(learner=learner, playbooks=PlaybookEngine())
        specialist = SpecialistRegistry().get("executor")
        mission = Mission(id="m1", description="Register account")
        step = PlanStep(id="s1", instruction="Submit signup form")

        for _ in range(2):
            runtime.observe_cycle(
                mission=mission,
                goal=None,
                task=None,
                step=step,
                specialist=specialist,
                exec_result=ExecutionResult(status=ExecutionStatus.BLOCKED, duration_ms=50),
                eval_result=EvalResult(
                    verdict=EvalVerdict.BLOCKED,
                    confidence=0.9,
                    reason="Captcha blocked signup",
                    blocker_type=BlockerType.CAPTCHA,
                ),
                decision="pause",
            )

        failure_insights = runtime.learner.get_insights("failure_pattern")
        assert any("captcha" in insight.description.lower() for insight in failure_insights)

    def test_observe_cycle_passes_factuality_signal(self):
        from remy.core_v3.agents.specialist_registry import SpecialistRegistry
        from remy.core_v3.evaluation.evaluation_engine import EvalResult, EvalVerdict
        from remy.core_v3.execution.execution_runtime import ExecutionResult, ExecutionStatus
        from remy.core_v3.improvement.outcome_learner import OutcomeLearner
        from remy.core_v3.improvement.playbook_engine import PlaybookEngine
        from remy.core_v3.missions.mission_models import Mission
        from remy.core_v3.planning.plan_models import PlanStep
        from remy.core_v3.runtime.learning_runtime import LearningRuntime

        learner = OutcomeLearner(min_evidence=2)
        learner.store_insights_to_memory = lambda: None
        runtime = LearningRuntime(learner=learner, playbooks=PlaybookEngine())
        specialist = SpecialistRegistry().get("researcher")
        mission = Mission(id="m1", description="Research")
        step = PlanStep(id="s1", instruction="Inspect repo")

        for _ in range(2):
            runtime.observe_cycle(
                mission=mission,
                goal=None,
                task=None,
                step=step,
                specialist=specialist,
                exec_result=ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    duration_ms=50,
                    unsupported_observed_claims=1,
                ),
                eval_result=EvalResult(
                    verdict=EvalVerdict.SUCCESS,
                    confidence=0.6,
                    reason="ok",
                ),
                decision="continue",
            )

        quality_insights = runtime.learner.get_insights("quality_pattern")
        assert any("unsupported observed claims" in insight.description.lower() for insight in quality_insights)


class TestMemoryRuntime:
    def test_build_cycle_context_includes_playbook_hint(self):
        from remy.core_v3.improvement.playbook_engine import PlaybookEngine
        from remy.core_v3.missions.mission_models import Goal, Mission, MissionMode
        from remy.core_v3.runtime.memory_runtime import MemoryRuntime

        class StubMemory:
            def recall(self, query, tags=None, limit=20):
                return []

        playbooks = PlaybookEngine()
        playbooks.create_from_execution(
            name="Research competitors",
            goal_description="Research AI SDK competitors",
            domain="research",
            steps=[{"action": "Search top competitors", "specialist": "researcher"}],
        )
        runtime = MemoryRuntime(memory=StubMemory(), playbooks=playbooks)
        mission = Mission(description="Research AI SDK competitors", mode=MissionMode.DEEP_RESEARCH)
        goal = Goal(description="Research AI SDK competitors")

        context = runtime.build_cycle_context(mission, goal=goal)

        assert len(context) == 1
        assert context[0]["type"] == "playbook"
        assert "Research competitors" in context[0]["content"]


class TestPersistenceRuntime:
    def test_store_goal_outcome_dedupes(self):
        from remy.core_v3.missions.mission_models import Goal
        from remy.core_v3.runtime.persistence_runtime import PersistenceRuntime

        class StubMissionPersistence:
            def save(self, missions, goals, tasks):
                return None

        class StubPlanPersistence:
            def save(self, plan):
                return None

        class StubMemory:
            def __init__(self):
                self.calls = []

            def store(self, content, tags, metadata=None, memory_class=None, deduplicate=False):
                self.calls.append((content, tags, metadata, memory_class, deduplicate))
                return "rec1"

        memory = StubMemory()
        runtime = PersistenceRuntime(
            mission_persistence=StubMissionPersistence(),
            plan_persistence=StubPlanPersistence(),
            memory=memory,
        )
        goal = Goal(id="g1", description="Verify signup flow", mission_id="m1")

        runtime.store_goal_outcome(goal, success=True, summary="Verified signup flow")
        runtime.store_goal_outcome(goal, success=True, summary="Verified signup flow")

        # In-memory guard collapses the second identical write within a process.
        assert len(memory.calls) == 1
        # Append-log outcome must also opt into backend dedup, so identical
        # repeats across process restarts (when the guard set is empty) collapse.
        assert memory.calls[0][4] is True

    def test_research_context_uses_persistence_runtime(self):
        from remy.core_v3.research.research_context import ResearchContextRuntime

        class StubPersistence:
            def __init__(self):
                self.project_ids = []

            def store_research_summary(self, project):
                self.project_ids.append(project.id)

        class StubProject:
            id = "rp1"
            synthesis = object()

        persistence = StubPersistence()
        ctx = ResearchContextRuntime(persistence_runtime=persistence)
        ctx.store_project_summary(StubProject())

        assert persistence.project_ids == ["rp1"]


class TestRecordingRuntime:
    def test_record_cycle_writes_recorder_and_audit(self):
        import tempfile
        from remy.core_v3.agents.specialist_registry import SpecialistRegistry
        from remy.core_v3.evaluation.evaluation_engine import EvalResult, EvalVerdict
        from remy.core_v3.execution.cycle_recorder import CycleRecorder
        from remy.core_v3.execution.execution_runtime import ExecutionResult, ExecutionStatus
        from remy.core_v3.governance.audit_engine import AuditEngine
        from remy.core_v3.missions.mission_models import Goal, Mission, Task
        from remy.core_v3.planning.plan_models import Plan, PlanStep
        from remy.core_v3.runtime.recording_runtime import RecordingRuntime

        recorder = CycleRecorder()
        audit = AuditEngine(log_path=tempfile.mktemp(suffix=".jsonl"))
        runtime = RecordingRuntime(recorder=recorder, audit=audit)
        specialist = SpecialistRegistry().get("executor")
        mission = Mission(id="m1", description="Verify signup flow")
        goal = Goal(id="g1", description="Verify signup flow", mission_id="m1")
        task = Task(id="t1", action="Open signup page", mission_id="m1", goal_id="g1")
        step = PlanStep(id="s1", instruction="Open signup page")
        plan = Plan(id="p1", mission_id="m1", steps=[step])

        runtime.record_cycle(
            cycle_num=1,
            mission=mission,
            goal=goal,
            task=task,
            plan=plan,
            step=step,
            specialist=specialist,
            exec_result=ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                duration_ms=80,
                tokens_used=12,
                cost_usd=0.01,
            ),
            eval_result=EvalResult(
                verdict=EvalVerdict.SUCCESS,
                confidence=0.8,
                reason="Verified page load",
                should_continue=False,
            ),
            decision="complete",
            memory_assisted=True,
            duration_ms=80,
        )

        assert recorder.stats()["cycles"] == 1
        assert audit.recent(1)[0].event_type == "cycle_completed"


class TestSpecialistRuntime:
    def test_resolve_prefers_task_specialist(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Goal, Mission, Task
        from remy.core_v3.planning.plan_models import PlanStep

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        goal = Goal(description="Research market shifts", mission_id=mission.id)
        task = Task(
            action="Browse to the signup page",
            mission_id=mission.id,
            goal_id=goal.id,
        )
        step = PlanStep(instruction="Analyze the result", specialist="analyst")

        specialist = chief.specialist_runtime.resolve(
            step=step,
            task=task,
            mission=mission,
            goal=goal,
        )
        assert specialist.id == "executor"

    def test_sensitive_work_prefers_higher_quality_specialist(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.evaluation.evaluation_engine import EvalResult, EvalVerdict
        from remy.core_v3.missions.mission_models import Goal, Mission, MissionMode
        from remy.core_v3.planning.plan_models import PlanStep

        chief = ChiefAgent()
        chief.evaluator._record_outcome("g1", "researcher", EvalResult(verdict=EvalVerdict.SUCCESS))
        chief.evaluator._record_outcome("g2", "researcher", EvalResult(verdict=EvalVerdict.SUCCESS))
        chief.evaluator._record_outcome(
            "g3",
            "analyst",
            EvalResult(verdict=EvalVerdict.SUCCESS, unsupported_observed_claims=3, factuality_penalty=0.3),
        )

        mission = chief.accept_mission(Mission(description="Research competitor claims", mode=MissionMode.DEEP_RESEARCH))
        goal = Goal(description="Verify profile facts", mission_id=mission.id)
        step = PlanStep(instruction="Research and verify profile facts from the website")

        specialist = chief.specialist_runtime.resolve(
            step=step,
            task=None,
            mission=mission,
            goal=goal,
        )
        assert specialist.id == "researcher"
        resolution = chief.specialist_runtime.last_resolution()
        assert resolution["specialist_id"] == "researcher"
        assert resolution["sensitive"] is True
        assert resolution["reason"].startswith("sensitive_quality_tiebreak:")

    def test_degraded_preferred_specialist_is_rerouted_to_better_candidate(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.evaluation.evaluation_engine import EvalResult, EvalVerdict
        from remy.core_v3.missions.mission_models import Goal, Mission, Task
        from remy.core_v3.planning.plan_models import PlanStep

        chief = ChiefAgent()
        chief.evaluator._record_outcome(
            "e1",
            "executor",
            EvalResult(verdict=EvalVerdict.SUCCESS, unsupported_observed_claims=3, factuality_penalty=0.35),
        )
        chief.evaluator._record_outcome("r1", "researcher", EvalResult(verdict=EvalVerdict.SUCCESS))
        chief.evaluator._record_outcome("r2", "researcher", EvalResult(verdict=EvalVerdict.SUCCESS))

        mission = chief.accept_mission(Mission(description="Investigate signup flow with verification"))
        goal = Goal(description="Verify signup evidence trail", mission_id=mission.id)
        task = Task(
            action="Browse to the signup page",
            mission_id=mission.id,
            goal_id=goal.id,
        )
        step = PlanStep(instruction="Research and verify the signup evidence trail")

        specialist = chief.specialist_runtime.resolve(
            step=step,
            task=task,
            mission=mission,
            goal=goal,
        )

        assert specialist.id == "researcher"
        resolution = chief.specialist_runtime.last_resolution()
        assert resolution["reason"].startswith("routing_pressure_override:task_specialist:")
        assert resolution["specialist_id"] == "researcher"

    def test_non_degraded_preferred_specialist_is_kept(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.evaluation.evaluation_engine import EvalResult, EvalVerdict
        from remy.core_v3.missions.mission_models import Goal, Mission, Task
        from remy.core_v3.planning.plan_models import PlanStep

        chief = ChiefAgent()
        chief.evaluator._record_outcome("e1", "executor", EvalResult(verdict=EvalVerdict.SUCCESS))
        chief.evaluator._record_outcome("e2", "executor", EvalResult(verdict=EvalVerdict.SUCCESS))
        chief.evaluator._record_outcome("r1", "researcher", EvalResult(verdict=EvalVerdict.SUCCESS))

        mission = chief.accept_mission(Mission(description="Browse the signup flow"))
        goal = Goal(description="Check signup page", mission_id=mission.id)
        task = Task(
            action="Browse to the signup page",
            mission_id=mission.id,
            goal_id=goal.id,
        )
        step = PlanStep(instruction="Analyze the result")

        specialist = chief.specialist_runtime.resolve(
            step=step,
            task=task,
            mission=mission,
            goal=goal,
        )

        assert specialist.id == "executor"
        resolution = chief.specialist_runtime.last_resolution()
        assert resolution["reason"] == "task_specialist"

    def test_routing_policy_memory_can_reroute_preferred_specialist(self, monkeypatch):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Goal, Mission, Task
        from remy.core_v3.planning.plan_models import PlanStep

        chief = ChiefAgent()

        def routing_policy_hint(specialist_id):
            if specialist_id == "executor":
                return {
                    "hint": "avoid",
                    "reason": "executor produced repeated failed routing outcomes",
                }
            if specialist_id == "researcher":
                return {
                    "hint": "prefer",
                    "reason": "researcher has supported this class of work",
                }
            return {"hint": ""}

        monkeypatch.setattr(chief.recorder, "routing_policy_hint", routing_policy_hint)

        mission = chief.accept_mission(Mission(description="Research competitor claims"))
        goal = Goal(description="Verify signup evidence trail", mission_id=mission.id)
        task = Task(
            action="Browse to the signup page",
            mission_id=mission.id,
            goal_id=goal.id,
        )
        step = PlanStep(instruction="Research and verify the signup evidence trail")

        specialist = chief.specialist_runtime.resolve(
            step=step,
            task=task,
            mission=mission,
            goal=goal,
        )

        assert specialist.id == "researcher"
        resolution = chief.specialist_runtime.last_resolution()
        assert resolution["reason"].startswith("routing_pressure_override:task_specialist:")
        assert resolution["routing_policy"] == "prefer"


class TestSpecialistInferenceRuntime:
    def test_infers_research_and_execution_specialists(self):
        from remy.core_v3.runtime.specialist_inference_runtime import SpecialistInferenceRuntime

        runtime = SpecialistInferenceRuntime()
        assert runtime.infer("Research the market shifts") == "researcher"
        assert runtime.infer("Browse and register the account") == "executor"
        assert runtime.infer("Compare the outputs") == "analyst"


class TestGoalContextRuntime:
    def test_builds_v2_goal_dict(self):
        from remy.core_v3.missions.mission_models import Mission
        from remy.core_v3.planning.plan_models import PlanStep
        from remy.core_v3.runtime.goal_context_runtime import GoalContextRuntime

        runtime = GoalContextRuntime()
        mission = Mission(description="Research market shifts", priority=2)
        step = PlanStep(instruction="Collect sources", specialist="researcher")

        goal_dict = runtime.build_goal_dict(step=step, mission=mission)
        assert goal_dict["content"] == "Collect sources"
        assert goal_dict["metadata"]["mission_id"] == mission.id
        assert "v3_step" in goal_dict["tags"]


class TestPlanBuilderRuntime:
    def test_builds_steps_from_tasks(self):
        from remy.core_v3.missions.mission_models import Task
        from remy.core_v3.runtime.plan_builder_runtime import PlanBuilderRuntime
        from remy.core_v3.runtime.specialist_inference_runtime import SpecialistInferenceRuntime

        runtime = PlanBuilderRuntime(SpecialistInferenceRuntime())
        steps = runtime.steps_from_tasks(
            [Task(id="t1", action="Research competitors", done_when="Find 5 sources")]
        )

        assert steps[0].id == "step_t1"
        assert steps[0].specialist == "researcher"
        assert steps[0].instruction == "Research competitors"


class TestCycleExecutionRuntime:
    async def test_chief_run_cycle_delegates_to_cycle_execution_runtime(self, monkeypatch):
        from remy.core_v3.agents.chief_agent import ChiefAgent, ChiefDecision, CycleResult
        from remy.core_v3.missions.mission_models import Mission

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))

        async def fake_run(bound_chief, *, mission, cycle_num):
            assert bound_chief is chief
            assert cycle_num == 1
            return CycleResult(mission_id=mission.id, decision=ChiefDecision.COMPLETE)

        monkeypatch.setattr(chief.cycle_execution_runtime, "run", fake_run)
        result = await chief.run_cycle(mission)

        assert result.mission_id == mission.id
        assert result.decision == ChiefDecision.COMPLETE


class TestProjectionRuntime:
    def test_mission_summary_builds_read_model(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Mission, Task

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        task = Task(action="Research competitor moves", mission_id=mission.id)
        chief.add_task(task)

        summary = chief.projection_runtime.mission_summary(mission)
        assert summary["id"] == mission.id
        assert summary["current_task"]["id"] == task.id


class TestRuntimeContainer:
    def test_container_builds_dict_view(self):
        from remy.core_v3.runtime.runtime_container import RuntimeContainer

        container = RuntimeContainer.build()
        runtime = container.as_dict()

        assert runtime["chief"] is container.chief
        assert runtime["loop"] is container.loop
        assert runtime["telemetry"] is container.telemetry

    def test_direct_chief_bootstraps_default_runtimes(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent

        chief = ChiefAgent()

        assert chief.cycle_execution_runtime is not None
        assert chief.specialist_inference_runtime is not None
        assert chief.goal_context_runtime is not None
        assert chief.plan_builder_runtime is not None

    def test_chief_can_skip_default_runtime_bootstrap(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent

        chief = ChiefAgent(bootstrap_defaults=False)

        assert chief.cycle_execution_runtime is None
        assert chief.specialist_inference_runtime is None
        assert chief.goal_context_runtime is None
        assert chief.plan_builder_runtime is None


# ---------------------------------------------------------------------------
# Phase 3: Agent routing
# ---------------------------------------------------------------------------

class TestAgentRouting:
    def test_researcher_can_handle(self):
        from remy.core_v3.agents.researcher import ResearcherAgent
        from remy.core_v3.agents.base_agent import AgentContext

        agent = ResearcherAgent()
        assert agent.can_handle(AgentContext(instruction="Research competitors"))
        assert not agent.can_handle(AgentContext(instruction="Summarize the findings"))

    def test_analyst_can_handle(self):
        from remy.core_v3.agents.analyst_agent import AnalystAgent
        from remy.core_v3.agents.base_agent import AgentContext

        agent = AnalystAgent()
        assert agent.can_handle(AgentContext(instruction="Analyze market trends"))
        assert not agent.can_handle(AgentContext(instruction="Research competitors"))

    def test_executor_can_handle(self):
        from remy.core_v3.agents.executor_agent import ExecutorAgent
        from remy.core_v3.agents.base_agent import AgentContext

        agent = ExecutorAgent()
        assert agent.can_handle(AgentContext(instruction="Browse to the signup page"))
        assert not agent.can_handle(AgentContext(instruction="Research competitors"))


# ---------------------------------------------------------------------------
# Phase 8: Telemetry
# ---------------------------------------------------------------------------

class TestTelemetry:
    def test_unbound(self):
        from remy.core_v3.observability.telemetry import Telemetry
        t = Telemetry()
        assert t.dashboard()["error"] == "Chief Agent not bound"
        assert t.health_check()["status"] == "unbound"

    def test_dashboard_reads_recorder(self):
        """Verify telemetry reads chief.recorder, not chief.cycle_recorder."""
        from remy.core_v3.observability.telemetry import Telemetry
        from remy.core_v3.execution.cycle_recorder import CycleRecorder, CycleRecord
        from remy.core_v3.governance.budget_engine import BudgetEngine
        from remy.core_v3.governance.policy_engine import PolicyEngine
        from remy.core_v3.governance.approval_engine import ApprovalEngine
        from remy.core_v3.governance.audit_engine import AuditEngine
        from remy.core_v3.evaluation.evaluation_engine import EvaluationEngine
        from remy.core_v3.agents.specialist_registry import SpecialistRegistry

        # Minimal chief-like object
        class FakeChief:
            budget = BudgetEngine()
            policy = PolicyEngine()
            approval = ApprovalEngine()
            audit = AuditEngine(log_path=tempfile.mktemp(suffix=".jsonl"))
            evaluator = EvaluationEngine()
            registry = SpecialistRegistry()
            recorder = CycleRecorder()

            def active_missions(self):
                return []
            def all_missions(self):
                return []

        chief = FakeChief()
        chief.recorder._records.append(CycleRecord(
            cycle_num=1, status="success", goal_description="test",
        ))

        t = Telemetry(chief)
        d = t.dashboard()

        # Execution data should be present (not missing)
        assert "execution" in d
        assert d["execution"]["cycles"] == 1

        # Health check should read recorder
        h = t.health_check()
        assert h["uptime_cycles"] == 1

        # Clean up
        if os.path.exists(chief.audit._path):
            os.unlink(chief.audit._path)

    def test_dashboard_exposes_active_task(self):
        from remy.core_v3.observability.telemetry import Telemetry
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Mission, MissionStatus, Task

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        mission.status = MissionStatus.PLANNING
        chief.activate_mission(mission)
        task = Task(action="Research competitor moves", mission_id=mission.id)
        chief.add_task(task)

        dashboard = Telemetry(chief).dashboard()
        assert dashboard["active_task"]["id"] == task.id


class TestChiefAgentLifecycle:
    def test_activate_mission_from_intake(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Mission, MissionStatus

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        assert mission.status == MissionStatus.INTAKE

        chief.activate_mission(mission)
        assert mission.status == MissionStatus.ACTIVE

    def test_pending_task_can_be_promoted_before_running(self):
        from remy.core_v3.missions.mission_models import Task, TaskStatus
        from remy.core_v3.runtime.state_machine import transition_task

        task = Task(action="Check GitHub stars")
        transition_task(task, TaskStatus.ACTIVE, "Selected for execution")
        transition_task(task, TaskStatus.RUNNING, "Delegated to specialist")
        assert task.status == TaskStatus.RUNNING


class TestOutcomeRuntime:
    def test_success_marks_task_complete_and_mission_complete(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.evaluation.evaluation_engine import EvalResult, EvalVerdict
        from remy.core_v3.execution.execution_runtime import ExecutionResult, ExecutionStatus
        from remy.core_v3.missions.mission_models import Goal, Mission, MissionStatus, Task, TaskStatus
        from remy.core_v3.planning.plan_models import StepStatus

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        chief.activate_mission(mission)
        goal = Goal(description="Research market shifts", mission_id=mission.id)
        chief.add_goal(goal)
        task = Task(action="Collect sources", mission_id=mission.id, goal_id=goal.id)
        chief.add_task(task)
        plan = chief.create_plan_from_tasks(mission)
        plan.risk_level = "medium"
        step = plan.steps[0]
        task.status = TaskStatus.RUNNING
        step.status = StepStatus.RUNNING

        outcome = chief.outcome_runtime.apply(
            mission=mission,
            goal=goal,
            task=task,
            plan=plan,
            step=step,
            exec_result=ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                evidence={"findings": [{"content": "a"}]},
                cost_usd=0.02,
                tokens_used=100,
            ),
            eval_result=EvalResult(
                verdict=EvalVerdict.SUCCESS,
                reason="Evidence verified",
                should_continue=False,
            ),
        )

        assert outcome.decision == "complete"
        assert mission.status == MissionStatus.COMPLETED

    def test_partial_keeps_task_pending(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.evaluation.evaluation_engine import EvalResult, EvalVerdict
        from remy.core_v3.execution.execution_runtime import ExecutionResult, ExecutionStatus
        from remy.core_v3.missions.mission_models import Goal, Mission, Task, TaskStatus
        from remy.core_v3.planning.plan_models import StepStatus

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        chief.activate_mission(mission)
        goal = Goal(description="Research market shifts", mission_id=mission.id)
        chief.add_goal(goal)
        task = Task(action="Collect sources", mission_id=mission.id, goal_id=goal.id)
        chief.add_task(task)
        plan = chief.create_plan_from_tasks(mission)
        plan.risk_level = "high"
        step = plan.steps[0]
        task.status = TaskStatus.RUNNING
        step.status = StepStatus.RUNNING

        outcome = chief.outcome_runtime.apply(
            mission=mission,
            goal=goal,
            task=task,
            plan=plan,
            step=step,
            exec_result=ExecutionResult(
                status=ExecutionStatus.PARTIAL,
                evidence={"findings": [{"content": "a"}]},
                cost_usd=0.01,
                tokens_used=50,
            ),
            eval_result=EvalResult(
                verdict=EvalVerdict.PARTIAL,
                reason="Need more sources",
                should_continue=True,
            ),
        )

        assert outcome.decision == "execute_step"
        assert task.status == TaskStatus.PENDING
        assert step.status == StepStatus.PENDING

    def test_wait_marks_task_waiting(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.evaluation.evaluation_engine import BlockerType, EvalResult, EvalVerdict
        from remy.core_v3.execution.execution_runtime import ExecutionResult, ExecutionStatus
        from remy.core_v3.missions.mission_models import Goal, Mission, Task, TaskStatus
        from remy.core_v3.planning.plan_models import StepStatus

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        chief.activate_mission(mission)
        goal = Goal(description="Research market shifts", mission_id=mission.id)
        chief.add_goal(goal)
        task = Task(action="Collect sources", mission_id=mission.id, goal_id=goal.id)
        chief.add_task(task)
        plan = chief.create_plan_from_tasks(mission)
        step = plan.steps[0]
        task.status = TaskStatus.RUNNING
        step.status = StepStatus.RUNNING

        outcome = chief.outcome_runtime.apply(
            mission=mission,
            goal=goal,
            task=task,
            plan=plan,
            step=step,
            exec_result=ExecutionResult(status=ExecutionStatus.FAILURE),
            eval_result=EvalResult(
                verdict=EvalVerdict.FAILURE,
                reason="429 rate limit",
                blocker_type=BlockerType.RATE_LIMIT,
            ),
        )

        assert outcome.decision == "pause"
        assert task.status == TaskStatus.WAITING
        assert "rate limit" in task.waiting_reason.lower()


class TestExecutionGateRuntime:
    def test_policy_deny_aborts_before_delegation(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Goal, Mission, Task, TaskStatus
        from remy.core_v3.governance.policy_engine import PolicyDecision, PolicyRule

        chief = ChiefAgent()
        chief.policy.add_rule(PolicyRule(
            id="deny_delete",
            action_pattern="Delete*",
            decision=PolicyDecision.DENY,
        ), priority=0)
        mission = chief.accept_mission(Mission(description="Delete everything now"))
        chief.activate_mission(mission)
        goal = Goal(description="Delete everything", mission_id=mission.id)
        chief.add_goal(goal)
        task = Task(action="Delete everything", mission_id=mission.id, goal_id=goal.id)
        chief.add_task(task)
        plan = chief.create_plan_from_tasks(mission)
        step = plan.steps[0]

        gate = chief.execution_gate.prepare(
            mission=mission,
            goal=goal,
            task=task,
            plan=plan,
            step=step,
            memory_context=[],
        )

        assert not gate.proceed
        assert gate.decision == "abort"
        assert task.status == TaskStatus.ABORTED

    def test_policy_approval_pauses_before_delegation(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Goal, Mission, Task, TaskStatus

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Transfer funds"))
        chief.activate_mission(mission)
        goal = Goal(description="Transfer funds", mission_id=mission.id)
        chief.add_goal(goal)
        task = Task(action="financial_transfer", mission_id=mission.id, goal_id=goal.id)
        chief.add_task(task)
        plan = chief.create_plan_from_tasks(mission)
        step = plan.steps[0]

        gate = chief.execution_gate.prepare(
            mission=mission,
            goal=goal,
            task=task,
            plan=plan,
            step=step,
            memory_context=[],
        )

        assert not gate.proceed
        assert gate.decision == "pause"
        assert task.status == TaskStatus.BLOCKED_APPROVAL

    def test_consequence_scar_blocks_refuted_planned_action_before_delegation(self, monkeypatch):
        from types import SimpleNamespace

        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Goal, Mission, Task, TaskStatus
        from remy.core_v3.planning.plan_models import StepStatus

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Medication follow-up"))
        chief.activate_mission(mission)
        goal = Goal(description="schedule patient medication reminder", mission_id=mission.id)
        chief.add_goal(goal)
        task = Task(action="double the dose automatically", mission_id=mission.id, goal_id=goal.id)
        chief.add_task(task)
        plan = chief.create_plan_from_tasks(mission)
        step = plan.steps[0]
        seen = {}

        def scar_check(situation, action):
            seen["situation"] = situation
            seen["action"] = action
            return SimpleNamespace(
                is_refuted=True,
                refutes=1,
                supports=20,
                scar=True,
            )

        monkeypatch.setattr(chief.recorder, "scar_check", scar_check)

        gate = chief.execution_gate.prepare(
            mission=mission,
            goal=goal,
            task=task,
            plan=plan,
            step=step,
            memory_context=[],
        )

        assert not gate.proceed
        assert gate.decision == "pause"
        assert "Consequence scar blocks execution" in gate.reason
        assert seen == {
            "situation": "schedule patient medication reminder",
            "action": "double the dose automatically",
        }
        assert task.status == TaskStatus.BLOCKED
        assert step.status == StepStatus.BLOCKED

    def test_policy_hint_requires_evidence_is_injected_into_execution_context(self, monkeypatch):
        from types import SimpleNamespace

        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Goal, Mission, Task, TaskStatus
        from remy.core_v3.planning.plan_models import StepStatus

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        chief.activate_mission(mission)
        goal = Goal(description="Research market shifts", mission_id=mission.id)
        chief.add_goal(goal)
        task = Task(action="Research competitor moves", mission_id=mission.id, goal_id=goal.id)
        chief.add_task(task)
        plan = chief.create_plan_from_tasks(mission)
        step = plan.steps[0]

        monkeypatch.setattr(
            chief.recorder,
            "scar_check",
            lambda *_args, **_kwargs: SimpleNamespace(is_refuted=False),
        )
        monkeypatch.setattr(
            chief.recorder,
            "policy_hint",
            lambda situation, action: {
                "type": "policy_hint",
                "hint": "requires_evidence",
                "situation": situation,
                "action": action,
                "reason": "factuality scar requires fresh evidence",
                "requires_evidence": True,
                "should_block": False,
            },
        )

        gate = chief.execution_gate.prepare(
            mission=mission,
            goal=goal,
            task=task,
            plan=plan,
            step=step,
            memory_context=[],
        )

        assert gate.proceed
        assert task.status == TaskStatus.RUNNING
        assert step.status == StepStatus.RUNNING
        assert gate.agent_ctx.policy_hints[0]["hint"] == "requires_evidence"
        assert "Runtime policy from lived consequence memory" in gate.agent_ctx.instruction
        assert "requires_evidence" in gate.agent_ctx.instruction
        debt_tasks = [
            item for item in chief._tasks.values()
            if item.id != task.id and "Verify evidence before relying on action" in item.action
        ]
        assert len(debt_tasks) == 1
        assert gate.agent_ctx.policy_hints[0]["evidence_debt_task_id"] == debt_tasks[0].id

    def test_gate_promotes_task_and_builds_context(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Goal, Mission, Task, TaskStatus
        from remy.core_v3.planning.plan_models import StepStatus

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        chief.activate_mission(mission)
        goal = Goal(description="Research market shifts", mission_id=mission.id)
        chief.add_goal(goal)
        task = Task(action="Research competitor moves", mission_id=mission.id, goal_id=goal.id)
        chief.add_task(task)
        plan = chief.create_plan_from_tasks(mission)
        step = plan.steps[0]

        gate = chief.execution_gate.prepare(
            mission=mission,
            goal=goal,
            task=task,
            plan=plan,
            step=step,
            memory_context=[{"content": "prior finding"}],
        )

        assert gate.proceed
        assert gate.specialist.id == "researcher"
        assert gate.agent_ctx is not None
        assert gate.agent_ctx.memory_context
        assert task.status == TaskStatus.RUNNING
        assert step.status == StepStatus.RUNNING

    def test_degraded_specialist_escalates_medium_risk_work_for_approval(self, monkeypatch):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Goal, Mission, RiskLevel, Task, TaskStatus

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Investigate counterparty profile", risk=RiskLevel.MEDIUM))
        chief.activate_mission(mission)
        goal = Goal(description="Investigate counterparty profile", mission_id=mission.id)
        chief.add_goal(goal)
        task = Task(action="Research counterparty profile", mission_id=mission.id, goal_id=goal.id)
        chief.add_task(task)
        plan = chief.create_plan_from_tasks(mission)
        plan.risk_level = "medium"
        step = plan.steps[0]

        monkeypatch.setattr(
            chief.execution_gate,
            "_resolve_specialist",
            lambda *_args: chief.registry.get("researcher"),
        )
        monkeypatch.setattr(
            chief.ops_query_runtime,
            "specialist_quality",
            lambda specialist_id: {
                "success_rate": 0.72,
                "quality_adjusted_success_rate": 0.49,
                "unsupported_claims": 3,
                "factuality_penalty": 0.23,
            },
        )

        gate = chief.execution_gate.prepare(
            mission=mission,
            goal=goal,
            task=task,
            plan=plan,
            step=step,
            memory_context=[],
        )

        pending = chief.approval.pending()
        assert not gate.proceed
        assert gate.decision == "pause"
        assert "routing pressure approval" in gate.reason.lower()
        assert task.status == TaskStatus.BLOCKED_APPROVAL
        assert len(pending) == 1
        assert pending[0].specialist == gate.specialist.id
        assert pending[0].risk_category == "medium"

    def test_healthy_specialist_does_not_trigger_extra_pressure_approval(self, monkeypatch):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Goal, Mission, RiskLevel, Task, TaskStatus
        from remy.core_v3.planning.plan_models import StepStatus

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Investigate market shifts", risk=RiskLevel.HIGH))
        chief.activate_mission(mission)
        goal = Goal(description="Investigate market shifts", mission_id=mission.id)
        chief.add_goal(goal)
        task = Task(action="Research market shifts", mission_id=mission.id, goal_id=goal.id)
        chief.add_task(task)
        plan = chief.create_plan_from_tasks(mission)
        plan.risk_level = "high"
        step = plan.steps[0]

        monkeypatch.setattr(
            chief.execution_gate,
            "_resolve_specialist",
            lambda *_args: chief.registry.get("researcher"),
        )
        monkeypatch.setattr(
            chief.ops_query_runtime,
            "specialist_quality",
            lambda specialist_id: {
                "success_rate": 0.81,
                "quality_adjusted_success_rate": 0.79,
                "unsupported_claims": 0,
                "factuality_penalty": 0.01,
            },
        )

        gate = chief.execution_gate.prepare(
            mission=mission,
            goal=goal,
            task=task,
            plan=plan,
            step=step,
            memory_context=[],
        )

        assert gate.proceed
        assert gate.decision == "execute_step"
        assert task.status == TaskStatus.RUNNING
        assert step.status == StepStatus.RUNNING
        assert chief.approval.pending() == []


class TestContextRuntime:
    def test_builds_agent_context_deterministically(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Goal, Mission, Task

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        goal = Goal(description="Research market shifts", mission_id=mission.id)
        chief.add_goal(goal)
        task = Task(action="Research competitor moves", mission_id=mission.id, goal_id=goal.id)
        chief.add_task(task)
        plan = chief.create_plan_from_tasks(mission)
        step = plan.steps[0]
        specialist = chief.specialist_runtime.resolve(step=step, task=task, mission=mission, goal=goal)

        ctx = chief.context_runtime.build(
            mission=mission,
            goal=goal,
            step=step,
            specialist=specialist,
            memory_context=[{"content": "prior finding"}],
        )

        assert ctx.instruction == step.instruction
        assert ctx.mission_id == mission.id
        assert ctx.goal_id == goal.id
        assert ctx.memory_context
        assert ctx.tools_allowed == specialist.tools


class TestTaskStateRuntime:
    def test_promotes_step_and_task_into_running(self):
        from remy.core_v3.missions.mission_models import Goal, Mission, Task, TaskStatus
        from remy.core_v3.planning.plan_models import StepStatus
        from remy.core_v3.runtime.task_state_runtime import TaskStateRuntime
        from remy.core_v3.agents.chief_agent import ChiefAgent

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        goal = Goal(description="Research market shifts", mission_id=mission.id)
        chief.add_goal(goal)
        task = Task(action="Research competitor moves", mission_id=mission.id, goal_id=goal.id)
        chief.add_task(task)
        plan = chief.create_plan_from_tasks(mission)
        step = plan.steps[0]

        TaskStateRuntime().promote_for_execution(step=step, task=task)
        assert step.status == StepStatus.RUNNING
        assert step.attempts == 1
        assert task.status == TaskStatus.RUNNING
        assert task.attempts == 1


class TestTaskDecisionRuntime:
    def test_denies_execution_and_blocks_for_approval(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Goal, Mission, Task, TaskStatus
        from remy.core_v3.planning.plan_models import StepStatus
        from remy.core_v3.runtime.task_decision_runtime import TaskDecisionRuntime

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        goal = Goal(description="Research market shifts", mission_id=mission.id)
        chief.add_goal(goal)
        task = Task(action="Research competitor moves", mission_id=mission.id, goal_id=goal.id)
        chief.add_task(task)
        plan = chief.create_plan_from_tasks(mission)
        step = plan.steps[0]

        runtime = TaskDecisionRuntime(goal_tracker=chief.goal_tracker)
        runtime.deny_execution(step=step, task=task, reason="policy denied")
        assert step.status == StepStatus.SKIPPED
        assert task.status == TaskStatus.ABORTED

        task.status = TaskStatus.PENDING
        runtime.block_for_approval(task=task, approval_id="approval_123")
        assert task.status == TaskStatus.BLOCKED_APPROVAL
        assert "approval_123" in task.blocker_reason


class TestStepStateRuntime:
    def test_marks_step_running_and_skipped(self):
        from remy.core_v3.planning.plan_models import PlanStep, StepStatus
        from remy.core_v3.runtime.step_state_runtime import StepStateRuntime

        step = PlanStep(instruction="Research competitor moves")
        runtime = StepStateRuntime()
        runtime.mark_running(step=step)
        assert step.status == StepStatus.RUNNING
        assert step.attempts == 1
        assert step.started_at > 0

        runtime.mark_skipped(step=step)
        assert step.status == StepStatus.SKIPPED


class TestPlanStateRuntime:
    def test_manages_step_completion_retry_failure_and_skip(self):
        from remy.core_v3.planning.plan_models import Plan, PlanStep, StepStatus
        from remy.core_v3.runtime.plan_state_runtime import PlanStateRuntime

        runtime = PlanStateRuntime()
        step = PlanStep(id="s1", instruction="Research competitor moves", status=StepStatus.RUNNING)
        plan = Plan(steps=[step])

        runtime.complete_step(plan=plan, step=step, result={"source_count": 3})
        assert step.status == StepStatus.COMPLETED
        assert step.result["source_count"] == 3
        assert "s1" in plan.completed_step_ids

        runtime.reset_step_for_retry(plan=plan, step=step)
        assert step.status == StepStatus.PENDING

        runtime.fail_step(plan=plan, step=step)
        assert step.status == StepStatus.FAILED
        assert "s1" in plan.failed_step_ids

        runtime.skip_step(plan=plan, step=step)
        assert step.status == StepStatus.SKIPPED


class TestTaskOutcomeRuntime:
    def test_applies_task_level_transitions(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Goal, Mission, Task, TaskStatus

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        goal = Goal(description="Research market shifts", mission_id=mission.id)
        chief.add_goal(goal)
        task = Task(action="Collect sources", mission_id=mission.id, goal_id=goal.id)
        chief.add_task(task)

        runtime = chief.task_outcome_runtime
        task.status = TaskStatus.RUNNING
        runtime.partial_continue(task=task)
        assert task.status == TaskStatus.PENDING

        task.status = TaskStatus.RUNNING
        runtime.wait(task=task, reason="rate_limit")
        assert task.status == TaskStatus.WAITING

        task.status = TaskStatus.RUNNING
        runtime.block_external(task=task, reason="captcha")
        assert task.status == TaskStatus.BLOCKED_EXTERNAL

        task.status = TaskStatus.RUNNING
        runtime.abort(task=task, reason="manual stop")
        assert task.status == TaskStatus.ABORTED


class TestGoalOutcomeRuntime:
    def test_applies_goal_completion_failure_and_task_done_completion(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Goal, Mission, Task, TaskStatus

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        goal = Goal(description="Research market shifts", mission_id=mission.id)
        chief.add_goal(goal)
        task = Task(action="Collect sources", mission_id=mission.id, goal_id=goal.id)
        chief.add_task(task)

        runtime = chief.goal_outcome_runtime
        runtime.complete(goal=goal, summary="Evidence verified")
        assert goal.status.value == "completed"

        goal.status = goal.status.ACTIVE
        runtime.fail(goal=goal, summary="No viable path")
        assert goal.status.value == "failed"

        goal.status = goal.status.ACTIVE
        task.status = TaskStatus.COMPLETED
        assert runtime.complete_if_all_tasks_done(goal=goal)
        assert goal.status.value == "completed"


class TestMissionOutcomeRuntime:
    def test_applies_mission_completion_and_escalation(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Mission, MissionStatus

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        chief.activate_mission(mission)
        plan = chief.create_plan(mission)

        runtime = chief.mission_outcome_runtime
        plan.steps.clear()
        assert runtime.should_complete(mission=mission, plan=plan)
        runtime.complete(mission=mission, reason="All steps done")
        assert mission.status == MissionStatus.COMPLETED

        mission.status = MissionStatus.ACTIVE
        runtime.escalate(mission=mission, reason="Needs human input")
        assert mission.status == MissionStatus.ESCALATED


class TestDecisionRuntime:
    def test_builds_success_partial_and_failure_decisions(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.evaluation.evaluation_engine import EvalResult, EvalVerdict
        from remy.core_v3.missions.mission_models import Goal, Mission, Task

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        chief.activate_mission(mission)
        goal = Goal(description="Research market shifts", mission_id=mission.id)
        chief.add_goal(goal)
        task = Task(action="Collect sources", mission_id=mission.id, goal_id=goal.id)
        chief.add_task(task)
        plan = chief.create_plan_from_tasks(mission)
        step = plan.steps[0]

        decision = chief.decision_runtime.decide(
            mission=mission,
            plan=plan,
            step=step,
            task=task,
            eval_result=EvalResult(
                verdict=EvalVerdict.PARTIAL,
                reason="Need more sources",
                should_continue=True,
            ),
        )
        assert decision.phase == "partial"

        decision = chief.decision_runtime.decide(
            mission=mission,
            plan=plan,
            step=step,
            task=task,
            eval_result=EvalResult(
                verdict=EvalVerdict.SUCCESS,
                reason="Evidence verified",
                should_continue=False,
            ),
        )
        assert decision.phase == "success"
        assert decision.task_completed

        decision = chief.decision_runtime.decide(
            mission=mission,
            plan=plan,
            step=step,
            task=task,
            eval_result=EvalResult(
                verdict=EvalVerdict.FAILURE,
                reason="No viable path",
            ),
        )
        assert decision.phase == "failure"
        assert bool(decision.replan_action)


class TestCompletionRuntime:
    def test_applies_success_and_partial_paths(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.evaluation.evaluation_engine import EvalResult, EvalVerdict
        from remy.core_v3.execution.execution_runtime import ExecutionResult, ExecutionStatus
        from remy.core_v3.missions.mission_models import Goal, Mission, MissionStatus, Task, TaskStatus

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        chief.activate_mission(mission)
        goal = Goal(description="Research market shifts", mission_id=mission.id)
        chief.add_goal(goal)
        task = Task(action="Collect sources", mission_id=mission.id, goal_id=goal.id)
        chief.add_task(task)
        plan = chief.create_plan_from_tasks(mission)
        step = plan.steps[0]
        task.status = TaskStatus.RUNNING

        app = type("App", (), {"decision": "pause", "reason": "", "next_action": ""})()
        chief.completion_runtime.apply_partial(
            plan=plan,
            task=task,
            step=step,
            decision=chief.decision_runtime.decide(
                mission=mission,
                plan=plan,
                step=step,
                task=task,
                eval_result=EvalResult(
                    verdict=EvalVerdict.PARTIAL,
                    reason="Need more sources",
                    should_continue=True,
                ),
            ),
            outcome=app,
        )
        assert app.decision == "execute_step"
        assert task.status == TaskStatus.PENDING

        task.status = TaskStatus.RUNNING
        app = type("App", (), {"decision": "pause", "reason": "", "next_action": ""})()
        chief.completion_runtime.apply_success(
            mission=mission,
            goal=goal,
            task=task,
            plan=plan,
            step=step,
            exec_result=ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                evidence={"findings": [{"content": "a"}]},
            ),
            eval_result=EvalResult(
                verdict=EvalVerdict.SUCCESS,
                reason="Evidence verified",
                should_continue=False,
            ),
            decision=chief.decision_runtime.decide(
                mission=mission,
                plan=plan,
                step=step,
                task=task,
                eval_result=EvalResult(
                    verdict=EvalVerdict.SUCCESS,
                    reason="Evidence verified",
                    should_continue=False,
                ),
            ),
            outcome=app,
        )
        assert app.decision == "complete"
        assert mission.status == MissionStatus.COMPLETED


class TestResultRuntime:
    def test_builds_normalized_outcome_results(self):
        from remy.core_v3.runtime.result_runtime import ResultRuntime

        runtime = ResultRuntime()
        assert runtime.initial("seed").reason == "seed"
        assert runtime.complete().decision == "complete"
        assert runtime.execute_step(reason="continue", next_action="next").next_action == "next"
        assert runtime.pause("wait").decision == "pause"
        assert runtime.escalate("human").decision == "escalate"
        assert runtime.abort("stop").decision == "abort"
        assert runtime.replan("retry").decision == "replan"


class TestCycleResultRuntime:
    def test_builds_and_updates_cycle_result_contract(self):
        from remy.core_v3.agents.chief_agent import CycleResult
        from remy.core_v3.runtime.cycle_result_runtime import CycleResultRuntime

        runtime = CycleResultRuntime(result_factory=CycleResult)
        result = runtime.initial(mission_id="m1")
        assert result.mission_id == "m1"

        prep = type("Prep", (), {"decision": "pause", "reason": "budget", "goal": None})()
        runtime.apply_cycle_prep(result, cycle_prep=prep)
        assert result.decision == "pause"
        assert result.reason == "budget"

        goal = type("Goal", (), {"id": "g1"})()
        runtime.apply_context(result, goal=goal, memory_context=[{"content": "x"}])
        assert result.goal_id == "g1"
        assert result.memory_context_used is True

        specialist = type("Specialist", (), {"id": "researcher"})()
        runtime.apply_specialist(result, specialist=specialist)
        assert result.specialist_used == "researcher"

        exec_result = type("Exec", (), {"cost_usd": 0.1, "tokens_used": 25})()
        runtime.apply_execution(result, exec_result=exec_result)
        assert result.cost_usd == 0.1
        assert result.tokens_used == 25

        eval_result = type("Eval", (), {"verdict": type("Verdict", (), {"value": "success"})()})()
        runtime.apply_evaluation(result, eval_result=eval_result, step_id="s1")
        assert result.eval_verdict == "success"
        assert result.step_executed == "s1"

        outcome = type("Outcome", (), {"decision": "complete", "reason": "done", "next_action": ""})()
        runtime.apply_outcome(result, outcome=outcome)
        assert result.decision == "complete"
        assert result.reason == "done"


class TestEvaluationRuntime:
    def test_selects_task_criteria_over_goal_and_calls_evaluator(self, monkeypatch):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.evaluation.evaluation_engine import EvalResult, EvalVerdict
        from remy.core_v3.execution.execution_runtime import ExecutionResult, ExecutionStatus
        from remy.core_v3.missions.mission_models import Goal, Mission, SuccessCriterion, Task

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        goal = Goal(
            description="Research market shifts",
            mission_id=mission.id,
            success_criteria=[SuccessCriterion(type="goal_check", description="goal")],
        )
        chief.add_goal(goal)
        task = Task(
            action="Collect sources",
            mission_id=mission.id,
            goal_id=goal.id,
            success_criteria=[SuccessCriterion(type="task_check", description="task")],
        )
        chief.add_task(task)

        exec_result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            session_log=[{"type": "tool", "tool": "web_search"}],
            evidence={"findings": [{"content": "a"}]},
        )
        criteria = chief.evaluation_runtime.criteria_dicts(goal=goal, task=task)
        assert criteria is not None
        assert criteria[0]["type"] == "task_check"

        captured = {}

        def fake_evaluate(execution_result, success_criteria=None, session_log=None, goal_id="", specialist=""):
            captured["execution_result"] = execution_result
            captured["success_criteria"] = success_criteria
            captured["session_log"] = session_log
            captured["goal_id"] = goal_id
            captured["specialist"] = specialist
            return EvalResult(verdict=EvalVerdict.SUCCESS, reason="ok")

        monkeypatch.setattr(chief.evaluation_runtime.evaluator, "evaluate", fake_evaluate)

        eval_result = chief.evaluation_runtime.evaluate(
            exec_result=exec_result,
            goal=goal,
            task=task,
            specialist_id="researcher",
        )
        assert captured["execution_result"] is exec_result
        assert captured["success_criteria"][0]["type"] == "task_check"
        assert captured["goal_id"] == goal.id
        assert captured["specialist"] == "researcher"
        assert eval_result.verdict == EvalVerdict.SUCCESS

    def test_passes_runtime_quality_signals_and_tolerates_legacy_signature(self, monkeypatch):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.execution.execution_runtime import ExecutionResult, ExecutionStatus
        from remy.core_v3.missions.mission_models import Goal, Mission, Task, TaskStatus

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Verify competitor claims"))
        goal = Goal(description="Verify competitor claims", mission_id=mission.id)
        chief.add_goal(goal)
        task = Task(
            action="Verify release notes",
            mission_id=mission.id,
            goal_id=goal.id,
            status=TaskStatus.BLOCKED_APPROVAL,
        )
        task.blocker_reason = "awaiting approval"
        chief.add_task(task)
        chief.approval.request_approval("publish", risk_category="high", cost_usd=0.1)
        chief.evaluator._failure_history.extend([
            {"goal_id": goal.id, "verdict": "failure"},
            {"goal_id": goal.id, "verdict": "blocked"},
        ])

        exec_result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            response="Reviewed the release page",
            session_log=[],
            had_external_evidence=True,
            evidence={"source": "missing direct link"},
        )

        captured = {}

        def fake_evaluate(execution_result, **kwargs):
            captured.update(kwargs)
            from remy.core_v3.evaluation.evaluation_engine import EvalResult, EvalVerdict
            return EvalResult(verdict=EvalVerdict.SUCCESS, confidence=0.8, reason="ok")

        monkeypatch.setattr(chief.evaluation_runtime.evaluator, "evaluate", fake_evaluate)

        chief.evaluation_runtime.evaluate(
            exec_result=exec_result,
            goal=goal,
            task=task,
            specialist_id="researcher",
        )
        assert captured["unsupported_observed_claims"] == 0
        assert captured["blocker_history_summary"]["recent_failures"] == 2
        assert captured["approval_state"]["pending_approvals"] >= 1
        assert captured["source_link_completeness"] == 0.0
        assert "status" in captured["budget_pressure_snapshot"]

        def legacy_evaluate(execution_result, success_criteria=None, session_log=None, goal_id="", specialist=""):
            from remy.core_v3.evaluation.evaluation_engine import EvalResult, EvalVerdict
            return EvalResult(verdict=EvalVerdict.SUCCESS, reason="legacy ok")

        monkeypatch.setattr(chief.evaluation_runtime.evaluator, "evaluate", legacy_evaluate)
        result = chief.evaluation_runtime.evaluate(
            exec_result=exec_result,
            goal=goal,
            task=task,
            specialist_id="researcher",
        )
        assert result.reason == "legacy ok"


class TestExecutionRuntime:
    async def test_delegates_and_normalizes_agent_output(self, monkeypatch):
        from remy.core_v3.agents.base_agent import AgentContext, AgentOutput
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.execution.execution_runtime import ExecutionStatus

        chief = ChiefAgent()

        async def fake_delegate(ctx):
            return AgentOutput(
                status="success",
                response=f"done:{ctx.instruction}",
                tool_calls=2,
                tokens_used=42,
                cost_usd=0.12,
                evidence={"findings": [{"content": "source"}]},
            )

        monkeypatch.setattr(chief.execution_runtime.delegation, "delegate", fake_delegate)
        exec_result = await chief.execution_runtime.execute(
            agent_ctx=AgentContext(instruction="Collect sources", mission_id="m1"),
        )

        assert exec_result.status == ExecutionStatus.SUCCESS
        assert exec_result.tool_calls == 2
        assert exec_result.tokens_used == 42


class TestAgentOutputRuntime:
    def test_normalizes_agent_output(self):
        from remy.core_v3.agents.base_agent import AgentOutput
        from remy.core_v3.execution.execution_runtime import ExecutionStatus
        from remy.core_v3.runtime.agent_output_runtime import AgentOutputRuntime

        runtime = AgentOutputRuntime()
        exec_result = runtime.normalize(
            AgentOutput(
                status="partial",
                response="need one more source",
                tool_calls=1,
                tokens_used=11,
                cost_usd=0.03,
                evidence={"findings": [{"content": "x"}]},
            )
        )

        assert exec_result.status == ExecutionStatus.PARTIAL
        assert exec_result.tool_calls == 1
        assert exec_result.tokens_used == 11


class TestCostRuntime:
    def test_applies_execution_cost_to_mission(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.execution.execution_runtime import ExecutionResult, ExecutionStatus
        from remy.core_v3.missions.mission_models import Mission

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Track ecosystem shifts"))
        exec_result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            tokens_used=55,
            cost_usd=0.17,
        )

        chief.cost_runtime.apply(mission=mission, exec_result=exec_result)

        assert mission.total_tokens == 55
        assert mission.total_cost_usd == 0.17
        assert mission.cycles_run == 1


class TestCheckpointRuntime:
    def test_early_exit_checkpoints_state(self, monkeypatch):
        from remy.core_v3.agents.chief_agent import ChiefAgent

        chief = ChiefAgent()
        called = {"save": 0}

        def fake_save():
            called["save"] += 1

        monkeypatch.setattr(chief, "save_state", fake_save)
        chief.checkpoint_runtime.early_exit(chief)

        assert called["save"] == 1

    def test_complete_cycle_saves_plan_and_state(self, monkeypatch):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Mission

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Track ecosystem shifts"))
        plan = chief.create_plan_from_tasks(mission)
        called = {"plan": 0, "save": 0}

        def fake_save_plan(saved_plan):
            assert saved_plan is plan
            called["plan"] += 1

        def fake_save():
            called["save"] += 1

        monkeypatch.setattr(chief.plan_persistence, "save", fake_save_plan)
        monkeypatch.setattr(chief, "save_state", fake_save)

        chief.checkpoint_runtime.complete_cycle(chief, plan=plan)

        assert called == {"plan": 1, "save": 1}


class TestPostCycleRuntime:
    def test_finalize_runs_record_learn_and_checkpoint(self):
        from remy.core_v3.runtime.post_cycle_runtime import PostCycleRuntime

        calls = []

        class Recording:
            def record_cycle(self, **kwargs):
                calls.append(("record", kwargs["decision"]))

        class Learning:
            def observe_cycle(self, **kwargs):
                calls.append(("learn", kwargs["decision"]))

        class Checkpoint:
            def complete_cycle(self, chief, *, plan):
                calls.append(("checkpoint", plan))

        runtime = PostCycleRuntime(
            recording_runtime=Recording(),
            checkpoint_runtime=Checkpoint(),
            learning_runtime=Learning(),
        )
        runtime.finalize(
            object(),
            cycle_num=1,
            mission=object(),
            goal=None,
            task=None,
            plan="plan1",
            step=object(),
            specialist=object(),
            exec_result=object(),
            eval_result=object(),
            decision="complete",
            memory_assisted=True,
            duration_ms=10,
        )

        assert calls == [
            ("record", "complete"),
            ("learn", "complete"),
            ("checkpoint", "plan1"),
        ]


class TestTimingRuntime:
    def test_elapsed_ms_uses_start_marker(self):
        from remy.core_v3.runtime.timing_runtime import TimingRuntime

        runtime = TimingRuntime()
        started = runtime.start_cycle()
        elapsed = runtime.elapsed_ms(started)

        assert isinstance(elapsed, int)
        assert elapsed >= 0


class TestRecoveryRuntime:
    def test_applies_wait_and_abort_paths(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.runtime.decision_runtime import OutcomeDecision
        from remy.core_v3.planning.replan_engine import ReplanAction, ReplanDecision
        from remy.core_v3.missions.mission_models import Goal, Mission, MissionStatus, Task, TaskStatus

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        chief.activate_mission(mission)
        goal = Goal(description="Research market shifts", mission_id=mission.id)
        chief.add_goal(goal)
        task = Task(action="Collect sources", mission_id=mission.id, goal_id=goal.id)
        chief.add_task(task)
        plan = chief.create_plan_from_tasks(mission)
        step = plan.steps[0]
        task.status = TaskStatus.RUNNING

        # Wait path
        app = type("App", (), {"decision": "pause", "reason": "", "next_action": ""})()
        chief.recovery_runtime.apply_failure(
            mission=mission,
            goal=goal,
            task=task,
            plan=plan,
            step=step,
            decision=OutcomeDecision(
                phase="failure",
                reason="Transient blocker",
                replan_action=ReplanAction.WAIT,
                replan_decision=ReplanDecision(
                    action=ReplanAction.WAIT,
                    reason="Transient blocker (rate_limit), waiting 30s",
                    wait_seconds=30,
                ),
            ),
            outcome=app,
        )
        assert app.decision == "pause"
        assert task.status == TaskStatus.WAITING

        # Abort path
        task.status = TaskStatus.RUNNING
        mission.status = MissionStatus.ACTIVE
        app = type("App", (), {"decision": "pause", "reason": "", "next_action": ""})()
        chief.recovery_runtime.apply_failure(
            mission=mission,
            goal=goal,
            task=task,
            plan=plan,
            step=step,
            decision=OutcomeDecision(
                phase="failure",
                reason="No viable path",
                replan_action=ReplanAction.ABORT,
                replan_decision=ReplanDecision(
                    action=ReplanAction.ABORT,
                    reason="No viable path",
                ),
            ),
            outcome=app,
        )
        assert app.decision == "abort"
        assert task.status == TaskStatus.ABORTED


class TestCycleRuntime:
    def test_budget_denied_pauses_before_resolution(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.governance.budget_engine import BudgetConfig
        from remy.core_v3.missions.mission_models import Mission

        chief = ChiefAgent()
        chief.budget.config = BudgetConfig(daily_usd=0.01, per_cycle_usd=0.01)
        chief.budget.record_spend(0.02)
        mission = chief.accept_mission(Mission(description="Research market shifts"))

        prep = chief.cycle_runtime.prepare(mission)
        assert not prep.proceed
        assert prep.decision == "pause"

    def test_no_runnable_work_pauses(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Mission

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        chief.activate_mission(mission)

        prep = chief.cycle_runtime.prepare(mission)
        assert not prep.proceed
        assert prep.decision == "pause"
        assert prep.reason == "No runnable task or goal"

    def test_cycle_runtime_resolves_task_goal_plan_step(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Goal, Mission, Task

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        goal = Goal(description="Research market shifts", mission_id=mission.id)
        chief.add_goal(goal)
        task = Task(action="Research competitor moves", mission_id=mission.id, goal_id=goal.id)
        chief.add_task(task)

        prep = chief.cycle_runtime.prepare(mission)
        assert prep.proceed
        assert prep.goal is not None and prep.goal.id == goal.id
        assert prep.task is not None and prep.task.id == task.id
        assert prep.plan is not None
        assert prep.step is not None
        assert prep.memory_context is not None


class TestSchedulerRuntime:
    def test_selects_mission_with_current_task_first(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Goal, Mission, Task
        from remy.core_v3.runtime.mission_state_runtime import MissionStateRuntime
        from remy.core_v3.runtime.scheduler_runtime import SchedulerRuntime

        chief = ChiefAgent()
        mission_a = chief.accept_mission(Mission(description="Passive mission"))
        mission_b = chief.accept_mission(Mission(description="Active research mission"))
        chief.activate_mission(mission_a)
        goal = Goal(description="Active research mission", mission_id=mission_b.id)
        chief.add_goal(goal)
        chief.add_task(Task(action="Research competitor moves", mission_id=mission_b.id, goal_id=goal.id))

        selection = SchedulerRuntime(
            chief.mission_query_runtime,
            chief.projection_runtime,
            mission_state_runtime=MissionStateRuntime(),
        ).next_mission()
        assert selection.mission is not None
        assert selection.mission.id == mission_b.id
        assert selection.runnable_count == 2

    def test_returns_reason_when_no_runnable_missions(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.runtime.scheduler_runtime import SchedulerRuntime

        chief = ChiefAgent()
        selection = SchedulerRuntime(chief.mission_query_runtime, chief.projection_runtime).next_mission()
        assert selection.mission is None
        assert selection.reason == "no_runnable_missions"

    def test_records_score_reason_and_stuck_pressure(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent, ChiefDecision, CycleResult
        from remy.core_v3.missions.mission_models import Goal, Mission, Task
        from remy.core_v3.runtime.loop_runtime import LoopRuntime
        from remy.core_v3.runtime.scheduler_runtime import SchedulerRuntime

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research competitor signals"))
        goal = Goal(description="Research competitor signals", mission_id=mission.id)
        chief.add_goal(goal)
        chief.add_task(Task(action="Research competitor releases", mission_id=mission.id, goal_id=goal.id))
        loop_runtime = LoopRuntime(chief)
        loop_runtime.handle_result(CycleResult(decision=ChiefDecision.REPLAN, mission_id=mission.id), consecutive_failures=0)
        loop_runtime.handle_result(CycleResult(decision=ChiefDecision.PAUSE, mission_id=mission.id), consecutive_failures=0)

        scheduler = SchedulerRuntime(
            chief.mission_query_runtime,
            chief.projection_runtime,
            mission_state_runtime=chief.mission_state_runtime,
            loop_runtime=loop_runtime,
            evaluator=chief.evaluator,
        )
        selection = scheduler.next_mission()
        assert selection.mission is not None
        assert selection.score != 0.0
        assert "stuck=2" in selection.reason
        recent = scheduler.recent_decisions(1)[0]
        assert recent["mission_id"] == mission.id
        assert recent["score"] == round(selection.score, 3)


class TestMissionStateRuntime:
    def test_activate_for_execution_promotes_intake_to_active(self):
        from remy.core_v3.missions.mission_models import Mission, MissionStatus
        from remy.core_v3.runtime.mission_state_runtime import MissionStateRuntime

        mission = Mission(description="Research market shifts")
        MissionStateRuntime().activate_for_execution(mission)
        assert mission.status == MissionStatus.ACTIVE

    def test_scheduler_uses_mission_state_runtime_for_schedulable_missions(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Mission, MissionStatus
        from remy.core_v3.runtime.mission_state_runtime import MissionStateRuntime
        from remy.core_v3.runtime.scheduler_runtime import SchedulerRuntime

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        selection = SchedulerRuntime(
            chief.mission_query_runtime,
            chief.projection_runtime,
            mission_state_runtime=MissionStateRuntime(),
        ).next_mission()
        assert selection.mission is not None
        assert selection.mission.id == mission.id
        assert mission.status == MissionStatus.ACTIVE

    def test_scheduler_deprioritizes_mission_with_degraded_preferred_specialist(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.evaluation.evaluation_engine import EvalResult, EvalVerdict
        from remy.core_v3.missions.mission_models import Goal, Mission, Task
        from remy.core_v3.runtime.scheduler_runtime import SchedulerRuntime

        chief = ChiefAgent()
        chief.evaluator._record_outcome(
            "e1",
            "executor",
            EvalResult(verdict=EvalVerdict.SUCCESS, unsupported_observed_claims=3, factuality_penalty=0.35),
        )
        chief.evaluator._record_outcome("r1", "researcher", EvalResult(verdict=EvalVerdict.SUCCESS))
        chief.evaluator._record_outcome("r2", "researcher", EvalResult(verdict=EvalVerdict.SUCCESS))

        browse_mission = chief.accept_mission(Mission(description="Browse signup page"))
        browse_goal = Goal(description="Check signup page", mission_id=browse_mission.id)
        chief.add_goal(browse_goal)
        chief.add_task(Task(action="Browse to the signup page", mission_id=browse_mission.id, goal_id=browse_goal.id))

        research_mission = chief.accept_mission(Mission(description="Research primary filings"))
        research_goal = Goal(description="Research primary filings", mission_id=research_mission.id)
        chief.add_goal(research_goal)
        chief.add_task(Task(action="Research primary filings", mission_id=research_mission.id, goal_id=research_goal.id))

        scheduler = SchedulerRuntime(
            chief.mission_query_runtime,
            chief.projection_runtime,
            mission_state_runtime=chief.mission_state_runtime,
            evaluator=chief.evaluator,
            ops_query_runtime=chief.ops_query_runtime,
        )
        selection = scheduler.next_mission()

        assert selection.mission is not None
        assert selection.mission.id == research_mission.id
        assert "routing_avoid=executor" in selection.reason or "routing_prefer=researcher" in selection.reason

    def test_scheduler_adds_routing_bonus_for_healthy_preferred_specialist(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.evaluation.evaluation_engine import EvalResult, EvalVerdict
        from remy.core_v3.missions.mission_models import Goal, Mission, Task
        from remy.core_v3.runtime.scheduler_runtime import SchedulerRuntime

        chief = ChiefAgent()
        chief.evaluator._record_outcome("r1", "researcher", EvalResult(verdict=EvalVerdict.SUCCESS))
        chief.evaluator._record_outcome("r2", "researcher", EvalResult(verdict=EvalVerdict.SUCCESS))

        mission = chief.accept_mission(Mission(description="Research primary filings"))
        goal = Goal(description="Research primary filings", mission_id=mission.id)
        chief.add_goal(goal)
        chief.add_task(Task(action="Research primary filings", mission_id=mission.id, goal_id=goal.id))

        scheduler = SchedulerRuntime(
            chief.mission_query_runtime,
            chief.projection_runtime,
            mission_state_runtime=chief.mission_state_runtime,
            evaluator=chief.evaluator,
            ops_query_runtime=chief.ops_query_runtime,
        )
        selection = scheduler.next_mission()

        assert selection.mission is not None
        assert selection.mission.id == mission.id
        assert "routing_prefer=researcher" in selection.reason
        recent = scheduler.recent_decisions(1)[0]
        assert recent["details"]["routing_reason"] == "routing_prefer=researcher"


class TestMissionQueryRuntime:
    def test_exposes_active_missions_and_plan_lookup(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Mission

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        chief.activate_mission(mission)
        plan = chief.create_plan(mission)

        query = chief.mission_query_runtime
        assert query.get_mission(mission.id) is mission
        assert query.get_plan(mission.id) is plan
        assert mission in query.active_missions()
        assert mission in query.all_missions()


class TestPlanQueryRuntime:
    def test_current_step_prefers_task_bound_step_then_falls_back(self):
        from remy.core_v3.planning.plan_models import Plan, PlanStep
        from remy.core_v3.missions.mission_models import Task
        from remy.core_v3.runtime.plan_query_runtime import PlanQueryRuntime

        plan = Plan(steps=[
            PlanStep(id="step_task_a", instruction="Task A"),
            PlanStep(id="step_task_b", instruction="Task B"),
        ])
        task = Task(id="task_b", action="Task B")
        query = PlanQueryRuntime(plans={"m1": plan})

        current = query.current_step(plan, task)
        assert current is not None
        assert current.id == "step_task_b"
        assert query.plan_steps(plan) == 2
        assert query.plan_progress(plan) == 0.0


class TestGoalQueryRuntime:
    def test_summary_reports_status_counts_and_blocked_reasons(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Goal, GoalStatus, Mission

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        goal = Goal(description="Research market shifts", mission_id=mission.id, status=GoalStatus.BLOCKED)
        goal.metadata["block_reason"] = "awaiting external data"
        chief.add_goal(goal)

        summary = chief.goal_query_runtime.summary(mission.id)
        assert summary["total"] == 1
        assert summary["by_status"]["blocked"] == 1
        assert summary["blocked"][0]["reason"] == "awaiting external data"


class TestOpsQueryRuntime:
    def test_exposes_budget_audit_approval_and_execution_reads(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent

        chief = ChiefAgent()
        chief.budget.record_spend(0.02, mission_id="m1", specialist="researcher")
        chief.audit.log_event("note", "something happened", mission_id="m1", actor="chief", cost_usd=0.02)
        chief.approval.request_approval("publish", risk_category="high", cost_usd=0.1)

        ops = chief.ops_query_runtime
        assert "daily_remaining_usd" in ops.budget_summary()
        assert ops.pending_approvals() >= 1
        assert "events_24h" in ops.audit_summary()
        assert ops.mission_cost("m1") >= 0.02
        assert "cycles" in ops.execution_stats()

    def test_exposes_health_governance_and_specialist_views(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent

        chief = ChiefAgent()
        chief.audit.log_event("step_failed", "research failed", mission_id="m1", actor="researcher")
        chief.approval.request_approval("publish", risk_category="high", cost_usd=0.1)

        ops = chief.ops_query_runtime
        governance = ops.governance_summary()
        assert governance["pending_approvals"] >= 1
        assert "approval_stats" in governance

        events = ops.specialist_recent_events("researcher", 5)
        assert events
        assert events[0]["event"] == "step_failed"

        health = ops.health_snapshot()
        assert "status" in health
        assert "budget" in health

    def test_exposes_stuck_missions_scheduler_decisions_and_quality_debt(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent, ChiefDecision, CycleResult
        from remy.core_v3.missions.mission_models import Goal, Mission, Task
        from remy.core_v3.runtime.loop_runtime import LoopRuntime
        from remy.core_v3.runtime.scheduler_runtime import SchedulerRuntime

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        goal = Goal(description="Research market shifts", mission_id=mission.id)
        chief.add_goal(goal)
        chief.add_task(Task(action="Research competitor threads", mission_id=mission.id, goal_id=goal.id))
        loop_runtime = LoopRuntime(chief)
        loop_runtime.handle_result(CycleResult(decision=ChiefDecision.REPLAN, mission_id=mission.id), consecutive_failures=0)
        loop_runtime.handle_result(CycleResult(decision=ChiefDecision.ESCALATE, mission_id=mission.id), consecutive_failures=0)
        scheduler = SchedulerRuntime(
            chief.mission_query_runtime,
            chief.projection_runtime,
            mission_state_runtime=chief.mission_state_runtime,
            loop_runtime=loop_runtime,
            evaluator=chief.evaluator,
        )
        scheduler.next_mission()
        chief.ops_query_runtime.bind_autonomy(
            loop_runtime=loop_runtime,
            scheduler_runtime=scheduler,
            mission_query_runtime=chief.mission_query_runtime,
            projection_runtime=chief.projection_runtime,
        )
        from remy.core_v3.evaluation.evaluation_engine import EvalResult, EvalVerdict
        chief.evaluator._record_outcome(
            goal.id,
            "researcher",
            EvalResult(verdict=EvalVerdict.SUCCESS, unsupported_observed_claims=2, factuality_penalty=0.12),
        )

        stuck = chief.ops_query_runtime.stuck_missions()
        decisions = chief.ops_query_runtime.scheduler_decisions_recent()
        debt = chief.ops_query_runtime.quality_debt_by_specialist()
        assert stuck and stuck[0]["mission"]["id"] == mission.id
        assert decisions and decisions[0]["mission_id"] == mission.id
        assert debt and debt[0]["id"] == "researcher"

    def test_exposes_evidence_debt_queue(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Goal, Mission, Task

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        goal = Goal(description="Research market shifts", mission_id=mission.id)
        chief.add_goal(goal)
        current = Task(action="Research competitor threads", mission_id=mission.id, goal_id=goal.id)
        chief.add_task(current)

        debt = chief.evidence_debt_runtime.open_debt(
            mission=mission,
            goal=goal,
            current_task=current,
            policy_hint={
                "hint": "requires_evidence",
                "situation": "Research market shifts",
                "action": "answer_without_evidence",
                "reason": "prior factuality scar",
            },
        )

        queue = chief.ops_query_runtime.evidence_debt_queue()
        assert debt is not None
        assert queue and queue[0]["id"] == debt.id
        assert queue[0]["mission_id"] == mission.id
        assert queue[0]["goal_id"] == goal.id
        assert "Verify evidence before relying on action" in queue[0]["action"]
        assert queue[0]["source_situation"] == "Research market shifts"
        assert queue[0]["source_action"] == "answer_without_evidence"

    def test_evidence_debt_resolution_records_consequence(self, monkeypatch):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.evaluation.evaluation_engine import EvalResult, EvalVerdict
        from remy.core_v3.execution.execution_runtime import ExecutionResult, ExecutionStatus
        from remy.core_v3.missions.mission_models import Goal, Mission, Task
        from remy.core_v3.planning.plan_models import PlanStep

        class FakeMemory:
            def __init__(self):
                self.units = []

            def capture_consequence(self, **kwargs):
                self.units.append(kwargs)
                return "cu-1"

        fake_memory = FakeMemory()
        monkeypatch.setattr(
            "remy.core_v3.memory.memory_api.get_memory",
            lambda: fake_memory,
        )

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        goal = Goal(description="Research market shifts", mission_id=mission.id)
        chief.add_goal(goal)
        current = Task(action="Research competitor threads", mission_id=mission.id, goal_id=goal.id)
        chief.add_task(current)
        debt = chief.evidence_debt_runtime.open_debt(
            mission=mission,
            goal=goal,
            current_task=current,
            policy_hint={
                "hint": "requires_evidence",
                "situation": "Research market shifts",
                "action": "answer_without_evidence",
                "reason": "prior factuality scar",
            },
        )

        record_id = chief.evidence_debt_runtime.resolve_after_evaluation(
            task=debt,
            mission=mission,
            goal=goal,
            step=PlanStep(instruction=debt.action),
            specialist=type("Specialist", (), {"id": "researcher"})(),
            exec_result=ExecutionResult(status=ExecutionStatus.SUCCESS, response="Verified with source."),
            eval_result=EvalResult(verdict=EvalVerdict.SUCCESS, reason="evidence found"),
        )

        assert record_id == "cu-1"
        assert fake_memory.units
        unit = fake_memory.units[0]
        assert unit["situation"] == "Research market shifts"
        assert unit["action"] == "answer_without_evidence"
        assert unit["consequence"] == "SUPPORTS"
        assert unit["trust"] == 1
        assert "evidence-debt-resolution" in unit["scope"]
        assert unit["namespace"] == "remy"

    def test_exposes_factuality_summary(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.execution.cycle_recorder import CycleRecord
        from remy.core_v3.evaluation.evaluation_engine import EvalResult, EvalVerdict

        chief = ChiefAgent()
        chief.recorder._records.append(CycleRecord(
            cycle_num=1,
            status="success",
            goal_description="fact-check",
            unsupported_observed_claims=2,
        ))
        chief.evaluator._record_outcome("g1", "researcher", EvalResult(verdict=EvalVerdict.SUCCESS))
        chief.evaluator._record_outcome(
            "g2",
            "analyst",
            EvalResult(verdict=EvalVerdict.SUCCESS, unsupported_observed_claims=2, factuality_penalty=0.24),
        )

        ops = chief.ops_query_runtime
        assert ops.factuality_summary()["unsupported_observed_claims_total"] == 2
        assert ops.factuality_summary()["per_specialist"]

    def test_exposes_specialist_quality(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.evaluation.evaluation_engine import EvalResult, EvalVerdict

        chief = ChiefAgent()
        chief.evaluator._record_outcome(
            "g1",
            "analyst",
            EvalResult(verdict=EvalVerdict.SUCCESS, unsupported_observed_claims=2, factuality_penalty=0.24),
        )
        quality = chief.ops_query_runtime.specialist_quality("analyst")
        assert quality["quality_adjusted_success_rate"] <= quality["success_rate"]


class TestDashboardRuntime:
    def test_dashboard_runtime_composes_operator_views(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.missions.mission_models import Mission

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        chief.activate_mission(mission)

        dashboard = chief.dashboard_runtime.dashboard()
        assert "missions" in dashboard
        assert "budget" in dashboard
        assert "governance" in dashboard
        assert "factuality" in dashboard
        assert "top_offenders" in dashboard["factuality"]
        assert dashboard["active_missions"] >= 1

    def test_dashboard_runtime_health_and_budget_views(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent

        chief = ChiefAgent()
        detail = chief.dashboard_runtime.budget_detail()
        health = chief.dashboard_runtime.health_check()
        assert "spending_history" in detail
        assert "status" in health

    def test_dashboard_runtime_specialist_detail_uses_quality_metrics(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.evaluation.evaluation_engine import EvalResult, EvalVerdict

        chief = ChiefAgent()
        chief.evaluator._record_outcome(
            "g1",
            "analyst",
            EvalResult(verdict=EvalVerdict.SUCCESS, unsupported_observed_claims=1, factuality_penalty=0.12),
        )
        detail = chief.dashboard_runtime.specialist_detail("analyst")
        assert "success_rate" in detail
        assert "quality_adjusted_success_rate" in detail
        assert "unsupported_observed_claims" in detail


class TestFactualityRuntime:
    def test_applies_guard_to_execution_result(self):
        from remy.core_v3.execution.execution_runtime import ExecutionResult, ExecutionStatus
        from remy.core_v3.runtime.factuality_runtime import FactualityRuntime

        runtime = FactualityRuntime()
        result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            response="I just reviewed your repository and it looks strong.",
            session_log=[],
        )

        updated = runtime.apply(result)
        assert updated.unsupported_observed_claims == 1
        assert updated.factuality_modified is True
        assert "reviewed your repository" not in updated.response.lower()


class TestAutonomyLoopStatus:
    def test_status_uses_query_runtimes_for_budget_and_execution(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.runtime.autonomy_loop import AutonomyLoop

        loop = AutonomyLoop(chief=ChiefAgent())
        status = loop.status()
        assert "daily_remaining_usd" in status["budget"]
        assert "cycles" in status["recorder"]

    def test_status_exposes_snapshot_fields(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent, CycleResult, ChiefDecision
        from remy.core_v3.missions.mission_models import Goal, Mission, Task
        from remy.core_v3.runtime.autonomy_loop import AutonomyLoop

        chief = ChiefAgent()
        mission = chief.accept_mission(Mission(description="Research market shifts"))
        goal = Goal(description="Research market shifts", mission_id=mission.id)
        chief.add_goal(goal)
        task = Task(action="Research competitor moves", mission_id=mission.id, goal_id=goal.id)
        chief.add_task(task)
        loop = AutonomyLoop(chief=chief)
        loop._last_selection = type("Selection", (), {"mission": mission, "reason": "priority=1", "score": 2.5})()
        loop._last_scheduler_reason = "priority=1"
        loop._last_result = CycleResult(
            decision=ChiefDecision.EXECUTE_STEP,
            mission_id=mission.id,
            goal_id=goal.id,
            step_executed="step1",
            reason="ok",
        )

        status = loop.status()
        assert status["current_mission"]["id"] == mission.id
        assert status["scheduler_reason"] == "priority=1"
        assert "stuck_missions_count" in status
        assert "approval_queue" in status
        assert "quality_debt_by_specialist" in status
        assert "evidence_debt_queue" in status


class TestLoopRuntime:

    def test_abort_increments_failure_counter(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent, ChiefDecision, CycleResult
        from remy.core_v3.runtime.loop_runtime import LoopRuntime

        loop_runtime = LoopRuntime(ChiefAgent())
        failures = loop_runtime.handle_result(
            CycleResult(decision=ChiefDecision.ABORT, mission_id="m1", reason="failed"),
            consecutive_failures=1,
        )
        assert failures == 2

    def test_execute_step_resets_failure_counter(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent, ChiefDecision, CycleResult
        from remy.core_v3.runtime.loop_runtime import LoopRuntime

        loop_runtime = LoopRuntime(ChiefAgent())
        failures = loop_runtime.handle_result(
            CycleResult(decision=ChiefDecision.EXECUTE_STEP, mission_id="m1"),
            consecutive_failures=2,
        )
        assert failures == 0

    def test_replan_pause_and_escalate_accumulate_pressure(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent, ChiefDecision, CycleResult
        from remy.core_v3.runtime.loop_runtime import LoopRuntime

        loop_runtime = LoopRuntime(ChiefAgent())
        for decision in (ChiefDecision.REPLAN, ChiefDecision.PAUSE, ChiefDecision.ESCALATE):
            loop_runtime.handle_result(CycleResult(decision=decision, mission_id="m1"), consecutive_failures=0)

        assert loop_runtime.pressure_for("m1") == 3
        assert loop_runtime.stuck_missions_count() == 1
        assert loop_runtime.stuck_missions()[0]["mission_id"] == "m1"
        assert loop_runtime.recent_decisions(1)[0]["decision"] == ChiefDecision.ESCALATE


class TestMaintenanceRuntime:
    def test_runs_archival_cleanup_and_persistence(self, monkeypatch):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.runtime.maintenance_runtime import MaintenanceRuntime

        chief = ChiefAgent()
        calls = {"archived": 0, "cleared": 0, "saved": 0}

        class FakeMemory:
            def run_maintenance(self):
                return {"consolidated": 2}

        monkeypatch.setattr(
            chief.goal_tracker,
            "archive_stale",
            lambda: calls.__setitem__("archived", calls["archived"] + 1),
        )
        monkeypatch.setattr(
            chief.approval,
            "clear_decided",
            lambda: calls.__setitem__("cleared", calls["cleared"] + 1),
        )
        monkeypatch.setattr(
            chief,
            "save_state",
            lambda: calls.__setitem__("saved", calls["saved"] + 1),
        )

        import remy.core_v3.memory.memory_api as memory_api

        monkeypatch.setattr(memory_api, "get_memory", lambda: FakeMemory())

        report = MaintenanceRuntime(chief).run()
        assert calls == {"archived": 1, "cleared": 1, "saved": 1}
        assert report["aura_maintenance"] == {"consolidated": 2}

    def test_maintenance_runtime_tolerates_memory_errors(self, monkeypatch):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.runtime.maintenance_runtime import MaintenanceRuntime

        chief = ChiefAgent()
        monkeypatch.setattr(chief.goal_tracker, "archive_stale", lambda: None)
        monkeypatch.setattr(chief.approval, "clear_decided", lambda: None)
        monkeypatch.setattr(chief, "save_state", lambda: None)

        import remy.core_v3.memory.memory_api as memory_api

        def _boom():
            raise RuntimeError("offline")

        monkeypatch.setattr(memory_api, "get_memory", _boom)

        report = MaintenanceRuntime(chief).run()
        assert report["aura_maintenance"] is None
        assert report["aura_maintenance_error"] == "offline"


class TestGuardRuntime:
    def test_budget_exhausted_blocks_and_requests_sleep(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.governance.budget_engine import BudgetConfig
        from remy.core_v3.runtime.guard_runtime import GuardRuntime

        chief = ChiefAgent()
        chief.budget.config = BudgetConfig(daily_usd=0.01, per_cycle_usd=1.0)
        chief.budget.record_spend(0.02)
        chief.budget.sync_from_v2 = lambda: None

        guard = GuardRuntime(chief).check(
            cycle_count=1,
            consecutive_failures=0,
            max_consecutive_failures=3,
            maintenance_only=False,
        )
        assert not guard.proceed
        assert guard.reason == "budget_exhausted"
        assert guard.sleep_sec == 300

    def test_failure_cooldown_resets_counter(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.runtime.guard_runtime import GuardRuntime

        chief = ChiefAgent()
        guard = GuardRuntime(chief).check(
            cycle_count=4,
            consecutive_failures=3,
            max_consecutive_failures=3,
            maintenance_only=False,
        )
        assert not guard.proceed
        assert guard.reason == "failure_cooldown"
        assert guard.sleep_sec == 600
        assert guard.consecutive_failures == 0

    def test_maintenance_only_short_circuits_without_sleep(self):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.runtime.guard_runtime import GuardRuntime

        chief = ChiefAgent()
        guard = GuardRuntime(chief).check(
            cycle_count=2,
            consecutive_failures=1,
            max_consecutive_failures=3,
            maintenance_only=True,
        )
        assert not guard.proceed
        assert guard.reason == "maintenance_only"
        assert guard.sleep_sec == 0


class TestLifecycleRuntime:
    def test_start_session_loads_state_archives_and_audits(self, monkeypatch):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.runtime.lifecycle_runtime import LifecycleRuntime

        chief = ChiefAgent()
        calls = {"loaded": 0, "archived": 0, "audited": 0}

        monkeypatch.setattr(
            chief,
            "load_state",
            lambda: calls.__setitem__("loaded", calls["loaded"] + 1),
        )
        monkeypatch.setattr(
            chief.goal_tracker,
            "archive_stale",
            lambda: calls.__setitem__("archived", calls["archived"] + 1),
        )
        monkeypatch.setattr(
            chief.audit,
            "log_event",
            lambda *args, **kwargs: calls.__setitem__("audited", calls["audited"] + 1),
        )

        LifecycleRuntime(chief).start_session()
        assert calls == {"loaded": 1, "archived": 1, "audited": 1}

    def test_stop_session_persists_state(self, monkeypatch):
        from remy.core_v3.agents.chief_agent import ChiefAgent
        from remy.core_v3.runtime.lifecycle_runtime import LifecycleRuntime

        chief = ChiefAgent()
        calls = {"saved": 0}
        monkeypatch.setattr(
            chief,
            "save_state",
            lambda: calls.__setitem__("saved", calls["saved"] + 1),
        )

        LifecycleRuntime(chief).stop_session()
        assert calls["saved"] == 1


class TestErrorRuntime:
    def test_cycle_exception_maps_to_pause_and_increments_failures(self):
        from remy.core_v3.agents.chief_agent import ChiefDecision
        from remy.core_v3.runtime.error_runtime import ErrorRuntime

        handled = ErrorRuntime().handle_cycle_exception(
            mission_id="m1",
            error=RuntimeError("boom"),
            consecutive_failures=2,
        )
        assert handled.cycle_result.decision == ChiefDecision.PAUSE
        assert handled.cycle_result.mission_id == "m1"
        assert "boom" in handled.cycle_result.reason
        assert handled.consecutive_failures == 3


# ---------------------------------------------------------------------------
# Cross-component integration
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_all_imports(self):
        """Verify all core_v3 modules import without error."""
        from remy.core_v3.governance.policy_engine import PolicyEngine
        from remy.core_v3.governance.budget_engine import BudgetEngine
        from remy.core_v3.governance.approval_engine import ApprovalEngine
        from remy.core_v3.governance.audit_engine import AuditEngine, EventType
        from remy.core_v3.evaluation.evaluation_engine import EvaluationEngine, BlockerType
        from remy.core_v3.planning.replan_engine import ReplanEngine, ReplanAction
        from remy.core_v3.research.research_runtime import ResearchRuntime
        from remy.core_v3.research.research_models import ResearchProject
        from remy.core_v3.research.source_ranking import SourceRanker
        from remy.core_v3.research.synthesis import SynthesisEngine
        from remy.core_v3.improvement.outcome_learner import OutcomeLearner
        from remy.core_v3.improvement.playbook_engine import PlaybookEngine
        from remy.core_v3.agents.base_agent import AgentContext, AgentOutput
        from remy.core_v3.agents.researcher import ResearcherAgent
        from remy.core_v3.agents.analyst_agent import AnalystAgent
        from remy.core_v3.agents.executor_agent import ExecutorAgent
        from remy.core_v3.missions.mission_models import Mission, Goal, Task
        from remy.core_v3.planning.plan_models import Plan, PlanStep
        from remy.core_v3.memory.memory_api import MemoryClass
        from remy.core_v3.runtime.state_machine import transition
        from remy.core_v3.execution.cycle_recorder import CycleRecorder
        from remy.core_v3.observability.telemetry import Telemetry

    def test_budget_not_double_charged(self):
        """Verify delegation_engine records spend, chief does not duplicate."""
        import inspect
        from remy.core_v3.agents import chief_agent

        source = inspect.getsource(chief_agent.ChiefAgent.run_cycle)
        # The chief should NOT call budget.record_spend after delegation
        assert "budget.record_spend" not in source, (
            "chief_agent.run_cycle() should not call budget.record_spend — "
            "delegation_engine already handles it"
        )
    def test_bootstrap_exposes_mission_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "mission_runtime" in runtime
        assert runtime["mission_runtime"] is runtime["chief"].mission_runtime

    def test_bootstrap_exposes_mission_state_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "mission_state_runtime" in runtime
        assert runtime["mission_state_runtime"] is runtime["chief"].mission_state_runtime

    def test_bootstrap_exposes_mission_query_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "mission_query_runtime" in runtime
        assert runtime["mission_query_runtime"] is runtime["chief"].mission_query_runtime

    def test_bootstrap_exposes_plan_query_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "plan_query_runtime" in runtime
        assert runtime["plan_query_runtime"] is runtime["chief"].plan_query_runtime

    def test_bootstrap_exposes_plan_state_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "plan_state_runtime" in runtime
        assert runtime["plan_state_runtime"] is runtime["chief"].plan_state_runtime

    def test_bootstrap_exposes_goal_query_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "goal_query_runtime" in runtime
        assert runtime["goal_query_runtime"] is runtime["chief"].goal_query_runtime

    def test_bootstrap_exposes_ops_query_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "ops_query_runtime" in runtime
        assert runtime["ops_query_runtime"] is runtime["chief"].ops_query_runtime

    def test_bootstrap_exposes_outcome_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "outcome_runtime" in runtime
        assert runtime["outcome_runtime"] is runtime["chief"].outcome_runtime

    def test_bootstrap_exposes_execution_gate(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "execution_gate" in runtime
        assert runtime["execution_gate"] is runtime["chief"].execution_gate

    def test_bootstrap_exposes_cycle_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "cycle_runtime" in runtime
        assert runtime["cycle_runtime"] is runtime["chief"].cycle_runtime

    def test_bootstrap_exposes_loop_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "loop_runtime" in runtime
        assert runtime["loop_runtime"] is runtime["loop"].loop_runtime

    def test_bootstrap_exposes_scheduler_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "scheduler_runtime" in runtime
        assert runtime["scheduler_runtime"] is runtime["loop"].scheduler_runtime

    def test_bootstrap_exposes_maintenance_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "maintenance_runtime" in runtime
        assert runtime["maintenance_runtime"] is runtime["loop"].maintenance_runtime

    def test_bootstrap_exposes_guard_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "guard_runtime" in runtime
        assert runtime["guard_runtime"] is runtime["loop"].guard_runtime

    def test_bootstrap_exposes_lifecycle_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "lifecycle_runtime" in runtime
        assert runtime["lifecycle_runtime"] is runtime["loop"].lifecycle_runtime

    def test_bootstrap_exposes_error_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "error_runtime" in runtime
        assert runtime["error_runtime"] is runtime["loop"].error_runtime

    def test_bootstrap_exposes_learning_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "learning_runtime" in runtime
        assert runtime["learning_runtime"] is runtime["chief"].learning_runtime

    def test_bootstrap_exposes_memory_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "memory_runtime" in runtime
        assert runtime["memory_runtime"] is runtime["chief"].memory_runtime

    def test_bootstrap_exposes_persistence_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "persistence_runtime" in runtime
        assert runtime["persistence_runtime"] is runtime["chief"].persistence_runtime

    def test_bootstrap_exposes_recording_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "recording_runtime" in runtime
        assert runtime["recording_runtime"] is runtime["chief"].recording_runtime

    def test_bootstrap_exposes_intake_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "intake_runtime" in runtime
        assert runtime["intake_runtime"] is runtime["chief"].intake_runtime

    def test_bootstrap_exposes_context_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "context_runtime" in runtime
        assert runtime["context_runtime"] is runtime["chief"].context_runtime

    def test_bootstrap_exposes_evaluation_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "evaluation_runtime" in runtime
        assert runtime["evaluation_runtime"] is runtime["chief"].evaluation_runtime

    def test_bootstrap_exposes_specialist_inference_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "specialist_inference_runtime" in runtime
        assert runtime["specialist_inference_runtime"] is runtime["chief"].specialist_inference_runtime

    def test_bootstrap_exposes_goal_context_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "goal_context_runtime" in runtime
        assert runtime["goal_context_runtime"] is runtime["chief"].goal_context_runtime

    def test_bootstrap_exposes_plan_builder_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "plan_builder_runtime" in runtime
        assert runtime["plan_builder_runtime"] is runtime["chief"].plan_builder_runtime

    def test_bootstrap_exposes_cost_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "cost_runtime" in runtime
        assert runtime["cost_runtime"] is runtime["chief"].cost_runtime

    def test_bootstrap_exposes_checkpoint_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "checkpoint_runtime" in runtime
        assert runtime["checkpoint_runtime"] is runtime["chief"].checkpoint_runtime

    def test_bootstrap_exposes_cycle_execution_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "cycle_execution_runtime" in runtime
        assert runtime["cycle_execution_runtime"] is runtime["chief"].cycle_execution_runtime

    def test_bootstrap_exposes_post_cycle_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "post_cycle_runtime" in runtime
        assert runtime["post_cycle_runtime"] is runtime["chief"].post_cycle_runtime

    def test_bootstrap_exposes_timing_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "timing_runtime" in runtime
        assert runtime["timing_runtime"] is runtime["chief"].timing_runtime

    def test_bootstrap_exposes_execution_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "execution_runtime" in runtime
        assert runtime["execution_runtime"] is runtime["chief"].execution_runtime

    def test_bootstrap_exposes_agent_output_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "agent_output_runtime" in runtime
        assert runtime["agent_output_runtime"] is runtime["chief"].agent_output_runtime

    def test_bootstrap_exposes_cycle_result_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "cycle_result_runtime" in runtime
        assert runtime["cycle_result_runtime"] is runtime["chief"].cycle_result_runtime

    def test_bootstrap_exposes_step_state_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "step_state_runtime" in runtime
        assert runtime["step_state_runtime"] is runtime["chief"].step_state_runtime

    def test_bootstrap_exposes_task_state_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "task_state_runtime" in runtime
        assert runtime["task_state_runtime"] is runtime["chief"].task_state_runtime

    def test_bootstrap_exposes_task_decision_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "task_decision_runtime" in runtime
        assert runtime["task_decision_runtime"] is runtime["chief"].task_decision_runtime

    def test_bootstrap_exposes_task_outcome_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "task_outcome_runtime" in runtime
        assert runtime["task_outcome_runtime"] is runtime["chief"].task_outcome_runtime

    def test_bootstrap_exposes_goal_outcome_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "goal_outcome_runtime" in runtime
        assert runtime["goal_outcome_runtime"] is runtime["chief"].goal_outcome_runtime

    def test_bootstrap_exposes_mission_outcome_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "mission_outcome_runtime" in runtime
        assert runtime["mission_outcome_runtime"] is runtime["chief"].mission_outcome_runtime

    def test_bootstrap_exposes_decision_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "decision_runtime" in runtime
        assert runtime["decision_runtime"] is runtime["chief"].decision_runtime

    def test_bootstrap_exposes_completion_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "completion_runtime" in runtime
        assert runtime["completion_runtime"] is runtime["chief"].completion_runtime

    def test_bootstrap_exposes_recovery_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "recovery_runtime" in runtime
        assert runtime["recovery_runtime"] is runtime["chief"].recovery_runtime

    def test_bootstrap_exposes_result_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "result_runtime" in runtime
        assert runtime["result_runtime"] is runtime["chief"].result_runtime

    def test_bootstrap_exposes_specialist_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "specialist_runtime" in runtime
        assert runtime["specialist_runtime"] is runtime["chief"].specialist_runtime

    def test_bootstrap_exposes_projection_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "projection_runtime" in runtime
        assert runtime["projection_runtime"] is runtime["chief"].projection_runtime

    def test_bootstrap_exposes_dashboard_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "dashboard_runtime" in runtime
        assert runtime["dashboard_runtime"] is runtime["chief"].dashboard_runtime

    def test_bootstrap_exposes_factuality_runtime(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "factuality_runtime" in runtime
        assert runtime["factuality_runtime"] is runtime["chief"].factuality_runtime
