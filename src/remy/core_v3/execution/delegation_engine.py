"""
Delegation Engine for Remy v3.

Routes execution from Chief Agent to the correct specialist agent.
Manages agent instantiation, context building, and result collection.

Replaces v2 orchestrator.dispatch_worker() as the primary execution path.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..agents.base_agent import BaseSpecialistAgent, AgentContext, AgentOutput
from ..agents.specialist_registry import SpecialistRegistry, SpecialistProfile
from ..agents.researcher import ResearcherAgent
from ..agents.executor_agent import ExecutorAgent
from ..agents.analyst_agent import AnalystAgent
from ..governance.budget_engine import BudgetEngine, BudgetAction
from ..governance.audit_engine import AuditEngine

log = logging.getLogger(__name__)


class DelegationEngine:
    """Routes tasks to specialist agents and collects results.

    Resolution priority:
    1. Explicit specialist ID in plan step
    2. can_handle() check on each agent
    3. Specialist registry resolve() (v2 pack-based fallback)
    4. Default to analyst
    """

    def __init__(
        self,
        registry: SpecialistRegistry | None = None,
        budget: BudgetEngine | None = None,
        audit: AuditEngine | None = None,
    ):
        self.registry = registry or SpecialistRegistry()
        self.budget = budget or BudgetEngine()
        self.audit = audit or AuditEngine()

        # Agent instances (singleton per type)
        self._agents: dict[str, BaseSpecialistAgent] = {}
        self._register_default_agents()

    def _register_default_agents(self):
        """Register MVP specialist agents."""
        self._agents["researcher"] = ResearcherAgent()
        self._agents["executor"] = ExecutorAgent()
        self._agents["analyst"] = AnalystAgent()

    def register_agent(self, agent: BaseSpecialistAgent):
        """Register a custom specialist agent."""
        self._agents[agent.agent_id] = agent

    # -------------------------------------------------------------------
    # Delegation
    # -------------------------------------------------------------------

    async def delegate(self, ctx: AgentContext) -> AgentOutput:
        """Delegate a task to the best specialist agent.

        Steps:
        1. Resolve agent
        2. Check budget
        3. Build context with constraints
        4. Execute
        5. Record audit
        6. Return result
        """
        start = time.time()

        # 1. Resolve agent
        agent = self._resolve_agent(ctx)
        log.info("Delegating to %s: %s", agent.agent_id, ctx.instruction[:60])

        # 2. Budget check
        budget_action, budget_reason = self.budget.check_budget(
            estimated_cost_usd=0.05,
            mission_id=ctx.mission_id,
        )
        if budget_action == BudgetAction.DENY:
            return AgentOutput(
                status="failure",
                response=f"Budget denied: {budget_reason}",
            )

        # Apply profile constraints to context
        profile = self.registry.get(agent.agent_id)
        if profile:
            ctx.step_budget = ctx.step_budget or profile.step_budget
            ctx.timeout_sec = ctx.timeout_sec or profile.timeout_sec
            ctx.tools_allowed = ctx.tools_allowed or profile.tools
            ctx.guardrails = ctx.guardrails or profile.guardrails
            ctx.approval_mode = ctx.approval_mode or profile.approval_mode

        if budget_action == BudgetAction.DEGRADE:
            ctx.use_cheap_model = True

        # 3. Execute
        output = await agent.run(ctx)

        # 4. Record spend
        self.budget.record_spend(
            output.cost_usd, output.tokens_used, ctx.mission_id
        )

        # 5. Audit
        self.audit.log_event(
            "delegation_completed",
            f"{agent.agent_id}: {output.status}",
            actor=agent.agent_id,
            mission_id=ctx.mission_id,
            cost_usd=output.cost_usd,
            details={
                "instruction": ctx.instruction[:80],
                "goal_id": ctx.goal_id,
                "tool_calls": output.tool_calls,
                "status": output.status,
                "duration_ms": output.duration_ms,
            },
        )

        return output

    # -------------------------------------------------------------------
    # Agent resolution
    # -------------------------------------------------------------------

    def _resolve_agent(self, ctx: AgentContext) -> BaseSpecialistAgent:
        """Find the best agent for this context.

        Priority:
        1. Explicit specialist from v2 goal_dict
        2. can_handle() on registered agents
        3. v2 pack resolution via registry
        4. Default to analyst
        """
        # 1. Check v2 goal_dict for explicit specialist hint
        if ctx.v2_goal_dict:
            template = (ctx.v2_goal_dict.get("metadata", {})
                       .get("goal_template", ""))
            agent = self._template_to_agent(template)
            if agent:
                return agent

        # 2. Ask agents if they can handle
        for agent in self._agents.values():
            if agent.can_handle(ctx):
                return agent

        # 3. Registry resolve (v2 pack-based)
        profile = self.registry.resolve(ctx.v2_goal_dict)
        agent_id = self._profile_to_agent_id(profile)
        if agent_id in self._agents:
            return self._agents[agent_id]

        # 4. Default
        return self._agents.get("analyst", AnalystAgent())

    def _template_to_agent(self, template: str) -> BaseSpecialistAgent | None:
        """Map v2 capability pack template to v3 agent."""
        mapping = {
            "market_research": "researcher",
            "monitoring": "researcher",
            "signup_operator": "executor",
            "publisher": "executor",
            "general": "analyst",
        }
        agent_id = mapping.get(template, "")
        return self._agents.get(agent_id)

    def _profile_to_agent_id(self, profile: SpecialistProfile) -> str:
        """Map specialist profile to agent ID."""
        worker_mapping = {
            "research_worker": "researcher",
            "browser_worker": "executor",
            "monitoring_worker": "researcher",
            "generic": "analyst",
        }
        return worker_mapping.get(profile.worker_type, "analyst")

    # -------------------------------------------------------------------
    # Query
    # -------------------------------------------------------------------

    def available_agents(self) -> list[str]:
        return list(self._agents.keys())

    def get_agent(self, agent_id: str) -> BaseSpecialistAgent | None:
        return self._agents.get(agent_id)

    def summary(self) -> list[dict[str, Any]]:
        return [
            {
                "id": a.agent_id,
                "label": a.agent_label,
                "type": type(a).__name__,
            }
            for a in self._agents.values()
        ]
