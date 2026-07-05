"""
Governed gateway for integration plugins.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..governance.approval_engine import ApprovalEngine
from ..governance.audit_engine import AuditEngine, EventType
from ..governance.budget_engine import BudgetAction, BudgetEngine
from ..governance.policy_engine import PolicyDecision, PolicyEngine
from .auth_store import AuthStore
from .contracts import IntegrationDecision, PluginContext, PluginRequest, PluginResult
from .health import IntegrationHealthBook
from .registry import IntegrationRegistry
from .router import choose_mode


@dataclass
class GatewayOutcome:
    decision: IntegrationDecision
    result: PluginResult | None = None
    reason: str = ""
    approval_id: str = ""


class IntegrationGateway:
    def __init__(
        self,
        *,
        registry: IntegrationRegistry,
        policy: PolicyEngine,
        budget: BudgetEngine,
        approval: ApprovalEngine,
        audit: AuditEngine,
        auth_store: AuthStore | None = None,
        health: IntegrationHealthBook | None = None,
    ):
        self.registry = registry
        self.policy = policy
        self.budget = budget
        self.approval = approval
        self.audit = audit
        self.auth_store = auth_store or AuthStore()
        self.health = health or IntegrationHealthBook()

    def execute(
        self,
        plugin_id: str,
        request: PluginRequest,
        *,
        ctx: PluginContext | None = None,
    ) -> GatewayOutcome:
        ctx = ctx or PluginContext()
        plugin = self.registry.get(plugin_id)
        if plugin is None:
            return GatewayOutcome(IntegrationDecision.BLOCKED, reason=f"Unknown plugin: {plugin_id}")
        if not plugin.supports(request.action):
            return GatewayOutcome(IntegrationDecision.BLOCKED, reason=f"{plugin_id} does not support {request.action}")

        request.mode = choose_mode(plugin.default_mode, request)

        manual_reason = plugin.requires_manual_assist(request)
        if manual_reason:
            self.audit.log_event(
                EventType.POLICY_VIOLATION,
                request.action,
                actor=ctx.actor,
                mission_id=ctx.mission_id,
                details={"plugin_id": plugin_id, "reason": manual_reason},
                risk_level=plugin.risk_level,
            )
            return GatewayOutcome(
                IntegrationDecision.MANUAL_ASSIST_REQUIRED,
                result=PluginResult(
                    ok=False,
                    status="manual_assist_required",
                    message=manual_reason,
                    requires_human=True,
                    manual_reason=manual_reason,
                ),
                reason=manual_reason,
            )

        cost_estimate = request.cost_estimate_usd or plugin.estimate_cost(request.action, request.payload)
        budget_action, budget_reason = self.budget.check_budget(cost_estimate, mission_id=ctx.mission_id)
        if budget_action == BudgetAction.DENY:
            return GatewayOutcome(IntegrationDecision.BLOCKED, reason=budget_reason)

        policy_tools = request.tools or [self._tool_name_for(plugin_id, request.action)]
        policy_decision, policy_reason = self.policy.evaluate(
            f"integration.{plugin_id}.{request.action}",
            cost_usd=cost_estimate,
            specialist=ctx.specialist or ctx.actor,
            tools=policy_tools,
            mission_id=ctx.mission_id,
        )
        if policy_decision == PolicyDecision.DENY:
            self.audit.log_event(
                EventType.POLICY_VIOLATION,
                request.action,
                actor=ctx.actor,
                mission_id=ctx.mission_id,
                details={"plugin_id": plugin_id, "reason": policy_reason},
                risk_level=plugin.risk_level,
            )
            return GatewayOutcome(IntegrationDecision.BLOCKED, reason=policy_reason)
        if policy_decision == PolicyDecision.APPROVE:
            approval = self.approval.request_approval(
                action=request.action,
                description=f"{plugin.label}: {request.action}",
                mission_id=ctx.mission_id,
                specialist=ctx.specialist or ctx.actor,
                risk_category=plugin.risk_level,
                cost_usd=cost_estimate,
                context={"plugin_id": plugin_id, "payload": request.payload},
            )
            self.audit.log_event(
                EventType.APPROVAL_REQUESTED,
                request.action,
                actor=ctx.actor,
                mission_id=ctx.mission_id,
                details={"plugin_id": plugin_id, "approval_id": approval.id},
                risk_level=plugin.risk_level,
                cost_usd=cost_estimate,
            )
            return GatewayOutcome(
                IntegrationDecision.APPROVAL_REQUIRED,
                reason=policy_reason,
                approval_id=approval.id,
            )

        start = time.perf_counter()
        result = plugin.execute(request, ctx)
        latency_ms = int((time.perf_counter() - start) * 1000)
        self.health.record(plugin_id, ok=result.ok, latency_ms=latency_ms, error=result.message if not result.ok else "")
        self.audit.log_event(
            EventType.STEP_EXECUTED if result.ok else EventType.STEP_FAILED,
            request.action,
            actor=ctx.actor,
            mission_id=ctx.mission_id,
            details={"plugin_id": plugin_id, "status": result.status},
            risk_level=plugin.risk_level,
            cost_usd=result.cost_usd or cost_estimate,
        )
        if result.ok:
            self.budget.record_spend(
                result.cost_usd or cost_estimate,
                mission_id=ctx.mission_id,
                specialist=ctx.specialist or ctx.actor,
                action=request.action,
                model=request.mode.value,
            )
        return GatewayOutcome(IntegrationDecision.ALLOW, result=result)

    @staticmethod
    def _tool_name_for(plugin_id: str, action: str) -> str:
        if plugin_id == "email_send":
            return "send_email"
        if plugin_id == "telegram":
            return "send_telegram"
        if action in {"github.create_account", "browser.create_account", "signup.create_account"}:
            return "create_account"
        return action.replace(".", "_")
