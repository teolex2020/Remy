"""
Policy Engine for Remy v3 Governance Layer.

Decides what can run silently, what requires confirmation,
and what is blocked entirely.

Phase 5: Enhanced with tool-level policies, dynamic mission rules,
cost-based escalation, and specialist constraints.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy decisions
# ---------------------------------------------------------------------------

class PolicyDecision(str, Enum):
    ALLOW = "allow"                # Run silently
    APPROVE = "approve"            # Requires human approval
    DENY = "deny"                  # Blocked entirely


class RiskCategory(str, Enum):
    SAFE = "safe"                  # Read-only, no external effects
    LOW = "low"                    # Reversible, low cost
    MEDIUM = "medium"              # External effects, moderate cost
    HIGH = "high"                  # Financial, publishing, irreversible
    CRITICAL = "critical"          # System-altering, security-sensitive


# ---------------------------------------------------------------------------
# Policy rule
# ---------------------------------------------------------------------------

@dataclass
class PolicyRule:
    """A single governance rule."""
    id: str = ""
    description: str = ""
    action_pattern: str = ""       # fnmatch pattern for action names
    risk_category: RiskCategory = RiskCategory.LOW
    decision: PolicyDecision = PolicyDecision.ALLOW

    # Conditions
    max_cost_usd: float = 0.0      # If cost > this, escalate
    requires_evidence: bool = False
    allowed_specialists: list[str] = field(default_factory=list)
    blocked_tools: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)

    # Scope
    mission_ids: list[str] = field(default_factory=list)  # Empty = all missions
    enabled: bool = True


# ---------------------------------------------------------------------------
# Default policies
# ---------------------------------------------------------------------------

DEFAULT_POLICIES: list[PolicyRule] = [
    PolicyRule(
        id="financial_actions",
        description="Any action involving money requires approval",
        action_pattern="financial_*",
        risk_category=RiskCategory.HIGH,
        decision=PolicyDecision.APPROVE,
        max_cost_usd=0.0,
    ),
    PolicyRule(
        id="wallet_operations",
        description="Wallet transfers always require approval",
        action_pattern="wallet_*",
        risk_category=RiskCategory.CRITICAL,
        decision=PolicyDecision.APPROVE,
    ),
    PolicyRule(
        id="publishing",
        description="Publishing content requires approval",
        action_pattern="publish_*",
        risk_category=RiskCategory.HIGH,
        decision=PolicyDecision.APPROVE,
    ),
    PolicyRule(
        id="browser_signup",
        description="Account creation requires approval",
        action_pattern="signup_*",
        risk_category=RiskCategory.MEDIUM,
        decision=PolicyDecision.APPROVE,
    ),
    PolicyRule(
        id="research",
        description="Research is safe to run autonomously",
        action_pattern="research_*",
        risk_category=RiskCategory.SAFE,
        decision=PolicyDecision.ALLOW,
    ),
    PolicyRule(
        id="memory_ops",
        description="Memory operations are safe",
        action_pattern="memory_*",
        risk_category=RiskCategory.SAFE,
        decision=PolicyDecision.ALLOW,
    ),
    PolicyRule(
        id="analysis",
        description="Analysis tasks are safe",
        action_pattern="analy*",
        risk_category=RiskCategory.SAFE,
        decision=PolicyDecision.ALLOW,
    ),
    PolicyRule(
        id="browser_navigation",
        description="Browsing is low risk but trackable",
        action_pattern="browse_*",
        risk_category=RiskCategory.LOW,
        decision=PolicyDecision.ALLOW,
    ),
    PolicyRule(
        id="high_cost_guard",
        description="Any action costing over $0.10 requires approval",
        action_pattern="*",
        risk_category=RiskCategory.MEDIUM,
        decision=PolicyDecision.ALLOW,
        max_cost_usd=0.10,
    ),
]

# Tools that are always blocked (safety guardrails)
BLOCKED_TOOLS = frozenset({
    "delete_all_memory",
    "factory_reset",
    "transfer_all_funds",
})

# Tools that always require approval
APPROVAL_TOOLS = frozenset({
    "send_transaction",
    "publish_content",
    "create_account",
    "send_email",
    "send_telegram",
})


# ---------------------------------------------------------------------------
# Policy Engine
# ---------------------------------------------------------------------------

class PolicyEngine:
    """Evaluates actions against governance rules."""

    def __init__(self, rules: list[PolicyRule] | None = None):
        self.rules = list(rules or DEFAULT_POLICIES)

    def evaluate(
        self,
        action: str,
        cost_usd: float = 0.0,
        specialist: str = "",
        tools: list[str] | None = None,
        mission_id: str = "",
    ) -> tuple[PolicyDecision, str]:
        """Evaluate an action against policies.

        Returns:
            (decision, reason)
        """
        tools = tools or []

        # Hard guardrails: blocked tools
        blocked = set(tools) & BLOCKED_TOOLS
        if blocked:
            return PolicyDecision.DENY, f"Tools {blocked} are permanently blocked"

        # Hard guardrails: approval-required tools
        needs_approval = set(tools) & APPROVAL_TOOLS
        if needs_approval:
            return PolicyDecision.APPROVE, (
                f"Tools {needs_approval} require human approval"
            )

        # Check rules in order (first match wins)
        for rule in self.rules:
            if not rule.enabled:
                continue
            if not self._matches(rule, action, mission_id):
                continue

            # Cost escalation
            if rule.max_cost_usd > 0 and cost_usd > rule.max_cost_usd:
                return PolicyDecision.APPROVE, (
                    f"Cost ${cost_usd:.2f} exceeds policy limit "
                    f"${rule.max_cost_usd:.2f} for {rule.id}"
                )

            # Blocked tools
            if rule.blocked_tools:
                blocked = set(tools) & set(rule.blocked_tools)
                if blocked:
                    return PolicyDecision.DENY, (
                        f"Tools {blocked} blocked by policy {rule.id}"
                    )

            # Allowed tools filter
            if rule.allowed_tools and tools:
                forbidden = set(tools) - set(rule.allowed_tools)
                if forbidden:
                    return PolicyDecision.DENY, (
                        f"Tools {forbidden} not in allowlist for {rule.id}"
                    )

            # Specialist check
            if rule.allowed_specialists and specialist:
                if specialist not in rule.allowed_specialists:
                    return PolicyDecision.DENY, (
                        f"Specialist '{specialist}' not allowed by {rule.id}"
                    )

            return rule.decision, rule.description

        # Default: allow
        return PolicyDecision.ALLOW, "no matching policy"

    def evaluate_tool(self, tool_name: str) -> tuple[PolicyDecision, str]:
        """Quick check: is this specific tool allowed?"""
        if tool_name in BLOCKED_TOOLS:
            return PolicyDecision.DENY, f"Tool '{tool_name}' is permanently blocked"
        if tool_name in APPROVAL_TOOLS:
            return PolicyDecision.APPROVE, f"Tool '{tool_name}' requires approval"
        return PolicyDecision.ALLOW, "tool allowed"

    def _matches(self, rule: PolicyRule, action: str, mission_id: str = "") -> bool:
        """Match rule against action using fnmatch patterns."""
        # Mission scope check
        if rule.mission_ids and mission_id:
            if mission_id not in rule.mission_ids:
                return False

        return fnmatch.fnmatch(action, rule.action_pattern)

    def add_rule(self, rule: PolicyRule, priority: int = 0):
        """Add a rule at given priority (0 = highest)."""
        self.rules.insert(priority, rule)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID."""
        for i, r in enumerate(self.rules):
            if r.id == rule_id:
                self.rules.pop(i)
                return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """Temporarily disable a rule."""
        for r in self.rules:
            if r.id == rule_id:
                r.enabled = False
                return True
        return False

    def get_rules(self) -> list[dict[str, Any]]:
        return [
            {
                "id": r.id,
                "description": r.description,
                "pattern": r.action_pattern,
                "risk": r.risk_category.value,
                "decision": r.decision.value,
                "enabled": r.enabled,
            }
            for r in self.rules
        ]

    def rules_for_action(self, action: str) -> list[PolicyRule]:
        """Return all rules matching an action (for debugging)."""
        return [r for r in self.rules if r.enabled and fnmatch.fnmatch(action, r.action_pattern)]
