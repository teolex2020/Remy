"""
Context runtime for Remy v3.

Builds AgentContext payloads for specialist execution so context assembly is
separate from execution gating policy.
"""

from __future__ import annotations

from ..agents.base_agent import AgentContext


class ContextRuntime:
    """Deterministic builder for specialist execution context."""

    def __init__(self, *, recorder, budget, goal_context_runtime):
        self.recorder = recorder
        self.budget = budget
        self.goal_context_runtime = goal_context_runtime

    def build(
        self,
        *,
        mission,
        goal,
        step,
        specialist,
        memory_context,
        policy_hints: list[dict] | None = None,
    ) -> AgentContext:
        routing_hint = {}
        try:
            routing_hint = self.recorder.model_routing_hint(
                getattr(specialist, "id", "") or ""
            )
        except Exception:
            routing_hint = {}

        policy_hints = list(policy_hints or [])
        if policy_hints:
            memory_context = list(memory_context or []) + [
                {
                    "content": self._render_policy_hint(hint),
                    "type": "policy_hint",
                    "score": 1.0,
                    "record_id": "",
                    "policy_hint": hint,
                }
                for hint in policy_hints
            ]

        return AgentContext(
            instruction=self._instruction_with_policy(step.instruction, policy_hints),
            mission_id=mission.id,
            goal_id=goal.id if goal else "",
            goal_description=goal.description if goal else mission.description,
            plan_step_id=step.id,
            step_budget=specialist.step_budget,
            timeout_sec=specialist.timeout_sec,
            tools_allowed=specialist.tools,
            guardrails=specialist.guardrails,
            approval_mode=specialist.approval_mode,
            memory_context=memory_context,
            past_outcomes=self.recorder.recent_outcomes_summary(5),
            policy_hints=policy_hints,
            budget_remaining_usd=max(
                0, self.budget.config.daily_usd - self.budget.state.daily_spent_usd
            ),
            preferred_model=str(routing_hint.get("preferred_model") or ""),
            avoid_models=tuple(routing_hint.get("avoid_models") or ()),
            v2_goal_dict=self.goal_context_runtime.build_goal_dict(step=step, mission=mission),
        )

    def _render_policy_hint(self, hint: dict) -> str:
        name = str(hint.get("hint") or "verify_first")
        action = str(hint.get("action") or "")
        reason = str(hint.get("reason") or "")
        if name == "prefer":
            return f"[POLICY:prefer] Prior consequence supports action '{action}'. {reason}".strip()
        if name == "avoid":
            return f"[POLICY:avoid] Prior consequence refutes action '{action}'. {reason}".strip()
        if name == "requires_evidence":
            return f"[POLICY:requires_evidence] Verify with evidence before trusting action '{action}'. {reason}".strip()
        return f"[POLICY:verify_first] Verify action '{action}' before treating it as known. {reason}".strip()

    def _instruction_with_policy(self, instruction: str, policy_hints: list[dict]) -> str:
        rendered = [self._render_policy_hint(hint) for hint in policy_hints if hint]
        if not rendered:
            return instruction
        return (
            f"{instruction}\n\n"
            "Runtime policy from lived consequence memory:\n"
            + "\n".join(f"- {line}" for line in rendered)
        )
