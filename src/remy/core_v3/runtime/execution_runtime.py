"""
Execution Runtime for Remy v3.

Owns delegation invocation and returns normalized execution-layer results so
ChiefAgent does not orchestrate specialist execution details inline.
"""

from __future__ import annotations

class ExecutionRuntime:
    """Execute a prepared specialist context through the delegation engine."""

    def __init__(self, delegation_engine, agent_output_runtime):
        self.delegation = delegation_engine
        self.agent_output_runtime = agent_output_runtime

    async def execute(self, *, agent_ctx):
        from remy.core.llm import model_routing_override

        with model_routing_override(
            preferred_model=getattr(agent_ctx, "preferred_model", "") or "",
            avoid_models=getattr(agent_ctx, "avoid_models", ()) or (),
        ):
            agent_output = await self.delegation.delegate(agent_ctx)
        return self.agent_output_runtime.normalize(agent_output)
