"""
Goal context runtime for Remy v3.

Builds v2-compatible goal dict payloads from plan-step and mission context.
"""

from __future__ import annotations


class GoalContextRuntime:
    """Build compatibility goal dicts for specialist and context bridges."""

    def build_goal_dict(self, *, step, mission) -> dict:
        return {
            "content": step.instruction,
            "metadata": {
                "status": "active",
                "priority": str(mission.priority),
                "goal_template": step.specialist or "",
                "mission_id": mission.id,
            },
            "tags": ["goal", "v3_step"],
        }
