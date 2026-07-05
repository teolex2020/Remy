"""
Autonomy Outcomes — action tracking and outcome recall.

ActionRecord dataclass + record_outcome + recall_similar_outcomes extracted from autonomy.py.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logger = logging.getLogger("Autonomy")


def _time_bucket_tags(timestamp_iso: str) -> list[str]:
    """Return week and month bucket tags for a given ISO timestamp."""
    try:
        dt = datetime.fromisoformat(timestamp_iso)
    except (ValueError, TypeError):
        dt = datetime.now()
    week = dt.strftime("week:%Y-W%W")   # e.g. week:2026-W11
    month = dt.strftime("month:%Y-%m")  # e.g. month:2026-03
    return [week, month]


def _get_autonomy():
    """Lazy accessor — reads from autonomy module (supports test patching)."""
    import remy.core.autonomy as _au

    return _au


@dataclass
class ActionRecord:
    """A single autonomous action taken by the agent."""

    action_id: str
    timestamp: str
    goal_id: Optional[str]
    action_type: str  # "agent_invoke" | "tool_call" | "research"
    description: str
    result: str
    success: bool
    tokens_used: int
    duration_ms: int
    turn_class: str = ""  # "productive" | "maintenance" | "idle"


def record_outcome(
    action: ActionRecord,
    goal_record_id: str | None = None,
) -> str:
    """Store an action outcome in brain for future learning. Returns record ID."""
    from remy.core.agent_tools import Level, brain_lock
    from remy.core.provenance import _stamp_provenance

    au = _get_autonomy()
    brain = au.brain

    content = (
        f"Action outcome [{action.action_type}]: {action.description[:100]}\n"
        f"Result: {'SUCCESS' if action.success else 'FAILURE'} - {action.result[:200]}\n"
        f"Tokens used: {action.tokens_used}"
    )

    tags = ["autonomous-outcome", action.action_type]
    if action.success:
        tags.append("outcome-success")
    else:
        tags.append("outcome-failure")
    tags.extend(_time_bucket_tags(action.timestamp))

    with brain_lock:
        # Compute running confidence for this action_type before storing
        prior = brain.search(query="", tags=["autonomous-outcome", action.action_type], limit=100)
        if prior:
            successes = sum(1 for r in prior if (r.metadata or {}).get("success") is True)
            confidence = round((successes + (1 if action.success else 0)) / (len(prior) + 1), 2)
        else:
            confidence = 1.0 if action.success else 0.0

        rec = brain.store(
            content=content,
            level=Level.DOMAIN,
            tags=tags,
            metadata=_stamp_provenance(
                {
                    "type": "autonomous_outcome",
                    "action_id": action.action_id,
                    "goal_id": action.goal_id,
                    "action_type": action.action_type,
                    "success": action.success,
                    "turn_class": action.turn_class,
                    "confidence": confidence,
                    "tokens_used": action.tokens_used,
                    "duration_ms": action.duration_ms,
                    "timestamp": action.timestamp,
                },
                "autonomous",
                tags=tags,
            ),
            auto_promote=False,
        )

        # Connect outcome to goal (promotion-gated)
        if goal_record_id:
            from remy.core.agent_tools import gated_connect
            connected = gated_connect(brain, rec.id, goal_record_id, weight=0.8)
            if not connected:
                # Outcome records are reflections, so factual-promotion policy can
                # block them. Keep a traceability edge for goal history navigation.
                try:
                    brain.connect(rec.id, goal_record_id, weight=0.8)
                except TypeError:
                    brain.connect(rec.id, goal_record_id, 0.8)
                except Exception:
                    pass

            goal_rec = brain.get(goal_record_id)
            if goal_rec:
                meta = dict(goal_rec.metadata or {})
                outcome_ids = meta.get("outcome_ids", [])
                outcome_ids.append(rec.id)
                meta["outcome_ids"] = outcome_ids[-10:]  # Keep last 10
                brain.update(goal_record_id, metadata=meta)

        tool_name = action.action_type or "unknown"
        try:
            if action.success:
                brain.record_tool_success(tool_name)
            else:
                brain.record_tool_failure(tool_name)
        except Exception:
            pass

    return rec.id


def recall_similar_outcomes(description: str, limit: int = 5) -> list[dict]:
    """Recall past outcomes similar to a planned action, for learning."""
    from remy.core.agent_tools import brain_lock

    au = _get_autonomy()
    brain = au.brain

    with brain_lock:
        results = brain.search(
            query=description,
            tags=["autonomous-outcome"],
            limit=limit,
        )
    return [
        {
            "content": r.content[:200],
            "success": r.metadata.get("success") if r.metadata else None,
            "action_type": r.metadata.get("action_type") if r.metadata else None,
            "confidence": r.metadata.get("confidence") if r.metadata else None,
        }
        for r in results
    ]
