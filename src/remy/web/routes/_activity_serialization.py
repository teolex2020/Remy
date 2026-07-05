"""Shared serializers and builders for activity payloads."""

from __future__ import annotations

from remy.web.routes._goal_serialization import serialize_goal_record


def serialize_outcome_record(rec) -> dict:
    meta = rec.metadata or {}
    return {
        "id": rec.id,
        "type": "outcome",
        "content": rec.content[:200],
        "success": meta.get("success", True),
        "action_type": meta.get("action_type", "unknown"),
        "tokens_used": meta.get("tokens_used", 0),
        "duration_ms": meta.get("duration_ms", 0),
        "goal_id": meta.get("goal_id"),
        "timestamp": meta.get("timestamp", ""),
    }


def serialize_reflection_record(rec) -> dict:
    meta = rec.metadata or {}
    return {
        "id": rec.id,
        "type": "reflection",
        "content": rec.content,
        "session_id": meta.get("session_id", ""),
        "action_count": meta.get("action_count", 0),
        "timestamp": meta.get("timestamp", ""),
    }


def serialize_proactive_record(rec) -> dict:
    meta = rec.metadata or {}
    return {
        "id": rec.id,
        "type": "proactive",
        "content": rec.content[:200],
        "trigger_reason": meta.get("trigger_reason", ""),
        "trigger_context": meta.get("trigger_context", ""),
        "timestamp": meta.get("timestamp", ""),
    }


def build_activity_payload(goals, outcomes, reflections, proactive) -> dict:
    success = sum(1 for o in outcomes if (o.metadata or {}).get("success"))
    failure = len(outcomes) - success
    total_tokens = sum((o.metadata or {}).get("tokens_used", 0) for o in outcomes)

    return {
        "summary": {
            "total_actions": len(outcomes),
            "success": success,
            "failure": failure,
            "success_rate": round(success / max(len(outcomes), 1) * 100),
            "total_tokens": total_tokens,
            "active_goals": sum(1 for g in goals if (g.metadata or {}).get("status") == "active"),
            "blocked_external_goals": sum(
                1 for g in goals if (g.metadata or {}).get("status") == "blocked_external"
            ),
            "blocked_user_goals": sum(
                1 for g in goals if (g.metadata or {}).get("status") == "blocked_by_user"
            ),
            "completed_goals": sum(
                1 for g in goals if (g.metadata or {}).get("status") == "completed"
            ),
        },
        "goals": [serialize_goal_record(g) for g in goals],
        "outcomes": [serialize_outcome_record(o) for o in outcomes],
        "reflections": [serialize_reflection_record(r) for r in reflections],
        "proactive": [serialize_proactive_record(p) for p in proactive],
    }
