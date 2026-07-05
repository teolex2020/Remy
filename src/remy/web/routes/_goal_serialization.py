"""Shared serializers for autonomous goal records across web routes."""

from __future__ import annotations

import re


def _goal_display_context(rec) -> dict:
    meta = rec.metadata or {}
    goal_template = meta.get("goal_template")

    cap_pack = None
    publisher_mode = ""
    publisher_channel = ""
    approval_mode = ""
    research_config = {}
    research_session = None
    if goal_template:
        try:
            from remy.core.capability_packs import (
                get_pack,
                infer_publisher_channel,
                infer_publisher_mode,
                pack_summary,
                resolve_research_config,
            )
            from remy.core.research_sessions import get_research_session_trace

            pack = get_pack(goal_template)
            research_config = resolve_research_config(
                {
                    "goal_template": goal_template,
                    "research_mode": meta.get("research_mode", ""),
                    "source_scope": meta.get("source_scope", ""),
                    "source_domains": meta.get("source_domains", []),
                    "citation_required": meta.get("citation_required", None),
                }
            )
            research_session = get_research_session_trace(meta.get("goal_id", "") or rec.id)
            if pack.id != "general":
                cap_pack = pack_summary(pack)
                approval_mode = pack.approval_mode
                if pack.id == "publisher":
                    goal_ctx = {
                        "description": rec.content,
                        "goal_template": goal_template,
                        "task_action": meta.get("task_action", ""),
                        "task_done_when": meta.get("task_done_when", ""),
                        "target_url": meta.get("target_url", ""),
                        "url": meta.get("target_url", ""),
                        "publisher_mode": meta.get("publisher_mode", ""),
                        "publisher_channel": meta.get("publisher_channel", ""),
                    }
                    publisher_mode = infer_publisher_mode(goal_ctx)
                    publisher_channel = infer_publisher_channel(goal_ctx)
        except Exception:
            pass

    return {
        "goal_template": goal_template,
        "capability_pack": cap_pack,
        "approval_mode": approval_mode,
        "publisher_mode": publisher_mode,
        "publisher_channel": publisher_channel,
        "research_mode": research_config.get("research_mode", ""),
        "source_scope": research_config.get("source_scope", ""),
        "source_domains": research_config.get("source_domains", []),
        "citation_required": research_config.get("citation_required", False),
        "research_session": research_session,
        "accepted_sources_count": (research_session or {}).get("accepted_sources_count", 0)
        if research_session
        else 0,
        "rejected_sources_count": (research_session or {}).get("rejected_sources_count", 0)
        if research_session
        else 0,
        "contradictions_count": (research_session or {}).get("contradictions_count", 0)
        if research_session
        else 0,
        "citation_coverage_rate": (research_session or {}).get("citation_coverage_rate", 0.0)
        if research_session
        else 0.0,
    }


def serialize_goal_record(rec) -> dict:
    meta = rec.metadata or {}
    context = _goal_display_context(rec)
    result = {
        "id": rec.id,
        "type": "goal",
        "content": rec.content,
        "status": meta.get("status", "active"),
        "priority": meta.get("priority", "medium"),
        "goal_type": meta.get("goal_type", "general"),
        "goal_template": context["goal_template"],
        "approval_mode": context["approval_mode"],
        "publisher_mode": context["publisher_mode"],
        "publisher_channel": context["publisher_channel"],
        "attempts": meta.get("attempts", 0),
        "blocked_reason": meta.get("blocked_reason", ""),
        "blocked_evidence": meta.get("blocked_evidence", ""),
        "resume_context": meta.get("resume_context", ""),
        "created_at": meta.get("created_at", ""),
        "timestamp": meta.get("updated_at") or meta.get("created_at", ""),
        "research_mode": context["research_mode"],
        "source_scope": context["source_scope"],
        "source_domains": context["source_domains"],
        "citation_required": context["citation_required"],
        "research_session": context["research_session"],
        "accepted_sources_count": context["accepted_sources_count"],
        "rejected_sources_count": context["rejected_sources_count"],
        "contradictions_count": context["contradictions_count"],
        "citation_coverage_rate": context["citation_coverage_rate"],
    }
    if context["capability_pack"]:
        result["capability_pack"] = context["capability_pack"]
        try:
            from remy.core.task_metrics import task_metrics

            fm = task_metrics.get_family(context["capability_pack"]["metrics_family"])
            if fm.get("total_cycles", 0) > 0:
                result["pack_metrics"] = fm
        except Exception:
            pass
    return result


def serialize_goal_as_todo(rec) -> dict | None:
    meta = rec.metadata or {}
    if meta.get("type") != "autonomous_goal":
        return None

    g_status = meta.get("status", "active")
    if g_status in ("active", "decomposed"):
        mapped = "in_progress"
    elif g_status in ("pending",):
        mapped = "pending"
    elif g_status in ("blocked_external", "blocked_by_user"):
        mapped = "in_progress"
    elif g_status in ("completed",):
        mapped = "done"
    else:
        return None

    context = _goal_display_context(rec)
    content = rec.content or ""
    title_match = re.match(r"Goal\s*\[[A-Z]+\]:\s*(.+)", content)
    title = title_match.group(1).strip() if title_match else content
    if len(title) > 120:
        title = title[:117] + "..."

    return {
        "id": rec.id,
        "todo_id": meta.get("goal_id"),
        "title": title,
        "priority": meta.get("priority", "medium"),
        "status": mapped,
        "category": "agent",
        "goal_type": meta.get("goal_type", "general"),
        "goal_template": context["goal_template"],
        "capability_pack": context["capability_pack"],
        "approval_mode": context["approval_mode"],
        "publisher_mode": context["publisher_mode"],
        "publisher_channel": context["publisher_channel"],
        "research_mode": context["research_mode"],
        "source_scope": context["source_scope"],
        "source_domains": context["source_domains"],
        "citation_required": context["citation_required"],
        "research_session": context["research_session"],
        "accepted_sources_count": context["accepted_sources_count"],
        "rejected_sources_count": context["rejected_sources_count"],
        "contradictions_count": context["contradictions_count"],
        "citation_coverage_rate": context["citation_coverage_rate"],
        "mission_id": meta.get("mission_id", ""),
        "mission_task_id": meta.get("mission_task_id", ""),
        "block_status": g_status if g_status.startswith("blocked") else "",
        "blocked_action_id": meta.get("blocked_action_id", ""),
        "blocked_reason": meta.get("blocked_reason", ""),
        "blocked_evidence": meta.get("blocked_evidence", ""),
        "resume_context": meta.get("resume_context", ""),
        "task_action": meta.get("task_action", ""),
        "task_depends_on": meta.get("task_depends_on", ""),
        "raw_status": g_status,
        "due_date": meta.get("deadline"),
        "repeat": None,
        "repeat_until": None,
        "last_completed_at": None,
        "created_by": meta.get("created_by", "system"),
        "created_at": meta.get("created_at"),
        "started_at": meta.get("last_attempt"),
        "updated_at": meta.get("updated_at"),
        "completed_at": meta.get("updated_at") if g_status == "completed" else None,
        "parent_todo_id": meta.get("parent_goal_id"),
        "source": "goal",
        "attempts": meta.get("attempts", 0),
        "immortal": meta.get("immortal", False),
    }


def serialize_goal_as_calendar_task(rec) -> dict | None:
    meta = rec.metadata or {}
    if meta.get("type") != "autonomous_goal":
        return None

    g_status = meta.get("status", "active")
    if g_status in ("active", "decomposed"):
        mapped = "active"
    elif g_status == "completed":
        mapped = "completed"
    else:
        return None

    date = meta.get("deadline")
    if not date:
        created = meta.get("created_at", "")
        date = created[:10] if len(created) >= 10 else None
    if not date:
        return None

    content = rec.content or ""
    title_match = re.match(r"Goal\s*\[[A-Z]+\]:\s*(.+)", content)
    title = title_match.group(1).strip() if title_match else content
    if len(title) > 120:
        title = title[:117] + "..."

    return {
        "id": rec.id,
        "description": title,
        "due_date": date,
        "repeat": meta.get("task_repeat"),
        "status": mapped,
        "source": "goal",
        "content": content,
    }
