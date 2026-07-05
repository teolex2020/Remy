"""Shared operator-facing activity state helpers."""

from __future__ import annotations


def build_mission_activity_state(
    mission_id: str,
    *,
    mission_data: dict | None = None,
    active_task_goal_records: list[dict] | None = None,
) -> dict | None:
    """Build a normalized mission activity summary for live operator surfaces."""
    mission_id = str(mission_id or "").strip()
    if not mission_id:
        return None

    mission_payload = {
        "id": mission_id,
        "description": mission_id,
    }
    if not isinstance(mission_data, dict):
        try:
            from remy.core.autonomy_goals import _load_missions

            mission_data = next(
                (mission for mission in _load_missions() if mission.get("id") == mission_id),
                None,
            )
        except Exception:
            mission_data = None

    if isinstance(mission_data, dict):
        mission_payload["description"] = str(mission_data.get("description") or mission_id)
        tasks = mission_data.get("tasks", []) if isinstance(mission_data.get("tasks"), list) else []
        if tasks:
            mission_payload["total_tasks"] = len(tasks)
            mission_payload["pending_task_items"] = [
                {
                    "goal_id": "",
                    "label": str(task.get("action") or task.get("id") or "task")[:100],
                    "status": "pending",
                    "detail": "",
                }
                for task in tasks[:3]
                if isinstance(task, dict)
            ]
            mission_payload["pending_task_labels"] = [
                str(task.get("action") or task.get("id") or "task")[:100]
                for task in tasks[:3]
                if isinstance(task, dict)
            ]

    try:
        from remy.core.agent_tools import brain, brain_lock
        from remy.core.orchestrator import get_focus_stale_cycles

        with brain_lock:
            mission_records = brain.search(query="", tags=[f"mission-{mission_id}"], limit=200)
        task_records = [
            rec for rec in (mission_records or [])
            if (getattr(rec, "metadata", None) or {}).get("mission_task_id")
        ]
        completed_tasks = sum(
            1 for rec in task_records
            if (getattr(rec, "metadata", None) or {}).get("status") == "completed"
        )
        pending_records = [
            rec for rec in task_records
            if (getattr(rec, "metadata", None) or {}).get("status") in ("pending", "active")
        ]
        blocked_tasks = sum(
            1 for rec in task_records
            if (getattr(rec, "metadata", None) or {}).get("status") in ("blocked_by_user", "blocked_external")
        )
        failed_tasks = sum(
            1 for rec in task_records
            if (getattr(rec, "metadata", None) or {}).get("status") == "failed"
        )
        queue_records = [
            rec for rec in task_records
            if (getattr(rec, "metadata", None) or {}).get("status") in (
                "active",
                "pending",
                "blocked_by_user",
                "blocked_external",
                "failed",
            )
        ]
        status_rank = {
            "active": 0,
            "pending": 1,
            "blocked_external": 2,
            "blocked_by_user": 3,
            "failed": 4,
        }
        queue_records.sort(
            key=lambda rec: status_rank.get(
                str((getattr(rec, "metadata", None) or {}).get("status") or "pending"),
                99,
            )
        )
        pending_labels = []
        pending_items = []
        for rec in queue_records:
            meta = (getattr(rec, "metadata", None) or {})
            content = str(getattr(rec, "content", "") or "").strip()
            if content.startswith("Task [") and "]: " in content:
                content = content.split("]: ", 1)[1]
            label = content[:100] or str(meta.get("mission_task_id") or "task")
            detail = str(
                meta.get("blocked_reason")
                or meta.get("status_notes")
                or meta.get("resume_context")
                or ""
            )[:140]
            pending_labels.append(label)
            pending_items.append({
                "goal_id": str(getattr(rec, "id", "") or ""),
                "label": label,
                "status": str(meta.get("status") or "pending"),
                "detail": detail,
            })

        if active_task_goal_records is not None:
            active_tasks = len(active_task_goal_records)
        else:
            active_tasks = sum(
                1 for rec in task_records
                if (getattr(rec, "metadata", None) or {}).get("status") == "active"
            )

        mission_payload["active_tasks"] = active_tasks
        mission_payload["pending_tasks"] = len(pending_records)
        mission_payload["blocked_tasks"] = blocked_tasks
        mission_payload["failed_tasks"] = failed_tasks
        mission_payload["completed_tasks"] = completed_tasks
        mission_payload["focus_stale_cycles"] = get_focus_stale_cycles(mission_id)
        if pending_labels:
            mission_payload["pending_task_labels"] = pending_labels[:3]
            mission_payload["pending_task_items"] = pending_items[:3]
        mission_payload.setdefault("total_tasks", len(task_records))
    except Exception:
        pass

    return mission_payload
