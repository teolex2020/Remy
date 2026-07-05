"""
Todo routes — CRUD for todo items stored in brain.
"""

import logging
import re
from datetime import datetime

from fastapi import APIRouter, HTTPException

from remy.web.routes._goal_serialization import serialize_goal_as_todo
from remy.web.routes._helpers import _get_api, run_in_thread

logger = logging.getLogger("WebAPI")

router = APIRouter()


@router.get("/todos")
async def list_todos(
    status: str = "active",
    category: str | None = None,
    days: int | None = None,
    limit: int = 50,
):
    """List todo items + autonomous goals. status: active/done/all. days: filter by created_at (1/3/7/30)."""
    api = _get_api()

    cutoff = None
    if days is not None and days > 0:
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=days)

    def _parse_dt(value: str | None):
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value)
            # Strip timezone to keep naive (cutoff is naive from datetime.now())
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            return dt
        except (ValueError, TypeError):
            return None

    def _query():
        with api.brain_lock:
            todos = api.brain.search(query="", tags=["todo-item"], limit=200)
            reminders = api.brain.search(query="", tags=["scheduled-task"], limit=200)
            goals = api.brain.search(query="", tags=["autonomous-goal"], limit=50)
        return todos, reminders, goals

    recs, reminder_recs, goal_recs = await run_in_thread(_query)
    items = []

    # --- Todo items ---
    for r in recs:
        meta = r.metadata or {}
        if meta.get("type") != "todo_item":
            continue
        s = meta.get("status", "pending")
        if status == "active" and s not in ("pending", "in_progress"):
            continue
        if status == "done" and s != "done":
            continue
        if category and category != "agent" and meta.get("category") != category:
            continue
        # Time period filter
        if cutoff:
            created_dt = _parse_dt(meta.get("created_at"))
            if created_dt and created_dt < cutoff:
                continue
        title = meta.get("title")
        if not title:
            content = r.content or ""
            m = re.match(r"Todo\s*\[[A-Z]+\]:\s*(.+?)(?:\s*\|\s*Due:.*)?$", content)
            title = m.group(1).strip() if m else content
        items.append(
            {
                "id": r.id,
                "todo_id": meta.get("todo_id"),
                "title": title,
                "priority": meta.get("priority", "medium"),
                "status": s,
                "category": meta.get("category", "personal"),
                "due_date": meta.get("due_date"),
                "repeat": meta.get("repeat"),
                "repeat_until": meta.get("repeat_until"),
                "last_completed_at": meta.get("last_completed_at"),
                "created_by": meta.get("created_by", "user"),
                "created_at": meta.get("created_at"),
                "started_at": meta.get("started_at"),
                "completed_at": meta.get("completed_at"),
                "parent_todo_id": meta.get("parent_todo_id"),
                "source": "todo",
            }
        )

    # --- Scheduled reminders --- 
    for r in reminder_recs:
        meta = r.metadata or {}
        if meta.get("type") != "scheduled_task":
            continue
        s = meta.get("status", "active")
        mapped = "done" if s == "done" else "pending"
        if status == "active" and mapped not in ("pending", "in_progress"):
            continue
        if status == "done" and mapped != "done":
            continue
        if category and category not in ("agent", "personal", "reminder"):
            continue
        if cutoff:
            created_dt = _parse_dt(meta.get("updated_at")) or _parse_dt(meta.get("timestamp"))
            if created_dt and created_dt < cutoff:
                continue

        title = meta.get("description") or r.content
        items.append(
            {
                "id": r.id,
                "todo_id": meta.get("task_id") or r.id,
                "title": title,
                "priority": meta.get("priority", "medium"),
                "status": mapped,
                "category": "reminder",
                "due_date": meta.get("due_date"),
                "repeat": meta.get("repeat"),
                "cron": meta.get("cron"),
                "repeat_until": meta.get("repeat_until"),
                "last_completed_at": meta.get("last_completed_at"),
                "created_by": "agent" if str(meta.get("source", "")).startswith("agent") else "user",
                "created_at": meta.get("timestamp") or meta.get("created_at"),
                "started_at": meta.get("started_at"),
                "completed_at": meta.get("completed_at"),
                "parent_todo_id": None,
                "source": "reminder",
            }
        )

    # --- Autonomous goals (shown as read-only items in "agent" category) ---
    if category is None or category == "agent":
        for g in goal_recs:
            meta = g.metadata or {}
            item = serialize_goal_as_todo(g)
            if not item:
                continue
            if status == "active" and item["status"] not in ("pending", "in_progress"):
                continue
            if status == "done" and item["status"] != "done":
                continue
            if cutoff:
                relevant_dt = (
                    _parse_dt(meta.get("updated_at"))
                    or _parse_dt(meta.get("last_attempt"))
                    or _parse_dt(meta.get("completed_at"))
                    or _parse_dt(meta.get("created_at"))
                )
                if relevant_dt and relevant_dt < cutoff:
                    continue
            items.append(item)

    priority_order = {"critical": -1, "high": 0, "medium": 1, "low": 2}
    status_order = {"in_progress": 0, "pending": 1, "done": 2, "archived": 3}
    items.sort(
        key=lambda x: (status_order.get(x["status"], 9), priority_order.get(x["priority"], 9))
    )
    return {"todos": items[:limit], "total": len(items)}


@router.post("/todos")
async def create_todo(body: dict):
    """Create a new todo item from the web UI."""
    import uuid as _uuid

    from remy.core.agent_tools import Level

    api = _get_api()
    title = (body.get("title") or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title is required")

    priority = (body.get("priority") or "medium").lower()
    if priority not in ("high", "medium", "low"):
        priority = "medium"
    due_date = (body.get("due_date") or "").strip() or None
    category = (body.get("category") or "personal").lower().strip() or "personal"
    repeat = (body.get("repeat") or "").strip().lower()
    if repeat and repeat not in ("daily", "weekly", "monthly"):
        repeat = ""
    repeat_until = (body.get("repeat_until") or "").strip() or None

    todo_id = f"todo-{_uuid.uuid4().hex[:12]}"
    tags = ["todo-item", f"cat-{category}", f"priority-{priority}"]
    if repeat:
        tags.append(f"repeat-{repeat}")

    content = f"Todo [{priority.upper()}]: {title}"
    if due_date:
        content += f" | Due: {due_date}"
    if repeat:
        content += f" | Repeats: {repeat}"

    meta = {
        "type": "todo_item",
        "todo_id": todo_id,
        "title": title,
        "priority": priority,
        "status": "pending",
        "category": category,
        "due_date": due_date,
        "repeat": repeat or None,
        "repeat_until": repeat_until,
        "created_by": "user",
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
    }

    def _store():
        with api.brain_lock:
            return api.brain.store(content=content, level=Level.DOMAIN, tags=tags, metadata=meta)

    rec = await run_in_thread(_store)
    return {
        "id": rec.id,
        "todo_id": todo_id,
        "title": title,
        "priority": priority,
        "category": category,
    }


@router.post("/todos/{record_id}/toggle")
async def toggle_todo(record_id: str):
    """Toggle a todo between done and pending."""
    api = _get_api()

    def _toggle():
        from datetime import timedelta

        with api.brain_lock:
            rec = api.brain.get(record_id)
        if not rec:
            return "not_found", None
        meta = rec.metadata or {}
        if meta.get("type") not in ("todo_item", "scheduled_task"):
            return "not_todo", None
        if meta.get("type") == "scheduled_task":
            current = meta.get("status", "active")
            repeat = meta.get("repeat")
            if current == "done":
                meta["status"] = "active"
                meta["completed_at"] = None
            else:
                if repeat:
                    old_due = meta.get("due_date")
                    try:
                        base = datetime.fromisoformat(old_due.replace("Z", "+00:00")) if old_due else datetime.now()
                    except Exception:
                        base = datetime.now()
                    if repeat == "daily":
                        next_due = base + timedelta(days=1)
                    elif repeat == "weekly":
                        next_due = base + timedelta(weeks=1)
                    elif repeat == "monthly":
                        month = base.month + 1
                        year = base.year
                        if month > 12:
                            month = 1
                            year += 1
                        try:
                            next_due = base.replace(year=year, month=month)
                        except ValueError:
                            next_due = base.replace(year=year, month=month, day=28)
                    else:
                        next_due = None
                    meta["last_completed_at"] = datetime.now().isoformat()
                    meta["status"] = "active" if next_due else "done"
                    meta["due_date"] = next_due.isoformat() if next_due else old_due
                else:
                    meta["status"] = "done"
                    meta["completed_at"] = datetime.now().isoformat()
            meta["updated_at"] = datetime.now().isoformat()

            title = meta.get("description") or rec.content
            content = f"Scheduled: {title}"
            if meta.get("due_date"):
                content += f" | Due: {meta['due_date']}"
            if meta.get("repeat"):
                content += f" | Repeats: {meta['repeat']}"
            if meta.get("cron"):
                content += f" | Cron: {meta['cron']}"
            if meta.get("status") == "done":
                content += " [DONE]"

            with api.brain_lock:
                api.brain.update(record_id, content=content, metadata=meta)
            return "ok", ("done" if meta["status"] == "done" else "pending")

        current = meta.get("status", "pending")
        if current == "done":
            new_status = "pending"
            meta["completed_at"] = None
            meta["started_at"] = None
        else:
            # Check if recurring task
            repeat = meta.get("repeat")
            if repeat:
                old_due = meta.get("due_date")
                base = datetime.fromisoformat(old_due) if old_due else datetime.now()
                if repeat == "daily":
                    next_due = base + timedelta(days=1)
                elif repeat == "weekly":
                    next_due = base + timedelta(weeks=1)
                elif repeat == "monthly":
                    month = base.month + 1
                    year = base.year
                    if month > 12:
                        month = 1
                        year += 1
                    try:
                        next_due = base.replace(year=year, month=month)
                    except ValueError:
                        next_due = base.replace(year=year, month=month, day=28)
                else:
                    next_due = None

                # Check repeat_until
                repeat_until = meta.get("repeat_until")
                if next_due and repeat_until:
                    try:
                        until_dt = datetime.fromisoformat(repeat_until)
                        if next_due > until_dt:
                            next_due = None
                    except (ValueError, TypeError):
                        pass

                if next_due:
                    # Advance to next occurrence, stay pending
                    new_status = "pending"
                    meta["due_date"] = next_due.strftime("%Y-%m-%d")
                    meta["started_at"] = None
                    meta["completed_at"] = None
                    meta["last_completed_at"] = datetime.now().isoformat()
                else:
                    # Repeat expired
                    new_status = "done"
                    meta["completed_at"] = datetime.now().isoformat()
                    meta["repeat"] = None
            else:
                new_status = "done"
                meta["completed_at"] = datetime.now().isoformat()

        meta["status"] = new_status
        meta["updated_at"] = datetime.now().isoformat()

        title = meta.get("title") or rec.content.split(": ", 1)[-1].split(" | ")[0]
        priority = meta.get("priority", "medium")
        content = f"Todo [{priority.upper()}]: {title}"
        if meta.get("due_date"):
            content += f" | Due: {meta['due_date']}"
        if meta.get("repeat"):
            content += f" | Repeats: {meta['repeat']}"
        if new_status == "done":
            content += " [DONE]"

        with api.brain_lock:
            api.brain.update(record_id, content=content, metadata=meta)
        return "ok", new_status

    result, new_status = await run_in_thread(_toggle)
    if result == "not_found":
        raise HTTPException(status_code=404, detail="Todo not found")
    if result == "not_todo":
        raise HTTPException(status_code=400, detail="Record is not a todo item")
    return {"id": record_id, "status": new_status}


@router.post("/todos/{record_id}/start")
async def start_todo(record_id: str):
    """Move a todo to in_progress status."""
    api = _get_api()

    def _start():
        with api.brain_lock:
            rec = api.brain.get(record_id)
        if not rec:
            return "not_found", None
        meta = rec.metadata or {}
        if meta.get("type") != "todo_item":
            return "not_todo", None
        if meta.get("status") == "done":
            return "already_done", None

        meta["status"] = "in_progress"
        meta["started_at"] = datetime.now().isoformat()
        meta["updated_at"] = datetime.now().isoformat()

        title = meta.get("title") or rec.content.split(": ", 1)[-1].split(" | ")[0]
        priority = meta.get("priority", "medium")
        content = f"Todo [{priority.upper()}]: {title}"
        if meta.get("due_date"):
            content += f" | Due: {meta['due_date']}"

        with api.brain_lock:
            api.brain.update(record_id, content=content, metadata=meta)
        return "ok", meta["started_at"]

    result, started_at = await run_in_thread(_start)
    if result == "not_found":
        raise HTTPException(status_code=404, detail="Todo not found")
    if result == "not_todo":
        raise HTTPException(status_code=400, detail="Record is not a todo item")
    if result == "already_done":
        raise HTTPException(status_code=400, detail="Cannot start a completed task")
    return {"id": record_id, "status": "in_progress", "started_at": started_at}


@router.delete("/todos/{record_id}")
async def delete_todo(record_id: str):
    """Delete a todo item."""
    api = _get_api()

    def _delete():
        with api.brain_lock:
            rec = api.brain.get(record_id)
        if not rec:
            return "not_found"
        meta = rec.metadata or {}
        if meta.get("type") not in ("todo_item", "scheduled_task"):
            return "not_todo"
        with api.brain_lock:
            api.brain.delete(record_id)
        return "ok"

    result = await run_in_thread(_delete)
    if result == "not_found":
        raise HTTPException(status_code=404, detail="Todo not found")
    if result == "not_todo":
        raise HTTPException(status_code=400, detail="Record is not a todo item")
    return {"deleted": True, "id": record_id}
