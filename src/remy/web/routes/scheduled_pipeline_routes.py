"""
Scheduled Pipeline routes — CRUD for pipeline schedules stored in brain.

Each schedule is stored as a brain record with tag "scheduled-pipeline".
The scheduler loop reads these records every minute and fires due pipelines.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from remy.web.routes._helpers import _get_api, run_in_thread

logger = logging.getLogger("ScheduledPipelines")

router = APIRouter()


# ── Pydantic models ────────────────────────────────────────────────────────────

class ScheduleCreate(BaseModel):
    name: str
    pipeline_id: str
    pipeline_name: str = ""
    input_text: str = ""
    # Simple schedule: "daily"/"weekly"/"hourly" OR advanced cron expression
    schedule_type: str = "daily"   # daily | weekly | hourly | custom
    cron: str = ""                  # e.g. "0 9 * * 1-5"  (only if schedule_type=="custom")
    time_of_day: str = "09:00"     # HH:MM — used for daily/weekly
    day_of_week: int = 0           # 0=Mon … 6=Sun — used for weekly
    enabled: bool = True


class ScheduleUpdate(BaseModel):
    name: str | None = None
    input_text: str | None = None
    schedule_type: str | None = None
    cron: str | None = None
    time_of_day: str | None = None
    day_of_week: int | None = None
    enabled: bool | None = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cron_from_schedule(s: dict) -> str:
    """Build a cron expression from the friendly schedule fields."""
    if s.get("schedule_type") == "custom":
        return s.get("cron", "0 9 * * *")
    hhmm = s.get("time_of_day", "09:00")
    try:
        hh, mm = hhmm.split(":")
        hh, mm = int(hh), int(mm)
    except Exception:
        hh, mm = 9, 0
    if s.get("schedule_type") == "hourly":
        return f"{mm} * * * *"
    if s.get("schedule_type") == "weekly":
        dow = int(s.get("day_of_week", 0))  # 0=Mon in our UI, cron 1=Mon
        return f"{mm} {hh} * * {dow + 1}"
    # daily (default)
    return f"{mm} {hh} * * *"


def _cron_is_due(cron: str, now: datetime) -> bool:
    """
    Minimal cron evaluator — checks if the given cron expression matches *now*
    (minute-level precision).  Supports */N step syntax and ranges (1-5).
    """
    try:
        parts = cron.strip().split()
        if len(parts) != 5:
            return False
        minute_f, hour_f, dom_f, month_f, dow_f = parts

        def _match(field: str, value: int) -> bool:
            if field == "*":
                return True
            for part in field.split(","):
                if part.startswith("*/"):
                    step = int(part[2:])
                    if value % step == 0:
                        return True
                elif "-" in part:
                    lo, hi = part.split("-")
                    if int(lo) <= value <= int(hi):
                        return True
                elif int(part) == value:
                    return True
            return False

        # cron dow: 0/7=Sun,1=Mon…6=Sat; Python weekday: 0=Mon…6=Sun
        py_dow = now.weekday()          # 0=Mon
        cron_dow = (py_dow + 1) % 7    # 1=Mon, 0=Sun

        return (
            _match(minute_f, now.minute)
            and _match(hour_f, now.hour)
            and _match(dom_f, now.day)
            and _match(month_f, now.month)
            and _match(dow_f, cron_dow)
        )
    except Exception:
        return False


def _record_to_schedule(r) -> dict:
    meta = r.metadata or {}
    return {
        "id": meta.get("schedule_id", ""),
        "record_id": getattr(r, "id", None) or meta.get("record_id", ""),
        "name": meta.get("name", ""),
        "pipeline_id": meta.get("pipeline_id", ""),
        "pipeline_name": meta.get("pipeline_name", ""),
        "input_text": meta.get("input_text", ""),
        "schedule_type": meta.get("schedule_type", "daily"),
        "cron": meta.get("cron", ""),
        "time_of_day": meta.get("time_of_day", "09:00"),
        "day_of_week": meta.get("day_of_week", 0),
        "enabled": meta.get("enabled", True),
        "created_at": meta.get("created_at", ""),
        "last_run_at": meta.get("last_run_at", None),
        "last_run_status": meta.get("last_run_status", None),
        "run_count": meta.get("run_count", 0),
    }


# ── API endpoints ──────────────────────────────────────────────────────────────

@router.get("/scheduled-pipelines")
async def list_scheduled_pipelines():
    api = _get_api()

    def _query():
        with api.brain_lock:
            return api.brain.search(query="", tags=["scheduled-pipeline"], limit=200)

    recs = await run_in_thread(_query)
    schedules = []
    for r in recs:
        meta = r.metadata or {}
        if meta.get("type") != "scheduled_pipeline":
            continue
        schedules.append(_record_to_schedule(r))

    schedules.sort(key=lambda s: s.get("created_at", ""), reverse=True)
    return {"schedules": schedules}


@router.post("/scheduled-pipelines")
async def create_scheduled_pipeline(body: ScheduleCreate):
    api = _get_api()
    schedule_id = f"sched-{uuid.uuid4().hex[:8]}"
    now = datetime.now().isoformat()
    cron = _cron_from_schedule(body.model_dump())

    meta = {
        "type": "scheduled_pipeline",
        "schedule_id": schedule_id,
        "name": body.name,
        "pipeline_id": body.pipeline_id,
        "pipeline_name": body.pipeline_name,
        "input_text": body.input_text,
        "schedule_type": body.schedule_type,
        "cron": cron,
        "time_of_day": body.time_of_day,
        "day_of_week": body.day_of_week,
        "enabled": body.enabled,
        "created_at": now,
        "last_run_at": None,
        "last_run_status": None,
        "run_count": 0,
    }

    summary = (
        f"Scheduled pipeline: {body.name} | pipeline_id={body.pipeline_id} | "
        f"cron={cron} | input={body.input_text[:80]}"
    )

    def _store():
        with api.brain_lock:
            api.brain.store(summary, tags=["scheduled-pipeline"], metadata=meta)

    await run_in_thread(_store)
    return {"ok": True, "schedule_id": schedule_id, "cron": cron}


@router.patch("/scheduled-pipelines/{schedule_id}")
async def update_scheduled_pipeline(schedule_id: str, body: ScheduleUpdate):
    api = _get_api()

    def _find():
        with api.brain_lock:
            return api.brain.search(query="", tags=["scheduled-pipeline"], limit=200)

    recs = await run_in_thread(_find)
    target = None
    for r in recs:
        meta = r.metadata or {}
        if meta.get("schedule_id") == schedule_id:
            target = r
            break

    if not target:
        raise HTTPException(status_code=404, detail="Schedule not found")

    meta = dict(target.metadata or {})
    update = body.model_dump(exclude_none=True)
    meta.update(update)

    # Recompute cron if any schedule field changed
    if any(k in update for k in ("schedule_type", "cron", "time_of_day", "day_of_week")):
        meta["cron"] = _cron_from_schedule(meta)

    summary = (
        f"Scheduled pipeline: {meta['name']} | pipeline_id={meta['pipeline_id']} | "
        f"cron={meta['cron']} | input={meta.get('input_text','')[:80]}"
    )

    record_id = getattr(target, "id", None)

    def _update():
        with api.brain_lock:
            if record_id:
                try:
                    api.brain.delete(record_id)
                except Exception:
                    pass
            api.brain.store(summary, tags=["scheduled-pipeline"], metadata=meta)

    await run_in_thread(_update)
    return {"ok": True, "cron": meta["cron"]}


@router.delete("/scheduled-pipelines/{schedule_id}")
async def delete_scheduled_pipeline(schedule_id: str):
    api = _get_api()

    def _find_and_delete():
        with api.brain_lock:
            recs = api.brain.search(query="", tags=["scheduled-pipeline"], limit=200)
            for r in recs:
                meta = r.metadata or {}
                if meta.get("schedule_id") == schedule_id:
                    rid = getattr(r, "id", None)
                    if rid:
                        try:
                            api.brain.delete(rid)
                        except Exception:
                            pass
                    return True
        return False

    found = await run_in_thread(_find_and_delete)
    if not found:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return {"ok": True}


@router.post("/scheduled-pipelines/{schedule_id}/run-now")
async def run_scheduled_pipeline_now(schedule_id: str):
    """Trigger an immediate run of a scheduled pipeline (manual fire)."""
    api = _get_api()

    def _find():
        with api.brain_lock:
            return api.brain.search(query="", tags=["scheduled-pipeline"], limit=200)

    recs = await run_in_thread(_find)
    target_meta = None
    for r in recs:
        meta = r.metadata or {}
        if meta.get("schedule_id") == schedule_id:
            target_meta = meta
            break

    if not target_meta:
        raise HTTPException(status_code=404, detail="Schedule not found")

    pipeline_id = target_meta.get("pipeline_id", "")
    input_text = target_meta.get("input_text", "")

    # Load pipeline steps from disk
    from remy.config.settings import settings
    import json

    pipeline_path = settings.DATA_DIR / "pipelines" / f"{pipeline_id}.json"
    if not pipeline_path.exists():
        raise HTTPException(status_code=404, detail="Pipeline file not found")

    with open(pipeline_path, "r", encoding="utf-8") as f:
        pipeline_data = json.load(f)

    steps = pipeline_data.get("steps", [])

    from remy.core.pipeline_runner import run_pipeline_steps
    import asyncio

    outputs = []
    async for event in run_pipeline_steps(steps, input_text):
        if event.get("type") == "step_done":
            outputs.append({"step": event.get("label", ""), "output": event.get("output", "")})

    # Update last_run_at in brain
    def _update_meta():
        with api.brain_lock:
            recs2 = api.brain.search(query="", tags=["scheduled-pipeline"], limit=200)
            for r2 in recs2:
                m = r2.metadata or {}
                if m.get("schedule_id") == schedule_id:
                    m2 = dict(m)
                    m2["last_run_at"] = datetime.now().isoformat()
                    m2["last_run_status"] = "ok"
                    m2["run_count"] = m2.get("run_count", 0) + 1
                    rid = getattr(r2, "id", None)
                    if rid:
                        try:
                            api.brain.delete(rid)
                        except Exception:
                            pass
                    summary = f"Scheduled pipeline: {m2['name']} | pipeline_id={m2['pipeline_id']} | cron={m2['cron']}"
                    api.brain.store(summary, tags=["scheduled-pipeline"], metadata=m2)
                    break

    await run_in_thread(_update_meta)
    return {"ok": True, "steps_run": len(outputs), "outputs": outputs}
