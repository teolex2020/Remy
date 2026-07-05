"""
Automation routes — CRUD + run for scheduled automations.

Each automation: trigger (schedule/on_start/manual) + steps + output_destination.
Stored in Brain with tag "automation".
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from remy.core.workflow_validation import AUTOMATION_WORKFLOW_STEP_TYPES
from remy.web.routes._helpers import _get_api, run_in_thread

logger = logging.getLogger("Automations")

router = APIRouter()

MAX_AUTOMATION_STEPS = 20
MAX_CONSECUTIVE_FAILURES = 3
MAX_RUN_OUTPUT_CHARS = 20000
MAX_STEP_OUTPUT_CHARS = 12000
ALLOWED_TRIGGER_TYPES = {"manual", "on_start", "schedule"}
ALLOWED_SCHEDULE_TYPES = {"daily", "weekly", "hourly", "custom"}
ALLOWED_STEP_TYPES = AUTOMATION_WORKFLOW_STEP_TYPES
ALLOWED_OUTPUT_TYPES = {"chat", "memory", "telegram", "email", "webhook"}
ERROR_PREFIXES = (
    "[Unknown block type:",
    "[Search error:",
    "[Memory search error:",
    "[Save error:",
    "[HTTP error:",
    "[Scrape error:",
    "[JSON parse error:",
    "[Regex error:",
    "[File read error:",
    "[File write error:",
)


def _guard_id(value: str):
    if not value or "/" in value or "\\" in value or ".." in value:
        raise HTTPException(status_code=400, detail="Invalid id")


class AutomationExecutionError(RuntimeError):
    """Raised when an automation cannot produce a trustworthy result."""

    def __init__(self, message: str, trace: list[dict] | None = None):
        super().__init__(message)
        self.trace = trace or []


# ── Models ─────────────────────────────────────────────────────────────────────

class AutomationSave(BaseModel):
    name: str
    description: str = ""
    enabled: bool = True
    trigger: dict = {}
    steps: list[dict] = []
    output_destination: dict = {}
    drawflow_data: dict | None = None
    source_template_id: str = ""
    source_template_name: str = ""


class AutomationTemplateSave(BaseModel):
    name: str
    description: str = ""
    trigger: dict = {}
    steps: list[dict] = []
    output_destination: dict = {}
    drawflow_data: dict | None = None


class AutomationTemplateInstantiate(BaseModel):
    name: str | None = None
    description: str | None = None
    enabled: bool = False
    inputs: dict[str, str] = {}


class AutomationPatch(BaseModel):
    name: str | None = None
    description: str | None = None
    enabled: bool | None = None
    trigger: dict | None = None
    steps: list[dict] | None = None
    output_destination: dict | None = None


# ── Cron helpers ───────────────────────────────────────────────────────────────

def _build_cron(trigger: dict) -> str:
    if trigger.get("type") != "schedule":
        return ""
    st = trigger.get("schedule_type", "daily")
    if st == "custom":
        return trigger.get("cron", "0 9 * * *")
    hhmm = trigger.get("time_of_day", "09:00")
    try:
        hh, mm = (int(x) for x in hhmm.split(":"))
    except Exception:
        hh, mm = 9, 0
    if st == "hourly":
        return f"{mm} * * * *"
    if st == "weekly":
        dow = int(trigger.get("day_of_week", 0))
        return f"{mm} {hh} * * {dow + 1}"
    return f"{mm} {hh} * * *"


def _is_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value.strip())
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _validate_automation_payload(
    *,
    name: str,
    trigger: dict,
    steps: list[dict],
    output_destination: dict,
    drawflow_data: dict | None = None,
) -> list[str]:
    errors: list[str] = []
    if not name or not name.strip():
        errors.append("Automation name is required.")

    trigger_type = (trigger or {}).get("type", "")
    if trigger_type not in ALLOWED_TRIGGER_TYPES:
        errors.append("Trigger type must be manual, on_start, or schedule.")
    if trigger_type == "schedule":
        schedule_type = (trigger or {}).get("schedule_type", "daily")
        if schedule_type not in ALLOWED_SCHEDULE_TYPES:
            errors.append("Schedule type must be daily, weekly, hourly, or custom.")
        if schedule_type == "custom" and not (trigger or {}).get("cron", "").strip():
            errors.append("Custom schedule requires a cron expression.")

    if not steps:
        errors.append("Add at least one action block between Trigger and Output.")
    if len(steps) > MAX_AUTOMATION_STEPS:
        errors.append(f"Automation cannot contain more than {MAX_AUTOMATION_STEPS} action blocks.")

    for index, step in enumerate(steps, start=1):
        step_type = step.get("type", "")
        config = step.get("config") or {}
        if step_type in {"http_request", "page_scrape"}:
            url = (config.get("url") or "").strip()
            label = "HTTP Request" if step_type == "http_request" else "Page Scraper"
            if not url:
                errors.append(f"Step {index} {label} requires a URL.")
            elif "{{" not in url and not _is_http_url(url):
                errors.append(f"Step {index} {label} requires a valid http(s) URL.")

    from remy.core.workflow_validation import validate_workflow_step_configs

    errors.extend(validate_workflow_step_configs(
        steps=steps,
        workflow_label="Automation",
        allowed_step_types=ALLOWED_STEP_TYPES,
    ))

    output_type = (output_destination or {}).get("type", "chat")
    if output_type not in ALLOWED_OUTPUT_TYPES:
        errors.append("Output destination must be chat, memory, telegram, email, or webhook.")
    if output_type == "telegram" and not (output_destination or {}).get("chat_id", "").strip():
        errors.append("Telegram output requires a chat id.")
    if output_type == "email" and not (output_destination or {}).get("to", "").strip():
        errors.append("Email output requires a recipient.")
    if output_type == "webhook":
        webhook_url = (output_destination or {}).get("url", "").strip()
        if not _is_http_url(webhook_url):
            errors.append("Webhook output requires a valid http(s) URL.")

    if drawflow_data:
        from remy.core.workflow_validation import validate_visual_workflow_graph

        errors.extend(validate_visual_workflow_graph(
            steps=steps,
            drawflow_data=drawflow_data,
            entry_name="trigger",
            terminal_name="output",
            workflow_label="Automation",
        ))

    return errors


def _raise_validation_errors(errors: list[str]) -> None:
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})


def _preview(text: str, limit: int = 500) -> str:
    value = text or ""
    return value[:limit]


def _clip(text: str, limit: int) -> tuple[str, bool]:
    value = text or ""
    return value[:limit], len(value) > limit


def _trace_item(
    *,
    index: int,
    step: dict,
    status: str,
    output: str = "",
    error: str = "",
) -> dict:
    clipped, truncated = _clip(output, MAX_STEP_OUTPUT_CHARS)
    return {
        "index": index,
        "id": step.get("id") or f"s{index}",
        "type": step.get("type", ""),
        "label": step.get("label") or step.get("type", f"Step {index}"),
        "status": status,
        "output": clipped,
        "output_length": len(output or ""),
        "output_truncated": truncated,
        "error": error,
    }


def _steps_in_execution_order(meta: dict) -> list[dict]:
    steps = list(meta.get("steps", []) or [])
    flow = meta.get("drawflow_data") or {}
    try:
        nodes = (flow.get("drawflow") or {}).get("Home", {}).get("data", {})
        if not nodes:
            return steps

        by_step_id = {step.get("id"): step for step in steps}
        ordered: list[dict] = []
        seen: set[str] = set()
        current_id = None
        for node_id, node in nodes.items():
            if (node or {}).get("name") == "trigger":
                current_id = str(node_id)
                break

        while current_id and current_id not in seen:
            seen.add(current_id)
            current = nodes.get(current_id) or nodes.get(int(current_id)) or {}
            connections = (
                (current.get("outputs") or {})
                .get("output_1", {})
                .get("connections", [])
            )
            if not connections:
                break
            next_id = str(connections[0].get("node", ""))
            if not next_id:
                break
            next_node = nodes.get(next_id) or nodes.get(int(next_id)) or {}
            next_name = next_node.get("name", "")
            if next_name == "output":
                break
            step = by_step_id.get(f"s{next_id}")
            if step:
                ordered.append(step)
            current_id = next_id

        if ordered:
            ordered_ids = {step.get("id") for step in ordered}
            ordered.extend(step for step in steps if step.get("id") not in ordered_ids)
            return ordered
    except Exception:
        logger.debug("Could not derive automation execution order from drawflow", exc_info=True)
    return steps


def cron_is_due(cron: str, now: datetime) -> bool:
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
                    if value % int(part[2:]) == 0:
                        return True
                elif "-" in part:
                    lo, hi = part.split("-")
                    if int(lo) <= value <= int(hi):
                        return True
                elif int(part) == value:
                    return True
            return False

        cron_dow = (now.weekday() + 1) % 7
        return (
            _match(minute_f, now.minute)
            and _match(hour_f, now.hour)
            and _match(dom_f, now.day)
            and _match(month_f, now.month)
            and _match(dow_f, cron_dow)
        )
    except Exception:
        return False


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _daily_due_at(now: datetime, hh: int, mm: int) -> datetime:
    return now.replace(hour=hh, minute=mm, second=0, microsecond=0)


def _weekly_due_at(now: datetime, hh: int, mm: int, day_of_week: int) -> datetime:
    days_back = (now.weekday() - day_of_week) % 7
    base = now - timedelta(days=days_back)
    return base.replace(hour=hh, minute=mm, second=0, microsecond=0)


def latest_missed_scheduled_run(meta: dict, now: datetime | None = None) -> datetime | None:
    """Return the most recent scheduled slot missed while the app was offline.

    This intentionally supports simple UI schedules only. Custom cron remains
    exact-time only until a cron parser is introduced.
    """
    now = now or datetime.now()
    trigger = meta.get("trigger", {}) or {}
    if trigger.get("type") != "schedule":
        return None
    if not meta.get("enabled", True):
        return None
    if trigger.get("catch_up", True) is False:
        return None

    schedule_type = trigger.get("schedule_type", "daily")
    if schedule_type == "custom":
        return None

    hhmm = trigger.get("time_of_day", "09:00")
    try:
        hh, mm = (int(part) for part in hhmm.split(":"))
    except Exception:
        hh, mm = 9, 0

    due_at: datetime | None = None
    if schedule_type == "hourly":
        due_at = now.replace(minute=mm, second=0, microsecond=0)
        if due_at > now:
            due_at -= timedelta(hours=1)
    elif schedule_type == "weekly":
        due_at = _weekly_due_at(now, hh, mm, int(trigger.get("day_of_week", 0)))
        if due_at > now:
            due_at -= timedelta(days=7)
    else:
        due_at = _daily_due_at(now, hh, mm)
        if due_at > now:
            due_at -= timedelta(days=1)

    last_run_at = _parse_iso_datetime(meta.get("last_run_at"))
    if last_run_at and last_run_at >= due_at:
        return None

    created_at = _parse_iso_datetime(meta.get("created_at"))
    if created_at and due_at < created_at:
        return None

    return due_at if due_at <= now else None


# ── Brain helpers ──────────────────────────────────────────────────────────────

def _record_to_automation(r) -> dict:
    meta = r.metadata or {}
    return {
        "id":                 meta.get("automation_id", ""),
        "record_id":          getattr(r, "id", None) or "",
        "name":               meta.get("name", ""),
        "description":        meta.get("description", ""),
        "enabled":            meta.get("enabled", True),
        "trigger":            meta.get("trigger", {}),
        "steps":              meta.get("steps", []),
        "output_destination": meta.get("output_destination", {}),
        "drawflow_data":      meta.get("drawflow_data", None),
        "cron":               meta.get("cron", ""),
        "created_at":         meta.get("created_at", ""),
        "last_run_at":        meta.get("last_run_at", None),
        "last_run_status":    meta.get("last_run_status", None),
        "last_run_error":     meta.get("last_run_error", None),
        "last_output_preview": meta.get("last_output_preview", ""),
        "last_steps_run":     meta.get("last_steps_run", 0),
        "run_count":          meta.get("run_count", 0),
        "failure_count":      meta.get("failure_count", 0),
        "consecutive_failures": meta.get("consecutive_failures", 0),
        "disabled_reason":    meta.get("disabled_reason", ""),
        "source_template_id": meta.get("source_template_id", ""),
        "source_template_name": meta.get("source_template_name", ""),
    }


def _find_automation(brain, automation_id: str):
    recs = brain.search(query="", tags=["automation"], limit=500)
    for r in recs:
        if (r.metadata or {}).get("automation_id") == automation_id:
            return r
    return None


def _delete_record(brain, r) -> None:
    rid = getattr(r, "id", None)
    if rid:
        try:
            brain.delete(rid)
        except Exception:
            pass


def _store_automation(brain, meta: dict) -> None:
    summary = (
        f"Automation: {meta['name']} | "
        f"trigger={meta.get('trigger',{}).get('type','?')} | "
        f"steps={len(meta.get('steps',[]))} | "
        f"output={meta.get('output_destination',{}).get('type','?')}"
    )
    brain.store(summary, tags=["automation"], metadata=meta)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/automations")
async def list_automations():
    api = _get_api()

    def _q():
        with api.brain_lock:
            return api.brain.search(query="", tags=["automation"], limit=500)

    recs = await run_in_thread(_q)
    items = []
    for r in recs:
        meta = r.metadata or {}
        if meta.get("type") != "automation":
            continue
        items.append(_record_to_automation(r))

    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"automations": items}


@router.get("/automations/templates/list")
async def list_automation_templates():
    from remy.core.workflow_templates import automation_templates

    return {"templates": automation_templates()}


@router.post("/automations/templates")
async def save_automation_template(body: AutomationTemplateSave):
    _raise_validation_errors(_validate_automation_payload(
        name=body.name,
        trigger=body.trigger,
        steps=body.steps,
        output_destination=body.output_destination,
        drawflow_data=body.drawflow_data,
    ))
    from remy.core.workflow_templates import save_custom_template

    template = save_custom_template("automation", {
        "name": body.name.strip() or "Untitled template",
        "description": body.description.strip(),
        "trigger": body.trigger,
        "steps": body.steps,
        "output_destination": body.output_destination,
        "drawflow_data": body.drawflow_data,
        "enabled": False,
    })
    return {"template": template}


@router.post("/automations/templates/{template_id}/instantiate")
async def instantiate_automation_template(template_id: str, body: AutomationTemplateInstantiate | None = None):
    _guard_id(template_id)
    from remy.core.workflow_templates import apply_template_inputs, find_automation_template

    template = find_automation_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Automation template not found")
    template = apply_template_inputs(template, body.inputs if body else {})

    name = (body.name if body else None) or template.get("name") or "Untitled automation"
    description = (body.description if body else None)
    if description is None:
        description = template.get("description", "")

    created = await create_automation(AutomationSave(
        name=name,
        description=description,
        enabled=bool(body.enabled) if body else False,
        trigger=template.get("trigger", {"type": "manual"}),
        steps=template.get("steps", []),
        output_destination=template.get("output_destination", {"type": "chat"}),
        drawflow_data=template.get("drawflow_data"),
        source_template_id=template_id,
        source_template_name=template.get("name", ""),
    ))
    automation = await get_automation(created["automation_id"])
    return {"automation": automation, "template_id": template_id}


@router.delete("/automations/templates/{template_id}")
async def delete_automation_template(template_id: str):
    _guard_id(template_id)
    from remy.core.workflow_templates import delete_custom_template

    if not delete_custom_template("automation", template_id):
        raise HTTPException(status_code=404, detail="Custom automation template not found")
    return {"deleted": True, "id": template_id}


@router.get("/automations/{automation_id}")
async def get_automation(automation_id: str):
    _guard_id(automation_id)
    api = _get_api()

    def _q():
        with api.brain_lock:
            return _find_automation(api.brain, automation_id)

    r = await run_in_thread(_q)
    if not r:
        raise HTTPException(status_code=404, detail="Automation not found")
    return _record_to_automation(r)


@router.post("/automations")
async def create_automation(body: AutomationSave):
    api = _get_api()
    _raise_validation_errors(_validate_automation_payload(
        name=body.name,
        trigger=body.trigger,
        steps=body.steps,
        output_destination=body.output_destination,
        drawflow_data=body.drawflow_data,
    ))
    automation_id = f"auto-{uuid.uuid4().hex[:8]}"
    cron = _build_cron(body.trigger)

    meta = {
        "type":               "automation",
        "automation_id":      automation_id,
        "name":               body.name,
        "description":        body.description,
        "enabled":            body.enabled,
        "trigger":            body.trigger,
        "steps":              body.steps,
        "output_destination": body.output_destination,
        "drawflow_data":      body.drawflow_data,
        "source_template_id": body.source_template_id.strip(),
        "source_template_name": body.source_template_name.strip(),
        "cron":               cron,
        "created_at":         datetime.now().isoformat(),
        "last_run_at":        None,
        "last_run_status":    None,
        "last_run_error":     None,
        "last_output_preview": "",
        "last_steps_run":     0,
        "run_count":          0,
        "failure_count":      0,
        "consecutive_failures": 0,
        "disabled_reason":    "",
    }

    def _store():
        with api.brain_lock:
            _store_automation(api.brain, meta)

    await run_in_thread(_store)
    return {"ok": True, "automation_id": automation_id}


@router.put("/automations/{automation_id}")
async def update_automation(automation_id: str, body: AutomationSave):
    _guard_id(automation_id)
    api = _get_api()
    _raise_validation_errors(_validate_automation_payload(
        name=body.name,
        trigger=body.trigger,
        steps=body.steps,
        output_destination=body.output_destination,
        drawflow_data=body.drawflow_data,
    ))

    def _replace():
        with api.brain_lock:
            r = _find_automation(api.brain, automation_id)
            if not r:
                return False
            old_meta = dict(r.metadata or {})
            _delete_record(api.brain, r)
            cron = _build_cron(body.trigger)
            new_meta = {
                **old_meta,
                "name":               body.name,
                "description":        body.description,
                "enabled":            body.enabled,
                "trigger":            body.trigger,
                "steps":              body.steps,
                "output_destination": body.output_destination,
                "drawflow_data":      body.drawflow_data,
                "source_template_id": body.source_template_id.strip() or old_meta.get("source_template_id", ""),
                "source_template_name": body.source_template_name.strip() or old_meta.get("source_template_name", ""),
                "cron":               cron,
            }
            _store_automation(api.brain, new_meta)
            return True

    found = await run_in_thread(_replace)
    if not found:
        raise HTTPException(status_code=404, detail="Automation not found")
    return {"ok": True}


@router.patch("/automations/{automation_id}")
async def patch_automation(automation_id: str, body: AutomationPatch):
    _guard_id(automation_id)
    api = _get_api()

    def _patch():
        with api.brain_lock:
            r = _find_automation(api.brain, automation_id)
            if not r:
                return False
            meta = dict(r.metadata or {})
            _delete_record(api.brain, r)
            update = body.model_dump(exclude_none=True)
            meta.update(update)
            if "trigger" in update:
                meta["cron"] = _build_cron(meta["trigger"])
            errors = _validate_automation_payload(
                name=meta.get("name", ""),
                trigger=meta.get("trigger", {}),
                steps=meta.get("steps", []),
                output_destination=meta.get("output_destination", {}),
                drawflow_data=meta.get("drawflow_data"),
            )
            if errors:
                _store_automation(api.brain, dict(r.metadata or {}))
                return {"found": True, "errors": errors}
            _store_automation(api.brain, meta)
            return {"found": True, "errors": []}

    result = await run_in_thread(_patch)
    if not result:
        raise HTTPException(status_code=404, detail="Automation not found")
    _raise_validation_errors(result.get("errors", []))
    return {"ok": True}


@router.delete("/automations/{automation_id}")
async def delete_automation(automation_id: str):
    _guard_id(automation_id)
    api = _get_api()

    def _del():
        with api.brain_lock:
            r = _find_automation(api.brain, automation_id)
            if not r:
                return False
            _delete_record(api.brain, r)
            return True

    found = await run_in_thread(_del)
    if not found:
        raise HTTPException(status_code=404, detail="Automation not found")
    return {"ok": True}


def _mark_automation_run(
    brain,
    automation_id: str,
    *,
    status: str,
    steps_run: int = 0,
    output: str = "",
    error: str = "",
) -> dict:
    r = _find_automation(brain, automation_id)
    if not r:
        return {}

    meta = dict(r.metadata or {})
    _delete_record(brain, r)
    meta["last_run_at"] = datetime.now().isoformat()
    meta["last_run_status"] = status
    meta["last_steps_run"] = steps_run
    meta["last_output_preview"] = _preview(output)
    meta["last_run_error"] = error or None

    if status == "ok":
        meta["run_count"] = meta.get("run_count", 0) + 1
        meta["consecutive_failures"] = 0
        meta["disabled_reason"] = ""
    else:
        meta["failure_count"] = meta.get("failure_count", 0) + 1
        meta["consecutive_failures"] = meta.get("consecutive_failures", 0) + 1
        if meta["consecutive_failures"] >= MAX_CONSECUTIVE_FAILURES:
            meta["enabled"] = False
            meta["disabled_reason"] = (
                f"Automatically paused after {meta['consecutive_failures']} consecutive failed runs."
            )

    _store_automation(brain, meta)
    return meta


async def run_automation_record(brain, brain_lock, meta: dict) -> dict:
    automation_id = meta.get("automation_id", "")
    from remy.core.workflow_runs import finish_workflow_run, start_workflow_run

    run_record = start_workflow_run(
        kind="automation",
        workflow_id=automation_id,
        workflow_name=meta.get("name", ""),
        input_text="",
        trigger=(meta.get("trigger") or {}).get("type", "manual"),
    )
    try:
        execution = await _execute_automation(meta)
        if len(execution) == 2:
            result_output, steps_run = execution
            trace = []
        else:
            result_output, steps_run, trace = execution
    except Exception as exc:
        error = str(exc) or exc.__class__.__name__
        trace = getattr(exc, "trace", [])

        def _fail():
            with brain_lock:
                return _mark_automation_run(
                    brain,
                    automation_id,
                    status="error",
                    steps_run=0,
                    error=error,
                )

        updated = await run_in_thread(_fail)
        finish_workflow_run(
            run_record,
            status="error",
            error=error,
            trace=trace,
            steps_run=0,
        )
        logger.error("Automation '%s' failed: %s", meta.get("name", automation_id), error)
        return {
            "ok": False,
            "run_id": run_record["run_id"],
            "steps_run": 0,
            "output": "",
            "error": error,
            "trace": trace,
            "enabled": updated.get("enabled", meta.get("enabled", True)),
            "disabled_reason": updated.get("disabled_reason", ""),
        }

    def _ok():
        with brain_lock:
            return _mark_automation_run(
                brain,
                automation_id,
                status="ok",
                steps_run=steps_run,
                output=result_output,
            )

    updated = await run_in_thread(_ok)
    output, output_truncated = _clip(result_output, MAX_RUN_OUTPUT_CHARS)
    finish_workflow_run(
        run_record,
        status="ok",
        output=result_output,
        trace=trace,
        steps_run=steps_run,
        output_truncated=output_truncated,
    )
    return {
        "ok": True,
        "run_id": run_record["run_id"],
        "steps_run": steps_run,
        "output": output,
        "output_length": len(result_output or ""),
        "output_truncated": output_truncated,
        "trace": trace,
        "enabled": updated.get("enabled", meta.get("enabled", True)),
        "disabled_reason": updated.get("disabled_reason", ""),
    }


@router.post("/automations/{automation_id}/run")
async def run_automation_now(automation_id: str):
    """Immediately execute an automation and deliver to output_destination."""
    _guard_id(automation_id)
    api = _get_api()

    def _load():
        with api.brain_lock:
            r = _find_automation(api.brain, automation_id)
            return (dict(r.metadata or {}) if r else None)

    meta = await run_in_thread(_load)
    if not meta:
        raise HTTPException(status_code=404, detail="Automation not found")

    result = await run_automation_record(api.brain, api.brain_lock, meta)
    if not result["ok"]:
        raise HTTPException(status_code=500, detail=result)
    return result


@router.get("/automations/{automation_id}/runs")
async def list_automation_runs(automation_id: str, limit: int = 50):
    _guard_id(automation_id)
    from remy.core.workflow_runs import list_workflow_runs

    return {"runs": list_workflow_runs("automation", automation_id, limit=limit)}


@router.get("/automations/{automation_id}/memory-report")
async def get_automation_memory_report(automation_id: str, limit: int = 20):
    _guard_id(automation_id)
    from remy.core.workflow_runs import summarize_workflow_memory

    return summarize_workflow_memory("automation", automation_id, limit=limit)


@router.get("/automations/{automation_id}/preflight")
async def get_automation_preflight(automation_id: str, mode: str = "manual_run"):
    _guard_id(automation_id)
    api = _get_api()

    def _load():
        with api.brain_lock:
            r = _find_automation(api.brain, automation_id)
            return (dict(r.metadata or {}) if r else None)

    meta = await run_in_thread(_load)
    if not meta:
        raise HTTPException(status_code=404, detail="Automation not found")

    from remy.core.workflow_runs import list_workflow_runs
    from remy.core.workflow_safety import assess_workflow_safety

    runs = list_workflow_runs("automation", automation_id, limit=20)
    has_successful_run = any(run.get("status") == "ok" for run in runs)
    return assess_workflow_safety(
        kind="automation",
        steps=meta.get("steps", []),
        trigger=meta.get("trigger", {}),
        output_destination=meta.get("output_destination", {}),
        has_successful_run=has_successful_run,
        mode=mode,
    )


@router.get("/automations/{automation_id}/runs/{run_id}")
async def get_automation_run(automation_id: str, run_id: str):
    _guard_id(automation_id)
    _guard_id(run_id)
    from remy.core.workflow_runs import get_workflow_run

    record = get_workflow_run("automation", automation_id, run_id)
    if not record:
        raise HTTPException(status_code=404, detail="Automation run not found")
    return record


# ── Execution engine ───────────────────────────────────────────────────────────

def _route_number(output: str) -> str:
    match = re.search(r"\d+", output or "")
    return match.group(0) if match else "1"


def _find_drawflow_node_id(nodes: dict, name: str) -> str | None:
    for node_id, node in nodes.items():
        if (node or {}).get("name") == name:
            return str(node_id)
    return None


def _route_output_names(output: str) -> list[str]:
    indexes = re.findall(r"\d+", output or "") or ["1"]
    names: list[str] = []
    seen: set[str] = set()
    for idx in indexes:
        name = f"output_{idx}"
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _next_drawflow_targets(nodes: dict, current_id: str, output_name: str) -> list[tuple[str, str]]:
    current = nodes.get(current_id) or nodes.get(int(current_id)) or {}
    connections = (
        (current.get("outputs") or {})
        .get(output_name, {})
        .get("connections", [])
    )
    targets: list[tuple[str, str]] = []
    for conn in connections:
        node_id = conn.get("node", "")
        if node_id != "":
            targets.append((str(node_id), str(conn.get("output", "input_1") or "input_1")))
    return targets


def _next_drawflow_node_ids(nodes: dict, current_id: str, output_name: str) -> list[str]:
    return [node_id for node_id, _input_name in _next_drawflow_targets(nodes, current_id, output_name)]


def _merge_required_inputs_from_node(node: dict, step: dict | None = None) -> list[str]:
    inputs = (node or {}).get("inputs") or {}
    required = [
        name
        for name, input_meta in sorted(inputs.items())
        if (input_meta or {}).get("connections")
    ]
    if required:
        return required
    try:
        count = int(((step or {}).get("config") or {}).get("input_count") or 2)
    except Exception:
        count = 2
    count = max(2, min(count, 10))
    return [f"input_{idx}" for idx in range(1, count + 1)]


def _merge_step_with_inputs(step: dict, inputs: list[str]) -> dict:
    return {
        **step,
        "config": {
            **(step.get("config") or {}),
            "_merge_inputs": inputs,
        },
    }


async def _execute_automation(meta: dict) -> tuple[str, int, list[dict]]:
    """Run all steps sequentially, passing output forward."""
    from remy.core.pipeline_runner import STOP_TOKEN, _execute_step

    steps = list(meta.get("steps", []) or [])
    ordered_steps = _steps_in_execution_order(meta)
    validation_errors = _validate_automation_payload(
        name=meta.get("name", ""),
        trigger=meta.get("trigger", {}),
        steps=steps,
        output_destination=meta.get("output_destination", {}),
        drawflow_data=meta.get("drawflow_data"),
    )
    if validation_errors:
        raise AutomationExecutionError("; ".join(validation_errors))

    ctx: dict[str, Any] = {"input": "", "prev": ""}
    last_output = ""
    steps_run = 0
    trace: list[dict] = []
    flow_nodes = (((meta.get("drawflow_data") or {}).get("drawflow") or {}).get("Home", {}) or {}).get("data", {})

    async def _run_step(step: dict, step_index: int, inherited_output: str | None = None) -> str:
        nonlocal last_output, steps_run
        step_id = step.get("id") or f"s{step_index}"
        source_output = last_output if inherited_output is None else inherited_output
        ctx["input"] = source_output
        ctx["prev"] = source_output
        ctx[f"s{step_index - 1}.output"] = source_output
        config = step.get("config") or {}
        retry_enabled = bool(config.get("_retry_enabled"))
        retry_count = max(0, min(int(config.get("_retry_count") or 0), 5)) if retry_enabled else 0
        retry_delay_ms = max(0, min(int(config.get("_retry_delay_ms") or 0), 30000))
        attempts = retry_count + 1
        last_error = ""
        try:
            output = ""
            for attempt in range(1, attempts + 1):
                try:
                    output = await _execute_step(step, ctx)
                    if any((output or "").startswith(prefix) for prefix in ERROR_PREFIXES):
                        raise AutomationExecutionError(output)
                    break
                except Exception as exc:
                    last_error = str(exc) or exc.__class__.__name__
                    if attempt >= attempts:
                        raise
                    if retry_delay_ms:
                        await asyncio.sleep(retry_delay_ms / 1000)
            ctx[step_id] = {"output": output}
            ctx[f"s{step_index}"] = {"output": output}
            ctx[f"s{step_index}.output"] = output
            if step.get("type") == "set_variable":
                variable_name = str((step.get("config") or {}).get("name", "") or "").strip()
                if variable_name:
                    ctx[variable_name] = output
            steps_run += 1
            if step.get("type") == "router":
                trace_output = f"Selected routes: {', '.join(_route_output_names(output))}"
            elif output == STOP_TOKEN:
                trace_output = "Stopped by filter"
            else:
                last_output = output
                trace_output = output
            trace.append(_trace_item(index=step_index, step=step, status="ok", output=trace_output))
            if retry_enabled and last_error:
                trace[-1]["retry_attempts"] = attempts
                trace[-1]["recovered_from_error"] = last_error
            return output
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
            ctx["error"] = error
            ctx["failed_step"] = step.get("label") or step.get("type", f"Step {step_index}")
            trace.append(_trace_item(index=step_index, step=step, status="error", error=error))
            if retry_enabled:
                trace[-1]["retry_attempts"] = attempts
            logger.error("Automation step %d failed: %s", step_index, exc)
            raise AutomationExecutionError(f"Step {step_index} failed: {exc}", trace=trace) from exc

    if flow_nodes:
        by_step_id = {step.get("id"): step for step in steps}
        trigger_id = _find_drawflow_node_id(flow_nodes, "trigger")
        queue: list[tuple[str, str, str]] = [(trigger_id, "output_1", "")] if trigger_id else []
        seen: set[tuple[str, str, str]] = set()
        terminal_outputs: list[str] = []
        merge_buffers: dict[str, dict[str, str]] = {}

        while queue and steps_run < MAX_AUTOMATION_STEPS:
            current_id, output_name, inherited_output = queue.pop(0)
            seen_key = (current_id, output_name, inherited_output[:500])
            if seen_key in seen:
                continue
            seen.add(seen_key)

            targets = _next_drawflow_targets(flow_nodes, current_id, output_name)
            if not targets:
                if inherited_output.strip():
                    terminal_outputs.append(inherited_output)
                continue

            for next_id, input_name in targets:
                next_node = flow_nodes.get(next_id) or flow_nodes.get(int(next_id)) or {}
                if next_node.get("name") == "output":
                    if inherited_output.strip():
                        terminal_outputs.append(inherited_output)
                    continue

                step = by_step_id.get(f"s{next_id}")
                if not step:
                    continue
                if step.get("type") == "merge":
                    buffer = merge_buffers.setdefault(next_id, {})
                    buffer[input_name] = inherited_output
                    required_inputs = _merge_required_inputs_from_node(next_node, step)
                    if not all(name in buffer for name in required_inputs):
                        continue
                    ordered_inputs = [buffer.get(name, "") for name in required_inputs]
                    merge_buffers.pop(next_id, None)
                    step = _merge_step_with_inputs(step, ordered_inputs)
                    inherited_for_step = "\n\n---\n\n".join(ordered_inputs)
                else:
                    inherited_for_step = inherited_output

                try:
                    output = await _run_step(step, steps_run + 1, inherited_for_step)
                except AutomationExecutionError as exc:
                    error_text = str(exc) or exc.__class__.__name__
                    error_targets = _next_drawflow_targets(flow_nodes, next_id, "output_2") if step.get("type") != "router" else []
                    if not error_targets:
                        raise
                    ctx["error"] = error_text
                    ctx["failed_step"] = step.get("label") or step.get("type", "Step")
                    queue.append((next_id, "output_2", error_text))
                    continue
                if output == STOP_TOKEN:
                    continue
                if step.get("type") == "router":
                    for route_output in _route_output_names(output):
                        queue.append((next_id, route_output, inherited_output))
                else:
                    queue.append((next_id, "output_1", output))

        if terminal_outputs:
            last_output = "\n\n---\n\n".join(terminal_outputs)
    else:
        for i, step in enumerate(ordered_steps):
            await _run_step(step, i + 1)

    if not (last_output or "").strip():
        raise AutomationExecutionError("Automation finished without output.", trace=trace)

    # Deliver to output destination
    try:
        await _deliver(
            last_output,
            meta.get("output_destination", {}),
            meta.get("name", "Automation"),
            meta.get("automation_id", ""),
        )
    except Exception as exc:
        error = str(exc) or exc.__class__.__name__
        trace.append({
            "index": steps_run + 1,
            "id": "output",
            "type": "output",
            "label": "Output",
            "status": "error",
            "output": "",
            "output_length": 0,
            "output_truncated": False,
            "error": error,
        })
        raise AutomationExecutionError(f"Output delivery failed: {error}", trace=trace) from exc

    return last_output, steps_run, trace


async def _deliver(output: str, dest: dict, automation_name: str, automation_id: str = "") -> None:
    dest_type = dest.get("type", "chat")

    if dest_type == "chat":
        # Store as a brain notification so user sees it when opening app
        from remy.core.agent_tools import brain, brain_lock
        def _save():
            with brain_lock:
                brain.store(
                    f"[Automation: {automation_name}]\n\n{output}",
                    tags=["automation-result", "notification"],
                    metadata={
                        "type": "automation_result",
                        "source": "automation",
                        "automation_id": automation_id,
                        "automation_name": automation_name,
                        "created_at": datetime.now().isoformat(),
                        "verified": False,
                    },
                )
        await asyncio.to_thread(_save)

    elif dest_type == "memory":
        from remy.core.agent_tools import brain, brain_lock
        tags = [t.strip() for t in dest.get("tags", "automation").split(",") if t.strip()]
        def _save():
            with brain_lock:
                brain.store(output, tags=tags, metadata={
                    "type": "automation_result",
                    "source": "automation",
                    "automation_id": automation_id,
                    "automation_name": automation_name,
                    "created_at": datetime.now().isoformat(),
                    "verified": False,
                })
        await asyncio.to_thread(_save)

    elif dest_type == "telegram":
        chat_id = dest.get("chat_id", "")
        if chat_id:
            try:
                from remy.config.settings import settings
                import httpx
                token = settings.TELEGRAM_BOT_TOKEN
                if not token:
                    raise AutomationExecutionError("Telegram bot token is not configured in Settings -> Integrations.")
                text = f"*{automation_name}*\n\n{output}"[:4096]
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.post(
                        f"https://api.telegram.org/bot{token}/sendMessage",
                        json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
                    )
                    resp.raise_for_status()
            except Exception as exc:
                logger.warning("Telegram delivery failed: %s", exc)
                raise

    elif dest_type == "email":
        to = dest.get("to", "")
        subject = dest.get("subject", f"Automation: {automation_name}")
        if to:
            try:
                import smtplib, ssl
                from email.mime.text import MIMEText
                from remy.config.settings import settings
                smtp_host = settings.SMTP_HOST or ""
                smtp_user = settings.SMTP_USER or ""
                smtp_pass = settings.SMTP_PASSWORD or ""
                if not smtp_host or not smtp_user or not smtp_pass:
                    raise AutomationExecutionError("Email SMTP credentials are not configured in Settings -> Integrations.")
                smtp_port = settings.SMTP_PORT or 587
                msg = MIMEText(output)
                msg["Subject"] = subject
                msg["From"]    = settings.SMTP_FROM or smtp_user
                msg["To"]      = to
                def _send():
                    ctx2 = ssl.create_default_context()
                    with smtplib.SMTP(smtp_host, smtp_port) as s:
                        s.starttls(context=ctx2)
                        s.login(smtp_user, smtp_pass)
                        s.send_message(msg)
                await asyncio.to_thread(_send)
            except Exception as exc:
                logger.warning("Email delivery failed: %s", exc)
                raise

    elif dest_type == "webhook":
        url = dest.get("url", "")
        if url:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.post(url, json={
                        "automation": automation_name,
                        "output": output,
                        "timestamp": datetime.now().isoformat(),
                    })
                    resp.raise_for_status()
            except Exception as exc:
                logger.warning("Webhook delivery failed: %s", exc)
                raise
