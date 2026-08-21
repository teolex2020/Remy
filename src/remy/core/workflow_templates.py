"""Built-in workflow template catalog shared by Home, Pipelines, and Automations."""

from __future__ import annotations

import json
import time
import uuid
from copy import deepcopy
from pathlib import Path

from remy.config.settings import settings
from remy.core.file_utils import atomic_write


MAX_CUSTOM_TEMPLATES_PER_KIND = 100


WORKFLOW_TEMPLATES: list[dict] = [
    {
        "id": "summarize-document",
        "name": "Summarize Document",
        "title": "Summarize Document",
        "pack": "Document Pack",
        "icon": "DOC",
        "description": "Import a local file, summarize it, extract action items, and keep a report.",
        "fields": [
            {"id": "source", "label": "Document name", "placeholder": "meeting-notes.md"},
            {"id": "goal", "label": "Focus", "placeholder": "decisions, risks, deadlines"},
        ],
        "home_steps": ["Document intake", "Evidence-bounded summary", "Action items", "Report preview"],
        "targetView": "documents",
        "pipeline_steps": [
            {"id": "s1", "type": "file_read", "label": "Read document", "config": {"filename": "{{input}}", "max_chars": 20000}},
            {"id": "s2", "type": "llm_call", "label": "Evidence-bounded summary", "config": {"prompt": "Summarize this document. Focus on decisions, risks, deadlines, and action items:\n\n{{s1.output}}", "model": ""}},
            {"id": "s3", "type": "memory_save", "label": "Save report", "config": {"text": "{{s2.output}}", "tags": "document,summary"}},
        ],
        "automation": {
            "trigger": {"type": "manual"},
            "steps": [
                {"id": "s1", "type": "file_read", "label": "Read document", "config": {"filename": "meeting-notes.md", "max_chars": 20000}},
                {"id": "s2", "type": "llm_call", "label": "Summarize", "config": {"system_prompt": "Summarize decisions, risks, deadlines, and action items.", "input_source": "{{prev}}", "model": ""}},
                {"id": "s3", "type": "memory_save", "label": "Save report", "config": {"tags": "document,summary", "input_source": "{{prev}}"}},
            ],
            "output_destination": {"type": "chat"},
        },
    },
    {
        "id": "daily-brief",
        "name": "Create Daily Brief",
        "title": "Create Daily Brief",
        "pack": "Personal Admin Pack",
        "icon": "DAY",
        "description": "Collect tasks, memory, and recent notes into a local morning brief.",
        "fields": [
            {"id": "time", "label": "Run time", "placeholder": "09:00"},
            {"id": "scope", "label": "Brief scope", "placeholder": "tasks, reminders, decisions"},
        ],
        "home_steps": ["Search tasks", "Search memory", "Draft brief", "Require approval before schedule"],
        "targetView": "automations",
        "pipeline_steps": [
            {"id": "s1", "type": "memory_search", "label": "Search tasks and notes", "config": {"query": "{{input}}", "limit": 10}},
            {"id": "s2", "type": "llm_call", "label": "Draft brief", "config": {"prompt": "Create a concise daily brief from these local notes and tasks:\n\n{{s1.output}}", "model": ""}},
        ],
        "automation": {
            "trigger": {"type": "schedule", "schedule_type": "daily", "time_of_day": "09:00", "catch_up": True},
            "steps": [
                {"id": "s1", "type": "memory_search", "label": "Search tasks", "config": {"limit": 10, "input_source": "tasks reminders decisions"}},
                {"id": "s2", "type": "llm_call", "label": "Draft brief", "config": {"system_prompt": "Create a concise daily brief with priorities and reminders.", "input_source": "{{prev}}", "model": ""}},
                {"id": "s3", "type": "notification", "label": "Approval note", "config": {"title": "Daily brief ready", "message": "{{prev}}"}},
            ],
            "output_destination": {"type": "chat"},
        },
    },
    {
        "id": "extract-deadlines",
        "name": "Extract Deadlines",
        "title": "Extract Deadlines",
        "pack": "Document Pack",
        "icon": "DATE",
        "description": "Find dates, commitments, and reminder candidates from pasted or imported text.",
        "fields": [
            {"id": "source", "label": "Source name", "placeholder": "contract.txt"},
            {"id": "notify", "label": "Output", "placeholder": "Run history, task list, Telegram later"},
        ],
        "home_steps": ["Read source", "Detect dates", "Mark unverified reminders", "Dry-run notification"],
        "targetView": "documents",
        "pipeline_steps": [
            {"id": "s1", "type": "file_read", "label": "Read source", "config": {"filename": "{{input}}", "max_chars": 20000}},
            {"id": "s2", "type": "llm_call", "label": "Extract deadlines", "config": {"prompt": "Extract dates, commitments, owners, and reminder candidates. Mark uncertain items as unverified:\n\n{{s1.output}}", "model": ""}},
        ],
        "automation": {
            "trigger": {"type": "manual"},
            "steps": [
                {"id": "s1", "type": "file_read", "label": "Read source", "config": {"filename": "contract.txt", "max_chars": 20000}},
                {"id": "s2", "type": "llm_call", "label": "Extract dates", "config": {"system_prompt": "Extract dates and commitments. Mark uncertain items unverified.", "input_source": "{{prev}}", "model": ""}},
                {"id": "s3", "type": "notification", "label": "Dry-run notification", "config": {"title": "Deadline candidates", "message": "{{prev}}"}},
            ],
            "output_destination": {"type": "chat"},
        },
    },
    {
        "id": "research-topic",
        "name": "Research Topic",
        "title": "Research Topic",
        "pack": "Research Pack",
        "icon": "SRC",
        "description": "Turn a research question into a source-backed finding set and saved notes.",
        "fields": [
            {"id": "topic", "label": "Topic", "placeholder": "local-first AI automation"},
            {"id": "question", "label": "Decision question", "placeholder": "what should we build first?"},
        ],
        "home_steps": ["Plan queries", "Collect sources", "Synthesize findings", "Save reviewable memory"],
        "targetView": "chat",
        "pipeline_steps": [
            {"id": "s1", "type": "web_search", "label": "Collect sources", "config": {"query": "{{input}}", "num_results": 5}},
            {"id": "s2", "type": "llm_call", "label": "Synthesize findings", "config": {"prompt": "Synthesize these sources into concise findings with caveats:\n\n{{s1.output}}", "model": ""}},
            {"id": "s3", "type": "memory_save", "label": "Save findings", "config": {"text": "{{s2.output}}", "tags": "research,finding"}},
        ],
        "automation": {
            "trigger": {"type": "manual"},
            "steps": [
                {"id": "s1", "type": "web_search", "label": "Collect sources", "config": {"num_results": 5, "input_source": "{{prev}}"}},
                {"id": "s2", "type": "llm_call", "label": "Synthesize findings", "config": {"system_prompt": "Synthesize sources into concise findings with caveats.", "input_source": "{{prev}}", "model": ""}},
                {"id": "s3", "type": "memory_save", "label": "Save findings", "config": {"tags": "research,finding", "input_source": "{{prev}}"}},
            ],
            "output_destination": {"type": "chat"},
        },
    },
    {
        "id": "monitor-website",
        "name": "Monitor Website",
        "title": "Monitor Website",
        "pack": "Research Pack",
        "icon": "URL",
        "description": "Check a URL on a schedule, compare changes, and pause on failures.",
        "fields": [
            {"id": "url", "label": "URL", "placeholder": "https://example.com/changelog"},
            {"id": "cadence", "label": "Cadence", "placeholder": "daily"},
        ],
        "home_steps": ["Fetch page", "Compare with previous run", "Summarize change", "Auto-pause on failure"],
        "targetView": "automations",
        "pipeline_steps": [
            {"id": "s1", "type": "page_scrape", "label": "Scrape page", "config": {"url": "{{input}}", "mode": "text", "max_chars": 12000}},
            {"id": "s2", "type": "llm_call", "label": "Summarize change", "config": {"prompt": "Summarize notable changes or say no meaningful change:\n\n{{s1.output}}", "model": ""}},
        ],
        "automation": {
            "trigger": {"type": "schedule", "schedule_type": "daily", "time_of_day": "09:00", "catch_up": True},
            "steps": [
                {"id": "s1", "type": "page_scrape", "label": "Scrape page", "config": {"url": "https://example.com/changelog", "mode": "text", "max_chars": 12000}},
                {"id": "s2", "type": "llm_call", "label": "Summarize change", "config": {"system_prompt": "Summarize notable changes or say no meaningful change.", "input_source": "{{prev}}", "model": ""}},
                {"id": "s3", "type": "notification", "label": "Notify if changed", "config": {"title": "Website monitor", "message": "{{prev}}"}},
            ],
            "output_destination": {"type": "chat"},
        },
    },
    {
        "id": "save-memory",
        "name": "Save Memory",
        "title": "Save Memory",
        "pack": "Memory Pack",
        "icon": "MEM",
        "description": "Save a note as a reviewable memory candidate without treating generated text as fact.",
        "fields": [
            {"id": "memory", "label": "Memory candidate", "placeholder": "Project fact, preference, decision"},
            {"id": "source", "label": "Source", "placeholder": "operator note"},
        ],
        "home_steps": ["Create candidate", "Mark source class", "Keep generated text unverified", "Queue for admission"],
        "targetView": "memory",
        "pipeline_steps": [
            {"id": "s1", "type": "template", "label": "Memory candidate", "config": {"text": "{{input}}"}},
            {"id": "s2", "type": "memory_save", "label": "Save candidate", "config": {"text": "{{s1.output}}", "tags": "memory-candidate,operator-note"}},
        ],
        "automation": {
            "trigger": {"type": "manual"},
            "steps": [
                {"id": "s1", "type": "template", "label": "Memory candidate", "config": {"text": "Project fact, preference, or decision"}},
                {"id": "s2", "type": "memory_save", "label": "Save candidate", "config": {"tags": "memory-candidate,operator-note", "input_source": "{{prev}}"}},
            ],
            "output_destination": {"type": "memory", "tags": "memory-candidate,operator-note"},
        },
    },
]


def home_templates() -> list[dict]:
    builtins = [
        {
            "id": item["id"],
            "title": item["title"],
            "name": item["name"],
            "pack": item["pack"],
            "icon": item["icon"],
            "description": item["description"],
            "fields": deepcopy(item["fields"]),
            "steps": list(item["home_steps"]),
            "targetView": item["targetView"],
        }
        for item in WORKFLOW_TEMPLATES
    ]
    custom = []
    for item in _custom_templates("pipeline"):
        custom.append(
            {
                "id": item["id"],
                "title": item.get("name", item["id"]),
                "name": item.get("name", item["id"]),
                "pack": item.get("pack", "custom"),
                "icon": item.get("icon", "workflow"),
                "description": item.get("description", "Custom pipeline template"),
                "fields": deepcopy(item.get("fields", [])),
                "steps": [step.get("label") or step.get("type", "Step") for step in item.get("steps", [])],
                "targetView": "pipelines",
                "source": "custom",
            }
        )
    return builtins + custom


def pipeline_templates() -> list[dict]:
    builtins = [
        {
            "id": item["id"],
            "name": item["name"],
            "description": item["description"],
            "pack": item["pack"],
            "source": "built-in",
            "steps": deepcopy(item["pipeline_steps"]),
        }
        for item in WORKFLOW_TEMPLATES
    ]
    return builtins + _custom_templates("pipeline")


def automation_templates() -> list[dict]:
    result = []
    for item in WORKFLOW_TEMPLATES:
        automation = deepcopy(item["automation"])
        result.append(
            {
                "id": item["id"],
                "name": item["name"],
                "description": item["description"],
                "pack": item["pack"],
                "source": "built-in",
                "trigger": automation["trigger"],
                "steps": automation["steps"],
                "output_destination": automation["output_destination"],
                "enabled": False,
            }
        )
    return result + _custom_templates("automation")


def find_pipeline_template(template_id: str) -> dict | None:
    wanted = str(template_id or "").strip()
    if not wanted:
        return None
    for template in pipeline_templates():
        if template.get("id") == wanted:
            return deepcopy(template)
    return None


def find_automation_template(template_id: str) -> dict | None:
    wanted = str(template_id or "").strip()
    if not wanted:
        return None
    for template in automation_templates():
        if template.get("id") == wanted:
            return deepcopy(template)
    return None


def apply_template_inputs(template: dict, inputs: dict | None = None) -> dict:
    """Return a template copy with first-run/home inputs applied to block config."""
    result = deepcopy(template)
    values = {
        str(key).strip(): str(value).strip()
        for key, value in (inputs or {}).items()
        if str(key).strip() and str(value).strip()
    }
    if not values:
        return result

    primary = _primary_template_input(values)
    substitutions = {"input": primary, **values}
    _substitute_values(result, substitutions)
    _apply_template_input_heuristics(result, values, primary)
    return result


def _primary_template_input(values: dict[str, str]) -> str:
    for key in ("url", "source", "topic", "memory", "scope", "question"):
        if values.get(key):
            return values[key]
    return next(iter(values.values()), "")


def _substitute_values(value, substitutions: dict[str, str]):
    if isinstance(value, dict):
        for key, item in list(value.items()):
            value[key] = _substitute_values(item, substitutions)
        return value
    if isinstance(value, list):
        return [_substitute_values(item, substitutions) for item in value]
    if isinstance(value, str):
        text = value
        for key, replacement in substitutions.items():
            text = text.replace(f"{{{{{key}}}}}", replacement)
        return text
    return value


def _apply_template_input_heuristics(template: dict, values: dict[str, str], primary: str) -> None:
    if trigger := template.get("trigger"):
        if values.get("time"):
            trigger["time_of_day"] = values["time"]

    steps = template.get("steps") or []
    for step in steps:
        config = step.get("config") or {}
        step_type = step.get("type", "")
        if step_type == "page_scrape" and values.get("url"):
            config["url"] = values["url"]
        elif step_type == "file_read" and values.get("source"):
            config["filename"] = values["source"]
        elif step_type == "memory_search":
            query = values.get("scope") or values.get("topic") or values.get("question") or primary
            if query:
                config["query"] = query
                config["input_source"] = query
        elif step_type == "web_search":
            query = " ".join(part for part in (values.get("topic"), values.get("question")) if part).strip() or primary
            if query:
                config["query"] = query
                config["input_source"] = query
        elif step_type == "template" and values.get("memory"):
            config["text"] = values["memory"]


def save_custom_template(kind: str, template: dict) -> dict:
    if kind not in {"pipeline", "automation"}:
        raise ValueError("kind must be pipeline or automation")
    data = _load_custom_templates()
    items = data.setdefault(kind, [])
    template_id = str(template.get("id") or f"custom-{kind}-{uuid.uuid4().hex[:8]}")
    saved = {
        **deepcopy(template),
        "id": template_id,
        "source": "custom",
        "pack": template.get("pack") or "My Templates",
        "created_at": template.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%S"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    items[:] = [item for item in items if item.get("id") != template_id]
    items.insert(0, saved)
    del items[MAX_CUSTOM_TEMPLATES_PER_KIND:]
    _save_custom_templates(data)
    return deepcopy(saved)


def delete_custom_template(kind: str, template_id: str) -> bool:
    if kind not in {"pipeline", "automation"}:
        raise ValueError("kind must be pipeline or automation")
    data = _load_custom_templates()
    items = data.setdefault(kind, [])
    before = len(items)
    items[:] = [item for item in items if item.get("id") != template_id]
    deleted = len(items) != before
    if deleted:
        _save_custom_templates(data)
    return deleted


def _custom_templates(kind: str) -> list[dict]:
    data = _load_custom_templates()
    items = data.get(kind, [])
    return [deepcopy(item) for item in items if isinstance(item, dict)]


def _custom_templates_path() -> Path:
    path = settings.DATA_DIR / "workflow_templates" / "custom.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_custom_templates() -> dict:
    path = _custom_templates_path()
    if not path.exists():
        return {"pipeline": [], "automation": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"pipeline": [], "automation": []}
        data.setdefault("pipeline", [])
        data.setdefault("automation", [])
        return data
    except Exception:
        return {"pipeline": [], "automation": []}


def _save_custom_templates(data: dict) -> None:
    atomic_write(_custom_templates_path(), json.dumps(data, ensure_ascii=False, indent=2))
