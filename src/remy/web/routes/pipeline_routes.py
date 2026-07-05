"""
Pipeline (Flow Builder) routes — save, load, delete, and run pipelines.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from remy.core.file_utils import atomic_write
from remy.core.workflow_validation import PIPELINE_WORKFLOW_STEP_TYPES

logger = logging.getLogger("PipelineRoutes")
router = APIRouter()
ALLOWED_PIPELINE_STEP_TYPES = PIPELINE_WORKFLOW_STEP_TYPES


def _pipelines_dir() -> Path:
    from remy.config.settings import settings
    d = settings.DATA_DIR / "pipelines"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_pipeline(pipeline_id: str) -> dict:
    path = _pipelines_dir() / f"{pipeline_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Pipeline not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read pipeline: {exc}")


def _is_http_url(value: str) -> bool:
    from urllib.parse import urlparse

    try:
        parsed = urlparse(value.strip())
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _validate_pipeline_payload(
    *,
    name: str,
    steps: list[dict],
    drawflow_data: dict | None = None,
) -> list[str]:
    errors: list[str] = []
    if not name or not name.strip():
        errors.append("Pipeline name is required.")
    if not steps:
        errors.append("Add at least one action block between Start and Result.")

    for index, step in enumerate(steps or [], start=1):
        step_type = step.get("type")
        if step_type in {"http_request", "page_scrape"}:
            url = ((step.get("config") or {}).get("url") or "").strip()
            label = "HTTP Request" if step_type == "http_request" else "Page Scraper"
            if not url:
                errors.append(f"Step {index} {label} requires a URL.")
            elif "{{" not in url and not _is_http_url(url):
                errors.append(f"Step {index} {label} requires a valid http(s) URL.")

    from remy.core.workflow_validation import validate_visual_workflow_graph, validate_workflow_step_configs

    errors.extend(validate_workflow_step_configs(
        steps=steps,
        workflow_label="Pipeline",
        allowed_step_types=ALLOWED_PIPELINE_STEP_TYPES,
    ))

    if drawflow_data:
        errors.extend(validate_visual_workflow_graph(
            steps=steps,
            drawflow_data=drawflow_data,
            entry_name="start",
            terminal_name="result",
            workflow_label="Pipeline",
        ))

    return errors


def _raise_validation_errors(errors: list[str]) -> None:
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})


# ── List ──────────────────────────────────────────────────────────────────────

@router.get("/pipelines")
async def list_pipelines():
    """Return all saved pipelines (id, name, description, step_count)."""
    result = []
    for f in sorted(_pipelines_dir().glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            result.append({
                "id": data.get("id", f.stem),
                "name": data.get("name", "Untitled"),
                "description": data.get("description", ""),
                "step_count": len(data.get("steps", [])),
                "source_template_id": data.get("source_template_id", ""),
                "source_template_name": data.get("source_template_name", ""),
                "created_at": data.get("created_at", ""),
            })
        except Exception:
            pass
    return {"pipelines": result}


# ── Get one ───────────────────────────────────────────────────────────────────

@router.get("/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: str):
    """Return full pipeline definition."""
    _guard_id(pipeline_id)
    return _load_pipeline(pipeline_id)


# ── Save / update ─────────────────────────────────────────────────────────────

class PipelineSaveRequest(BaseModel):
    id: str | None = None
    name: str
    description: str = ""
    steps: list[dict]
    drawflow_data: dict | None = None  # raw Drawflow export for canvas restore
    source_template_id: str = ""
    source_template_name: str = ""


class PipelineTemplateSaveRequest(BaseModel):
    name: str
    description: str = ""
    steps: list[dict]
    drawflow_data: dict | None = None


class PipelineTemplateInstantiateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    inputs: dict[str, str] = {}


@router.post("/pipelines")
async def save_pipeline(body: PipelineSaveRequest):
    """Create or update a pipeline. Returns the saved pipeline with id."""
    pipeline_id = body.id or str(uuid.uuid4())[:8]
    _guard_id(pipeline_id)
    _raise_validation_errors(_validate_pipeline_payload(
        name=body.name,
        steps=body.steps,
        drawflow_data=body.drawflow_data,
    ))

    path = _pipelines_dir() / f"{pipeline_id}.json"
    existing: dict[str, Any] = {}
    if body.id and path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    data: dict[str, Any] = {
        "id": pipeline_id,
        "name": body.name.strip() or "Untitled",
        "description": body.description.strip(),
        "steps": body.steps,
        "drawflow_data": body.drawflow_data,
        "source_template_id": body.source_template_id.strip() or str(existing.get("source_template_id", "") or ""),
        "source_template_name": body.source_template_name.strip() or str(existing.get("source_template_name", "") or ""),
        "created_at": existing.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2))
    logger.info("Saved pipeline %s (%s)", pipeline_id, data["name"])
    return data


# ── Delete ────────────────────────────────────────────────────────────────────

@router.delete("/pipelines/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    """Delete a saved pipeline."""
    _guard_id(pipeline_id)
    path = _pipelines_dir() / f"{pipeline_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Pipeline not found")
    path.unlink()
    return {"deleted": True, "id": pipeline_id}


# ── Run ───────────────────────────────────────────────────────────────────────

class PipelineRunRequest(BaseModel):
    pipeline_id: str
    input_text: str = ""


@router.post("/pipelines/run")
async def run_pipeline(body: PipelineRunRequest):
    """Execute a pipeline. Returns SSE stream of step results."""
    _guard_id(body.pipeline_id)
    pipeline = _load_pipeline(body.pipeline_id)
    _raise_validation_errors(_validate_pipeline_payload(
        name=pipeline.get("name", ""),
        steps=pipeline.get("steps", []),
        drawflow_data=pipeline.get("drawflow_data"),
    ))

    async def _stream():
        from remy.core.pipeline_runner import run_pipeline_steps
        from remy.core.workflow_runs import finish_workflow_run, start_workflow_run

        run_record = start_workflow_run(
            kind="pipeline",
            workflow_id=body.pipeline_id,
            workflow_name=pipeline.get("name", ""),
            input_text=body.input_text,
            trigger="manual",
        )
        trace: list[dict] = []
        steps_run = 0
        final_output = ""
        yield f"data: {json.dumps({'type': 'run_started', 'run_id': run_record['run_id']}, ensure_ascii=False)}\n\n"
        try:
            async for event in run_pipeline_steps(pipeline["steps"], body.input_text):
                if event.get("type") == "step_done":
                    steps_run += 1
                    trace.append(_pipeline_trace_item(event, "ok"))
                elif event.get("type") == "step_error":
                    trace.append(_pipeline_trace_item(event, "error"))
                elif event.get("type") == "done":
                    final_output = event.get("output", "") or ""
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            finish_workflow_run(
                run_record,
                status="ok",
                output=final_output,
                trace=trace,
                steps_run=steps_run,
            )
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
            finish_workflow_run(
                run_record,
                status="error",
                error=error,
                trace=trace,
                steps_run=steps_run,
            )
            yield f"data: {json.dumps({'type': 'error', 'error': error}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class HomeTemplateRunRequest(BaseModel):
    template_id: str
    title: str
    pack: str = ""
    mode: str = "dry_run"
    inputs: dict[str, str] = {}
    steps: list[str] = []


@router.post("/pipelines/home-templates/run")
async def run_home_template(body: HomeTemplateRunRequest):
    """Record a first-screen template dry run or approval-gated run.

    Home templates are intentionally shallow: they prove the workflow shape,
    required fields, and safety gates before a user opens the full pipeline
    editor or enables scheduling.
    """
    from remy.core.workflow_runs import finish_workflow_run, start_workflow_run

    template_id = (body.template_id or "").strip()
    _guard_id(template_id)
    mode = (body.mode or "dry_run").strip().lower()
    if mode not in {"dry_run", "run"}:
        raise HTTPException(status_code=400, detail="mode must be dry_run or run")

    title = (body.title or "").strip() or template_id
    inputs = {
        str(key).strip(): str(value).strip()
        for key, value in (body.inputs or {}).items()
        if str(key).strip()
    }
    if not inputs:
        raise HTTPException(status_code=400, detail="At least one template input is required")

    steps = [str(step).strip() for step in (body.steps or []) if str(step).strip()]
    if not steps:
        steps = ["Validate inputs", "Prepare workflow", "Wait for approval"]

    status = "dry_run_ready" if mode == "dry_run" else "queued_for_human_approval"
    preview = " -> ".join(steps)
    trace = [
        {
            "index": index,
            "id": f"home-step-{index}",
            "type": "template_preview",
            "label": step,
            "status": "ok",
            "output": "Ready for dry run." if mode == "dry_run" else "Queued for human approval.",
            "output_length": 0,
            "output_truncated": False,
            "error": "",
        }
        for index, step in enumerate(steps, start=1)
    ]
    run_record = start_workflow_run(
        kind="home_template",
        workflow_id="templates",
        workflow_name=title,
        input_text=json.dumps(inputs, ensure_ascii=False),
        trigger=mode,
    )
    run_record.update(
        {
            "template_id": template_id,
            "title": title,
            "pack": body.pack.strip(),
            "mode": mode,
            "inputs": inputs,
            "cost": "$0.00 local",
            "retry_count": 0,
            "auto_paused": False,
        }
    )
    finished = finish_workflow_run(
        run_record,
        status=status,
        output=preview,
        trace=trace,
        steps_run=len(steps),
    )
    return {"run": _home_template_run_summary(finished)}


@router.get("/pipelines/home-templates/runs")
async def list_home_template_runs(limit: int = 20):
    from remy.core.workflow_runs import list_workflow_runs

    runs = list_workflow_runs("home_template", "templates", limit=limit)
    return {"runs": runs}


@router.delete("/pipelines/home-templates/runs")
async def clear_home_template_runs():
    from remy.core.workflow_runs import delete_workflow_runs

    return {"deleted": delete_workflow_runs("home_template", "templates")}


def _home_template_run_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": record.get("run_id", ""),
        "template_id": record.get("template_id", ""),
        "title": record.get("title") or record.get("workflow_name", ""),
        "pack": record.get("pack", ""),
        "mode": record.get("mode", ""),
        "status": record.get("status", ""),
        "duration_ms": record.get("duration_ms"),
        "cost": record.get("cost", "$0.00 local"),
        "retry_count": record.get("retry_count", 0),
        "auto_paused": record.get("auto_paused", False),
        "created_at": record.get("started_at", ""),
        "preview": record.get("output", ""),
        "inputs": record.get("inputs", {}),
    }


def _pipeline_trace_item(event: dict, status: str) -> dict:
    output = event.get("output", "") or ""
    return {
        "index": event.get("index", 0),
        "id": event.get("id", ""),
        "type": event.get("step_type", ""),
        "label": event.get("label", ""),
        "status": status,
        "output": output[:12000],
        "output_length": len(output),
        "output_truncated": len(output) > 12000,
        "error": event.get("error", "") or "",
    }


@router.get("/pipelines/{pipeline_id}/runs")
async def list_pipeline_runs(pipeline_id: str, limit: int = 50):
    _guard_id(pipeline_id)
    from remy.core.workflow_runs import list_workflow_runs

    return {"runs": list_workflow_runs("pipeline", pipeline_id, limit=limit)}


@router.get("/pipelines/{pipeline_id}/memory-report")
async def get_pipeline_memory_report(pipeline_id: str, limit: int = 20):
    _guard_id(pipeline_id)
    from remy.core.workflow_runs import summarize_workflow_memory

    return summarize_workflow_memory("pipeline", pipeline_id, limit=limit)


@router.get("/pipelines/{pipeline_id}/preflight")
async def get_pipeline_preflight(pipeline_id: str):
    _guard_id(pipeline_id)
    pipeline = _load_pipeline(pipeline_id)
    from remy.core.workflow_runs import list_workflow_runs
    from remy.core.workflow_safety import assess_workflow_safety

    runs = list_workflow_runs("pipeline", pipeline_id, limit=20)
    has_successful_run = any(run.get("status") == "ok" for run in runs)
    return assess_workflow_safety(
        kind="pipeline",
        steps=pipeline.get("steps", []),
        has_successful_run=has_successful_run,
        mode="manual_run",
    )


@router.get("/pipelines/{pipeline_id}/runs/{run_id}")
async def get_pipeline_run(pipeline_id: str, run_id: str):
    _guard_id(pipeline_id)
    _guard_id(run_id)
    from remy.core.workflow_runs import get_workflow_run

    record = get_workflow_run("pipeline", pipeline_id, run_id)
    if not record:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    return record


# ── Test single step ──────────────────────────────────────────────────────────

class StepTestRequest(BaseModel):
    step: dict
    input_text: str = ""


class HttpConnectionTestRequest(BaseModel):
    url: str
    method: str = "GET"
    body: str = ""
    headers: str = ""
    auth_secret_key: str = ""
    auth_scheme: str = "Bearer"


class ScrapeTestRequest(BaseModel):
    url: str
    mode: str = "text"
    max_chars: int = 12000


@router.post("/pipelines/test-step")
async def test_step(body: StepTestRequest):
    """Run a single step in isolation and return result."""
    from remy.core.pipeline_runner import run_single_step
    try:
        result = await run_single_step(body.step, body.input_text)
        return {"ok": True, "output": result}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ── Built-in templates ────────────────────────────────────────────────────────

@router.post("/workflows/http-test")
async def test_http_connection(body: HttpConnectionTestRequest):
    """Test an HTTP Request block without exposing local secret values."""
    import httpx

    url = body.url.strip()
    method = body.method.strip().upper() or "GET"
    if not _is_http_url(url):
        raise HTTPException(status_code=400, detail="Enter a valid http(s) URL.")
    if method not in {"GET", "POST"}:
        raise HTTPException(status_code=400, detail="HTTP test supports GET and POST only.")

    request_headers: dict[str, str] = {}
    for raw_line in (body.headers or "").splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            request_headers[key] = value

    secret_key = body.auth_secret_key.strip()
    if secret_key:
        from remy.core.pipeline_runner import _resolve_local_secret

        secret_value = _resolve_local_secret(secret_key)
        if not secret_value:
            raise HTTPException(status_code=400, detail="Selected Authorization secret is not configured.")
        scheme = (body.auth_scheme or "Bearer").strip()
        request_headers["Authorization"] = f"{scheme} {secret_value}" if scheme else secret_value

    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.request(
                method,
                url,
                headers=request_headers,
                content=body.body.encode("utf-8") if method == "POST" else None,
            )
    except Exception as exc:
        return {
            "ok": False,
            "status_code": None,
            "duration_ms": int((time.perf_counter() - started) * 1000),
            "preview": "",
            "error": _friendly_route_error(exc, kind="http"),
            "authorized": bool(secret_key),
        }

    preview = (response.text or "")[:700]
    ok = 200 <= response.status_code < 300
    return {
        "ok": ok,
        "status_code": response.status_code,
        "duration_ms": int((time.perf_counter() - started) * 1000),
        "preview": preview,
        "error": "" if ok else _friendly_status_error(response.status_code, kind="http"),
        "authorized": bool(secret_key),
    }


@router.post("/workflows/scrape-test")
async def test_page_scrape(body: ScrapeTestRequest):
    """Test a Page Scraper block with the same extraction path used by workflows."""
    url = body.url.strip()
    mode = body.mode.strip().lower() or "text"
    if not _is_http_url(url):
        raise HTTPException(status_code=400, detail="Enter a valid http(s) URL.")
    if mode not in {"text", "title", "links"}:
        raise HTTPException(status_code=400, detail="Extract mode must be text, title, or links.")

    started = time.perf_counter()
    from remy.core.pipeline_runner import _run_page_scrape

    output = await _run_page_scrape({
        "url": url,
        "mode": mode,
        "max_chars": body.max_chars,
    })
    ok = not output.startswith("[Scrape error:")
    return {
        "ok": ok,
        "duration_ms": int((time.perf_counter() - started) * 1000),
        "preview": output[:1200],
        "error": "" if ok else _strip_error_prefix(output),
        "mode": mode,
    }


def _friendly_status_error(status_code: int, *, kind: str) -> str:
    if kind == "http":
        if status_code in {401, 403}:
            return f"HTTP {status_code}: check Authorization secret, auth scheme, and endpoint permissions."
        if status_code == 404:
            return "HTTP 404: endpoint was not found. Check URL path and method."
        if 400 <= status_code < 500:
            return f"HTTP {status_code}: request was rejected. Check URL, method, headers, and body."
        if status_code >= 500:
            return f"HTTP {status_code}: server error. Retry later or enable block retry."
    return f"HTTP {status_code}"


def _friendly_route_error(exc: Exception, *, kind: str) -> str:
    detail = str(exc).strip()
    if "timed out" in detail.lower() or "timeout" in detail.lower():
        return "Request timed out. Check the URL or retry later."
    if kind == "http":
        return f"Request could not be completed. Check URL, network, and method. Detail: {detail or exc.__class__.__name__}"
    return f"Page could not be read. Check URL or try a different extract mode. Detail: {detail or exc.__class__.__name__}"


def _strip_error_prefix(value: str) -> str:
    text = str(value or "").strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if text.lower().startswith("scrape error:"):
        text = text.split(":", 1)[1].strip()
    if text.lower().startswith("http error:"):
        text = text.split(":", 1)[1].strip()
    return text or "Workflow test failed."


TEMPLATES = [
    {
        "id": "_tpl_research",
        "name": "Research topic",
        "description": "Web search → AI analysis → save to memory",
        "steps": [
            {"id": "s1", "type": "web_search", "label": "Web search",
             "config": {"query": "{{input}}", "num_results": 5}},
            {"id": "s2", "type": "llm_call", "label": "AI analysis",
             "config": {"prompt": "Analyze these search results and provide a concise summary:\n\n{{s1.output}}",
                        "model": ""}},
            {"id": "s3", "type": "memory_save", "label": "Save to memory",
             "config": {"text": "{{s2.output}}", "tags": "research"}},
        ],
    },
    {
        "id": "_tpl_summarize",
        "name": "Summarize and remember",
        "description": "Summarize text with AI and save to memory",
        "steps": [
            {"id": "s1", "type": "llm_call", "label": "AI summarize",
             "config": {"prompt": "Summarize this text into key points (3-5 sentences):\n\n{{input}}",
                        "model": ""}},
            {"id": "s2", "type": "memory_save", "label": "Save to memory",
             "config": {"text": "{{s1.output}}", "tags": "summary"}},
        ],
    },
    {
        "id": "_tpl_memory_search",
        "name": "Memory search + answer",
        "description": "Find relevant memories and generate a response",
        "steps": [
            {"id": "s1", "type": "memory_search", "label": "Memory search",
             "config": {"query": "{{input}}", "limit": 5}},
            {"id": "s2", "type": "llm_call", "label": "Generate answer",
             "config": {"prompt": "Based on this information from memory, answer the question '{{input}}':\n\n{{s1.output}}",
                        "model": ""}},
        ],
    },
]


@router.get("/pipelines/templates/list")
async def list_templates():
    from remy.core.workflow_templates import pipeline_templates

    return {"templates": pipeline_templates()}


@router.post("/pipelines/templates")
async def save_pipeline_template(body: PipelineTemplateSaveRequest):
    _raise_validation_errors(_validate_pipeline_payload(
        name=body.name,
        steps=body.steps,
        drawflow_data=body.drawflow_data,
    ))
    from remy.core.workflow_templates import save_custom_template

    template = save_custom_template("pipeline", {
        "name": body.name.strip() or "Untitled template",
        "description": body.description.strip(),
        "steps": body.steps,
        "drawflow_data": body.drawflow_data,
    })
    return {"template": template}


@router.post("/pipelines/templates/{template_id}/instantiate")
async def instantiate_pipeline_template(template_id: str, body: PipelineTemplateInstantiateRequest | None = None):
    _guard_id(template_id)
    from remy.core.workflow_templates import apply_template_inputs, find_pipeline_template

    template = find_pipeline_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Pipeline template not found")
    template = apply_template_inputs(template, body.inputs if body else {})

    name = (body.name if body else None) or template.get("name") or "Untitled pipeline"
    description = (body.description if body else None)
    if description is None:
        description = template.get("description", "")

    pipeline = await save_pipeline(PipelineSaveRequest(
        name=name,
        description=description,
        steps=template.get("steps", []),
        drawflow_data=template.get("drawflow_data"),
        source_template_id=template_id,
        source_template_name=template.get("name", ""),
    ))
    return {"pipeline": pipeline, "template_id": template_id}


@router.delete("/pipelines/templates/{template_id}")
async def delete_pipeline_template(template_id: str):
    _guard_id(template_id)
    from remy.core.workflow_templates import delete_custom_template

    if not delete_custom_template("pipeline", template_id):
        raise HTTPException(status_code=404, detail="Custom pipeline template not found")
    return {"deleted": True, "id": template_id}


@router.get("/pipelines/home-templates/list")
async def list_home_templates():
    from remy.core.workflow_templates import home_templates

    return {"templates": home_templates()}


# ── Guard ─────────────────────────────────────────────────────────────────────

def _guard_id(pipeline_id: str):
    if not pipeline_id or "/" in pipeline_id or "\\" in pipeline_id or ".." in pipeline_id:
        raise HTTPException(status_code=400, detail="Invalid pipeline id")


# ── Workflow files (shared storage for File Read / File Write blocks) ─────────
#
# Makes the data/workflow_files directory visible: pipelines' File Read picks
# from these, File Write results can be downloaded. Names are sanitized by the
# same helper the runner uses (`_safe_workflow_file`), so the API and the
# blocks agree on what a valid file is.

_WORKFLOW_FILE_MAX_BYTES = 5 * 1024 * 1024  # 5 MB — these are text inputs, not media


@router.get("/workflow-files")
async def list_workflow_files():
    from remy.core.pipeline_runner import _workflow_files_dir

    files = []
    for path in sorted(_workflow_files_dir().iterdir()):
        if not path.is_file():
            continue
        stat = path.stat()
        files.append({
            "name": path.name,
            "size": stat.st_size,
            "modified": stat.st_mtime,
        })
    return {"files": files}


@router.post("/workflow-files")
async def upload_workflow_file(file: UploadFile = File(...)):
    from remy.core.pipeline_runner import _safe_workflow_file

    try:
        path = _safe_workflow_file(file.filename or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    content = await file.read()
    if len(content) > _WORKFLOW_FILE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 5 MB)")

    path.write_bytes(content)
    return {"uploaded": True, "name": path.name, "size": len(content)}


@router.get("/workflow-files/{filename}")
async def download_workflow_file(filename: str):
    from remy.core.pipeline_runner import _safe_workflow_file

    try:
        path = _safe_workflow_file(filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, filename=path.name)


@router.delete("/workflow-files/{filename}")
async def delete_workflow_file(filename: str):
    from remy.core.pipeline_runner import _safe_workflow_file

    try:
        path = _safe_workflow_file(filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    path.unlink()
    return {"deleted": True, "name": filename}
