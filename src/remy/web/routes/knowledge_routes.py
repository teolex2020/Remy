"""
Knowledge routes — research, health, facts, identity, stats, knowledge base CRUD.
"""

import json
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from remy.core.tool_handlers.profile import (
    is_valid_person_payload,
    merge_identity_people,
    sanitize_person_payload,
    sanitize_identity_profile_payload,
    _format_profile_content,
)
from remy.web.routes._goal_serialization import serialize_goal_as_calendar_task
from remy.web.routes._helpers import _get_api, run_in_thread, _TIMEOUT_SLOW
from remy.web.routes._research_serialization import serialize_completed_research_project

logger = logging.getLogger("WebAPI")

router = APIRouter()

_IDENTITY_PROFILE_FIELDS = (
    "name",
    "age",
    "location",
    "occupation",
    "languages",
    "family",
    "email",
    "phone",
    "social",
    "personal_focus",
    "interests",
    "notes",
)

_ALLOWED_EXTENSIONS = {".txt", ".md", ".csv"}
_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# ============== RESEARCH ==============


@router.get("/knowledge/research")
async def get_research_projects():
    """List all research projects (active and completed)."""
    api = _get_api()
    active = api.get_active_research_projects()

    def _query():
        with api.brain_lock:
            return api.brain.search(query="", tags=["research-project", "completed"], limit=50)

    completed_recs = await run_in_thread(_query)
    completed = [serialize_completed_research_project(r) for r in completed_recs]

    return {"active": active, "completed": completed}


# ============== METRICS ==============


@router.get("/knowledge/metrics")
async def get_metric_data(limit: int = 200):
    """Get recent tracked metrics (deduplicated, sorted by timestamp desc)."""
    api = _get_api()

    def _query():
        with api.brain_lock:
            metrics = api.brain.search(query="", tags=["metric"], limit=limit)
            legacy_metrics = api.brain.search(query="", tags=["health-metric"], limit=limit)
            return list(metrics or []) + list(legacy_metrics or [])

    recs = await run_in_thread(_query)
    data = []
    seen: set[str] = set()  # Dedup key: metric+value+date (ignore time)
    for r in recs:
        meta = r.metadata or {}
        metric = meta.get("metric", "")
        value = meta.get("value")
        ts = meta.get("timestamp", "")
        # Dedup key: same metric + same value + same date (YYYY-MM-DD)
        date_key = ts[:10] if ts else ""
        dedup_key = f"{metric}|{value}|{date_key}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        data.append(
            {
                "metric": metric,
                "value": value,
                "unit": meta.get("unit"),
                "timestamp": ts,
                "notes": meta.get("notes"),
            }
        )

    # Sort by timestamp descending (newest first)
    data.sort(key=lambda x: x.get("timestamp") or "", reverse=True)
    return {"data": data}


# ============== FACTS ==============


@router.get("/knowledge/facts")
async def get_extracted_facts(limit: int = 50):
    """Get recent extracted facts."""
    api = _get_api()

    def _query():
        with api.brain_lock:
            return api.brain.search(query="", tags=["extracted-fact"], limit=limit)

    recs = await run_in_thread(_query)
    data = []
    for r in recs:
        meta = r.metadata or {}
        data.append(
            {
                "content": r.content,
                "structure": meta.get("structure"),
                "source": meta.get("source"),
                "extracted_at": meta.get("extracted_at"),
            }
        )
    return {"data": data}


# ============== IDENTITY ==============


@router.get("/knowledge/identity")
async def get_identity():
    """Get user profile and people records for the Identity tab."""
    api = _get_api()

    def _query():
        from remy.core.brain_tools import get_user_profile_record
        _prof = get_user_profile_record(api.brain, api.brain_lock)
        with api.brain_lock:
            _people = api.brain.search(query="", tags=["person"], limit=50)
        return _prof, _people

    profile_rec, people = await run_in_thread(_query)

    profile_data = {}
    if profile_rec:
        raw_meta = profile_rec.metadata or {}
        meta = sanitize_identity_profile_payload(raw_meta)
        repaired_meta = dict(raw_meta)
        changed = False
        for key, value in meta.items():
            if raw_meta.get(key, "") != value:
                repaired_meta[key] = value
                changed = True
        if changed:
            repaired_meta["protected_fields"] = sorted(
                field for field in ("phone", "email") if repaired_meta.get(field)
            )
            try:
                with api.brain_lock:
                    api.brain.update(
                        profile_rec.id,
                        content=_format_profile_content(repaired_meta),
                        metadata=repaired_meta,
                    )
            except Exception:
                logger.debug("Identity profile auto-repair failed", exc_info=True)
        for key in _IDENTITY_PROFILE_FIELDS:
            profile_data[key] = meta.get(key, "")
        profile_data["id"] = profile_rec.id
        profile_data["verified"] = bool(raw_meta.get("verified", False))

    people_list = []
    for p in people:
        meta = p.metadata or {}
        if not is_valid_person_payload(meta, getattr(p, "content", "")):
            repaired_person = sanitize_person_payload(meta, getattr(p, "content", ""))
            if repaired_person:
                try:
                    with api.brain_lock:
                        api.brain.update(p.id, metadata=repaired_person)
                    meta = repaired_person
                except Exception:
                    logger.debug("Person auto-repair failed for %s", p.id, exc_info=True)
            else:
                try:
                    with api.brain_lock:
                        tags = [tag for tag in list(getattr(p, "tags", []) or []) if tag != "person"]
                        if "identity-invalid" not in tags:
                            tags.append("identity-invalid")
                        bad_meta = dict(meta)
                        bad_meta["type"] = "invalid_person_capture"
                        api.brain.update(p.id, metadata=bad_meta, tags=tags)
                except Exception:
                    logger.debug("Invalid person quarantine failed for %s", p.id, exc_info=True)
                continue
        people_list.append(
            {
                "id": p.id,
                "full_name": meta.get("full_name", p.content[:100]),
                "role": meta.get("role", ""),
                "birth_date": meta.get("birth_date", ""),
                "birth_place": meta.get("birth_place", ""),
                "verified": meta.get("verified", False),
                "trust_score": meta.get("trust_score", 0.5),
            }
        )

    people_list = merge_identity_people(people_list, profile_data.get("family", ""))

    return {"profile": profile_data, "people": people_list}


@router.put("/knowledge/identity/profile")
async def update_identity_profile(body: dict):
    """Update user profile from the Identity tab."""
    api = _get_api()
    result = api.execute_tool("store_user_profile", body)
    return json.loads(result)


@router.put("/knowledge/identity/person/{record_id}")
async def update_identity_person(record_id: str, body: dict):
    """Update a person record. Marks as verified (user-confirmed, trust=1.0)."""
    api = _get_api()

    def _update():
        with api.brain_lock:
            rec = api.brain.get(record_id)
            if not rec:
                return "not_found"
            if "person" not in (rec.tags or []):
                return "not_person"
            meta = dict(rec.metadata) if rec.metadata else {}
            for key in ("full_name", "role", "birth_date", "birth_place"):
                if key in body and body[key] is not None:
                    meta[key] = body[key]
            meta["verified"] = True
            meta["source"] = "user-confirmed"
            meta["trust_score"] = 1.0
            parts = [meta.get("full_name", "")]
            if meta.get("role"):
                parts.append(meta["role"])
            if meta.get("birth_date"):
                parts.append(f"born {meta['birth_date']}")
            if meta.get("birth_place"):
                parts.append(f"in {meta['birth_place']}")
            api.brain.update(record_id, content=", ".join(parts), metadata=meta)
            return "ok"

    result = await run_in_thread(_update)
    if result == "not_found":
        raise HTTPException(status_code=404, detail="Person not found")
    if result == "not_person":
        raise HTTPException(status_code=400, detail="Record is not a person")
    return {"updated": True, "id": record_id}


# ============== CALENDAR ==============


@router.get("/knowledge/calendar")
async def get_calendar_tasks():
    """Return scheduled tasks, todos, and goals for calendar display."""
    import re as _re

    api = _get_api()

    def _query():
        with api.brain_lock:
            scheduled = api.brain.search(query="", tags=["scheduled-task"], limit=200)
            todos = api.brain.search(query="", tags=["todo-item"], limit=200)
            goals = api.brain.search(query="", tags=["autonomous-goal"], limit=100)
        return scheduled, todos, goals

    scheduled_recs, todo_recs, goal_recs = await run_in_thread(_query)
    tasks = []

    # --- Scheduled tasks (existing) ---
    for r in scheduled_recs:
        meta = r.metadata or {}
        date = meta.get("due_date")
        if not date:
            continue
        tasks.append(
            {
                "id": r.id,
                "description": meta.get("description", r.content),
                "due_date": date,
                "repeat": meta.get("repeat"),
                "status": meta.get("status", "active"),
                "source": "scheduled",
                "content": r.content,
            }
        )

    # --- Todo items ---
    for r in todo_recs:
        meta = r.metadata or {}
        if meta.get("type") != "todo_item":
            continue
        s = meta.get("status", "pending")
        if s in ("done",):
            mapped = "completed"
        else:
            mapped = "active"
        date = meta.get("due_date")
        if not date:
            created = meta.get("created_at", "")
            date = created[:10] if len(created) >= 10 else None
        if not date:
            continue
        title = meta.get("title")
        if not title:
            content = r.content or ""
            m = _re.match(r"Todo\s*\[[A-Z]+\]:\s*(.+?)(?:\s*\|\s*Due:.*)?$", content)
            title = m.group(1).strip() if m else content
        tasks.append(
            {
                "id": r.id,
                "description": title,
                "due_date": date,
                "repeat": meta.get("repeat"),
                "status": mapped,
                "source": "todo",
                "content": r.content,
            }
        )

    # --- Autonomous goals ---
    for r in goal_recs:
        item = serialize_goal_as_calendar_task(r)
        if item:
            tasks.append(item)

    return {"tasks": tasks}


# ============== KNOWLEDGE STATS ==============


@router.get("/knowledge/stats")
async def get_knowledge_stats():
    """Get high-level brain statistics."""
    from remy.core.usage_stats import usage_tracker

    stats = usage_tracker.get_stats()

    return {
        "status": "online",
        "memory_backend": "AuraMemory",
        "version": "2.0.0",
        "usage": {
            "user_tokens": stats["user_tokens"],
            "autonomy_tokens": stats["autonomy_tokens"],
            "total_tokens": stats["user_tokens"] + stats["autonomy_tokens"],
            "last_updated": stats["last_updated"],
        },
    }


# ============== KNOWLEDGE BASE ==============


@router.get("/knowledge/base")
async def list_knowledge_base(limit: int = 100, offset: int = 0, query: str = ""):
    """List records in the knowledge base (all brain records)."""
    from remy.core.agent_tools import brain, brain_lock

    if brain is None:
        raise HTTPException(status_code=503, detail="Brain is not available")

    q = query.strip()

    def _query():
        with brain_lock:
            if q:
                results = brain.search(query=q, limit=limit)
            else:
                results = brain.search(limit=limit)
            total = brain.count() if not q else len(results)
            return results, total

    results, total = await run_in_thread(_query)

    items = []
    for r in results:
        meta = r.metadata or {} if hasattr(r, "metadata") else {}
        # Extract timestamp: metadata.timestamp > metadata.created_at > record.created_at
        ts = (
            meta.get("timestamp")
            or meta.get("created_at")
            or (r.created_at if hasattr(r, "created_at") else None)
            or ""
        )
        items.append(
            {
                "id": r.id,
                "text": r.content,
                "content": r.content,
                "level": str(r.level) if hasattr(r, "level") else "DOMAIN",
                "tags": list(r.tags) if hasattr(r, "tags") else [],
                "strength": round(r.strength, 3) if hasattr(r, "strength") else 1.0,
                "metadata": meta,
                "timestamp": ts,
            }
        )

    return {"items": items, "total": total, "offset": offset, "limit": limit}


@router.post("/knowledge/ingest")
async def ingest_knowledge(body: dict):
    """Ingest text into the knowledge base via brain.store()."""
    from remy.core.agent_tools import Level, brain, brain_lock

    if brain is None:
        raise HTTPException(status_code=503, detail="Brain is not available")

    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    pin = bool(body.get("pin", False))
    level = Level.IDENTITY if pin else Level.DOMAIN

    def _ingest():
        if len(text) > 1000:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [text]
            count = 0
            with brain_lock:
                for para in paragraphs:
                    brain.store(para, level=level, tags=["kb-ingested"])
                    count += 1
                brain.flush()
                total = brain.count()
            return {"ingested": count, "pinned": pin, "total": total}
        else:
            with brain_lock:
                brain.store(text, level=level, tags=["kb-ingested"])
                brain.flush()
                total = brain.count()
            return {"ingested": 1, "pinned": pin, "total": total}

    return await run_in_thread(_ingest)


@router.post("/knowledge/upload")
async def upload_knowledge_file(file: UploadFile = File(...), pin: bool = Form(False)):
    """Upload a file to the knowledge base (.txt, .md, .csv)."""
    from remy.core.agent_tools import Level, brain, brain_lock

    if brain is None:
        raise HTTPException(status_code=503, detail="Brain is not available")

    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(_ALLOWED_EXTENSIONS)}",
        )

    content_bytes = await file.read()
    if len(content_bytes) > _MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {len(content_bytes)} bytes (max {_MAX_FILE_SIZE})",
        )

    try:
        text = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text")

    if ext == ".csv":
        import csv
        import io

        reader = csv.reader(io.StringIO(text))
        rows = [" | ".join(cell.strip() for cell in row if cell.strip()) for row in reader]
        chunks = [r for r in rows if r]
    else:
        chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not chunks:
            chunks = [text.strip()] if text.strip() else []

    if not chunks:
        raise HTTPException(status_code=400, detail="File is empty or has no parseable content")

    level = Level.IDENTITY if pin else Level.DOMAIN

    def _upload():
        count = 0
        with brain_lock:
            for chunk in chunks:
                brain.store(chunk, level=level, tags=["kb-ingested"])
                count += 1
            brain.flush()
            total = brain.count()
        return count, total

    count, total = await run_in_thread(_upload)

    return {
        "filename": file.filename,
        "extension": ext,
        "chunks": count,
        "pinned": pin,
        "total": total,
    }


_BULK_ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst",
    ".pdf", ".docx", ".xlsx", ".csv",
    ".html", ".htm", ".xml", ".json", ".jsonl",
    ".zip",
}
_BULK_MAX_FILE_SIZE = 512 * 1024 * 1024   # 512 MB per file
_BULK_MAX_TOTAL_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB total


@router.post("/knowledge/bulk-ingest")
async def bulk_ingest_knowledge(
    files: list[UploadFile] = File(...),
    pin: bool = Form(False),
    tags: str = Form(""),
    source_label: str = Form("bulk-upload"),
):
    """Bulk ingest documents (PDF, DOCX, XLSX, CSV, HTML, XML, TXT, ZIP) into the brain.

    Returns an SSE stream of progress events. Each event is a JSON object:
      {phase, file, stored, skipped, message, ...}
    Final event has phase="done" with totals.
    """
    from remy.core.agent_tools import brain
    from remy.core.bulk_ingestion import SUPPORTED_EXTENSIONS, sse_bulk_ingest_stream

    if brain is None:
        raise HTTPException(status_code=503, detail="Brain is not available")

    # Validate files
    total_size = 0
    saved_paths: list[Path] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="remy_bulk_upload_"))

    try:
        for uf in files:
            ext = Path(uf.filename or "").suffix.lower()
            if ext not in _BULK_ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(_BULK_ALLOWED_EXTENSIONS))}",
                )
            content = await uf.read()
            if len(content) > _BULK_MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"{uf.filename}: file too large ({len(content) // 1024 // 1024} MB, max 512 MB)",
                )
            total_size += len(content)
            if total_size > _BULK_MAX_TOTAL_SIZE:
                raise HTTPException(status_code=400, detail="Total upload size exceeds 2 GB limit")
            dest = tmp_dir / (uf.filename or f"file_{len(saved_paths)}{ext}")
            dest.write_bytes(content)
            saved_paths.append(dest)

    except HTTPException:
        import shutil
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        raise

    extra_tags = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    async def _stream_and_cleanup():
        import shutil as _shutil
        try:
            async for event in sse_bulk_ingest_stream(
                saved_paths,
                tags=extra_tags,
                pin=pin,
                source_label=source_label,
            ):
                yield event
        finally:
            _shutil.rmtree(str(tmp_dir), ignore_errors=True)

    return StreamingResponse(
        _stream_and_cleanup(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/knowledge/base/{record_id}")
async def delete_knowledge_item(record_id: str):
    """Delete a record from the knowledge base."""
    from remy.core.agent_tools import brain, brain_lock

    if brain is None:
        raise HTTPException(status_code=503, detail="Brain is not available")

    def _delete():
        with brain_lock:
            return brain.delete(record_id)

    success = await run_in_thread(_delete)
    if not success:
        raise HTTPException(status_code=404, detail=f"Record '{record_id}' not found")
    return {"deleted": True, "id": record_id}
