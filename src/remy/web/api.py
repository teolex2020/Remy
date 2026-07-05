"""
FastAPI routes for the web/desktop GUI — REST + WebSocket.
"""

import base64
import json
import logging
import os
import platform
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import asyncio

from remy.config.settings import settings
from remy.core.agent_tools import brain, brain_lock, brain_run, level_name, get_brain_startup_status
from remy.core.brain_tools import execute_tool
from remy.core.event_bus import event_bus
from remy.core.brain_tools import get_active_research_projects
from remy.core.usage_stats import usage_tracker
from remy.core.metrics import metrics_collector
from remy.web.routes._research_serialization import serialize_completed_research_project

logger = logging.getLogger("WebAPI")

router = APIRouter(prefix="/api")

_start_time = time.time()

# Will be set by desktop_gui.py at startup
_session_manager = None



class RateLimitMiddleware:
    """Simple in-memory rate limiter: 120 requests/minute per IP."""

    _MAX_IPS = 1000  # Cap dict size to prevent memory growth from many IPs

    def __init__(self, app, max_requests: int = 120, window_seconds: int = 60):
        self.app = app
        self.max_requests = max_requests
        self.window = window_seconds
        self._requests: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        # Skip rate limiting for static assets
        if not path.startswith("/api"):
            await self.app(scope, receive, send)
            return

        # Get client IP
        client = scope.get("client")
        ip = client[0] if client else "unknown"

        async with self._lock:
            now = time.time()
            # Clean old entries
            if ip in self._requests:
                self._requests[ip] = [t for t in self._requests[ip] if now - t < self.window]
            else:
                self._requests[ip] = []

            # Cap dict size — evict oldest IPs
            if len(self._requests) > self._MAX_IPS:
                sorted_ips = sorted(
                    self._requests.items(),
                    key=lambda x: max(x[1]) if x[1] else 0,
                    reverse=True,
                )
                self._requests = dict(sorted_ips[:500])

            if len(self._requests[ip]) >= self.max_requests:
                response = JSONResponse(
                    {"detail": "Rate limit exceeded. Try again later."},
                    status_code=429
                )
                await response(scope, receive, send)
                return

            self._requests[ip].append(now)

        await self.app(scope, receive, send)


class NoCacheStaticMiddleware:
    """ASGI middleware: sets Cache-Control: no-cache for JS/CSS files.

    This forces the browser to revalidate with the server (ETag/If-None-Match)
    on every request, preventing stale cached JS from causing errors like
    'window.apiClient.getKnowledgeBase is not a function'.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        needs_no_cache = path.endswith((".js", ".css"))

        if not needs_no_cache:
            await self.app(scope, receive, send)
            return

        # Intercept response headers to inject Cache-Control
        async def send_with_no_cache(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                # Remove any existing Cache-Control
                headers = [
                    (k, v) for k, v in headers
                    if k.lower() != b"cache-control"
                ]
                headers.append((b"cache-control", b"no-cache, must-revalidate"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_no_cache)


class RequestLoggingMiddleware:
    """ASGI middleware: logs HTTP requests with method, path, status, duration.

    Sets request_id and channel context vars for all downstream loggers.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if not path.startswith("/api"):
            await self.app(scope, receive, send)
            return

        from remy.core.logging_config import (
            ctx_channel, ctx_request_id, generate_request_id,
        )

        request_id = generate_request_id()
        method = scope.get("method", "?")

        req_token = ctx_request_id.set(request_id)
        ch_token = ctx_channel.set("web")

        start = time.time()
        status_code = 0

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 0)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            status_code = 500
            raise
        finally:
            duration_sec = time.time() - start
            duration_ms = int(duration_sec * 1000)
            logger.info(
                "HTTP %s %s %d %dms",
                method, path, status_code, duration_ms,
            )
            metrics_collector.record_http_request(method, path, status_code, duration_sec)
            ctx_request_id.reset(req_token)
            ctx_channel.reset(ch_token)


# ============== SCHEDULER LIFECYCLE ==============

from remy.web.scheduler import Scheduler

# Single global scheduler instance
_scheduler = Scheduler()

async def start_scheduler():
    if _scheduler.running:
        return
    logger.info("Starting background scheduler...")
    await _scheduler.start()

async def load_push_subscription():
    try:
        from remy.web.push import load_subscription
        load_subscription()
    except Exception as e:
        logger.warning(f"Failed to load push subscription: {e}")

async def shutdown_cleanup():
    logger.info("Server shutting down — closing active session...")
    try:
        manager = get_session_manager()
        await manager.close_session()
    except Exception as e:
        logger.warning(f"Session close on shutdown failed: {e}")
    logger.info("Stopping background scheduler...")
    try:
        await _scheduler.stop()
    except Exception as e:
        logger.warning(f"Scheduler stop on shutdown failed: {e}")
    try:
        from remy.core.agent_tools import close_brain

        close_brain()
    except Exception as e:
        logger.warning(f"Brain close on shutdown failed: {e}")



def set_session_manager(manager):
    global _session_manager
    _session_manager = manager


def get_session_manager():
    if _session_manager is None:
        raise RuntimeError("Session manager not initialized")
    return _session_manager


# ============== PYDANTIC MODELS ==============


class SearchQuery(BaseModel):
    query: str
    tags: str | None = None


class UpdateRecordPayload(BaseModel):
    content: str | None = None
    tags: str | None = None
    level: str | None = None


class CreateRecordPayload(BaseModel):
    content: str
    tags: str | None = None
    level: str | None = None


class SettingsPayload(BaseModel):
    gemini_api_key: str | None = None
    summary_model: str | None = None
    gemini_voice: str | None = None
    telegram_bot_token: str | None = None
    proactive_chat_id: int | None = None
    review_model: str | None = None


# ============== REST ENDPOINTS ==============


@router.get("/llm-optimization/measurements")
async def get_llm_optimization_measurements(limit: int = 100):
    """Return persisted LLM optimization measurements and aggregate summary."""
    from remy.core.llm_optimization_metrics import list_measurements, summarize_measurements

    items = list_measurements(limit=limit)
    return {"items": items, "summary": summarize_measurements(items)}


@router.delete("/llm-optimization/measurements")
async def clear_llm_optimization_measurements():
    """Clear persisted LLM optimization measurements."""
    from remy.core.llm_optimization_metrics import clear_measurements

    clear_measurements()
    return {"ok": True}


@router.get("/llm-optimization/models")
async def get_llm_optimization_models():
    """Return registered models with input/output prices for the lab selector.

    The lab window prices context reduction against the selected model; cost is
    computed live from these registry prices, so the same reduction shows a
    different dollar saving per model.
    """
    from remy.core.model_registry import list_registered_models

    models = []
    for entry in list_registered_models():
        name = entry.get("model_name") or entry.get("name") or ""
        if not name:
            continue
        models.append({
            "model": name,
            "provider": entry.get("provider"),
            "input_price_per_1m_usd": entry.get("input_price"),
            "output_price_per_1m_usd": entry.get("output_price"),
        })
    return {"models": models}


@router.get("/stats")
async def get_brain_stats():
    """Get brain statistics."""
    """Get brain statistics."""
    try:
        count = await brain_run(brain.count, timeout=3.0)
        stats  = await brain_run(brain.stats,  timeout=3.0)
    except Exception:
        return {"error": "busy", "message": "Agent is busy, try again in a moment"}
    
    # Get token usage
    usage = usage_tracker.get_stats()
    # Calculate total if not present (handled by JS but good to have)
    usage["total_tokens"] = usage.get("user_tokens", 0) + usage.get("autonomy_tokens", 0)
    
    return {"total_records": count, "stats": stats, "usage": usage}


@router.get("/records")
async def list_records(tags: str | None = None, offset: int = 0, limit: int = 50):
    """List memory records."""
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    try:
        records = await brain_run(brain.list_records, tags=tag_list, min_strength=0.01, timeout=4.0)
    except Exception:
        return {"items": [], "total": 0, "busy": True}

    items = []
    # Simple slicing for now since brain doesn't have offset support
    for r in records[offset : offset + limit]:
        item = {
            "id": r.id,
            "content": r.content[:300],
            "tags": list(r.tags) if r.tags else [],
            "level": str(r.level),
            "strength": round(r.strength, 3),
            "activation_count": r.activation_count,
            "verified": (r.metadata or {}).get("verified"),
            "source": (r.metadata or {}).get("source"),
            "trust_score": (r.metadata or {}).get("trust_score"),
        }
        if hasattr(r, "importance"):
            item["importance"] = round(r.importance, 4)
        items.append(item)

    return {"records": items, "total": len(records)}


@router.post("/records")
async def create_record(payload: CreateRecordPayload):
    """Create a new record."""
    manager = get_session_manager()
    session = manager.get_or_create_session()

    args = {"content": payload.content}
    if payload.tags:
        args["tags"] = payload.tags
    if payload.level:
        level_map = {
            "working": "L1_WORKING",
            "decisions": "L2_DECISIONS",
            "domain": "L3_DOMAIN",
            "identity": "L4_IDENTITY",
        }
        args["level"] = level_map.get(payload.level.lower(), "L3_DOMAIN")

    result = execute_tool("store", args, session.session_id)
    data = json.loads(result)
    return data


@router.get("/records/{record_id}")
async def get_record(record_id: str):
    """Get single record details."""
    rec = await brain_run(brain.get, record_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Record '{record_id}' not found")

    connections = []
    for conn_id, weight in rec.connections.items():
        conn_rec = await brain_run(brain.get, conn_id)
        if conn_rec:
            connections.append({
                "id": conn_id,
                "content": conn_rec.content[:150],
                "tags": list(conn_rec.tags) if conn_rec.tags else [],
                "weight": round(weight, 3),
            })

    result = {
        "id": rec.id,
        "content": rec.content,
        "tags": list(rec.tags) if rec.tags else [],
        "level": str(rec.level),
        "strength": round(rec.strength, 3),
        "activation_count": rec.activation_count,
        "metadata": rec.metadata or {},
        "connections": connections,
        "verified": (rec.metadata or {}).get("verified"),
        "source": (rec.metadata or {}).get("source"),
        "trust_score": (rec.metadata or {}).get("trust_score"),
    }
    if hasattr(rec, "importance"):
        result["importance"] = round(rec.importance, 4)
    return result


@router.put("/records/{record_id}")
async def update_record(record_id: str, payload: UpdateRecordPayload):
    """Update a record."""
    manager = get_session_manager()
    session = manager.get_or_create_session()

    args = {"record_id": record_id}
    if payload.content:
        args["content"] = payload.content
    if payload.tags:
        args["tags"] = payload.tags
    if payload.level:
        args["level"] = payload.level

    result = execute_tool("update_record", args, session.session_id)
    data = json.loads(result)

    if "error" in data:
        raise HTTPException(status_code=404, detail=data["error"])

    return data


@router.delete("/records/{record_id}")
async def delete_record(record_id: str):
    """Delete a record."""
    manager = get_session_manager()
    session = manager.get_or_create_session()

    result = execute_tool("delete_record", {"record_id": record_id}, session.session_id)
    data = json.loads(result)

    if "error" in data:
        raise HTTPException(status_code=404, detail=data["error"])

    return data


@router.post("/search")
async def search_records(query: SearchQuery):
    """Search records."""
    manager = get_session_manager()
    session = manager.get_or_create_session()

    args = {"query": query.query}
    if query.tags:
        args["tags"] = query.tags

    result = execute_tool("search", args, session.session_id)

    if "No results" in result:
        return {"results": []}

    try:
        items = json.loads(result)
        return {"results": items}
    except json.JSONDecodeError:
        return {"results": [], "raw": result}


@router.post("/consolidate")
async def consolidate_records():
    """Run native memory consolidation — merge similar records (MinHash ≥ 85%)."""
    """Run native memory consolidation — merge similar records (MinHash ≥ 85%)."""
    try:
        with brain_lock:
            result = brain.consolidate()
        return {
            "merged": result.get("merged", 0),
            "llm_merged": result.get("llm_merged", 0),
        }
    except AttributeError:
        raise HTTPException(status_code=501, detail="consolidate() not available — update aura to v1.4.1+")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Consolidation failed: {e}")


@router.post("/fix-memory-levels")
async def fix_memory_levels_endpoint():
    """Downgrade records incorrectly stuck at IDENTITY level back to DOMAIN."""
    """Downgrade records incorrectly stuck at IDENTITY level back to DOMAIN."""
    from remy.core.background_brain import fix_memory_levels
    # fix_memory_levels should handle locking internally if possible, but let's wrap it here to be safe
    # Actually fix_memory_levels takes brain as arg and likely iterates.
    with brain_lock:
        result = fix_memory_levels(brain)
    return result


_INTERNAL_GRAPH_TAGS = {
    "background-insights-latest",
    "session-summary",
    "web-search-cache",
    "autonomous-outcome",
    "outcome-failure",
}

_INTERNAL_GRAPH_TYPES = {
    "background_insights",
    "session_summary",
    "autonomous_outcome",
    "web_search_cache",
}


def _include_in_graph(record, mode: str) -> bool:
    if mode == "full":
        return True

    tags = {str(t or "").strip().lower() for t in (getattr(record, "tags", None) or [])}
    if tags & _INTERNAL_GRAPH_TAGS:
        return False

    metadata = getattr(record, "metadata", None) or {}
    record_type = str(metadata.get("type", "") or "").strip().lower()
    if record_type in _INTERNAL_GRAPH_TYPES:
        return False

    return True


@router.get("/graph")
async def get_graph_data(mode: str = "user", scope: str | None = None):
    """Get knowledge graph data for visualization."""
    graph_mode = (scope or mode or "user").strip().lower()
    if graph_mode not in {"user", "full"}:
        raise HTTPException(status_code=400, detail="Invalid graph mode")

    try:
        all_records = await brain_run(brain.list_records, min_strength=0.01, timeout=4.0)
        if not all_records:
            all_records = await brain_run(brain.list_records, min_strength=0.0, timeout=4.0)
    except Exception:
        all_records = []
    records = [r for r in all_records if _include_in_graph(r, graph_mode)]

    nodes = []
    edges = []
    seen_edges = set()

    valid_ids = {r.id for r in records}

    for r in records:
        node = {
            "id": r.id,
            "label": r.content[:60],
            "level": level_name(r.level) if hasattr(r, "level") else "",
            "tags": list(r.tags) if r.tags else [],
            "strength": round(r.strength, 3),
        }
        if hasattr(r, "importance"):
            node["importance"] = round(r.importance, 4)
        nodes.append(node)

        for conn_id, weight in r.connections.items():
            if conn_id not in valid_ids:
                continue
                
            edge_key = tuple(sorted([r.id, conn_id]))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append({
                    "source": r.id,
                    "target": conn_id,
                    "weight": round(weight, 3),
                })

    return {"nodes": nodes, "edges": edges}


# ============== SETTINGS ==============


@router.get("/settings")
async def get_settings():
    """Get current settings (API key masked)."""
    api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
    masked_key = ""
    if api_key:
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"

    bot_token = settings.TELEGRAM_BOT_TOKEN
    masked_bot = ""
    if bot_token:
        masked_bot = bot_token[:4] + "..." + bot_token[-4:] if len(bot_token) > 10 else "***"

    return {
        "gemini_api_key_masked": masked_key,
        "has_api_key": bool(api_key),
        "summary_model": settings.SUMMARY_MODEL,
        "gemini_voice": settings.GEMINI_VOICE,
        "has_telegram": bool(settings.TELEGRAM_BOT_TOKEN),
        "telegram_bot_masked": masked_bot,
        "proactive_chat_id": settings.PROACTIVE_CHAT_ID,
        "web_host": settings.WEB_HOST,
        "web_port": settings.WEB_PORT,
        "review_model": settings.REVIEW_MODEL,
        "review_enabled": settings.REVIEW_ENABLED,
    }


@router.put("/settings")
async def update_settings(payload: SettingsPayload):
    """Update runtime settings without rewriting .env."""
    from remy.config.settings import set_runtime_setting

    updated = []

    if payload.gemini_api_key is not None:
        set_runtime_setting("GEMINI_API_KEY", payload.gemini_api_key, target=settings)
        try:
            manager = get_session_manager()
            refresh = getattr(manager, "refresh_credentials", None)
            if refresh:
                refresh()
        except Exception as e:
            logger.warning("Failed to refresh web session credentials: %s", e)
        updated.append("GEMINI_API_KEY")

    if payload.summary_model is not None:
        set_runtime_setting("SUMMARY_MODEL", payload.summary_model, target=settings)
        updated.append("SUMMARY_MODEL")

    if payload.gemini_voice is not None:
        set_runtime_setting("GEMINI_VOICE", payload.gemini_voice, target=settings)
        updated.append("GEMINI_VOICE")

    if payload.telegram_bot_token is not None:
        set_runtime_setting("TELEGRAM_BOT_TOKEN", payload.telegram_bot_token, target=settings)
        updated.append("TELEGRAM_BOT_TOKEN")

    if payload.proactive_chat_id is not None:
        set_runtime_setting("PROACTIVE_CHAT_ID", payload.proactive_chat_id, target=settings)
        updated.append("PROACTIVE_CHAT_ID")

    if payload.review_model is not None:
        set_runtime_setting("REVIEW_MODEL", payload.review_model, target=settings)
        updated.append("REVIEW_MODEL")

    return {
        "updated": updated,
        "note": "Changes apply immediately and are saved to data/runtime_settings.json.",
    }


# ============== EXPORT ==============


@router.post("/import")
async def import_brain(payload: dict):
    """Import brain from JSON export."""
    from remy.core.agent_tools import Level
    
    # Check if 'records' key exists, otherwise assume root is list
    if "records" in payload:
        records = payload["records"]
    elif isinstance(payload, list):
        records = payload
    else:
        raise HTTPException(status_code=400, detail="Invalid format. Expected JSON with 'records' list.")

    old_to_new = {}
    success_count = 0
    errors = []

    # Level mapper
    level_map = {
        "WORKING": Level.WORKING,
        "DECISIONS": Level.DECISIONS,
        "DOMAIN": Level.DOMAIN,
        "IDENTITY": Level.IDENTITY,
        "L1_WORKING": Level.WORKING,
        "L2_DECISIONS": Level.DECISIONS,
        "L3_DOMAIN": Level.DOMAIN,
        "L4_IDENTITY": Level.IDENTITY,
    }

    # Pass 1: Create records (without connections)
    for r in records:
        try:
            old_id = r.get("id")
            if not old_id:
                continue

            lvl_str = str(r.get("level", "L3_DOMAIN")).replace("Level.", "")
            level = level_map.get(lvl_str, Level.DOMAIN)
            
            # Import trusts user intent — duplicates handled by brain's dedup on recall.
            
            with brain_lock:
                new_rec = brain.store(
                    content=r.get("content", ""),
                    level=level,
                    tags=r.get("tags", []),
                    metadata=r.get("metadata", None),
                )
            old_to_new[old_id] = new_rec.id
            success_count += 1
        except Exception as e:
            errors.append(f"Failed to store record {r.get('id')}: {e}")

    # Pass 2: Restore connections
    connections_restored = 0
    for r in records:
        old_id = r.get("id")
        new_id = old_to_new.get(old_id)
        if not new_id:
            continue
            
        conns = r.get("connections", {})
        # conns can be dict {target_id: weight} or list of objects
        
        target_items = []
        if isinstance(conns, dict):
            target_items = conns.items()
        elif isinstance(conns, list):
            # If export format changed to list of edges
            pass 

        for target_old, weight in target_items:
            target_new = old_to_new.get(target_old)
            if target_new:
                try:
                    with brain_lock:
                        brain.connect(new_id, target_new, weight=float(weight))
                    connections_restored += 1
                except Exception:
                    pass

    return {
        "imported": success_count,
        "connections_restored": connections_restored,
        "errors": errors[:10], # Limit error output
    }


@router.get("/export")
async def export_brain():
    """Export all brain records as JSON."""
    with brain_lock:
        records = brain.list_records(min_strength=0.0)

    items = []
    for r in records:
        item = {
            "id": r.id,
            "content": r.content,
            "tags": list(r.tags) if r.tags else [],
            "level": str(r.level),
            "strength": round(r.strength, 4),
            "activation_count": r.activation_count,
            "metadata": r.metadata or {},
            "connections": {cid: round(w, 3) for cid, w in r.connections.items()},
        }
        if hasattr(r, "importance"):
            item["importance"] = round(r.importance, 4)
        items.append(item)

    return JSONResponse(
        content={"exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "count": len(items), "records": items},
        headers={"Content-Disposition": "attachment; filename=remy-brain-export.json"},
    )


# ============== HISTORY ==============

@router.get("/history")
async def list_history():
    """List past session logs."""
    history_dir = settings.DATA_DIR / "history"
    if not history_dir.exists():
        return {"sessions": []}

    sessions = []
    for f in history_dir.glob("*.json"):
        try:
            stat = f.stat()
            # Parse filename for timestamp if possible, or use file creation time
            # Format: YYYY-mm-ddTHH-MM-SS_uuid.json
            
            # Read first few bytes to get session ID or summary?
            # actually let's just use filename and generic info to be fast
            sessions.append({
                "filename": f.name,
                "timestamp": stat.st_mtime,
                "size": stat.st_size,
                "date_str": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            })
        except Exception:
            pass

    # Sort by timestamp desc
    sessions.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"sessions": sessions}


@router.get("/history/{filename}")
async def get_history_session(filename: str):
    """Get specific session log."""
    # Sanitize filename to prevent path traversal
    safe_name = Path(filename).name
    if safe_name != filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    history_dir = settings.DATA_DIR / "history"
    filepath = history_dir / safe_name

    if not filepath.exists() or not filepath.is_file():
         raise HTTPException(status_code=404, detail="History not found")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load history: {e}")

# ============== SANDBOX TOOLS ==============

@router.get("/sandbox/tools")
async def list_sandbox_tools():
    """List all sandbox tools and their status."""
    from remy.sandbox.manifest import SandboxManifest
    manifest = SandboxManifest(settings.SANDBOX_DIR / "manifest.json")
    return {"tools": manifest.summary()}


@router.put("/sandbox/tools/{tool_name}/toggle")
async def toggle_sandbox_tool(tool_name: str):
    """Toggle a sandbox tool between approved and rejected."""
    from remy.sandbox.manifest import SandboxManifest
    from remy.core.brain_tools import reload_tools

    manifest = SandboxManifest(settings.SANDBOX_DIR / "manifest.json")

    tool = manifest.get_tool(tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    current = tool["status"]
    if current == "approved":
        tool["status"] = "rejected"
        manifest.save()
        reload_tools()
        return {"name": tool_name, "status": "rejected", "note": "Tool deactivated."}
    elif current == "rejected":
        tool["status"] = "approved"
        manifest.save()
        reload_tools()
        return {"name": tool_name, "status": "approved", "note": "Tool reactivated."}
    elif current in ("tested", "pending"):
        tool["status"] = "approved"
        manifest.save()
        reload_tools()
        return {"name": tool_name, "status": "approved", "note": "Tool approved and loaded."}
    else:
        return {"name": tool_name, "status": current, "note": f"Cannot toggle from '{current}' status."}


# ============== METRICS ==============


@router.get("/metrics")
async def get_metrics():
    """Prometheus text exposition format metrics."""
    from remy.core.metrics import collect_metrics
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(
        content=collect_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# ============== DIAGNOSTICS ==============


@router.post("/end-session")
async def end_session():
    """Close active session: save history, generate summary, end brain session.

    Called by navigator.sendBeacon() on page unload as a last-resort backup.
    The primary close path is the WebSocket disconnect handler.
    """
    try:
        manager = get_session_manager()
        await manager.close_session()
        return {"ok": True}
    except Exception as e:
        logger.warning(f"end-session failed: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/diagnostics")
async def get_diagnostics():
    """System diagnostics and health check."""
    from remy.core.brain_tools import get_registry

    api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")

    # Brain stats
    startup_status = get_brain_startup_status()
    try:
        with brain_lock:
            record_count = brain.count()
        brain_status = "ok" if not startup_status.get("quarantined_at_startup") else "recovered_after_quarantine"
    except Exception as e:
        record_count = 0
        brain_status = f"error: {e}"

    # Tool count (cheap — no full reload)
    try:
        registry = get_registry()
        tool_count = registry.tool_count
    except Exception:
        tool_count = 0

    uptime_sec = int(time.time() - _start_time)
    hours, remainder = divmod(uptime_sec, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Knowledge base (Aura Memory) stats
    kb_status = "unified into brain"
    kb_records = 0

    return {
        "status": "ok" if api_key and brain_status == "ok" else "degraded",
        "uptime": f"{hours}h {minutes}m {seconds}s",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "api_key_configured": bool(api_key),
        "model": settings.SUMMARY_MODEL,
        "brain": {
            "status": brain_status,
            "records": record_count,
            "path": str(settings.AURA_BRAIN_PATH),
            "startup": startup_status,
        },
        "knowledge": {
            "status": kb_status,
            "records": kb_records,
            "path": str(settings.AURA_MEMORY_PATH) if settings.AURA_MEMORY_ENABLED else "disabled",
        },
        "tools": tool_count,
        "telegram_configured": bool(settings.TELEGRAM_BOT_TOKEN),
        "sandbox_dir": str(settings.SANDBOX_DIR),
    }


# ============== AUDIT TRAIL ==============


@router.get("/audit/logs")
async def get_audit_logs(n: int = 20, tool: str | None = None):
    """Recent audit log entries for critical tool executions."""
    from remy.core.audit_trail import get_audit_logger
    return {"logs": get_audit_logger().get_recent_logs(n=n, tool_name=tool)}


@router.get("/audit/integrity")
async def get_audit_integrity():
    """Check audit log integrity (SHA-256 checksums)."""
    from remy.core.audit_trail import get_audit_logger
    return get_audit_logger().verify_integrity()


@router.get("/audit/summary")
async def get_audit_summary():
    """Aggregate audit stats by tool and status."""
    from remy.core.audit_trail import get_audit_logger
    return get_audit_logger().get_summary()


# ============== EVALUATION METRICS ==============


@router.get("/eval-metrics")
async def get_eval_metrics(channel: str | None = None, limit: int = 50):
    """Aggregated evaluation metrics for agent responses."""
    from remy.core.eval_metrics import get_metrics_summary
    return get_metrics_summary(channel=channel, limit=limit)


# ============== ACTIVITY LOG ==============
@router.get("/activity")
async def get_activity():
    """Aggregate autonomous agent activity data for the Activity Log view."""
    from remy.core.combined_runner import get_activity_feed_snapshot

    return await asyncio.to_thread(
        get_activity_feed_snapshot,
        brain,
        brain_lock,
        goal_limit=50,
        outcome_limit=100,
        reflection_limit=10,
        proactive_limit=20,
    )


# ============== WEBSOCKET ==============


def _classify_error(error_text: str) -> dict:
    """Classify error for user-friendly message + recovery estimation."""
    e = error_text.lower()
    if "quota" in e or "429" in e or "resource_exhausted" in e:
        return {
            "message": "API rate limit reached. Please wait a moment and try again.",
            "retryable": True,
            "error_class": "rate_limit",
        }
    if "402" in e or "insufficient" in e or "credits" in e or "can only afford" in e:
        return {
            "message": "Insufficient credits on OpenRouter. Top up at openrouter.ai/settings/credits or switch to a free model (add :free suffix).",
            "retryable": False,
            "error_class": "billing",
        }
    if "api key" in e or "401" in e or "403" in e or "permission" in e:
        return {
            "message": "API authentication error. Check your API key in Settings.",
            "retryable": False,
            "error_class": "auth",
        }
    if "timeout" in e or "deadline" in e:
        return {
            "message": "Request timed out. Try again or simplify your message.",
            "retryable": True,
            "error_class": "timeout",
        }
    if "connect" in e or "network" in e or "unreachable" in e or "getaddrinfo" in e:
        return {
            "message": "Network error. Check your internet connection.",
            "retryable": True,
            "error_class": "network",
        }
    if "subscriptable" in e:
        return {
            "message": "API response parsing error. Retrying automatically...",
            "retryable": True,
            "error_class": "transient",
        }
    return {
        "message": "Something went wrong. Try again.",
        "retryable": True,
        "error_class": "unknown",
    }


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """Real-time chat via WebSocket."""
    # Auth disabled — accept all connections
    await websocket.accept()
    metrics_collector.ws_connected("chat")
    manager = get_session_manager()

    logger.info("WebSocket chat connected")

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "message":
                user_text = data.get("text", "").strip()
                if not user_text:
                    continue

                # Send typing indicator
                await websocket.send_json({"type": "typing"})

                if data.get("context_reducer_compare"):
                    try:
                        from remy.core.context_reducer import (
                            compare_context_reducer,
                            make_gemini_llm_func,
                        )

                        session = manager.get_or_create_session()
                        # Lab window may pick the model to price against. When a
                        # Gemini model is selected we call it directly so token
                        # counts and cost reflect THAT model; otherwise the
                        # default chain is used and cost stays $0 (unknown price).
                        lab_model = str(data.get("model") or "").strip() or None
                        lab_llm_func = None
                        if lab_model and lab_model.lower().startswith("gemini"):
                            lab_llm_func = make_gemini_llm_func(lab_model)
                        report = await compare_context_reducer(
                            user_text=user_text,
                            session_log=session.session_log,
                            history=session.history,
                            session_id=session.session_id,
                            model=lab_model,
                            llm_func=lab_llm_func,
                        )
                        await websocket.send_json({"type": "context_reducer_compare", "report": report})
                    except Exception as e:
                        logger.error(f"ContextReducer compare error: {e}")
                        err = _classify_error(str(e))
                        await websocket.send_json({
                            "type": "error",
                            "content": err["message"],
                            "retryable": err["retryable"],
                            "error_class": err["error_class"],
                        })
                    await websocket.send_json({"type": "done"})
                    continue

                if data.get("context_reducer_apply"):
                    try:
                        from remy.core.context_reducer import apply_context_reducer

                        session = manager.get_or_create_session()
                        session.session_log.append({"type": "user_text", "text": user_text[:200]})
                        result = await apply_context_reducer(
                            user_text=user_text,
                            session_log=session.session_log,
                            history=session.history,
                            session_id=session.session_id,
                        )
                        answer = str(result.get("answer") or "")
                        report = result.get("report") or {}
                        session.session_log.append({
                            "type": "model_response",
                            "text": answer[:200],
                            "full_text": answer,
                            "source": "context_reducer_apply",
                        })
                        await websocket.send_json({"type": "token", "content": answer})
                        await websocket.send_json({"type": "llm_optimization_apply", "report": report})
                    except Exception as e:
                        logger.error(f"ContextReducer apply error: {e}")
                        err = _classify_error(str(e))
                        await websocket.send_json({
                            "type": "error",
                            "content": err["message"],
                            "retryable": err["retryable"],
                            "error_class": err["error_class"],
                        })
                    await websocket.send_json({"type": "done"})
                    continue

                try:
                    streamed_any = False
                    async for event in manager.gemini_respond_stream(user_text):
                        if event["type"] == "token":
                            await websocket.send_json({
                                "type": "token",
                                "content": event["content"]
                            })
                            streamed_any = True
                        elif event["type"] == "tool_start":
                            await websocket.send_json({
                                "type": "tool_start",
                                "content": event["tool"]
                            })
                        elif event["type"] == "tool_end":
                            await websocket.send_json({
                                "type": "tool_end",
                                "content": event["tool"]
                            })
                        elif event["type"] == "final":
                            # Only send full text if streaming didn't yield tokens
                            if not streamed_any and event.get("text"):
                                await websocket.send_json({
                                    "type": "text",
                                    "content": event["text"]
                                })
                            
                except Exception as e:
                    logger.error(f"Gemini respond error: {e}")
                    err = _classify_error(str(e))
                    await websocket.send_json({
                        "type": "error",
                        "content": err["message"],
                        "retryable": err["retryable"],
                        "error_class": err["error_class"],
                    })

                await websocket.send_json({"type": "done"})

            elif data.get("type") == "voice":
                audio_b64 = data.get("audio", "")
                mime_type = data.get("mime_type", "audio/webm")

                if not audio_b64:
                    await websocket.send_json({"type": "error", "content": "No audio data received."})
                    continue

                try:
                    audio_bytes = base64.b64decode(audio_b64)
                except Exception:
                    await websocket.send_json({"type": "error", "content": "Invalid audio encoding."})
                    continue

                await websocket.send_json({"type": "typing"})

                try:
                    result = await manager.gemini_respond_multimodal(
                        attachments=[{"mime_type": mime_type, "data": audio_bytes}],
                        is_voice=True,
                    )
                    await websocket.send_json({
                        "type": "text",
                        "content": result["response"],
                        "speak": True,
                    })
                except Exception as e:
                    logger.error(f"Voice respond error: {e}")
                    err = _classify_error(str(e))
                    await websocket.send_json({
                        "type": "error",
                        "content": err["message"],
                        "retryable": err["retryable"],
                        "error_class": err["error_class"],
                    })

                await websocket.send_json({"type": "done"})

            elif data.get("type") == "file":
                file_b64 = data.get("data", "")
                mime_type = data.get("mime_type", "application/octet-stream")
                file_name = data.get("name", "unknown")
                accompanying_text = data.get("text", "")

                if not file_b64:
                    await websocket.send_json({"type": "error", "content": "No file data received."})
                    continue

                try:
                    file_bytes = base64.b64decode(file_b64)
                except Exception:
                    await websocket.send_json({"type": "error", "content": "Invalid file encoding."})
                    continue

                await websocket.send_json({"type": "typing"})

                prompt = accompanying_text or f"The user uploaded a file named '{file_name}'. Analyze it and respond."

                try:
                    result = await manager.gemini_respond_multimodal(
                        text=prompt,
                        attachments=[{"mime_type": mime_type, "data": file_bytes}],
                    )
                    await websocket.send_json({"type": "text", "content": result["response"]})
                except Exception as e:
                    logger.error(f"File respond error: {e}")
                    err = _classify_error(str(e))
                    await websocket.send_json({
                        "type": "error",
                        "content": err["message"],
                        "retryable": err["retryable"],
                        "error_class": err["error_class"],
                    })

                await websocket.send_json({"type": "done"})

            elif data.get("type") == "new_session":
                await manager.close_session()
                manager.get_or_create_session()
                await websocket.send_json({"type": "session_reset"})

    except WebSocketDisconnect:
        logger.info("WebSocket chat disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Graceful session close: save history JSON + generate summary + brain.end_session
        # Timeout prevents blocking shutdown (Ctrl+C) if summary generation is slow.
        try:
            await asyncio.wait_for(manager.close_session(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Session close timed out (10s) — skipping summary")
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("Session close interrupted by shutdown")
        except Exception as e:
            logger.warning(f"Session close on disconnect failed: {e}")
        metrics_collector.ws_disconnected("chat")


# ============== HUMAN-IN-THE-LOOP APPROVAL API ==============

@router.get("/approvals")
async def get_pending_approvals():
    """List all currently pending approval actions."""
    from remy.core.combined_runner import get_approval_runtime_snapshot

    approvals = get_approval_runtime_snapshot(goal_limit=3, approval_limit=100).get("pending", [])
    return {"pending": approvals}


@router.post("/approvals/{action_id}/approve")
async def approve_action(action_id: str):
    """Approve a pending action by ID (full UUID or first-8 prefix)."""
    from remy.core.combined_runner import resolve_operator_approval
    return resolve_operator_approval(action_id, approved=True, decided_by="web")


@router.post("/approvals/{action_id}/reject")
async def reject_action(action_id: str):
    """Reject a pending action by ID (full UUID or first-8 prefix)."""
    from remy.core.combined_runner import resolve_operator_approval
    return resolve_operator_approval(action_id, approved=False, decided_by="web")


@router.websocket("/ws/approvals")
async def websocket_approvals(websocket: WebSocket):
    """Push approval.pending / approval.resolved events to the Web GUI in real-time."""
    from remy.core.combined_runner import get_runtime_transport_snapshot
    await websocket.accept()
    queue = event_bus.subscribe()
    logger.info("Approvals WebSocket connected (%d subscribers)", get_runtime_transport_snapshot().get("subscribers", 0))

    # Send snapshot of any already-pending actions so the UI catches up on reconnect
    try:
        from remy.core.combined_runner import get_approval_runtime_snapshot

        approvals = get_approval_runtime_snapshot(goal_limit=3, approval_limit=50).get("pending", [])
        for action in approvals:
            if action.get("action_id"):
                await websocket.send_json({
                    "type": "approval.pending",
                    "action_id": action.get("action_id"),
                    "description": action.get("description"),
                    "timeout_sec": action.get("timeout_sec"),
                    "created_at": action.get("created_at"),
                })
    except Exception as e:
        logger.debug("Could not send approval snapshot: %s", e)

    try:
        async def _forward_events():
            while True:
                event = await queue.get()
                if event.get("type") in ("approval.pending", "approval.resolved"):
                    await websocket.send_json(event)

        async def _listen_client():
            while True:
                try:
                    await websocket.receive_text()
                except WebSocketDisconnect:
                    return

        done, pending_tasks = await asyncio.wait(
            [
                asyncio.create_task(_forward_events()),
                asyncio.create_task(_listen_client()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending_tasks:
            task.cancel()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("Approvals WebSocket error: %s", e)
    finally:
        event_bus.unsubscribe(queue)
        logger.info("Approvals WebSocket disconnected (%d subscribers)", get_runtime_transport_snapshot().get("subscribers", 0))


@router.websocket("/ws/activity")
async def websocket_activity(websocket: WebSocket):
    """Real-time autonomous thought stream via WebSocket."""
    # Auth disabled — accept all connections
    await websocket.accept()
    metrics_collector.ws_connected("activity")
    queue = event_bus.subscribe()
    from remy.core.combined_runner import get_runtime_transport_snapshot
    logger.info("Activity WebSocket connected (%d subscribers)", get_runtime_transport_snapshot().get("subscribers", 0))

    # Send initial budget snapshot so UI doesn't show "--" until first cycle
    try:
        from remy.core.combined_runner import get_budget_runtime_snapshot
        from remy.core.runtime_event_contract import build_runtime_event
        budget = get_budget_runtime_snapshot(goal_limit=5, approval_limit=10)
        await websocket.send_json(build_runtime_event(
            "budget_init",
            event_domain="budget",
            payload={"budget": budget},
            legacy_fields={"budget": budget},
            timestamp=time.time(),
        ))
    except Exception as e:
        logger.debug("Could not send budget_init: %s", e)

    try:
        async def forward_events():
            while True:
                event = await queue.get()
                await websocket.send_json(event)

        async def listen_client():
            while True:
                try:
                    await websocket.receive_text()
                except WebSocketDisconnect:
                    return

        done, pending = await asyncio.wait(
            [
                asyncio.create_task(forward_events()),
                asyncio.create_task(listen_client()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("Activity WebSocket error: %s", e)
    finally:
        metrics_collector.ws_disconnected("activity")
        event_bus.unsubscribe(queue)
        logger.info("Activity WebSocket disconnected (%d subscribers)", get_runtime_transport_snapshot().get("subscribers", 0))
# ... existing code ...

# ============== KNOWLEDGE DASHBOARD API (RM-8) ==============

@router.get("/knowledge/research")
async def get_research_projects():
    """List all research projects (active and completed)."""
    # Active
    active = get_active_research_projects()
    
    # Completed (last 50)
    with brain_lock:
        completed_recs = brain.search(query="", tags=["research-project", "completed"], limit=50)
    completed = [serialize_completed_research_project(r) for r in completed_recs]
        
    return {"active": active, "completed": completed}

@router.get("/knowledge/metrics")
async def get_metric_data(limit: int = 50):
    """Get recent tracked metrics."""
    with brain_lock:
        recs = brain.search(query="", tags=["metric"], limit=limit)
        legacy_recs = brain.search(query="", tags=["health-metric"], limit=limit)
    recs = list(recs or []) + list(legacy_recs or [])
    data = []
    seen: set[str] = set()
    for r in recs:
        meta = r.metadata or {}
        date_key = str(meta.get("timestamp") or "")[:10]
        dedupe_key = f"{meta.get('metric')}|{meta.get('value')}|{date_key}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        data.append({
            "metric": meta.get("metric"),
            "value": meta.get("value"),
            "unit": meta.get("unit"),
            "timestamp": meta.get("timestamp"),
            "notes": meta.get("notes"),
        })
    return {"data": data}

@router.get("/knowledge/facts")
async def get_extracted_facts(limit: int = 50):
    """Get recent extracted facts."""
    with brain_lock:
        recs = brain.search(query="", tags=["extracted-fact"], limit=limit)
    data = []
    for r in recs:
        meta = r.metadata or {}
        data.append({
            "content": r.content,
            "structure": meta.get("structure"),
            "source": meta.get("source"),
            "extracted_at": meta.get("extracted_at"),
        })
    return {"data": data}


# ============== IDENTITY (Profile + People) ==============

_IDENTITY_PROFILE_FIELDS = ("name", "age", "location", "occupation", "languages",
                            "family", "personal_focus", "interests", "notes", "email", "phone")


@router.get("/knowledge/identity")
async def get_identity():
    """Get user profile and people records for the Identity tab."""
    from remy.core.brain_tools import get_user_profile_record
    profile_rec = get_user_profile_record(brain, brain_lock)
    with brain_lock:
        people = brain.search(query="", tags=["person"], limit=50)

    profile_data = {}
    if profile_rec:
        meta = profile_rec.metadata or {}
        for key in _IDENTITY_PROFILE_FIELDS:
            if key == "personal_focus":
                profile_data[key] = meta.get("personal_focus") or meta.get("health_focus", "")
            else:
                profile_data[key] = meta.get(key, "")
        profile_data["id"] = profiles[0].id
        profile_data["verified"] = meta.get("verified", False)

    people_list = []
    for p in people:
        meta = p.metadata or {}
        people_list.append({
            "id": p.id,
            "full_name": meta.get("full_name", p.content[:100]),
            "role": meta.get("role", ""),
            "birth_date": meta.get("birth_date", ""),
            "birth_place": meta.get("birth_place", ""),
            "verified": meta.get("verified", False),
            "trust_score": meta.get("trust_score", 0.5),
        })

    return {"profile": profile_data, "people": people_list}


@router.put("/knowledge/identity/profile")
async def update_identity_profile(body: dict):
    """Update user profile from the Identity tab. Marks all fields as user-confirmed."""
    result = execute_tool("store_user_profile", body)
    return json.loads(result)


@router.put("/knowledge/identity/person/{record_id}")
async def update_identity_person(record_id: str, body: dict):
    """Update a person record. Marks as verified (user-confirmed, trust=1.0)."""
    with brain_lock:
        rec = brain.get(record_id)
        if not rec:
            raise HTTPException(status_code=404, detail="Person not found")
        if "person" not in (rec.tags or []):
            raise HTTPException(status_code=400, detail="Record is not a person")
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
        brain.update(record_id, content=", ".join(parts), metadata=meta)
    return {"updated": True, "id": record_id}


@router.get("/todos")
async def list_todos(status: str = "active", category: str | None = None, limit: int = 50):
    """List todo items. status: active (pending+in_progress), done, all."""
    with brain_lock:
        recs = brain.search(query="", tags=["todo-item"], limit=200)
    items = []
    for r in recs:
        meta = r.metadata or {}
        if meta.get("type") != "todo_item":
            continue
        s = meta.get("status", "pending")
        # Filter by status
        if status == "active" and s not in ("pending", "in_progress"):
            continue
        if status == "done" and s != "done":
            continue
        # Filter by category
        if category and meta.get("category") != category:
            continue
        items.append({
            "id": r.id,
            "todo_id": meta.get("todo_id"),
            "title": meta.get("title", r.content),
            "priority": meta.get("priority", "medium"),
            "status": s,
            "category": meta.get("category", "personal"),
            "due_date": meta.get("due_date"),
            "created_by": meta.get("created_by", "user"),
            "created_at": meta.get("created_at"),
            "completed_at": meta.get("completed_at"),
            "parent_todo_id": meta.get("parent_todo_id"),
        })
    # Sort: in_progress first, then pending, then done; within group by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    status_order = {"in_progress": 0, "pending": 1, "done": 2, "archived": 3}
    items.sort(key=lambda x: (status_order.get(x["status"], 9), priority_order.get(x["priority"], 9)))
    return {"todos": items[:limit], "total": len(items)}


@router.post("/todos")
async def create_todo(body: dict):
    """Create a new todo item from the web UI."""
    from remy.core.agent_tools import Level
    import uuid as _uuid

    title = (body.get("title") or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title is required")

    priority = (body.get("priority") or "medium").lower()
    if priority not in ("high", "medium", "low"):
        priority = "medium"
    due_date = (body.get("due_date") or "").strip() or None
    category = (body.get("category") or "personal").lower().strip() or "personal"

    todo_id = f"todo-{_uuid.uuid4().hex[:12]}"
    tags = ["todo-item", f"cat-{category}", f"priority-{priority}"]

    content = f"Todo [{priority.upper()}]: {title}"
    if due_date:
        content += f" | Due: {due_date}"

    meta = {
        "type": "todo_item",
        "todo_id": todo_id,
        "title": title,
        "priority": priority,
        "status": "pending",
        "category": category,
        "due_date": due_date,
        "created_by": "user",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
    }

    with brain_lock:
        rec = brain.store(content=content, level=Level.DOMAIN, tags=tags, metadata=meta)
    return {"id": rec.id, "todo_id": todo_id, "title": title, "priority": priority, "category": category}


@router.post("/todos/{record_id}/toggle")
async def toggle_todo(record_id: str):
    """Toggle a todo between done and pending."""
    with brain_lock:
        rec = brain.get(record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Todo not found")
    meta = rec.metadata or {}
    if meta.get("type") != "todo_item":
        raise HTTPException(status_code=400, detail="Record is not a todo item")
    current = meta.get("status", "pending")
    if current == "done":
        new_status = "pending"
        meta["completed_at"] = None
    else:
        new_status = "done"
        meta["completed_at"] = datetime.now().isoformat()
    meta["status"] = new_status
    with brain_lock:
        brain.update(record_id, metadata=meta)
    return {"id": record_id, "status": new_status}


@router.delete("/todos/{record_id}")
async def delete_todo(record_id: str):
    """Delete a todo item."""
    with brain_lock:
        rec = brain.get(record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Todo not found")
    meta = rec.metadata or {}
    if meta.get("type") != "todo_item":
        raise HTTPException(status_code=400, detail="Record is not a todo item")
    with brain_lock:
        brain.delete(record_id)
    return {"deleted": True, "id": record_id}


@router.get("/knowledge/stats")
async def get_brain_stats():
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
            "last_updated": stats["last_updated"]
        }
    }


# ============== KNOWLEDGE BASE (Aura Memory) ==============

def _get_knowledge():
    """Knowledge base was unified into brain (Aura SDK v1.0.5+)."""
    raise HTTPException(status_code=503, detail="Knowledge base (Aura Memory) is not available — unified into brain")


@router.get("/knowledge/base")
async def list_knowledge_base(limit: int = 100, offset: int = 0, query: str = ""):
    """List records in the semantic knowledge base."""
    kb, lock = _get_knowledge()
    if query.strip():
        from remy.core.brain_tools import _kb_retrieve
        items = _kb_retrieve(query.strip(), top_k=limit)
        return {"items": items, "total": len(items), "query": query}
    else:
        with lock:
            records, total = kb.list_memories(offset, limit, None)
        items = records if isinstance(records, list) else list(records)
        return {"items": items, "total": total, "offset": offset, "limit": limit}


@router.post("/knowledge/ingest")
async def ingest_knowledge(body: dict):
    """Ingest text into the knowledge base."""
    kb, lock = _get_knowledge()
    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    pin = bool(body.get("pin", False))

    if len(text) > 1000:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text]
        with lock:
            if pin:
                count = kb.ingest_batch_pinned(paragraphs)
            else:
                count = kb.ingest_batch(paragraphs)
            kb.flush()
            total = kb.count()
        return {"ingested": count, "pinned": pin, "total": total}
    else:
        with lock:
            status = kb.process(text, pin=pin)
            kb.flush()
            total = kb.count()
        return {"ingested": 1, "status": status, "pinned": pin, "total": total}


_ALLOWED_EXTENSIONS = {".txt", ".md", ".csv"}
_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/knowledge/upload")
async def upload_knowledge_file(file: UploadFile = File(...), pin: bool = Form(False)):
    """Upload a file to the knowledge base (.txt, .md, .csv)."""
    kb, lock = _get_knowledge()

    # Validate extension
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Allowed: {', '.join(_ALLOWED_EXTENSIONS)}")

    # Read content
    content_bytes = await file.read()
    if len(content_bytes) > _MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large: {len(content_bytes)} bytes (max {_MAX_FILE_SIZE})")

    try:
        text = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text")

    # Parse based on extension
    if ext == ".csv":
        import csv
        import io
        reader = csv.reader(io.StringIO(text))
        rows = [" | ".join(cell.strip() for cell in row if cell.strip()) for row in reader]
        chunks = [r for r in rows if r]
    else:
        # .txt / .md — split by double newlines
        chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not chunks:
            chunks = [text.strip()] if text.strip() else []

    if not chunks:
        raise HTTPException(status_code=400, detail="File is empty or has no parseable content")

    # Ingest
    with lock:
        if pin:
            count = kb.ingest_batch_pinned(chunks)
        else:
            count = kb.ingest_batch(chunks)
        kb.flush()
        total = kb.count()

    return {
        "filename": file.filename,
        "extension": ext,
        "chunks": count,
        "pinned": pin,
        "total": total,
    }


@router.delete("/knowledge/base/{record_id}")
async def delete_knowledge_item(record_id: str):
    """Delete a record from the knowledge base."""
    kb, lock = _get_knowledge()
    with lock:
        success = kb.delete_memory(record_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Record '{record_id}' not found")
    return {"deleted": True, "id": record_id}


# ============== GENERATED IMAGES ==============

@router.get("/generated_images/{filename}")
async def serve_generated_image(filename: str):
    """Serve a generated image file."""
    from fastapi.responses import FileResponse
    image_dir = Path(settings.DATA_DIR) / "generated_images"
    filepath = (image_dir / filename).resolve()
    if not filepath.exists() or not filepath.is_relative_to(image_dir.resolve()):
        raise HTTPException(status_code=404, detail="Image not found")
    media_type = "image/png"
    if filepath.suffix.lower() in (".jpg", ".jpeg"):
        media_type = "image/jpeg"
    elif filepath.suffix.lower() == ".webp":
        media_type = "image/webp"
    return FileResponse(filepath, media_type=media_type)


# ============== BROWSER SCREENSHOTS ==============

@router.get("/browser_screenshots/{filename}")
async def serve_browser_screenshot(filename: str):
    """Serve a browser screenshot file."""
    from fastapi.responses import FileResponse
    image_dir = Path(settings.DATA_DIR) / "browser_screenshots"
    filepath = (image_dir / filename).resolve()
    if not filepath.exists() or not filepath.is_relative_to(image_dir.resolve()):
        raise HTTPException(status_code=404, detail="Screenshot not found")
    return FileResponse(filepath, media_type="image/png")


# ============== GENERATED REPORTS ==============

@router.get("/reports/{filename}")
async def serve_report(filename: str):
    """Serve a generated PDF report."""
    from fastapi.responses import FileResponse
    reports_dir = Path(settings.DATA_DIR) / "reports"
    filepath = (reports_dir / filename).resolve()
    if not filepath.exists() or not filepath.is_relative_to(reports_dir.resolve()):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(filepath, media_type="application/pdf")


# ============== GENERATED PRESENTATIONS ==============

@router.get("/presentations/{filename}")
async def serve_presentation(filename: str):
    """Serve a generated PPTX presentation."""
    from fastapi.responses import FileResponse
    pres_dir = Path(settings.DATA_DIR) / "presentations"
    filepath = (pres_dir / filename).resolve()
    if not filepath.exists() or not filepath.is_relative_to(pres_dir.resolve()):
        raise HTTPException(status_code=404, detail="Presentation not found")
    return FileResponse(
        filepath,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ============== AURA MEMORY PACKAGE MANAGEMENT ==============

@router.get("/aura/status")
async def aura_status():
    """Check installed and latest available version of aura-memory."""
    import subprocess
    import sys
    import importlib.metadata
    import re

    installed_version = None
    try:
        installed_version = importlib.metadata.version("aura-memory")
    except Exception:
        pass

    latest_version = None
    latest_error = None
    try:
        # pip index versions: works without network only if package cached
        result = subprocess.run(
            [sys.executable, "-m", "pip", "index", "versions", "aura-memory"],
            capture_output=True, text=True, timeout=15
        )
        m = re.search(r"aura-memory \(([^)]+)\)", result.stdout)
        if m:
            latest_version = m.group(1).split(",")[0].strip()
    except Exception as e:
        latest_error = str(e)

    return {
        "installed": installed_version,
        "latest": latest_version,
        "latest_error": latest_error,
        "up_to_date": installed_version == latest_version if (installed_version and latest_version) else None,
        "pypi_url": "https://pypi.org/project/aura-memory/",
    }


@router.post("/aura/update")
async def aura_update():
    """
    Install/upgrade aura-memory from PyPI, then schedule a process restart.

    The aura_memory Rust extension (.pyd) cannot be hot-swapped while loaded —
    the process must restart for the new version to take effect.
    We return the pip output first, then restart in a background task so the
    HTTP response reaches the browser before the server goes down.
    """
    import subprocess
    import sys
    import asyncio
    import os

    # Run pip install in a thread to avoid blocking the event loop
    loop = asyncio.get_event_loop()

    def _run_pip():
        return subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "aura-memory",
             "--no-warn-script-location"],
            capture_output=True, text=True, timeout=180
        )

    try:
        result = await loop.run_in_executor(None, _run_pip)
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "Timeout — pip не відповідає.", "stdout": "", "stderr": "", "restarting": False}
    except Exception as e:
        return {"success": False, "message": str(e), "stdout": "", "stderr": "", "restarting": False}

    success = result.returncode == 0

    if success:
        # Schedule restart after 1.5 s so this response completes first
        async def _restart():
            await asyncio.sleep(1.5)
            import subprocess as sp

            if getattr(sys, "frozen", False):
                # Frozen exe: spawn a new instance of the exe, then exit this one.
                # os.execv cannot replace a PyInstaller exe with a fresh Python process.
                sp.Popen([sys.executable] + sys.argv[1:])
                os._exit(0)
            else:
                # Development: os.execv replaces the process — new .pyd loads cleanly.
                try:
                    os.execv(sys.executable, [sys.executable] + sys.argv)
                except Exception:
                    sp.Popen([sys.executable] + sys.argv)
                    os._exit(0)

        asyncio.create_task(_restart())

    return {
        "success": success,
        "restarting": success,
        "stdout": result.stdout[-3000:] if result.stdout else "",
        "stderr": result.stderr[-1000:] if result.stderr else "",
        "message": "Оновлено! Remy перезапускається…" if success else "Помилка встановлення.",
    }
