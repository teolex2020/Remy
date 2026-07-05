"""
Memory routes — brain CRUD, search, graph, consolidation, import/export.
"""

import json
import logging
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from remy.core.agent_tools import level_name, tier_of
from remy.web.routes._helpers import _get_api, run_in_thread, run_lambda_in_thread, _TIMEOUT_FAST, _TIMEOUT_SLOW

logger = logging.getLogger("WebAPI")

router = APIRouter()

# ============== LIST RECORDS CACHE ==============
# Avoids hitting brain.list_records(5000) on every page refresh.
# Invalidated on store/update/delete. TTL is a safety net.
_RECORDS_CACHE_TTL = 10.0
_records_cache: dict = {"ts": 0.0, "data": None}


def _invalidate_records_cache() -> None:
    _records_cache["ts"] = 0.0
    _records_cache["data"] = None


async def _get_cached_all_records(api) -> list:
    """Return cached list_records(5000) or fetch fresh if stale."""
    now = time.time()
    if _records_cache["data"] is not None and (now - _records_cache["ts"]) < _RECORDS_CACHE_TTL:
        return _records_cache["data"]

    def _fetch():
        with api.brain_lock:
            return api.brain.list_records(min_strength=0.01, limit=5000)

    records = await run_in_thread(_fetch, error_msg="Memory records fetch timed out")
    _records_cache["ts"] = time.time()
    _records_cache["data"] = records
    return records


# ============== PYDANTIC MODELS ==============


class SearchQuery(BaseModel):
    query: str
    tags: str | None = None
    tier: str | None = "all"
    period: str | None = "all"
    mode: str | None = "hybrid"


class UpdateRecordPayload(BaseModel):
    content: str | None = None
    tags: str | None = None
    level: str | None = None


class CreateRecordPayload(BaseModel):
    content: str
    tags: str | None = None
    level: str | None = None


class RecordFeedbackPayload(BaseModel):
    useful: bool
    reason: str | None = None


def _normalize_tier_stats(stats: dict | None) -> dict:
    """Return a stable tier-stats payload for the web UI."""
    stats = stats or {}

    # Aura may return flat keys like "core_identity", "cognitive_working",
    # or nested dicts like {"levels": {...}}, {"core": {...}, "cognitive": {...}}.
    # Handle all variants.
    levels = stats.get("levels")
    if isinstance(levels, dict):
        identity = int(levels.get("identity", levels.get("IDENTITY", 0)) or 0)
        domain = int(levels.get("domain", levels.get("DOMAIN", 0)) or 0)
        decisions = int(levels.get("decisions", levels.get("DECISIONS", 0)) or 0)
        working = int(levels.get("working", levels.get("WORKING", 0)) or 0)
    else:
        # Try flat keys first (core_identity, cognitive_working, etc.)
        identity = int(stats.get("core_identity", 0) or 0)
        domain = int(stats.get("core_domain", 0) or 0)
        decisions = int(stats.get("cognitive_decisions", 0) or 0)
        working = int(stats.get("cognitive_working", 0) or 0)

        # Fall back to nested dicts if flat keys are all zero
        if not any([identity, domain, decisions, working]):
            cognitive = stats.get("cognitive") or {}
            core = stats.get("core") or {}
            identity = int(core.get("identity", stats.get("identity", stats.get("IDENTITY", 0))) or 0)
            domain = int(core.get("domain", stats.get("domain", stats.get("DOMAIN", 0))) or 0)
            decisions = int(cognitive.get("decisions", stats.get("decisions", stats.get("DECISIONS", 0))) or 0)
            working = int(cognitive.get("working", stats.get("working", stats.get("WORKING", 0))) or 0)

    normalized = {
        "levels": {
            "identity": identity,
            "domain": domain,
            "decisions": decisions,
            "working": working,
            "IDENTITY": identity,
            "DOMAIN": domain,
            "DECISIONS": decisions,
            "WORKING": working,
        },
        "cognitive": {
            "total": working + decisions,
            "working": working,
            "decisions": decisions,
        },
        "core": {
            "total": domain + identity,
            "domain": domain,
            "identity": identity,
        },
        "total": identity + domain + decisions + working,
    }

    if "strength_distribution" in stats:
        normalized["strength_distribution"] = stats["strength_distribution"]
    return normalized


# ============== ENDPOINTS ==============


@router.get("/stats")
async def get_brain_stats():
    """Get brain statistics."""
    api = _get_api()

    def _query():
        with api.brain_lock:
            return api.brain.count(), api.brain.stats()

    count, stats = await run_in_thread(_query, timeout=_TIMEOUT_FAST, error_msg="Brain stats timed out")
    usage = api.usage_tracker.get_stats()
    usage["total_tokens"] = usage.get("user_tokens", 0) + usage.get("autonomy_tokens", 0)

    return {"total_records": count, "stats": stats, "usage": usage}


@router.get("/records")
async def list_records(tags: str | None = None, tier: str = "all", period: str = "all", offset: int = 0, limit: int = 50):
    """List memory records."""
    import re
    from datetime import datetime
    api = _get_api()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

    # Use cache for the full list, then filter in Python (no brain_lock needed)
    all_records = await _get_cached_all_records(api)

    records = all_records
    if tag_list:
        tag_set = set(tag_list)
        records = [r for r in records if tag_set & set(r.tags or [])]

    if tier and tier != "all":
        records = [r for r in records if tier_of(r.level) == tier]

    if period and period != "all":
        now = time.time()
        try:
            days = int(period)
            cutoff = now - (days * 86400)
            filtered = []
            for r in records:
                ts = (getattr(r, "metadata", {}) or {}).get("timestamp")
                if ts:
                    if float(ts) > 1e11:
                        ts = float(ts) / 1000.0
                    if float(ts) >= cutoff:
                        filtered.append(r)
                    continue
                match = re.search(r'^\[(\d{4}-\d{2}-\d{2})', getattr(r, "content", ""))
                if match:
                    try:
                        dt = datetime.strptime(match.group(1), "%Y-%m-%d").timestamp()
                        if dt >= cutoff:
                            filtered.append(r)
                    except Exception:
                        filtered.append(r)
                else:
                    filtered.append(r)
            records = filtered
        except ValueError:
            pass

    items = []
    for r in records[offset : offset + limit]:
        meta = r.metadata or {}
        # resolve a unix timestamp from whichever field is available
        raw_ts = (
            meta.get("timestamp")
            or meta.get("created_at")
            or meta.get("updated_at")
            or getattr(r, "created_at", None)
            or getattr(r, "updated_at", None)
        )
        if raw_ts:
            try:
                raw_ts = float(raw_ts)
                if raw_ts > 1e11:
                    raw_ts = raw_ts / 1000.0
            except (TypeError, ValueError):
                raw_ts = None
        # fallback: parse date from content like "[2026-04-13 ..."
        if not raw_ts:
            match = re.search(r'^\[(\d{4}-\d{2}-\d{2})', getattr(r, "content", ""))
            if match:
                try:
                    raw_ts = datetime.strptime(match.group(1), "%Y-%m-%d").timestamp()
                except Exception:
                    pass
        item = {
            "id": r.id,
            "content": r.content[:300],
            "tags": list(r.tags) if r.tags else [],
            "level": level_name(r.level),
            "strength": round(r.strength, 3),
            "activation_count": r.activation_count,
            "created_at": raw_ts,
        }
        items.append(item)

    return {"records": items, "total": len(records)}


@router.post("/records")
async def create_record(payload: CreateRecordPayload):
    """Create a new record."""
    api = _get_api()
    manager = api.get_session_manager()
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

    result = api.execute_tool("store", args, session.session_id)
    _invalidate_records_cache()
    data = json.loads(result)
    return data


@router.get("/records/{record_id}")
async def get_record(record_id: str):
    """Get single record details."""
    api = _get_api()

    def _query():
        with api.brain_lock:
            rec = api.brain.get(record_id)
            if not rec:
                return None, {}
            conn_recs = {cid: api.brain.get(cid) for cid in rec.connections}
            return rec, conn_recs

    rec, conn_recs = await run_in_thread(_query, timeout=_TIMEOUT_FAST, error_msg="Record fetch timed out")
    if not rec:
        raise HTTPException(status_code=404, detail=f"Record '{record_id}' not found")

    connections = []
    for conn_id, weight in rec.connections.items():
        conn_rec = conn_recs.get(conn_id)
        if conn_rec:
            connections.append(
                {
                    "id": conn_id,
                    "content": conn_rec.content[:150],
                    "tags": list(conn_rec.tags) if conn_rec.tags else [],
                    "weight": round(weight, 3),
                }
            )

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
    api = _get_api()
    manager = api.get_session_manager()
    session = manager.get_or_create_session()

    args = {"record_id": record_id}
    if payload.content:
        args["content"] = payload.content
    if payload.tags:
        args["tags"] = payload.tags
    if payload.level:
        args["level"] = payload.level

    result = api.execute_tool("update_record", args, session.session_id)
    _invalidate_records_cache()
    data = json.loads(result)

    if "error" in data:
        raise HTTPException(status_code=404, detail=data["error"])

    return data


@router.delete("/records/{record_id}")
async def delete_record(record_id: str):
    """Delete a record."""
    api = _get_api()
    manager = api.get_session_manager()
    session = manager.get_or_create_session()

    result = api.execute_tool("delete_record", {"record_id": record_id}, session.session_id)
    _invalidate_records_cache()
    data = json.loads(result)

    if "error" in data:
        raise HTTPException(status_code=404, detail=data["error"])

    return data


@router.post("/records/{record_id}/feedback")
async def submit_record_feedback(record_id: str, payload: RecordFeedbackPayload):
    """Submit explicit positive/negative feedback for a record."""
    api = _get_api()
    manager = api.get_session_manager()
    session = manager.get_or_create_session()

    result = api.execute_tool(
        "memory_feedback",
        {
            "record_id": record_id,
            "useful": bool(payload.useful),
            "reason": (payload.reason or "").strip(),
        },
        session.session_id,
    )
    data = json.loads(result)
    if "error" in data:
        raise HTTPException(status_code=400, detail=data["error"])
    return data


@router.post("/search")
async def search_records(query: SearchQuery):
    """Search records."""
    api = _get_api()
    manager = api.get_session_manager()
    session = manager.get_or_create_session()

    args = {"query": query.query}
    if query.tags:
        args["tags"] = query.tags
    mode = (query.mode or "hybrid").strip().lower()
    tool_name = "search_exact" if mode == "exact" else "search"

    result = api.execute_tool(tool_name, args, session.session_id)

    if "No results" in result:
        return {"results": []}

    try:
        items = json.loads(result)
        
        # Apply tier and period filters to search results
        if query.tier and query.tier != "all":
            items = [item for item in items if tier_of(item.get("level")) == query.tier]
            
        if query.period and query.period != "all":
            now = time.time()
            try:
                days = int(query.period)
                cutoff = now - (days * 86400)
                filtered = []
                import re
                from datetime import datetime
                for item in items:
                    # Try metadata timestamp
                    ts = (item.get("metadata", {}) or {}).get("timestamp")
                    if ts:
                        if float(ts) > 1e11:
                            ts = float(ts) / 1000.0
                        if float(ts) >= cutoff:
                            filtered.append(item)
                        continue
                        
                    # Fallback to date in content
                    match = re.search(r'^\[(\d{4}-\d{2}-\d{2})', item.get("content", ""))
                    if match:
                        try:
                            dt = datetime.strptime(match.group(1), "%Y-%m-%d").timestamp()
                            if dt >= cutoff:
                                filtered.append(item)
                        except:
                            filtered.append(item)
                    else:
                        filtered.append(item)
                items = filtered
            except ValueError:
                pass
                
        return {"results": items}
    except json.JSONDecodeError:
        return {"results": [], "raw": result}


@router.post("/consolidate")
async def consolidate_records():
    """Run native memory consolidation."""
    api = _get_api()
    try:

        def _consolidate():
            with api.brain_lock:
                return api.brain.consolidate()

        result = await run_in_thread(_consolidate, timeout=_TIMEOUT_SLOW, error_msg="Consolidation timed out")
        return {
            "merged": result.get("merged", 0),
            "llm_merged": result.get("llm_merged", 0),
        }
    except AttributeError:
        raise HTTPException(
            status_code=501, detail="consolidate() not available in this Aura SDK version"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Consolidation failed: {e}")


@router.post("/fix-memory-levels")
async def fix_memory_levels_endpoint():
    """Downgrade records incorrectly stuck at IDENTITY level back to DOMAIN."""
    api = _get_api()
    from remy.core.background_brain import fix_memory_levels

    def _fix():
        with api.brain_lock:
            return fix_memory_levels(api.brain)

    result = await run_in_thread(_fix, timeout=_TIMEOUT_SLOW, error_msg="Fix memory levels timed out")
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
async def get_graph_data(mode: str = "user"):
    """Get knowledge graph data for visualization."""
    api = _get_api()
    graph_mode = (mode or "user").strip().lower()
    if graph_mode not in {"user", "full"}:
        raise HTTPException(status_code=400, detail="Invalid graph mode")

    def _query():
        with api.brain_lock:
            return api.brain.list_records(min_strength=0.01)

    all_records = await run_in_thread(_query, timeout=_TIMEOUT_SLOW, error_msg="Graph data fetch timed out")
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
        metadata = getattr(r, "metadata", None) or {}
        node["timestamp"] = (
            metadata.get("timestamp")
            or metadata.get("created_at")
            or metadata.get("updated_at")
            or metadata.get("recovered_at")
            or metadata.get("last_updated_at")
            or getattr(r, "created_at", None)
            or getattr(r, "updated_at", None)
        )
        if hasattr(r, "importance"):
            node["importance"] = round(r.importance, 4)
        nodes.append(node)

        for conn_id, weight in r.connections.items():
            if conn_id not in valid_ids:
                continue

            edge_key = tuple(sorted([r.id, conn_id]))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append(
                    {
                        "source": r.id,
                        "target": conn_id,
                        "weight": round(weight, 3),
                    }
                )

    return {"nodes": nodes, "edges": edges}


@router.post("/import")
async def import_brain(request: Request):
    """Import brain from JSON export."""
    from remy.core.agent_tools import Level

    api = _get_api()

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Payload too large (max 50 MB)")

    payload = await request.json()

    if isinstance(payload, dict) and "records" in payload:
        records = payload["records"]
    elif isinstance(payload, list):
        records = payload
    else:
        raise HTTPException(
            status_code=400, detail="Invalid format. Expected JSON with 'records' list."
        )

    if len(records) > 50_000:
        raise HTTPException(
            status_code=400, detail=f"Too many records ({len(records)}). Max 50,000 per import."
        )

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

    def _do_import():
        old_to_new = {}
        success_count = 0
        errors = []

        for r in records:
            try:
                old_id = r.get("id")
                if not old_id:
                    continue

                lvl_str = str(r.get("level", "L3_DOMAIN")).replace("Level.", "")
                level = level_map.get(lvl_str, Level.DOMAIN)

                with api.brain_lock:
                    new_rec = api.brain.store(
                        content=r.get("content", ""),
                        level=level,
                        tags=r.get("tags", []),
                        metadata=r.get("metadata", None),
                    )
                old_to_new[old_id] = new_rec.id
                success_count += 1
            except Exception as e:
                errors.append(f"Failed to store record {r.get('id')}: {e}")

        connections_restored = 0
        for r in records:
            old_id = r.get("id")
            new_id = old_to_new.get(old_id)
            if not new_id:
                continue

            conns = r.get("connections", {})

            target_items = []
            if isinstance(conns, dict):
                target_items = conns.items()

            for target_old, weight in target_items:
                target_new = old_to_new.get(target_old)
                if target_new:
                    try:
                        with api.brain_lock:
                            api.brain.connect(new_id, target_new, weight=float(weight))
                        connections_restored += 1
                    except Exception:
                        pass

        return {
            "imported": success_count,
            "connections_restored": connections_restored,
            "errors": errors[:10],
        }

    result = await run_in_thread(_do_import, timeout=_TIMEOUT_SLOW, error_msg="Import timed out — try smaller batches")
    _invalidate_records_cache()
    return result


@router.get("/export")
async def export_brain():
    """Export all brain records as JSON."""
    api = _get_api()

    def _query():
        with api.brain_lock:
            return api.brain.list_records(min_strength=0.0)

    records = await run_in_thread(_query, timeout=_TIMEOUT_SLOW, error_msg="Export timed out")

    items = []
    for r in records:
        item = {
            "id": r.id,
            "content": r.content,
            "tags": list(r.tags) if r.tags else [],
            "level": level_name(r.level),
            "strength": round(r.strength, 4),
            "activation_count": r.activation_count,
            "metadata": r.metadata or {},
            "connections": {cid: round(w, 3) for cid, w in r.connections.items()},
        }
        if hasattr(r, "importance"):
            item["importance"] = round(r.importance, 4)
        items.append(item)

    return JSONResponse(
        content={
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "count": len(items),
            "records": items,
        },
        headers={"Content-Disposition": "attachment; filename=remy-brain-export.json"},
    )


# ============== MEMORY TIERS ==============


@router.get("/memory/tier-stats")
async def get_tier_stats():
    """Memory breakdown by cognitive/core tier."""
    api = _get_api()

    def _query():
        with api.brain_lock:
            return _normalize_tier_stats(api.brain.tier_stats())

    return await run_in_thread(_query, timeout=_TIMEOUT_FAST, error_msg="Tier stats timed out")


@router.get("/memory/cognitive")
async def get_cognitive_records(offset: int = 0, limit: int = 50):
    """Cognitive tier records (WORKING + DECISIONS)."""
    api = _get_api()

    def _query():
        with api.brain_lock:
            return api.brain.recall_cognitive(limit=5000)

    records = await run_in_thread(_query, error_msg="Cognitive records timed out")

    items = []
    for r in records[offset : offset + limit]:
        item = {
            "id": r.id,
            "content": r.content[:300],
            "tags": list(r.tags) if r.tags else [],
            "level": level_name(r.level),
            "tier": "cognitive",
            "strength": round(r.strength, 3),
            "activation_count": r.activation_count,
        }
        items.append(item)
    return {"records": items, "total": len(records), "tier": "cognitive"}


@router.get("/memory/core")
async def get_core_records(offset: int = 0, limit: int = 50):
    """Core tier records (DOMAIN + IDENTITY)."""
    api = _get_api()

    def _query():
        with api.brain_lock:
            return api.brain.recall_core(limit=5000)

    records = await run_in_thread(_query, error_msg="Core records timed out")

    items = []
    for r in records[offset : offset + limit]:
        item = {
            "id": r.id,
            "content": r.content[:300],
            "tags": list(r.tags) if r.tags else [],
            "level": level_name(r.level),
            "tier": "core",
            "strength": round(r.strength, 3),
            "activation_count": r.activation_count,
        }
        items.append(item)
    return {"records": items, "total": len(records), "tier": "core"}


@router.get("/memory/promotion-candidates")
async def get_promotion_candidates():
    """Cognitive records that are candidates for promotion to core."""
    api = _get_api()

    def _query():
        with api.brain_lock:
            return api.brain.promotion_candidates()

    candidates = await run_in_thread(_query, timeout=_TIMEOUT_FAST, error_msg="Promotion candidates timed out")

    items = []
    for r in candidates:
        items.append(
            {
                "id": r.id,
                "content": r.content[:300],
                "level": level_name(r.level),
                "strength": round(r.strength, 3),
                "activation_count": r.activation_count,
                "tags": list(r.tags) if r.tags else [],
            }
        )
    return {"candidates": items, "count": len(items)}
