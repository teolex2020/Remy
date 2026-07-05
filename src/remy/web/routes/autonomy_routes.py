"""
Autonomy routes — status, toggle, activity log.
"""

import asyncio
import logging
from html import escape
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse

from remy.web.routes._helpers import _get_api, run_in_thread, _TIMEOUT_SLOW, _TIMEOUT_EVAL

logger = logging.getLogger("WebAPI")

router = APIRouter()

async def build_autonomy_status_payload():
    """Return current autonomous loop status payload."""
    defaults = {
        "running": False,
        "version": "v2",
        "transport_connected": False,
        "session_id": None,
        "budget": None,
        "current_goal": None,
        "current_mission": None,
        "current_task": None,
        "current_step": None,
        "last_cycle_result": None,
        "current_role": "",
        "last_agent_response": None,
        "last_research_activity": None,
        "research_session": None,
        "pending_approvals": 0,
        "approval_queue": [],
        "scheduler_reason": "",
        "scheduler_selection": {},
        "scheduler_decisions_recent": [],
        "stuck_missions_count": 0,
        "stuck_missions": [],
        "specialist_resolution": {},
        "quality_debt_by_specialist": [],
        "evidence_debt_queue": [],
        "routing_pressure": {},
        "evaluation": {},
        "factuality": {},
        "active_harness_module": {},
    }
    try:
        from remy.core.combined_runner import (
            get_activity_runtime_snapshot,
            is_runtime_transport_connected,
        )

        status = dict(defaults)
        status.update(
            get_activity_runtime_snapshot(
                goal_limit=3,
                approval_limit=10,
                transport_connected=is_runtime_transport_connected(),
            )
            or {}
        )
        if (
            status.get("session_id") is not None
            or status.get("running")
            or status.get("pending_approvals")
            or status.get("budget") is not None
        ):
            return status
    except ImportError:
        pass
    return defaults


@router.get("/autonomy/status")
async def get_autonomy_status():
    """Return current autonomous loop status."""
    return await build_autonomy_status_payload()


_toggle_lock = asyncio.Lock()
_last_toggle_time: float = 0.0
_TOGGLE_DEBOUNCE_SEC = 2.0  # Reject toggles within 2s of previous toggle


@router.post("/autonomy/toggle")
async def toggle_autonomy():
    """Start or stop the autonomous loop at runtime."""
    import time

    global _last_toggle_time

    now = time.monotonic()
    if now - _last_toggle_time < _TOGGLE_DEBOUNCE_SEC:
        # Debounce: too soon after previous toggle — return current state
        try:
            from remy.core.combined_runner import is_autonomy_running

            running = is_autonomy_running()
        except ImportError:
            running = False
        return {"running": running, "action": "debounced"}

    if _toggle_lock.locked():
        try:
            from remy.core.combined_runner import is_autonomy_running

            running = is_autonomy_running()
        except ImportError:
            running = False
        return {"running": running, "action": "debounced"}

    async with _toggle_lock:
        _last_toggle_time = time.monotonic()
        try:
            from remy.core.combined_runner import is_autonomy_running, start_autonomy, stop_autonomy

            if is_autonomy_running():
                await stop_autonomy()
                return {"running": False, "action": "stopped"}
            else:
                await start_autonomy()
                return {"running": True, "action": "started"}
        except ImportError:
            return {"error": "Combined runner not active", "running": False}


@router.post("/server/shutdown")
async def shutdown_server():
    """Gracefully shut down the entire server (all channels)."""
    logger.info("Server shutdown requested via API")

    try:
        from remy.core.combined_runner import request_graceful_shutdown

        if request_graceful_shutdown():
            return {"ok": True, "message": "Server shutting down gracefully..."}
    except Exception as e:
        logger.warning("Graceful shutdown request failed: %s", e)

    return {
        "ok": False,
        "message": "Combined runner shutdown path is not active. Stop the process manually.",
    }


@router.get("/survival")
async def get_survival_status():
    """Return agent's financial survival status — balance, runway, spending."""
    import asyncio as _aio

    def _query():
        from remy.core.combined_runner import get_budget_runtime_snapshot
        from remy.core.survival import check_wallet_balance, estimate_runway

        # Always fetch fresh balance for the API
        balance = check_wallet_balance()
        trx_price_usd = 0.12
        total_usd = balance["usdt"] + (balance["trx"] * trx_price_usd)

        # LLM cost data (load first — needed for runway calculation)
        budget = get_budget_runtime_snapshot(goal_limit=3, approval_limit=10)
        llm_cost_today = float(budget.get("llm_cost_today") or 0.0)
        runway_days = estimate_runway(total_usd, llm_cost_today)

        return {
            "wallet": {
                "address": balance.get("address", "TNjyL4vZwBQg1tzudWWM8aFavPCYZTRAJY"),
                "trx": round(balance["trx"], 4),
                "usdt": round(balance["usdt"], 4),
                "total_usd": round(total_usd, 2),
                "error": balance.get("error"),
            },
            "runway": {
                "days": round(runway_days, 1),
                "daily_burn_usd": round(llm_cost_today + 0.50, 2),
                "status": (
                    "CRITICAL" if runway_days < 0.5 else "WARNING" if runway_days < 2 else "HEALTHY"
                ),
            },
            "spending": {
                "llm_cost_today_usd": round(llm_cost_today, 6),
                "llm_cost_lifetime_usd": round(float(budget.get("llm_cost_lifetime_usd") or 0.0), 6),
                "llm_tokens_today": int(budget.get("llm_tokens_today") or 0),
                "llm_tokens_lifetime": int(budget.get("llm_tokens_lifetime") or 0),
            },
            "last_check": budget,
        }

    result = await _aio.to_thread(_query)
    return result


@router.get("/activity")
async def get_activity():
    """Aggregate autonomous agent activity data."""
    api = _get_api()
    from remy.core.combined_runner import get_activity_feed_snapshot

    return await run_in_thread(
        get_activity_feed_snapshot,
        api.brain,
        api.brain_lock,
        goal_limit=50,
        outcome_limit=100,
        reflection_limit=10,
        proactive_limit=20,
    )


@router.get("/autonomy/research-artifacts/{record_id}/markdown", response_class=PlainTextResponse)
async def get_research_artifact_markdown(record_id: str):
    """Return the canonical markdown body for a research artifact."""
    api = _get_api()
    target_id = str(record_id or "").strip()
    if not target_id:
        raise HTTPException(status_code=400, detail="Missing record id")

    def _load_markdown():
        with api.brain_lock:
            rec = api.brain.get(target_id)
        if not rec:
            return None
        metadata = rec.metadata or {}
        markdown_body = str(metadata.get("markdown_body", "") or "")
        if markdown_body:
            return markdown_body
        content = str(getattr(rec, "content", "") or "")
        if content:
            return content
        return None

    markdown = await run_in_thread(_load_markdown)
    if not markdown:
        raise HTTPException(status_code=404, detail="Research artifact markdown not found")
    return PlainTextResponse(markdown, media_type="text/markdown; charset=utf-8")


@router.get("/autonomy/research-artifacts/{record_id}/view", response_class=HTMLResponse)
async def get_research_artifact_viewer(record_id: str):
    """Return a readable viewer shell for a research markdown artifact."""
    api = _get_api()
    target_id = str(record_id or "").strip()
    if not target_id:
        raise HTTPException(status_code=400, detail="Missing record id")

    def _load_artifact_meta():
        with api.brain_lock:
            rec = api.brain.get(target_id)
        if not rec:
            return None
        metadata = rec.metadata or {}
        source_domains = []
        for source in list(metadata.get("sources") or []):
            url = str(source or "").strip()
            if not url:
                continue
            try:
                host = (urlparse(url).netloc or "").lower().strip()
            except Exception:
                host = ""
            if host and host not in source_domains:
                source_domains.append(host)
        markdown_body = str(metadata.get("markdown_body", "") or getattr(rec, "content", "") or "")
        title = str(metadata.get("title", "") or "").strip()
        if not title:
            first_heading = next(
                (line.strip().lstrip("#").strip() for line in markdown_body.splitlines() if line.strip().startswith("#")),
                "",
            )
            title = first_heading or f"Research Report {target_id}"
        return {
            "title": title,
            "pdf_url": str(metadata.get("pdf_url", "") or ""),
            "pdf_filename": str(metadata.get("pdf_filename", "") or "report.pdf"),
            "citation_complete": bool(metadata.get("citation_complete", False)),
            "citation_count": int(metadata.get("citation_count") or 0),
            "findings_count": int(metadata.get("findings_count") or 0),
            "confidence_avg": float(metadata.get("confidence_avg") or 0.0),
            "evidence_note": str(metadata.get("evidence_note", "") or ""),
            "source_domains": source_domains[:4],
            "contradictions_count": int(metadata.get("contradictions_count") or 0),
        }

    artifact_meta = await run_in_thread(_load_artifact_meta)
    if not artifact_meta:
        raise HTTPException(status_code=404, detail="Research artifact not found")

    title = artifact_meta["title"]
    pdf_url = artifact_meta["pdf_url"]
    pdf_filename = artifact_meta["pdf_filename"]
    citation_status = "complete" if artifact_meta["citation_complete"] else "partial"
    confidence_pct = round(float(artifact_meta["confidence_avg"] or 0.0) * 100)
    evidence_note = artifact_meta["evidence_note"]
    source_domains = ", ".join(artifact_meta["source_domains"]) or "n/a"
    contradictions_count = int(artifact_meta["contradictions_count"] or 0)
    warning_badges = []
    if not artifact_meta["citation_complete"]:
        warning_badges.append("partial citations")
    if contradictions_count > 0:
        warning_badges.append(f"{contradictions_count} contradictions")
    if evidence_note:
        warning_badges.append("review evidence note")
    badges_html = ""
    if warning_badges:
        badges_html = '<div class="research-viewer-evidence-badges">' + "".join(
            f'<span class="research-viewer-evidence-badge">{escape(badge)}</span>'
            for badge in warning_badges
        ) + "</div>"
    evidence_note_html = (
        f'<div class="research-viewer-evidence-note">{escape(evidence_note)}</div>'
        if evidence_note
        else ""
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escape(title)} - Remy</title>
    <link rel="stylesheet" href="/css/main.css?v=1.15">
</head>
<body class="research-viewer-page">
    <main class="research-viewer-shell" data-record-id="{escape(target_id)}">
        <header class="research-viewer-header">
            <div>
                <div class="research-viewer-kicker">Research Artifact</div>
                <h1 class="research-viewer-title">{escape(title)}</h1>
            </div>
            <div class="research-viewer-actions">
                <a class="research-viewer-link" href="/api/autonomy/research-artifacts/{escape(target_id)}/markdown" download="{escape(target_id)}.md">Download Markdown</a>
                {f'<a class="research-viewer-link" href="{escape(pdf_url)}" download="{escape(pdf_filename)}">Download PDF</a>' if pdf_url else ''}
                <a class="research-viewer-link" href="/" target="_blank" rel="noreferrer">Remy</a>
            </div>
        </header>
        <section class="research-viewer-status" id="research-viewer-status">Loading report...</section>
        <section class="research-viewer-evidence">
            <div class="research-viewer-evidence-card">
                <div class="research-viewer-evidence-label">Citation Status</div>
                <div class="research-viewer-evidence-value">{escape(citation_status)}</div>
            </div>
            <div class="research-viewer-evidence-card">
                <div class="research-viewer-evidence-label">Sources</div>
                <div class="research-viewer-evidence-value">{artifact_meta["citation_count"]}</div>
            </div>
            <div class="research-viewer-evidence-card">
                <div class="research-viewer-evidence-label">Findings</div>
                <div class="research-viewer-evidence-value">{artifact_meta["findings_count"]}</div>
            </div>
            <div class="research-viewer-evidence-card">
                <div class="research-viewer-evidence-label">Avg Confidence</div>
                <div class="research-viewer-evidence-value">{confidence_pct}%</div>
            </div>
            <div class="research-viewer-evidence-card">
                <div class="research-viewer-evidence-label">Source Domains</div>
                <div class="research-viewer-evidence-value research-viewer-evidence-value--small">{escape(source_domains)}</div>
            </div>
            {badges_html}
            {evidence_note_html}
        </section>
        <div class="research-viewer-layout">
            <aside class="research-viewer-sidebar" id="research-viewer-sidebar" hidden>
                <div class="research-viewer-sidebar-title">Sections</div>
                <nav class="research-viewer-sections" id="research-viewer-sections"></nav>
                <div class="research-viewer-sidebar-group" id="research-viewer-sources-wrap" hidden>
                    <div class="research-viewer-sidebar-title">Sources</div>
                    <nav class="research-viewer-sections" id="research-viewer-sources"></nav>
                </div>
            </aside>
            <article class="research-viewer-document markdown-body" id="research-viewer-document" hidden></article>
        </div>
    </main>
    <script type="module" src="/js/research-viewer.js?v=1.15"></script>
</body>
</html>"""
    return HTMLResponse(html)


@router.get("/autonomy/capability-packs")
async def get_capability_packs():
    """Return all registered capability packs."""
    from remy.core.capability_packs import get_all_packs, pack_summary

    return {"packs": {pid: pack_summary(p) for pid, p in get_all_packs().items()}}


@router.get("/autonomy/benchmarks")
async def get_autonomy_benchmarks():
    """Return the latest deterministic autonomy benchmark report."""
    from remy.core.autonomy_benchmarks import load_benchmark_report, run_autonomy_benchmarks

    report = await run_in_thread(load_benchmark_report)
    if report is None:
        report = await run_in_thread(run_autonomy_benchmarks)
    return report


@router.post("/autonomy/benchmarks/run")
async def run_autonomy_benchmarks_now():
    """Run the deterministic benchmark pack immediately and return the report."""
    from remy.core.autonomy_benchmarks import run_autonomy_benchmarks

    return await run_in_thread(run_autonomy_benchmarks)


@router.get("/autonomy/live-validation")
async def get_autonomy_live_validation():
    """Return the latest live validation readiness report."""
    from remy.core.autonomy_live_validation import (
        load_live_validation_report,
        run_live_validation_pack,
    )

    report = await run_in_thread(load_live_validation_report)
    if report is None:
        report = await run_in_thread(run_live_validation_pack)
    return report


@router.post("/autonomy/live-validation/run")
async def run_autonomy_live_validation_now():
    """Run the live validation readiness pack immediately and return the report."""
    from remy.core.autonomy_live_validation import run_live_validation_pack

    return await run_in_thread(run_live_validation_pack)


@router.get("/autonomy/live-validation/scenarios")
async def get_autonomy_live_validation_scenarios():
    """Return editable live validation scenarios."""
    from remy.core.autonomy_live_validation import load_live_validation_scenarios

    return {"scenarios": await run_in_thread(load_live_validation_scenarios)}


@router.post("/autonomy/live-validation/scenarios")
async def save_autonomy_live_validation_scenarios(body: dict | None = None):
    """Replace the live validation scenario pack and return the normalized scenarios."""
    from remy.core.autonomy_live_validation import (
        load_live_validation_scenarios,
        run_live_validation_pack,
        save_live_validation_scenarios,
    )

    scenarios = (body or {}).get("scenarios")
    if not isinstance(scenarios, list):
        raise HTTPException(status_code=400, detail="scenarios must be a list")

    await run_in_thread(save_live_validation_scenarios, scenarios)
    normalized = await run_in_thread(load_live_validation_scenarios)
    report = await run_in_thread(run_live_validation_pack)
    return {"scenarios": normalized, "report": report}


@router.post("/autonomy/goals/{record_id}/resume")
async def resume_autonomy_goal(record_id: str, body: dict | None = None):
    """Resume a blocked goal while preserving a continuation note."""
    from remy.core.autonomy_goals import resume_goal_from_blocker

    api = _get_api()
    note = ((body or {}).get("note") or "").strip()

    def _resume():
        with api.brain_lock:
            rec = api.brain.get(record_id)
        if not rec:
            return "not_found", None
        meta = rec.metadata or {}
        if meta.get("type") != "autonomous_goal":
            return "not_goal", None
        if meta.get("status") not in ("blocked_external", "blocked_by_user"):
            return "not_blocked", meta
        resumed = resume_goal_from_blocker(record_id, note=note)
        if not resumed:
            return "resume_failed", meta
        with api.brain_lock:
            updated = api.brain.get(record_id)
        return "ok", updated.metadata if updated else meta

    result, meta = await run_in_thread(_resume)
    if result == "not_found":
        raise HTTPException(status_code=404, detail="Goal not found")
    if result == "not_goal":
        raise HTTPException(status_code=400, detail="Record is not an autonomous goal")
    if result == "not_blocked":
        raise HTTPException(status_code=400, detail="Goal is not blocked")
    if result == "resume_failed":
        raise HTTPException(status_code=409, detail="Goal could not be resumed")
    return {
        "ok": True,
        "id": record_id,
        "status": meta.get("status", "active"),
        "resume_context": meta.get("resume_context", ""),
    }


@router.post("/autonomy/goals/{record_id}/unblock")
async def unblock_autonomy_goal(record_id: str):
    """Unblock a goal and mark it active without adding a continuation note."""
    from remy.core.autonomy_goals import unblock_goal

    api = _get_api()

    def _unblock():
        with api.brain_lock:
            rec = api.brain.get(record_id)
        if not rec:
            return "not_found", None
        meta = rec.metadata or {}
        if meta.get("type") != "autonomous_goal":
            return "not_goal", None
        if meta.get("status") not in ("blocked_external", "blocked_by_user"):
            return "not_blocked", meta
        unblock_goal(record_id)
        with api.brain_lock:
            updated = api.brain.get(record_id)
        return "ok", updated.metadata if updated else meta

    result, meta = await run_in_thread(_unblock)
    if result == "not_found":
        raise HTTPException(status_code=404, detail="Goal not found")
    if result == "not_goal":
        raise HTTPException(status_code=400, detail="Record is not an autonomous goal")
    if result == "not_blocked":
        raise HTTPException(status_code=400, detail="Goal is not blocked")
    return {
        "ok": True,
        "id": record_id,
        "status": meta.get("status", "active"),
    }


@router.post("/autonomy/goals/{record_id}/archive")
async def archive_autonomy_goal(record_id: str, body: dict | None = None):
    """Archive a goal so the autonomous loop will no longer execute it."""
    from remy.core.autonomy_goals import archive_goal

    api = _get_api()
    reason = ((body or {}).get("reason") or "archived_by_user").strip() or "archived_by_user"

    def _archive():
        with api.brain_lock:
            rec = api.brain.get(record_id)
        if not rec:
            return "not_found", None
        meta = rec.metadata or {}
        if meta.get("type") != "autonomous_goal":
            return "not_goal", None
        ok, updated = archive_goal(record_id, reason=reason)
        if not ok:
            return "archive_failed", meta
        return "ok", updated

    result, meta = await run_in_thread(_archive)
    if result == "not_found":
        raise HTTPException(status_code=404, detail="Goal not found")
    if result == "not_goal":
        raise HTTPException(status_code=400, detail="Record is not an autonomous goal")
    if result == "archive_failed":
        raise HTTPException(status_code=409, detail="Goal could not be archived")
    return {
        "ok": True,
        "id": record_id,
        "status": meta.get("status", "archived"),
        "archived_reason": meta.get("archived_reason", reason),
    }
