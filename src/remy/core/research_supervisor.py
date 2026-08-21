"""Durable background execution for chat-created research projects.

The project record in Aura is the queue.  A tick claims one pending query,
persists its receipt, and yields, so ordinary chat work remains responsive.
Interrupted ``running`` jobs are re-queued on process startup.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from remy.core.execution_ledger import get_execution_ledger

logger = logging.getLogger("ResearchSupervisor")

_task: asyncio.Task | None = None
_stop_event: asyncio.Event | None = None
_wake_event: asyncio.Event | None = None
_loop: asyncio.AbstractEventLoop | None = None
_MAX_QUERY_ATTEMPTS = 2
_IDLE_SECONDS = 8.0


@dataclass(slots=True)
class _OwnedResearchRecord:
    """Minimal Aura record view carrying its authoritative MicroBrain owner."""

    id: str
    metadata: dict[str, Any]


def _now() -> str:
    return datetime.now().isoformat()


def _owner_identity(meta: dict) -> tuple[str, str]:
    """Validate the immutable Project -> MicroBrain ownership pair."""
    from remy.core.project_store import get_project_store

    owner_project_id = str(meta.get("owner_project_id") or "").strip()
    brain_id = str(meta.get("brain_id") or "").strip()
    if not owner_project_id or not brain_id:
        raise ValueError("Research job has no project/MicroBrain owner")
    owner = get_project_store().require_project(
        owner_project_id,
        include_archived=False,
    )
    if owner.brain_id != brain_id:
        raise ValueError("Research job brain_id does not match its project owner")
    return owner_project_id, brain_id


def wake_research_supervisor() -> bool:
    """Wake the worker after ``start_research`` commits a queue record."""
    if not _loop or not _wake_event or _loop.is_closed():
        return False
    _loop.call_soon_threadsafe(_wake_event.set)
    return bool(_task and not _task.done())


def _project_records() -> list[Any]:
    from remy.core.agent_tools import brain, brain_lock
    from remy.core.microbrain import bind_project
    from remy.core.project_store import get_project_store

    records: list[_OwnedResearchRecord] = []
    for owner in get_project_store().list_projects(include_archived=False):
        with bind_project(owner.project_id), brain_lock:
            found = list(
                brain.search(query="", tags=["research-project"], limit=100) or []
            )
        for rec in found:
            metadata = dict(rec.metadata or {})
            # The catalog location is authoritative. This also upgrades legacy
            # research records in memory without mutating their historical data.
            metadata["owner_project_id"] = owner.project_id
            metadata["brain_id"] = owner.brain_id
            records.append(
                _OwnedResearchRecord(
                    id=str(rec.id),
                    metadata=metadata,
                )
            )
    return records


def _find_project(project_id: str):
    for rec in _project_records():
        if str((rec.metadata or {}).get("project_id") or "") == project_id:
            return rec
    return None


def _update_project(project_id: str, **changes) -> dict:
    from remy.core.agent_tools import brain, brain_lock
    from remy.core.microbrain import bind_project

    rec = _find_project(project_id)
    if not rec:
        return {}
    owner_project_id = str((rec.metadata or {}).get("owner_project_id") or "")
    if not owner_project_id:
        return {}
    with bind_project(owner_project_id), brain_lock:
        meta = dict(rec.metadata or {})
        meta.update(changes)
        meta["owner_project_id"] = owner_project_id
        brain.update(rec.id, metadata=meta)
        return meta


def _append_receipt(project_id: str, receipt: dict, **changes) -> dict:
    rec = _find_project(project_id)
    if not rec:
        return {}
    receipts = list((rec.metadata or {}).get("receipts") or [])
    receipts.append(receipt)
    return _update_project(project_id, receipts=receipts[-100:], **changes)


def _sync_research_run(meta: dict, *, status: str, phase: str, step: str) -> dict:
    attempt_id = str(meta.get("execution_attempt_id") or "")
    if not attempt_id:
        return {}
    ledger = get_execution_ledger()
    getter = getattr(ledger, "get", None)
    attempt = getter(attempt_id) if callable(getter) else None
    if not attempt or not isinstance((attempt.get("metadata") or {}).get("run_envelope"), dict):
        return {}
    from remy.core.run_envelope import update_run

    try:
        envelope = update_run(
            attempt_id,
            status=status,
            phase=phase,
            current_step=step,
            event="research_progress",
        )
    except (KeyError, RuntimeError):
        return {}
    project_id = str(meta.get("project_id") or envelope.get("source_id") or "")
    if project_id:
        _update_project(project_id, run_id=envelope.get("run_id", ""), run_envelope=envelope)
    return envelope


def _checkpoint(project_id: str, node: str, status: str, detail: str = "", **changes) -> dict:
    """Persist a user-visible, query-boundary research checkpoint."""
    rec = _find_project(project_id)
    if not rec:
        return {}
    meta = dict(rec.metadata or {})
    now = _now()
    checkpoint = {
        "node": node,
        "status": status,
        "detail": detail,
        "query_index": int(meta.get("next_query_index", meta.get("queries_done", 0)) or 0),
        "queries_done": int(meta.get("queries_done", 0) or 0),
        "findings_count": int(meta.get("findings_count", 0) or 0),
        "updated_at": now,
    }
    history = list(meta.get("checkpoint_history") or [])
    history.append(checkpoint)
    updated = _update_project(
        project_id,
        durable_checkpoint=checkpoint,
        checkpoint_history=history[-50:],
        worker_heartbeat=now,
        **changes,
    )
    phase = {
        "paused": "paused",
        "pausing": "pausing",
        "queued": "queued",
        "retrying": "retrying",
        "recovered": "resuming",
    }.get(status, "executing")
    run_status = "paused" if status == "paused" else "running"
    _sync_research_run(
        {**meta, **updated, "project_id": project_id},
        status=run_status,
        phase=phase,
        step=detail or node,
    )
    return updated


def pause_research_project(project_id: str) -> dict:
    """Request a safe pause after the current network operation/query."""
    rec = _find_project(project_id)
    if not rec:
        raise KeyError(project_id)
    meta = dict(rec.metadata or {})
    state = str(meta.get("job_state") or "queued")
    if meta.get("status") in {"complete", "abandoned", "failed"} or state in {
        "completed", "completed_with_limits", "failed", "cancelled", "blocked",
    }:
        raise ValueError("Research project is already finished.")
    pausing = state in {"running", "retrying"}
    updated = _checkpoint(
        project_id,
        "pause_boundary",
        "pausing" if pausing else "paused",
        "Waiting for the current query to commit." if pausing else "Paused between queries.",
        pause_requested=True,
        job_state="pausing" if pausing else "paused",
    )
    wake_research_supervisor()
    return updated


def resume_research_project(project_id: str) -> dict:
    """Resume a project from its last committed query boundary."""
    rec = _find_project(project_id)
    if not rec:
        raise KeyError(project_id)
    meta = dict(rec.metadata or {})
    if str(meta.get("job_state") or "") not in {"paused", "pausing"}:
        raise ValueError("Research project is not paused.")
    updated = _checkpoint(
        project_id,
        "query_queue",
        "queued",
        "Resumed from the last committed query.",
        pause_requested=False,
        job_state="queued",
        current_query="",
    )
    wake_research_supervisor()
    return updated


def _recover_interrupted_projects() -> int:
    ledger = get_execution_ledger()
    ledger.recover_orphans(kind="research")
    recovered = 0
    for rec in _project_records():
        meta = dict(rec.metadata or {})
        if meta.get("status") in {"complete", "abandoned"}:
            continue
        state = str(meta.get("job_state") or "")
        if state in {"completed", "failed", "cancelled", "blocked"}:
            continue
        # Legacy projects had only status=researching and were never executed.
        if not state or state in {"running", "retrying"}:
            project_id = str(meta.get("project_id") or "")
            if not project_id:
                continue
            attempt_id = str(meta.get("execution_attempt_id") or "")
            attempt = ledger.get(attempt_id) if attempt_id else None
            if attempt and attempt.get("state") in {"claimed", "running"}:
                continue
            _append_receipt(
                project_id,
                {
                    "at": _now(),
                    "event": "recovered",
                    "previous_state": state or "legacy",
                    "previous_attempt_state": (attempt or {}).get("state", "missing"),
                },
                job_version=1,
                job_state="queued",
                execution_attempt_id="",
                next_query_index=int(meta.get("queries_done", 0) or 0),
                worker_heartbeat=_now(),
            )
            _checkpoint(
                project_id,
                "query_queue",
                "recovered",
                "Recovered after process restart; continuing from the last committed query.",
            )
            recovered += 1
    return recovered


def _next_project():
    candidates = []
    for rec in _project_records():
        meta = rec.metadata or {}
        if meta.get("status") in {"complete", "abandoned"}:
            continue
        if meta.get("job_state") in {"queued", "running", "retrying"}:
            candidates.append(rec)
    return min(candidates, key=lambda r: str((r.metadata or {}).get("started_at") or ""), default=None)


def _ensure_execution_attempt(project_id: str, meta: dict) -> str:
    ledger = get_execution_ledger()
    owner_project_id, brain_id = _owner_identity(meta)
    attempt_id = str(meta.get("execution_attempt_id") or "")
    attempt = ledger.get(attempt_id) if attempt_id else None
    if attempt and attempt.get("state") in {"claimed", "running"}:
        if attempt.get("state") == "claimed":
            ledger.mark_running(attempt_id)
        return attempt_id
    from remy.core.run_envelope import RunLimits, make_run_envelope

    run_id = f"run-{uuid.uuid4().hex[:12]}"
    query_count = max(1, len(meta.get("query_plan") or []))
    envelope = make_run_envelope(
        run_id=run_id,
        kind="research",
        source_id=project_id,
        goal=str(meta.get("topic") or meta.get("context") or "Background research"),
        owner_project_id=owner_project_id,
        brain_id=brain_id,
        conversation_id=str(meta.get("session_id") or f"research:{project_id}"),
        limits=RunLimits(
            max_turns=query_count + 10,
            token_budget=max(250_000, query_count * 64_000),
            max_parallel_workers=1,
            loop_repeat_limit=max(4, min(query_count + 2, 12)),
        ),
        metadata={"topic": meta.get("topic", "")},
    )
    attempt = ledger.claim(
        kind="research",
        job_id=project_id,
        idempotency_class="idempotent",
        owner_project_id=owner_project_id,
        brain_id=brain_id,
        session_id=str(meta.get("session_id") or f"research:{project_id}"),
        channel=str(meta.get("channel") or meta.get("delivery_target") or "web"),
        metadata={
            "topic": meta.get("topic", ""),
            "project_id": project_id,
            "owner_project_id": owner_project_id,
            "brain_id": brain_id,
            "run_envelope": envelope,
        },
    )
    attempt_id = str(attempt["attempt_id"])
    ledger.mark_running(attempt_id)
    envelope.update({"attempt_id": attempt_id, "ledger_state": "running"})
    _update_project(
        project_id,
        execution_attempt_id=attempt_id,
        run_id=run_id,
        run_envelope=envelope,
    )
    return attempt_id


def _research_run_step(attempt_id: str, label: str, *, signature: str = "") -> None:
    """Update new envelopes while remaining compatible with legacy attempts."""
    ledger = get_execution_ledger()
    getter = getattr(ledger, "get", None)
    attempt = getter(attempt_id) if callable(getter) else None
    if not attempt or not isinstance((attempt.get("metadata") or {}).get("run_envelope"), dict):
        return
    from remy.core.run_envelope import RunCoordinator

    envelope = RunCoordinator(attempt_id).step(label, signature=signature)
    project_id = str(envelope.get("source_id") or "")
    if project_id:
        _update_project(project_id, run_id=envelope.get("run_id", ""), run_envelope=envelope)


def _finish_execution_attempt(
    meta: dict,
    state: str,
    *,
    output_ref: str = "",
    error: str = "",
) -> None:
    attempt_id = str(meta.get("execution_attempt_id") or "")
    if not attempt_id:
        return
    try:
        ledger = get_execution_ledger()
        getter = getattr(ledger, "get", None)
        attempt = getter(attempt_id) if callable(getter) else None
        if attempt and isinstance((attempt.get("metadata") or {}).get("run_envelope"), dict):
            from remy.core.run_envelope import finish_run

            envelope = finish_run(
                attempt_id,
                status="interrupted" if state == "unknown" else state,
                stop_reason=("partial_results" if state == "completed_with_limits" else ""),
                output_ref=output_ref,
                error=error,
            )
            project_id = str(meta.get("project_id") or envelope.get("source_id") or "")
            if project_id:
                _update_project(
                    project_id,
                    run_id=envelope.get("run_id", ""),
                    run_envelope=envelope,
                )
        else:
            ledger.finish(
                attempt_id,
                state,
                output_ref=output_ref,
                error=error,
                metadata={"project_id": meta.get("project_id", "")},
            )
    except (KeyError, RuntimeError):
        logger.debug("Research attempt %s was already finalized", attempt_id)


def _queue_continuation(meta: dict, content: str, *, status: str, report_id: str = "") -> str:
    session_id = str(meta.get("session_id") or "")
    if not session_id:
        return ""
    owner_project_id, brain_id = _owner_identity(meta)
    return get_execution_ledger().enqueue_continuation(
        session_id=session_id,
        owner_project_id=owner_project_id,
        brain_id=brain_id,
        kind="research_result",
        source_id=str(meta.get("project_id") or ""),
        content=content,
        metadata={
            "status": status,
            "topic": meta.get("topic", ""),
            "report_id": report_id,
            "owner_project_id": owner_project_id,
            "brain_id": brain_id,
            "delivery_target": meta.get("delivery_target") or meta.get("channel") or "web",
        },
    )


def _discover(query: str, context: str = "") -> list[dict]:
    direct = re.findall(r"https?://[^\s)\]>'\"]+", f"{context} {query}")
    results = [{"uri": url.rstrip(".,;"), "title": "Direct source"} for url in direct]
    if results:
        return results[:3]
    from ddgs import DDGS
    raw = DDGS(timeout=15).text(
        query,
        max_results=6,
        backend="duckduckgo,brave,google,mojeek,yahoo",
    )
    return [
        {"uri": str(item.get("href") or ""), "title": str(item.get("title") or "")}
        for item in (raw or []) if item.get("href")
    ]


def _fetch_source(source: dict) -> tuple[str, str, str]:
    from remy.core.cancellation import check_cancelled
    from remy.core.web_content import extract_visible_text, fetch_html

    check_cancelled()
    url = str(source.get("uri") or "")
    text = ""
    page_title = ""
    try:
        html = fetch_html(url, timeout=20)
        text, page_title = extract_visible_text(html)
    except Exception:
        logger.debug("Static fetch failed for %s; trying Chromium", url, exc_info=True)
    if len(text.strip()) < 80:
        # Isolated Chromium fallback for JavaScript-rendered pages. It does not
        # reuse or disturb the interactive browser session owned by the user.
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            check_cancelled()
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                try:
                    page.wait_for_load_state("networkidle", timeout=5000)
                except Exception:
                    pass
                page_title = page.title()
                text = page.locator("body").inner_text(timeout=10000)
            finally:
                browser.close()
        check_cancelled()
    if len(text.strip()) < 80:
        raise ValueError("page returned too little visible text")
    title = page_title or str(source.get("title") or "Untitled source")
    # Keep the finding extractive. Synthesis happens only after all queries.
    excerpt = re.sub(r"\s+", " ", text).strip()[:2400]
    return url, title, f"{title}. Extracted page evidence: {excerpt}"


async def _run_query(project_id: str, meta: dict) -> None:
    """Run one durable step inside the research project's owning MicroBrain."""
    from remy.core.microbrain import bind_project

    owner_project_id, _ = _owner_identity(meta)
    with bind_project(owner_project_id):
        await _run_query_in_project(project_id, meta)


async def _run_query_in_project(project_id: str, meta: dict) -> None:
    from remy.core.brain_tools import _add_research_finding, _complete_research
    from remy.core.claim_provenance import record_turn_fetch_evidence

    attempt_id = await asyncio.to_thread(_ensure_execution_attempt, project_id, meta)
    meta = {**meta, "project_id": project_id, "execution_attempt_id": attempt_id}
    attempt_getter = getattr(get_execution_ledger(), "get", None)
    attempt_record = attempt_getter(attempt_id) if callable(attempt_getter) else None
    if attempt_record:
        envelope = (attempt_record.get("metadata") or {}).get("run_envelope")
        if isinstance(envelope, dict):
            meta.update({"run_id": envelope.get("run_id", ""), "run_envelope": envelope})
    plan = list(meta.get("query_plan") or [])
    index = int(meta.get("next_query_index", meta.get("queries_done", 0)) or 0)
    if index >= len(plan):
        await _finish_project(project_id, meta, _complete_research)
        return

    query = str(plan[index])
    context = str(meta.get("context") or "")
    required_urls = re.findall(r"https?://[^\s)\]>'\"]+", context)
    discovery_context = context if index == 0 else ""
    if index > 0 and required_urls:
        host = urlparse(required_urls[0]).hostname or ""
        if host:
            query = f"site:{host} {query}"
    session_id = str(meta.get("session_id") or f"research:{project_id}")
    attempts = dict(meta.get("query_attempts") or {})
    attempt = int(attempts.get(str(index), 0) or 0) + 1
    attempts[str(index)] = attempt
    _update_project(
        project_id,
        job_state="running",
        current_query=query,
        query_attempts=attempts,
        worker_heartbeat=_now(),
        last_error="",
    )
    _checkpoint(
        project_id,
        f"query:{index + 1}:discover",
        "running",
        query,
        job_state="running",
    )
    try:
        await asyncio.to_thread(
            _research_run_step,
            attempt_id,
            f"Research query {index + 1}: {query}",
            signature=f"query:{query}",
        )
        await asyncio.to_thread(
            get_execution_ledger().heartbeat,
            attempt_id,
            {"event": "query_started", "index": index, "query": query},
        )
        sources = await asyncio.to_thread(_discover, query, discovery_context)
        _checkpoint(
            project_id,
            f"query:{index + 1}:fetch",
            "running",
            f"Checking up to {min(len(sources), 3)} candidate source(s).",
        )
        errors: list[str] = []
        stored = None
        used_url = ""
        for source in sources[:3]:
            try:
                from remy.core.cancellation import check_cancelled
                check_cancelled()
                url, _title, evidence = await asyncio.to_thread(_fetch_source, source)
                record_turn_fetch_evidence(
                    session_id, tool="research_supervisor", url=url, title=_title
                )
                from remy.core.memory_write_queue import get_memory_write_queue
                stored = json.loads(await get_memory_write_queue().run(
                    _add_research_finding,
                    {"project_id": project_id, "content": evidence, "source_url": url},
                    session_id,
                    "worker-research-supervisor",
                ))
                if stored.get("stored"):
                    used_url = url
                    break
                errors.append(str(stored.get("error") or "finding was rejected"))
            except Exception as exc:
                errors.append(str(exc))
        if not stored or not stored.get("stored"):
            raise RuntimeError("; ".join(errors[-3:]) or "no fetchable sources found")
        latest = _find_project(project_id)
        pause_requested = bool(latest and (latest.metadata or {}).get("pause_requested"))
        next_state = "paused" if pause_requested else "queued"
        _append_receipt(
            project_id,
            {"at": _now(), "event": "query_completed", "index": index, "query": query, "source_url": used_url},
            job_state=next_state,
            current_query="",
            next_query_index=index + 1,
            queries_done=index + 1,
            queries_succeeded=int(meta.get("queries_succeeded", 0) or 0) + 1,
            worker_heartbeat=_now(),
        )
        _checkpoint(
            project_id,
            "pause_boundary" if pause_requested else "query_queue",
            next_state,
            "Paused after committing the query." if pause_requested else "Query committed; ready for the next step.",
        )
        await asyncio.to_thread(
            get_execution_ledger().append_receipt,
            attempt_id,
            "query_completed",
            {"index": index, "query": query, "source_url": used_url},
        )
    except asyncio.CancelledError:
        _append_receipt(
            project_id,
            {"at": _now(), "event": "interrupted", "index": index, "query": query},
            job_state="queued",
            current_query="",
            execution_attempt_id="",
            worker_heartbeat=_now(),
        )
        await asyncio.to_thread(
            _finish_execution_attempt,
            meta,
            "cancelled",
            error="Research worker stopped before the query completed",
        )
        raise
    except Exception as exc:
        error = str(exc)[:1000]
        latest = _find_project(project_id)
        pause_requested = bool(latest and (latest.metadata or {}).get("pause_requested"))
        if attempt < _MAX_QUERY_ATTEMPTS:
            _append_receipt(
                project_id,
                {"at": _now(), "event": "query_retry", "index": index, "query": query, "attempt": attempt, "error": error},
                job_state="paused" if pause_requested else "retrying",
                current_query="" if pause_requested else query,
                last_error=error,
                worker_heartbeat=_now(),
            )
        else:
            _append_receipt(
                project_id,
                {"at": _now(), "event": "query_failed", "index": index, "query": query, "attempt": attempt, "error": error},
                job_state="queued",
                current_query="",
                next_query_index=index + 1,
                queries_done=index + 1,
                queries_failed=int(meta.get("queries_failed", 0) or 0) + 1,
                last_error=error,
                worker_heartbeat=_now(),
            )
        _checkpoint(
            project_id,
            "pause_boundary" if pause_requested else f"query:{index + 1}:retry",
            "paused" if pause_requested else ("retrying" if attempt < _MAX_QUERY_ATTEMPTS else "queued"),
            "Paused after the query error." if pause_requested else error,
        )


async def _finish_project(project_id: str, meta: dict, complete_fn) -> None:
    from remy.core.notification_router import notify

    if int(meta.get("findings_count", 0) or 0) <= 0:
        reason = str(meta.get("last_error") or "No grounded findings could be collected")
        _append_receipt(
            project_id, {"at": _now(), "event": "failed", "error": reason},
            job_state="failed", status="failed", current_query="", worker_heartbeat=_now(),
        )
        _finish_execution_attempt(meta, "failed", error=reason)
        message = f"Дослідження «{meta.get('topic', project_id)}» завершилося без перевірених джерел: {reason}"
        _queue_continuation(meta, message, status="failed")
        notify(
            message,
            level="warning", event_type="research.failed",
            event_data={
                "project_id": project_id,
                "owner_project_id": meta.get("owner_project_id", ""),
                "brain_id": meta.get("brain_id", ""),
                "topic": meta.get("topic", ""),
                "status": "failed",
                "session_id": meta.get("session_id", ""),
            },
        )
        return
    from remy.core.memory_write_queue import get_memory_write_queue
    _checkpoint(
        project_id,
        "synthesis",
        "running",
        "Building the grounded final report from committed findings.",
        job_state="running",
    )
    result = json.loads(await get_memory_write_queue().run(
        complete_fn,
        {"project_id": project_id},
        str(meta.get("session_id") or f"research:{project_id}"),
        "worker-research-supervisor",
    ))
    if not result.get("completed"):
        error = str(result.get("error") or "report verification failed")
        _append_receipt(
            project_id, {"at": _now(), "event": "completion_blocked", "error": error},
            job_state="blocked", last_error=error, worker_heartbeat=_now(),
        )
        _finish_execution_attempt(meta, "blocked", error=error)
        message = f"Дослідження «{meta.get('topic', project_id)}» потребує уваги: {error}"
        _queue_continuation(meta, message, status="blocked")
        notify(
            message,
            level="warning", event_type="research.failed",
            event_data={
                "project_id": project_id,
                "owner_project_id": meta.get("owner_project_id", ""),
                "brain_id": meta.get("brain_id", ""),
                "topic": meta.get("topic", ""),
                "status": "blocked",
                "error": error,
                "session_id": meta.get("session_id", ""),
            },
        )
        return
    limited = int(meta.get("queries_failed", 0) or 0) > 0
    terminal_state = "completed_with_limits" if limited else "completed"
    _append_receipt(
        project_id, {"at": _now(), "event": terminal_state, "report_id": result.get("report_id")},
        job_state=terminal_state, status="complete", current_query="", notification_sent=True,
        worker_heartbeat=_now(),
    )
    _checkpoint(
        project_id,
        "complete",
        terminal_state,
        "Final report committed and queued for delivery.",
    )
    _finish_execution_attempt(
        meta,
        terminal_state,
        output_ref=f"research-report:{result.get('report_id') or ''}",
    )
    limitation = " Частину запитів не вдалося перевірити; звіт містить явні обмеження." if limited else ""
    message = f"Дослідження «{result.get('topic', project_id)}» завершено.{limitation}\n\n{result.get('markdown') or result.get('report', '')}"
    continuation_id = _queue_continuation(
        meta,
        message,
        status=terminal_state,
        report_id=str(result.get("report_id") or ""),
    )
    notify(
        message,
        event_type="research.complete",
        event_data={
            "project_id": project_id,
            "owner_project_id": meta.get("owner_project_id", ""),
            "brain_id": meta.get("brain_id", ""),
            "topic": result.get("topic", ""),
            "status": terminal_state,
            "report_id": result.get("report_id"), "markdown": result.get("markdown", ""),
            "session_id": meta.get("session_id", ""), "continuation_id": continuation_id,
        },
    )


async def run_research_tick() -> bool:
    rec = await asyncio.to_thread(_next_project)
    if not rec:
        return False
    project_id = str((rec.metadata or {}).get("project_id") or "")
    if not project_id:
        return False
    await _run_query(project_id, dict(rec.metadata or {}))
    return True


async def _worker_loop() -> None:
    recovered = await asyncio.to_thread(_recover_interrupted_projects)
    if recovered:
        logger.info("Recovered %d research project(s)", recovered)
    while _stop_event and not _stop_event.is_set():
        try:
            worked = await run_research_tick()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Research supervisor tick failed")
            worked = False
        if worked:
            await asyncio.sleep(0)
            continue
        if not _wake_event:
            return
        _wake_event.clear()
        try:
            await asyncio.wait_for(_wake_event.wait(), timeout=_IDLE_SECONDS)
        except asyncio.TimeoutError:
            pass


async def start_research_supervisor() -> None:
    global _task, _stop_event, _wake_event, _loop
    if _task and not _task.done():
        return
    _loop = asyncio.get_running_loop()
    _stop_event = asyncio.Event()
    _wake_event = asyncio.Event()
    _task = asyncio.create_task(_worker_loop(), name="research-supervisor")
    logger.info("Research supervisor started")


async def stop_research_supervisor() -> None:
    global _task
    if not _task:
        return
    if _stop_event:
        _stop_event.set()
    if _wake_event:
        _wake_event.set()
    try:
        await asyncio.wait_for(_task, timeout=30)
    except asyncio.TimeoutError:
        _task.cancel()
        await asyncio.gather(_task, return_exceptions=True)
    finally:
        _task = None
    logger.info("Research supervisor stopped")
