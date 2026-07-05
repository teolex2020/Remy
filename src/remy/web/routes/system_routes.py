"""
System routes — unified control-plane status endpoint.

Single source of truth for runtime state: channels, approvals,
active goals, maintenance, budget, browser.
"""

import asyncio
import logging
import time
from contextlib import nullcontext
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from remy.config.settings import set_runtime_setting, settings
from remy.web.routes._helpers import _get_api, run_in_thread, _TIMEOUT_SLOW

logger = logging.getLogger("SystemRoutes")

router = APIRouter()
_MEMORY_STATUS_TTL_SEC = 60.0  # serve stale cache, refresh in background
_memory_status_cache: dict = {"ts": 0.0, "data": None, "refreshing": False}
_last_reconstruction_verification: dict | None = None
_last_harness_eval_runs: dict[str, dict] = {}
_last_startup_recovery_apply: dict = {}
_last_startup_recovery_reconcile: dict = {}
_last_startup_recovery_cleanup: dict = {}


def _safe_runtime_call(fn, default):
    try:
        value = fn()
        return default if value is None else value
    except Exception:
        return default


class PackToggleRequest(BaseModel):
    enabled: bool


class MemoryReconstructRequest(BaseModel):
    candidate_ids: list[str] = []


def _invalidate_memory_status_cache() -> None:
    _memory_status_cache["ts"] = 0.0
    _memory_status_cache["data"] = None


def _harness_eval_dedupe_key(scenario_id: str) -> str:
    return f"harness_eval|{str(scenario_id or '').strip()}"


def _harness_eval_action_target(scenario_id: str) -> str:
    scenario = str(scenario_id or "").strip()
    if scenario == "verify_gate_ablation":
        return "open_memory_verification"
    if scenario == "recovery_replay_ablation":
        return "open_missing_memory_review"
    if scenario == "correction_loop_ablation":
        return "open_correction_review"
    if scenario == "decision_dossier_ablation":
        return "open_decision_dossier"
    return ""


def _emit_harness_eval_alert(result: dict) -> None:
    """Raise an operator-visible alert when a harness eval surfaces a real risk."""
    try:
        from remy.core.notification_router import notify

        scenario_id = str(result.get("id") or "").strip()
        status = str(result.get("status") or "").strip()
        action_target = _harness_eval_action_target(scenario_id)
        message = ""
        level = "warning"

        if status == "not_enough_data":
            message = f"Harness eval {scenario_id} does not have enough recent data."
            level = "info"
        elif scenario_id == "verify_gate_ablation":
            prevented = int(((result.get("delta") or {}).get("prevented_false_successes", 0)) or 0)
            if prevented > 0:
                message = f"Verify-gate eval found {prevented} prevented false success claim(s) that merit review."
        elif scenario_id == "recovery_replay_ablation":
            unresolved = int(((result.get("ablation") or {}).get("assumed_unresolved_missing", 0)) or 0)
            if unresolved > 0:
                message = f"Recovery/replay eval found {unresolved} missing memory candidate(s) that would remain unresolved."
        elif scenario_id == "correction_loop_ablation":
            missed = int(((result.get("ablation") or {}).get("assumed_missed_corrections", 0)) or 0)
            if missed > 0:
                message = f"Correction-loop eval found {missed} correction item(s) that would be missed without repair pressure."
        elif scenario_id == "decision_dossier_ablation":
            missing = int(((result.get("ablation") or {}).get("assumed_missing_snapshots", 0)) or 0)
            goals = int(((result.get("baseline") or {}).get("active_goals", 0)) or 0)
            if missing > 0 and goals > 0:
                message = f"Decision-dossier eval found {missing} review snapshot(s) backing {goals} active goal(s)."

        dedupe_key = _harness_eval_dedupe_key(scenario_id)
        if not message:
            notify(
                f"Harness eval {scenario_id} no longer shows an active operator risk.",
                level="info",
                event_type="harness_eval.resolved",
                event_data={
                    "source": "harness_eval",
                    "scenario_id": scenario_id,
                    "action_target": action_target,
                    "eval_status": status,
                    "resolves": [dedupe_key],
                },
                parse_mode="",
            )
            return

        notify(
            message,
            level=level,
            event_type="operator_alert",
            event_data={
                "source": "harness_eval",
                "scenario_id": scenario_id,
                "action_target": action_target,
                "eval_status": status,
                "dedupe_key": dedupe_key,
            },
            parse_mode="",
        )
    except Exception:
        pass


def _store_operator_artifact(*, api, content: str, tags: list[str], metadata: dict | None = None) -> str:
    """Persist an operator-visible artifact into memory when the brain is writable."""
    if not api or not getattr(api, "brain", None) or not hasattr(api.brain, "store"):
        return ""
    try:
        from remy.core.agent_tools import Level
        from remy.core.provenance import _stamp_provenance

        brain_lock = getattr(api, "brain_lock", None) or nullcontext()
        stamped = _stamp_provenance(metadata or {}, "system", tags=tags)
        with brain_lock:
            rec = api.brain.store(
                content=content,
                level=Level.DECISIONS,
                tags=tags,
                metadata=stamped,
            )
        return str(getattr(rec, "id", "") or "")
    except Exception:
        return ""


def _store_reconstruction_operator_artifact(*, api, stats: dict, label_by_id: dict[str, str] | None = None) -> str:
    applied = int(stats.get("applied", 0) or 0)
    skipped = int(stats.get("skipped", 0) or 0)
    if applied <= 0 and skipped <= 0:
        return ""

    label_by_id = label_by_id or {}
    applied_ids = [str(item).strip() for item in (stats.get("applied_candidate_ids") or []) if str(item).strip()]
    skipped_ids = [str(item).strip() for item in (stats.get("skipped_candidate_ids") or []) if str(item).strip()]
    verification = stats.get("verification") or {}
    dedupe_key = "|".join(
        [
            "reconstruction_review",
            str(int(stats.get("requested", 0) or 0)),
            str(applied),
            str(skipped),
            str(verification.get("status") or "").strip(),
            ",".join(applied_ids),
            ",".join(skipped_ids),
        ]
    )

    if hasattr(api.brain, "search"):
        existing = _safe_runtime_call(
            lambda: api.brain.search(query=dedupe_key, tags=["reconstruction_review"], limit=1),
            [],
        )
        if existing:
            return str(getattr(existing[0], "id", "") or "")

    content = "\n".join(
        [
            "Reconstruction Review Snapshot",
            "",
            f"Requested: {int(stats.get('requested', 0) or 0)}",
            f"Applied: {applied}",
            f"Skipped: {skipped}",
            f"Verification: {str(verification.get('status') or '').strip()}",
            f"Reason: {str(verification.get('reason') or '').strip()}",
            "",
            "Restored:",
            *[f"- {label_by_id.get(item, item)}" for item in applied_ids],
            "",
            "Skipped:",
            *[f"- {label_by_id.get(item, item)}" for item in skipped_ids],
            "",
            f"Dedupe key: {dedupe_key}",
        ]
    ).strip()

    return _store_operator_artifact(
        api=api,
        content=content,
        tags=["operator", "reconstruction_review", "review"],
        metadata={
            "type": "reconstruction_review",
            "requested": int(stats.get("requested", 0) or 0),
            "applied": applied,
            "skipped": skipped,
            "verification": {
                "status": str(verification.get("status") or "").strip(),
                "reason": str(verification.get("reason") or "").strip(),
            },
            "dedupe_key": dedupe_key,
            "applied_labels": [label_by_id.get(item, item) for item in applied_ids],
            "skipped_labels": [label_by_id.get(item, item) for item in skipped_ids],
        },
    )


def _build_startup_recovery_status(startup_status: dict) -> dict:
    backup_path = str((startup_status or {}).get("backup_path") or "").strip()
    preview = {}
    if backup_path:
        try:
            from remy.core.startup_recovery import inspect_backup_recovery

            preview = inspect_backup_recovery(Path(backup_path))
        except Exception as exc:
            preview = {"error": str(exc), "backup_path": backup_path}
    return {
        "available": bool(backup_path),
        "backup_path": backup_path,
        "preview": preview,
        "last_apply": dict(_last_startup_recovery_apply),
        "last_reconcile": dict(_last_startup_recovery_reconcile),
        "last_cleanup": dict(_last_startup_recovery_cleanup),
    }


def _store_startup_recovery_operator_artifact(*, api, title: str, tags: list[str], result: dict) -> str:
    lines = [
        title,
        "",
        f"Backup path: {result.get('backup_path', '')}",
        f"Imported count: {result.get('imported_count', '')}",
        f"Recovered records: {result.get('recovered_records', '')}",
        f"Changes needed: {result.get('changes_needed', '')}",
        f"Cleanup candidates: {result.get('cleanup_candidates', '')}",
        f"Applied: {result.get('applied', '')}",
    ]
    return _store_operator_artifact(
        api=api,
        content="\n".join(str(line) for line in lines).strip(),
        tags=["operator", "incident_snapshot", "review", *tags],
        metadata={
            "type": tags[0] if tags else "startup_recovery",
            "source": "startup",
            "result": result,
        },
    )


def _emit_startup_recovery_alert(*, message: str, status: str, artifact_id: str, result: dict) -> None:
    try:
        from remy.core.notification_router import notify

        notify(
            message,
            level="warning",
            event_type="operator_alert",
            event_data={
                "source": "startup",
                "action_target": "open_memory_verification",
                "artifact_ids": [artifact_id] if artifact_id else [],
                "verification_status": status,
                "backup_path": result.get("backup_path", ""),
                "imported_count": result.get("imported_count"),
                "recovered_records": result.get("recovered_records"),
                "changes_needed": result.get("changes_needed"),
                "cleanup_candidates": result.get("cleanup_candidates"),
            },
            parse_mode="",
        )
    except Exception:
        pass


def _build_verification_memory_summary(api) -> dict:
    recent: list[dict] = []

    def _collect(records, source_type: str) -> None:
        for rec in records or []:
            meta = getattr(rec, "metadata", None) or {}
            verification = meta.get("verification") or {}
            if not isinstance(verification, dict) or not verification.get("status"):
                continue
            recent.append(
                {
                    "record_id": getattr(rec, "id", ""),
                    "label": meta.get("title") or meta.get("topic") or getattr(rec, "content", "")[:80] or source_type,
                    "status": str(verification.get("status") or ""),
                    "reason": str(verification.get("reason") or ""),
                    "source_type": source_type,
                    "artifact_ids": list(verification.get("artifact_ids") or []),
                }
            )

    generated_reports = _safe_runtime_call(
        lambda: api.brain.search(query="", tags=["generated-report"], limit=5),
        [],
    )
    research_reports = _safe_runtime_call(
        lambda: [
            rec for rec in (api.brain.search(query="", tags=["research"], limit=8) or [])
            if ((getattr(rec, "metadata", None) or {}).get("type") == "research_report")
        ],
        [],
    )
    _collect(generated_reports, "generated_report")
    _collect(research_reports, "research_report")

    verified_count = sum(1 for item in recent if item["status"] == "verified")
    repair_required_count = sum(1 for item in recent if item["status"] == "repair_required")
    return {
        "verified_count": verified_count,
        "repair_required_count": repair_required_count,
        "recent": recent[:4],
        "last_reconstruction": _last_reconstruction_verification or {},
    }


def _build_memory_status_sync(api) -> dict:
    from remy.core.history_replay import analyze_history_memory_gaps

    count = api.brain.count()
    salience_summary = _safe_runtime_call(
        lambda: api.brain.salience_summary() if hasattr(api.brain, "salience_summary") else {},
        {},
    )
    high_salience_records = _safe_runtime_call(
        lambda: api.brain.high_salience_records(limit=5) if hasattr(api.brain, "high_salience_records") else [],
        [],
    )
    latest_reflection_digest = _safe_runtime_call(
        lambda: api.brain.latest_reflection_digest() if hasattr(api.brain, "latest_reflection_digest") else None,
        None,
    )
    reflection_digest = latest_reflection_digest or _safe_runtime_call(
        lambda: api.brain.reflection_digest(limit=10) if hasattr(api.brain, "reflection_digest") else {},
        {},
    )
    contradiction_review_queue = _safe_runtime_call(
        lambda: api.brain.contradiction_review_queue(limit=5) if hasattr(api.brain, "contradiction_review_queue") else [],
        [],
    )
    contradiction_clusters = _safe_runtime_call(
        lambda: api.brain.contradiction_clusters(limit=5) if hasattr(api.brain, "contradiction_clusters") else [],
        [],
    )
    instability_summary = _safe_runtime_call(
        lambda: api.brain.belief_instability_summary() if hasattr(api.brain, "belief_instability_summary") else {},
        {},
    )
    correction_review_queue = _safe_runtime_call(
        lambda: api.brain.get_correction_review_queue(limit=5) if hasattr(api.brain, "get_correction_review_queue") else [],
        [],
    )
    suggested_corrections = _safe_runtime_call(
        lambda: api.brain.get_suggested_corrections(limit=5) if hasattr(api.brain, "get_suggested_corrections") else [],
        [],
    )
    recently_corrected = _safe_runtime_call(
        lambda: api.brain.get_recently_corrected_beliefs(limit=5) if hasattr(api.brain, "get_recently_corrected_beliefs") else [],
        [],
    )
    correction_report = _safe_runtime_call(
        lambda: api.brain.get_suggested_corrections_report(limit=5) if hasattr(api.brain, "get_suggested_corrections_report") else {"entries": []},
        {"entries": []},
    )
    history_review = _safe_runtime_call(
        lambda: analyze_history_memory_gaps(
            lambda **search_kwargs: api.brain.search(**search_kwargs),
            history_dir=api.settings.DATA_DIR / "history" if hasattr(api.settings, "DATA_DIR") else settings.DATA_DIR / "history",
            sample_limit=5,
        ),
        {},
    )
    verification = _build_verification_memory_summary(api)
    return {
        "status": "ok",
        "records": count,
        "path": str(api.settings.AURA_BRAIN_PATH),
        "salience": {
            "high_count": int(salience_summary.get("high_salience_count", len(high_salience_records)) or 0),
            "avg_salience": salience_summary.get("avg_salience", 0.0),
            "max_salience": salience_summary.get("max_salience", 0.0),
            "top_records": high_salience_records[:3],
        },
        "reflection": {
            "summary_count": int(
                reflection_digest.get("summary_count")
                or len(reflection_digest.get("top_findings", []) or [])
                or 0
            ),
            "high_severity_count": int(reflection_digest.get("high_severity_count", 0) or 0),
            "top_findings": (reflection_digest.get("top_findings") or [])[:3],
        },
        "contradictions": {
            "review_queue_count": len(contradiction_review_queue),
            "cluster_count": int(
                instability_summary.get("contradiction_cluster_count", len(contradiction_clusters)) or 0
            ),
            "top_review": contradiction_review_queue[:3],
        },
        "corrections": {
            "suggestions_count": int(
                correction_report.get("entry_count")
                or len(correction_report.get("entries", []) or [])
                or len(suggested_corrections)
                or 0
            ),
            "review_queue_count": len(correction_review_queue),
            "recently_corrected_count": len(recently_corrected),
            "top_suggestions": (suggested_corrections or [])[:3],
            "top_review": (correction_review_queue or [])[:3],
            "recently_corrected": (recently_corrected or [])[:3],
        },
        "history_review": {
            "missing_candidates_count": int(history_review.get("missing_candidates_count", 0) or 0),
            "review_candidates_count": int(history_review.get("review_candidates_count", 0) or 0),
            "missing_by_tool": history_review.get("missing_by_tool", {}) or {},
            "recent_missing": (history_review.get("recent_missing") or [])[:3],
            "review_candidates": (history_review.get("review_candidates") or [])[:3],
            "recommended_actions": (history_review.get("recommended_actions") or [])[:3],
        },
        "verification": verification,
    }


async def _refresh_memory_status_bg(api) -> None:
    """Background refresh — updates cache without blocking the caller."""
    if _memory_status_cache.get("refreshing"):
        return
    _memory_status_cache["refreshing"] = True
    try:
        data = await run_in_thread(_build_memory_status_sync, api, timeout=_TIMEOUT_SLOW, error_msg="Memory status refresh timed out")
        _memory_status_cache["ts"] = time.time()
        _memory_status_cache["data"] = data
    except Exception as e:
        logger.warning("Background memory status refresh failed: %s", e)
    finally:
        _memory_status_cache["refreshing"] = False


async def _get_cached_memory_status(api) -> dict:
    now = time.time()
    cached = _memory_status_cache.get("data")
    cached_ts = float(_memory_status_cache.get("ts") or 0.0)
    age = now - cached_ts
    if cached is not None:
        if age < _MEMORY_STATUS_TTL_SEC:
            return cached
        # Serve stale cache immediately, refresh in background
        asyncio.ensure_future(_refresh_memory_status_bg(api))
        return cached
    # No cache at all — must wait for first load
    data = await run_in_thread(_build_memory_status_sync, api, timeout=_TIMEOUT_SLOW, error_msg="Memory status build timed out")
    _memory_status_cache["ts"] = now
    _memory_status_cache["data"] = data
    return data


@router.get("/system/status")
async def build_system_status_payload(*, include_packs: bool = False):
    """Unified runtime status — all channels, goals, approvals, budget in one call."""
    api = _get_api()
    now = time.time()

    result = {
        "uptime_sec": int(now - api._start_time),
        "channels": {},
        "memory": {},
        "harness": {},
        "autonomy": {},
        "approvals": {},
        "operator_alerts": {},
        "budget": {},
        "evaluation": {},
        "factuality": {},
        "browser": {},
        "maintenance": {},
        "improvement": {},
    }

    # --- Channels ---
    from remy.core.combined_runner import get_operator_console_snapshot

    runtime_snapshot = {}
    try:
        runtime_snapshot = get_operator_console_snapshot()
    except Exception:
        runtime_snapshot = {}
    control_state = runtime_snapshot.get("control", {})

    result["channels"] = runtime_snapshot.get("channels", {})
    result["gateway"] = runtime_snapshot.get("gateway", {})

    # --- Harness / runtime contract ---
    try:
        from remy.core.agent_tools import get_brain_startup_status
        from remy.core.failure_taxonomy import (
            classify_startup_incident,
            get_failure_taxonomy_summary,
        )
        from remy.core.harness_ablation import get_harness_ablation_summary
        from remy.core.harness_charter import get_harness_charter_summary
        from remy.core.harness_eval_history import get_harness_eval_history_summary
        from remy.core.harness_eval_matrix import get_harness_eval_matrix_summary
        from remy.core.harness_modules import get_harness_modules_summary
        from remy.core.harness_migration import get_harness_migration_summary
        from remy.core.role_contracts import get_role_contracts_summary
        from remy.core.script_adapter_registry import get_script_adapter_registry_summary
        from remy.core.state_semantics import get_state_semantics_summary
        from remy.core.runtime_contract import get_runtime_contract_summary

        startup_status = _safe_runtime_call(get_brain_startup_status, {})
        incident = classify_startup_incident(startup_status)
        current_role = str((runtime_snapshot.get("autonomy", {}) or {}).get("current_role") or "").strip()
        eval_matrix = get_harness_eval_matrix_summary()
        eval_matrix["latest_runs"] = dict(_last_harness_eval_runs)
        eval_matrix["history"] = get_harness_eval_history_summary()
        result["harness"] = {
            "contract": get_runtime_contract_summary(),
            "charter": get_harness_charter_summary(),
            "modules": get_harness_modules_summary(),
            "migration": get_harness_migration_summary(),
            "role_contracts": get_role_contracts_summary(current_role=current_role),
            "script_adapter_registry": get_script_adapter_registry_summary(),
            "state_semantics": get_state_semantics_summary(),
            "ablation": get_harness_ablation_summary(api=api, runtime_snapshot=runtime_snapshot),
            "eval_matrix": eval_matrix,
            "failure_taxonomy": get_failure_taxonomy_summary(),
            "startup": startup_status,
            "incident": incident.to_dict() if incident else None,
        }
    except Exception as e:
        result["harness"] = {"error": str(e)}

    # --- Runtime autonomy loop state ---
    try:
        result["autonomy"] = runtime_snapshot.get("autonomy", {"running": False, "session_id": None})
        result["approvals"] = runtime_snapshot.get("approvals", {"pending_count": 0, "pending": []})
        result["budget"] = runtime_snapshot.get("budget", {})
        result["evaluation"] = runtime_snapshot.get("evaluation", {})
        result["factuality"] = runtime_snapshot.get("factuality", {})
    except Exception as e:
        result["autonomy"] = {"running": False, "error": str(e)}
        result["approvals"] = {"pending_count": 0, "pending": [], "error": str(e)}
        result["budget"] = {"error": str(e)}
        result["evaluation"] = {"error": str(e)}
        result["factuality"] = {"error": str(e)}

    # --- Memory ---
    try:
        result["memory"] = await _get_cached_memory_status(api)
    except Exception as e:
        result["memory"] = {"status": "error", "error": str(e)}

    try:
        from remy.core.harness_modules import derive_active_harness_module

        if isinstance(result.get("harness"), dict):
            result["harness"]["active_module"] = derive_active_harness_module(
                runtime_snapshot=runtime_snapshot,
                memory_status=result.get("memory", {}),
            )
    except Exception:
        if isinstance(result.get("harness"), dict):
            result["harness"]["active_module"] = {}

    # --- Recent operator alerts ---
    try:
        from remy.core.notification_router import get_recent_notifications

        recent_alerts = get_recent_notifications(event_type="operator_alert", limit=8)
        result["operator_alerts"] = {
            "count": len(recent_alerts),
            "unacknowledged_count": sum(1 for item in recent_alerts if not item.get("acknowledged") and not item.get("resolved")),
            "items": [
                {
                    "id": item.get("id", ""),
                    "type": item.get("type", "operator_alert"),
                    "level": item.get("level", "info"),
                    "message": str(item.get("message", ""))[:280],
                    "timestamp": item.get("timestamp"),
                    "acknowledged": bool(item.get("acknowledged")),
                    "resolved": bool(item.get("resolved")),
                    "resolved_at": item.get("resolved_at"),
                    "repeat_count": int(item.get("repeat_count", 1) or 1),
                    "gateway_health": item.get("gateway_health", ""),
                    "health_level": item.get("health_level", ""),
                    "source": item.get("source", ""),
                    "scenario_id": item.get("scenario_id", ""),
                    "action_target": item.get("action_target", ""),
                    "artifact_ids": list(item.get("artifact_ids") or []),
                    "failure_code": item.get("failure_code", ""),
                    "verification_status": item.get("verification_status", ""),
                    "verification_reason": item.get("verification_reason", ""),
                    "eval_status": item.get("eval_status", ""),
                    "requested": item.get("requested"),
                    "applied": item.get("applied"),
                    "skipped": item.get("skipped"),
                }
                for item in recent_alerts
            ],
        }
    except Exception as e:
        result["operator_alerts"] = {"count": 0, "unacknowledged_count": 0, "items": [], "error": str(e)}

    # --- Browser ---
    try:
        from remy.core.pinchtab_service import _pinchtab_process
        result["browser"] = {
            "running": _pinchtab_process is not None and _pinchtab_process.returncode is None
        }
    except Exception as e:
        result["browser"] = {"running": False, "error": str(e)}

    # --- Maintenance (background brain last run) ---
    try:
        from remy.core import background_brain as _bb
        last = getattr(_bb, "_last_report", None)
        result["maintenance"] = last or {"status": "no data yet"}
    except Exception as e:
        result["maintenance"] = {"error": str(e)}

    # --- Self-improvement / playbooks ---
    try:
        result["improvement"] = runtime_snapshot.get("improvement", {
            "learning": {},
            "playbooks": {},
            "reviewable_insights": [],
            "top_playbooks": [],
        })
    except Exception as e:
        result["improvement"] = {
            "learning": {},
            "playbooks": {},
            "reviewable_insights": [],
            "top_playbooks": [],
            "error": str(e),
        }

    if include_packs:
        result["packs"] = (await build_capability_packs_payload()).get("packs", [])
    return result


@router.get("/system/status")
async def get_system_status():
    return await build_system_status_payload()


@router.post("/system/memory/reconstruct-missing")
async def reconstruct_missing_memory(payload: MemoryReconstructRequest):
    try:
        from remy.core.history_replay import analyze_history_memory_gaps, reconstruct_history_candidates
        from remy.core.verification_gate import emit_verification_incident, resolve_verification_incident

        api = _get_api()
        history_review = {}
        candidate_ids = [str(item).strip() for item in (payload.candidate_ids or []) if str(item).strip()]
        if not candidate_ids:
            history_review = _safe_runtime_call(
                lambda: analyze_history_memory_gaps(
                    lambda **search_kwargs: api.brain.search(**search_kwargs),
                    history_dir=settings.DATA_DIR / "history",
                    sample_limit=5,
                ),
                {},
            )
            candidate_ids = [
                str(item.get("candidate_id", "")).strip()
                for item in (history_review.get("recent_missing") or [])
                if str(item.get("candidate_id", "")).strip()
            ]
        if not candidate_ids:
            return {"ok": False, "error": "No missing memory candidates selected."}

        def _execute(tool: str, tool_args: dict):
            from remy.core.tool_dispatch import execute_tool

            return execute_tool(tool, tool_args, session_id="system-reconstruct", channel="system")

        stats = reconstruct_history_candidates(
            _execute,
            candidate_ids=candidate_ids,
            history_dir=settings.DATA_DIR / "history",
        )
        label_by_id = {
            str(item.get("candidate_id", "")).strip(): str(item.get("label") or item.get("tool") or "Candidate").strip()
            for item in (history_review.get("recent_missing") or [])
            if str(item.get("candidate_id", "")).strip()
        }
        artifact_record_id = _store_reconstruction_operator_artifact(
            api=api,
            stats=stats,
            label_by_id=label_by_id,
        )
        verification = stats.get("verification") or {}
        if isinstance(verification, dict) and verification.get("status") == "repair_required":
            from remy.core.verification_gate import VerificationResult

            emit_verification_incident(
                source="reconstruct_missing_memory",
                verification=VerificationResult(
                    status=str(verification.get("status") or ""),
                    verified=bool(verification.get("verified")),
                    failure_code=str(verification.get("failure_code") or "") or None,
                    reason=str(verification.get("reason") or ""),
                    artifact_ids=list(verification.get("artifact_ids") or []),
                    repair_required=bool(verification.get("repair_required")),
                ),
                artifact_label="missing memory reconstruction",
                extra={
                    "requested": int(stats.get("requested", 0) or 0),
                    "applied": int(stats.get("applied", 0) or 0),
                    "skipped": int(stats.get("skipped", 0) or 0),
                    "artifact_ids": list(verification.get("artifact_ids") or []) + ([artifact_record_id] if artifact_record_id else []),
                },
            )
        elif isinstance(verification, dict) and verification.get("status") == "verified":
            resolve_verification_incident(
                source="reconstruct_missing_memory",
                artifact_label="missing memory reconstruction",
                extra={
                    "requested": int(stats.get("requested", 0) or 0),
                    "applied": int(stats.get("applied", 0) or 0),
                    "skipped": int(stats.get("skipped", 0) or 0),
                    "artifact_ids": ([artifact_record_id] if artifact_record_id else []),
                },
            )
        global _last_reconstruction_verification
        _last_reconstruction_verification = verification
        _invalidate_memory_status_cache()
        return {"ok": True, "stats": stats}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/system/harness/evals/verify-gate/run")
async def run_verify_gate_eval():
    try:
        from remy.core.harness_eval_history import store_harness_eval_run
        from remy.core.harness_eval_matrix import run_verify_gate_ablation_eval

        api = _get_api()
        memory_status = await _get_cached_memory_status(api)
        result = run_verify_gate_ablation_eval(
            memory_verification=memory_status.get("verification") if isinstance(memory_status, dict) else {},
        )
        result["executed_at"] = time.time()
        _last_harness_eval_runs["verify_gate_ablation"] = result
        store_harness_eval_run(result)
        _emit_harness_eval_alert(result)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/system/harness/evals/recovery-replay/run")
async def run_recovery_replay_eval():
    try:
        from remy.core.harness_eval_history import store_harness_eval_run
        from remy.core.harness_eval_matrix import run_recovery_replay_ablation_eval

        api = _get_api()
        memory_status = await _get_cached_memory_status(api)
        verification = (
            memory_status.get("verification")
            if isinstance(memory_status, dict) and isinstance(memory_status.get("verification"), dict)
            else {}
        )
        result = run_recovery_replay_ablation_eval(
            history_review=memory_status.get("history_review") if isinstance(memory_status, dict) else {},
            last_reconstruction=verification.get("last_reconstruction") if isinstance(verification, dict) else {},
        )
        result["executed_at"] = time.time()
        _last_harness_eval_runs["recovery_replay_ablation"] = result
        store_harness_eval_run(result)
        _emit_harness_eval_alert(result)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/system/harness/evals/correction-loop/run")
async def run_correction_loop_eval():
    try:
        from remy.core.harness_eval_history import store_harness_eval_run
        from remy.core.harness_eval_matrix import run_correction_loop_ablation_eval

        api = _get_api()
        memory_status = await _get_cached_memory_status(api)
        result = run_correction_loop_ablation_eval(
            corrections=memory_status.get("corrections") if isinstance(memory_status, dict) else {},
        )
        result["executed_at"] = time.time()
        _last_harness_eval_runs["correction_loop_ablation"] = result
        store_harness_eval_run(result)
        _emit_harness_eval_alert(result)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/system/harness/evals/decision-dossier/run")
async def run_decision_dossier_eval():
    try:
        from remy.core.combined_runner import get_operator_console_snapshot
        from remy.core.harness_eval_history import store_harness_eval_run
        from remy.core.harness_eval_matrix import run_decision_dossier_ablation_eval

        api = _get_api()
        decision_records = _safe_runtime_call(
            lambda: api.brain.search(query="", tags=["decision_dossier"], limit=50),
            [],
        )
        decision_snapshot_count = len(decision_records or [])
        pinned_snapshot_count = sum(
            1
            for rec in (decision_records or [])
            if "pinned_snapshot" in {str(tag) for tag in (getattr(rec, "tags", []) or [])}
        )
        runtime_snapshot = _safe_runtime_call(get_operator_console_snapshot, {})
        autonomy = runtime_snapshot.get("autonomy", {}) if isinstance(runtime_snapshot, dict) else {}
        goals = autonomy.get("goals", {}) if isinstance(autonomy, dict) else {}
        active_goal_count = int(goals.get("active", 0) or len(goals.get("active_list", []) or []))

        result = run_decision_dossier_ablation_eval(
            decision_snapshot_count=decision_snapshot_count,
            pinned_snapshot_count=pinned_snapshot_count,
            active_goal_count=active_goal_count,
        )
        result["executed_at"] = time.time()
        _last_harness_eval_runs["decision_dossier_ablation"] = result
        store_harness_eval_run(result)
        _emit_harness_eval_alert(result)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/system/approvals/{action_id}/approve")
async def approve_action(action_id: str):
    """Approve a pending action."""
    try:
        from remy.core.combined_runner import resolve_operator_approval
        return resolve_operator_approval(action_id, approved=True, decided_by="web")
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/system/approvals/{action_id}/deny")
async def deny_action(action_id: str):
    """Deny a pending action."""
    try:
        from remy.core.combined_runner import resolve_operator_approval
        return resolve_operator_approval(action_id, approved=False, decided_by="web")
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/system/operator-alerts/{alert_id}/ack")
async def acknowledge_operator_alert(alert_id: str):
    """Acknowledge a recent operator alert."""
    try:
        from remy.core.notification_router import acknowledge_notification

        ok = acknowledge_notification(alert_id)
        return {"ok": ok, "alert_id": alert_id}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/system/startup-recovery/status")
async def get_startup_recovery_status():
    try:
        from remy.core.agent_tools import get_brain_startup_status

        startup_status = get_brain_startup_status()
        return {"ok": True, "startup": startup_status, "recovery": _build_startup_recovery_status(startup_status)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/system/startup-recovery/recover")
async def recover_startup_backup():
    try:
        from remy.core.agent_tools import get_brain_startup_status
        from remy.core.startup_recovery import apply_backup_recovery

        startup_status = get_brain_startup_status()
        backup_path = str(startup_status.get("backup_path") or "").strip()
        if not backup_path:
            return {"ok": False, "error": "No startup backup is available."}

        api = _get_api()
        result = apply_backup_recovery(Path(backup_path))
        artifact_id = _store_startup_recovery_operator_artifact(
            api=api,
            title="Startup Backup Recovery Applied",
            tags=["startup_backup_recovery"],
            result=result,
        )
        result["operator_artifact_id"] = artifact_id
        global _last_startup_recovery_apply
        _last_startup_recovery_apply = dict(result)
        _invalidate_memory_status_cache()
        _emit_startup_recovery_alert(
            message="Startup backup recovery was applied.",
            status="backup_recovery_applied",
            artifact_id=artifact_id,
            result=result,
        )
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/system/startup-recovery/reconcile")
async def reconcile_startup_recovery():
    try:
        from remy.core.startup_recovery import reconcile_recovered_records

        api = _get_api()
        result = reconcile_recovered_records(apply=True)
        artifact_id = _store_startup_recovery_operator_artifact(
            api=api,
            title="Startup Backup Recovery Reconciliation Applied",
            tags=["startup_backup_reconciliation"],
            result=result,
        )
        result["operator_artifact_id"] = artifact_id
        global _last_startup_recovery_reconcile
        _last_startup_recovery_reconcile = dict(result)
        _invalidate_memory_status_cache()
        _emit_startup_recovery_alert(
            message="Startup backup recovery reconciliation was applied.",
            status="backup_reconciliation_applied",
            artifact_id=artifact_id,
            result=result,
        )
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/system/startup-recovery/cleanup")
async def cleanup_startup_recovery():
    try:
        from remy.core.startup_recovery import cleanup_recovered_records

        api = _get_api()
        result = cleanup_recovered_records(apply=True)
        artifact_id = _store_startup_recovery_operator_artifact(
            api=api,
            title="Startup Backup Recovery Cleanup Applied",
            tags=["startup_backup_recovery_cleanup"],
            result=result,
        )
        result["operator_artifact_id"] = artifact_id
        global _last_startup_recovery_cleanup
        _last_startup_recovery_cleanup = dict(result)
        _invalidate_memory_status_cache()
        _emit_startup_recovery_alert(
            message="Startup backup recovery cleanup was applied.",
            status="backup_recovery_cleanup_applied",
            artifact_id=artifact_id,
            result=result,
        )
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/system/policy")
async def get_policy():
    """Get current governance policy rules."""
    try:
        from remy.core_v3.governance.policy_engine import PolicyEngine
        engine = PolicyEngine()
        return {"rules": engine.get_rules()}
    except Exception as e:
        return {"rules": [], "error": str(e)}


async def build_capability_packs_payload():
    """List all capability packs with metadata."""
    try:
        from remy.core.capability_packs import get_all_packs, pack_summary
        packs = get_all_packs()
        result = []
        for pack_id, pack in packs.items():
            s = pack_summary(pack)
            result.append({
                "id": pack_id,
                "name": s.get("label") or pack_id,
                "enabled": s.get("enabled", True),
                "approval_mode": s.get("approval_mode", "none"),
                "tools": s.get("tools", []),
                "guardrails": s.get("guardrails", []),
                "metrics_family": getattr(pack, "metrics_family", None),
                "worker": getattr(pack, "worker", None),
                "description": s.get("description", ""),
                "risk_profile": s.get("risk_profile", "unknown"),
                "budget_profile": s.get("budget_profile", "unknown"),
                "tool_scope": s.get("tool_scope", ""),
                "step_budget": s.get("step_budget", 0),
                "timeout_sec": s.get("timeout_sec", 0),
            })
        return {"packs": result}
    except Exception as e:
        return {"packs": [], "error": str(e)}


@router.get("/system/packs")
async def get_capability_packs():
    return await build_capability_packs_payload()


@router.post("/system/packs/{pack_id}")
async def set_capability_pack_state(pack_id: str, body: PackToggleRequest):
    """Enable or disable a capability pack through runtime settings."""
    try:
        from remy.core.capability_packs import get_all_packs, get_disabled_pack_ids

        all_packs = get_all_packs()
        if pack_id not in all_packs:
            return {"ok": False, "error": f"Unknown pack: {pack_id}"}
        if pack_id == "general" and not body.enabled:
            return {"ok": False, "error": "The general pack cannot be disabled."}

        disabled = get_disabled_pack_ids()
        if body.enabled:
            disabled.discard(pack_id)
        else:
            disabled.add(pack_id)
        set_runtime_setting("PACKS_DISABLED", sorted(disabled), target=settings)
        return {
            "ok": True,
            "pack_id": pack_id,
            "enabled": body.enabled,
            "disabled_packs": sorted(disabled),
        }
    except Exception as e:
        logger.exception("Failed to update pack state")
        return {"ok": False, "error": str(e)}
