"""Conversation-scoped Trajectory ledger and Inspector data."""

from __future__ import annotations

import json
import sqlite3
from importlib import metadata as importlib_metadata
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from remy.core.conversation_store import get_conversation_store
from remy.core.microbrain import current_project_id
from remy.core.project_store import LEGACY_PROJECT_ID
from remy.core.trajectory_diagnostics import (
    analyze_project_trajectory,
    analyze_trajectory,
    analyze_trajectory_slo,
    analyze_model_promotion_canary,
    build_model_promotion_recommendation,
    build_incident_dossier,
    derive_incident_eval_criteria,
    evaluate_trajectory_regression,
    render_incident_dossier_markdown,
)
from remy.core.trajectory_store import get_trajectory_store
from remy.core.session_event_store import SessionEventStore
from remy.core.trajectory_replay import run_trajectory_sandbox_replay
from remy.core.transcript_store import get_transcript_store
from remy.web.routes._helpers import _get_api
from remy.config.settings import set_runtime_setting, settings


router = APIRouter()


class TrajectoryAnnotationPayload(BaseModel):
    label: str = Field(default="", max_length=32)
    note: str = Field(default="", max_length=4000)
    bookmarked: bool = False


class TrajectoryForkPayload(BaseModel):
    title: str = Field(default="", max_length=120)
    preferred_model: str = Field(default="", max_length=240)
    next_prompt: str = Field(default="", max_length=20000)
    activate: bool = True


class TrajectoryBaselinePayload(BaseModel):
    name: str = Field(default="", max_length=120)
    days: int = Field(default=30, ge=1, le=365)
    activate: bool = True


class TrajectoryAlertPayload(BaseModel):
    status: str = Field(max_length=32)


class TrajectoryAlertPolicyPayload(BaseModel):
    name: str = Field(default="", max_length=120)
    scope_type: str = Field(default="project", max_length=32)
    scope_value: str = Field(default="*", max_length=240)
    failure_rate_warning: float = Field(default=0, ge=0, le=1)
    failure_rate_critical: float = Field(default=0, ge=0, le=1)
    latency_warning_ms: float = Field(default=0, ge=0, le=86_400_000)
    latency_critical_ms: float = Field(default=0, ge=0, le=86_400_000)
    tokens_warning: float = Field(default=0, ge=0, le=100_000_000)
    tokens_critical: float = Field(default=0, ge=0, le=100_000_000)
    enabled: bool = True


class TrajectorySloPayload(BaseModel):
    target_success_rate: float = Field(default=0.99, ge=0.5, lt=1)
    window_days: int = Field(default=30, ge=1, le=365)
    min_operations: int = Field(default=5, ge=1, le=100_000)


class TrajectoryEvalCasePayload(BaseModel):
    name: str = Field(default="", max_length=160)


class TrajectoryEvalRunPayload(BaseModel):
    conversation_id: str = Field(default="", max_length=160)


class TrajectoryEvalMatrixPayload(BaseModel):
    name: str = Field(default="", max_length=160)
    preferred_model: str = Field(default="", max_length=240)
    case_ids: list[str] = Field(default_factory=list, max_length=25)


class TrajectoryEvalComparisonPayload(BaseModel):
    name: str = Field(default="", max_length=160)
    models: list[str] = Field(default_factory=list, max_length=4)
    case_ids: list[str] = Field(default_factory=list, max_length=25)


class TrajectoryModelPromotionPayload(BaseModel):
    candidate_model: str = Field(max_length=240)
    confirm_model: str = Field(max_length=240)
    canary_percent: int = Field(default=10, ge=5, le=50)


class TrajectoryModelPromotionActionPayload(BaseModel):
    confirm_model: str = Field(max_length=240)


def _policy_thresholds(payload: TrajectoryAlertPolicyPayload) -> dict[str, float]:
    return {
        "failure_rate_warning": payload.failure_rate_warning,
        "failure_rate_critical": payload.failure_rate_critical,
        "latency_warning_ms": payload.latency_warning_ms,
        "latency_critical_ms": payload.latency_critical_ms,
        "tokens_warning": payload.tokens_warning,
        "tokens_critical": payload.tokens_critical,
    }


def _runtime_agent_version() -> str:
    try:
        return importlib_metadata.version("remy")
    except importlib_metadata.PackageNotFoundError:
        try:
            from remy import __version__

            return str(__version__)
        except Exception:
            return "unknown"


def _timestamp(value: str) -> float | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _fork_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict) and "content" in value:
        return _fork_text(value.get("content"))
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value).strip()


def _incident_dossier_data(*, project_id: str, incident_id: str) -> tuple[dict, str, list]:
    trajectory_store = get_trajectory_store()
    regression = next(
        (
            row for row in trajectory_store.list_regression_alerts(
                project_id=project_id, limit=500
            )
            if row["alert_id"] == incident_id
        ),
        None,
    )
    slo_incident = None
    if regression is None:
        slo_incident = next(
            (
                row for row in trajectory_store.list_slo_incidents(
                    project_id=project_id, limit=500
                )
                if row["incident_id"] == incident_id
            ),
            None,
        )
    if regression is None and slo_incident is None:
        raise KeyError(incident_id)
    incident = dict(regression or slo_incident or {})
    if slo_incident is not None:
        incident["source_type"] = "slo"
        incident["metric"] = "slo.burn_rate"
        incident["threshold"] = 1.0
    conversation_id = str(incident.get("conversation_id") or "")
    records = trajectory_store.list_events(
        project_id=project_id,
        session_id=conversation_id,
        limit=5000,
    ) if conversation_id else []
    dossier = build_incident_dossier(records, incident=incident)
    return dossier, render_incident_dossier_markdown(dossier), records


def _fork_dialogue(
    records: list[dict], boundary_event_id: str
) -> tuple[list[dict], str, dict]:
    """Project a causal prefix into completed chat pairs plus a pending prompt."""
    ordered = sorted(records, key=lambda row: int(row.get("sequence") or 0))
    boundary_index = next(
        (
            index
            for index, row in enumerate(ordered)
            if str(row.get("event_id") or "") == boundary_event_id
        ),
        -1,
    )
    if boundary_index < 0:
        raise KeyError(boundary_event_id)
    prefix = ordered[: boundary_index + 1]
    boundary = prefix[-1]
    boundary_turn_id = str(boundary.get("turn_id") or "")

    grouped: dict[str, list[dict]] = {}
    for index, row in enumerate(prefix):
        turn_id = str(row.get("turn_id") or f"event-{index}")
        grouped.setdefault(turn_id, []).append(row)

    dialogue: list[dict] = []
    pending_prompt = ""
    for turn_id, rows in grouped.items():
        user = next((row for row in rows if row.get("kind") == "USER"), None)
        assistants = [row for row in rows if row.get("kind") == "ASSISTANT"]
        user_text = _fork_text((user or {}).get("input") or (user or {}).get("output"))
        assistant = assistants[-1] if assistants else None
        assistant_text = _fork_text((assistant or {}).get("output"))
        if user_text and assistant_text:
            dialogue.extend((
                {
                    "role": "user",
                    "content": user_text,
                    "source_event_id": str(user.get("event_id") or ""),
                    "source_turn_id": turn_id,
                },
                {
                    "role": "assistant",
                    "content": assistant_text,
                    "source_event_id": str(assistant.get("event_id") or ""),
                    "source_turn_id": turn_id,
                },
            ))
        elif turn_id == boundary_turn_id and user_text:
            # A half-finished turn is not inserted into history twice. The UI
            # receives it as an editable prompt and the operator decides when
            # to execute it in the new branch.
            pending_prompt = user_text

    return dialogue, pending_prompt, boundary


def _legacy_projection(project_id: str, conversation_id: str, limit: int) -> list[dict]:
    rows = get_transcript_store().list_session(
        conversation_id,
        owner_project_id=project_id,
        include_legacy_unscoped=project_id == LEGACY_PROJECT_ID,
        limit=limit,
    )
    result = []
    turn_number = 0
    turn_id = ""
    for sequence, row in enumerate(rows, start=1):
        role = str(row.get("role") or "").lower()
        if role == "user" or not turn_id:
            turn_number += 1
            turn_id = f"legacy-turn-{turn_number}"
        kind = {
            "user": "USER",
            "assistant": "ASSISTANT",
            "system": "CONTEXT",
            "tool": "TOOL",
        }.get(role, "CONTEXT")
        content = row.get("content") or ""
        created = _timestamp(row.get("created_at", ""))
        result.append({
            "sequence": sequence,
            "event_id": row.get("message_id") or f"legacy-{sequence}",
            "session_id": conversation_id,
            "project_id": project_id,
            "turn_id": turn_id,
            "step_id": "",
            "request_id": "",
            "call_id": "",
            "parent_id": "",
            "kind": kind,
            "status": "completed",
            "source": {
                "kind": "transcript",
                "message_type": row.get("message_type", "text"),
                "metadata": row.get("metadata") or {},
                "trust_tier": "user" if role == "user" else "recorded-output",
            },
            "input": content if kind == "USER" else None,
            "output": content,
            "schema": None,
            "details": {
                "preview": " ".join(str(content).split())[:260],
                "legacy_projection": True,
            },
            "started_at": created,
            "first_output_at": created,
            "completed_at": created,
            "duration_ms": 0,
            "error": "",
            "created_at": row.get("created_at", ""),
            "started_at_iso": row.get("created_at", ""),
            "completed_at_iso": row.get("created_at", ""),
        })
    return result


def _summary(records: list[dict]) -> dict:
    turns = {row.get("turn_id") for row in records if row.get("turn_id")}
    requests = [row for row in records if row.get("kind") == "REQUEST"]
    tools = [row for row in records if row.get("kind") in {"TOOL", "SUBTOOL"}]
    failures = [row for row in records if row.get("status") == "failed" or row.get("error")]
    timestamps = [
        float(value)
        for row in records
        for value in (row.get("started_at"), row.get("completed_at"))
        if value is not None
    ]
    return {
        "records": len(records),
        "turns": len(turns),
        "requests": len(requests),
        "tool_calls": len(tools),
        "failures": len(failures),
        "duration_ms": (
            max(0, int((max(timestamps) - min(timestamps)) * 1000))
            if len(timestamps) >= 2 else 0
        ),
    }


def _turn_summaries(records: list[dict]) -> list[dict]:
    """Project the causal ledger into compact replay/focus metadata."""
    grouped: dict[str, list[dict]] = {}
    for row in records:
        turn_id = str(row.get("turn_id") or "")
        if turn_id:
            grouped.setdefault(turn_id, []).append(row)

    result = []
    for index, (turn_id, rows) in enumerate(grouped.items(), start=1):
        meaningful = [row for row in rows if row.get("kind") != "ATTEMPT"]
        observed = meaningful or rows
        has_failed = any(row.get("status") == "failed" or row.get("error") for row in observed)
        has_running = any(row.get("status") == "running" for row in observed)
        timestamps = [
            float(value)
            for row in rows
            for value in (row.get("started_at"), row.get("completed_at"))
            if value is not None
        ]
        requests = [row for row in rows if row.get("kind") == "REQUEST"]
        tools = [row for row in rows if row.get("kind") in {"TOOL", "SUBTOOL"}]
        total_tokens = sum(
            int(((row.get("details") or {}).get("usage") or {}).get("total_tokens") or 0)
            for row in requests
        )

        def preview(kind: str) -> str:
            row = next((item for item in rows if item.get("kind") == kind), None)
            if not row:
                return ""
            details = row.get("details") if isinstance(row.get("details"), dict) else {}
            value = details.get("preview") or row.get("output") or row.get("input") or ""
            return " ".join(str(value).split())[:180]

        result.append({
            "turn_id": turn_id,
            "index": index,
            "status": "failed" if has_failed else "running" if has_running else "completed",
            "started_at": min(timestamps) if timestamps else None,
            "completed_at": max(timestamps) if timestamps else None,
            "duration_ms": (
                max(0, int((max(timestamps) - min(timestamps)) * 1000))
                if len(timestamps) >= 2 else 0
            ),
            "record_count": len(rows),
            "request_count": len(requests),
            "tool_count": len(tools),
            "attempt_count": sum(1 for row in rows if row.get("kind") == "ATTEMPT"),
            "failure_count": sum(
                1 for row in rows if row.get("status") == "failed" or row.get("error")
            ),
            "error_count": sum(
                1 for row in rows if (row.get("diagnostic") or {}).get("severity") == "error"
            ),
            "warning_count": sum(
                1 for row in rows if (row.get("diagnostic") or {}).get("severity") == "warning"
            ),
            "total_tokens": total_tokens,
            "user_preview": preview("USER"),
            "assistant_preview": preview("ASSISTANT"),
        })
    return result


@router.get("/trajectory/analytics")
async def get_project_trajectory_analytics(
    days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=20_000, ge=1, le=25_000),
):
    """Return project-scoped, aggregate-only execution analytics."""
    project_id = current_project_id()
    conversation_store = get_conversation_store(project_id)
    titles = {
        record.conversation_id: record.title
        for record in conversation_store.list(include_archived=True)
    }
    trajectory_store = get_trajectory_store()
    records = trajectory_store.list_project_events(project_id=project_id, limit=limit)
    active_baseline = trajectory_store.get_active_analytics_baseline(project_id=project_id)
    count_project_events = getattr(trajectory_store, "count_project_events", None)
    stored_total = (
        int(count_project_events(project_id=project_id))
        if callable(count_project_events) else len(records)
    )
    analytics = analyze_project_trajectory(
        records,
        conversation_titles=titles,
        days=days,
        baseline_metrics=(active_baseline or {}).get("metrics") if active_baseline else None,
    )
    baselines = trajectory_store.list_analytics_baselines(project_id=project_id)
    policies = trajectory_store.list_alert_policies(project_id=project_id)
    policies_by_id = {row["policy_id"]: row for row in policies}
    alerts = trajectory_store.list_regression_alerts(project_id=project_id, limit=100)
    for alert in alerts:
        alert["title"] = titles.get(alert["conversation_id"], "Conversation")
        alert["policy"] = policies_by_id.get(alert["baseline_id"])
    slo_incidents = trajectory_store.list_slo_incidents(project_id=project_id, limit=100)
    for incident in slo_incidents:
        incident["title"] = titles.get(incident["conversation_id"], "Conversation")
    slo_config = trajectory_store.get_slo_config(project_id=project_id)
    slo = analyze_trajectory_slo(records, **{
        "target_success_rate": slo_config["target_success_rate"],
        "window_days": slo_config["window_days"],
        "min_operations": slo_config["min_operations"],
    })
    alert_history = trajectory_store.list_alert_history(project_id=project_id, limit=100)
    for item in alert_history:
        conversation_id = str((item.get("details") or {}).get("conversation_id") or "")
        item["title"] = titles.get(conversation_id, "Conversation") if conversation_id else ""
    eval_cases = trajectory_store.list_eval_cases(project_id=project_id, limit=100)
    eval_matrices = trajectory_store.list_eval_matrices(project_id=project_id, limit=20)
    eval_comparisons = trajectory_store.list_eval_comparisons(
        project_id=project_id, limit=20
    )
    promotion_recommendation = build_model_promotion_recommendation(
        eval_comparisons,
        current_model=str(settings.SUMMARY_MODEL or ""),
    )
    model_promotions = trajectory_store.list_model_promotions(
        project_id=project_id, limit=20
    )
    telemetry_promotion = model_promotions[0] if model_promotions else None
    canary_telemetry = analyze_model_promotion_canary(
        records,
        promotion=telemetry_promotion,
    )
    analytics.update({
        "project_id": project_id,
        "days": days,
        "active_baseline": active_baseline,
        "baselines": baselines,
        "policies": policies,
        "alerts": alerts,
        "slo_incidents": slo_incidents,
        "alert_history": alert_history,
        "slo": slo,
        "slo_config": slo_config,
        "eval_cases": eval_cases,
        "eval_matrices": eval_matrices,
        "eval_comparisons": eval_comparisons,
        "promotion_recommendation": promotion_recommendation,
        "model_promotions": model_promotions,
        "active_model_promotion": model_promotions[0]
        if model_promotions and model_promotions[0]["status"] in {
            "canary", "ready", "promoted", "rollback_pending"
        }
        else None,
        "canary_telemetry": canary_telemetry,
        "pagination": {
            "limit": limit,
            "returned": len(records),
            "estimated_total": max(stored_total, len(records)),
            "window_truncated": stored_total > len(records),
        },
    })
    return analytics


@router.post("/trajectory/analytics/baselines")
async def create_project_trajectory_baseline(payload: TrajectoryBaselinePayload):
    project_id = current_project_id()
    trajectory_store = get_trajectory_store()
    records = trajectory_store.list_project_events(project_id=project_id, limit=25_000)
    analytics = analyze_project_trajectory(records, days=payload.days)
    if not analytics["summary"].get("requests"):
        raise HTTPException(
            status_code=422,
            detail="A baseline requires at least one recorded model request",
        )
    baseline = trajectory_store.create_analytics_baseline(
        project_id=project_id,
        name=payload.name,
        days=payload.days,
        metrics=analytics["window_baseline"],
        summary=analytics["summary"],
        source_event_count=len(records),
        activate=payload.activate,
    )
    return {"baseline": baseline}


@router.get("/trajectory/analytics/baselines")
async def list_project_trajectory_baselines():
    project_id = current_project_id()
    return {
        "project_id": project_id,
        "baselines": get_trajectory_store().list_analytics_baselines(project_id=project_id),
    }


@router.put("/trajectory/analytics/baselines/window-median")
async def use_project_trajectory_window_median():
    project_id = current_project_id()
    get_trajectory_store().deactivate_analytics_baseline(project_id=project_id)
    return {"active_baseline": None}


@router.put("/trajectory/analytics/baselines/{baseline_id}/activate")
async def activate_project_trajectory_baseline(baseline_id: str):
    project_id = current_project_id()
    try:
        baseline = get_trajectory_store().activate_analytics_baseline(
            project_id=project_id,
            baseline_id=baseline_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Trajectory baseline not found") from exc
    return {"baseline": baseline}


@router.delete("/trajectory/analytics/baselines/{baseline_id}")
async def delete_project_trajectory_baseline(baseline_id: str):
    project_id = current_project_id()
    deleted = get_trajectory_store().delete_analytics_baseline(
        project_id=project_id,
        baseline_id=baseline_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Trajectory baseline not found")
    return {"deleted": True, "baseline_id": baseline_id}


@router.get("/trajectory/analytics/alerts")
async def list_project_trajectory_alerts(
    status: str = Query(default="", max_length=32),
    limit: int = Query(default=100, ge=1, le=500),
):
    project_id = current_project_id()
    allowed_status = {"", "open", "acknowledged", "resolved"}
    if status not in allowed_status:
        raise HTTPException(status_code=422, detail="Unsupported regression alert status")
    titles = {
        record.conversation_id: record.title
        for record in get_conversation_store(project_id).list(include_archived=True)
    }
    trajectory_store = get_trajectory_store()
    policies = trajectory_store.list_alert_policies(project_id=project_id)
    policies_by_id = {row["policy_id"]: row for row in policies}
    alerts = trajectory_store.list_regression_alerts(
        project_id=project_id,
        status=status,
        limit=limit,
    )
    for alert in alerts:
        alert["title"] = titles.get(alert["conversation_id"], "Conversation")
        alert["policy"] = policies_by_id.get(alert["baseline_id"])
    return {"project_id": project_id, "alerts": alerts}


@router.patch("/trajectory/analytics/alerts/{alert_id}")
async def update_project_trajectory_alert(
    alert_id: str,
    payload: TrajectoryAlertPayload,
):
    project_id = current_project_id()
    try:
        alert = get_trajectory_store().update_regression_alert(
            project_id=project_id,
            alert_id=alert_id,
            status=payload.status,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Regression alert not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"alert": alert}


@router.get("/trajectory/analytics/alert-history")
async def list_project_trajectory_alert_history(
    alert_id: str = Query(default="", max_length=80),
    limit: int = Query(default=200, ge=1, le=1000),
):
    project_id = current_project_id()
    return {
        "project_id": project_id,
        "history": get_trajectory_store().list_alert_history(
            project_id=project_id,
            alert_id=alert_id,
            limit=limit,
        ),
    }


@router.put("/trajectory/analytics/slo")
async def update_project_trajectory_slo(payload: TrajectorySloPayload):
    project_id = current_project_id()
    try:
        config = get_trajectory_store().update_slo_config(
            project_id=project_id,
            target_success_rate=payload.target_success_rate,
            window_days=payload.window_days,
            min_operations=payload.min_operations,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"slo_config": config}


@router.get("/trajectory/analytics/slo/incidents")
async def list_project_trajectory_slo_incidents(
    status: str = Query(default="", max_length=32),
    limit: int = Query(default=100, ge=1, le=500),
):
    if status not in {"", "open", "acknowledged", "resolved"}:
        raise HTTPException(status_code=422, detail="Unsupported SLO incident status")
    project_id = current_project_id()
    return {
        "project_id": project_id,
        "incidents": get_trajectory_store().list_slo_incidents(
            project_id=project_id,
            status=status,
            limit=limit,
        ),
    }


@router.patch("/trajectory/analytics/slo/incidents/{incident_id}")
async def update_project_trajectory_slo_incident(
    incident_id: str,
    payload: TrajectoryAlertPayload,
):
    project_id = current_project_id()
    try:
        incident = get_trajectory_store().update_slo_incident(
            project_id=project_id,
            incident_id=incident_id,
            status=payload.status,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="SLO incident not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"incident": incident}


@router.get("/trajectory/analytics/incidents/{incident_id}/dossier")
async def get_project_trajectory_incident_dossier(incident_id: str):
    project_id = current_project_id()
    try:
        dossier, markdown, _ = _incident_dossier_data(
            project_id=project_id, incident_id=incident_id
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Trajectory incident not found")
    return {
        "project_id": project_id,
        "dossier": dossier,
        "markdown": markdown,
    }


@router.post("/trajectory/analytics/incidents/{incident_id}/eval-cases")
async def create_project_trajectory_eval_case(
    incident_id: str,
    payload: TrajectoryEvalCasePayload,
):
    project_id = current_project_id()
    try:
        dossier, _, records = _incident_dossier_data(
            project_id=project_id, incident_id=incident_id
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Trajectory incident not found")
    trajectory_store = get_trajectory_store()
    slo_config = trajectory_store.get_slo_config(project_id=project_id)
    criteria = derive_incident_eval_criteria(
        dossier,
        slo_target_success_rate=slo_config["target_success_rate"],
    )
    baseline = evaluate_trajectory_regression(records, criteria=criteria)
    incident = dossier.get("incident") or {}
    try:
        case = trajectory_store.create_eval_case(
            project_id=project_id,
            incident_id=incident_id,
            name=payload.name or f"Regression: {incident.get('metric') or 'trajectory incident'}",
            source_conversation_id=str(incident.get("conversation_id") or ""),
            criteria=criteria,
            baseline_snapshot=baseline,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"case": case}


@router.get("/trajectory/analytics/eval-cases")
async def list_project_trajectory_eval_cases(
    limit: int = Query(default=100, ge=1, le=500),
):
    project_id = current_project_id()
    return {
        "project_id": project_id,
        "cases": get_trajectory_store().list_eval_cases(
            project_id=project_id, limit=limit
        ),
    }


@router.post("/trajectory/analytics/eval-cases/{case_id}/runs")
async def run_project_trajectory_eval_case(
    case_id: str,
    payload: TrajectoryEvalRunPayload,
):
    project_id = current_project_id()
    trajectory_store = get_trajectory_store()
    try:
        case = trajectory_store.get_eval_case(project_id=project_id, case_id=case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Trajectory eval case not found") from exc
    conversation_id = str(payload.conversation_id or case["source_conversation_id"])
    try:
        conversation = get_conversation_store(project_id).require(conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Candidate conversation not found") from exc
    if str(conversation.project_id or "") != project_id:
        raise HTTPException(status_code=404, detail="Candidate conversation not found")
    records = trajectory_store.list_events(
        project_id=project_id,
        session_id=conversation_id,
        limit=5000,
    )
    evaluation = evaluate_trajectory_regression(
        records,
        criteria=case["criteria"],
        baseline_snapshot=case["baseline_snapshot"],
    )
    run = trajectory_store.record_eval_run(
        project_id=project_id,
        case_id=case_id,
        candidate_conversation_id=conversation_id,
        evaluation=evaluation,
    )
    return {"case_id": case_id, "run": run}


@router.post("/trajectory/analytics/eval-cases/{case_id}/sandbox-replay")
async def sandbox_replay_project_trajectory_eval_case(case_id: str):
    """Run the incident turn on the current agent with all tools intercepted."""
    project_id = current_project_id()
    trajectory_store = get_trajectory_store()
    try:
        case = trajectory_store.get_eval_case(project_id=project_id, case_id=case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Trajectory eval case not found") from exc
    try:
        dossier, _, source_records = _incident_dossier_data(
            project_id=project_id, incident_id=case["incident_id"]
        )
        replay_result = await run_trajectory_sandbox_replay(
            project_id=project_id,
            case=case,
            source_records=source_records,
            selected_event=dossier.get("selected_event") or {},
            trajectory_store=trajectory_store,
            conversation_store=get_conversation_store(project_id),
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=409,
            detail="The source incident is no longer available for sandbox replay",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    evaluation = evaluate_trajectory_regression(
        replay_result["records"],
        criteria={**case["criteria"], "max_sandbox_blocked": 0},
        baseline_snapshot=case["baseline_snapshot"],
    )
    conversation = replay_result["conversation"]
    run = trajectory_store.record_eval_run(
        project_id=project_id,
        case_id=case_id,
        candidate_conversation_id=str(conversation.conversation_id),
        evaluation=evaluation,
        mode="sandbox-replay",
        replay=replay_result["replay"],
    )
    return {
        "case_id": case_id,
        "run": run,
        "replay": run["replay"],
        "conversation": conversation.to_dict(),
    }


@router.get("/trajectory/analytics/eval-cases/{case_id}/runs")
async def list_project_trajectory_eval_runs(
    case_id: str,
    limit: int = Query(default=50, ge=1, le=500),
):
    project_id = current_project_id()
    try:
        get_trajectory_store().get_eval_case(project_id=project_id, case_id=case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Trajectory eval case not found") from exc
    return {
        "project_id": project_id,
        "case_id": case_id,
        "runs": get_trajectory_store().list_eval_runs(
            project_id=project_id, case_id=case_id, limit=limit
        ),
    }


@router.post("/trajectory/analytics/eval-matrices")
async def run_project_trajectory_eval_matrix(payload: TrajectoryEvalMatrixPayload):
    """Run selected regression cases sequentially and return a durable release gate."""
    project_id = current_project_id()
    trajectory_store = get_trajectory_store()
    cases = _selected_eval_cases(
        trajectory_store=trajectory_store,
        project_id=project_id,
        case_ids=payload.case_ids,
    )
    preferred_model = str(payload.preferred_model or settings.SUMMARY_MODEL or "")[:240]
    matrix = await _run_eval_matrix(
        project_id=project_id,
        trajectory_store=trajectory_store,
        cases=cases,
        name=payload.name or "Trajectory release gate",
        preferred_model=preferred_model,
    )
    return {"matrix": matrix}


def _selected_eval_cases(
    *,
    trajectory_store,
    project_id: str,
    case_ids: list[str],
) -> list[dict]:
    cases = [
        case for case in trajectory_store.list_eval_cases(project_id=project_id, limit=500)
        if case.get("enabled")
    ]
    requested = {str(case_id) for case_id in case_ids if str(case_id)}
    if requested:
        cases = [case for case in cases if case["case_id"] in requested]
        missing = requested - {case["case_id"] for case in cases}
        if missing:
            raise HTTPException(status_code=404, detail="One or more eval cases were not found")
    if not cases:
        raise HTTPException(status_code=422, detail="No enabled regression eval cases")
    return cases[:25]


async def _run_eval_matrix(
    *,
    project_id: str,
    trajectory_store,
    cases: list[dict],
    name: str,
    preferred_model: str,
    agent_version: str = "",
) -> dict:
    matrix = trajectory_store.create_eval_matrix(
        project_id=project_id,
        name=name,
        preferred_model=preferred_model,
        agent_version=str(agent_version or _runtime_agent_version())[:80],
        case_count=len(cases),
    )
    conversation_store = get_conversation_store(project_id)
    for case in cases:
        try:
            dossier, _, source_records = _incident_dossier_data(
                project_id=project_id, incident_id=case["incident_id"]
            )
            replay_result = await run_trajectory_sandbox_replay(
                project_id=project_id,
                case=case,
                source_records=source_records,
                selected_event=dossier.get("selected_event") or {},
                trajectory_store=trajectory_store,
                conversation_store=conversation_store,
                preferred_model=preferred_model,
            )
            evaluation = evaluate_trajectory_regression(
                replay_result["records"],
                criteria={**case["criteria"], "max_sandbox_blocked": 0},
                baseline_snapshot=case["baseline_snapshot"],
            )
            conversation = replay_result["conversation"]
            run = trajectory_store.record_eval_run(
                project_id=project_id,
                case_id=case["case_id"],
                candidate_conversation_id=str(conversation.conversation_id),
                evaluation=evaluation,
                mode="sandbox-replay",
                replay=replay_result["replay"],
            )
            execution_error = str(
                (replay_result.get("replay") or {}).get("execution_error_type") or ""
            )
            metrics = evaluation.get("metrics") or {}
            trajectory_store.record_eval_matrix_entry(
                project_id=project_id,
                matrix_id=matrix["matrix_id"],
                case_id=case["case_id"],
                run_id=run["run_id"],
                candidate_conversation_id=str(conversation.conversation_id),
                status="error" if execution_error else evaluation["status"],
                score=evaluation["score"],
                failure_rate=metrics.get("failure_rate") or 0,
                avg_request_ms=metrics.get("avg_request_ms") or 0,
                tokens_per_request=metrics.get("tokens_per_request") or 0,
                error_type=execution_error,
            )
        except Exception as exc:
            trajectory_store.record_eval_matrix_entry(
                project_id=project_id,
                matrix_id=matrix["matrix_id"],
                case_id=case["case_id"],
                status="error",
                error_type=type(exc).__name__,
            )
    return trajectory_store.complete_eval_matrix(
        project_id=project_id, matrix_id=matrix["matrix_id"]
    )


@router.get("/trajectory/analytics/eval-matrices")
async def list_project_trajectory_eval_matrices(
    limit: int = Query(default=20, ge=1, le=100),
):
    project_id = current_project_id()
    return {
        "project_id": project_id,
        "matrices": get_trajectory_store().list_eval_matrices(
            project_id=project_id, limit=limit
        ),
    }


@router.post("/trajectory/analytics/eval-comparisons")
async def run_project_trajectory_eval_comparison(payload: TrajectoryEvalComparisonPayload):
    """Compare the same sandbox replay suite across bounded model candidates."""
    project_id = current_project_id()
    trajectory_store = get_trajectory_store()
    models = []
    seen = set()
    for value in payload.models:
        model = str(value or "").strip()[:240]
        if model and model not in seen:
            models.append(model)
            seen.add(model)
    if len(models) < 2:
        raise HTTPException(status_code=422, detail="Choose at least two unique models")
    cases = _selected_eval_cases(
        trajectory_store=trajectory_store,
        project_id=project_id,
        case_ids=payload.case_ids,
    )
    if len(models) * len(cases) > 50:
        raise HTTPException(status_code=422, detail="Model comparison is limited to 50 replays")
    name = payload.name or "Trajectory model comparison"
    comparison = trajectory_store.create_eval_comparison(
        project_id=project_id,
        name=name,
        agent_version=_runtime_agent_version(),
        model_count=len(models),
        case_count=len(cases),
    )
    for model in models:
        matrix = await _run_eval_matrix(
            project_id=project_id,
            trajectory_store=trajectory_store,
            cases=cases,
            name=f"{name} · {model}",
            preferred_model=model,
        )
        trajectory_store.record_eval_comparison_model(
            project_id=project_id,
            comparison_id=comparison["comparison_id"],
            matrix=matrix,
        )
    completed = trajectory_store.complete_eval_comparison(
        project_id=project_id,
        comparison_id=comparison["comparison_id"],
    )
    active_before = trajectory_store.get_active_model_promotion(project_id=project_id)
    promotion = trajectory_store.observe_model_promotion_comparison(
        project_id=project_id,
        comparison=completed,
    )
    rollback_pending = bool(
        active_before
        and active_before.get("status") in {"promoted", "rollback_pending"}
        and promotion
        and promotion.get("status") == "rollback_pending"
    )
    auto_rollback = False
    if rollback_pending:
        set_runtime_setting(
            "SUMMARY_MODEL",
            active_before["previous_model"],
            target=settings,
        )
        promotion = trajectory_store.complete_model_promotion_auto_rollback(
            project_id=project_id,
            promotion_id=active_before["promotion_id"],
        )
        auto_rollback = True
    return {
        "comparison": completed,
        "promotion": promotion,
        "auto_rollback": auto_rollback,
    }


@router.get("/trajectory/analytics/eval-comparisons")
async def list_project_trajectory_eval_comparisons(
    limit: int = Query(default=20, ge=1, le=100),
):
    project_id = current_project_id()
    return {
        "project_id": project_id,
        "comparisons": get_trajectory_store().list_eval_comparisons(
            project_id=project_id, limit=limit
        ),
    }


@router.post("/trajectory/analytics/model-promotions")
async def start_project_model_promotion(payload: TrajectoryModelPromotionPayload):
    """Start a confirmed, session-stable canary from current recommendation evidence."""
    project_id = current_project_id()
    candidate = str(payload.candidate_model or "").strip()
    if candidate != str(payload.confirm_model or "").strip():
        raise HTTPException(status_code=422, detail="Model confirmation does not match")
    trajectory_store = get_trajectory_store()
    comparisons = trajectory_store.list_eval_comparisons(project_id=project_id, limit=20)
    recommendation = build_model_promotion_recommendation(
        comparisons,
        current_model=str(settings.SUMMARY_MODEL or ""),
    )
    if (
        recommendation.get("status") != "promote"
        or recommendation.get("recommended_model") != candidate
    ):
        raise HTTPException(status_code=409, detail="Model is not currently eligible for promotion")
    try:
        promotion = trajectory_store.create_model_promotion(
            project_id=project_id,
            previous_model=str(settings.SUMMARY_MODEL or ""),
            candidate_model=candidate,
            canary_percent=10,
            evidence_comparison_ids=recommendation.get("evidence_comparison_ids") or [],
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"promotion": promotion, "summary_model_changed": False}


@router.get("/trajectory/analytics/model-promotions")
async def list_project_model_promotions(
    limit: int = Query(default=20, ge=1, le=100),
):
    project_id = current_project_id()
    return {
        "project_id": project_id,
        "promotions": get_trajectory_store().list_model_promotions(
            project_id=project_id, limit=limit
        ),
    }


@router.post("/trajectory/analytics/model-promotions/{promotion_id}/promote")
async def finalize_project_model_promotion(
    promotion_id: str,
    payload: TrajectoryModelPromotionActionPayload,
):
    """Explicitly promote a canary only after its new comparison evidence is ready."""
    project_id = current_project_id()
    trajectory_store = get_trajectory_store()
    try:
        promotion = trajectory_store.get_model_promotion(
            project_id=project_id, promotion_id=promotion_id
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Model promotion not found") from exc
    if str(payload.confirm_model or "").strip() != promotion["candidate_model"]:
        raise HTTPException(status_code=422, detail="Model confirmation does not match")
    if promotion["status"] != "ready" or not promotion.get("ramp_complete"):
        raise HTTPException(
            status_code=409,
            detail="Canary comparisons and progressive traffic ramp must both be complete",
        )
    try:
        set_runtime_setting(
            "SUMMARY_MODEL", promotion["candidate_model"], target=settings
        )
        promoted = trajectory_store.mark_model_promotion_promoted(
            project_id=project_id, promotion_id=promotion_id
        )
    except Exception as exc:
        try:
            set_runtime_setting(
                "SUMMARY_MODEL", promotion["previous_model"], target=settings
            )
            trajectory_store.rollback_model_promotion(
                project_id=project_id,
                promotion_id=promotion_id,
                reason="promotion_apply_failed",
            )
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="Could not safely apply model promotion") from exc
    return {"promotion": promoted, "summary_model": str(settings.SUMMARY_MODEL)}


@router.post("/trajectory/analytics/model-promotions/{promotion_id}/rollback")
async def rollback_project_model_promotion(
    promotion_id: str,
    payload: TrajectoryModelPromotionActionPayload,
):
    """Stop a canary or explicitly restore the saved pre-promotion model."""
    project_id = current_project_id()
    trajectory_store = get_trajectory_store()
    try:
        promotion = trajectory_store.get_model_promotion(
            project_id=project_id, promotion_id=promotion_id
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Model promotion not found") from exc
    if str(payload.confirm_model or "").strip() != promotion["previous_model"]:
        raise HTTPException(status_code=422, detail="Rollback model confirmation does not match")
    if promotion["status"] in {"promoted", "rollback_pending"}:
        try:
            set_runtime_setting(
                "SUMMARY_MODEL", promotion["previous_model"], target=settings
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Could not restore rollback model") from exc
    try:
        rolled_back = trajectory_store.rollback_model_promotion(
            project_id=project_id,
            promotion_id=promotion_id,
            reason="manual_rollback",
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"promotion": rolled_back, "summary_model": str(settings.SUMMARY_MODEL)}


@router.delete("/trajectory/analytics/eval-cases/{case_id}")
async def delete_project_trajectory_eval_case(case_id: str):
    project_id = current_project_id()
    deleted = get_trajectory_store().delete_eval_case(
        project_id=project_id, case_id=case_id
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Trajectory eval case not found")
    return {"deleted": True, "case_id": case_id}


@router.post("/trajectory/analytics/policies")
async def create_project_trajectory_policy(payload: TrajectoryAlertPolicyPayload):
    project_id = current_project_id()
    try:
        policy = get_trajectory_store().create_alert_policy(
            project_id=project_id,
            name=payload.name,
            scope_type=payload.scope_type,
            scope_value=payload.scope_value,
            thresholds=_policy_thresholds(payload),
            enabled=payload.enabled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"policy": policy}


@router.get("/trajectory/analytics/policies")
async def list_project_trajectory_policies():
    project_id = current_project_id()
    return {
        "project_id": project_id,
        "policies": get_trajectory_store().list_alert_policies(project_id=project_id),
    }


@router.put("/trajectory/analytics/policies/{policy_id}")
async def update_project_trajectory_policy(
    policy_id: str,
    payload: TrajectoryAlertPolicyPayload,
):
    project_id = current_project_id()
    try:
        policy = get_trajectory_store().update_alert_policy(
            project_id=project_id,
            policy_id=policy_id,
            name=payload.name,
            scope_type=payload.scope_type,
            scope_value=payload.scope_value,
            thresholds=_policy_thresholds(payload),
            enabled=payload.enabled,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Trajectory alert policy not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"policy": policy}


@router.delete("/trajectory/analytics/policies/{policy_id}")
async def delete_project_trajectory_policy(policy_id: str):
    project_id = current_project_id()
    deleted = get_trajectory_store().delete_alert_policy(
        project_id=project_id,
        policy_id=policy_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Trajectory alert policy not found")
    return {"deleted": True, "policy_id": policy_id}


@router.get("/conversations/{conversation_id}/trajectory")
async def get_conversation_trajectory(
    conversation_id: str,
    limit: int = Query(default=1500, ge=1, le=5000),
):
    project_id = current_project_id()
    try:
        conversation = get_conversation_store(project_id).require(conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc

    trajectory_store = get_trajectory_store()
    records = trajectory_store.list_events(
        project_id=conversation.project_id,
        session_id=conversation.conversation_id,
        limit=limit,
    )
    count_events = getattr(trajectory_store, "count_events", None)
    stored_total = (
        int(count_events(
            project_id=conversation.project_id,
            session_id=conversation.conversation_id,
        ))
        if callable(count_events) else len(records)
    )
    legacy_projection = not records
    historical_count = 0
    if legacy_projection:
        records = _legacy_projection(conversation.project_id, conversation.conversation_id, limit)
        historical_count = len(records)
    else:
        # Preserve older transcript-only turns after upgrading to the richer
        # event store. New transcript rows are written after begin_turn, so a
        # strict timestamp boundary avoids duplicating the instrumented turn.
        first_started = min(
            (float(row["started_at"]) for row in records if row.get("started_at") is not None),
            default=0.0,
        )
        if first_started:
            historical = [
                row
                for row in _legacy_projection(
                    conversation.project_id,
                    conversation.conversation_id,
                    limit,
                )
                if row.get("started_at") is not None
                and float(row["started_at"]) < first_started
            ]
            records = historical + records
            historical_count = len(historical)
            for sequence, row in enumerate(records, start=1):
                row["sequence"] = sequence
    diagnostics = analyze_trajectory(records)
    event_log = {
        "status": "legacy" if legacy_projection else "unavailable",
        "append_only": False,
        "contract_version": 0,
        "event_versions": 0,
        "subjects": 0,
        "last_sequence": 0,
        "integrity": {"ok": legacy_projection, "drifted": []},
        "projections": [],
    }
    trajectory_path = getattr(trajectory_store, "path", None)
    if not legacy_projection and (
        isinstance(trajectory_path, str) or hasattr(trajectory_path, "__fspath__")
    ):
        try:
            unified_store = SessionEventStore(trajectory_path)
            journal = unified_store.list_events(
                project_id=conversation.project_id,
                session_id=conversation.conversation_id,
            )
            verification = unified_store.verify_materialized_projection(
                project_id=conversation.project_id,
                session_id=conversation.conversation_id,
            )
            event_log = {
                "status": "available",
                "append_only": True,
                "contract_version": 1,
                "event_versions": len(journal),
                "subjects": int(verification["subjects"]),
                "last_sequence": int(journal[-1]["sequence"] if journal else 0),
                "integrity": verification,
                "projections": [
                    "transcript",
                    "tool_runs",
                    "metrics",
                    "recovery",
                    "model_history",
                    "checkpoint",
                    "execution_ledger",
                    "audit",
                ],
            }
        except (OSError, ValueError, sqlite3.Error):
            pass
    estimated_total = (
        historical_count if legacy_projection else stored_total + historical_count
    )
    loaded_stored = min(stored_total, limit)
    loaded_total = len(records)
    may_have_more_legacy = historical_count >= limit
    truncated = stored_total > loaded_stored or may_have_more_legacy
    can_load_more = truncated and limit < 5000
    diagnostics["integrity"]["window_truncated"] = truncated
    return {
        "conversation_id": conversation.conversation_id,
        "project_id": conversation.project_id,
        "records": records,
        "summary": _summary(records),
        "turns": _turn_summaries(records),
        "diagnostics": diagnostics,
        "event_log": event_log,
        "pagination": {
            "limit": limit,
            "returned": loaded_total,
            "estimated_total": max(estimated_total, loaded_total),
            "has_more": can_load_more,
            "window_truncated": truncated,
            "next_limit": min(5000, max(limit + 500, limit * 2)) if can_load_more else limit,
        },
        "legacy_projection": legacy_projection,
    }


def _read_only_trajectory_projection(
    *,
    project_id: str,
    session_id: str,
    execution: dict,
    limit: int,
) -> dict:
    trajectory_store = get_trajectory_store()
    records = trajectory_store.list_events(
        project_id=project_id,
        session_id=session_id,
        limit=limit,
    )
    if not records:
        raise HTTPException(status_code=404, detail="Execution trajectory not found")
    count_events = getattr(trajectory_store, "count_events", None)
    stored_total = (
        int(count_events(project_id=project_id, session_id=session_id))
        if callable(count_events)
        else len(records)
    )
    diagnostics = analyze_trajectory(records)
    truncated = stored_total > len(records)
    diagnostics["integrity"]["window_truncated"] = truncated
    can_load_more = truncated and limit < 5000
    return {
        "conversation_id": "",
        "execution_session_id": session_id,
        "execution": {**execution, "read_only": True},
        "project_id": project_id,
        "records": records,
        "summary": _summary(records),
        "turns": _turn_summaries(records),
        "diagnostics": diagnostics,
        "event_log": {
            "status": "available",
            "append_only": True,
            "contract_version": 1,
            "event_versions": stored_total,
            "subjects": stored_total,
            "last_sequence": int(records[-1].get("sequence") or 0),
            "integrity": {"ok": True, "drifted": []},
            "projections": ["execution"],
        },
        "pagination": {
            "limit": limit,
            "returned": len(records),
            "estimated_total": stored_total,
            "has_more": can_load_more,
            "window_truncated": truncated,
            "next_limit": min(5000, max(limit + 500, limit * 2)) if can_load_more else limit,
        },
        "legacy_projection": False,
    }


@router.get("/trajectory/executions/{scope}/{source_id}/{run_id}")
async def get_execution_trajectory(
    scope: str,
    source_id: str,
    run_id: str,
    limit: int = Query(default=1500, ge=1, le=5000),
):
    """Return a read-only Trajectory projection for a non-chat execution."""
    normalized_scope = str(scope or "").strip().lower()
    if normalized_scope not in {"experiment", "automation", "agent_lab"}:
        raise HTTPException(status_code=404, detail="Execution trajectory scope not found")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    if any(not value or any(ch not in allowed for ch in value) for value in (source_id, run_id)):
        raise HTTPException(status_code=400, detail="Invalid execution trajectory identifier")

    project_id = current_project_id()
    session_id = f"{normalized_scope}:{source_id}:{run_id}"
    return _read_only_trajectory_projection(
        project_id=project_id,
        session_id=session_id,
        execution={"scope": normalized_scope, "source_id": source_id, "run_id": run_id},
        limit=limit,
    )


@router.get("/trajectory/self-modifications/{proposal_id}")
async def get_self_modification_trajectory(
    proposal_id: str,
    limit: int = Query(default=1500, ge=1, le=5000),
):
    """Project an immutable guidance proposal lifecycle into the shared Inspector."""
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    if not proposal_id or any(ch not in allowed for ch in proposal_id):
        raise HTTPException(status_code=400, detail="Invalid self-modification proposal identifier")
    project_id = current_project_id()
    try:
        from remy.core.self_modification_lab import get_self_modification_lab

        proposal = get_self_modification_lab().get(
            project_id=project_id,
            proposal_id=proposal_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Self-modification proposal not found") from exc
    return _read_only_trajectory_projection(
        project_id=project_id,
        session_id=str(proposal["trajectory_session_id"]),
        execution={
            "scope": "self-modification",
            "source_id": proposal_id,
            "run_id": "lifecycle",
        },
        limit=limit,
    )


@router.post("/conversations/{conversation_id}/trajectory/{event_id}/fork")
async def fork_conversation_from_trajectory(
    conversation_id: str,
    event_id: str,
    payload: TrajectoryForkPayload,
):
    """Create a non-executing conversation branch at a causal event boundary."""
    project_id = current_project_id()
    conversation_store = get_conversation_store(project_id)
    try:
        source = conversation_store.require(conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc

    trajectory = await get_conversation_trajectory(conversation_id, limit=5000)
    if payload.activate and any(
        str(record.get("status") or "") == "running"
        for record in trajectory.get("records") or []
    ):
        raise HTTPException(
            status_code=409,
            detail="Wait for the active turn to finish or stop it before switching to a fork",
        )
    try:
        dialogue, suggested_prompt, boundary = _fork_dialogue(
            trajectory.get("records") or [], event_id
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Trajectory boundary not found") from exc

    title = str(payload.title or "").strip() or f"Fork · {source.title}"
    next_prompt = str(payload.next_prompt or "").strip() or suggested_prompt
    boundary_summary = {
        "event_id": str(boundary.get("event_id") or event_id),
        "sequence": int(boundary.get("sequence") or 0),
        "kind": str(boundary.get("kind") or ""),
        "turn_id": str(boundary.get("turn_id") or ""),
        "status": str(boundary.get("status") or ""),
    }
    fork = conversation_store.create(
        title,
        activate=False,
        metadata={
            "fork": {
                "source_conversation_id": source.conversation_id,
                "boundary": boundary_summary,
                "copied_messages": len(dialogue),
                "preferred_model": str(payload.preferred_model or ""),
                "auto_executed": False,
            }
        },
    )

    transcript_store = get_transcript_store()
    try:
        for message in dialogue:
            transcript_store.append(
                session_id=fork.conversation_id,
                owner_project_id=fork.project_id,
                brain_id=fork.brain_id,
                role=message["role"],
                content=message["content"],
                metadata={
                    "project_id": fork.project_id,
                    "fork_source_conversation_id": source.conversation_id,
                    "fork_boundary_event_id": event_id,
                    "fork_source_event_id": message["source_event_id"],
                    "fork_source_turn_id": message["source_turn_id"],
                },
            )
        get_trajectory_store().record_fork(
            session_id=fork.conversation_id,
            project_id=fork.project_id,
            source_session_id=source.conversation_id,
            boundary_event_id=event_id,
            boundary_sequence=boundary_summary["sequence"],
            boundary_kind=boundary_summary["kind"],
            copied_messages=len(dialogue),
            preferred_model=payload.preferred_model,
        )
    except Exception as exc:
        # Keep a failed partial branch out of the active chat list while
        # preserving it as a recoverable archived record for diagnostics.
        conversation_store.archive(fork.conversation_id)
        raise HTTPException(status_code=500, detail="Could not persist trajectory fork") from exc

    active_id = conversation_store.get_active_id()
    if payload.activate:
        session = await _get_api().get_session_manager().switch_conversation(
            project_id,
            fork.conversation_id,
        )
        active_id = session.session_id

    return {
        "conversation": {
            **fork.to_dict(),
            "active": fork.conversation_id == active_id,
        },
        "active_conversation_id": active_id,
        "chat_reset": bool(payload.activate),
        "fork": {
            "source_conversation_id": source.conversation_id,
            "boundary": boundary_summary,
            "copied_messages": len(dialogue),
            "preferred_model": str(payload.preferred_model or ""),
            "next_prompt": next_prompt,
            "auto_executed": False,
        },
    }


@router.put("/conversations/{conversation_id}/trajectory/{event_id}/annotation")
async def update_trajectory_annotation(
    conversation_id: str,
    event_id: str,
    payload: TrajectoryAnnotationPayload,
):
    project_id = current_project_id()
    try:
        conversation = get_conversation_store(project_id).require(conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    try:
        annotation = get_trajectory_store().set_annotation(
            project_id=conversation.project_id,
            session_id=conversation.conversation_id,
            event_id=event_id,
            label=payload.label,
            note=payload.note,
            bookmarked=payload.bookmarked,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Trajectory record not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"event_id": event_id, "annotation": annotation}


@router.delete("/conversations/{conversation_id}/trajectory/{event_id}/annotation")
async def delete_trajectory_annotation(conversation_id: str, event_id: str):
    project_id = current_project_id()
    try:
        conversation = get_conversation_store(project_id).require(conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    deleted = get_trajectory_store().delete_annotation(
        project_id=conversation.project_id,
        session_id=conversation.conversation_id,
        event_id=event_id,
    )
    return {"event_id": event_id, "deleted": deleted}
