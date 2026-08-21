"""Persistent execution history for pipelines and automations."""

from __future__ import annotations

import json
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from remy.core.execution_ledger import get_execution_ledger
from remy.core.file_utils import atomic_write
from remy.core.workflow_memory_evaluator import evaluate_workflow_memory

MAX_RUN_RECORDS_PER_WORKFLOW = 100
DEFAULT_RUN_LIST_LIMIT = 50
MAX_RUN_LIST_LIMIT = 100


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_id(value: str) -> str:
    safe = "".join(ch for ch in (value or "") if ch.isalnum() or ch in {"-", "_"})
    return safe[:80] or "unknown"


def _runs_root() -> Path:
    from remy.core.project_store import project_data_root

    root = project_data_root() / "workflow_runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _workflow_dir(kind: str, workflow_id: str) -> Path:
    path = _runs_root() / _safe_id(kind) / _safe_id(workflow_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_path(kind: str, workflow_id: str, run_id: str) -> Path:
    return _workflow_dir(kind, workflow_id) / f"{_safe_id(run_id)}.json"


def start_workflow_run(
    *,
    kind: str,
    workflow_id: str,
    workflow_name: str = "",
    input_text: str = "",
    trigger: str = "manual",
    idempotency_class: str = "side_effecting",
    session_id: str = "",
    channel: str = "",
) -> dict[str, Any]:
    from remy.core.microbrain import current_project_id
    from remy.core.project_store import get_project_store

    owner = get_project_store().require_project(current_project_id())
    run_id = f"run-{uuid.uuid4().hex[:12]}"
    attempt = get_execution_ledger().claim(
        kind=kind,
        job_id=f"{workflow_id}:{run_id}",
        idempotency_class=idempotency_class,
        owner_project_id=owner.project_id,
        brain_id=owner.brain_id,
        session_id=session_id,
        channel=channel,
        metadata={
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "trigger": trigger,
            "owner_project_id": owner.project_id,
            "brain_id": owner.brain_id,
        },
    )
    get_execution_ledger().mark_running(attempt["attempt_id"])
    from remy.core.run_envelope import attach_attempt_envelope

    envelope = attach_attempt_envelope(
        attempt_id=attempt["attempt_id"],
        run_id=run_id,
        kind=kind,
        source_id=workflow_id,
        goal=input_text or workflow_name or f"Run {kind}",
        owner_project_id=owner.project_id,
        brain_id=owner.brain_id,
        conversation_id=session_id,
        metadata={"workflow_name": workflow_name, "trigger": trigger},
    )
    record: dict[str, Any] = {
        "run_id": run_id,
        "execution_attempt_id": attempt["attempt_id"],
        "idempotency_class": idempotency_class,
        "kind": kind,
        "owner_project_id": owner.project_id,
        "brain_id": owner.brain_id,
        "workflow_id": workflow_id,
        "workflow_name": workflow_name,
        "trigger": trigger,
        "status": "running",
        "started_at": _now(),
        "finished_at": "",
        "duration_ms": None,
        "input": input_text,
        "output": "",
        "output_length": 0,
        "output_truncated": False,
        "error": "",
        "steps_run": 0,
        "trace": [],
        "run_envelope": envelope,
    }
    try:
        save_workflow_run(record)
        _prune_workflow_runs(kind, workflow_id)
    except BaseException as exc:
        get_execution_ledger().finish(
            attempt["attempt_id"],
            "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=f"Could not persist workflow run: {exc}",
        )
        raise
    return record


def finish_workflow_run(
    record: dict[str, Any],
    *,
    status: str,
    output: str = "",
    error: str = "",
    trace: list[dict] | None = None,
    steps_run: int | None = None,
    output_truncated: bool = False,
) -> dict[str, Any]:
    started_at = _parse_dt(record.get("started_at", ""))
    finished_at = datetime.now()
    record["status"] = status
    record["finished_at"] = finished_at.isoformat(timespec="seconds")
    record["duration_ms"] = int((finished_at - started_at).total_seconds() * 1000) if started_at else None
    record["output"] = output or ""
    record["output_length"] = len(output or "")
    record["output_truncated"] = bool(output_truncated)
    record["error"] = error or ""
    record["trace"] = list(trace or [])
    record["steps_run"] = int(steps_run if steps_run is not None else len(record["trace"]))
    record["memory_evaluation"] = evaluate_workflow_memory(record["trace"])
    attempt_id = str(record.get("execution_attempt_id") or "")
    try:
        save_workflow_run(record)
    except BaseException as exc:
        if attempt_id:
            try:
                get_execution_ledger().finish(
                    attempt_id,
                    "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                    error=f"Could not persist terminal workflow record: {exc}",
                )
            except RuntimeError:
                pass
        raise
    if attempt_id:
        from remy.core.run_envelope import finish_run

        envelope_status = _envelope_terminal_status(status, output_truncated=output_truncated)
        try:
            record["run_envelope"] = finish_run(
                attempt_id,
                status=envelope_status,
                output_ref=f"workflow:{record.get('kind', '')}:{record.get('workflow_id', '')}:{record.get('run_id', '')}",
                error=error,
                stop_reason=(
                    "output_limit" if output_truncated else
                    "user_stopped" if envelope_status == "cancelled" else ""
                ),
            )
            save_workflow_run(record)
        except RuntimeError:
            # Preserve compatibility if a caller repeats finalization; the ledger's
            # first terminal receipt remains authoritative.
            pass
    return record


def update_workflow_run_progress(
    record: dict[str, Any],
    *,
    step: str,
    signature: str = "",
) -> dict[str, Any]:
    """Commit a bounded workflow step to the shared run envelope."""
    from remy.core.run_envelope import RunCoordinator

    attempt_id = str(record.get("execution_attempt_id") or "")
    if not attempt_id:
        return {}
    envelope = RunCoordinator(attempt_id).step(step, signature=signature)
    record["run_envelope"] = envelope
    save_workflow_run(record)
    return envelope


def _ledger_terminal_state(status: str, *, output_truncated: bool = False) -> str:
    normalized = str(status or "").lower()
    if normalized in {"ok", "success", "completed", "complete"}:
        return "completed_with_limits" if output_truncated else "completed"
    if normalized in {"partial", "completed_with_limits"}:
        return "completed_with_limits"
    if normalized in {"cancelled", "canceled", "stopped"}:
        return "cancelled"
    if normalized == "blocked":
        return "blocked"
    return "failed"


def _envelope_terminal_status(status: str, *, output_truncated: bool = False) -> str:
    state = _ledger_terminal_state(status, output_truncated=output_truncated)
    return "interrupted" if state == "unknown" else state


def save_workflow_run(record: dict[str, Any]) -> None:
    path = _run_path(str(record.get("kind", "")), str(record.get("workflow_id", "")), str(record.get("run_id", "")))
    atomic_write(path, json.dumps(record, ensure_ascii=False, indent=2))


def get_workflow_run(kind: str, workflow_id: str, run_id: str) -> dict[str, Any] | None:
    path = _run_path(kind, workflow_id, run_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_workflow_runs(kind: str, workflow_id: str, limit: int = 50) -> list[dict[str, Any]]:
    limit = _normalize_limit(limit, default=DEFAULT_RUN_LIST_LIMIT, maximum=MAX_RUN_LIST_LIMIT)
    path = _workflow_dir(kind, workflow_id)
    records: list[dict[str, Any]] = []
    for file in sorted(path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except Exception:
            continue
        records.append({
            "run_id": data.get("run_id", file.stem),
            "execution_attempt_id": data.get("execution_attempt_id", ""),
            "idempotency_class": data.get("idempotency_class", ""),
            "kind": data.get("kind", kind),
            "workflow_id": data.get("workflow_id", workflow_id),
            "workflow_name": data.get("workflow_name", ""),
            "template_id": data.get("template_id", ""),
            "title": data.get("title", data.get("workflow_name", "")),
            "pack": data.get("pack", ""),
            "mode": data.get("mode", ""),
            "trigger": data.get("trigger", ""),
            "status": data.get("status", ""),
            "started_at": data.get("started_at", ""),
            "created_at": data.get("started_at", ""),
            "finished_at": data.get("finished_at", ""),
            "duration_ms": data.get("duration_ms"),
            "cost": data.get("cost", ""),
            "retry_count": data.get("retry_count", 0),
            "auto_paused": data.get("auto_paused", False),
            "steps_run": data.get("steps_run", 0),
            "memory_evaluation": data.get("memory_evaluation", {}),
            "output_preview": (data.get("output", "") or "")[:500],
            "preview": (data.get("output", "") or "")[:500],
            "error": data.get("error", ""),
            "run_envelope": data.get("run_envelope", {}),
            "trajectory_session_id": data.get("trajectory_session_id", ""),
            "trajectory_run_event_id": data.get("trajectory_run_event_id", ""),
            "trajectory_result_event_id": data.get("trajectory_result_event_id", ""),
        })
        if len(records) >= limit:
            break
    return records


def summarize_workflow_memory(kind: str, workflow_id: str, limit: int = 20) -> dict[str, Any]:
    limit = _normalize_limit(limit, default=20, maximum=MAX_RUN_LIST_LIMIT)
    runs = list_workflow_runs(kind, workflow_id, limit=limit)
    reports = [
        run.get("memory_evaluation") or {}
        for run in runs
        if isinstance(run.get("memory_evaluation"), dict) and run.get("memory_evaluation")
    ]
    if not reports:
        return {
            "kind": kind,
            "workflow_id": workflow_id,
            "run_count": len(runs),
            "evaluated_run_count": 0,
            "average_score": None,
            "status": "no_data",
            "totals": {
                "memory_search_count": 0,
                "memory_save_count": 0,
                "empty_search_count": 0,
                "duplicate_save_candidate_count": 0,
                "missed_search_before_save_count": 0,
            },
            "top_recommendations": [],
            "trend": {
                "latest_score": None,
                "previous_score": None,
                "delta": None,
                "direction": "unknown",
                "latest_run_id": "",
                "previous_run_id": "",
            },
        }

    totals = {
        "memory_search_count": sum(_int_report_value(report, "memory_search_count") for report in reports),
        "memory_save_count": sum(_int_report_value(report, "memory_save_count") for report in reports),
        "empty_search_count": sum(_int_report_value(report, "empty_search_count") for report in reports),
        "duplicate_save_candidate_count": sum(
            _int_report_value(report, "duplicate_save_candidate_count") for report in reports
        ),
        "missed_search_before_save_count": sum(1 for report in reports if report.get("missed_search_before_save")),
    }
    scores = [_int_report_value(report, "score") for report in reports]
    average_score = round(sum(scores) / len(scores), 1)
    recommendation_counts: Counter[str] = Counter()
    for report in reports:
        for recommendation in report.get("recommendations") or []:
            text = str(recommendation).strip()
            if text:
                recommendation_counts[text] += 1

    return {
        "kind": kind,
        "workflow_id": workflow_id,
        "run_count": len(runs),
        "evaluated_run_count": len(reports),
        "average_score": average_score,
        "status": _memory_report_status(average_score, totals),
        "totals": totals,
        "top_recommendations": [
            {"text": text, "count": count}
            for text, count in recommendation_counts.most_common(5)
        ],
        "trend": _memory_report_trend(runs),
    }


def delete_workflow_runs(kind: str, workflow_id: str) -> int:
    path = _workflow_dir(kind, workflow_id)
    deleted = 0
    for file in path.glob("*.json"):
        try:
            file.unlink()
            deleted += 1
        except Exception:
            pass
    return deleted


def _normalize_limit(value: int, *, default: int, maximum: int) -> int:
    try:
        limit = int(value)
    except Exception:
        return default
    if limit <= 0:
        return default
    return min(limit, maximum)


def _int_report_value(report: dict[str, Any], key: str) -> int:
    try:
        return int(report.get(key) or 0)
    except Exception:
        return 0


def _memory_report_status(average_score: float, totals: dict[str, int]) -> str:
    if average_score < 70:
        return "needs_attention"
    if (
        totals.get("empty_search_count", 0)
        or totals.get("duplicate_save_candidate_count", 0)
        or totals.get("missed_search_before_save_count", 0)
    ):
        return "watch"
    return "ok"


def _memory_report_trend(runs: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated: list[tuple[dict[str, Any], int]] = []
    for run in runs:
        report = run.get("memory_evaluation") or {}
        if not isinstance(report, dict):
            continue
        try:
            score = int(report.get("score"))
        except Exception:
            continue
        evaluated.append((run, score))
        if len(evaluated) >= 2:
            break

    if not evaluated:
        return {
            "latest_score": None,
            "previous_score": None,
            "delta": None,
            "direction": "unknown",
            "latest_run_id": "",
            "previous_run_id": "",
        }

    latest_run, latest_score = evaluated[0]
    previous_run = {}
    previous_score: int | None = None
    delta: int | None = None
    direction = "baseline"
    if len(evaluated) > 1:
        previous_run, previous_score = evaluated[1]
        delta = latest_score - previous_score
        if delta > 0:
            direction = "improved"
        elif delta < 0:
            direction = "regressed"
        else:
            direction = "unchanged"

    return {
        "latest_score": latest_score,
        "previous_score": previous_score,
        "delta": delta,
        "direction": direction,
        "latest_run_id": latest_run.get("run_id", ""),
        "previous_run_id": previous_run.get("run_id", ""),
    }


def _parse_dt(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _prune_workflow_runs(kind: str, workflow_id: str) -> None:
    path = _workflow_dir(kind, workflow_id)
    files = sorted(path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[MAX_RUN_RECORDS_PER_WORKFLOW:]:
        try:
            old.unlink()
        except Exception:
            pass
