"""Deterministic diagnostics derived from a conversation trajectory."""

from __future__ import annotations

import hashlib
import math
import re
import statistics
import time
from datetime import datetime, timezone
from typing import Any


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and "content" in value:
        return _text(value.get("content"))
    return str(value)


def _finding(
    record: dict[str, Any],
    *,
    severity: str,
    category: str,
    title: str,
    explanation: str,
    next_check: str,
) -> dict[str, Any]:
    return {
        "finding_id": f"{category}:{record.get('event_id') or record.get('sequence')}",
        "event_id": str(record.get("event_id") or ""),
        "turn_id": str(record.get("turn_id") or ""),
        "kind": str(record.get("kind") or "EVENT"),
        "severity": severity,
        "category": category,
        "title": title,
        "explanation": explanation,
        "next_check": next_check,
    }


def _failure_finding(record: dict[str, Any]) -> dict[str, Any]:
    kind = str(record.get("kind") or "EVENT")
    error = str(record.get("error") or "Unknown failure")
    interrupted = any(word in error.lower() for word in ("interrupt", "cancel", "stopped"))
    severity = "warning" if interrupted else "error"
    if kind == "REQUEST":
        explanation = "The model request ended before a usable assistant result was produced."
        next_check = "Inspect Options, model/provider, Timing, and the effective Input."
    elif kind in {"TOOL", "SUBTOOL"}:
        explanation = "The tool lifecycle completed with an error or was interrupted."
        next_check = "Compare Payload with Schema, then inspect Result, Policy, and parent Request."
    elif kind == "VERIFICATION":
        explanation = "Post-response verification did not accept the generated claims."
        next_check = "Inspect Claims and Evidence, then repair the unsupported statements."
    elif kind in {"POLICY", "APPROVAL"}:
        explanation = "A policy or approval decision prevented normal execution."
        next_check = "Inspect the governing rule, risk signal, and recorded decision."
    else:
        explanation = "This execution record did not complete successfully."
        next_check = "Inspect Raw, Source, Timing, and its causal parent."
    return _finding(
        record,
        severity=severity,
        category="interruption" if interrupted else "failure",
        title=f"{kind} {'interrupted' if interrupted else 'failed'}: {error[:120]}",
        explanation=explanation,
        next_check=next_check,
    )


def _timing_breakdown(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize leaf execution spans without double-counting parent requests."""
    attempt_request_ids = {
        str(row.get("request_id") or "")
        for row in records
        if row.get("kind") == "ATTEMPT" and row.get("request_id")
    }
    operations = []
    for row in records:
        kind = str(row.get("kind") or "")
        if kind in {"ATTEMPT", "TOOL", "SUBTOOL"} or (
            kind == "REQUEST" and str(row.get("request_id") or "") not in attempt_request_ids
        ):
            start = row.get("started_at")
            if start is None:
                continue
            end = row.get("completed_at")
            if end is None:
                end = float(start) + max(0, int(row.get("duration_ms") or 0)) / 1000
            operations.append((float(start), max(float(start), float(end)), row))

    all_times = [
        float(value)
        for row in records
        for value in (row.get("started_at"), row.get("completed_at"))
        if value is not None
    ]
    wall_clock_ms = (
        max(0, int((max(all_times) - min(all_times)) * 1000))
        if len(all_times) >= 2 else 0
    )
    intervals = sorted((start, end) for start, end, _ in operations if end > start)
    merged: list[list[float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    busy_ms = int(sum((end - start) * 1000 for start, end in merged))
    summed_operation_ms = sum(
        max(0, int((end - start) * 1000)) for start, end, _ in operations
    )
    sweep = []
    for start, end in intervals:
        sweep.append((start, 1))
        sweep.append((end, -1))
    concurrency = 0
    max_concurrency = 0
    for _, delta in sorted(sweep, key=lambda item: (item[0], item[1])):
        concurrency += delta
        max_concurrency = max(max_concurrency, concurrency)

    def operation_duration(kinds: set[str]) -> int:
        return sum(
            max(0, int((end - start) * 1000))
            for start, end, row in operations
            if row.get("kind") in kinds
        )

    return {
        "wall_clock_ms": wall_clock_ms,
        "busy_ms": busy_ms,
        "idle_ms": max(0, wall_clock_ms - busy_ms),
        "overlap_ms": max(0, summed_operation_ms - busy_ms),
        "model_ms": operation_duration({"REQUEST", "ATTEMPT"}),
        "tool_ms": operation_duration({"TOOL", "SUBTOOL"}),
        "operation_count": len(operations),
        "max_concurrency": max_concurrency,
    }


def _trace_integrity(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate that the trace itself is sufficiently complete and causal."""
    issues: list[dict[str, Any]] = []
    event_ids = [str(row.get("event_id") or "") for row in records if row.get("event_id")]
    event_id_set = set(event_ids)
    request_ids = {
        str(row.get("request_id") or row.get("event_id") or "")
        for row in records if row.get("kind") == "REQUEST"
    }
    known_refs = event_id_set | request_ids | {
        str(row.get("call_id") or "") for row in records if row.get("call_id")
    }

    def issue(
        row: dict[str, Any] | None,
        *,
        severity: str,
        category: str,
        title: str,
        explanation: str,
    ) -> None:
        event_id = str((row or {}).get("event_id") or "")
        issues.append({
            "issue_id": f"{category}:{event_id or len(issues) + 1}",
            "event_id": event_id,
            "turn_id": str((row or {}).get("turn_id") or ""),
            "severity": severity,
            "category": category,
            "title": title,
            "explanation": explanation,
        })

    seen_ids: set[str] = set()
    seen_sequences: set[int] = set()
    prior_sequence: int | None = None
    for row in records:
        event_id = str(row.get("event_id") or "")
        if event_id and event_id in seen_ids:
            issue(
                row,
                severity="error",
                category="duplicate-event",
                title=f"Duplicate event ID: {event_id}",
                explanation="Two ledger rows share an identity, so causal links are ambiguous.",
            )
        seen_ids.add(event_id)

        sequence = row.get("sequence")
        if sequence is not None:
            sequence = int(sequence)
            if sequence in seen_sequences or (
                prior_sequence is not None and sequence <= prior_sequence
            ):
                issue(
                    row,
                    severity="error",
                    category="sequence-order",
                    title=f"Invalid ledger sequence: {sequence}",
                    explanation="Trajectory rows are duplicated or no longer strictly ordered.",
                )
            seen_sequences.add(sequence)
            prior_sequence = sequence

        parent_id = str(row.get("parent_id") or "")
        if parent_id and parent_id not in known_refs:
            issue(
                row,
                severity="warning",
                category="orphan-parent",
                title=f"Missing causal parent: {parent_id}",
                explanation="The record references a parent that is absent from the loaded trace window.",
            )
        request_id = str(row.get("request_id") or "")
        if row.get("kind") != "REQUEST" and request_id and request_id not in request_ids:
            issue(
                row,
                severity="warning",
                category="orphan-request",
                title=f"Missing request: {request_id}",
                explanation="The record cannot be correlated with its originating model request.",
            )
        start = row.get("started_at")
        first = row.get("first_output_at")
        completed = row.get("completed_at")
        if start is not None and completed is not None and float(completed) < float(start):
            issue(
                row,
                severity="error",
                category="negative-duration",
                title="Completion precedes start",
                explanation="This record has an invalid clock ordering and its duration is unreliable.",
            )
        if start is not None and first is not None and float(first) < float(start):
            issue(
                row,
                severity="warning",
                category="invalid-first-output",
                title="First output precedes start",
                explanation="TTFT cannot be trusted for this record.",
            )
        if not row.get("turn_id") and row.get("kind") not in {"SYSTEM"}:
            issue(
                row,
                severity="warning",
                category="missing-turn",
                title="Record is not assigned to a turn",
                explanation="Turn replay and comparison cannot place this event reliably.",
            )

    total = len(records)
    with_source = sum(1 for row in records if row.get("source"))
    with_timing = sum(1 for row in records if row.get("started_at") is not None)
    child_kinds = {"SYSTEM", "CONTEXT", "ATTEMPT", "ASSISTANT", "TOOL", "SUBTOOL"}
    children = [row for row in records if row.get("kind") in child_kinds]
    correlated = sum(
        1 for row in children
        if row.get("turn_id") and (
            row.get("kind") == "SYSTEM"
            or row.get("request_id") in request_ids
            or not request_ids
        )
    )
    requests = [row for row in records if row.get("kind") == "REQUEST"]
    attempted_requests = {
        str(row.get("request_id") or "")
        for row in records if row.get("kind") == "ATTEMPT" and row.get("request_id")
    }

    def percent(numerator: int, denominator: int) -> int:
        return 100 if denominator <= 0 else round((numerator / denominator) * 100)

    coverage = {
        "source_percent": percent(with_source, total),
        "timing_percent": percent(with_timing, total),
        "correlation_percent": percent(correlated, len(children)),
        "attempt_percent": percent(len(attempted_requests), len(requests)),
    }
    coverage["score"] = round(
        coverage["source_percent"] * 0.20
        + coverage["timing_percent"] * 0.25
        + coverage["correlation_percent"] * 0.30
        + coverage["attempt_percent"] * 0.25
    )
    error_count = sum(1 for row in issues if row["severity"] == "error")
    warning_count = sum(1 for row in issues if row["severity"] == "warning")
    status = (
        "invalid" if error_count
        else "degraded" if warning_count or coverage["score"] < 75
        else "reliable"
    )
    return {
        "status": status,
        "coverage": coverage,
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": issues,
    }


def _error_clusters(
    records: list[dict[str, Any]],
    requests_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group repeated failures while ignoring volatile IDs and numbers."""

    def normalized_error(value: Any) -> str:
        text = str(value or "unknown failure").strip().lower()
        text = re.sub(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
            "<uuid>",
            text,
        )
        text = re.sub(r"\b[0-9a-f]{16,}\b", "<hex>", text)
        text = re.sub(r"\b\d+(?:\.\d+)?\b", "<n>", text)
        return " ".join(text.split())[:500]

    def component(row: dict[str, Any]) -> str:
        details = row.get("details") if isinstance(row.get("details"), dict) else {}
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        kind = str(row.get("kind") or "EVENT")
        if kind == "ATTEMPT":
            provider = details.get("provider") or source.get("provider") or "provider"
            model = details.get("model") or source.get("model") or "model"
            return f"{provider}/{model}"
        if kind in {"TOOL", "SUBTOOL"}:
            return str(details.get("name") or source.get("name") or kind)
        if kind == "REQUEST":
            return str(details.get("model") or source.get("model") or "model-request")
        return kind

    grouped: dict[str, dict[str, Any]] = {}
    for row in records:
        if not (row.get("status") == "failed" or row.get("error")):
            continue
        error = str(row.get("error") or "Unknown failure")
        identity = f"{row.get('kind')}|{component(row)}|{normalized_error(error)}"
        fingerprint = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        request = requests_by_id.get(str(row.get("request_id") or ""), {})
        recovered = row.get("kind") == "ATTEMPT" and request.get("status") == "completed"
        annotation = row.get("annotation") if isinstance(row.get("annotation"), dict) else {}
        cluster = grouped.setdefault(fingerprint, {
            "fingerprint": fingerprint,
            "kind": str(row.get("kind") or "EVENT"),
            "component": component(row),
            "sample_error": error[:300],
            "event_ids": [],
            "turn_ids": [],
            "occurrence_count": 0,
            "recovered_count": 0,
            "triaged_count": 0,
            "resolved_count": 0,
            "total_duration_ms": 0,
            "first_sequence": row.get("sequence"),
            "last_sequence": row.get("sequence"),
            "first_seen_at": row.get("started_at"),
            "last_seen_at": row.get("started_at"),
        })
        cluster["event_ids"].append(str(row.get("event_id") or ""))
        turn_id = str(row.get("turn_id") or "")
        if turn_id and turn_id not in cluster["turn_ids"]:
            cluster["turn_ids"].append(turn_id)
        cluster["occurrence_count"] += 1
        cluster["recovered_count"] += int(bool(recovered))
        cluster["triaged_count"] += int(bool(annotation))
        cluster["resolved_count"] += int(annotation.get("label") == "resolved")
        cluster["total_duration_ms"] += max(0, int(row.get("duration_ms") or 0))
        if row.get("sequence") is not None:
            cluster["last_sequence"] = row.get("sequence")
        if row.get("started_at") is not None:
            cluster["last_seen_at"] = row.get("started_at")

    clusters = list(grouped.values())
    for cluster in clusters:
        occurrences = int(cluster["occurrence_count"])
        cluster["recurring"] = occurrences > 1
        cluster["fully_recovered"] = cluster["recovered_count"] == occurrences
        cluster["resolution"] = (
            "resolved" if cluster["resolved_count"] == occurrences
            else "triaged" if cluster["triaged_count"]
            else "untriaged"
        )
        cluster["severity"] = (
            "info" if cluster["fully_recovered"]
            else "error"
        )
    return sorted(
        clusters,
        key=lambda item: (
            not item["recurring"],
            -int(item["occurrence_count"]),
            -int(item["last_sequence"] or 0),
        ),
    )


def analyze_project_trajectory(
    records: list[dict[str, Any]],
    *,
    conversation_titles: dict[str, str] | None = None,
    days: int = 30,
    now: float | None = None,
    baseline_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an aggregate-only project view across trajectory sessions.

    Raw prompts, outputs, schemas, error messages, and annotation notes are
    deliberately excluded from the returned shape.  Event IDs remain so an
    operator can jump from a regression to its full conversation inspector.
    """
    observed_now = float(now if now is not None else time.time())
    window_days = max(1, min(int(days), 365))
    window_start = observed_now - window_days * 86_400
    titles = conversation_titles or {}

    def timestamp(row: dict[str, Any]) -> float | None:
        value = row.get("started_at")
        if value is None:
            value = row.get("completed_at")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    scoped = [row for row in records if (timestamp(row) or 0) >= window_start]

    def duration(row: dict[str, Any]) -> int:
        try:
            explicit = int(row.get("duration_ms") or 0)
        except (TypeError, ValueError):
            explicit = 0
        if explicit:
            return max(0, explicit)
        try:
            return max(0, int((float(row["completed_at"]) - float(row["started_at"])) * 1000))
        except (KeyError, TypeError, ValueError):
            return 0

    def usage(row: dict[str, Any]) -> int:
        details = row.get("details") if isinstance(row.get("details"), dict) else {}
        values = details.get("usage") if isinstance(details.get("usage"), dict) else {}
        for key in ("total_tokens", "total_token_count"):
            try:
                if values.get(key) not in (None, ""):
                    return max(0, int(values[key]))
            except (TypeError, ValueError):
                pass
        total = 0
        for keys in (
            ("input_tokens", "prompt_tokens", "prompt_token_count"),
            ("output_tokens", "completion_tokens", "candidates_token_count"),
        ):
            for key in keys:
                try:
                    if values.get(key) not in (None, ""):
                        total += max(0, int(values[key]))
                        break
                except (TypeError, ValueError):
                    continue
        return total

    def percentile_95(values: list[int]) -> int:
        if not values:
            return 0
        ordered = sorted(values)
        return int(ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)])

    def component(row: dict[str, Any], fallback: str) -> str:
        details = row.get("details") if isinstance(row.get("details"), dict) else {}
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        return str(details.get("name") or source.get("name") or fallback)

    requests_by_id = {
        str(row.get("request_id") or row.get("event_id") or ""): row
        for row in scoped if row.get("kind") == "REQUEST"
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in scoped:
        session_id = str(row.get("session_id") or "")
        if session_id:
            grouped.setdefault(session_id, []).append(row)

    sessions = []
    for session_id, rows in grouped.items():
        requests = [row for row in rows if row.get("kind") == "REQUEST"]
        tools = [row for row in rows if row.get("kind") in {"TOOL", "SUBTOOL"}]
        attempts = [row for row in rows if row.get("kind") == "ATTEMPT"]
        failed_attempts = [row for row in attempts if row.get("status") == "failed" or row.get("error")]
        recovered_attempts = [
            row for row in failed_attempts
            if requests_by_id.get(str(row.get("request_id") or ""), {}).get("status") == "completed"
        ]
        failed_operations = [
            row for row in requests + tools
            if row.get("status") == "failed" or row.get("error")
        ]
        effective_failures = len(failed_operations) + len(failed_attempts) - len(recovered_attempts)
        operations = max(1, len(requests) + len(tools))
        request_durations = [duration(row) for row in requests if duration(row) > 0]
        timestamps = [value for row in rows for value in (timestamp(row), row.get("completed_at")) if value is not None]
        models = []
        providers = []
        for row in attempts or requests:
            details = row.get("details") if isinstance(row.get("details"), dict) else {}
            source = row.get("source") if isinstance(row.get("source"), dict) else {}
            model = str(details.get("model") or source.get("model") or "")
            provider = str(details.get("provider") or source.get("provider") or "")
            if model and model not in models:
                models.append(model)
            if provider and provider not in providers:
                providers.append(provider)
        latest = max(rows, key=lambda row: int(row.get("sequence") or 0))
        latest_failure = max(
            failed_operations or failed_attempts or rows,
            key=lambda row: int(row.get("sequence") or 0),
        )
        request_tokens = sum(usage(row) for row in requests)
        sessions.append({
            "session_id": session_id,
            "conversation_id": session_id,
            "title": str(titles.get(session_id) or "Conversation"),
            "started_at": min(float(value) for value in timestamps) if timestamps else None,
            "completed_at": max(float(value) for value in timestamps) if timestamps else None,
            "turns": len({str(row.get("turn_id")) for row in rows if row.get("turn_id")}),
            "events": len(rows),
            "requests": len(requests),
            "tool_calls": len(tools),
            "failures": effective_failures,
            "recovered_failures": len(recovered_attempts),
            "failure_rate": round(effective_failures / operations, 4),
            "total_tokens": request_tokens,
            "tokens_per_request": round(request_tokens / max(1, len(requests))),
            "avg_request_ms": round(statistics.fmean(request_durations)) if request_durations else 0,
            "p95_request_ms": percentile_95(request_durations),
            "providers": providers,
            "models": models,
            "latest_event_id": str(latest.get("event_id") or ""),
            "problem_event_id": str(latest_failure.get("event_id") or ""),
        })
    sessions.sort(key=lambda row: (float(row.get("started_at") or 0), row["session_id"]), reverse=True)

    metric_sessions = [row for row in sessions if row["requests"] or row["tool_calls"]]
    window_baseline = {
        "failure_rate": round(statistics.median([row["failure_rate"] for row in metric_sessions]), 4) if metric_sessions else 0,
        "avg_request_ms": round(statistics.median([row["avg_request_ms"] for row in metric_sessions if row["avg_request_ms"]])) if any(row["avg_request_ms"] for row in metric_sessions) else 0,
        "tokens_per_request": round(statistics.median([row["tokens_per_request"] for row in metric_sessions if row["requests"]])) if any(row["requests"] for row in metric_sessions) else 0,
    }
    if baseline_metrics is not None:
        baseline = {}
        for key in ("failure_rate", "avg_request_ms", "tokens_per_request"):
            try:
                baseline[key] = max(0.0, float(baseline_metrics.get(key) or 0))
            except (AttributeError, TypeError, ValueError):
                baseline[key] = 0.0
        baseline_source = "saved"
    else:
        baseline = window_baseline
        baseline_source = "window_median"
    regressions = []
    if metric_sessions and (baseline_metrics is not None or len(metric_sessions) >= 2):
        for row in metric_sessions:
            reasons = []
            failure_delta = row["failure_rate"] - baseline["failure_rate"]
            latency_delta = row["avg_request_ms"] - baseline["avg_request_ms"]
            token_delta = row["tokens_per_request"] - baseline["tokens_per_request"]
            if row["failures"] and failure_delta >= 0.15:
                reasons.append({
                    "metric": "failure_rate", "delta": round(failure_delta, 4),
                    "observed": row["failure_rate"], "baseline": baseline["failure_rate"],
                })
            if row["avg_request_ms"] > max(baseline["avg_request_ms"] * 1.5, baseline["avg_request_ms"] + 1000):
                reasons.append({
                    "metric": "latency", "delta": latency_delta,
                    "observed": row["avg_request_ms"], "baseline": baseline["avg_request_ms"],
                })
            if baseline["tokens_per_request"] and row["tokens_per_request"] > baseline["tokens_per_request"] * 1.5:
                reasons.append({
                    "metric": "tokens", "delta": token_delta,
                    "observed": row["tokens_per_request"],
                    "baseline": baseline["tokens_per_request"],
                })
            if reasons:
                regressions.append({
                    "conversation_id": row["conversation_id"],
                    "title": row["title"],
                    "severity": "error" if row["failures"] and failure_delta >= 0.30 else "warning",
                    "reasons": reasons,
                    "event_id": row["problem_event_id"] or row["latest_event_id"],
                })
    regressions.sort(key=lambda row: (row["severity"] != "error", -len(row["reasons"])))

    provider_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in scoped:
        if row.get("kind") != "ATTEMPT":
            continue
        details = row.get("details") if isinstance(row.get("details"), dict) else {}
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        key = (
            str(details.get("provider") or source.get("provider") or "unknown"),
            str(details.get("model") or source.get("model") or "unknown"),
        )
        provider_groups.setdefault(key, []).append(row)
    providers = []
    for (provider, model), rows in provider_groups.items():
        failed = [row for row in rows if row.get("status") == "failed" or row.get("error")]
        recovered = sum(
            requests_by_id.get(str(row.get("request_id") or ""), {}).get("status") == "completed"
            for row in failed
        )
        durations = [duration(row) for row in rows if duration(row) > 0]
        token_request_ids = {
            str(row.get("request_id") or "")
            for row in rows
            if row.get("status") == "completed" and row.get("request_id")
        }
        provider_tokens = sum(
            usage(requests_by_id[request_id])
            for request_id in token_request_ids if request_id in requests_by_id
        )
        providers.append({
            "provider": provider, "model": model, "attempts": len(rows),
            "failures": len(failed), "recovered": recovered,
            "avg_ms": round(statistics.fmean(durations)) if durations else 0,
            "p95_ms": percentile_95(durations),
            "token_requests": len(token_request_ids),
            "total_tokens": provider_tokens,
            "tokens_per_request": round(provider_tokens / max(1, len(token_request_ids))),
        })
    providers.sort(key=lambda row: (-row["attempts"], -row["failures"], row["provider"], row["model"]))

    tool_groups: dict[str, list[dict[str, Any]]] = {}
    for row in scoped:
        if row.get("kind") in {"TOOL", "SUBTOOL"}:
            tool_groups.setdefault(component(row, str(row.get("kind") or "tool")), []).append(row)
    tools = []
    for name, rows in tool_groups.items():
        durations = [duration(row) for row in rows if duration(row) > 0]
        tools.append({
            "name": name, "calls": len(rows),
            "failures": sum(row.get("status") == "failed" or bool(row.get("error")) for row in rows),
            "avg_ms": round(statistics.fmean(durations)) if durations else 0,
            "p95_ms": percentile_95(durations), "total_ms": sum(durations),
        })
    tools.sort(key=lambda row: (-row["calls"], -row["failures"], row["name"]))

    daily: dict[str, dict[str, Any]] = {}
    for row in scoped:
        stamp = timestamp(row)
        if stamp is None:
            continue
        day = datetime.fromtimestamp(stamp, tz=timezone.utc).date().isoformat()
        bucket = daily.setdefault(day, {
            "date": day, "session_ids": set(), "requests": 0,
            "tool_calls": 0, "failures": 0, "tokens": 0, "request_ms": [],
        })
        bucket["session_ids"].add(str(row.get("session_id") or ""))
        if row.get("kind") == "REQUEST":
            bucket["requests"] += 1
            bucket["tokens"] += usage(row)
            if duration(row) > 0:
                bucket["request_ms"].append(duration(row))
        if row.get("kind") in {"TOOL", "SUBTOOL"}:
            bucket["tool_calls"] += 1
        if row.get("status") == "failed" or row.get("error"):
            bucket["failures"] += 1
    trends = [{
        "date": day, "sessions": len(bucket["session_ids"]),
        "requests": bucket["requests"], "tool_calls": bucket["tool_calls"],
        "failures": bucket["failures"], "tokens": bucket["tokens"],
        "avg_request_ms": round(statistics.fmean(bucket["request_ms"])) if bucket["request_ms"] else 0,
    } for day, bucket in sorted(daily.items())]

    event_sessions = {str(row.get("event_id") or ""): str(row.get("session_id") or "") for row in scoped}
    clusters = []
    for cluster in _error_clusters(scoped, requests_by_id)[:20]:
        session_ids = {event_sessions.get(event_id, "") for event_id in cluster["event_ids"]}
        latest_event_id = cluster["event_ids"][-1] if cluster["event_ids"] else ""
        clusters.append({
            "fingerprint": cluster["fingerprint"], "kind": cluster["kind"],
            "component": cluster["component"], "occurrence_count": cluster["occurrence_count"],
            "recovered_count": cluster["recovered_count"], "session_count": len(session_ids - {""}),
            "recurring": cluster["recurring"], "resolution": cluster["resolution"],
            "severity": cluster["severity"], "last_seen_at": cluster["last_seen_at"],
            "conversation_id": event_sessions.get(latest_event_id, ""),
            "event_id": latest_event_id,
        })

    all_requests = [row for row in scoped if row.get("kind") == "REQUEST"]
    all_tools = [row for row in scoped if row.get("kind") in {"TOOL", "SUBTOOL"}]
    request_durations = [duration(row) for row in all_requests if duration(row) > 0]
    total_tokens = sum(usage(row) for row in all_requests)
    total_failures = sum(row["failures"] for row in sessions)
    total_recovered = sum(row["recovered_failures"] for row in sessions)
    completed_requests = sum(row.get("status") == "completed" for row in all_requests)
    return {
        "summary": {
            "sessions": len(sessions), "events": len(scoped),
            "turns": len({(row.get("session_id"), row.get("turn_id")) for row in scoped if row.get("turn_id")}),
            "requests": len(all_requests), "tool_calls": len(all_tools),
            "failures": total_failures, "recovered_failures": total_recovered,
            "success_rate": round(completed_requests / max(1, len(all_requests)), 4),
            "recovery_rate": round(total_recovered / max(1, total_recovered + total_failures), 4),
            "total_tokens": total_tokens,
            "avg_request_ms": round(statistics.fmean(request_durations)) if request_durations else 0,
            "p95_request_ms": percentile_95(request_durations),
            "window_start": window_start, "window_end": observed_now,
        },
        "baseline": baseline,
        "window_baseline": window_baseline,
        "baseline_source": baseline_source,
        "sessions": sessions,
        "regressions": regressions,
        "trends": trends,
        "providers": providers,
        "tools": tools,
        "error_clusters": clusters,
    }


def evaluate_trajectory_policies(
    analytics: dict[str, Any],
    policies: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate aggregate session/component metrics against explicit policies."""
    sessions = analytics.get("sessions") or []
    if not sessions:
        return []
    session = sessions[0]
    event_id = str(session.get("problem_event_id") or session.get("latest_event_id") or "")
    conversation_id = str(session.get("conversation_id") or "")
    providers = analytics.get("providers") or []
    tools = analytics.get("tools") or []
    results = []

    def matching_component(policy: dict[str, Any]) -> dict[str, float] | None:
        scope = str(policy.get("scope_type") or "")
        value = str(policy.get("scope_value") or "")
        if scope == "project":
            return {
                "failure_rate": float(session.get("failure_rate") or 0),
                "latency": float(session.get("avg_request_ms") or 0),
                "tokens": float(session.get("tokens_per_request") or 0),
            }
        if scope in {"provider", "model"}:
            key = scope
            rows = [
                row for row in providers
                if str(row.get(key) or "").casefold() == value.casefold()
            ]
            if not rows:
                return None
            attempts = sum(int(row.get("attempts") or 0) for row in rows)
            failures = sum(int(row.get("failures") or 0) for row in rows)
            return {
                "failure_rate": failures / max(1, attempts),
                "latency": max(float(row.get("p95_ms") or 0) for row in rows),
                "tokens": sum(float(row.get("total_tokens") or 0) for row in rows)
                / max(1, sum(int(row.get("token_requests") or 0) for row in rows)),
            }
        if scope == "tool":
            rows = [
                row for row in tools
                if str(row.get("name") or "").casefold() == value.casefold()
            ]
            if not rows:
                return None
            calls = sum(int(row.get("calls") or 0) for row in rows)
            failures = sum(int(row.get("failures") or 0) for row in rows)
            return {
                "failure_rate": failures / max(1, calls),
                "latency": max(float(row.get("p95_ms") or 0) for row in rows),
                "tokens": 0,
            }
        return None

    for policy in policies:
        if not policy.get("enabled"):
            continue
        observed = matching_component(policy)
        if observed is None:
            continue
        thresholds = policy.get("thresholds") if isinstance(policy.get("thresholds"), dict) else {}
        reasons = []
        severity = "warning"
        definitions = (
            ("failure_rate", "failure_rate_warning", "failure_rate_critical"),
            ("latency", "latency_warning_ms", "latency_critical_ms"),
            ("tokens", "tokens_warning", "tokens_critical"),
        )
        for metric, warning_key, critical_key in definitions:
            value = float(observed.get(metric) or 0)
            warning = float(thresholds.get(warning_key) or 0)
            critical = float(thresholds.get(critical_key) or 0)
            threshold = 0.0
            metric_severity = "warning"
            if critical and value >= critical:
                threshold = critical
                metric_severity = "error"
            elif warning and value >= warning:
                threshold = warning
            if not threshold:
                continue
            if metric_severity == "error":
                severity = "error"
            reasons.append({
                "metric": f"{policy.get('scope_type')}.{metric}",
                "observed": round(value, 4),
                "baseline": threshold,
                "delta": round(value - threshold, 4),
                "threshold_level": "critical" if metric_severity == "error" else "warning",
            })
        if reasons and event_id and conversation_id:
            results.append({
                "policy_id": str(policy.get("policy_id") or ""),
                "policy_name": str(policy.get("name") or "Alert policy"),
                "scope_type": str(policy.get("scope_type") or "project"),
                "scope_value": str(policy.get("scope_value") or "*"),
                "regression": {
                    "conversation_id": conversation_id,
                    "event_id": event_id,
                    "severity": severity,
                    "reasons": reasons,
                },
            })
    return results


def analyze_trajectory_slo(
    records: list[dict[str, Any]],
    *,
    target_success_rate: float = 0.99,
    window_days: int = 30,
    min_operations: int = 5,
    now: float | None = None,
) -> dict[str, Any]:
    """Calculate error-budget consumption and multi-window burn rate."""
    observed_now = float(now if now is not None else time.time())
    target = min(0.99999, max(0.5, float(target_success_rate)))
    budget_rate = max(0.00001, 1 - target)
    days = max(1, min(int(window_days), 365))
    minimum = max(1, int(min_operations))
    operations = [
        row for row in records
        if row.get("kind") in {"REQUEST", "TOOL", "SUBTOOL"}
        and row.get("started_at") is not None
    ]

    def window(label: str, seconds: int) -> dict[str, Any]:
        start = observed_now - seconds
        rows = [row for row in operations if float(row.get("started_at") or 0) >= start]
        failures = sum(row.get("status") == "failed" or bool(row.get("error")) for row in rows)
        count = len(rows)
        error_rate = failures / max(1, count)
        burn_rate = error_rate / budget_rate
        return {
            "label": label,
            "seconds": seconds,
            "operations": count,
            "failures": failures,
            "success_rate": round(1 - error_rate, 6) if count else None,
            "error_rate": round(error_rate, 6) if count else None,
            "burn_rate": round(burn_rate, 3) if count else None,
            "sufficient_data": count >= minimum,
        }

    definitions = [
        ("1h", 3_600), ("6h", 21_600), ("24h", 86_400),
        ("7d", 604_800), (f"{days}d", days * 86_400),
    ]
    windows = []
    seen_seconds = set()
    for label, seconds in definitions:
        if seconds in seen_seconds or seconds > days * 86_400:
            continue
        seen_seconds.add(seconds)
        windows.append(window(label, seconds))
    by_label = {row["label"]: row for row in windows}
    overall = windows[-1] if windows else window(f"{days}d", days * 86_400)

    def burns(label: str, threshold: float) -> bool:
        row = by_label.get(label) or {}
        return bool(row.get("sufficient_data") and float(row.get("burn_rate") or 0) >= threshold)

    if burns("1h", 14.4) and burns("6h", 6):
        alert = {"status": "critical", "reason": "fast-burn", "windows": ["1h", "6h"]}
    elif burns("6h", 6) and burns("24h", 3):
        alert = {"status": "warning", "reason": "sustained-burn", "windows": ["6h", "24h"]}
    elif burns("7d", 1) and bool(overall.get("sufficient_data")) and float(overall.get("burn_rate") or 0) >= 1:
        alert = {"status": "warning", "reason": "budget-burn", "windows": ["7d", overall["label"]]}
    else:
        alert = {
            "status": "healthy" if overall.get("sufficient_data") else "insufficient-data",
            "reason": "within-budget" if overall.get("sufficient_data") else "minimum-operations",
            "windows": [],
        }

    allowed_failures = float(overall.get("operations") or 0) * budget_rate
    failures = float(overall.get("failures") or 0)
    consumed = failures / max(budget_rate, allowed_failures) if overall.get("operations") else 0
    reference = by_label.get("6h") or by_label.get("24h") or overall
    reference_burn = float(reference.get("burn_rate") or 0)
    return {
        "target_success_rate": target,
        "window_days": days,
        "min_operations": minimum,
        "status": alert["status"],
        "alert": alert,
        "windows": windows,
        "operations": int(overall.get("operations") or 0),
        "failures": int(overall.get("failures") or 0),
        "success_rate": overall.get("success_rate"),
        "error_budget_rate": round(budget_rate, 6),
        "allowed_failures": round(allowed_failures, 3),
        "budget_consumed": round(consumed, 3),
        "budget_remaining": round(1 - consumed, 3),
        "projected_exhaustion_hours": (
            round(days * 24 / reference_burn, 1) if reference_burn > 0 else None
        ),
    }


def build_incident_dossier(
    records: list[dict[str, Any]],
    *,
    incident: dict[str, Any],
    generated_at: float | None = None,
) -> dict[str, Any]:
    """Build a privacy-bounded incident evidence package from trajectory metadata."""
    diagnostics = analyze_trajectory(records, now=generated_at)
    event_id = str(incident.get("event_id") or "")
    selected = next(
        (row for row in records if str(row.get("event_id") or "") == event_id),
        {},
    )
    selected_turn = str(selected.get("turn_id") or "")
    selected_request = str(selected.get("request_id") or "")
    chain_rows = []
    for row in records:
        row_event_id = str(row.get("event_id") or "")
        related = bool(row_event_id == event_id)
        related = related or bool(selected_turn and str(row.get("turn_id") or "") == selected_turn)
        related = related or bool(
            selected_request and str(row.get("request_id") or "") == selected_request
        )
        related = related or bool(event_id and str(row.get("parent_id") or "") == event_id)
        if not related:
            continue
        chain_rows.append({
            "event_id": row_event_id,
            "kind": str(row.get("kind") or "EVENT"),
            "status": str(row.get("status") or "unknown"),
            "duration_ms": int(row.get("duration_ms") or 0),
            "turn_id": str(row.get("turn_id") or ""),
            "request_id": str(row.get("request_id") or ""),
            "parent_id": str(row.get("parent_id") or ""),
        })
    chain_rows = chain_rows[-100:]

    selected_source = selected.get("source") if isinstance(selected.get("source"), dict) else {}
    selected_details = selected.get("details") if isinstance(selected.get("details"), dict) else {}
    selected_event = {
        "event_id": event_id,
        "kind": str(selected.get("kind") or "EVENT"),
        "status": str(selected.get("status") or "unknown"),
        "duration_ms": int(selected.get("duration_ms") or 0),
        "turn_id": selected_turn,
        "request_id": selected_request,
        "call_id": str(selected.get("call_id") or ""),
        "component": str(
            selected_details.get("name") or selected_source.get("name") or ""
        )[:160],
        "provider": str(
            selected_details.get("provider") or selected_source.get("provider") or ""
        )[:160],
        "model": str(
            selected_details.get("model") or selected_source.get("model") or ""
        )[:240],
    }

    root_causes = [{
        "root_cause_id": str(row.get("root_cause_id") or ""),
        "title": str(row.get("title") or "Execution failure"),
        "severity": str(row.get("severity") or "info"),
        "confidence": str(row.get("confidence") or "signal"),
        "score": int(row.get("score") or 0),
        "evidence_event_ids": list(row.get("evidence_event_ids") or [])[:50],
        "turn_ids": list(row.get("turn_ids") or [])[:50],
    } for row in (diagnostics.get("root_causes") or [])[:5]]
    findings = [{
        "finding_id": str(row.get("finding_id") or ""),
        "event_id": str(row.get("event_id") or ""),
        "kind": str(row.get("kind") or "EVENT"),
        "severity": str(row.get("severity") or "info"),
        "category": str(row.get("category") or "execution"),
        "next_check": str(row.get("next_check") or "")[:280],
    } for row in (diagnostics.get("findings") or [])[:50]]
    recovery_paths = []
    for path in (diagnostics.get("recovery") or {}).get("paths") or []:
        recovery_paths.append({
            "request_id": str(path.get("request_id") or ""),
            "turn_id": str(path.get("turn_id") or ""),
            "outcome": str(path.get("outcome") or "unknown"),
            "recovered": bool(path.get("recovered")),
            "attempt_count": int(path.get("attempt_count") or 0),
            "failed_attempt_count": int(path.get("failed_attempt_count") or 0),
            "final_model": str(path.get("final_model") or "")[:240],
            "steps": [{
                "event_id": str(step.get("event_id") or ""),
                "provider": str(step.get("provider") or "")[:160],
                "model": str(step.get("model") or "")[:240],
                "status": str(step.get("status") or "unknown"),
                "duration_ms": int(step.get("duration_ms") or 0),
                "retry_action": str(step.get("retry_action") or "")[:120],
            } for step in (path.get("steps") or [])[:20]],
        })
    error_clusters = [{
        "fingerprint": str(row.get("fingerprint") or ""),
        "kind": str(row.get("kind") or "EVENT"),
        "component": str(row.get("component") or "")[:160],
        "event_ids": [str(value) for value in (row.get("event_ids") or [])[:50]],
        "turn_ids": [str(value) for value in (row.get("turn_ids") or [])[:50]],
        "occurrence_count": int(row.get("occurrence_count") or 0),
        "recovered_count": int(row.get("recovered_count") or 0),
        "triaged_count": int(row.get("triaged_count") or 0),
        "resolved_count": int(row.get("resolved_count") or 0),
        "total_duration_ms": int(row.get("total_duration_ms") or 0),
        "recurring": bool(row.get("recurring")),
        "fully_recovered": bool(row.get("fully_recovered")),
        "resolution": str(row.get("resolution") or "untriaged"),
        "severity": str(row.get("severity") or "warning"),
    } for row in (diagnostics.get("error_clusters") or [])[:20]]

    root_ids = {row["root_cause_id"] for row in root_causes}
    metric = str(incident.get("metric") or "slo.burn_rate")
    recommendations = []
    if "latency" in metric or "latency" in root_ids:
        recommendations.append("Inspect the critical-path timing and the slowest provider or tool span.")
    if "failure_rate" in metric or {"tool-execution", "provider-reliability"} & root_ids:
        recommendations.append("Compare failed component fingerprints with the last healthy trajectory.")
    if "tokens" in metric:
        recommendations.append("Review context growth, compaction boundaries, and tokens per successful request.")
    if metric == "slo.burn_rate":
        recommendations.append("Confirm the fast and slow burn windows before changing the SLO target.")
    if recovery_paths:
        recommendations.append("Validate that the observed retry or fallback path is deterministic before closing.")
    if not recommendations:
        recommendations.append("Open the evidence event and inspect its causal parents and children.")

    return {
        "dossier_version": 1,
        "generated_at": datetime.fromtimestamp(
            float(generated_at if generated_at is not None else time.time()),
            tz=timezone.utc,
        ).isoformat(),
        "incident": {
            "incident_id": str(incident.get("incident_id") or incident.get("alert_id") or ""),
            "source_type": str(incident.get("source_type") or "trajectory"),
            "status": str(incident.get("status") or "open"),
            "severity": str(incident.get("severity") or "warning"),
            "reason": str(incident.get("reason") or ""),
            "metric": metric,
            "observed": float(incident.get("observed") or incident.get("burn_rate") or 0),
            "threshold": float(incident.get("baseline") or incident.get("threshold") or 0),
            "delta": float(incident.get("delta") or 0),
            "conversation_id": str(incident.get("conversation_id") or ""),
            "event_id": event_id,
            "detected_at": str(incident.get("detected_at") or ""),
            "updated_at": str(incident.get("updated_at") or ""),
        },
        "summary": {
            "health": str(diagnostics.get("health") or "unknown"),
            "errors": int(diagnostics.get("error_count") or 0),
            "warnings": int(diagnostics.get("warning_count") or 0),
            "trace_integrity": str((diagnostics.get("integrity") or {}).get("status") or "unknown"),
            "trace_coverage": int(
                ((diagnostics.get("integrity") or {}).get("coverage") or {}).get("score") or 0
            ),
            "record_count": len(records),
        },
        "selected_event": selected_event,
        "causal_chain": chain_rows,
        "root_causes": root_causes,
        "findings": findings,
        "recovery_paths": recovery_paths[:20],
        "error_clusters": error_clusters,
        "critical_path": list(diagnostics.get("critical_path") or [])[:100],
        "timing_breakdown": diagnostics.get("timing_breakdown") or {},
        "recommendations": recommendations,
        "privacy": {
            "raw_inputs_included": False,
            "raw_outputs_included": False,
            "raw_errors_included": False,
            "annotation_notes_included": False,
        },
    }


def render_incident_dossier_markdown(dossier: dict[str, Any]) -> str:
    """Render the bounded dossier without reintroducing raw trajectory payloads."""
    incident = dossier.get("incident") or {}
    summary = dossier.get("summary") or {}
    selected = dossier.get("selected_event") or {}
    lines = [
        "# Trajectory incident dossier",
        "",
        f"- Incident: `{incident.get('incident_id') or 'unknown'}`",
        f"- Source: **{incident.get('source_type') or 'trajectory'}**",
        f"- Status: **{incident.get('status') or 'unknown'}**",
        f"- Severity: **{incident.get('severity') or 'unknown'}**",
        f"- Metric: `{incident.get('metric') or 'unknown'}`",
        f"- Observed / threshold: {incident.get('observed', 0)} / {incident.get('threshold', 0)}",
        f"- Conversation: `{incident.get('conversation_id') or 'unknown'}`",
        f"- Evidence event: `{incident.get('event_id') or 'unknown'}`",
        "",
        "## Health",
        "",
        f"- Trajectory: **{summary.get('health') or 'unknown'}**",
        f"- Findings: {summary.get('errors', 0)} errors, {summary.get('warnings', 0)} warnings",
        f"- Trace integrity: **{summary.get('trace_integrity') or 'unknown'}** ({summary.get('trace_coverage', 0)}%)",
        f"- Selected event: `{selected.get('kind') or 'EVENT'}` / `{selected.get('status') or 'unknown'}` / {selected.get('duration_ms', 0)} ms",
        "",
        "## Root causes",
        "",
    ]
    causes = dossier.get("root_causes") or []
    lines.extend(
        f"- **{row.get('title') or 'Execution failure'}** — {row.get('confidence') or 'signal'}; evidence: "
        + ", ".join(f"`{event_id}`" for event_id in row.get("evidence_event_ids") or [])
        for row in causes
    )
    if not causes:
        lines.append("- No structured root cause identified.")
    lines.extend(["", "## Recommended checks", ""])
    lines.extend(f"- {item}" for item in dossier.get("recommendations") or [])
    lines.extend([
        "",
        "## Privacy boundary",
        "",
        "This dossier excludes raw inputs, outputs, error text, and annotation notes.",
        "",
    ])
    return "\n".join(lines)


def derive_incident_eval_criteria(
    dossier: dict[str, Any],
    *,
    slo_target_success_rate: float = 0.99,
) -> dict[str, Any]:
    """Turn a bounded incident dossier into deterministic regression gates."""
    incident = dossier.get("incident") if isinstance(dossier.get("incident"), dict) else {}
    metric = str(incident.get("metric") or "")
    threshold = max(0.0, float(incident.get("threshold") or 0))
    criteria: dict[str, Any] = {
        "max_error_count": 0,
        "max_sandbox_blocked": 0,
        "min_trace_coverage": 80,
        "blocked_fingerprints": sorted({
            str(row.get("fingerprint") or "")
            for row in (dossier.get("error_clusters") or [])
            if str(row.get("fingerprint") or "")
        }),
    }
    if "failure_rate" in metric:
        criteria["max_failure_rate"] = threshold
    elif "latency" in metric:
        criteria["max_avg_request_ms"] = threshold
    elif "tokens" in metric:
        criteria["max_tokens_per_request"] = threshold
    elif metric == "slo.burn_rate":
        target = min(0.99999, max(0.5, float(slo_target_success_rate)))
        criteria["max_failure_rate"] = round(1 - target, 6)
    if str((dossier.get("selected_event") or {}).get("kind") or "") == "REQUEST":
        criteria["require_completed_request"] = True
    return criteria


def evaluate_trajectory_regression(
    records: list[dict[str, Any]],
    *,
    criteria: dict[str, Any],
    baseline_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one recorded conversation against a durable regression case."""
    timestamps = [
        float(row.get("completed_at") or row.get("started_at") or 0)
        for row in records
        if row.get("completed_at") is not None or row.get("started_at") is not None
    ]
    observed_now = max(timestamps, default=time.time()) + 1
    project = analyze_project_trajectory(records, days=365, now=observed_now)
    diagnostics = analyze_trajectory(records, now=observed_now)
    summary = project.get("summary") or {}
    sessions = project.get("sessions") or []
    session = sessions[0] if sessions else {}
    fingerprints = sorted({
        str(row.get("fingerprint") or "")
        for row in (diagnostics.get("error_clusters") or [])
        if str(row.get("fingerprint") or "")
    })
    replay_tool_rows = [
        row for row in records
        if str(row.get("kind") or "") in {"TOOL", "SUBTOOL"}
        and isinstance((row.get("details") or {}).get("policy"), dict)
        and bool(((row.get("details") or {}).get("policy") or {}).get("sandbox_replay"))
    ]
    sandbox_blocked = sum(
        str(((row.get("details") or {}).get("policy") or {}).get("action") or "")
        == "blocked"
        for row in replay_tool_rows
    )
    metrics = {
        "failure_rate": float(session.get("failure_rate") or summary.get("failure_rate") or 0),
        "avg_request_ms": float(session.get("avg_request_ms") or summary.get("avg_request_ms") or 0),
        "tokens_per_request": float(
            session.get("tokens_per_request") or summary.get("tokens_per_request") or 0
        ),
        "error_count": int(diagnostics.get("error_count") or 0),
        "warning_count": int(diagnostics.get("warning_count") or 0),
        "trace_coverage": int(
            ((diagnostics.get("integrity") or {}).get("coverage") or {}).get("score") or 0
        ),
        "completed_requests": int((diagnostics.get("successes") or {}).get("completed_requests") or 0),
        "fingerprints": fingerprints,
        "health": str(diagnostics.get("health") or "unknown"),
        "record_count": len(records),
        "sandbox_tool_calls": len(replay_tool_rows),
        "sandbox_blocked_calls": sandbox_blocked,
    }
    checks = []

    def maximum(key: str, metric_key: str, label: str) -> None:
        if key not in criteria:
            return
        expected = float(criteria.get(key) or 0)
        observed = float(metrics.get(metric_key) or 0)
        checks.append({
            "check_id": key, "label": label, "operator": "<=",
            "expected": expected, "observed": observed,
            "passed": observed <= expected,
        })

    maximum("max_failure_rate", "failure_rate", "Failure rate")
    maximum("max_avg_request_ms", "avg_request_ms", "Average request latency")
    maximum("max_tokens_per_request", "tokens_per_request", "Tokens per request")
    maximum("max_error_count", "error_count", "Error findings")
    maximum("max_sandbox_blocked", "sandbox_blocked_calls", "Blocked sandbox calls")
    if "min_trace_coverage" in criteria:
        expected = int(criteria.get("min_trace_coverage") or 0)
        observed = int(metrics["trace_coverage"])
        checks.append({
            "check_id": "min_trace_coverage", "label": "Trace coverage",
            "operator": ">=", "expected": expected, "observed": observed,
            "passed": observed >= expected,
        })
    if criteria.get("require_completed_request"):
        observed = int(metrics["completed_requests"])
        checks.append({
            "check_id": "require_completed_request", "label": "Completed request",
            "operator": ">=", "expected": 1, "observed": observed,
            "passed": observed >= 1,
        })
    blocked = {
        str(value) for value in (criteria.get("blocked_fingerprints") or []) if str(value)
    }
    if blocked:
        present = sorted(blocked & set(fingerprints))
        checks.append({
            "check_id": "blocked_fingerprints", "label": "Incident fingerprints absent",
            "operator": "none", "expected": sorted(blocked), "observed": present,
            "passed": not present,
        })
    passed_count = sum(bool(row["passed"]) for row in checks)
    passed = bool(checks) and passed_count == len(checks)
    baseline = baseline_snapshot if isinstance(baseline_snapshot, dict) else {}
    baseline_metrics = baseline.get("metrics") if isinstance(baseline.get("metrics"), dict) else {}
    baseline_fingerprints = {
        str(value) for value in (baseline_metrics.get("fingerprints") or []) if str(value)
    }
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "score": round(passed_count / max(1, len(checks)) * 100, 1),
        "checks": checks,
        "metrics": metrics,
        "comparison": {
            "failure_rate_delta": round(
                metrics["failure_rate"] - float(baseline_metrics.get("failure_rate") or 0), 6
            ),
            "avg_request_ms_delta": round(
                metrics["avg_request_ms"] - float(baseline_metrics.get("avg_request_ms") or 0), 3
            ),
            "tokens_per_request_delta": round(
                metrics["tokens_per_request"] - float(
                    baseline_metrics.get("tokens_per_request") or 0
                ), 3
            ),
            "error_count_delta": metrics["error_count"] - int(
                baseline_metrics.get("error_count") or 0
            ),
            "removed_fingerprints": sorted(baseline_fingerprints - set(fingerprints)),
            "new_fingerprints": sorted(set(fingerprints) - baseline_fingerprints),
        },
        "evaluated_at": datetime.fromtimestamp(observed_now, tz=timezone.utc).isoformat(),
        "privacy": {
            "raw_inputs_included": False,
            "raw_outputs_included": False,
            "raw_errors_included": False,
        },
    }


def build_model_promotion_recommendation(
    comparisons: list[dict[str, Any]],
    *,
    current_model: str = "",
    required_wins: int = 3,
    window: int = 10,
) -> dict[str, Any]:
    """Derive a conservative, evidence-only model promotion recommendation."""
    required = max(2, min(int(required_wins), 10))
    bounded_window = max(required, min(int(window), 50))
    history = sorted(
        [row for row in comparisons if str(row.get("status") or "") != "running"],
        key=lambda row: str(row.get("completed_at") or row.get("created_at") or ""),
        reverse=True,
    )[:bounded_window]
    configured_model = str(current_model or "")[:240]
    candidate = str((history[0] if history else {}).get("winner_model") or "")[:240]
    streak_rows = []
    if candidate:
        for comparison in history:
            if (
                str(comparison.get("status") or "") != "completed"
                or str(comparison.get("winner_model") or "") != candidate
            ):
                break
            streak_rows.append(comparison)

    evidence = streak_rows[:required]
    candidate_rows = []
    current_rows = []
    for comparison in evidence:
        models = comparison.get("models") if isinstance(comparison.get("models"), list) else []
        candidate_row = next(
            (row for row in models if str(row.get("preferred_model") or "") == candidate),
            None,
        )
        if candidate_row:
            candidate_rows.append(candidate_row)
        if configured_model and configured_model != candidate:
            current_row = next(
                (
                    row for row in models
                    if str(row.get("preferred_model") or "") == configured_model
                ),
                None,
            )
            if current_row:
                current_rows.append(current_row)

    candidate_gates_passed = len(candidate_rows) == required and all(
        bool(row.get("gate_passed")) and int(row.get("error_count") or 0) == 0
        for row in candidate_rows
    )
    current_model_compared = (
        not configured_model
        or configured_model == candidate
        or len(current_rows) == required
    )
    outranked_current = (
        not configured_model
        or configured_model == candidate
        or (
            len(current_rows) == required
            and all(
                int(candidate_row.get("rank") or 999999)
                < int(current_row.get("rank") or 999999)
                for candidate_row, current_row in zip(candidate_rows, current_rows)
            )
        )
    )

    status = "hold"
    reason = "A candidate needs more consecutive comparison wins."
    if len(history) < required:
        status = "insufficient-data"
        reason = f"Complete at least {required} model comparisons."
    elif not candidate:
        reason = "The latest comparison has no viable winner."
    elif len(streak_rows) < required:
        reason = f"{candidate} has {len(streak_rows)} of {required} required consecutive wins."
    elif not candidate_gates_passed:
        reason = "The winning streak contains a blocked release gate or replay error."
    elif not current_model_compared:
        reason = "Include the configured model in every comparison used as promotion evidence."
    elif not outranked_current:
        reason = "The candidate did not consistently outrank the configured model."
    elif configured_model == candidate:
        status = "current-model-leading"
        reason = "The configured model is already the stable comparison leader."
    else:
        status = "promote"
        reason = f"{candidate} earned {required} consecutive safe comparison wins."

    divisor = max(1, len(candidate_rows))
    return {
        "status": status,
        "recommended_model": candidate if status in {"promote", "current-model-leading"} else "",
        "candidate_model": candidate,
        "current_model": configured_model,
        "consecutive_wins": len(streak_rows),
        "required_wins": required,
        "comparisons_analyzed": len(history),
        "current_model_compared": current_model_compared,
        "all_evidence_gates_passed": candidate_gates_passed,
        "avg_score": round(
            sum(float(row.get("avg_score") or 0) for row in candidate_rows) / divisor, 2
        ),
        "avg_request_ms": round(
            sum(float(row.get("avg_request_ms") or 0) for row in candidate_rows) / divisor,
            2,
        ),
        "avg_tokens_per_request": round(
            sum(float(row.get("avg_tokens_per_request") or 0) for row in candidate_rows)
            / divisor,
            2,
        ),
        "evidence_comparison_ids": [
            str(row.get("comparison_id") or "")[:100] for row in evidence
        ],
        "reason": reason,
        "automatic_change_applied": False,
        "privacy": {
            "raw_inputs_included": False,
            "raw_outputs_included": False,
            "raw_errors_included": False,
        },
    }


def analyze_model_promotion_canary(
    records: list[dict[str, Any]],
    *,
    promotion: dict[str, Any] | None,
    min_requests_per_arm: int = 10,
    max_requests_per_arm: int = 200,
    familywise_alpha: float = 0.05,
) -> dict[str, Any]:
    """Sequentially compare production candidate/control requests for one canary."""
    required = max(5, min(int(min_requests_per_arm), 1000))
    maximum = max(required, min(int(max_requests_per_arm), 10_000))
    alpha = max(0.001, min(float(familywise_alpha), 0.25))
    planning_horizon_hours = 24
    if not promotion:
        return {
            "status": "inactive",
            "promotion_id": "",
            "min_requests_per_arm": required,
            "max_requests_per_arm": maximum,
            "candidate": {},
            "control": {},
            "reasons": [],
            "confidence": {
                "method": "alpha-spending repeated-look guardrail",
                "decision": "continue",
                "familywise_confidence": round(1 - alpha, 4),
                "look": 0,
                "samples_per_arm": 0,
                "intervals": {},
            },
            "sample_plan": {
                "status": "inactive",
                "projected_decision": "continue",
                "planning_horizon_hours": planning_horizon_hours,
                "metric_plans": [],
            },
            "privacy": {
                "raw_inputs_included": False,
                "raw_outputs_included": False,
                "raw_errors_included": False,
            },
        }
    try:
        maximum = max(
            required,
            min(int(promotion.get("max_requests_per_arm") or maximum), 10_000),
        )
    except (TypeError, ValueError):
        pass
    try:
        alpha = max(
            0.001,
            min(float(promotion.get("familywise_alpha") or alpha), 0.25),
        )
    except (TypeError, ValueError):
        pass
    try:
        planning_horizon_hours = max(
            1,
            min(int(promotion.get("max_canary_hours") or 24), 24 * 30),
        )
    except (TypeError, ValueError):
        pass
    promotion_id = str(promotion.get("promotion_id") or "")
    ramp_stage = int(promotion.get("ramp_stage") or 0)

    def request_usage(row: dict[str, Any]) -> int:
        details = row.get("details") if isinstance(row.get("details"), dict) else {}
        usage = details.get("usage") if isinstance(details.get("usage"), dict) else {}
        for key in ("total_tokens", "total_token_count"):
            try:
                if usage.get(key) not in (None, ""):
                    return max(0, int(usage[key]))
            except (TypeError, ValueError):
                pass
        total = 0
        for keys in (
            ("input_tokens", "prompt_tokens", "prompt_token_count"),
            ("output_tokens", "completion_tokens", "candidates_token_count"),
        ):
            for key in keys:
                try:
                    if usage.get(key) not in (None, ""):
                        total += max(0, int(usage[key]))
                        break
                except (TypeError, ValueError):
                    continue
        return total

    arms: dict[str, list[dict[str, Any]]] = {"candidate": [], "control": []}
    for row in records:
        if str(row.get("kind") or "") != "REQUEST":
            continue
        details = row.get("details") if isinstance(row.get("details"), dict) else {}
        options = details.get("options") if isinstance(details.get("options"), dict) else {}
        if str(options.get("promotion_id") or "") != promotion_id:
            continue
        try:
            request_stage = int(options.get("ramp_stage") or 0)
        except (TypeError, ValueError):
            request_stage = 0
        if request_stage != ramp_stage:
            continue
        arm = "candidate" if bool(options.get("canary_applied")) else "control"
        arms[arm].append(row)

    def arm_values(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
        durations = [float(max(0, int(row.get("duration_ms") or 0))) for row in rows]
        return {
            "duration": [value for value in durations if value > 0],
            "tokens": [float(request_usage(row)) for row in rows],
        }

    def summarize(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
        fallbacks = sum(
            bool((row.get("details") or {}).get("fallback_used"))
            for row in rows
            if isinstance(row.get("details"), dict)
        )
        failures = sum(
            str(row.get("status") or "") == "failed"
            or bool(row.get("error"))
            or bool((row.get("details") or {}).get("fallback_used"))
            for row in rows
        )
        values = arm_values(rows)
        durations = values["duration"]
        token_total = sum(values["tokens"])
        return {
            "model": str(model or "")[:240],
            "requests": len(rows),
            "sessions": len({str(row.get("session_id") or "") for row in rows}),
            "failures": failures,
            "fallbacks": fallbacks,
            "failure_rate": round(failures / max(1, len(rows)), 4),
            "avg_request_ms": round(statistics.fmean(durations), 2) if durations else 0,
            "tokens_per_request": round(token_total / max(1, len(rows)), 2),
        }

    candidate = summarize(arms["candidate"], str(promotion.get("candidate_model") or ""))
    control = summarize(arms["control"], str(promotion.get("previous_model") or ""))
    enough_data = candidate["requests"] >= required and control["requests"] >= required
    samples_per_arm = min(int(candidate["requests"]), int(control["requests"]))
    look = int(candidate["requests"]) + int(control["requests"])
    # Allocate alpha across every possible sample-size look. Since
    # sum(1 / (t * (t + 1))) == 1, repeated evaluation keeps the configured
    # family-wise false-decision budget instead of treating every refresh as a
    # fresh fixed-sample test.
    look_alpha = alpha / max(1, look * (look + 1))
    boundary_z = statistics.NormalDist().inv_cdf(1 - look_alpha / 2)

    def interval_payload(
        *, metric: str, difference: float, standard_error: float, margin: float,
        candidate_value: float, control_value: float, samples: int,
    ) -> dict[str, Any]:
        lower = difference - boundary_z * standard_error
        upper = difference + boundary_z * standard_error
        digits = 4 if metric == "failure_rate" else 2
        return {
            "metric": metric,
            "difference": round(difference, digits),
            "lower": round(lower, digits),
            "upper": round(upper, digits),
            "margin": round(margin, digits),
            "candidate": round(candidate_value, digits),
            "control": round(control_value, digits),
            "samples_per_arm": samples,
            "non_inferior": upper <= margin,
            "harm_confirmed": lower > margin,
        }

    intervals: dict[str, dict[str, Any]] = {}
    planning_inputs: dict[str, dict[str, float | str]] = {}
    if enough_data:
        candidate_n = max(1, int(candidate["requests"]))
        control_n = max(1, int(control["requests"]))
        # Add-one smoothing keeps the uncertainty honest when an arm has zero
        # observed failures; the displayed difference remains the raw rate.
        candidate_p = (int(candidate["failures"]) + 1) / (candidate_n + 2)
        control_p = (int(control["failures"]) + 1) / (control_n + 2)
        failure_se = math.sqrt(
            candidate_p * (1 - candidate_p) / (candidate_n + 2)
            + control_p * (1 - control_p) / (control_n + 2)
        )
        intervals["failure_rate"] = interval_payload(
            metric="failure_rate",
            difference=candidate["failure_rate"] - control["failure_rate"],
            standard_error=failure_se,
            margin=0.10,
            candidate_value=candidate["failure_rate"],
            control_value=control["failure_rate"],
            samples=samples_per_arm,
        )
        planning_inputs["failure_rate"] = {
            "kind": "rate",
            "difference": candidate["failure_rate"] - control["failure_rate"],
            "margin": 0.10,
            "candidate_rate": candidate["failure_rate"],
            "control_rate": control["failure_rate"],
        }

        candidate_values = arm_values(arms["candidate"])
        control_values = arm_values(arms["control"])

        def add_mean_interval(
            metric: str,
            candidate_samples: list[float],
            control_samples: list[float],
            absolute_margin: float,
        ) -> None:
            if not candidate_samples or not control_samples:
                return
            candidate_mean = statistics.fmean(candidate_samples)
            control_mean = statistics.fmean(control_samples)
            candidate_variance = (
                statistics.variance(candidate_samples) if len(candidate_samples) > 1 else 0
            )
            control_variance = (
                statistics.variance(control_samples) if len(control_samples) > 1 else 0
            )
            standard_error = math.sqrt(
                candidate_variance / len(candidate_samples)
                + control_variance / len(control_samples)
            )
            intervals[metric] = interval_payload(
                metric=metric,
                difference=candidate_mean - control_mean,
                standard_error=standard_error,
                margin=max(absolute_margin, control_mean * 0.5),
                candidate_value=candidate_mean,
                control_value=control_mean,
                samples=min(len(candidate_samples), len(control_samples)),
            )
            planning_inputs[metric] = {
                "kind": "mean",
                "difference": candidate_mean - control_mean,
                "margin": max(absolute_margin, control_mean * 0.5),
                "candidate_variance": candidate_variance,
                "control_variance": control_variance,
            }

        add_mean_interval(
            "avg_request_ms",
            candidate_values["duration"],
            control_values["duration"],
            1000,
        )
        add_mean_interval(
            "tokens_per_request",
            candidate_values["tokens"],
            control_values["tokens"],
            500,
        )

    harmful = [row for row in intervals.values() if row["harm_confirmed"]]
    non_inferior = bool(intervals) and all(
        row["non_inferior"] for row in intervals.values()
    )
    if harmful:
        decision = "rollback"
    elif enough_data and non_inferior:
        decision = "advance"
    elif samples_per_arm >= maximum:
        decision = "inconclusive"
    else:
        decision = "continue"
    reasons = [
        {
            "metric": row["metric"],
            "candidate": row["candidate"],
            "control": row["control"],
            "threshold": row["margin"],
            "lower": row["lower"],
            "upper": row["upper"],
            "decision": "rollback",
        }
        for row in harmful
    ]
    status = {
        "advance": "healthy",
        "rollback": "regressed",
        "inconclusive": "inconclusive",
    }.get(decision, "collecting-data")

    def projected_boundary_z(projected_samples_per_arm: int) -> float:
        projected_look = max(1, projected_samples_per_arm * 2)
        projected_alpha = alpha / (projected_look * (projected_look + 1))
        return statistics.NormalDist().inv_cdf(1 - projected_alpha / 2)

    metric_plans: list[dict[str, Any]] = []
    if enough_data:
        for metric, inputs in planning_inputs.items():
            difference = float(inputs["difference"])
            margin = float(inputs["margin"])
            projected_metric_decision = (
                "advance" if difference < margin else "rollback" if difference > margin
                else "inconclusive"
            )
            target_samples: int | None = None
            for projected_samples in range(samples_per_arm, maximum + 1):
                if str(inputs["kind"]) == "rate":
                    projected_candidate_p = (
                        float(inputs["candidate_rate"]) * projected_samples + 1
                    ) / (projected_samples + 2)
                    projected_control_p = (
                        float(inputs["control_rate"]) * projected_samples + 1
                    ) / (projected_samples + 2)
                    standard_error = math.sqrt(
                        projected_candidate_p * (1 - projected_candidate_p)
                        / (projected_samples + 2)
                        + projected_control_p * (1 - projected_control_p)
                        / (projected_samples + 2)
                    )
                else:
                    standard_error = math.sqrt(
                        float(inputs["candidate_variance"]) / projected_samples
                        + float(inputs["control_variance"]) / projected_samples
                    )
                half_width = projected_boundary_z(projected_samples) * standard_error
                if (
                    projected_metric_decision == "advance"
                    and difference + half_width <= margin
                ) or (
                    projected_metric_decision == "rollback"
                    and difference - half_width > margin
                ):
                    target_samples = projected_samples
                    break
            current_interval = intervals[metric]
            uncertainty_ratio = (
                (float(current_interval["upper"]) - float(current_interval["lower"]))
                / max(abs(margin), 0.0001)
            )
            metric_plans.append({
                "metric": metric,
                "projected_decision": (
                    projected_metric_decision if target_samples is not None else "inconclusive"
                ),
                "target_requests_per_arm": target_samples,
                "additional_requests_per_arm": (
                    max(0, target_samples - samples_per_arm)
                    if target_samples is not None else None
                ),
                "uncertainty_ratio": round(uncertainty_ratio, 3),
            })

    rollback_plans = [
        row for row in metric_plans
        if row["projected_decision"] == "rollback"
        and row["target_requests_per_arm"] is not None
    ]
    advance_plans = [
        row for row in metric_plans
        if row["projected_decision"] == "advance"
        and row["target_requests_per_arm"] is not None
    ]
    projected_decision = "inconclusive"
    target_requests_per_arm: int | None = None
    limiting_metric = ""
    if decision in {"advance", "rollback"}:
        projected_decision = decision
        target_requests_per_arm = samples_per_arm
        matching = [
            row for row in metric_plans if row["projected_decision"] == decision
        ]
        if matching:
            limiting_metric = str(max(
                matching, key=lambda row: float(row["uncertainty_ratio"])
            )["metric"])
    elif rollback_plans:
        limiting = min(
            rollback_plans,
            key=lambda row: int(row["target_requests_per_arm"] or maximum),
        )
        projected_decision = "rollback"
        target_requests_per_arm = int(limiting["target_requests_per_arm"])
        limiting_metric = str(limiting["metric"])
    elif metric_plans and len(advance_plans) == len(metric_plans):
        limiting = max(
            advance_plans,
            key=lambda row: int(row["target_requests_per_arm"] or 0),
        )
        projected_decision = "advance"
        target_requests_per_arm = int(limiting["target_requests_per_arm"])
        limiting_metric = str(limiting["metric"])
    elif metric_plans:
        limiting_metric = str(max(
            metric_plans, key=lambda row: float(row["uncertainty_ratio"])
        )["metric"])

    if not enough_data:
        plan_status = "collecting-minimum"
        projected_decision = "continue"
        target_requests_per_arm = required
    elif decision in {"advance", "rollback"}:
        plan_status = "resolved"
    elif projected_decision == "inconclusive":
        plan_status = "unresolved-within-budget"
    else:
        plan_status = "projected"

    candidate_additional = max(
        0,
        int(target_requests_per_arm or maximum) - int(candidate["requests"]),
    )
    control_additional = max(
        0,
        int(target_requests_per_arm or maximum) - int(control["requests"]),
    )

    def observed_timestamp(row: dict[str, Any]) -> float | None:
        value = row.get("started_at")
        try:
            parsed = float(value)
            return parsed if math.isfinite(parsed) and parsed > 0 else None
        except (TypeError, ValueError):
            pass
        try:
            return datetime.fromisoformat(str(value)).timestamp()
        except (TypeError, ValueError):
            return None

    all_timestamps = [
        timestamp
        for row in (*arms["candidate"], *arms["control"])
        if (timestamp := observed_timestamp(row)) is not None
    ]
    candidate_rate = 0.0
    control_rate = 0.0
    if len(all_timestamps) >= 2 and max(all_timestamps) > min(all_timestamps):
        observed_hours = (max(all_timestamps) - min(all_timestamps)) / 3600
        candidate_rate = max(0, len(arms["candidate"]) - 1) / observed_hours
        control_rate = max(0, len(arms["control"]) - 1) / observed_hours
    estimated_hours: float | None = None
    if target_requests_per_arm is not None and candidate_rate > 0 and control_rate > 0:
        estimated_hours = max(
            candidate_additional / candidate_rate,
            control_additional / control_rate,
        )

    sample_plan = {
        "status": plan_status,
        "projected_decision": projected_decision,
        "limiting_metric": limiting_metric,
        "target_requests_per_arm": target_requests_per_arm,
        "additional_candidate_requests": candidate_additional,
        "additional_control_requests": control_additional,
        "max_requests_per_arm": maximum,
        "within_request_budget": (
            projected_decision != "inconclusive"
            and target_requests_per_arm is not None
            and target_requests_per_arm <= maximum
        ),
        "planning_horizon_hours": planning_horizon_hours,
        "estimated_hours": round(estimated_hours, 2) if estimated_hours is not None else None,
        "within_time_budget": (
            estimated_hours <= planning_horizon_hours
            if estimated_hours is not None else None
        ),
        "request_rates_per_hour": {
            "candidate": round(candidate_rate, 2),
            "control": round(control_rate, 2),
        },
        "metric_plans": metric_plans,
        "assumption": "Observed effect, variance, and traffic mix remain stable.",
    }
    return {
        "status": status,
        "promotion_id": promotion_id[:100],
        "ramp_stage": ramp_stage,
        "canary_percent": int(promotion.get("canary_percent") or 0),
        "min_requests_per_arm": required,
        "max_requests_per_arm": maximum,
        "sample_complete": enough_data,
        "candidate": candidate,
        "control": control,
        "deltas": {
            "failure_rate": round(candidate["failure_rate"] - control["failure_rate"], 4),
            "avg_request_ms": round(
                candidate["avg_request_ms"] - control["avg_request_ms"], 2
            ),
            "tokens_per_request": round(
                candidate["tokens_per_request"] - control["tokens_per_request"], 2
            ),
        },
        "reasons": reasons,
        "confidence": {
            "method": "alpha-spending repeated-look guardrail",
            "decision": decision,
            "familywise_confidence": round(1 - alpha, 4),
            "look": look,
            "samples_per_arm": samples_per_arm,
            "look_alpha": round(look_alpha, 8),
            "boundary_z": round(boundary_z, 4),
            "intervals": intervals,
        },
        "sample_plan": sample_plan,
        "privacy": {
            "raw_inputs_included": False,
            "raw_outputs_included": False,
            "raw_errors_included": False,
        },
    }


def analyze_trajectory(
    records: list[dict[str, Any]],
    *,
    now: float | None = None,
) -> dict[str, Any]:
    """Return causal findings, bottlenecks, and positive execution signals."""
    observed_now = float(now if now is not None else time.time())
    findings: list[dict[str, Any]] = []
    by_event: dict[str, list[dict[str, Any]]] = {}
    requests_by_id = {
        str(record.get("request_id") or record.get("event_id") or ""): record
        for record in records
        if record.get("kind") == "REQUEST"
    }

    def add(item: dict[str, Any]) -> None:
        key = str(item.get("event_id") or "")
        signature = (key, item.get("category"))
        if any((row.get("event_id"), row.get("category")) == signature for row in findings):
            return
        findings.append(item)
        if key:
            by_event.setdefault(key, []).append(item)

    for record in records:
        kind = str(record.get("kind") or "EVENT")
        status = str(record.get("status") or "")
        error = str(record.get("error") or "")
        details = record.get("details") if isinstance(record.get("details"), dict) else {}
        duration_ms = max(0, int(record.get("duration_ms") or 0))

        if kind == "ATTEMPT" and (status == "failed" or error):
            request = requests_by_id.get(str(record.get("request_id") or ""), {})
            recovered = request.get("status") == "completed"
            retry_action = str(details.get("retry_action") or "retry/fallback")
            add(_finding(
                record,
                severity="info" if recovered else "warning",
                category="recovered-attempt" if recovered else "provider-attempt",
                title=(
                    f"Provider attempt recovered via {retry_action}"
                    if recovered else f"Provider attempt failed: {error[:120]}"
                ),
                explanation=(
                    "This provider attempt failed, but a later attempt completed the parent request."
                    if recovered else "The provider attempt failed before the parent request completed."
                ),
                next_check="Inspect the attempt model, provider, timing, error, and retry action.",
            ))
            continue

        if status == "failed" or error:
            add(_failure_finding(record))
            continue

        if status == "running" and record.get("started_at") is not None:
            running_ms = max(0, int((observed_now - float(record["started_at"])) * 1000))
            if running_ms >= 120_000:
                add(_finding(
                    record,
                    severity="warning",
                    category="stalled",
                    title=f"{kind} may be stalled",
                    explanation=f"The record has remained running for {running_ms // 1000} seconds.",
                    next_check="Inspect provider/tool health and consider cancellation before retrying.",
                ))

        if kind == "REQUEST":
            if record.get("first_output_at") is not None and record.get("started_at") is not None:
                ttft_ms = max(
                    0,
                    int((float(record["first_output_at"]) - float(record["started_at"])) * 1000),
                )
                if ttft_ms >= 8_000:
                    add(_finding(
                        record,
                        severity="warning",
                        category="slow-ttft",
                        title=f"Slow first token: {ttft_ms / 1000:.1f}s",
                        explanation="Most of the wait happened before visible generation began.",
                        next_check="Inspect provider/model selection, request size, cache usage, and retries.",
                    ))
            if duration_ms >= 60_000:
                add(_finding(
                    record,
                    severity="warning",
                    category="slow-request",
                    title=f"Slow model request: {duration_ms / 1000:.1f}s",
                    explanation="This model step is a major latency contributor.",
                    next_check="Inspect Timing, Usage, Input size, model routing, and tool count.",
                ))
            if details.get("fallback_used"):
                add(_finding(
                    record,
                    severity="info",
                    category="fallback",
                    title="Fallback model was used",
                    explanation="The preferred provider/model did not serve the final request.",
                    next_check="Inspect model routing and provider availability if this was unexpected.",
                ))

        if kind in {"TOOL", "SUBTOOL"} and duration_ms >= 30_000:
            add(_finding(
                record,
                severity="warning",
                category="slow-tool",
                title=f"Slow tool call: {duration_ms / 1000:.1f}s",
                explanation="This tool is a major latency contributor on the critical path.",
                next_check="Inspect Payload, downstream service latency, result size, and retries.",
            ))

        if kind == "ASSISTANT" and status == "completed" and not _text(record.get("output")).strip():
            add(_finding(
                record,
                severity="warning",
                category="empty-output",
                title="Assistant completed with empty output",
                explanation="The request completed but produced no visible assistant content.",
                next_check="Inspect the source Request and tool calls; the model may have emitted only a tool call.",
            ))

        if kind == "VERIFICATION":
            output = record.get("output") if isinstance(record.get("output"), dict) else {}
            unsupported = int(output.get("unsupported_claims_total") or 0)
            if unsupported:
                add(_finding(
                    record,
                    severity="warning",
                    category="unsupported-claims",
                    title=f"{unsupported} unsupported claim(s)",
                    explanation="Verification found claims without sufficient recorded evidence.",
                    next_check="Open Claims and Evidence, then repair or qualify the response.",
                ))

    severity_order = {"error": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda row: (severity_order.get(str(row.get("severity")), 9), row["finding_id"]))

    duration_candidates = [
        record for record in records
        if record.get("kind") in {
            "REQUEST", "ATTEMPT", "TOOL", "SUBTOOL", "COMPACTED", "VERIFICATION"
        }
        and int(record.get("duration_ms") or 0) > 0
    ]
    critical_path = sorted(
        duration_candidates,
        key=lambda row: int(row.get("duration_ms") or 0),
        reverse=True,
    )[:5]
    bottleneck = critical_path[0] if critical_path else None

    error_count = sum(1 for row in findings if row["severity"] == "error")
    warning_count = sum(1 for row in findings if row["severity"] == "warning")
    running_count = sum(1 for row in records if row.get("status") == "running")
    health = "failed" if error_count else "degraded" if warning_count else "running" if running_count else "healthy"
    completed_requests = sum(
        1 for row in records if row.get("kind") == "REQUEST" and row.get("status") == "completed"
    )
    completed_tools = sum(
        1 for row in records
        if row.get("kind") in {"TOOL", "SUBTOOL"} and row.get("status") == "completed"
    )
    completed_verifications = sum(
        1 for row in records if row.get("kind") == "VERIFICATION" and row.get("status") == "completed"
    )

    records_by_event = {
        str(row.get("event_id") or ""): row for row in records if row.get("event_id")
    }

    def root_group(finding: dict[str, Any]) -> tuple[str, str]:
        category = str(finding.get("category") or "")
        kind = str(finding.get("kind") or "")
        if category.startswith("slow-") or category == "stalled":
            return "latency", "Latency bottleneck"
        if category == "interruption":
            return "runtime-interruption", "Runtime interruption"
        if category in {"empty-output", "unsupported-claims"}:
            return "response-quality", "Response quality"
        if kind in {"TOOL", "SUBTOOL"}:
            return "tool-execution", "Tool execution"
        if kind in {"POLICY", "APPROVAL"}:
            return "policy-gate", "Policy or approval gate"
        if category in {"fallback", "recovered-attempt", "provider-attempt"} or kind in {
            "REQUEST", "ATTEMPT"
        }:
            return "provider-reliability", "Provider/model reliability"
        if kind == "VERIFICATION":
            return "response-quality", "Response quality"
        return "execution", "Execution failure"

    grouped_causes: dict[str, dict[str, Any]] = {}
    severity_score = {"error": 100, "warning": 40, "info": 5}
    for finding in findings:
        cause_id, label = root_group(finding)
        cause = grouped_causes.setdefault(cause_id, {
            "root_cause_id": cause_id,
            "title": label,
            "severity": "info",
            "score": 0,
            "evidence_event_ids": [],
            "turn_ids": [],
            "signals": [],
        })
        severity = str(finding.get("severity") or "info")
        cause["score"] += severity_score.get(severity, 0)
        if severity_order.get(severity, 9) < severity_order.get(cause["severity"], 9):
            cause["severity"] = severity
        event_id = str(finding.get("event_id") or "")
        turn_id = str(finding.get("turn_id") or "")
        if event_id and event_id not in cause["evidence_event_ids"]:
            cause["evidence_event_ids"].append(event_id)
            cause["score"] += min(
                30,
                int((records_by_event.get(event_id) or {}).get("duration_ms") or 0) // 2000,
            )
        if turn_id and turn_id not in cause["turn_ids"]:
            cause["turn_ids"].append(turn_id)
        cause["signals"].append(str(finding.get("title") or ""))

    root_causes = sorted(
        grouped_causes.values(), key=lambda item: (-int(item["score"]), item["title"])
    )[:5]
    for cause in root_causes:
        cause["confidence"] = (
            "observed" if cause["severity"] == "error"
            else "likely" if cause["severity"] == "warning"
            else "signal"
        )
        cause["explanation"] = (
            f"{len(cause['evidence_event_ids'])} recorded signal(s): "
            + "; ".join(cause.pop("signals")[:3])
        )

    attempts_by_request: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        if row.get("kind") == "ATTEMPT" and row.get("request_id"):
            attempts_by_request.setdefault(str(row["request_id"]), []).append(row)
    recovery_paths = []
    for request_id, attempts in attempts_by_request.items():
        request = requests_by_id.get(request_id, {})
        failed_attempts = [
            row for row in attempts if row.get("status") == "failed" or row.get("error")
        ]
        if len(attempts) < 2 and not failed_attempts:
            continue
        steps = []
        for attempt in attempts:
            details = attempt.get("details") if isinstance(attempt.get("details"), dict) else {}
            steps.append({
                "event_id": str(attempt.get("event_id") or ""),
                "provider": str(details.get("provider") or (attempt.get("source") or {}).get("provider") or ""),
                "model": str(details.get("model") or (attempt.get("source") or {}).get("model") or ""),
                "status": str(attempt.get("status") or ""),
                "duration_ms": int(attempt.get("duration_ms") or 0),
                "retry_action": str(details.get("retry_action") or ""),
                "error": str(attempt.get("error") or "")[:240],
            })
        recovery_paths.append({
            "request_id": request_id,
            "turn_id": str(request.get("turn_id") or ""),
            "outcome": str(request.get("status") or "unknown"),
            "recovered": request.get("status") == "completed" and bool(failed_attempts),
            "attempt_count": len(attempts),
            "failed_attempt_count": len(failed_attempts),
            "final_model": next(
                (step["model"] for step in reversed(steps) if step["status"] == "completed"),
                "",
            ),
            "steps": steps,
        })

    for record in records:
        event_id = str(record.get("event_id") or "")
        event_findings = by_event.get(event_id, [])
        record["diagnostic"] = {
            "severity": event_findings[0]["severity"] if event_findings else "ok",
            "findings": event_findings,
        }

    return {
        "health": health,
        "error_count": error_count,
        "warning_count": warning_count,
        "running_count": running_count,
        "findings": findings,
        "bottleneck": (
            {
                "event_id": bottleneck.get("event_id"),
                "kind": bottleneck.get("kind"),
                "duration_ms": int(bottleneck.get("duration_ms") or 0),
                "preview": str((bottleneck.get("details") or {}).get("preview") or "")[:160],
            }
            if bottleneck else None
        ),
        "critical_path": [str(row.get("event_id") or "") for row in critical_path],
        "timing_breakdown": _timing_breakdown(records),
        "integrity": _trace_integrity(records),
        "error_clusters": _error_clusters(records, requests_by_id),
        "root_causes": root_causes,
        "recovery": {
            "paths": recovery_paths,
            "recovered_requests": sum(1 for path in recovery_paths if path["recovered"]),
            "failed_attempts": sum(path["failed_attempt_count"] for path in recovery_paths),
        },
        "successes": {
            "completed_requests": completed_requests,
            "completed_tools": completed_tools,
            "completed_verifications": completed_verifications,
        },
    }
