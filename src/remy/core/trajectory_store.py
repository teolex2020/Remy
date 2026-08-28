"""Durable, causal execution history for the conversation Trajectory inspector."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from remy.config.settings import settings


_SCHEMA_LOCK = threading.Lock()

_PIPELINE_SENSITIVE_KEY_PARTS = (
    "authorization", "password", "api_key", "apikey", "secret", "token",
    "private_key", "seed_phrase", "cookie",
)
_PIPELINE_INLINE_SECRET_RE = re.compile(
    r"(?i)\b(authorization|api[ _-]?key|password|secret|token)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)
_PIPELINE_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_EXECUTION_EVENT_KINDS = {
    "experiment": {
        "EXPERIMENT_PHASE", "EXPERIMENT_MODEL", "EXPERIMENT_DECISION",
        "EXPERIMENT_INTERVENTION",
    },
    "automation": {
        "AUTOMATION_TRIGGER", "AUTOMATION_STEP", "AUTOMATION_DELIVERY",
    },
    "agent_lab": {
        "AGENT_LAB_PHASE", "AGENT_LAB_TEAM", "AGENT_LAB_DECISION",
        "AGENT_LAB_ARTIFACT", "AGENT_LAB_VERIFICATION",
    },
}
_EXECUTION_SCOPE_MODULES = {
    "experiment": "remy.core.experiment_lab",
    "automation": "remy.web.routes.automation_routes",
    "agent_lab": "remy.core.agent_lab",
}
_SELF_MODIFICATION_EVENT_KINDS = {
    "SELF_MOD_PROPOSAL",
    "SELF_MOD_EVAL",
    "SELF_MOD_APPROVAL",
    "SELF_MOD_CANARY",
    "SELF_MOD_PROMOTION",
    "SELF_MOD_ROLLBACK",
}


def _now_iso(timestamp: float | None = None) -> str:
    return datetime.fromtimestamp(
        timestamp if timestamp is not None else time.time(),
        tz=timezone.utc,
    ).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            pass
    return str(value)


def _json_dump(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, separators=(",", ":"))


def _message_role(message: Any) -> str:
    name = type(message).__name__.lower()
    if "system" in name:
        return "system"
    if "human" in name or name == "usermessage":
        return "user"
    if "tool" in name:
        return "tool"
    if "ai" in name or "assistant" in name:
        return "assistant"
    return str(getattr(message, "type", "message") or "message").lower()


def _message_content(message: Any) -> Any:
    return _json_safe(getattr(message, "content", message))


def _message_payload(message: Any) -> dict[str, Any]:
    payload = {
        "role": _message_role(message),
        "content": _message_content(message),
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = _json_safe(tool_calls)
    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        payload["tool_call_id"] = str(tool_call_id)
    return payload


def _tool_payload(tool: Any) -> dict[str, Any]:
    name = str(getattr(tool, "name", "") or type(tool).__name__)
    description = str(getattr(tool, "description", "") or "")
    schema: Any = {}
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None:
        try:
            schema = args_schema.model_json_schema()
        except Exception:
            try:
                schema = args_schema.schema()
            except Exception:
                schema = str(args_schema)
    return {"name": name, "description": description, "parameters": _json_safe(schema)}


def _preview(value: Any, limit: int = 260) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(_json_safe(value), ensure_ascii=False)
        except Exception:
            text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _sanitize_pipeline_value(value: Any, *, depth: int = 0) -> Any:
    """Bound and redact pipeline telemetry before it reaches Trajectory."""
    if depth > 8:
        return "[depth limit]"
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for raw_key, item in list(value.items())[:100]:
            key = str(raw_key)
            lowered = key.lower()
            if any(part in lowered for part in _PIPELINE_SENSITIVE_KEY_PARTS):
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = _sanitize_pipeline_value(item, depth=depth + 1)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        return [
            _sanitize_pipeline_value(item, depth=depth + 1)
            for item in list(value)[:100]
        ]
    if isinstance(value, str):
        text = _PIPELINE_BEARER_RE.sub("Bearer ***REDACTED***", value)
        text = _PIPELINE_INLINE_SECRET_RE.sub(
            lambda match: f"{match.group(1)}{match.group(2)}***REDACTED***",
            text,
        )
        return text if len(text) <= 8_000 else text[:8_000] + "…[truncated]"
    return _json_safe(value)


def _context_source(content: Any) -> dict[str, Any]:
    text = str(content or "")
    lowered = text.lower()
    if "cognitive snapshot" in lowered:
        component = "acl-cognitive-brief"
    elif "background task completed" in lowered:
        component = "background-continuation"
    elif "factuality" in lowered or "evidence" in lowered:
        component = "factuality-contract"
    elif "scratchpad" in lowered:
        component = "agent-scratchpad"
    elif "session" in lowered:
        component = "session-context"
    else:
        component = "runtime-context"
    return {
        "kind": "component",
        "name": component,
        "module": "remy.core.agent",
        "trust_tier": "internal",
        "admission_reason": "Included in the effective model request",
    }


class TrajectoryStore:
    """Append-oriented SQLite store with lightweight in-process correlation."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else settings.DATA_DIR / "trajectory.sqlite3"
        self._state_lock = threading.RLock()
        self._active_turns: dict[str, str] = {}
        self._active_requests: dict[str, str] = {}
        self._last_requests: dict[str, str] = {}
        self._ensure_schema()
        self._recover_interrupted()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    @staticmethod
    def _emit_change(
        *,
        project_id: str,
        session_id: str,
        event_id: str,
        change: str,
        kind: str = "",
        status: str = "",
        sequence: int | None = None,
        fields: Iterable[str] | None = None,
    ) -> None:
        """Publish a payload-free hint after a durable trajectory mutation."""
        try:
            from remy.core.event_bus import event_bus

            payload = {
                "conversation_id": session_id,
                "project_id": project_id,
                "event_id": event_id,
                "change": change,
                "kind": str(kind or "").upper(),
                "status": str(status or ""),
                "sequence": sequence,
                "fields": sorted(str(field) for field in (fields or [])),
            }
            event_bus.emit(
                "trajectory.changed",
                {
                    "event_name": "trajectory.changed",
                    "event_domain": "trajectory",
                    "owner_project_id": project_id,
                    "payload": payload,
                    **payload,
                },
            )
        except Exception:
            # Observability must never interrupt the agent's execution path.
            pass

    @staticmethod
    def _emit_analytics_change(*, project_id: str, change: str) -> None:
        """Publish a payload-free hint for baseline and alert dashboards."""
        try:
            from remy.core.event_bus import event_bus

            event_bus.emit(
                "trajectory.analytics.changed",
                {
                    "event_name": "trajectory.analytics.changed",
                    "event_domain": "trajectory",
                    "owner_project_id": project_id,
                    "project_id": project_id,
                    "change": change,
                    "payload": {"project_id": project_id, "change": change},
                },
            )
        except Exception:
            pass

    def _ensure_schema(self) -> None:
        with _SCHEMA_LOCK, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS trajectory_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL DEFAULT '',
                    step_id TEXT NOT NULL DEFAULT '',
                    request_id TEXT NOT NULL DEFAULT '',
                    call_id TEXT NOT NULL DEFAULT '',
                    parent_id TEXT NOT NULL DEFAULT '',
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_json TEXT NOT NULL DEFAULT '{}',
                    input_json TEXT NOT NULL DEFAULT 'null',
                    output_json TEXT NOT NULL DEFAULT 'null',
                    schema_json TEXT NOT NULL DEFAULT 'null',
                    details_json TEXT NOT NULL DEFAULT '{}',
                    started_at REAL,
                    first_output_at REAL,
                    completed_at REAL,
                    duration_ms INTEGER,
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_trajectory_session_seq
                    ON trajectory_events(project_id, session_id, id);
                CREATE INDEX IF NOT EXISTS idx_trajectory_request
                    ON trajectory_events(request_id, id);
                CREATE INDEX IF NOT EXISTS idx_trajectory_call
                    ON trajectory_events(call_id, id);
                CREATE TABLE IF NOT EXISTS session_event_log (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    ledger_event_id TEXT NOT NULL UNIQUE,
                    subject_event_id TEXT NOT NULL,
                    event_version INTEGER NOT NULL,
                    project_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL DEFAULT '',
                    event_type TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    changed_fields_json TEXT NOT NULL DEFAULT '[]',
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    UNIQUE(subject_event_id, event_version)
                );
                CREATE INDEX IF NOT EXISTS idx_session_event_log_stream
                    ON session_event_log(project_id, session_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_session_event_log_subject
                    ON session_event_log(subject_event_id, event_version);
                CREATE TABLE IF NOT EXISTS trajectory_annotations (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    label TEXT NOT NULL DEFAULT '',
                    note TEXT NOT NULL DEFAULT '',
                    bookmarked INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_trajectory_annotations_session
                    ON trajectory_annotations(project_id, session_id, updated_at);
                CREATE TABLE IF NOT EXISTS trajectory_analytics_baselines (
                    baseline_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    days INTEGER NOT NULL,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    summary_json TEXT NOT NULL DEFAULT '{}',
                    source_event_count INTEGER NOT NULL DEFAULT 0,
                    active INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_trajectory_baselines_project
                    ON trajectory_analytics_baselines(project_id, active, created_at);
                CREATE TABLE IF NOT EXISTS trajectory_regression_alerts (
                    alert_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    baseline_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    observed REAL NOT NULL DEFAULT 0,
                    baseline REAL NOT NULL DEFAULT 0,
                    delta REAL NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'open',
                    detected_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(project_id, baseline_id, session_id, event_id, metric)
                );
                CREATE INDEX IF NOT EXISTS idx_trajectory_alerts_project
                    ON trajectory_regression_alerts(project_id, status, updated_at);
                CREATE TABLE IF NOT EXISTS trajectory_alert_policies (
                    policy_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    scope_type TEXT NOT NULL,
                    scope_value TEXT NOT NULL DEFAULT '*',
                    thresholds_json TEXT NOT NULL DEFAULT '{}',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_trajectory_policies_project
                    ON trajectory_alert_policies(project_id, enabled, scope_type, updated_at);
                CREATE TABLE IF NOT EXISTS trajectory_alert_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    status TEXT NOT NULL,
                    actor TEXT NOT NULL DEFAULT 'system',
                    details_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_trajectory_alert_history_project
                    ON trajectory_alert_history(project_id, history_id);
                CREATE TABLE IF NOT EXISTS trajectory_slo_config (
                    project_id TEXT PRIMARY KEY,
                    target_success_rate REAL NOT NULL DEFAULT 0.99,
                    window_days INTEGER NOT NULL DEFAULT 30,
                    min_operations INTEGER NOT NULL DEFAULT 5,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trajectory_slo_incidents (
                    incident_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    burn_rate REAL NOT NULL DEFAULT 0,
                    windows_json TEXT NOT NULL DEFAULT '[]',
                    operations INTEGER NOT NULL DEFAULT 0,
                    failures INTEGER NOT NULL DEFAULT 0,
                    conversation_id TEXT NOT NULL DEFAULT '',
                    event_id TEXT NOT NULL DEFAULT '',
                    detected_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(project_id, reason)
                );
                CREATE INDEX IF NOT EXISTS idx_trajectory_slo_incidents_project
                    ON trajectory_slo_incidents(project_id, status, updated_at);
                CREATE TABLE IF NOT EXISTS trajectory_eval_cases (
                    case_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    incident_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    source_conversation_id TEXT NOT NULL,
                    criteria_json TEXT NOT NULL DEFAULT '{}',
                    baseline_snapshot_json TEXT NOT NULL DEFAULT '{}',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(project_id, incident_id)
                );
                CREATE INDEX IF NOT EXISTS idx_trajectory_eval_cases_project
                    ON trajectory_eval_cases(project_id, enabled, updated_at);
                CREATE TABLE IF NOT EXISTS trajectory_eval_runs (
                    run_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    candidate_conversation_id TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'recorded',
                    status TEXT NOT NULL,
                    score REAL NOT NULL DEFAULT 0,
                    checks_json TEXT NOT NULL DEFAULT '[]',
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    comparison_json TEXT NOT NULL DEFAULT '{}',
                    replay_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_trajectory_eval_runs_case
                    ON trajectory_eval_runs(project_id, case_id, created_at);
                CREATE TABLE IF NOT EXISTS trajectory_replay_sessions (
                    session_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_trajectory_replay_sessions_project
                    ON trajectory_replay_sessions(project_id, created_at);
                CREATE TABLE IF NOT EXISTS trajectory_eval_matrices (
                    matrix_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    preferred_model TEXT NOT NULL DEFAULT '',
                    agent_version TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'running',
                    case_count INTEGER NOT NULL DEFAULT 0,
                    passed_count INTEGER NOT NULL DEFAULT 0,
                    failed_count INTEGER NOT NULL DEFAULT 0,
                    error_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_trajectory_eval_matrices_project
                    ON trajectory_eval_matrices(project_id, created_at);
                CREATE TABLE IF NOT EXISTS trajectory_eval_matrix_entries (
                    entry_id TEXT PRIMARY KEY,
                    matrix_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    run_id TEXT NOT NULL DEFAULT '',
                    candidate_conversation_id TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    score REAL NOT NULL DEFAULT 0,
                    failure_rate REAL NOT NULL DEFAULT 0,
                    avg_request_ms REAL NOT NULL DEFAULT 0,
                    tokens_per_request REAL NOT NULL DEFAULT 0,
                    error_type TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    UNIQUE(matrix_id, case_id)
                );
                CREATE INDEX IF NOT EXISTS idx_trajectory_eval_matrix_entries_matrix
                    ON trajectory_eval_matrix_entries(project_id, matrix_id, created_at);
                CREATE TABLE IF NOT EXISTS trajectory_eval_comparisons (
                    comparison_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    agent_version TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'running',
                    model_count INTEGER NOT NULL DEFAULT 0,
                    case_count INTEGER NOT NULL DEFAULT 0,
                    winner_model TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_trajectory_eval_comparisons_project
                    ON trajectory_eval_comparisons(project_id, created_at);
                CREATE TABLE IF NOT EXISTS trajectory_eval_comparison_models (
                    comparison_model_id TEXT PRIMARY KEY,
                    comparison_id TEXT NOT NULL,
                    matrix_id TEXT NOT NULL DEFAULT '',
                    project_id TEXT NOT NULL,
                    preferred_model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    gate_passed INTEGER NOT NULL DEFAULT 0,
                    passed_count INTEGER NOT NULL DEFAULT 0,
                    failed_count INTEGER NOT NULL DEFAULT 0,
                    error_count INTEGER NOT NULL DEFAULT 0,
                    avg_score REAL NOT NULL DEFAULT 0,
                    avg_failure_rate REAL NOT NULL DEFAULT 0,
                    avg_request_ms REAL NOT NULL DEFAULT 0,
                    avg_tokens_per_request REAL NOT NULL DEFAULT 0,
                    rank INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    UNIQUE(comparison_id, preferred_model)
                );
                CREATE INDEX IF NOT EXISTS idx_trajectory_eval_comparison_models
                    ON trajectory_eval_comparison_models(project_id, comparison_id, rank);
                CREATE TABLE IF NOT EXISTS trajectory_model_promotions (
                    promotion_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    previous_model TEXT NOT NULL,
                    candidate_model TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'canary',
                    canary_percent INTEGER NOT NULL DEFAULT 10,
                    ramp_stage INTEGER NOT NULL DEFAULT 0,
                    ramp_complete INTEGER NOT NULL DEFAULT 0,
                    healthy_windows INTEGER NOT NULL DEFAULT 0,
                    required_healthy_windows INTEGER NOT NULL DEFAULT 2,
                    health_window_requests INTEGER NOT NULL DEFAULT 1,
                    health_window_pause_seconds INTEGER NOT NULL DEFAULT 300,
                    max_requests_per_arm INTEGER NOT NULL DEFAULT 200,
                    familywise_alpha REAL NOT NULL DEFAULT 0.05,
                    max_canary_hours INTEGER NOT NULL DEFAULT 24,
                    last_health_candidate_requests INTEGER NOT NULL DEFAULT 0,
                    last_health_control_requests INTEGER NOT NULL DEFAULT 0,
                    last_telemetry_candidate_requests INTEGER NOT NULL DEFAULT 0,
                    last_telemetry_control_requests INTEGER NOT NULL DEFAULT 0,
                    last_health_window_at TEXT NOT NULL DEFAULT '',
                    stage_started_at TEXT NOT NULL DEFAULT '',
                    ramp_history_json TEXT NOT NULL DEFAULT '[]',
                    healthy_comparisons INTEGER NOT NULL DEFAULT 0,
                    required_healthy_comparisons INTEGER NOT NULL DEFAULT 2,
                    evidence_json TEXT NOT NULL DEFAULT '[]',
                    last_comparison_id TEXT NOT NULL DEFAULT '',
                    rollback_reason TEXT NOT NULL DEFAULT '',
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    promoted_at TEXT NOT NULL DEFAULT '',
                    completed_at TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_trajectory_model_promotions_project
                    ON trajectory_model_promotions(project_id, started_at);
                """
            )
            eval_run_columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(trajectory_eval_runs)").fetchall()
            }
            if "mode" not in eval_run_columns:
                conn.execute(
                    "ALTER TABLE trajectory_eval_runs ADD COLUMN mode TEXT NOT NULL DEFAULT 'recorded'"
                )
            if "replay_json" not in eval_run_columns:
                conn.execute(
                    "ALTER TABLE trajectory_eval_runs ADD COLUMN replay_json TEXT NOT NULL DEFAULT '{}'"
                )
            matrix_entry_columns = {
                str(row["name"])
                for row in conn.execute(
                    "PRAGMA table_info(trajectory_eval_matrix_entries)"
                ).fetchall()
            }
            for column in ("failure_rate", "avg_request_ms", "tokens_per_request"):
                if column not in matrix_entry_columns:
                    conn.execute(
                        f"ALTER TABLE trajectory_eval_matrix_entries "
                        f"ADD COLUMN {column} REAL NOT NULL DEFAULT 0"
                    )
            promotion_columns = {
                str(row["name"])
                for row in conn.execute(
                    "PRAGMA table_info(trajectory_model_promotions)"
                ).fetchall()
            }
            promotion_migrations = {
                "ramp_stage": "INTEGER NOT NULL DEFAULT 0",
                "ramp_complete": "INTEGER NOT NULL DEFAULT 0",
                "healthy_windows": "INTEGER NOT NULL DEFAULT 0",
                "required_healthy_windows": "INTEGER NOT NULL DEFAULT 2",
                "health_window_requests": "INTEGER NOT NULL DEFAULT 1",
                "health_window_pause_seconds": "INTEGER NOT NULL DEFAULT 300",
                "max_requests_per_arm": "INTEGER NOT NULL DEFAULT 200",
                "familywise_alpha": "REAL NOT NULL DEFAULT 0.05",
                "max_canary_hours": "INTEGER NOT NULL DEFAULT 24",
                "last_health_candidate_requests": "INTEGER NOT NULL DEFAULT 0",
                "last_health_control_requests": "INTEGER NOT NULL DEFAULT 0",
                "last_telemetry_candidate_requests": "INTEGER NOT NULL DEFAULT 0",
                "last_telemetry_control_requests": "INTEGER NOT NULL DEFAULT 0",
                "last_health_window_at": "TEXT NOT NULL DEFAULT ''",
                "stage_started_at": "TEXT NOT NULL DEFAULT ''",
                "ramp_history_json": "TEXT NOT NULL DEFAULT '[]'",
            }
            for column, definition in promotion_migrations.items():
                if column not in promotion_columns:
                    conn.execute(
                        f"ALTER TABLE trajectory_model_promotions "
                        f"ADD COLUMN {column} {definition}"
                    )
            legacy_active_promotions = conn.execute(
                """SELECT promotion_id, started_at FROM trajectory_model_promotions
                   WHERE status IN ('canary', 'ready') AND ramp_history_json='[]'"""
            ).fetchall()
            for row in legacy_active_promotions:
                started_at = str(row["started_at"] or _now_iso())
                conn.execute(
                    """UPDATE trajectory_model_promotions SET
                           canary_percent=10, ramp_stage=0, ramp_complete=0,
                           healthy_windows=0, health_window_requests=1,
                           last_telemetry_candidate_requests=0,
                           last_telemetry_control_requests=0, stage_started_at=?,
                           ramp_history_json=?, updated_at=?
                       WHERE promotion_id=?""",
                    (
                        started_at,
                        _json_dump([{
                            "stage": 0, "percent": 10, "at": started_at,
                            "reason": "progressive_ramp_migration",
                        }]),
                        _now_iso(), row["promotion_id"],
                    ),
                )
            self._backfill_session_event_log(conn)

    @staticmethod
    def _session_event_payload(row: sqlite3.Row) -> dict[str, Any]:
        """Normalize one materialized trajectory row into the replay contract."""
        payload = dict(row)
        payload["projection_sequence"] = int(payload.pop("id"))
        for column in (
            "source_json", "input_json", "output_json", "schema_json", "details_json"
        ):
            target = column.removesuffix("_json")
            raw = payload.pop(column, None)
            try:
                payload[target] = json.loads(raw or "null")
            except (json.JSONDecodeError, TypeError):
                payload[target] = raw
        payload["model_visible"] = str(payload.get("kind") or "") in {
            "SYSTEM", "USER", "CONTEXT", "REQUEST", "ASSISTANT", "TOOL", "SUBTOOL"
        }
        payload["contract_version"] = 1
        return payload

    @classmethod
    def _append_session_event_log(
        cls,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        event_type: str,
        changed_fields: Iterable[str] = (),
    ) -> int:
        previous = conn.execute(
            """SELECT MAX(event_version) AS version FROM session_event_log
               WHERE subject_event_id=?""",
            (row["event_id"],),
        ).fetchone()
        version = int(previous["version"] or 0) + 1
        payload = cls._session_event_payload(row)
        encoded = _json_dump(payload)
        cursor = conn.execute(
            """INSERT INTO session_event_log(
                   ledger_event_id, subject_event_id, event_version,
                   project_id, session_id, turn_id, event_type, kind, status,
                   changed_fields_json, payload_json, payload_sha256, recorded_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                f"session-event-{uuid.uuid4().hex}",
                str(row["event_id"]),
                version,
                str(row["project_id"]),
                str(row["session_id"]),
                str(row["turn_id"] or ""),
                str(event_type),
                str(row["kind"]),
                str(row["status"]),
                _json_dump(sorted({str(field) for field in changed_fields})),
                encoded,
                hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                _now_iso(),
            ),
        )
        return int(cursor.lastrowid)

    @classmethod
    def _backfill_session_event_log(cls, conn: sqlite3.Connection) -> None:
        """Adopt pre-ledger trajectory rows once without inventing old revisions."""
        rows = conn.execute(
            """SELECT trajectory_events.* FROM trajectory_events
               LEFT JOIN session_event_log
                 ON session_event_log.subject_event_id=trajectory_events.event_id
               WHERE session_event_log.subject_event_id IS NULL
               ORDER BY trajectory_events.id"""
        ).fetchall()
        for row in rows:
            cls._append_session_event_log(
                conn,
                row,
                event_type="event.imported",
                changed_fields=("legacy_snapshot",),
            )

    def _recover_interrupted(self) -> None:
        """Close records left running by a previous process without replaying work."""
        now = time.time()
        with self._connect() as conn:
            interrupted = conn.execute(
                "SELECT event_id FROM trajectory_events WHERE status='running'"
            ).fetchall()
            conn.execute(
                """UPDATE trajectory_events
                   SET status='failed', completed_at=?,
                       duration_ms=CASE
                           WHEN started_at IS NULL THEN 0
                           ELSE MAX(0, CAST((? - started_at) * 1000 AS INTEGER))
                       END,
                       error=CASE WHEN error='' THEN 'interrupted by process restart' ELSE error END
                   WHERE status='running'""",
                (now, now),
            )
            for interrupted_row in interrupted:
                recovered = conn.execute(
                    "SELECT * FROM trajectory_events WHERE event_id=?",
                    (interrupted_row["event_id"],),
                ).fetchone()
                if recovered:
                    self._append_session_event_log(
                        conn,
                        recovered,
                        event_type="event.recovered",
                        changed_fields=("status", "completed_at", "duration_ms", "error"),
                    )
            conn.execute(
                """UPDATE trajectory_eval_matrices
                   SET status='failed',
                       error_count=MAX(
                           error_count,
                           case_count - passed_count - failed_count - error_count
                       ),
                       completed_at=?
                   WHERE status='running'""",
                (_now_iso(now),),
            )
            conn.execute(
                """UPDATE trajectory_eval_comparisons
                   SET status='failed', completed_at=?
                   WHERE status='running'""",
                (_now_iso(now),),
            )

    def _append(
        self,
        *,
        session_id: str,
        project_id: str,
        kind: str,
        status: str = "completed",
        turn_id: str = "",
        step_id: str = "",
        request_id: str = "",
        call_id: str = "",
        parent_id: str = "",
        source: Any = None,
        input_value: Any = None,
        output_value: Any = None,
        schema: Any = None,
        details: Any = None,
        started_at: float | None = None,
        first_output_at: float | None = None,
        completed_at: float | None = None,
        duration_ms: int | None = None,
        error: str = "",
        event_id: str | None = None,
    ) -> str:
        event_id = event_id or f"evt-{uuid.uuid4().hex}"
        created = time.time()
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO trajectory_events(
                    event_id, session_id, project_id, turn_id, step_id,
                    request_id, call_id, parent_id, kind, status,
                    source_json, input_json, output_json, schema_json,
                    details_json, started_at, first_output_at, completed_at,
                    duration_ms, error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event_id, session_id, project_id, turn_id, step_id,
                    request_id, call_id, parent_id, kind.upper(), status,
                    _json_dump(source or {}), _json_dump(input_value),
                    _json_dump(output_value), _json_dump(schema),
                    _json_dump(details or {}), started_at, first_output_at,
                    completed_at, duration_ms, str(error or ""), _now_iso(created),
                ),
            )
            sequence = int(cursor.lastrowid)
            row = conn.execute(
                "SELECT * FROM trajectory_events WHERE event_id=?",
                (event_id,),
            ).fetchone()
            self._append_session_event_log(
                conn,
                row,
                event_type="event.created",
                changed_fields=("*",),
            )
        self._emit_change(
            project_id=project_id,
            session_id=session_id,
            event_id=event_id,
            change="append",
            kind=kind,
            status=status,
            sequence=sequence,
        )
        return event_id

    def _update(self, event_id: str, **changes: Any) -> None:
        allowed = {
            "status", "output_json", "details_json", "first_output_at",
            "completed_at", "duration_ms", "error",
        }
        assignments = []
        values = []
        for key, value in changes.items():
            if key not in allowed:
                continue
            if key in {"output_json", "details_json"}:
                value = _json_dump(value)
            assignments.append(f"{key}=?")
            values.append(value)
        if not assignments:
            return
        row = None
        with self._connect() as conn:
            conn.execute(
                f"UPDATE trajectory_events SET {', '.join(assignments)} WHERE event_id=?",
                (*values, event_id),
            )
            row = conn.execute(
                """SELECT * FROM trajectory_events WHERE event_id=?""",
                (event_id,),
            ).fetchone()
            if row:
                self._append_session_event_log(
                    conn,
                    row,
                    event_type="event.updated",
                    changed_fields=(key for key in changes if key in allowed),
                )
        if row:
            self._emit_change(
                project_id=str(row["project_id"]),
                session_id=str(row["session_id"]),
                event_id=event_id,
                change="update",
                kind=str(row["kind"]),
                status=str(row["status"]),
                sequence=int(row["id"]),
                fields=(key for key in changes if key in allowed),
            )

    def begin_turn(
        self,
        *,
        session_id: str,
        project_id: str,
        content: Any,
        source: dict[str, Any] | None = None,
        input_kind: str = "USER",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        turn_id = f"turn-{uuid.uuid4().hex}"
        started = time.time()
        with self._state_lock:
            self._active_turns[session_id] = turn_id
            self._active_requests.pop(session_id, None)
            self._last_requests.pop(session_id, None)
        self._append(
            session_id=session_id,
            project_id=project_id,
            kind=input_kind,
            turn_id=turn_id,
            status="completed",
            source=source or {
                "kind": "user",
                "channel": "desktop",
                "trust_tier": "user",
            },
            input_value=content,
            output_value=content,
            details={"preview": _preview(content), **dict(metadata or {})},
            started_at=started,
            completed_at=started,
            duration_ms=0,
        )
        return turn_id

    def record_context(
        self,
        *,
        session_id: str,
        project_id: str,
        content: Any,
        source: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        with self._state_lock:
            turn_id = self._active_turns.get(session_id, "")
        if not turn_id:
            return ""
        now = time.time()
        return self._append(
            session_id=session_id,
            project_id=project_id,
            kind="CONTEXT",
            turn_id=turn_id,
            status="completed",
            source=source,
            input_value=content,
            output_value=content,
            details={"preview": _preview(content), **dict(metadata or {})},
            started_at=now,
            completed_at=now,
            duration_ms=0,
        )

    def record_fork(
        self,
        *,
        session_id: str,
        project_id: str,
        source_session_id: str,
        boundary_event_id: str,
        boundary_sequence: int,
        boundary_kind: str,
        copied_messages: int,
        preferred_model: str = "",
    ) -> str:
        """Record branch provenance without replaying any source side effect."""
        now = time.time()
        turn_id = f"fork-{uuid.uuid4().hex}"
        source_ref = {
            "conversation_id": source_session_id,
            "event_id": boundary_event_id,
            "sequence": int(boundary_sequence or 0),
            "kind": str(boundary_kind or ""),
        }
        return self._append(
            session_id=session_id,
            project_id=project_id,
            kind="FORK",
            status="completed",
            turn_id=turn_id,
            source={
                "kind": "session-fork",
                "module": "remy.web.routes.trajectory_routes",
                "trust_tier": "operator-controlled",
            },
            input_value=source_ref,
            output_value={
                "conversation_id": session_id,
                "copied_messages": max(0, int(copied_messages or 0)),
                "auto_executed": False,
            },
            details={
                "preview": (
                    f"Forked at {boundary_kind or 'event'} "
                    f"#{int(boundary_sequence or 0)} · {copied_messages} messages"
                ),
                "source_conversation_id": source_session_id,
                "boundary_event_id": boundary_event_id,
                "boundary_sequence": int(boundary_sequence or 0),
                "boundary_kind": str(boundary_kind or ""),
                "copied_messages": max(0, int(copied_messages or 0)),
                "preferred_model": str(preferred_model or ""),
                "auto_executed": False,
            },
            started_at=now,
            completed_at=now,
            duration_ms=0,
        )

    def record_child_lifecycle(
        self,
        *,
        session_id: str,
        project_id: str,
        child_id: str,
        event_type: str,
        status: str,
        run_id: str = "",
        attempt_id: str = "",
        generation: int = 0,
        summary: str = "",
        details: dict[str, Any] | None = None,
    ) -> str:
        """Expose child lifecycle transitions in the parent Trajectory stream."""
        now = time.time()
        lifecycle = str(event_type or "child.updated").strip().lower()
        kind = "SETTLEMENT" if lifecycle == "child.settled" else "CHILD"
        safe_details = dict(details or {})
        safe_details.update(
            {
                "child_id": str(child_id),
                "lifecycle_event": lifecycle,
                "run_id": str(run_id or ""),
                "attempt_id": str(attempt_id or ""),
                "generation": max(0, int(generation or 0)),
                "preview": _preview(summary or lifecycle),
            }
        )
        return self._append(
            session_id=str(session_id or f"child:{child_id}"),
            project_id=str(project_id),
            kind=kind,
            status=str(status or "completed"),
            parent_id=str(child_id),
            source={
                "kind": "child-session",
                "module": "remy.core.child_sessions",
                "trust_tier": "internal-derived",
            },
            input_value={
                "child_id": str(child_id),
                "event": lifecycle,
                "run_id": str(run_id or ""),
            },
            output_value={
                "status": str(status or ""),
                "summary": str(summary or "")[:4_000],
            },
            details=safe_details,
            started_at=now,
            completed_at=now,
            duration_ms=0,
        )

    def record_ptc_run(
        self,
        *,
        project_id: str,
        session_id: str,
        run_id: str,
        program_hash: str,
        receipt: dict[str, Any],
    ) -> str:
        """Expose one bounded PTC execution as an inspectable Trajectory event."""
        now = time.time()
        status = str(receipt.get("status") or "completed")
        usage = dict(receipt.get("usage") or {})
        limits = dict(receipt.get("limits") or {})
        step_count = len(receipt.get("steps") or [])
        stop_reason = str(receipt.get("stop_reason") or "")
        preview = f"PTC {status} · {step_count} read-only step(s)"
        if stop_reason:
            preview += f" · {stop_reason}"
        return self._append(
            session_id=str(session_id or f"ptc:{run_id}"),
            project_id=str(project_id),
            kind="PTC",
            status=status,
            parent_id=str(run_id),
            source={
                "kind": "ptc-pilot",
                "module": "remy.core.ptc_pilot",
                "trust_tier": "internal-policy",
            },
            input_value={
                "run_id": str(run_id),
                "program_hash": str(program_hash),
                "read_only": True,
            },
            output_value=receipt,
            details={
                "preview": preview,
                "run_id": str(run_id),
                "program_hash": str(program_hash),
                "read_only": True,
                "limits": limits,
                "usage": usage,
                "stop_reason": stop_reason,
                "step_count": step_count,
            },
            started_at=now - (max(0, int(usage.get("elapsed_ms") or 0)) / 1_000),
            completed_at=now,
            duration_ms=max(0, int(usage.get("elapsed_ms") or 0)),
        )

    def begin_pipeline_run(
        self,
        *,
        project_id: str,
        session_id: str,
        pipeline_id: str,
        pipeline_name: str,
        run_id: str,
        attempt_id: str = "",
        input_value: Any = None,
        definition_hash: str = "",
        steps: list[dict[str, Any]] | None = None,
        trigger: str = "manual",
    ) -> str:
        """Open a pipeline span under the active conversation turn."""
        with self._state_lock:
            turn_id = self._active_turns.get(session_id, "")
        if not turn_id:
            return ""
        safe_steps = [
            {
                "id": str(step.get("id") or ""),
                "type": str(step.get("type") or ""),
                "label": str(step.get("label") or ""),
            }
            for step in list(steps or [])[:100]
            if isinstance(step, dict)
        ]
        name = str(pipeline_name or pipeline_id or "Pipeline")
        return self._append(
            event_id=f"pipeline-run-{uuid.uuid4().hex}",
            session_id=str(session_id),
            project_id=str(project_id),
            kind="PIPELINE_RUN",
            status="running",
            turn_id=turn_id,
            source={
                "kind": "pipeline",
                "module": "remy.web.routes.pipeline_routes",
                "trust_tier": "operator-configured",
                "pipeline_id": str(pipeline_id),
                "pipeline_name": name,
                "run_id": str(run_id),
                "attempt_id": str(attempt_id or ""),
                "definition_hash": str(definition_hash or ""),
                "trigger": str(trigger or "manual"),
            },
            input_value=_sanitize_pipeline_value(input_value),
            schema=safe_steps,
            details={
                "name": name,
                "preview": f"{name} · {len(safe_steps)} step(s)",
                "pipeline_id": str(pipeline_id),
                "pipeline_name": name,
                "run_id": str(run_id),
                "attempt_id": str(attempt_id or ""),
                "definition_hash": str(definition_hash or ""),
                "trigger": str(trigger or "manual"),
                "step_count": len(safe_steps),
            },
            started_at=time.time(),
        )

    def begin_pipeline_step(
        self,
        *,
        parent_event_id: str,
        step_id: str,
        step_type: str,
        label: str,
        index: int,
        input_value: Any = None,
    ) -> str:
        """Open one inspectable pipeline block execution."""
        parent = self._row_for_event(event_id=parent_event_id)
        if not parent or str(parent["kind"]) != "PIPELINE_RUN":
            return ""
        parent_details = json.loads(parent["details_json"] or "{}")
        safe_label = str(label or step_id or f"Step {index + 1}")
        return self._append(
            event_id=f"pipeline-step-{uuid.uuid4().hex}",
            session_id=str(parent["session_id"]),
            project_id=str(parent["project_id"]),
            kind="PIPELINE_STEP",
            status="running",
            turn_id=str(parent["turn_id"] or ""),
            step_id=str(step_id or ""),
            parent_id=str(parent_event_id),
            source={
                "kind": "pipeline-block",
                "module": "remy.core.pipeline_runner",
                "trust_tier": "operator-configured",
                "step_type": str(step_type or "step"),
                "step_id": str(step_id or ""),
                "pipeline_id": str(parent_details.get("pipeline_id") or ""),
                "run_id": str(parent_details.get("run_id") or ""),
                "definition_hash": str(parent_details.get("definition_hash") or ""),
            },
            input_value=_sanitize_pipeline_value(input_value),
            details={
                "name": safe_label,
                "preview": f"{safe_label} · running",
                "index": max(0, int(index)),
                "step_id": str(step_id or ""),
                "step_type": str(step_type or "step"),
                "pipeline_id": str(parent_details.get("pipeline_id") or ""),
                "run_id": str(parent_details.get("run_id") or ""),
                "definition_hash": str(parent_details.get("definition_hash") or ""),
            },
            started_at=time.time(),
        )

    def complete_pipeline_step(
        self,
        *,
        event_id: str,
        output: Any = None,
        error: str = "",
        route_outputs: list[str] | None = None,
    ) -> None:
        """Close a pipeline block while preserving its redacted input/output."""
        row = self._row_for_event(event_id=event_id)
        if not row or str(row["kind"]) != "PIPELINE_STEP":
            return
        now = time.time()
        started = float(row["started_at"] or now)
        details = json.loads(row["details_json"] or "{}")
        selected_routes = [str(item) for item in list(route_outputs or [])[:20]]
        safe_output = _sanitize_pipeline_value(output)
        details.update({
            "preview": _preview(error or safe_output or details.get("name") or "Pipeline step"),
            "route_outputs": selected_routes,
        })
        self._update(
            event_id,
            status="failed" if error else "completed",
            output_json=safe_output,
            details_json=details,
            first_output_at=now if not error and output not in (None, "") else None,
            completed_at=now,
            duration_ms=max(0, int((now - started) * 1000)),
            error=str(error or ""),
        )

    def record_pipeline_route(
        self,
        *,
        step_event_id: str,
        selected_outputs: list[str],
    ) -> str:
        """Record an explicit branch decision made by a pipeline block."""
        step = self._row_for_event(event_id=step_event_id)
        if not step or str(step["kind"]) != "PIPELINE_STEP":
            return ""
        details = json.loads(step["details_json"] or "{}")
        step_source = json.loads(step["source_json"] or "{}")
        selected = [str(item) for item in list(selected_outputs or [])[:20]]
        now = time.time()
        label = str(details.get("name") or step["step_id"] or "Route")
        return self._append(
            session_id=str(step["session_id"]),
            project_id=str(step["project_id"]),
            kind="PIPELINE_ROUTE",
            status="completed",
            turn_id=str(step["turn_id"] or ""),
            step_id=str(step["step_id"] or ""),
            parent_id=str(step_event_id),
            source={
                "kind": "pipeline-routing",
                "module": "remy.core.pipeline_runner",
                "trust_tier": "internal-derived",
                "pipeline_id": str(step_source.get("pipeline_id") or ""),
                "run_id": str(step_source.get("run_id") or ""),
                "definition_hash": str(step_source.get("definition_hash") or ""),
                "step_id": str(step["step_id"] or ""),
            },
            input_value={"step_id": str(step["step_id"] or ""), "label": label},
            output_value={"selected_outputs": selected},
            details={
                "name": f"Route · {label}",
                "preview": f"Selected: {', '.join(selected) if selected else 'no output'}",
                "selected_outputs": selected,
                "pipeline_id": str(step_source.get("pipeline_id") or ""),
                "run_id": str(step_source.get("run_id") or ""),
                "step_id": str(step["step_id"] or ""),
            },
            started_at=now,
            completed_at=now,
            duration_ms=0,
        )

    def complete_pipeline_run(
        self,
        *,
        event_id: str,
        status: str,
        output: Any = None,
        error: str = "",
        steps_run: int = 0,
    ) -> str:
        """Close the parent span and append a separately selectable result."""
        row = self._row_for_event(event_id=event_id)
        if not row or str(row["kind"]) != "PIPELINE_RUN":
            return ""
        now = time.time()
        started = float(row["started_at"] or now)
        normalized = str(status or "completed").lower()
        event_status = (
            "completed" if normalized in {"ok", "success", "completed", "complete"}
            else "cancelled" if normalized in {"cancelled", "canceled", "stopped"}
            else "failed"
        )
        safe_output = _sanitize_pipeline_value(output)
        details = json.loads(row["details_json"] or "{}")
        details.update({
            "preview": (
                f"{details.get('pipeline_name') or 'Pipeline'} · {event_status} · "
                f"{max(0, int(steps_run))} step(s)"
            ),
            "steps_run": max(0, int(steps_run)),
            "result_status": event_status,
        })
        self._update(
            event_id,
            status=event_status,
            output_json=safe_output,
            details_json=details,
            first_output_at=now if safe_output not in (None, "") else None,
            completed_at=now,
            duration_ms=max(0, int((now - started) * 1000)),
            error=str(error or ""),
        )
        return self._append(
            session_id=str(row["session_id"]),
            project_id=str(row["project_id"]),
            kind="PIPELINE_RESULT",
            status=event_status,
            turn_id=str(row["turn_id"] or ""),
            parent_id=str(event_id),
            source={
                "kind": "pipeline-result",
                "module": "remy.web.routes.pipeline_routes",
                "trust_tier": "internal-derived",
                "pipeline_id": str(details.get("pipeline_id") or ""),
                "run_id": str(details.get("run_id") or ""),
                "definition_hash": str(details.get("definition_hash") or ""),
            },
            input_value={
                "pipeline_id": str(details.get("pipeline_id") or ""),
                "run_id": str(details.get("run_id") or ""),
            },
            output_value=safe_output,
            details={
                "name": f"Result · {details.get('pipeline_name') or 'Pipeline'}",
                "preview": _preview(error or safe_output or event_status),
                "pipeline_id": str(details.get("pipeline_id") or ""),
                "run_id": str(details.get("run_id") or ""),
                "definition_hash": str(details.get("definition_hash") or ""),
                "steps_run": max(0, int(steps_run)),
            },
            started_at=now,
            completed_at=now,
            duration_ms=0,
            error=str(error or ""),
        )

    def begin_execution_run(
        self,
        *,
        scope: str,
        project_id: str,
        source_id: str,
        source_name: str,
        run_id: str,
        attempt_id: str = "",
        goal: Any = None,
        definition_hash: str = "",
        schema: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Open a non-chat Experiment or Automation execution session."""
        normalized_scope = str(scope or "").strip().lower()
        if normalized_scope not in _EXECUTION_EVENT_KINDS:
            raise ValueError(f"Unsupported execution trajectory scope: {scope}")
        safe_source_id = str(source_id or "").strip()
        safe_run_id = str(run_id or "").strip()
        if not safe_source_id or not safe_run_id:
            raise ValueError("Execution trajectory requires source_id and run_id")
        session_id = f"{normalized_scope}:{safe_source_id}:{safe_run_id}"
        name = str(source_name or safe_source_id)
        details = {
            "name": name,
            "preview": f"{name} · running",
            "scope": normalized_scope,
            "source_id": safe_source_id,
            "run_id": safe_run_id,
            "attempt_id": str(attempt_id or ""),
            "definition_hash": str(definition_hash or ""),
            **_sanitize_pipeline_value(dict(metadata or {})),
        }
        return self._append(
            event_id=f"{normalized_scope}-run-{uuid.uuid4().hex}",
            session_id=session_id,
            project_id=str(project_id),
            kind=f"{normalized_scope.upper()}_RUN",
            status="running",
            turn_id=f"{normalized_scope}-turn-{safe_run_id}",
            source={
                "kind": normalized_scope,
                "module": _EXECUTION_SCOPE_MODULES[normalized_scope],
                "trust_tier": "operator-configured",
                "source_id": safe_source_id,
                "run_id": safe_run_id,
                "attempt_id": str(attempt_id or ""),
                "definition_hash": str(definition_hash or ""),
            },
            input_value=_sanitize_pipeline_value(goal),
            schema=_sanitize_pipeline_value(schema),
            details=details,
            started_at=time.time(),
        )

    def record_execution_event(
        self,
        *,
        parent_event_id: str,
        event_kind: str,
        status: str = "completed",
        name: str = "",
        input_value: Any = None,
        output_value: Any = None,
        details: dict[str, Any] | None = None,
        source_kind: str = "execution-stage",
        error: str = "",
        duration_ms: int = 0,
    ) -> str:
        """Append one bounded child observation to an execution trajectory."""
        parent = self._row_for_event(event_id=parent_event_id)
        if not parent:
            return ""
        parent_kind = str(parent["kind"] or "")
        scope = parent_kind.removesuffix("_RUN").lower()
        normalized_kind = str(event_kind or "").strip().upper()
        if normalized_kind not in _EXECUTION_EVENT_KINDS.get(scope, set()):
            raise ValueError(
                f"Unsupported {scope or 'execution'} trajectory event: {event_kind}"
            )
        parent_details = json.loads(parent["details_json"] or "{}")
        safe_details = _sanitize_pipeline_value(dict(details or {}))
        safe_details.update({
            "name": str(name or normalized_kind.replace("_", " ").title()),
            "scope": scope,
            "source_id": str(parent_details.get("source_id") or ""),
            "run_id": str(parent_details.get("run_id") or ""),
            "definition_hash": str(parent_details.get("definition_hash") or ""),
        })
        safe_output = _sanitize_pipeline_value(output_value)
        safe_details.setdefault(
            "preview",
            _preview(error or safe_output or safe_details["name"]),
        )
        now = time.time()
        bounded_duration = max(0, int(duration_ms or 0))
        return self._append(
            session_id=str(parent["session_id"]),
            project_id=str(parent["project_id"]),
            kind=normalized_kind,
            status=str(status or "completed"),
            turn_id=str(parent["turn_id"] or ""),
            step_id=str(safe_details.get("step_id") or safe_details.get("phase") or ""),
            parent_id=str(parent_event_id),
            source={
                "kind": str(source_kind or "execution-stage"),
                "module": _EXECUTION_SCOPE_MODULES.get(scope, "remy.core.agent_lab"),
                "trust_tier": "internal-derived",
                "scope": scope,
                "source_id": str(parent_details.get("source_id") or ""),
                "run_id": str(parent_details.get("run_id") or ""),
            },
            input_value=_sanitize_pipeline_value(input_value),
            output_value=safe_output,
            details=safe_details,
            started_at=now - (bounded_duration / 1_000),
            first_output_at=now if safe_output not in (None, "") else None,
            completed_at=now,
            duration_ms=bounded_duration,
            error=str(error or ""),
        )

    def complete_execution_run(
        self,
        *,
        event_id: str,
        status: str,
        output: Any = None,
        error: str = "",
        details: dict[str, Any] | None = None,
    ) -> str:
        """Close an Experiment/Automation run and append its result record."""
        row = self._row_for_event(event_id=event_id)
        if not row:
            return ""
        parent_kind = str(row["kind"] or "")
        scope = parent_kind.removesuffix("_RUN").lower()
        if scope not in _EXECUTION_EVENT_KINDS:
            return ""
        now = time.time()
        started = float(row["started_at"] or now)
        normalized = str(status or "completed").lower()
        event_status = (
            "completed"
            if normalized in {"ok", "success", "completed", "complete", "completed_with_limits"}
            else "cancelled"
            if normalized in {"cancelled", "canceled", "stopped"}
            else "failed"
        )
        safe_output = _sanitize_pipeline_value(output)
        parent_details = json.loads(row["details_json"] or "{}")
        safe_extra = _sanitize_pipeline_value(dict(details or {}))
        parent_details.update(safe_extra)
        parent_details.update({
            "preview": f"{parent_details.get('name') or scope.title()} · {event_status}",
            "result_status": event_status,
        })
        self._update(
            event_id,
            status=event_status,
            output_json=safe_output,
            details_json=parent_details,
            first_output_at=now if safe_output not in (None, "") else None,
            completed_at=now,
            duration_ms=max(0, int((now - started) * 1000)),
            error=str(error or ""),
        )
        return self._append(
            session_id=str(row["session_id"]),
            project_id=str(row["project_id"]),
            kind=f"{scope.upper()}_RESULT",
            status=event_status,
            turn_id=str(row["turn_id"] or ""),
            parent_id=str(event_id),
            source={
                "kind": f"{scope}-result",
                "module": _EXECUTION_SCOPE_MODULES.get(scope, "remy.core.agent_lab"),
                "trust_tier": "internal-derived",
                "scope": scope,
                "source_id": str(parent_details.get("source_id") or ""),
                "run_id": str(parent_details.get("run_id") or ""),
            },
            input_value={
                "source_id": str(parent_details.get("source_id") or ""),
                "run_id": str(parent_details.get("run_id") or ""),
            },
            output_value=safe_output,
            details={
                "name": f"Result · {parent_details.get('name') or scope.title()}",
                "preview": _preview(error or safe_output or event_status),
                "scope": scope,
                "source_id": str(parent_details.get("source_id") or ""),
                "run_id": str(parent_details.get("run_id") or ""),
                "definition_hash": str(parent_details.get("definition_hash") or ""),
                **safe_extra,
            },
            started_at=now,
            completed_at=now,
            duration_ms=0,
            error=str(error or ""),
        )

    def record_team_event(
        self,
        *,
        project_id: str,
        session_id: str,
        event_type: str,
        status: str,
        payload: dict[str, Any],
        run_id: str = "",
        duration_ms: int = 0,
    ) -> str:
        """Record planner, policy-gate and fan-in decisions in Trajectory."""
        event_name = str(event_type or "team_event").strip().upper()
        if event_name not in {"TEAM_PLAN", "TEAM_GATE", "TEAM_RESULT"}:
            raise ValueError(f"Unsupported team event type: {event_type}")
        now = time.time()
        safe_payload = dict(payload or {})
        members = list(
            safe_payload.get("members")
            or (safe_payload.get("plan") or {}).get("members")
            or []
        )
        return self._append(
            session_id=str(session_id),
            project_id=str(project_id),
            kind=event_name,
            status=str(status or "completed"),
            parent_id=str(run_id or ""),
            source={
                "kind": "agent-team",
                "module": "remy.core.team_planner",
                "trust_tier": "internal-policy",
            },
            input_value={
                "mode": str(safe_payload.get("mode") or ""),
                "plan_hash": str(safe_payload.get("plan_hash") or ""),
                "member_count": len(members),
            },
            output_value=safe_payload,
            details={
                "preview": (
                    f"{event_name.replace('_', ' ').title()} · "
                    f"{len(members)} member(s) · {status}"
                ),
                "run_id": str(run_id or ""),
                "mode": str(safe_payload.get("mode") or ""),
                "member_count": len(members),
                "read_only": bool(safe_payload.get("read_only", True)),
            },
            started_at=now - (max(0, int(duration_ms or 0)) / 1_000),
            completed_at=now,
            duration_ms=max(0, int(duration_ms or 0)),
            error=str(safe_payload.get("error") or ""),
        )

    def record_self_modification_event(
        self,
        *,
        project_id: str,
        proposal_id: str,
        event_type: str,
        status: str,
        payload: dict[str, Any],
    ) -> str:
        """Record a hash/metrics-only Self-modification Lab transition."""
        event_name = str(event_type or "").strip().upper()
        if event_name not in _SELF_MODIFICATION_EVENT_KINDS:
            raise ValueError(f"Unsupported self-modification event type: {event_type}")
        safe_payload = _sanitize_pipeline_value(dict(payload or {}))
        # Proposal text is intentionally excluded even if a caller passes it.
        for key in ("candidate_text", "baseline_text", "prompt", "content"):
            safe_payload.pop(key, None)
        now = time.time()
        session_id = f"self-mod:{proposal_id}"
        return self._append(
            session_id=session_id,
            project_id=str(project_id),
            kind=event_name,
            status=str(status or "completed"),
            turn_id=f"self-mod-turn:{proposal_id}",
            parent_id=str(proposal_id),
            source={
                "kind": "self-modification-lab",
                "module": "remy.core.self_modification_lab",
                "trust_tier": "operator-reviewed",
            },
            input_value={
                "proposal_id": str(proposal_id),
                "candidate_hash": str(safe_payload.get("candidate_hash") or ""),
                "baseline_hash": str(safe_payload.get("baseline_hash") or ""),
            },
            output_value=safe_payload,
            details={
                "name": event_name.replace("_", " ").title(),
                "preview": str(
                    safe_payload.get("summary")
                    or safe_payload.get("decision")
                    or status
                )[:260],
                "proposal_id": str(proposal_id),
                "candidate_hash": str(safe_payload.get("candidate_hash") or ""),
                "baseline_hash": str(safe_payload.get("baseline_hash") or ""),
            },
            started_at=now,
            completed_at=now,
            duration_ms=0,
            error=str(safe_payload.get("error") or ""),
        )

    def begin_request(
        self,
        *,
        session_id: str,
        messages: Iterable[Any],
        tools: Iterable[Any],
        routing: dict[str, Any] | None = None,
        context_sources: Iterable[dict[str, Any]] | None = None,
    ) -> str:
        with self._state_lock:
            turn_id = self._active_turns.get(session_id, "")
        if not turn_id:
            return ""

        rows = list(messages)
        tool_rows = [_tool_payload(tool) for tool in tools]
        request_id = f"req-{uuid.uuid4().hex}"
        step_id = f"step-{uuid.uuid4().hex}"
        started = time.time()
        project_id = self._project_for_turn(turn_id)
        payload = [_message_payload(message) for message in rows]
        self._append(
            event_id=request_id,
            session_id=session_id,
            project_id=project_id,
            kind="REQUEST",
            status="running",
            turn_id=turn_id,
            step_id=step_id,
            request_id=request_id,
            source={"kind": "model-request", "module": "remy.core.agent"},
            input_value=payload,
            schema=tool_rows,
            details={
                "options": _json_safe(routing or {}),
                "tool_count": len(tool_rows),
                "message_count": len(payload),
                "preview": _preview(payload[-1] if payload else ""),
            },
            started_at=started,
        )

        system_messages = [row for row in rows if _message_role(row) == "system"]
        if system_messages:
            self._record_system_change(
                session_id=session_id,
                project_id=project_id,
                turn_id=turn_id,
                request_id=request_id,
                prompt=_message_content(system_messages[0]),
                tools=tool_rows,
                started=started,
            )
            explicit_sources = list(context_sources or [])
            for context_index, context in enumerate(system_messages[1:]):
                content = _message_content(context)
                source = (
                    explicit_sources[context_index]
                    if context_index < len(explicit_sources)
                    else _context_source(content)
                )
                self._append(
                    session_id=session_id,
                    project_id=project_id,
                    kind="CONTEXT",
                    status="completed",
                    turn_id=turn_id,
                    step_id=step_id,
                    request_id=request_id,
                    parent_id=request_id,
                    source=source,
                    input_value=content,
                    output_value=content,
                    details={"preview": _preview(content)},
                    started_at=started,
                    completed_at=started,
                    duration_ms=0,
                )

        with self._state_lock:
            self._active_requests[session_id] = request_id
            self._last_requests[session_id] = request_id
        return request_id

    def _project_for_turn(self, turn_id: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT project_id FROM trajectory_events WHERE turn_id=? ORDER BY id LIMIT 1",
                (turn_id,),
            ).fetchone()
        return str(row["project_id"] if row else "")

    def _record_system_change(
        self,
        *,
        session_id: str,
        project_id: str,
        turn_id: str,
        request_id: str,
        prompt: Any,
        tools: list[dict[str, Any]],
        started: float,
    ) -> None:
        fingerprint = hashlib.sha256(
            _json_dump({"prompt": prompt, "tools": tools}).encode("utf-8")
        ).hexdigest()
        with self._connect() as conn:
            previous = conn.execute(
                """SELECT event_id, input_json, schema_json, details_json
                   FROM trajectory_events
                   WHERE project_id=? AND session_id=? AND kind='SYSTEM'
                   ORDER BY id DESC LIMIT 1""",
                (project_id, session_id),
            ).fetchone()
        previous_details = json.loads(previous["details_json"] or "{}") if previous else {}
        if previous_details.get("fingerprint") == fingerprint:
            return
        previous_prompt = json.loads(previous["input_json"] or "null") if previous else None
        previous_tools = json.loads(previous["schema_json"] or "null") if previous else None
        change_kind = "initial"
        if previous:
            prompt_changed = previous_prompt != prompt
            tools_changed = previous_tools != tools
            if prompt_changed and tools_changed:
                change_kind = "system-and-tools"
            elif prompt_changed:
                change_kind = "system"
            else:
                change_kind = "tools"
        self._append(
            session_id=session_id,
            project_id=project_id,
            kind="SYSTEM",
            status="completed",
            turn_id=turn_id,
            request_id=request_id,
            parent_id=request_id,
            source={
                "kind": "component",
                "name": "system-instruction",
                "module": "remy.core.agent",
                "trust_tier": "internal",
            },
            input_value=prompt,
            output_value=prompt,
            schema=tools,
            details={
                "fingerprint": fingerprint,
                "change_kind": change_kind,
                "previous_event_id": previous["event_id"] if previous else "",
                "previous_prompt": previous_prompt,
                "previous_tools": previous_tools,
                "preview": _preview(prompt),
            },
            started_at=started,
            completed_at=started,
            duration_ms=0,
        )

    def complete_request(self, *, session_id: str, response: Any) -> str:
        with self._state_lock:
            request_id = self._active_requests.pop(session_id, "")
            turn_id = self._active_turns.get(session_id, "")
        if not request_id:
            return ""
        now = time.time()
        request = self._row_for_event(request_id=request_id, kind="REQUEST")
        started = float(request["started_at"] or now) if request else now
        first_output = (
            float(request["first_output_at"])
            if request and request["first_output_at"] is not None
            else now
        )
        duration_ms = max(0, int((now - started) * 1000))
        usage = getattr(response, "usage_metadata", None)
        metadata = getattr(response, "response_metadata", None) or {}
        tool_calls = getattr(response, "tool_calls", None) or []
        output = {
            "content": _message_content(response),
            "tool_calls": _json_safe(tool_calls),
        }
        self._update(
            request_id,
            status="completed",
            output_json=output,
            details_json={
                **(json.loads(request["details_json"] or "{}") if request else {}),
                "provider": metadata.get("_provider") or metadata.get("provider") or "",
                "model": metadata.get("_served_by") or metadata.get("model_name") or metadata.get("model") or "",
                "usage": _json_safe(usage or metadata.get("usage_metadata") or metadata.get("token_usage") or {}),
                "fallback_used": bool(
                    metadata.get("_fallback_used")
                    or metadata.get("fallback_used")
                    or metadata.get("fallback")
                ),
            },
            first_output_at=first_output,
            completed_at=now,
            duration_ms=duration_ms,
        )
        project_id = str(request["project_id"] if request else self._project_for_turn(turn_id))
        assistant_id = self._append(
            session_id=session_id,
            project_id=project_id,
            kind="ASSISTANT",
            status="completed",
            turn_id=turn_id,
            step_id=str(request["step_id"] if request else ""),
            request_id=request_id,
            parent_id=request_id,
            source={
                "kind": "model",
                "provider": metadata.get("_provider") or metadata.get("provider") or "",
                "model": metadata.get("_served_by") or metadata.get("model_name") or metadata.get("model") or "",
            },
            input_value={"request_id": request_id},
            output_value=output,
            details={
                "preview": _preview(output["content"]),
                "usage": _json_safe(usage or metadata.get("usage_metadata") or metadata.get("token_usage") or {}),
                "tool_call_count": len(tool_calls),
            },
            started_at=started,
            first_output_at=first_output,
            completed_at=now,
            duration_ms=duration_ms,
        )
        return assistant_id

    def mark_first_output(
        self,
        *,
        session_id: str,
        timestamp: float | None = None,
    ) -> None:
        """Capture first visible streamed output without writing every token."""
        with self._state_lock:
            request_id = (
                self._active_requests.get(session_id)
                or self._last_requests.get(session_id)
                or ""
            )
        if not request_id:
            return
        observed = float(timestamp if timestamp is not None else time.time())
        row = None
        with self._connect() as conn:
            previous = conn.execute(
                "SELECT first_output_at FROM trajectory_events WHERE event_id=?",
                (request_id,),
            ).fetchone()
            conn.execute(
                """UPDATE trajectory_events
                   SET first_output_at=CASE
                       WHEN first_output_at IS NULL OR ? < first_output_at THEN ?
                       ELSE first_output_at END
                   WHERE event_id=? AND kind='REQUEST'""",
                (observed, observed, request_id),
            )
            row = conn.execute(
                """SELECT * FROM trajectory_events WHERE event_id=?""",
                (request_id,),
            ).fetchone()
            if (
                row
                and row["first_output_at"] is not None
                and (not previous or previous["first_output_at"] != row["first_output_at"])
            ):
                self._append_session_event_log(
                    conn,
                    row,
                    event_type="event.updated",
                    changed_fields=("first_output_at",),
                )
        if row:
            self._emit_change(
                project_id=str(row["project_id"]),
                session_id=str(row["session_id"]),
                event_id=request_id,
                change="update",
                kind=str(row["kind"]),
                status=str(row["status"]),
                sequence=int(row["id"]),
                fields=("first_output_at",),
            )

    def fail_request(self, *, session_id: str, error: Any) -> None:
        with self._state_lock:
            request_id = self._active_requests.pop(session_id, "")
        if not request_id:
            return
        now = time.time()
        request = self._row_for_event(request_id=request_id, kind="REQUEST")
        started = float(request["started_at"] or now) if request else now
        self._update(
            request_id,
            status="failed",
            completed_at=now,
            duration_ms=max(0, int((now - started) * 1000)),
            error=str(error),
        )

    def begin_model_attempt(
        self,
        *,
        session_id: str,
        model: str,
        provider: str,
        model_index: int,
        attempt: int,
        purpose: str,
        tools_enabled: bool,
        compatibility_mode: bool = False,
    ) -> str:
        with self._state_lock:
            turn_id = self._active_turns.get(session_id, "")
            request_id = self._active_requests.get(session_id, "")
        if not turn_id or not request_id:
            return ""
        project_id = self._project_for_turn(turn_id)
        return self._append(
            session_id=session_id,
            project_id=project_id,
            kind="ATTEMPT",
            status="running",
            turn_id=turn_id,
            request_id=request_id,
            parent_id=request_id,
            source={
                "kind": "provider-attempt",
                "provider": provider,
                "model": model,
            },
            input_value={
                "purpose": purpose,
                "tools_enabled": bool(tools_enabled),
                "compatibility_mode": bool(compatibility_mode),
            },
            details={
                "model": model,
                "provider": provider,
                "model_index": int(model_index),
                "attempt": int(attempt),
                "fallback": int(model_index) > 0,
                "preview": f"{provider} · {model} · attempt {attempt}",
            },
            started_at=time.time(),
        )

    def complete_model_attempt(
        self,
        *,
        event_id: str,
        success: bool,
        duration_ms: int,
        error: Any = None,
        retry_action: str = "",
        retry_delay_seconds: int = 0,
        output: Any = None,
    ) -> None:
        if not event_id:
            return
        row = self._row_for_event(event_id=event_id)
        if not row:
            return
        details = json.loads(row["details_json"] or "{}")
        details.update({
            "retry_action": str(retry_action or ""),
            "retry_delay_seconds": max(0, int(retry_delay_seconds or 0)),
        })
        completed = time.time()
        self._update(
            event_id,
            status="completed" if success else "failed",
            output_json=output,
            details_json=details,
            first_output_at=completed if success else None,
            completed_at=completed,
            duration_ms=max(0, int(duration_ms or 0)),
            error=str(error or ""),
        )

    def begin_tool(
        self,
        *,
        session_id: str,
        call_id: str,
        name: str,
        payload: Any,
        schema: Any = None,
        parent_call_id: str = "",
    ) -> str:
        with self._state_lock:
            turn_id = self._active_turns.get(session_id, "")
            request_id = self._last_requests.get(session_id, "")
        if not turn_id:
            return ""
        project_id = self._project_for_turn(turn_id)
        started = time.time()
        if schema is not None and not isinstance(schema, (dict, list)):
            schema = _tool_payload(schema)
        return self._append(
            event_id=f"tool-{uuid.uuid4().hex}",
            session_id=session_id,
            project_id=project_id,
            kind="SUBTOOL" if parent_call_id else "TOOL",
            status="running",
            turn_id=turn_id,
            request_id=request_id,
            call_id=str(call_id),
            parent_id=str(parent_call_id or request_id),
            source={"kind": "tool", "name": name, "module": "remy.core.agent_tools"},
            input_value=payload,
            schema=schema,
            details={"name": name, "preview": _preview(payload)},
            started_at=started,
        )

    def complete_tool(
        self,
        *,
        event_id: str,
        result: Any,
        error: str = "",
        policy: Any = None,
        artifacts: Any = None,
    ) -> None:
        if not event_id:
            return
        now = time.time()
        row = self._row_for_event(event_id=event_id)
        if not row:
            return
        started = float(row["started_at"] or now)
        details = json.loads(row["details_json"] or "{}")
        details.update({
            "result_preview": _preview(result),
            "policy": _json_safe(policy),
            "artifacts": _json_safe(artifacts),
        })
        self._update(
            event_id,
            status="failed" if error else "completed",
            output_json=result,
            details_json=details,
            completed_at=now,
            duration_ms=max(0, int((now - started) * 1000)),
            error=error,
        )

    def record_diagnostics(
        self,
        *,
        session_id: str,
        entries: Iterable[dict[str, Any]],
    ) -> None:
        with self._state_lock:
            turn_id = self._active_turns.get(session_id, "")
        if not turn_id:
            return
        project_id = self._project_for_turn(turn_id)
        now = time.time()
        mapping = {
            "factuality_analysis": "VERIFICATION",
            "claim_source_matrix": "VERIFICATION",
            "claim_lifecycle": "VERIFICATION",
            "research_execution_schedule": "VERIFICATION",
            "research_same_run_recovery": "VERIFICATION",
            "research_prefetch_queue": "VERIFICATION",
            "marginal_evidence_analysis": "VERIFICATION",
            "source_provenance_graph": "VERIFICATION",
            "epistemic_governance": "POLICY",
            "memory_retrieval": "MEMORY",
            "approval": "APPROVAL",
            "subagent": "SUBAGENT",
            "compaction": "COMPACTED",
        }
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            kind = mapping.get(str(entry.get("type") or ""))
            if not kind:
                continue
            self._append(
                session_id=session_id,
                project_id=project_id,
                kind=kind,
                status="failed" if entry.get("error") else "completed",
                turn_id=turn_id,
                parent_id=self._last_requests.get(session_id, ""),
                source={"kind": "component", "name": entry.get("type"), "trust_tier": "internal"},
                input_value=entry,
                output_value=entry,
                details={"preview": _preview(entry)},
                started_at=now,
                completed_at=now,
                duration_ms=int(entry.get("duration_ms") or 0),
                error=str(entry.get("error") or ""),
            )

    def finish_turn(
        self,
        *,
        session_id: str,
        error: Any = None,
        evaluate_regressions: bool = True,
    ) -> None:
        if error:
            self.fail_request(session_id=session_id, error=error)
        with self._state_lock:
            turn_id = self._active_turns.pop(session_id, None)
            self._active_requests.pop(session_id, None)
            self._last_requests.pop(session_id, None)
        if turn_id:
            project_id = self._project_for_turn(turn_id)
            now = time.time()
            interruption = str(error or "interrupted before turn completion")
            interrupted_rows = []
            with self._connect() as conn:
                interrupted_rows = conn.execute(
                    """SELECT id, event_id, project_id, session_id, kind
                       FROM trajectory_events
                       WHERE turn_id=? AND status='running'""",
                    (turn_id,),
                ).fetchall()
                conn.execute(
                    """UPDATE trajectory_events
                       SET status='failed', completed_at=?,
                           duration_ms=CASE
                               WHEN started_at IS NULL THEN 0
                               ELSE MAX(0, CAST((? - started_at) * 1000 AS INTEGER))
                           END,
                           error=CASE WHEN error='' THEN ? ELSE error END
                       WHERE turn_id=? AND status='running'""",
                    (now, now, interruption, turn_id),
                )
                for interrupted_row in interrupted_rows:
                    updated = conn.execute(
                        "SELECT * FROM trajectory_events WHERE event_id=?",
                        (interrupted_row["event_id"],),
                    ).fetchone()
                    if updated:
                        self._append_session_event_log(
                            conn,
                            updated,
                            event_type="event.interrupted",
                            changed_fields=(
                                "status", "completed_at", "duration_ms", "error"
                            ),
                        )
            for row in interrupted_rows:
                self._emit_change(
                    project_id=str(row["project_id"]),
                    session_id=str(row["session_id"]),
                    event_id=str(row["event_id"]),
                    change="update",
                    kind=str(row["kind"]),
                    status="failed",
                    sequence=int(row["id"]),
                    fields=("status", "completed_at", "duration_ms", "error"),
                )
            if project_id and evaluate_regressions:
                self._evaluate_session_regressions(
                    project_id=project_id,
                    session_id=session_id,
                )

    def _row_for_event(
        self,
        *,
        event_id: str = "",
        request_id: str = "",
        kind: str = "",
    ) -> sqlite3.Row | None:
        where = []
        values: list[Any] = []
        if event_id:
            where.append("event_id=?")
            values.append(event_id)
        if request_id:
            where.append("request_id=?")
            values.append(request_id)
        if kind:
            where.append("kind=?")
            values.append(kind)
        if not where:
            return None
        with self._connect() as conn:
            return conn.execute(
                f"SELECT * FROM trajectory_events WHERE {' AND '.join(where)} ORDER BY id DESC LIMIT 1",
                values,
            ).fetchone()

    def list_events(
        self,
        *,
        project_id: str,
        session_id: str,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM (
                    SELECT * FROM trajectory_events
                    WHERE project_id=? AND session_id=?
                    ORDER BY id DESC LIMIT ?
                ) ORDER BY id""",
                (project_id, session_id, max(1, min(int(limit), 5000))),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            for column in ("source_json", "input_json", "output_json", "schema_json", "details_json"):
                target = column.removesuffix("_json")
                try:
                    item[target] = json.loads(item.pop(column) or "null")
                except json.JSONDecodeError:
                    item[target] = item.pop(column)
            item["sequence"] = item.pop("id")
            item["started_at_iso"] = _now_iso(item["started_at"]) if item.get("started_at") else ""
            item["completed_at_iso"] = _now_iso(item["completed_at"]) if item.get("completed_at") else ""
            result.append(item)
        with self._connect() as conn:
            annotation_rows = conn.execute(
                """SELECT * FROM trajectory_annotations
                   WHERE project_id=? AND session_id=?""",
                (project_id, session_id),
            ).fetchall()
        annotations = {
            str(row["event_id"]): {
                "label": str(row["label"] or ""),
                "note": str(row["note"] or ""),
                "bookmarked": bool(row["bookmarked"]),
                "updated_at": str(row["updated_at"] or ""),
            }
            for row in annotation_rows
        }
        for item in result:
            item["annotation"] = annotations.get(str(item.get("event_id") or ""))
        cumulative = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        attempts_by_request: dict[str, list[dict[str, Any]]] = {}
        for item in result:
            if item.get("kind") == "ATTEMPT" and item.get("request_id"):
                attempts_by_request.setdefault(str(item["request_id"]), []).append(item)
        for item in result:
            if item.get("kind") != "REQUEST":
                continue
            details = item.get("details") if isinstance(item.get("details"), dict) else {}
            usage = details.get("usage") if isinstance(details.get("usage"), dict) else {}

            def _usage_value(*keys: str) -> int:
                for key in keys:
                    value = usage.get(key)
                    if value not in (None, ""):
                        try:
                            return max(0, int(value))
                        except (TypeError, ValueError):
                            continue
                return 0

            request_usage = {
                "input_tokens": _usage_value("input_tokens", "prompt_tokens", "prompt_token_count"),
                "output_tokens": _usage_value(
                    "output_tokens", "completion_tokens", "candidates_token_count"
                ),
                "total_tokens": _usage_value("total_tokens", "total_token_count"),
            }
            if not request_usage["total_tokens"]:
                request_usage["total_tokens"] = (
                    request_usage["input_tokens"] + request_usage["output_tokens"]
                )
            for key in cumulative:
                cumulative[key] += request_usage[key]
            details["usage"] = request_usage
            details["session_cumulative_usage"] = dict(cumulative)
            attempts = attempts_by_request.get(str(item.get("request_id") or ""), [])
            details["attempt_count"] = len(attempts)
            details["retry_count"] = max(0, len(attempts) - 1)
            details["attempt_models"] = list(dict.fromkeys(
                str((attempt.get("details") or {}).get("model") or "")
                for attempt in attempts
                if (attempt.get("details") or {}).get("model")
            ))
            item["details"] = details
        return result

    def list_project_events(
        self,
        *,
        project_id: str,
        limit: int = 20_000,
    ) -> list[dict[str, Any]]:
        """Return a recent cross-session window for aggregate diagnostics.

        This intentionally does not attach annotation notes or build session
        cumulative usage.  Cross-session callers receive the minimum decoded
        event shape needed to calculate project metrics, while the API layer
        is responsible for returning aggregate-only data.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM (
                    SELECT * FROM trajectory_events
                    WHERE project_id=? AND session_id NOT IN (
                        SELECT session_id FROM trajectory_replay_sessions WHERE project_id=?
                    )
                    ORDER BY id DESC LIMIT ?
                ) ORDER BY id""",
                (project_id, project_id, max(1, min(int(limit), 25_000))),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            for column in ("source_json", "input_json", "output_json", "schema_json", "details_json"):
                target = column.removesuffix("_json")
                try:
                    item[target] = json.loads(item.pop(column) or "null")
                except json.JSONDecodeError:
                    item[target] = item.pop(column)
            item["sequence"] = item.pop("id")
            item["started_at_iso"] = _now_iso(item["started_at"]) if item.get("started_at") else ""
            item["completed_at_iso"] = _now_iso(item["completed_at"]) if item.get("completed_at") else ""
            result.append(item)
        return result

    def count_project_events(self, *, project_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT COUNT(*) AS total FROM trajectory_events
                   WHERE project_id=? AND session_id NOT IN (
                       SELECT session_id FROM trajectory_replay_sessions WHERE project_id=?
                   )""",
                (project_id, project_id),
            ).fetchone()
        return int(row["total"] if row else 0)

    def list_model_promotion_requests(
        self,
        *,
        project_id: str,
        promotion_id: str,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        """Return only aggregate-safe production requests assigned to one canary."""
        promotion = str(promotion_id or "")[:100]
        if not promotion:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM (
                    SELECT * FROM trajectory_events
                    WHERE project_id=? AND kind='REQUEST'
                      AND details_json LIKE ?
                      AND session_id NOT IN (
                          SELECT session_id FROM trajectory_replay_sessions WHERE project_id=?
                      )
                    ORDER BY id DESC LIMIT ?
                ) ORDER BY id""",
                (
                    project_id,
                    f'%"promotion_id":"{promotion}"%',
                    project_id,
                    max(1, min(int(limit), 10_000)),
                ),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            for column in (
                "source_json", "input_json", "output_json", "schema_json", "details_json"
            ):
                target = column.removesuffix("_json")
                try:
                    item[target] = json.loads(item.pop(column) or "null")
                except json.JSONDecodeError:
                    item[target] = item.pop(column)
            item["sequence"] = item.pop("id")
            result.append(item)
        return result

    def list_self_modification_canary_events(
        self,
        *,
        project_id: str,
        proposal_id: str,
        limit: int = 10_000,
    ) -> list[dict[str, Any]]:
        """Return aggregate-safe production requests and their verification events."""
        proposal = str(proposal_id or "")[:160]
        if not proposal:
            return []
        pattern = f'%"proposal_id":"{proposal}"%'
        bounded_limit = max(1, min(int(limit), 20_000))
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM (
                       SELECT * FROM trajectory_events
                       WHERE project_id=?
                         AND session_id NOT IN (
                             SELECT session_id FROM trajectory_replay_sessions
                             WHERE project_id=?
                         )
                         AND (
                             (kind='REQUEST' AND details_json LIKE ?)
                             OR (
                                 kind='VERIFICATION' AND parent_id IN (
                                     SELECT event_id FROM trajectory_events
                                     WHERE project_id=? AND kind='REQUEST'
                                       AND details_json LIKE ?
                                 )
                             )
                         )
                       ORDER BY id DESC LIMIT ?
                   ) ORDER BY id""",
                (
                    project_id,
                    project_id,
                    pattern,
                    project_id,
                    pattern,
                    bounded_limit,
                ),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            for column in (
                "source_json", "input_json", "output_json", "schema_json", "details_json"
            ):
                target = column.removesuffix("_json")
                raw = item.pop(column)
                try:
                    item[target] = json.loads(raw or "null")
                except json.JSONDecodeError:
                    item[target] = raw
            item["sequence"] = item.pop("id")
            if item.get("kind") == "REQUEST":
                details = item.get("details") if isinstance(item.get("details"), dict) else {}
                options = details.get("options") if isinstance(details.get("options"), dict) else {}
                self_mod = (
                    options.get("self_modification")
                    if isinstance(options.get("self_modification"), dict)
                    else {}
                )
                item["details"] = {"options": {"self_modification": self_mod}}
                item["input"] = None
                item["output"] = None
                item["schema"] = None
                item["error"] = ""
            elif item.get("kind") == "VERIFICATION":
                source = item.get("output") if isinstance(item.get("output"), dict) else {}
                counters = {
                    key: source.get(key)
                    for key in (
                        "unsupported_observed_claims",
                        "unsupported_claims_total",
                        "unsupported",
                    )
                    if key in source
                }
                item["input"] = counters
                item["output"] = counters
                item["details"] = {}
                item["error"] = ""
            result.append(item)
        return result

    def register_replay_session(
        self,
        *,
        session_id: str,
        project_id: str,
        case_id: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO trajectory_replay_sessions(
                       session_id, project_id, case_id, created_at
                   ) VALUES (?, ?, ?, ?)""",
                (session_id, project_id, case_id, _now_iso()),
            )

    def is_replay_session(self, *, project_id: str, session_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT 1 FROM trajectory_replay_sessions
                   WHERE project_id=? AND session_id=?""",
                (project_id, session_id),
            ).fetchone()
        return bool(row)

    @staticmethod
    def _baseline_metrics(value: Any) -> dict[str, float]:
        source = value if isinstance(value, dict) else {}
        result = {}
        for key in ("failure_rate", "avg_request_ms", "tokens_per_request"):
            try:
                result[key] = max(0.0, float(source.get(key) or 0))
            except (TypeError, ValueError):
                result[key] = 0.0
        return result

    @staticmethod
    def _baseline_summary(value: Any) -> dict[str, Any]:
        source = value if isinstance(value, dict) else {}
        allowed = (
            "sessions", "events", "turns", "requests", "tool_calls",
            "failures", "recovered_failures", "success_rate", "recovery_rate",
            "total_tokens", "avg_request_ms", "p95_request_ms",
            "window_start", "window_end",
        )
        return {key: _json_safe(source.get(key)) for key in allowed if key in source}

    def create_analytics_baseline(
        self,
        *,
        project_id: str,
        name: str,
        days: int,
        metrics: dict[str, Any],
        summary: dict[str, Any],
        source_event_count: int,
        activate: bool = True,
    ) -> dict[str, Any]:
        baseline_id = f"baseline-{uuid.uuid4().hex}"
        created_at = _now_iso()
        clean_name = (str(name or "").strip() or "Trajectory baseline")[:120]
        clean_days = max(1, min(int(days), 365))
        clean_metrics = self._baseline_metrics(metrics)
        clean_summary = self._baseline_summary(summary)
        resolved_alert_ids: list[str] = []
        with self._connect() as conn:
            if activate:
                conn.execute(
                    "UPDATE trajectory_analytics_baselines SET active=0 WHERE project_id=?",
                    (project_id,),
                )
                resolved_alert_ids = self._resolve_alerts_with_history(
                    conn,
                    project_id=project_id,
                    where_sql="baseline_id LIKE 'baseline-%'",
                    values=(),
                    action="baseline-replaced",
                    created_at=created_at,
                )
            conn.execute(
                """INSERT INTO trajectory_analytics_baselines(
                       baseline_id, project_id, name, days, metrics_json,
                       summary_json, source_event_count, active, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    baseline_id, project_id, clean_name, clean_days,
                    _json_dump(clean_metrics), _json_dump(clean_summary),
                    max(0, int(source_event_count)), int(bool(activate)), created_at,
                ),
            )
        for alert_id in resolved_alert_ids:
            self._notify_trajectory_alert(
                {"alert_id": alert_id, "project_id": project_id}, resolved=True
            )
        self._emit_analytics_change(project_id=project_id, change="baseline-created")
        return {
            "baseline_id": baseline_id,
            "project_id": project_id,
            "name": clean_name,
            "days": clean_days,
            "metrics": clean_metrics,
            "summary": clean_summary,
            "source_event_count": max(0, int(source_event_count)),
            "active": bool(activate),
            "created_at": created_at,
        }

    def list_analytics_baselines(self, *, project_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM trajectory_analytics_baselines
                   WHERE project_id=? ORDER BY active DESC, created_at DESC""",
                (project_id,),
            ).fetchall()
        result = []
        for row in rows:
            result.append({
                "baseline_id": str(row["baseline_id"]),
                "project_id": str(row["project_id"]),
                "name": str(row["name"]),
                "days": int(row["days"]),
                "metrics": self._baseline_metrics(json.loads(row["metrics_json"] or "{}")),
                "summary": self._baseline_summary(json.loads(row["summary_json"] or "{}")),
                "source_event_count": int(row["source_event_count"] or 0),
                "active": bool(row["active"]),
                "created_at": str(row["created_at"]),
            })
        return result

    def get_active_analytics_baseline(self, *, project_id: str) -> dict[str, Any] | None:
        return next(
            (row for row in self.list_analytics_baselines(project_id=project_id) if row["active"]),
            None,
        )

    def activate_analytics_baseline(self, *, project_id: str, baseline_id: str) -> dict[str, Any]:
        resolved_alert_ids: list[str] = []
        with self._connect() as conn:
            row = conn.execute(
                """SELECT baseline_id FROM trajectory_analytics_baselines
                   WHERE project_id=? AND baseline_id=?""",
                (project_id, baseline_id),
            ).fetchone()
            if not row:
                raise KeyError(baseline_id)
            conn.execute(
                "UPDATE trajectory_analytics_baselines SET active=0 WHERE project_id=?",
                (project_id,),
            )
            conn.execute(
                """UPDATE trajectory_analytics_baselines SET active=1
                   WHERE project_id=? AND baseline_id=?""",
                (project_id, baseline_id),
            )
            resolved_alert_ids = self._resolve_alerts_with_history(
                conn,
                project_id=project_id,
                where_sql="baseline_id LIKE 'baseline-%' AND baseline_id!=?",
                values=(baseline_id,),
                action="baseline-activated",
            )
        for alert_id in resolved_alert_ids:
            self._notify_trajectory_alert(
                {"alert_id": alert_id, "project_id": project_id}, resolved=True
            )
        self._emit_analytics_change(project_id=project_id, change="baseline-activated")
        return self.get_active_analytics_baseline(project_id=project_id) or {}

    def deactivate_analytics_baseline(self, *, project_id: str) -> None:
        resolved_alert_ids: list[str] = []
        with self._connect() as conn:
            conn.execute(
                "UPDATE trajectory_analytics_baselines SET active=0 WHERE project_id=?",
                (project_id,),
            )
            resolved_alert_ids = self._resolve_alerts_with_history(
                conn,
                project_id=project_id,
                where_sql="baseline_id LIKE 'baseline-%'",
                values=(),
                action="baseline-deactivated",
            )
        for alert_id in resolved_alert_ids:
            self._notify_trajectory_alert(
                {"alert_id": alert_id, "project_id": project_id}, resolved=True
            )
        self._emit_analytics_change(project_id=project_id, change="baseline-deactivated")

    def delete_analytics_baseline(self, *, project_id: str, baseline_id: str) -> bool:
        resolved_alert_ids: list[str] = []
        with self._connect() as conn:
            row = conn.execute(
                """SELECT active FROM trajectory_analytics_baselines
                   WHERE project_id=? AND baseline_id=?""",
                (project_id, baseline_id),
            ).fetchone()
            if not row:
                return False
            was_active = bool(row["active"])
            conn.execute(
                """DELETE FROM trajectory_analytics_baselines
                   WHERE project_id=? AND baseline_id=?""",
                (project_id, baseline_id),
            )
            resolved_alert_ids = self._resolve_alerts_with_history(
                conn,
                project_id=project_id,
                where_sql="baseline_id=?",
                values=(baseline_id,),
                action="baseline-deleted",
            )
            if was_active:
                fallback = conn.execute(
                    """SELECT baseline_id FROM trajectory_analytics_baselines
                       WHERE project_id=? ORDER BY created_at DESC LIMIT 1""",
                    (project_id,),
                ).fetchone()
                if fallback:
                    conn.execute(
                        "UPDATE trajectory_analytics_baselines SET active=1 WHERE baseline_id=?",
                        (fallback["baseline_id"],),
                    )
        for alert_id in resolved_alert_ids:
            self._notify_trajectory_alert(
                {"alert_id": alert_id, "project_id": project_id}, resolved=True
            )
        self._emit_analytics_change(project_id=project_id, change="baseline-deleted")
        return True

    @staticmethod
    def _alert_policy_values(
        *,
        scope_type: str,
        scope_value: str,
        thresholds: dict[str, Any],
    ) -> tuple[str, str, dict[str, float]]:
        normalized_scope = str(scope_type or "").strip().lower()
        if normalized_scope not in {"project", "provider", "model", "tool"}:
            raise ValueError("Unsupported trajectory alert policy scope")
        normalized_value = str(scope_value or "").strip()
        if normalized_scope == "project":
            normalized_value = "*"
        if not normalized_value:
            raise ValueError("Provider, model, and tool policies require a scope value")
        source = thresholds if isinstance(thresholds, dict) else {}
        limits = {
            "failure_rate_warning": 1.0,
            "failure_rate_critical": 1.0,
            "latency_warning_ms": 86_400_000.0,
            "latency_critical_ms": 86_400_000.0,
            "tokens_warning": 100_000_000.0,
            "tokens_critical": 100_000_000.0,
        }
        clean = {}
        for key, maximum in limits.items():
            try:
                clean[key] = min(maximum, max(0.0, float(source.get(key) or 0)))
            except (TypeError, ValueError):
                clean[key] = 0.0
        for warning_key, critical_key in (
            ("failure_rate_warning", "failure_rate_critical"),
            ("latency_warning_ms", "latency_critical_ms"),
            ("tokens_warning", "tokens_critical"),
        ):
            warning = clean[warning_key]
            critical = clean[critical_key]
            if warning and critical and critical < warning:
                raise ValueError("Critical policy thresholds must be at least warning thresholds")
        if not any(clean.values()):
            raise ValueError("At least one trajectory alert threshold must be greater than zero")
        return normalized_scope, normalized_value[:240], clean

    def create_alert_policy(
        self,
        *,
        project_id: str,
        name: str,
        scope_type: str,
        scope_value: str,
        thresholds: dict[str, Any],
        enabled: bool = True,
    ) -> dict[str, Any]:
        normalized_scope, normalized_value, clean = self._alert_policy_values(
            scope_type=scope_type,
            scope_value=scope_value,
            thresholds=thresholds,
        )
        now = _now_iso()
        policy_id = f"policy-{uuid.uuid4().hex}"
        clean_name = (str(name or "").strip() or f"{normalized_scope.title()} policy")[:120]
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO trajectory_alert_policies(
                       policy_id, project_id, name, scope_type, scope_value,
                       thresholds_json, enabled, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    policy_id, project_id, clean_name, normalized_scope,
                    normalized_value, _json_dump(clean), int(bool(enabled)), now, now,
                ),
            )
        self._emit_analytics_change(project_id=project_id, change="policy-created")
        return {
            "policy_id": policy_id, "project_id": project_id, "name": clean_name,
            "scope_type": normalized_scope, "scope_value": normalized_value,
            "thresholds": clean, "enabled": bool(enabled),
            "created_at": now, "updated_at": now,
        }

    def list_alert_policies(self, *, project_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM trajectory_alert_policies
                   WHERE project_id=? ORDER BY enabled DESC, updated_at DESC""",
                (project_id,),
            ).fetchall()
        return [{
            "policy_id": str(row["policy_id"]),
            "project_id": str(row["project_id"]),
            "name": str(row["name"]),
            "scope_type": str(row["scope_type"]),
            "scope_value": str(row["scope_value"]),
            "thresholds": json.loads(row["thresholds_json"] or "{}"),
            "enabled": bool(row["enabled"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        } for row in rows]

    def update_alert_policy(
        self,
        *,
        project_id: str,
        policy_id: str,
        name: str,
        scope_type: str,
        scope_value: str,
        thresholds: dict[str, Any],
        enabled: bool,
    ) -> dict[str, Any]:
        normalized_scope, normalized_value, clean = self._alert_policy_values(
            scope_type=scope_type,
            scope_value=scope_value,
            thresholds=thresholds,
        )
        now = _now_iso()
        clean_name = (str(name or "").strip() or f"{normalized_scope.title()} policy")[:120]
        resolved_alert_ids: list[str] = []
        with self._connect() as conn:
            cursor = conn.execute(
                """UPDATE trajectory_alert_policies SET name=?, scope_type=?,
                       scope_value=?, thresholds_json=?, enabled=?, updated_at=?
                   WHERE project_id=? AND policy_id=?""",
                (
                    clean_name, normalized_scope, normalized_value,
                    _json_dump(clean), int(bool(enabled)), now, project_id, policy_id,
                ),
            )
            if not cursor.rowcount:
                raise KeyError(policy_id)
            resolved_alert_ids = self._resolve_alerts_with_history(
                conn,
                project_id=project_id,
                where_sql="baseline_id=?",
                values=(policy_id,),
                action="policy-updated",
                created_at=now,
            )
        for alert_id in resolved_alert_ids:
            self._notify_trajectory_alert(
                {"alert_id": alert_id, "project_id": project_id}, resolved=True
            )
        self._emit_analytics_change(project_id=project_id, change="policy-updated")
        return next(
            row for row in self.list_alert_policies(project_id=project_id)
            if row["policy_id"] == policy_id
        )

    def delete_alert_policy(self, *, project_id: str, policy_id: str) -> bool:
        resolved_alert_ids: list[str] = []
        with self._connect() as conn:
            cursor = conn.execute(
                """DELETE FROM trajectory_alert_policies
                   WHERE project_id=? AND policy_id=?""",
                (project_id, policy_id),
            )
            if not cursor.rowcount:
                return False
            resolved_alert_ids = self._resolve_alerts_with_history(
                conn,
                project_id=project_id,
                where_sql="baseline_id=?",
                values=(policy_id,),
                action="policy-deleted",
            )
        for alert_id in resolved_alert_ids:
            self._notify_trajectory_alert(
                {"alert_id": alert_id, "project_id": project_id}, resolved=True
            )
        self._emit_analytics_change(project_id=project_id, change="policy-deleted")
        return True

    @staticmethod
    def _append_alert_history(
        conn: sqlite3.Connection,
        *,
        alert_id: str,
        project_id: str,
        action: str,
        status: str,
        actor: str = "system",
        details: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> None:
        allowed_details = {}
        for key in (
            "metric", "severity", "observed", "threshold", "delta",
            "conversation_id", "event_id", "source_type",
        ):
            if key in (details or {}):
                allowed_details[key] = _json_safe((details or {}).get(key))
        conn.execute(
            """INSERT INTO trajectory_alert_history(
                   alert_id, project_id, action, status, actor, details_json, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                alert_id, project_id, str(action or "updated")[:64],
                str(status or "open")[:32], str(actor or "system")[:64],
                _json_dump(allowed_details), created_at or _now_iso(),
            ),
        )

    def _resolve_alerts_with_history(
        self,
        conn: sqlite3.Connection,
        *,
        project_id: str,
        where_sql: str,
        values: tuple[Any, ...],
        action: str,
        created_at: str | None = None,
    ) -> list[str]:
        """Resolve a trusted internal alert selection and retain why it changed."""
        now = created_at or _now_iso()
        rows = conn.execute(
            f"""SELECT * FROM trajectory_regression_alerts
                WHERE project_id=? AND {where_sql} AND status!='resolved'""",
            (project_id, *values),
        ).fetchall()
        critical_ids: list[str] = []
        for row in rows:
            alert_id = str(row["alert_id"])
            conn.execute(
                """UPDATE trajectory_regression_alerts
                   SET status='resolved', updated_at=? WHERE alert_id=?""",
                (now, alert_id),
            )
            self._append_alert_history(
                conn,
                alert_id=alert_id,
                project_id=project_id,
                action=action,
                status="resolved",
                details={
                    "metric": row["metric"], "severity": row["severity"],
                    "observed": row["observed"], "threshold": row["baseline"],
                    "conversation_id": row["session_id"], "event_id": row["event_id"],
                    "source_type": (
                        "policy" if str(row["baseline_id"]).startswith("policy-")
                        else "baseline"
                    ),
                },
                created_at=now,
            )
            if str(row["severity"]) == "error":
                critical_ids.append(alert_id)
        return critical_ids

    def list_alert_history(
        self,
        *,
        project_id: str,
        alert_id: str = "",
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        where = "WHERE project_id=?"
        values: list[Any] = [project_id]
        if alert_id:
            where += " AND alert_id=?"
            values.append(alert_id)
        values.append(max(1, min(int(limit), 1000)))
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT * FROM trajectory_alert_history {where}
                    ORDER BY history_id DESC LIMIT ?""",
                values,
            ).fetchall()
        return [{
            "history_id": int(row["history_id"]),
            "alert_id": str(row["alert_id"]),
            "project_id": str(row["project_id"]),
            "action": str(row["action"]),
            "status": str(row["status"]),
            "actor": str(row["actor"]),
            "details": json.loads(row["details_json"] or "{}"),
            "created_at": str(row["created_at"]),
        } for row in rows]

    def get_slo_config(self, *, project_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM trajectory_slo_config WHERE project_id=?",
                (project_id,),
            ).fetchone()
        if not row:
            return {
                "project_id": project_id,
                "target_success_rate": 0.99,
                "window_days": 30,
                "min_operations": 5,
                "updated_at": "",
            }
        return {
            "project_id": str(row["project_id"]),
            "target_success_rate": float(row["target_success_rate"]),
            "window_days": int(row["window_days"]),
            "min_operations": int(row["min_operations"]),
            "updated_at": str(row["updated_at"]),
        }

    def update_slo_config(
        self,
        *,
        project_id: str,
        target_success_rate: float,
        window_days: int,
        min_operations: int,
    ) -> dict[str, Any]:
        target = float(target_success_rate)
        if not 0.5 <= target < 1:
            raise ValueError("SLO target success rate must be between 0.5 and 1.0")
        days = max(1, min(int(window_days), 365))
        minimum = max(1, min(int(min_operations), 100_000))
        now = _now_iso()
        resolved_incident_ids: list[str] = []
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO trajectory_slo_config(
                       project_id, target_success_rate, window_days,
                       min_operations, updated_at
                   ) VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(project_id) DO UPDATE SET
                       target_success_rate=excluded.target_success_rate,
                       window_days=excluded.window_days,
                       min_operations=excluded.min_operations,
                       updated_at=excluded.updated_at""",
                (project_id, target, days, minimum, now),
            )
            rows = conn.execute(
                """SELECT * FROM trajectory_slo_incidents
                   WHERE project_id=? AND status!='resolved'""",
                (project_id,),
            ).fetchall()
            for row in rows:
                incident_id = str(row["incident_id"])
                conn.execute(
                    """UPDATE trajectory_slo_incidents
                       SET status='resolved', updated_at=? WHERE incident_id=?""",
                    (now, incident_id),
                )
                self._append_alert_history(
                    conn,
                    alert_id=incident_id,
                    project_id=project_id,
                    action="slo-config-updated",
                    status="resolved",
                    details={
                        "metric": "slo.burn_rate", "severity": row["severity"],
                        "observed": row["burn_rate"], "threshold": 1,
                        "conversation_id": row["conversation_id"],
                        "event_id": row["event_id"], "source_type": "slo",
                    },
                    created_at=now,
                )
                resolved_incident_ids.append(incident_id)
        for incident_id in resolved_incident_ids:
            self._notify_slo_incident(
                {"incident_id": incident_id, "project_id": project_id}, resolved=True
            )
        self._emit_analytics_change(project_id=project_id, change="slo-updated")
        return self.get_slo_config(project_id=project_id)

    def list_slo_incidents(
        self,
        *,
        project_id: str,
        status: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        where = "WHERE project_id=?"
        values: list[Any] = [project_id]
        if status:
            where += " AND status=?"
            values.append(status)
        values.append(max(1, min(int(limit), 500)))
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT * FROM trajectory_slo_incidents {where}
                    ORDER BY CASE status WHEN 'open' THEN 0 WHEN 'acknowledged' THEN 1 ELSE 2 END,
                             updated_at DESC LIMIT ?""",
                values,
            ).fetchall()
        return [{
            "incident_id": str(row["incident_id"]),
            "project_id": str(row["project_id"]),
            "reason": str(row["reason"]),
            "severity": str(row["severity"]),
            "status": str(row["status"]),
            "burn_rate": float(row["burn_rate"] or 0),
            "windows": json.loads(row["windows_json"] or "[]"),
            "operations": int(row["operations"] or 0),
            "failures": int(row["failures"] or 0),
            "conversation_id": str(row["conversation_id"]),
            "event_id": str(row["event_id"]),
            "detected_at": str(row["detected_at"]),
            "updated_at": str(row["updated_at"]),
        } for row in rows]

    def sync_slo_incidents(
        self,
        *,
        project_id: str,
        conversation_id: str,
        event_id: str,
        slo: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Synchronize the active project-level SLO incident, if any."""
        now = _now_iso()
        incident_status = str(slo.get("status") or "insufficient-data")
        alert = slo.get("alert") if isinstance(slo.get("alert"), dict) else {}
        reason = str(alert.get("reason") or "within-budget")
        is_active = incident_status in {"warning", "critical"}
        active_incident_id = ""
        changed = False
        notify_opened: list[dict[str, Any]] = []
        notify_resolved: list[dict[str, Any]] = []
        selected_windows = {
            str(label) for label in (alert.get("windows") or []) if str(label)
        }
        window_rows = [
            row for row in (slo.get("windows") or [])
            if isinstance(row, dict) and (
                not selected_windows or str(row.get("label") or "") in selected_windows
            )
        ]
        burn_rate = max(
            (float(row.get("burn_rate") or 0) for row in window_rows),
            default=0.0,
        )
        with self._connect() as conn:
            if is_active:
                changed = True
                active_incident_id = "slo-incident-" + hashlib.sha256(
                    f"{project_id}|{reason}".encode("utf-8")
                ).hexdigest()[:24]
                prior = conn.execute(
                    """SELECT * FROM trajectory_slo_incidents
                       WHERE project_id=? AND reason=?""",
                    (project_id, reason),
                ).fetchone()
                conn.execute(
                    """INSERT INTO trajectory_slo_incidents(
                           incident_id, project_id, reason, severity, status,
                           burn_rate, windows_json, operations, failures,
                           conversation_id, event_id, detected_at, updated_at
                       ) VALUES (?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(project_id, reason) DO UPDATE SET
                           severity=excluded.severity,
                           status=CASE
                               WHEN trajectory_slo_incidents.status='resolved' THEN 'open'
                               ELSE trajectory_slo_incidents.status
                           END,
                           burn_rate=excluded.burn_rate,
                           windows_json=excluded.windows_json,
                           operations=excluded.operations,
                           failures=excluded.failures,
                           conversation_id=excluded.conversation_id,
                           event_id=excluded.event_id,
                           updated_at=excluded.updated_at""",
                    (
                        active_incident_id, project_id, reason, incident_status,
                        burn_rate, _json_dump(window_rows),
                        int(slo.get("operations") or 0), int(slo.get("failures") or 0),
                        conversation_id, event_id, now, now,
                    ),
                )
                if prior is None or str(prior["status"]) == "resolved":
                    self._append_alert_history(
                        conn,
                        alert_id=active_incident_id,
                        project_id=project_id,
                        action="triggered" if prior is None else "reopened",
                        status="open",
                        details={
                            "metric": "slo.burn_rate", "severity": incident_status,
                            "observed": burn_rate, "threshold": 1,
                            "conversation_id": conversation_id, "event_id": event_id,
                            "source_type": "slo",
                        },
                        created_at=now,
                    )
                    notify_opened.append({
                        "incident_id": active_incident_id, "project_id": project_id,
                        "reason": reason, "severity": incident_status,
                        "burn_rate": burn_rate, "conversation_id": conversation_id,
                        "event_id": event_id,
                    })
            stale = conn.execute(
                """SELECT * FROM trajectory_slo_incidents
                   WHERE project_id=? AND status!='resolved' AND incident_id!=?""",
                (project_id, active_incident_id),
            ).fetchall()
            for row in stale:
                changed = True
                incident_id = str(row["incident_id"])
                conn.execute(
                    """UPDATE trajectory_slo_incidents
                       SET status='resolved', updated_at=? WHERE incident_id=?""",
                    (now, incident_id),
                )
                self._append_alert_history(
                    conn,
                    alert_id=incident_id,
                    project_id=project_id,
                    action="auto-resolved",
                    status="resolved",
                    details={
                        "metric": "slo.burn_rate", "severity": row["severity"],
                        "observed": row["burn_rate"], "threshold": 1,
                        "conversation_id": row["conversation_id"],
                        "event_id": row["event_id"], "source_type": "slo",
                    },
                    created_at=now,
                )
                notify_resolved.append({
                    "incident_id": incident_id, "project_id": project_id,
                })
        for incident in notify_opened:
            self._notify_slo_incident(incident, resolved=False)
        for incident in notify_resolved:
            self._notify_slo_incident(incident, resolved=True)
        if changed:
            self._emit_analytics_change(project_id=project_id, change="slo-incidents-synced")
        return self.list_slo_incidents(project_id=project_id)

    def update_slo_incident(
        self,
        *,
        project_id: str,
        incident_id: str,
        status: str,
    ) -> dict[str, Any]:
        normalized = str(status or "").strip().lower()
        if normalized not in {"open", "acknowledged", "resolved"}:
            raise ValueError("Unsupported SLO incident status")
        with self._connect() as conn:
            prior = conn.execute(
                """SELECT * FROM trajectory_slo_incidents
                   WHERE project_id=? AND incident_id=?""",
                (project_id, incident_id),
            ).fetchone()
            if not prior:
                raise KeyError(incident_id)
            now = _now_iso()
            conn.execute(
                """UPDATE trajectory_slo_incidents SET status=?, updated_at=?
                   WHERE project_id=? AND incident_id=?""",
                (normalized, now, project_id, incident_id),
            )
            if str(prior["status"]) != normalized:
                self._append_alert_history(
                    conn,
                    alert_id=incident_id,
                    project_id=project_id,
                    action={
                        "acknowledged": "acknowledged",
                        "resolved": "resolved",
                        "open": "reopened",
                    }[normalized],
                    status=normalized,
                    actor="operator",
                    details={
                        "metric": "slo.burn_rate", "severity": prior["severity"],
                        "observed": prior["burn_rate"], "threshold": 1,
                        "conversation_id": prior["conversation_id"],
                        "event_id": prior["event_id"], "source_type": "slo",
                    },
                    created_at=now,
                )
        if normalized == "resolved":
            self._notify_slo_incident(
                {"incident_id": incident_id, "project_id": project_id}, resolved=True
            )
        elif normalized == "open" and str(prior["status"]) == "resolved":
            self._notify_slo_incident({
                "incident_id": incident_id, "project_id": project_id,
                "reason": prior["reason"], "severity": prior["severity"],
                "burn_rate": prior["burn_rate"],
                "conversation_id": prior["conversation_id"],
                "event_id": prior["event_id"],
            }, resolved=False)
        self._emit_analytics_change(project_id=project_id, change="slo-incident-updated")
        return next(
            row for row in self.list_slo_incidents(project_id=project_id, limit=500)
            if row["incident_id"] == incident_id
        )

    @staticmethod
    def _eval_criteria(value: dict[str, Any] | None) -> dict[str, Any]:
        source = value if isinstance(value, dict) else {}
        result: dict[str, Any] = {}
        for key in (
            "max_failure_rate", "max_avg_request_ms", "max_tokens_per_request",
            "max_error_count", "max_sandbox_blocked", "min_trace_coverage",
        ):
            if key in source:
                result[key] = max(0.0, float(source.get(key) or 0))
        result["require_completed_request"] = bool(source.get("require_completed_request"))
        result["blocked_fingerprints"] = sorted({
            str(item)[:64] for item in (source.get("blocked_fingerprints") or [])
            if str(item).strip()
        })[:100]
        return result

    @staticmethod
    def _eval_metrics(value: dict[str, Any] | None) -> dict[str, Any]:
        source = value if isinstance(value, dict) else {}
        metrics: dict[str, Any] = {}
        for key in ("failure_rate", "avg_request_ms", "tokens_per_request"):
            if key in source:
                metrics[key] = float(source.get(key) or 0)
        for key in (
            "error_count", "warning_count", "trace_coverage",
            "completed_requests", "record_count",
            "sandbox_tool_calls", "sandbox_blocked_calls",
        ):
            if key in source:
                metrics[key] = max(0, int(source.get(key) or 0))
        if "health" in source:
            metrics["health"] = str(source.get("health") or "unknown")[:32]
        metrics["fingerprints"] = sorted({
            str(item)[:64] for item in (source.get("fingerprints") or [])
            if str(item).strip()
        })[:100]
        return metrics

    @staticmethod
    def _eval_checks(value: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        labels = {
            "max_failure_rate": "Failure rate",
            "max_avg_request_ms": "Average request latency",
            "max_tokens_per_request": "Tokens per request",
            "max_error_count": "Error findings",
            "max_sandbox_blocked": "Blocked sandbox calls",
            "min_trace_coverage": "Trace coverage",
            "require_completed_request": "Completed request",
            "blocked_fingerprints": "Incident fingerprints absent",
        }
        result = []
        for source in value or []:
            if not isinstance(source, dict):
                continue
            check_id = str(source.get("check_id") or "")
            if check_id not in labels:
                continue
            expected = source.get("expected")
            observed = source.get("observed")
            if check_id == "blocked_fingerprints":
                expected = sorted({
                    str(item)[:64] for item in (expected or []) if str(item).strip()
                })[:100]
                observed = sorted({
                    str(item)[:64] for item in (observed or []) if str(item).strip()
                })[:100]
            else:
                expected = float(expected or 0)
                observed = float(observed or 0)
            result.append({
                "check_id": check_id,
                "label": labels[check_id],
                "operator": "none" if check_id == "blocked_fingerprints" else str(
                    source.get("operator") or ""
                )[:4],
                "expected": expected,
                "observed": observed,
                "passed": bool(source.get("passed")),
            })
        return result

    @staticmethod
    def _eval_comparison(value: dict[str, Any] | None) -> dict[str, Any]:
        source = value if isinstance(value, dict) else {}
        result: dict[str, Any] = {}
        for key in (
            "failure_rate_delta", "avg_request_ms_delta",
            "tokens_per_request_delta", "error_count_delta",
        ):
            if key in source:
                result[key] = float(source.get(key) or 0)
        for key in ("removed_fingerprints", "new_fingerprints"):
            result[key] = sorted({
                str(item)[:64] for item in (source.get(key) or []) if str(item).strip()
            })[:100]
        return result

    @staticmethod
    def _eval_replay(value: dict[str, Any] | None) -> dict[str, Any]:
        source = value if isinstance(value, dict) else {}
        decisions = []
        for row in source.get("decisions") or []:
            if not isinstance(row, dict):
                continue
            action = str(row.get("action") or "")
            if action not in {"fixture-hit", "blocked"}:
                continue
            decisions.append({
                "tool": str(row.get("tool") or "")[:120],
                "action": action,
                "args_fingerprint": str(row.get("args_fingerprint") or "")[:64],
                "fixture_event_id": str(row.get("fixture_event_id") or "")[:100],
                "source_status": str(row.get("source_status") or "")[:32],
                "side_effect_executed": False,
            })
        return {
            "sandboxed": bool(source.get("sandboxed")),
            "status": "failed" if str(source.get("status") or "") == "failed" else "completed",
            "side_effects_executed": 0,
            "tool_calls": max(0, int(source.get("tool_calls") or 0)),
            "fixture_hits": max(0, int(source.get("fixture_hits") or 0)),
            "blocked_calls": max(0, int(source.get("blocked_calls") or 0)),
            "execution_error_type": str(source.get("execution_error_type") or "")[:120],
            "source_turn_id": str(source.get("source_turn_id") or "")[:100],
            "source_event_id": str(source.get("source_event_id") or "")[:100],
            "preferred_model": str(source.get("preferred_model") or "")[:240],
            "decisions": decisions[:100],
        }

    def create_eval_case(
        self,
        *,
        project_id: str,
        incident_id: str,
        name: str,
        source_conversation_id: str,
        criteria: dict[str, Any],
        baseline_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        now = _now_iso()
        case_id = f"trajectory-eval-{uuid.uuid4().hex}"
        clean_name = (str(name or "").strip() or "Trajectory regression")[:160]
        clean_criteria = self._eval_criteria(criteria)
        safe_baseline = {
            "status": (
                "passed" if str(baseline_snapshot.get("status") or "") == "passed"
                else "failed"
            ),
            "score": float(baseline_snapshot.get("score") or 0),
            "metrics": self._eval_metrics(baseline_snapshot.get("metrics")),
        }
        with self._connect() as conn:
            existing = conn.execute(
                """SELECT case_id FROM trajectory_eval_cases
                   WHERE project_id=? AND incident_id=?""",
                (project_id, incident_id),
            ).fetchone()
            if existing:
                raise ValueError("This incident already has a regression eval case")
            conn.execute(
                """INSERT INTO trajectory_eval_cases(
                       case_id, project_id, incident_id, name,
                       source_conversation_id, criteria_json,
                       baseline_snapshot_json, enabled, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                (
                    case_id, project_id, incident_id, clean_name,
                    source_conversation_id, _json_dump(clean_criteria),
                    _json_dump(safe_baseline), now, now,
                ),
            )
        self._emit_analytics_change(project_id=project_id, change="eval-case-created")
        return self.get_eval_case(project_id=project_id, case_id=case_id)

    def get_eval_case(self, *, project_id: str, case_id: str) -> dict[str, Any]:
        for row in self.list_eval_cases(project_id=project_id, limit=500):
            if row["case_id"] == case_id:
                return row
        raise KeyError(case_id)

    def list_eval_cases(
        self,
        *,
        project_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM trajectory_eval_cases
                   WHERE project_id=? ORDER BY enabled DESC, updated_at DESC LIMIT ?""",
                (project_id, max(1, min(int(limit), 500))),
            ).fetchall()
            result = []
            for row in rows:
                latest = conn.execute(
                    """SELECT * FROM trajectory_eval_runs
                       WHERE project_id=? AND case_id=?
                       ORDER BY created_at DESC LIMIT 1""",
                    (project_id, row["case_id"]),
                ).fetchone()
                item = {
                    "case_id": str(row["case_id"]),
                    "project_id": str(row["project_id"]),
                    "incident_id": str(row["incident_id"]),
                    "name": str(row["name"]),
                    "source_conversation_id": str(row["source_conversation_id"]),
                    "criteria": json.loads(row["criteria_json"] or "{}"),
                    "baseline_snapshot": json.loads(row["baseline_snapshot_json"] or "{}"),
                    "enabled": bool(row["enabled"]),
                    "created_at": str(row["created_at"]),
                    "updated_at": str(row["updated_at"]),
                    "latest_run": self._eval_run_row(latest) if latest else None,
                }
                result.append(item)
        return result

    @staticmethod
    def _eval_run_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "run_id": str(row["run_id"]),
            "case_id": str(row["case_id"]),
            "project_id": str(row["project_id"]),
            "candidate_conversation_id": str(row["candidate_conversation_id"]),
            "mode": str(row["mode"] or "recorded"),
            "status": str(row["status"]),
            "score": float(row["score"] or 0),
            "checks": json.loads(row["checks_json"] or "[]"),
            "metrics": json.loads(row["metrics_json"] or "{}"),
            "comparison": json.loads(row["comparison_json"] or "{}"),
            "replay": json.loads(row["replay_json"] or "{}"),
            "created_at": str(row["created_at"]),
        }

    def list_eval_runs(
        self,
        *,
        project_id: str,
        case_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM trajectory_eval_runs
                   WHERE project_id=? AND case_id=?
                   ORDER BY created_at DESC LIMIT ?""",
                (project_id, case_id, max(1, min(int(limit), 500))),
            ).fetchall()
        return [self._eval_run_row(row) for row in rows]

    def record_eval_run(
        self,
        *,
        project_id: str,
        case_id: str,
        candidate_conversation_id: str,
        evaluation: dict[str, Any],
        mode: str = "recorded",
        replay: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.get_eval_case(project_id=project_id, case_id=case_id)
        run_id = f"trajectory-eval-run-{uuid.uuid4().hex}"
        created_at = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO trajectory_eval_runs(
                       run_id, case_id, project_id, candidate_conversation_id,
                       mode, status, score, checks_json, metrics_json,
                       comparison_json, replay_json, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, case_id, project_id, candidate_conversation_id,
                    "sandbox-replay" if mode == "sandbox-replay" else "recorded",
                    str(evaluation.get("status") or "failed"),
                    float(evaluation.get("score") or 0),
                    _json_dump(self._eval_checks(evaluation.get("checks"))),
                    _json_dump(self._eval_metrics(evaluation.get("metrics"))),
                    _json_dump(self._eval_comparison(evaluation.get("comparison"))),
                    _json_dump(self._eval_replay(replay)),
                    created_at,
                ),
            )
            row = conn.execute(
                "SELECT * FROM trajectory_eval_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
        self._emit_analytics_change(project_id=project_id, change="eval-run-recorded")
        return self._eval_run_row(row)

    def delete_eval_case(self, *, project_id: str, case_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM trajectory_eval_cases WHERE project_id=? AND case_id=?",
                (project_id, case_id),
            )
            if not cursor.rowcount:
                return False
            conn.execute(
                "DELETE FROM trajectory_eval_runs WHERE project_id=? AND case_id=?",
                (project_id, case_id),
            )
        self._emit_analytics_change(project_id=project_id, change="eval-case-deleted")
        return True

    @staticmethod
    def _eval_matrix_entry_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "entry_id": str(row["entry_id"]),
            "matrix_id": str(row["matrix_id"]),
            "project_id": str(row["project_id"]),
            "case_id": str(row["case_id"]),
            "run_id": str(row["run_id"] or ""),
            "candidate_conversation_id": str(row["candidate_conversation_id"] or ""),
            "status": str(row["status"]),
            "score": float(row["score"] or 0),
            "failure_rate": float(row["failure_rate"] or 0),
            "avg_request_ms": float(row["avg_request_ms"] or 0),
            "tokens_per_request": float(row["tokens_per_request"] or 0),
            "error_type": str(row["error_type"] or ""),
            "created_at": str(row["created_at"]),
        }

    def _eval_matrix_row(
        self,
        row: sqlite3.Row,
        *,
        conn: sqlite3.Connection,
    ) -> dict[str, Any]:
        entries = conn.execute(
            """SELECT * FROM trajectory_eval_matrix_entries
               WHERE project_id=? AND matrix_id=? ORDER BY created_at, entry_id""",
            (row["project_id"], row["matrix_id"]),
        ).fetchall()
        completed_entries = [item for item in entries if str(item["status"]) != "error"]
        divisor = max(1, len(completed_entries))
        return {
            "matrix_id": str(row["matrix_id"]),
            "project_id": str(row["project_id"]),
            "name": str(row["name"]),
            "preferred_model": str(row["preferred_model"] or ""),
            "agent_version": str(row["agent_version"] or ""),
            "status": str(row["status"]),
            "gate_passed": str(row["status"]) == "passed",
            "case_count": int(row["case_count"] or 0),
            "passed_count": int(row["passed_count"] or 0),
            "failed_count": int(row["failed_count"] or 0),
            "error_count": int(row["error_count"] or 0),
            "avg_score": round(
                sum(float(item["score"] or 0) for item in completed_entries) / divisor, 2
            ),
            "avg_failure_rate": round(
                sum(float(item["failure_rate"] or 0) for item in completed_entries) / divisor,
                6,
            ),
            "avg_request_ms": round(
                sum(float(item["avg_request_ms"] or 0) for item in completed_entries) / divisor,
                2,
            ),
            "avg_tokens_per_request": round(
                sum(float(item["tokens_per_request"] or 0) for item in completed_entries)
                / divisor,
                2,
            ),
            "created_at": str(row["created_at"]),
            "completed_at": str(row["completed_at"] or ""),
            "entries": [self._eval_matrix_entry_row(item) for item in entries],
        }

    def create_eval_matrix(
        self,
        *,
        project_id: str,
        name: str,
        preferred_model: str,
        agent_version: str,
        case_count: int,
    ) -> dict[str, Any]:
        matrix_id = f"trajectory-matrix-{uuid.uuid4().hex}"
        created_at = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO trajectory_eval_matrices(
                       matrix_id, project_id, name, preferred_model, agent_version,
                       status, case_count, created_at
                   ) VALUES (?, ?, ?, ?, ?, 'running', ?, ?)""",
                (
                    matrix_id, project_id,
                    (str(name or "").strip() or "Trajectory release gate")[:160],
                    str(preferred_model or "")[:240],
                    str(agent_version or "")[:80],
                    max(0, int(case_count)), created_at,
                ),
            )
            row = conn.execute(
                "SELECT * FROM trajectory_eval_matrices WHERE matrix_id=?",
                (matrix_id,),
            ).fetchone()
            result = self._eval_matrix_row(row, conn=conn)
        self._emit_analytics_change(project_id=project_id, change="eval-matrix-created")
        return result

    def record_eval_matrix_entry(
        self,
        *,
        project_id: str,
        matrix_id: str,
        case_id: str,
        status: str,
        score: float = 0,
        failure_rate: float = 0,
        avg_request_ms: float = 0,
        tokens_per_request: float = 0,
        run_id: str = "",
        candidate_conversation_id: str = "",
        error_type: str = "",
    ) -> dict[str, Any]:
        normalized = str(status or "failed").lower()
        if normalized not in {"passed", "failed", "error"}:
            normalized = "error"
        raw_error_type = str(error_type or "")
        clean_error_type = (
            raw_error_type[:120]
            if raw_error_type and all(
                character.isalnum() or character in {"_", "-", "."}
                for character in raw_error_type
            )
            else "ReplayError" if raw_error_type else ""
        )
        entry_id = f"matrix-entry-{uuid.uuid4().hex}"
        with self._connect() as conn:
            matrix = conn.execute(
                """SELECT matrix_id FROM trajectory_eval_matrices
                   WHERE project_id=? AND matrix_id=?""",
                (project_id, matrix_id),
            ).fetchone()
            if not matrix:
                raise KeyError(matrix_id)
            conn.execute(
                """INSERT INTO trajectory_eval_matrix_entries(
                       entry_id, matrix_id, project_id, case_id, run_id,
                       candidate_conversation_id, status, score, failure_rate,
                       avg_request_ms, tokens_per_request, error_type, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry_id, matrix_id, project_id, case_id,
                    str(run_id or "")[:100], str(candidate_conversation_id or "")[:160],
                    normalized, max(0.0, min(float(score or 0), 100.0)),
                    max(0.0, min(float(failure_rate or 0), 1.0)),
                    max(0.0, float(avg_request_ms or 0)),
                    max(0.0, float(tokens_per_request or 0)),
                    clean_error_type, _now_iso(),
                ),
            )
            row = conn.execute(
                "SELECT * FROM trajectory_eval_matrix_entries WHERE entry_id=?",
                (entry_id,),
            ).fetchone()
        return self._eval_matrix_entry_row(row)

    def complete_eval_matrix(self, *, project_id: str, matrix_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            matrix = conn.execute(
                """SELECT * FROM trajectory_eval_matrices
                   WHERE project_id=? AND matrix_id=?""",
                (project_id, matrix_id),
            ).fetchone()
            if not matrix:
                raise KeyError(matrix_id)
            counts = conn.execute(
                """SELECT status, COUNT(*) AS total
                   FROM trajectory_eval_matrix_entries
                   WHERE project_id=? AND matrix_id=? GROUP BY status""",
                (project_id, matrix_id),
            ).fetchall()
            by_status = {str(row["status"]): int(row["total"] or 0) for row in counts}
            expected = int(matrix["case_count"] or 0)
            passed = by_status.get("passed", 0)
            failed = by_status.get("failed", 0)
            errors = by_status.get("error", 0)
            gate_passed = expected > 0 and passed == expected and not failed and not errors
            conn.execute(
                """UPDATE trajectory_eval_matrices SET
                       status=?, passed_count=?, failed_count=?, error_count=?, completed_at=?
                   WHERE project_id=? AND matrix_id=?""",
                (
                    "passed" if gate_passed else "failed",
                    passed, failed, errors, _now_iso(), project_id, matrix_id,
                ),
            )
            updated = conn.execute(
                "SELECT * FROM trajectory_eval_matrices WHERE matrix_id=?",
                (matrix_id,),
            ).fetchone()
            result = self._eval_matrix_row(updated, conn=conn)
        self._emit_analytics_change(project_id=project_id, change="eval-matrix-completed")
        return result

    def list_eval_matrices(
        self,
        *,
        project_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM trajectory_eval_matrices
                   WHERE project_id=? ORDER BY created_at DESC LIMIT ?""",
                (project_id, max(1, min(int(limit), 100))),
            ).fetchall()
            return [self._eval_matrix_row(row, conn=conn) for row in rows]

    @staticmethod
    def _eval_comparison_model_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "comparison_model_id": str(row["comparison_model_id"]),
            "comparison_id": str(row["comparison_id"]),
            "matrix_id": str(row["matrix_id"] or ""),
            "project_id": str(row["project_id"]),
            "preferred_model": str(row["preferred_model"]),
            "status": str(row["status"]),
            "gate_passed": bool(row["gate_passed"]),
            "passed_count": int(row["passed_count"] or 0),
            "failed_count": int(row["failed_count"] or 0),
            "error_count": int(row["error_count"] or 0),
            "avg_score": float(row["avg_score"] or 0),
            "avg_failure_rate": float(row["avg_failure_rate"] or 0),
            "avg_request_ms": float(row["avg_request_ms"] or 0),
            "avg_tokens_per_request": float(row["avg_tokens_per_request"] or 0),
            "rank": int(row["rank"] or 0),
            "created_at": str(row["created_at"]),
        }

    def _eval_comparison_row(
        self,
        row: sqlite3.Row,
        *,
        conn: sqlite3.Connection,
    ) -> dict[str, Any]:
        models = conn.execute(
            """SELECT * FROM trajectory_eval_comparison_models
               WHERE project_id=? AND comparison_id=?
               ORDER BY CASE WHEN rank=0 THEN 999999 ELSE rank END, created_at""",
            (row["project_id"], row["comparison_id"]),
        ).fetchall()
        return {
            "comparison_id": str(row["comparison_id"]),
            "project_id": str(row["project_id"]),
            "name": str(row["name"]),
            "agent_version": str(row["agent_version"] or ""),
            "status": str(row["status"]),
            "model_count": int(row["model_count"] or 0),
            "case_count": int(row["case_count"] or 0),
            "winner_model": str(row["winner_model"] or ""),
            "created_at": str(row["created_at"]),
            "completed_at": str(row["completed_at"] or ""),
            "models": [self._eval_comparison_model_row(item) for item in models],
        }

    def create_eval_comparison(
        self,
        *,
        project_id: str,
        name: str,
        agent_version: str,
        model_count: int,
        case_count: int,
    ) -> dict[str, Any]:
        comparison_id = f"trajectory-comparison-{uuid.uuid4().hex}"
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO trajectory_eval_comparisons(
                       comparison_id, project_id, name, agent_version, status,
                       model_count, case_count, created_at
                   ) VALUES (?, ?, ?, ?, 'running', ?, ?, ?)""",
                (
                    comparison_id,
                    project_id,
                    (str(name or "").strip() or "Trajectory model comparison")[:160],
                    str(agent_version or "")[:80],
                    max(0, int(model_count)),
                    max(0, int(case_count)),
                    _now_iso(),
                ),
            )
            row = conn.execute(
                "SELECT * FROM trajectory_eval_comparisons WHERE comparison_id=?",
                (comparison_id,),
            ).fetchone()
            result = self._eval_comparison_row(row, conn=conn)
        self._emit_analytics_change(project_id=project_id, change="eval-comparison-created")
        return result

    def record_eval_comparison_model(
        self,
        *,
        project_id: str,
        comparison_id: str,
        matrix: dict[str, Any],
    ) -> dict[str, Any]:
        model_id = f"comparison-model-{uuid.uuid4().hex}"
        with self._connect() as conn:
            parent = conn.execute(
                """SELECT comparison_id FROM trajectory_eval_comparisons
                   WHERE project_id=? AND comparison_id=?""",
                (project_id, comparison_id),
            ).fetchone()
            if not parent:
                raise KeyError(comparison_id)
            conn.execute(
                """INSERT INTO trajectory_eval_comparison_models(
                       comparison_model_id, comparison_id, matrix_id, project_id,
                       preferred_model, status, gate_passed, passed_count,
                       failed_count, error_count, avg_score, avg_failure_rate,
                       avg_request_ms, avg_tokens_per_request, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    model_id, comparison_id, str(matrix.get("matrix_id") or "")[:100],
                    project_id, str(matrix.get("preferred_model") or "")[:240],
                    str(matrix.get("status") or "failed")[:32],
                    int(bool(matrix.get("gate_passed"))),
                    max(0, int(matrix.get("passed_count") or 0)),
                    max(0, int(matrix.get("failed_count") or 0)),
                    max(0, int(matrix.get("error_count") or 0)),
                    max(0.0, min(float(matrix.get("avg_score") or 0), 100.0)),
                    max(0.0, min(float(matrix.get("avg_failure_rate") or 0), 1.0)),
                    max(0.0, float(matrix.get("avg_request_ms") or 0)),
                    max(0.0, float(matrix.get("avg_tokens_per_request") or 0)),
                    _now_iso(),
                ),
            )
            row = conn.execute(
                "SELECT * FROM trajectory_eval_comparison_models WHERE comparison_model_id=?",
                (model_id,),
            ).fetchone()
        return self._eval_comparison_model_row(row)

    def complete_eval_comparison(
        self,
        *,
        project_id: str,
        comparison_id: str,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            comparison = conn.execute(
                """SELECT * FROM trajectory_eval_comparisons
                   WHERE project_id=? AND comparison_id=?""",
                (project_id, comparison_id),
            ).fetchone()
            if not comparison:
                raise KeyError(comparison_id)
            rows = conn.execute(
                """SELECT * FROM trajectory_eval_comparison_models
                   WHERE project_id=? AND comparison_id=?""",
                (project_id, comparison_id),
            ).fetchall()

            def ranking_key(item: sqlite3.Row) -> tuple[Any, ...]:
                latency = float(item["avg_request_ms"] or 0)
                tokens = float(item["avg_tokens_per_request"] or 0)
                return (
                    -int(bool(item["gate_passed"])),
                    -int(item["passed_count"] or 0),
                    int(item["error_count"] or 0),
                    int(item["failed_count"] or 0),
                    -float(item["avg_score"] or 0),
                    latency if latency > 0 else float("inf"),
                    tokens if tokens > 0 else float("inf"),
                    str(item["preferred_model"]),
                )

            ranked = sorted(rows, key=ranking_key)
            viable = [
                item for item in ranked
                if int(item["passed_count"] or 0) + int(item["failed_count"] or 0) > 0
            ]
            winner = str(viable[0]["preferred_model"]) if viable else ""
            for rank, item in enumerate(ranked, start=1):
                conn.execute(
                    "UPDATE trajectory_eval_comparison_models SET rank=? WHERE comparison_model_id=?",
                    (rank, item["comparison_model_id"]),
                )
            conn.execute(
                """UPDATE trajectory_eval_comparisons
                   SET status=?, winner_model=?, completed_at=?
                   WHERE project_id=? AND comparison_id=?""",
                (
                    "completed" if winner else "failed", winner, _now_iso(),
                    project_id, comparison_id,
                ),
            )
            updated = conn.execute(
                "SELECT * FROM trajectory_eval_comparisons WHERE comparison_id=?",
                (comparison_id,),
            ).fetchone()
            result = self._eval_comparison_row(updated, conn=conn)
        self._emit_analytics_change(project_id=project_id, change="eval-comparison-completed")
        return result

    def list_eval_comparisons(
        self,
        *,
        project_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM trajectory_eval_comparisons
                   WHERE project_id=? ORDER BY created_at DESC LIMIT ?""",
                (project_id, max(1, min(int(limit), 100))),
            ).fetchall()
            return [self._eval_comparison_row(row, conn=conn) for row in rows]

    @staticmethod
    def _model_promotion_row(row: sqlite3.Row) -> dict[str, Any]:
        evidence = json.loads(row["evidence_json"] or "[]")
        ramp_history = json.loads(row["ramp_history_json"] or "[]")
        return {
            "promotion_id": str(row["promotion_id"]),
            "project_id": str(row["project_id"]),
            "previous_model": str(row["previous_model"]),
            "candidate_model": str(row["candidate_model"]),
            "status": str(row["status"]),
            "canary_percent": int(row["canary_percent"] or 0),
            "ramp_stage": int(row["ramp_stage"] or 0),
            "ramp_complete": bool(row["ramp_complete"]),
            "healthy_windows": int(row["healthy_windows"] or 0),
            "required_healthy_windows": int(row["required_healthy_windows"] or 0),
            "health_window_requests": int(row["health_window_requests"] or 0),
            "health_window_pause_seconds": int(row["health_window_pause_seconds"] or 0),
            "max_requests_per_arm": int(row["max_requests_per_arm"] or 0),
            "familywise_alpha": float(row["familywise_alpha"] or 0),
            "max_canary_hours": int(row["max_canary_hours"] or 0),
            "last_health_candidate_requests": int(
                row["last_health_candidate_requests"] or 0
            ),
            "last_health_control_requests": int(row["last_health_control_requests"] or 0),
            "last_telemetry_candidate_requests": int(
                row["last_telemetry_candidate_requests"] or 0
            ),
            "last_telemetry_control_requests": int(
                row["last_telemetry_control_requests"] or 0
            ),
            "last_health_window_at": str(row["last_health_window_at"] or ""),
            "stage_started_at": str(row["stage_started_at"] or ""),
            "ramp_history": ramp_history if isinstance(ramp_history, list) else [],
            "healthy_comparisons": int(row["healthy_comparisons"] or 0),
            "required_healthy_comparisons": int(
                row["required_healthy_comparisons"] or 0
            ),
            "evidence_comparison_ids": [
                str(value)[:100] for value in evidence if str(value)
            ],
            "last_comparison_id": str(row["last_comparison_id"] or ""),
            "rollback_reason": str(row["rollback_reason"] or ""),
            "started_at": str(row["started_at"]),
            "updated_at": str(row["updated_at"]),
            "promoted_at": str(row["promoted_at"] or ""),
            "completed_at": str(row["completed_at"] or ""),
        }

    def create_model_promotion(
        self,
        *,
        project_id: str,
        previous_model: str,
        candidate_model: str,
        canary_percent: int,
        evidence_comparison_ids: list[str],
        required_healthy_comparisons: int = 2,
        required_healthy_windows: int = 2,
        health_window_requests: int = 1,
        health_window_pause_seconds: int = 300,
        max_requests_per_arm: int = 200,
        familywise_alpha: float = 0.05,
        max_canary_hours: int = 24,
    ) -> dict[str, Any]:
        previous = str(previous_model or "").strip()[:240]
        candidate = str(candidate_model or "").strip()[:240]
        if not previous or not candidate or previous == candidate:
            raise ValueError("Promotion requires distinct current and candidate models")
        promotion_id = f"trajectory-promotion-{uuid.uuid4().hex}"
        now = _now_iso()
        evidence = [str(value)[:100] for value in evidence_comparison_ids if str(value)][:10]
        initial_history = [{
            "stage": 0,
            "percent": 10,
            "at": now,
            "reason": "canary_started",
        }]
        with self._connect() as conn:
            active = conn.execute(
                """SELECT promotion_id FROM trajectory_model_promotions
                   WHERE project_id=? AND status IN ('canary', 'ready', 'promoted', 'rollback_pending')
                   LIMIT 1""",
                (project_id,),
            ).fetchone()
            if active:
                raise ValueError("An active model promotion already exists")
            conn.execute(
                """INSERT INTO trajectory_model_promotions(
                       promotion_id, project_id, previous_model, candidate_model,
                       status, canary_percent, ramp_stage, ramp_complete,
                       healthy_windows, required_healthy_windows,
                       health_window_requests, health_window_pause_seconds,
                       max_requests_per_arm, familywise_alpha, max_canary_hours,
                       stage_started_at, ramp_history_json, healthy_comparisons,
                       required_healthy_comparisons, evidence_json, started_at, updated_at
                   ) VALUES (?, ?, ?, ?, 'canary', 10, 0, 0, 0, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)""",
                (
                    promotion_id, project_id, previous, candidate,
                    max(1, min(int(required_healthy_windows), 10)),
                    max(1, min(int(health_window_requests), 1000)),
                    max(60, min(int(health_window_pause_seconds), 86_400)),
                    max(10, min(int(max_requests_per_arm), 10_000)),
                    max(0.001, min(float(familywise_alpha), 0.25)),
                    max(1, min(int(max_canary_hours), 24 * 30)),
                    now, _json_dump(initial_history),
                    max(1, min(int(required_healthy_comparisons), 10)),
                    _json_dump(evidence), now, now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM trajectory_model_promotions WHERE promotion_id=?",
                (promotion_id,),
            ).fetchone()
            result = self._model_promotion_row(row)
        self._emit_analytics_change(project_id=project_id, change="model-promotion-started")
        return result

    def get_active_model_promotion(self, *, project_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM trajectory_model_promotions
                   WHERE project_id=? AND status IN ('canary', 'ready', 'promoted', 'rollback_pending')
                   ORDER BY started_at DESC LIMIT 1""",
                (project_id,),
            ).fetchone()
            return self._model_promotion_row(row) if row else None

    def get_model_promotion(
        self,
        *,
        project_id: str,
        promotion_id: str,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM trajectory_model_promotions
                   WHERE project_id=? AND promotion_id=?""",
                (project_id, promotion_id),
            ).fetchone()
            if not row:
                raise KeyError(promotion_id)
            return self._model_promotion_row(row)

    def list_model_promotions(
        self,
        *,
        project_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM trajectory_model_promotions
                   WHERE project_id=? ORDER BY started_at DESC LIMIT ?""",
                (project_id, max(1, min(int(limit), 100))),
            ).fetchall()
            return [self._model_promotion_row(row) for row in rows]

    def observe_model_promotion_comparison(
        self,
        *,
        project_id: str,
        comparison: dict[str, Any],
    ) -> dict[str, Any] | None:
        comparison_id = str(comparison.get("comparison_id") or "")[:100]
        if not comparison_id:
            return self.get_active_model_promotion(project_id=project_id)
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM trajectory_model_promotions
                   WHERE project_id=? AND status IN ('canary', 'ready', 'promoted', 'rollback_pending')
                   ORDER BY started_at DESC LIMIT 1""",
                (project_id,),
            ).fetchone()
            if not row:
                return None
            if str(row["status"]) == "rollback_pending":
                return self._model_promotion_row(row)
            if str(row["last_comparison_id"] or "") == comparison_id:
                return self._model_promotion_row(row)
            comparison_time = str(
                comparison.get("completed_at") or comparison.get("created_at") or ""
            )
            if comparison_time and comparison_time < str(row["started_at"]):
                return self._model_promotion_row(row)
            models = comparison.get("models") if isinstance(comparison.get("models"), list) else []
            candidate = str(row["candidate_model"])
            previous = str(row["previous_model"])
            candidate_row = next(
                (item for item in models if str(item.get("preferred_model") or "") == candidate),
                None,
            )
            previous_row = next(
                (item for item in models if str(item.get("preferred_model") or "") == previous),
                None,
            )
            rollback_reason = ""
            if not candidate_row:
                rollback_reason = "candidate_missing"
            elif not previous_row:
                rollback_reason = "rollback_model_missing"
            elif int(candidate_row.get("error_count") or 0) > 0:
                rollback_reason = "candidate_replay_error"
            elif not bool(candidate_row.get("gate_passed")):
                rollback_reason = "candidate_gate_failed"
            elif str(comparison.get("winner_model") or "") != candidate:
                rollback_reason = "candidate_not_winner"
            elif int(candidate_row.get("rank") or 999999) >= int(
                previous_row.get("rank") or 999999
            ):
                rollback_reason = "candidate_not_better"
            now = _now_iso()
            if rollback_reason:
                if str(row["status"]) == "promoted":
                    conn.execute(
                        """UPDATE trajectory_model_promotions SET
                               status='rollback_pending', rollback_reason=?,
                               last_comparison_id=?, updated_at=?
                           WHERE promotion_id=?""",
                        (rollback_reason, comparison_id, now, row["promotion_id"]),
                    )
                else:
                    conn.execute(
                        """UPDATE trajectory_model_promotions SET
                               status='rolled_back', rollback_reason=?,
                               last_comparison_id=?, updated_at=?, completed_at=?
                           WHERE promotion_id=?""",
                        (rollback_reason, comparison_id, now, now, row["promotion_id"]),
                    )
            else:
                healthy = int(row["healthy_comparisons"] or 0)
                if str(row["status"]) in {"canary", "ready"}:
                    healthy += 1
                required = int(row["required_healthy_comparisons"] or 0)
                status = "ready" if healthy >= required else str(row["status"])
                conn.execute(
                    """UPDATE trajectory_model_promotions SET
                           status=?, healthy_comparisons=?, last_comparison_id=?, updated_at=?
                       WHERE promotion_id=?""",
                    (status, healthy, comparison_id, now, row["promotion_id"]),
                )
            updated = conn.execute(
                "SELECT * FROM trajectory_model_promotions WHERE promotion_id=?",
                (row["promotion_id"],),
            ).fetchone()
            result = self._model_promotion_row(updated)
        self._emit_analytics_change(project_id=project_id, change="model-promotion-observed")
        return result

    def observe_model_promotion_telemetry(
        self,
        *,
        project_id: str,
        promotion_id: str,
        telemetry: dict[str, Any],
        observed_at: str | None = None,
    ) -> dict[str, Any]:
        """Count independent healthy windows and advance the 10/25/50 ramp."""
        if str(telemetry.get("status") or "") == "regressed":
            return self.rollback_model_promotion(
                project_id=project_id,
                promotion_id=promotion_id,
                reason="canary_telemetry_regression",
            )
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM trajectory_model_promotions
                   WHERE project_id=? AND promotion_id=?""",
                (project_id, promotion_id),
            ).fetchone()
            if not row:
                raise KeyError(promotion_id)
            if str(row["status"]) not in {"canary", "ready"}:
                return self._model_promotion_row(row)
            if bool(row["ramp_complete"]) or str(telemetry.get("status") or "") != "healthy":
                return self._model_promotion_row(row)
            candidate_requests = int((telemetry.get("candidate") or {}).get("requests") or 0)
            control_requests = int((telemetry.get("control") or {}).get("requests") or 0)
            last_telemetry_candidate = int(
                row["last_telemetry_candidate_requests"] or 0
            )
            last_telemetry_control = int(row["last_telemetry_control_requests"] or 0)
            if (
                candidate_requests == last_telemetry_candidate
                and control_requests == last_telemetry_control
            ):
                return self._model_promotion_row(row)
            conn.execute(
                """UPDATE trajectory_model_promotions SET
                       last_telemetry_candidate_requests=?,
                       last_telemetry_control_requests=?
                   WHERE promotion_id=?""",
                (candidate_requests, control_requests, promotion_id),
            )
            last_candidate = int(row["last_health_candidate_requests"] or 0)
            last_control = int(row["last_health_control_requests"] or 0)
            if (
                candidate_requests <= last_candidate
                or control_requests <= last_control
            ):
                updated = conn.execute(
                    "SELECT * FROM trajectory_model_promotions WHERE promotion_id=?",
                    (promotion_id,),
                ).fetchone()
                return self._model_promotion_row(updated)
            now = str(observed_at or _now_iso())
            last_window = str(row["last_health_window_at"] or "")
            if last_window:
                try:
                    elapsed = (
                        datetime.fromisoformat(now) - datetime.fromisoformat(last_window)
                    ).total_seconds()
                except ValueError:
                    elapsed = 0
                if elapsed < int(row["health_window_pause_seconds"] or 300):
                    updated = conn.execute(
                        "SELECT * FROM trajectory_model_promotions WHERE promotion_id=?",
                        (promotion_id,),
                    ).fetchone()
                    return self._model_promotion_row(updated)
            healthy_windows = int(row["healthy_windows"] or 0) + 1
            required_windows = int(row["required_healthy_windows"] or 2)
            stage = int(row["ramp_stage"] or 0)
            schedule = (10, 25, 50)
            history = json.loads(row["ramp_history_json"] or "[]")
            history = history if isinstance(history, list) else []
            if healthy_windows >= required_windows and stage < len(schedule) - 1:
                next_stage = stage + 1
                history.append({
                    "stage": next_stage,
                    "percent": schedule[next_stage],
                    "at": now,
                    "reason": "healthy_windows_complete",
                    "candidate_requests": candidate_requests,
                    "control_requests": control_requests,
                })
                conn.execute(
                    """UPDATE trajectory_model_promotions SET
                           canary_percent=?, ramp_stage=?, healthy_windows=0,
                           last_health_candidate_requests=0,
                           last_health_control_requests=0,
                           last_telemetry_candidate_requests=0,
                           last_telemetry_control_requests=0,
                           last_health_window_at='', stage_started_at=?,
                           ramp_history_json=?, updated_at=?
                       WHERE promotion_id=?""",
                    (
                        schedule[next_stage], next_stage, now,
                        _json_dump(history[-10:]), now, promotion_id,
                    ),
                )
            elif healthy_windows >= required_windows:
                history.append({
                    "stage": stage,
                    "percent": schedule[stage],
                    "at": now,
                    "reason": "ramp_complete",
                    "candidate_requests": candidate_requests,
                    "control_requests": control_requests,
                })
                conn.execute(
                    """UPDATE trajectory_model_promotions SET
                           ramp_complete=1, healthy_windows=?,
                           last_health_candidate_requests=?,
                           last_health_control_requests=?, last_health_window_at=?,
                           ramp_history_json=?, updated_at=?
                       WHERE promotion_id=?""",
                    (
                        healthy_windows, candidate_requests, control_requests, now,
                        _json_dump(history[-10:]), now, promotion_id,
                    ),
                )
            else:
                conn.execute(
                    """UPDATE trajectory_model_promotions SET
                           healthy_windows=?, last_health_candidate_requests=?,
                           last_health_control_requests=?, last_health_window_at=?, updated_at=?
                       WHERE promotion_id=?""",
                    (
                        healthy_windows, candidate_requests, control_requests,
                        now, now, promotion_id,
                    ),
                )
            updated = conn.execute(
                "SELECT * FROM trajectory_model_promotions WHERE promotion_id=?",
                (promotion_id,),
            ).fetchone()
            result = self._model_promotion_row(updated)
        self._emit_analytics_change(project_id=project_id, change="model-promotion-ramped")
        return result

    def mark_model_promotion_promoted(
        self,
        *,
        project_id: str,
        promotion_id: str,
    ) -> dict[str, Any]:
        now = _now_iso()
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM trajectory_model_promotions
                   WHERE project_id=? AND promotion_id=?""",
                (project_id, promotion_id),
            ).fetchone()
            if not row:
                raise KeyError(promotion_id)
            if str(row["status"]) != "ready" or not bool(row["ramp_complete"]):
                raise ValueError("Model promotion is not ready")
            conn.execute(
                """UPDATE trajectory_model_promotions
                   SET status='promoted', promoted_at=?, updated_at=?
                   WHERE promotion_id=?""",
                (now, now, promotion_id),
            )
            updated = conn.execute(
                "SELECT * FROM trajectory_model_promotions WHERE promotion_id=?",
                (promotion_id,),
            ).fetchone()
            result = self._model_promotion_row(updated)
        self._emit_analytics_change(project_id=project_id, change="model-promoted")
        return result

    def rollback_model_promotion(
        self,
        *,
        project_id: str,
        promotion_id: str,
        reason: str = "manual_rollback",
    ) -> dict[str, Any]:
        allowed_reasons = {
            "manual_rollback", "promotion_apply_failed", "canary_telemetry_regression"
        }
        clean_reason = reason if reason in allowed_reasons else "manual_rollback"
        now = _now_iso()
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM trajectory_model_promotions
                   WHERE project_id=? AND promotion_id=?""",
                (project_id, promotion_id),
            ).fetchone()
            if not row:
                raise KeyError(promotion_id)
            if str(row["status"]) not in {
                "canary", "ready", "promoted", "rollback_pending"
            }:
                raise ValueError("Model promotion is not active")
            conn.execute(
                """UPDATE trajectory_model_promotions SET
                       status='rolled_back', rollback_reason=?, updated_at=?, completed_at=?
                   WHERE promotion_id=?""",
                (clean_reason, now, now, promotion_id),
            )
            updated = conn.execute(
                "SELECT * FROM trajectory_model_promotions WHERE promotion_id=?",
                (promotion_id,),
            ).fetchone()
            result = self._model_promotion_row(updated)
        self._emit_analytics_change(project_id=project_id, change="model-promotion-rolled-back")
        return result

    def complete_model_promotion_auto_rollback(
        self,
        *,
        project_id: str,
        promotion_id: str,
    ) -> dict[str, Any]:
        now = _now_iso()
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM trajectory_model_promotions
                   WHERE project_id=? AND promotion_id=?""",
                (project_id, promotion_id),
            ).fetchone()
            if not row:
                raise KeyError(promotion_id)
            if str(row["status"]) != "rollback_pending":
                raise ValueError("Automatic rollback is not pending")
            conn.execute(
                """UPDATE trajectory_model_promotions
                   SET status='rolled_back', updated_at=?, completed_at=?
                   WHERE promotion_id=?""",
                (now, now, promotion_id),
            )
            updated = conn.execute(
                "SELECT * FROM trajectory_model_promotions WHERE promotion_id=?",
                (promotion_id,),
            ).fetchone()
            result = self._model_promotion_row(updated)
        self._emit_analytics_change(
            project_id=project_id, change="model-promotion-auto-rollback-completed"
        )
        return result

    def sync_regression_alerts(
        self,
        *,
        project_id: str,
        baseline_id: str,
        session_id: str,
        regressions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Upsert current regression signals and resolve stale session alerts."""
        now = _now_iso()
        active_keys: set[tuple[str, str]] = set()
        notify_opened: list[dict[str, Any]] = []
        notify_resolved: list[dict[str, Any]] = []
        with self._connect() as conn:
            for regression in regressions:
                if str(regression.get("conversation_id") or "") != session_id:
                    continue
                event_id = str(regression.get("event_id") or "")
                severity = str(regression.get("severity") or "warning")
                for reason in regression.get("reasons") or []:
                    metric = str(reason.get("metric") or "")
                    if not event_id or not metric:
                        continue
                    active_keys.add((event_id, metric))
                    alert_id = "alert-" + hashlib.sha256(
                        f"{project_id}|{baseline_id}|{session_id}|{event_id}|{metric}".encode("utf-8")
                    ).hexdigest()[:24]
                    observed = float(reason.get("observed") or 0)
                    baseline = float(reason.get("baseline") or 0)
                    delta = float(reason.get("delta") or 0)
                    prior = conn.execute(
                        """SELECT status, severity FROM trajectory_regression_alerts
                           WHERE project_id=? AND baseline_id=? AND session_id=?
                             AND event_id=? AND metric=?""",
                        (project_id, baseline_id, session_id, event_id, metric),
                    ).fetchone()
                    conn.execute(
                        """INSERT INTO trajectory_regression_alerts(
                               alert_id, project_id, baseline_id, session_id,
                               event_id, metric, severity, observed, baseline,
                               delta, status, detected_at, updated_at
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)
                           ON CONFLICT(project_id, baseline_id, session_id, event_id, metric)
                           DO UPDATE SET severity=excluded.severity,
                               observed=excluded.observed, baseline=excluded.baseline,
                               delta=excluded.delta,
                               status=CASE
                                   WHEN trajectory_regression_alerts.status='resolved' THEN 'open'
                                   ELSE trajectory_regression_alerts.status
                               END,
                               updated_at=excluded.updated_at""",
                        (
                            alert_id, project_id, baseline_id, session_id,
                            event_id, metric, severity, observed, baseline,
                            delta, now, now,
                        ),
                    )
                    if prior is None or str(prior["status"]) == "resolved":
                        action = "triggered" if prior is None else "reopened"
                        self._append_alert_history(
                            conn,
                            alert_id=alert_id,
                            project_id=project_id,
                            action=action,
                            status="open",
                            details={
                                "metric": metric, "severity": severity,
                                "observed": observed, "threshold": baseline,
                                "delta": delta, "conversation_id": session_id,
                                "event_id": event_id,
                                "source_type": "policy" if baseline_id.startswith("policy-") else "baseline",
                            },
                            created_at=now,
                        )
                        if severity == "error":
                            notify_opened.append({
                                "alert_id": alert_id, "metric": metric,
                                "observed": observed, "baseline": baseline,
                                "conversation_id": session_id, "event_id": event_id,
                                "project_id": project_id,
                            })
            existing = conn.execute(
                """SELECT alert_id, event_id, metric, severity, observed, baseline
                   FROM trajectory_regression_alerts
                   WHERE project_id=? AND baseline_id=? AND session_id=?
                     AND status!='resolved'""",
                (project_id, baseline_id, session_id),
            ).fetchall()
            for row in existing:
                if (str(row["event_id"]), str(row["metric"])) not in active_keys:
                    conn.execute(
                        """UPDATE trajectory_regression_alerts
                           SET status='resolved', updated_at=? WHERE alert_id=?""",
                        (now, row["alert_id"]),
                    )
                    self._append_alert_history(
                        conn,
                        alert_id=str(row["alert_id"]),
                        project_id=project_id,
                        action="auto-resolved",
                        status="resolved",
                        details={
                            "metric": row["metric"], "severity": row["severity"],
                            "observed": row["observed"], "threshold": row["baseline"],
                            "conversation_id": session_id, "event_id": row["event_id"],
                        },
                        created_at=now,
                    )
                    if str(row["severity"]) == "error":
                        notify_resolved.append({"alert_id": str(row["alert_id"]), "project_id": project_id})
        for alert in notify_opened:
            self._notify_trajectory_alert(alert, resolved=False)
        for alert in notify_resolved:
            self._notify_trajectory_alert(alert, resolved=True)
        self._emit_analytics_change(project_id=project_id, change="alerts-synced")
        return self.list_regression_alerts(project_id=project_id)

    def list_regression_alerts(
        self,
        *,
        project_id: str,
        status: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        where = "WHERE project_id=?"
        values: list[Any] = [project_id]
        if status:
            where += " AND status=?"
            values.append(status)
        values.append(max(1, min(int(limit), 500)))
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT * FROM trajectory_regression_alerts {where}
                    ORDER BY CASE status WHEN 'open' THEN 0 WHEN 'acknowledged' THEN 1 ELSE 2 END,
                             updated_at DESC LIMIT ?""",
                values,
            ).fetchall()
        return [{
            "alert_id": str(row["alert_id"]),
            "project_id": str(row["project_id"]),
            "baseline_id": str(row["baseline_id"]),
            "source_type": "policy" if str(row["baseline_id"]).startswith("policy-") else "baseline",
            "conversation_id": str(row["session_id"]),
            "event_id": str(row["event_id"]),
            "metric": str(row["metric"]),
            "severity": str(row["severity"]),
            "observed": float(row["observed"] or 0),
            "baseline": float(row["baseline"] or 0),
            "delta": float(row["delta"] or 0),
            "status": str(row["status"]),
            "detected_at": str(row["detected_at"]),
            "updated_at": str(row["updated_at"]),
        } for row in rows]

    def update_regression_alert(
        self,
        *,
        project_id: str,
        alert_id: str,
        status: str,
    ) -> dict[str, Any]:
        normalized = str(status or "").strip().lower()
        if normalized not in {"open", "acknowledged", "resolved"}:
            raise ValueError("Unsupported regression alert status")
        prior = None
        with self._connect() as conn:
            prior = conn.execute(
                """SELECT * FROM trajectory_regression_alerts
                   WHERE project_id=? AND alert_id=?""",
                (project_id, alert_id),
            ).fetchone()
            if not prior:
                raise KeyError(alert_id)
            cursor = conn.execute(
                """UPDATE trajectory_regression_alerts SET status=?, updated_at=?
                   WHERE project_id=? AND alert_id=?""",
                (normalized, _now_iso(), project_id, alert_id),
            )
            if not cursor.rowcount:
                raise KeyError(alert_id)
            if str(prior["status"]) != normalized:
                action = {
                    "acknowledged": "acknowledged",
                    "resolved": "resolved",
                    "open": "reopened",
                }[normalized]
                self._append_alert_history(
                    conn,
                    alert_id=alert_id,
                    project_id=project_id,
                    action=action,
                    status=normalized,
                    actor="operator",
                    details={
                        "metric": prior["metric"], "severity": prior["severity"],
                        "observed": prior["observed"], "threshold": prior["baseline"],
                        "conversation_id": prior["session_id"], "event_id": prior["event_id"],
                    },
                )
        if str(prior["severity"]) == "error":
            if normalized == "resolved":
                self._notify_trajectory_alert(
                    {"alert_id": alert_id, "project_id": project_id}, resolved=True
                )
            elif normalized == "open" and str(prior["status"]) == "resolved":
                self._notify_trajectory_alert({
                    "alert_id": alert_id, "project_id": project_id,
                    "metric": prior["metric"], "observed": prior["observed"],
                    "baseline": prior["baseline"],
                    "conversation_id": prior["session_id"], "event_id": prior["event_id"],
                }, resolved=False)
        self._emit_analytics_change(project_id=project_id, change="alert-updated")
        return next(
            row for row in self.list_regression_alerts(project_id=project_id, limit=500)
            if row["alert_id"] == alert_id
        )

    @staticmethod
    def _notify_trajectory_alert(alert: dict[str, Any], *, resolved: bool) -> None:
        """Bridge critical trajectory incidents into the shared operator center."""
        try:
            from remy.core.notification_router import notify

            alert_id = str(alert.get("alert_id") or "")
            dedupe_key = f"trajectory:{alert_id}"
            if resolved:
                notify(
                    "Trajectory regression resolved",
                    level="info",
                    event_type="operator_alert",
                    event_data={
                        "dedupe_key": f"trajectory-recovery:{alert_id}",
                        "resolves": [dedupe_key],
                        "resolved": True,
                        "source": "trajectory",
                        "action_target": "open_trajectory_analytics",
                        "artifact_ids": [alert_id],
                    },
                    parse_mode="",
                )
                return
            metric = str(alert.get("metric") or "regression")
            notify(
                f"Trajectory critical regression: {metric}",
                level="critical",
                event_type="operator_alert",
                event_data={
                    "dedupe_key": dedupe_key,
                    "source": "trajectory",
                    "action_target": "open_trajectory_alert",
                    "artifact_ids": [
                        str(alert.get("conversation_id") or ""),
                        str(alert.get("event_id") or ""),
                        alert_id,
                    ],
                    "failure_code": "trajectory_regression",
                    "trajectory_alert_id": alert_id,
                    "project_id": str(alert.get("project_id") or ""),
                    "conversation_id": str(alert.get("conversation_id") or ""),
                    "event_id": str(alert.get("event_id") or ""),
                    "metric": metric,
                    "observed": float(alert.get("observed") or 0),
                    "threshold": float(alert.get("baseline") or 0),
                },
                parse_mode="",
            )
        except Exception:
            return

    @staticmethod
    def _notify_slo_incident(incident: dict[str, Any], *, resolved: bool) -> None:
        """Project SLO incidents share the durable operator notification center."""
        try:
            from remy.core.notification_router import notify

            incident_id = str(incident.get("incident_id") or "")
            dedupe_key = f"trajectory-slo:{incident_id}"
            if resolved:
                notify(
                    "Trajectory SLO incident resolved",
                    level="info",
                    event_type="operator_alert",
                    event_data={
                        "dedupe_key": f"trajectory-slo-recovery:{incident_id}",
                        "resolves": [dedupe_key],
                        "resolved": True,
                        "source": "trajectory",
                        "action_target": "open_trajectory_analytics",
                        "artifact_ids": [incident_id],
                    },
                    parse_mode="",
                )
                return
            reason = str(incident.get("reason") or "budget-burn")
            severity = str(incident.get("severity") or "warning")
            burn_rate = float(incident.get("burn_rate") or 0)
            notify(
                f"Trajectory SLO {reason}: {burn_rate:.1f}x burn rate",
                level="critical" if severity == "critical" else "warning",
                event_type="operator_alert",
                event_data={
                    "dedupe_key": dedupe_key,
                    "source": "trajectory",
                    "action_target": "open_trajectory_alert",
                    "artifact_ids": [
                        str(incident.get("conversation_id") or ""),
                        str(incident.get("event_id") or ""),
                        incident_id,
                    ],
                    "failure_code": "trajectory_slo_burn",
                    "trajectory_alert_id": incident_id,
                    "project_id": str(incident.get("project_id") or ""),
                    "conversation_id": str(incident.get("conversation_id") or ""),
                    "event_id": str(incident.get("event_id") or ""),
                    "metric": "slo.burn_rate",
                    "observed": burn_rate,
                    "threshold": 1.0,
                },
                parse_mode="",
            )
        except Exception:
            return

    def _evaluate_session_regressions(self, *, project_id: str, session_id: str) -> None:
        """Compare a completed session with the active baseline, best-effort."""
        try:
            baseline = self.get_active_analytics_baseline(project_id=project_id)
            policies = self.list_alert_policies(project_id=project_id)
            slo_config = self.get_slo_config(project_id=project_id)
            from remy.core.trajectory_diagnostics import (
                analyze_project_trajectory,
                analyze_trajectory_slo,
                evaluate_trajectory_policies,
            )

            records = self.list_events(
                project_id=project_id,
                session_id=session_id,
                limit=5000,
            )
            analytics = analyze_project_trajectory(
                records,
                days=365,
                baseline_metrics=baseline["metrics"] if baseline else None,
            )
            if baseline:
                self.sync_regression_alerts(
                    project_id=project_id,
                    baseline_id=baseline["baseline_id"],
                    session_id=session_id,
                    regressions=analytics.get("regressions") or [],
                )
            signals = {
                row["policy_id"]: [row["regression"]]
                for row in evaluate_trajectory_policies(analytics, policies)
            }
            for policy in policies:
                self.sync_regression_alerts(
                    project_id=project_id,
                    baseline_id=policy["policy_id"],
                    session_id=session_id,
                    regressions=signals.get(policy["policy_id"], []),
                )
            project_records = self.list_project_events(project_id=project_id, limit=20_000)
            slo = analyze_trajectory_slo(
                project_records,
                target_success_rate=slo_config["target_success_rate"],
                window_days=slo_config["window_days"],
                min_operations=slo_config["min_operations"],
            )
            latest = next(
                (
                    row for row in reversed(project_records)
                    if str(row.get("session_id") or "") == session_id
                ),
                {},
            )
            self.sync_slo_incidents(
                project_id=project_id,
                conversation_id=session_id,
                event_id=str(latest.get("event_id") or ""),
                slo=slo,
            )
        except Exception:
            # Alerting is observability and must never fail the agent turn.
            return

    def count_events(self, *, project_id: str, session_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT COUNT(*) AS total FROM trajectory_events
                   WHERE project_id=? AND session_id=?""",
                (project_id, session_id),
            ).fetchone()
        return int(row["total"] if row else 0)

    def set_annotation(
        self,
        *,
        project_id: str,
        session_id: str,
        event_id: str,
        label: str = "",
        note: str = "",
        bookmarked: bool = False,
    ) -> dict[str, Any]:
        normalized_label = str(label or "").strip().lower()
        if normalized_label not in {"", "investigate", "bug", "expected", "resolved"}:
            raise ValueError("Unsupported trajectory annotation label")
        normalized_note = str(note or "").strip()
        if len(normalized_note) > 4000:
            raise ValueError("Trajectory annotation note exceeds 4000 characters")
        with self._connect() as conn:
            event = conn.execute(
                """SELECT event_id FROM trajectory_events
                   WHERE event_id=? AND project_id=? AND session_id=?""",
                (event_id, project_id, session_id),
            ).fetchone()
            if not event:
                raise KeyError(event_id)
            updated_at = _now_iso(time.time())
            conn.execute(
                """INSERT INTO trajectory_annotations(
                       event_id, session_id, project_id, label, note, bookmarked, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(event_id) DO UPDATE SET
                       label=excluded.label,
                       note=excluded.note,
                       bookmarked=excluded.bookmarked,
                       updated_at=excluded.updated_at""",
                (
                    event_id, session_id, project_id, normalized_label,
                    normalized_note, int(bool(bookmarked)), updated_at,
                ),
            )
        self._emit_change(
            project_id=project_id,
            session_id=session_id,
            event_id=event_id,
            change="annotation",
            fields=("label", "note", "bookmarked"),
        )
        return {
            "label": normalized_label,
            "note": normalized_note,
            "bookmarked": bool(bookmarked),
            "updated_at": updated_at,
        }

    def delete_annotation(
        self, *, project_id: str, session_id: str, event_id: str
    ) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                """DELETE FROM trajectory_annotations
                   WHERE event_id=? AND project_id=? AND session_id=?""",
                (event_id, project_id, session_id),
            )
        deleted = cursor.rowcount > 0
        if deleted:
            self._emit_change(
                project_id=project_id,
                session_id=session_id,
                event_id=event_id,
                change="annotation_deleted",
                fields=("annotation",),
            )
        return deleted


_STORE: TrajectoryStore | None = None
_STORE_LOCK = threading.Lock()


def get_trajectory_store() -> TrajectoryStore:
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                _STORE = TrajectoryStore()
    return _STORE
