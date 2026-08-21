"""Approval-gated, additive prompt experiments with canary and rollback.

This module deliberately does not edit source files, base prompts, policies,
tool permissions, approval rules, or sandbox configuration.  The only mutable
runtime surface is a bounded additive guidance overlay selected per project and
stable session bucket after an existing Trajectory eval matrix passes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any

from remy.config.settings import settings


TARGET = "agent.guidance"
EMPTY_OVERLAY_HASH = hashlib.sha256(b"").hexdigest()
_MAX_CANDIDATE_CHARS = 4_000
_MIN_EVAL_CASES = 3
_MIN_CANARY_REQUESTS = 5
_TARGET_CANARY_REQUESTS = 20
_MIN_CANARY_OBSERVATION_SECONDS = 300
_MIN_VERIFICATION_COVERAGE = 0.8
_CONFIDENCE_Z = 1.96
_INCONCLUSIVE_ALERT_SECONDS = 1_800
_LOCK = threading.RLock()
_EVAL_OVERLAY: ContextVar[dict[str, Any] | None] = ContextVar(
    "remy_self_modification_eval_overlay",
    default=None,
)
_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|system|developer)", re.I),
    re.compile(r"(?:bypass|disable|override).{0,40}(?:approval|sandbox|security|policy)", re.I),
    re.compile(r"(?:reveal|print|expose).{0,30}(?:secret|api.?key|system prompt)", re.I),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _default_canary_policy() -> dict[str, Any]:
    return {
        "minimum_requests_per_cohort": _MIN_CANARY_REQUESTS,
        "target_requests_per_cohort": _TARGET_CANARY_REQUESTS,
        "minimum_observation_seconds": _MIN_CANARY_OBSERVATION_SECONDS,
        "minimum_verification_coverage": _MIN_VERIFICATION_COVERAGE,
        "confidence_level": 0.95,
        "failure_rate_margin": 0.02,
        "unsupported_rate_margin": 0.02,
        "latency_multiplier": 1.5,
        "inconclusive_alert_seconds": _INCONCLUSIVE_ALERT_SECONDS,
        "alerts_enabled": True,
    }


def _validate_canary_policy(payload: dict[str, Any]) -> dict[str, Any]:
    defaults = _default_canary_policy()
    values = {**defaults, **dict(payload or {})}
    clean = {
        "minimum_requests_per_cohort": int(values["minimum_requests_per_cohort"]),
        "target_requests_per_cohort": int(values["target_requests_per_cohort"]),
        "minimum_observation_seconds": int(values["minimum_observation_seconds"]),
        "minimum_verification_coverage": float(values["minimum_verification_coverage"]),
        "confidence_level": float(values["confidence_level"]),
        "failure_rate_margin": float(values["failure_rate_margin"]),
        "unsupported_rate_margin": float(values["unsupported_rate_margin"]),
        "latency_multiplier": float(values["latency_multiplier"]),
        "inconclusive_alert_seconds": int(values["inconclusive_alert_seconds"]),
        "alerts_enabled": bool(values["alerts_enabled"]),
    }
    if not 5 <= clean["minimum_requests_per_cohort"] <= 100:
        raise ValueError("minimum_requests_per_cohort must be between 5 and 100")
    if not clean["minimum_requests_per_cohort"] <= clean["target_requests_per_cohort"] <= 10_000:
        raise ValueError("target_requests_per_cohort must be at least the safety floor and at most 10000")
    if not 0 <= clean["minimum_observation_seconds"] <= 604_800:
        raise ValueError("minimum_observation_seconds must be between 0 and 604800")
    if not 0.5 <= clean["minimum_verification_coverage"] <= 1:
        raise ValueError("minimum_verification_coverage must be between 0.5 and 1")
    if not 0.80 <= clean["confidence_level"] <= 0.999:
        raise ValueError("confidence_level must be between 0.80 and 0.999")
    if not 0 <= clean["failure_rate_margin"] <= 0.25:
        raise ValueError("failure_rate_margin must be between 0 and 0.25")
    if not 0 <= clean["unsupported_rate_margin"] <= 0.25:
        raise ValueError("unsupported_rate_margin must be between 0 and 0.25")
    if not 1 <= clean["latency_multiplier"] <= 5:
        raise ValueError("latency_multiplier must be between 1 and 5")
    if not clean["minimum_observation_seconds"] <= clean["inconclusive_alert_seconds"] <= 2_592_000:
        raise ValueError(
            "inconclusive_alert_seconds must be at least the observation window and at most 2592000"
        )
    return clean


def _validate_candidate(target: str, text: str) -> str:
    if str(target or "").strip() != TARGET:
        raise ValueError(
            "Only the additive agent.guidance target is supported; code, tools, "
            "policy, approval, sandbox, and base prompts are immutable."
        )
    candidate = str(text or "").strip()
    if len(candidate) < 20:
        raise ValueError("Candidate guidance must contain at least 20 characters")
    if len(candidate) > _MAX_CANDIDATE_CHARS:
        raise ValueError(f"Candidate guidance exceeds {_MAX_CANDIDATE_CHARS} characters")
    if any(pattern.search(candidate) for pattern in _INJECTION_PATTERNS):
        raise ValueError("Candidate guidance contains a forbidden policy-bypass instruction")
    return candidate


class SelfModificationLab:
    """Durable proposal state machine for one local Remy installation."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or settings.DATA_DIR / "self_modification_lab.sqlite3")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _ensure_schema(self) -> None:
        with _LOCK, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS self_modification_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    target TEXT NOT NULL,
                    candidate_text TEXT NOT NULL,
                    candidate_hash TEXT NOT NULL,
                    baseline_hash TEXT NOT NULL,
                    previous_proposal_id TEXT NOT NULL DEFAULT '',
                    rationale TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT 'operator',
                    source_session_id TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    eval_matrix_id TEXT NOT NULL DEFAULT '',
                    eval_json TEXT NOT NULL DEFAULT '{}',
                    canary_percent INTEGER NOT NULL DEFAULT 0,
                    canary_json TEXT NOT NULL DEFAULT '{}',
                    approved_by TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    approved_at TEXT NOT NULL DEFAULT '',
                    activated_at TEXT NOT NULL DEFAULT '',
                    rolled_back_at TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_self_mod_project_status
                    ON self_modification_proposals(project_id, status, updated_at DESC);
                CREATE TABLE IF NOT EXISTS self_modification_canary_policies (
                    project_id TEXT PRIMARY KEY,
                    policy_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS self_modification_canary_alerts (
                    project_id TEXT NOT NULL,
                    proposal_id TEXT NOT NULL,
                    alert_code TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    opened_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    resolved_at TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(project_id, proposal_id, alert_code)
                );
                CREATE INDEX IF NOT EXISTS idx_self_mod_canary_alerts_project
                    ON self_modification_canary_alerts(project_id, status, updated_at DESC);
                """
            )

    def get_canary_policy(self, *, project_id: str) -> dict[str, Any]:
        project_id = str(project_id or "").strip()
        if not project_id:
            raise ValueError("project_id is required")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT policy_json, updated_at FROM self_modification_canary_policies WHERE project_id=?",
                (project_id,),
            ).fetchone()
        if not row:
            return {**_default_canary_policy(), "source": "default", "updated_at": ""}
        try:
            policy = _validate_canary_policy(json.loads(row["policy_json"] or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Stored canary policy is invalid for {project_id}") from exc
        return {**policy, "source": "project", "updated_at": str(row["updated_at"] or "")}

    def update_canary_policy(
        self, *, project_id: str, policy: dict[str, Any]
    ) -> dict[str, Any]:
        project_id = str(project_id or "").strip()
        if not project_id:
            raise ValueError("project_id is required")
        clean = _validate_canary_policy(policy)
        now = _now()
        with _LOCK, self._connect() as conn:
            conn.execute(
                """INSERT INTO self_modification_canary_policies(project_id, policy_json, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(project_id) DO UPDATE SET
                       policy_json=excluded.policy_json, updated_at=excluded.updated_at""",
                (project_id, json.dumps(clean, ensure_ascii=False), now),
            )
        return {**clean, "source": "project", "updated_at": now}

    def list_canary_alerts(
        self, *, project_id: str, proposal_id: str = "", status: str = ""
    ) -> list[dict[str, Any]]:
        clauses = ["project_id=?"]
        values: list[Any] = [str(project_id)]
        if proposal_id:
            clauses.append("proposal_id=?")
            values.append(str(proposal_id))
        if status:
            clauses.append("status=?")
            values.append(str(status))
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT * FROM self_modification_canary_alerts
                    WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC LIMIT 100""",
                values,
            ).fetchall()
        return [{
            "project_id": str(row["project_id"]),
            "proposal_id": str(row["proposal_id"]),
            "alert_code": str(row["alert_code"]),
            "severity": str(row["severity"]),
            "status": str(row["status"]),
            "details": json.loads(row["details_json"] or "{}"),
            "opened_at": str(row["opened_at"]),
            "updated_at": str(row["updated_at"]),
            "resolved_at": str(row["resolved_at"] or ""),
        } for row in rows]

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        item = dict(row)
        item["evaluation"] = json.loads(item.pop("eval_json") or "{}")
        item["canary_evaluation"] = json.loads(item.pop("canary_json") or "{}")
        item["trajectory_session_id"] = f"self-mod:{item['proposal_id']}"
        return item

    def get(self, *, project_id: str, proposal_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM self_modification_proposals
                   WHERE project_id=? AND proposal_id=?""",
                (str(project_id), str(proposal_id)),
            ).fetchone()
        item = self._decode(row)
        if item is None:
            raise KeyError(proposal_id)
        return item

    def list(self, *, project_id: str, status: str = "", limit: int = 100) -> list[dict[str, Any]]:
        sql = "SELECT * FROM self_modification_proposals WHERE project_id=?"
        args: list[Any] = [str(project_id)]
        if status and status != "all":
            sql += " AND status=?"
            args.append(str(status))
        sql += " ORDER BY updated_at DESC LIMIT ?"
        args.append(max(1, min(int(limit), 500)))
        with self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [self._decode(row) for row in rows]

    def _current_active(self, conn: sqlite3.Connection, project_id: str) -> sqlite3.Row | None:
        return conn.execute(
            """SELECT * FROM self_modification_proposals
               WHERE project_id=? AND status='active'
               ORDER BY activated_at DESC, updated_at DESC LIMIT 1""",
            (project_id,),
        ).fetchone()

    def _emit(self, item: dict[str, Any], event_type: str, **payload: Any) -> None:
        try:
            from remy.core.trajectory_store import get_trajectory_store

            get_trajectory_store().record_self_modification_event(
                project_id=item["project_id"],
                proposal_id=item["proposal_id"],
                event_type=event_type,
                status="failed" if item["status"] in {"eval_failed", "rolled_back"} else "completed",
                payload={
                    "candidate_hash": item["candidate_hash"],
                    "baseline_hash": item["baseline_hash"],
                    "target": item["target"],
                    "decision": item["status"],
                    **payload,
                },
            )
        except Exception:
            pass

    def create_proposal(
        self,
        *,
        project_id: str,
        candidate_text: str,
        rationale: str,
        target: str = TARGET,
        source: str = "operator",
        source_session_id: str = "",
    ) -> dict[str, Any]:
        project_id = str(project_id or "").strip()
        if not project_id:
            raise ValueError("project_id is required")
        candidate = _validate_candidate(target, candidate_text)
        source = str(source or "operator").strip().lower()
        if source not in {"operator", "agent", "experiment"}:
            raise ValueError("Unsupported proposal source")
        proposal_id = f"self-mod-{uuid.uuid4().hex}"
        now = _now()
        with _LOCK, self._connect() as conn:
            active = self._current_active(conn, project_id)
            baseline_hash = str(active["candidate_hash"]) if active else EMPTY_OVERLAY_HASH
            previous_id = str(active["proposal_id"]) if active else ""
            conn.execute(
                """INSERT INTO self_modification_proposals(
                       proposal_id, project_id, target, candidate_text, candidate_hash,
                       baseline_hash, previous_proposal_id, rationale, source,
                       source_session_id, status, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?, ?)""",
                (
                    proposal_id, project_id, TARGET, candidate, _hash(candidate),
                    baseline_hash, previous_id, str(rationale or "")[:4_000], source,
                    str(source_session_id or "")[:160], now, now,
                ),
            )
        item = self.get(project_id=project_id, proposal_id=proposal_id)
        self._emit(item, "SELF_MOD_PROPOSAL", summary="Immutable proposal created")
        return item

    @staticmethod
    def _matrix_for_project(project_id: str, matrix_id: str) -> dict[str, Any]:
        from remy.core.trajectory_store import get_trajectory_store

        matrices = get_trajectory_store().list_eval_matrices(project_id=project_id, limit=100)
        matrix = next((item for item in matrices if item["matrix_id"] == matrix_id), None)
        if matrix is None:
            raise KeyError(matrix_id)
        return matrix

    def record_evaluation(
        self, *, project_id: str, proposal_id: str, matrix_id: str
    ) -> dict[str, Any]:
        matrix = self._matrix_for_project(project_id, matrix_id)
        proposal = self.get(project_id=project_id, proposal_id=proposal_id)
        expected_agent_version = f"self-mod:{proposal['candidate_hash']}"
        candidate_bound = str(matrix.get("agent_version") or "") == expected_agent_version
        gate_passed = bool(
            matrix.get("gate_passed")
            and candidate_bound
            and int(matrix.get("case_count") or 0) >= _MIN_EVAL_CASES
            and int(matrix.get("passed_count") or 0) == int(matrix.get("case_count") or 0)
            and not int(matrix.get("failed_count") or 0)
            and not int(matrix.get("error_count") or 0)
        )
        evaluation = {
            "matrix_id": matrix_id,
            "matrix_status": str(matrix.get("status") or ""),
            "case_count": int(matrix.get("case_count") or 0),
            "passed_count": int(matrix.get("passed_count") or 0),
            "failed_count": int(matrix.get("failed_count") or 0),
            "error_count": int(matrix.get("error_count") or 0),
            "avg_score": float(matrix.get("avg_score") or 0),
            "gate_passed": gate_passed,
            "candidate_bound": candidate_bound,
            "expected_agent_version": expected_agent_version,
            "required_min_cases": _MIN_EVAL_CASES,
        }
        with _LOCK, self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM self_modification_proposals WHERE project_id=? AND proposal_id=?",
                (project_id, proposal_id),
            ).fetchone()
            if not row:
                raise KeyError(proposal_id)
            if str(row["status"]) not in {"draft", "eval_failed"}:
                raise ValueError("Only draft or failed-eval proposals can be evaluated")
            conn.execute(
                """UPDATE self_modification_proposals SET status=?, eval_matrix_id=?,
                       eval_json=?, updated_at=? WHERE project_id=? AND proposal_id=?""",
                (
                    "eval_passed" if gate_passed else "eval_failed",
                    matrix_id, json.dumps(evaluation, ensure_ascii=False), _now(),
                    project_id, proposal_id,
                ),
            )
        item = self.get(project_id=project_id, proposal_id=proposal_id)
        self._emit(item, "SELF_MOD_EVAL", evaluation=evaluation)
        return item

    def _assert_fresh_baseline(self, conn: sqlite3.Connection, row: sqlite3.Row) -> None:
        active = self._current_active(conn, str(row["project_id"]))
        active_hash = str(active["candidate_hash"]) if active else EMPTY_OVERLAY_HASH
        if active_hash != str(row["baseline_hash"]):
            raise ValueError("Proposal baseline is stale; create a new proposal against the active overlay")

    def approve(
        self, *, project_id: str, proposal_id: str, confirm_candidate_hash: str, approved_by: str
    ) -> dict[str, Any]:
        with _LOCK, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM self_modification_proposals WHERE project_id=? AND proposal_id=?",
                (project_id, proposal_id),
            ).fetchone()
            if not row:
                raise KeyError(proposal_id)
            if str(row["status"]) != "eval_passed":
                raise ValueError("A passing Trajectory eval matrix is required before approval")
            if str(confirm_candidate_hash) != str(row["candidate_hash"]):
                raise ValueError("Candidate hash confirmation does not match")
            if not str(approved_by or "").strip():
                raise ValueError("approved_by is required")
            self._assert_fresh_baseline(conn, row)
            now = _now()
            conn.execute(
                """UPDATE self_modification_proposals SET status='approved', approved_by=?,
                       approved_at=?, updated_at=? WHERE proposal_id=?""",
                (str(approved_by)[:160], now, now, proposal_id),
            )
        item = self.get(project_id=project_id, proposal_id=proposal_id)
        self._emit(item, "SELF_MOD_APPROVAL", summary="Operator approved exact candidate hash")
        return item

    def start_canary(
        self, *, project_id: str, proposal_id: str, confirm_candidate_hash: str,
        canary_percent: int = 10,
    ) -> dict[str, Any]:
        percent = max(5, min(int(canary_percent), 25))
        with _LOCK, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM self_modification_proposals WHERE project_id=? AND proposal_id=?",
                (project_id, proposal_id),
            ).fetchone()
            if not row:
                raise KeyError(proposal_id)
            if str(row["status"]) != "approved":
                raise ValueError("Only an approved proposal can enter canary")
            if str(confirm_candidate_hash) != str(row["candidate_hash"]):
                raise ValueError("Candidate hash confirmation does not match")
            self._assert_fresh_baseline(conn, row)
            competing = conn.execute(
                """SELECT proposal_id FROM self_modification_proposals
                   WHERE project_id=? AND status IN ('canary','canary_passed') LIMIT 1""",
                (project_id,),
            ).fetchone()
            if competing:
                raise ValueError("Another prompt-overlay canary is already active")
            conn.execute(
                """UPDATE self_modification_proposals SET status='canary',
                       canary_percent=?, activated_at=?, updated_at=? WHERE proposal_id=?""",
                (percent, _now(), _now(), proposal_id),
            )
        item = self.get(project_id=project_id, proposal_id=proposal_id)
        self._emit(item, "SELF_MOD_CANARY", canary_percent=percent, summary="Canary started")
        return item

    @staticmethod
    def _canary_gate(
        metrics: dict[str, Any], policy: dict[str, Any] | None = None
    ) -> tuple[bool, dict[str, Any]]:
        policy = _validate_canary_policy(policy or {})
        clean = {
            "candidate_requests": max(0, int(metrics.get("candidate_requests") or 0)),
            "baseline_requests": max(0, int(metrics.get("baseline_requests") or 0)),
            "candidate_failure_rate": max(0.0, float(metrics.get("candidate_failure_rate") or 0)),
            "baseline_failure_rate": max(0.0, float(metrics.get("baseline_failure_rate") or 0)),
            "candidate_unsupported_rate": max(0.0, float(metrics.get("candidate_unsupported_rate") or 0)),
            "baseline_unsupported_rate": max(0.0, float(metrics.get("baseline_unsupported_rate") or 0)),
            "candidate_avg_request_ms": max(0.0, float(metrics.get("candidate_avg_request_ms") or 0)),
            "baseline_avg_request_ms": max(0.0, float(metrics.get("baseline_avg_request_ms") or 0)),
            "candidate_verified_requests": max(0, int(metrics.get("candidate_verified_requests") or 0)),
            "baseline_verified_requests": max(0, int(metrics.get("baseline_verified_requests") or 0)),
            "candidate_verification_coverage": max(
                0.0, min(1.0, float(metrics.get("candidate_verification_coverage") or 0))
            ),
            "baseline_verification_coverage": max(
                0.0, min(1.0, float(metrics.get("baseline_verification_coverage") or 0))
            ),
            "telemetry_source": str(metrics.get("telemetry_source") or "manual")[:80],
            "collected_at": str(metrics.get("collected_at") or "")[:80],
            "statistical_gate_passed": bool(metrics.get("statistical_gate_passed", True)),
            "confidence_score": max(
                0.0, min(1.0, float(metrics.get("confidence_score") or 0))
            ),
            "observation_seconds": max(
                0.0, float(metrics.get("observation_seconds") or 0)
            ),
        }
        checks = {
            "enough_candidate_requests": (
                clean["candidate_requests"] >= policy["minimum_requests_per_cohort"]
            ),
            "enough_baseline_requests": (
                clean["baseline_requests"] >= policy["minimum_requests_per_cohort"]
            ),
            "failure_rate_not_regressed": (
                clean["candidate_failure_rate"]
                <= clean["baseline_failure_rate"] + policy["failure_rate_margin"]
            ),
            "unsupported_rate_not_regressed": (
                clean["candidate_unsupported_rate"]
                <= clean["baseline_unsupported_rate"] + policy["unsupported_rate_margin"]
            ),
            "latency_not_regressed": (
                not clean["baseline_avg_request_ms"]
                or clean["candidate_avg_request_ms"]
                <= clean["baseline_avg_request_ms"] * policy["latency_multiplier"]
            ),
        }
        if clean["telemetry_source"] == "trajectory-production":
            checks["statistical_non_inferiority"] = clean["statistical_gate_passed"]
        return all(checks.values()), {**clean, "checks": checks}

    @staticmethod
    def _wilson_interval(
        successes: int, total: int, *, z_score: float | None = None
    ) -> dict[str, float]:
        z_score = _CONFIDENCE_Z if z_score is None else float(z_score)
        if total <= 0:
            return {"rate": 0.0, "lower": 0.0, "upper": 1.0}
        rate = max(0.0, min(1.0, successes / total))
        z2 = z_score * z_score
        denominator = 1 + z2 / total
        center = (rate + z2 / (2 * total)) / denominator
        spread = (
            z_score
            * math.sqrt((rate * (1 - rate) + z2 / (4 * total)) / total)
            / denominator
        )
        return {
            "rate": rate,
            "lower": max(0.0, center - spread),
            "upper": min(1.0, center + spread),
        }

    @staticmethod
    def _mean_interval(
        values: list[float], *, z_score: float | None = None
    ) -> dict[str, float]:
        z_score = _CONFIDENCE_Z if z_score is None else float(z_score)
        clean = [max(0.0, float(value)) for value in values]
        if not clean:
            return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
        mean = sum(clean) / len(clean)
        if len(clean) < 2:
            return {"mean": mean, "lower": mean, "upper": mean}
        variance = sum((value - mean) ** 2 for value in clean) / (len(clean) - 1)
        margin = z_score * math.sqrt(variance / len(clean))
        return {
            "mean": mean,
            "lower": max(0.0, mean - margin),
            "upper": mean + margin,
        }

    @staticmethod
    def _verification_is_unsupported(record: dict[str, Any]) -> bool:
        payloads = [record.get("output"), record.get("input")]
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            for key in (
                "unsupported_observed_claims",
                "unsupported_claims_total",
                "unsupported",
            ):
                try:
                    if float(payload.get(key) or 0) > 0:
                        return True
                except (TypeError, ValueError):
                    continue
        return False

    def collect_canary_telemetry(
        self,
        *,
        project_id: str,
        proposal_id: str,
        trajectory_store=None,
    ) -> dict[str, Any]:
        """Aggregate candidate/control outcomes without exposing request content."""
        proposal = self.get(project_id=project_id, proposal_id=proposal_id)
        policy = self.get_canary_policy(project_id=project_id)
        confidence_level = float(policy["confidence_level"])
        z_score = (
            _CONFIDENCE_Z
            if confidence_level == 0.95
            else NormalDist().inv_cdf(0.5 + confidence_level / 2)
        )
        if trajectory_store is None:
            from remy.core.trajectory_store import get_trajectory_store

            trajectory_store = get_trajectory_store()
        records = trajectory_store.list_self_modification_canary_events(
            project_id=project_id,
            proposal_id=proposal_id,
            limit=10_000,
        )
        requests: list[dict[str, Any]] = []
        verifications: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            if record.get("kind") == "VERIFICATION" and record.get("parent_id"):
                verifications.setdefault(str(record["parent_id"]), []).append(record)
                continue
            if record.get("kind") != "REQUEST":
                continue
            details = record.get("details") if isinstance(record.get("details"), dict) else {}
            options = details.get("options") if isinstance(details.get("options"), dict) else {}
            self_mod = (
                options.get("self_modification")
                if isinstance(options.get("self_modification"), dict)
                else {}
            )
            if str(self_mod.get("proposal_id") or "") != proposal_id:
                continue
            cohort = str(self_mod.get("cohort") or "")
            if cohort not in {"candidate", "baseline"}:
                continue
            if str(record.get("status") or "") not in {"completed", "failed"}:
                continue
            requests.append({**record, "self_modification": self_mod, "cohort": cohort})

        # Tool-using turns can contain several model REQUEST events, while the
        # factuality audit belongs to the terminal answer. Count one terminal
        # request per turn so coverage and quality rates describe user-visible
        # outcomes instead of internal tool-call hops.
        terminal_by_turn: dict[str, dict[str, Any]] = {}
        for request in requests:
            turn_key = str(request.get("turn_id") or request.get("event_id") or "")
            previous = terminal_by_turn.get(turn_key)
            if previous is None or int(request.get("sequence") or 0) > int(
                previous.get("sequence") or 0
            ):
                terminal_by_turn[turn_key] = request
        requests = list(terminal_by_turn.values())

        def summarize(cohort: str) -> dict[str, Any]:
            rows = [record for record in requests if record["cohort"] == cohort]
            completed = [record for record in rows if record.get("status") == "completed"]
            failed = len(rows) - len(completed)
            verified = 0
            unsupported = 0
            for record in completed:
                audits = verifications.get(str(record.get("event_id") or ""), [])
                if not audits:
                    continue
                verified += 1
                if any(self._verification_is_unsupported(audit) for audit in audits):
                    unsupported += 1
            durations = [max(0.0, float(record.get("duration_ms") or 0)) for record in rows]
            duration_total = sum(durations)
            timestamps = [
                float(record.get("started_at"))
                for record in rows
                if record.get("started_at") is not None
            ]
            verification_coverage = verified / len(completed) if completed else 1.0
            return {
                "requests": len(rows),
                "completed_requests": len(completed),
                "failed_requests": failed,
                "verified_requests": verified,
                "unsupported_requests": unsupported,
                "failure_rate": failed / len(rows) if rows else 0.0,
                "unsupported_rate": unsupported / verified if verified else 0.0,
                "avg_request_ms": duration_total / len(rows) if rows else 0.0,
                "verification_coverage": verification_coverage,
                "_durations": durations,
                "_timestamps": timestamps,
            }

        candidate = summarize("candidate")
        baseline = summarize("baseline")
        timestamps = [*candidate["_timestamps"], *baseline["_timestamps"]]
        observation_seconds = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else 0.0
        safety_readiness_checks = {
            "enough_candidate_requests": (
                candidate["requests"] >= policy["minimum_requests_per_cohort"]
            ),
            "enough_baseline_requests": (
                baseline["requests"] >= policy["minimum_requests_per_cohort"]
            ),
            "candidate_verification_coverage": (
                candidate["verification_coverage"] >= policy["minimum_verification_coverage"]
            ),
            "baseline_verification_coverage": (
                baseline["verification_coverage"] >= policy["minimum_verification_coverage"]
            ),
        }
        safety_ready = all(safety_readiness_checks.values())
        promotion_readiness_checks = {
            "candidate_sample_target": (
                candidate["requests"] >= policy["target_requests_per_cohort"]
            ),
            "baseline_sample_target": (
                baseline["requests"] >= policy["target_requests_per_cohort"]
            ),
            "minimum_observation_window": (
                observation_seconds >= policy["minimum_observation_seconds"]
            ),
            "candidate_verification_coverage": safety_readiness_checks[
                "candidate_verification_coverage"
            ],
            "baseline_verification_coverage": safety_readiness_checks[
                "baseline_verification_coverage"
            ],
        }
        promotion_ready = all(promotion_readiness_checks.values())
        failure_candidate = self._wilson_interval(
            candidate["failed_requests"], candidate["requests"], z_score=z_score
        )
        failure_baseline = self._wilson_interval(
            baseline["failed_requests"], baseline["requests"], z_score=z_score
        )
        unsupported_candidate = self._wilson_interval(
            candidate["unsupported_requests"], candidate["verified_requests"], z_score=z_score
        )
        unsupported_baseline = self._wilson_interval(
            baseline["unsupported_requests"], baseline["verified_requests"], z_score=z_score
        )
        latency_candidate = self._mean_interval(candidate["_durations"], z_score=z_score)
        latency_baseline = self._mean_interval(baseline["_durations"], z_score=z_score)
        statistical_checks = {
            "failure_rate_non_inferior": (
                failure_candidate["upper"]
                <= failure_baseline["lower"] + policy["failure_rate_margin"]
            ),
            "unsupported_rate_non_inferior": (
                unsupported_candidate["upper"]
                <= unsupported_baseline["lower"] + policy["unsupported_rate_margin"]
            ),
            "latency_non_inferior": (
                not latency_baseline["lower"]
                or latency_candidate["upper"]
                <= latency_baseline["lower"] * policy["latency_multiplier"]
            ),
        }
        statistical_gate_passed = all(statistical_checks.values())
        sample_score = min(
            1.0,
            min(candidate["requests"], baseline["requests"])
            / max(1, policy["target_requests_per_cohort"]),
        )
        time_score = (
            1.0
            if policy["minimum_observation_seconds"] <= 0
            else min(1.0, observation_seconds / policy["minimum_observation_seconds"])
        )
        audit_score = min(
            candidate["verification_coverage"], baseline["verification_coverage"]
        )
        failure_gap = max(0.0, failure_candidate["upper"] - failure_baseline["lower"])
        unsupported_gap = max(
            0.0, unsupported_candidate["upper"] - unsupported_baseline["lower"]
        )
        failure_margin = policy["failure_rate_margin"]
        unsupported_margin = policy["unsupported_rate_margin"]
        failure_precision = (
            1.0 if failure_gap <= failure_margin
            else min(1.0, failure_margin / max(1e-9, failure_gap))
        )
        unsupported_precision = (
            1.0 if unsupported_gap <= unsupported_margin
            else min(1.0, unsupported_margin / max(1e-9, unsupported_gap))
        )
        latency_limit = latency_baseline["lower"] * policy["latency_multiplier"]
        latency_precision = (
            1.0
            if not latency_candidate["upper"] or latency_candidate["upper"] <= latency_limit
            else min(1.0, latency_limit / latency_candidate["upper"])
        )
        confidence_score = min(
            sample_score,
            time_score,
            audit_score,
            failure_precision,
            unsupported_precision,
            latency_precision,
        )
        metrics = {
            "candidate_requests": candidate["requests"],
            "baseline_requests": baseline["requests"],
            "candidate_failure_rate": candidate["failure_rate"],
            "baseline_failure_rate": baseline["failure_rate"],
            "candidate_unsupported_rate": candidate["unsupported_rate"],
            "baseline_unsupported_rate": baseline["unsupported_rate"],
            "candidate_avg_request_ms": candidate["avg_request_ms"],
            "baseline_avg_request_ms": baseline["avg_request_ms"],
            "candidate_verified_requests": candidate["verified_requests"],
            "baseline_verified_requests": baseline["verified_requests"],
            "candidate_verification_coverage": candidate["verification_coverage"],
            "baseline_verification_coverage": baseline["verification_coverage"],
            "telemetry_source": "trajectory-production",
            "collected_at": _now(),
            "statistical_gate_passed": statistical_gate_passed,
            "confidence_score": confidence_score,
            "observation_seconds": observation_seconds,
        }
        hard_gate_passed = None
        gate_checks: dict[str, bool] = {}
        if safety_ready:
            # Preview the original hard safety thresholds without allowing the
            # statistical promotion gate to mask an early obvious regression.
            hard_metrics = {**metrics, "telemetry_source": "trajectory-safety-preview"}
            hard_gate_passed, gate_report = self._canary_gate(hard_metrics, policy)
            gate_checks = dict(gate_report.get("checks") or {})
        hard_regression = safety_ready and hard_gate_passed is False
        ready = bool(hard_regression or (promotion_ready and statistical_gate_passed))
        gate_passed = (
            False
            if hard_regression
            else True if promotion_ready and statistical_gate_passed else None
        )
        candidate.pop("_durations", None)
        candidate.pop("_timestamps", None)
        baseline.pop("_durations", None)
        baseline.pop("_timestamps", None)
        return {
            "proposal_id": proposal_id,
            "candidate_hash": proposal["candidate_hash"],
            "baseline_hash": proposal["baseline_hash"],
            "status": (
                "regressed"
                if hard_regression
                else "healthy"
                if ready and gate_passed
                else "inconclusive"
                if promotion_ready
                else "collecting"
            ),
            "ready": ready,
            "safety_ready": safety_ready,
            "promotion_ready": promotion_ready,
            "hard_regression": hard_regression,
            "readiness_checks": safety_readiness_checks,
            "promotion_readiness_checks": promotion_readiness_checks,
            "gate_passed": gate_passed,
            "gate_checks": gate_checks,
            "statistics": {
                "confidence_level": confidence_level,
                "confidence_score": confidence_score,
                "gate_passed": statistical_gate_passed,
                "checks": statistical_checks,
                "failure_rate": {
                    "candidate": failure_candidate,
                    "baseline": failure_baseline,
                    "allowed_margin": policy["failure_rate_margin"],
                },
                "unsupported_rate": {
                    "candidate": unsupported_candidate,
                    "baseline": unsupported_baseline,
                    "allowed_margin": policy["unsupported_rate_margin"],
                },
                "latency_ms": {
                    "candidate": latency_candidate,
                    "baseline": latency_baseline,
                    "allowed_multiplier": policy["latency_multiplier"],
                },
            },
            "observation": {
                "seconds": observation_seconds,
                "minimum_seconds": policy["minimum_observation_seconds"],
            },
            "requirements": policy,
            "candidate": candidate,
            "baseline": baseline,
            "metrics": metrics,
            "alerts": self.list_canary_alerts(
                project_id=project_id, proposal_id=proposal_id
            ),
            "privacy": "aggregate-only; prompts and responses excluded",
        }

    @staticmethod
    def _notify_canary_alert(alert: dict[str, Any], *, resolved: bool) -> None:
        try:
            from remy.core.notification_router import notify

            project_id = str(alert.get("project_id") or "")
            proposal_id = str(alert.get("proposal_id") or "")
            alert_code = str(alert.get("alert_code") or "canary_attention")
            dedupe_key = f"self-mod-canary:{project_id}:{proposal_id}:{alert_code}"
            if resolved:
                notify(
                    "Self-improvement canary incident resolved",
                    level="info",
                    event_type="operator_alert",
                    event_data={
                        "dedupe_key": f"self-mod-canary-recovery:{project_id}:{proposal_id}:{alert_code}",
                        "resolves": [dedupe_key],
                        "resolved": True,
                        "source": "self-improvement-lab",
                        "action_target": "open_self_modification_lab",
                        "artifact_ids": [proposal_id],
                        "project_id": project_id,
                        "proposal_id": proposal_id,
                    },
                    parse_mode="",
                )
                return
            regressed = alert_code == "canary_regressed"
            notify(
                (
                    "Self-improvement canary auto-rolled back after regression"
                    if regressed
                    else "Self-improvement canary remains statistically inconclusive"
                ),
                level="critical" if regressed else "warning",
                event_type="operator_alert",
                event_data={
                    "dedupe_key": dedupe_key,
                    "source": "self-improvement-lab",
                    "action_target": "open_self_modification_lab",
                    "artifact_ids": [proposal_id],
                    "failure_code": alert_code,
                    "project_id": project_id,
                    "proposal_id": proposal_id,
                },
                parse_mode="",
            )
        except Exception:
            return

    def _sync_canary_alerts(
        self,
        *,
        project_id: str,
        proposal_id: str,
        telemetry: dict[str, Any],
        policy: dict[str, Any],
    ) -> list[dict[str, Any]]:
        active: dict[str, dict[str, Any]] = {}
        if policy["alerts_enabled"]:
            if telemetry.get("status") == "regressed":
                active["canary_regressed"] = {
                    "severity": "critical",
                    "status": "regressed",
                    "candidate_requests": int((telemetry.get("candidate") or {}).get("requests") or 0),
                    "baseline_requests": int((telemetry.get("baseline") or {}).get("requests") or 0),
                }
            elif (
                telemetry.get("status") == "inconclusive"
                and float((telemetry.get("observation") or {}).get("seconds") or 0)
                >= policy["inconclusive_alert_seconds"]
            ):
                active["canary_inconclusive_stale"] = {
                    "severity": "warning",
                    "status": "inconclusive",
                    "confidence_score": float(
                        (telemetry.get("statistics") or {}).get("confidence_score") or 0
                    ),
                    "observation_seconds": float(
                        (telemetry.get("observation") or {}).get("seconds") or 0
                    ),
                }
        now = _now()
        notify_opened: list[dict[str, Any]] = []
        notify_resolved: list[dict[str, Any]] = []
        with _LOCK, self._connect() as conn:
            existing = conn.execute(
                """SELECT * FROM self_modification_canary_alerts
                   WHERE project_id=? AND proposal_id=?""",
                (project_id, proposal_id),
            ).fetchall()
            existing_by_code = {str(row["alert_code"]): row for row in existing}
            for alert_code, details in active.items():
                prior = existing_by_code.get(alert_code)
                conn.execute(
                    """INSERT INTO self_modification_canary_alerts(
                           project_id, proposal_id, alert_code, severity, status,
                           details_json, opened_at, updated_at, resolved_at
                       ) VALUES (?, ?, ?, ?, 'open', ?, ?, ?, '')
                       ON CONFLICT(project_id, proposal_id, alert_code) DO UPDATE SET
                           severity=excluded.severity, status='open',
                           details_json=excluded.details_json, updated_at=excluded.updated_at,
                           opened_at=CASE
                               WHEN self_modification_canary_alerts.status='resolved'
                               THEN excluded.opened_at
                               ELSE self_modification_canary_alerts.opened_at
                           END,
                           resolved_at=''""",
                    (
                        project_id, proposal_id, alert_code, details["severity"],
                        json.dumps(details, ensure_ascii=False), now, now,
                    ),
                )
                if prior is None or str(prior["status"]) == "resolved":
                    notify_opened.append({
                        "project_id": project_id,
                        "proposal_id": proposal_id,
                        "alert_code": alert_code,
                    })
            for alert_code, prior in existing_by_code.items():
                if alert_code in active or str(prior["status"]) == "resolved":
                    continue
                conn.execute(
                    """UPDATE self_modification_canary_alerts
                       SET status='resolved', resolved_at=?, updated_at=?
                       WHERE project_id=? AND proposal_id=? AND alert_code=?""",
                    (now, now, project_id, proposal_id, alert_code),
                )
                notify_resolved.append({
                    "project_id": project_id,
                    "proposal_id": proposal_id,
                    "alert_code": alert_code,
                })
        for alert in notify_opened:
            self._notify_canary_alert(alert, resolved=False)
        for alert in notify_resolved:
            self._notify_canary_alert(alert, resolved=True)
        return self.list_canary_alerts(project_id=project_id, proposal_id=proposal_id)

    def observe_canary_telemetry(
        self,
        *,
        project_id: str,
        proposal_id: str,
        trajectory_store=None,
    ) -> dict[str, Any]:
        """Collect production telemetry and enforce the gate once evidence is ready."""
        proposal = self.get(project_id=project_id, proposal_id=proposal_id)
        telemetry = self.collect_canary_telemetry(
            project_id=project_id,
            proposal_id=proposal_id,
            trajectory_store=trajectory_store,
        )
        evaluated = False
        if proposal["status"] == "canary" and telemetry["ready"]:
            try:
                proposal = self.evaluate_canary(
                    project_id=project_id,
                    proposal_id=proposal_id,
                    metrics=telemetry["metrics"],
                )
                evaluated = True
            except ValueError:
                proposal = self.get(project_id=project_id, proposal_id=proposal_id)
                if proposal["status"] == "canary":
                    raise
        policy = self.get_canary_policy(project_id=project_id)
        telemetry["alerts"] = self._sync_canary_alerts(
            project_id=project_id,
            proposal_id=proposal_id,
            telemetry=telemetry,
            policy=policy,
        )
        return {"proposal": proposal, "telemetry": telemetry, "evaluated": evaluated}

    def observe_active_canary_telemetry(
        self,
        *,
        project_id: str,
        trajectory_store=None,
    ) -> dict[str, Any]:
        active = self.list(project_id=project_id, status="canary", limit=1)
        if not active:
            return {"proposal": None, "telemetry": None, "evaluated": False}
        return self.observe_canary_telemetry(
            project_id=project_id,
            proposal_id=active[0]["proposal_id"],
            trajectory_store=trajectory_store,
        )

    def evaluate_canary(
        self, *, project_id: str, proposal_id: str, metrics: dict[str, Any]
    ) -> dict[str, Any]:
        passed, report = self._canary_gate(
            metrics, self.get_canary_policy(project_id=project_id)
        )
        with _LOCK, self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM self_modification_proposals WHERE project_id=? AND proposal_id=?",
                (project_id, proposal_id),
            ).fetchone()
            if not row:
                raise KeyError(proposal_id)
            if str(row["status"]) != "canary":
                raise ValueError("Only a running canary can be evaluated")
            now = _now()
            conn.execute(
                """UPDATE self_modification_proposals SET status=?, canary_json=?,
                       rolled_back_at=?, updated_at=? WHERE proposal_id=?""",
                (
                    "canary_passed" if passed else "rolled_back",
                    json.dumps({**report, "gate_passed": passed}, ensure_ascii=False),
                    "" if passed else now, now, proposal_id,
                ),
            )
        item = self.get(project_id=project_id, proposal_id=proposal_id)
        self._emit(
            item,
            "SELF_MOD_CANARY" if passed else "SELF_MOD_ROLLBACK",
            metrics=report,
            summary="Canary gate passed" if passed else "Canary auto-rolled back",
        )
        return item

    def promote(
        self, *, project_id: str, proposal_id: str, confirm_candidate_hash: str
    ) -> dict[str, Any]:
        with _LOCK, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM self_modification_proposals WHERE project_id=? AND proposal_id=?",
                (project_id, proposal_id),
            ).fetchone()
            if not row:
                raise KeyError(proposal_id)
            if str(row["status"]) != "canary_passed":
                raise ValueError("A passing canary evaluation is required before promotion")
            if str(confirm_candidate_hash) != str(row["candidate_hash"]):
                raise ValueError("Candidate hash confirmation does not match")
            conn.execute(
                """UPDATE self_modification_proposals SET status='superseded', updated_at=?
                   WHERE project_id=? AND status='active'""",
                (_now(), project_id),
            )
            conn.execute(
                """UPDATE self_modification_proposals SET status='active', canary_percent=100,
                       activated_at=?, updated_at=? WHERE proposal_id=?""",
                (_now(), _now(), proposal_id),
            )
        item = self.get(project_id=project_id, proposal_id=proposal_id)
        self._emit(item, "SELF_MOD_PROMOTION", summary="Candidate promoted to active overlay")
        return item

    def rollback(self, *, project_id: str, proposal_id: str, reason: str = "operator") -> dict[str, Any]:
        with _LOCK, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM self_modification_proposals WHERE project_id=? AND proposal_id=?",
                (project_id, proposal_id),
            ).fetchone()
            if not row:
                raise KeyError(proposal_id)
            if str(row["status"]) not in {"canary", "canary_passed", "active"}:
                raise ValueError("Only canary or active proposals can be rolled back")
            previous_id = str(row["previous_proposal_id"] or "")
            conn.execute(
                """UPDATE self_modification_proposals SET status='rolled_back',
                       rolled_back_at=?, updated_at=? WHERE proposal_id=?""",
                (_now(), _now(), proposal_id),
            )
            if str(row["status"]) == "active" and previous_id:
                conn.execute(
                    """UPDATE self_modification_proposals SET status='active',
                           activated_at=?, updated_at=?
                       WHERE project_id=? AND proposal_id=? AND status='superseded'""",
                    (_now(), _now(), project_id, previous_id),
                )
        item = self.get(project_id=project_id, proposal_id=proposal_id)
        self._emit(item, "SELF_MOD_ROLLBACK", summary=str(reason or "operator")[:260])
        return item

    def resolve_overlay(self, *, project_id: str, session_id: str) -> dict[str, Any]:
        """Resolve a stable candidate/baseline cohort without mutating settings."""
        evaluation = _EVAL_OVERLAY.get()
        if evaluation and str(evaluation.get("project_id") or "") == str(project_id):
            return {
                "enabled": True,
                "cohort": "evaluation",
                "proposal_id": str(evaluation["proposal_id"]),
                "candidate_hash": str(evaluation["candidate_hash"]),
                "text": str(evaluation["candidate_text"]),
                "bucket": -1,
                "canary_percent": 100,
            }
        with self._connect() as conn:
            canary = conn.execute(
                """SELECT * FROM self_modification_proposals
                   WHERE project_id=? AND status IN ('canary','canary_passed')
                   ORDER BY activated_at DESC LIMIT 1""",
                (project_id,),
            ).fetchone()
            active = self._current_active(conn, project_id)
        if canary:
            bucket = int(
                hashlib.sha256(
                    f"{canary['proposal_id']}|{session_id or '__anonymous__'}".encode("utf-8")
                ).hexdigest()[:8],
                16,
            ) % 100
            if bucket < int(canary["canary_percent"] or 0):
                return {
                    "enabled": True,
                    "tracked": True,
                    "cohort": "candidate",
                    "proposal_id": str(canary["proposal_id"]),
                    "candidate_hash": str(canary["candidate_hash"]),
                    "baseline_hash": str(canary["baseline_hash"]),
                    "overlay_proposal_id": str(canary["proposal_id"]),
                    "overlay_hash": str(canary["candidate_hash"]),
                    "text": str(canary["candidate_text"]),
                    "bucket": bucket,
                    "canary_percent": int(canary["canary_percent"] or 0),
                }
            return {
                "enabled": bool(active),
                "tracked": True,
                "cohort": "baseline",
                # Telemetry is keyed to the canary proposal, even when the
                # effective overlay is the previous active version (or none).
                "proposal_id": str(canary["proposal_id"]),
                "candidate_hash": str(canary["candidate_hash"]),
                "baseline_hash": str(canary["baseline_hash"]),
                "overlay_proposal_id": str(active["proposal_id"]) if active else "",
                "overlay_hash": str(active["candidate_hash"]) if active else str(canary["baseline_hash"]),
                "text": str(active["candidate_text"]) if active else "",
                "bucket": bucket,
                "canary_percent": int(canary["canary_percent"] or 0),
            }
        if active:
            return {
                "enabled": True,
                "tracked": False,
                "cohort": "baseline",
                "proposal_id": str(active["proposal_id"]),
                "candidate_hash": str(active["candidate_hash"]),
                "baseline_hash": str(active["baseline_hash"]),
                "overlay_proposal_id": str(active["proposal_id"]),
                "overlay_hash": str(active["candidate_hash"]),
                "text": str(active["candidate_text"]),
                "bucket": -1,
                "canary_percent": 100,
            }
        return {"enabled": False, "cohort": "none", "text": ""}

    @contextmanager
    def evaluation_overlay(self, *, project_id: str, proposal_id: str):
        """Temporarily bind an immutable candidate to sandbox replay requests."""
        proposal = self.get(project_id=project_id, proposal_id=proposal_id)
        if proposal["status"] not in {"draft", "eval_failed"}:
            raise ValueError("Only draft or failed-eval proposals can run evaluation")
        token = _EVAL_OVERLAY.set({
            "project_id": project_id,
            "proposal_id": proposal_id,
            "candidate_hash": proposal["candidate_hash"],
            "candidate_text": proposal["candidate_text"],
        })
        try:
            yield proposal
        finally:
            _EVAL_OVERLAY.reset(token)


_DEFAULT_LAB: SelfModificationLab | None = None


def get_self_modification_lab() -> SelfModificationLab:
    global _DEFAULT_LAB
    expected = Path(settings.DATA_DIR / "self_modification_lab.sqlite3")
    if _DEFAULT_LAB is None or _DEFAULT_LAB.db_path != expected:
        _DEFAULT_LAB = SelfModificationLab(expected)
    return _DEFAULT_LAB
