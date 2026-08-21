"""Unified observable execution contract for every Remy run.

The envelope intentionally builds on :mod:`execution_ledger`.  It does not
replace experiment/workflow/chat persistence and it does not depend on an
external agent framework.  Its job is to answer the operator's questions:
what is running, what is it doing, what limits apply, and why did it stop?
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from remy.core.execution_ledger import get_execution_ledger

RUN_SCHEMA = "remy.run-envelope"
RUN_SCHEMA_VERSION = 1

ACTIVE_RUN_STATUSES = {
    "queued",
    "planning",
    "running",
    "waiting_tool",
    "waiting_approval",
    "paused",
    "stopping",
}
TERMINAL_RUN_STATUSES = {
    "completed",
    "completed_with_limits",
    "failed",
    "cancelled",
    "interrupted",
    "blocked",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True, frozen=True)
class RunLimits:
    max_turns: int = 80
    token_budget: int = 128_000
    max_parallel_workers: int = 1
    loop_repeat_limit: int = 8

    def normalized(self) -> "RunLimits":
        return RunLimits(
            max_turns=max(1, min(int(self.max_turns), 10_000)),
            token_budget=max(1_000, min(int(self.token_budget), 20_000_000)),
            max_parallel_workers=max(1, min(int(self.max_parallel_workers), 16)),
            loop_repeat_limit=max(2, min(int(self.loop_repeat_limit), 50)),
        )


DEFAULT_LIMITS: dict[str, RunLimits] = {
    "chat": RunLimits(max_turns=80, token_budget=128_000, max_parallel_workers=1),
    "research": RunLimits(max_turns=120, token_budget=500_000, max_parallel_workers=3),
    "experiment": RunLimits(max_turns=100, token_budget=750_000, max_parallel_workers=3),
    "pipeline": RunLimits(max_turns=200, token_budget=250_000, max_parallel_workers=1),
    "ptc_pilot": RunLimits(max_turns=12, token_budget=128_000, max_parallel_workers=1),
    "automation": RunLimits(max_turns=200, token_budget=250_000, max_parallel_workers=1),
    "worker_group": RunLimits(max_turns=80, token_budget=192_000, max_parallel_workers=3),
    "agent_team": RunLimits(max_turns=18, token_budget=144_000, max_parallel_workers=3),
}


class RunLimitExceeded(RuntimeError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def make_run_envelope(
    *,
    run_id: str,
    kind: str,
    source_id: str,
    goal: str,
    owner_project_id: str,
    brain_id: str,
    conversation_id: str = "",
    limits: RunLimits | None = None,
    status: str = "queued",
    phase: str = "queued",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_limits = (limits or DEFAULT_LIMITS.get(kind) or RunLimits()).normalized()
    now = _now()
    return {
        "schema": RUN_SCHEMA,
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": str(run_id),
        "kind": str(kind),
        "source_id": str(source_id),
        "owner_project_id": str(owner_project_id),
        "brain_id": str(brain_id),
        "conversation_id": str(conversation_id or ""),
        "goal": str(goal or "").strip()[:4_000],
        "status": status,
        "phase": phase,
        "current_step": "",
        "limits": asdict(resolved_limits),
        "usage": {
            "turns": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "active_workers": 0,
            "peak_workers": 0,
        },
        "started_at": now,
        "updated_at": now,
        "heartbeat_at": now,
        "finished_at": "",
        "stop_reason": "",
        "error": "",
        "artifacts": [],
        "recent_signatures": [],
        "metadata": dict(metadata or {}),
    }


def start_run(
    *,
    kind: str,
    source_id: str,
    goal: str,
    owner_project_id: str,
    brain_id: str,
    conversation_id: str = "",
    channel: str = "",
    limits: RunLimits | None = None,
    idempotency_class: str = "side_effecting",
    metadata: dict[str, Any] | None = None,
    run_id: str = "",
) -> dict[str, Any]:
    run_id = str(run_id or f"run-{uuid.uuid4().hex[:12]}")
    envelope = make_run_envelope(
        run_id=run_id,
        kind=kind,
        source_id=source_id,
        goal=goal,
        owner_project_id=owner_project_id,
        brain_id=brain_id,
        conversation_id=conversation_id,
        limits=limits,
        metadata=metadata,
    )
    attempt = get_execution_ledger().claim(
        kind=kind,
        job_id=f"{source_id}:{run_id}",
        idempotency_class=idempotency_class,
        owner_project_id=owner_project_id,
        brain_id=brain_id,
        session_id=conversation_id,
        channel=channel,
        metadata={"run_envelope": envelope},
    )
    get_execution_ledger().mark_running(attempt["attempt_id"])
    return update_run(
        attempt["attempt_id"], status="running", phase="starting", event="run_started"
    )


def attach_attempt_envelope(
    *,
    attempt_id: str,
    run_id: str,
    kind: str,
    source_id: str,
    goal: str,
    owner_project_id: str,
    brain_id: str,
    conversation_id: str = "",
    limits: RunLimits | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    envelope = make_run_envelope(
        run_id=run_id,
        kind=kind,
        source_id=source_id,
        goal=goal,
        owner_project_id=owner_project_id,
        brain_id=brain_id,
        conversation_id=conversation_id,
        limits=limits,
        status="running",
        phase="starting",
        metadata=metadata,
    )
    return get_execution_ledger().patch_metadata(
        attempt_id,
        {"run_envelope": envelope},
        event="run_started",
        event_data={"run_id": run_id, "kind": kind},
    )


def _attempt_envelope(attempt: dict[str, Any] | None) -> dict[str, Any]:
    if not attempt:
        raise KeyError("Run attempt not found")
    envelope = (attempt.get("metadata") or {}).get("run_envelope")
    if not isinstance(envelope, dict):
        raise KeyError("Attempt has no run envelope")
    return dict(envelope)


def update_run(
    attempt_id: str,
    *,
    status: str | None = None,
    phase: str | None = None,
    current_step: str | None = None,
    usage: dict[str, int] | None = None,
    stop_reason: str | None = None,
    error: str | None = None,
    artifacts: list[dict[str, Any]] | None = None,
    recent_signatures: list[str] | None = None,
    event: str = "run_progress",
) -> dict[str, Any]:
    ledger = get_execution_ledger()
    attempt = ledger.get(attempt_id)
    envelope = _attempt_envelope(attempt)
    now = _now()
    if status is not None:
        if status not in ACTIVE_RUN_STATUSES | TERMINAL_RUN_STATUSES:
            raise ValueError(f"Unsupported run status: {status}")
        envelope["status"] = status
    if phase is not None:
        envelope["phase"] = str(phase)[:120]
    if current_step is not None:
        envelope["current_step"] = str(current_step)[:500]
    if usage:
        merged_usage = dict(envelope.get("usage") or {})
        merged_usage.update({key: max(0, int(value)) for key, value in usage.items()})
        envelope["usage"] = merged_usage
    if stop_reason is not None:
        envelope["stop_reason"] = str(stop_reason)[:200]
    if error is not None:
        envelope["error"] = str(error)[:4_000]
    if artifacts is not None:
        envelope["artifacts"] = list(artifacts)[-100:]
    if recent_signatures is not None:
        envelope["recent_signatures"] = [str(item)[:300] for item in recent_signatures[-20:]]
    envelope["updated_at"] = now
    envelope["heartbeat_at"] = now
    updated = ledger.patch_metadata(
        attempt_id,
        {"run_envelope": envelope},
        event=event,
        event_data={
            "run_id": envelope.get("run_id", ""),
            "status": envelope.get("status", ""),
            "phase": envelope.get("phase", ""),
            "current_step": envelope.get("current_step", ""),
        },
    )
    return public_envelope(updated)


def finish_run(
    attempt_id: str,
    *,
    status: str,
    stop_reason: str = "",
    error: str = "",
    output_ref: str = "",
    artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if status not in TERMINAL_RUN_STATUSES:
        raise ValueError(f"Run status is not terminal: {status}")
    ledger = get_execution_ledger()
    attempt = ledger.get(attempt_id)
    envelope = _attempt_envelope(attempt)
    now = _now()
    envelope.update(
        {
            "status": status,
            "phase": "finished",
            "updated_at": now,
            "heartbeat_at": now,
            "finished_at": now,
            "stop_reason": str(stop_reason or "")[:200],
            "error": str(error or "")[:4_000],
            "current_step": "",
        }
    )
    if artifacts is not None:
        envelope["artifacts"] = list(artifacts)[-100:]
    ledger_state = {
        "completed": "completed",
        "completed_with_limits": "completed_with_limits",
        "cancelled": "cancelled",
        "blocked": "blocked",
        "interrupted": "unknown",
    }.get(status, "failed")
    finished = ledger.finish(
        attempt_id,
        ledger_state,
        output_ref=output_ref,
        error=error,
        metadata={"run_envelope": envelope},
    )
    return public_envelope(finished)


def public_envelope(attempt: dict[str, Any]) -> dict[str, Any]:
    envelope = _attempt_envelope(attempt)
    envelope.pop("recent_signatures", None)
    envelope["attempt_id"] = attempt.get("attempt_id", "")
    envelope["ledger_state"] = attempt.get("state", "")
    envelope["receipts"] = list(attempt.get("receipts") or [])[-50:]
    return envelope


def get_run(run_id: str, *, owner_project_id: str = "") -> dict[str, Any] | None:
    for attempt in get_execution_ledger().list_attempts(
        owner_project_id=owner_project_id, limit=500
    ):
        envelope = (attempt.get("metadata") or {}).get("run_envelope")
        if isinstance(envelope, dict) and envelope.get("run_id") == run_id:
            detailed = get_execution_ledger().get(attempt["attempt_id"])
            return public_envelope(detailed or attempt)
    return None


def list_runs(
    *, owner_project_id: str, status: str = "", kind: str = "", limit: int = 100
) -> list[dict[str, Any]]:
    items = []
    for attempt in get_execution_ledger().list_attempts(
        kind=kind, owner_project_id=owner_project_id, limit=limit
    ):
        envelope = (attempt.get("metadata") or {}).get("run_envelope")
        if not isinstance(envelope, dict):
            continue
        if status and envelope.get("status") != status:
            continue
        items.append(public_envelope(attempt))
    return items


def recover_interrupted_runs() -> list[dict[str, Any]]:
    recovered = get_execution_ledger().recover_orphans()
    results = []
    for attempt in recovered:
        envelope = (attempt.get("metadata") or {}).get("run_envelope")
        if not isinstance(envelope, dict):
            continue
        now = _now()
        envelope.update(
            {
                "status": "interrupted",
                "phase": "finished",
                "finished_at": now,
                "updated_at": now,
                "heartbeat_at": now,
                "stop_reason": "process_restarted",
                "error": "Remy stopped before this run produced a terminal receipt.",
                "current_step": "",
            }
        )
        # recover_orphans already made the ledger row terminal, so update the
        # metadata directly through the terminal-safe helper below.
        get_execution_ledger().replace_terminal_metadata(
            attempt["attempt_id"], {"run_envelope": envelope}
        )
        refreshed = get_execution_ledger().get(attempt["attempt_id"])
        results.append(public_envelope(refreshed or attempt))
    return results


class RunCoordinator:
    """Enforce bounded work while keeping every decision operator-visible."""

    def __init__(self, attempt_id: str):
        self.attempt_id = attempt_id
        self._lock = threading.RLock()

    def snapshot(self) -> dict[str, Any]:
        attempt = get_execution_ledger().get(self.attempt_id)
        if not attempt:
            raise KeyError(self.attempt_id)
        return public_envelope(attempt)

    def heartbeat(self, *, phase: str = "running", step: str = "") -> dict[str, Any]:
        return update_run(
            self.attempt_id,
            status="running" if phase not in {"waiting_tool", "waiting_approval", "paused"} else phase,
            phase=phase,
            current_step=step,
        )

    def step(self, label: str, *, signature: str = "") -> dict[str, Any]:
        with self._lock:
            attempt = get_execution_ledger().get(self.attempt_id)
            envelope = _attempt_envelope(attempt)
            usage = dict(envelope.get("usage") or {})
            limits = dict(envelope.get("limits") or {})
            turns = int(usage.get("turns") or 0) + 1
            if turns > int(limits.get("max_turns") or 1):
                self._limit("turn_limit", "Run reached its maximum number of steps.")
            recent = list(envelope.get("recent_signatures") or [])
            clean_signature = str(signature or "").strip()[:300]
            if clean_signature:
                recent.append(clean_signature)
                repeat_limit = int(limits.get("loop_repeat_limit") or 8)
                if len(recent) >= repeat_limit and len(set(recent[-repeat_limit:])) == 1:
                    self._limit("loop_detected", "The same operation repeated without visible progress.")
            usage["turns"] = turns
            return update_run(
                self.attempt_id,
                status="running",
                phase="executing",
                current_step=label,
                usage=usage,
                recent_signatures=recent,
                event="run_step",
            )

    def consume_tokens(self, *, input_tokens: int = 0, output_tokens: int = 0) -> dict[str, Any]:
        with self._lock:
            attempt = get_execution_ledger().get(self.attempt_id)
            envelope = _attempt_envelope(attempt)
            usage = dict(envelope.get("usage") or {})
            usage["input_tokens"] = int(usage.get("input_tokens") or 0) + max(0, int(input_tokens))
            usage["output_tokens"] = int(usage.get("output_tokens") or 0) + max(0, int(output_tokens))
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
            budget = int((envelope.get("limits") or {}).get("token_budget") or 1)
            if usage["total_tokens"] > budget:
                self._limit("token_budget", "Run reached its token budget.")
            return update_run(self.attempt_id, usage=usage, event="run_usage")

    def worker_started(self, label: str = "") -> dict[str, Any]:
        with self._lock:
            attempt = get_execution_ledger().get(self.attempt_id)
            envelope = _attempt_envelope(attempt)
            usage = dict(envelope.get("usage") or {})
            active = int(usage.get("active_workers") or 0) + 1
            limit = int((envelope.get("limits") or {}).get("max_parallel_workers") or 1)
            if active > limit:
                raise RunLimitExceeded(
                    "parallel_worker_limit", f"Only {limit} parallel workers are allowed."
                )
            usage["active_workers"] = active
            usage["peak_workers"] = max(int(usage.get("peak_workers") or 0), active)
            return update_run(
                self.attempt_id,
                phase="delegating",
                current_step=label,
                usage=usage,
                event="worker_started",
            )

    def worker_finished(self) -> dict[str, Any]:
        with self._lock:
            attempt = get_execution_ledger().get(self.attempt_id)
            envelope = _attempt_envelope(attempt)
            usage = dict(envelope.get("usage") or {})
            usage["active_workers"] = max(0, int(usage.get("active_workers") or 0) - 1)
            return update_run(self.attempt_id, usage=usage, event="worker_finished")

    def _limit(self, reason: str, message: str) -> None:
        update_run(
            self.attempt_id,
            status="stopping",
            phase="limit_reached",
            stop_reason=reason,
            error=message,
            event="run_limit_reached",
        )
        raise RunLimitExceeded(reason, message)


def extract_token_usage(metadata: dict[str, Any] | None) -> tuple[int, int]:
    metadata = dict(metadata or {})
    usage = metadata.get("usage_metadata") or metadata.get("token_usage") or metadata
    input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0
    output_tokens = (
        usage.get("completion_tokens", 0)
        or usage.get("candidates_tokens", 0)
        or usage.get("output_tokens", 0)
        or 0
    )
    return int(input_tokens or 0), int(output_tokens or 0)


_CONTROL_LOCK = threading.RLock()
_STOP_CALLBACKS: dict[str, Callable[[str], Any]] = {}


def register_run_stop(run_id: str, callback: Callable[[str], Any]) -> None:
    with _CONTROL_LOCK:
        _STOP_CALLBACKS[run_id] = callback


def unregister_run_stop(run_id: str) -> None:
    with _CONTROL_LOCK:
        _STOP_CALLBACKS.pop(run_id, None)


def request_run_stop(run_id: str, *, owner_project_id: str, reason: str = "Stopped by user") -> dict[str, Any]:
    run = get_run(run_id, owner_project_id=owner_project_id)
    if not run:
        raise KeyError(run_id)
    if run.get("status") in TERMINAL_RUN_STATUSES:
        return run
    update_run(
        run["attempt_id"],
        status="stopping",
        phase="stopping",
        stop_reason="user_stopped",
        current_step=reason,
        event="stop_requested",
    )
    with _CONTROL_LOCK:
        callback = _STOP_CALLBACKS.get(run_id)
    if callback is not None:
        callback(reason)
    return get_run(run_id, owner_project_id=owner_project_id) or run
