"""
Per-goal execution history — structured attempt tracking.

Persists a capped log of execution attempts per goal so the agent
can see what it already tried, what failed, and why. Prevents
blind retries and surfaces failure patterns to the decision prompt.

Storage: data/goal_history/{goal_id}.json
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from remy.config.settings import settings

logger = logging.getLogger("TaskMemory")

_MAX_ATTEMPTS_PER_GOAL = 50
_LOCK = threading.Lock()


# ============== Failure Classification ==============


class FailureReason:
    """Structured failure categories for goal attempts."""

    TIMEOUT = "timeout"
    BLOCKED_EXTERNAL = "blocked_external"
    TOOL_ERROR = "tool_error"
    NO_ACTION = "no_action"
    ZERO_TOOL = "zero_tool"
    VALIDATION_FAILED = "validation_failed"
    REPEATED_FAILURE = "repeated_failure"
    UNKNOWN = "unknown"
    SUCCESS = "success"


def classify_failure(
    evaluation: dict | None,
    session_log: list[dict] | None = None,
    timeout: bool = False,
    blocked_external: bool = False,
    zero_tool: bool = False,
    worker_status: str = "",
) -> str:
    """Categorize the failure reason from execution signals."""
    if timeout:
        return FailureReason.TIMEOUT
    if blocked_external:
        return FailureReason.BLOCKED_EXTERNAL
    if zero_tool:
        return FailureReason.ZERO_TOOL

    if evaluation and evaluation.get("success"):
        return FailureReason.SUCCESS

    if worker_status in ("no_action",):
        return FailureReason.NO_ACTION

    # Check session log for tool errors
    if session_log:
        error_count = 0
        for entry in session_log:
            if not isinstance(entry, dict):
                continue
            if entry.get("type") == "tool_call" and entry.get("error"):
                error_count += 1
        if error_count > 0:
            return FailureReason.TOOL_ERROR

    # Check evaluation reason
    if evaluation:
        reason = (evaluation.get("reason") or "").lower()
        if "validation" in reason or "criteria" in reason:
            return FailureReason.VALIDATION_FAILED
        if "repeat" in reason or "same" in reason or "again" in reason:
            return FailureReason.REPEATED_FAILURE

    return FailureReason.UNKNOWN


# ============== Attempt Record ==============


@dataclass
class GoalAttemptRecord:
    """One execution attempt for a goal."""

    timestamp: str
    worker: str
    status: str
    failure_reason: str
    success: bool
    duration_ms: int = 0
    tokens_used: int = 0
    blocker: str = ""
    evidence_summary: str = ""
    goal_template: str = ""


# ============== Storage ==============


def _goal_history_dir() -> Path:
    return settings.DATA_DIR / "goal_history"


def _goal_history_path(goal_id: str) -> Path:
    # Sanitize goal_id for filesystem
    safe_id = "".join(c for c in goal_id if c.isalnum() or c in "-_")[:64]
    return _goal_history_dir() / f"{safe_id}.json"


def log_goal_attempt(
    goal_id: str,
    *,
    worker: str = "",
    status: str = "",
    failure_reason: str = "",
    success: bool = False,
    duration_ms: int = 0,
    tokens_used: int = 0,
    blocker: str = "",
    evidence_summary: str = "",
    goal_template: str = "",
) -> None:
    """Append an attempt record to the goal's history file."""
    if not goal_id:
        return

    record = GoalAttemptRecord(
        timestamp=datetime.now().isoformat(),
        worker=worker,
        status=status,
        failure_reason=failure_reason,
        success=success,
        duration_ms=duration_ms,
        tokens_used=tokens_used,
        blocker=blocker,
        evidence_summary=evidence_summary[:300],
        goal_template=goal_template,
    )

    path = _goal_history_path(goal_id)

    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        attempts = []
        if path.exists():
            try:
                attempts = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                attempts = []

        attempts.append(asdict(record))

        # Cap at max attempts — keep most recent
        if len(attempts) > _MAX_ATTEMPTS_PER_GOAL:
            attempts = attempts[-_MAX_ATTEMPTS_PER_GOAL:]

        try:
            path.write_text(json.dumps(attempts, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to write goal history for %s: %s", goal_id, e)


def get_goal_history(goal_id: str, limit: int = 50) -> list[dict]:
    """Get the execution history for a goal."""
    if not goal_id:
        return []

    path = _goal_history_path(goal_id)

    with _LOCK:
        if not path.exists():
            return []
        try:
            attempts = json.loads(path.read_text(encoding="utf-8"))
            return attempts[-limit:]
        except Exception:
            return []


def get_goal_history_summary(goal_id: str) -> dict:
    """Get aggregated stats for a goal's execution history."""
    attempts = get_goal_history(goal_id)
    if not attempts:
        return {"goal_id": goal_id, "total_attempts": 0}

    total = len(attempts)
    successes = sum(1 for a in attempts if a.get("success"))
    failures = total - successes

    # Count failure reasons
    reason_counts: dict[str, int] = {}
    for a in attempts:
        reason = a.get("failure_reason", "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    # Most common failure
    top_failure = ""
    if reason_counts:
        non_success = {k: v for k, v in reason_counts.items() if k != FailureReason.SUCCESS}
        if non_success:
            top_failure = max(non_success, key=lambda k: non_success[k])

    # Total time and tokens
    total_duration = sum(a.get("duration_ms", 0) for a in attempts)
    total_tokens = sum(a.get("tokens_used", 0) for a in attempts)

    # Last attempt info
    last = attempts[-1]

    return {
        "goal_id": goal_id,
        "total_attempts": total,
        "successes": successes,
        "failures": failures,
        "completion_rate": round(successes / total, 3) if total else 0.0,
        "failure_reasons": reason_counts,
        "top_failure": top_failure,
        "total_duration_ms": total_duration,
        "total_tokens": total_tokens,
        "last_status": last.get("status", ""),
        "last_failure_reason": last.get("failure_reason", ""),
        "last_attempt": last.get("timestamp", ""),
    }


# ============== Prompt Formatting ==============


def format_goal_history_for_prompt(goal_id: str, limit: int = 5) -> str:
    """Format recent goal attempts as compact context for the decision prompt.

    Shows what the agent already tried, so it doesn't repeat the same approach.
    """
    attempts = get_goal_history(goal_id, limit=limit)
    if not attempts:
        return ""

    # Take last N attempts
    recent = attempts[-limit:]
    total = len(get_goal_history(goal_id))

    lines = [f"\nPREVIOUS ATTEMPTS ({len(recent)} of {total} total):"]

    for i, a in enumerate(recent, 1):
        ok = "OK" if a.get("success") else "FAIL"
        reason = a.get("failure_reason", "")
        worker = a.get("worker", "agent")
        status = a.get("status", "")
        evidence = a.get("evidence_summary", "")
        blocker = a.get("blocker", "")
        duration = a.get("duration_ms", 0)

        line = f"  {i}. [{ok}] {worker}"
        if status:
            line += f" status={status}"
        if reason and reason != FailureReason.SUCCESS:
            line += f" reason={reason}"
        if blocker:
            line += f" blocker={blocker[:80]}"
        if evidence:
            line += f" | {evidence[:100]}"
        if duration:
            line += f" ({duration}ms)"
        lines.append(line)

    # Add pattern warning if repeating
    failure_reasons = [a.get("failure_reason") for a in recent if not a.get("success")]
    if len(failure_reasons) >= 3:
        # Check if same reason repeats
        last_3 = failure_reasons[-3:]
        if len(set(last_3)) == 1 and last_3[0] != FailureReason.SUCCESS:
            lines.append(
                f"  WARNING: Same failure reason '{last_3[0]}' repeated 3+ times. Change approach or mark blocked."
            )

    lines.append(
        "  Do NOT repeat a failed approach. Try a different strategy or report blocked_external."
    )
    return "\n".join(lines) + "\n"
