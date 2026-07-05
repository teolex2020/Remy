"""
Execution Log — structured per-cycle run logging.

Records every autonomy cycle with:
- timestamp, duration, tokens, cost
- goal_id, pack_id, worker type
- tool calls made (name + success)
- outcome (success/failure/timeout/blocked)
- step count vs budget

Persists to data/execution_log.jsonl (append-only, line-delimited JSON).
Keeps last N entries in memory for fast queries.

Inspired by eggent's run-log.ts pattern.

Status taxonomy (canonical — all modules must use this):
  success          — measurable progress, tool called successfully
  partial_progress — timeout/incomplete but real tools ran and data stored
  failure          — tool errors, no progress, wrong approach
  timeout          — timed out with zero real tool calls
  blocked          — external blocker (captcha, auth, payment)
  zero_tool        — no tools called at all, pure filler
  error            — internal error (exception in framework)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field

logger = logging.getLogger("ExecutionLog")

LOG_FILE = "execution_log.jsonl"
MAX_IN_MEMORY = 200

# ── Canonical status taxonomy ──────────────────────────────────────
# Every module (autonomy, critique, reporter, metrics) should use
# these constants instead of ad-hoc string comparisons.

STATUS_SUCCESS = "success"
STATUS_PARTIAL = "partial_progress"
STATUS_FAILURE = "failure"
STATUS_TIMEOUT = "timeout"
STATUS_BLOCKED = "blocked"
STATUS_ZERO_TOOL = "zero_tool"
STATUS_ERROR = "error"

# Statuses that count as "positive outcome" for completion_rate
POSITIVE_STATUSES = frozenset({STATUS_SUCCESS, STATUS_PARTIAL})

# Statuses that count as "negative outcome"
NEGATIVE_STATUSES = frozenset({STATUS_FAILURE, STATUS_TIMEOUT, STATUS_ZERO_TOOL, STATUS_ERROR})


def derive_cycle_status(
    *,
    worker_status: str,
    eval_success: bool,
    has_real_tools: bool,
) -> str:
    """Single source of truth for cycle status derivation.

    Args:
        worker_status: raw status from WorkerExecutionResult (e.g. "timeout",
                       "partial_progress", "findings_collected", "completed",
                       "blocked_external", "searching", "attempted", "")
        eval_success:  evaluation.get("success")
        has_real_tools: whether any non-sentinel tool calls were made
    """
    # 1. Worker explicitly reports partial_progress or findings_collected
    if worker_status in ("partial_progress", "findings_collected"):
        return STATUS_PARTIAL

    # 2. Worker timed out
    if worker_status == "timeout":
        return STATUS_PARTIAL if has_real_tools else STATUS_TIMEOUT

    # 3. External blocker
    if worker_status == "blocked_external":
        return STATUS_BLOCKED

    # 4. No tools at all + eval says failure
    if not has_real_tools and not eval_success:
        return STATUS_ZERO_TOOL

    # 5. Eval says success but worker is still mid-work (searching/attempted)
    if eval_success and has_real_tools and worker_status in ("searching", "attempted"):
        return STATUS_PARTIAL

    # 6. Normal success/failure from evaluation
    return STATUS_SUCCESS if eval_success else STATUS_FAILURE


@dataclass
class ToolCallEntry:
    """One tool call within a cycle."""

    tool: str
    success: bool = True
    duration_ms: int = 0


@dataclass
class ExecutionEntry:
    """One complete autonomy cycle execution record."""

    timestamp: float
    cycle_num: int = 0
    goal_id: str = ""
    goal_description: str = ""
    pack_id: str = ""
    pack_label: str = ""
    worker: str = ""
    status: str = ""  # success | failure | timeout | blocked | zero_tool | error
    duration_ms: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    tool_calls: list[ToolCallEntry] = field(default_factory=list)
    tool_count: int = 0
    step_budget: int = 0  # max allowed steps for this pack
    steps_used: int = 0  # actual steps taken
    evaluation_confidence: float = 0.0
    evaluation_reason: str = ""
    turn_class: str = ""  # productive | maintenance | idle
    verified: bool = False
    repeated_failure: bool = False
    memory_assisted: bool = False


class ExecutionLog:
    """Append-only execution log with in-memory tail for fast queries."""

    def __init__(self):
        from remy.core.meta_store import resolve_path

        self._path = resolve_path(LOG_FILE, "metrics")
        self._lock = threading.Lock()
        self._entries: list[dict] = []
        self._load_tail()

    def _load_tail(self):
        """Load last MAX_IN_MEMORY entries from disk."""
        if not self._path.exists():
            return
        try:
            lines = self._path.read_text(encoding="utf-8").strip().split("\n")
            tail = lines[-MAX_IN_MEMORY:] if len(lines) > MAX_IN_MEMORY else lines
            for line in tail:
                if line.strip():
                    try:
                        self._entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            logger.warning("Failed to load execution log tail: %s", e)

    def record(self, entry: ExecutionEntry) -> bool:
        """Append one execution entry to log."""
        data = asdict(entry)
        # Flatten tool_calls for compact storage
        data["tool_calls"] = [
            {"tool": tc["tool"], "ok": tc["success"]} for tc in data.get("tool_calls", [])
        ]
        data["tool_count"] = len(data["tool_calls"])

        with self._lock:
            self._entries.append(data)
            if len(self._entries) > MAX_IN_MEMORY:
                self._entries = self._entries[-MAX_IN_MEMORY:]

            # Append to file
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data, default=str) + "\n")
                return True
            except Exception as e:
                logger.warning("Failed to append to execution log: %s", e)
                return False

    def get_recent(self, limit: int = 50) -> list[dict]:
        """Get most recent entries."""
        with self._lock:
            return list(self._entries[-limit:])

    def get_by_pack(self, pack_id: str, limit: int = 20) -> list[dict]:
        """Get recent entries for a specific pack."""
        with self._lock:
            matched = [e for e in self._entries if e.get("pack_id") == pack_id]
            return matched[-limit:]

    def get_by_goal(self, goal_id: str, limit: int = 20) -> list[dict]:
        """Get recent entries for a specific goal."""
        with self._lock:
            matched = [e for e in self._entries if e.get("goal_id") == goal_id]
            return matched[-limit:]

    def get_pack_summary(self) -> dict:
        """Aggregate stats per pack from in-memory entries."""
        with self._lock:
            packs: dict[str, dict] = {}
            for entry in self._entries:
                pid = entry.get("pack_id") or "unknown"
                if pid not in packs:
                    packs[pid] = {
                        "pack_id": pid,
                        "pack_label": entry.get("pack_label", pid),
                        "total_runs": 0,
                        "successes": 0,
                        "partial_progress": 0,
                        "failures": 0,
                        "timeouts": 0,
                        "blocked": 0,
                        "total_duration_ms": 0,
                        "total_tokens": 0,
                        "total_cost_usd": 0.0,
                        "total_steps": 0,
                        "total_step_budget": 0,
                        "avg_step_utilization": 0.0,
                    }
                p = packs[pid]
                p["total_runs"] += 1
                p["total_duration_ms"] += entry.get("duration_ms", 0)
                p["total_tokens"] += entry.get("tokens_used", 0)
                p["total_cost_usd"] += entry.get("cost_usd", 0.0)
                p["total_steps"] += entry.get("steps_used", 0)
                p["total_step_budget"] += entry.get("step_budget", 0)

                status = entry.get("status", "")
                if status == STATUS_SUCCESS:
                    p["successes"] += 1
                elif status == STATUS_PARTIAL:
                    p["partial_progress"] += 1
                elif status == STATUS_TIMEOUT:
                    p["timeouts"] += 1
                elif status == STATUS_BLOCKED:
                    p["blocked"] += 1
                else:
                    p["failures"] += 1

            # Compute averages
            for p in packs.values():
                n = p["total_runs"]
                if n:
                    p["avg_duration_ms"] = round(p["total_duration_ms"] / n)
                    p["avg_tokens"] = round(p["total_tokens"] / n)
                    p["avg_cost_usd"] = round(p["total_cost_usd"] / n, 6)
                    p["completion_rate"] = round((p["successes"] + p["partial_progress"]) / n, 3)
                if p["total_step_budget"] > 0:
                    p["avg_step_utilization"] = round(p["total_steps"] / p["total_step_budget"], 3)
                p["total_cost_usd"] = round(p["total_cost_usd"], 6)

            return packs

    def get_step_efficiency(self) -> dict:
        """Step budget utilization stats — are we burning steps or under-using them?"""
        with self._lock:
            by_pack: dict[str, list[float]] = {}
            for entry in self._entries:
                budget = entry.get("step_budget", 0)
                used = entry.get("steps_used", 0)
                if budget <= 0:
                    continue
                pid = entry.get("pack_id") or "unknown"
                if pid not in by_pack:
                    by_pack[pid] = []
                by_pack[pid].append(used / budget)

            result = {}
            for pid, ratios in by_pack.items():
                n = len(ratios)
                avg = sum(ratios) / n if n else 0
                maxed_out = sum(1 for r in ratios if r >= 0.95)
                result[pid] = {
                    "runs": n,
                    "avg_utilization": round(avg, 3),
                    "maxed_out_runs": maxed_out,
                    "maxed_out_rate": round(maxed_out / n, 3) if n else 0,
                }
            return result


# Singleton
execution_log = ExecutionLog()


def record_cycle_execution(
    *,
    cycle_num: int,
    goal: dict | None,
    worker_result,
    session_log: list[dict] | None,
    evaluation: dict,
    duration_ms: int,
    tokens_used: int,
    cost_usd: float,
    turn_class: str = "",
    verified: bool = False,
    repeated_failure: bool = False,
    memory_assisted: bool = False,
):
    """Record one autonomy cycle to the execution log with consistent fallbacks."""
    try:
        from remy.core.capability_packs import resolve_pack

        goal = goal or {}
        pack = resolve_pack(goal)
        tool_entries: list[ToolCallEntry] = []
        for entry in session_log or []:
            if not isinstance(entry, dict) or entry.get("type") != "tool_call":
                continue
            tool_entries.append(
                ToolCallEntry(
                    tool=entry.get("tool", ""),
                    success=not str(entry.get("result", "")).startswith("Error"),
                )
            )

        worker_status = getattr(worker_result, "status", "") or ""
        real_tools = [t for t in tool_entries if t.tool != "worker_timeout"]
        status = derive_cycle_status(
            worker_status=worker_status,
            eval_success=bool(evaluation.get("success")),
            has_real_tools=bool(real_tools),
        )

        entry = ExecutionEntry(
            timestamp=time.time(),
            cycle_num=cycle_num,
            goal_id=goal.get("goal_id", ""),
            goal_description=goal.get("description", "")[:120],
            pack_id=goal.get("_pack_id") or pack.id,
            pack_label=goal.get("_pack_label") or pack.label,
            worker=getattr(worker_result, "worker", ""),
            status=status,
            duration_ms=duration_ms,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            tool_calls=tool_entries,
            tool_count=len(tool_entries),
            step_budget=goal.get("_pack_step_budget") or getattr(pack, "step_budget", 0),
            steps_used=getattr(worker_result, "tool_calls", 0) or len(tool_entries),
            evaluation_confidence=evaluation.get("confidence", 0.0),
            evaluation_reason=(evaluation.get("reason") or "")[:120],
            turn_class=turn_class,
            verified=bool(verified),
            repeated_failure=bool(repeated_failure),
            memory_assisted=bool(memory_assisted),
        )
        if not execution_log.record(entry):
            logger.warning(
                "Execution log write failed for goal=%s path=%s",
                goal.get("goal_id", ""),
                execution_log._path,
            )
    except Exception as e:
        logger.warning(
            "Execution log recording failed for goal=%s path=%s: %s",
            (goal or {}).get("goal_id", ""),
            getattr(execution_log, "_path", "unknown"),
            e,
        )
