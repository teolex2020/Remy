"""
Prometheus-style metrics collector for Remy.

Exposes application health and performance data in Prometheus text format.
No external dependencies — plain string formatting.
"""

import logging
import threading
import time
from collections import defaultdict

logger = logging.getLogger("Metrics")

_DURATION_CAP = 1000  # Max stored durations per bucket


class MetricsCollector:
    """Thread-safe in-memory counters for request-scoped metrics.

    Updated by middleware and WebSocket handlers.
    Pulled by collect_metrics() on each /api/metrics scrape.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._http_requests_total: dict[tuple, int] = defaultdict(int)
        self._http_duration_seconds: dict[tuple, list[float]] = defaultdict(list)
        self._active_ws: dict[str, int] = defaultdict(int)
        self._llm_calls_total: int = 0
        self._llm_duration_seconds: list[float] = []

    def record_http_request(
        self, method: str, path: str, status_code: int, duration_sec: float
    ):
        """Called by RequestLoggingMiddleware after each request."""
        path_prefix = _normalize_path(path)
        status_class = f"{status_code // 100}xx"
        with self._lock:
            self._http_requests_total[(method, path_prefix, status_class)] += 1
            bucket = self._http_duration_seconds[(method, path_prefix)]
            bucket.append(duration_sec)
            if len(bucket) > _DURATION_CAP:
                self._http_duration_seconds[(method, path_prefix)] = bucket[-_DURATION_CAP:]

    def ws_connected(self, ws_type: str):
        """Called when a WebSocket client connects. ws_type: 'chat' or 'activity'."""
        with self._lock:
            self._active_ws[ws_type] += 1

    def ws_disconnected(self, ws_type: str):
        """Called when a WebSocket client disconnects."""
        with self._lock:
            self._active_ws[ws_type] = max(0, self._active_ws[ws_type] - 1)

    def record_llm_call(self, duration_sec: float):
        """Called after an LLM invocation completes."""
        with self._lock:
            self._llm_calls_total += 1
            self._llm_duration_seconds.append(duration_sec)
            if len(self._llm_duration_seconds) > _DURATION_CAP:
                self._llm_duration_seconds = self._llm_duration_seconds[-_DURATION_CAP:]

    def get_snapshot(self) -> dict:
        """Return a copy of all accumulated counters."""
        with self._lock:
            return {
                "http_requests_total": dict(self._http_requests_total),
                "http_duration_seconds": {
                    k: list(v) for k, v in self._http_duration_seconds.items()
                },
                "active_ws": dict(self._active_ws),
                "llm_calls_total": self._llm_calls_total,
                "llm_duration_seconds": list(self._llm_duration_seconds),
            }


def _normalize_path(path: str) -> str:
    """Reduce path cardinality for Prometheus labels.

    /api/records/abc123def456 -> /api/records/:id
    /api/todos/rec-xyz/toggle -> /api/todos/:id/toggle
    /api/stats -> /api/stats
    """
    parts = path.strip("/").split("/")
    normalized = []
    for i, part in enumerate(parts):
        if i <= 1:
            # Keep "api" and resource name
            normalized.append(part)
        elif len(part) > 12 or any(c.isdigit() for c in part[:8]):
            normalized.append(":id")
        else:
            normalized.append(part)
    return "/" + "/".join(normalized)


# Module-level singleton
metrics_collector = MetricsCollector()


def collect_metrics() -> str:
    """Gather all metrics and return Prometheus text exposition format.

    Data sources:
    - MetricsCollector (request counters, WS connections, LLM calls)
    - brain.stats() / brain.count() (memory records)
    - tool_health.get_health_report() (circuit breaker state)
    - usage_tracker.get_stats() (token usage)
    - Autonomy budget file (if exists)
    - Goal/outcome counts from brain tags
    """
    lines: list[str] = []
    seen_types: set[str] = set()

    def _metric(name: str, mtype: str, help_text: str, value, labels: dict | None = None):
        if name not in seen_types:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {mtype}")
            seen_types.add(name)
        if labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
            lines.append(f"{name}{{{label_str}}} {value}")
        else:
            lines.append(f"{name} {value}")

    snap = metrics_collector.get_snapshot()

    # ---- 1. Uptime ----
    try:
        from remy.web.api import _start_time
        uptime = time.time() - _start_time
        _metric("remy_uptime_seconds", "gauge", "Time since server start in seconds", f"{uptime:.1f}")
    except Exception:
        pass

    # ---- 2. HTTP request counters ----
    if snap["http_requests_total"]:
        lines.append("# HELP remy_http_requests_total Total HTTP requests")
        lines.append("# TYPE remy_http_requests_total counter")
        seen_types.add("remy_http_requests_total")
        for (method, path, status_class), count in sorted(snap["http_requests_total"].items()):
            lines.append(
                f'remy_http_requests_total{{method="{method}",path="{path}",status="{status_class}"}} {count}'
            )

    # ---- 3. HTTP latency (summary: p50, p99) ----
    if snap["http_duration_seconds"]:
        lines.append("# HELP remy_http_duration_seconds HTTP request duration in seconds")
        lines.append("# TYPE remy_http_duration_seconds summary")
        seen_types.add("remy_http_duration_seconds")
        for (method, path), durations in sorted(snap["http_duration_seconds"].items()):
            if not durations:
                continue
            sorted_d = sorted(durations)
            n = len(sorted_d)
            p50 = sorted_d[int(n * 0.5)]
            p99 = sorted_d[min(int(n * 0.99), n - 1)]
            lb = f'method="{method}",path="{path}"'
            lines.append(f'remy_http_duration_seconds{{{lb},quantile="0.5"}} {p50:.4f}')
            lines.append(f'remy_http_duration_seconds{{{lb},quantile="0.99"}} {p99:.4f}')
            lines.append(f"remy_http_duration_seconds_sum{{{lb}}} {sum(sorted_d):.4f}")
            lines.append(f"remy_http_duration_seconds_count{{{lb}}} {n}")

    # ---- 4. Active WebSockets ----
    for ws_type in ("chat", "activity"):
        count = snap["active_ws"].get(ws_type, 0)
        _metric("remy_active_websockets", "gauge", "Active WebSocket connections", count, {"type": ws_type})

    # ---- 5. LLM call metrics ----
    _metric("remy_llm_calls_total", "counter", "Total LLM invocations", snap["llm_calls_total"])
    if snap["llm_duration_seconds"]:
        durations = snap["llm_duration_seconds"]
        avg = sum(durations) / len(durations)
        _metric("remy_llm_duration_seconds_avg", "gauge", "Average LLM call duration in seconds", f"{avg:.3f}")

    # ---- 6. Brain record stats ----
    try:
        from remy.core.agent_tools import brain, brain_lock
        with brain_lock:
            record_count = brain.count()
            stats = brain.stats()
        _metric("remy_brain_records_total", "gauge", "Total brain memory records", record_count)
        if isinstance(stats, dict):
            for key, val in stats.items():
                if isinstance(val, bool):
                    _metric("remy_brain_stats", "gauge", "Brain statistics", int(val), {"key": str(key)})
                elif isinstance(val, (int, float)):
                    _metric("remy_brain_stats", "gauge", "Brain statistics", val, {"key": str(key)})
    except Exception:
        pass

    # ---- 7. Tool health (circuit breaker) ----
    try:
        from remy.core.brain_tools import tool_health
        report = tool_health.get_health_report()
        for tool_name, status in report.items():
            if "UNAVAILABLE" in status:
                val = 1
            elif "degraded" in status:
                val = 0.5
            else:
                val = 0
            _metric(
                "remy_tool_circuit_open", "gauge",
                "Tool circuit breaker state (1=open, 0.5=degraded, 0=healthy)",
                val, {"tool": tool_name},
            )
    except Exception:
        pass

    # ---- 8. Token usage ----
    try:
        from remy.core.usage_stats import usage_tracker
        usage = usage_tracker.get_stats()
        user_tokens = usage.get("user_tokens", 0)
        autonomy_tokens = usage.get("autonomy_tokens", 0)
        _metric("remy_tokens_total", "counter", "Total tokens consumed", user_tokens + autonomy_tokens)
        _metric("remy_tokens_user", "gauge", "User token usage", user_tokens)
        _metric("remy_tokens_autonomy", "gauge", "Autonomy token usage", autonomy_tokens)
    except Exception:
        pass

    # ---- 9. Autonomy budget (from shared operator snapshot) ----
    try:
        from remy.core.combined_runner import get_budget_runtime_snapshot

        budget = get_budget_runtime_snapshot(goal_limit=5, approval_limit=10)
        if budget:
            _metric("remy_autonomy_tokens_today", "gauge", "Autonomy tokens used today", int(budget.get("llm_tokens_today") or 0))
            _metric("remy_autonomy_tokens_this_hour", "gauge", "Autonomy tokens used this hour", int(budget.get("llm_tokens_this_hour") or 0))
            _metric("remy_autonomy_tokens_lifetime", "gauge", "Lifetime autonomy tokens", int(budget.get("llm_tokens_lifetime") or 0))
    except Exception:
        pass

    # ---- 10. Goal stats ----
    try:
        from remy.core.combined_runner import get_goal_runtime_snapshot

        goals = get_goal_runtime_snapshot(goal_limit=5, approval_limit=10)
        active = int(goals.get("active", 0) or 0)
        blocked = int(goals.get("blocked", 0) or 0)
        total_goals = int(goals.get("total", 0) or 0)
        completed = max(total_goals - active - blocked, 0)

        status_counts = {
            "active": active,
            "blocked": blocked,
            "completed": completed,
        }
        for status, count in sorted(status_counts.items()):
            if count:
                _metric("remy_goals", "gauge", "Autonomous goals by status", count, {"status": status})

        if total_goals > 0:
            _metric("remy_goal_completion_rate", "gauge", "Fraction of goals completed", f"{completed / total_goals:.3f}")
    except Exception:
        pass

    # ---- 11. Outcome stats ----
    try:
        from remy.core.agent_tools import brain as _brain, brain_lock as _bl
        with _bl:
            outcomes = _brain.search(query="", tags=["autonomous-outcome"], limit=200)
        success_count = sum(1 for o in outcomes if (o.metadata or {}).get("success"))
        _metric("remy_autonomy_outcomes_total", "counter", "Total autonomous outcomes", len(outcomes))
        _metric("remy_autonomy_outcomes_success", "gauge", "Successful autonomous outcomes", success_count)
        _metric("remy_autonomy_outcomes_failure", "gauge", "Failed autonomous outcomes", len(outcomes) - success_count)
        if outcomes:
            _metric("remy_autonomy_success_rate", "gauge", "Autonomous action success rate", f"{success_count / len(outcomes):.3f}")
    except Exception:
        pass

    # ---- 12. Event bus subscribers ----
    try:
        from remy.core.combined_runner import get_runtime_transport_snapshot

        transport = get_runtime_transport_snapshot()
        _metric("remy_event_bus_subscribers", "gauge", "Active event bus subscribers", transport.get("subscribers", 0))
    except Exception:
        pass

    lines.append("")  # Trailing newline
    return "\n".join(lines)
