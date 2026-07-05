"""
Tool Health & Circuit Breaker — per-tool failure tracking with auto-recovery.

Tracks consecutive failures per tool. After FAILURE_THRESHOLD failures within
RECOVERY_SEC, the tool's circuit opens (unavailable) until cooldown expires.
Also defines retry constants for transient-failure tools.
"""

import logging
import threading
import time

logger = logging.getLogger("BrainTools")

# Tools that may have transient failures (network, API rate limits)
_RETRYABLE_TOOLS = frozenset({"web_search", "http_get"})
_MAX_RETRIES = 2
_RETRY_DELAYS = [2, 5]  # seconds between retries


class ToolHealth:
    """Per-tool circuit breaker and health tracking. Thread-safe."""

    FAILURE_THRESHOLD = 3  # failures before circuit opens
    RECOVERY_SEC = 600  # 10 min cooldown

    def __init__(self):
        self._failures: dict[str, list[float]] = {}  # tool -> [timestamps]
        self._circuit_open_until: dict[str, float] = {}  # tool -> timestamp
        self._lock = threading.Lock()

    def record_failure(self, tool_name: str):
        """Record a tool failure. Opens circuit if threshold reached."""
        now = time.time()
        with self._lock:
            if tool_name not in self._failures:
                self._failures[tool_name] = []

            # Keep only recent failures (last 10 min)
            self._failures[tool_name] = [t for t in self._failures[tool_name] if now - t < 600] + [
                now
            ]

            if len(self._failures[tool_name]) >= self.FAILURE_THRESHOLD:
                self._circuit_open_until[tool_name] = now + self.RECOVERY_SEC
                logger.warning(
                    "Circuit OPEN for tool '%s' until %s (%d recent failures)",
                    tool_name,
                    time.strftime("%H:%M:%S", time.localtime(now + self.RECOVERY_SEC)),
                    len(self._failures[tool_name]),
                )

        # Mirror to Aura brain for persistent cross-session tracking
        try:
            from remy.core.agent_tools import brain
            brain.record_tool_failure(tool_name)
        except Exception:
            pass

    def record_success(self, tool_name: str):
        """Record a tool success. Clears failure history."""
        with self._lock:
            self._failures.pop(tool_name, None)
            self._circuit_open_until.pop(tool_name, None)

        # Mirror to Aura brain for persistent cross-session tracking
        try:
            from remy.core.agent_tools import brain
            brain.record_tool_success(tool_name)
        except Exception:
            pass

    def is_available(self, tool_name: str) -> bool:
        """Check if a tool's circuit is closed (available)."""
        with self._lock:
            open_until = self._circuit_open_until.get(tool_name, 0)
            if time.time() >= open_until:
                # Circuit has recovered — clear it
                self._circuit_open_until.pop(tool_name, None)
                return True
            return False

    def get_health_report(self) -> dict[str, str]:
        """Get health status for all tracked tools. Returns {tool: status_str}."""
        now = time.time()
        report = {}
        with self._lock:
            for tool_name in set(
                list(self._failures.keys()) + list(self._circuit_open_until.keys())
            ):
                open_until = self._circuit_open_until.get(tool_name, 0)
                recent_failures = [t for t in self._failures.get(tool_name, []) if now - t < 600]
                if now < open_until:
                    remaining = int(open_until - now)
                    report[tool_name] = (
                        f"UNAVAILABLE ({remaining}s cooldown, {len(recent_failures)} failures)"
                    )
                elif recent_failures:
                    report[tool_name] = f"degraded ({len(recent_failures)} recent failures)"
                # Only report tools with issues — healthy tools omitted
        return report


# Module-level singleton
tool_health = ToolHealth()
