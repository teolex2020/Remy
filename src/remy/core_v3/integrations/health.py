"""
Health tracking for integration plugins.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .contracts import HealthStatus


@dataclass
class PluginHealth:
    status: HealthStatus = HealthStatus.HEALTHY
    last_error: str = ""
    call_count: int = 0
    error_count: int = 0
    last_latency_ms: int = 0
    updated_at: float = field(default_factory=time.time)


class IntegrationHealthBook:
    def __init__(self):
        self._health: dict[str, PluginHealth] = {}

    def record(self, plugin_id: str, *, ok: bool, latency_ms: int, error: str = "") -> PluginHealth:
        state = self._health.setdefault(plugin_id, PluginHealth())
        state.call_count += 1
        state.last_latency_ms = latency_ms
        state.updated_at = time.time()
        if ok:
            state.status = HealthStatus.HEALTHY if state.error_count < 3 else HealthStatus.DEGRADED
            state.last_error = ""
        else:
            state.error_count += 1
            state.last_error = error
            state.status = HealthStatus.UNAVAILABLE if state.error_count >= 3 else HealthStatus.DEGRADED
        return state

    def get(self, plugin_id: str) -> PluginHealth:
        return self._health.get(plugin_id, PluginHealth())

    def summary(self) -> dict[str, dict]:
        return {
            plugin_id: {
                "status": health.status.value,
                "call_count": health.call_count,
                "error_count": health.error_count,
                "last_latency_ms": health.last_latency_ms,
                "last_error": health.last_error,
            }
            for plugin_id, health in self._health.items()
        }

