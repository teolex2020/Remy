"""
Remy Gateway — unified runtime channel registry with health model.

Single place to:
- register/unregister channels (web, telegram, autonomy, browser)
- query live channel status
- emit lifecycle events to the event bus
- restart individual channels independently
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Awaitable

logger = logging.getLogger("Gateway")


class ChannelStatus(str, Enum):
    STARTING  = "starting"
    RUNNING   = "running"
    DEGRADED  = "degraded"
    STOPPED   = "stopped"
    ERROR     = "error"


@dataclass
class ChannelHealth:
    name: str
    status: ChannelStatus = ChannelStatus.STOPPED
    started_at: float | None = None
    stopped_at: float | None = None
    error: str | None = None
    restart_count: int = 0
    last_event: str | None = None

    def uptime_sec(self) -> float | None:
        if self.started_at and self.status == ChannelStatus.RUNNING:
            return time.time() - self.started_at
        return None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "uptime_sec": self.uptime_sec(),
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "error": self.error,
            "restart_count": self.restart_count,
            "last_event": self.last_event,
        }


class ChannelRegistry:
    """
    Central registry for all runtime channels.
    Channels register themselves here; anyone can query health.
    """

    def __init__(self):
        self._channels: dict[str, ChannelHealth] = {}
        self._lock = asyncio.Lock()

    def register(self, name: str) -> ChannelHealth:
        """Register a channel (idempotent)."""
        if name not in self._channels:
            self._channels[name] = ChannelHealth(name=name)
            logger.debug("Channel registered: %s", name)
        return self._channels[name]

    def set_status(self, name: str, status: ChannelStatus, error: str | None = None, event: str | None = None):
        """Update channel status. Creates channel if not registered."""
        ch = self._channels.setdefault(name, ChannelHealth(name=name))
        ch.status = status
        ch.last_event = event or status.value
        if status == ChannelStatus.RUNNING:
            ch.started_at = time.time()
            ch.stopped_at = None
            ch.error = None
        elif status in (ChannelStatus.STOPPED, ChannelStatus.ERROR):
            ch.stopped_at = time.time()
            if error:
                ch.error = error
        if error:
            ch.error = error

        # Emit to event bus
        try:
            from remy.core.event_bus import event_bus
            event_bus.emit("channel_status", {"channel": name, "status": status.value, "error": error})
        except Exception:
            pass

        logger.info("Channel %s -> %s%s", name, status.value, f" ({error})" if error else "")

    def mark_restart(self, name: str):
        ch = self._channels.get(name)
        if ch:
            ch.restart_count += 1

    def get(self, name: str) -> ChannelHealth | None:
        return self._channels.get(name)

    def all(self) -> dict[str, dict]:
        return {name: ch.to_dict() for name, ch in self._channels.items()}

    def summary(self) -> dict:
        """Quick health summary for System dashboard."""
        statuses = {name: ch.status.value for name, ch in self._channels.items()}
        any_error = any(ch.status == ChannelStatus.ERROR for ch in self._channels.values())
        any_degraded = any(ch.status == ChannelStatus.DEGRADED for ch in self._channels.values())
        all_running = all(
            ch.status in (ChannelStatus.RUNNING, ChannelStatus.STOPPED)
            for ch in self._channels.values()
        )
        health = "error" if any_error else "degraded" if any_degraded else "ok"
        return {"health": health, "channels": statuses}


# ============== Global singleton ==============

_registry = ChannelRegistry()


def get_registry() -> ChannelRegistry:
    return _registry


# ============== Convenience wrappers ==============

def channel_starting(name: str):
    _registry.set_status(name, ChannelStatus.STARTING)


def channel_running(name: str):
    _registry.set_status(name, ChannelStatus.RUNNING)


def channel_stopped(name: str, error: str | None = None):
    status = ChannelStatus.ERROR if error else ChannelStatus.STOPPED
    _registry.set_status(name, status, error=error)


def channel_degraded(name: str, reason: str):
    _registry.set_status(name, ChannelStatus.DEGRADED, event=reason)
