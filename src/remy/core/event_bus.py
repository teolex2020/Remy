"""
Lightweight asyncio event bus for real-time thought streaming.

Zero external dependencies. Events are fire-and-forget: if no
subscribers are connected, events are silently dropped.
"""

import asyncio
import logging
import threading
import time

logger = logging.getLogger("EventBus")


class EventBus:
    """Pub/sub event bus backed by asyncio.Queue per subscriber. Thread-safe."""

    def __init__(self, max_queue_size: int = 256):
        self._subscribers: list[asyncio.Queue] = []
        self._max_queue_size = max_queue_size
        self._lock = threading.Lock()

    def subscribe(self) -> asyncio.Queue:
        """Register a new subscriber. Returns a Queue to read events from."""
        q: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue_size)
        with self._lock:
            self._subscribers.append(q)
            count = len(self._subscribers)
        logger.debug("Subscriber added (%d total)", count)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        """Remove a subscriber."""
        with self._lock:
            try:
                self._subscribers.remove(q)
                count = len(self._subscribers)
            except ValueError:
                return
        logger.debug("Subscriber removed (%d remaining)", count)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def emit(self, event_type: str, data: dict | None = None):
        """Emit an event to all subscribers. Non-blocking, drops on full queue."""
        with self._lock:
            subscribers = list(self._subscribers)

        if not subscribers:
            return

        event = {
            "type": event_type,
            "timestamp": time.time(),
            **(data or {}),
        }

        for q in subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass


# Global singleton
event_bus = EventBus()
