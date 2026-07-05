"""Tests for the asyncio event bus."""

import asyncio
import pytest

from remy.core.event_bus import EventBus, event_bus


class TestEventBus:

    def test_emit_no_subscribers(self):
        """Events silently dropped with no subscribers."""
        bus = EventBus()
        bus.emit("test_event", {"key": "value"})

    def test_subscribe_and_emit(self):
        bus = EventBus()
        q = bus.subscribe()
        bus.emit("test_event", {"key": "value"})
        event = q.get_nowait()
        assert event["type"] == "test_event"
        assert event["key"] == "value"
        assert "timestamp" in event

    def test_multiple_subscribers(self):
        bus = EventBus()
        q1 = bus.subscribe()
        q2 = bus.subscribe()
        bus.emit("test", {"n": 1})
        assert q1.get_nowait()["n"] == 1
        assert q2.get_nowait()["n"] == 1

    def test_unsubscribe(self):
        bus = EventBus()
        q = bus.subscribe()
        assert bus.subscriber_count == 1
        bus.unsubscribe(q)
        assert bus.subscriber_count == 0
        bus.emit("test", {})
        assert q.empty()

    def test_unsubscribe_nonexistent(self):
        bus = EventBus()
        q = asyncio.Queue()
        bus.unsubscribe(q)  # Should not raise

    def test_full_queue_drops_oldest(self):
        bus = EventBus(max_queue_size=2)
        q = bus.subscribe()
        bus.emit("e1", {"n": 1})
        bus.emit("e2", {"n": 2})
        bus.emit("e3", {"n": 3})  # Should drop e1
        first = q.get_nowait()
        second = q.get_nowait()
        assert first["n"] == 2
        assert second["n"] == 3

    def test_subscriber_count(self):
        bus = EventBus()
        assert bus.subscriber_count == 0
        q1 = bus.subscribe()
        assert bus.subscriber_count == 1
        q2 = bus.subscribe()
        assert bus.subscriber_count == 2
        bus.unsubscribe(q1)
        assert bus.subscriber_count == 1

    def test_emit_with_no_data(self):
        bus = EventBus()
        q = bus.subscribe()
        bus.emit("ping")
        event = q.get_nowait()
        assert event["type"] == "ping"
        assert "timestamp" in event

    def test_global_singleton_exists(self):
        assert event_bus is not None
        assert hasattr(event_bus, "emit")
        assert hasattr(event_bus, "subscribe")
