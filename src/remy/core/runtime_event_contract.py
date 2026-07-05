"""Typed runtime/UI event envelope helpers."""

from __future__ import annotations

import time
from typing import Any

SCHEMA_NAME = "remy.runtime.event"
SCHEMA_VERSION = 1


def build_runtime_event(
    event_name: str,
    *,
    event_domain: str,
    payload: dict[str, Any] | None = None,
    level: str | None = None,
    timestamp: float | None = None,
    legacy_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a typed event envelope while preserving legacy top-level fields."""
    event = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "type": event_name,
        "event_name": event_name,
        "event_domain": event_domain,
        "timestamp": float(timestamp if timestamp is not None else time.time()),
        "payload": dict(payload or {}),
    }
    if level is not None:
        event["level"] = level
    if legacy_fields:
        event.update(legacy_fields)
    return event
