"""Small helpers for AuraSDK business-time memory APIs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def parse_memory_timestamp(value: Any, *, field_name: str) -> float:
    """Parse a finite Unix timestamp or ISO-8601 datetime into UTC seconds."""
    if isinstance(value, bool) or value is None or value == "":
        raise ValueError(f"{field_name} is required")

    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} must be a Unix timestamp or ISO-8601 datetime"
            ) from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        timestamp = parsed.timestamp()

    if not float("-inf") < timestamp < float("inf"):
        raise ValueError(f"{field_name} must be finite")
    return timestamp


def temporal_store_kwargs(args: dict[str, Any]) -> dict[str, float]:
    """Extract and validate optional validity boundaries from tool arguments."""
    result: dict[str, float] = {}
    if args.get("valid_from") not in (None, ""):
        result["valid_from"] = parse_memory_timestamp(
            args["valid_from"], field_name="valid_from"
        )
    if args.get("valid_until") not in (None, ""):
        result["valid_until"] = parse_memory_timestamp(
            args["valid_until"], field_name="valid_until"
        )
    if (
        "valid_from" in result
        and "valid_until" in result
        and result["valid_from"] >= result["valid_until"]
    ):
        raise ValueError("valid_from must be earlier than valid_until")
    return result
