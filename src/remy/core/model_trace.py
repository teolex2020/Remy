"""Small helpers for recording which model produced an LLM step."""

from __future__ import annotations

from typing import Any


def _metadata_value(meta: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = meta.get(key)
        if value not in (None, ""):
            return value
    return ""


def extract_token_usage(response: Any) -> dict[str, int] | None:
    """Normalize token usage reported by LangChain model integrations.

    Providers expose usage either on ``AIMessage.usage_metadata`` or inside
    ``response_metadata`` using OpenAI, Gemini, or Anthropic field names.
    Returning ``None`` is intentional: callers must not present an estimate as
    provider-reported usage.
    """
    direct = getattr(response, "usage_metadata", None)
    meta = getattr(response, "response_metadata", None) or {}
    if not isinstance(meta, dict):
        meta = {}
    usage = direct if isinstance(direct, dict) and direct else (
        meta.get("usage_metadata") or meta.get("token_usage") or {}
    )
    if not isinstance(usage, dict) or not usage:
        return None

    def _integer(*keys: str) -> int:
        for key in keys:
            value = usage.get(key)
            if value not in (None, ""):
                try:
                    return max(0, int(value))
                except (TypeError, ValueError):
                    continue
        return 0

    input_tokens = _integer("input_tokens", "prompt_tokens", "prompt_token_count")
    output_tokens = _integer(
        "output_tokens", "completion_tokens", "candidates_tokens", "candidates_token_count"
    )
    total_tokens = _integer("total_tokens", "total_token_count")
    if not total_tokens:
        total_tokens = input_tokens + output_tokens
    if not total_tokens:
        return None
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def model_call_event(
    response: Any,
    *,
    purpose: str,
    channel: str = "",
    duration_ms: int = 0,
) -> dict[str, Any]:
    """Build a compact session-log event from a LangChain model response.

    `remy.core.llm.call_llm` stores provider/fallback metadata on
    `response.response_metadata`. Keeping that metadata in the session log lets
    v3 memory learn which model caused a later success or failure.
    """
    meta = getattr(response, "response_metadata", None) or {}
    if not isinstance(meta, dict):
        meta = {}

    model = str(
        _metadata_value(
            meta,
            "_served_by",
            "served_by",
            "model_name",
            "model",
            "model_id",
        )
        or ""
    )
    fallback_used = bool(
        meta.get("_fallback_used")
        or meta.get("fallback_used")
        or meta.get("fallback")
    )
    tool_calls = getattr(response, "tool_calls", None) or []

    event: dict[str, Any] = {
        "type": "llm_call",
        "purpose": purpose,
        "model": model,
        "fallback_used": fallback_used,
        "tool_calls_requested": len(tool_calls),
    }
    token_usage = extract_token_usage(response)
    if token_usage:
        event["token_usage"] = token_usage
    if channel:
        event["channel"] = channel
    if duration_ms:
        event["duration_ms"] = int(duration_ms)
    return event


def extract_model_runtime(session_log: list[dict] | None) -> tuple[str, bool]:
    """Return the last observed model and whether any fallback occurred."""
    model = ""
    fallback_used = False
    for entry in session_log or []:
        if not isinstance(entry, dict) or entry.get("type") != "llm_call":
            continue
        fallback_used = fallback_used or bool(entry.get("fallback_used"))
        if entry.get("model"):
            model = str(entry.get("model") or "")
    return model, fallback_used
