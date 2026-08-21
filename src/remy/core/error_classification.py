"""Shared user-facing classification for provider/runtime failures."""

from __future__ import annotations


def classify_llm_error(error_text: str) -> dict[str, object]:
    text = str(error_text or "")
    lowered = text.lower()
    if "quota" in lowered or "429" in lowered or "resource_exhausted" in lowered:
        return {
            "message": "API rate limit reached. Please wait a moment and try again.",
            "retryable": True,
            "error_class": "rate_limit",
        }
    if "provider returned error" in lowered or "overloaded" in lowered or "capacity" in lowered:
        return {
            "message": "The selected model provider is temporarily overloaded. Remy will retry or use a configured fallback model.",
            "retryable": True,
            "error_class": "provider_capacity",
        }
    if "402" in lowered or "insufficient" in lowered or "credits" in lowered or "can only afford" in lowered:
        return {
            "message": "Insufficient provider credits. Add credits or switch to another configured model.",
            "retryable": False,
            "error_class": "billing",
        }
    if "api key" in lowered or "401" in lowered or "403" in lowered or "permission" in lowered:
        return {
            "message": "API authentication error. Check your API key in Settings.",
            "retryable": False,
            "error_class": "auth",
        }
    if any(marker in lowered for marker in (
        "400 bad request",
        "error code: 400",
        "invalid_request_error",
        "not supported",
        "unsupported parameter",
    )):
        return {
            "message": (
                "The selected model rejected this request configuration. "
                "Retry will not help; switch models or update the model adapter."
            ),
            "retryable": False,
            "error_class": "configuration",
        }
    if "timeout" in lowered or "deadline" in lowered:
        return {
            "message": "Request timed out. Try again or simplify your message.",
            "retryable": True,
            "error_class": "timeout",
        }
    if "connect" in lowered or "network" in lowered or "unreachable" in lowered or "getaddrinfo" in lowered:
        return {
            "message": "Network error. Check your internet connection.",
            "retryable": True,
            "error_class": "network",
        }
    if "subscriptable" in lowered:
        return {
            "message": "API response parsing error. Retrying automatically...",
            "retryable": True,
            "error_class": "transient",
        }
    if "nonetype" in lowered and "has no attribute" in lowered:
        return {
            "message": "Remy could not process the model response. Check the server log for the original provider error.",
            "retryable": False,
            "error_class": "internal",
        }
    return {
        "message": "The request failed. Check the server log for the exact provider error.",
        "retryable": False,
        "error_class": "unknown",
    }
