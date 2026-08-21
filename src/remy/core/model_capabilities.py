"""Dependency-free model capability profiles for provider-safe requests."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Request and response features supported by one model endpoint."""

    model: str
    provider: str
    native_tool_calling: bool | None = None
    parallel_tool_calls: bool | None = None
    structured_output: bool | None = None
    reasoning_content: bool | None = None
    vision: bool | None = None
    prompt_caching: bool | None = None
    native_streaming: bool | None = True
    system_messages: bool | None = True
    context_window: int | None = None
    local: bool = False
    source: str = "provider-default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_runtime_overrides: dict[str, dict[str, Any]] = {}
_override_lock = threading.RLock()


def _provider_defaults(model: str, provider: str) -> ModelCapabilities:
    normalized = model.lower()
    if provider == "llamacpp":
        # GGUF templates vary. None means "probe once"; an actual template
        # failure records a process-local negative capability afterwards.
        return ModelCapabilities(
            model=model,
            provider=provider,
            native_tool_calling=None,
            parallel_tool_calls=False,
            structured_output=None,
            reasoning_content="deepseek" in normalized,
            vision=any(marker in normalized for marker in ("vision", "vl", "gemma-3n")),
            context_window=None,
            local=True,
        )
    if provider == "google":
        return ModelCapabilities(
            model=model,
            provider=provider,
            native_tool_calling=True,
            parallel_tool_calls=True,
            structured_output=True,
            reasoning_content="thinking" in normalized or "pro" in normalized,
            vision=True,
            prompt_caching=True,
        )
    if provider == "anthropic":
        return ModelCapabilities(
            model=model,
            provider=provider,
            native_tool_calling=True,
            parallel_tool_calls=True,
            structured_output=True,
            reasoning_content=True,
            vision=True,
            prompt_caching=True,
        )
    if provider in {"openai", "openrouter", "deepseek", "xai", "nvidia"}:
        return ModelCapabilities(
            model=model,
            provider=provider,
            native_tool_calling=True,
            parallel_tool_calls=True,
            structured_output=True,
            reasoning_content=provider in {"deepseek", "nvidia"}
            or any(marker in normalized for marker in ("deepseek", "reason", "r1")),
            vision=any(marker in normalized for marker in ("vision", "vl", "gpt-4o", "gpt-5")),
            prompt_caching=provider in {"openai", "openrouter"},
        )
    return ModelCapabilities(model=model, provider=provider)


def get_model_capabilities(
    model: str,
    *,
    provider: str | None = None,
) -> ModelCapabilities:
    """Resolve provider defaults plus learned runtime compatibility facts."""
    clean_model = str(model or "").strip()
    if provider is None:
        from remy.core.model_registry import get_provider_for_model

        provider = get_provider_for_model(clean_model)
    profile = _provider_defaults(clean_model, str(provider or "unknown").lower())
    with _override_lock:
        overrides = dict(_runtime_overrides.get(clean_model, {}))
    if overrides:
        profile = replace(profile, **overrides, source="runtime-observation")
    return profile


def mark_model_capability(model: str, **capabilities: Any) -> ModelCapabilities:
    """Record a capability learned from an actual provider response."""
    allowed = set(ModelCapabilities.__dataclass_fields__) - {
        "model",
        "provider",
        "source",
    }
    clean = {key: value for key, value in capabilities.items() if key in allowed}
    if not clean:
        return get_model_capabilities(model)
    with _override_lock:
        _runtime_overrides.setdefault(str(model), {}).update(clean)
    return get_model_capabilities(model)


def clear_runtime_capability_overrides() -> None:
    """Reset learned observations (primarily for tests and model reloads)."""
    with _override_lock:
        _runtime_overrides.clear()
