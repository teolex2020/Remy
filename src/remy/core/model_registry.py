"""
Model Registry — per-model API key storage with provider auto-detection.

Stores model → {api_key, provider} mappings in data/model_registry.json.
Falls back to settings.GEMINI_API_KEY / OPENAI_API_KEY for backward compat.
"""

import json
import logging
from threading import Lock

from remy.config.settings import settings

logger = logging.getLogger("ModelRegistry")

_REGISTRY_PATH = settings.DATA_DIR / "model_registry.json"
_lock = Lock()
_cache: dict | None = None

# Models that are auto-populated from global API keys — not manually added by user
_AUTO_MIGRATED_MODELS = frozenset({
    "gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.5-pro",
    "gemini-2.0-flash", "gemini-flash-lite-latest", "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite", "gpt-4o", "gpt-4o-mini", "o3-mini",
})


# ============== PROVIDER DETECTION ==============


def detect_provider(model_name: str) -> str:
    """Auto-detect provider from model name prefix."""
    name = model_name.lower().strip()
    if name.startswith("ollama:"):
        return "ollama"
    if name.startswith(("gpt-", "o1-", "o3-", "o4-", "chatgpt-")):
        return "openai"
    if name.startswith("claude-"):
        return "anthropic"
    if name.startswith("deepseek"):
        return "deepseek"
    if name.startswith("grok-"):
        return "xai"
    if name.startswith(("gemini-", "gemma-")):
        return "google"
    # Models with slash are OpenRouter format: "provider/model-name"
    if "/" in name:
        return "openrouter"
    # Default to google for unknown models
    return "google"


# ============== REGISTRY I/O ==============


def load_registry() -> dict:
    """Load model registry from disk, with auto-migration on first run."""
    global _cache
    with _lock:
        if _cache is not None:
            return _cache
        if _REGISTRY_PATH.exists():
            try:
                data = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
                # Back-fill auto_migrated flag for entries that predate this field
                dirty = False
                for model_name, entry in data.items():
                    if "auto_migrated" not in entry and model_name in _AUTO_MIGRATED_MODELS:
                        entry["auto_migrated"] = True
                        dirty = True
                if dirty:
                    _save_to_disk(data)
                _cache = data
                return _cache
            except Exception as e:
                logger.error("Failed to load model registry: %s", e)

        # First run or corrupted file — migrate from .env
        _cache = _migrate_from_env()
        _save_to_disk(_cache)
        return _cache


def save_registry(data: dict) -> None:
    """Save registry to disk and update cache."""
    global _cache
    with _lock:
        _cache = data
        _save_to_disk(data)


def _save_to_disk(data: dict) -> None:
    """Write registry JSON to disk."""
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REGISTRY_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def invalidate_cache() -> None:
    """Force reload from disk on next access."""
    global _cache
    with _lock:
        _cache = None


def _migrate_from_env() -> dict:
    """Create initial registry from existing .env API keys."""
    registry = {}
    gemini_key = settings.GEMINI_API_KEY
    if gemini_key:
        for model in [
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-flash-lite-latest",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash-lite",
        ]:
            registry[model] = {"api_key": gemini_key, "provider": "google", "auto_migrated": True}
        logger.info("Migrated GEMINI_API_KEY to model registry (%d models)", len(registry))

    openai_key = settings.OPENAI_API_KEY
    if openai_key:
        for model in ["gpt-4o", "gpt-4o-mini", "o3-mini"]:
            registry[model] = {"api_key": openai_key, "provider": "openai", "auto_migrated": True}
        logger.info("Migrated OPENAI_API_KEY to model registry")

    return registry


# ============== API KEY RESOLUTION ==============


def get_api_key_for_model(model_name: str) -> str | None:
    """Get API key for a specific model, with fallback chain.

    Resolution order:
    1. Exact match in registry
    2. Any key from the same provider in registry
    3. Fallback to settings (GEMINI_API_KEY, OPENAI_API_KEY)
    """
    registry = load_registry()

    # 1. Exact match
    entry = registry.get(model_name)
    if entry and entry.get("api_key"):
        return entry["api_key"]

    # 2. Same-provider fallback
    provider = detect_provider(model_name)
    for entry in registry.values():
        if entry.get("provider") == provider and entry.get("api_key"):
            return entry["api_key"]

    # 3. Settings fallback
    if provider == "google" and settings.GEMINI_API_KEY:
        return settings.GEMINI_API_KEY
    if provider == "openai" and settings.OPENAI_API_KEY:
        return settings.OPENAI_API_KEY
    if provider == "openrouter" and settings.OPENROUTER_API_KEY:
        return settings.OPENROUTER_API_KEY

    return None


# ============== CRUD ==============


def register_model(
    model_name: str,
    api_key: str,
    provider: str | None = None,
    input_price: float | None = None,
    output_price: float | None = None,
) -> None:
    """Add or update a model in the registry."""
    registry = load_registry()
    entry = registry.get(model_name, {})
    entry["api_key"] = api_key
    entry["provider"] = provider or detect_provider(model_name)
    if input_price is not None:
        entry["input_price"] = input_price
    if output_price is not None:
        entry["output_price"] = output_price
    registry[model_name] = entry
    save_registry(registry)
    logger.info(
        "Registered model '%s' (provider: %s, in=$%.2f/M, out=$%.2f/M)",
        model_name,
        entry["provider"],
        entry.get("input_price", 0),
        entry.get("output_price", 0),
    )


def unregister_model(model_name: str) -> bool:
    """Remove a model from the registry. Returns True if existed."""
    registry = load_registry()
    if model_name in registry:
        del registry[model_name]
        save_registry(registry)
        logger.info("Unregistered model '%s'", model_name)
        return True
    return False


def _mask_key(key: str) -> str:
    """Mask API key for display: first 6 + ... + last 4."""
    if not key:
        return ""
    if len(key) <= 12:
        return "***"
    return key[:6] + "..." + key[-4:]


def list_registered_models() -> list[dict]:
    """List all registered models with masked keys."""
    registry = load_registry()
    result = []
    for model_name, entry in sorted(registry.items()):
        result.append(
            {
                "name": model_name,
                "provider": entry.get("provider", detect_provider(model_name)),
                "api_key_masked": _mask_key(entry.get("api_key", "")),
                "has_key": bool(entry.get("api_key")),
                "input_price": entry.get("input_price"),
                "output_price": entry.get("output_price"),
                "auto_migrated": entry.get("auto_migrated", False),
            }
        )
    return result


def get_model_pricing(model_name: str) -> tuple[float, float]:
    """Return (input_price, output_price) in $/M tokens for a model. 0.0 if unknown."""
    registry = load_registry()
    entry = registry.get(model_name, {})
    return (entry.get("input_price") or 0.0, entry.get("output_price") or 0.0)


def get_all_providers_with_keys() -> dict[str, str]:
    """Return {provider: api_key} for all providers that have at least one key."""
    registry = load_registry()
    providers = {}
    for entry in registry.values():
        p = entry.get("provider", "")
        if p and entry.get("api_key") and p not in providers:
            providers[p] = entry["api_key"]

    # Fallbacks from settings
    if "google" not in providers and settings.GEMINI_API_KEY:
        providers["google"] = settings.GEMINI_API_KEY
    if "openai" not in providers and settings.OPENAI_API_KEY:
        providers["openai"] = settings.OPENAI_API_KEY
    if "openrouter" not in providers and settings.OPENROUTER_API_KEY:
        providers["openrouter"] = settings.OPENROUTER_API_KEY

    return providers
