"""
Multi-model LLM abstraction with automatic fallback.

Provides get_llm(), call_llm(), and call_llm_async() for all text LLM call sites.
On transient errors (429/500/503/connection), automatically retries with fallback models.
"""

import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from typing import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from remy.config.settings import settings

logger = logging.getLogger("LLM")

_MODEL_ROUTING_OVERRIDE: ContextVar[dict[str, Any] | None] = ContextVar(
    "remy_model_routing_override",
    default=None,
)


# ============== ERROR DETECTION ==============


def _is_transient_error(exc: Exception) -> bool:
    """Determine if an exception is transient and worth retrying with a fallback.

    Catches:
    - google.genai.errors.ServerError (500, 503)
    - ChatGoogleGenerativeAIError wrapping a 429/500/503 ClientError
    - httpx connection/timeout errors
    - Generic connection errors
    """
    # 1. google.genai ServerError (500, 503)
    try:
        from google.genai.errors import ServerError
        if isinstance(exc, ServerError):
            return True
    except ImportError:
        pass

    # 2. ChatGoogleGenerativeAIError wrapping a rate-limit or server error
    try:
        from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
        if isinstance(exc, ChatGoogleGenerativeAIError):
            cause = exc.__cause__
            if cause and hasattr(cause, "code") and cause.code in (429, 500, 503):
                return True
            if "429" in str(exc):
                return True
    except ImportError:
        pass

    # 3. httpx connection/timeout errors
    try:
        import httpx
        if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
            return True
    except ImportError:
        pass

    # 4. Generic connection errors
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True

    # 5. TypeError from langchain-google-genai response parsing bugs
    # e.g. "'Response' object is not subscriptable" — happens sporadically
    # with preview models when the library can't parse an otherwise valid response.
    if isinstance(exc, TypeError) and "subscriptable" in str(exc):
        return True

    return False


# ============== LLM FACTORY ==============


def _get_model_provider(model_name: str) -> str:
    """Infer provider from model name prefix."""
    from remy.core.model_registry import detect_provider
    return detect_provider(model_name)


_THINKING_MODELS = {"gemini-3.1-pro-preview", "gemini-3-pro-preview", "gemini-2.5-pro-preview"}


def get_llm(model_name: str | None = None) -> BaseChatModel:
    """Create a LangChain chat model for the given model name.

    Supports Google Gemini (default) and OpenAI (optional).
    API key is resolved from Model Registry first, then settings fallback.
    Pro/thinking models automatically get thinking_config=HIGH injected.
    """
    from remy.core.model_registry import get_api_key_for_model

    model = model_name or settings.SUMMARY_MODEL
    provider = _get_model_provider(model)
    api_key = get_api_key_for_model(model)

    if provider != "ollama" and not api_key:
        raise ValueError(
            f"No API key found for model '{model}' (provider: {provider}). "
            "Add the model in Settings → Model Registry."
        )

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            try:
                from langchain_community.chat_models import ChatOllama
            except ImportError:
                raise ImportError(
                    "langchain-ollama package required for Ollama models. "
                    "Install with: pip install langchain-ollama"
                )
        model_id = model.removeprefix("ollama:")
        return ChatOllama(model=model_id, base_url=settings.OLLAMA_BASE_URL)

    if provider in ("openai", "openrouter", "deepseek", "xai"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                f"langchain-openai package required for model '{model}'. "
                "Install with: pip install langchain-openai"
            )
        kwargs: dict = {"model": model, "api_key": api_key}
        if provider == "openrouter":
            kwargs["base_url"] = "https://openrouter.ai/api/v1"
        elif provider == "deepseek":
            kwargs["base_url"] = "https://api.deepseek.com"
        elif provider == "xai":
            kwargs["base_url"] = "https://api.x.ai/v1"
        return ChatOpenAI(**kwargs)
    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                f"langchain-anthropic package required for model '{model}'. "
                "Install with: pip install langchain-anthropic"
            )
        return ChatAnthropic(model=model, api_key=api_key)
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs: dict = {"model": model, "api_key": api_key}
        if model in _THINKING_MODELS:
            kwargs["thinking_level"] = "high"
        return ChatGoogleGenerativeAI(**kwargs)


def _get_fallback_chain() -> list[str]:
    """Return the ordered list of fallback model names from settings."""
    models = settings.FALLBACK_MODELS
    if not models:
        return []
    return [m.strip() for m in models if m.strip()]


@contextmanager
def model_routing_override(
    *,
    preferred_model: str = "",
    avoid_models: Sequence[str] | None = None,
):
    """Temporarily reorder the LLM chain for a single execution path.

    The override is intentionally soft: avoided models are moved to the end of
    the chain, not removed, so fallback safety still works if every better model
    fails. A preferred model is tried first when memory has positive evidence.
    """
    token = _MODEL_ROUTING_OVERRIDE.set({
        "preferred_model": (preferred_model or "").strip(),
        "avoid_models": [m.strip() for m in (avoid_models or []) if str(m).strip()],
    })
    try:
        yield
    finally:
        _MODEL_ROUTING_OVERRIDE.reset(token)


def _apply_model_routing(models: list[str]) -> list[str]:
    routing = _MODEL_ROUTING_OVERRIDE.get() or {}
    preferred = str(routing.get("preferred_model") or "").strip()
    avoid = {str(m).strip() for m in routing.get("avoid_models", []) if str(m).strip()}

    ordered: list[str] = []
    if preferred:
        ordered.append(preferred)
        avoid.discard(preferred)

    for model in models:
        if model and model not in ordered and model not in avoid:
            ordered.append(model)
    for model in models:
        if model and model not in ordered and model in avoid:
            ordered.append(model)

    return ordered or models


# ============== COST TRACKING ==============


def _record_cost(result, model: str, purpose: str) -> None:
    """Extract token counts from LLM response and record in CostTracker."""
    try:
        meta = getattr(result, "response_metadata", None) or {}
        usage = meta.get("usage_metadata") or meta.get("token_usage") or {}

        input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0) or usage.get("candidates_tokens", 0) or usage.get("output_tokens", 0)

        if input_tokens or output_tokens:
            from remy.core.cost_tracker import get_cost_tracker
            get_cost_tracker().record(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                purpose=purpose,
            )
    except Exception:
        pass  # Cost tracking is best-effort, never blocks LLM calls


# ============== CORE CALL FUNCTIONS ==============


def call_llm(
    prompt: str | list[BaseMessage],
    *,
    tools: Sequence | None = None,
    purpose: str = "general",
    channel: str | None = None,
) -> "AIMessage":
    """Invoke an LLM with automatic fallback on transient errors.

    Args:
        prompt: Text string or list of LangChain messages.
        tools: If provided, calls llm.bind_tools(tools) before invoke.
        purpose: Logging label (e.g., "agent", "research", "evaluation").

    Returns:
        AIMessage from whichever model succeeds.

    Raises:
        The original exception if ALL models fail.
    """
    primary = settings.SUMMARY_MODEL
    models_to_try = [primary] + _get_fallback_chain()

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_models: list[str] = []
    for m in models_to_try:
        if m not in seen:
            seen.add(m)
            unique_models.append(m)
    unique_models = _apply_model_routing(unique_models)

    last_exception = None
    max_retries_per_model = 3
    retry_delays = [2, 5, 10]  # seconds between retries

    for i, model in enumerate(unique_models):
        for attempt in range(max_retries_per_model):
            try:
                llm = get_llm(model)

                if tools:
                    llm = llm.bind_tools(tools)

                _start = time.time()
                result = llm.invoke(prompt)
                _duration = time.time() - _start

                if i > 0 or attempt > 0:
                    logger.info(
                        "Model '%s' succeeded for [%s] (%.1fs, attempt %d)",
                        model, purpose, _duration, attempt + 1,
                    )
                else:
                    logger.debug(
                        "Primary model '%s' succeeded for [%s] (%.1fs)",
                        model, purpose, _duration,
                    )

                # Attach metadata about which model served the request
                if hasattr(result, "response_metadata") and isinstance(
                    result.response_metadata, dict
                ):
                    result.response_metadata["_served_by"] = model
                    result.response_metadata["_fallback_used"] = i > 0

                # Record cost in real-time tracker
                _record_cost(result, model, purpose)

                return result

            except Exception as e:
                last_exception = e
                if not _is_transient_error(e):
                    logger.error(
                        "Non-transient error from '%s' for [%s]: %s",
                        model, purpose, e,
                    )
                    raise

                # Retry same model with backoff before moving to fallback
                if attempt < max_retries_per_model - 1:
                    delay = retry_delays[attempt]
                    logger.warning(
                        "Transient error from '%s' for [%s] (attempt %d/%d): %s. "
                        "Retrying in %ds...",
                        model, purpose, attempt + 1, max_retries_per_model,
                        e, delay,
                    )
                    time.sleep(delay)
                    continue

                # All retries for this model exhausted — try next model
                remaining = len(unique_models) - i - 1
                if remaining > 0:
                    logger.warning(
                        "Model '%s' failed %d attempts for [%s]: %s. "
                        "Falling back (%d model(s) remaining).",
                        model, max_retries_per_model, purpose, e, remaining,
                    )
                else:
                    logger.error(
                        "All models exhausted for [%s]. Last error from '%s': %s",
                        purpose, model, e,
                    )

    raise last_exception


async def call_llm_async(
    prompt: str | list,
    *,
    tools: Sequence | None = None,
    purpose: str = "general",
    channel: str | None = None,
) -> "AIMessage":
    """Async wrapper: runs call_llm in a thread to preserve async semantics."""
    import asyncio

    return await asyncio.to_thread(
        call_llm, prompt, tools=tools, purpose=purpose, channel=channel,
    )
