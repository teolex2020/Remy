"""
Multi-model LLM abstraction with automatic fallback.

Provides get_llm(), call_llm(), and call_llm_async() for all text LLM call sites.
On transient errors (429/500/503/connection), automatically retries with fallback models.
"""

import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from remy.config.settings import settings

logger = logging.getLogger("LLM")

_MODEL_ROUTING_OVERRIDE: ContextVar[dict[str, Any] | None] = ContextVar(
    "remy_model_routing_override",
    default=None,
)
_LOCAL_MODELS_WITHOUT_NATIVE_TOOLS: set[str] = set()


# ============== ERROR DETECTION ==============

class EmptyModelCompletion(RuntimeError):
    """Provider returned neither visible text nor a tool call."""


def _is_llamacpp_template_compatibility_error(exc: Exception) -> bool:
    """Return whether llama.cpp rejected a model's strict Jinja chat template."""
    lowered = str(exc).lower()
    return any(
        marker in lowered
        for marker in (
            "unable to generate parser for this template",
            "conversation roles must alternate",
            "jinja exception",
        )
    )


def _message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("output_text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _prepare_llamacpp_prompt(
    prompt: str | list[BaseMessage],
    *,
    plain_chat: bool = False,
) -> str | list[BaseMessage]:
    """Adapt Remy's rich history to strict local GGUF chat templates.

    Gemma-family templates accept a single system instruction and enforce
    user/assistant alternation. Native tool histories are flattened only for
    the compatibility retry used when a model cannot parse tool schemas.
    """
    if not isinstance(prompt, list):
        return prompt

    system_parts: list[str] = []
    turns: list[BaseMessage] = []
    for message in prompt:
        text = _message_text(message)
        if isinstance(message, SystemMessage):
            if text:
                system_parts.append(text)
            continue
        if not plain_chat:
            turns.append(message)
            continue
        if isinstance(message, ToolMessage):
            label = getattr(message, "name", None) or "tool"
            turns.append(HumanMessage(content=f"[Tool result: {label}]\n{text}".strip()))
        elif isinstance(message, AIMessage):
            if not text and getattr(message, "tool_calls", None):
                text = "[Tool request completed.]"
            if text:
                turns.append(AIMessage(content=text))
        elif isinstance(message, HumanMessage):
            if text:
                turns.append(HumanMessage(content=text))
        elif text:
            turns.append(HumanMessage(content=text))

    if plain_chat:
        alternating: list[BaseMessage] = []
        for message in turns:
            role = AIMessage if isinstance(message, AIMessage) else HumanMessage
            if alternating and isinstance(alternating[-1], role):
                merged = "\n\n".join(
                    part
                    for part in (
                        _message_text(alternating[-1]),
                        _message_text(message),
                    )
                    if part
                )
                alternating[-1] = role(content=merged)
            else:
                alternating.append(message)
        if alternating and isinstance(alternating[0], AIMessage):
            alternating.insert(0, HumanMessage(content="[Continue the conversation.]"))
        turns = alternating

    prepared: list[BaseMessage] = []
    if system_parts:
        prepared.append(SystemMessage(content="\n\n".join(system_parts)))
    prepared.extend(turns)
    return prepared


def _content_has_visible_text(content: Any) -> bool:
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str) and part.strip():
                return True
            if isinstance(part, dict):
                for key in ("text", "content", "output_text"):
                    value = part.get(key)
                    if isinstance(value, str) and value.strip():
                        return True
            else:
                value = getattr(part, "text", None)
                if isinstance(value, str) and value.strip():
                    return True
    return False


def _response_has_output(result: Any) -> bool:
    if _content_has_visible_text(getattr(result, "content", None)):
        return True
    tool_calls = getattr(result, "tool_calls", None)
    return isinstance(tool_calls, (list, tuple)) and bool(tool_calls)


def _is_transient_error(exc: Exception) -> bool:
    """Determine if an exception is transient and worth retrying with a fallback.

    Catches:
    - google.genai.errors.ServerError (500, 503)
    - ChatGoogleGenerativeAIError wrapping a 429/500/503 ClientError
    - httpx connection/timeout errors
    - Generic connection errors
    """
    if isinstance(exc, EmptyModelCompletion):
        return True
    # Provider SDKs disagree on exception classes but usually expose an HTTP
    # status on the exception or its response. Inspect the cause chain first so
    # NVIDIA/OpenRouter/OpenAI adapters share the same recovery behavior.
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        candidates = [getattr(current, "status_code", None), getattr(current, "code", None)]
        response = getattr(current, "response", None)
        candidates.append(getattr(response, "status_code", None))
        for candidate in candidates:
            try:
                if int(candidate) in (408, 409, 425, 429, 500, 502, 503, 504):
                    return True
            except (TypeError, ValueError):
                pass
        current = current.__cause__ or current.__context__

    lowered = str(exc).lower()
    if any(marker in lowered for marker in (
        "429", "rate limit", "too many requests", "resource_exhausted",
        "provider returned error", "temporarily unavailable", "overloaded", "capacity",
    )):
        return True

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


def _openai_requires_responses_api(model: str) -> bool:
    """Use Responses API for reasoning-family OpenAI models with tools.

    Chat Completions rejects function tools combined with reasoning_effort for
    these models.  ChatOpenAI can preserve the LangChain interface while using
    the supported endpoint underneath.
    """
    normalized = str(model or "").strip().lower()
    return normalized.startswith(("gpt-5", "o1", "o3", "o4"))


def get_llm(model_name: str | None = None) -> BaseChatModel:
    """Create a LangChain chat model for the given model name.

    Supports Google Gemini (default) and OpenAI (optional).
    API key is resolved from Model Registry first, then settings fallback.
    Pro/thinking models automatically get thinking_config=HIGH injected.
    """
    from remy.core.model_registry import get_api_key_for_model, get_provider_for_model

    model = model_name or settings.SUMMARY_MODEL
    provider = get_provider_for_model(model)
    api_key = get_api_key_for_model(model)

    if provider != "llamacpp" and not api_key:
        raise ValueError(
            f"No API key found for model '{model}' (provider: {provider}). "
            "Add the model in Settings → Model Registry."
        )

    if provider == "llamacpp":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("langchain-openai is required for local llama.cpp models")
        from remy.core.llama_cpp_service import llama_cpp_service

        model_id = model.removeprefix("llamacpp:")
        llama_cpp_service.start_model(model_id)
        return ChatOpenAI(
            model=model_id,
            api_key="local-llama-cpp",
            base_url=settings.LLAMA_CPP_BASE_URL,
        )

    if provider == "nvidia":
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
        except ImportError:
            raise ImportError(
                "langchain-nvidia-ai-endpoints is required for NVIDIA NIM models. "
                "Install Remy dependencies again with: pip install -e ."
            )
        return ChatNVIDIA(model=model, api_key=api_key)

    if provider in ("openai", "openrouter", "deepseek", "xai"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                f"langchain-openai package required for model '{model}'. "
                "Install with: pip install langchain-openai"
            )
        kwargs: dict = {"model": model, "api_key": api_key}
        if provider == "openai" and _openai_requires_responses_api(model):
            kwargs["use_responses_api"] = True
            kwargs["output_version"] = "responses/v1"
        elif provider == "openrouter":
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
        output_tokens = (
            usage.get("completion_tokens", 0)
            or usage.get("candidates_tokens", 0)
            or usage.get("output_tokens", 0)
        )

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


def _begin_trajectory_attempt(
    *,
    model: str,
    provider: str,
    model_index: int,
    attempt: int,
    purpose: str,
    tools_enabled: bool,
    compatibility_mode: bool,
) -> str:
    try:
        from remy.core.logging_config import ctx_session_id
        from remy.core.trajectory_store import get_trajectory_store

        session_id = str(ctx_session_id.get() or "")
        if not session_id:
            return ""
        return get_trajectory_store().begin_model_attempt(
            session_id=session_id,
            model=model,
            provider=provider,
            model_index=model_index,
            attempt=attempt,
            purpose=purpose,
            tools_enabled=tools_enabled,
            compatibility_mode=compatibility_mode,
        )
    except Exception as exc:
        logger.debug("Trajectory provider attempt start skipped: %s", exc)
        return ""


def _complete_trajectory_attempt(event_id: str, **changes: Any) -> None:
    if not event_id:
        return
    try:
        from remy.core.trajectory_store import get_trajectory_store

        get_trajectory_store().complete_model_attempt(event_id=event_id, **changes)
    except Exception as exc:
        logger.debug("Trajectory provider attempt completion skipped: %s", exc)


def _record_compaction_trajectory(result: Any, *, purpose: str) -> None:
    """Publish structural compaction metrics to the active request timeline."""
    if not getattr(result, "compacted", False):
        return
    try:
        from remy.core.logging_config import ctx_session_id
        from remy.core.context_compaction import record_compaction_event

        session_id = str(ctx_session_id.get() or "")
        record_compaction_event(
            result,
            session_id=session_id,
            purpose=purpose,
        )
    except Exception as exc:
        logger.debug("Trajectory compaction telemetry skipped: %s", exc)


def _estimate_tool_schema_tokens(tools: Sequence | None) -> int:
    """Estimate provider input occupied by bound tool declarations."""
    if not tools:
        return 0
    from remy.core.context_compaction import estimate_text_tokens

    payload: list[dict[str, Any]] = []
    for tool in tools:
        schema = getattr(tool, "args_schema", None)
        if schema is not None and callable(getattr(schema, "model_json_schema", None)):
            try:
                schema = schema.model_json_schema()
            except Exception:
                schema = str(schema)
        payload.append({
            "name": str(getattr(tool, "name", "") or "")[:240],
            "description": str(getattr(tool, "description", "") or "")[:4_000],
            "schema": schema,
        })
    return estimate_text_tokens(payload)


# ============== CORE CALL FUNCTIONS ==============


def call_llm(
    prompt: str | list[BaseMessage],
    *,
    tools: Sequence | None = None,
    purpose: str = "general",
    channel: str | None = None,
    preferred_model: str | None = None,
    allow_fallback: bool = True,
) -> "AIMessage":
    """Invoke an LLM with automatic fallback on transient errors.

    Args:
        prompt: Text string or list of LangChain messages.
        tools: If provided, calls llm.bind_tools(tools) before invoke.
        purpose: Logging label (e.g., "agent", "research", "evaluation").
        preferred_model: Exact first model for an explicit orchestrator assignment.
        allow_fallback: Whether an explicitly assigned model may use the normal fallback chain.

    Returns:
        AIMessage from whichever model succeeds.

    Raises:
        The original exception if ALL models fail.
    """
    assigned = str(preferred_model or "").strip()
    primary = assigned or settings.SUMMARY_MODEL
    models_to_try = [primary] + (_get_fallback_chain() if (not assigned or allow_fallback) else [])

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_models: list[str] = []
    for m in models_to_try:
        if m not in seen:
            seen.add(m)
            unique_models.append(m)
    # Explicit assignments are already a routing decision. Do not let the
    # adaptive router silently replace or reorder the selected worker model.
    if not assigned:
        unique_models = _apply_model_routing(unique_models)

    last_exception = None
    max_retries_per_model = 3
    retry_delays = [2, 5, 10]  # seconds between retries
    tool_schema_tokens = _estimate_tool_schema_tokens(tools)

    for i, model in enumerate(unique_models):
        from remy.core.model_capabilities import get_model_capabilities

        capabilities = get_model_capabilities(model)
        is_llamacpp = str(model).startswith("llamacpp:")
        model_prompt = prompt
        if (
            isinstance(prompt, list)
            and getattr(settings, "CONTEXT_COMPACTION_ENABLED", True)
        ):
            from remy.core.context_compaction import compact_messages_for_model

            initial_compaction = compact_messages_for_model(
                prompt,
                model=model,
                reserved_input_tokens=tool_schema_tokens,
            )
            model_prompt = initial_compaction.messages
            _record_compaction_trajectory(initial_compaction, purpose=purpose)

        retry_without_tools = bool(tools) and (
            capabilities.native_tool_calling is False
            or (is_llamacpp and model in _LOCAL_MODELS_WITHOUT_NATIVE_TOOLS)
        )
        overflow_retries = 0
        for attempt in range(max_retries_per_model):
            _attempt_started = time.time()
            _attempt_event_id = _begin_trajectory_attempt(
                model=model,
                provider=capabilities.provider,
                model_index=i,
                attempt=attempt + 1,
                purpose=purpose,
                tools_enabled=bool(tools) and not retry_without_tools,
                compatibility_mode=retry_without_tools,
            )
            try:
                llm = get_llm(model)

                if tools and not retry_without_tools:
                    llm = llm.bind_tools(tools)

                _start = time.time()
                effective_prompt = (
                    _prepare_llamacpp_prompt(model_prompt, plain_chat=retry_without_tools)
                    if is_llamacpp
                    else model_prompt
                )
                result = llm.invoke(effective_prompt)
                _duration = time.time() - _start
                if not _response_has_output(result):
                    raise EmptyModelCompletion(
                        f"Model '{model}' returned an empty completion"
                    )

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
                    result.response_metadata["_model_capabilities"] = capabilities.to_dict()

                # Record cost in real-time tracker
                _record_cost(result, model, purpose)

                _complete_trajectory_attempt(
                    _attempt_event_id,
                    success=True,
                    duration_ms=int((time.time() - _attempt_started) * 1000),
                    output={
                        "served_by": model,
                        "has_output": True,
                    },
                )

                return result

            except Exception as e:
                last_exception = e
                from remy.core.context_compaction import is_context_overflow_error

                context_overflow = is_context_overflow_error(e)
                if (
                    isinstance(model_prompt, list)
                    and attempt < max_retries_per_model - 1
                ):
                    from remy.core.context_compaction import (
                        compact_messages_for_model,
                        estimate_messages_tokens,
                    )

                    configured_retries = getattr(
                        settings, "CONTEXT_COMPACTION_OVERFLOW_RETRIES", 1
                    )
                    if not isinstance(configured_retries, int):
                        configured_retries = 1
                    if (
                        overflow_retries < max(0, configured_retries)
                        and context_overflow
                    ):
                        overflow_target = getattr(
                            settings, "CONTEXT_COMPACTION_OVERFLOW_TARGET_RATIO", 0.35
                        )
                        if not isinstance(overflow_target, (int, float)):
                            overflow_target = 0.35
                        overflow_compaction = compact_messages_for_model(
                            model_prompt,
                            model=model,
                            force=True,
                            reason="provider_overflow",
                            target_ratio=float(overflow_target),
                            reserved_input_tokens=tool_schema_tokens,
                        )
                        if overflow_compaction.tokens_after < estimate_messages_tokens(
                            model_prompt
                        ):
                            overflow_retries += 1
                            model_prompt = overflow_compaction.messages
                            _record_compaction_trajectory(
                                overflow_compaction,
                                purpose=purpose,
                            )
                            _complete_trajectory_attempt(
                                _attempt_event_id,
                                success=False,
                                duration_ms=int((time.time() - _attempt_started) * 1000),
                                error=e,
                                retry_action="retry-after-context-compaction",
                            )
                            logger.warning(
                                "Model '%s' exceeded its context for [%s]; compacted "
                                "the prompt from %d to %d estimated tokens and retrying.",
                                model,
                                purpose,
                                overflow_compaction.tokens_before,
                                overflow_compaction.tokens_after,
                            )
                            continue
                if context_overflow:
                    remaining = len(unique_models) - i - 1
                    _complete_trajectory_attempt(
                        _attempt_event_id,
                        success=False,
                        duration_ms=int((time.time() - _attempt_started) * 1000),
                        error=e,
                        retry_action=(
                            "fallback-model-after-context-overflow"
                            if remaining > 0
                            else "context-overflow-exhausted"
                        ),
                    )
                    if remaining > 0:
                        logger.warning(
                            "Model '%s' still exceeded its context for [%s] after "
                            "compaction; trying the next fallback model.",
                            model,
                            purpose,
                        )
                        break
                    raise
                if (
                    is_llamacpp
                    and not retry_without_tools
                    and _is_llamacpp_template_compatibility_error(e)
                ):
                    from remy.core.model_capabilities import mark_model_capability

                    retry_without_tools = True
                    _LOCAL_MODELS_WITHOUT_NATIVE_TOOLS.add(model)
                    capabilities = mark_model_capability(
                        model,
                        native_tool_calling=False,
                        parallel_tool_calls=False,
                    )
                    logger.warning(
                        "Local model '%s' rejected its native tool/chat template for [%s]. "
                        "Retrying in strict alternating chat mode without native tools.",
                        model,
                        purpose,
                    )
                    _complete_trajectory_attempt(
                        _attempt_event_id,
                        success=False,
                        duration_ms=int((time.time() - _attempt_started) * 1000),
                        error=e,
                        retry_action="retry-without-native-tools",
                    )
                    continue
                if not _is_transient_error(e):
                    _complete_trajectory_attempt(
                        _attempt_event_id,
                        success=False,
                        duration_ms=int((time.time() - _attempt_started) * 1000),
                        error=e,
                        retry_action="abort-non-transient",
                    )
                    logger.error(
                        "Non-transient error from '%s' for [%s]: %s",
                        model, purpose, e,
                    )
                    raise

                # Empty completions are usually a provider streaming/adapter
                # anomaly. Retry once, then move to the fallback model instead
                # of making the user wait through the full network backoff.
                attempts_for_error = (
                    2 if isinstance(e, EmptyModelCompletion) else max_retries_per_model
                )

                # Retry same model with backoff before moving to fallback
                if attempt < attempts_for_error - 1:
                    if isinstance(e, EmptyModelCompletion) and tools:
                        # Several OpenAI-compatible endpoints (notably some
                        # NVIDIA NIM models) accept a large tool schema but
                        # return an empty assistant message. The conversation
                        # may already contain retrieved memory/context, so make
                        # the single retry a plain chat completion instead of
                        # repeating the incompatible tool-bound request.
                        retry_without_tools = True
                    delay = 1 if isinstance(e, EmptyModelCompletion) else retry_delays[attempt]
                    _complete_trajectory_attempt(
                        _attempt_event_id,
                        success=False,
                        duration_ms=int((time.time() - _attempt_started) * 1000),
                        error=e,
                        retry_action=(
                            "retry-same-model-without-tools"
                            if retry_without_tools else "retry-same-model"
                        ),
                        retry_delay_seconds=delay,
                    )
                    logger.warning(
                        "Transient error from '%s' for [%s] (attempt %d/%d): %s. "
                        "Retrying in %ds%s...",
                        model, purpose, attempt + 1, max_retries_per_model,
                        e, delay,
                        " without tool binding" if retry_without_tools else "",
                    )
                    time.sleep(delay)
                    continue

                # All retries for this model exhausted — try next model
                remaining = len(unique_models) - i - 1
                _complete_trajectory_attempt(
                    _attempt_event_id,
                    success=False,
                    duration_ms=int((time.time() - _attempt_started) * 1000),
                    error=e,
                    retry_action="fallback-model" if remaining > 0 else "exhausted",
                )
                if remaining > 0:
                    logger.warning(
                        "Model '%s' failed %d attempts for [%s]: %s. "
                        "Falling back (%d model(s) remaining).",
                        model, attempts_for_error, purpose, e, remaining,
                    )
                else:
                    logger.error(
                        "All models exhausted for [%s]. Last error from '%s': %s",
                        purpose, model, e,
                    )
                break

    raise last_exception


async def call_llm_async(
    prompt: str | list,
    *,
    tools: Sequence | None = None,
    purpose: str = "general",
    channel: str | None = None,
    preferred_model: str | None = None,
    allow_fallback: bool = True,
) -> "AIMessage":
    """Async wrapper: runs call_llm in a thread to preserve async semantics."""
    import asyncio

    return await asyncio.to_thread(
        call_llm,
        prompt,
        tools=tools,
        purpose=purpose,
        channel=channel,
        preferred_model=preferred_model,
        allow_fallback=allow_fallback,
    )
