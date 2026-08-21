"""Token-aware context compaction for provider-bound model requests.

The legacy agent compactor is intentionally message-count based for backwards
compatibility.  This module owns the runtime policy used when a concrete model
is known: estimate the complete message payload, compare it with that model's
usable context budget, preserve a recent suffix, and expose privacy-safe
metrics for Trajectory.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from remy.config.settings import settings


_PROVIDER_CONTEXT_FALLBACKS = {
    "google": 131_072,
    "anthropic": 131_072,
    "openai": 131_072,
    "xai": 131_072,
    "deepseek": 65_536,
    "openrouter": 65_536,
    "nvidia": 65_536,
    "llamacpp": 32_768,
    "unknown": 32_768,
}

_OVERFLOW_MARKERS = (
    "maximum context length",
    "max context length",
    "context length exceeded",
    "context_length_exceeded",
    "context window exceeded",
    "exceeds the context window",
    "prompt is too long",
    "input is too long",
    "too many input tokens",
    "too many tokens",
    "input token count",
    "reduce the length of the messages",
    "request too large for model",
)


@dataclass(frozen=True, slots=True)
class CompactionPolicy:
    """Resolved limits for one concrete model request."""

    model: str
    provider: str
    context_window_tokens: int
    output_reserve_tokens: int
    threshold_ratio: float
    target_ratio: float
    recent_ratio: float
    max_recent_messages: int
    min_recent_messages: int
    tool_prune_threshold_chars: int
    tool_prune_head_chars: int
    tool_prune_tail_chars: int

    @property
    def usable_tokens(self) -> int:
        return max(1_024, self.context_window_tokens - self.output_reserve_tokens)

    @property
    def threshold_tokens(self) -> int:
        return max(1, int(self.usable_tokens * self.threshold_ratio))

    @property
    def target_tokens(self) -> int:
        return max(1, int(self.usable_tokens * self.target_ratio))


@dataclass(frozen=True, slots=True)
class CompactionResult:
    """Compacted messages plus metadata safe to write to Trajectory."""

    messages: list[BaseMessage]
    compacted: bool
    reason: str
    model: str
    provider: str
    tokens_before: int
    tokens_after: int
    context_window_tokens: int
    threshold_tokens: int
    target_tokens: int
    reserved_input_tokens: int
    messages_before: int
    messages_after: int
    summarized_messages: int
    pruned_tool_results: int
    overflow_retry: bool = False

    def trajectory_entry(self, *, purpose: str = "") -> dict[str, Any]:
        """Return bounded structural telemetry without prompt or result text."""
        payload = asdict(self)
        payload.pop("messages", None)
        payload.update({"type": "compaction", "purpose": str(purpose or "")[:80]})
        return payload


def record_compaction_event(
    result: CompactionResult,
    *,
    session_id: str,
    purpose: str,
) -> None:
    """Best-effort publication of one privacy-safe ``COMPACTED`` event."""
    if not result.compacted or not session_id:
        return
    try:
        from remy.core.trajectory_store import get_trajectory_store

        get_trajectory_store().record_diagnostics(
            session_id=session_id,
            entries=[result.trajectory_entry(purpose=purpose)],
        )
    except Exception:
        # Observability must never make a model request fail.
        return


def estimate_text_tokens(value: Any) -> int:
    """Conservatively estimate tokens without a provider tokenizer dependency."""
    if value is None:
        return 0
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            text = str(value)
    if not text:
        return 0
    # UTF-8 bytes are more conservative than characters for Cyrillic/CJK while
    # remaining close to the common four-English-characters-per-token rule.
    return max(1, math.ceil(len(text.encode("utf-8")) / 3.5))


def estimate_message_tokens(message: BaseMessage) -> int:
    tokens = 5 + estimate_text_tokens(getattr(message, "content", ""))
    if isinstance(message, AIMessage):
        tokens += estimate_text_tokens(getattr(message, "tool_calls", None))
    if isinstance(message, ToolMessage):
        tokens += estimate_text_tokens(getattr(message, "tool_call_id", ""))
        tokens += estimate_text_tokens(getattr(message, "name", ""))
    return tokens


def estimate_messages_tokens(messages: Sequence[BaseMessage]) -> int:
    return 3 + sum(estimate_message_tokens(message) for message in messages)


def is_context_overflow_error(exc: BaseException) -> bool:
    """Recognize provider-specific context-window rejection messages."""
    current: BaseException | None = exc
    visited: set[int] = set()
    chunks: list[str] = []
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        chunks.append(str(current).lower())
        body = getattr(current, "body", None)
        if body:
            chunks.append(str(body).lower())
        current = current.__cause__ or current.__context__
    text = " ".join(chunks)
    return any(marker in text for marker in _OVERFLOW_MARKERS)


def _configured_context_window(model: str, provider: str) -> int | None:
    configured = getattr(settings, "MODEL_CONTEXT_WINDOWS", {}) or {}
    if not isinstance(configured, dict):
        return None
    normalized = str(model or "").strip().lower()
    candidates = (model, normalized, f"provider:{provider}", provider, "default")
    for key in candidates:
        try:
            value = int(configured.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if value >= 1_024:
            return value
    return None


def resolve_compaction_policy(
    model: str,
    *,
    max_recent_messages: int | None = None,
    context_window_tokens: int | None = None,
    target_ratio: float | None = None,
) -> CompactionPolicy:
    """Resolve explicit, configured, learned, then conservative model limits."""
    from remy.core.model_capabilities import get_model_capabilities

    capabilities = get_model_capabilities(model)
    provider = str(capabilities.provider or "unknown").lower()
    window = int(context_window_tokens or 0)
    if window <= 0:
        window = _configured_context_window(model, provider) or 0
    if window <= 0:
        window = int(capabilities.context_window or 0)
    if window <= 0:
        window = int(
            _PROVIDER_CONTEXT_FALLBACKS.get(
                provider,
                getattr(settings, "CONTEXT_COMPACTION_DEFAULT_WINDOW", 32_768),
            )
        )

    threshold_ratio = float(
        getattr(settings, "CONTEXT_COMPACTION_THRESHOLD_RATIO", 0.80)
    )
    resolved_target = float(
        target_ratio
        if target_ratio is not None
        else getattr(settings, "CONTEXT_COMPACTION_TARGET_RATIO", 0.55)
    )
    threshold_ratio = min(0.95, max(0.20, threshold_ratio))
    resolved_target = min(threshold_ratio - 0.05, max(0.10, resolved_target))

    max_recent = int(
        max_recent_messages
        if max_recent_messages is not None
        else getattr(settings, "CONTEXT_COMPACTION_MAX_RECENT_MESSAGES", 32)
    )
    min_recent = int(getattr(settings, "CONTEXT_COMPACTION_MIN_RECENT_MESSAGES", 6))
    max_recent = max(1, max_recent)
    min_recent = max(1, min(max_recent, min_recent))

    reserve = int(getattr(settings, "CONTEXT_COMPACTION_OUTPUT_RESERVE_TOKENS", 8_192))
    reserve = max(256, min(reserve, max(256, window // 2)))

    return CompactionPolicy(
        model=str(model or ""),
        provider=provider,
        context_window_tokens=max(1_024, window),
        output_reserve_tokens=reserve,
        threshold_ratio=threshold_ratio,
        target_ratio=resolved_target,
        recent_ratio=float(getattr(settings, "CONTEXT_COMPACTION_RECENT_RATIO", 0.16)),
        max_recent_messages=max_recent,
        min_recent_messages=min_recent,
        tool_prune_threshold_chars=int(
            getattr(settings, "CONTEXT_TOOL_PRUNE_THRESHOLD_CHARS", 8_192)
        ),
        tool_prune_head_chars=int(getattr(settings, "CONTEXT_TOOL_PRUNE_HEAD_CHARS", 4_096)),
        tool_prune_tail_chars=int(getattr(settings, "CONTEXT_TOOL_PRUNE_TAIL_CHARS", 1_024)),
    )


def _copy_with_content(message: BaseMessage, content: Any) -> BaseMessage:
    if hasattr(message, "model_copy"):
        return message.model_copy(update={"content": content})
    if hasattr(message, "copy"):
        return message.copy(update={"content": content})
    raise TypeError(f"Unsupported message type: {type(message).__name__}")


def _prune_large_tool_results(
    messages: Sequence[BaseMessage], policy: CompactionPolicy
) -> tuple[list[BaseMessage], int]:
    pruned: list[BaseMessage] = []
    count = 0
    threshold = max(512, policy.tool_prune_threshold_chars)
    head_chars = max(128, policy.tool_prune_head_chars)
    tail_chars = max(64, policy.tool_prune_tail_chars)
    for message in messages:
        content = getattr(message, "content", None)
        if (
            isinstance(message, ToolMessage)
            and isinstance(content, str)
            and len(content) > threshold
            and not getattr(message, "artifact", None)
        ):
            omitted = max(0, len(content) - head_chars - tail_chars)
            replacement = (
                content[:head_chars]
                + f"\n...[tool result compacted; {omitted} characters omitted]...\n"
                + content[-tail_chars:]
            )
            pruned.append(_copy_with_content(message, replacement))
            count += 1
        else:
            pruned.append(message)
    return pruned, count


def _safe_recent_start(messages: Sequence[BaseMessage], start: int) -> int:
    """Move a suffix boundary so it cannot begin with an orphan tool result."""
    start = max(0, min(len(messages), start))
    if start >= len(messages) or not isinstance(messages[start], ToolMessage):
        return start
    while start > 0 and isinstance(messages[start], ToolMessage):
        start -= 1
    if isinstance(messages[start], AIMessage) and getattr(messages[start], "tool_calls", None):
        return start
    while start < len(messages) and isinstance(messages[start], ToolMessage):
        start += 1
    return start


def _next_safe_recent_start(messages: Sequence[BaseMessage], start: int) -> int:
    """Drop the oldest item without leaving its tool responses orphaned."""
    next_start = min(len(messages), max(0, start) + 1)
    if (
        start < len(messages)
        and isinstance(messages[start], AIMessage)
        and getattr(messages[start], "tool_calls", None)
    ):
        while next_start < len(messages) and isinstance(messages[next_start], ToolMessage):
            next_start += 1
        return next_start
    while next_start < len(messages) and isinstance(messages[next_start], ToolMessage):
        next_start += 1
    return next_start


def _summary_message(old_messages: Sequence[BaseMessage], budget_tokens: int) -> SystemMessage | None:
    lines: list[str] = []
    for message in old_messages:
        content = getattr(message, "content", "")
        if isinstance(message, HumanMessage):
            text = content if isinstance(content, str) else "[multimodal]"
            lines.append(f"User: {text[:300]}")
        elif isinstance(message, AIMessage) and content:
            text = content if isinstance(content, str) else str(content)
            if text.strip():
                lines.append(f"Remy: {text[:300]}")
    if not lines:
        return None
    prefix = "Earlier in this conversation:\n"
    max_chars = max(128, int(max(32, budget_tokens) * 3.2))
    text = prefix + "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 16)].rstrip() + "\n...[truncated]"
    return SystemMessage(content=text)


def _trim_oversized_recent_message(message: BaseMessage, max_tokens: int) -> BaseMessage:
    content = getattr(message, "content", None)
    if not isinstance(content, str) or estimate_message_tokens(message) <= max_tokens:
        return message
    # Leave room for role/metadata overhead and the explicit omission marker.
    max_chars = max(96, int(max(24, max_tokens - 16) * 2.5))
    if len(content) <= max_chars:
        return message
    head = max_chars * 3 // 4
    tail = max_chars - head
    clipped = (
        content[:head]
        + "\n...[message center compacted to fit model context]...\n"
        + content[-tail:]
    )
    return _copy_with_content(message, clipped)


def compact_messages_for_model(
    messages: Sequence[BaseMessage],
    *,
    model: str,
    max_recent_messages: int | None = None,
    context_window_tokens: int | None = None,
    force: bool = False,
    reason: str = "threshold",
    target_ratio: float | None = None,
    reserved_input_tokens: int = 0,
) -> CompactionResult:
    """Compact a prompt only when its model-specific budget requires it."""
    original = list(messages)
    policy = resolve_compaction_policy(
        model,
        max_recent_messages=max_recent_messages,
        context_window_tokens=context_window_tokens,
        target_ratio=target_ratio,
    )
    tokens_before = estimate_messages_tokens(original)
    reserved_input_tokens = max(0, int(reserved_input_tokens or 0))
    threshold_tokens = max(256, policy.threshold_tokens - reserved_input_tokens)
    target_tokens = max(128, policy.target_tokens - reserved_input_tokens)
    pruned, pruned_tools = _prune_large_tool_results(original, policy)
    pruned_tokens = estimate_messages_tokens(pruned)
    needs_summary = force or pruned_tokens > threshold_tokens

    if not needs_summary:
        compacted = pruned_tools > 0
        return CompactionResult(
            messages=pruned,
            compacted=compacted,
            reason="tool_result_pruning" if compacted else "below_threshold",
            model=policy.model,
            provider=policy.provider,
            tokens_before=tokens_before,
            tokens_after=pruned_tokens,
            context_window_tokens=policy.context_window_tokens,
            threshold_tokens=threshold_tokens,
            target_tokens=target_tokens,
            reserved_input_tokens=reserved_input_tokens,
            messages_before=len(original),
            messages_after=len(pruned),
            summarized_messages=0,
            pruned_tool_results=pruned_tools,
            overflow_retry=reason == "provider_overflow",
        )

    recent_budget = max(
        512,
        min(target_tokens, int(policy.usable_tokens * policy.recent_ratio)),
    )
    start = len(pruned)
    recent_tokens = 3
    while start > 0:
        candidate_tokens = estimate_message_tokens(pruned[start - 1])
        kept_count = len(pruned) - start
        if kept_count >= policy.max_recent_messages:
            break
        if kept_count >= policy.min_recent_messages and recent_tokens + candidate_tokens > recent_budget:
            break
        start -= 1
        recent_tokens += candidate_tokens
    start = _safe_recent_start(pruned, start)
    old = pruned[:start]
    recent = list(pruned[start:])

    # Keep reducing the oldest complete edge until summary + recent fits the
    # target. The newest guaranteed window is never discarded wholesale.
    while old and len(recent) > policy.min_recent_messages:
        summary_budget = max(64, target_tokens - estimate_messages_tokens(recent))
        summary = _summary_message(old, summary_budget)
        candidate = ([summary] if summary else []) + recent
        if estimate_messages_tokens(candidate) <= target_tokens:
            break
        next_start = _next_safe_recent_start(pruned, start)
        start = min(len(pruned), next_start)
        old = pruned[:start]
        recent = list(pruned[start:])

    # A single very large latest message can exceed any budget. Preserve its
    # head and tail and mark the omission explicitly instead of failing again.
    if recent:
        summary_allowance = min(256, max(64, target_tokens // 8)) if old else 0
        per_message_cap = max(
            64,
            (target_tokens - summary_allowance) // max(1, len(recent)),
        )
        recent = [_trim_oversized_recent_message(msg, per_message_cap) for msg in recent]

    summary_budget = max(64, target_tokens - estimate_messages_tokens(recent))
    summary = _summary_message(old, summary_budget)
    result_messages = ([summary] if summary else []) + recent
    if not result_messages and pruned:
        result_messages = [pruned[-1]]

    tokens_after = estimate_messages_tokens(result_messages)
    return CompactionResult(
        messages=result_messages,
        compacted=(result_messages != original),
        reason=str(reason or "threshold"),
        model=policy.model,
        provider=policy.provider,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        context_window_tokens=policy.context_window_tokens,
        threshold_tokens=threshold_tokens,
        target_tokens=target_tokens,
        reserved_input_tokens=reserved_input_tokens,
        messages_before=len(original),
        messages_after=len(result_messages),
        summarized_messages=len(old),
        pruned_tool_results=pruned_tools,
        overflow_retry=reason == "provider_overflow",
    )
