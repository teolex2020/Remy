"""Structural + consequence-aware model routing.

The router deliberately avoids keyword/NLP routing. It uses:
- lived model outcome memory supplied by the caller,
- structural task pressure,
- dynamic model registry + pricing metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from remy.config.settings import settings


@dataclass(frozen=True)
class TaskComplexity:
    bucket: str
    score: int
    reasons: tuple[str, ...]


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return str(content or "")


def _latest_human_text(messages: Sequence[Any]) -> str:
    for message in reversed(list(messages or [])):
        if message.__class__.__name__ == "HumanMessage":
            return _message_text(message)
    for message in reversed(list(messages or [])):
        if message.__class__.__name__ != "SystemMessage":
            text = _message_text(message).strip()
            if text:
                return text
    return ""


def _has_active_tool_context(messages: Sequence[Any]) -> bool:
    for message in messages or []:
        if message.__class__.__name__ == "ToolMessage":
            return True
        if getattr(message, "tool_calls", None):
            return True
    return False


def _add_unique(models: list[str], model: str) -> None:
    model = str(model or "").strip()
    if model and model not in models:
        models.append(model)


def _available_models() -> tuple[str, ...]:
    models: list[str] = []
    _add_unique(models, settings.SUMMARY_MODEL)
    for model in list(settings.FALLBACK_MODELS or []):
        _add_unique(models, model)
    try:
        from remy.core.model_registry import list_registered_models

        for item in list_registered_models():
            name = str(item.get("name") or "").strip()
            provider = str(item.get("provider") or "").strip()
            if bool(item.get("has_key")) or provider == "ollama":
                _add_unique(models, name)
    except Exception:
        pass
    return tuple(models)


def _model_price(model: str) -> float:
    try:
        from remy.core.model_registry import get_model_pricing
        from remy.core.pricing import pricing_registry

        input_price, output_price = get_model_pricing(model)
        if input_price or output_price:
            return float(input_price or 0.0) + float(output_price or 0.0)
        input_price, output_price = pricing_registry.get_price(model)
        return float(input_price or 0.0) + float(output_price or 0.0)
    except Exception:
        return 0.0


def _cheapest_model(models: Sequence[str], avoid: set[str]) -> str:
    candidates = [model for model in models if model and model not in avoid]
    if not candidates:
        return ""
    indexed = list(enumerate(candidates))
    indexed.sort(key=lambda item: (_model_price(item[1]), item[0]))
    return indexed[0][1]


def _strongest_price_prior(models: Sequence[str], avoid: set[str]) -> str:
    candidates = [model for model in models if model and model not in avoid]
    if not candidates:
        return ""
    indexed = list(enumerate(candidates))
    indexed.sort(key=lambda item: (-_model_price(item[1]), item[0]))
    return indexed[0][1]


def estimate_task_complexity(
    messages: Sequence[Any],
    *,
    channel: str = "",
) -> TaskComplexity:
    """Estimate model pressure from task shape, not semantic labels."""
    text = _latest_human_text(messages)
    words = [part for part in text.replace("\n", " ").split(" ") if part.strip()]
    word_count = len(words)
    line_count = len([line for line in text.splitlines() if line.strip()]) or (1 if text.strip() else 0)
    question_count = text.count("?")
    structural_separators = text.count("\n") + text.count(";") + text.count(":")
    has_tool_context = _has_active_tool_context(messages)

    score = 0
    reasons: list[str] = []
    if word_count > 24:
        score += 1
        reasons.append("longer_than_short_turn")
    if word_count > 80:
        score += 2
        reasons.append("large_turn")
    if line_count > 2:
        score += 1
        reasons.append("multi_line")
    if line_count > 6:
        score += 1
        reasons.append("many_lines")
    if question_count > 1:
        score += 1
        reasons.append("multiple_questions")
    if structural_separators > 8:
        score += 1
        reasons.append("many_structural_separators")
    if has_tool_context:
        score += 2
        reasons.append("active_tool_context")
    if channel in {"autonomous", "proactive"}:
        score += 2
        reasons.append("autonomous_channel")

    if score <= 1:
        bucket = "simple"
    elif score >= 4:
        bucket = "demanding"
    else:
        bucket = "normal"
    return TaskComplexity(bucket=bucket, score=score, reasons=tuple(reasons or ("low_pressure",)))


def build_adaptive_model_routing(
    *,
    messages: Sequence[Any],
    channel: str = "",
    task_type: str = "",
    base_routing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge consequence-memory routing with dynamic structural routing.

    Lived consequence memory always wins. Price is only a fallback prior:
    cheapest for low-pressure turns, highest configured price for demanding
    turns when no lived outcome exists yet.
    """
    base = dict(base_routing or {})
    avoid = {str(model or "").strip() for model in base.get("avoid_models", ()) if str(model or "").strip()}
    preferred = str(base.get("preferred_model") or "").strip()
    complexity = estimate_task_complexity(messages, channel=channel)

    result: dict[str, Any] = {
        "preferred_model": preferred,
        "avoid_models": tuple(sorted(avoid)),
        "routing_source": "consequence_memory" if preferred else "none",
        "complexity_bucket": complexity.bucket,
        "complexity_score": complexity.score,
        "routing_reasons": complexity.reasons,
        "task_type": task_type or "",
    }
    if preferred:
        return result
    if not bool(getattr(settings, "MODEL_ROUTER_ENABLED", True)):
        result["routing_source"] = "disabled"
        return result

    available = _available_models()
    selected = ""
    source = "none"
    if complexity.bucket == "simple":
        selected = _cheapest_model(available, avoid)
        source = "structural_low_cost_prior" if selected else "none"
    elif complexity.bucket == "demanding":
        selected = _strongest_price_prior(available, avoid)
        source = "structural_high_capability_prior" if selected else "none"

    result["preferred_model"] = selected
    result["routing_source"] = source
    result["available_models"] = available
    return result
