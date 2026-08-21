"""ContextReducer A/B measurement for Remy.

This module is intentionally small and product-facing. It does not import Aura
lab code. It builds a compact context slice from existing session history and
compares a raw-context LLM call against a reduced-context LLM call.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import time
from typing import Any, Callable


_WORD_RE = re.compile(r"[\w\-]{3,}", re.UNICODE)
_STOP = {
    "the", "and", "for", "with", "from", "this", "that", "what", "which", "where",
    "when", "how", "why", "your", "you", "are", "was", "were", "have", "has",
    "about", "into", "using", "please", "tell", "show", "user", "assistant",
}


def token_estimate(text: str) -> int:
    return max(1, len(re.findall(r"[\w\-]+", text or "")))


def _provider_prompt_tokens(usage: dict[str, Any]) -> int | None:
    """Read prompt/input token count across provider naming conventions.

    Gemini uses ``prompt_token_count``; OpenAI-style uses ``prompt_tokens`` /
    ``input_tokens``. Returns None when the provider did not report usage.
    """
    for key in ("prompt_token_count", "prompt_tokens", "input_tokens"):
        val = usage.get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
    return None


def _provider_output_tokens(usage: dict[str, Any]) -> int | None:
    for key in ("candidates_token_count", "completion_tokens", "output_tokens"):
        val = usage.get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
    return None


def make_gemini_llm_func(model: str):
    """Build an llm_func that calls a specific Gemini model directly.

    Returns a callable ``(prompt) -> (text, {"usage_metadata": {...}})`` so the
    lab can measure real provider token counts for the *selected* model. Used by
    the LLM Optimization Lab window where the operator picks the model to price.
    """
    from remy.config.settings import settings
    from google import genai

    def _call(prompt: str):
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        resp = client.models.generate_content(model=model, contents=prompt)
        um = getattr(resp, "usage_metadata", None)
        usage: dict[str, Any] = {}
        if um is not None:
            usage = {
                "prompt_token_count": getattr(um, "prompt_token_count", None),
                "candidates_token_count": getattr(um, "candidates_token_count", None),
            }
        return (getattr(resp, "text", "") or "").strip(), {"usage_metadata": usage}

    return _call


def _model_input_price(model: str | None) -> float:
    """Input price in $/1M tokens for ``model`` from the model registry.

    Returns 0.0 when unknown so cost math degrades to zero rather than lying.
    The registry is the single source of truth for pricing; this module never
    hardcodes model prices.
    """
    if not model:
        return 0.0
    try:
        from remy.core.model_registry import get_model_pricing

        input_price, _ = get_model_pricing(model)
        return float(input_price or 0.0)
    except Exception:
        return 0.0


def _terms(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text or "") if t.lower() not in _STOP}


def _message_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    content = getattr(value, "content", "")
    if isinstance(content, str):
        return content
    return str(content or "")


def build_context_lines(session_log: list[dict] | None = None, history: list[Any] | None = None) -> list[str]:
    lines: list[str] = []
    for item in session_log or []:
        kind = str(item.get("type") or "")
        if kind == "user_text":
            text = str(item.get("text") or "").strip()
            if text:
                lines.append(f"user: {text}")
        elif kind == "model_response":
            text = str(item.get("full_text") or item.get("text") or "").strip()
            if text:
                lines.append(f"assistant: {text}")
        elif kind in {"factuality_analysis", "epistemic_governance"}:
            text = str(item)[:500]
            if text:
                lines.append(f"audit: {text}")

    if not lines:
        for msg in history or []:
            text = _message_text(msg).strip()
            if text:
                role = type(msg).__name__.replace("Message", "").lower() or "message"
                lines.append(f"{role}: {text}")
    return lines


def reduce_context_lines(lines: list[str], query: str, *, max_lines: int = 8, keep_recent: int = 2) -> list[str]:
    if not lines:
        return []
    query_terms = _terms(query)
    scored: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        line_terms = _terms(line)
        score = len(query_terms & line_terms)
        if score:
            scored.append((score, index, line))

    selected: dict[int, str] = {}
    for _score, index, line in sorted(scored, key=lambda row: (-row[0], -row[1]))[:max_lines]:
        selected[index] = line
    for index in range(max(0, len(lines) - keep_recent), len(lines)):
        selected[index] = lines[index]

    return [selected[index] for index in sorted(selected)][:max_lines]


def build_prompt(user_text: str, context_lines: list[str], *, mode: str) -> str:
    context = "\n".join(context_lines).strip() or "No prior context."
    return (
        "You are Remy, a careful assistant. Use the context only when it is relevant.\n"
        f"Context mode: {mode}.\n\n"
        "CONTEXT:\n"
        f"{context}\n\n"
        "USER REQUEST:\n"
        f"{user_text}\n\n"
        "ANSWER:"
    )


def _compact_factuality_report(report: Any | None) -> dict[str, Any]:
    if report is None:
        return {
            "checked": False,
            "unsupported_claims_total": 0,
            "unverified_current_claims": 0,
            "brain_storage_unsafe": False,
        }
    return {
        "checked": True,
        "unsupported_claims_total": int(getattr(report, "unsupported_claims_total", 0) or 0),
        "unverified_current_claims": int(getattr(report, "unverified_current_claims", 0) or 0),
        "unsupported_observed_claims": int(getattr(report, "unsupported_observed_claims", 0) or 0),
        "supported_claims_total": int(getattr(report, "supported_claims_total", 0) or 0),
        "supported_internal": int(getattr(report, "supported_internal", 0) or 0),
        "supported_external_verified": int(getattr(report, "supported_external_verified", 0) or 0),
        "unverified_external": int(getattr(report, "unverified_external", 0) or 0),
        "unsupported": int(getattr(report, "unsupported", 0) or 0),
        "brain_storage_unsafe": bool(getattr(report, "brain_storage_unsafe", False)),
        "modified": bool(getattr(report, "modified", False)),
        "evidence_record_ids": list(getattr(report, "evidence_record_ids", []) or [])[:20],
        "external_citations_total": int(getattr(report, "external_citations_total", 0) or 0),
        "external_citations_grounded": int(getattr(report, "external_citations_grounded", 0) or 0),
        "external_citations_phantom": int(getattr(report, "external_citations_phantom", 0) or 0),
        "missing_source_links": bool(getattr(report, "missing_source_links", False)),
    }


def _verify_answer_with_text(
    answer: str,
    session_log: list[dict] | None,
    *,
    session_id: str | None,
) -> tuple[str, dict[str, Any]]:
    try:
        from remy.core.factuality import enforce_factuality

        verified_text, report = enforce_factuality(answer or "", list(session_log or []), session_id=session_id)
        return verified_text, _compact_factuality_report(report)
    except Exception as exc:
        return answer or "", {
            "checked": False,
            "error": str(exc)[:180],
            "unsupported_claims_total": 0,
            "unverified_current_claims": 0,
            "brain_storage_unsafe": False,
            "modified": False,
        }


def _verify_answer(answer: str, session_log: list[dict] | None, *, session_id: str | None) -> dict[str, Any]:
    _verified_text, report = _verify_answer_with_text(answer, session_log, session_id=session_id)
    return report


def _route_recommendation(user_text: str, *, session_log: list[dict] | None = None) -> dict[str, Any]:
    try:
        from langchain_core.messages import HumanMessage
        from remy.core.adaptive_model_router import build_adaptive_model_routing

        routing = build_adaptive_model_routing(
            messages=[HumanMessage(content=user_text)],
            channel="desktop",
            task_type="context_reducer_compare",
            base_routing=None,
        )
        preferred = str(routing.get("preferred_model") or "")
        source = str(routing.get("routing_source") or "none")
        bucket = str(routing.get("complexity_bucket") or "")
        if source == "structural_low_cost_prior":
            decision = "cheap_llm"
        elif source == "structural_high_capability_prior":
            decision = "strong_llm"
        elif preferred:
            decision = "preferred_llm"
        else:
            decision = "default_llm_chain"
        if not user_text.strip() and session_log:
            decision = "memory_only"
        return {
            "enabled": True,
            "decision": decision,
            "preferred_model": preferred,
            "routing_source": source,
            "complexity_bucket": bucket,
            "complexity_score": routing.get("complexity_score", 0),
            "routing_reasons": list(routing.get("routing_reasons", ())),
        }
    except Exception as exc:
        return {"enabled": False, "decision": "default_llm_chain", "error": str(exc)[:180]}


def _write_measurement_event(
    session_log: list[dict] | None,
    *,
    user_text: str,
    raw: dict[str, Any],
    reduced: dict[str, Any],
    delta: dict[str, Any],
    verifier: dict[str, Any],
    router: dict[str, Any],
    report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event = {
        "type": "llm_optimization_measurement",
        "question": user_text[:500],
        "raw_prompt_tokens_estimate": raw.get("prompt_tokens_estimate"),
        "reduced_prompt_tokens_estimate": reduced.get("prompt_tokens_estimate"),
        "tokens_saved_estimate": delta.get("prompt_tokens_saved_estimate"),
        "latency_saved_seconds": delta.get("latency_saved_seconds"),
        "verifier": verifier,
        "router": router,
        "durable_memory_written": False,
        "storage_policy": "session_log_measurement_only",
    }
    if session_log is not None:
        session_log.append(event)
    persisted = False
    answer_memory_written = False
    answer_memory_error = ""
    persist_error = ""
    if report is not None:
        try:
            from remy.core.llm_optimization_metrics import append_answer_memory, append_measurement

            append_measurement(report)
            persisted = True
            append_answer_memory(report)
            answer_memory_written = True
        except Exception as exc:
            persist_error = str(exc)[:180]
            answer_memory_error = persist_error
    return {
        "enabled": True,
        "target": "session_log+measurement_store+answer_memory_store",
        "event_written": session_log is not None,
        "persistent_measurement_written": persisted,
        "answer_memory_written": answer_memory_written,
        "answer_memory_error": answer_memory_error,
        "persistent_error": persist_error,
        "durable_memory_written": False,
        "policy": "measurement metadata plus verifier-safe answer memory; unsafe answer text suppressed",
    }


async def _call_text(
    prompt: str,
    llm_func: Callable[[str], Any] | None,
    *,
    session_id: str | None = None,
    purpose: str = "context_reducer_compare",
    router: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    vault = None
    llm_prompt = prompt
    if llm_func is None:
        from remy.config.settings import settings
        from remy.core.llm import call_llm

        if settings.PII_SHIELD_ENABLED:
            try:
                from remy.core.pii_vault import get_vault, shield

                vault = get_vault(session_id or "__default__")
                llm_prompt = shield(prompt, vault)
            except Exception:
                vault = None
                llm_prompt = prompt
        preferred = str((router or {}).get("preferred_model") or "")
        avoid = tuple((router or {}).get("avoid_models") or ())
        if preferred or avoid:
            from remy.core.llm import model_routing_override

            with model_routing_override(preferred_model=preferred, avoid_models=avoid):
                result = await asyncio.to_thread(call_llm, llm_prompt, purpose=purpose)
        else:
            result = await asyncio.to_thread(call_llm, llm_prompt, purpose=purpose)
    else:
        result = llm_func(llm_prompt)
        if inspect.isawaitable(result):
            result = await result
    # Accept three llm_func shapes: a LangChain-style message object
    # (``.content`` / ``.response_metadata``), a ``(text, metadata)`` tuple, or a
    # bare string. Tuple form lets callers pass provider usage directly.
    tuple_metadata: dict[str, Any] = {}
    if isinstance(result, tuple) and len(result) == 2:
        content = str(result[0] or "")
        if isinstance(result[1], dict):
            tuple_metadata = result[1]
    else:
        content = str(getattr(result, "content", result) or "")
    if vault is not None:
        try:
            from remy.core.pii_vault import restore

            content = restore(content, vault)
        except Exception:
            pass
    metadata = tuple_metadata or getattr(result, "response_metadata", {}) or {}
    return content, dict(metadata) if isinstance(metadata, dict) else {}


async def apply_context_reducer(
    *,
    user_text: str,
    session_log: list[dict] | None = None,
    history: list[Any] | None = None,
    llm_func: Callable[[str], Any] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Answer with a reduced prompt and record the optimization measurement.

    This is not a response cache. The LLM is still called once; only the prompt
    context is reduced before the call. The raw prompt is estimated but skipped.
    """
    lines = build_context_lines(session_log=session_log, history=history)
    raw_lines = lines
    raw_prompt = build_prompt(user_text, raw_lines, mode="raw")
    raw_tokens = token_estimate(raw_prompt)

    try:
        from remy.core.llm_optimization_metrics import find_memory_only_answer

        memory_hit = find_memory_only_answer(user_text)
    except Exception:
        memory_hit = None

    if memory_hit:
        answer = str(memory_hit.get("answer") or "")
        raw = {
            "mode": "raw",
            "llm_call_skipped": True,
            "answer": "",
            "answer_preview": "",
            "context_lines": len(raw_lines),
            "context_tokens_estimate": token_estimate("\n".join(raw_lines)),
            "prompt_tokens_estimate": raw_tokens,
            "elapsed_seconds": 0.0,
            "provider_prompt_tokens": None,
            "provider_output_tokens": None,
        }
        reduced = {
            "mode": "memory_only",
            "llm_call_skipped": True,
            "answer": answer,
            "answer_preview": answer[:600],
            "context_lines": 0,
            "context_tokens_estimate": 0,
            "prompt_tokens_estimate": 0,
            "elapsed_seconds": 0.0,
            "provider_prompt_tokens": 0,
            "provider_output_tokens": 0,
            "memory_source_created_at": memory_hit.get("created_at"),
            "memory_source_schema": memory_hit.get("source_schema"),
            "memory_source_store": memory_hit.get("source_store"),
        }
        delta = {
            "prompt_tokens_saved_estimate": raw_tokens,
            "prompt_token_reduction_ratio": raw_tokens,
            "latency_saved_seconds": 0.0,
            "latency_reduction_ratio": 0.0,
            "context_lines_removed": len(raw_lines),
            "wrong_answers_avoided_estimate": 0,
            "llm_calls_saved": 1,
        }
        verifier = {
            "enabled": True,
            "raw": {"checked": False, "llm_call_skipped": True},
            "reduced": {"checked": False, "memory_only_exact_match": True, "brain_storage_unsafe": False},
            "blocked": False,
            "policy": "exact repeated prompt only; complete verifier-safe prior answer required",
        }
        router = {
            "enabled": True,
            "decision": "memory_only",
            "preferred_model": "",
            "routing_source": "llm_optimization_measurement_store",
            "routing_reasons": ["exact_prompt_memory_hit", "raw_llm_call_skipped"],
        }
        tracker = {
            "enabled": True,
            "tokens_saved_estimate": delta["prompt_tokens_saved_estimate"],
            "latency_saved_seconds": delta["latency_saved_seconds"],
            "wrong_answers_avoided_estimate": 0,
            "llm_calls_saved": 1,
            "provider_prompt_tokens_raw": None,
            "provider_prompt_tokens_reduced": 0,
        }
        report: dict[str, Any] = {
            "schema": "remy_llm_optimization_memory_only_v1",
            "enabled": True,
            "user_text_preview": user_text[:240],
            "answer": answer,
            "raw": raw,
            "reduced": reduced,
            "delta": delta,
            "blocks": {
                "context_reducer": {
                    "enabled": True,
                    "applied_to_answer": False,
                    "memory_only_fast_path": True,
                    "raw_context_lines": len(raw_lines),
                    "reduced_context_lines": 0,
                    "prompt_tokens_saved_estimate": delta["prompt_tokens_saved_estimate"],
                },
                "memory_writer": {},
                "verifier": verifier,
                "model_router": router,
                "cost_latency_tracker": tracker,
            },
            "claims": {
                "kv_cache_internal_replacement": False,
                "active_prompt_reduction": False,
                "apply_context_reducer": False,
                "memory_only_cache_hit": True,
                "raw_llm_call_skipped": True,
                "measured_ab_comparison": False,
                "five_block_pipeline_report": True,
                "durable_fact_memory_written": False,
            },
        }
        memory_writer = _write_measurement_event(
            session_log,
            user_text=user_text,
            raw=raw,
            reduced=reduced,
            delta=delta,
            verifier=verifier,
            router=router,
            report=report,
        )
        report["blocks"]["memory_writer"] = memory_writer
        return {"answer": answer, "report": report}

    reduced_lines = reduce_context_lines(lines, user_text)
    reduced_prompt = build_prompt(user_text, reduced_lines, mode="context_reducer_apply")
    router = _route_recommendation(user_text, session_log=session_log)

    started = time.perf_counter()
    text, metadata = await _call_text(
        reduced_prompt,
        llm_func,
        session_id=session_id,
        purpose="context_reducer_apply",
        router=router,
    )
    elapsed = round(time.perf_counter() - started, 4)
    usage = metadata.get("usage_metadata") or metadata.get("token_usage") or {}
    original_text = text
    text, reduced_verifier = _verify_answer_with_text(text, session_log, session_id=session_id)

    reduced_tokens = token_estimate(reduced_prompt)
    raw = {
        "mode": "raw",
        "llm_call_skipped": True,
        "answer": "",
        "answer_preview": "",
        "context_lines": len(raw_lines),
        "context_tokens_estimate": token_estimate("\n".join(raw_lines)),
        "prompt_tokens_estimate": raw_tokens,
        "elapsed_seconds": 0.0,
        "provider_prompt_tokens": None,
        "provider_output_tokens": None,
    }
    reduced = {
        "mode": "context_reducer_apply",
        "answer": text,
        "answer_preview": text[:600],
        "original_answer_preview": original_text[:600],
        "verifier_modified_answer": bool(reduced_verifier.get("modified", False)),
        "context_lines": len(reduced_lines),
        "context_tokens_estimate": token_estimate("\n".join(reduced_lines)),
        "prompt_tokens_estimate": reduced_tokens,
        "elapsed_seconds": elapsed,
        "provider_prompt_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
        "provider_output_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
    }
    delta = {
        "prompt_tokens_saved_estimate": raw_tokens - reduced_tokens,
        "prompt_token_reduction_ratio": round(raw_tokens / max(1, reduced_tokens), 2),
        "latency_saved_seconds": 0.0,
        "latency_reduction_ratio": 0.0,
        "context_lines_removed": max(0, len(raw_lines) - len(reduced_lines)),
        "wrong_answers_avoided_estimate": 0,
    }
    verifier = {
        "enabled": True,
        "raw": {"checked": False, "llm_call_skipped": True},
        "reduced": reduced_verifier,
        "blocked": bool(reduced_verifier.get("brain_storage_unsafe", False)),
        "policy": "mark unsafe/unverified claims; do not store unsafe facts",
    }
    router["actual_model"] = metadata.get("_served_by") or ""
    router["fallback_used"] = bool(metadata.get("_fallback_used", False))
    router["router_applied_to_llm_call"] = bool(router.get("preferred_model"))
    tracker = {
        "enabled": True,
        "tokens_saved_estimate": delta["prompt_tokens_saved_estimate"],
        "latency_saved_seconds": delta["latency_saved_seconds"],
        "wrong_answers_avoided_estimate": 0,
        "provider_prompt_tokens_raw": None,
        "provider_prompt_tokens_reduced": reduced.get("provider_prompt_tokens"),
    }
    report: dict[str, Any] = {
        "schema": "remy_llm_optimization_apply_v1",
        "enabled": True,
        "user_text_preview": user_text[:240],
        "answer": text,
        "raw": raw,
        "reduced": reduced,
        "delta": delta,
        "blocks": {
            "context_reducer": {
                "enabled": True,
                "applied_to_answer": True,
                "raw_context_lines": len(raw_lines),
                "reduced_context_lines": len(reduced_lines),
                "prompt_tokens_saved_estimate": delta["prompt_tokens_saved_estimate"],
            },
            "memory_writer": {},
            "verifier": verifier,
            "model_router": router,
            "cost_latency_tracker": tracker,
        },
        "claims": {
            "kv_cache_internal_replacement": False,
            "active_prompt_reduction": True,
            "apply_context_reducer": True,
            "model_router_applied": bool(router.get("preferred_model")),
            "verifier_modified_answer": bool(reduced_verifier.get("modified", False)),
            "raw_llm_call_skipped": True,
            "measured_ab_comparison": False,
            "five_block_pipeline_report": True,
            "durable_fact_memory_written": False,
        },
    }
    memory_writer = _write_measurement_event(
        session_log,
        user_text=user_text,
        raw=raw,
        reduced=reduced,
        delta=delta,
        verifier=verifier,
        router=router,
        report=report,
    )
    report["blocks"]["memory_writer"] = memory_writer
    return {"answer": text, "report": report}


async def compare_context_reducer(
    *,
    user_text: str,
    session_log: list[dict] | None = None,
    history: list[Any] | None = None,
    llm_func: Callable[[str], Any] | None = None,
    session_id: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    lines = build_context_lines(session_log=session_log, history=history)
    raw_lines = lines
    reduced_lines = reduce_context_lines(lines, user_text)

    raw_prompt = build_prompt(user_text, raw_lines, mode="raw")
    reduced_prompt = build_prompt(user_text, reduced_lines, mode="context_reducer")

    async def run_one(name: str, prompt: str, context_lines: list[str]) -> dict[str, Any]:
        started = time.perf_counter()
        text, metadata = await _call_text(prompt, llm_func, session_id=session_id)
        elapsed = round(time.perf_counter() - started, 4)
        usage = metadata.get("usage_metadata") or metadata.get("token_usage") or {}
        return {
            "mode": name,
            "answer": text,
            "answer_preview": text[:600],
            "context_lines": len(context_lines),
            "context_tokens_estimate": token_estimate("\n".join(context_lines)),
            "prompt_tokens_estimate": token_estimate(prompt),
            "elapsed_seconds": elapsed,
            "provider_prompt_tokens": _provider_prompt_tokens(usage),
            "provider_output_tokens": _provider_output_tokens(usage),
        }

    raw = await run_one("raw", raw_prompt, raw_lines)
    reduced = await run_one("context_reducer", reduced_prompt, reduced_lines)

    raw_tokens = int(raw["prompt_tokens_estimate"] or 1)
    reduced_tokens = int(reduced["prompt_tokens_estimate"] or 1)
    raw_latency = float(raw["elapsed_seconds"] or 0.0)
    reduced_latency = float(reduced["elapsed_seconds"] or 0.0)

    # Prefer real provider token counts for cost; fall back to estimate only
    # when the provider did not report usage. Cost is computed live from the
    # model registry price for THIS model — never a hardcoded ratio.
    raw_billable = int(raw.get("provider_prompt_tokens") or raw_tokens)
    reduced_billable = int(reduced.get("provider_prompt_tokens") or reduced_tokens)
    tokens_source = "provider" if raw.get("provider_prompt_tokens") else "estimate"
    input_price = _model_input_price(model)  # $/1M input tokens
    cost_raw_usd = raw_billable * input_price / 1_000_000.0
    cost_reduced_usd = reduced_billable * input_price / 1_000_000.0
    cost_saved_usd = max(0.0, cost_raw_usd - cost_reduced_usd)
    raw_verifier = _verify_answer(raw.get("answer", ""), session_log, session_id=session_id)
    reduced_verifier = _verify_answer(reduced.get("answer", ""), session_log, session_id=session_id)
    wrong_answers_avoided = max(
        0,
        int(raw_verifier.get("unsupported_claims_total", 0) or 0)
        - int(reduced_verifier.get("unsupported_claims_total", 0) or 0),
    )
    delta = {
        "prompt_tokens_saved_estimate": raw_tokens - reduced_tokens,
        "prompt_token_reduction_ratio": round(raw_tokens / max(1, reduced_tokens), 2),
        "latency_saved_seconds": round(raw_latency - reduced_latency, 4),
        "latency_reduction_ratio": round(raw_latency / max(0.001, reduced_latency), 2),
        "context_lines_removed": max(0, len(raw_lines) - len(reduced_lines)),
        "wrong_answers_avoided_estimate": wrong_answers_avoided,
        # Live cost math from the model registry — varies by model and request.
        "model": model,
        "input_price_per_1m_usd": round(input_price, 6),
        "billable_tokens_source": tokens_source,
        "billable_prompt_tokens_raw": raw_billable,
        "billable_prompt_tokens_reduced": reduced_billable,
        "prompt_tokens_saved_billable": raw_billable - reduced_billable,
        "prompt_token_reduction_ratio_billable": round(
            raw_billable / max(1, reduced_billable), 2
        ),
        "cost_raw_usd": round(cost_raw_usd, 8),
        "cost_reduced_usd": round(cost_reduced_usd, 8),
        "cost_saved_usd": round(cost_saved_usd, 8),
    }
    verifier = {
        "enabled": True,
        "raw": raw_verifier,
        "reduced": reduced_verifier,
        "blocked": bool(reduced_verifier.get("brain_storage_unsafe", False)),
        "policy": "mark unsafe/unverified claims; do not store unsafe facts",
    }
    router = _route_recommendation(user_text, session_log=session_log)
    tracker = {
        "enabled": True,
        "tokens_saved_estimate": delta["prompt_tokens_saved_estimate"],
        "latency_saved_seconds": delta["latency_saved_seconds"],
        "wrong_answers_avoided_estimate": wrong_answers_avoided,
        "provider_prompt_tokens_raw": raw.get("provider_prompt_tokens"),
        "provider_prompt_tokens_reduced": reduced.get("provider_prompt_tokens"),
        "model": model,
        "input_price_per_1m_usd": delta["input_price_per_1m_usd"],
        "cost_raw_usd": delta["cost_raw_usd"],
        "cost_reduced_usd": delta["cost_reduced_usd"],
        "cost_saved_usd": delta["cost_saved_usd"],
        "billable_tokens_source": tokens_source,
    }
    report: dict[str, Any] = {
        "schema": "remy_llm_optimization_compare_v1",
        "enabled": True,
        "user_text_preview": user_text[:240],
        "raw": raw,
        "reduced": reduced,
        "delta": delta,
        "blocks": {
            "context_reducer": {
                "enabled": True,
                "raw_context_lines": len(raw_lines),
                "reduced_context_lines": len(reduced_lines),
                "prompt_tokens_saved_estimate": delta["prompt_tokens_saved_estimate"],
            },
            "memory_writer": {},
            "verifier": verifier,
            "model_router": router,
            "cost_latency_tracker": tracker,
        },
        "claims": {
            "kv_cache_internal_replacement": False,
            "active_prompt_reduction": True,
            "measured_ab_comparison": True,
            "five_block_pipeline_report": True,
            "durable_fact_memory_written": False,
        },
    }
    memory_writer = _write_measurement_event(
        session_log,
        user_text=user_text,
        raw=raw,
        reduced=reduced,
        delta=delta,
        verifier=verifier,
        router=router,
        report=report,
    )
    report["blocks"]["memory_writer"] = memory_writer
    return report
