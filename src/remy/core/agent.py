"""
LangGraph Agent — ReAct pattern StateGraph for text channels (web, telegram).

Uses ChatGoogleGenerativeAI as the LLM, with all brain tools + sandbox tools
bound via LangChain StructuredTool wrappers.

Main entry point specific for text channels (web, telegram).
Voice input on these channels is converted to multimodal messages and passed here.
"""

import json
import logging
import re
import threading
import time
import urllib.parse
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from remy.core.brain_tools import build_system_instruction, CORE_TOOL_NAMES
from remy.core.factuality import enforce_factuality, summarize_claim_details
from remy.core.langgraph_tools import (
    get_all_tools,
    get_tools_by_names,
    set_channel,
    set_session_id,
)
from remy.config.settings import settings

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 30
INSIGHT_CHECK_INTERVAL = 5  # Check brain insights every N messages
_KEEP_RECENT = {
    "autonomous": 20,
    "research": 24,
    "default": 16,
}

# Adaptive recursion limits per channel/task type
# Each graph node invocation = +1, so model→tools→model = 3.
# 35 ≈ 11 tool iterations, 70 ≈ 23, 100 ≈ 33
RECURSION_LIMITS = {
    "quick": 35,       # simple Q&A
    "normal": 70,      # regular conversation
    "research": 150,   # browser, web_search, multi-step research
    "autonomous": 150,  # autonomous mode
}

# Tool names that signal a research/complex task requiring higher recursion limit
_RESEARCH_TOOLS = frozenset({
    "browse_page", "browser_act", "browser_close",
    "start_research", "add_research_finding", "complete_research",
    "delegate_task",
})

# Keywords in user message that signal research intent
_RESEARCH_KEYWORDS = frozenset({
    "досліди", "дослідж", "research", "browse", "знайди", "find",
    "зареєструй", "register", "sign up", "відкрий сайт", "open site",
    "зайди на", "go to", "навігуй", "navigate",
})


_TEMPORAL_KEYWORDS = frozenset({
    "today", "yesterday", "tomorrow", "week", "month", "recent", "latest",
    "сьогодні", "вчора", "завтра", "тиж", "місяц", "остан", "недавно",
})
_FACTUALITY_CONTRACT_TOOLS = frozenset({
    "recall",
    "recall_full",
    "web_search",
    "browse_page",
    "extract_content",
    "http_get",
    "search_web",
})


def _message_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    content = getattr(value, "content", "")
    if isinstance(content, str):
        return content
    return str(content or "")


def _model_routing_situation(channel: str, task_type: str = "") -> str:
    base = f"model-routing:purpose:agent|channel:{channel or 'unknown'}"
    task = (task_type or "").strip()
    if task:
        return f"{base}|task_type:{task}"
    return base


def _store_factuality_consequence(
    *,
    user_message: Any,
    session_id: str,
    channel: str,
    factuality_report,
) -> None:
    """Persist factuality audit failures as lived consequence scars."""
    unsupported = int(getattr(factuality_report, "unsupported_observed_claims", 0) or 0)
    unsupported_total = int(getattr(factuality_report, "unsupported_claims_total", 0) or 0)
    unverified_current = int(getattr(factuality_report, "unverified_current_claims", 0) or 0)
    if max(unsupported, unsupported_total, unverified_current) <= 0:
        return
    try:
        from remy.core_v3.memory.memory_api import get_memory

        situation = _message_text(user_message).strip()[:240] or f"agent_response:{channel}"
        memory = get_memory()
        claim_class_counts: dict[str, int] = {}
        claim_samples: dict[str, str] = {}
        for claim in list(getattr(factuality_report, "claim_details", None) or []):
            if bool(getattr(claim, "supported", False)):
                continue
            claim_class = str(getattr(claim, "claim_class", "") or "unsupported_claim")
            claim_class_counts[claim_class] = claim_class_counts.get(claim_class, 0) + 1
            claim_samples.setdefault(claim_class, _message_text(getattr(claim, "text", ""))[:160])

        memory.capture_consequence(
            situation=situation,
            action="answer_without_evidence",
            consequence="REFUTES",
            trust=-1,
            scope=[
                "factuality-scar",
                f"unsupported_observed_claims:{unsupported}",
                f"unsupported_claims_total:{unsupported_total}",
                f"unverified_current_claims:{unverified_current}",
                *[f"claim_class:{name}:{count}" for name, count in sorted(claim_class_counts.items())],
                f"channel:{channel}" if channel else "channel:",
                f"session:{session_id}" if session_id else "session:",
            ],
            provenance=[
                "remy:core_agent_factuality",
                f"session:{session_id}" if session_id else "session:",
            ],
            links={"session": session_id},
            namespace="remy-factuality",
        )
        for claim_class, count in sorted(claim_class_counts.items()):
            memory.capture_consequence(
                situation=f"factuality-claim-type:{claim_class}|channel:{channel or 'unknown'}",
                action=f"answer_claim_type:{claim_class}:without_evidence",
                consequence="REFUTES",
                trust=-1,
                scope=[
                    "factuality-scar",
                    "claim-type-scar",
                    f"claim_class:{claim_class}",
                    f"unsupported_claims:{count}",
                    f"channel:{channel}" if channel else "channel:",
                    f"session:{session_id}" if session_id else "session:",
                    f"sample:{claim_samples.get(claim_class, '')}",
                ],
                provenance=[
                    "remy:core_agent_factuality_claim_type",
                    f"session:{session_id}" if session_id else "session:",
                ],
                links={"session": session_id, "claim_class": claim_class},
                namespace="remy-factuality",
            )
    except Exception as exc:
        logger.debug("Failed to store factuality consequence: %s", exc)


def _store_model_outcome_consequence(
    *,
    user_message: Any,
    session_id: str,
    channel: str,
    task_type: str = "",
    session_log: list[dict],
    response_text: str,
    factuality_report,
    governance_decision,
) -> None:
    """Persist model outcome memory for the old LangGraph agent path."""
    try:
        from remy.core.model_trace import extract_model_runtime
        from remy.core_v3.memory.memory_api import get_memory

        model, fallback_used = extract_model_runtime(session_log)
        if not model:
            return

        unsupported = int(getattr(factuality_report, "unsupported_observed_claims", 0) or 0)
        governance_mode = str(getattr(governance_decision, "mode", "") or "")
        refuted = unsupported > 0 or governance_mode == "block" or not (response_text or "").strip()
        memory = get_memory()
        memory.capture_consequence(
            situation=_model_routing_situation(channel, task_type),
            action=f"model:{model}",
            consequence="REFUTES" if refuted else "SUPPORTS",
            trust=-1 if refuted else 1,
            scope=[
                "model-outcome",
                f"model:{model}",
                f"task_type:{task_type}" if task_type else "task_type:",
                f"fallback:{str(fallback_used).lower()}",
                f"channel:{channel}" if channel else "channel:",
                f"session:{session_id}" if session_id else "session:",
                f"unsupported_observed_claims:{unsupported}",
                f"governance:{governance_mode or 'none'}",
                "status:refuted" if refuted else "status:supported",
                f"user:{_message_text(user_message).strip()[:120]}",
            ],
            provenance=[
                "remy:core_agent_model_outcome",
                f"session:{session_id}" if session_id else "session:",
            ],
            links={"session": session_id or "", "model": model},
            namespace="remy-models",
        )
    except Exception as exc:
        logger.debug("Failed to store model outcome consequence: %s", exc)


def _model_routing_from_consequence_memory(channel: str, task_type: str = "") -> dict[str, Any]:
    """Build a soft model-routing override from lived model outcomes."""
    try:
        from remy.config.settings import settings
        from remy.core.consequence_gate import consult_policy_hint
        from remy.core.model_registry import list_registered_models
        from remy.core_v3.memory.memory_api import get_memory

        models: list[str] = []
        for model in [settings.SUMMARY_MODEL, *list(settings.FALLBACK_MODELS or [])]:
            model = str(model or "").strip()
            if model and model not in models:
                models.append(model)
        for item in list_registered_models():
            model = str(item.get("name") or "").strip()
            provider = str(item.get("provider") or "").strip()
            if model and model not in models and (bool(item.get("has_key")) or provider == "ollama"):
                models.append(model)
        if not models:
            return {}

        memory = get_memory()
        situations = [_model_routing_situation(channel, task_type)]
        fallback_situation = _model_routing_situation(channel)
        if fallback_situation not in situations:
            situations.append(fallback_situation)
        avoid: list[str] = []
        preferred = ""
        preferred_score = 0
        for model in models:
            context = {}
            for situation in situations:
                hint = consult_policy_hint(
                    memory,
                    situation=situation,
                    action=f"model:{model}",
                    namespace="remy-models",
                )
                context = hint.to_context() if hasattr(hint, "to_context") else dict(hint or {})
                if int(context.get("supports", 0) or 0) or int(context.get("refutes", 0) or 0):
                    break
            supports = int(context.get("supports", 0) or 0)
            refutes = int(context.get("refutes", 0) or 0)
            policy = str(context.get("hint") or "")
            if policy == "avoid" and refutes > 0:
                avoid.append(model)
                continue
            if policy == "prefer" and supports > 0:
                score = supports - refutes
                if not preferred or score > preferred_score:
                    preferred = model
                    preferred_score = score
        return {
            "preferred_model": preferred,
            "avoid_models": tuple(avoid),
        }
    except Exception as exc:
        logger.debug("Model consequence routing skipped: %s", exc)
        return {}



_FETCH_EVIDENCE_TOOLS = frozenset({
    "browse_page",
    "browser_act",
    "http_get",
    "extract_content",
    "fetch_url",
})
_TRUSTED_SOURCE_EXACT_HOSTS = {
    "arxiv.org",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "docs.python.org",
    "developer.mozilla.org",
    "w3.org",
    "github.com",
    "docs.github.com",
    "learn.microsoft.com",
}
_TRUSTED_SOURCE_HOST_HINTS = (
    ".gov", ".edu", "docs.", "developer.", "arxiv", "pubmed", "ncbi",
    "nature", "science.org", "springer", "acm", "ieee", "nih.gov",
)
_WEAK_SOURCE_HOST_HINTS = (
    "mirror", "blog", "blogspot", "wordpress", "medium.com", "substack",
    "reddit", "quora", "pinterest", "researchgate.net", "vertexaisearch.cloud.google.com",
)
_WEAK_SOURCE_TITLE_HINTS = ("mirror", "cached", "sponsored", "affiliate", "seo")


def _domain_root(host: str) -> str:
    host = (host or "").lower().strip(".")
    parts = [part for part in host.split(".") if part]
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _extract_query_site_constraints(query: str) -> set[str]:
    query_lower = (query or "").lower()
    constraints: set[str] = set()
    for raw in re.findall(r"site:([^\s]+)", query_lower):
        host = raw.strip().strip("\"'").strip()
        if host:
            constraints.add(host.removeprefix("www."))
    return constraints


def _source_matches_query_constraints(source: dict, query: str) -> bool:
    constraints = _extract_query_site_constraints(query)
    if not constraints:
        return True
    uri = str((source or {}).get("uri") or "").strip()
    host = urllib.parse.urlsplit(uri).netloc.lower().removeprefix("www.")
    root = _domain_root(host)
    for constraint in constraints:
        c = constraint.lower().removeprefix("www.")
        if host == c or host.endswith(f".{c}") or root == c:
            return True
    return False


def _source_host(source: dict) -> str:
    uri = str((source or {}).get("uri") or (source or {}).get("url") or "").strip()
    return urllib.parse.urlsplit(uri).netloc.lower().removeprefix("www.")


def _filter_candidate_sources_for_query(sources: list[dict], query: str) -> list[dict]:
    constraints = _extract_query_site_constraints(query)
    if not constraints:
        return list(sources or [])
    return [dict(source) for source in (sources or []) if _source_matches_query_constraints(source, query)]


def _source_class_for_memory(source: dict) -> str:
    try:
        from remy.core.retrieval.source_filter import classify

        return str(classify(source or {}).source_class or "unknown")
    except Exception:
        return "unknown"


def _source_memory_bias(source: dict) -> tuple[int, str]:
    """Return a small score adjustment from lived source consequences."""
    try:
        from remy.core.consequence_gate import consult_policy_hint
        from remy.core_v3.memory.memory_api import get_memory

        memory = get_memory()
        source_class = _source_class_for_memory(source)
        host = _source_host(source)
        bias = 0
        reasons: list[str] = []
        for action in (
            f"source_class:{source_class}",
            f"source_host:{host}" if host else "",
        ):
            if not action:
                continue
            hint = consult_policy_hint(
                memory,
                situation="source-selection:global",
                action=action,
                namespace="remy-sources",
            )
            context = hint.to_context() if hasattr(hint, "to_context") else dict(hint or {})
            supports = int(context.get("supports", 0) or 0)
            refutes = int(context.get("refutes", 0) or 0)
            policy = str(context.get("hint") or "")
            if policy == "avoid" and refutes > 0:
                bias -= min(80, 30 + refutes * 10)
                reasons.append(f"memory_avoid:{action}")
            elif policy == "prefer" and supports > 0:
                bias += min(50, 15 + supports * 5)
                reasons.append(f"memory_prefer:{action}")
        return bias, "+".join(reasons)
    except Exception as exc:
        logger.debug("Source memory bias skipped: %s", exc)
        return 0, ""


def _score_candidate_source(source: dict) -> tuple[int, str]:
    uri = str((source or {}).get("uri") or "").strip()
    title = str((source or {}).get("title") or "").strip().lower()
    host = urllib.parse.urlsplit(uri).netloc.lower()
    root = _domain_root(host)
    score = 0
    reason = "fallback"
    if host in _TRUSTED_SOURCE_EXACT_HOSTS or root in _TRUSTED_SOURCE_EXACT_HOSTS:
        score += 110
        reason = "trusted-domain"
    elif host.endswith((".gov", ".edu")):
        score += 100
        reason = "official-domain"
    elif any(hint in host for hint in _TRUSTED_SOURCE_HOST_HINTS):
        score += 80
        reason = "trusted-host-hint"
    elif any(tok in title for tok in ("official", "documentation", "docs", "journal", "paper", "research")):
        score += 30
        reason = "trusted-title-hint"
    else:
        score += 10
    if any(hint in host for hint in _WEAK_SOURCE_HOST_HINTS):
        score -= 45
        reason = "weak-host"
    if any(hint in title for hint in _WEAK_SOURCE_TITLE_HINTS):
        score -= 25
        reason = "weak-title"
    if not host:
        score -= 20
        reason = "missing-host"
    return score, reason


def _choose_best_candidate_source(sources: list[dict], query: str = "") -> dict | None:
    best_source = None
    best_score = None
    best_reason = "fallback"
    candidate_sources = _filter_candidate_sources_for_query(sources, query)
    for source in candidate_sources or []:
        score, reason = _score_candidate_source(source)
        memory_bias, memory_reason = _source_memory_bias(source)
        score += memory_bias
        if memory_reason:
            reason = f"{reason}+{memory_reason}"
        if best_score is None or score > best_score:
            best_source = dict(source)
            best_score = score
            best_reason = reason
    if best_source is None:
        return None
    best_source["trust_score"] = int(best_score or 0)
    best_source["trust_reason"] = best_reason
    return best_source


def _record_fetch_evidence(session_id: str | None, tool_name: str, raw_result, *, explicit_url: str = "") -> None:
    if not session_id or tool_name not in _FETCH_EVIDENCE_TOOLS:
        return
    try:
        from remy.core.claim_provenance import record_turn_fetch_evidence
        payload = raw_result
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {"raw": payload}
        if not isinstance(payload, dict):
            payload = {"raw": str(payload)}
        url = str(payload.get("url") or explicit_url or "").strip()
        title = str(payload.get("title") or "").strip()
        site = str(payload.get("site") or payload.get("source_name") or "").strip()
        if url:
            record_turn_fetch_evidence(session_id, tool=tool_name, url=url, title=title, site=site)
    except Exception as e:
        logger.debug("Failed to record fetch evidence for %s: %s", tool_name, e)


def _store_source_grounding_consequences(
    *,
    session_id: str,
    channel: str,
    factuality_report,
    governance_decision,
) -> None:
    """Persist which fetched source classes/hosts helped or failed grounding."""
    try:
        from remy.core.claim_provenance import get_turn_fetch_evidence
        from remy.core_v3.memory.memory_api import get_memory

        fetches = get_turn_fetch_evidence(session_id)
        if not fetches or factuality_report is None:
            return

        governance_mode = str(getattr(governance_decision, "mode", "") or "")
        unsupported_total = int(getattr(factuality_report, "unsupported_claims_total", 0) or 0)
        unsupported_observed = int(getattr(factuality_report, "unsupported_observed_claims", 0) or 0)
        unverified_current = int(getattr(factuality_report, "unverified_current_claims", 0) or 0)
        unverified_external = int(getattr(factuality_report, "unverified_external", 0) or 0)
        phantom = int(getattr(factuality_report, "external_citations_phantom", 0) or 0)
        unsafe = bool(getattr(factuality_report, "brain_storage_unsafe", False)) or governance_mode == "block"

        refuted = unsafe or phantom > 0 or unverified_external > 0
        supported = (
            bool(getattr(factuality_report, "had_external_evidence", False))
            and unsupported_total == 0
            and unsupported_observed == 0
            and unverified_current == 0
            and not refuted
        )
        if not supported and not refuted:
            return

        memory = get_memory()
        consequence = "REFUTES" if refuted else "SUPPORTS"
        trust = -1 if refuted else 1
        seen_actions: set[str] = set()
        for item in fetches:
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            source = {"uri": url, "title": str(item.get("title") or "")}
            source_class = _source_class_for_memory(source)
            host = _source_host(source)
            for action in (
                f"source_class:{source_class}",
                f"source_host:{host}" if host else "",
                f"source_tool:{str(item.get('tool') or 'unknown')}",
            ):
                if not action or action in seen_actions:
                    continue
                seen_actions.add(action)
                memory.capture_consequence(
                    situation="source-selection:global",
                    action=action,
                    consequence=consequence,
                    trust=trust,
                    scope=[
                        "source-grounding",
                        f"source_class:{source_class}",
                        f"host:{host}" if host else "host:",
                        f"tool:{str(item.get('tool') or '')}",
                        f"channel:{channel}" if channel else "channel:",
                        f"session:{session_id}" if session_id else "session:",
                        f"unsupported_claims_total:{unsupported_total}",
                        f"unverified_current_claims:{unverified_current}",
                        f"unverified_external:{unverified_external}",
                        f"phantom_citations:{phantom}",
                        "status:refuted" if refuted else "status:supported",
                    ],
                    provenance=[
                        "remy:core_agent_source_grounding",
                        f"session:{session_id}" if session_id else "session:",
                    ],
                    links={"session": session_id or "", "url": url, "host": host},
                    namespace="remy-sources",
                )
    except Exception as exc:
        logger.debug("Failed to store source grounding consequence: %s", exc)


def _tool_gate_situation(messages: list, *, session_id: str | None, channel: str) -> str:
    for msg in reversed(messages or []):
        if isinstance(msg, HumanMessage):
            text = _message_text(msg).strip()
            if text:
                return text[:240]
    return f"tool_call:{channel or 'unknown'}:{session_id or 'unknown'}"


def _tool_gate_action(tool_name: str, tool_args: dict | None) -> str:
    compact_args = {
        str(key): str(value)[:160]
        for key, value in sorted((tool_args or {}).items(), key=lambda item: str(item[0]))
    }
    if not compact_args:
        return f"tool:{tool_name}"
    return f"tool:{tool_name}:{json.dumps(compact_args, ensure_ascii=False, sort_keys=True)}"


def _blocked_tool_policy_hint(
    *,
    messages: list,
    session_id: str | None,
    channel: str,
    tool_name: str,
    tool_args: dict | None,
) -> dict | None:
    """Return an avoid policy if this exact tool action was refuted before."""
    try:
        from remy.core.consequence_gate import consult_policy_hint
        from remy.core_v3.memory.memory_api import get_memory

        situation = _tool_gate_situation(messages, session_id=session_id, channel=channel)
        action = _tool_gate_action(tool_name, tool_args)
        hint = consult_policy_hint(
            get_memory(),
            situation=situation,
            action=action,
            namespace="remy-tools",
        )
        context = hint.to_context() if hasattr(hint, "to_context") else dict(hint or {})
        if context.get("hint") == "avoid" or context.get("should_block"):
            return context
    except Exception as exc:
        logger.debug("Tool consequence gate skipped for %s: %s", tool_name, exc)
    return None


def _tool_result_refuted(result: Any) -> bool:
    text = str(result or "")
    if text.startswith(("Error:", "Unknown tool:", "Blocked by consequence memory:")):
        return True
    try:
        parsed = json.loads(text) if isinstance(result, str) else result
        if isinstance(parsed, dict) and parsed.get("error"):
            return True
    except Exception:
        pass
    return False


def _store_tool_consequence(
    *,
    messages: list,
    session_id: str | None,
    channel: str,
    tool_name: str,
    tool_args: dict | None,
    result: Any,
) -> None:
    """Persist tool call outcomes so future calls can be gated by scars."""
    try:
        from remy.core_v3.memory.memory_api import get_memory

        refuted = _tool_result_refuted(result)
        memory = get_memory()
        memory.capture_consequence(
            situation=_tool_gate_situation(messages, session_id=session_id, channel=channel),
            action=_tool_gate_action(tool_name, tool_args),
            consequence="REFUTES" if refuted else "SUPPORTS",
            trust=-1 if refuted else 1,
            scope=[
                "tool-call",
                f"tool:{tool_name}",
                f"channel:{channel}" if channel else "channel:",
                f"session:{session_id}" if session_id else "session:",
                "result:error" if refuted else "result:ok",
            ],
            provenance=[
                "remy:core_agent_tool_call",
                f"session:{session_id}" if session_id else "session:",
            ],
            links={"session": session_id or ""},
            namespace="remy-tools",
        )
    except Exception as exc:
        logger.debug("Failed to store tool consequence for %s: %s", tool_name, exc)


def _extract_total_usage_tokens(result) -> int:
    """Best-effort token extraction across provider-specific metadata shapes."""
    meta = getattr(result, "response_metadata", None) or {}
    usage = meta.get("usage_metadata") or meta.get("token_usage") or {}

    total_tokens = usage.get("total_tokens", 0)
    if total_tokens:
        return int(total_tokens)

    input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
    output_tokens = (
        usage.get("completion_tokens", 0)
        or usage.get("candidates_tokens", 0)
        or usage.get("output_tokens", 0)
    )
    return int(input_tokens or 0) + int(output_tokens or 0)


def _estimate_recursion_limit(channel: str, user_message: str | HumanMessage) -> int:
    """Choose recursion limit based on channel and message content."""
    return RECURSION_LIMITS[_estimate_task_type(channel, user_message)]


def _extract_text(user_message: str | HumanMessage) -> str:
    if isinstance(user_message, str):
        return user_message
    return user_message.content if isinstance(user_message.content, str) else ""


def _latest_human_text(messages: list) -> str:
    for msg in reversed(messages or []):
        if isinstance(msg, HumanMessage):
            return _extract_text(msg)
    return ""


def _estimate_task_type(channel: str, user_message: str | HumanMessage) -> str:
    if channel == "autonomous":
        return "autonomous"
    text = _extract_text(user_message)
    text_lower = text.lower()
    for kw in _RESEARCH_KEYWORDS:
        if kw in text_lower:
            return "research"
    if len(text.split()) < 8:
        return "quick"
    return "normal"


def _detect_turn_locale(user_message: str | HumanMessage) -> str:
    # Interim heuristic for Stabilization A4 DS-2 fix.
    # Phase A.7 Brain-Native Mouth will replace this with structured events
    # rendered by the SLM in turn language; brain itself stays language-agnostic.
    text = _extract_text(user_message)
    for ch in text:
        if "\u0400" <= ch <= "\u04ff":
            return "ua"
    return "en"


def _needs_factuality_contract(messages: list, session_log: list) -> bool:
    user_text = _latest_human_text(messages)
    text_lower = user_text.lower()
    if any(kw in text_lower for kw in _TEMPORAL_KEYWORDS):
        return True
    if any(kw in text_lower for kw in _RESEARCH_KEYWORDS):
        return True
    if any(token in text_lower for token in ("market", "pricing", "price", "benchmark", "accuracy", "latest", "current", "research")):
        return True
    if re.search(r"(?:\$|€|£)\s?\d|\b\d+\s?%|\b\d{4}\b", user_text):
        return True

    recent_tools = [
        entry.get("tool", "")
        for entry in (session_log or [])[-8:]
        if isinstance(entry, dict) and entry.get("type") == "tool_call"
    ]
    return any(tool in _FACTUALITY_CONTRACT_TOOLS for tool in recent_tools)


def _factuality_claim_type_policy_lines(channel: str) -> list[str]:
    """Read lived factuality scars and turn them into pre-answer policy lines."""
    try:
        from remy.core.consequence_gate import consult_policy_hint
        from remy.core_v3.memory.memory_api import get_memory

        memory = get_memory()
        lines: list[str] = []
        for claim_class in (
            "observed_fact",
            "unverified_current_fact",
            "memory_fact",
            "inference",
        ):
            hint = consult_policy_hint(
                memory,
                situation=f"factuality-claim-type:{claim_class}|channel:{channel or 'unknown'}",
                action=f"answer_claim_type:{claim_class}:without_evidence",
                namespace="remy-factuality",
            )
            context = hint.to_context() if hasattr(hint, "to_context") else dict(hint or {})
            refutes = int(context.get("refutes", 0) or 0)
            policy = str(context.get("hint") or "")
            if refutes <= 0 and policy not in {"avoid", "requires_evidence"}:
                continue
            lines.append(
                f"- Prior consequence memory refuted unsupported {claim_class}; "
                "require fresh evidence before stating this claim type as fact."
            )
        return lines
    except Exception as exc:
        logger.debug("Factuality claim-type policy skipped: %s", exc)
        return []


def _build_factuality_contract_message(state: "AgentState") -> SystemMessage | None:
    if not _needs_factuality_contract(state.get("messages", []), state.get("session_log", [])):
        return None

    claim_policy_lines = _factuality_claim_type_policy_lines(str(state.get("channel") or ""))
    policy_block = ""
    if claim_policy_lines:
        policy_block = (
            "\nLived factuality memory for this channel:\n"
            + "\n".join(claim_policy_lines)
            + "\n"
        )

    return SystemMessage(
        content=(
            "Factuality contract for this turn:\n"
            "- Do not present unsupported claims as facts.\n"
            "- If a claim depends on recalled memory, prefer statements backed by recalled records from this turn.\n"
            "- If a claim is inferred, label it as an inference.\n"
            "- If a claim is current, market-related, numeric, or research-heavy and not strongly verified, put it in needs_verification.\n"
            "- If the answer mixes supported and unsupported claims, format it as:\n"
            "  Facts:\n"
            "  Inferences:\n"
            "  Unknowns:\n"
            "  Needs verification:\n"
            "- If little is verified, be brief and say that verification is still needed."
            f"{policy_block}"
        )
    )


def _estimate_keep_recent(channel: str, user_message: str | HumanMessage) -> int:
    """Choose compact_history budget based on channel and task complexity."""
    text = _extract_text(user_message)
    text_lower = text.lower()

    if channel == "autonomous":
        try:
            from remy.core.context_window import dynamic_keep_recent

            return dynamic_keep_recent(channel, text)
        except Exception:
            return _KEEP_RECENT["autonomous"]

    if any(kw in text_lower for kw in _RESEARCH_KEYWORDS):
        return _KEEP_RECENT["research"]

    return _KEEP_RECENT["default"]

# Per-session system instruction cache — avoids rebuilding 3-5K token prompt
# on every tool iteration within the same request.
# Key: (session_id, channel) → Value: cached instruction str
_sys_instruction_cache: dict[tuple[str, str], str] = {}
_sys_instruction_cache_lock = threading.Lock()


def _build_metric_injection(session_id: str) -> str:
    """Build per-turn self-metric block (never cached — must be fresh).

    Primary defense against self-metric hallucination: the LLM gets a
    deterministic snapshot + contract to reference metrics via tokens, not
    write numbers in prose. Collection failures produce a partial snapshot
    and never crash the turn.
    """
    try:
        from remy.core.agent_tools import brain, brain_lock
        from remy.core.metric_render import (
            build_compact_injection,
            build_full_injection,
        )
        from remy.core.metric_snapshot import collect_metric_snapshot

        with brain_lock:
            snapshot = collect_metric_snapshot(brain, session_id=session_id)
        if not snapshot.values:
            return ""
        # Compact line goes to every turn; full contract block gives the LLM
        # the rules for citing these metrics by token. Together ~150-250 tokens.
        compact = build_compact_injection(snapshot)
        full = build_full_injection(snapshot)
        return f"\n{compact}\n\n{full}\n"
    except Exception as exc:
        logger.debug("metric injection failed: %s", exc)
        return ""


def _get_cached_system_instruction(session_id: str, channel: str) -> str:
    """Get system instruction, cached per (session_id, channel) pair, with session directives."""
    key = (session_id, channel)
    metric_block = _build_metric_injection(session_id)
    with _sys_instruction_cache_lock:
        if key in _sys_instruction_cache:
            cached = _sys_instruction_cache[key]
            # Always append live session directives (not cached, may change mid-session)
            try:
                from remy.core.runtime_directives import format_directives_for_instruction
                directives = format_directives_for_instruction(session_id)
                if directives:
                    cached = cached + "\n\n" + directives
            except Exception:
                pass
            return cached + metric_block
    instruction = build_system_instruction(channel=channel)
    # Append session directives
    try:
        from remy.core.runtime_directives import format_directives_for_instruction
        directives = format_directives_for_instruction(session_id)
        if directives:
            instruction = instruction + "\n\n" + directives
    except Exception:
        pass
    with _sys_instruction_cache_lock:
        # Evict old entries to prevent unbounded growth
        if len(_sys_instruction_cache) > 50:
            _sys_instruction_cache.clear()
        # Cache base instruction without directives (directives are injected live)
        _sys_instruction_cache[key] = build_system_instruction(channel=channel)
    return instruction + metric_block


def invalidate_system_instruction_cache(session_id: str | None = None) -> None:
    """Invalidate cached system instructions. Call when brain context changes mid-session."""
    with _sys_instruction_cache_lock:
        if session_id:
            keys_to_remove = [k for k in _sys_instruction_cache if k[0] == session_id]
            for k in keys_to_remove:
                del _sys_instruction_cache[k]
        else:
            _sys_instruction_cache.clear()
    try:
        from remy.core.proactive_context import _proactive_context_cache

        _proactive_context_cache.clear()
    except Exception:
        pass
    try:
        from remy.core import brain_tools as _bt

        if hasattr(_bt, "_proactive_context_cache"):
            _bt._proactive_context_cache.clear()
    except Exception:
        pass


# ============== STATE ==============


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    session_id: str
    channel: str  # "desktop" | "telegram"
    session_log: list
    enabled_tools: set  # extended tool names enabled via enable_tools meta-tool
    _cached_session_ctx: str
    _cached_scratchpad: str
    _context_injected: bool  # True if _inject_context returned a non-None result


def _restore_pii_value(value, vault):
    """Restore PII tokens in strings, lists, and dict-shaped tool-call args."""
    from remy.core.pii_vault import restore

    if isinstance(value, str):
        return restore(value, vault)
    if isinstance(value, list):
        return [_restore_pii_value(item, vault) for item in value]
    if isinstance(value, dict):
        return {key: _restore_pii_value(item, vault) for key, item in value.items()}
    return value


def _restore_ai_message_pii(message: AIMessage, vault) -> AIMessage:
    """Restore PII tokens in an AIMessage before the app executes or displays it."""
    try:
        message.content = _restore_pii_value(message.content, vault)
        if getattr(message, "tool_calls", None):
            message.tool_calls = _restore_pii_value(message.tool_calls, vault)
        if getattr(message, "additional_kwargs", None):
            message.additional_kwargs = _restore_pii_value(message.additional_kwargs, vault)
    except Exception as exc:
        logger.warning("PII restore failed: %s", exc)
    return message


# ============== GRAPH NODES ==============


def call_model(state: AgentState) -> dict:
    """Invoke the LLM with system instruction and bound tools."""
    messages = state["messages"]
    channel = state.get("channel", "desktop")
    session_id = state.get("session_id", "")
    task_type = _estimate_task_type(channel, _latest_human_text(messages))

    # Per-iteration history compaction.
    # Without this, autonomous cycles grow messages unbounded across tool
    # iterations and can exceed Gemini's 1M token input limit. The entry-point
    # compaction (agent_runtime) only runs once per user turn, not per tool step.
    # Strip any leading SystemMessage first — it is re-added below with the
    # current cached instruction, so letting compact_history treat it as
    # history would confuse the summary pass.
    if messages and isinstance(messages[0], SystemMessage):
        trimmed = list(messages[1:])
    else:
        trimmed = list(messages)
    _keep_recent = _KEEP_RECENT.get(channel, _KEEP_RECENT["default"])
    if channel == "autonomous":
        try:
            from remy.core.context_window import dynamic_keep_recent
            _keep_recent = dynamic_keep_recent(channel, "")
        except Exception:
            pass
    messages = compact_history(trimmed, keep_recent=_keep_recent)

    # ACL cognitive brief — replaces replayed history with a typed snapshot
    # of current brain state. Only for autonomous channel, behind feature
    # flag. Falls back to pure compact_history on any failure.
    import os as _os
    _cognitive_brief = ""
    _brief_flag = _os.environ.get("ACL_BRIEF_ENABLED", "0") == "1"
    _brief_error: str | None = None
    if channel == "autonomous" and _brief_flag:
        try:
            from remy.core.agent_tools import brain as _brain
            from remy.core.cognitive_brief import build_cognitive_brief
            _cognitive_brief = build_cognitive_brief(_brain, locale="ua")
        except Exception as _exc:
            logger.debug("ACL brief disabled: %s", _exc)
            _cognitive_brief = ""
            _brief_error = str(_exc)

    # 24h trial metric — one JSONL record per autonomous-channel call_model.
    # Captures whether the brief was used, its size, and what the full
    # transcript would have cost so we can quantify savings tomorrow.
    if channel == "autonomous":
        try:
            from remy.core.cognitive_brief import (
                _estimate_messages_tokens,
                estimate_tokens as _est_tok,
                log_brief_metric,
            )
            log_brief_metric(
                enabled=_brief_flag,
                brief_used=bool(_cognitive_brief),
                brief_chars=len(_cognitive_brief),
                brief_tokens=_est_tok(_cognitive_brief) if _cognitive_brief else 0,
                transcript_tokens_estimate=_estimate_messages_tokens(messages),
                session_id=session_id,
                channel=channel,
                error=_brief_error,
            )
        except Exception as _metric_exc:
            logger.debug("acl metric skipped: %s", _metric_exc)

    # Ensure system instruction is the first message (cached per session)
    sys_instruction = _get_cached_system_instruction(session_id, channel)
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=sys_instruction)] + list(messages)
    else:
        # Update system message in case channel changed
        messages = [SystemMessage(content=sys_instruction)] + list(messages[1:])

    # Inject ACL cognitive brief right after sys instruction, so downstream
    # injectors (context, session, scratchpad, factuality) shift naturally.
    brief_msg = None
    if _cognitive_brief:
        brief_header = "=== COGNITIVE SNAPSHOT (current brain state) ==="
        brief_msg = SystemMessage(content=f"{brief_header}\n{_cognitive_brief}")
        messages.insert(1, brief_msg)

    # RM-7: Inject relevant context from brain
    context_msg = _inject_context(state)
    _context_injected_this_turn = False
    if context_msg:
        # Insert after system instruction (and brief if present), before history
        insert_pos = 2 if brief_msg else 1
        messages.insert(insert_pos, context_msg)
        _context_injected_this_turn = True

    # RM-11: Inject session context to prevent self-contradiction
    session_ctx = state.get("_cached_session_ctx") or ""
    if session_ctx:
        insert_pos = 1
        if brief_msg:
            insert_pos += 1
        if context_msg:
            insert_pos += 1
        messages.insert(insert_pos, SystemMessage(content=session_ctx))

    scratchpad_ctx = state.get("_cached_scratchpad") or ""
    if scratchpad_ctx:
        insert_pos = 1
        if brief_msg:
            insert_pos += 1
        if context_msg:
            insert_pos += 1
        if session_ctx:
            insert_pos += 1
        messages.insert(insert_pos, SystemMessage(content=scratchpad_ctx))

    factuality_contract_msg = _build_factuality_contract_message(state)
    if factuality_contract_msg:
        insert_pos = 1
        if brief_msg:
            insert_pos += 1
        if context_msg:
            insert_pos += 1
        if session_ctx:
            insert_pos += 1
        if scratchpad_ctx:
            insert_pos += 1
        messages.insert(insert_pos, factuality_contract_msg)

    # Strip text from intermediate AIMessages that also have tool_calls.
    # When the model generates text + tool_calls in one response, that text
    # is an intermediate "thinking" artifact. If passed to the next LLM call,
    # the model sees it and repeats it (causing duplicate responses).
    # We ALWAYS strip it regardless of length — tool_calls + ToolMessages
    # provide enough context for the model to generate a coherent final answer.
    cleaned = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls and msg.content:
            cleaned.append(AIMessage(content="", tool_calls=msg.tool_calls, id=msg.id))
        else:
            cleaned.append(msg)
    messages = cleaned

    # Fix message turn ordering for Gemini's strict requirements:
    # - no orphan ToolMessages, no broken tool sequences
    # - no SystemMessage between AIMessage(tool_calls) and ToolMessage
    # - no consecutive AIMessages without user/tool turns between them
    messages = _fix_gemini_turns(messages)

    # Selective tool loading: autonomous/proactive get all tools,
    # interactive channels start with core-only, extend via enable_tools
    if channel in ("autonomous", "proactive"):
        tools = get_all_tools()
    else:
        enabled = state.get("enabled_tools") or set()
        tool_names = CORE_TOOL_NAMES | enabled
        tools = get_tools_by_names(tool_names)

    from remy.core.adaptive_model_router import build_adaptive_model_routing
    from remy.core.llm import call_llm, model_routing_override
    from remy.core.model_trace import model_call_event

    _llm_start = time.time()
    _model_routing = build_adaptive_model_routing(
        messages=messages,
        channel=channel,
        task_type=task_type,
        base_routing=_model_routing_from_consequence_memory(channel, task_type),
    )
    with model_routing_override(
        preferred_model=str(_model_routing.get("preferred_model") or ""),
        avoid_models=tuple(_model_routing.get("avoid_models") or ()),
    ):
        pii_vault = None
        llm_messages = messages
        if settings.PII_SHIELD_ENABLED:
            try:
                from remy.core.pii_vault import get_vault, shield_messages

                pii_vault = get_vault(session_id or "__default__")
                llm_messages = shield_messages(messages, pii_vault)
            except Exception as exc:
                logger.warning("PII shield failed, sending unshielded LLM payload: %s", exc)
                pii_vault = None
                llm_messages = messages

        response = call_llm(llm_messages, tools=tools, purpose="agent")
        if pii_vault is not None and isinstance(response, AIMessage):
            response = _restore_ai_message_pii(response, pii_vault)
    _llm_duration = time.time() - _llm_start
    raw_response = response

    # Guard: ensure response is an AIMessage (langchain-google-genai may return raw Response)
    if not isinstance(response, AIMessage):
        logger.warning("LLM returned %s instead of AIMessage, wrapping", type(response).__name__)
        content = ""
        if hasattr(response, "content"):
            content = str(response.content)
        elif hasattr(response, "text"):
            content = str(response.text)
        response = AIMessage(content=content or "I encountered an issue processing your request.")

    # Track LLM call duration for metrics
    try:
        from remy.core.metrics import metrics_collector
        metrics_collector.record_llm_call(_llm_duration)
    except Exception:
        pass

    # Track usage (User)
    try:
        from remy.core.usage_stats import usage_tracker
        total_tokens = _extract_total_usage_tokens(response)
        if total_tokens > 0:
            usage_tracker.record_usage("user", total_tokens)
    except Exception as e:
        logger.warning(f"Failed to track usage: {e}")

    session_log = list(state.get("session_log", []))
    _model_call_event = model_call_event(
        raw_response,
        purpose="agent",
        channel=channel,
        duration_ms=int(_llm_duration * 1000),
    )
    _model_call_event["model_routing"] = {
        "preferred_model": str(_model_routing.get("preferred_model") or ""),
        "avoid_models": list(_model_routing.get("avoid_models") or ()),
        "source": str(_model_routing.get("routing_source") or ""),
        "complexity_bucket": str(_model_routing.get("complexity_bucket") or ""),
        "complexity_score": int(_model_routing.get("complexity_score") or 0),
        "reasons": list(_model_routing.get("routing_reasons") or ()),
        "task_type": str(_model_routing.get("task_type") or ""),
    }
    session_log.append(_model_call_event)

    return {
        "messages": [response],
        "session_log": session_log,
        "_context_injected": _context_injected_this_turn,
    }


def call_tools(state: AgentState) -> dict:
    """Execute tool calls from the last AI message."""
    from remy.core.event_bus import event_bus

    messages = state["messages"]
    session_log = list(state.get("session_log", []))
    session_id = state.get("session_id")
    channel = state.get("channel", "")

    # Set session_id + channel for tool execution & provenance tracking
    set_session_id(session_id)
    set_channel(channel)

    last_message = messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"messages": [], "session_log": session_log}

    tool_map = {t.name: t for t in get_all_tools()}
    tool_messages = []
    _emit = channel == "autonomous"
    newly_enabled: set[str] = set()

    for tc in last_message.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]

        logger.info("Tool call: %s(%s)", tool_name, tool_args)

        if _emit:
            # Build a readable summary of args for Activity stream
            args_summary = ", ".join(
                f"{k}={str(v)[:120]}" for k, v in tool_args.items()
            )
            event_bus.emit("tool_call", {
                "tool": tool_name,
                "args_summary": args_summary[:500],
            })

        tool = tool_map.get(tool_name)
        policy_block = (
            _blocked_tool_policy_hint(
                messages=messages,
                session_id=session_id,
                channel=channel,
                tool_name=tool_name,
                tool_args=tool_args,
            )
            if tool
            else None
        )
        if policy_block:
            result = (
                "Blocked by consequence memory: this exact tool action was "
                f"previously refuted. Reason: {policy_block.get('reason') or 'prior refutation'}"
            )
        elif tool:
            try:
                result = tool.invoke(tool_args)
            except Exception as e:
                logger.error("Tool %s error: %s", tool_name, e)
                result = f"Error: {e}"
        else:
            result = f"Unknown tool: {tool_name}"

        auto_extract_log_entry = None
        if not policy_block:
            _store_tool_consequence(
                messages=messages,
                session_id=session_id,
                channel=channel,
                tool_name=tool_name,
                tool_args=tool_args,
                result=result,
            )
        if not policy_block and tool_name in _FETCH_EVIDENCE_TOOLS:
            _record_fetch_evidence(session_id, tool_name, result, explicit_url=str(tool_args.get("url", "") or ""))
        if not policy_block and tool_name == "web_search":
            try:
                parsed_result = json.loads(str(result)) if isinstance(result, str) else result
                if isinstance(parsed_result, dict) and parsed_result.get("mode") == "candidate_discovery":
                    query = str(tool_args.get("query") or "")
                    sources = parsed_result.get("sources") or []
                    aligned_sources = _filter_candidate_sources_for_query(sources, query)
                    parsed_result.setdefault("candidate_count", len(sources))
                    if aligned_sources != list(sources or []):
                        parsed_result["raw_candidate_count"] = len(sources)
                        parsed_result["filtered_candidate_count"] = len(aligned_sources)
                        parsed_result["sources"] = aligned_sources
                        parsed_result["candidate_count"] = len(aligned_sources)
                        if not aligned_sources:
                            parsed_result["answer"] = (
                                "No candidate sources matched the query constraints yet. "
                                "Nothing is verified. Refine the query or search again."
                            )
                    selected_source = _choose_best_candidate_source(aligned_sources, query=query)
                    if selected_source:
                        parsed_result["selected_source"] = {
                            "title": selected_source.get("title", ""),
                            "uri": selected_source.get("uri", ""),
                            "trust_score": selected_source.get("trust_score", 0),
                            "trust_reason": selected_source.get("trust_reason", "fallback"),
                        }
                    top_source = selected_source or (aligned_sources[0] if aligned_sources else None)
                    top_url = str((top_source or {}).get("uri") or "").strip()
                    extract_tool = tool_map.get("extract_content")
                    if top_url and extract_tool:
                        try:
                            extracted = extract_tool.invoke({"url": top_url})
                            _record_fetch_evidence(session_id, "extract_content", extracted, explicit_url=top_url)
                            # Auto-fetch counts as forward progress too — clear the
                            # web_search-without-fetch counter so the agent isn't
                            # blocked on an already-fetched candidate.
                            try:
                                from remy.core.brain_tools import _reset_web_search_no_fetch
                                _reset_web_search_no_fetch(session_id)
                            except Exception:
                                pass
                            auto_extract_log_entry = {
                                "type": "tool_call",
                                "tool": "extract_content",
                                "args": {"url": str(top_url)[:100]},
                                "args_full": {"url": str(top_url)},
                                "result": str(extracted)[:200],
                                "result_full": str(extracted),
                                "auto_follow_from": "web_search",
                            }
                            parsed_result["auto_extract"] = {
                                "url": top_url,
                                "result": json.loads(str(extracted)) if isinstance(extracted, str) else extracted,
                            }
                            parsed_result["answer"] = (
                                parsed_result.get("answer", "")
                                + " Auto-fetched the best trusted candidate with extract_content for evidence-first follow-up."
                            ).strip()
                        except Exception as follow_err:
                            auto_extract_log_entry = {
                                "type": "tool_call",
                                "tool": "extract_content",
                                "args": {"url": str(top_url)[:100]},
                                "args_full": {"url": str(top_url)},
                                "result": f"Error: {follow_err}"[:200],
                                "result_full": f"Error: {follow_err}",
                                "auto_follow_from": "web_search",
                            }
                            parsed_result["auto_extract"] = {"url": top_url, "error": str(follow_err)}
                            parsed_result["answer"] = (
                                parsed_result.get("answer", "")
                                + " Auto-follow fetch failed; candidate discovery remains unverified until a fetch succeeds."
                            ).strip()
                    result = json.dumps(parsed_result, ensure_ascii=False)
            except Exception as e:
                logger.debug("Auto extract after web_search failed: %s", e)

        logger.info("Tool result: %s", str(result)[:200])

        if _emit:
            event_bus.emit("tool_result", {
                "tool": tool_name,
                "result": str(result)[:800],
                "is_error": str(result).startswith("Error:") or '"error"' in str(result)[:100],
            })

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )

        # Log for session summary
        session_log.append({
            "type": "tool_call",
            "tool": tool_name,
            "args": {k: str(v)[:100] for k, v in tool_args.items()},
            "args_full": {k: str(v) for k, v in tool_args.items()},
            "result": str(result)[:200],
            "result_full": str(result),
        })
        if policy_block:
            session_log[-1]["consequence_gate"] = {
                "blocked": True,
                "policy_hint": policy_block,
            }
        if auto_extract_log_entry is not None:
            session_log.append(auto_extract_log_entry)

        # Track enable_tools calls — update state so next call_model sees them
        if tool_name == "enable_tools":
            try:
                import json as _json
                parsed = _json.loads(result)
                newly_enabled.update(parsed.get("enabled", []))
            except Exception:
                pass

    ret: dict = {"messages": tool_messages, "session_log": session_log}

    if newly_enabled:
        enabled = set(state.get("enabled_tools") or set()) | newly_enabled
        ret["enabled_tools"] = enabled

    return ret


# ============== ROUTING ==============

_tool_call_count: dict[str, int] = {}
_tool_call_lock = threading.Lock()


def should_continue(state: AgentState) -> str:
    """Route: if last message has tool calls → 'tools', else → END."""
    messages = state["messages"]
    session_id = state.get("session_id", "default")

    if not messages:
        return END

    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # Check if current tools are research/browser — allow more iterations
        current_tools = {tc["name"] for tc in last_message.tool_calls}
        has_research = bool(current_tools & _RESEARCH_TOOLS)

        with _tool_call_lock:
            # Guard against infinite loops
            prev = _tool_call_count.get(session_id, 0)

            # Already sent wrap-up but model still wants tools? Hard stop.
            if prev == -1:
                logger.warning("Model ignored wrap-up instruction, forcing END")
                _tool_call_count.pop(session_id, None)
                return END

            count = prev + 1
            _tool_call_count[session_id] = count

        # Research/browser tasks get a higher ceiling than normal turns.
        effective_limit = 50 if has_research else MAX_TOOL_ITERATIONS
        if count > effective_limit:
            logger.warning("Max tool iterations (%d) reached for session %s",
                           MAX_TOOL_ITERATIONS, session_id[:8])
            with _tool_call_lock:
                # Mark as "wrap-up sent" so next time we hard-stop
                _tool_call_count[session_id] = -1
            # Force a final response instead of silence — provide dummy ToolMessages
            # for all pending tool calls, then route back to model for a wrap-up answer
            wrap_msg = ("[SYSTEM: Tool limit reached. Summarize everything you have done "
                        "so far and respond to the user. Do NOT call any more tools.]")
            for tc in last_message.tool_calls:
                state["messages"].append(
                    ToolMessage(content=wrap_msg, tool_call_id=tc["id"])
                )
            return "model"

        return "tools"

    # Reset counter on completion
    with _tool_call_lock:
        _tool_call_count.pop(session_id, None)
    return END


# ============== GRAPH BUILDER ==============

_compiled_graphs: dict[str, object] = {}
_graph_lock = threading.Lock()


def invalidate_graph_cache() -> None:
    """Clear compiled graph cache so tools are re-read on next invocation."""
    with _graph_lock:
        _compiled_graphs.clear()


def build_agent_graph(channel: str = "desktop"):
    """Build and compile the LangGraph agent. Cached per channel."""
    with _graph_lock:
        if channel in _compiled_graphs:
            return _compiled_graphs[channel]

        graph = StateGraph(AgentState)

        graph.add_node("model", call_model)
        graph.add_node("tools", call_tools)

        graph.set_entry_point("model")

        graph.add_conditional_edges(
            "model",
            should_continue,
            {"tools": "tools", END: END, "model": "model"},
        )
        graph.add_edge("tools", "model")

        compiled = graph.compile()
        _compiled_graphs[channel] = compiled
        logger.info("Agent graph compiled for channel: %s", channel)
        return compiled


# ============== IN-SESSION THINKING ==============

_message_counts: dict[str, int] = {}
_message_counts_lock = threading.Lock()


def check_session_insights(session_id: str, messages: list) -> list:
    """Periodically check brain for insights and inject into conversation.

    Runs every INSIGHT_CHECK_INTERVAL messages. Zero LLM calls — pure Python.
    Returns messages list with insight injected (or unchanged).
    """
    with _message_counts_lock:
        # Prevent unbounded dict growth in long-running server deployments
        if len(_message_counts) > 100:
            # Keep only the 50 most recent entries
            sorted_keys = sorted(_message_counts, key=_message_counts.get, reverse=True)
            for k in sorted_keys[50:]:
                del _message_counts[k]

        count = _message_counts.get(session_id, 0) + 1
        _message_counts[session_id] = count

    if count % INSIGHT_CHECK_INTERVAL != 0:
        return messages

    try:
        from remy.core.agent_tools import Level, brain, brain_lock_read
        from remy.core.hybrid_search import recall_cognitive_structured, search_exact_structured
        from remy.core.memory_policy import sanitize_memory_content, sanitize_memory_metadata

        # Extract last user text for context search
        _last_user_text = ""
        for m in reversed(messages):
            c = getattr(m, "content", "")
            if isinstance(c, str) and c.strip():
                _last_user_text = c.strip()
                break

        try:
            with brain_lock_read(timeout=1.5):
                exact_hits = search_exact_structured(
                    brain,
                    _last_user_text,
                    top_k=6,
                )
        except RuntimeError:
            return messages  # brain busy — skip insights, respond immediately
        exact_lines = []
        for hit in exact_hits:
            level = str(hit.get("level", ""))
            if "IDENTITY" not in level and "DOMAIN" not in level:
                continue
            content = (hit.get("content") or "").strip()
            if not content:
                continue
            line_ids = _extract_record_ids(content)
            if line_ids and line_ids.issubset(system_ids):
                continue
            meta = sanitize_memory_metadata(hit.get("metadata") or {}, tags=hit.get("tags") or [])
            tag_str = f" [{', '.join(hit.get('tags', []))}]" if hit.get("tags") else ""
            source = meta.get("source")
            prefix = f"- ({source}) " if source else "- "
            exact_lines.append(f"{prefix}{content[:220]}{tag_str}")
        if exact_lines:
            context_parts.append("[Exact Memory]\n" + "\n".join(exact_lines[:4]))

        try:
            with brain_lock_read(timeout=1.5):
                insights = brain.insights()
        except RuntimeError:
            return messages  # brain busy — skip insights
        if not insights:
            return messages

        important = [i for i in insights if i["type"] in (
            "decay_risk", "conflict", "hot_topic"
        )]

        if not important:
            return messages

        parts = []
        for ins in important[:2]:
            if ins["type"] == "decay_risk":
                records = ins["details"].get("records", [])
                names = [r["content"][:50] for r in records[:2]]
                if names:
                    parts.append(f"Fading memories: {', '.join(names)}")
            elif ins["type"] == "conflict":
                pairs = ins["details"].get("pairs", [])
                if pairs:
                    p = pairs[0]
                    parts.append(f"Possible contradiction: '{p['content_a'][:40]}' vs '{p['content_b'][:40]}'")
            elif ins["type"] == "hot_topic":
                topics = ins["details"].get("topics", [])
                names = [t["tag"] for t in topics[:3]]
                if names:
                    parts.append(f"Active topics: {', '.join(names)}")

        if parts:
            insight_text = "[INTERNAL BRAIN INSIGHT — mention naturally if relevant]: " + "; ".join(parts)
            messages.append(SystemMessage(content=insight_text))
            logger.info("In-session insight injected: %s", insight_text[:100])

    except Exception as e:
        logger.debug("In-session insight check failed: %s", e)

    return messages


# ============== HISTORY COMPRESSION ==============


def compact_history(messages: list, keep_recent: int = 16) -> list:
    """Compact history to reduce context window usage.

    1. Truncate long ToolMessage content (>300 chars)
    2. If messages > keep_recent: compress older messages into a summary
    3. Never break tool call sequences (AIMessage+ToolMessages stay together)
    """
    if not messages:
        return messages

    # Step 1: Truncate tool results
    compacted = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str) and len(msg.content) > 300:
            compacted.append(ToolMessage(
                content=msg.content[:300] + "...[truncated]",
                tool_call_id=msg.tool_call_id,
            ))
        else:
            compacted.append(msg)

    # Step 2: If short enough, return as-is
    if len(compacted) <= keep_recent:
        return compacted

    # Step 3: Split into old + recent, keeping tool sequences intact
    split = len(compacted) - keep_recent
    while split < len(compacted) and isinstance(compacted[split], ToolMessage):
        split += 1

    old_part = compacted[:split]
    recent_part = compacted[split:]

    # Step 4: Summarize old part (extract user + AI text, skip tool details)
    # Use 300 chars per message to preserve more context for self-consistency
    summary_lines = []
    for msg in old_part:
        if isinstance(msg, HumanMessage):
            text = msg.content if isinstance(msg.content, str) else "[multimodal]"
            summary_lines.append(f"User: {text[:300]}")
        elif isinstance(msg, AIMessage) and msg.content:
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            if text.strip():
                summary_lines.append(f"Remy: {text[:300]}")

    if summary_lines:
        summary = "Earlier in this conversation:\n" + "\n".join(summary_lines)
        if len(summary) > 8000:
            summary = summary[:8000] + "\n...[truncated]"
        recent_part = _sanitize_tool_sequences(recent_part)
        if any(isinstance(m, HumanMessage) for m in recent_part):
            aligned = list(recent_part)
            while aligned and not isinstance(aligned[0], HumanMessage):
                aligned.pop(0)
            recent_part = aligned or recent_part
        return [SystemMessage(content=summary)] + recent_part

    return recent_part


def _sanitize_tool_sequences(messages: list) -> list:
    """Strip leading orphan ToolMessages after compaction."""
    cleaned = list(messages)
    while cleaned and isinstance(cleaned[0], ToolMessage):
        cleaned.pop(0)
    return cleaned


def _fix_gemini_turns(messages: list) -> list:
    """Fix message sequence to satisfy Gemini's strict turn ordering.

    Gemini requires:
    - A function call (AIMessage with tool_calls) must come right after
      a user turn (HumanMessage) or a function response (ToolMessage).
    - ToolMessages must immediately follow the AIMessage with tool_calls.
    - No two AIMessages in a row without a HumanMessage/ToolMessage between them.
    - SystemMessages must not appear between AIMessage(tool_calls) and ToolMessages.

    This function:
    1. Removes orphan ToolMessages (no preceding AIMessage with matching tool_calls)
    2. Removes AIMessages with tool_calls that lack corresponding ToolMessages
    3. Ensures no SystemMessage breaks an AIMessage→ToolMessage sequence
    4. Collapses consecutive AIMessages (keeps the last one)
    """
    if not messages:
        return messages

    # --- Pass 1: collect valid tool_call_ids ---
    # Build a set of tool_call_ids from AIMessages that have tool_calls
    ai_tool_call_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                ai_tool_call_ids.add(tc["id"])

    # Build a set of tool_call_ids that have matching ToolMessages
    tool_response_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_response_ids.add(msg.tool_call_id)

    # IDs that have both an AIMessage tool_call and a ToolMessage response
    valid_ids = ai_tool_call_ids & tool_response_ids

    # --- Pass 2: filter out broken sequences ---
    filtered = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            if msg.tool_call_id not in valid_ids:
                continue  # orphan ToolMessage
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # Keep only tool_calls that have matching responses
            good_calls = [tc for tc in msg.tool_calls if tc["id"] in valid_ids]
            if not good_calls:
                # No valid tool calls — convert to plain AIMessage if has content
                if msg.content:
                    filtered.append(AIMessage(content=msg.content, id=msg.id))
                continue  # drop entirely if no content either
            if len(good_calls) != len(msg.tool_calls):
                # Partial match — rebuild with only valid calls
                filtered.append(AIMessage(
                    content=msg.content or "",
                    tool_calls=good_calls,
                    id=msg.id,
                ))
                continue
        filtered.append(msg)

    # --- Pass 3: ensure SystemMessages don't break AI→Tool sequences ---
    # Move any SystemMessage that sits between AIMessage(tool_calls) and ToolMessage
    result = []
    deferred_system: list = []
    expecting_tool_response = False

    for msg in filtered:
        if expecting_tool_response:
            if isinstance(msg, ToolMessage):
                result.append(msg)
                continue
            elif isinstance(msg, SystemMessage):
                # Defer — will insert before the AIMessage(tool_calls)
                deferred_system.append(msg)
                continue
            else:
                # Unexpected message — tool sequence ended
                expecting_tool_response = False
                # Flush deferred system messages
                result.extend(deferred_system)
                deferred_system.clear()

        if isinstance(msg, AIMessage) and msg.tool_calls:
            # Flush any deferred system messages BEFORE this AI message
            if deferred_system:
                result.extend(deferred_system)
                deferred_system.clear()
            result.append(msg)
            expecting_tool_response = True
        else:
            if deferred_system:
                result.extend(deferred_system)
                deferred_system.clear()
            result.append(msg)

    if deferred_system:
        result.extend(deferred_system)

    # --- Pass 4: collapse consecutive AIMessages ---
    final = []
    for msg in result:
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            if final and isinstance(final[-1], AIMessage) and not final[-1].tool_calls:
                # Two plain AI messages in a row — keep the last one
                final[-1] = msg
                continue
        final.append(msg)

    # --- Pass 5: ensure first non-System message is HumanMessage ---
    # Gemini wants the conversation to start with a user turn (after system messages)
    first_non_system = None
    for i, msg in enumerate(final):
        if not isinstance(msg, SystemMessage):
            first_non_system = i
            break

    if first_non_system is not None and isinstance(final[first_non_system], AIMessage):
        # Insert a synthetic user message
        final.insert(first_non_system, HumanMessage(content="[continue]"))

    return final


# ============== CONVENIENCE FUNCTION ==============


async def invoke_agent(
    user_message: str | HumanMessage,
    session_id: str,
    channel: str,
    session_log: list,
    history: list | None = None,
) -> tuple[str, list, list]:
    return await _invoke_agent_inner(user_message, session_id, channel, session_log, history)


async def _invoke_agent_inner(
    user_message: str | HumanMessage,
    session_id: str,
    channel: str,
    session_log: list,
    history: list | None = None,
) -> tuple[str, list, list]:
    """Run the agent and return (response_text, updated_messages, updated_session_log).

    Args:
        user_message: Text string or HumanMessage (for multimodal).
        session_id: Session ID for tool context.
        channel: "desktop" or "telegram".
        session_log: Activity log for session summary.
        history: Previous message history (LangChain messages).

    Returns:
        (response_text, messages_list, session_log)
    """
    import asyncio

    _invoke_start = time.time()
    try:
        from remy.core.eval_metrics import snapshot_recall_stats
        _recall_stats_start = snapshot_recall_stats()
    except Exception:
        _recall_stats_start = {}

    # Invalidate system instruction cache so it's rebuilt once this turn,
    # then stays cached across tool iterations within the same request.
    invalidate_system_instruction_cache(session_id)

    graph = build_agent_graph(channel)

    # Build messages list
    messages = list(history) if history else []

    # Compact history — truncate tool results, compress old messages
    keep_recent = _estimate_keep_recent(channel, user_message)
    messages = compact_history(messages, keep_recent=keep_recent)

    # In-session thinking — periodically inject brain insights
    messages = check_session_insights(session_id, messages)

    # Add user message
    if isinstance(user_message, str):
        messages.append(HumanMessage(content=user_message))
    else:
        messages.append(user_message)

    state = AgentState(
        messages=messages,
        session_id=session_id,
        channel=channel,
        session_log=list(session_log),
        enabled_tools=set(),
        _cached_session_ctx="",
        _cached_scratchpad="",
    )

    session_ctx = _build_session_context(messages)
    if session_ctx:
        state["_cached_session_ctx"] = session_ctx.content
    try:
        from remy.core.scratchpad import get_scratchpad_context

        scratchpad_ctx = get_scratchpad_context(
            query=_extract_text(user_message),
            auto_filter=True,
            session_id=session_id,
        )
        if scratchpad_ctx:
            state["_cached_scratchpad"] = scratchpad_ctx
    except Exception:
        pass

    # Run graph in thread to avoid blocking event loop
    # Adaptive recursion limit based on channel + message content
    rec_limit = _estimate_recursion_limit(channel, user_message)
    config = {"recursion_limit": rec_limit}
    logger.debug("Recursion limit: %d (channel=%s)", rec_limit, channel)

    try:
        result = await asyncio.to_thread(graph.invoke, state, config)
    except Exception as e:
        # Handle GraphRecursionError — extract partial results
        if "recursion limit" in str(e).lower() or "Recursion limit" in str(e):
            logger.warning("Recursion limit (%d) hit for session %s — extracting partial response",
                           rec_limit, session_id[:8])
            # Gather what the agent accomplished from session_log
            result = state  # state has messages accumulated so far
        else:
            raise

    # Extract response text from last AI message
    response_text = ""
    result_messages = result.get("messages", [])
    for msg in reversed(result_messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            content = msg.content
            # content can be str or list of dicts (multimodal response)
            if isinstance(content, str):
                response_text = content.strip()
            elif isinstance(content, list):
                # Extract text parts from list content
                text_parts = []
                for part in content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and part.get("text"):
                        text_parts.append(part["text"])
                response_text = "\n".join(text_parts).strip()
            break

    if not response_text:
        # Build a summary from session_log so the user sees what was accomplished
        log_items = result.get("session_log", session_log)
        tool_names = [item["tool"] for item in log_items if item.get("type") == "tool_call"]
        if tool_names:
            summary = ", ".join(dict.fromkeys(tool_names))  # unique, preserving order
            response_text = (
                f"I reached the step limit while working on your request. "
                f"Tools used: {summary}. Please send a follow-up message "
                f"to continue where I left off."
            )
        else:
            response_text = "I couldn't generate a response. Please try again."

    # Filter messages for history (exclude SystemMessage)
    clean_messages = [
        m for m in result_messages if not isinstance(m, SystemMessage)
    ]

    updated_log = list(result.get("session_log", session_log))

    # Pre-mouth epistemic governance (Phase A.7 Step 4 — brain-native mouth).
    #
    # Block path: brain decides epistemic state via decide_governance(), then
    # the mouth organ renders an honest-uncertainty reply in the turn's
    # surface language via the SLM. The contaminated draft is discarded and
    # never reaches the user or enforce_factuality.
    #
    # Soft / aggressive paths: still use the legacy govern_response adapter
    # until A.7 Step 3 cleanup after streaming/non-streaming unification.
    governance_decision = None
    governance_output = None
    mouth_render_source = None
    try:
        from remy.core.epistemic_governance import (
            decide_governance, decision_from_output,
        )
        governance_output = decide_governance(
            response_text, updated_log, session_id=session_id,
        )
        if governance_output.mode == "block":
            from remy.core.mouth import render_block_response
            mouth_result = render_block_response(user_message, governance_output)
            response_text = mouth_result.text
            mouth_render_source = mouth_result.source
            governance_decision = decision_from_output(governance_output)
        else:
            from remy.core.epistemic_governance import govern_response
            response_text, governance_decision = govern_response(
                response_text, updated_log,
                locale=_detect_turn_locale(user_message),
                session_id=session_id,
            )
    except Exception:
        governance_decision = None
        governance_output = None

    # DS-1 short-circuit: once governance hard-blocks, enforce_factuality must
    # not overwrite the honest fallback with a rewritten version of the phantom.
    factuality_report = None
    if governance_decision is None or governance_decision.mode != "block":
        try:
            response_text, factuality_report = enforce_factuality(
                response_text, updated_log, session_id=session_id,
            )
        except Exception:
            factuality_report = None
    factuality_claims = summarize_claim_details(factuality_report)
    _store_factuality_consequence(
        user_message=user_message,
        session_id=session_id,
        channel=channel,
        factuality_report=factuality_report,
    )
    _store_model_outcome_consequence(
        user_message=user_message,
        session_id=session_id,
        channel=channel,
        task_type=_estimate_task_type(channel, user_message),
        session_log=updated_log,
        response_text=response_text,
        factuality_report=factuality_report,
        governance_decision=governance_decision,
    )
    _store_source_grounding_consequences(
        session_id=session_id,
        channel=channel,
        factuality_report=factuality_report,
        governance_decision=governance_decision,
    )

    if response_text:
        last_model_entry = updated_log[-1] if updated_log else None
        if not (
            isinstance(last_model_entry, dict)
            and last_model_entry.get("type") == "model_response"
            and (last_model_entry.get("text") or "") == response_text[:200]
        ):
            updated_log.append({
                "type": "model_response",
                "text": response_text[:200],
                "full_text": response_text,
            })
    if factuality_report:
        updated_log.append({
            "type": "factuality_analysis",
            "unsupported_observed_claims": factuality_report.unsupported_observed_claims,
            "unverified_current_claims": factuality_report.unverified_current_claims,
            "supported_claims_total": factuality_report.supported_claims_total,
            "unsupported_claims_total": factuality_report.unsupported_claims_total,
            "evidence_record_ids": list(factuality_report.evidence_record_ids),
            "claims": factuality_claims,
            # Split evidence counts (Phase A.3).
            "supported_internal": factuality_report.supported_internal,
            "supported_external_verified": factuality_report.supported_external_verified,
            "unverified_external": factuality_report.unverified_external,
            "unsupported": factuality_report.unsupported,
            "brain_storage_unsafe": factuality_report.brain_storage_unsafe,
        })
    if governance_decision is not None and governance_decision.mode != "none":
        entry = {
            "type": "epistemic_governance",
            **governance_decision.to_dict(),
        }
        if mouth_render_source:
            entry["mouth_render_source"] = mouth_render_source
        if governance_output is not None:
            entry["structured"] = governance_output.to_dict()
        updated_log.append(entry)

    # F5: Collect and store evaluation metrics (non-critical)
    try:
        from remy.core.eval_metrics import compute_response_metrics, store_eval_metrics
        _duration_ms = int((time.time() - _invoke_start) * 1000)
        _was_context_injected = bool(result.get("_context_injected", False))
        _metrics = compute_response_metrics(
            session_id=session_id,
            channel=channel,
            messages=result_messages,
            session_log=updated_log,
            response_text=response_text,
            duration_ms=_duration_ms,
            context_injected=_was_context_injected,
            unsupported_observed_claims=(
                factuality_report.unsupported_observed_claims if factuality_report else 0
            ),
            recall_stats_before=_recall_stats_start,
        )
        store_eval_metrics(_metrics)
    except Exception:
        pass

    # F3: Detect implicit feedback signals (user-facing channels only)
    if channel in ("desktop", "telegram", "voice"):
        try:
            from remy.core.brain_tools import (
                apply_latest_user_correction_feedback,
                detect_feedback_signals,
                store_feedback_signal,
            )
            _signals = detect_feedback_signals(result_messages, channel)
            for _sig in _signals:
                store_feedback_signal(_sig)
            apply_latest_user_correction_feedback(
                result_messages,
                channel,
                session_log=updated_log,
            )
        except Exception:
            pass

    return response_text, clean_messages, updated_log


async def invoke_agent_stream(
    user_message: str | HumanMessage,
    session_id: str,
    channel: str,
    session_log: list,
    history: list | None = None,
):
    """Run the agent and yield events for streaming response.

    Yields:
        dict: {
            "type": "token" | "tool_start" | "tool_end" | "final",
            "content": str | dict,
            ...
        }
    """
    import asyncio
    from langchain_core.messages import AIMessageChunk

    # Invalidate system instruction cache for fresh rebuild this turn
    invalidate_system_instruction_cache(session_id)

    graph = build_agent_graph(channel)

    # Build messages list
    messages = list(history) if history else []

    # Compact history — truncate tool results, compress old messages
    keep_recent = _estimate_keep_recent(channel, user_message)
    messages = compact_history(messages, keep_recent=keep_recent)

    # In-session thinking
    messages = check_session_insights(session_id, messages)

    # Add user message
    if isinstance(user_message, str):
        messages.append(HumanMessage(content=user_message))
    else:
        messages.append(user_message)

    state = AgentState(
        messages=messages,
        session_id=session_id,
        channel=channel,
        session_log=list(session_log),
        enabled_tools=set(),
        _cached_session_ctx="",
        _cached_scratchpad="",
    )

    session_ctx = _build_session_context(messages)
    if session_ctx:
        state["_cached_session_ctx"] = session_ctx.content
    try:
        from remy.core.scratchpad import get_scratchpad_context

        scratchpad_ctx = get_scratchpad_context(
            query=_extract_text(user_message),
            auto_filter=True,
            session_id=session_id,
        )
        if scratchpad_ctx:
            state["_cached_scratchpad"] = scratchpad_ctx
    except Exception:
        pass

    final_text = ""
    tool_inputs = {}
    final_session_log = list(session_log)

    # Stream events from the graph
    last_ai_message = None
    # Buffer tokens per model call — only yield when model call is final (no tool calls).
    # This prevents intermediate model calls (which produce text + tool calls) from
    # being streamed to the user, causing "triple response" artifacts.
    _token_buffer = []  # tokens buffered from current model call
    _model_had_tool_calls = False  # whether current model call produced tool calls
    _flushed = False  # guard: True after buffer was flushed, reset on new tokens
    _recursion_hit = False
    _tools_ran = False  # True after at least one tool ran — used to emit "thinking" on next LLM call
    _thinking_yielded = False  # prevent duplicate "thinking" events per model call
    _pii_stream_restorer = None
    if settings.PII_SHIELD_ENABLED:
        try:
            from remy.core.pii_vault import StreamingRestorer, get_vault

            _pii_stream_restorer = StreamingRestorer(get_vault(session_id or "__default__"))
        except Exception as exc:
            logger.debug("PII stream restorer disabled: %s", exc)
            _pii_stream_restorer = None

    def _restore_stream_token(token: str) -> str:
        if _pii_stream_restorer is None:
            return token
        return _pii_stream_restorer.feed(token)

    def _flush_stream_restorer() -> str:
        if _pii_stream_restorer is None:
            return ""
        return _pii_stream_restorer.flush()

    rec_limit = _estimate_recursion_limit(channel, user_message)
    config = {"recursion_limit": rec_limit}
    logger.debug("Stream recursion limit: %d (channel=%s)", rec_limit, channel)

    try:
        async for event in graph.astream_events(state, version="v1", config=config):
            kind = event["event"]

            # Stream LLM tokens — buffer them, don't yield yet
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if isinstance(chunk, AIMessageChunk):
                    # Emit "thinking" once when LLM starts generating after tools ran
                    if _tools_ran and not _thinking_yielded:
                        _thinking_yielded = True
                        yield {"type": "thinking", "content": "Analyzing results..."}
                    # New tokens arriving → reset flush guard (new model call started)
                    _flushed = False
                    # Detect if this model call has tool calls
                    if chunk.tool_call_chunks:
                        _model_had_tool_calls = True
                    if chunk.content and not chunk.tool_call_chunks:
                        content = chunk.content
                        # langchain-google-genai may return content as list of dicts
                        if isinstance(content, list):
                            content = "".join(
                                part.get("text", "") if isinstance(part, dict) else str(part)
                                for part in content
                            )
                        if isinstance(content, str) and content:
                            _token_buffer.append(content)

            # When a model call ends, decide: flush buffer or discard
            # Note: LangGraph uses NODE name ("model"), not function name ("call_model")
            elif kind == "on_chain_end":
                if event.get("name") == "tools":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict) and output.get("session_log") is not None:
                        final_session_log = output["session_log"]
                elif event.get("name") == "model":
                    output = event.get("data", {}).get("output", {})
                    out_messages = output.get("messages", [])
                    if out_messages:
                        last_ai_message = out_messages[-1]

                    # Skip duplicate on_chain_end events for the same model call
                    if _flushed:
                        continue

                    # Check if this model call had tool calls (intermediate step)
                    has_tools = (
                        _model_had_tool_calls
                        or (last_ai_message and isinstance(last_ai_message, AIMessage) and last_ai_message.tool_calls)
                    )

                    if has_tools:
                        # Intermediate model call — discard buffered tokens
                        logger.debug("Discarding %d buffered tokens from intermediate model call (has tool_calls)", len(_token_buffer))
                        _token_buffer.clear()
                    else:
                        # Final model call: restore PII before streaming to the UI.
                        for token in _token_buffer:
                            restored = _restore_stream_token(token)
                            if restored:
                                yield {"type": "token", "content": restored}
                                final_text += restored
                        tail = _flush_stream_restorer()
                        if tail:
                            yield {"type": "token", "content": tail}
                            final_text += tail
                        _token_buffer.clear()

                    _flushed = True
                    # Reset for next model call
                    _model_had_tool_calls = False

            # Tool usage feedback — enriched with args/result summaries
            elif kind == "on_tool_start":
                tool_name = event["name"]
                if not tool_name.startswith("_"):
                    # Extract args summary from input
                    args_summary = ""
                    try:
                        tool_input = event.get("data", {}).get("input", {})
                        if isinstance(tool_input, dict):
                            tool_inputs[tool_name] = tool_input
                            parts = []
                            for k, v in tool_input.items():
                                sv = str(v)
                                if len(sv) > 80:
                                    sv = sv[:77] + "..."
                                parts.append(f"{k}={sv}")
                            args_summary = ", ".join(parts)[:200]
                    except Exception:
                        pass
                    yield {"type": "tool_start", "tool": tool_name, "args": args_summary}

            elif kind == "on_tool_end":
                tool_name = event["name"]
                if not tool_name.startswith("_"):
                    _tools_ran = True
                    _thinking_yielded = False  # reset so next LLM call emits "thinking"
                    # Extract result summary
                    result_summary = ""
                    try:
                        output = event.get("data", {}).get("output", "")
                        if hasattr(output, "content"):
                            output = output.content
                        result_str = str(output)
                        if len(result_str) > 150:
                            result_summary = result_str[:147] + "..."
                        else:
                            result_summary = result_str
                    except Exception:
                        pass
                    yield {"type": "tool_end", "tool": tool_name, "result": result_summary}

    except Exception as e:
        if "recursion limit" in str(e).lower() or "Recursion limit" in str(e):
            _recursion_hit = True
            logger.warning("Stream recursion limit (%d) hit — yielding partial response", rec_limit)
        elif "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            logger.error("LLM quota exhausted during stream: %s", e)
            final_text = (
                "API quota exhausted. "
                "Please wait and try again later, or switch to a different model."
            )
            yield {"type": "token", "content": final_text}
        else:
            raise

    # Flush any remaining buffered tokens (safety net)
    if _token_buffer:
        for token in _token_buffer:
            restored = _restore_stream_token(token)
            if restored:
                yield {"type": "token", "content": restored}
                final_text += restored
        tail = _flush_stream_restorer()
        if tail:
            yield {"type": "token", "content": tail}
            final_text += tail
        _token_buffer.clear()

    logger.info(f"Stream done: final_text length={len(final_text)}, has_last_ai={last_ai_message is not None}")

    # If streaming didn't capture tokens, extract text from the last AI message
    if not final_text and last_ai_message:
        content = last_ai_message.content
        logger.debug(f"Fallback: last_ai_message content type={type(content).__name__}, value={repr(content)[:200]}")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        if isinstance(content, str):
            final_text = content
        logger.info(f"Fallback extracted final_text length={len(final_text)}")

    # If we ended up with no text for any reason (recursion limit, tool-iteration
    # hard-stop, streaming anomaly), surface a visible message instead of silence.
    if not final_text:
        if _recursion_hit:
            final_text = (
                "I reached the step limit while working on your request. "
                "Please send a follow-up message to continue where I left off."
            )
        else:
            final_text = (
                "I couldn't produce a final answer this turn (likely hit the "
                "tool-call limit while searching). Please rephrase or narrow "
                "the request."
            )
        yield {"type": "token", "content": final_text}

    # Pre-mouth epistemic governance for streaming path too.
    stream_governance_decision = None
    try:
        from remy.core.epistemic_governance import govern_response
        final_text, stream_governance_decision = govern_response(
            final_text, final_session_log,
            locale=_detect_turn_locale(user_message),
            session_id=session_id,
        )
    except Exception:
        stream_governance_decision = None

    # DS-1 short-circuit: see non-streaming path.
    factuality_report = None
    if stream_governance_decision is None or stream_governance_decision.mode != "block":
        try:
            final_text, factuality_report = enforce_factuality(
                final_text, final_session_log, session_id=session_id,
            )
        except Exception:
            factuality_report = None
    factuality_claims = summarize_claim_details(factuality_report)
    _store_factuality_consequence(
        user_message=user_message,
        session_id=session_id,
        channel=channel,
        factuality_report=factuality_report,
    )
    _store_model_outcome_consequence(
        user_message=user_message,
        session_id=session_id,
        channel=channel,
        task_type=_estimate_task_type(channel, user_message),
        session_log=final_session_log,
        response_text=final_text,
        factuality_report=factuality_report,
        governance_decision=stream_governance_decision,
    )
    _store_source_grounding_consequences(
        session_id=session_id,
        channel=channel,
        factuality_report=factuality_report,
        governance_decision=stream_governance_decision,
    )
    if stream_governance_decision is not None and stream_governance_decision.mode != "none":
        final_session_log.append({
            "type": "epistemic_governance",
            **stream_governance_decision.to_dict(),
        })

    if final_text:
        last_model_entry = final_session_log[-1] if final_session_log else None
        if not (
            isinstance(last_model_entry, dict)
            and last_model_entry.get("type") == "model_response"
            and (last_model_entry.get("text") or "") == final_text[:200]
        ):
            final_session_log.append({
                "type": "model_response",
                "text": final_text[:200],
                "full_text": final_text,
            })
    if factuality_report:
        final_session_log.append({
            "type": "factuality_analysis",
            "unsupported_observed_claims": factuality_report.unsupported_observed_claims,
            "unverified_current_claims": factuality_report.unverified_current_claims,
            "supported_claims_total": factuality_report.supported_claims_total,
            "unsupported_claims_total": factuality_report.unsupported_claims_total,
            "evidence_record_ids": list(factuality_report.evidence_record_ids),
            "claims": factuality_claims,
        })

    if channel in ("desktop", "telegram", "voice"):
        try:
            from remy.core.brain_tools import (
                apply_latest_user_correction_feedback,
                detect_feedback_signals,
                store_feedback_signal,
            )
            final_messages = messages + [AIMessage(content=final_text)]
            _signals = detect_feedback_signals(final_messages, channel)
            for _sig in _signals:
                store_feedback_signal(_sig)
            apply_latest_user_correction_feedback(
                final_messages,
                channel,
                session_log=final_session_log,
            )
        except Exception:
            pass

    # Metric rendering — primary defense. Substitute {{metric:id}} tokens the
    # LLM (hopefully) emitted against a fresh snapshot. Unknown/stale ids get
    # deterministic placeholders. The auditor below runs AFTER this so it sees
    # substituted values, not slot tokens.
    metric_render_summary = None
    try:
        from remy.core.agent_tools import brain, brain_lock
        from remy.core.metric_render import render_metrics
        from remy.core.metric_snapshot import collect_metric_snapshot

        with brain_lock:
            _snapshot = collect_metric_snapshot(brain, session_id=session_id)
        if _snapshot.values:
            _rendered = render_metrics(final_text, _snapshot)
            final_text = _rendered.text
            metric_render_summary = {
                "used": list(_rendered.used_metric_ids),
                "unknown": list(_rendered.unknown_metric_ids),
                "stale": list(_rendered.stale_metric_ids),
                "available": list(_snapshot.available_ids),
                "missing_sources": list(_snapshot.missing_metric_ids),
            }
    except Exception as _e:
        logger.debug("metric render failed: %s", _e)

    # Epistemic response auditor — detect fabricated arXiv IDs, live metrics,
    # entitlement claims without tool backing. Warn mode logs to JSONL; block
    # mode rewrites final_text. Mode controlled via RESPONSE_AUDITOR_MODE env.
    audit_summary = None
    try:
        from remy.core.retrieval.evidence_resolver import TurnContext
        from remy.core.retrieval.response_auditor import audit_response

        _turn = TurnContext(session_log=final_session_log, session_id=session_id)
        _audit = audit_response(final_text, turn=_turn, turn_id=session_id)
        if _audit.has_violations:
            audit_summary = {
                "violations": len(_audit.violations),
                "types": sorted({v.claim.claim_type for v in _audit.violations}),
                "mode": _audit.actions[0].mode if _audit.actions else "warn",
            }
            if _audit.rewritten_text is not None:
                final_text = _audit.rewritten_text
    except Exception as _e:
        logger.debug("response auditor failed: %s", _e)

    # Yield final packet so the caller (session.py) can update session history
    yield {
        "type": "final",
        "text": final_text,
        "messages": messages + [AIMessage(content=final_text)],
        "session_log": final_session_log,
        "metric_render": metric_render_summary,
        "epistemic_audit": audit_summary,
        "factuality": (
            {
                "unsupported_observed_claims": factuality_report.unsupported_observed_claims,
                "unverified_current_claims": factuality_report.unverified_current_claims,
                "supported_claims_total": factuality_report.supported_claims_total,
                "unsupported_claims_total": factuality_report.unsupported_claims_total,
                "modified": factuality_report.modified,
                "had_external_evidence": factuality_report.had_external_evidence,
                "citations_added": factuality_report.citations_added,
                "missing_source_links": factuality_report.missing_source_links,
                "evidence_record_ids": list(factuality_report.evidence_record_ids),
                "claims": factuality_claims,
            }
            if factuality_report
            else None
        ),
    }


def _build_session_context(messages: list) -> SystemMessage | None:
    """RM-11: Extract key facts from current conversation to prevent self-contradiction.

    Scans the conversation history and extracts:
    - Decisions and commitments made by the agent
    - Key facts stated by the user
    - Actions already taken (tool calls and their outcomes)

    Returns a SystemMessage with session context, or None if conversation is too short.
    Zero LLM calls — pure text extraction.
    """
    if len(messages) < 4:
        return None

    user_facts = []
    actions_taken = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            text = msg.content if isinstance(msg.content, str) else ""
            if len(text) < 10:
                continue
            # Extract short summary of user statements (first sentence or up to 120 chars)
            snippet = text.split("\n")[0][:120].strip()
            if snippet:
                user_facts.append(snippet)

        elif isinstance(msg, AIMessage):
            # Track tool calls as actions taken (check before content filter)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    actions_taken.append(f"{tc['name']}({', '.join(f'{k}={str(v)[:40]}' for k, v in tc['args'].items())})")

        elif isinstance(msg, ToolMessage):
            text = msg.content if isinstance(msg.content, str) else ""
            if len(text) > 20:
                # Capture abbreviated tool result
                result_snippet = text[:80].strip()
                if result_snippet:
                    actions_taken.append(f"  → {result_snippet}")

    # Only build context if there's meaningful content
    if not user_facts and not actions_taken:
        return None

    parts = []
    parts.append("[SESSION CONTEXT — what already happened in this conversation]")

    if user_facts:
        # Keep last 8 user statements (most relevant)
        recent_facts = user_facts[-8:]
        parts.append("User stated:")
        for fact in recent_facts:
            parts.append(f"  - {fact}")

    if actions_taken:
        # Keep last 10 actions
        recent_actions = actions_taken[-10:]
        parts.append("Actions already taken:")
        for action in recent_actions:
            parts.append(f"  - {action}")

    parts.append("")
    parts.append("IMPORTANT: Do NOT propose actions that were already completed or contradict what was discussed above.")

    context = "\n".join(parts)
    logger.debug("Session context built: %d user facts, %d actions", len(user_facts), len(actions_taken))
    return SystemMessage(content=context)


def _extract_record_ids(text: str) -> set[str]:
    return set(re.findall(r"\[id:([^\]]+)\]", text or ""))


def _has_temporal_signal(text: str) -> bool:
    text_lower = (text or "").lower()
    return any(kw in text_lower for kw in _TEMPORAL_KEYWORDS)


def _expand_relative_dates(user_text: str) -> str:
    """Expand relative date words into absolute anchors for better recall."""
    if not user_text:
        return user_text

    from datetime import datetime, timedelta

    now = datetime.now()
    replacements = {
        "today": now.strftime("%Y-%m-%d"),
        "yesterday": (now - timedelta(days=1)).strftime("%Y-%m-%d"),
        "tomorrow": (now + timedelta(days=1)).strftime("%Y-%m-%d"),
        "last week": f"week around {(now - timedelta(days=7)).strftime('%Y-%m-%d')}",
        "this week": f"week of {now.strftime('%Y-%m-%d')}",
        "сьогодні": now.strftime("%Y-%m-%d"),
        "вчора": (now - timedelta(days=1)).strftime("%Y-%m-%d"),
        "завтра": (now + timedelta(days=1)).strftime("%Y-%m-%d"),
    }
    expanded = user_text
    for needle, replacement in replacements.items():
        expanded = re.sub(
            rf"\b{re.escape(needle)}\b",
            f"{needle} ({replacement})",
            expanded,
            flags=re.IGNORECASE,
        )
    return expanded


def _inject_context(state: AgentState) -> SystemMessage | None:
    """Build safe memory context from exact retrieval plus cognitive recall."""
    messages = state["messages"]
    if not messages:
        return None

    last_msg = messages[-1]
    if not isinstance(last_msg, HumanMessage):
        return None

    user_text = last_msg.content
    if not isinstance(user_text, str) or len(user_text) < 5:
        return None

    if len(user_text.split()) < 2:
        return None

    context_parts = []
    recall_query = _expand_relative_dates(user_text)
    existing_ids = set()
    for message in messages:
        if isinstance(message, SystemMessage):
            existing_ids |= _extract_record_ids(getattr(message, "content", "") or "")

    def _render_hits(title: str, hits: list[dict]) -> str | None:
        lines: list[str] = []
        seen_ids: set[str] = set()
        for hit in hits or []:
            hit_id = str(hit.get("id") or "").strip()
            if hit_id and (hit_id in existing_ids or hit_id in seen_ids):
                continue
            text = str(hit.get("content") or "").strip()
            if not text:
                continue
            if hit_id and f"[id:{hit_id}]" not in text:
                text = f"[id:{hit_id}] {text}"
            # Epistemic annotations — help LLM gauge reliability
            annotations = []
            conf = hit.get("confidence")
            if conf is not None and conf < 0.5:
                annotations.append("low-confidence")
            conflict = hit.get("conflict_mass")
            if conflict and int(conflict) > 0:
                annotations.append("contested")
            if annotations:
                text = f"{text} [{', '.join(annotations)}]"
            lines.append(f"- {text[:360]}")
            if hit_id:
                seen_ids.add(hit_id)
        if not lines:
            return None
        return f"[{title}]\n" + "\n".join(lines)

    # 1. Episodic memory — Phase A.8: factual turns use evidence-safe path
    try:
        from remy.core.agent_tools import brain
        from remy.core.hybrid_search import build_evidence_packet, search_exact_structured

        is_factual = _needs_factuality_contract(messages, state.get("session_log", []))
        if is_factual:
            # Factual/citation/verify turns: forbidden classes excluded, provenance attached
            exact_hits = build_evidence_packet(
                brain,
                recall_query,
                top_k=3,
                session_id=state.get("session_id"),
            )
            block_title = "Evidence"
        else:
            exact_hits = search_exact_structured(
                brain,
                recall_query,
                top_k=3,
                lexical_limit=6,
            )
            block_title = "Exact Memory"
        exact_block = _render_hits(block_title, exact_hits)
        if exact_block:
            context_parts.append(exact_block)
    except Exception as e:
        logger.debug("Structured exact recall failed: %s", e)

    # 2. Semantic KB — top 3 results. Cognitive recall is now provenance-aware
    #    (memories born from a lived consequence outrank model-generated ones);
    #    see recall_cognitive_structured. Fail-soft on older AuraSDK wheels.
    try:
        from remy.core.agent_tools import brain
        from remy.core.hybrid_search import recall_cognitive_structured

        cognitive_hits = recall_cognitive_structured(
            brain,
            recall_query,
            top_k=4,
            min_strength=0.05,
            session_id=state.get("session_id"),
        )
        cognitive_block = _render_hits("Cognitive Recall", cognitive_hits)
        if cognitive_block:
            context_parts.append(cognitive_block)
    except Exception as e:
        logger.debug("Structured cognitive recall failed: %s", e)

    # 3. Cognitive state — contradictions, patterns, epistemic caution
    if context_parts:
        try:
            from remy.core.agent_tools import brain as _brain
            from remy.core.hybrid_search import build_cognitive_context

            cog_block = build_cognitive_context(_brain)
            if cog_block:
                context_parts.append(cog_block)
        except Exception as e:
            logger.debug("Cognitive context build failed: %s", e)

    if not context_parts:
        return None

    combined = "Relevant Memory Context:\n\n" + "\n\n".join(context_parts)
    # Token budget guard: ~4800 chars max
    if len(combined) > 4800:
        combined = combined[:4800] + "\n...[truncated]"

    logger.info("Memory context injected for: %s... (%d chars)", user_text[:30], len(combined))
    return SystemMessage(content=combined)
