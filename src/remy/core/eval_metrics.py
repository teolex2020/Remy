"""
Evaluation Metrics - zero-LLM quality tracking for agent responses.

Tracks tool efficiency, memory utilization, response size, and latency.
Stored in data/eval_metrics.jsonl (not in brain - avoids memory pollution).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

from remy.config.settings import settings

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-zA-Zа-яА-ЯіїєґІЇЄҐ0-9]{3,}")
_RECALL_LINE_RE = re.compile(r"^\[[^\]]+\]\s*(.+)$")
_TRAILING_TAG_RE = re.compile(r"\s*\[[^\]]+\]\s*$")
_STOP_WORDS = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "have", "your",
        "about", "into", "what", "when", "where", "will", "just", "then",
        "user", "agent", "notes", "note", "summary", "working", "memory",
        "про", "для", "щоб", "коли", "після", "його", "її", "було", "бути",
        "якщо", "цей", "ця", "ці", "так", "але", "або", "про", "мене",
    }
)


@dataclass
class ResponseMetrics:
    """Metrics for a single agent response. Zero LLM calls."""

    session_id: str
    channel: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    response_length: int = 0
    response_word_count: int = 0
    tools_called: int = 0
    tools_succeeded: int = 0
    tools_failed: int = 0
    unique_tools: list[str] = field(default_factory=list)
    context_injected: bool = False
    recall_used: bool = False
    store_used: bool = False
    total_duration_ms: int = 0
    llm_calls: int = 0
    store_attempts: int = 0
    duplicate_store_count: int = 0
    duplicate_store_rate: float = 0.0
    recalled_items: int = 0
    recalled_items_hit: int = 0
    recall_hit_rate: float = 0.0
    scratchpad_compression_ratio: float = 0.0
    working_memory_total: int = 0
    working_memory_active: int = 0
    working_memory_bloat: int = 0
    unsupported_observed_claims: int = 0
    recall_calls: int = 0
    empty_recall_count: int = 0
    empty_recall_rate: float = 0.0


def snapshot_recall_stats() -> dict:
    """Snapshot the brain's cumulative recall/search hit counters.

    Backed by AuraSDK's ``recall_hit_stats()`` (counters are process-wide and
    cumulative). Callers snapshot before an invocation and pass the result to
    ``compute_response_metrics`` so a per-response empty-recall rate can be
    derived as the delta. Fail-soft: returns ``{}`` if the installed SDK
    predates ``recall_hit_stats`` (≤1.5.4) or the brain is unavailable, in
    which case the empty-recall metric is simply omitted.
    """
    try:
        from remy.core.agent_tools import brain

        fn = getattr(brain, "recall_hit_stats", None)
        if fn is None:
            return {}
        return dict(fn() or {})
    except Exception as e:
        logger.debug("recall_hit_stats snapshot unavailable: %s", e)
        return {}


def _recall_stats_delta(before: dict | None) -> dict | None:
    """Delta of recall/search counters between a prior snapshot and now.

    Returns None when telemetry is unavailable (missing snapshot or SDK
    support), so callers can distinguish "no recalls this turn" (a real 0)
    from "cannot measure" (omit the field).
    """
    if not before:
        return None
    after = snapshot_recall_stats()
    if not after:
        return None
    return {k: int(after.get(k, 0)) - int(before.get(k, 0)) for k in after}


def compute_response_metrics(
    session_id: str,
    channel: str,
    messages: list,
    session_log: list,
    response_text: str,
    duration_ms: int = 0,
    context_injected: bool = False,
    unsupported_observed_claims: int = 0,
    recall_stats_before: dict | None = None,
) -> ResponseMetrics:
    """Compute metrics from a completed agent invocation. Zero LLM calls."""
    from langchain_core.messages import AIMessage
    from remy.core.scratchpad import get_scratchpad_metrics

    metrics = ResponseMetrics(
        session_id=session_id,
        channel=channel,
        response_length=len(response_text),
        response_word_count=len(response_text.split()),
        total_duration_ms=duration_ms,
        context_injected=context_injected,
        unsupported_observed_claims=unsupported_observed_claims,
    )

    tool_names: list[str] = []
    recalled_candidates: list[str] = []
    for entry in session_log:
        if not isinstance(entry, dict) or entry.get("type") != "tool_call":
            continue

        tool_name = entry.get("tool", "")
        tool_names.append(tool_name)
        result_text = str(entry.get("result", ""))
        if result_text.startswith("Error:") or '"error"' in result_text[:100]:
            metrics.tools_failed += 1
        else:
            metrics.tools_succeeded += 1

        if tool_name.startswith("store"):
            metrics.store_attempts += 1
            if "similar_existing" in result_text:
                metrics.duplicate_store_count += 1

        if tool_name == "recall":
            recalled_candidates.extend(_extract_recall_candidates(result_text))

    metrics.tools_called = len(tool_names)
    metrics.unique_tools = list(set(tool_names))
    metrics.recall_used = "recall" in tool_names or "recall_knowledge" in tool_names
    metrics.store_used = any(t.startswith("store") for t in tool_names)
    metrics.llm_calls = sum(1 for m in messages if isinstance(m, AIMessage))
    if metrics.store_attempts:
        metrics.duplicate_store_rate = round(
            metrics.duplicate_store_count / metrics.store_attempts * 100, 1
        )

    deduped_candidates = []
    seen = set()
    for item in recalled_candidates:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped_candidates.append(item)
    metrics.recalled_items = len(deduped_candidates)
    metrics.recalled_items_hit = sum(
        1 for item in deduped_candidates if _response_mentions(item, response_text)
    )
    if metrics.recalled_items:
        metrics.recall_hit_rate = round(
            metrics.recalled_items_hit / metrics.recalled_items * 100, 1
        )

    scratchpad_meta = get_scratchpad_metrics(session_id)
    metrics.scratchpad_compression_ratio = float(
        scratchpad_meta.get("scratchpad_compression_ratio", 0.0) or 0.0
    )
    metrics.working_memory_total = int(
        scratchpad_meta.get("scratchpad_working_total", scratchpad_meta.get("working_memory_total", 0)) or 0
    )
    metrics.working_memory_active = int(
        scratchpad_meta.get("scratchpad_working_active", scratchpad_meta.get("working_memory_active", 0)) or 0
    )
    metrics.working_memory_bloat = int(
        scratchpad_meta.get("scratchpad_working_bloat", scratchpad_meta.get("working_memory_bloat", 0)) or 0
    )

    # Empty-recall telemetry: SDK-counted recalls that returned zero records
    # this turn. Distinct from recall_hit_rate (did the model USE what it
    # recalled) — this is "did recall find anything at all". Only set when the
    # SDK exposes the counters and a before-snapshot was provided.
    delta = _recall_stats_delta(recall_stats_before)
    if delta is not None:
        metrics.recall_calls = max(0, delta.get("recall_total", 0))
        metrics.empty_recall_count = max(0, delta.get("recall_empty", 0))
        if metrics.recall_calls:
            metrics.empty_recall_rate = round(
                metrics.empty_recall_count / metrics.recall_calls * 100, 1
            )

    return metrics


def _rotate_metrics_file(metrics_file) -> None:
    """Keep only the last 3 days of entries. Trims in-place."""
    try:
        from datetime import datetime, timedelta
        cutoff = (datetime.now() - timedelta(days=3)).isoformat()
        lines = metrics_file.read_text(encoding="utf-8").strip().split("\n")
        kept = []
        for line in lines:
            if not line.strip():
                continue
            try:
                ts = json.loads(line).get("timestamp", "")
                if ts >= cutoff:
                    kept.append(line)
            except json.JSONDecodeError:
                pass
        metrics_file.write_text("\n".join(kept) + "\n", encoding="utf-8")
    except Exception as e:
        logger.debug("Metrics rotation failed: %s", e)


_metrics_write_count = 0
_ROTATE_EVERY_N = 50  # rotate at most once every 50 writes


def _metrics_file_path():
    return settings.DATA_DIR / "eval_metrics.jsonl"


def store_eval_metrics(metrics: ResponseMetrics) -> None:
    """Store metrics to a JSON file (not brain - avoids memory pollution)."""
    global _metrics_write_count
    try:
        metrics_file = _metrics_file_path()
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "session_id": metrics.session_id,
            "channel": metrics.channel,
            "timestamp": metrics.timestamp,
            "response_length": metrics.response_length,
            "response_word_count": metrics.response_word_count,
            "tools_called": metrics.tools_called,
            "tools_succeeded": metrics.tools_succeeded,
            "tools_failed": metrics.tools_failed,
            "unique_tools": metrics.unique_tools,
            "context_injected": metrics.context_injected,
            "recall_used": metrics.recall_used,
            "store_used": metrics.store_used,
            "total_duration_ms": metrics.total_duration_ms,
            "llm_calls": metrics.llm_calls,
            "store_attempts": metrics.store_attempts,
            "duplicate_store_count": metrics.duplicate_store_count,
            "duplicate_store_rate": metrics.duplicate_store_rate,
            "recalled_items": metrics.recalled_items,
            "recalled_items_hit": metrics.recalled_items_hit,
            "recall_hit_rate": metrics.recall_hit_rate,
            "scratchpad_compression_ratio": metrics.scratchpad_compression_ratio,
            "working_memory_total": metrics.working_memory_total,
            "working_memory_active": metrics.working_memory_active,
            "working_memory_bloat": metrics.working_memory_bloat,
            "unsupported_observed_claims": metrics.unsupported_observed_claims,
            "recall_calls": metrics.recall_calls,
            "empty_recall_count": metrics.empty_recall_count,
            "empty_recall_rate": metrics.empty_recall_rate,
        }
        with open(metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        _metrics_write_count += 1
        if _metrics_write_count % _ROTATE_EVERY_N == 0:
            _rotate_metrics_file(metrics_file)
    except Exception as e:
        logger.debug("Failed to store eval metrics: %s", e)


def get_metrics_summary(channel: str | None = None, limit: int = 50) -> dict:
    """Aggregate recent evaluation metrics from JSONL file. Zero LLM calls."""
    try:
        metrics_file = _metrics_file_path()
        if not metrics_file.exists():
            return {"total_responses": 0}

        lines = metrics_file.read_text(encoding="utf-8").strip().split("\n")
        entries = []
        for line in lines[-limit:]:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if channel and entry.get("channel") != channel:
                    continue
                entries.append(entry)
            except json.JSONDecodeError:
                continue

        if not entries:
            return {"total_responses": 0}

        total = len(entries)
        total_tools = 0
        total_tools_ok = 0
        total_tools_fail = 0
        total_words = 0
        recall_count = 0
        store_count = 0
        context_count = 0
        total_duration = 0
        total_llm = 0
        total_dup_rate = 0.0
        total_recall_hit_rate = 0.0
        total_compression = 0.0
        total_working_bloat = 0
        total_unsupported_claims = 0
        total_recall_calls = 0
        total_empty_recalls = 0

        for meta in entries:
            total_tools += meta.get("tools_called", 0)
            total_tools_ok += meta.get("tools_succeeded", 0)
            total_tools_fail += meta.get("tools_failed", 0)
            total_words += meta.get("response_word_count", 0)
            total_dup_rate += float(meta.get("duplicate_store_rate", 0.0) or 0.0)
            total_recall_hit_rate += float(meta.get("recall_hit_rate", 0.0) or 0.0)
            total_compression += float(meta.get("scratchpad_compression_ratio", 0.0) or 0.0)
            total_working_bloat += int(meta.get("working_memory_bloat", 0) or 0)
            total_unsupported_claims += int(meta.get("unsupported_observed_claims", 0) or 0)
            total_recall_calls += int(meta.get("recall_calls", 0) or 0)
            total_empty_recalls += int(meta.get("empty_recall_count", 0) or 0)
            if meta.get("recall_used"):
                recall_count += 1
            if meta.get("store_used"):
                store_count += 1
            if meta.get("context_injected"):
                context_count += 1
            total_duration += meta.get("total_duration_ms", 0)
            total_llm += meta.get("llm_calls", 0)

        return {
            "total_responses": total,
            "avg_word_count": round(total_words / total, 1),
            "avg_tools_per_response": round(total_tools / total, 2),
            "tool_success_rate": round(total_tools_ok / max(total_tools, 1) * 100, 1),
            "tool_failure_rate": round(total_tools_fail / max(total_tools, 1) * 100, 1),
            "recall_usage_rate": round(recall_count / total * 100, 1),
            "store_usage_rate": round(store_count / total * 100, 1),
            "context_injection_rate": round(context_count / total * 100, 1),
            "avg_duration_ms": round(total_duration / total),
            "avg_llm_calls": round(total_llm / total, 1),
            "avg_duplicate_store_rate": round(total_dup_rate / total, 1),
            "avg_recall_hit_rate": round(total_recall_hit_rate / total, 1),
            "avg_scratchpad_compression_ratio": round(total_compression / total, 3),
            "avg_working_memory_bloat": round(total_working_bloat / total, 1),
            "unsupported_observed_claims_total": total_unsupported_claims,
            "recall_calls_total": total_recall_calls,
            "empty_recall_count_total": total_empty_recalls,
            "empty_recall_rate": (
                round(total_empty_recalls / total_recall_calls * 100, 1)
                if total_recall_calls
                else 0.0
            ),
        }
    except Exception as e:
        logger.debug("Metrics summary failed: %s", e)
        return {"total_responses": 0, "error": str(e)}


def _extract_recall_candidates(result_text: str) -> list[str]:
    items = []
    for line in result_text.splitlines():
        line = line.strip()
        if not line or line.startswith("No relevant"):
            continue
        match = _RECALL_LINE_RE.match(line)
        if not match:
            continue
        text = _TRAILING_TAG_RE.sub("", match.group(1)).strip()
        if len(text) >= 12:
            items.append(text[:180])
    return items


def _response_mentions(candidate: str, response_text: str) -> bool:
    response_norm = _normalize_text(response_text)
    candidate_norm = _normalize_text(candidate)
    if not response_norm or not candidate_norm:
        return False
    if candidate_norm in response_norm:
        return True

    candidate_words = [
        word for word in _WORD_RE.findall(candidate.lower()) if word not in _STOP_WORDS
    ]
    if not candidate_words:
        return False
    hits = sum(1 for word in candidate_words[:6] if word in response_norm)
    needed = 2 if len(candidate_words) >= 3 else 1
    return hits >= needed


def _normalize_text(text: str) -> str:
    words = [word for word in _WORD_RE.findall((text or "").lower()) if word not in _STOP_WORDS]
    return " ".join(words)
