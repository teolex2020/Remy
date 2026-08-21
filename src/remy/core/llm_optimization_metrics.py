"""Persistent measurements for the LLM optimization pipeline.

Stores product metrics and compact answer-memory records for the optimization
pipeline. Answer memory is conservative: unsafe answers never store answer text,
and stored records remain product cache/audit data, not durable domain facts.
"""

from __future__ import annotations

import json
import threading
import re
import time
from pathlib import Path
from typing import Any

from remy.config.settings import settings

_LOCK = threading.Lock()
MAX_STORED_RUNS = 500


def _question_key(text: str) -> str:
    return " ".join(re.findall(r"[\w\-]+", (text or "").lower(), re.UNICODE))


def _store_dir() -> Path:
    path = settings.DATA_DIR / "llm_optimization"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _store_path() -> Path:
    return _store_dir() / "measurements.jsonl"


def _answer_memory_path() -> Path:
    return _store_dir() / "answer_memory.jsonl"


def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
    raw = dict(report.get("raw") or {})
    reduced = dict(report.get("reduced") or {})
    for payload in (raw, reduced):
        answer = str(payload.get("answer") or "")
        preview = str(payload.get("answer_preview") or answer[:600])[:600]
        payload["answer_preview"] = preview
        payload["answer_truncated"] = bool(answer and len(answer) > len(preview))
        payload.pop("answer", None)

    return {
        "schema": "remy_llm_optimization_measurement_v1",
        "created_at": report.get("created_at") or time.time(),
        "user_text_preview": str(report.get("user_text_preview") or "")[:240],
        "raw": raw,
        "reduced": reduced,
        "delta": dict(report.get("delta") or {}),
        "blocks": dict(report.get("blocks") or {}),
        "claims": dict(report.get("claims") or {}),
    }


def _compact_answer_memory(report: dict[str, Any]) -> dict[str, Any]:
    blocks = dict(report.get("blocks") or {})
    verifier = dict(blocks.get("verifier") or {})
    reduced_verifier = dict(verifier.get("reduced") or {})
    reduced = dict(report.get("reduced") or {})
    claims = dict(report.get("claims") or {})
    blocked = bool(verifier.get("blocked") or reduced_verifier.get("brain_storage_unsafe"))
    answer = str(reduced.get("answer") or report.get("answer") or "")
    answer_preview = "" if blocked else str(reduced.get("answer_preview") or answer[:600])[:600]
    evidence_ids = reduced_verifier.get("evidence_record_ids") or []
    if not isinstance(evidence_ids, list):
        evidence_ids = []
    return {
        "schema": "remy_llm_optimization_answer_memory_v1",
        "created_at": report.get("created_at") or time.time(),
        "question": str(report.get("user_text_preview") or "")[:500],
        "mode": (
            "memory_only" if claims.get("memory_only_cache_hit")
            else "apply" if claims.get("apply_context_reducer")
            else "compare"
        ),
        "answer_preview": answer_preview,
        "answer_text_stored": bool(answer_preview),
        "unsafe_answer_text_suppressed": blocked,
        "sources": {
            "evidence_record_ids": [str(x)[:120] for x in evidence_ids[:20]],
            "external_citations_total": int(reduced_verifier.get("external_citations_total") or 0),
            "external_citations_grounded": int(reduced_verifier.get("external_citations_grounded") or 0),
            "external_citations_phantom": int(reduced_verifier.get("external_citations_phantom") or 0),
        },
        "confirmed": {
            "supported_claims_total": int(reduced_verifier.get("supported_claims_total") or 0),
            "supported_internal": int(reduced_verifier.get("supported_internal") or 0),
            "supported_external_verified": int(reduced_verifier.get("supported_external_verified") or 0),
        },
        "mistakes": {
            "unsupported_claims_total": int(reduced_verifier.get("unsupported_claims_total") or 0),
            "unverified_current_claims": int(reduced_verifier.get("unverified_current_claims") or 0),
            "unverified_external": int(reduced_verifier.get("unverified_external") or 0),
            "unsupported": int(reduced_verifier.get("unsupported") or 0),
            "verifier_modified_answer": bool(reduced.get("verifier_modified_answer") or claims.get("verifier_modified_answer")),
            "blocked": blocked,
        },
    }


def append_answer_memory(report: dict[str, Any]) -> dict[str, Any]:
    item = _compact_answer_memory(report)
    line = json.dumps(item, ensure_ascii=False, sort_keys=True)
    with _LOCK:
        path = _answer_memory_path()
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        _trim_locked(path)
    return item


def list_answer_memory(limit: int = 100) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit or 100), MAX_STORED_RUNS))
    path = _answer_memory_path()
    if not path.exists():
        return []
    with _LOCK:
        lines = path.read_text(encoding="utf-8").splitlines()
    items: list[dict[str, Any]] = []
    for line in reversed(lines[-limit:]):
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items


def append_measurement(report: dict[str, Any]) -> dict[str, Any]:
    item = _compact_report(report)
    line = json.dumps(item, ensure_ascii=False, sort_keys=True)
    with _LOCK:
        path = _store_path()
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        _trim_locked(path)
    return item


def _trim_locked(path: Path) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return
    if len(lines) <= MAX_STORED_RUNS:
        return
    path.write_text("\n".join(lines[-MAX_STORED_RUNS:]) + "\n", encoding="utf-8")


def list_measurements(limit: int = 100) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit or 100), MAX_STORED_RUNS))
    path = _store_path()
    if not path.exists():
        return []
    with _LOCK:
        lines = path.read_text(encoding="utf-8").splitlines()
    items: list[dict[str, Any]] = []
    for line in reversed(lines[-limit:]):
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items


def find_memory_only_answer(user_text: str, *, limit: int = MAX_STORED_RUNS) -> dict[str, Any] | None:
    """Return a conservative exact-match answer from MemoryWriter output.

    This is a product optimization cache, not durable factual memory. It first
    reads answer_memory.jsonl and only falls back to older measurement records
    for backward compatibility.
    """
    key = _question_key(user_text)
    if not key:
        return None

    for item in list_answer_memory(limit=limit):
        if _question_key(str(item.get("question") or "")) != key:
            continue
        mistakes = dict(item.get("mistakes") or {})
        answer = str(item.get("answer_preview") or "").strip()
        if (
            not answer
            or not item.get("answer_text_stored")
            or item.get("unsafe_answer_text_suppressed")
            or mistakes.get("blocked")
        ):
            continue
        return {
            "answer": answer,
            "created_at": item.get("created_at"),
            "source_schema": item.get("schema"),
            "source_store": "answer_memory",
            "source_user_text_preview": item.get("question"),
            "source_reduced_prompt_tokens_estimate": None,
        }

    for item in list_measurements(limit=limit):
        if _question_key(str(item.get("user_text_preview") or "")) != key:
            continue
        reduced = dict(item.get("reduced") or {})
        answer = str(reduced.get("answer_preview") or "").strip()
        if not answer or reduced.get("answer_truncated"):
            continue
        blocks = dict(item.get("blocks") or {})
        verifier = dict(blocks.get("verifier") or {})
        reduced_verifier = dict(verifier.get("reduced") or {})
        if verifier.get("blocked") or reduced_verifier.get("brain_storage_unsafe"):
            continue
        return {
            "answer": answer,
            "created_at": item.get("created_at"),
            "source_schema": item.get("schema"),
            "source_store": "measurement_fallback",
            "source_user_text_preview": item.get("user_text_preview"),
            "source_reduced_prompt_tokens_estimate": reduced.get("prompt_tokens_estimate"),
        }
    return None


def clear_measurements() -> None:
    with _LOCK:
        path = _store_path()
        if path.exists():
            path.write_text("", encoding="utf-8")
        answer_path = _answer_memory_path()
        if answer_path.exists():
            answer_path.write_text("", encoding="utf-8")


def summarize_measurements(items: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(items)
    if not count:
        return {
            "count": 0,
            "token_saving_wins": 0,
            "five_block_reports": 0,
            "compare_runs": 0,
            "apply_runs": 0,
            "memory_only_hits": 0,
            "llm_calls_saved": 0,
            "answer_memory_writes": 0,
            "router_decisions": {},
            "avg_tokens_saved": 0,
            "avg_token_ratio": 0.0,
            "avg_latency_saved_seconds": 0.0,
            "wrong_answers_avoided_estimate": 0,
        }
    saved_total = 0.0
    ratio_total = 0.0
    latency_total = 0.0
    wins = 0
    five_block = 0
    wrong_avoided = 0
    compare_runs = 0
    apply_runs = 0
    memory_only_hits = 0
    llm_calls_saved = 0
    answer_memory_writes = 0
    router_decisions: dict[str, int] = {}
    for item in items:
        delta = item.get("delta") or {}
        saved = float(delta.get("prompt_tokens_saved_estimate") or 0)
        ratio = float(delta.get("prompt_token_reduction_ratio") or 0)
        latency = float(delta.get("latency_saved_seconds") or 0)
        saved_total += saved
        ratio_total += ratio
        latency_total += latency
        wins += 1 if saved > 0 else 0
        five_block += 1 if (item.get("claims") or {}).get("five_block_pipeline_report") else 0
        wrong_avoided += int(delta.get("wrong_answers_avoided_estimate") or 0)
        claims = item.get("claims") or {}
        compare_runs += 1 if claims.get("measured_ab_comparison") else 0
        apply_runs += 1 if claims.get("apply_context_reducer") else 0
        memory_only_hits += 1 if claims.get("memory_only_cache_hit") else 0
        llm_calls_saved += int(delta.get("llm_calls_saved") or (1 if claims.get("memory_only_cache_hit") else 0))
        blocks = item.get("blocks") or {}
        writer = (blocks.get("memory_writer") or {})
        answer_memory_writes += 1 if writer.get("answer_memory_written") else 0
        decision = str((blocks.get("model_router") or {}).get("decision") or "unknown")
        router_decisions[decision] = router_decisions.get(decision, 0) + 1
    return {
        "count": count,
        "token_saving_wins": wins,
        "five_block_reports": five_block,
        "compare_runs": compare_runs,
        "apply_runs": apply_runs,
        "memory_only_hits": memory_only_hits,
        "llm_calls_saved": llm_calls_saved,
        "answer_memory_writes": answer_memory_writes,
        "router_decisions": router_decisions,
        "avg_tokens_saved": round(saved_total / count, 2),
        "avg_token_ratio": round(ratio_total / count, 2),
        "avg_latency_saved_seconds": round(latency_total / count, 4),
        "wrong_answers_avoided_estimate": wrong_avoided,
    }
