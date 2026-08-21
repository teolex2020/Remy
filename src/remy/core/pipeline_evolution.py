"""Turn successful repeated work into approval-gated workflow drafts."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from remy.config.settings import settings

_LOCK = threading.Lock()
_URL_RE = re.compile(r"https?://[^\s)\]>]+", re.IGNORECASE)
_WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)
_REPEAT_RE = re.compile(
    r"\b(щодня|щотижня|щогодини|кожн(?:ого|і|у)|регулярно|автоматично|"
    r"daily|weekly|hourly|every|regularly|automatically)\b", re.IGNORECASE
)
_STOP = {
    "і", "та", "а", "але", "це", "я", "ти", "мені", "для", "про", "що", "щоб",
    "the", "a", "an", "and", "or", "to", "for", "of", "me", "please", "this", "that",
}
_SIDE_EFFECT_TOOLS = {
    "store", "store_knowledge", "send_email", "send_telegram", "write_file",
    "generate_report", "generate_presentation", "schedule_task", "add_todo",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_path() -> Path:
    return settings.DATA_DIR / "pipeline_evolution.sqlite3"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def _ensure_schema() -> None:
    with _LOCK, _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS pipeline_candidates (
                candidate_id TEXT PRIMARY KEY,
                intent_key TEXT NOT NULL,
                normalized_request TEXT NOT NULL,
                title TEXT NOT NULL,
                status TEXT NOT NULL,
                occurrence_count INTEGER NOT NULL DEFAULT 1,
                session_id TEXT NOT NULL DEFAULT '',
                source_request TEXT NOT NULL,
                source_trace_json TEXT NOT NULL DEFAULT '[]',
                pipeline_json TEXT NOT NULL DEFAULT '{}',
                trigger_json TEXT NOT NULL DEFAULT '{}',
                success_criteria_json TEXT NOT NULL DEFAULT '[]',
                risk_json TEXT NOT NULL DEFAULT '{}',
                estimate_json TEXT NOT NULL DEFAULT '{}',
                dry_run_json TEXT NOT NULL DEFAULT '{}',
                pipeline_id TEXT NOT NULL DEFAULT '',
                automation_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                decided_at TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_pipeline_candidate_status
                ON pipeline_candidates(status, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_pipeline_candidate_intent
                ON pipeline_candidates(intent_key, updated_at DESC);
            """
        )


def _normalize(text: str) -> str:
    value = _URL_RE.sub(" <url> ", text.lower())
    words = [word for word in _WORD_RE.findall(value) if word not in _STOP and len(word) > 1]
    return " ".join(words[:40])


def _intent_key(normalized: str) -> str:
    stable = " ".join(sorted(set(normalized.split()))[:24])
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()[:20]


def _title(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned[:72] + ("…" if len(cleaned) > 72 else "")


def _turn_trace(session_log: list[dict]) -> list[dict]:
    start = 0
    for index, item in enumerate(session_log or []):
        if item.get("type") in {"user_text", "user_voice"}:
            start = index + 1
    return [
        {
            "tool": str(item.get("tool") or ""),
            "args": dict(item.get("args_full") or item.get("args") or {}),
            "result_preview": str(item.get("result") or "")[:500],
        }
        for item in (session_log or [])[start:]
        if item.get("type") == "tool_call" and item.get("tool")
    ]


def _infer_trigger(text: str) -> dict:
    lower = text.lower()
    time_match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", lower)
    time_of_day = time_match.group(0) if time_match else "09:00"
    if re.search(r"щогод|every\s+hour|hourly", lower):
        return {"type": "schedule", "schedule_type": "hourly", "time_of_day": time_of_day}
    if re.search(r"щотиж|кожн\w*\s+тиж|weekly|every\s+week", lower):
        return {"type": "schedule", "schedule_type": "weekly", "time_of_day": time_of_day, "day_of_week": 0}
    if re.search(r"щодня|кожн\w*\s+день|daily|every\s+day", lower):
        return {"type": "schedule", "schedule_type": "daily", "time_of_day": time_of_day}
    return {"type": "manual"}


def _tool_step(tool: str, args: dict, index: int) -> dict | None:
    step_id = f"s{index}"
    if tool == "web_search":
        return {"id": step_id, "type": "web_search", "label": "Search the web", "config": {"query": args.get("query") or "{{input}}", "num_results": 5, "fetch_content": True}}
    if tool in {"browse_page", "extract_content"} and args.get("url"):
        return {"id": step_id, "type": "page_scrape", "label": "Read source page", "config": {"url": args["url"], "mode": "text", "max_chars": 12000}}
    if tool in {"search", "search_exact", "recall", "recall_knowledge"}:
        return {"id": step_id, "type": "memory_search", "label": "Search memory", "config": {"query": args.get("query") or "{{input}}", "limit": 5}}
    if tool in {"store", "store_knowledge"}:
        return {"id": step_id, "type": "memory_save", "label": "Save verified result", "config": {"input_source": "{{prev}}", "tags": "generated-pipeline", "dedup_guard": True}}
    return None


def compile_pipeline(source_request: str, trace: list[dict]) -> dict:
    steps: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for item in trace:
        mapped = _tool_step(item.get("tool", ""), item.get("args") or {}, len(steps) + 1)
        if not mapped:
            continue
        signature = (mapped["type"], json.dumps(mapped["config"], sort_keys=True, ensure_ascii=False))
        if signature in seen:
            continue
        seen.add(signature)
        steps.append(mapped)
    prompt = (
        "Execute this recurring task reliably: " + source_request[:800] +
        "\n\nCurrent input: {{input}}\nPrevious step evidence: {{prev}}\n"
        "Return a concise result. Do not claim facts not supported by the evidence."
    )
    steps.append({"id": f"s{len(steps)+1}", "type": "llm_call", "label": "Produce verified result", "config": {"prompt": prompt, "model": ""}})
    side_effects = sorted({item.get("tool", "") for item in trace if item.get("tool") in _SIDE_EFFECT_TOOLS})
    risk_level = "high" if any(name in side_effects for name in {"send_email", "send_telegram", "write_file"}) else "medium" if side_effects else "low"
    return {
        "name": _title(source_request),
        "description": "Generated from a repeatedly successful conversation trace. Requires explicit approval.",
        "steps": steps,
        "trigger": _infer_trigger(source_request),
        "output_destination": {"type": "chat"},
        "success_criteria": [
            "Every required step completes without an error",
            "The final output is non-empty",
            *( ["Claims from the web are supported by fetched source content"] if any(s["type"] in {"web_search", "page_scrape"} for s in steps) else [] ),
        ],
        "risk": {"level": risk_level, "side_effect_tools": side_effects, "approval_required": True},
        "estimate": {
            "llm_calls_per_run": sum(1 for step in steps if step["type"] == "llm_call"),
            "external_reads_per_run": sum(1 for step in steps if step["type"] in {"web_search", "page_scrape", "http_request"}),
            "estimated_input_tokens": 1500,
        },
    }


def observe_successful_turn(
    *, session_id: str, user_text: str, session_log: list[dict], force_draft: bool = False
) -> dict | None:
    """Record a successful turn and produce a draft after a repeated intent."""
    text = str(user_text or "").strip()
    normalized = _normalize(text)
    if len(normalized) < 12 or len(normalized.split()) < 3:
        return None
    trace = _turn_trace(session_log)
    explicit_repeat = bool(_REPEAT_RE.search(text))
    # A repeated chat phrase is not a workflow. Automatic candidates require
    # evidence that Remy actually executed work (a tool trace), unless the user
    # explicitly requested recurring automation or the agent deliberately
    # proposed a draft through its approval-gated tool.
    if not trace and not explicit_repeat and not force_draft:
        return None
    _ensure_schema()
    key = _intent_key(normalized)
    became_draft = False
    with _LOCK, _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM pipeline_candidates WHERE status IN ('observing','draft','dry_run_passed') ORDER BY updated_at DESC LIMIT 200"
        ).fetchall()
        match = None
        for row in rows:
            similarity = SequenceMatcher(None, normalized, row["normalized_request"]).ratio()
            if row["intent_key"] == key or similarity >= 0.74:
                match = row
                break
        now = _now()
        if match:
            count = int(match["occurrence_count"] or 1) + 1
            status = match["status"]
            compiled = compile_pipeline(text, trace)
            if status == "observing" and (count >= 2 or explicit_repeat or force_draft):
                status = "draft"
                became_draft = True
            conn.execute(
                """UPDATE pipeline_candidates SET occurrence_count=?, status=?, session_id=?,
                    source_request=?, source_trace_json=?, pipeline_json=?, trigger_json=?,
                    success_criteria_json=?, risk_json=?, estimate_json=?, updated_at=? WHERE candidate_id=?""",
                (count, status, session_id, text, json.dumps(trace, ensure_ascii=False),
                 json.dumps({"name": compiled["name"], "description": compiled["description"], "steps": compiled["steps"]}, ensure_ascii=False),
                 json.dumps(compiled["trigger"], ensure_ascii=False), json.dumps(compiled["success_criteria"], ensure_ascii=False),
                 json.dumps(compiled["risk"], ensure_ascii=False), json.dumps(compiled["estimate"], ensure_ascii=False), now, match["candidate_id"]),
            )
            candidate_id = match["candidate_id"]
        else:
            compiled = compile_pipeline(text, trace)
            candidate_id = "pipeline-candidate-" + hashlib.sha256(f"{session_id}\0{text}".encode("utf-8")).hexdigest()[:20]
            status = "draft" if (explicit_repeat or force_draft) else "observing"
            became_draft = status == "draft"
            conn.execute(
                """INSERT INTO pipeline_candidates(candidate_id,intent_key,normalized_request,title,status,
                    session_id,source_request,source_trace_json,pipeline_json,trigger_json,success_criteria_json,
                    risk_json,estimate_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (candidate_id, key, normalized, _title(text), status, session_id, text,
                 json.dumps(trace, ensure_ascii=False),
                 json.dumps({"name": compiled["name"], "description": compiled["description"], "steps": compiled["steps"]}, ensure_ascii=False),
                 json.dumps(compiled["trigger"], ensure_ascii=False), json.dumps(compiled["success_criteria"], ensure_ascii=False),
                 json.dumps(compiled["risk"], ensure_ascii=False), json.dumps(compiled["estimate"], ensure_ascii=False), now, now),
            )
    item = get_candidate(candidate_id)
    if item and became_draft:
        try:
            from remy.core.notification_router import notify
            notify(
                f"Remy created a draft pipeline from repeated work: {item['title']}",
                level="info", event_type="pipeline.candidate",
                event_data={"candidate_id": candidate_id, "status": "draft"},
            )
        except Exception:
            pass
    return item


def _decode(row: sqlite3.Row | dict) -> dict:
    item = dict(row)
    for source, target in (
        ("source_trace_json", "source_trace"), ("pipeline_json", "pipeline"),
        ("trigger_json", "trigger"), ("success_criteria_json", "success_criteria"),
        ("risk_json", "risk"), ("estimate_json", "estimate"), ("dry_run_json", "dry_run"),
    ):
        item[target] = json.loads(item.pop(source, "{}") or ("[]" if target in {"source_trace", "success_criteria"} else "{}"))
    return item


def get_candidate(candidate_id: str) -> dict | None:
    _ensure_schema()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM pipeline_candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
    return _decode(row) if row else None


def list_candidates(*, status: str = "draft", limit: int = 100) -> list[dict]:
    _ensure_schema()
    sql = "SELECT * FROM pipeline_candidates"
    args: list[Any] = []
    if status and status != "all":
        sql += " WHERE status=?"
        args.append(status)
    sql += " ORDER BY updated_at DESC LIMIT ?"
    args.append(max(1, min(int(limit), 500)))
    with _connect() as conn:
        return [_decode(row) for row in conn.execute(sql, args).fetchall()]


def record_dry_run(candidate_id: str, result: dict, *, passed: bool) -> dict | None:
    _ensure_schema()
    with _LOCK, _connect() as conn:
        conn.execute(
            "UPDATE pipeline_candidates SET status=?, dry_run_json=?, updated_at=? WHERE candidate_id=? AND status IN ('draft','dry_run_passed')",
            ("dry_run_passed" if passed else "draft", json.dumps(result, ensure_ascii=False), _now(), candidate_id),
        )
    return get_candidate(candidate_id)


def decide_candidate(candidate_id: str, *, status: str, pipeline_id: str = "", automation_id: str = "") -> dict | None:
    if status not in {"approved", "rejected", "activated"}:
        raise ValueError("Unsupported decision")
    _ensure_schema()
    with _LOCK, _connect() as conn:
        conn.execute(
            "UPDATE pipeline_candidates SET status=?, pipeline_id=?, automation_id=?, decided_at=?, updated_at=? WHERE candidate_id=?",
            (status, pipeline_id, automation_id, _now(), _now(), candidate_id),
        )
    return get_candidate(candidate_id)
