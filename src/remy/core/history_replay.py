from __future__ import annotations

import json
import logging
from pathlib import Path

from remy.config.settings import settings

logger = logging.getLogger("HistoryReplay")

SAFE_REPLAY_TOOLS = {
    "store",
    "store_person",
    "store_research",
    "store_story",
    "store_user_profile",
    "schedule_task",
}


def _candidate_id(source_file: str, entry_index: int, tool: str) -> str:
    return f"{source_file}:{entry_index}:{tool}"


def history_files(history_dir: Path | None = None) -> list[Path]:
    base = history_dir or (settings.DATA_DIR / "history")
    return sorted(base.glob("*.json")) if base.exists() else []


def load_history_log(path: Path) -> list[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    entries = payload.get("log")
    return entries if isinstance(entries, list) else []


def tool_call_succeeded(result) -> bool:
    if result is None:
        return False
    if isinstance(result, str):
        text = result.strip()
        if not text or text.startswith("Error:") or "validation error for" in text.lower():
            return False
        try:
            parsed = json.loads(text)
        except Exception:
            return True
        return not (isinstance(parsed, dict) and parsed.get("error"))
    if isinstance(result, dict):
        return not bool(result.get("error"))
    return True


def normalize_replay_args(tool: str, args: dict) -> dict:
    normalized = dict(args or {})
    if tool == "store_person":
        if "relation" in normalized and "role" not in normalized:
            normalized["role"] = normalized.pop("relation")
        if "birthday" in normalized and "birth_date" not in normalized:
            normalized["birth_date"] = normalized.pop("birthday")
    return normalized


def iter_history_tool_calls(history_dir: Path | None = None):
    for history_file in history_files(history_dir):
        entries = load_history_log(history_file)
        for entry_index, entry in enumerate(entries):
            if entry.get("type") != "tool_call":
                continue
            tool = str(entry.get("tool") or "").strip()
            if tool not in SAFE_REPLAY_TOOLS:
                continue
            yield {
                "source_file": history_file.name,
                "entry_index": entry_index,
                "candidate_id": _candidate_id(history_file.name, entry_index, tool),
                "tool": tool,
                "timestamp": str(entry.get("timestamp") or ""),
                "args": normalize_replay_args(tool, entry.get("args") or {}),
                "result": entry.get("result"),
            }


def _clip(value: str, limit: int = 180) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _profile_snapshot(search_fn) -> dict:
    try:
        from remy.core.brain_tools import get_user_profile_record
        rec = get_user_profile_record()
    except Exception:
        rec = None
    if not rec:
        return {}
    return dict(getattr(rec, "metadata", {}) or {})


def _list_existing_people(search_fn) -> list[dict]:
    try:
        people = search_fn(query="", tags=["person"], limit=50) or []
    except Exception:
        people = []
    results = []
    for rec in people:
        meta = dict(getattr(rec, "metadata", {}) or {})
        results.append(
            {
                "id": getattr(rec, "id", None),
                "name": str(meta.get("full_name") or meta.get("name") or "").strip(),
                "birth_date": str(meta.get("birth_date") or "").strip(),
                "aliases": [str(a).strip() for a in (meta.get("aliases") or []) if str(a).strip()],
                "content": str(getattr(rec, "content", "") or ""),
            }
        )
    return results


def _person_exists(existing_people: list[dict], name: str, birth_date: str = "") -> bool:
    norm_name = name.casefold().strip()
    norm_birth = birth_date.strip()
    if not norm_name:
        return False
    for person in existing_people:
        candidates = [person.get("name", ""), *person.get("aliases", [])]
        if norm_name in {str(c).casefold().strip() for c in candidates if c}:
            if not norm_birth or not person.get("birth_date") or person.get("birth_date") == norm_birth:
                return True
    return False


def _task_exists(search_fn, description: str) -> bool:
    query = description.strip()
    if not query:
        return False
    try:
        tasks = search_fn(query=query, tags=["scheduled-task"], limit=10) or []
    except Exception:
        tasks = []
    lowered = query.casefold()
    for rec in tasks:
        content = str(getattr(rec, "content", "") or "").casefold()
        if lowered in content:
            return True
    return False


def _topic_exists(search_fn, topic: str) -> bool:
    query = topic.strip()
    if not query:
        return False
    try:
        results = search_fn(query=query, limit=10) or []
    except Exception:
        results = []
    lowered = query.casefold()
    for rec in results:
        content = str(getattr(rec, "content", "") or "").casefold()
        tags = {str(tag).casefold() for tag in (getattr(rec, "tags", []) or [])}
        if lowered in content:
            return True
        if "research-project" in tags or "research-finding" in tags or "research-report" in tags:
            return True
    return False


def analyze_history_memory_gaps(
    search_fn,
    *,
    history_dir: Path | None = None,
    sample_limit: int = 12,
) -> dict:
    profile = _profile_snapshot(search_fn)
    people = _list_existing_people(search_fn)
    try:
        current_records = len(search_fn(query="", limit=50) or [])
    except Exception:
        current_records = 0

    report = {
        "files": 0,
        "entries": 0,
        "tool_calls_seen": 0,
        "successful_memory_writes": 0,
        "current_brain_records_sampled": current_records,
        "current_profile_fields": sorted(key for key, value in profile.items() if value),
        "current_people_count": len(people),
        "missing_candidates_count": 0,
        "review_candidates_count": 0,
        "missing_by_tool": {},
        "recent_missing": [],
        "review_candidates": [],
        "recommended_actions": [],
    }

    missing: list[dict] = []
    review: list[dict] = []

    seen_files: set[str] = set()
    for item in iter_history_tool_calls(history_dir):
        seen_files.add(item["source_file"])
        report["entries"] += 1
        report["tool_calls_seen"] += 1
        if not tool_call_succeeded(item.get("result")):
            continue
        report["successful_memory_writes"] += 1

        tool = item["tool"]
        args = item["args"]
        timestamp = item["timestamp"]
        candidate_id = item["candidate_id"]
        source_file = item["source_file"]

        if tool == "store_user_profile":
            missing_fields = [
                key for key, value in args.items()
                if value and not profile.get(key)
            ]
            if missing_fields:
                missing.append(
                    {
                        "candidate_id": candidate_id,
                        "tool": tool,
                        "label": f"profile fields: {', '.join(missing_fields)}",
                        "reason": "fields appear in history but are missing in current profile",
                        "timestamp": timestamp,
                        "source_file": source_file,
                    }
                )
            continue

        if tool == "store_person":
            name = str(args.get("full_name") or args.get("name") or "").strip()
            birth_date = str(args.get("birth_date") or "").strip()
            if name and not _person_exists(people, name, birth_date):
                missing.append(
                    {
                        "candidate_id": candidate_id,
                        "tool": tool,
                        "label": name,
                        "reason": "person exists in history but no matching person is present in current memory",
                        "timestamp": timestamp,
                        "source_file": source_file,
                    }
                )
            continue

        if tool == "schedule_task":
            description = str(args.get("description") or args.get("task") or args.get("title") or "").strip()
            if description and not _task_exists(search_fn, description):
                missing.append(
                    {
                        "candidate_id": candidate_id,
                        "tool": tool,
                        "label": _clip(description, 90),
                        "reason": "scheduled task appears in history but not in current active memory",
                        "timestamp": timestamp,
                        "source_file": source_file,
                    }
                )
            continue

        if tool == "store_research":
            topic = str(
                args.get("topic")
                or args.get("project_name")
                or args.get("title")
                or args.get("subject")
                or ""
            ).strip()
            if topic and not _topic_exists(search_fn, topic):
                missing.append(
                    {
                        "candidate_id": candidate_id,
                        "tool": tool,
                        "label": topic,
                        "reason": "research artifact appears in history but not in current active memory",
                        "timestamp": timestamp,
                        "source_file": source_file,
                    }
                )
            review.append(
                {
                    "candidate_id": candidate_id,
                    "tool": tool,
                    "label": topic or "research finding",
                    "summary": _clip(args.get("summary") or args.get("content") or args.get("report") or ""),
                    "timestamp": timestamp,
                }
            )
            continue

        if tool in {"store", "store_story"}:
            label = str(args.get("title") or args.get("content") or "").strip()
            review.append(
                {
                    "candidate_id": candidate_id,
                    "tool": tool,
                    "label": _clip(label, 90) or tool,
                    "summary": _clip(args.get("content") or args.get("title") or ""),
                    "timestamp": timestamp,
                }
            )

    report["files"] = len(seen_files)

    report["missing_candidates_count"] = len(missing)
    report["review_candidates_count"] = len(review)
    report["recent_missing"] = missing[:sample_limit]
    report["review_candidates"] = review[:sample_limit]

    missing_by_tool: dict[str, int] = {}
    for item in missing:
        missing_by_tool[item["tool"]] = missing_by_tool.get(item["tool"], 0) + 1
    report["missing_by_tool"] = missing_by_tool

    actions = []
    if missing:
        actions.append("Replay or selectively reconstruct missing memory from history.")
    if review:
        actions.append("Re-read review candidates to promote durable facts, people, tasks, and research artifacts.")
    if not missing and not review:
        actions.append("No obvious history gaps detected in the sampled active brain.")
    report["recommended_actions"] = actions
    return report


def reconstruct_history_candidates(
    execute_fn,
    *,
    candidate_ids: list[str],
    history_dir: Path | None = None,
) -> dict:
    from remy.core.verification_gate import run_reconstruction_verification_gate

    requested = {str(item).strip() for item in (candidate_ids or []) if str(item).strip()}
    stats = {
        "requested": len(requested),
        "applied": 0,
        "skipped": 0,
        "applied_candidate_ids": [],
        "skipped_candidate_ids": [],
        "missing_candidate_ids": [],
        "tool_errors": [],
    }
    if not requested:
        stats["verification"] = run_reconstruction_verification_gate(requested=0).to_dict()
        return stats

    matched = 0
    for item in iter_history_tool_calls(history_dir):
        candidate_id = item["candidate_id"]
        if candidate_id not in requested:
            continue
        matched += 1
        if not tool_call_succeeded(item.get("result")):
            stats["skipped"] += 1
            stats["skipped_candidate_ids"].append(candidate_id)
            continue
        result = execute_fn(item["tool"], item["args"])
        if tool_call_succeeded(result):
            stats["applied"] += 1
            stats["applied_candidate_ids"].append(candidate_id)
        else:
            stats["skipped"] += 1
            stats["skipped_candidate_ids"].append(candidate_id)
            stats["tool_errors"].append(
                {
                    "candidate_id": candidate_id,
                    "tool": item["tool"],
                    "result": str(result)[:400],
                }
            )

    if matched < len(requested):
        known_ids = {item["candidate_id"] for item in iter_history_tool_calls(history_dir)}
        stats["missing_candidate_ids"] = sorted(requested - known_ids)
    stats["verification"] = run_reconstruction_verification_gate(
        requested=stats["requested"],
        applied_candidate_ids=stats["applied_candidate_ids"],
        skipped_candidate_ids=stats["skipped_candidate_ids"],
        missing_candidate_ids=stats["missing_candidate_ids"],
        tool_errors=stats["tool_errors"],
    ).to_dict()
    return stats


def replay_history(
    execute_fn,
    *,
    history_dir: Path | None = None,
    dry_run: bool = False,
    force: bool = False,
    count_records_fn=None,
) -> dict:
    existing_records = 0
    if count_records_fn is not None:
        try:
            existing_records = int(count_records_fn() or 0)
        except Exception:
            existing_records = 0
    if existing_records and not force:
        raise RuntimeError(
            f"Active brain is not empty ({existing_records} records found). "
            "Use force=True if you really want to replay into the current store."
        )

    stats = {
        "files": 0,
        "entries": 0,
        "tool_calls_seen": 0,
        "tool_calls_replayed": 0,
        "tool_calls_skipped": 0,
        "tool_errors": [],
        "existing_records_before": existing_records,
    }

    for history_file in history_files(history_dir):
        stats["files"] += 1
        for entry in load_history_log(history_file):
            stats["entries"] += 1
            if entry.get("type") != "tool_call":
                continue
            tool = str(entry.get("tool") or "").strip()
            if tool not in SAFE_REPLAY_TOOLS:
                continue
            stats["tool_calls_seen"] += 1
            if not tool_call_succeeded(entry.get("result")):
                stats["tool_calls_skipped"] += 1
                continue

            args = normalize_replay_args(tool, entry.get("args") or {})
            if dry_run:
                stats["tool_calls_replayed"] += 1
                continue

            result = execute_fn(tool, args)
            if tool_call_succeeded(result):
                stats["tool_calls_replayed"] += 1
            else:
                stats["tool_calls_skipped"] += 1
                stats["tool_errors"].append(
                    {
                        "file": history_file.name,
                        "tool": tool,
                        "args": args,
                        "result": str(result)[:400],
                    }
                )

    logger.info("History replay completed: %s", stats)
    return stats
