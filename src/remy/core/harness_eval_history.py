"""Persistence and summary helpers for harness eval runs."""

from __future__ import annotations

import json
import logging
from collections import Counter

from remy.core.meta_store import resolve_path

logger = logging.getLogger("HarnessEvalHistory")

_HISTORY_FILE = "harness_eval_history.jsonl"


def store_harness_eval_run(entry: dict) -> None:
    """Append a harness eval run to metrics storage."""
    try:
        path = resolve_path(_HISTORY_FILE, "metrics")
        payload = {
            "id": str(entry.get("id") or "").strip(),
            "label": str(entry.get("label") or "").strip(),
            "status": str(entry.get("status") or "").strip(),
            "summary": str(entry.get("summary") or "").strip(),
            "executed_at": entry.get("executed_at"),
        }
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug("Failed to store harness eval run: %s", e)


def get_harness_eval_history_summary(limit: int = 20) -> dict:
    """Read recent harness eval history and compute lightweight trends."""
    try:
        path = resolve_path(_HISTORY_FILE, "metrics")
        if not path.exists():
            return {"total_runs": 0, "scenario_counts": {}, "status_counts": {}, "latest_entries": []}

        lines = path.read_text(encoding="utf-8").strip().split("\n")
        entries: list[dict] = []
        for line in lines[-limit:]:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            entries.append(item)

        counts = Counter(str(item.get("id") or "").strip() for item in entries if item.get("id"))
        status_counts = Counter(str(item.get("status") or "").strip() for item in entries if item.get("status"))
        latest_entries = [
            {
                "id": str(item.get("id") or ""),
                "status": str(item.get("status") or ""),
                "summary": str(item.get("summary") or ""),
                "executed_at": item.get("executed_at"),
            }
            for item in entries[-5:]
        ][::-1]
        return {
            "total_runs": len(entries),
            "scenario_counts": dict(counts),
            "status_counts": dict(status_counts),
            "latest_entries": latest_entries,
        }
    except Exception as e:
        logger.debug("Failed to summarize harness eval history: %s", e)
        return {"total_runs": 0, "scenario_counts": {}, "status_counts": {}, "latest_entries": [], "error": str(e)}
