"""
Monitoring Store — persisted snapshots and change history for monitored URLs.

Stores page content snapshots in data/monitoring_snapshots.json.
Detects changes by comparing current content against last snapshot.
Maintains a change history log per target URL.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher

logger = logging.getLogger("MonitoringStore")

SNAPSHOTS_FILE = "monitoring_snapshots.json"
MAX_HISTORY_PER_TARGET = 50


@dataclass
class Snapshot:
    """One content snapshot of a monitored URL."""

    url: str
    content_hash: str
    content_summary: str  # first 500 chars of extracted text
    timestamp: float
    content_length: int = 0


@dataclass
class ChangeEvent:
    """One detected change between two snapshots."""

    url: str
    timestamp: float
    change_type: (
        str  # "new" | "minor_update" | "significant_change" | "content_removed" | "unreachable"
    )
    similarity: float  # 0.0-1.0, where 1.0 = identical
    before_summary: str
    after_summary: str
    diff_highlights: list[str] = field(default_factory=list)  # key changed sections


@dataclass
class MonitorTarget:
    """State for one monitored URL."""

    url: str
    last_snapshot: Snapshot | None = None
    change_history: list[dict] = field(default_factory=list)
    check_count: int = 0


class MonitoringStore:
    """Persisted store for monitoring snapshots and change detection."""

    def __init__(self):
        from remy.core.meta_store import resolve_path

        self._path = resolve_path(SNAPSHOTS_FILE, "monitoring")
        self._targets: dict[str, MonitorTarget] = {}
        self._load()

    def _load(self):
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for url, target_data in data.get("targets", {}).items():
                snap_data = target_data.get("last_snapshot")
                snap = None
                if snap_data:
                    snap = Snapshot(
                        url=snap_data["url"],
                        content_hash=snap_data["content_hash"],
                        content_summary=snap_data["content_summary"],
                        timestamp=snap_data["timestamp"],
                        content_length=snap_data.get("content_length", 0),
                    )
                self._targets[url] = MonitorTarget(
                    url=url,
                    last_snapshot=snap,
                    change_history=target_data.get("change_history", []),
                    check_count=target_data.get("check_count", 0),
                )
        except Exception as e:
            logger.warning("Failed to load monitoring store: %s", e)

    def _save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "targets": {},
                "saved_at": time.time(),
            }
            for url, target in self._targets.items():
                td: dict = {
                    "check_count": target.check_count,
                    "change_history": target.change_history[-MAX_HISTORY_PER_TARGET:],
                }
                if target.last_snapshot:
                    td["last_snapshot"] = asdict(target.last_snapshot)
                data["targets"][url] = td

            from remy.core.file_utils import atomic_write

            atomic_write(self._path, json.dumps(data, indent=2))
        except Exception as e:
            logger.warning("Failed to save monitoring store: %s", e)

    @staticmethod
    def _hash_content(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Quick similarity ratio between two texts."""
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        # Use first 3000 chars for performance
        return SequenceMatcher(None, a[:3000], b[:3000]).ratio()

    @staticmethod
    def _classify_change(similarity: float, old_len: int, new_len: int) -> str:
        if old_len == 0:
            return "new"
        if new_len == 0:
            return "content_removed"
        if similarity >= 0.95:
            return "no_change"
        if similarity >= 0.75:
            return "minor_update"
        return "significant_change"

    @staticmethod
    def _extract_diff_highlights(
        old_text: str, new_text: str, max_highlights: int = 5
    ) -> list[str]:
        """Extract the most notable differences between old and new text."""
        old_lines = old_text.split("\n")
        new_lines = new_text.split("\n")

        highlights = []
        sm = SequenceMatcher(None, old_lines, new_lines)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            if tag == "replace":
                old_chunk = " ".join(old_lines[i1:i2]).strip()[:120]
                new_chunk = " ".join(new_lines[j1:j2]).strip()[:120]
                highlights.append(f"Changed: '{old_chunk}' → '{new_chunk}'")
            elif tag == "insert":
                new_chunk = " ".join(new_lines[j1:j2]).strip()[:120]
                highlights.append(f"Added: '{new_chunk}'")
            elif tag == "delete":
                old_chunk = " ".join(old_lines[i1:i2]).strip()[:120]
                highlights.append(f"Removed: '{old_chunk}'")
            if len(highlights) >= max_highlights:
                break
        return highlights

    def check_for_changes(self, url: str, current_content: str) -> ChangeEvent:
        """Compare current content against stored snapshot. Returns a ChangeEvent.

        Also updates the stored snapshot and change history.
        """
        target = self._targets.get(url)
        if not target:
            target = MonitorTarget(url=url)
            self._targets[url] = target

        target.check_count += 1
        now = time.time()
        content_hash = self._hash_content(current_content)
        summary = current_content[:500].strip()

        old_snapshot = target.last_snapshot

        # No previous snapshot — first check
        if not old_snapshot:
            event = ChangeEvent(
                url=url,
                timestamp=now,
                change_type="new",
                similarity=0.0,
                before_summary="(first check)",
                after_summary=summary,
            )
        elif old_snapshot.content_hash == content_hash:
            # Identical content
            event = ChangeEvent(
                url=url,
                timestamp=now,
                change_type="no_change",
                similarity=1.0,
                before_summary=old_snapshot.content_summary,
                after_summary=summary,
            )
        else:
            # Content changed — compute diff
            sim = self._similarity(old_snapshot.content_summary, summary)
            change_type = self._classify_change(
                sim,
                old_snapshot.content_length,
                len(current_content),
            )
            highlights = self._extract_diff_highlights(
                old_snapshot.content_summary,
                summary,
            )
            event = ChangeEvent(
                url=url,
                timestamp=now,
                change_type=change_type,
                similarity=round(sim, 3),
                before_summary=old_snapshot.content_summary,
                after_summary=summary,
                diff_highlights=highlights,
            )

        # Update snapshot
        target.last_snapshot = Snapshot(
            url=url,
            content_hash=content_hash,
            content_summary=summary,
            timestamp=now,
            content_length=len(current_content),
        )

        # Log change (skip no_change to save space)
        if event.change_type != "no_change":
            target.change_history.append(asdict(event))
            if len(target.change_history) > MAX_HISTORY_PER_TARGET:
                target.change_history = target.change_history[-MAX_HISTORY_PER_TARGET:]

        self._save()
        return event

    def record_unreachable(self, url: str) -> ChangeEvent:
        """Record that a URL was unreachable."""
        target = self._targets.get(url)
        if not target:
            target = MonitorTarget(url=url)
            self._targets[url] = target

        target.check_count += 1
        now = time.time()

        event = ChangeEvent(
            url=url,
            timestamp=now,
            change_type="unreachable",
            similarity=0.0,
            before_summary=(target.last_snapshot.content_summary if target.last_snapshot else ""),
            after_summary="(page unreachable)",
        )
        target.change_history.append(asdict(event))
        self._save()
        return event

    def get_target_status(self, url: str) -> dict:
        """Get monitoring status for a URL."""
        target = self._targets.get(url)
        if not target:
            return {"url": url, "monitored": False}
        return {
            "url": url,
            "monitored": True,
            "check_count": target.check_count,
            "last_check": target.last_snapshot.timestamp if target.last_snapshot else None,
            "last_hash": target.last_snapshot.content_hash if target.last_snapshot else None,
            "recent_changes": len(
                [
                    c
                    for c in target.change_history[-10:]
                    if c.get("change_type") not in ("no_change", "new")
                ]
            ),
        }

    def get_all_targets(self) -> list[dict]:
        """Get status of all monitored targets."""
        return [self.get_target_status(url) for url in self._targets]

    def get_change_history(self, url: str, limit: int = 10) -> list[dict]:
        """Get recent change history for a URL."""
        target = self._targets.get(url)
        if not target:
            return []
        return target.change_history[-limit:]


# Singleton
monitoring_store = MonitoringStore()
