"""
Live task metrics — per-family success tracking.

Tracks completion rate, blocked rate, retry rate, and time-to-completion
for each task family (signup_operator, publisher, market_research, general).

Persists to data/task_metrics.json. Thread-safe.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass

logger = logging.getLogger("TaskMetrics")

METRICS_FILE = "task_metrics.json"

# Known task families
TASK_FAMILIES = ("signup_operator", "publisher", "market_research", "monitoring", "general")


@dataclass
class FamilyMetrics:
    """Aggregated metrics for one task family."""

    total_cycles: int = 0
    successes: int = 0
    failures: int = 0
    blocked_external: int = 0
    zero_tool_cycles: int = 0
    timeouts: int = 0
    total_duration_ms: int = 0
    total_tokens: int = 0
    memory_assisted: int = 0
    retry_shaped: int = 0
    verified_completions: int = 0
    repeated_failures: int = 0
    accepted_sources_total: int = 0
    rejected_sources_total: int = 0
    citation_coverage_total: float = 0.0
    contradictions_total: int = 0
    session_resumes: int = 0

    @property
    def completion_rate(self) -> float:
        return self.successes / self.total_cycles if self.total_cycles else 0.0

    @property
    def blocked_rate(self) -> float:
        return self.blocked_external / self.total_cycles if self.total_cycles else 0.0

    @property
    def retry_rate(self) -> float:
        """Fraction of cycles that failed (potential retries)."""
        return self.failures / self.total_cycles if self.total_cycles else 0.0

    @property
    def avg_duration_ms(self) -> float:
        return self.total_duration_ms / self.total_cycles if self.total_cycles else 0.0

    @property
    def avg_tokens(self) -> float:
        return self.total_tokens / self.total_cycles if self.total_cycles else 0.0

    @property
    def memory_assist_rate(self) -> float:
        """Fraction of cycles where browser memory influenced execution."""
        return self.memory_assisted / self.total_cycles if self.total_cycles else 0.0

    @property
    def retry_shaped_rate(self) -> float:
        """Fraction of cycles where retry was shaped by memory (avoided bad selectors)."""
        return self.retry_shaped / self.total_cycles if self.total_cycles else 0.0

    @property
    def verified_completion_rate(self) -> float:
        """Fraction of successes that were verified with evidence."""
        return self.verified_completions / self.successes if self.successes else 0.0

    @property
    def repeated_failure_rate(self) -> float:
        """Fraction of failures that were repeated (same reason 3+ times)."""
        return self.repeated_failures / self.failures if self.failures else 0.0

    @property
    def avg_accepted_sources(self) -> float:
        return self.accepted_sources_total / self.total_cycles if self.total_cycles else 0.0

    @property
    def avg_rejected_sources(self) -> float:
        return self.rejected_sources_total / self.total_cycles if self.total_cycles else 0.0

    @property
    def citation_coverage_rate(self) -> float:
        return self.citation_coverage_total / self.total_cycles if self.total_cycles else 0.0

    @property
    def contradiction_resolution_rate(self) -> float:
        return self.contradictions_total / self.total_cycles if self.total_cycles else 0.0

    @property
    def session_resume_rate(self) -> float:
        return self.session_resumes / self.total_cycles if self.total_cycles else 0.0

    def to_summary(self) -> dict:
        return {
            "total_cycles": self.total_cycles,
            "successes": self.successes,
            "failures": self.failures,
            "blocked_external": self.blocked_external,
            "zero_tool_cycles": self.zero_tool_cycles,
            "timeouts": self.timeouts,
            "completion_rate": round(self.completion_rate, 3),
            "blocked_rate": round(self.blocked_rate, 3),
            "retry_rate": round(self.retry_rate, 3),
            "avg_duration_ms": round(self.avg_duration_ms),
            "avg_tokens": round(self.avg_tokens),
            "memory_assisted": self.memory_assisted,
            "retry_shaped": self.retry_shaped,
            "memory_assist_rate": round(self.memory_assist_rate, 3),
            "retry_shaped_rate": round(self.retry_shaped_rate, 3),
            "verified_completions": self.verified_completions,
            "verified_completion_rate": round(self.verified_completion_rate, 3),
            "repeated_failures": self.repeated_failures,
            "repeated_failure_rate": round(self.repeated_failure_rate, 3),
            "accepted_sources_total": self.accepted_sources_total,
            "rejected_sources_total": self.rejected_sources_total,
            "avg_accepted_sources": round(self.avg_accepted_sources, 2),
            "avg_rejected_sources": round(self.avg_rejected_sources, 2),
            "citation_coverage_rate": round(self.citation_coverage_rate, 3),
            "contradictions_total": self.contradictions_total,
            "contradiction_resolution_rate": round(self.contradiction_resolution_rate, 3),
            "session_resumes": self.session_resumes,
            "session_resume_rate": round(self.session_resume_rate, 3),
        }


@dataclass
class CycleOutcome:
    """Input event for recording one execution cycle's result."""

    family: str
    success: bool
    blocked_external: bool = False
    zero_tool: bool = False
    timeout: bool = False
    duration_ms: int = 0
    tokens_used: int = 0
    worker: str = ""
    memory_assisted: bool = False
    retry_shaped: bool = False
    verified: bool = False
    repeated_failure: bool = False
    accepted_sources_count: int = 0
    rejected_sources_count: int = 0
    citation_coverage_rate: float = 0.0
    contradictions_count: int = 0
    session_resumed: bool = False


def _pack_metadata(family: str) -> dict:
    """Get capability pack label/description for a metrics family."""
    try:
        from remy.core.capability_packs import get_pack

        pack = get_pack(family)
        return {"pack_label": pack.label, "pack_description": pack.description}
    except Exception:
        return {}


class TaskMetricsTracker:
    """Thread-safe per-family metrics tracker with persistence."""

    def __init__(self, path=None):
        if path is not None:
            from pathlib import Path

            self._path = Path(path) / METRICS_FILE if Path(path).is_dir() else Path(path)
        else:
            from remy.core.meta_store import resolve_path

            self._path = resolve_path(METRICS_FILE, "metrics")
        self._lock = threading.Lock()
        self._families: dict[str, FamilyMetrics] = {}
        self._last_save = 0.0
        self._load()

    def _load(self):
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for family, vals in data.get("families", {}).items():
                self._families[family] = FamilyMetrics(
                    total_cycles=vals.get("total_cycles", 0),
                    successes=vals.get("successes", 0),
                    failures=vals.get("failures", 0),
                    blocked_external=vals.get("blocked_external", 0),
                    zero_tool_cycles=vals.get("zero_tool_cycles", 0),
                    timeouts=vals.get("timeouts", 0),
                    total_duration_ms=vals.get("total_duration_ms", 0),
                    total_tokens=vals.get("total_tokens", 0),
                    memory_assisted=vals.get("memory_assisted", 0),
                    retry_shaped=vals.get("retry_shaped", 0),
                    verified_completions=vals.get("verified_completions", 0),
                    repeated_failures=vals.get("repeated_failures", 0),
                    accepted_sources_total=vals.get("accepted_sources_total", 0),
                    rejected_sources_total=vals.get("rejected_sources_total", 0),
                    citation_coverage_total=vals.get("citation_coverage_total", 0.0),
                    contradictions_total=vals.get("contradictions_total", 0),
                    session_resumes=vals.get("session_resumes", 0),
                )
        except Exception as e:
            logger.warning("Failed to load task metrics: %s", e)

    def _save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "families": {name: asdict(fm) for name, fm in self._families.items()},
                "saved_at": time.time(),
            }
            from remy.core.file_utils import atomic_write

            atomic_write(self._path, json.dumps(data, indent=2))
            self._last_save = time.time()
        except Exception as e:
            logger.warning("Failed to save task metrics: %s", e)

    def record(self, outcome: CycleOutcome):
        """Record one execution cycle outcome."""
        family = outcome.family if outcome.family in TASK_FAMILIES else "general"

        with self._lock:
            fm = self._families.setdefault(family, FamilyMetrics())
            fm.total_cycles += 1
            fm.total_duration_ms += outcome.duration_ms
            fm.total_tokens += outcome.tokens_used

            if outcome.blocked_external:
                fm.blocked_external += 1
            elif outcome.timeout:
                fm.timeouts += 1
            elif outcome.zero_tool:
                fm.zero_tool_cycles += 1
            elif outcome.success:
                fm.successes += 1
            else:
                fm.failures += 1

            if outcome.memory_assisted:
                fm.memory_assisted += 1
            if outcome.retry_shaped:
                fm.retry_shaped += 1
            if outcome.verified and outcome.success:
                fm.verified_completions += 1
            if outcome.repeated_failure and not outcome.success:
                fm.repeated_failures += 1
            fm.accepted_sources_total += max(0, int(outcome.accepted_sources_count or 0))
            fm.rejected_sources_total += max(0, int(outcome.rejected_sources_count or 0))
            fm.citation_coverage_total += max(0.0, float(outcome.citation_coverage_rate or 0.0))
            fm.contradictions_total += max(0, int(outcome.contradictions_count or 0))
            if outcome.session_resumed:
                fm.session_resumes += 1

            # Persist every 5 cycles or every 60s
            if fm.total_cycles % 5 == 0 or time.time() - self._last_save > 60:
                self._save()

    def get_family(self, family: str) -> dict:
        """Get summary for one task family."""
        with self._lock:
            fm = self._families.get(family)
            if not fm:
                return {"family": family, "total_cycles": 0}
            summary = fm.to_summary()
            summary["family"] = family
            summary.update(_pack_metadata(family))
            return summary

    def get_all(self) -> dict:
        """Get summary for all families + totals."""
        with self._lock:
            result = {}
            totals = FamilyMetrics()

            for family in TASK_FAMILIES:
                fm = self._families.get(family)
                if fm and fm.total_cycles > 0:
                    summary = fm.to_summary()
                    summary.update(_pack_metadata(family))
                    result[family] = summary
                    totals.total_cycles += fm.total_cycles
                    totals.successes += fm.successes
                    totals.failures += fm.failures
                    totals.blocked_external += fm.blocked_external
                    totals.zero_tool_cycles += fm.zero_tool_cycles
                    totals.timeouts += fm.timeouts
                    totals.total_duration_ms += fm.total_duration_ms
                    totals.total_tokens += fm.total_tokens
                    totals.memory_assisted += fm.memory_assisted
                    totals.retry_shaped += fm.retry_shaped
                    totals.verified_completions += fm.verified_completions
                    totals.repeated_failures += fm.repeated_failures
                    totals.accepted_sources_total += fm.accepted_sources_total
                    totals.rejected_sources_total += fm.rejected_sources_total
                    totals.citation_coverage_total += fm.citation_coverage_total
                    totals.contradictions_total += fm.contradictions_total
                    totals.session_resumes += fm.session_resumes

            return {
                "families": result,
                "totals": totals.to_summary(),
            }

    def flush(self):
        """Force-save to disk."""
        with self._lock:
            self._save()


def resolve_family(goal: dict | None, worker: str = "") -> str:
    """Determine task family from goal metadata via capability pack resolution.

    Uses the same pack resolution as the orchestrator so metrics align
    with the pack that actually ran the goal.
    """
    try:
        from remy.core.capability_packs import resolve_pack

        pack = resolve_pack(goal)
        family = pack.metrics_family
        if family in TASK_FAMILIES:
            return family
    except Exception:
        pass
    # Fallback: explicit goal_template
    if goal:
        template = goal.get("goal_template", "")
        if template in TASK_FAMILIES:
            return template
    return "general"


def detect_memory_signals(session_log: list[dict]) -> tuple[bool, bool]:
    """Check session log for memory involvement (browser or research).

    Returns (memory_assisted, retry_shaped) booleans.
    """
    memory_assisted = False
    retry_shaped = False
    for entry in session_log or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("execution_memory"):
            memory_assisted = True
        if entry.get("retry_shaped"):
            retry_shaped = True
        # Research memory: auto-contradictions detected during add_research_finding
        if entry.get("type") == "tool_call" and entry.get("tool") == "add_research_finding":
            result = entry.get("result") or {}
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except Exception:
                    result = {}
            if result.get("auto_contradictions"):
                memory_assisted = True
        if memory_assisted and retry_shaped:
            break
    return memory_assisted, retry_shaped


# Singleton
task_metrics = TaskMetricsTracker()
