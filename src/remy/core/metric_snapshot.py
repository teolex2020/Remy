"""Deterministic collection of agent self-metrics for prompt injection.

Primary defense against self-metric hallucination. Metrics are collected from a
single canonical source each, stamped, and exposed to the LLM via structured
tokens (see metric_render). The LLM must reference metrics by id
(`{{metric:X}}`); it does not write the numbers itself.

Design rules:
    - Every metric has exactly one source function. No guessing, no aggregation.
    - If a source fails, the metric is omitted and the rest of the snapshot
      still renders. A partial snapshot never crashes a turn.
    - Adding a new metric requires an explicit entry in METRIC_SCHEMA. This is a
      schema, not a keyword list. The set must stay small and unambiguous.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricValue:
    value: int | float | str
    source_method: str
    source_path: str
    collected_at: float
    stale_after_sec: float


@dataclass(frozen=True)
class MetricSnapshot:
    values: dict[str, MetricValue]
    collected_at: float
    session_id: str | None = None
    turn_id: str | None = None
    missing_metric_ids: tuple[str, ...] = field(default_factory=tuple)

    def get(self, metric_id: str) -> MetricValue | None:
        return self.values.get(metric_id)

    @property
    def available_ids(self) -> tuple[str, ...]:
        return tuple(self.values.keys())


@dataclass(frozen=True)
class MetricSource:
    id: str
    extract: Callable[[Any], int | float | str]
    source_method: str
    source_path: str
    stale_after_sec: float = 60.0


def _ext_recent_corrections(brain: Any) -> int:
    mh = brain.get_memory_health_digest()
    val = getattr(mh, "recent_correction_count", None)
    if val is None and isinstance(mh, dict):
        val = mh.get("recent_correction_count")
    if val is None:
        raise AttributeError("recent_correction_count missing from memory_health_digest")
    return int(val)


def _thermal_summary(brain: Any) -> Any:
    aura = getattr(brain, "_aura", None)
    if aura is not None and hasattr(aura, "export_acl_thermal_summary"):
        return aura.export_acl_thermal_summary()
    raise AttributeError("brain._aura.export_acl_thermal_summary unavailable")


def _ext_total_records(brain: Any) -> int:
    try:
        summary = _thermal_summary(brain)
        total = getattr(summary, "total_records", None)
        if total is None and isinstance(summary, dict):
            total = summary.get("total_records")
        if total is not None:
            return int(total)
    except Exception:
        if not hasattr(brain, "tier_stats"):
            raise
    stats = brain.tier_stats()
    if isinstance(stats, dict):
        total = stats.get("total", stats.get("total_records"))
        if total is not None:
            return int(total)
    raise AttributeError("total not available from thermal summary or brain.tier_stats")


def _ext_hot_zones(brain: Any) -> int:
    summary = _thermal_summary(brain)
    hot_zones = getattr(summary, "hot_zones", None)
    if hot_zones is None and isinstance(summary, dict):
        hot_zones = summary.get("hot_zones")
    if hot_zones is None:
        raise AttributeError("hot_zones missing from thermal summary")
    return len(hot_zones)


def _ext_volatile_beliefs(brain: Any) -> int:
    summary = _thermal_summary(brain)
    val = getattr(summary, "high_volatility_belief_count", None)
    if val is None and isinstance(summary, dict):
        val = summary.get("high_volatility_belief_count")
    if val is None:
        raise AttributeError("high_volatility_belief_count missing from thermal summary")
    return int(val)


def _ext_conflict_clusters(brain: Any) -> int:
    summary = _thermal_summary(brain)
    val = getattr(summary, "contradiction_cluster_count", None)
    if val is None and isinstance(summary, dict):
        val = summary.get("contradiction_cluster_count")
    if val is None:
        raise AttributeError("contradiction_cluster_count missing from thermal summary")
    return int(val)


METRIC_SCHEMA: tuple[MetricSource, ...] = (
    MetricSource(
        id="total_records",
        extract=_ext_total_records,
        source_method="brain._aura.export_acl_thermal_summary.total_records",
        source_path="brain._aura",
    ),
    MetricSource(
        id="hot_zones",
        extract=_ext_hot_zones,
        source_method="brain._aura.export_acl_thermal_summary.hot_zones",
        source_path="brain._aura",
    ),
    MetricSource(
        id="volatile_beliefs",
        extract=_ext_volatile_beliefs,
        source_method="brain._aura.export_acl_thermal_summary.high_volatility_belief_count",
        source_path="brain._aura",
    ),
    MetricSource(
        id="conflict_clusters",
        extract=_ext_conflict_clusters,
        source_method="brain._aura.export_acl_thermal_summary.contradiction_cluster_count",
        source_path="brain._aura",
    ),
    MetricSource(
        id="recent_corrections",
        extract=_ext_recent_corrections,
        source_method="brain.get_memory_health_digest.recent_correction_count",
        source_path="brain",
    ),
)


def _log_metric_failure(src: MetricSource, exc: Exception) -> None:
    message = "metric_snapshot: %s failed (%s: %s) - omitted"
    args = (src.id, type(exc).__name__, exc)
    if isinstance(exc, AttributeError):
        logger.debug(message, *args)
    else:
        logger.warning(message, *args)


def collect_metric_snapshot(
    brain: Any,
    session_id: str | None = None,
    turn_id: str | None = None,
    schema: tuple[MetricSource, ...] = METRIC_SCHEMA,
) -> MetricSnapshot:
    """Collect all metrics from *schema*. Failures are logged, not raised."""
    now = time.time()
    values: dict[str, MetricValue] = {}
    missing: list[str] = []

    for src in schema:
        try:
            raw = src.extract(brain)
        except Exception as exc:
            _log_metric_failure(src, exc)
            missing.append(src.id)
            continue
        values[src.id] = MetricValue(
            value=raw,
            source_method=src.source_method,
            source_path=src.source_path,
            collected_at=now,
            stale_after_sec=src.stale_after_sec,
        )

    return MetricSnapshot(
        values=values,
        collected_at=now,
        session_id=session_id,
        turn_id=turn_id,
        missing_metric_ids=tuple(missing),
    )
