"""Render {{metric:ID}} tokens against a MetricSnapshot, and build the
compact machine-like line the LLM sees in every prompt.

Strict rules:
    - Only exact metric ids from the snapshot are substituted.
    - Unknown ids → "[невідома метрика]" + log. No fallback guessing.
    - Stale values (collected_at older than stale_after_sec) → "[застаріле]".
    - No normalization of "almost similar" ids. If LLM writes `{{metric:HotZones}}`
      that's unknown; only `{{metric:hot_zones}}` resolves.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from remy.core.metric_snapshot import MetricSnapshot

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\{\{metric:([a-zA-Z0-9_]+)\}\}")

UNKNOWN_PLACEHOLDER = "[невідома метрика]"
STALE_PLACEHOLDER = "[застаріле]"


@dataclass(frozen=True)
class RenderResult:
    text: str
    used_metric_ids: tuple[str, ...]
    unknown_metric_ids: tuple[str, ...]
    stale_metric_ids: tuple[str, ...]


def render_metrics(
    text: str,
    snapshot: MetricSnapshot,
    now: float | None = None,
) -> RenderResult:
    """Substitute {{metric:id}} tokens in *text*. Never raises.

    Returns the rewritten text plus accounting of what was used, unknown, or stale.
    """
    if not text:
        return RenderResult(text="", used_metric_ids=(), unknown_metric_ids=(), stale_metric_ids=())

    now_ts = now if now is not None else time.time()
    used: list[str] = []
    unknown: list[str] = []
    stale: list[str] = []

    def _replace(match: re.Match) -> str:
        metric_id = match.group(1)
        value = snapshot.get(metric_id)
        if value is None:
            unknown.append(metric_id)
            logger.info("metric_render: unknown id %r — replaced with placeholder", metric_id)
            return UNKNOWN_PLACEHOLDER
        age = now_ts - value.collected_at
        if age > value.stale_after_sec:
            stale.append(metric_id)
            logger.info(
                "metric_render: id %r is stale (age=%.1fs > ttl=%.1fs)",
                metric_id, age, value.stale_after_sec,
            )
            return STALE_PLACEHOLDER
        used.append(metric_id)
        return str(value.value)

    rendered = _TOKEN_RE.sub(_replace, text)
    return RenderResult(
        text=rendered,
        used_metric_ids=tuple(used),
        unknown_metric_ids=tuple(unknown),
        stale_metric_ids=tuple(stale),
    )


def build_compact_injection(snapshot: MetricSnapshot) -> str:
    """Return a single machine-like line for every turn. No prose.

    Format:
        [metrics] total_records=681 hot_zones=6 volatile_beliefs=53 ...

    Missing metrics are omitted silently from the line (they're still recorded
    on the snapshot object for observability).
    """
    if not snapshot.values:
        return ""
    parts = [f"{mid}={snapshot.values[mid].value}" for mid in snapshot.available_ids]
    return "[metrics] " + " ".join(parts)


def build_full_injection(snapshot: MetricSnapshot) -> str:
    """Return the full prompt block for self-introspection turns.

    Includes explicit contract so the LLM knows HOW to cite these metrics.
    """
    if not snapshot.values:
        return ""
    header = (
        "=== AVAILABLE INTERNAL METRICS ===\n"
        "Use ONLY these ids when citing your internal state.\n"
        "Write {{metric:ID}} in your reply — never write the number yourself.\n"
        "If a metric you need is NOT listed here, say \"я не маю цієї метрики\"\n"
        "instead of inventing a value.\n"
    )
    lines = [
        f"  {{metric:{mid}}}  =  {v.value}    ({v.source_method})"
        for mid, v in snapshot.values.items()
    ]
    footer = "=== END METRICS ==="
    return "\n".join([header, *lines, footer])
