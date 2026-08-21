"""
Generic metric and event handlers.

These replace the old health-specific tools with neutral workflow primitives.
Compatibility aliases are handled in tool_dispatch.py and health.py.
"""

import json
import logging
from datetime import datetime

logger = logging.getLogger("BrainTools")


def _get_brain():
    """Lazy accessor; reads brain from brain_tools to support test patching."""
    import remy.core.brain_tools as _bt

    return _bt.brain


def _track_metric(args: dict, channel: str | None = None) -> str:
    """Track a user-reported numeric metric."""
    from remy.core.agent_tools import Level, brain_lock
    from remy.core.provenance import _stamp_provenance
    from remy.core.tool_dispatch import _NON_INTERACTIVE_CHANNELS

    if channel in _NON_INTERACTIVE_CHANNELS:
        return json.dumps(
            {
                "error": (
                    "track_metric is only available in interactive channels. "
                    "Tracked metrics must come from explicit user input."
                )
            },
            ensure_ascii=False,
        )

    brain = _get_brain()

    metric_type = str(args.get("metric_type") or args.get("metric") or "").lower().strip()
    if not metric_type:
        return json.dumps({"error": "track_metric requires metric_type."}, ensure_ascii=False)

    value = args.get("value")
    unit = str(args.get("unit") or "").strip()
    notes = str(args.get("notes") or "")

    content = f"Metric: {metric_type} = {value} {unit}".strip()
    if notes:
        content += f" ({notes})"

    timestamp = datetime.now().isoformat()
    tags = ["metric", metric_type, "tracked-metric"]

    with brain_lock:
        rec = brain.store(
            content=content,
            level=Level.DOMAIN,
            tags=tags,
            metadata=_stamp_provenance(
                {
                    "type": "metric",
                    "metric": metric_type,
                    "value": value,
                    "unit": unit,
                    "notes": notes,
                    "timestamp": timestamp,
                },
                channel,
                tags=tags,
            ),
            deduplicate=False,
        )

        trend = "data point recorded"
        try:
            history = brain.search(query="", tags=[metric_type], limit=5)
            if len(history) > 1:
                sorted_hist = sorted(
                    history, key=lambda x: (x.metadata or {}).get("timestamp", ""), reverse=True
                )
                prev = sorted_hist[1]
                prev_val = (prev.metadata or {}).get("value")
                if prev_val is not None:
                    diff = float(value) - float(prev_val)
                    trend = f"change: {diff:+.2f} vs last"
        except Exception:
            pass

    return f"Recorded {metric_type}: {value} {unit}. ({trend})"


def _metric_summary(args: dict) -> str:
    """Summarize tracked metrics and events."""
    brain = _get_brain()
    period = args.get("period", "week")
    limit = int(args.get("limit") or 100)

    metrics = brain.search(query="", tags=["metric"], limit=limit)
    legacy_metrics = brain.search(query="", tags=["health-metric"], limit=limit)
    events = brain.search(query="", tags=["event"], limit=limit)
    legacy_events = brain.search(query="", tags=["symptom"], limit=limit)

    all_metrics = list(metrics or []) + list(legacy_metrics or [])
    all_events = list(events or []) + list(legacy_events or [])

    if not all_metrics and not all_events:
        return "No tracked metrics or events found."

    lines = [f"Metric Summary ({period}):"]

    if all_metrics:
        lines.append("\nMetrics:")
        by_type: dict[str, list[float]] = {}
        for item in all_metrics:
            meta = item.metadata or {}
            metric_type = str(meta.get("metric") or "unknown")
            by_type.setdefault(metric_type, [])
            value = meta.get("value")
            try:
                by_type[metric_type].append(float(value))
            except (TypeError, ValueError):
                pass

        for metric_type, values in by_type.items():
            if values:
                avg = sum(values) / len(values)
                lines.append(f"- {metric_type}: {len(values)} entries, avg {avg:.1f}")

    if all_events:
        lines.append("\nRecent Events:")
        for item in all_events[:5]:
            lines.append(f"- {item.content}")

    return "\n".join(lines)


def _event_correlate(args: dict) -> str:
    """Find possible correlations for an event using memory records."""
    brain = _get_brain()
    event = str(args.get("event") or args.get("symptom") or "").strip()
    if not event:
        return json.dumps({"error": "event_correlate requires event."}, ensure_ascii=False)

    related = brain.recall(event, token_budget=1000)

    prompt = (
        f"Analyze this event: {event}\n"
        f"Based on the following related memory records:\n{related}\n\n"
        "Suggest potential correlations based on the memory data. "
        "Be explicit that this is correlation analysis, not a verified cause."
    )

    try:
        from remy.core.llm import call_llm

        analysis = call_llm(prompt, purpose="event_correlate").content
        return f"Correlation analysis for '{event}':\n{analysis}"
    except Exception as e:
        return f"Correlation analysis failed: {e}"


def _track_health_metric(args: dict, channel: str | None = None) -> str:
    """Deprecated compatibility alias for _track_metric."""
    return _track_metric(args, channel)


def _health_summary(args: dict) -> str:
    """Deprecated compatibility alias for _metric_summary."""
    return _metric_summary(args)


def _symptom_correlate(args: dict) -> str:
    """Deprecated compatibility alias for _event_correlate."""
    if "event" not in args and "symptom" in args:
        args = {**args, "event": args.get("symptom")}
    return _event_correlate(args)
