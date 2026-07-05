"""Deprecated compatibility module for old health-specific imports."""


def _track_health_metric(args: dict, channel: str | None = None) -> str:
    """Deprecated alias for generic metric tracking."""
    from remy.core.tool_handlers.metrics import _track_metric

    return _track_metric(args, channel)


def _health_summary(args: dict) -> str:
    """Deprecated alias for generic metric summary."""
    from remy.core.tool_handlers.metrics import _metric_summary

    return _metric_summary(args)


def _symptom_correlate(args: dict) -> str:
    """Deprecated alias for generic event correlation."""
    from remy.core.tool_handlers.metrics import _event_correlate

    if "event" not in args and "symptom" in args:
        args = {**args, "event": args.get("symptom")}
    return _event_correlate(args)


def _extract_facts(
    args: dict, channel: str | None = None, session_id: str | None = None
) -> str:
    """Deprecated alias for neutral fact extraction."""
    from remy.core.tool_handlers.facts import _extract_facts as _handler

    return _handler(args, channel, session_id)
