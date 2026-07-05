"""Cognitive Brief — compress autonomous-channel history to ACL snapshot.

Instead of replaying accumulated ToolMessage/AIMessage history at every
LangGraph iteration (which inflates past 1M tokens), we project the SDK's
cognitive state through ACL (Aura Cognitive Language) and render it to
a small, deterministic brief: a few hundred tokens of *what the brain
knows right now*, in plain language.

This turns the autonomous loop's context from "transcript of every
previous step" into "current cognitive snapshot" — effectively an
infinite context window, since brief size is bounded by cognitive
state, not conversation length.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

_METRICS_LOCK = Lock()
_METRICS_PATH = Path("data") / "acl_brief_metrics.jsonl"


def build_cognitive_brief(brain: Any, locale: str = "ua", *, max_conflicts: int = 5, max_gaps: int = 5) -> str:
    """Build a deterministic cognitive snapshot from brain state.

    Args:
        brain: _AuraCompat instance exposing `_aura` (the Rust Aura handle).
        locale: "ua" or "en" — rendering language.
        max_conflicts: cap on conflict lines.
        max_gaps: cap on gap lines.

    Returns:
        Multi-line string; empty string if SDK lacks ACL support (old wheel).
    """
    try:
        from aura._core import (
            render_thermal_summary,
            render_conflict_report,
            render_gap_report,
        )
    except ImportError:
        return ""

    aura = getattr(brain, "_aura", None) or brain
    if not hasattr(aura, "export_acl_thermal_summary"):
        return ""

    sections: list[str] = []

    try:
        thermal = aura.export_acl_thermal_summary()
        sections.append(render_thermal_summary(thermal, locale))
    except Exception as exc:
        logger.debug("cognitive_brief: thermal export failed: %s", exc)

    try:
        conflicts = aura.export_acl_conflicts(max_conflicts)
        if conflicts:
            sections.append("")
            sections.append("=== Active conflicts ===" if locale != "ua" else "=== Активні конфлікти ===")
            for c in conflicts:
                sections.append(render_conflict_report(c, locale))
    except Exception as exc:
        logger.debug("cognitive_brief: conflicts export failed: %s", exc)

    try:
        gaps = aura.export_acl_gaps(max_gaps)
        if gaps:
            sections.append("")
            sections.append("=== Knowledge gaps ===" if locale != "ua" else "=== Прогалини знань ===")
            for g in gaps:
                sections.append(render_gap_report(g, locale))
    except Exception as exc:
        logger.debug("cognitive_brief: gaps export failed: %s", exc)

    return "\n".join(sections).strip()


def estimate_tokens(text: str) -> int:
    """Rough token count (4 chars/token heuristic)."""
    return max(1, len(text) // 4)


def _estimate_messages_tokens(messages) -> int:
    """Estimate tokens for a LangGraph message list (before brief injection)."""
    total = 0
    for m in messages or []:
        content = getattr(m, "content", "") or ""
        if isinstance(content, list):
            content = " ".join(str(p) for p in content)
        total += len(str(content))
    return max(1, total // 4)


def log_brief_metric(
    *,
    enabled: bool,
    brief_used: bool,
    brief_chars: int,
    brief_tokens: int,
    transcript_tokens_estimate: int,
    session_id: str = "",
    channel: str = "autonomous",
    error: str | None = None,
) -> None:
    """Append one metric record for the 24h ACL brief trial.

    Non-blocking on failure — metrics must never break the agent loop.
    """
    try:
        _METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": time.time(),
            "time_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "channel": channel,
            "session_id": session_id,
            "enabled": enabled,
            "brief_used": brief_used,
            "brief_chars": brief_chars,
            "brief_tokens": brief_tokens,
            "transcript_tokens_estimate": transcript_tokens_estimate,
            "savings_tokens": max(0, transcript_tokens_estimate - brief_tokens),
            "error": error,
        }
        line = json.dumps(entry, ensure_ascii=False)
        with _METRICS_LOCK:
            with _METRICS_PATH.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as exc:
        logger.debug("acl metric log failed: %s", exc)
