"""Temporal Orientation Layer — gives the agent a sense of WHERE and WHEN it is.

Collects from AuraSDK methods that were available but never wired into the agent's
context. Injected into system_instruction on every cycle so the agent always knows:

  NOW:      date, weekday, week number, position in seed period
  STATE:    memory health, trend direction, record counts
  HORIZON:  tasks due today / this week / overdue
  MOMENTUM: days active, hours since last action
  SIGNALS:  surfaced concepts, policy hints, top memory issues
"""

import logging
from datetime import datetime, date, timedelta

logger = logging.getLogger(__name__)

# Cache: avoid hitting brain on every single token generation
_cache: dict = {}
_CACHE_TTL_SEC = 120  # refresh every 2 minutes


def get_temporal_orientation() -> str:
    """Return a compact orientation block for injection into system prompt."""
    import time
    now_ts = time.time()
    if _cache.get("ts") and (now_ts - _cache["ts"]) < _CACHE_TTL_SEC:
        return _cache["text"]

    try:
        from remy.core.agent_tools import brain, brain_lock
        with brain_lock:
            text = _build_orientation(brain)
        _cache["ts"] = now_ts
        _cache["text"] = text
        return text
    except Exception as e:
        logger.debug("temporal_orientation failed: %s", e)
        return ""


def _build_orientation(brain) -> str:
    parts = []

    # --- 1. NOW ---
    now = datetime.now()
    week_num = now.isocalendar()[1]
    weekday = now.strftime("%A")
    date_str = now.strftime("%d %B %Y")

    # Seed period position
    seed_line = ""
    try:
        from remy.core.survival import SEED_START_DATE, SEED_PERIOD_DAYS
        seed_start = datetime.fromisoformat(SEED_START_DATE)
        day_num = (now - seed_start).days + 1
        remaining = SEED_PERIOD_DAYS - day_num
        if 0 < day_num <= SEED_PERIOD_DAYS:
            seed_line = f" | Seed day {day_num}/{SEED_PERIOD_DAYS} ({remaining} left)"
    except Exception:
        pass

    parts.append(f"NOW: {weekday}, {date_str} | Week {week_num}{seed_line}")

    # --- 2. MEMORY STATE ---
    try:
        mh = brain.get_memory_health_digest() if hasattr(brain, 'get_memory_health_digest') else None
        if mh is None and hasattr(brain, '_aura'):
            raw = brain._aura.get_memory_health_digest()
            mh = brain._normalize_runtime_payload(raw) if hasattr(brain, '_normalize_runtime_payload') else raw

        if mh:
            total = getattr(mh, 'total_records', None) or (mh.get('total_records') if isinstance(mh, dict) else None) or 0
            trend = getattr(mh, 'maintenance_trend_direction', None) or (mh.get('maintenance_trend_direction') if isinstance(mh, dict) else None) or 'unknown'
            issues = getattr(mh, 'top_issues', None) or (mh.get('top_issues') if isinstance(mh, dict) else None) or []
            hi_vol = getattr(mh, 'high_volatility_belief_count', None) or (mh.get('high_volatility_belief_count') if isinstance(mh, dict) else 0) or 0
            corrections = getattr(mh, 'recent_correction_count', None) or (mh.get('recent_correction_count') if isinstance(mh, dict) else 0) or 0

            state_line = f"MEMORY: {total} records | trend: {trend}"
            if hi_vol:
                state_line += f" | volatile beliefs: {hi_vol}"
            if corrections:
                state_line += f" | recent corrections: {corrections}"
            parts.append(state_line)

            if issues:
                issue_strs = []
                for iss in issues[:2]:
                    if isinstance(iss, dict):
                        issue_strs.append(iss.get('description') or iss.get('title') or str(iss)[:60])
                    elif hasattr(iss, 'title'):
                        severity = getattr(iss, 'severity', '')
                        title = getattr(iss, 'title', '')
                        s = f"[{severity}] {title}" if severity else title
                        issue_strs.append(s[:80])
                    else:
                        issue_strs.append(str(iss)[:60])
                if issue_strs:
                    parts.append(f"  ATTENTION: {' | '.join(issue_strs)}")
    except Exception as e:
        logger.debug("memory health failed: %s", e)

    # --- 3. HORIZON (tasks due) ---
    try:
        from remy.core.agent_tools import brain_lock as _bl
        today = date.today()
        tomorrow = today + timedelta(days=1)
        week_end = today + timedelta(days=7)

        with _bl:
            tasks = brain.search(query="", tags=["scheduled-task"], limit=50)

        due_today, due_week, overdue = [], [], []
        for t in tasks:
            meta = t.metadata or {}
            if meta.get("status") not in ("active", "pending"):
                continue
            due_str = meta.get("due_date")
            if not due_str:
                continue
            try:
                due_dt = datetime.fromisoformat(due_str).date()
                days_ago = (today - due_dt).days
                desc = (meta.get("description") or t.content or "")[:50]

                # Include event context so the agent can reason about relevance
                event_date = meta.get("event_date") or meta.get("occasion_date")
                event_type = meta.get("event_type") or meta.get("occasion_type")
                context_suffix = ""
                if event_date:
                    try:
                        ev_dt = datetime.fromisoformat(event_date).date()
                        ev_days_ago = (today - ev_dt).days
                        ev_label = event_type or "event"
                        if ev_days_ago > 0:
                            context_suffix = f" [event: {ev_label} was {ev_days_ago}d ago on {ev_dt}]"
                        else:
                            context_suffix = f" [event: {ev_label} on {ev_dt}]"
                    except Exception:
                        pass

                if due_dt < today:
                    overdue.append(f"{desc} [{days_ago}d overdue{context_suffix}]")
                elif due_dt == today:
                    due_today.append(f"{desc}{context_suffix}")
                elif due_dt <= week_end:
                    due_week.append(f"{desc}{context_suffix}")
            except Exception:
                continue

        horizon_parts = []
        if overdue:
            horizon_parts.append(f"OVERDUE ({len(overdue)}): " + "; ".join(overdue[:3]))
        if due_today:
            horizon_parts.append(f"TODAY: {', '.join(due_today[:2])}")
        if due_week:
            horizon_parts.append(f"THIS WEEK: {len(due_week)} tasks")
        if horizon_parts:
            parts.append("HORIZON: " + " | ".join(horizon_parts))
    except Exception as e:
        logger.debug("horizon failed: %s", e)

    # --- 4. MOMENTUM ---
    try:
        from remy.core.execution_log import execution_log
        entries = execution_log.get_recent(limit=100)

        # Hours since last action
        if entries:
            last_ts = entries[-1].get("timestamp") or entries[-1].get("started_at", "")
            if last_ts:
                last_dt = datetime.fromisoformat(last_ts)
                hours_ago = (datetime.now() - last_dt).total_seconds() / 3600
                if hours_ago < 1:
                    last_str = f"{int(hours_ago * 60)}m ago"
                else:
                    last_str = f"{hours_ago:.1f}h ago"
            else:
                last_str = "unknown"
        else:
            last_str = "no actions yet"

        # Days active this week
        today_str = date.today().isoformat()
        week_start = (date.today() - timedelta(days=date.today().weekday())).isoformat()
        active_days = set()
        for e in entries:
            ts = e.get("timestamp") or e.get("started_at", "")
            if ts and ts[:10] >= week_start:
                active_days.add(ts[:10])

        parts.append(f"MOMENTUM: last action {last_str} | active {len(active_days)} day(s) this week")
    except Exception as e:
        logger.debug("momentum failed: %s", e)

    # --- 5. SURFACED SIGNALS (concepts + policy hints) ---
    try:
        concepts = brain.get_surfaced_concepts(limit=5) if hasattr(brain, 'get_surfaced_concepts') else []
        if concepts:
            concept_strs = []
            for c in concepts[:3]:
                if isinstance(c, dict):
                    concept_strs.append(c.get('label') or c.get('key') or str(c)[:40])
                elif hasattr(c, 'key'):
                    # SurfacedConcept PyO3 object — extract key and tags
                    tags = getattr(c, 'tags', [])
                    state = getattr(c, 'state', '')
                    label = ", ".join(tags[:3]) if tags else str(getattr(c, 'key', ''))[:40]
                    if state:
                        label += f" ({state})"
                    concept_strs.append(label[:50])
                elif hasattr(c, 'label'):
                    concept_strs.append(c.label[:40])
                else:
                    concept_strs.append(str(c)[:40])
            if concept_strs:
                parts.append(f"EMERGING CONCEPTS: {', '.join(concept_strs)}")
    except Exception as e:
        logger.debug("concepts failed: %s", e)

    try:
        hints = brain.get_surfaced_policy_hints(limit=3) if hasattr(brain, 'get_surfaced_policy_hints') else []
        if hints:
            hint_strs = []
            for h in hints[:2]:
                if isinstance(h, dict):
                    hint_strs.append(h.get('content') or h.get('key') or str(h)[:60])
                elif hasattr(h, 'key'):
                    hint_strs.append(str(getattr(h, 'key', ''))[:60])
                elif hasattr(h, 'content'):
                    hint_strs.append(h.content[:60])
                else:
                    hint_strs.append(str(h)[:60])
            if hint_strs:
                parts.append(f"POLICY SIGNALS: {' | '.join(hint_strs)}")
    except Exception as e:
        logger.debug("policy hints failed: %s", e)

    # --- 6. SPATIAL ORIENTATION (reads from brain, never fabricated) ---
    try:
        spatial = _get_spatial_context(brain)
        if spatial:
            parts.append(spatial)
    except Exception as e:
        logger.debug("spatial orientation failed: %s", e)

    if not parts:
        return ""

    lines = ["=== TEMPORAL & SPATIAL ORIENTATION ==="] + parts + ["=== END ORIENTATION ==="]
    return "\n".join(lines)


def _get_spatial_context(brain) -> str:
    """Read spatial/location context from brain memory.

    IMPORTANT: Only reads what was actually stored. Never fabricates.
    Returns empty string if nothing stored yet.
    """
    try:
        records = brain.search(query="location spatial orientation", tags=["spatial-context"], limit=5)
        if not records:
            # Fallback: search by tag only
            records = brain.search(query="", tags=["spatial-context"], limit=5)
        if not records:
            return ""

        lines = []
        for rec in records[:3]:
            meta = rec.metadata or {}
            content = rec.content or ""
            # Extract key fields from content/metadata
            loc_type = meta.get("location_type", "")
            if loc_type == "home_base":
                lines.append(f"HOME BASE: {content[:120]}")
            elif loc_type == "operational_zone":
                lines.append(f"ZONE: {content[:120]}")
            elif loc_type == "timezone":
                lines.append(f"TIMEZONE: {content[:80]}")
            else:
                lines.append(f"SPATIAL: {content[:120]}")

        if lines:
            return "LOCATION: " + " | ".join(lines)
    except Exception:
        pass
    return ""
