"""
Proactive Context — session-start context and active todos for system instruction.

Provides background context (scheduled tasks, session summaries, failure history,
dialogue continuity) and active todo tracking for the system prompt.
"""

import json
import logging

from remy.config.settings import settings

logger = logging.getLogger("BrainTools")


def _get_brain():
    """Lazy accessor — reads brain from brain_tools (supports test patching)."""
    import remy.core.brain_tools as _bt

    return _bt.brain


def _get_brain_lock():
    """Lazy accessor — reads brain_lock from brain_tools (supports test patching)."""
    import remy.core.brain_tools as _bt

    return _bt.brain_lock


def _get_active_todos_context() -> str:
    """Get active todo items for system instruction context.

    Cross-references each todo with failure outcomes to prevent
    the agent from blindly re-proposing failed tasks.
    """
    brain = _get_brain()

    try:
        records = brain.search(query="", tags=["todo-item"], limit=50)
        if not records:
            return ""

        # Pre-load recent failures for cross-referencing
        failure_records = []
        try:
            failure_records = brain.search(query="", tags=["outcome-failure"], limit=20)
        except Exception:
            pass
        failure_texts = [f.content.lower() for f in failure_records]

        from datetime import datetime

        today = datetime.now().strftime("%Y-%m-%d")
        active = []
        overdue = []

        for r in records:
            meta = getattr(r, "metadata", None) or {}
            if meta.get("type") != "todo_item":
                continue
            status = meta.get("status", "pending")
            if status in ("done", "archived"):
                continue

            title = r.content.split(": ", 1)[-1].split(" | ")[0] if ": " in r.content else r.content
            priority = meta.get("priority", "medium")
            due = meta.get("due_date")
            cat = meta.get("category", "personal")

            entry = f"[{priority.upper()}] {title}"
            if due:
                entry += f" (due: {due})"
            if cat != "personal":
                entry += f" [{cat}]"
            if status == "in_progress":
                entry += " *IN PROGRESS*"

            # Cross-reference with failures
            title_lower = title.lower()
            title_words = [w for w in title_lower.split() if len(w) > 3]
            for ft in failure_texts:
                if title_lower in ft or any(w in ft for w in title_words):
                    entry += " ⚠ PREVIOUSLY FAILED — check failures before retrying"
                    break

            if due and due < today:
                overdue.append(entry)
            else:
                active.append(entry)

        if not active and not overdue:
            return ""

        parts = ["\n## ACTIVE TODOS (source of truth — only these tasks need work)"]
        parts.append("Tasks NOT listed here are already DONE. Do NOT re-propose completed tasks.")
        if overdue:
            parts.append("OVERDUE:")
            parts.extend(f"  - ⚠ {t}" for t in overdue)
        if active:
            parts.extend(f"  - {t}" for t in active[:10])
        if len(active) > 10:
            parts.append(f"  ... and {len(active) - 10} more")
        parts.append("")
        return "\n".join(parts)
    except Exception:
        return ""


_proactive_context_cache: dict[str, tuple[float, str]] = {}  # {"context": (timestamp, text)}
_PROACTIVE_CACHE_TTL_SEC = 300  # 5 minutes


def get_proactive_context() -> str:
    """Generate proactive context for session start (Wake Up Routine).

    Cached for 5 minutes to avoid repeated heavy brain searches on every request.
    Checks:
    1. Scheduled tasks for today/tomorrow.
    2. Recent session summaries.
    3. Background insights.
    """
    import time as _time

    brain_lock = _get_brain_lock()

    cached = _proactive_context_cache.get("context")
    if cached:
        ts, text = cached
        if _time.time() - ts < _PROACTIVE_CACHE_TTL_SEC:
            return text

    with brain_lock:
        result = _get_proactive_context_locked()
    _proactive_context_cache["context"] = (_time.time(), result)
    return result


def _get_proactive_context_locked() -> str:
    """Inner get_proactive_context, called under brain_lock."""
    from datetime import datetime, timedelta

    brain = _get_brain()
    context_parts = []

    # 0. Personal events today (birthdays / anniversaries) - shown in first reply of the day.
    try:
        now = datetime.now()
        today_md = (now.month, now.day)
        person_records = brain.search(query="", tags=["person"], limit=50)
        today_events = []
        for rec in person_records:
            meta = getattr(rec, "metadata", None) or {}
            if meta.get("type") != "person":
                continue
            birth_date = meta.get("birth_date")
            if not birth_date:
                continue
            try:
                parsed = datetime.fromisoformat(str(birth_date)).date()
            except Exception:
                continue
            if (parsed.month, parsed.day) != today_md:
                continue

            full_name = meta.get("full_name") or rec.content.split(",")[0].strip()
            role = (meta.get("role") or "").strip()
            years = now.year - parsed.year
            if (now.month, now.day) < (parsed.month, parsed.day):
                years -= 1
            detail = f"{full_name} birthday"
            if role:
                detail += f" ({role})"
            detail += f" - turns {years}"
            today_events.append(detail)

        if today_events:
            context_parts.append(f"PERSONAL EVENTS TODAY ({now.strftime('%Y-%m-%d')}):")
            for item in today_events[:3]:
                context_parts.append(f"- {item}")

        # Also surface birthdays that passed in the last 14 days (missed/overdue awareness)
        from datetime import timedelta
        recent_passed = []
        for rec in person_records:
            meta = getattr(rec, "metadata", None) or {}
            if meta.get("type") != "person":
                continue
            birth_date = meta.get("birth_date")
            if not birth_date:
                continue
            try:
                parsed = datetime.fromisoformat(str(birth_date)).date()
            except Exception:
                continue
            # Build this year's birthday date
            try:
                this_year_bday = parsed.replace(year=now.year)
            except ValueError:
                continue  # Feb 29 edge case
            days_ago = (now.date() - this_year_bday).days
            if 1 <= days_ago <= 14:
                full_name = meta.get("full_name") or rec.content.split(",")[0].strip()
                role = (meta.get("role") or "").strip()
                years = now.year - parsed.year
                detail = f"{full_name}"
                if role:
                    detail += f" ({role})"
                detail += f" — birthday was {days_ago} day(s) ago ({this_year_bday.strftime('%d %b')}), turned {years}"
                recent_passed.append(detail)
        if recent_passed:
            context_parts.append(f"RECENT BIRTHDAYS PASSED (last 14 days — for your awareness):")
            for item in recent_passed[:3]:
                context_parts.append(f"- {item}")
    except Exception as e:
        logger.warning(f"Failed to get personal events context: {e}")

    # 1. Scheduled Tasks
    try:
        now = datetime.now()
        tomorrow = now + timedelta(days=1)
        tasks = brain.search(query="", tags=["scheduled-task"], limit=20)

        due_today = []
        due_tomorrow = []

        for task in tasks:
            meta = task.metadata or {}
            if meta.get("status") != "active":
                continue

            due_str = meta.get("due_date")
            if not due_str:
                continue

            try:
                due_date = datetime.fromisoformat(due_str)
                if due_date.date() == now.date():
                    due_today.append(meta.get("description", task.content))
                elif due_date.date() == tomorrow.date():
                    due_tomorrow.append(meta.get("description", task.content))
            except Exception:
                continue

        if due_today:
            context_parts.append(f"TASKS FOR TODAY ({now.strftime('%Y-%m-%d')}):")
            for t in due_today:
                context_parts.append(f"- {t}")

        if due_tomorrow:
            context_parts.append("UPCOMING - TOMORROW:")
            for t in due_tomorrow:
                context_parts.append(f"- {t}")

    except Exception as e:
        logger.warning(f"Failed to get scheduled tasks: {e}")

    # 2. Recent Session Summaries (Continuation)
    try:
        summaries = brain.search(query="", tags=["session-summary"], limit=2)
        if summaries:
            context_parts.append("\nPREVIOUS CONTEXT (What happened last time):")
            for s in summaries:
                context_parts.append(f"- {s.content}")
    except Exception:
        pass

    # 2b. Last dialogue from previous session (cheap — JSON file read, no LLM)
    try:
        brain_path = str(getattr(brain, "path", ""))
        if brain_path and brain_path == str(settings.AURA_BRAIN_PATH):
            history_dir = settings.DATA_DIR / "history"
        else:
            history_dir = None
        if history_dir and history_dir.exists():
            files = sorted(history_dir.glob("*.json"), key=lambda f: f.name, reverse=True)
            if files:
                with open(files[0], "r", encoding="utf-8") as _f:
                    prev = json.load(_f)
                dialogue = [
                    e
                    for e in prev.get("log", [])
                    if e.get("type") in ("user_text", "model_response") and e.get("text")
                ]
                if dialogue:
                    last_turns = dialogue[-8:]
                    context_parts.append("\nLAST DIALOGUE (previous session):")
                    for turn in last_turns:
                        role = "User" if turn["type"] == "user_text" else "Assistant"
                        context_parts.append(f'- {role}: "{turn["text"][:150]}"')
    except Exception:
        pass

    # 3. Recent Failures & Outcomes
    try:
        failures = brain.search(query="", tags=["outcome-failure"], limit=5)
        if failures:
            context_parts.append(
                "\n⚠ RECENT FAILURES (DO NOT repeat these without a new strategy):"
            )
            for f in failures:
                meta = getattr(f, "metadata", None) or {}
                reason = meta.get("reason", "")
                ts = meta.get("timestamp", "")
                summary = f.content[:200]
                line = f"- {summary}"
                if reason:
                    line += f" | Reason: {reason}"
                if ts:
                    line += f" | When: {ts[:10]}"
                context_parts.append(line)
            context_parts.append("Note: Do not retry the same approach unless something changed.")

        outcomes = brain.search(query="", tags=["autonomous-outcome"], limit=5)
        recent_outcomes = [o for o in outcomes if o.id not in {f.id for f in (failures or [])}]
        if recent_outcomes:
            context_parts.append("\nRECENT AUTONOMOUS OUTCOMES:")
            for o in recent_outcomes[:3]:
                meta = getattr(o, "metadata", None) or {}
                status = "✓" if "outcome-success" in (getattr(o, "tags", None) or []) else "✗"
                context_parts.append(f"- {status} {o.content[:150]}")
    except Exception as e:
        logger.warning(f"Failed to get failure context: {e}")

    # 4. Cognitive Layer Signals — reflection, contradictions, tool health
    try:
        # Latest reflection digest — what AuraSDK learned from maintenance
        rd = brain.latest_reflection_digest() if hasattr(brain, "latest_reflection_digest") else None
        if rd:
            findings = getattr(rd, "findings", None) or (rd.get("findings") if isinstance(rd, dict) else None) or []
            if findings:
                high = [f for f in findings if (getattr(f, "severity", None) or (f.get("severity") if isinstance(f, dict) else "")) in ("high", "critical")]
                if high:
                    context_parts.append("\nCOGNITIVE ALERTS (from last maintenance):")
                    for finding in high[:2]:
                        msg = getattr(finding, "message", None) or (finding.get("message") if isinstance(finding, dict) else str(finding)[:100])
                        context_parts.append(f"- {msg}")
    except Exception:
        pass

    try:
        # Contradiction clusters — conflicting beliefs that need resolution
        clusters = brain.contradiction_clusters(limit=3) if hasattr(brain, "contradiction_clusters") else []
        if clusters:
            context_parts.append(f"\nCONTRADICTION SIGNALS: {len(clusters)} conflict cluster(s) detected in memory.")
            for cl in clusters[:1]:
                label = getattr(cl, "label", None) or (cl.get("label") if isinstance(cl, dict) else str(cl)[:60])
                if label:
                    context_parts.append(f"- Topic: {label}")
    except Exception:
        pass

    try:
        # Tool health — warn if key tools are degraded
        th = brain.tool_health() if hasattr(brain, "tool_health") else {}
        degraded = [t for t, s in th.items() if "degraded" in str(s)]
        if degraded:
            context_parts.append(f"\nTOOL HEALTH WARNING: {', '.join(degraded)} showing failures — consider alternative approach.")
    except Exception:
        pass

    if not context_parts:
        return ""

    header = (
        "\n\n=== PROACTIVE AWAKENING CONTEXT (reference only) ===\n"
        "This is background information for YOUR awareness. "
        "Do NOT recite it to the user. Do NOT append tasks or reminders to unrelated answers. "
        "Only mention tasks/outcomes if the user ASKS about them or if you are greeting the user.\n"
    )
    return header + "\n".join(context_parts) + "\n===================================\n"
