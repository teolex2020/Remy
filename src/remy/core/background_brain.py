"""
Background Brain — offline memory maintenance, pattern discovery, and proactive notifications.

Run between sessions: decay, reflect, discover insights, consolidate memories,
check scheduled tasks, send notifications.
Usage: remy --background
"""

import logging
import random
import threading
from collections import defaultdict
from datetime import datetime, timedelta

from remy.config.settings import settings

logger = logging.getLogger("BackgroundBrain")

_background_lock = threading.Lock()

# Transient insights from the last background run — NOT stored as records.
# Consumed by build_system_instruction() and notifications, then overwritten next cycle.
# Protected by _insights_lock against concurrent reads from web/telegram while
# run_background() writes new values.
_insights_lock = threading.Lock()
_last_insights: list[str] = []
_last_cross_connections: list[str] = []
_insights_loaded: bool = False

_INSIGHTS_RECORD_TAG = "background-insights-latest"
_LLM_CONSOLIDATION_INTERVAL_SEC = 3600
_last_llm_consolidation_time: float = 0.0


def _load_persisted_insights(brain) -> None:
    """Load insights lazily so they survive restarts."""
    global _last_insights, _last_cross_connections, _insights_loaded
    if _insights_loaded:
        return
    try:
        records = brain.search(query="", tags=[_INSIGHTS_RECORD_TAG], limit=1)
        if records:
            meta = records[0].metadata or {}
            _last_insights = meta.get("insights", []) or []
            _last_cross_connections = meta.get("cross_connections", []) or []
    except Exception as e:
        logger.debug("Could not load persisted insights: %s", e)
    _insights_loaded = True


def _persist_insights(brain, insights: list[str], cross_connections: list[str]) -> None:
    """Store the latest background insights as a single upserted record."""
    if not insights and not cross_connections:
        return
    try:
        from remy.core.agent_tools import Level
        records = brain.search(query="", tags=[_INSIGHTS_RECORD_TAG], limit=1)
        payload = {
            "type": "background_insights",
            "insights": insights,
            "cross_connections": cross_connections,
            "updated_at": datetime.now().isoformat(),
        }
        content = f"Background insights ({len(insights)} items)"
        if records:
            brain.update(records[0].id, content=content, metadata=payload)
        else:
            brain.store(content=content, level=Level.WORKING, tags=[_INSIGHTS_RECORD_TAG], metadata=payload)
    except Exception as e:
        logger.debug("Could not persist insights: %s", e)


def _level_attr(name: str):
    from remy.core.agent_tools import Level

    return getattr(Level, name, getattr(Level, name.capitalize()))


def _all_brain_records(brain, *, min_strength: float = 0.0):
    try:
        return list(brain.records.values())
    except Exception:
        return list(brain.list_records(min_strength=min_strength))


def _get_brain_record(brain, rec_id: str):
    try:
        return brain.records.get(rec_id)
    except Exception:
        try:
            return brain.get(rec_id)
        except Exception:
            return None


def get_transient_insights() -> list[str]:
    """Get insights from the last background run (persisted across restart)."""
    with _insights_lock:
        if not _insights_loaded:
            try:
                try:
                    import remy.core.brain_tools as _bt

                    brain = _bt.brain
                except Exception:
                    from remy.core.agent_tools import brain
                _load_persisted_insights(brain)
            except Exception:
                pass
        return list(_last_insights)


def get_transient_cross_connections() -> list[str]:
    """Get cross-connections from the last background run (persisted across restart)."""
    with _insights_lock:
        if not _insights_loaded:
            try:
                try:
                    import remy.core.brain_tools as _bt

                    brain = _bt.brain
                except Exception:
                    from remy.core.agent_tools import brain
                _load_persisted_insights(brain)
            except Exception:
                pass
        return list(_last_cross_connections)


def _guarded_reflect(brain) -> dict:
    """Run brain.reflect() with overpromotion protection."""
    LEVEL_IDENTITY = _level_attr("IDENTITY")
    LEVEL_WORKING = _level_attr("WORKING")

    original_levels: dict[str, int] = {}
    for rec in _all_brain_records(brain):
        if rec.level != LEVEL_IDENTITY:
            original_levels[rec.id] = rec.level

    try:
        result = brain.reflect()
    finally:
        restored = 0
        for rec_id, orig_level in original_levels.items():
            rec = _get_brain_record(brain, rec_id)
            if rec is None:
                continue

            should_restore = False
            if orig_level == LEVEL_WORKING and rec.level != LEVEL_WORKING:
                should_restore = True
            elif rec.level == LEVEL_IDENTITY and (
                rec.activation_count < 20 or rec.strength < 0.9
            ):
                should_restore = True

            if should_restore:
                try:
                    brain.update(rec.id, level=orig_level)
                    restored += 1
                except Exception:
                    logger.debug("Could not restore level for %s", rec.id)

        if restored:
            logger.info("Overpromotion guard: restored %d records", restored)

    return result


# Tags that should NEVER be at IDENTITY level (they are transient/domain data)
_NON_IDENTITY_TAGS = {
    "autonomous-outcome", "autonomous-goal", "session-summary",
    "research-project", "research-finding", "web-search-cache",
    "proactive-session", "action-plan", "session-reflection",
    "scheduled-task", "health-metric", "fact", "extracted-fact",
    "consolidated-meta", "person", "story", "research-report",
    "push-subscription", "background-insights-latest",
}

# Tags that legitimately belong at IDENTITY
_IDENTITY_TAGS = {"user-profile", "identity"}


def fix_memory_levels(brain=None) -> dict:
    """Mass-fix records incorrectly stuck at IDENTITY level."""
    if brain is None:
        from remy.core.agent_tools import brain as default_brain
        brain = default_brain

    LEVEL_IDENTITY = _level_attr("IDENTITY")
    LEVEL_DOMAIN = _level_attr("DOMAIN")

    stats = {"total_identity": 0, "downgraded_to_domain": 0, "kept_identity": 0, "errors": 0}

    for rec in _all_brain_records(brain):
        if rec.level != LEVEL_IDENTITY:
            continue
        stats["total_identity"] += 1

        rec_tags = set(rec.tags or [])
        if rec_tags & _IDENTITY_TAGS:
            stats["kept_identity"] += 1
            continue

        if rec_tags & _NON_IDENTITY_TAGS:
            try:
                brain.update(rec.id, level=LEVEL_DOMAIN)
                stats["downgraded_to_domain"] += 1
            except Exception:
                stats["errors"] += 1
            continue

        if rec.activation_count >= 20 and rec.strength >= 0.9:
            stats["kept_identity"] += 1
            continue

        try:
            brain.update(rec.id, level=LEVEL_DOMAIN)
            stats["downgraded_to_domain"] += 1
        except Exception:
            stats["errors"] += 1

    logger.info(
        "Memory level fix: %d IDENTITY records - %d downgraded to DOMAIN, %d kept, %d errors",
        stats["total_identity"], stats["downgraded_to_domain"],
        stats["kept_identity"], stats["errors"],
    )
    return stats


def run_background(brain=None) -> dict:
    """Run one cycle of background brain processing."""
    if not _background_lock.acquire(blocking=False):
        logger.debug("Background maintenance already running, skipping.")
        return {"timestamp": datetime.now().isoformat(), "skipped": True}

    try:
        return _run_background_locked(brain)
    finally:
        _background_lock.release()


def _run_background_locked(brain=None) -> dict:
    """Internal: runs background processing while holding the lock."""
    if brain is None:
        from remy.core.agent_tools import brain as default_brain
        brain = default_brain

    from remy.core.agent_tools import brain_lock
    with brain_lock:
        return _run_background_inner(brain)


def _run_background_inner(brain) -> dict:
    """Innermost background loop - runs under both locks."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "decay": {"decayed": 0, "archived": 0},
        "reflect": {"promoted": 0, "connected": 0, "archived": 0},
        "insights_found": 0,
        "consolidation": {"clusters_found": 0, "records_merged": 0, "meta_records_created": 0, "native_merged": 0},
        "cross_connections": 0,
        "task_reminders": [],
        "total_records": 0,
    }

    try:
        report["level_fix"] = fix_memory_levels(brain)

        # ── TCS Step 1: Compute thermal map FIRST ──
        _thermal_report = None
        _cycle_class = None
        try:
            from remy.core.thermal_advisor import (
                compute_thermal_map, format_thermal_report_json, append_thermal_observation,
                classify_cycle, _cycle_count,
            )
            _thermal_report = compute_thermal_map(str(settings.AURA_BRAIN_PATH))
            if _thermal_report:
                report["thermal"] = format_thermal_report_json(_thermal_report)
                report["_thermal_report_obj"] = _thermal_report
                if _thermal_report.hot_zone_count > 0:
                    logger.info(
                        "Thermal: %d hot zones, %d clusters, energy=%.1f",
                        _thermal_report.hot_zone_count, len(_thermal_report.clusters),
                        _thermal_report.total_energy,
                    )
        except Exception as e:
            logger.debug("Thermal advisor skipped: %s", e)

        # ── TCS Step 2: Classify cycle ──
        try:
            from remy.core.thermal_advisor import classify_cycle, _cycle_count
            _cycle_class = classify_cycle(_thermal_report, _cycle_count)
            report["cycle_classification"] = {
                "type": _cycle_class.cycle_type,
                "reason": _cycle_class.reason,
                "skip_diagnostics": _cycle_class.skip_diagnostics,
                "skip_insights": _cycle_class.skip_insights,
                "skip_llm": _cycle_class.skip_llm,
            }
            logger.info("Cycle classified: %s (%s)", _cycle_class.cycle_type, _cycle_class.reason)
        except Exception:
            pass

        # Plasticity summary (plasticity runs inside compute_thermal_map)
        try:
            from remy.core.synaptic_plasticity import get_plasticity_summary
            plast = get_plasticity_summary(str(settings.AURA_BRAIN_PATH))
            if plast.get("total_tracked", 0) > 0:
                report["plasticity"] = plast
                if plast.get("pruned", 0) > 0:
                    logger.info(
                        "Plasticity: %d tracked, %d weakened, %d pruned",
                        plast["total_tracked"], plast["weakened"], plast["pruned"],
                    )
        except Exception:
            pass

        # ── TCS Step 3: Run maintenance (always — Rust substrate handles internal phases) ──
        maint = brain.run_maintenance()
        report["decay"] = {
            "decayed": getattr(getattr(maint, "decay", None), "decayed", 0),
            "archived": getattr(getattr(maint, "decay", None), "archived", 0),
        }
        report["reflect"] = {
            "promoted": getattr(getattr(maint, "reflect", None), "promoted", 0),
            "connected": 0,
            "archived": getattr(getattr(maint, "reflect", None), "archived", 0),
        }
        report["insights_found"] = getattr(maint, "insights_found", 0)
        report["cross_connections"] = getattr(maint, "cross_connections", 0)

        report["consolidation"] = {
            "clusters_found": getattr(getattr(maint, "consolidation", None), "clusters_found", 0),
            "records_merged": 0,
            "meta_records_created": getattr(getattr(maint, "consolidation", None), "meta_created", 0),
            "native_merged": getattr(getattr(maint, "consolidation", None), "native_merged", 0),
        }

        # ── TCS Step 4: Cognitive diagnostics (skipped on cold cycles) ──
        _skip_diag = _cycle_class and _cycle_class.skip_diagnostics

        if not _skip_diag:
            # Phase 3 cognitive diagnostics — trajectory delta + pending conflicts
            try:
                delta = brain.get_trajectory_delta()
                if delta is not None:
                    report["identity_trajectory"] = {
                        "assessment": delta.assessment,
                        "total_preference_change": delta.total_preference_change,
                        "stability_delta": round(delta.stability_delta, 3),
                        "cycles_elapsed": delta.cycles_elapsed,
                    }
                    if delta.assessment != "stable":
                        logger.info(
                            "Identity trajectory: %s (pref_change=%d, stability_delta=%.3f)",
                            delta.assessment, delta.total_preference_change, delta.stability_delta
                        )
            except Exception:
                pass

            try:
                conflicts = brain.get_conflict_cases(10)
                if conflicts:
                    report["pending_conflicts"] = len(conflicts)
                    logger.info("Pending identity conflicts: %d", len(conflicts))
            except Exception:
                pass

            # Memory drift monitoring (AuraSDK 2.1.0)
            try:
                drift_alert = brain.get_drift_alert()
                if drift_alert:
                    report["drift_alert"] = str(drift_alert)
                    logger.warning("Memory drift alert: %s", drift_alert)
                dr = brain.get_drift_report()
                if dr:
                    report["drift"] = {
                        "score": round(getattr(dr, "drift_score", 0), 4),
                        "assessment": getattr(dr, "assessment", ""),
                        "cycles": getattr(dr, "cycles_measured", 0),
                    }
            except Exception:
                pass

            # V16: Autonomic drive diagnostics (zero LLM cost)
            try:
                from remy.core.v16_proposal import collect_v16_diagnostics
                v16 = collect_v16_diagnostics(brain)
                if v16:
                    report["v16_autonomic"] = v16
                    if v16.get("active_drives", 0) > 0:
                        logger.info("V16 autonomic: %d active drives", v16["active_drives"])
            except Exception:
                pass
        else:
            report["diagnostics_skipped"] = True
            logger.info("Diagnostics skipped: %s", _cycle_class.reason)

        # ── TCS Step 5: Insights (skipped on cold cycles) ──
        _skip_insights = _cycle_class and _cycle_class.skip_insights

        if not _skip_insights:
            formatted = []
            try:
                formatted = _format_all_insights(brain.insights())
                formatted.extend(_format_correction_review_insights(brain))
            except Exception:
                formatted = []
            with _insights_lock:
                global _last_insights, _last_cross_connections
                _last_insights = formatted
                _last_cross_connections = []
            _persist_insights(brain, _last_insights, _last_cross_connections)

        # ── TCS Step 6: LLM gating (cold cycles skip entirely) ──
        _skip_llm = _cycle_class and _cycle_class.skip_llm

        import time as _time
        global _last_llm_consolidation_time
        now = _time.time()
        time_since_last = now - _last_llm_consolidation_time

        if _skip_llm:
            report.setdefault("llm_gating", {})["skipped"] = True
            report["llm_gating"]["reason"] = "cold_cycle"
            logger.info("LLM consolidation skipped: cold cycle")
        elif time_since_last >= _LLM_CONSOLIDATION_INTERVAL_SEC:
            # Thermal LLM gating: cold graph + no drought = skip
            llm_gate_skip = False
            _MAX_LLM_DROUGHT_SEC = _LLM_CONSOLIDATION_INTERVAL_SEC * 3
            thermal_data = report.get("thermal")
            if (thermal_data
                    and thermal_data.get("hot_zone_count", 0) == 0
                    and time_since_last < _MAX_LLM_DROUGHT_SEC):
                llm_gate_skip = True
                report.setdefault("llm_gating", {})["skipped"] = True
                report["llm_gating"]["reason"] = "cold_graph"
                logger.info("LLM consolidation skipped: graph cold, no hot zones")

            if not llm_gate_skip:
                consolidation = _consolidate_records(brain)
                _last_llm_consolidation_time = now
                consolidation["native_merged"] = report["consolidation"]["native_merged"]
                report["consolidation"] = consolidation
                if report.get("thermal", {}).get("hot_zone_count", 0) > 0:
                    report.setdefault("llm_gating", {})["triggered_by"] = "hot_zones"
                else:
                    report.setdefault("llm_gating", {})["triggered_by"] = "drought_fallback"

        # ── Always-run phases (not gated by TCS) ──
        reminders = _check_scheduled_tasks(brain)
        todo_reminders = _check_overdue_todos(brain)
        reminders.extend(todo_reminders)
        report["task_reminders"] = reminders

        expired = _expire_search_cache(brain)
        if expired:
            report["cache_expired"] = expired

        archived_count = _archive_old_records(brain)
        if archived_count:
            report["records_archived"] = archived_count

        try:
            from remy.core.autonomy_rules import decay_stale_rules
            rules_archived = decay_stale_rules()
            if rules_archived:
                report["rules_archived"] = rules_archived
        except Exception as e:
            logger.debug("Rule decay skipped: %s", e)

        kb_synced = _sync_knowledge(brain)
        if kb_synced:
            report["knowledge_synced"] = kb_synced

        # ── TCS Step 7: Observation log — after all decisions ──
        try:
            if _thermal_report is not None:
                append_thermal_observation(
                    str(settings.AURA_BRAIN_PATH),
                    _thermal_report,
                    routing=report.get("consolidation", {}).get("thermal_routing"),
                    llm_gating=report.get("llm_gating"),
                )
        except Exception:
            pass

        report["total_records"] = brain.count()

    except OSError as e:
        import sys
        if sys.platform == "win32" and getattr(e, "winerror", None) in (5, 32):
            logger.warning("Background brain: compaction skipped (Windows file lock): %s", e)
        else:
            logger.error("Background processing failed: %s", e)
        report["error"] = str(e)
    except Exception as e:
        logger.error("Background processing failed: %s", e)
        report["error"] = str(e)

    try:
        cleanup_stats = _cleanup_files()
        if cleanup_stats["deleted_count"] > 0:
            report["file_cleanup"] = cleanup_stats
            logger.info("File cleanup: %s", cleanup_stats)
    except Exception as e:
        logger.error("File cleanup failed: %s", e)

    return report


# ============== RECORD ARCHIVAL ==============

# Tags → max age in days. Records older than this are deleted.
_ARCHIVAL_RULES: list[tuple[str, int, int]] = [
    # (tag, max_age_days, keep_recent_n)
    ("autonomous-outcome", 7, 50),    # Keep last 50 or 7 days
    ("session-summary", 14, 20),      # Keep last 20 or 14 days
    ("research-finding", 30, 100),    # Keep last 100 or 30 days
    ("web-search-cache", 1, 0),       # Already handled by Phase 7, but catch stragglers
    ("proactive-session", 7, 20),     # Keep last 20 or 7 days
    ("action-plan", 14, 10),          # Keep last 10 or 14 days
    ("eval-metric", 0, 0),            # Immediate cleanup (unwanted noise)
]


def _archive_old_records(brain) -> int:
    """Delete old transient records to prevent unbounded growth.

    For each tag category, keeps the N most recent records OR records
    younger than max_age_days, whichever is more generous.

    Thermal acceleration: tags in the cold zone get halved retention
    (archive sooner), because thermal mechanics confirmed they are
    stable and not contributing to active cognitive work.
    """
    # Build cold-tag set from thermal map (if available)
    cold_tags = set()
    try:
        from remy.core.thermal_advisor import get_maintenance_routing
        routing = get_maintenance_routing(str(settings.AURA_BRAIN_PATH))
        cold_tags = set(routing.cold_skip_tags)
    except Exception:
        pass

    total_archived = 0

    for tag, max_age_days, keep_recent_n in _ARCHIVAL_RULES:
        try:
            records = brain.search(query="", tags=[tag], limit=500)
            if len(records) <= keep_recent_n:
                continue

            # Thermal acceleration: cold-zone tags get halved retention
            effective_age = max_age_days
            if cold_tags and tag in cold_tags and max_age_days > 1:
                effective_age = max(1, max_age_days // 2)

            cutoff = (datetime.now() - timedelta(days=effective_age)).isoformat()

            # Records are sorted by recency (most recent first).
            # Keep the first keep_recent_n, then archive the rest if older than cutoff.
            candidates = records[keep_recent_n:] if keep_recent_n > 0 else records
            archived = 0

            for rec in candidates:
                meta = rec.metadata or {}
                created = meta.get("timestamp") or meta.get("created_at", "")
                if not created or created < cutoff:
                    try:
                        brain.delete(rec.id)
                        archived += 1
                    except Exception:
                        pass

            if archived:
                accel = " (thermal-accelerated)" if tag in cold_tags else ""
                logger.info("Archived %d old '%s' records%s", archived, tag, accel)
                total_archived += archived

        except Exception as e:
            logger.debug("Archival for tag '%s' failed: %s", tag, e)

    return total_archived


# ============== KNOWLEDGE SYNC (Phase 9) ==============

# Tags that should NOT be mirrored to knowledge (transient/system records)
_KB_SYNC_SKIP_TAGS = frozenset({
    "web-search-cache", "session-summary", "proactive-session",
    "action-plan", "session-reflection", "consolidated-meta",
    "autonomous-outcome", "autonomous-goal", "relationship",
    "todo-item", "push-subscription", "feedback-signal",
    # Operational noise — not long-term knowledge
    "eval-metric", "agent_invoke", "autonomous-outcome", "autonomous",
    "sandbox", "generation", "stability", "test",
})

# Tags that explicitly opt a record IN regardless of level
_KB_SYNC_FORCE_TAGS = frozenset({
    "fact", "extracted-fact", "knowledge", "permanent",
    "research-report", "store_research", "user-profile", "identity",
    "health-metric", "person", "story",
})

_KB_SYNC_MAX_PER_RUN = 50


def _sync_knowledge(brain) -> int:
    """Phase 9: Mirror meaningful long-term brain records to Aura Memory.

    Only syncs records that are DOMAIN (L3) or IDENTITY (L4) level,
    OR are explicitly tagged with knowledge-worthy tags (fact, research-report, etc.).
    WORKING (L1) and DECISIONS (L2) records are skipped unless force-tagged.

    Returns count of records synced.
    """
    try:
        from remy.core.agent_tools import knowledge, knowledge_lock
        if knowledge is None:
            return 0
    except ImportError:
        return 0

    try:
        from remy.core.event_bus import event_bus
    except ImportError:
        event_bus = None

    from remy.core.agent_tools import Level

    records = brain.list_records(min_strength=0.1)
    if not records:
        return 0

    candidates = []
    for rec in records:
        tags = set(rec.tags or [])
        meta = rec.metadata or {}

        if meta.get("mirrored_to_kb"):
            continue
        if not rec.content or len(rec.content.strip()) < 20:
            continue

        # Force-include explicitly marked knowledge records regardless of level
        force_include = bool(tags & _KB_SYNC_FORCE_TAGS)

        # Skip transient/system tags (unless force-included)
        if not force_include and (tags & _KB_SYNC_SKIP_TAGS):
            continue

        # Level gate: only DOMAIN (3) and IDENTITY (4) unless force-included
        if not force_include and rec.level not in (Level.DOMAIN, Level.IDENTITY):
            continue

        candidates.append(rec)
        if len(candidates) >= _KB_SYNC_MAX_PER_RUN:
            break

    if not candidates:
        return 0

    if event_bus:
        event_bus.emit("knowledge_sync_start", {"candidates": len(candidates)})

    synced = 0
    for rec in candidates:
        try:
            pin = (rec.level == Level.IDENTITY)
            text = (rec.content or "").strip()[:2000]
            # Skip JSON-wrapped content blocks (Gemini API parts mistakenly serialized)
            if not text or len(text) < 20:
                continue
            if text.startswith("{'type':") or text.startswith('{"type":'):
                logger.debug("KB sync: skipping JSON-wrapped content for %s", rec.id)
                continue
            with knowledge_lock:
                knowledge.process(text, pin=pin)
            meta = dict(rec.metadata or {})
            meta["mirrored_to_kb"] = True
            brain.update(rec.id, metadata=meta)
            synced += 1
        except Exception as e:
            logger.debug("KB sync failed for %s: %s", rec.id, e)

    if synced > 0:
        try:
            with knowledge_lock:
                knowledge.flush()
        except Exception:
            pass
        if event_bus:
            event_bus.emit("knowledge_sync_end", {"synced": synced})
        logger.info("Knowledge sync: %d records mirrored to KB", synced)

    return synced


def _format_all_insights(insights: list[dict]) -> list[str]:
    """Format insights into human-readable strings for transient context.

    Returns list of formatted insight strings. NOT stored as brain records.
    """
    result = []
    for ins in insights:
        content = _format_insight(ins)
        if content:
            result.append(content)
            logger.debug("Transient insight: %s", content[:80])
    return result[:5]  # Max 5 transient insights


def _format_correction_review_insights(brain) -> list[str]:
    """Surface bounded correction pressure as transient background insight."""
    result = []
    try:
        suggestions = brain.get_suggested_corrections(limit=2) if hasattr(brain, "get_suggested_corrections") else []
        queue = brain.get_correction_review_queue(limit=2) if hasattr(brain, "get_correction_review_queue") else []
        if suggestions:
            top = suggestions[0]
            if isinstance(top, dict):
                text = top.get("suggested_action") or top.get("reason_detail") or "review suggested correction"
                result.append(f"Correction pressure: {text[:90]}")
        if queue:
            result.append(f"Correction review queue has {len(queue)} candidate(s).")
    except Exception as e:
        logger.debug("correction review insight failed: %s", e)
    return result[:2]


def _format_insight(ins: dict) -> str | None:
    """Format an insight into a natural language string for storage.

    Returns None for insight types we don't want to store.
    Uses ACL renderer for deterministic, locale-aware output.
    """
    from remy.core.acl_renderer import Locale, render_insight

    # Skip insight records to avoid recursive nesting
    _insight_prefixes = ("Memories fading:", "Memories matured", "Knowledge cluster",
                         "Possible contradiction", "Indirect link:",
                         "Memories fading:", "Спогади згасають:",
                         "CONFLICT:", "КОНФЛIКТ:")

    def _filter_records(recs):
        """Filter out records that are themselves insights/cross-connections."""
        return [r for r in recs
                if not any(r.get("content", "").startswith(p) for p in _insight_prefixes)]

    ins_type = ins["type"]
    details = ins.get("details", {})

    if ins_type == "decay_risk":
        records = _filter_records(details.get("records", []))
        if not records:
            return None
        names = [r["content"][:60] for r in records[:3]]
        return render_insight("decay", {"names": names}, Locale.EN)

    elif ins_type == "conflict":
        pairs = details.get("pairs", [])
        if not pairs:
            return None
        p = pairs[0]
        if any(p.get(k, "").startswith(_insight_prefixes) for k in ("content_a", "content_b")):
            return None
        return render_insight("conflict", {
            "content_a": p.get("content_a", "?"),
            "content_b": p.get("content_b", "?"),
        }, Locale.EN)

    elif ins_type == "cluster":
        tags = details.get("dominant_tags", [])
        size = details.get("size", 0)
        if not tags:
            return None
        return render_insight("cluster", {"tags": tags[:5], "size": size}, Locale.EN)

    elif ins_type == "promotion":
        records = details.get("records", [])
        ready = _filter_records([r for r in records if r.get("can_promote")])
        if not ready:
            return None
        names = [r["content"][:60] for r in ready[:3]]
        return render_insight("promotion", {"names": names}, Locale.EN)

    return None


# ============== MEMORY CONSOLIDATION ==============

# Tags that should never be consolidated (system/transient records)
_CONSOLIDATION_SKIP_TAGS = frozenset([
    "user-profile", "identity", "session-summary", "scheduled-task",
    "autonomous-goal", "autonomous-outcome", "action-plan",
    "sandbox", "background-insight", "cross-connection",
    "session-reflection", "consolidated-meta", "web-search-cache",
    "research-project", "research-finding",
    "health-metric", "extracted-fact", "todo-item",
])

# Minimum cluster size to trigger consolidation
_MIN_CLUSTER_SIZE = 3
# Max clusters to process per run (limit LLM calls)
_MAX_CLUSTERS_PER_RUN = 3
# Min content similarity score to include in a cluster
_MIN_SIMILARITY_SCORE = 0.3


def _consolidate_records(brain) -> dict:
    """Merge clusters of similar records into summarized meta-records.

    Process:
    1. Group records by shared tags (find natural clusters)
    2. Verify content similarity within clusters via recall_structured
    3. LLM-summarize each cluster into a meta-record
    4. Connect meta-record to originals, demote originals

    Returns dict with counts: clusters_found, records_merged, meta_records_created.
    """
    from remy.core.event_bus import event_bus

    result = {"clusters_found": 0, "records_merged": 0, "meta_records_created": 0}

    try:
        clusters = _find_consolidation_clusters(brain)
        result["clusters_found"] = len(clusters)

        if not clusters:
            return result

        # Thermal routing: hot-first ordering is the default maintenance path.
        # Falls back to original order only if thermal_advisor is unavailable.
        from remy.core.thermal_advisor import get_maintenance_routing, sort_clusters_by_thermal_priority
        routing = get_maintenance_routing(str(settings.AURA_BRAIN_PATH))
        clusters, deferred, routing_stats = sort_clusters_by_thermal_priority(clusters, routing)
        result["thermal_routing"] = routing_stats
        if deferred:
            logger.info(
                "Thermal routing: %d prioritized, %d cold-deferred (cycle %d, mode=%s)",
                len(clusters), len(deferred), routing.cycle_number, routing.mode,
            )

        event_bus.emit("consolidation_start", {
            "clusters_found": len(clusters),
        })

        processed = 0
        for tag_key, record_ids in clusters:
            if processed >= _MAX_CLUSTERS_PER_RUN:
                break

            meta_id = _merge_cluster(brain, tag_key, record_ids)
            if meta_id:
                result["records_merged"] += len(record_ids)
                result["meta_records_created"] += 1
                processed += 1

                event_bus.emit("consolidation_merged", {
                    "tags": tag_key,
                    "records_merged": len(record_ids),
                    "meta_record_id": meta_id,
                })

        if result["meta_records_created"] > 0:
            event_bus.emit("consolidation_end", {
                "clusters_processed": result["meta_records_created"],
                "total_records_merged": result["records_merged"],
            })

    except Exception as e:
        logger.error("Memory consolidation failed: %s", e)

    return result


def _find_consolidation_clusters(brain) -> list[tuple[str, list[str]]]:
    """Find groups of similar records that should be consolidated.

    Groups records by their tag sets (excluding system tags), then filters
    to clusters of MIN_CLUSTER_SIZE+ records. Skips records that are already
    consolidated meta-records or belong to skip-list tag categories.

    Returns list of (tag_key, [record_ids]) sorted by cluster size descending.
    """
    records = brain.list_records(min_strength=0.3)
    if len(records) < _MIN_CLUSTER_SIZE:
        return []

    # Group by user-facing tags (sorted, frozen for hashing)
    tag_groups: dict[str, list] = defaultdict(list)

    for rec in records:
        tags = set(rec.tags or [])

        # Skip system records
        if tags & _CONSOLIDATION_SKIP_TAGS:
            continue

        # Skip records already consolidated
        meta = rec.metadata or {}
        if meta.get("consolidated_into"):
            continue

        # Build tag key from user-facing tags only
        user_tags = sorted(tags - _CONSOLIDATION_SKIP_TAGS)
        if not user_tags:
            continue

        tag_key = ",".join(user_tags)
        tag_groups[tag_key].append(rec)

    # Filter to clusters of sufficient size, sort by size descending
    clusters = []
    for tag_key, recs in tag_groups.items():
        if len(recs) >= _MIN_CLUSTER_SIZE:
            # Verify content similarity using recall_structured
            verified = _verify_cluster_similarity(brain, recs)
            if len(verified) >= _MIN_CLUSTER_SIZE:
                clusters.append((tag_key, [r.id for r in verified]))

    clusters.sort(key=lambda x: len(x[1]), reverse=True)
    return clusters


def _verify_cluster_similarity(brain, records: list) -> list:
    """Verify that records in a tag-based cluster are actually content-similar.

    Uses the first record's content as a query for recall_structured,
    then keeps only records that score above the similarity threshold.
    Falls back to tag-based grouping if recall doesn't return enough results.
    """
    if len(records) < _MIN_CLUSTER_SIZE:
        return records

    # Use the longest record as the representative query
    anchor = max(records, key=lambda r: len(r.content or ""))
    query = (anchor.content or "")[:200]

    if not query.strip():
        return records  # Can't verify, assume tag-match is good enough

    try:
        similar = brain.recall_structured(query, top_k=50, min_strength=0.1)
        similar_ids = {r["id"] for r in similar if r.get("score", 0) >= _MIN_SIMILARITY_SCORE}
        # Keep records that appear in similarity results
        verified = [r for r in records if r.id in similar_ids]
        if len(verified) >= _MIN_CLUSTER_SIZE:
            return verified
        # Recall may not find all records (embedding dedup, small corpus).
        # Trust tag-based grouping if records share the same tag set.
        return records
    except Exception:
        # If recall fails, trust tag-based grouping
        return records


def _merge_cluster(brain, tag_key: str, record_ids: list[str]) -> str | None:
    """Merge a cluster of records into a single meta-record.

    1. Collect all record contents
    2. LLM-summarize into a consolidated record
    3. Store as DOMAIN-level meta-record with consolidated-meta tag
    4. Connect meta-record to all originals
    5. Mark originals with consolidated_into metadata

    Returns meta-record ID on success, None on failure.
    """
    from remy.core.agent_tools import Level

    # Collect source content
    sources = []
    for rid in record_ids:
        rec = brain.get(rid)
        if rec:
            sources.append({
                "id": rec.id,
                "content": rec.content,
                "tags": rec.tags,
                "level": rec.level,
            })

    if len(sources) < _MIN_CLUSTER_SIZE:
        return None

    # Generate consolidated summary via LLM
    summary = _generate_consolidation_summary(tag_key, sources)
    if not summary:
        return None

    # Check for data loss — abort if summary drops important values
    lost = _check_data_loss(sources, summary)
    if lost:
        logger.warning(
            "Consolidation would lose data — keeping originals. Lost values: %s",
            lost[:5],
        )
        return None

    # Store meta-record
    tags_list = [t.strip() for t in tag_key.split(",") if t.strip()]
    meta_record = brain.store(
        content=summary,
        level=Level.DOMAIN,
        tags=tags_list + ["consolidated-meta"],
        metadata={
            "type": "consolidation",
            "source_count": len(sources),
            "source_ids": [s["id"] for s in sources],
            "consolidated_at": datetime.now().isoformat(),
        },
    )

    meta_id = meta_record.id

    # Connect meta-record to all sources and mark them (promotion-gated)
    from remy.core.agent_tools import gated_connect
    for source in sources:
        try:
            gated_connect(brain, meta_id, source["id"], weight=0.8)
            brain.update(
                source["id"],
                metadata={
                    **(brain.get(source["id"]).metadata or {}),
                    "consolidated_into": meta_id,
                },
            )
        except Exception as e:
            logger.warning("Failed to link source %s to meta %s: %s", source["id"], meta_id, e)

    logger.info(
        "Consolidated %d records [%s] into meta-record %s",
        len(sources), tag_key, meta_id,
    )
    return meta_id


def _generate_consolidation_summary(tag_key: str, sources: list[dict]) -> str | None:
    """Generate a consolidated summary of multiple similar records via LLM.

    Returns the summary text, or None if LLM call fails.
    """
    # Build the source content block (cap total to ~2000 chars)
    content_parts = []
    total_chars = 0
    for s in sources:
        snippet = (s["content"] or "")[:300]
        if total_chars + len(snippet) > 2000:
            break
        content_parts.append(f"- {snippet}")
        total_chars += len(snippet)

    sources_text = "\n".join(content_parts)

    prompt = (
        "You are consolidating multiple related memory records into one concise summary.\n\n"
        f"Topic tags: {tag_key}\n"
        f"Number of records: {len(sources)}\n\n"
        f"Records:\n{sources_text}\n\n"
        "Write a single consolidated summary that:\n"
        "1. Preserves ALL unique facts and details from the records\n"
        "2. Removes redundancy and duplication\n"
        "3. Organizes information logically\n"
        "4. Keeps the same language as the original records\n"
        "5. Is concise but complete (aim for 2-4 sentences)\n\n"
        "Consolidated summary:"
    )

    try:
        from remy.core.llm import call_llm

        result = call_llm(prompt, purpose="consolidation")
        content = result.content
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        summary = str(content).strip()

        if len(summary) < 10:
            logger.warning("Consolidation summary too short, skipping")
            return None

        return summary

    except Exception as e:
        logger.error("Consolidation LLM call failed: %s", e)
        return None


def _check_data_loss(sources: list[dict], summary: str) -> list[str]:
    """Check if consolidation summary lost important structured values.

    Extracts emails, phone numbers, URLs, and monetary amounts from sources
    and verifies they appear in the summary. Returns list of lost values.
    """
    import re

    original = "\n".join(s.get("content", "") or "" for s in sources)

    # URL regex — only track URLs with a meaningful path (not bare domains like https://dev.to/)
    _bare_domain_re = re.compile(r"^https?://[^/]+/?$")

    extractors = {
        "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        "url": re.compile(r"https?://[^\s)<>\"]+"),
        "money": re.compile(r"\$[\d,.]+|\d+\s*(?:USD|EUR|UAH|TRX|USDT)"),
    }

    # Phone regex with date exclusion
    phone_re = re.compile(r"\b\d[\d\s\-]{8,}\d\b")
    date_re = re.compile(r"\d{4}-\d{2}-\d{2}")

    lost = []
    for label, pattern in extractors.items():
        for m in pattern.finditer(original):
            val = m.group().strip()
            if label == "url" and _bare_domain_re.match(val):
                continue  # skip bare domains — not unique identifiers
            if val not in summary:
                lost.append(f"{label}:{val}")

    # Check phones separately — skip ISO dates
    for m in phone_re.finditer(original):
        matched = m.group().strip()
        if date_re.match(matched):
            continue
        digits = re.sub(r"\D", "", matched)
        if len(digits) >= 7 and digits not in re.sub(r"\D", "", summary):
            lost.append(f"phone:{matched}")

    return lost


# ============== KNOWLEDGE SYNTHESIS ==============


def _discover_cross_connections(brain) -> list[str]:
    """2-hop graph walk to find indirect connections worth noting.

    Zero LLM calls — pure Python graph traversal.
    Returns list of discovery strings.
    """
    records = brain.list_records(min_strength=0.3)
    if len(records) < 5:
        return []

    # Only consider records that have connections AND are not already cross-connections
    connected = [
        r for r in records
        if r.connections and "cross-connection" not in (r.tags or [])
    ]
    if len(connected) < 3:
        return []

    sample = random.sample(connected, min(10, len(connected)))
    discoveries = []
    seen = set()

    for rec in sample:
        for hop1_id in rec.connections:
            hop1 = brain.get(hop1_id)
            if not hop1 or not hop1.connections:
                continue
            if "cross-connection" in (hop1.tags or []):
                continue
            for hop2_id in hop1.connections:
                # Skip if hop2 is the original record or already directly connected
                if hop2_id == rec.id or hop2_id in rec.connections:
                    continue
                # Avoid duplicates
                pair_key = tuple(sorted([rec.id, hop2_id]))
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                hop2 = brain.get(hop2_id)
                if hop2:
                    discoveries.append(
                        f"Indirect link: '{rec.content[:50]}' "
                        f"-> '{hop1.content[:50]}' "
                        f"-> '{hop2.content[:50]}'"
                    )

    return discoveries[:3]  # Max 3 per run



# ============== SEARCH CACHE EXPIRY ==============


def _expire_search_cache(brain) -> int:
    """Delete web-search-cache records older than 24 hours.

    Prevents unbounded growth in long-running server deployments.
    Returns number of expired records deleted.
    """
    try:
        cache_records = brain.search(query="", tags=["web-search-cache"], limit=200)
        if not cache_records:
            return 0

        now = datetime.now()
        expired = 0
        for r in cache_records:
            meta = getattr(r, "metadata", None) or {}
            cached_at = meta.get("cached_at", "")
            if not cached_at:
                # No timestamp — delete if record is old (fallback to created_at)
                created = getattr(r, "created_at", None)
                if created and (now - created) > timedelta(hours=24):
                    brain.delete(r.id)
                    expired += 1
                continue

            try:
                cache_time = datetime.fromisoformat(cached_at)
                if (now - cache_time) > timedelta(hours=24):
                    brain.delete(r.id)
                    expired += 1
            except (ValueError, TypeError):
                pass

        if expired:
            logger.info("Cache expiry: deleted %d stale web-search-cache records", expired)
        return expired
    except Exception as e:
        logger.debug("Cache expiry check failed: %s", e)
        return 0


# ============== SCHEDULED TASK CHECKER ==============


def _check_scheduled_tasks(brain) -> list[str]:
    """Check for due scheduled tasks and generate reminders.

    Zero LLM calls — uses brain.recall for context enrichment.
    Returns list of reminder strings.
    """
    tasks = brain.search(query="", tags=["scheduled-task"], limit=200)
    if not tasks:
        return []

    now = datetime.now()
    tomorrow = now + timedelta(days=1)
    reminders = []

    for task in tasks:
        meta = task.metadata or {}
        if meta.get("status") != "active":
            continue

        due_str = meta.get("due_date", "")
        if not due_str:
            continue

        try:
            due_date = datetime.fromisoformat(due_str)
        except (ValueError, TypeError):
            continue

        # Check if task is due (today or tomorrow for advance warning)
        if due_date.date() > tomorrow.date():
            continue

        # Skip if already reminded today (persistent dedup)
        today_str = now.date().isoformat()
        if meta.get("last_reminded_date") == today_str:
            continue

        description = meta.get("description", task.content)

        # Get context from memory (cheap recall, no LLM)
        context = ""
        try:
            recall_result = brain.recall(description, token_budget=256)
            if recall_result and "No relevant" not in recall_result:
                context = f" Context: {recall_result[:200]}"
        except Exception:
            pass

        if due_date.date() == now.date():
            reminder = f"Due today: {description}.{context}"
        else:
            reminder = f"Due tomorrow: {description}.{context}"

        reminders.append(reminder)

        # Mark as reminded today (persists across restarts)
        meta["last_reminded_date"] = today_str
        brain.update(task.id, metadata=meta)

        # Handle recurring tasks — advance the due_date
        repeat = meta.get("repeat")
        if repeat and due_date.date() <= now.date():
            new_due = _advance_due_date(due_date, repeat)
            if new_due:
                brain.update(
                    task.id,
                    metadata={**meta, "due_date": new_due.isoformat()},
                )
                logger.info(f"Advanced recurring task '{description}' to {new_due.date()}")
        elif not repeat and due_date.date() < now.date():
            # One-time task that is past due — mark as done so it doesn't trigger forever
            days_overdue = (now.date() - due_date.date()).days
            if days_overdue >= 2:
                meta["status"] = "done"
                meta["completed_at"] = now.isoformat()
                meta["updated_at"] = now.isoformat()
                meta["status_notes"] = f"Auto-completed: one-time task {days_overdue}d overdue"
                brain.update(task.id, metadata=meta)
                logger.info(
                    "Auto-completed one-time scheduled task '%s' (%dd overdue)",
                    description[:60], days_overdue,
                )

    return reminders


def _check_overdue_todos(brain) -> list[str]:
    """Check for overdue todo items and generate reminders.

    Returns list of reminder strings for todos past their due_date.
    """
    todos = brain.search(query="", tags=["todo-item"], limit=100)
    if not todos:
        return []

    now = datetime.now()
    reminders = []

    for rec in todos:
        meta = rec.metadata or {}
        if meta.get("type") != "todo_item":
            continue
        status = meta.get("status", "pending")
        if status in ("done", "archived"):
            continue
        due_str = meta.get("due_date")
        if not due_str:
            continue
        try:
            due_date = datetime.fromisoformat(due_str)
        except (ValueError, TypeError):
            continue
        if due_date.date() < now.date():
            title = meta.get("title", rec.content)
            priority = meta.get("priority", "medium")
            reminders.append(f"Overdue todo [{priority}]: {title} (was due {due_date.date()})")
        elif due_date.date() == now.date():
            title = meta.get("title", rec.content)
            reminders.append(f"Todo due today: {title}")

    return reminders


def _advance_due_date(current: datetime, repeat: str) -> datetime | None:
    """Calculate the next due date for a recurring task."""
    if repeat == "daily":
        return current + timedelta(days=1)
    elif repeat == "weekly":
        return current + timedelta(weeks=1)
    elif repeat == "monthly":
        # Simple month advance
        month = current.month + 1
        year = current.year
        if month > 12:
            month = 1
            year += 1
        try:
            return current.replace(year=year, month=month)
        except ValueError:
            # Handle edge case like Jan 31 -> Feb 28
            return current.replace(year=year, month=month, day=28)
    return None


# ============== PROACTIVE NOTIFICATIONS ==============


def build_notification_message(report: dict, brain) -> str | None:
    """Build a conversational notification message with follow-up questions.

    Zero LLM calls — pure Python template from brain data.
    Returns None if nothing worth notifying about.
    """
    has_content = (
        report.get("insights_found", 0) > 0
        or report.get("cross_connections", 0) > 0
        or report.get("task_reminders")
    )
    if not has_content:
        return None

    lines = []

    # Task reminders with follow-up questions (highest priority)
    task_reminders = report.get("task_reminders", [])
    for reminder in task_reminders[:3]:
        lines.append(reminder)

    # Generate follow-up question from last session
    try:
        summaries = brain.search(query="", tags=["session-summary"], limit=1)
        if summaries:
            last_summary = summaries[0].content[:200]
            lines.append(f"\nLast time we talked about: {last_summary}")
            lines.append("Want to continue where we left off?")
    except Exception:
        pass

    # Add insight-based question if no tasks (from transient insights, not brain records)
    if not task_reminders:
        transient = get_transient_insights()
        if transient:
            lines.append(f"\nI noticed something: {transient[0][:150]}")
            lines.append("Want me to look into this?")

    if not lines:
        return None

    # Conversational header instead of "Background Report"
    header = "Hey! Remy here with a quick update:"
    return header + "\n\n" + "\n".join(lines)


async def send_notifications(report: dict, brain=None):
    """Send notifications via Telegram and/or Web Push.

    Zero LLM calls — sends pre-formatted conversational text.
    """
    if brain is None:
        from remy.core.agent_tools import brain as default_brain
        brain = default_brain

    message = build_notification_message(report, brain)
    if not message:
        logger.info("Nothing to notify about")
        return

    try:
        from remy.core.notification_router import notify

        notify(message, level="info", event_type="background.report", parse_mode="")
    except Exception as e:
        logger.error(f"Failed to send notification: {e}")


# ============== REPORT ==============


def print_report(report: dict):
    """Print a human-readable report of background processing."""
    print("=" * 50)
    print("BACKGROUND BRAIN — Processing Report")
    print(f"Time: {report['timestamp']}")
    print(f"Total records: {report['total_records']}")
    print(f"Decay: {report['decay']['decayed']} decayed, {report['decay']['archived']} archived")
    print(f"Reflect: {report['reflect']['promoted']} promoted, {report['reflect']['connected']} connected")
    print(f"Insights found: {report.get('insights_found', 0)} (transient, not stored)")
    cons = report.get("consolidation", {})
    if cons.get("clusters_found", 0) > 0 or cons.get("native_merged", 0) > 0:
        print(f"Consolidation: {cons.get('native_merged', 0)} native merged, "
              f"{cons['clusters_found']} clusters found, "
              f"{cons['records_merged']} records merged into "
              f"{cons['meta_records_created']} meta-records")
    print(f"Cross-connections: {report.get('cross_connections', 0)}")
    reminders = report.get("task_reminders", [])
    if reminders:
        print(f"Task reminders: {len(reminders)}")
        for r in reminders:
            print(f"  - {r[:100]}")
    if "error" in report:
        print(f"ERROR: {report['error']}")
    print("=" * 50)


# ============== FILE CLEANUP (Phase 10) ==============


def _cleanup_files() -> dict:
    """"Cleanup old temporary files (screenshots, images, browser profile).

    Rules:
    1. Browser Screenshots: Delete if older than 1 hour (effectively session-end).
    2. Generated Images: Delete if older than 30 days.
    3. Browser Profile Cache: Delete Cache, Code Cache, GPUCache (preserve login).
    """
    import shutil
    import time

    stats = {"deleted_count": 0, "freed_bytes": 0, "errors": 0}
    now = time.time()

    # ensure we don't delete files currently in use (simple grace period)
    ONE_HOUR = 3600
    THIRTY_DAYS = 30 * 24 * 3600

    # 1. Browser Screenshots (data/browser_screenshots)
    ss_dir = settings.DATA_DIR / "browser_screenshots"
    if ss_dir.exists():
        for f in ss_dir.iterdir():
            if f.is_file() and (now - f.stat().st_mtime) > ONE_HOUR:
                try:
                    size = f.stat().st_size
                    f.unlink()
                    stats["deleted_count"] += 1
                    stats["freed_bytes"] += size
                except Exception:
                    stats["errors"] += 1

    # 2. Generated Images (data/generated_images)
    img_dir = settings.DATA_DIR / "generated_images"
    if img_dir.exists():
        for f in img_dir.iterdir():
            if f.is_file() and (now - f.stat().st_mtime) > THIRTY_DAYS:
                try:
                    size = f.stat().st_size
                    f.unlink()
                    stats["deleted_count"] += 1
                    stats["freed_bytes"] += size
                except Exception:
                    stats["errors"] += 1

    # 3. Browser Profile Cache (data/browser_profile/Default/...)
    # Target folders: Cache, Code Cache, GPUCache, DawnGraphiteCache
    profile_dir = settings.DATA_DIR / "browser_profile" / "Default"
    cache_folders = ["Cache", "Code Cache", "GPUCache", "DawnGraphiteCache"]

    if profile_dir.exists():
        for folder_name in cache_folders:
            folder_path = profile_dir / folder_name
            if folder_path.exists() and folder_path.is_dir():
                try:
                    # Calculate size before deletion for stats
                    folder_size = sum(f.stat().st_size for f in folder_path.glob('**/*') if f.is_file())
                    shutil.rmtree(folder_path)
                    stats["deleted_count"] += 1  # Count 1 folder as 1 item
                    stats["freed_bytes"] += folder_size
                except PermissionError:
                    # Browser is likely running and locking files
                    stats["errors"] += 1
                except Exception:
                    stats["errors"] += 1

    return stats

