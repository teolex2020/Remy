"""
Brain Voice V1 — proactive, locale-agnostic messages from the brain.

Emits structured BrainVoiceEvents when TCS signals cross significance
thresholds (hot-zone spikes, pruning bursts, routing shifts). Events
are stored structurally; rendering to natural language happens at the
edge via acl_renderer.render_brain_voice(event, locale).

Design rules:
- Zero LLM calls — deterministic event detection from already-persisted
  thermal + plasticity artifacts.
- No hardcoded user-facing language. Events carry typed fields only.
- Per-kind debounce + global rate limit so the brain never spams.
- State persists across restarts via brain record with tag
  `brain-voice-state` (single row, overwritten each cycle).
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional

logger = logging.getLogger("BrainVoice")

# ── Tunables ────────────────────────────────────────────────────────────────

HOT_ZONE_SPIKE_MIN_DELTA = 2           # new hot zones since last cycle
PRUNING_BURST_MIN_DELTA = 5            # new pruned edges since last cycle
ROUTING_DEBOUNCE_SEC = 6 * 3600        # 6 hours between routing-shift speaks
PER_KIND_DEBOUNCE_SEC = {
    "hot_zone_spike": 2 * 3600,
    "pruning_burst":  4 * 3600,
    "routing_shift":  ROUTING_DEBOUNCE_SEC,
}
GLOBAL_RATE_MAX_PER_HOUR = 3           # at most 3 proactive messages / hour
RECENT_WINDOW_LIMIT = 64               # keep last N events in outbox

# ── Event model ─────────────────────────────────────────────────────────────

EventKind = Literal["hot_zone_spike", "pruning_burst", "routing_shift"]
Severity = Literal["info", "notable", "urgent"]


@dataclass
class BrainVoiceEvent:
    event_id: str
    kind: EventKind
    severity: Severity
    payload: dict          # typed structural fields — NO prose
    timestamp: float       # unix seconds
    acknowledged: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BrainVoiceEvent":
        return cls(
            event_id=d["event_id"],
            kind=d["kind"],
            severity=d.get("severity", "notable"),
            payload=d.get("payload", {}),
            timestamp=float(d.get("timestamp", 0.0)),
            acknowledged=bool(d.get("acknowledged", False)),
        )


# ── Persistent state (between cycles) ───────────────────────────────────────

_STATE_FILENAME = "brain_voice_state.json"


@dataclass
class _CycleState:
    last_hot_zones: int = 0
    last_pruned: int = 0
    last_routing_mode: str = ""
    last_speak_ts_by_kind: dict = field(default_factory=dict)
    recent_events: list = field(default_factory=list)   # list of dict

    def to_dict(self) -> dict:
        return {
            "last_hot_zones": self.last_hot_zones,
            "last_pruned": self.last_pruned,
            "last_routing_mode": self.last_routing_mode,
            "last_speak_ts_by_kind": self.last_speak_ts_by_kind,
            "recent_events": self.recent_events,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "_CycleState":
        return cls(
            last_hot_zones=int(d.get("last_hot_zones", 0)),
            last_pruned=int(d.get("last_pruned", 0)),
            last_routing_mode=str(d.get("last_routing_mode", "")),
            last_speak_ts_by_kind=dict(d.get("last_speak_ts_by_kind", {})),
            recent_events=list(d.get("recent_events", [])),
        )


def _state_path(data_dir: str) -> str:
    return os.path.join(data_dir, _STATE_FILENAME)


def _load_state(data_dir: str) -> _CycleState:
    path = _state_path(data_dir)
    if not os.path.exists(path):
        return _CycleState()
    try:
        with open(path, encoding="utf-8") as f:
            return _CycleState.from_dict(json.load(f))
    except Exception as e:
        logger.warning("Failed to read brain voice state: %s — starting fresh", e)
        return _CycleState()


def _save_state(data_dir: str, state: _CycleState) -> None:
    try:
        os.makedirs(data_dir, exist_ok=True)
        with open(_state_path(data_dir), "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, ensure_ascii=True)
    except Exception as e:
        logger.warning("Failed to persist brain voice state: %s", e)


# ── Detection helpers ───────────────────────────────────────────────────────


def _global_rate_ok(state: _CycleState, now: float) -> bool:
    window_start = now - 3600
    recent_speaks = sum(
        1 for e in state.recent_events
        if float(e.get("timestamp", 0)) >= window_start
    )
    return recent_speaks < GLOBAL_RATE_MAX_PER_HOUR


def _kind_debounce_ok(state: _CycleState, kind: EventKind, now: float) -> bool:
    debounce = PER_KIND_DEBOUNCE_SEC.get(kind, 3600)
    last = float(state.last_speak_ts_by_kind.get(kind, 0))
    return (now - last) >= debounce


def _append_event(state: _CycleState, event: BrainVoiceEvent) -> None:
    state.recent_events.append(event.to_dict())
    if len(state.recent_events) > RECENT_WINDOW_LIMIT:
        state.recent_events = state.recent_events[-RECENT_WINDOW_LIMIT:]
    state.last_speak_ts_by_kind[event.kind] = event.timestamp


# ── Main entrypoint ─────────────────────────────────────────────────────────


def detect_and_record(data_dir: str) -> list[BrainVoiceEvent]:
    """Run detection against current TCS state; append new events; persist.

    Returns only NEWLY emitted events (so caller can route them).
    Called by the scheduler after each maintenance cycle.
    """
    from remy.core.thermal_advisor import compute_thermal_map, get_maintenance_routing
    from remy.core.synaptic_plasticity import get_plasticity_summary

    thermal = compute_thermal_map(data_dir)
    if thermal is None:
        return []

    routing = get_maintenance_routing(data_dir)
    plasticity = get_plasticity_summary(data_dir) or {}

    state = _load_state(data_dir)
    now = time.time()
    new_events: list[BrainVoiceEvent] = []

    cur_hot = int(thermal.hot_zone_count)
    cur_pruned = int(plasticity.get("pruned", 0))
    cur_mode = str(getattr(routing, "mode", "") or "")

    # ── Trigger 1: hot-zone spike ──────────────────────────────────────────
    if _global_rate_ok(state, now) and _kind_debounce_ok(state, "hot_zone_spike", now):
        delta = cur_hot - state.last_hot_zones
        conflict_clusters = sum(
            1 for c in thermal.clusters if getattr(c, "has_conflict", False)
        )
        if delta >= HOT_ZONE_SPIKE_MIN_DELTA and conflict_clusters > 0:
            severity: Severity = "urgent" if delta >= 5 else "notable"
            ev = BrainVoiceEvent(
                event_id=uuid.uuid4().hex[:16],
                kind="hot_zone_spike",
                severity=severity,
                payload={
                    "hot_zones_now": cur_hot,
                    "hot_zones_prev": state.last_hot_zones,
                    "delta": delta,
                    "conflict_clusters": conflict_clusters,
                    "mean_temperature": round(float(thermal.mean_temperature), 4),
                },
                timestamp=now,
            )
            _append_event(state, ev)
            new_events.append(ev)

    # ── Trigger 2: pruning burst ──────────────────────────────────────────
    if _global_rate_ok(state, now) and _kind_debounce_ok(state, "pruning_burst", now):
        pruned_delta = cur_pruned - state.last_pruned
        if pruned_delta >= PRUNING_BURST_MIN_DELTA:
            severity = "urgent" if pruned_delta >= 20 else "notable"
            ev = BrainVoiceEvent(
                event_id=uuid.uuid4().hex[:16],
                kind="pruning_burst",
                severity=severity,
                payload={
                    "pruned_now": cur_pruned,
                    "pruned_prev": state.last_pruned,
                    "delta": pruned_delta,
                    "healthy": int(plasticity.get("healthy", 0)),
                    "weakened": int(plasticity.get("weakened", 0)),
                },
                timestamp=now,
            )
            _append_event(state, ev)
            new_events.append(ev)

    # ── Trigger 3: routing shift ──────────────────────────────────────────
    if (
        cur_mode
        and state.last_routing_mode
        and cur_mode != state.last_routing_mode
        and _global_rate_ok(state, now)
        and _kind_debounce_ok(state, "routing_shift", now)
    ):
        ev = BrainVoiceEvent(
            event_id=uuid.uuid4().hex[:16],
            kind="routing_shift",
            severity="info",
            payload={
                "from_mode": state.last_routing_mode,
                "to_mode": cur_mode,
                "cycle_number": int(getattr(routing, "cycle_number", 0)),
            },
            timestamp=now,
        )
        _append_event(state, ev)
        new_events.append(ev)

    # Update baseline regardless of whether we spoke (so deltas stay sane)
    state.last_hot_zones = cur_hot
    state.last_pruned = cur_pruned
    if cur_mode:
        state.last_routing_mode = cur_mode

    _save_state(data_dir, state)

    if new_events:
        logger.info(
            "brain_voice emitted %d event(s): %s",
            len(new_events),
            [e.kind for e in new_events],
        )
    return new_events


# ── Outbox (consumed by API / UI) ───────────────────────────────────────────


def recent_events(data_dir: str, *, since_ts: Optional[float] = None, limit: int = 20) -> list[dict]:
    """Return recent brain voice events (newest first) for UI polling."""
    state = _load_state(data_dir)
    events = list(state.recent_events)
    if since_ts is not None:
        events = [e for e in events if float(e.get("timestamp", 0)) > since_ts]
    events.sort(key=lambda e: float(e.get("timestamp", 0)), reverse=True)
    return events[:limit]


def acknowledge(data_dir: str, event_id: str) -> bool:
    """Mark an event as acknowledged so the UI can hide it."""
    state = _load_state(data_dir)
    changed = False
    for e in state.recent_events:
        if e.get("event_id") == event_id and not e.get("acknowledged"):
            e["acknowledged"] = True
            changed = True
            break
    if changed:
        _save_state(data_dir, state)
    return changed
