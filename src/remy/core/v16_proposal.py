"""
V16 Proposal Mode — autonomic drive awareness and proposal rendering for Remi.

This module connects the AuraSDK V16 autonomic drive layer to Remi's autonomy loop.
It does NOT execute actions — it reports what the cognitive substrate wants,
why it wants it, and what safety gates would allow or block.

Integration points:
  - background_brain.py: collect_v16_diagnostics() — runs each maintenance cycle
  - autonomy.py _cycle(): render_proposals() — after _decide_and_act()
  - system_instruction.py: proposal context for LLM awareness
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("V16Proposal")

# ── JSONL log path ─────────────────────────────────────────────────────────

_LOG_DIR = Path("data/logs")
_PROPOSAL_LOG = _LOG_DIR / "v16_proposals.jsonl"

# ── Allowed action classes in proposal mode ────────────────────────────────

DIRECTLY_ALLOWED = {"Internal", "Observe", "Report"}
PROPOSAL_ONLY = {"Propose", "Optimize", "Respond", "Reinforce"}


def _serialize(obj, depth=0):
    """Serialize PyO3 object to JSON-safe dict."""
    if depth > 8:
        return str(obj)
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_serialize(x, depth + 1) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _serialize(v, depth + 1) for k, v in obj.items()}
    obj_repr = repr(obj)
    if "." in obj_repr and obj_repr.split(".")[-1].isidentifier():
        attrs = [a for a in dir(obj) if not a.startswith("_") and not callable(getattr(obj, a, None))]
        if attrs and all(type(getattr(obj, a, None)) is type(obj) for a in attrs[:3]):
            return str(obj).split(".")[-1] if "." in str(obj) else str(obj)
    attrs = [a for a in dir(obj) if not a.startswith("_")]
    if attrs:
        result = {}
        for a in attrs:
            v = getattr(obj, a, None)
            if callable(v):
                continue
            result[a] = _serialize(v, depth + 1)
        return result if result else str(obj)
    return str(obj)


def _log_jsonl(entry: dict):
    """Append one JSON line to the V16 proposal log."""
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(_PROPOSAL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug("V16 JSONL log write failed: %s", e)


# ── Runtime signal reporting ───────────────────────────────────────────────

def report_wake_signal(brain, wake_reason: str = "timer", last_activity_secs: int = 0):
    """Report a wake-cycle RuntimeSignal to the cognitive substrate.

    Called at the start of each autonomy cycle to let V16 detect
    orientation loss and generate appropriate tensions.
    """
    try:
        from aura import RuntimeSignal, SignalPayload, WakeReason

        now = int(time.time())

        # Map string wake reason to enum
        reason_map = {
            "timer": WakeReason.Timer,
            "user_message": WakeReason.UserMessage,
            "system_event": WakeReason.SystemEvent,
            "manual": WakeReason.Manual,
        }
        reason = reason_map.get(wake_reason, WakeReason.Timer)

        # Report wake reason (informational)
        wake_signal = RuntimeSignal("remi", now, 0.95, SignalPayload.wake_reason(reason))
        brain.report_runtime_signal(wake_signal)

        # Report inactivity gap if significant
        if last_activity_secs > 0:
            gap_signal = RuntimeSignal("remi", now, 0.90,
                                       SignalPayload.inactivity_gap(last_activity_secs))
            brain.report_runtime_signal(gap_signal)

    except Exception as e:
        logger.debug("V16 wake signal failed: %s", e)


def report_resource_signal(brain, budget_remaining_ratio: float, tokens_used: int,
                           budget_ceiling: int | None = None):
    """Report current resource budget state to V16."""
    try:
        from aura import RuntimeSignal, SignalPayload

        now = int(time.time())
        signal = RuntimeSignal("remi", now, 0.90,
                               SignalPayload.resource_report(budget_remaining_ratio,
                                                            tokens_used,
                                                            budget_ceiling))
        brain.report_runtime_signal(signal)
    except Exception as e:
        logger.debug("V16 resource signal failed: %s", e)


# ── Diagnostics collection ─────────────────────────────────────────────────

def collect_v16_diagnostics(brain) -> dict:
    """Collect V16 autonomic state for the background maintenance report.

    Returns a dict suitable for inclusion in the background_brain report.
    Zero LLM cost — all data comes from the cognitive substrate.
    """
    result = {}
    try:
        # Kill switch / freeze state
        ks = _serialize(brain.get_kill_switch_state())
        result["kill_switch"] = ks

        # Active drives by priority
        drives = brain.get_active_drives_by_priority(10)
        drives_s = [_serialize(d) for d in drives]
        result["active_drives"] = len(drives_s)
        result["top_drives"] = drives_s[:5]

        # Tension diagnostics
        diag = _serialize(brain.get_tension_diagnostics())
        result["tension_diagnostics"] = diag

        # Safety gates for top drives
        gates = []
        for d in drives[:3]:
            try:
                gate = brain.get_safety_gate(d.id)
                if gate:
                    gates.append(_serialize(gate))
            except Exception:
                pass
        result["safety_gates"] = gates

    except Exception as e:
        result["error"] = str(e)
        logger.debug("V16 diagnostics collection failed: %s", e)

    return result


# ── Proposal rendering ─────────────────────────────────────────────────────

def render_proposals(brain, cycle_num: int = 0, session_id: str = "") -> list[dict]:
    """Generate proposals from active drives.

    Called after _decide_and_act() in the autonomy cycle.
    Returns list of proposal dicts (also logged to JSONL).
    """
    proposals = []
    try:
        drives = brain.get_active_drives_by_priority(5)
        if not drives:
            return proposals

        for drive in drives:
            ds = _serialize(drive)

            # Get safety gate
            gate = None
            gate_s = {}
            try:
                gate = brain.get_safety_gate(drive.id)
                gate_s = _serialize(gate) if gate else {}
            except Exception:
                pass

            allowed_actions = gate_s.get("allowed_actions", [])
            frozen = gate_s.get("frozen", False)
            requires_approval = gate_s.get("requires_approval", False)

            # Classify: can we execute directly, or is this proposal-only?
            directly_executable = all(a in DIRECTLY_ALLOWED for a in allowed_actions)
            proposal_only_actions = [a for a in allowed_actions if a in PROPOSAL_ONLY]

            status = "blocked" if frozen else (
                "executable" if directly_executable else "proposal_only"
            )

            proposal = {
                "drive_id": ds.get("id", ""),
                "tension_source": ds.get("tension_source", ""),
                "priority_class": ds.get("priority_class", ""),
                "priority": ds.get("priority", 0),
                "imperative": ds.get("imperative", ""),
                "linked_goal": ds.get("linked_goal"),
                "safety_status": status,
                "allowed_actions": allowed_actions,
                "proposal_only_actions": proposal_only_actions,
                "frozen": frozen,
                "freeze_reason": gate_s.get("freeze_reason"),
                "requires_approval": requires_approval,
                "approval_reason": gate_s.get("approval_reason"),
                "token_budget_remaining": gate_s.get("token_budget_remaining"),
            }
            proposals.append(proposal)

        # Log to JSONL
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "proposal_cycle",
            "cycle_num": cycle_num,
            "session_id": session_id,
            "proposals_count": len(proposals),
            "proposals": proposals,
        }
        _log_jsonl(entry)

        if proposals:
            logger.info("V16 proposals: %d drives active (top: %s %s)",
                        len(proposals),
                        proposals[0]["priority_class"],
                        proposals[0]["tension_source"])

    except Exception as e:
        logger.debug("V16 proposal rendering failed: %s", e)

    return proposals


def format_proposal_summary(proposals: list[dict]) -> str:
    """Format proposals into a human-readable summary for operator display."""
    if not proposals:
        return ""

    lines = ["[Autonomic Drive Status]"]
    for i, p in enumerate(proposals):
        status_icon = {
            "executable": "[OK]",
            "proposal_only": "[PROPOSAL]",
            "blocked": "[BLOCKED]",
        }.get(p["safety_status"], "[?]")

        lines.append(
            f"  {status_icon} {p['priority_class']} | {p['tension_source']} | "
            f"priority={p['priority']:.2f}"
        )
        if p["imperative"]:
            lines.append(f"    Why: {p['imperative'][:120]}")
        if p["proposal_only_actions"]:
            lines.append(f"    Proposed actions: {', '.join(p['proposal_only_actions'])}")
        if p["frozen"]:
            lines.append(f"    Blocked: {p.get('freeze_reason', 'frozen')}")
        if p["requires_approval"]:
            lines.append(f"    Requires approval: {p.get('approval_reason', 'yes')}")

    return "\n".join(lines)


# ── Wake cycle JSONL logging ──────────────────────────────────────────────

def log_wake_cycle(brain, cycle_num: int, session_id: str,
                   wake_reason: str = "timer",
                   signals_reported: list[str] | None = None,
                   action_taken: str | None = None,
                   action_result: str | None = None):
    """Log a complete wake cycle entry per the V16 validation plan requirements.

    Each entry captures: wake timestamp, wake reason, runtime signals,
    active tensions, drives, proposals, safety gates, freeze state.
    """
    try:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "wake_cycle",
            "cycle_num": cycle_num,
            "session_id": session_id,
            "wake_reason": wake_reason,
            "signals_reported": signals_reported or [],
        }

        # Collect cognitive state
        diag = collect_v16_diagnostics(brain)
        entry["active_drives"] = diag.get("active_drives", 0)
        entry["top_drives"] = diag.get("top_drives", [])
        entry["tension_diagnostics"] = diag.get("tension_diagnostics", {})
        entry["safety_gates"] = diag.get("safety_gates", [])
        entry["kill_switch"] = diag.get("kill_switch", {})

        # Action info
        entry["action_taken"] = action_taken
        entry["action_result"] = action_result

        _log_jsonl(entry)

    except Exception as e:
        logger.debug("V16 wake cycle log failed: %s", e)
