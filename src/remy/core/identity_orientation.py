"""Identity Orientation Layer — gives the agent self-awareness and metacognition.

Injected into system_instruction alongside temporal_orientation so the agent
understands WHO it is, how confident its knowledge is, and whether its
identity is drifting or under pressure.

Sections:
  NARRATIVE:   compact self-model from AuraSDK narrative engine
  METACOG:     epistemic confidence, conflicts, action guidance
  ANCHORS:     top identity anchors (most stable beliefs)
  DRIFT:       memory drift alert if detected
"""

import logging
import time

logger = logging.getLogger(__name__)

_cache: dict = {}
_CACHE_TTL_SEC = 120  # refresh every 2 minutes


def get_identity_orientation() -> str:
    """Return a compact identity block for injection into system prompt."""
    now_ts = time.time()
    if _cache.get("ts") and (now_ts - _cache["ts"]) < _CACHE_TTL_SEC:
        return _cache["text"]

    try:
        from remy.core.agent_tools import brain, brain_lock
        with brain_lock:
            text = _build_identity_block(brain)
        _cache["ts"] = now_ts
        _cache["text"] = text
        return text
    except Exception as e:
        logger.debug("identity_orientation failed: %s", e)
        return ""


def invalidate_cache():
    """Clear cached identity block (call after persona/identity changes)."""
    _cache.clear()


def _build_identity_block(brain) -> str:
    parts = []

    # --- 1. NARRATIVE SELF (pre-formatted by AuraSDK) ---
    try:
        narrative = brain.get_narrative_self_formatted()
        if narrative and isinstance(narrative, str) and len(narrative) > 10:
            # Truncate to keep token budget reasonable
            lines = narrative.strip().splitlines()
            compact = "\n".join(lines[:8])  # max 8 lines
            if len(lines) > 8:
                compact += f"\n  ... ({len(lines) - 8} more lines)"
            parts.append(compact)
    except Exception as e:
        logger.debug("narrative self failed: %s", e)

    # --- 2. METACOGNITIVE STATE ---
    try:
        mc = brain.get_metacognitive_context()
        if mc:
            conf = getattr(mc, "confidence_score", None)
            conflicts = getattr(mc, "conflict_count", 0)
            unstable = getattr(mc, "has_unstable_beliefs", False)
            guidance = getattr(mc, "action_guidance", "")
            dominant = getattr(mc, "dominant_finding_kind", "")
            repetition = getattr(mc, "repetition_detected", False)

            mc_line = f"METACOG: confidence={conf:.2f}" if conf is not None else "METACOG:"
            if conflicts:
                mc_line += f" | {conflicts} conflict(s)"
            if unstable:
                mc_line += " | unstable beliefs present"
            if repetition:
                mc_line += " | repetition detected"
            if dominant:
                mc_line += f" | dominant signal: {dominant}"
            if guidance:
                mc_line += f" | recommendation: {guidance}"
            parts.append(mc_line)
    except Exception as e:
        logger.debug("metacognitive context failed: %s", e)

    # --- 3. IDENTITY ANCHORS (top 5 most stable beliefs) ---
    try:
        anchors = brain.get_identity_anchors()
        if anchors:
            anchor_strs = []
            for a in anchors[:5]:
                key = getattr(a, "key", "")
                stability = getattr(a, "stability", 0)
                shielded = getattr(a, "is_shielded", False)
                # Extract readable label from key (format: namespace:tags:kind)
                label = key.split(":")[-2] if ":" in key else key
                label = label.replace(",", ", ")[:40]
                s = f"{label} (stab={stability:.0f}"
                if shielded:
                    s += ", shielded"
                s += ")"
                anchor_strs.append(s)
            parts.append("ANCHORS: " + " | ".join(anchor_strs))
    except Exception as e:
        logger.debug("identity anchors failed: %s", e)

    # --- 4. DRIFT ALERT (critical — always show if detected) ---
    try:
        drift_alert = brain.get_drift_alert()
        if drift_alert:
            parts.append(f"⚠ DRIFT ALERT: {drift_alert}")
    except Exception as e:
        logger.debug("drift alert failed: %s", e)

    # --- 5. DRIFT REPORT (compact summary even when no alert) ---
    try:
        dr = brain.get_drift_report()
        if dr:
            score = getattr(dr, "drift_score", None)
            assessment = getattr(dr, "assessment", "")
            cycles = getattr(dr, "cycles_measured", 0)
            if score is not None and cycles > 0:
                parts.append(f"DRIFT: score={score:.3f} ({assessment}) over {cycles} cycles")
    except Exception as e:
        logger.debug("drift report failed: %s", e)

    if not parts:
        return ""

    lines = ["=== IDENTITY CONTEXT ==="] + parts + ["=== END IDENTITY ==="]
    return "\n".join(lines)
