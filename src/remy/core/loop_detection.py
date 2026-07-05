"""Loop Detection — detects repetitive tool patterns across autonomy cycles.

Two-layer detection:

Layer 1 (tool-name fingerprint): sorted set of tool names per cycle.
  - warning (3+ repeats): hint injected into decision prompt
  - force_change (5+ repeats): strong warning + must use different tools
  - skip (7+ repeats): triggers existing backoff mechanism

Layer 2 (SHA-256 call fingerprint): hashes actual tool name+args per call.
  Detects exact repeated calls (same args) even within a single cycle.
  If the same SHA-256 appears 3+ times in the rolling window → circuit breaker.

Cost: zero LLM calls — pure deque + set + hashlib comparison.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass, field

# ── Configuration ─────────────────────────────────────────────────────────
CYCLE_HISTORY_SIZE = 10  # Track last N fingerprints
WARN_THRESHOLD = 3  # Consecutive identical cycles → warning
FORCE_CHANGE_THRESHOLD = 5  # → strong warning
SKIP_THRESHOLD = 7  # → skip cycle + trigger backoff

# Layer 2: SHA-256 exact-call detection
CALL_HASH_WINDOW = 50  # Rolling window of individual call hashes
CALL_REPEAT_THRESHOLD = 3  # Same exact call hash N times → circuit breaker
CALL_HASH_COOLDOWN_SEC = 300  # 5 min cooldown after circuit breaker fires


# ── Data Structures ───────────────────────────────────────────────────────


@dataclass
class CycleFingerprint:
    """Fingerprint of tools called in a single autonomy cycle."""

    tool_names: tuple[str, ...]  # sorted, hashable
    sha256: str  # SHA-256 of all (tool_name, sorted_args) tuples
    cycle_num: int
    timestamp: float


@dataclass
class LoopDetectionState:
    """Tracks tool patterns across autonomy cycles."""

    history: deque = field(default_factory=lambda: deque(maxlen=CYCLE_HISTORY_SIZE))
    current_repetition_count: int = 0
    # Layer 2: individual call hashes
    call_hashes: deque = field(default_factory=lambda: deque(maxlen=CALL_HASH_WINDOW))
    circuit_breaker_until: float = 0.0  # timestamp when CB expires
    circuit_breaker_count: int = 0  # lifetime CB triggers
    blocked_hashes: set = field(default_factory=set)  # currently blocked SHA-256s


# ── SHA-256 Helpers ──────────────────────────────────────────────────────


def _hash_tool_call(tool_name: str, args: dict | None) -> str:
    """SHA-256 hash of a single tool call (name + sorted args). Truncated to 16 hex."""
    payload = json.dumps({"t": tool_name, "a": args or {}}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _hash_cycle(session_log: list[dict]) -> str:
    """SHA-256 of all tool calls in order (name + args). Full 64 hex."""
    parts = []
    for e in session_log:
        if e.get("type") == "tool_call":
            parts.append(json.dumps(
                {"t": e.get("tool", ""), "a": e.get("args", {})},
                sort_keys=True, ensure_ascii=False,
            ))
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()


# ── Core Functions ────────────────────────────────────────────────────────


def extract_fingerprint(session_log: list[dict], cycle_num: int) -> CycleFingerprint:
    """Extract a CycleFingerprint from this cycle's session_log.

    Layer 1: sorted deduplicated tool names.
    Layer 2: SHA-256 of all (tool_name, args) tuples in call order.
    """
    tool_names = sorted(set(e["tool"] for e in session_log if e.get("type") == "tool_call"))
    sha = _hash_cycle(session_log)
    return CycleFingerprint(
        tool_names=tuple(tool_names),
        sha256=sha,
        cycle_num=cycle_num,
        timestamp=time.time(),
    )


def detect_loop(state: LoopDetectionState, new_fp: CycleFingerprint) -> dict:
    """Check if the new fingerprint matches recent consecutive cycles.

    Layer 1: tool-name set comparison (original behavior).
    Layer 2: SHA-256 exact match — detects identical call sequences even faster.

    Updates state in-place and returns detection result:
        {
            "level": "none" | "warning" | "force_change" | "skip",
            "repetition_count": int,
            "repeated_tools": tuple[str, ...],
            "sha256_match": bool,
            "message": str,
        }
    """
    # Layer 2 (exact SHA-256): count consecutive SHA matches
    sha_consecutive = 0
    for old_fp in reversed(state.history):
        if old_fp.sha256 == new_fp.sha256:
            sha_consecutive += 1
        else:
            break

    # Layer 1 (tool-name set): count consecutive name matches
    name_consecutive = 0
    for old_fp in reversed(state.history):
        if old_fp.tool_names == new_fp.tool_names:
            name_consecutive += 1
        else:
            break

    state.history.append(new_fp)

    # Use whichever layer detected more repeats
    # SHA match is stricter (exact args) so it gets a -1 bonus on thresholds
    if sha_consecutive >= name_consecutive:
        consecutive = sha_consecutive
        sha_match = True
        # Exact repeats are worse — lower thresholds by 1
        warn_at = max(WARN_THRESHOLD - 1, 2)
        force_at = max(FORCE_CHANGE_THRESHOLD - 1, 3)
        skip_at = max(SKIP_THRESHOLD - 1, 5)
    else:
        consecutive = name_consecutive
        sha_match = False
        warn_at = WARN_THRESHOLD
        force_at = FORCE_CHANGE_THRESHOLD
        skip_at = SKIP_THRESHOLD

    state.current_repetition_count = consecutive + 1  # include this cycle
    count = state.current_repetition_count

    exact_label = " (exact same calls)" if sha_match else ""

    if count >= skip_at:
        return {
            "level": "skip",
            "repetition_count": count,
            "repeated_tools": new_fp.tool_names,
            "sha256_match": sha_match,
            "message": (
                f"LOOP DETECTED: Same tool pattern {list(new_fp.tool_names)} "
                f"repeated {count} times{exact_label}. Skipping cycle and backing off."
            ),
        }
    elif count >= force_at:
        return {
            "level": "force_change",
            "repetition_count": count,
            "repeated_tools": new_fp.tool_names,
            "sha256_match": sha_match,
            "message": (
                f"LOOP WARNING: Pattern {list(new_fp.tool_names)} repeated {count} times{exact_label}. "
                f"You MUST use different tools or a different approach."
            ),
        }
    elif count >= warn_at:
        return {
            "level": "warning",
            "repetition_count": count,
            "repeated_tools": new_fp.tool_names,
            "sha256_match": sha_match,
            "message": (
                f"Repetition notice: Same tools {list(new_fp.tool_names)} "
                f"for {count} consecutive cycles{exact_label}. Consider varying your approach."
            ),
        }

    return {
        "level": "none",
        "repetition_count": count,
        "repeated_tools": (),
        "sha256_match": False,
        "message": "",
    }


# ── Layer 2: Per-call circuit breaker ─────────────────────────────────────


def check_call_loop(
    state: LoopDetectionState,
    tool_name: str,
    tool_args: dict | None = None,
) -> dict:
    """Check if a single tool call is being repeated excessively.

    Called BEFORE executing each tool call. If the same (tool + args) SHA-256
    appears CALL_REPEAT_THRESHOLD times in the rolling window, returns a
    circuit-breaker result.

    Returns:
        {"blocked": bool, "hash": str, "count": int, "message": str}
    """
    now = time.time()

    # Check if circuit breaker cooldown is active
    if state.circuit_breaker_until > now:
        return {
            "blocked": True,
            "hash": "",
            "count": 0,
            "message": f"Circuit breaker active (cooldown {int(state.circuit_breaker_until - now)}s remaining). "
                       "All tool calls paused.",
        }

    call_hash = _hash_tool_call(tool_name, tool_args)
    state.call_hashes.append((call_hash, now))

    # Count occurrences of this hash in the window
    count = sum(1 for h, _ in state.call_hashes if h == call_hash)

    if count >= CALL_REPEAT_THRESHOLD:
        state.circuit_breaker_until = now + CALL_HASH_COOLDOWN_SEC
        state.circuit_breaker_count += 1
        state.blocked_hashes.add(call_hash)
        return {
            "blocked": True,
            "hash": call_hash,
            "count": count,
            "message": (
                f"CIRCUIT BREAKER: {tool_name}() called {count} times with identical args. "
                f"Blocking for {CALL_HASH_COOLDOWN_SEC}s to prevent token waste."
            ),
        }

    return {"blocked": False, "hash": call_hash, "count": count, "message": ""}


def format_loop_warning_for_prompt(detection_result: dict) -> str:
    """Format a loop detection result as text to inject into the decision prompt.

    Returns empty string if level is 'none'.
    """
    if detection_result["level"] == "none":
        return ""

    level = detection_result["level"].upper()
    tools = ", ".join(detection_result["repeated_tools"])
    count = detection_result["repetition_count"]

    lines = [
        f"\n=== LOOP DETECTION: {level} ===",
        f"You have called [{tools}] for {count} consecutive cycles.",
    ]

    if detection_result["level"] == "force_change":
        lines.append(
            "You MUST call DIFFERENT tools this cycle. Repeating the same pattern = failure."
        )
        lines.append(
            "If stuck, use 'scratchpad' to note what's blocking you, "
            "then try a completely different approach."
        )
    elif detection_result["level"] == "skip":
        lines.append("Cycle will be skipped. Entering backoff sleep.")
    elif detection_result["level"] == "warning":
        lines.append("Consider trying a different approach or different tools.")

    lines.append("===\n")
    return "\n".join(lines)
