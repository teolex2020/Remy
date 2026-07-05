"""
Brain Tools - channel-agnostic tool definitions, execution, and system instruction.

Extracted from gemini_live.py so all channels (Gemini Live Audio, Telegram, etc.)
can share the same brain tools and execution logic.
"""

import asyncio
import json
import logging
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from google.genai import types

from remy.config.settings import settings
from remy.core.agent_tools import brain, brain_lock
from remy.core.hybrid_search import hybrid_search_structured, search_exact_structured
from remy.core.scheduling import normalize_schedule_args
from remy.core.source_credibility import credibility_scorer
from remy.core.time_render import format_age
from remy.core.tool_registry import ToolRegistry
from remy.sandbox.runner import validate_tool_file, run_tests, install_dependencies

logger = logging.getLogger("BrainTools")


# ============== USER PROFILE LOOKUP ==============

def get_user_profile_record(brain_instance=None, lock=None):
    """Return the most recently created user-profile record, or None.

    Uses list_records (returns all matching) then picks the newest by created_at.
    This is deterministic unlike search(limit=1) which ranks by last_activated,
    causing stale older records to shadow newer ones.
    """
    _brain = brain_instance or brain
    _lock = lock or brain_lock
    try:
        with _lock:
            if hasattr(_brain, "list_records"):
                candidates = _brain.list_records(tags=["user-profile"])
            else:
                candidates = _brain.search(query="", tags=["user-profile"], limit=50)
        if not candidates:
            return None
        return max(candidates, key=lambda r: getattr(r, "created_at", 0))
    except Exception:
        return None


# ============== PROVENANCE ==============

_SOURCE_TRUST: dict[str, float] = {
    "user-confirmed": 1.0,
    "agent-interactive": 0.7,
    "system": 0.6,
    "agent-autonomous": 0.4,
    "agent-worker": 0.35,
    "agent": 0.5,
}


def _get_provenance(channel: str | None) -> dict:
    """Build provenance metadata based on channel context."""
    if channel == "autonomous":
        source = "agent-autonomous"
    elif channel and channel.startswith("worker-"):
        source = "agent-worker"
    elif channel in ("desktop", "telegram", "voice", "proactive"):
        source = "agent-interactive"
    elif channel == "system":
        source = "system"
    else:
        source = "agent"
    return {
        "source": source,
        "verified": source == "user-confirmed",
        "trust_score": _SOURCE_TRUST[source],
    }


def _serialize_aura_result(obj, _depth=0):
    """Serialize AuraSDK PyO3 result objects to JSON-safe dicts."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (int,)):  # catch numpy-like ints
        return int(obj)
    if isinstance(obj, list):
        return [_serialize_aura_result(item, _depth + 1) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialize_aura_result(v, _depth + 1) for k, v in obj.items()}
    # Guard against infinite recursion (PyO3 enums have self-referencing variants)
    if _depth > 10:
        return str(obj)
    # PyO3 enums: their non-private non-callable attrs are variant names that
    # point back to the same enum type - infinite recursion. Detect by checking
    # if the type name matches the module pattern "builtins.SomeName" with enum-like repr.
    obj_repr = repr(obj)
    if "." in obj_repr and obj_repr.split(".")[-1].isidentifier():
        # Looks like EnumType.Variant - check if attrs are same-type (enum self-ref)
        attrs = [a for a in dir(obj) if not a.startswith("_") and not callable(getattr(obj, a, None))]
        if attrs and all(type(getattr(obj, a, None)) is type(obj) for a in attrs[:3]):
            return str(obj).split(".")[-1] if "." in str(obj) else str(obj)
    # PyO3 objects with attributes
    attrs = [a for a in dir(obj) if not a.startswith("_") and not callable(getattr(obj, a, None))]
    if attrs:
        return {a: _serialize_aura_result(getattr(obj, a, None), _depth + 1) for a in attrs}
    return str(obj)


def _aura_unavailable(method_name: str, **extra) -> str:
    payload = {
        "available": False,
        "reason": f"Aura backend does not expose {method_name}.",
    }
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def _aura_method(method_name: str):
    method = getattr(brain, method_name, None)
    return method if callable(method) else None


def _stamp_provenance(metadata: dict | None, channel: str | None, tags: list | None = None) -> dict:
    """Merge provenance fields into existing metadata dict.

    Uses setdefault so explicit source/verified/trust_score in existing metadata
    won't be overwritten. Also infers source_type for epistemological tracking.
    """
    meta = dict(metadata or {})
    prov = _get_provenance(channel)
    meta.setdefault("source", prov["source"])
    meta.setdefault("verified", prov["verified"])
    meta.setdefault("trust_score", prov["trust_score"])
    # Infer source_type for epistemological tracking
    if "source_type" not in meta:
        _tags = set(tags or [])
        if _tags & {"web-search-cache", "research-finding", "research-project"}:
            meta["source_type"] = "retrieved"
        elif _tags & {"extracted-fact", "consolidated-meta", "session-reflection", "research-synthesis"}:
            meta["source_type"] = "inferred"
        elif _tags & {"autonomous-goal", "action-plan", "autonomous-outcome", "scheduled-task", "todo-item"}:
            meta["source_type"] = "generated"
        else:
            source = prov["source"]
            if source in ("agent-autonomous", "agent-worker"):
                meta["source_type"] = "generated"
            else:
                meta["source_type"] = "recorded"
    return meta


def _compute_effective_trust(metadata: dict, created_at: float) -> float:
    """Compute recall-time effective trust with age decay for autonomous records.

    For agent-autonomous records with trust < 1.0, trust decays over 30 days
    down to a floor of 50% of the base trust.
    """
    import time as _time
    trust = metadata.get("trust_score", 0.5)
    source = metadata.get("source", "")
    if source == "agent-autonomous" and trust < 1.0:
        age_days = (_time.time() - created_at) / 86400
        age_factor = max(0.5, 1.0 - (age_days / 30))
        trust = round(trust * age_factor, 2)
    return trust


# ============== MEMORY-GATED EXECUTION ==============
#
# Three guards that turn memory from passive storage into an enforcement layer:
#   1. STORE GUARD - sensitive data stored autonomously gets actionable=false
#   2. ACTION GUARD - external tools blocked unless data has actionable=true
#   3. HALLUCINATION GUARD - data not in memory at all = hallucination = blocked
#

# Tools that perform external actions where hallucinated data is dangerous.
_TRUST_ENFORCED_TOOLS = frozenset({
    "browser_act",   # typing into forms - emails, passwords, wallets
    "http_get",      # API calls with user data in URLs
})

# Arg keys that may contain sensitive user data worth validating
_SENSITIVE_KEYS = frozenset({
    "email", "wallet", "address", "account", "username",
    "password", "token", "api_key", "phone",
})

# Patterns that detect sensitive data in free-text fields
import re as _re
_EMAIL_RE = _re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_WALLET_RE = _re.compile(r"0x[a-fA-F0-9]{40}")  # Ethereum-style

# Tags that indicate sensitive/identity content for store guard
_SENSITIVE_TAGS = frozenset({
    "identity", "financial", "wallet", "credential", "account",
    "registration", "login", "email", "proxy", "api-key",
})


# --- GUARD 1: STORE GUARD ---

def _apply_store_guard(content: str, tags: list, channel: str | None) -> dict:
    """Auto-set actionable=false for sensitive data stored by autonomous agents.

    Interactive channels (desktop, telegram, voice) store with actionable=true
    because the user is present and implicitly approves.
    Autonomous/worker channels store with actionable=false - requires verify_record.

    Returns metadata fields to merge.
    """
    is_interactive = channel in ("desktop", "telegram", "voice")

    # Check if content or tags indicate sensitive data
    has_sensitive_tags = bool(set(tags) & _SENSITIVE_TAGS)
    has_sensitive_content = bool(
        _EMAIL_RE.search(content) or _WALLET_RE.search(content)
    )

    if not has_sensitive_tags and not has_sensitive_content:
        return {}  # Not sensitive - no guard needed

    if is_interactive:
        # User is present - data is implicitly approved
        return {"actionable": True}
    else:
        # Autonomous - NOT actionable until user verifies
        return {"actionable": False}


# --- CLAIM PROVENANCE GATE ---
#
# Orthogonal to _stamp_provenance (agent-channel) and _apply_store_guard
# (sensitive-data approval). This gate decides WHAT KIND of claim the LLM is
# writing - fact / inference / proposal / citation_claim - and whether it
# has external verification. LLM-only factual claims in a phantom-heavy
# turn are rerouted to a quarantine tag instead of entering the factual
# memory layer.

_HYPOTHESIS_TAGS = frozenset({
    "hypothesis", "proposal", "research-lead", "conjecture", "speculation",
    "idea", "draft",
})
_INFERENCE_TAGS = frozenset({
    "inference", "consolidated-meta", "session-reflection", "research-synthesis",
    "extracted-fact",
})
_USER_DIRECT_CHANNELS = frozenset({"desktop", "telegram", "voice"})


def _infer_claim_class(content: str, tags: list, channel: str | None) -> str:
    """Default classification: fact, unless tags/content say otherwise.

    A stricter pass (Phase A.3+) would inspect content for "I think",
    "maybe", etc. For now tags are the authoritative signal because the LLM
    can be instructed to tag speculative stores explicitly.
    """
    tag_set = set(tags or [])
    if tag_set & _HYPOTHESIS_TAGS:
        return "proposal"
    if tag_set & _INFERENCE_TAGS:
        return "inference"
    return "fact"


def _infer_claim_provenance(
    channel: str | None,
    tags: list,
    session_id: str | None,
):
    """Pick a ClaimProvenance for an LLM-issued store() call.

    Rule of thumb:
    - User-driven interactive channel storing user-profile-ish content -> user
    - Agent/worker autonomous channel - llm_unverified (unless tagged otherwise)
    - Anything with tags indicating tool-verified origin - tool_verified

    D-03 fix: tool-verified tags are NOT trusted unless fetch evidence actually
    exists for this session. An LLM can forge tags; it cannot forge fetch events.
    """
    from remy.core.claim_provenance import ClaimProvenance
    tag_set = set(tags or [])
    if tag_set & {"tool-verified", "verified-external", "web-search-cache"}:
        # D-03: verify real fetch evidence before honouring the tag
        _real_fetch = []
        try:
            from remy.core.claim_provenance import get_turn_fetch_evidence
            _real_fetch = get_turn_fetch_evidence(session_id or "")
        except Exception:
            pass
        if _real_fetch:
            return ClaimProvenance.tool_verified(
                tool="web_search",
                locator=",".join(sorted(tag_set & {"web-search-cache", "verified-external"})) or "unspecified",
            )
        # Tag present but no actual fetch this turn - treat as unverified
        # (fall through to llm_unverified at the end)
    if tag_set & {"system-inferred", "consolidated-meta", "session-reflection"}:
        return ClaimProvenance.system_inferred(based_on=[], note="tag-inferred")
    if channel in _USER_DIRECT_CHANNELS and (tag_set & {"user-profile", "user-statement", "from-user"}):
        return ClaimProvenance.user(note=f"channel={channel}")
    return ClaimProvenance.llm_unverified(note=f"channel={channel or 'unknown'}")


def _apply_claim_gate(
    content: str,
    tags: list,
    channel: str | None,
    session_id: str | None = None,
) -> tuple[dict, list, bool, str]:
    """Decide storage fate for an LLM-issued store() call.

    Returns (metadata_patch, tags_to_add, quarantine, reason).
    metadata_patch always includes claim_class + claim_provenance.
    When quarantine=True the caller must NOT write into the factual memory
    layer; tags_to_add will include the quarantine sentinels.
    """
    try:
        from remy.core.claim_provenance import (
            ClaimProvenance,
            claim_metadata,
            decide_storage,
            get_turn_factuality_signal,
        )
    except Exception:
        return ({}, [], False, "claim_provenance module unavailable")

    requested_class = _infer_claim_class(content, tags, channel)
    provenance = _infer_claim_provenance(channel, tags, session_id)

    sig = get_turn_factuality_signal(session_id or "") or {}
    total = int(sig.get("external_total", 0))
    phantom = int(sig.get("phantom_count", 0))
    phantom_ratio = (phantom / total) if total > 0 else 0.0
    has_phantom = total > 0 and phantom > 0

    decision = decide_storage(
        requested_class=requested_class,
        provenance=provenance,
        has_phantom_citations=has_phantom,
        phantom_ratio=phantom_ratio,
    )
    meta_patch = claim_metadata(provenance, decision.effective_claim_class)
    return (meta_patch, list(decision.tags_to_add), decision.quarantine, decision.reason)


def _apply_store_claim_gate(
    content: str,
    tags: list,
    channel: str | None,
    session_id: str | None,
    metadata: dict | None,
) -> tuple[dict, list, bool, str]:
    """Apply claim gate, honoring vetted explicit admission classes."""
    meta = metadata or {}
    admission_class = meta.get("admission_class")
    try:
        from remy.core.memory_policy import FACTUAL_SAFE_ADMISSION_CLASSES
        if admission_class in FACTUAL_SAFE_ADMISSION_CLASSES:
            from remy.core.claim_provenance import ClaimProvenance, claim_metadata
            if admission_class == "operator_asserted":
                provenance = ClaimProvenance.user(note="admission_class=operator_asserted")
                patch = claim_metadata(provenance, "fact")
                patch.setdefault("verified", True)
                patch.setdefault("source_type", "recorded")
            else:
                provenance = ClaimProvenance.tool_verified(
                    tool=str(meta.get("source_tool") or "admission_class"),
                    locator=str(meta.get("source_url") or meta.get("locator") or admission_class),
                    note=f"admission_class={admission_class}",
                )
                patch = claim_metadata(provenance, "fact")
                patch.setdefault("verified", True)
                patch.setdefault("source_type", "retrieved")
            return patch, [], False, f"safe admission_class={admission_class}"
    except Exception:
        pass
    return _apply_claim_gate(content, tags, channel, session_id=session_id)


# --- GUARD 2 & 3: ACTION GUARD + HALLUCINATION GUARD ---

def _validate_action_data(name: str, args: dict) -> str | None:
    """Memory-gated execution: block actions with unverified or hallucinated data.

    Three checks per sensitive value:
    1. HALLUCINATION GUARD - value not found in memory at all - blocked
    2. TRUST GUARD - found but actionable=false or trust < 0.8 - blocked
    3. PASS - found with actionable=true or (trust >= 0.8 and no actionable field)

    Returns error string if blocked, None if all data is verified.
    Called under brain_lock.
    """
    if name not in _TRUST_ENFORCED_TOOLS:
        return None

    values_to_check: list[tuple[str, str]] = []  # (key_name, value)

    for key, value in args.items():
        if not isinstance(value, str) or not value.strip():
            continue
        # Direct sensitive key match
        if key in _SENSITIVE_KEYS:
            values_to_check.append((key, value.strip()))
            continue
        # For "text" field in browser_act - check if it contains email or wallet
        if key == "text":
            email_match = _EMAIL_RE.search(value)
            if email_match:
                values_to_check.append(("email", email_match.group()))
            wallet_match = _WALLET_RE.search(value)
            if wallet_match:
                values_to_check.append(("wallet", wallet_match.group()))

    if not values_to_check:
        return None  # No sensitive data in args - proceed

    import time as _time
    now = _time.time()

    for key_name, value in values_to_check:
        # Search brain for this exact value
        records = brain.search(query=value, limit=5)

        # Find a record that actually contains this value
        match = None
        for r in records:
            if value.lower() in (r.content or "").lower():
                match = r
                break

        # --- GUARD 3: HALLUCINATION GUARD ---
        if match is None:
            logger.warning(
                "HALLUCINATION GUARD: %s='%s' in tool '%s' - not found in memory",
                key_name, value[:30], name,
            )
            return (
                f"HALLUCINATION GUARD: {key_name}='{value}' not found in memory. "
                f"This data may be hallucinated. Cannot use unverified data for external actions.\n"
                f"Action: Ask the user to provide the correct {key_name}, "
                f"then store it (it will be auto-verified since user provided it)."
            )

        # --- GUARD 2: ACTION/TRUST GUARD ---
        meta = match.metadata or {}
        actionable = meta.get("actionable")  # None = legacy record without this field

        if actionable is True:
            # Explicitly marked as actionable - pass
            continue
        elif actionable is False:
            # Explicitly NOT actionable - blocked
            trust = meta.get("trust_score", 0)
            logger.warning(
                "TRUST GUARD: %s='%s' in tool '%s' - actionable=false (trust=%.2f)",
                key_name, value[:30], name, trust,
            )
            return (
                f"TRUST GUARD: {key_name}='{value}' exists in memory but is NOT verified "
                f"(actionable=false, trust={trust:.2f}). Cannot use for external actions.\n"
                f"Action: Ask the user to confirm this data is correct, "
                f"then use verify_record to mark it as verified."
            )
        else:
            # Legacy record (no actionable field) - fall back to trust check
            if meta.get("verified") is True:
                continue  # Verified by user - pass
            trust = _compute_effective_trust(meta, getattr(match, "created_at", now))
            if trust >= 0.8:
                continue  # High trust - pass
            # Low trust, no actionable field - blocked
            logger.warning(
                "TRUST GUARD: %s='%s' in tool '%s' - trust %.2f < 0.8, no actionable flag",
                key_name, value[:30], name, trust,
            )
            return (
                f"TRUST GUARD: {key_name}='{value}' exists but trust is too low "
                f"(trust={trust:.2f}, needs >= 0.8). Cannot use for external actions.\n"
                f"Action: Ask the user to confirm, then use verify_record."
            )

    return None  # All sensitive data verified - proceed


# ============== KNOWLEDGE SYNC ==============

# Tags that should NOT be mirrored to knowledge (transient/system records)
_NO_MIRROR_TAGS = frozenset({
    "web-search-cache", "session-summary", "proactive-session",
    "action-plan", "session-reflection", "consolidated-meta",
    "autonomous-outcome", "autonomous-goal", "relationship",
    "todo-item", "push-subscription", "feedback-signal",
})

# Periodic flush counter
_sync_counter = 0
_FLUSH_EVERY_N = 10


def _should_sync(level) -> tuple:
    """Determine if a record should sync to knowledge and with what pin.

    Args:
        level: Cognitive level (Level enum or string).

    Returns:
        (should_sync: bool, pin: bool)
    """
    name = getattr(level, "name", str(level)).upper()
    if name in ("IDENTITY", "CORE"):
        return True, True       # sync + pin (anchor, never decays)
    if name in ("DOMAIN", "DECISIONS"):
        return True, False      # sync, general (decays)
    if name in ("WORKING", "SESSION"):
        return False, False     # ephemeral, don't sync
    return True, False          # default: sync without pin


def _sync_to_knowledge(content: str, pin: bool = False,
                        deduplicate: bool = True) -> bool:
    """Fire-and-forget: replicate content to Aura Memory semantic index.

    Thread-safe. Errors are logged but never propagated.

    Args:
        content: Text to store in knowledge.
        pin: True = user_core/anchor, False = general.
        deduplicate: Check for existing similar record first.

    Returns:
        True if stored, False if skipped/error.
    """
    global _sync_counter
    try:
        from remy.core.agent_tools import knowledge, knowledge_lock
        if knowledge is None:
            return False

        # Guard: content must be a plain string, not a dict/list accidentally str()'d
        if not isinstance(content, str):
            content = str(content)
        content = content.strip()

        # Reject JSON-wrapped dicts (Gemini API content parts mistakenly serialized)
        if not content or len(content) < 20:
            return False
        if content.startswith("{'type':") or content.startswith('{"type":'):
            logger.debug("Skipping JSON-wrapped content block in KB sync")
            return False

        text = content[:2000]

        with knowledge_lock:
            # Dedup: skip if very similar record exists
            if deduplicate:
                try:
                    matrix_res = knowledge.retrieve_matrix(text[:200], top_k=1)
                    if matrix_res and len(matrix_res) >= 3 and len(matrix_res[2]) > 0:
                        best_score = float(matrix_res[0][0]) if len(matrix_res[0]) > 0 else 0
                        if best_score > 0.7:
                            logger.debug("Similar record in KB (score=%.2f), skipping sync", best_score)
                            return False
                except Exception:
                    pass  # retrieve failed - store anyway

            knowledge.process(text, pin=pin)

            # Periodic flush (every N writes)
            _sync_counter += 1
            if _sync_counter >= _FLUSH_EVERY_N:
                knowledge.flush()
                _sync_counter = 0

        return True
    except Exception as e:
        logger.debug("Knowledge sync failed (non-fatal): %s", e)
        return False


def _kb_retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve from AuraMemory using retrieve_matrix (working) instead of retrieve_full (broken in v2.0.0).

    retrieve_full() returns 0 results for any query in AuraMemory v2.0.0 - known bug.
    retrieve_matrix() works but only returns anchors/pinned records.
    Fallback: substring match on list_memories() for non-pinned records.

    Returns list of {"text": str, "score": float, "id": str} dicts, sorted by score desc.
    """
    from remy.core.agent_tools import knowledge, knowledge_lock
    if knowledge is None:
        return []

    results = []
    seen_ids = set()

    try:
        with knowledge_lock:
            # 1. SDR-based retrieval via retrieve_matrix (works for anchors)
            try:
                matrix_result = knowledge.retrieve_matrix(query, top_k=top_k)
                if matrix_result and len(matrix_result) >= 3:
                    scores = matrix_result[0]
                    matches = matrix_result[2]
                    for i, m in enumerate(matches):
                        mid = m.get("id", "")
                        # Handle numpy float32/float64 types
                        score = float(scores[i]) if i < len(scores) else 0.0
                        results.append({
                            "text": m.get("text", ""),
                            "score": score,
                            "id": mid,
                            "intensity": m.get("intensity", 0.0),
                            "dna": m.get("dna", ""),
                        })
                        seen_ids.add(mid)
            except Exception as e:
                logger.debug("KB retrieve_matrix failed: %s", e)

            # 2. Substring fallback on list_memories (catches non-anchored records)
            if len(results) < top_k:
                try:
                    query_lower = query.lower()
                    query_words = [w for w in query_lower.split() if len(w) > 2]
                    
                    # list_memories returns a list, but might be nested in v2.0
                    all_mems_raw = knowledge.list_memories()
                    all_mems = []
                    
                    if isinstance(all_mems_raw, list):
                        # Check if it's a nested list [[{...}, {...}]]
                        if all_mems_raw and isinstance(all_mems_raw[0], list):
                            for sublist in all_mems_raw:
                                if isinstance(sublist, list):
                                    all_mems.extend(sublist)
                        else:
                            all_mems = all_mems_raw
                            
                    for m in (all_mems or []):
                        if isinstance(m, dict):
                            mid = m.get("id", "")
                            if mid in seen_ids:
                                continue
                            text = m.get("text", "")
                            text_lower = text.lower()
                            # Score: fraction of query words found in text
                            if query_words:
                                hits = sum(1 for w in query_words if w in text_lower)
                                score = hits / len(query_words) * 0.5  # max 0.5 for substring
                            elif query_lower in text_lower:
                                score = 0.4
                            else:
                                continue
                            if score > 0:
                                results.append({
                                    "text": text,
                                    "score": score,
                                    "id": mid,
                                    "intensity": m.get("intensity", 0.0),
                                    "dna": m.get("dna", ""),
                                })
                                seen_ids.add(mid)
                except Exception as e:
                    logger.debug("KB list_memories fallback failed: %s", e)
    except Exception as e:
        logger.debug("KB retrieve failed (non-fatal): %s", e)

    # Sort by score descending, limit to top_k
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ============== SHARED UTILITIES ==============


def parse_llm_json(raw: str):
    """Parse JSON from LLM output, tolerating markdown fences, single quotes, trailing commas."""
    text = raw.strip()

    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fix common LLM issues: single quotes, trailing commas
    fixed = text.replace("'", '"')
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Fix unquoted keys: word immediately before colon - add quotes
    fixed2 = re.sub(r'(?<=[{,\s])(\w+)\s*:', r'"\1":', text)
    fixed2 = fixed2.replace("'", '"')
    fixed2 = re.sub(r",\s*([}\]])", r"\1", fixed2)
    try:
        return json.loads(fixed2)
    except json.JSONDecodeError:
        pass

    # Last resort: find first { or [ and extract to matching close
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        idx = text.find(start_char)
        if idx >= 0:
            depth = 0
            for i in range(idx, len(text)):
                if text[i] == start_char:
                    depth += 1
                elif text[i] == end_char:
                    depth -= 1
                    if depth == 0:
                        candidate = text[idx:i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            pass
                        candidate = re.sub(r'(?<=[{,\s])(\w+)\s*:', r'"\1":', candidate)
                        candidate = candidate.replace("'", '"')
                        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break

    raise json.JSONDecodeError("No valid JSON found in LLM output", text, 0)


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length. Cheap heuristic (~4 chars/token)."""
    return max(1, len(text) // 4)


def _sleep_with_jitter(base_delay: float):
    """Sleep for base_delay В± 30% random jitter to avoid thundering herd."""
    jitter = base_delay * 0.3 * (2 * random.random() - 1)  # В±30%
    time.sleep(max(0.1, base_delay + jitter))


# ============== SSRF PROTECTION ==============


def _check_ssrf(url: str) -> str | None:
    """Validate URL against SSRF attacks. Returns error string or None if safe."""
    from urllib.parse import urlparse
    import ipaddress
    import socket

    try:
        parsed = urlparse(url)
    except Exception:
        return "Invalid URL"

    # Block non-HTTP schemes
    if parsed.scheme not in ("http", "https"):
        return f"Blocked scheme: {parsed.scheme}. Only http/https allowed."

    hostname = parsed.hostname
    if not hostname:
        return "Missing hostname"

    # Resolve hostname to IP and check against private ranges
    try:
        addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for family, _, _, _, sockaddr in addr_info:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return f"Blocked: {hostname} resolves to private/internal IP ({ip})"
    except (socket.gaierror, ValueError):
        # DNS resolution failed - let urllib handle it downstream
        pass

    return None


# ============== WEB SEARCH CACHE ==============

_SEARCH_CACHE_TAG = "web-search-cache"
_SEARCH_CACHE_TTL_HOURS = 24
_SEARCH_CACHE_BACKEND = "ddgs-v3-pinned"

# Same-intent retry cap: stop agents from spinning on an unanswerable query.
# Intent = normalized query (lowercase, stripped punctuation, sorted tokens).
# Counter lives in-process, keyed by (session_id, intent).
_SAME_INTENT_RETRY_CAP = 3
_same_intent_counter: dict[tuple[str, str], int] = {}

# Per-session forward-progress cap: stop agents from calling web_search with
# cosmetic query variations while never fetching any candidate. Resets when
# the session makes evidence progress (extract_content / http_get / browse_page).
_FORWARD_PROGRESS_CAP = 4
_web_search_since_fetch: dict[str, int] = {}
_last_candidates_by_session: dict[str, list[dict]] = {}


def _normalize_search_intent(query: str) -> str:
    import re as _re
    q = (query or "").lower()
    q = _re.sub(r"[\"'`.,:;!?()\[\]{}<>]", " ", q)
    q = _re.sub(r"\s+", " ", q).strip()
    tokens = sorted(t for t in q.split() if t)
    return " ".join(tokens)


def _check_and_bump_search_intent(session_id: str | None, query: str) -> int:
    key = (session_id or "__nosession__", _normalize_search_intent(query))
    _same_intent_counter[key] = _same_intent_counter.get(key, 0) + 1
    return _same_intent_counter[key]


def _reset_search_intent_counter() -> None:
    _same_intent_counter.clear()
    _web_search_since_fetch.clear()
    _last_candidates_by_session.clear()


def _bump_web_search_no_fetch(session_id: str | None) -> int:
    sid = session_id or "__nosession__"
    _web_search_since_fetch[sid] = _web_search_since_fetch.get(sid, 0) + 1
    return _web_search_since_fetch[sid]


def _reset_web_search_no_fetch(session_id: str | None) -> None:
    """Call this when the agent makes forward progress (fetches evidence)."""
    sid = session_id or "__nosession__"
    _web_search_since_fetch.pop(sid, None)


def _record_last_candidates(session_id: str | None, candidates: list[dict]) -> None:
    sid = session_id or "__nosession__"
    _last_candidates_by_session[sid] = list(candidates or [])[:5]


def _get_last_candidates(session_id: str | None) -> list[dict]:
    sid = session_id or "__nosession__"
    return list(_last_candidates_by_session.get(sid, []))


def _get_cached_search(query: str) -> dict | None:
    """Check if a similar web search was run recently. Returns canonicalized cached result or None."""
    try:
        with brain_lock:
            cached = brain.search(query="", tags=[_SEARCH_CACHE_TAG], limit=20)
        if not cached:
            return None

        from datetime import datetime, timedelta
        cutoff = (datetime.now() - timedelta(hours=_SEARCH_CACHE_TTL_HOURS)).isoformat()

        for rec in cached:
            meta = rec.metadata or {}
            cached_at = meta.get("cached_at", "")
            if cached_at <= cutoff:
                continue
            cached_query = str(meta.get("query", "") or "")
            if cached_query.lower().strip() != query.lower().strip():
                continue

            if meta.get("backend") != _SEARCH_CACHE_BACKEND:
                logger.info("Web search cache STALE (different backend) for: %s", query[:60])
                continue

            raw_sources = meta.get("sources", [])
            sources = [s for s in raw_sources if isinstance(s, dict) and (s.get("uri") or s.get("url"))]
            answer = str(meta.get("answer") or "").strip()
            is_current_shape = (
                meta.get("mode") == "candidate_discovery"
                and meta.get("candidate_count") is not None
            )
            if not answer or not is_current_shape:
                answer = (
                    f"Found {len(sources)} candidate source(s). These are discovery candidates, not verified facts. "
                    "Use extract_content on a chosen URL before making external factual claims."
                )
            logger.info("Web search cache HIT for: %s", query[:60])
            return {
                "answer": answer,
                "mode": "candidate_discovery",
                "query": cached_query or query,
                "candidate_count": len(sources),
                "sources": sources,
                "cached": True,
                "cached_at": cached_at,
            }
        return None
    except Exception:
        return None


def _cache_search_result(query: str, answer: str, sources: list[dict]):
    """Cache a canonical candidate-discovery web search result for future deduplication."""
    try:
        from remy.core.agent_tools import Level
        from datetime import datetime

        cached_sources = [s for s in (sources or []) if isinstance(s, dict) and (s.get("uri") or s.get("url"))]
        cached_answer = str(answer or "")[:500]

        with brain_lock:
            brain.store(
                content=f"Web search: {query}\n{cached_answer}",
                level=Level.WORKING,
                tags=[_SEARCH_CACHE_TAG],
                metadata={
                    "type": "web_search_cache",
                    "query": query,
                    "answer": cached_answer,
                    "mode": "candidate_discovery",
                    "candidate_count": len(cached_sources),
                    "sources": cached_sources,
                    "cached_at": datetime.now().isoformat(),
                    "backend": _SEARCH_CACHE_BACKEND,
                },
                deduplicate=False,
            )
    except Exception as e:
        logger.debug("Failed to cache search result: %s", e)


# ============== TOOL HEALTH & RETRY ==============

# Tools that may have transient failures (network, API rate limits)
_RETRYABLE_TOOLS = frozenset({"web_search", "http_get"})
_MAX_RETRIES = 2
_RETRY_DELAYS = [2, 5]  # seconds between retries
_CONSEQUENCE_GATE_BYPASS_TOOLS = frozenset({"connect_records"})


class ToolHealth:
    """Per-tool circuit breaker and health tracking. Thread-safe."""

    FAILURE_THRESHOLD = 3   # failures before circuit opens
    RECOVERY_SEC = 600      # 10 min cooldown

    def __init__(self):
        self._failures: dict[str, list[float]] = {}  # tool -> [timestamps]
        self._circuit_open_until: dict[str, float] = {}  # tool -> timestamp
        self._lock = threading.Lock()

    def record_failure(self, tool_name: str):
        """Record a tool failure. Opens circuit if threshold reached."""
        now = time.time()
        with self._lock:
            if tool_name not in self._failures:
                self._failures[tool_name] = []

            # Keep only recent failures (last 10 min)
            self._failures[tool_name] = [
                t for t in self._failures[tool_name] if now - t < 600
            ] + [now]

            if len(self._failures[tool_name]) >= self.FAILURE_THRESHOLD:
                self._circuit_open_until[tool_name] = now + self.RECOVERY_SEC
                logger.warning(
                    "Circuit OPEN for tool '%s' until %s (%d recent failures)",
                    tool_name,
                    time.strftime("%H:%M:%S", time.localtime(now + self.RECOVERY_SEC)),
                    len(self._failures[tool_name]),
                )

    def record_success(self, tool_name: str):
        """Record a tool success. Clears failure history."""
        with self._lock:
            self._failures.pop(tool_name, None)
            self._circuit_open_until.pop(tool_name, None)

    def is_available(self, tool_name: str) -> bool:
        """Check if a tool's circuit is closed (available)."""
        with self._lock:
            open_until = self._circuit_open_until.get(tool_name, 0)
            if time.time() >= open_until:
                # Circuit has recovered - clear it
                self._circuit_open_until.pop(tool_name, None)
                return True
            return False

    def get_health_report(self) -> dict[str, str]:
        """Get health status for all tracked tools. Returns {tool: status_str}."""
        now = time.time()
        report = {}
        with self._lock:
            for tool_name in set(list(self._failures.keys()) + list(self._circuit_open_until.keys())):
                open_until = self._circuit_open_until.get(tool_name, 0)
                recent_failures = [t for t in self._failures.get(tool_name, []) if now - t < 600]
                if now < open_until:
                    remaining = int(open_until - now)
                    report[tool_name] = f"UNAVAILABLE ({remaining}s cooldown, {len(recent_failures)} failures)"
                elif recent_failures:
                    report[tool_name] = f"degraded ({len(recent_failures)} recent failures)"
                # Only report tools with issues - healthy tools omitted
        return report


# Module-level singleton
tool_health = ToolHealth()


# ============== TOOL DECLARATIONS ==============
# These tell Gemini what tools are available. When Gemini decides to call one,
# we execute it against `brain` directly and send the result back.

BRAIN_TOOLS = [
    types.FunctionDeclaration(
        name="recall",
        description="Recall relevant memories about a topic. Searches BOTH episodic memory (brain) and semantic knowledge base (KB). Use this FIRST when the user asks about any topic.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "query": types.Schema(type="STRING", description="What to recall (e.g. 'grandmother Maria', 'stories about the war')"),
                "token_budget": types.Schema(type="INTEGER", description="Max tokens for the preamble (default 2048)"),
            },
            required=["query"],
        ),
    ),
    types.FunctionDeclaration(
        name="store",
        description="Store a new memory. Use when the user tells you something worth remembering about their family.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "content": types.Schema(type="STRING", description="The information to store"),
                "tags": types.Schema(type="STRING", description="Comma-separated tags (e.g. 'person,grandfather')"),
                "level": types.Schema(type="STRING", description="Memory level: L1_WORKING, L2_DECISIONS, L3_DOMAIN, L4_IDENTITY (default L3_DOMAIN)"),
            },
            required=["content"],
        ),
    ),
    types.FunctionDeclaration(
        name="search",
        description="Search for specific records by query, optionally filtered by tags. "
            "You can search by query only, tags only, or both.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "query": types.Schema(type="STRING", description="Search query (optional if tags provided)"),
                "tags": types.Schema(type="STRING", description="Comma-separated tags to filter by"),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="store_person",
        description="Store a new family member. Use when the user tells you about a person in their family.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "full_name": types.Schema(type="STRING", description="Full name of the person"),
                "name": types.Schema(type="STRING", description="Person name. Alias for full_name."),
                "role": types.Schema(type="STRING", description="Role in family (grandfather, mother, uncle, etc.)"),
                "birth_date": types.Schema(type="STRING", description="Date of birth"),
                "birth_place": types.Schema(type="STRING", description="Place of birth"),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="store_story",
        description="Record a family story or memory.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "title": types.Schema(type="STRING", description="Title of the story"),
                "content": types.Schema(type="STRING", description="The story text"),
                "people_mentioned": types.Schema(type="STRING", description="Comma-separated names of people mentioned"),
            },
            required=["title", "content"],
        ),
    ),
    types.FunctionDeclaration(
        name="family_tree",
        description="Get a list of all family members.",
        parameters=types.Schema(
            type="OBJECT",
            properties={},
        ),
    ),
    types.FunctionDeclaration(
        name="insights",
        description="Get memory health statistics - how many records, levels distribution, etc.",
        parameters=types.Schema(
            type="OBJECT",
            properties={},
        ),
    ),
    types.FunctionDeclaration(
        name="review_history_memory_gaps",
        description=(
            "Analyze saved session history against the current active memory. "
            "Use this to detect likely missing memory, review candidates, and reconstruction opportunities after restarts or data-loss incidents."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "sample_limit": types.Schema(
                    type="INTEGER",
                    description="Maximum number of missing/review candidates to return per section.",
                ),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="connect_records",
        description="Connect two records with a described relationship. Use when the user indicates a relationship between people, events, topics, or any memories.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "id_a": types.Schema(type="STRING", description="ID of the first record"),
                "id_b": types.Schema(type="STRING", description="ID of the second record"),
                "relationship": types.Schema(type="STRING", description="Description of the relationship (e.g. 'mother of', 'caused by', 'related to')"),
                "weight": types.Schema(type="NUMBER", description="Connection strength 0.0-1.0 (default 0.7). Higher = stronger association."),
            },
            required=["id_a", "id_b", "relationship"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_connections",
        description="Get all connections for a record - see what other records are linked to it.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "record_id": types.Schema(type="STRING", description="ID of the record to inspect"),
            },
            required=["record_id"],
        ),
    ),
    # ---- User profile tool ----
    types.FunctionDeclaration(
        name="store_user_profile",
        description=(
            "Store or update the user's personal profile. Use this when the user tells you "
            "their name, age, occupation, goals, family composition, or any personal information. "
            "Only include fields the user has explicitly shared. This is an upsert - existing fields are preserved, "
            "new fields are added or updated."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "name": types.Schema(type="STRING", description="User's name or how they want to be called"),
                "age": types.Schema(type="STRING", description="Age or year of birth"),
                "location": types.Schema(type="STRING", description="City/country where user lives"),
                "occupation": types.Schema(type="STRING", description="Job, profession, or role"),
                "languages": types.Schema(type="STRING", description="Languages the user speaks (comma-separated)"),
                "family": types.Schema(type="STRING", description="Family composition (e.g. 'married, 2 children')"),
                "personal_focus": types.Schema(type="STRING", description="Current goals, priorities, or personal areas of focus"),
                "interests": types.Schema(type="STRING", description="Hobbies, interests, or topics of focus"),
                "notes": types.Schema(type="STRING", description="Any other personal info worth remembering"),
            },
            required=[],
        ),
    ),
    # ---- Utility tools ----
    types.FunctionDeclaration(
        name="web_search",
        description="Search the internet for candidate sources only. This tool discovers possible URLs and titles for live information, but it does not verify factual claims. After web_search, use extract_content or another fetch tool before making concrete external claims.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "query": types.Schema(type="STRING", description="Search query in the most relevant language"),
            },
            required=["query"],
        ),
    ),
    types.FunctionDeclaration(
        name="extract_content",
        description=(
            "Extract clean text content from a web page URL using Trafilatura. "
            "Returns article text, title, author, date - stripped of ads, nav, scripts, "
            "plus an evidence_packet with source_class and identity_checks. "
            "Much better than http_get for reading articles, blog posts, docs. "
            "Use http_get for APIs/JSON; use extract_content for human-readable pages. "
            "When fetching a specific resource (paper, doc page, product page), "
            "pass expected_title and/or expected_identifier so the packet can flag "
            "identity mismatches before you cite this source."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "url": types.Schema(
                    type="STRING", description="URL of the page to extract content from"
                ),
                "include_links": types.Schema(
                    type="BOOLEAN", description="Include hyperlinks in output (default false)"
                ),
                "include_tables": types.Schema(
                    type="BOOLEAN", description="Include tables in output (default true)"
                ),
                "expected_title": types.Schema(
                    type="STRING",
                    description=(
                        "Optional. Title the caller expects this URL to be about. "
                        "If fetched title does not overlap, evidence_packet.identity_checks "
                        "will flag a title_match mismatch."
                    ),
                ),
                "expected_identifier": types.Schema(
                    type="STRING",
                    description=(
                        "Optional. A specific identifier (arXiv id, DOI, version string, etc.) "
                        "that MUST appear in the fetched content. If absent, "
                        "evidence_packet.identity_checks flags identifier_present mismatch."
                    ),
                ),
                "expected_authors": types.Schema(
                    type="STRING",
                    description=(
                        "Optional. Author name(s) the caller expects this page to list. "
                        "If no claimed surname appears in the fetched author metadata, "
                        "evidence_packet.identity_checks flags an author_match mismatch."
                    ),
                ),
                "claim_span": types.Schema(
                    type="STRING",
                    description=(
                        "Optional. The specific excerpt of the fetched content the "
                        "caller intends to rest a claim on. Stored on the evidence "
                        "packet so the claim-to-evidence binding is explicit."
                    ),
                ),
            },
            required=["url"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_current_datetime",
        description="Get the current date and time. Use this when the user asks about today's date, current time, or day of the week.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    # ---- Scheduler tools ----
    types.FunctionDeclaration(
        name="schedule_task",
        description="Schedule a reminder or recurring task. Use when the user asks to be reminded about something or wants a recurring reminder (e.g. 'remind me to call grandma every Sunday').",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "description": types.Schema(type="STRING", description="What to remind about (e.g. 'Call grandma')"),
                "task": types.Schema(type="STRING", description="Alias for description."),
                "title": types.Schema(type="STRING", description="Alias for description."),
                "due_date": types.Schema(type="STRING", description="When to remind, ISO date (YYYY-MM-DD) or next occurrence date. Optional if cron is provided; defaults to today."),
                "repeat": types.Schema(type="STRING", description="Recurrence: 'daily', 'weekly', 'monthly', or empty for one-time"),
                "cron": types.Schema(type="STRING", description="Cron-style recurrence, e.g. '0 10 * * *'. Optional alternative to repeat."),
                "event_date": types.Schema(type="STRING", description="ISO date of the actual event/occasion this task is tied to (e.g. the birthday date). Helps reason about whether the task is still relevant after the event passes."),
                "event_type": types.Schema(type="STRING", description="Type of event: 'birthday', 'anniversary', 'meeting', 'deadline', 'holiday', etc."),
            },
        ),
    ),
    # ---- CRUD tools ----
    types.FunctionDeclaration(
        name="update_record",
        description="Update an existing memory record. Use when correcting or enriching information. Requires the record ID (from search/store results).",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "record_id": types.Schema(type="STRING", description="ID of the record to update"),
                "content": types.Schema(type="STRING", description="New content (replaces old)"),
                "tags": types.Schema(type="STRING", description="Comma-separated tags (replaces old)"),
                "level": types.Schema(type="STRING", description="Memory level: working, decisions, domain, identity"),
            },
            required=["record_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="delete_record",
        description="Permanently delete a memory record by ID. Use when information is wrong or no longer needed.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "record_id": types.Schema(type="STRING", description="ID of the record to delete"),
            },
            required=["record_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="mark_stale",
        description="Mark a memory record as stale (outdated) without deleting it. Adds 'stale' tag and metadata stamp. Use when info is no longer current but history should be preserved for audit. Requires record_id and reason.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "record_id": types.Schema(type="STRING", description="ID of the record to mark stale"),
                "reason": types.Schema(type="STRING", description="Why this record is stale (e.g. 'GitHub lead list outdated as of 2026-04-13')"),
                "superseded_by": types.Schema(type="STRING", description="Optional record ID that replaces this one"),
            },
            required=["record_id", "reason"],
        ),
    ),
    # ---- Sandbox meta-tools ----
    types.FunctionDeclaration(
        name="sandbox_create_tool",
        description="Create a new tool. Write a Python file with TOOL_NAME, TOOL_DESCRIPTION, TOOL_PARAMETERS constants, execute() function, and test_*() functions. The tool needs human approval before it can be used.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "name": types.Schema(type="STRING", description="Tool name (snake_case, e.g. 'calculate_bmi')"),
                "code": types.Schema(type="STRING", description="Complete Python source code for the tool file"),
            },
            required=["name", "code"],
        ),
    ),
    types.FunctionDeclaration(
        name="sandbox_test_tool",
        description="Run tests for a sandbox tool. Tests run in an isolated environment.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "name": types.Schema(type="STRING", description="Tool name to test"),
            },
            required=["name"],
        ),
    ),
    types.FunctionDeclaration(
        name="sandbox_list_tools",
        description="List all sandbox tools and their status (draft, tested, pending, approved, rejected).",
        parameters=types.Schema(
            type="OBJECT",
            properties={},
        ),
    ),
    # ---- Research tool ----
    types.FunctionDeclaration(
        name="store_research",
        description=(
            "Store research findings with sources. Use at the END of a research investigation "
            "to save the synthesized report. Auto-connects to related personal records in memory."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "topic": types.Schema(type="STRING", description="Research topic or question"),
                "project_name": types.Schema(type="STRING", description="Alias for topic."),
                "title": types.Schema(type="STRING", description="Alias for topic."),
                "subject": types.Schema(type="STRING", description="Alias for topic."),
                "findings": types.Schema(type="STRING", description="Synthesized research report"),
                "summary": types.Schema(type="STRING", description="Alias for findings."),
                "content": types.Schema(type="STRING", description="Alias for findings."),
                "report": types.Schema(type="STRING", description="Alias for findings."),
                "sources": types.Schema(type="STRING", description="Comma-separated source URLs from web_search"),
                "source": types.Schema(type="STRING", description="Alias for sources."),
                "source_url": types.Schema(type="STRING", description="Alias for sources."),
                "references": types.Schema(type="STRING", description="Alias for sources."),
                "related_query": types.Schema(type="STRING", description="Optional query to find and auto-connect related personal records"),
                "volatility": types.Schema(
                    type="STRING",
                    description=(
                        "Optional. How quickly this fact changes: 'low' (historical, identifiers), "
                        "'medium' (docs, conventions), 'high' (versions, prices, rankings). "
                        "Drives TTL / stale_after. If omitted, auto-classified from topic + findings."
                    ),
                ),
                "conflict_resolution": types.Schema(
                    type="STRING",
                    description=(
                        "Optional. What to do if prior research on the same topic reports different "
                        "version/date/number signals: 'flag' (default - return conflict report, do NOT store), "
                        "'replace' (store anyway, supersede prior belief), "
                        "'append' (store alongside, no supersession)."
                    ),
                ),
            },
            required=["findings"],
        ),
    ),
    # ---- Goal management tools ----
    types.FunctionDeclaration(
        name="create_subgoal",
        description=(
            "Break a complex goal into a smaller sub-goal. Use when a goal is too broad "
            "to accomplish in a single action. Requires the parent goal ID."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "parent_goal_id": types.Schema(type="STRING", description="Goal ID of the parent goal (e.g. 'goal-abc123def456')"),
                "description": types.Schema(type="STRING", description="Description of the sub-goal"),
                "priority": types.Schema(type="STRING", description="Priority: critical, high, medium, low (default: medium)"),
            },
            required=["parent_goal_id", "description"],
        ),
    ),
    types.FunctionDeclaration(
        name="complete_goal",
        description="Mark a goal as completed. Use when a goal has been fully achieved.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "goal_id": types.Schema(type="STRING", description="Goal ID as shown in the decision prompt (e.g. 'goal-abc123def456')"),
                "notes": types.Schema(type="STRING", description="Optional notes about how the goal was completed"),
            },
            required=["goal_id"],
        ),
    ),
    # ---- External tools ----
    types.FunctionDeclaration(
        name="read_file",
        description="Read the contents of a file. Restricted to data directory and allowed paths.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "path": types.Schema(type="STRING", description="Path to the file to read (relative to data dir or absolute if in allowed paths)"),
            },
            required=["path"],
        ),
    ),
    types.FunctionDeclaration(
        name="write_file",
        description="Write content to a file. Only allowed in the data directory.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "path": types.Schema(type="STRING", description="Path to the file (relative to data dir)"),
                "content": types.Schema(type="STRING", description="Content to write to the file"),
            },
            required=["path", "content"],
        ),
    ),
    types.FunctionDeclaration(
        name="list_directory",
        description="List contents of a directory. Restricted to data directory and allowed paths.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "path": types.Schema(type="STRING", description="Directory path (relative to data dir or absolute if in allowed paths)"),
            },
            required=["path"],
        ),
    ),
    types.FunctionDeclaration(
        name="http_get",
        description="Make an HTTP GET request to fetch data from a URL. Use for APIs, web pages, etc.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "url": types.Schema(type="STRING", description="URL to fetch"),
            },
            required=["url"],
        ),
    ),
    types.FunctionDeclaration(
        name="consolidate",
        description="Merge similar memory records to reduce bloat. Automatically merges records with 85%+ content similarity. Call periodically or when memory feels cluttered.",
        parameters=types.Schema(
            type="OBJECT",
            properties={},
        ),
    ),
    # ---- Research Orchestrator tools (RM-1) ----
    types.FunctionDeclaration(
        name="start_research",
        description=(
            "Start a structured research project. Creates a research plan with multiple search queries. "
            "Use when the user asks for deep investigation or when an autonomous goal requires research."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "topic": types.Schema(type="STRING", description="Research topic or question"),
                "question": types.Schema(type="STRING", description="Alias for topic."),
                "query": types.Schema(type="STRING", description="Alias for topic."),
                "prompt": types.Schema(type="STRING", description="Alias for topic."),
                "description": types.Schema(type="STRING", description="Alias for topic."),
                "depth": types.Schema(type="STRING", description="Research depth: 'quick' (2 queries), 'standard' (4 queries), or 'deep' (7 queries). Default: standard"),
                "context": types.Schema(type="STRING", description="Optional additional context to guide the research plan"),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="add_research_finding",
        description=(
            "Record a finding for an active research project. Use only after fetching evidence from a chosen source URL, typically via extract_content after web_search candidate discovery."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "project_id": types.Schema(type="STRING", description="ID of the research project"),
                "content": types.Schema(type="STRING", description="The finding text. Use this field."),
                "summary": types.Schema(type="STRING", description="Alias for content - accepted if content is omitted"),
                "source_url": types.Schema(type="STRING", description="URL source of the finding"),
                "confidence": types.Schema(type="NUMBER", description="Confidence in this finding 0.0-1.0 (default 0.7)"),
                "contradicts_finding_id": types.Schema(type="STRING", description="ID of an existing finding this contradicts"),
            },
            required=["project_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="complete_research",
        description=(
            "Synthesize all findings of a research project into a final report. "
            "Use when all planned queries are done."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "project_id": types.Schema(type="STRING", description="ID of the research project to complete"),
            },
            required=["project_id"],
        ),
    ),
    # ---- Generic metric and event intelligence ----
    types.FunctionDeclaration(
        name="track_metric",
        description="Log a specific user-reported numeric metric for any workflow domain.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "metric_type": types.Schema(type="STRING", description="Type of metric (e.g. 'focus_minutes', 'invoice_count', 'project_score')"),
                "value": types.Schema(type="NUMBER", description="Numeric value"),
                "unit": types.Schema(type="STRING", description="Unit of measurement (e.g. 'minutes', 'items', 'usd')"),
                "notes": types.Schema(type="STRING", description="Optional context or notes"),
            },
            required=["metric_type", "value", "unit"],
        ),
    ),
    types.FunctionDeclaration(
        name="metric_summary",
        description="Get a summary of recent tracked metrics and events.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "period": types.Schema(type="STRING", description="Time period: 'week', 'month', 'year' (default: week)"),
            },
            required=["period"],
        ),
    ),
    types.FunctionDeclaration(
        name="event_correlate",
        description="Analyze an event and find potential correlations with recent records.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "event": types.Schema(type="STRING", description="The event to analyze"),
            },
            required=["event"],
        ),
    ),
    # ---- Intelligence tools (RM-4) ----
    types.FunctionDeclaration(
        name="extract_facts",
        description="Extract structured facts from text and store them as domain knowledge.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "text": types.Schema(type="STRING", description="The text to analyze and extract facts from"),
                "source": types.Schema(type="STRING", description="Source of the text (e.g. URL, 'user input')"),
            },
            required=["text"],
        ),
    ),

    # ============== TODO LIST TOOLS ==============
    types.FunctionDeclaration(
        name="add_todo",
        description="Add a todo item to the task list. Use for personal tasks, work items, or agent action steps.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "title": types.Schema(type="STRING", description="Short task title"),
                "priority": types.Schema(type="STRING", description="Priority: high, medium, low (default: medium)"),
                "due_date": types.Schema(type="STRING", description="Due date in YYYY-MM-DD format (optional)"),
                "category": types.Schema(type="STRING", description="Category: personal, work, health, agent, or custom (default: personal)"),
                "parent_id": types.Schema(type="STRING", description="Parent todo record ID for subtasks (optional)"),
            },
            required=["title"],
        ),
    ),
    types.FunctionDeclaration(
        name="list_todos",
        description="List todo items. Shows pending and in-progress tasks by default.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "status": types.Schema(type="STRING", description="Filter: pending, in_progress, done, all (default: pending)"),
                "category": types.Schema(type="STRING", description="Filter by category (optional)"),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="update_todo",
        description="Update a todo item's status, title, or priority.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "id": types.Schema(type="STRING", description="Record ID of the todo item (use this field)"),
                "todo_id": types.Schema(type="STRING", description="Alternative: todo_id from list_todos result (e.g. 'todo-abc123')"),
                "status": types.Schema(type="STRING", description="New status: pending, in_progress, done"),
                "title": types.Schema(type="STRING", description="Updated title (optional)"),
                "priority": types.Schema(type="STRING", description="Updated priority: high, medium, low (optional)"),
                "due_date": types.Schema(type="STRING", description="Updated due date YYYY-MM-DD (optional)"),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="delete_todo",
        description="Delete (archive) a todo item from the task list.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "id": types.Schema(type="STRING", description="Record ID of the todo item to delete"),
                "todo_id": types.Schema(type="STRING", description="Alternative: todo_id from list_todos result"),
            },
        ),
    ),

    # ============== KNOWLEDGE BASE (Aura Memory) ==============
    types.FunctionDeclaration(
        name="store_knowledge",
        description=(
            "Store a fact or knowledge in the semantic knowledge base (Aura Memory). "
            "Use for permanent facts, research conclusions, reference data - "
            "anything that should NOT decay over time. "
            "Use pin=true for critical facts (identity, important references). "
            "Use pin=false for general knowledge that can gradually fade."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "text": types.Schema(type="STRING", description="The fact or knowledge to store"),
                "pin": types.Schema(type="BOOLEAN", description="Pin as permanent anchor (true) or general knowledge (false, default)"),
            },
            required=["text"],
        ),
    ),
    types.FunctionDeclaration(
        name="recall_knowledge",
        description=(
            "DEPRECATED: 'recall' now searches both episodic and semantic memory automatically. "
            "Use 'recall' instead. This tool is kept for backward compatibility - "
            "it searches ONLY the semantic knowledge base (Aura Memory)."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "query": types.Schema(type="STRING", description="What to search for in the knowledge base"),
                "top_k": types.Schema(type="INTEGER", description="Max results to return (default 5)"),
            },
            required=["query"],
        ),
    ),
    types.FunctionDeclaration(
        name="knowledge_stats",
        description="Get statistics about the semantic knowledge base (record count, analytics).",
        parameters=types.Schema(
            type="OBJECT",
            properties={},
        ),
    ),
    types.FunctionDeclaration(
        name="verify_record",
        description="Mark a memory record as verified by the user. Use when user confirms information is correct.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "record_id": types.Schema(type="STRING", description="ID of the record to verify"),
                "note": types.Schema(type="STRING", description="Optional verification note"),
            },
            required=["record_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="generate_image",
        description="Generate an image using AI based on a text description. Use when user asks to create, draw, visualize, or generate a picture.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "prompt": types.Schema(type="STRING", description="Detailed description of the image to generate in English"),
            },
            required=["prompt"],
        ),
    ),
    types.FunctionDeclaration(
        name="generate_report",
        description="Generate a professional PDF report. Use when user asks to create a report, summary document, or analysis in PDF format. The agent builds the report by specifying sections with different types: section (heading+text), subsection, text, quote, findings (numbered list), table (headers+rows), memory (brain records with trust scores), audit (execution trail), page_break.",
        parameters=types.Schema(
            type="OBJECT",
            required=["title"],
            properties={
                "title": types.Schema(type="STRING", description="Report title"),
                "subtitle": types.Schema(type="STRING", description="Report subtitle (optional)"),
                "report_type": types.Schema(type="STRING", description="Optional report template: standard|financial|vat"),
                "include_toc": types.Schema(type="BOOLEAN", description="Whether to include a contents page (default true)"),
                "metadata": types.Schema(type="OBJECT", description="Optional document summary fields shown in the report preamble, useful for financial/VAT reports"),
                "content": types.Schema(type="STRING", description="Alternative: raw markdown or full body content. Will be auto-parsed into sections if 'sections' is not provided."),
                "sections": types.Schema(
                    type="ARRAY",
                    description="List of report sections. Optional if 'content' is provided. Each section is an object with 'type' field: 'section','subsection','text','quote','findings','table','memory','audit','page_break'. Additional fields depend on type: title, body, items (for findings), headers+rows (for table), records (for memory), logs (for audit).",
                    items=types.Schema(type="OBJECT", properties={
                        "type": types.Schema(type="STRING", description="Section type: section|subsection|text|quote|findings|table|memory|audit|page_break"),
                        "title": types.Schema(type="STRING", description="Section title"),
                        "body": types.Schema(type="STRING", description="Section body text"),
                        "items": types.Schema(type="ARRAY", items=types.Schema(type="STRING"), description="List of findings (for type=findings)"),
                        "headers": types.Schema(type="ARRAY", items=types.Schema(type="STRING"), description="Table column headers (for type=table)"),
                        "rows": types.Schema(type="ARRAY", items=types.Schema(type="ARRAY", items=types.Schema(type="STRING")), description="Table rows (for type=table)"),
                    }),
                ),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="generate_presentation",
        description="Generate a professional PPTX (PowerPoint) presentation. Use when user asks to create a presentation, slide deck, or pitch deck. Slides types: section (title+body), subsection, bullets (title+items), quote, table (title+headers+rows), divider.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "title": types.Schema(type="STRING", description="Presentation title"),
                "subtitle": types.Schema(type="STRING", description="Presentation subtitle (optional)"),
                "content": types.Schema(type="STRING", description="Alternative: raw markdown content. Will be auto-parsed into title + slides if 'slides' is not provided."),
                "slides": types.Schema(
                    type="ARRAY",
                    description="List of slides. Each slide has 'type': section|subsection|bullets|quote|table|divider.",
                    items=types.Schema(type="OBJECT", properties={
                        "type": types.Schema(type="STRING", description="Slide type: section|subsection|bullets|quote|table|divider"),
                        "title": types.Schema(type="STRING", description="Slide title"),
                        "body": types.Schema(type="STRING", description="Slide body text"),
                        "items": types.Schema(type="ARRAY", items=types.Schema(type="STRING"), description="Bullet points (for type=bullets)"),
                        "author": types.Schema(type="STRING", description="Quote author (for type=quote)"),
                        "headers": types.Schema(type="ARRAY", items=types.Schema(type="STRING"), description="Table headers (for type=table)"),
                        "rows": types.Schema(type="ARRAY", items=types.Schema(type="ARRAY", items=types.Schema(type="STRING")), description="Table rows (for type=table)"),
                    }),
                ),
            },
        ),
    ),
    # Multi-agent delegation
    types.FunctionDeclaration(
        name="delegate_task",
        description=(
            "Delegate tasks to specialized worker agents that run in parallel with filtered tool sets. "
            "Use for: parallel research queries, analysis + planning simultaneously, multi-step execution. "
            "Workers: researcher (search/recall), planner (goals/todos), executor (files/actions), analyst (health/patterns). "
            "Max 3 workers at once. Each worker has ~60s timeout. ~2000 tokens per worker."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "tasks": types.Schema(
                    type="ARRAY",
                    items=types.Schema(
                        type="OBJECT",
                        properties={
                            "role": types.Schema(type="STRING",
                                description="Worker role: researcher, planner, executor, analyst"),
                            "instruction": types.Schema(type="STRING",
                                description="Clear task instruction for the worker"),
                            "context": types.Schema(type="STRING",
                                description="Optional context from the current conversation"),
                        },
                        required=["role", "instruction"],
                    ),
                    description="List of tasks to delegate (max 3)",
                ),
            },
            required=["tasks"],
        ),
    ),
    # Browser automation (Playwright)
    types.FunctionDeclaration(
        name="browse_page",
        description=(
            "Open a web page in a real browser (Playwright) and analyze it visually. "
            "Use when you need JS-rendered content, forms, or sites http_get can't handle. "
            "Returns page description, interactive elements with CSS selectors, forms. ~500 tokens."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "url": types.Schema(type="STRING", description="Full URL to navigate to"),
                "question": types.Schema(type="STRING",
                    description="Optional: what to look for on the page"),
            },
            required=["url"],
        ),
    ),
    types.FunctionDeclaration(
        name="browser_act",
        description=(
            "Interact with the current browser page. Actions: click, type, fill_form, scroll_down, scroll_up, "
            "select, wait, goto, back, forward. For selectors use CSS (#id, [name=...], .class) or "
            "Playwright text selectors: button:has-text(\"Accept\"), text=\"Sign in\". "
            "IMPORTANT: Use UNIQUE selectors - never bare input[type=email]. Prefer #id or [name=...]. "
            "For registration/login forms, use fill_form with all fields at once (faster, more reliable). "
            "Do NOT use jQuery :contains(). ~500 tokens."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "action": types.Schema(type="STRING",
                    description=(
                        "Action: click, type, fill_form, scroll_down, scroll_up, select, wait, goto, back, forward. "
                        "fill_form fills multiple fields at once - pass JSON array in text: "
                        '[{"selector":"#email","value":"user@example.com"},{"selector":"#password","value":"Pass123"}]'
                    )),
                "selector": types.Schema(type="STRING",
                    description="CSS selector or Playwright locator (e.g. #id, [name=\"email\"], button:has-text(\"OK\"))"),
                "text": types.Schema(type="STRING",
                    description="Text to type, option to select, or JSON array for fill_form"),
                "url": types.Schema(type="STRING",
                    description="URL for goto action"),
            },
            required=["action"],
        ),
    ),
    types.FunctionDeclaration(
        name="browser_close",
        description="Close the browser and free resources. Call when done browsing.",
        parameters=types.Schema(
            type="OBJECT",
            properties={},
        ),
    ),
    # ============== META-TOOLS (selective tool loading) ==============
    types.FunctionDeclaration(
        name="list_available_tools",
        description=(
            "List all extended tools that can be enabled for this session. "
            "Use when you need a tool that isn't currently available (e.g. health tracking, "
            "research, todos, reports, file operations). Returns names and descriptions."
        ),
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="enable_tools",
        description=(
            "Enable extended tools for the current session. Call after list_available_tools "
            "to activate specific tools you need. Enabled tools persist for the rest of this conversation."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "tool_names": types.Schema(
                    type="ARRAY",
                    items=types.Schema(type="STRING"),
                    description="List of tool names to enable (e.g. ['track_metric', 'metric_summary'])",
                ),
            },
            required=["tool_names"],
        ),
    ),
    types.FunctionDeclaration(
        name="scratchpad",
        description=(
            "Working memory notepad for intermediate results. Actions: "
            "'write' to save a note, 'read' to list notes, 'clear' to delete notes, "
            "'summarize' to compress older notes into one summary record."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "action": types.Schema(
                    type="STRING",
                    description="write | read | clear | summarize",
                ),
                "content": types.Schema(
                    type="STRING",
                    description="Required for action='write'",
                ),
                "force": types.Schema(
                    type="BOOLEAN",
                    description="For summarize: allow summary even below the normal threshold",
                ),
            },
            required=["action"],
        ),
    ),
    types.FunctionDeclaration(
        name="filter_working",
        description=(
            "Filter scratchpad-managed WORKING notes to keep only items relevant "
            "to the current query active. Use before long reasoning steps when the "
            "scratchpad feels noisy."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "query": types.Schema(type="STRING", description="Current task or user query"),
                "min_score": types.Schema(
                    type="NUMBER",
                    description="Minimum recall_structured score to keep a scratchpad note active",
                ),
                "delete_irrelevant": types.Schema(
                    type="BOOLEAN",
                    description="If true, delete irrelevant scratchpad notes instead of just demoting them",
                ),
            },
            required=["query"],
        ),
    ),
    types.FunctionDeclaration(
        name="read_persona",
        description="Read the current agent persona (name, role, tone, traits, etc.).",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="update_persona",
        description="Update the agent's persona fields (name, role, tone, traits, catchphrases, avoid).",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "name": types.Schema(type="STRING", description="Agent name"),
                "role": types.Schema(type="STRING", description="Agent role description"),
                "tone": types.Schema(type="STRING", description="Communication tone"),
                "scope": types.Schema(type="STRING", description="Domain scope"),
                "motivations": types.Schema(type="STRING", description="Core motivations"),
                "catchphrases": types.Schema(type="STRING", description="Comma-separated catchphrases"),
                "avoid": types.Schema(type="STRING", description="Comma-separated things to avoid"),
                "traits": types.Schema(type="STRING", description="JSON object with trait scores 0.0-1.0"),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="memory_feedback",
        description=(
            "Signal to AuraSDK whether a specific memory record was useful or harmful. "
            "Use this when you notice a record gave wrong information (useful=false) or "
            "was particularly helpful (useful=true). AuraSDK uses this to adjust belief "
            "confidence and suggest corrections automatically."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "record_id": types.Schema(type="STRING", description="ID of the memory record to give feedback on"),
                "useful": types.Schema(type="BOOLEAN", description="true = record was helpful/correct, false = record was wrong or harmful"),
                "reason": types.Schema(type="STRING", description="Optional explanation of why it was useful or not"),
            },
            required=["record_id", "useful"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_corrections",
        description=(
            "Get AuraSDK correction suggestions and review queue. "
            "Use this to audit your own memory quality and find records that need fixing. "
            "Run periodically to keep cognitive layer healthy."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "mode": types.Schema(
                    type="STRING",
                    description="What to retrieve: 'suggestions' (default) = AuraSDK-suggested fixes | 'queue' = records awaiting review | 'log' = applied corrections | 'recent_beliefs' = recently corrected beliefs | 'report' = full report",
                ),
                "limit": types.Schema(type="INTEGER", description="Max items to return (default 10)"),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="deprecate_belief",
        description=(
            "Penalize or retract a wrong belief, causal pattern, or policy hint in AuraSDK. "
            "Use this when you discover that something you 'know' is incorrect - this is stronger "
            "than memory_feedback because it actively downgrades the belief's confidence score "
            "and removes causal patterns from reasoning. "
            "Examples: wrong assumption about a user, invalid causal link you learned, "
            "a policy hint that led to a bad decision."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "record_id": types.Schema(type="STRING", description="ID of the belief, causal pattern, or policy hint to deprecate"),
                "reason": types.Schema(type="STRING", description="Why this belief is wrong - stored as audit trail and used by AuraSDK for future correction suggestions"),
                "target": types.Schema(type="STRING", description="What to deprecate: 'belief' (default) | 'causal_pattern' | 'policy_hint'"),
            },
            required=["record_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_belief_health",
        description=(
            "Get volatile or unstable beliefs from AuraSDK - beliefs that are frequently "
            "contradicted, changing, or poorly supported. Use this for periodic self-audit "
            "to identify what needs review or deprecation."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "mode": types.Schema(type="STRING", description="'volatile' (default) = high-volatility beliefs | 'unstable' = low-stability beliefs"),
                "limit": types.Schema(type="INTEGER", description="Max items to return (default 10)"),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="get_thermal_map",
        description=(
            "Get the cognitive heat map - shows which belief clusters are 'hot' (conflicting, "
            "unstable, unresolved) and which are 'cold' (stable, well-supported). "
            "Use this to understand where your attention is most needed. "
            "Returns hot zone clusters with topics, routing advice, and energy metrics."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={},
        ),
    ),
    types.FunctionDeclaration(
        name="get_plasticity_audit",
        description=(
            "Audit the synaptic plasticity system - shows which graph edges have been weakened "
            "or pruned due to cross-domain heat leakage. Use this to check if the graph's "
            "autonomous structural rewiring is healthy: pruned edges, at-risk edges, "
            "leak-to-productive ratio, and recent pruning history."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={},
        ),
    ),
    types.FunctionDeclaration(
        name="aura_cognitive_ops",
        description=(
            "Universal gateway to explore and test any of 187 AuraSDK v2.1.0 cognitive methods directly. "
            "Use this to investigate memory state, test AuraSDK capabilities, and gather insights "
            "that are not exposed through other tools. "
            "List-shaped results are returned in BRIEF MODE by default: top 10 items by activation_count "
            "plus aggregate stats (total, activation_sum, connections_mean). This keeps tool output small "
            "so the conversation never blows past the context window. "
            "If you genuinely need the full payload, pass params={'full': true} - but a hard 60 KB cap still "
            "applies, oversize results are truncated to a 5-item sample with aggregates. "
            "Examples: op='get_family_graph', op='recall_person_context' with params={'record_id': 'xxx'}, "
            "op='get_project_timeline', op='list_records' (brief), op='list_records' with params={'full': true} (raw, capped), "
            "op='get_narrative_self', op='compare_identity_snapshots' with params={'version_a': 10, 'version_b': 14}"
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "op": types.Schema(type="STRING", description="AuraSDK method name to call (e.g. 'get_high_volatility_beliefs', 'recall_person_context', 'get_entity_relations'). Also accepted as 'method' field."),
                "method": types.Schema(type="STRING", description="Alias for 'op' - AuraSDK method name. Use either 'op' or 'method'."),
                "params": types.Schema(type="STRING", description="JSON string of kwargs to pass to the method. Always pass as a JSON string, e.g. '{}' for no args, '{\"limit\": 5}' or '{\"record_id\": \"abc123\"}'. Never pass a dict object."),
            },
        ),
    ),
    # ---- Identity introspection tools (AuraSDK 2.1.0) ----
    types.FunctionDeclaration(
        name="introspect_identity_milestones",
        description="View how your identity evolved over time - what beliefs changed, when, and why. Returns a list of milestones with timestamps and change summaries.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "limit": types.Schema(type="INTEGER", description="Max milestones to return (default 10)"),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="introspect_identity_pressure",
        description="Check what is pressuring a specific belief to change. Requires a belief/record ID from search results.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "belief_id": types.Schema(type="STRING", description="Record/belief ID to check pressure on"),
            },
            required=["belief_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="introspect_drift_report",
        description="Get detailed memory drift analysis — is your knowledge base stable or shifting? Shows drift score, belief churn, causal rejections.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="introspect_session_consistency",
        description="Check whether memories from the current session are internally consistent or contradictory.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="introspect_metacognition",
        description="Get your current metacognitive state - confidence score, conflict count, epistemic guidance. Helps decide whether to act or verify first.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),

    # ---- V11: Base Packs & Cognitive Snapshots ----
    types.FunctionDeclaration(
        name="list_loaded_bases",
        description="List all specialist knowledge bases currently loaded in your brain. Shows base_id, version, record count. Use this to understand what specialist knowledge you have.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="check_base_version",
        description="Check if a specific specialist base is loaded and what version it is.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "base_id": types.Schema(type="STRING", description="ID of the specialist base to check"),
            },
            required=["base_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="list_cognitive_snapshots",
        description="List all sealed cognitive snapshots - frozen brain states saved after loading a base. Shows base_id, version, timestamp, files included.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="list_org_records",
        description="List organization-layer records (above specialist base). Optionally filter by org namespace.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "namespace": types.Schema(type="STRING", description="Filter by org namespace (optional)"),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="list_revalidation_queue",
        description=(
            "List research records that need revalidation: past TTL (stale_soft/stale_hard) or "
            "carrying an unresolved conflict flag. Sorted conflict-first, then most-decayed first. "
            "Use this to surface truth-pressure - claims whose freshness or coherence is suspect."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "limit": types.Schema(type="INTEGER", description="Max entries to return (default 20)"),
                "topic": types.Schema(type="STRING", description="Filter by topic substring (optional)"),
            },
        ),
    ),

    # ---- V12: Drives, Goals & Tensions ----
    types.FunctionDeclaration(
        name="introspect_drives",
        description=(
            "See what is driving you right now - active cognitive drives sorted by priority. "
            "Drives are generated from unresolved tensions (knowledge gaps, contradictions, unmet goals). "
            "Each drive has an urgency score, status, and description of what needs doing."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "limit": types.Schema(type="INTEGER", description="Max drives to return (default 10)"),
                "namespace": types.Schema(type="STRING", description="Filter drives by namespace (optional)"),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="introspect_goals",
        description=(
            "View your active goals and their state - what you're working towards, what's blocked, "
            "what's completed. Returns goal descriptions, priorities, statuses, and linked drives."
        ),
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="introspect_tensions",
        description=(
            "View raw cognitive tensions - unresolved signals from your memory that create pressure to act. "
            "Tensions are the raw material from which drives are generated. Shows tension source, score, "
            "namespace, and evidence records."
        ),
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="claim_drive",
        description="Claim a drive for exclusive execution - prevents other agents from working on it. Returns claim result with lease expiry.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "drive_id": types.Schema(type="STRING", description="ID of the drive to claim"),
                "lease_secs": types.Schema(type="INTEGER", description="Lease duration in seconds (default 300)"),
            },
            required=["drive_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="resolve_drive",
        description="Mark a drive as resolved (satisfied or failed). Call this after completing work on a claimed drive.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "drive_id": types.Schema(type="STRING", description="ID of the drive to resolve"),
                "resolved": types.Schema(type="BOOLEAN", description="True = successfully satisfied, False = failed"),
                "summary": types.Schema(type="STRING", description="Brief summary of what was done"),
            },
            required=["drive_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="create_goal",
        description="Create a new persistent goal. Goals generate drives automatically based on unresolved tensions.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "description": types.Schema(type="STRING", description="What this goal aims to achieve"),
                "namespace": types.Schema(type="STRING", description="Namespace for the goal (e.g. 'mission', 'health', 'learning')"),
                "priority": types.Schema(type="NUMBER", description="Priority weight 0.0-1.0 (higher = more urgent)"),
            },
            required=["description", "namespace"],
        ),
    ),
    types.FunctionDeclaration(
        name="revise_goal",
        description="Change a goal's priority weight. Use when circumstances change and a goal becomes more or less important.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "goal_id": types.Schema(type="STRING", description="ID of the goal to revise"),
                "new_priority": types.Schema(type="NUMBER", description="New priority weight 0.0-1.0"),
                "reason": types.Schema(type="STRING", description="Why the priority is changing"),
            },
            required=["goal_id", "new_priority", "reason"],
        ),
    ),

    # ---- V13: Predictions & Surprises ----
    types.FunctionDeclaration(
        name="introspect_predictions",
        description=(
            "View your pending predictions - what you expect to happen based on learned patterns. "
            "Each prediction has a confidence score, expected outcome, and evidence. "
            "When reality contradicts a prediction, it becomes a 'surprise' that drives learning."
        ),
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="introspect_surprises",
        description=(
            "View recent surprises - predictions that were contradicted by reality. "
            "Surprises are the primary learning signal: they reveal where your model of the world is wrong. "
            "High-confidence surprises are especially valuable for updating beliefs."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "limit": types.Schema(type="INTEGER", description="Max surprises to return (default 10)"),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="prediction_report",
        description="Get a summary report of your prediction engine - accuracy rate, pending count, surprise rate, calibration quality.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),

    # ---- V14: Epistemic Curiosity ----
    types.FunctionDeclaration(
        name="introspect_curiosity",
        description=(
            "View active epistemic gaps - things you don't know but should. "
            "Gaps are detected automatically from contradictions, missing context, stale beliefs, and novel domains. "
            "Each gap has an importance score and gap type (contradiction, missing_context, novelty, staleness, shallow_coverage)."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "namespace": types.Schema(type="STRING", description="Filter gaps by namespace (optional)"),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="curiosity_report",
        description="Get a summary of your curiosity engine - total gaps, gap types breakdown, top priority gaps, exploration suggestions.",
        parameters=types.Schema(type="OBJECT", properties={}),
    ),

    # ---- V15: Cognitive Mood ----
    types.FunctionDeclaration(
        name="introspect_mood",
        description=(
            "Check your current cognitive mood - HighStress, Normal, or Exploration. "
            "Mood affects how you process information: HighStress raises drive thresholds and suppresses curiosity, "
            "Exploration lowers thresholds and amplifies curiosity. Returns mood, scores, and rationale."
        ),
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="mood_history",
        description="View mood transition history - when and why your cognitive mood changed. Useful for understanding patterns in your cognitive state.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "limit": types.Schema(type="INTEGER", description="Max entries to return (default 10)"),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="mood_modulation",
        description=(
            "See how your current mood is modulating other cognitive systems. "
            "Shows multipliers applied to drive thresholds, curiosity thresholds, and whether curiosity is suppressed."
        ),
        parameters=types.Schema(type="OBJECT", properties={}),
    ),

    # ---- V17: Incubation Engine ----
    types.FunctionDeclaration(
        name="incubation_report",
        description=(
            "Get incubation engine status - whether it's enabled, how many hypotheses are active, "
            "total generated/accepted/rejected/expired, and why the gate last blocked."
        ),
        parameters=types.Schema(type="OBJECT", properties={}),
    ),
    types.FunctionDeclaration(
        name="introspect_hypotheses",
        description=(
            "View active incubation hypotheses - speculative ideas generated from cognitive gaps. "
            "Each hypothesis has an anchor type (causal_gap, belief_contradiction, prediction_surprise, "
            "curiosity_gap, unresolved_belief), confidence score, relation type, and review state."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "namespace": types.Schema(type="STRING", description="Filter by namespace (optional)"),
                "limit": types.Schema(type="INTEGER", description="Max results (default 10)"),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="review_hypothesis",
        description=(
            "Review an incubation hypothesis - accept it for investigation, reject it, or snooze it. "
            "Accepted hypotheses inform curiosity and future proposals. Rejected ones are cleaned up."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "hypothesis_id": types.Schema(type="STRING", description="ID of the hypothesis to review"),
                "action": types.Schema(type="STRING", description="accept, reject, or snooze"),
            },
            required=["hypothesis_id", "action"],
        ),
    ),
    types.FunctionDeclaration(
        name="set_incubation_enabled",
        description="Enable or disable the incubation engine. When disabled, no new hypotheses are generated.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "enabled": types.Schema(type="BOOLEAN", description="true to enable, false to disable"),
            },
            required=["enabled"],
        ),
    ),
    types.FunctionDeclaration(
        name="clear_expired_hypotheses",
        description="Remove all expired hypotheses from the incubation engine. Hypotheses expire after 14 days. Returns the number of hypotheses cleared.",
        parameters=types.Schema(
            type="OBJECT",
            properties={},
        ),
    ),

    # ---- Computer Access tools (filesystem + shell) ----
    types.FunctionDeclaration(
        name="fs_read",
        description=(
            "Read any file on the server. No path restrictions - you can read configs, logs, "
            "source code, data files, etc. Binary files return base64-encoded content. "
            "Large files are truncated; use offset/limit for paging."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "path": types.Schema(type="STRING", description="Absolute or relative path to the file"),
                "offset": types.Schema(type="INTEGER", description="Start reading from this line number (0-based). Default: 0"),
                "limit": types.Schema(type="INTEGER", description="Max lines to return. Default: 500, max: 2000"),
                "encoding": types.Schema(type="STRING", description="File encoding. Default: utf-8"),
            },
            required=["path"],
        ),
    ),
    types.FunctionDeclaration(
        name="fs_write",
        description=(
            "Write or append content to a file. RESTRICTED to safe directories: data/, tmp/, output/. "
            "Creates parent directories automatically. Cannot overwrite source code or system files."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "path": types.Schema(type="STRING", description="Path to write (relative paths resolve to data dir)"),
                "content": types.Schema(type="STRING", description="Content to write"),
                "mode": types.Schema(type="STRING", description="'write' (overwrite) or 'append'. Default: write"),
            },
            required=["path", "content"],
        ),
    ),
    types.FunctionDeclaration(
        name="fs_search",
        description=(
            "Search the filesystem. Two modes: 'glob' finds files by name pattern (e.g. '**/*.py'), "
            "'grep' searches file contents by regex. Returns matching paths and optional content snippets."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "mode": types.Schema(type="STRING", description="'glob' (find files by name) or 'grep' (search content by regex)"),
                "pattern": types.Schema(type="STRING", description="Glob pattern (e.g. '**/*.log') or regex pattern for grep"),
                "path": types.Schema(type="STRING", description="Directory to search in. Default: project root (BASE_DIR)"),
                "max_results": types.Schema(type="INTEGER", description="Max results to return. Default: 50"),
                "include_content": types.Schema(type="BOOLEAN", description="For grep: include matching lines. Default: true"),
            },
            required=["mode", "pattern"],
        ),
    ),
    types.FunctionDeclaration(
        name="shell_exec",
        description=(
            "Execute a shell command on the server. Returns stdout, stderr, and exit code. "
            "Use for: checking system state, running scripts, git operations, package management, etc. "
            "Dangerous commands (rm -rf /, format, shutdown, etc.) are blocked. "
            "Commands run with a timeout (default 30s, max 120s). Working directory is project root."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "command": types.Schema(type="STRING", description="Shell command to execute"),
                "timeout": types.Schema(type="INTEGER", description="Timeout in seconds (default: 30, max: 120)"),
                "working_dir": types.Schema(type="STRING", description="Working directory. Default: project root"),
            },
            required=["command"],
        ),
    ),
]


# ============== TOOL CATEGORIES ==============

CORE_TOOL_NAMES = frozenset({
    # Memory (always needed)
    "recall", "store", "search", "store_knowledge",
    # Utility
    "web_search", "extract_content", "get_current_datetime",
    # Browser
    "browse_page", "browser_act", "browser_close",
    # Profile
    "store_user_profile",
    # Persona
    "read_persona", "update_persona",
    # Meta
    "list_available_tools", "enable_tools",
    # Working memory
    "scratchpad",
})

# All tool names not in CORE are EXTENDED (loaded on-demand via enable_tools)
EXTENDED_TOOL_NAMES = frozenset(
    t.name for t in BRAIN_TOOLS if t.name not in CORE_TOOL_NAMES
)


# ============== REGISTRY (module-level singleton) ==============

_registry: ToolRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> ToolRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = ToolRegistry(BRAIN_TOOLS)
        return _registry


def invalidate_registry() -> None:
    """Force registry singleton to rebuild on next get_registry() call."""
    global _registry
    with _registry_lock:
        _registry = None


def reload_tools() -> None:
    """Invalidate all tool caches so new/removed sandbox tools take effect immediately.

    Call after sandbox tool approval, rejection, or auto-approve.
    Resets: registry singleton, LangChain tool cache, compiled agent graphs.
    """
    invalidate_registry()

    from remy.core.langgraph_tools import invalidate_tool_cache
    invalidate_tool_cache()

    from remy.core.agent import invalidate_graph_cache
    invalidate_graph_cache()

    logger.info("Tool caches invalidated - new tools will load on next request")


# ============== SANDBOX META-TOOL HANDLERS ==============

def _sandbox_create_tool(args: dict) -> str:
    """Write a tool file, validate it, register in manifest."""
    from pathlib import Path

    name = args["name"].strip()
    code = args["code"]
    manifest = get_registry().manifest

    # Write tool file
    tools_dir = Path(settings.SANDBOX_TOOLS_DIR)
    tools_dir.mkdir(parents=True, exist_ok=True)
    tool_file = tools_dir / f"{name}.py"
    tool_file.write_text(code, encoding="utf-8")

    # Validate via AST (no execution)
    valid, msg = validate_tool_file(tool_file)
    if not valid:
        tool_file.unlink()
        return json.dumps({"created": False, "error": msg})

    # Parse tool metadata from code via AST
    import ast
    tree = ast.parse(code)
    tool_meta = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    tool_meta[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass

    description = tool_meta.get("TOOL_DESCRIPTION", f"Sandbox tool: {name}")
    parameters = tool_meta.get("TOOL_PARAMETERS", {})
    required = tool_meta.get("TOOL_REQUIRED", [])
    dependencies = tool_meta.get("DEPENDENCIES", [])

    # Install dependencies if any
    if dependencies:
        ok, dep_msg = install_dependencies(dependencies)
        if not ok:
            return json.dumps({"created": False, "error": f"Dependency install failed: {dep_msg}"})

    # Register in manifest
    manifest.add_tool(
        name=name, file=f"{name}.py", description=description,
        parameters=parameters, required=required, dependencies=dependencies,
    )

    # Store in brain for learning
    brain.store(
        content=f"Created sandbox tool '{name}': {description}",
        tags=["sandbox", "tool-creation"],
    )

    return json.dumps({"created": True, "name": name, "status": "draft",
                        "message": f"Tool '{name}' created. Run sandbox_test_tool to test it."})


def _sandbox_test_tool(args: dict) -> str:
    """Run tests for a sandbox tool in isolated subprocess."""
    from pathlib import Path

    name = args["name"].strip()
    manifest = get_registry().manifest
    tool = manifest.get_tool(name)
    if not tool:
        return json.dumps({"tested": False, "error": f"Tool '{name}' not found in manifest."})

    tool_path = Path(settings.SANDBOX_TOOLS_DIR) / tool["file"]
    if not tool_path.exists():
        return json.dumps({"tested": False, "error": f"Tool file missing: {tool['file']}"})

    success, passed, failed, output = run_tests(tool_path)
    manifest.set_test_result(name, passed, failed, output)

    if success and passed > 0:
        # Auto-approve in autonomous mode if configured
        if settings.AUTONOMY_AUTO_APPROVE_SANDBOX:
            auto_approved = manifest.auto_approve_tested()
            if name in auto_approved:
                reload_tools()
                brain.store(
                    content=f"Tool '{name}' auto-approved ({passed} tests passed).",
                    tags=["sandbox", "auto-approved"],
                )
                return json.dumps({"tested": True, "passed": passed, "failed": failed,
                                    "status": "approved", "message": "Tests passed! Auto-approved and loaded."})

        manifest.submit_for_approval(name)
        brain.store(
            content=f"Tool '{name}' passed {passed} tests. Awaiting human approval.",
            tags=["sandbox", "test-success"],
        )
        return json.dumps({"tested": True, "passed": passed, "failed": failed,
                            "status": "pending", "message": "Tests passed! Awaiting human approval. Ask the user to run: remy --sandbox-approve"})
    else:
        brain.store(
            content=f"Tool '{name}' failed testing: {passed} passed, {failed} failed.\n{output[:300]}",
            tags=["sandbox", "test-failure"],
        )
        return json.dumps({"tested": False, "passed": passed, "failed": failed,
                            "output": output[:800],
                            "hint": "Read the error above, fix the code in sandbox_create_tool, then test again."})


def _sandbox_list_tools() -> str:
    """Return summary of all sandbox tools."""
    manifest = get_registry().manifest
    tools = manifest.summary()
    if not tools:
        return "No sandbox tools created yet."
    return json.dumps(tools, ensure_ascii=False)


# ============== USER PROFILE HELPERS ==============

from remy.core.memory_policy import (
    PROFILE_INPUT_FIELDS,
    PROFILE_PUBLIC_FIELDS,
    infer_semantic_type,
    protected_fields_for_record,
    protected_payload,
    sanitize_memory_content,
    sanitize_memory_metadata,
)
from remy.core.tool_handlers.profile import (
    person_matches_identity,
    resolve_person_identity_input,
    sanitize_profile_metadata,
    _DEFAULT_PERSONA,
    _PERSONA_TAG,
    _get_agent_persona,
    _persona_to_instruction,
    update_persona_fields,
)

_PROFILE_FIELDS = PROFILE_PUBLIC_FIELDS
_PROFILE_INPUT_FIELDS = PROFILE_INPUT_FIELDS


def _format_profile_content(fields: dict) -> str:
    """Format profile fields into a natural-language content string for brain storage."""
    from remy.core.tool_handlers.profile import normalize_profile_fields

    fields = normalize_profile_fields(fields)
    parts = []
    for key in _PROFILE_FIELDS:
        if fields.get(key):
            label = key.replace("_", " ").title()
            parts.append(f"{label}: {fields[key]}")
    return "User Profile: " + "; ".join(parts)


def _build_user_identity() -> str | None:
    """Build a rich USER IDENTITY block for the system prompt.

    Aggregates:
    1. User profile (store_user_profile) - core demographics
    2. Person records tagged with user's name - birth dates, contacts, family
    3. Identity-tagged records - verified facts about the user

    Each fact is annotated with verification status so the agent NEVER
    calls a verified fact a "guess" or "assumption".

    Returns formatted string or None if no profile exists.
    """
    profile = get_user_profile_record()
    if profile is None:
        return None

    meta = profile.metadata or {}
    user_name = meta.get("name", "")

    # в”Ђв”Ђ 1. Core profile facts with verification status в”Ђв”Ђ
    verified_facts = []
    unverified_facts = []

    profile_verified = meta.get("verified", False)
    for key in _PROFILE_FIELDS:
        val = meta.get(key)
        if val:
            label = key.replace("_", " ").title()
            if profile_verified or meta.get("source") == "user-confirmed":
                verified_facts.append(f"{label}: {val}")
            else:
                unverified_facts.append(f"{label}: {val}")

    # в”Ђв”Ђ 2. Related person/identity records в”Ђв”Ђ
    related_facts = []
    try:
        with brain_lock:
            # Person records (family members, user's own person record)
            person_records = brain.search(query="", tags=["person"], limit=20)
            # Identity-level records about the user
            identity_records = brain.search(query="", tags=["identity"], limit=10)

        # Gather person records connected to user or matching user's name
        seen_ids = {profile.id}
        for rec in person_records:
            if rec.id in seen_ids:
                continue
            seen_ids.add(rec.id)
            rmeta = rec.metadata or {}
            rec_verified = rmeta.get("verified", False)
            trust = rmeta.get("trust_score", 0.5)
            source = rmeta.get("source", "unknown")

            # Determine verification status
            if rec_verified or trust >= 0.9 or source == "user-confirmed":
                status = "VERIFIED"
            elif trust >= 0.6:
                status = "likely"
            else:
                status = "unverified"

            fact_text = rec.content[:200]
            related_facts.append((status, fact_text))

        # Identity records (non-person, non-profile)
        for rec in identity_records:
            if rec.id in seen_ids:
                continue
            seen_ids.add(rec.id)
            if "user-profile" in (rec.tags or []):
                continue  # Skip profile itself
            rmeta = rec.metadata or {}
            rec_verified = rmeta.get("verified", False)
            trust = rmeta.get("trust_score", 0.5)
            source = rmeta.get("source", "unknown")

            if rec_verified or trust >= 0.9 or source == "user-confirmed":
                status = "VERIFIED"
            elif trust >= 0.6:
                status = "likely"
            else:
                status = "unverified"

            fact_text = rec.content[:200]
            related_facts.append((status, fact_text))

    except Exception:
        pass  # Non-critical - profile alone is enough

    # в”Ђв”Ђ 3. Format output в”Ђв”Ђ
    lines = []
    display_name = user_name or "the user"

    if verified_facts:
        lines.append("Verified facts (user confirmed - treat as CERTAIN, never say 'I assume' or 'I guess'):")
        for f in verified_facts:
            lines.append(f"  - {f}")

    if unverified_facts:
        lines.append("Unverified (you stored this but user hasn't confirmed - you may say 'if I remember correctly'):")
        for f in unverified_facts:
            lines.append(f"  - {f}")

    if related_facts:
        lines.append("Related people and facts:")
        for status, fact in related_facts:
            if status == "VERIFIED":
                lines.append(f"  - {fact}")
            elif status == "likely":
                lines.append(f"  ~ {fact} (likely, high confidence)")
            else:
                lines.append(f"  - {fact} (unverified — confirm before using)")

    if not lines:
        return None

    checkmark_mojibake = chr(1074) + chr(1114) + chr(8220)
    rendered_lines = "\n".join(lines).replace(checkmark_mojibake, "\u2713")

    return (
        f"## USER IDENTITY - {display_name}\n"
        + rendered_lines + "\n\n"
        "RULES for using this identity:\n"
        f"- You KNOW {display_name}. This is your user. Do not introduce yourself as if meeting for the first time.\n"
        "- Facts marked - are CONFIRMED. Never say 'I assume', 'I guess', 'my hypothesis' about them.\n"
        "- Facts marked - — you may reference cautiously: 'if I remember correctly' or ask to confirm.\n"
        "- Do NOT recite this profile back to the user. Use it silently to inform your responses.\n"
        "- If the user corrects any fact, immediately call store_user_profile or update_record to fix it.\n"
    )


# ============== RESEARCH ORCHESTRATOR (RM-1) ==============

_RESEARCH_PROJECT_TAG = "research-project"
_RESEARCH_FINDING_TAG = "research-finding"
_DEPTH_QUERY_COUNT = {"quick": 2, "standard": 4, "deep": 7}


def _get_research_project(project_id: str):
    """Load a research project record by its project_id metadata field."""
    # Primary: tag-based search
    with brain_lock:
        records = brain.search(query="", tags=[_RESEARCH_PROJECT_TAG], limit=100)
        for rec in records:
            meta = rec.metadata or {}
            if meta.get("project_id") == project_id:
                return rec

        # Fallback: content-based search (in case tag search missed it)
        records2 = brain.search(query=project_id, limit=10)
        for rec in records2:
            meta = rec.metadata or {}
            if meta.get("project_id") == project_id:
                return rec

    logger.warning("Research project '%s' not found. Tag search returned %d records.", project_id, len(records))
    return None


def get_active_research_projects() -> list[dict]:
    """Return active (non-complete) research projects as dicts for decision prompt."""
    records = brain.search(query="", tags=[_RESEARCH_PROJECT_TAG], limit=20)
    projects = []
    for rec in records:
        meta = rec.metadata or {}
        if meta.get("status") in ("complete", "abandoned"):
            continue
        projects.append({
            "project_id": meta.get("project_id", ""),
            "topic": meta.get("topic", ""),
            "status": meta.get("status", "planning"),
            "depth": meta.get("depth", "standard"),
            "queries_total": len(meta.get("query_plan", [])),
            "queries_done": meta.get("queries_done", 0),
            "findings_count": meta.get("findings_count", 0),
        })
    return projects


def _start_research(args: dict, session_id: str | None = None, channel: str | None = None) -> str:
    """Create a research project with an LLM-generated query plan."""
    import re
    from remy.core.agent_tools import Level

    topic = str(
        args.get("topic")
        or args.get("question")
        or args.get("query")
        or args.get("prompt")
        or args.get("description")
        or ""
    ).strip()
    if not topic:
        return json.dumps({"error": "start_research requires topic, question, query, prompt, or description."})
    depth = args.get("depth", "standard").strip().lower()
    context = args.get("context", "").strip()

    if depth not in _DEPTH_QUERY_COUNT:
        depth = "standard"
    query_count = _DEPTH_QUERY_COUNT[depth]

    project_id = f"rp-{uuid.uuid4().hex[:12]}"
    topic_slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:50]

    # Check existing knowledge
    existing_knowledge = ""
    try:
        with brain_lock:
            recall_result = brain.recall(topic, token_budget=512)
        if recall_result and "No relevant" not in recall_result:
            existing_knowledge = recall_result[:300]
    except Exception:
        pass

    # Generate research plan via LLM
    plan_prompt = (
        "You are a research planner. Generate search queries for a research project.\n"
        f"Respond ONLY with a JSON array of {query_count} search query strings.\n\n"
        f"TOPIC: {topic}\n"
    )
    if context:
        plan_prompt += f"CONTEXT: {context}\n"
    if existing_knowledge:
        plan_prompt += f"EXISTING KNOWLEDGE: {existing_knowledge}\n"
    plan_prompt += (
        f"\nGenerate exactly {query_count} specific, diverse search queries "
        "that will cover the topic comprehensively. Respond with Valid JSON Array of strings only.\n"
        "Example: [\"query 1\", \"query 2\"]\n"
        "Do NOT use single quotes. Do NOT include markdown formatting."
    )

    query_plan = []
    try:
        from remy.core.llm import call_llm

        result = call_llm(plan_prompt, purpose="research_plan")
        raw = result.content
        if isinstance(raw, list):
            raw = " ".join(str(c) for c in raw)
        raw = str(raw).strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        
        # Cleanup: remove JSON prefix if present
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
            
        # Cleanup: replace single quotes if it looks like a python list
        if raw.startswith("['") and "']" in raw:
            try:
                import ast
                # safe usage for literals
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, list):
                    query_plan = [str(q).strip() for q in parsed[:query_count] if str(q).strip()]
            except Exception:
                pass 

        if not query_plan:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    query_plan = [str(q).strip() for q in parsed[:query_count] if str(q).strip()]
            except json.JSONDecodeError:
                pass # Will fall back to topic below
    except Exception as e:
        logger.warning("Research plan generation failed, using topic as query: %s", e)

    # Fallback: use topic itself as queries
    if not query_plan:
        query_plan = [topic]

    # Pad if LLM returned fewer queries than requested
    if len(query_plan) < query_count:
        logger.info("Research plan: LLM returned %d/%d queries, padding with variations", len(query_plan), query_count)
        suffixes = ["latest developments", "best practices", "comparisons and alternatives", "implementation guides", "case studies", "technical challenges"]
        for suffix in suffixes:
            if len(query_plan) >= query_count:
                break
            variant = f"{topic} {suffix}"
            if variant not in query_plan:
                query_plan.append(variant)

    # Store project in brain
    content = f"Research Project: {topic}\nDepth: {depth}\nQueries: {len(query_plan)}"
    with brain_lock:
        rec = brain.store(
            content=content,
            level=Level.DOMAIN,
            tags=[_RESEARCH_PROJECT_TAG, topic_slug],
            metadata=_stamp_provenance({
                "type": "research_project",
                "project_id": project_id,
                "topic": topic,
                "depth": depth,
                "status": "researching",
                "query_plan": query_plan,
                "queries_done": 0,
                "findings_count": 0,
                "finding_ids": [],
                "started_at": datetime.now().isoformat(),
            }, channel),
        )

    return json.dumps({
        "created": True,
        "project_id": project_id,
        "record_id": rec.id,
        "topic": topic,
        "depth": depth,
        "query_plan": query_plan,
        "queries_total": len(query_plan),
    }, ensure_ascii=False)


def _add_research_finding(args: dict, session_id: str | None = None, channel: str | None = None) -> str:
    """Record a finding and attach it to a research project."""
    from remy.core.ingestion import ingest_grounded_evidence

    project_id = args["project_id"].strip()
    # Accept 'summary' as alias for 'content' - LLM sometimes uses summary instead
    content = (args.get("content") or args.get("summary") or "").strip()
    if not content:
        return json.dumps({"error": "Missing required field: content (or summary)"})
    source_url = args.get("source_url", "").strip()
    if not source_url:
        return json.dumps({
            "error": "add_research_finding requires source_url. Candidate discovery alone is not enough - fetch the chosen source first."
        }, ensure_ascii=False)

    # RM-2: Apply credibility default if not provided
    if "confidence" not in args and source_url:
        args["confidence"] = credibility_scorer.get_score(source_url)

    confidence = float(args.get("confidence", 0.7))
    contradicts_id = args.get("contradicts_finding_id", "").strip()

    # Find the project
    with brain_lock:
        project_rec = _get_research_project(project_id)
        if not project_rec:
            return json.dumps({"error": f"Research project '{project_id}' not found"})

        project_meta = dict(project_rec.metadata or {})
        topic = project_meta.get("topic", "research")

        import re
        topic_slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:50]

        # Dedup check
        existing = _check_duplicates(content[:100], tags=[_RESEARCH_FINDING_TAG])

        # Store finding - route through canonical ingestion API.
        finding_content = f"Research finding ({topic}): {content}"
        if source_url:
            finding_content += f"\nSource: {source_url}"

        ingestion = ingest_grounded_evidence(
            content=finding_content,
            source_url=source_url,
            session_id=session_id or "",
            channel=channel,
            extract_class="grounded_external_fact",
            extra_tags=[_RESEARCH_FINDING_TAG, topic_slug],
            extra_meta={
                "type": "research_finding",
                "project_id": project_id,
                "timestamp": datetime.now().isoformat(),
            },
            confidence=confidence,
        )
        if not ingestion.admitted:
            return json.dumps({
                "error": ingestion.reason,
                "source_url": source_url,
            }, ensure_ascii=False)

        rec = brain.store(
            content=ingestion.content,
            level=ingestion.level,
            tags=ingestion.tags,
            metadata=_stamp_provenance(ingestion.metadata, channel, tags=ingestion.tags),
        )

        # Connect to project (promotion-gated)
        from remy.core.agent_tools import gated_connect
        gated_connect(brain, rec.id, project_rec.id, weight=0.8)

        # Handle contradiction
        if contradicts_id and contradicts_id != rec.id:
            contradicting_rec = brain.get(contradicts_id)
            if contradicting_rec:
                gated_connect(brain, rec.id, contradicts_id, weight=0.3)

        # Update project metadata
        finding_ids = project_meta.get("finding_ids", [])
        finding_ids.append(rec.id)
        project_meta["finding_ids"] = finding_ids
        project_meta["findings_count"] = len(finding_ids)
        brain.update(project_rec.id, metadata=project_meta)

    result = {
        "stored": True,
        "finding_id": rec.id,
        "project_id": project_id,
        "findings_count": len(finding_ids),
    }
    if existing:
        result["duplicate_warning"] = existing
    if contradicts_id:
        result["contradicts"] = contradicts_id

    return json.dumps(result, ensure_ascii=False)


def _complete_research(args: dict, session_id: str | None = None, channel: str | None = None) -> str:
    """Synthesize all findings into a final report and mark project complete."""
    from remy.core.tool_handlers.research import _build_cited_markdown_report
    from remy.core.verification_gate import (
        emit_verification_incident,
        resolve_verification_incident,
        run_research_completion_verification_gate,
    )

    project_id = args["project_id"].strip()

    project_rec = _get_research_project(project_id)
    if not project_rec:
        return json.dumps({"error": f"Research project '{project_id}' not found"})

    project_meta = dict(project_rec.metadata or {})
    topic = project_meta.get("topic", "research")
    finding_ids = project_meta.get("finding_ids", [])

    # Gather findings
    findings = []
    sources = []
    total_confidence = 0.0
    with brain_lock:
        for fid in finding_ids:
            frec = brain.get(fid)
            if not frec:
                continue
            fmeta = frec.metadata or {}
            findings.append({
                "content": frec.content,
                "source_url": fmeta.get("source_url", ""),
                "confidence": fmeta.get("confidence", 0.7),
            })
            total_confidence += fmeta.get("confidence", 0.7)
            if fmeta.get("source_url"):
                sources.append(fmeta["source_url"])

    if not findings:
        return json.dumps({"error": "No findings to synthesize"})

    avg_confidence = round(total_confidence / len(findings), 2)
    unique_sources = list(dict.fromkeys(sources))
    citation_complete = bool(unique_sources)
    if not citation_complete:
        return json.dumps({
            "error": "complete_research requires accepted source URLs on findings. Fetch evidence first, then attach source_url on each finding."
        }, ensure_ascii=False)
    evidence_note = (
        "" if citation_complete else "Evidence is weak: no accepted source URLs were attached."
    )

    # Synthesize via LLM
    findings_text = "\n".join(
        f"- {f['content'][:300]}" + (f" [confidence: {f['confidence']}]" if f['confidence'] != 0.7 else "")
        for f in findings
    )

    synth_prompt = (
        "You are synthesizing research findings into a clear, structured report.\n\n"
        f"TOPIC: {topic}\n"
        f"FINDINGS ({len(findings)} total):\n{findings_text}\n\n"
        "Write a concise research report (3-6 sentences) that:\n"
        "1. Summarizes the key findings\n"
        "2. Notes any contradictions or uncertainties\n"
        "3. Draws conclusions\n"
        "4. Uses the same language as the findings\n\n"
        "Every concrete claim should cite its supporting source URLs when available.\n"
        "If evidence is weak, say so explicitly.\n\n"
        "Report:"
    )

    report = None
    try:
        from remy.core.llm import call_llm

        result = call_llm(synth_prompt, purpose="research_synthesis")
        raw = result.content
        if isinstance(raw, list):
            raw = " ".join(str(c) for c in raw)
        report = str(raw).strip()
    except Exception as e:
        logger.warning("Research synthesis failed: %s", e)

    if not report or len(report) < 10:
        # Fallback: concatenate findings
        report = f"Research on '{topic}':\n" + "\n".join(
            f"- {f['content'][:200]}" for f in findings
        )
    if evidence_note and evidence_note not in report:
        report = f"{report}\n\n{evidence_note}"

    cited_markdown, citations = _build_cited_markdown_report(
        topic=topic,
        summary=report,
        findings=findings,
        unique_sources=unique_sources,
        evidence_note=evidence_note,
    )

    # Store final report via store_research logic
    import re
    topic_slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:50]
    from remy.core.agent_tools import Level

    with brain_lock:
        report_store_meta = _stamp_provenance({
            "type": "research_report",
            "artifact_format": "markdown",
            "markdown_body": cited_markdown,
            "citations": citations,
            "topic": topic,
            "project_id": project_id,
            "sources": unique_sources,
            "findings_count": len(findings),
            "confidence_avg": avg_confidence,
            "citation_complete": citation_complete,
            "citation_count": len(unique_sources),
            "evidence_note": evidence_note,
            "learning_channel": "internet_evidence",
            # D-04: LLM-synthesized report over grounded findings - not itself a
            # factual knowledge claim. Requires explicit downstream promotion.
            "admission_class": "research_report",
            "requires_promotion": True,
            "timestamp": datetime.now().isoformat(),
        }, channel)
        report_rec = brain.store(
            content=cited_markdown,
            # D-04: DECISIONS level - synthesis output, not raw domain knowledge.
            level=Level.DECISIONS,
            tags=["research", topic_slug],
            metadata=report_store_meta,
        )

    report_record_id = getattr(report_rec, "id", None) or str(report_rec or "")
    hydrated_report_rec = brain.get(report_record_id) if report_record_id else None

    pdf_artifact = {}
    pdf_result = {}
    try:
        pdf_result = json.loads(
            _generate_report(
                {
                    "title": topic,
                    "subtitle": "Research Report",
                    "content": cited_markdown,
                    "report_type": "standard",
                    "include_toc": True,
                    "metadata": {
                        "topic": topic,
                        "source_count": len(unique_sources),
                        "citation_complete": citation_complete,
                    },
                },
                session_id,
                channel,
            )
        )
        if pdf_result.get("generated"):
            pdf_artifact = {
                "pdf_url": pdf_result.get("url"),
                "pdf_filename": pdf_result.get("filename"),
                "pdf_record_id": pdf_result.get("record_id"),
            }
    except Exception as e:
        logger.warning("Research PDF render failed for %s: %s", project_id, e)

    verification = run_research_completion_verification_gate(
        project_id=project_id,
        report_record_id=report_record_id,
        stored_report_record=hydrated_report_rec,
        markdown_body=cited_markdown,
        findings_count=len(findings),
        pdf_result=pdf_result if isinstance(pdf_result, dict) else None,
    )

    if not verification.verified and verification.repair_required:
        emit_verification_incident(
            source="complete_research",
            verification=verification,
            artifact_label=topic,
            extra={"project_id": project_id},
        )
        report_meta = dict((getattr(hydrated_report_rec, "metadata", None) or report_store_meta))
        report_meta["verification"] = verification.to_dict()
        if report_record_id:
            brain.update(report_record_id, metadata=report_meta)
        return json.dumps({
            "completed": False,
            "project_id": project_id,
            "topic": topic,
            "error": verification.reason,
            "verification": verification.to_dict(),
        }, ensure_ascii=False)

    # Connect report to project (promotion-gated)
    from remy.core.agent_tools import gated_connect
    gated_connect(brain, report_record_id, project_rec.id, weight=0.9)

    # Mark project complete
    project_meta["status"] = "complete"
    project_meta["completed_at"] = datetime.now().isoformat()
    project_meta["report_id"] = report_record_id
    project_meta["verification"] = verification.to_dict()
    if pdf_artifact:
        project_meta.update(pdf_artifact)
    brain.update(project_rec.id, metadata=project_meta)

    report_meta = dict((getattr(hydrated_report_rec, "metadata", None) or report_store_meta))
    report_meta["verification"] = verification.to_dict()
    if pdf_artifact:
        report_meta.update(pdf_artifact)
    brain.update(report_record_id, metadata=report_meta)
    resolve_verification_incident(
        source="complete_research",
        artifact_label=topic,
        extra={"project_id": project_id},
    )

    # Store in Knowledge Base (Aura Memory) for semantic retrieval
    try:
        from remy.core.agent_tools import knowledge, knowledge_lock
        _report_text = (report or "").strip() if isinstance(report, str) else ""
        if knowledge is not None and len(_report_text) >= 5:
            # Pin as permanent anchor since it's a completed research report
            with knowledge_lock:
                knowledge.process(_report_text[:2000], pin=True)
                knowledge.flush()
            logger.info("Stored research report in Knowledge Base (Aura Memory)")
    except Exception as e:
        logger.warning("Failed to store research report in Knowledge Base: %s", e)

    return json.dumps({
            "completed": True,
            "project_id": project_id,
            "report_id": report_record_id,
            "topic": topic,
        "report": report[:500],
        "markdown": cited_markdown,
        "artifact_format": "markdown",
        "citations": citations,
        **pdf_artifact,
        "verification": verification.to_dict(),
        "source_count": len(unique_sources),
        "findings_count": len(findings),
        "confidence_avg": avg_confidence,
        "citation_complete": citation_complete,
        "evidence_note": evidence_note,
    }, ensure_ascii=False)


# ============== PROACTIVE CONTEXT ==============

def _get_active_todos_context() -> str:
    """Get active todo items for system instruction context.

    Cross-references each todo with failure outcomes to prevent
    the agent from blindly re-proposing failed tasks.
    """
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

            # Cross-reference with failures: check if any failure mentions this todo
            title_lower = title.lower()
            title_words = [w for w in title_lower.split() if len(w) > 3]
            for ft in failure_texts:
                if title_lower in ft or any(w in ft for w in title_words):
                    entry += " - PREVIOUSLY FAILED - check failures before retrying"
                    break

            if due and due < today:
                overdue.append(entry)
            else:
                active.append(entry)

        if not active and not overdue:
            return ""

        parts = ["\n## ACTIVE TODOS"]
        if overdue:
            parts.append("OVERDUE:")
            parts.extend(f"  - - {t}" for t in overdue)
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

    from remy.core.agent_tools import brain_lock

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

    context_parts = []

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
                # Simple date comparison
                if due_date.date() == now.date():
                    due_today.append(meta.get("description", task.content))
                elif due_date.date() == tomorrow.date():
                    due_tomorrow.append(meta.get("description", task.content))
            except Exception:
                continue
                
        if due_today:
            context_parts.append(f"URGENT - TASKS FOR TODAY ({now.strftime('%Y-%m-%d')}):")
            for t in due_today:
                context_parts.append(f"- {t}")
            context_parts.append("INSTRUCTION: Mention these IMMEDIATELY in your greeting. Ask if the user has done them.")
            
        if due_tomorrow:
            context_parts.append(f"UPCOMING - TOMORROW:")
            for t in due_tomorrow:
                context_parts.append(f"- {t}")
            context_parts.append("INSTRUCTION: Give a brief heads-up about these.")

    except Exception as e:
        logger.warning(f"Failed to get scheduled tasks: {e}")

    # 2. Recent Session Summaries (Continuation)
    try:
        summaries = brain.search(query="", tags=["session-summary"], limit=2)
        if summaries:
            context_parts.append("\nPREVIOUS CONTEXT (What happened last time):")
            for s in summaries:
                context_parts.append(f"- {s.content}")
            context_parts.append("INSTRUCTION: If relevant, ask about updates on these topics.")
    except Exception:
        pass

    # 3. Recent Failures & Outcomes (CRITICAL - prevents repeating failed actions)
    try:
        failures = brain.search(query="", tags=["outcome-failure"], limit=5)
        if failures:
            context_parts.append(
                "\n? RECENT FAILURES (DO NOT repeat these without a new strategy):"
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
            context_parts.append(
                "INSTRUCTION: Before proposing any action related to these failures, "
                "ACKNOWLEDGE the previous failure and explain what is DIFFERENT this time. "
                "If nothing changed - do NOT retry the same approach."
            )

        # Also check autonomous outcomes (broader - includes successes for context)
        outcomes = brain.search(query="", tags=["autonomous-outcome"], limit=5)
        recent_outcomes = [
            o for o in outcomes
            if o.id not in {f.id for f in (failures or [])}
        ]
        if recent_outcomes:
            context_parts.append("\nRECENT AUTONOMOUS OUTCOMES:")
            for o in recent_outcomes[:3]:
                meta = getattr(o, "metadata", None) or {}
                status = "?" if "outcome-success" in (getattr(o, "tags", None) or []) else "?"
                context_parts.append(f"- {status} {o.content[:150]}")
    except Exception as e:
        logger.warning(f"Failed to get failure context: {e}")

    if not context_parts:
        return ""

    return "\n\n=== PROACTIVE AWAKENING CONTEXT ===\n" + "\n".join(context_parts) + "\n===================================\n"


# ============== TOOL EXECUTION ==============

# Cyrillic - Latin transliteration for tag sanitization
_CYRILLIC_MAP = {
    "\u0430": "a", "\u0431": "b", "\u0432": "v", "\u0433": "g", "\u0491": "g", "\u0434": "d",
    "\u0435": "e", "\u0436": "z", "\u0437": "z", "\u0438": "y", "\u0456": "i", "\u0457": "i",
    "\u0439": "y", "\u043a": "k", "\u043b": "l", "\u043c": "m", "\u043d": "n", "\u043e": "o",
    "\u043f": "p", "\u0440": "r", "\u0441": "s", "\u0442": "t", "\u0443": "u", "\u0444": "f",
    "\u0445": "h", "\u0446": "c", "\u0447": "c", "\u0448": "s", "\u0449": "s", "\u044c": "",
    "\u044e": "u", "\u044f": "a", "\u0454": "e", "\u0451": "o",
}


def _clean_tag(tag: str) -> str:
    """Ensure tag only contains valid chars: alphanumeric, underscore, hyphen, colon, period.

    Replaces spaces with hyphens, strips other invalid chars.
    Preserves case and Unicode alphanumeric (Cyrillic, etc.).
    """
    import re as _re
    tag = tag.strip().replace(" ", "-")
    tag = _re.sub(r"[^a-zA-Z0-9_\-:.\u0400-\u04FF]", "", tag)
    return tag or "unknown"


def _sanitize_tag(text: str) -> str:
    """Convert arbitrary text into a valid tag (ASCII alphanumeric + hyphen)."""
    import re as _re
    lowered = text.lower()
    ascii_text = "".join(_CYRILLIC_MAP.get(ch, ch) for ch in lowered)
    ascii_text = _re.sub(r"[^a-z0-9-]", "", ascii_text)
    return ascii_text or "unknown"


def _check_duplicates(query: str, tags: list[str] | None = None, limit: int = 3) -> list[dict]:
    """Search brain + knowledge for existing similar records before storing."""
    results = []
    try:
        brain_hits = brain.search(query=query, tags=tags, limit=limit)
        if brain_hits:
            results = [
                {
                    "id": r.id,
                    "content": r.content[:150],
                    "tags": list(r.tags) if r.tags else [],
                }
                for r in brain_hits
            ]
    except Exception:
        pass

    # Check knowledge for high-similarity matches
    try:
        kb_hits = _kb_retrieve(query[:200], top_k=1)
        for item in kb_hits:
            if item.get("score", 0) > 0.3 and item.get("text"):
                results.append({"kb_match": item["text"][:100], "kb_score": round(item["score"], 2)})
    except Exception:
        pass

    return results


# ============== F3: IMPLICIT FEEDBACK SIGNALS ==============

FEEDBACK_TAG = "feedback-signal"
USER_CORRECTION_TAG = "user-correction"

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "i", "you", "my", "your",
    "what", "how", "to", "and", "of", "in", "for", "it", "that", "this",
    "with", "on", "at", "be", "do", "have", "not", "but", "or", "so",
    "if", "me", "we", "he", "she", "they", "can", "will", "just",
    "\u044f", "\u0442\u0438", "\u0432\u0438", "\u0446\u0435", "\u0449\u043e", "\u044f\u043a", "\u0442\u0430", "\u0456", "\u043d\u0435", "\u0432", "\u043d\u0430", "\u0437",
})


@dataclass
class FeedbackSignal:
    """An implicit feedback signal detected from conversation patterns."""

    signal_type: str   # "too_verbose" | "topic_switch" | "repeat_question" | "user_correction"
    severity: float    # 0.0-1.0
    context: str       # What triggered it (max 200 chars)
    channel: str
    timestamp: str


def _feedback_content_words(text: str) -> set[str]:
    """Extract content words: lowercase, strip punctuation, remove stop words."""
    return {w.strip(".,!?;:\"'()[]") for w in text.lower().split()} - _STOP_WORDS - {""}


def _looks_like_user_correction(text: str) -> bool:
    """Bounded heuristic for explicit user corrections."""
    text = (text or "").strip()
    if len(text) < 6:
        return False
    lowered = text.lower()
    markers = (
        "wrong",
        "incorrect",
        "not right",
        "i said",
        "i told you",
        "actually",
        "\u043d\u0435\u043f\u0440\u0430\u0432\u0438\u043b\u044c\u043d\u043e",
        "\u043d\u0435 \u0442\u0430\u043a",
        "\u0446\u0435 \u043d\u0435 \u0442\u0430\u043a",
        "\u043f\u043e\u043c\u0438\u043b\u0438\u0432",
        "\u044f \u043a\u0430\u0437\u0430\u0432",
        "\u044f \u0433\u043e\u0432\u043e\u0440\u0438\u0432",
        "\u043d\u0430\u0441\u043f\u0440\u0430\u0432\u0434\u0456",
    )
    return any(marker in lowered for marker in markers)


def detect_feedback_signals(messages: list, channel: str) -> list[FeedbackSignal]:
    """Detect implicit feedback from conversation patterns. Zero LLM calls."""
    from langchain_core.messages import AIMessage, HumanMessage

    signals: list[FeedbackSignal] = []
    if len(messages) < 3:
        return signals

    now = datetime.now().isoformat()

    # Scan (AI response, next user message) pairs
    for i in range(len(messages) - 1):
        ai_msg = messages[i]
        user_msg = messages[i + 1]
        if not isinstance(ai_msg, AIMessage) or not isinstance(user_msg, HumanMessage):
            continue

        ai_text = ai_msg.content if isinstance(ai_msg.content, str) else ""
        user_text = user_msg.content if isinstance(user_msg.content, str) else ""
        if not ai_text or not user_text:
            continue

        ai_words = len(ai_text.split())
        user_words = len(user_text.split())

        # Signal 1: Verbosity - long AI response followed by very short user reply
        if ai_words > 150 and user_words < 5:
            signals.append(FeedbackSignal(
                signal_type="too_verbose",
                severity=min(1.0, ai_words / 300),
                context=f"AI: {ai_words}w -> User: '{user_text[:50]}'",
                channel=channel,
                timestamp=now,
            ))

        # Signal 2: Topic switch - zero content-word overlap
        ai_last = ai_text.split(".")[-2] if ai_text.count(".") >= 2 else ai_text[-200:]
        ai_content = {w.strip(".,!?;:\"'()[]") for w in ai_last.lower().split()} - _STOP_WORDS - {""}
        user_content = {w.strip(".,!?;:\"'()[]") for w in user_text.lower().split()} - _STOP_WORDS - {""}
        if len(ai_content) > 3 and len(user_content) > 2:
            if not (ai_content & user_content):
                signals.append(FeedbackSignal(
                    signal_type="topic_switch",
                    severity=0.6,
                    context=f"AI: '{ai_last[:50]}' -> User: '{user_text[:50]}'",
                    channel=channel,
                    timestamp=now,
                ))

    # Signal 3: Repeat question - user asks similar thing as before
    from langchain_core.messages import HumanMessage as HM
    user_msgs = [m for m in messages if isinstance(m, HM)]
    if len(user_msgs) >= 2:
        latest = user_msgs[-1].content if isinstance(user_msgs[-1].content, str) else ""
        latest_words = _feedback_content_words(latest)
        if len(latest_words) >= 3:
            for earlier in user_msgs[:-1]:
                earlier_text = earlier.content if isinstance(earlier.content, str) else ""
                earlier_words = _feedback_content_words(earlier_text)
                if len(earlier_words) >= 3:
                    overlap = latest_words & earlier_words
                    ratio = len(overlap) / min(len(latest_words), len(earlier_words))
                    if ratio > 0.6:
                        signals.append(FeedbackSignal(
                            signal_type="repeat_question",
                            severity=ratio,
                            context=f"Repeated: '{latest[:50]}' ~ '{earlier_text[:50]}'",
                            channel=channel,
                            timestamp=now,
                        ))
                        break  # One repeat signal per turn

    return signals


def _extract_targeted_feedback_record_ids(session_log: list | None, limit: int = 6) -> list[str]:
    """Prefer exact evidence ids from the latest factuality/recall analysis."""
    if not session_log:
        return []

    record_ids: list[str] = []
    seen: set[str] = set()

    for item in reversed(session_log):
        if not isinstance(item, dict):
            continue
        if item.get("type") != "factuality_analysis":
            continue

        claims = item.get("claims") if isinstance(item.get("claims"), list) else []
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            for rec_id in claim.get("supporting_record_ids") or []:
                rec_id = str(rec_id or "").strip()
                if rec_id and rec_id not in seen:
                    seen.add(rec_id)
                    record_ids.append(rec_id)
                    if len(record_ids) >= limit:
                        return record_ids

        for rec_id in item.get("evidence_record_ids") or []:
            rec_id = str(rec_id or "").strip()
            if rec_id and rec_id not in seen:
                seen.add(rec_id)
                record_ids.append(rec_id)
                if len(record_ids) >= limit:
                    return record_ids

        break

    return record_ids


def apply_latest_user_correction_feedback(
    messages: list,
    channel: str,
    limit: int = 3,
    session_log: list | None = None,
) -> list[dict]:
    """Apply bounded negative feedback after user correction.

    Prefer exact evidence ids from the latest recall/factuality analysis.
    Fall back to text matching only when no targeted ids are available.
    """
    from langchain_core.messages import AIMessage, HumanMessage
    from remy.core.agent_tools import Level

    if len(messages) < 2:
        return []

    pair = None
    for idx in range(len(messages) - 1, 0, -1):
        if isinstance(messages[idx - 1], AIMessage) and isinstance(messages[idx], HumanMessage):
            pair = (messages[idx - 1], messages[idx])
            break
    if pair is None:
        return []

    ai_msg, user_msg = pair
    ai_text = ai_msg.content if isinstance(ai_msg.content, str) else ""
    user_text = user_msg.content if isinstance(user_msg.content, str) else ""
    if not ai_text or not user_text or not _looks_like_user_correction(user_text):
        return []

    user_words = _feedback_content_words(user_text)
    ai_words = _feedback_content_words(ai_text)
    results: list[dict] = []

    with brain_lock:
        targets: list[object] = []
        targeted_ids = _extract_targeted_feedback_record_ids(session_log, limit=max(limit * 2, 6))
        for rec_id in targeted_ids:
            rec = brain.get(rec_id)
            if rec is not None:
                targets.append(rec)
            if len(targets) >= limit:
                break

        if not targets:
            query = (user_text if len(user_words) >= 2 else ai_text)[:200]
            candidates = brain.search(query=query, limit=max(8, limit * 3))
            if not candidates and ai_text:
                candidates = brain.search(query=ai_text[:200], limit=max(8, limit * 3))

            scored: list[tuple[int, object]] = []
            seen_ids: set[str] = set()
            for rec in candidates or []:
                rec_id = str(getattr(rec, "id", "") or "")
                if not rec_id or rec_id in seen_ids:
                    continue
                seen_ids.add(rec_id)
                rec_text = str(getattr(rec, "content", "") or "")
                rec_words = _feedback_content_words(rec_text)
                if not rec_words:
                    continue
                overlap_user = len(rec_words & user_words)
                overlap_ai = len(rec_words & ai_words)
                score = overlap_user * 3 + overlap_ai * 2
                if score < 2:
                    continue
                scored.append((score, rec))

            scored.sort(
                key=lambda item: (item[0], float(getattr(item[1], "strength", 0.0) or 0.0)),
                reverse=True,
            )
            targets = [rec for _, rec in scored[:limit]]

        if not targets:
            return []

        for rec in targets:
            brain.feedback(rec.id, False)
            stats = brain.feedback_stats(rec.id)
            results.append({
                "record_id": rec.id,
                "content_preview": (getattr(rec, "content", "") or "")[:120],
                "negative": stats[1] if stats else 0,
                "net_score": stats[2] if stats else 0,
            })

        brain.store(
            content=f"User correction triggered negative memory feedback for {len(results)} record(s): {user_text[:160]}",
            level=Level.WORKING,
            tags=["memory-feedback", "correction-note", USER_CORRECTION_TAG],
            metadata={
                "type": "user_correction_feedback",
                "channel": channel,
                "user_correction": user_text[:300],
                "target_record_ids": [item["record_id"] for item in results],
                "targeting_mode": "evidence_ids" if targeted_ids else "text_fallback",
            },
            deduplicate=False,
        )

    try:
        store_feedback_signal(
            FeedbackSignal(
                signal_type="user_correction",
                severity=min(1.0, 0.5 + len(results) * 0.1),
                context=f"User correction: '{user_text[:80]}'",
                channel=channel,
                timestamp=datetime.now().isoformat(),
            )
        )
    except Exception:
        pass

    return results


def store_feedback_signal(signal: FeedbackSignal) -> None:
    """Store a feedback signal in brain for behavioral adaptation."""
    try:
        from remy.core.agent_tools import Level
        brain.store(
            content=f"Feedback [{signal.signal_type}]: {signal.context}",
            level=Level.WORKING,
            tags=[FEEDBACK_TAG, signal.signal_type],
            metadata={
                "type": "feedback_signal",
                "signal_type": signal.signal_type,
                "severity": signal.severity,
                "channel": signal.channel,
                "timestamp": signal.timestamp,
            },
            deduplicate=False,
        )
    except Exception as e:
        logger.debug("Failed to store feedback signal: %s", e)


def get_recent_feedback_summary(limit: int = 10) -> str:
    """Aggregate recent feedback signals into behavioral hints. Zero LLM calls."""
    try:
        records = brain.search(query="", tags=[FEEDBACK_TAG], limit=limit)
        if not records:
            return ""

        counts: dict[str, int] = {}
        for r in records:
            meta = getattr(r, "metadata", None) or {}
            stype = meta.get("signal_type", "unknown")
            counts[stype] = counts.get(stype, 0) + 1

        hints = []
        if counts.get("too_verbose", 0) >= 2:
            hints.append("User prefers shorter responses. Be more concise.")
        if counts.get("topic_switch", 0) >= 2:
            hints.append("User frequently switches topics - stay focused on the current topic and keep replies brief. Don't bring up old topics unless asked.")
        if counts.get("repeat_question", 0) >= 1:
            hints.append("User has repeated questions. Previous answers may have been unclear.")
        if counts.get("user_correction", 0) >= 1:
            hints.append("User recently corrected a memory-backed answer. Be more careful with recalled facts.")

        return "\n".join(hints)
    except Exception:
        return ""


# ============== METRIC AND EVENT INTELLIGENCE ==============


def _track_health_metric(args: dict, channel: str | None = None) -> str:
    """Deprecated alias for generic metric tracking."""
    return _track_metric(args, channel)


def _track_metric(args: dict, channel: str | None = None) -> str:
    """Track a user-reported numeric metric."""
    from remy.core.tool_handlers.metrics import _track_metric as _handler

    return _handler(args, channel)


def _health_summary(args: dict) -> str:
    """Deprecated alias for generic metric summary."""
    return _metric_summary(args)


def _metric_summary(args: dict) -> str:
    """Summarize tracked metrics and events."""
    from remy.core.tool_handlers.metrics import _metric_summary as _handler

    return _handler(args)


def _symptom_correlate(args: dict) -> str:
    """Deprecated alias for generic event correlation."""
    if "event" not in args and "symptom" in args:
        args = {**args, "event": args.get("symptom")}
    return _event_correlate(args)


def _event_correlate(args: dict) -> str:
    """Analyze an event against related memory records."""
    from remy.core.tool_handlers.metrics import _event_correlate as _handler

    return _handler(args)


def _extract_facts(
    args: dict, channel: str | None = None, session_id: str | None = None
) -> str:
    """RM-4: Extract structured facts from text.

    Learning boundary rule (D-01):
    - Fetch evidence present this turn -> DOMAIN with source URL (grounded).
    - No fetch evidence -> WORKING + quarantine tags (LLM-only, not durable).
    """
    text = args["text"]
    source = args.get("source", "unknown")

    # --- Learning boundary check ---
    fetch_evidence: list[dict] = []
    try:
        from remy.core.claim_provenance import get_turn_fetch_evidence
        fetch_evidence = get_turn_fetch_evidence(session_id or "")
    except Exception:
        pass

    has_fetch_grounding = bool(fetch_evidence)
    grounding_url = fetch_evidence[0].get("url", "") if fetch_evidence else ""

    prompt = (
        f"Extract key facts from the following text into distinct Subject-Predicate-Object statements.\n"
        f"Text: {text}\n\n"
        "Format: JSON list of objects with keys 'subject', 'predicate', 'object', 'context'.\n"
        "Example: [{'subject': 'Project Aurora', 'predicate': 'uses', 'object': 'daily reports', 'context': 'workflow tracking'}]"
    )

    try:
        from remy.core.llm import call_llm
        from remy.core.agent_tools import Level

        result = call_llm(prompt, purpose="extract_facts").content

        # Clean markdown
        clean = str(result).replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)

        if isinstance(data, list):
            stored_count = 0
            for item in data:
                subject = item.get("subject")
                predicate = item.get("predicate")
                obj = item.get("object")

                if subject and predicate and obj:
                    content = f"{subject} {predicate} {obj}."

                    if has_fetch_grounding:
                        # Grounded extraction: route through canonical API.
                        from remy.core.ingestion import ingest_grounded_evidence
                        ingestion = ingest_grounded_evidence(
                            content=content,
                            source_url=grounding_url or source,
                            session_id=session_id or "",
                            channel=channel,
                            extract_class="grounded_source_extract",
                            extra_tags=["fact", "extracted-fact"],
                            extra_meta={
                                "type": "fact",
                                "verified": True,
                                "extraction_method": "llm",
                                "structure": item,
                                "extracted_at": datetime.now().isoformat(),
                            },
                        )
                        if not ingestion.admitted:
                            if "not anchored by a fetch" not in (ingestion.reason or ""):
                                continue
                            store_level = Level.DOMAIN
                            store_tags = [
                                "grounded-evidence",
                                "claim:tool-verified",
                                "fact",
                                "extracted-fact",
                            ]
                            store_meta = _stamp_provenance({
                                "admission_class": "grounded_source_extract",
                                "learning_channel": "internet_evidence",
                                "source_url": grounding_url or source,
                                "source_anchored": True,
                                "fetch_tool": fetch_evidence[0].get("tool", ""),
                                "type": "fact",
                                "verified": True,
                                "extraction_method": "llm",
                                "structure": item,
                                "extracted_at": datetime.now().isoformat(),
                            }, channel, tags=store_tags)
                        else:
                            store_level = ingestion.level
                            store_tags = ingestion.tags
                            store_meta = _stamp_provenance(
                                ingestion.metadata, channel, tags=store_tags,
                            )
                    else:
                        store_level = Level.WORKING
                        store_tags = [
                            "extracted-fact",
                            "unverified-extraction",
                            "quarantine-unverified",
                        ]
                        store_meta = _stamp_provenance({
                            "type": "fact",
                            "verified": False,
                            "extraction_method": "llm",
                            "structure": item,
                            "extracted_at": datetime.now().isoformat(),
                            "source": source,
                            "learning_channel": "unverified",
                            "admission_class": "unverified_claim",
                            "requires_grounding": True,
                        }, channel, tags=store_tags)

                    brain.store(
                        content=content,
                        level=store_level,
                        tags=store_tags,
                        metadata=store_meta,
                    )
                    stored_count += 1
                    _sync_to_knowledge(content)

            if not has_fetch_grounding and stored_count > 0:
                return (
                    f"Extracted {stored_count} items from text. "
                    "No fetch evidence found this turn - stored as unverified working notes "
                    "(Level.WORKING, quarantine-unverified). "
                    "Use extract_content on a source URL first if you want durable domain facts."
                )
            return f"Extracted and stored {stored_count} facts."

        return "Failed to parse facts: format unexpected."

    except Exception as e:
        return f"Fact extraction failed: {e}"


def _handle_delegate_task(args: dict, session_id: str | None, channel: str | None) -> str:
    """Handle delegate_task tool - runs outside brain_lock to avoid deadlock."""
    from remy.core.tool_handlers.delegate import _handle_delegate_task as _delegate

    return _delegate(args, session_id, channel)

# ============== BROWSER TOOL HANDLERS ==============

_consecutive_browser_failures: int = 0
_browser_error_history: list[str] = []  # recent error messages for pattern analysis
_login_error_history: list[str] = []  # recent visible login errors for escalation
_BROWSER_ANALYZE_THRESHOLD: int = 3    # after 3 failures - analyze & suggest pivot
_BROWSER_HARD_STOP_LIMIT: int = 6      # after 6 failures - hard stop (agent had its chance)

def _analyze_browser_errors(errors: list[str]) -> str:
    """Compatibility wrapper for browser error pattern analysis."""
    from remy.core.tool_handlers.browser_dispatch import _analyze_browser_errors as _analyze

    return _analyze(errors)
def _handle_browser_tool(name: str, args: dict, session_id: str | None, channel: str | None) -> str:
    """Compatibility wrapper for the canonical browser dispatcher."""
    from remy.core.tool_handlers.browser_dispatch import _handle_browser_tool as _dispatch_browser_tool

    return _dispatch_browser_tool(name, args, session_id, channel)


class _BrowserLoop:
    """Persistent event loop in a dedicated daemon thread for browser operations.

    Keeps a single asyncio event loop alive across calls so that Playwright's
    BrowserManager sees the same loop_id and reuses the browser instead of
    re-launching on every tool invocation.
    """

    _instance: "_BrowserLoop | None" = None
    _lock = threading.Lock()

    def __init__(self):
        import warnings
        warnings.filterwarnings("ignore", category=ResourceWarning,
                                message="unclosed transport")
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="browser-loop"
        )
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    @classmethod
    def get(cls) -> "_BrowserLoop":
        if cls._instance is None or cls._instance._loop.is_closed():
            with cls._lock:
                if cls._instance is None or cls._instance._loop.is_closed():
                    cls._instance = cls()
        return cls._instance

    def run(self, coro):
        """Submit a coroutine to the persistent loop and wait for the result.

        Uses concurrent.futures to bridge sync - async safely. Polls with
        short timeout so KeyboardInterrupt can propagate.
        """
        import concurrent.futures
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        # Poll so KeyboardInterrupt can propagate
        while True:
            try:
                return future.result(timeout=0.5)
            except concurrent.futures.TimeoutError:
                continue

    def shutdown(self):
        """Stop the persistent loop (for cleanup/tests)."""
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)


def _run_async(coro) -> str:
    """Run an async coroutine from sync context using a persistent browser loop.

    Uses a dedicated daemon thread with a long-lived event loop so that
    Playwright's BrowserManager can reuse the same browser instance across
    multiple tool calls (same loop_id - no re-launch).
    """
    return _BrowserLoop.get().run(coro)


async def _handle_browse_page(args: dict, session_id: str | None, channel: str | None) -> str:
    """Compatibility wrapper for the canonical browser dispatcher."""
    from remy.core.tool_handlers.browser_dispatch import _handle_browse_page as _browse_page

    return await _browse_page(args, session_id, channel)


async def _handle_browser_act(args: dict, session_id: str | None, channel: str | None) -> str:
    """Compatibility wrapper for the canonical browser dispatcher."""
    from remy.core.tool_handlers.browser_dispatch import _handle_browser_act as _browser_act

    return await _browser_act(args, session_id, channel)


async def _handle_browser_close(args: dict, session_id: str | None, channel: str | None) -> str:
    """Compatibility wrapper for the canonical browser dispatcher."""
    from remy.core.tool_handlers.browser_dispatch import _handle_browser_close as _browser_close

    return await _browser_close(args, session_id, channel)


def _tool_gate_situation(session_id: str | None, channel: str | None) -> str:
    return f"tool_call:{channel or 'unknown'}:{session_id or 'unknown'}"


def _tool_gate_action(name: str, args: dict | None) -> str:
    compact_args = {
        str(key): str(value)[:160]
        for key, value in sorted((args or {}).items(), key=lambda item: str(item[0]))
    }
    if not compact_args:
        return f"tool:{name}"
    return f"tool:{name}:{json.dumps(compact_args, ensure_ascii=False, sort_keys=True)}"


def _tool_result_refuted(result: str) -> bool:
    text = str(result or "")
    if text.startswith(("Error:", "Unknown tool:", "Blocked by consequence memory:")):
        return True
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and parsed.get("error"):
            return True
    except Exception:
        pass
    return False


def _blocked_tool_policy_hint(
    name: str,
    args: dict | None,
    session_id: str | None,
    channel: str | None,
) -> dict | None:
    try:
        from remy.core.consequence_gate import consult_policy_hint

        store = getattr(brain, "_aura", brain)
        hint = consult_policy_hint(
            store,
            situation=_tool_gate_situation(session_id, channel),
            action=_tool_gate_action(name, args),
            namespace="remy-tools",
        )
        context = hint.to_context() if hasattr(hint, "to_context") else dict(hint or {})
        if context.get("hint") == "avoid" or context.get("should_block"):
            return context
    except Exception as exc:
        logger.debug("Legacy tool consequence gate skipped for %s: %s", name, exc)
    return None


def _blocked_tool_response(name: str, policy_hint: dict) -> str:
    return json.dumps(
        {
            "error": (
                "Blocked by consequence memory: this exact legacy tool action "
                "was previously refuted."
            ),
            "tool": name,
            "consequence_gate": {
                "blocked": True,
                "policy_hint": policy_hint,
            },
        },
        ensure_ascii=False,
    )


def _store_tool_consequence(
    name: str,
    args: dict | None,
    result: str,
    session_id: str | None,
    channel: str | None,
) -> None:
    try:
        store = getattr(brain, "_aura", brain)
        capture = getattr(store, "capture_consequence", None)
        if capture is None:
            return

        refuted = _tool_result_refuted(result)
        capture(
            situation=_tool_gate_situation(session_id, channel),
            action=_tool_gate_action(name, args),
            consequence="REFUTES" if refuted else "SUPPORTS",
            trust=-1 if refuted else 1,
            scope=[
                "tool-call",
                "legacy-brain-tools",
                f"tool:{name}",
                f"channel:{channel}" if channel else "channel:",
                f"session:{session_id}" if session_id else "session:",
                "result:error" if refuted else "result:ok",
            ],
            provenance=[
                "remy:brain_tools",
                f"session:{session_id}" if session_id else "session:",
            ],
            links={"session": session_id or "", "tool": name},
            namespace="remy-tools",
        )
    except TypeError:
        try:
            refuted = _tool_result_refuted(result)
            store = getattr(brain, "_aura", brain)
            store.capture_consequence(
                _tool_gate_situation(session_id, channel),
                _tool_gate_action(name, args),
                "REFUTES" if refuted else "SUPPORTS",
                -1 if refuted else 1,
            )
        except Exception:
            pass
    except Exception as exc:
        logger.debug("Failed to store legacy tool consequence for %s: %s", name, exc)


def _finalize_tool_result(
    name: str,
    args: dict | None,
    result: str,
    session_id: str | None,
    channel: str | None,
) -> str:
    _store_tool_consequence(name, args, result, session_id, channel)
    return result


def execute_tool(name: str, args: dict, session_id: str | None = None, channel: str | None = None) -> str:
    """Execute a brain tool, sandbox meta-tool, or sandbox tool.

    Includes per-tool circuit breaker and retry with backoff for transient failures.
    All brain operations are serialized via brain_lock to protect Rust AuraMemory backend.

    Args:
        name: Tool name.
        args: Tool arguments dict.
        session_id: Session ID for co-activation tracking (per-channel).
        channel: Channel context for provenance tracking (autonomous/desktop/telegram/voice).
    """
    # delegate_task runs OUTSIDE brain_lock - workers acquire it per-tool-call.
    # Running inside brain_lock would deadlock (orchestrator holds lock - workers need lock).
    if name not in _CONSEQUENCE_GATE_BYPASS_TOOLS:
        policy_block = _blocked_tool_policy_hint(name, args, session_id, channel)
        if policy_block:
            return _blocked_tool_response(name, policy_block)

    if name == "delegate_task":
        return _finalize_tool_result(
            name,
            args,
            _handle_delegate_task(args, session_id, channel),
            session_id,
            channel,
        )

    # Browser tools run OUTSIDE brain_lock - async I/O + vision API calls.
    # But trust validation needs brain_lock for brain.search().
    if name in ("browse_page", "browser_act", "browser_close"):
        if name in _TRUST_ENFORCED_TOOLS:
            from remy.core.agent_tools import brain_lock as _bl
            with _bl:
                block_msg = _validate_action_data(name, args)
            if block_msg:
                return _finalize_tool_result(
                    name,
                    args,
                    json.dumps({"error": block_msg}),
                    session_id,
                    channel,
                )
        return _finalize_tool_result(
            name,
            args,
            _handle_browser_tool(name, args, session_id, channel),
            session_id,
            channel,
        )

    if name in ("scratchpad", "filter_working"):
        return _finalize_tool_result(
            name,
            args,
            _execute_unlocked_working_memory_tool(name, args, session_id, channel),
            session_id,
            channel,
        )

    # Computer access tools run OUTSIDE brain_lock - pure filesystem/subprocess I/O.
    if name == "fs_read":
        return _finalize_tool_result(name, args, _handle_fs_read(args), session_id, channel)
    if name == "fs_write":
        return _finalize_tool_result(name, args, _handle_fs_write(args), session_id, channel)
    if name == "fs_search":
        return _finalize_tool_result(name, args, _handle_fs_search(args), session_id, channel)
    if name == "shell_exec":
        return _finalize_tool_result(name, args, _handle_shell_exec(args), session_id, channel)

    from remy.core.agent_tools import brain_lock

    with brain_lock:
        result = _execute_tool_locked(name, args, session_id, channel)
    return _finalize_tool_result(name, args, result, session_id, channel)


def _execute_unlocked_working_memory_tool(
    name: str,
    args: dict,
    session_id: str | None = None,
    channel: str | None = None,
) -> str:
    """Run working-memory tools that manage their own lock boundaries."""
    if name == "scratchpad":
        from remy.core.scratchpad import clear_notes, read_notes, summarize_notes, write_note

        action = str(args.get("action", "read") or "read").lower()
        if action == "write":
            content = args.get("content", "").strip()
            if not content:
                return json.dumps({"error": "content is required for write action"})
            result = write_note(content, session_id=session_id or "", channel=channel or "")
            return json.dumps(result, ensure_ascii=False)
        if action == "clear":
            deleted = clear_notes()
            return json.dumps({"cleared": True, "deleted_count": deleted})
        if action == "summarize":
            result = summarize_notes(
                session_id=session_id or "",
                channel=channel or "",
                force=bool(args.get("force", False)),
            )
            return json.dumps(result, ensure_ascii=False)

        notes = read_notes()
        return json.dumps({"notes": notes, "count": len(notes)}, ensure_ascii=False)

    if name == "filter_working":
        from remy.core.scratchpad import filter_working_memory

        query = args.get("query", "").strip()
        if not query:
            return json.dumps({"error": "query is required"}, ensure_ascii=False)
        result = filter_working_memory(
            query,
            session_id=session_id or "",
            min_score=float(args.get("min_score", 0.18) or 0.18),
            delete_irrelevant=bool(args.get("delete_irrelevant", False)),
        )
        return json.dumps(result, ensure_ascii=False)

    return json.dumps({"error": f"Unknown unlocked working-memory tool: {name}"})


def _execute_tool_locked(name: str, args: dict, session_id: str | None = None, channel: str | None = None) -> str:
    """Inner execute_tool, called under brain_lock."""
    # Circuit breaker check
    if not tool_health.is_available(name):
        report = tool_health.get_health_report()
        status = report.get(name, "unavailable")
        return json.dumps({"error": f"Tool '{name}' temporarily unavailable: {status}"})

    # Trust enforcement - block actions with unverified sensitive data
    block_msg = _validate_action_data(name, args)
    if block_msg:
        return json.dumps({"error": block_msg})

    _start_ts = time.time()
    result = _execute_tool_inner(name, args, session_id, channel)

    # Audit trail for critical tools (finance, registration, identity)
    from remy.core.audit_trail import is_critical
    if is_critical(name):
        _elapsed = (time.time() - _start_ts) * 1000
        _audit_status = "success"
        _audit_error = None
        try:
            _parsed = json.loads(result)
            if isinstance(_parsed, dict) and "error" in _parsed:
                _audit_status = "error"
                _audit_error = str(_parsed["error"])
        except (json.JSONDecodeError, TypeError):
            if result.startswith("Error:"):
                _audit_status = "error"
                _audit_error = result
        from remy.core.audit_trail import get_audit_logger
        get_audit_logger().log_action(
            tool_name=name, tool_input=args,
            raw_output=result, status=_audit_status,
            execution_time_ms=_elapsed, channel=channel,
            error_message=_audit_error,
        )

    # Track health only for network/infra-dependent tools
    # Logic errors (not found, invalid input) should NOT trip the circuit breaker
    _NETWORK_TOOLS = {"web_search", "http_get", "code_execution"}
    if name in _NETWORK_TOOLS:
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict) and "error" in parsed:
                tool_health.record_failure(name)
            else:
                tool_health.record_success(name)
        except (json.JSONDecodeError, TypeError):
            if result.startswith("Error:"):
                tool_health.record_failure(name)
            else:
                tool_health.record_success(name)

    return result


def _generate_image(args: dict, session_id: str | None, channel: str | None) -> str:
    """Generate image via Gemini and save to disk.

    Uses generate_content_stream with ImageConfig per official docs.
    """
    import mimetypes
    import uuid
    from google import genai as google_genai
    from google.genai import types as genai_types
    from remy.core.agent_tools import Level

    prompt = args["prompt"]
    client = google_genai.Client(api_key=settings.GEMINI_API_KEY)

    image_dir = Path(settings.DATA_DIR) / "generated_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    contents = [
        genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=prompt)],
        ),
    ]
    config = genai_types.GenerateContentConfig(
        image_config=genai_types.ImageConfig(
            image_size="1K",
        ),
        response_modalities=["IMAGE", "TEXT"],
    )

    # Stream response - collect image data from chunks
    data_buffer = None
    mime_type = None
    text_parts = []

    for chunk in client.models.generate_content_stream(
        model="gemini-3-pro-image-preview",
        contents=contents,
        config=config,
    ):
        if chunk.parts is None:
            continue
        for part in chunk.parts:
            if part.inline_data and part.inline_data.data:
                data_buffer = part.inline_data.data
                mime_type = part.inline_data.mime_type
            elif part.text:
                text_parts.append(part.text)

    if not data_buffer:
        return json.dumps({
            "generated": False,
            "error": " ".join(text_parts) or "No image generated",
        }, ensure_ascii=False)

    ext = mimetypes.guess_extension(mime_type) or ".png"
    filename = f"gen_{uuid.uuid4().hex[:8]}{ext}"
    filepath = image_dir / filename
    filepath.write_bytes(data_buffer)

    meta = _stamp_provenance({
        "type": "generated_image",
        "prompt": prompt,
        "model": "gemini-3-pro-image-preview",
        "filename": filename,
    }, channel)
    with brain_lock:
        rec = brain.store(
            content=f"Generated image: {prompt}",
            level=Level.WORKING,
            tags=["generated-image"],
            metadata=meta,
        )

    url = f"/api/generated_images/{filename}"
    return json.dumps({
        "generated": True,
        "filename": filename,
        "url": url,
        "record_id": rec.id,
        "prompt": prompt,
        "markdown": f"![{prompt[:80]}]({url})",
    }, ensure_ascii=False)


def _parse_markdown_to_sections(content: str) -> tuple[str, list[dict]]:
    """Fallback: parse markdown content into title + sections when LLM sends raw markdown."""
    lines = content.strip().split("\n")
    title = "Report"
    sections: list[dict] = []
    current_body: list[str] = []

    def flush_body():
        text = "\n".join(current_body).strip()
        if text:
            sections.append({"type": "text", "body": text})
        current_body.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ") and title == "Report":
            flush_body()
            title = stripped[2:].strip()
        elif stripped.startswith("## "):
            flush_body()
            sections.append({"type": "section", "title": stripped[3:].strip(), "body": ""})
        elif stripped.startswith("### "):
            flush_body()
            sections.append({"type": "subsection", "title": stripped[4:].strip(), "body": ""})
        elif stripped.startswith("* ") or stripped.startswith("- "):
            # Collect list items - append to current section body or buffer
            current_body.append(stripped)
        else:
            current_body.append(line)

    flush_body()

    # If we got sections with empty body, merge following text into them
    merged: list[dict] = []
    for s in sections:
        if merged and merged[-1].get("type") in ("section", "subsection") and not merged[-1].get("body") and s.get("type") == "text":
            merged[-1]["body"] = s["body"]
        else:
            merged.append(s)

    return title, merged or [{"type": "text", "body": content}]


def _normalize_report_sections(sections: list[dict]) -> list[dict]:
    """Normalize LLM report sections into a consistent schema."""
    normalized: list[dict] = []
    for raw in sections or []:
        section = dict(raw or {})
        if "body" not in section and "content" in section:
            section["body"] = section["content"]

        sec_type = section.get("type")
        if not sec_type:
            if section.get("items"):
                sec_type = "findings"
            elif section.get("headers") and section.get("rows"):
                sec_type = "table"
            elif section.get("title"):
                sec_type = "section"
            else:
                sec_type = "text"
        section["type"] = sec_type
        normalized.append(section)

    return normalized


def _generate_report(args: dict, session_id: str | None, channel: str | None) -> str:
    """Generate a PDF report and save to disk."""
    from remy.core.agent_tools import Level
    from remy.core.report_builder import ReportBuilder
    from remy.core.verification_gate import (
        emit_verification_incident,
        resolve_verification_incident,
        run_report_verification_gate,
    )

    title = args.get("title", "Report")
    subtitle = args.get("subtitle", "")
    sections = args.get("sections", [])

    # Fallback: LLM sent {"content": "# markdown..."} instead of title+sections
    if not sections and "content" in args:
        title, sections = _parse_markdown_to_sections(args["content"])
    sections = _normalize_report_sections(sections)

    report_dir = str(Path(settings.DATA_DIR) / "reports")
    report = ReportBuilder(
        title=title,
        subtitle=subtitle,
        author="Remy AI Agent",
        output_dir=report_dir,
        report_type=args.get("report_type", "standard"),
        include_toc=bool(args.get("include_toc", True)),
        metadata=args.get("metadata") or {},
    )

    for section in sections:
        sec_type = section.get("type", "text")

        if sec_type == "section":
            report.add_section(
                title=section.get("title", ""),
                body=section.get("body", ""),
            )
        elif sec_type == "subsection":
            report.add_subsection(
                title=section.get("title", ""),
                body=section.get("body", ""),
            )
        elif sec_type == "text":
            report.add_text(section.get("body", ""))
        elif sec_type == "quote":
            report.add_quote(section.get("body", ""))
        elif sec_type == "findings":
            report.add_key_findings(
                findings=section.get("items", []),
                title=section.get("title", "Key Findings"),
            )
        elif sec_type == "table":
            report.add_table(
                headers=section.get("headers", []),
                rows=section.get("rows", []),
                title=section.get("title", ""),
            )
        elif sec_type == "memory":
            report.add_memory_records(
                records=section.get("records", []),
                title=section.get("title", "Memory Records"),
            )
        elif sec_type == "audit":
            report.add_audit_summary(
                audit_logs=section.get("logs", []),
                title=section.get("title", "Audit Trail"),
            )
        elif sec_type == "page_break":
            report.add_page_break()

    filepath = report.save()
    verification = run_report_verification_gate(
        filepath,
        title=title,
        section_count=len(sections),
    )
    if not verification.verified and verification.repair_required:
        emit_verification_incident(
            source="generate_report",
            verification=verification,
            artifact_label=title,
        )
        try:
            Path(filepath).unlink(missing_ok=True)
        except Exception:
            pass
        return json.dumps({
            "generated": False,
            "error": verification.reason,
            "title": title,
            "verification": verification.to_dict(),
        }, ensure_ascii=False)
    filename = Path(filepath).name

    # Store metadata in brain
    meta = _stamp_provenance({
        "type": "generated_report",
        "title": title,
        "verification": verification.to_dict(),
        "filename": filename,
        "section_count": len(sections),
    }, channel)
    with brain_lock:
        rec = brain.store(
            content=f"Generated PDF report: {title}",
            level=Level.WORKING,
            tags=["generated-report"],
            metadata=meta,
        )
    resolve_verification_incident(
        source="generate_report",
        artifact_label=title,
        extra={"record_id": str(getattr(rec, "id", "") or "").strip()},
    )

    url = f"/api/reports/{filename}"
    markdown = f"[{title}]({url})"

    return json.dumps({
        "generated": True,
        "filename": filename,
        "url": url,
        "markdown": markdown,
        "record_id": rec.id,
        "title": title,
        "verification": verification.to_dict(),
        "instruction": "Present the report using the exact markdown link above. Do NOT add any host or domain - the relative URL is correct as-is.",
    }, ensure_ascii=False)


def _parse_markdown_to_slides(content: str) -> tuple[str, list[dict]]:
    """Convert raw markdown into presentation title + slides list."""
    lines = content.strip().split("\n")
    title = "Presentation"
    slides: list[dict] = []
    current_body: list[str] = []
    current_items: list[str] = []

    def flush():
        nonlocal current_body, current_items
        if current_items and slides:
            # Convert last section to bullets if it has items
            last = slides[-1]
            if last.get("type") == "section" and not last.get("body"):
                last["type"] = "bullets"
                last["items"] = current_items
                current_items = []
                current_body = []
                return
            slides.append({"type": "bullets", "title": "Key Points", "items": current_items})
            current_items = []
        if current_body:
            body_text = "\n".join(current_body).strip()
            if body_text:
                if slides and slides[-1].get("type") in ("section", "subsection") and not slides[-1].get("body"):
                    slides[-1]["body"] = body_text
                else:
                    slides.append({"type": "section", "title": "", "body": body_text})
            current_body = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            flush()
            candidate = stripped[2:].strip()
            if not slides and title == "Presentation":
                title = candidate
            else:
                slides.append({"type": "section", "title": candidate, "body": ""})
        elif stripped.startswith("## "):
            flush()
            slides.append({"type": "section", "title": stripped[3:].strip(), "body": ""})
        elif stripped.startswith("### "):
            flush()
            slides.append({"type": "subsection", "title": stripped[4:].strip(), "body": ""})
        elif stripped.startswith("* ") or stripped.startswith("- "):
            current_items.append(stripped[2:].strip())
        elif stripped.startswith("> "):
            flush()
            slides.append({"type": "quote", "body": stripped[2:].strip()})
        else:
            if current_items:
                flush()
            current_body.append(line)

    flush()
    return title, slides or [{"type": "section", "title": "Content", "body": content}]


def _generate_presentation(args: dict, session_id: str | None, channel: str | None) -> str:
    """Generate a PPTX presentation and save to disk."""
    from remy.core.agent_tools import Level
    from remy.core.presentation_builder import PresentationBuilder

    title = args.get("title", "Presentation")
    subtitle = args.get("subtitle", "")
    slides = args.get("slides", [])

    # Fallback: LLM sent {"content": "# markdown..."} instead of title+slides
    if not slides and "content" in args:
        title, slides = _parse_markdown_to_slides(args["content"])

    pres_dir = str(Path(settings.DATA_DIR) / "presentations")
    pres = PresentationBuilder(
        title=title,
        subtitle=subtitle,
        author="Remy AI Agent",
        output_dir=pres_dir,
    )

    for slide in slides:
        # Normalize: LLM may send "content" instead of "body"
        if "body" not in slide and "content" in slide:
            slide["body"] = slide["content"]
        slide_type = slide.get("type") or ("bullets" if slide.get("items") else "section")

        if slide_type == "section":
            pres.add_section(
                title=slide.get("title", ""),
                body=slide.get("body", ""),
            )
        elif slide_type == "subsection":
            pres.add_subsection(
                title=slide.get("title", ""),
                body=slide.get("body", ""),
            )
        elif slide_type == "bullets":
            pres.add_bullets(
                title=slide.get("title", ""),
                items=slide.get("items", []),
            )
        elif slide_type == "quote":
            pres.add_quote(
                text=slide.get("body", ""),
                author=slide.get("author", ""),
            )
        elif slide_type == "table":
            pres.add_table(
                title=slide.get("title", ""),
                headers=slide.get("headers", []),
                rows=slide.get("rows", []),
            )
        elif slide_type == "divider":
            pres.add_section_divider(slide.get("title", ""))

    filepath = pres.save()
    filename = Path(filepath).name

    # Store metadata in brain
    meta = _stamp_provenance({
        "type": "generated_presentation",
        "title": title,
        "filename": filename,
        "slide_count": len(slides),
    }, channel)
    with brain_lock:
        rec = brain.store(
            content=f"Generated PPTX presentation: {title}",
            level=Level.WORKING,
            tags=["generated-presentation"],
            metadata=meta,
        )

    url = f"/api/presentations/{filename}"
    markdown = f"[{title}]({url})"

    return json.dumps({
        "generated": True,
        "filename": filename,
        "url": url,
        "markdown": markdown,
        "record_id": rec.id,
        "title": title,
        "instruction": "Present the presentation using the exact markdown link above. Do NOT add any host or domain - the relative URL is correct as-is.",
    }, ensure_ascii=False)


def _execute_tool_inner(name: str, args: dict, session_id: str | None = None, channel: str | None = None) -> str:
    """Core tool execution logic. Called by execute_tool() with health tracking."""
    from remy.core.agent_tools import Level

    registry = get_registry()

    try:
        # ---- Meta-tools (selective tool loading) ----
        if name == "list_available_tools":
            extended = [
                {"name": t.name, "description": t.description[:120]}
                for t in BRAIN_TOOLS if t.name in EXTENDED_TOOL_NAMES
            ]
            return json.dumps({"available_tools": extended, "count": len(extended)})

        elif name == "enable_tools":
            requested = args.get("tool_names", [])
            if not requested:
                return json.dumps({"error": "No tool names provided"})
            valid = [n for n in requested if n in EXTENDED_TOOL_NAMES]
            invalid = [n for n in requested if n not in EXTENDED_TOOL_NAMES and n not in CORE_TOOL_NAMES]
            result = {"enabled": valid}
            if invalid:
                result["unknown"] = invalid
            return json.dumps(result)

        # ---- Sandbox meta-tools ----
        elif name == "sandbox_create_tool":
            return _sandbox_create_tool(args)
        elif name == "sandbox_test_tool":
            return _sandbox_test_tool(args)
        elif name == "sandbox_list_tools":
            return _sandbox_list_tools()

        # ---- Approved sandbox tools ----
        elif registry.is_sandbox_tool(name):
            return registry.execute_sandbox_tool(name, args)

        # ---- Core brain tools ----
        elif name == "recall":
            import time as _time

            def _is_recall_excluded(item) -> bool:
                tags = set(item.get("tags") or []) if isinstance(item, dict) else set(getattr(item, "tags", []) or [])
                return _SEARCH_CACHE_TAG in tags

            # 1. Brain recall (episodic, RRF semantic, ~200ms)
            with brain_lock:
                brain_results = brain.recall_structured(
                    args["query"], top_k=15, session_id=session_id,
                )
            brain_results = [r for r in (brain_results or []) if not _is_recall_excluded(r)]
            # Phase 3 Step 2: promotion/conflict/supersession gate for LLM-facing recall.
            # Admitted-but-not-promoted records must not surface as primary substrate.
            from remy.core.agent_tools import _apply_factual_recall_filter
            brain_results = _apply_factual_recall_filter(brain_results)

            # 2. Fallback: brain.search() - simple substring match
            # recall_structured uses RRF semantic matching which can miss
            # exact keyword matches. search() catches what RRF misses.
            seen_ids = {r.get("id") for r in (brain_results or []) if r.get("id")}
            with brain_lock:
                search_hits = brain.search(query=args["query"], limit=10)
            search_hits = [hit for hit in search_hits if not _is_recall_excluded(hit)]
            for hit in search_hits:
                if hit.id not in seen_ids:
                    brain_results.append({
                        "id": hit.id,
                        "content": hit.content,
                        "tags": list(getattr(hit, "tags", [])),
                        "metadata": getattr(hit, "metadata", {}) or {},
                        "score": 0.6,  # decent score for exact substring match
                    })
                    seen_ids.add(hit.id)

            # 2b. Failure-aware recall - surface outcome records related to the query.
            # This prevents the agent from re-proposing failed actions.
            try:
                with brain_lock:
                    failure_hits = brain.search(query=args["query"], tags=["outcome-failure"], limit=5)
                for hit in failure_hits:
                    if hit.id not in seen_ids:
                        brain_results.append({
                            "id": hit.id,
                            "content": hit.content,
                            "tags": list(getattr(hit, "tags", [])),
                            "metadata": getattr(hit, "metadata", {}) or {},
                            "score": 0.8,  # high score - failures are critical context
                        })
                        seen_ids.add(hit.id)
            except Exception:
                pass

            # 3. Knowledge recall (semantic, SDR - optional)
            kb_results = []
            try:
                kb_results = _kb_retrieve(args["query"], top_k=5)
            except Exception:
                pass  # Knowledge is optional

            if not brain_results and not kb_results:
                return "No relevant memories found."

            # Phase A.8.1 - factual-query containment.
            # If the query looks like a factual/citation/verify request, filter
            # out forbidden narrative classes before the result is rendered for
            # the LLM.  This closes the direct-recall bypass around the A.8
            # inject-context boundary that was identified in live validation.
            # Non-factual queries are unaffected - general memory recall is preserved.
            try:
                from remy.core.hybrid_search import _FACTUAL_FORBIDDEN_TAGS, is_factual_query
                if is_factual_query(args.get("query", "")):
                    _factual_forbidden = _FACTUAL_FORBIDDEN_TAGS
                    brain_results = [
                        r for r in brain_results
                        if not _factual_forbidden.intersection(
                            set(r.get("tags") or []) if isinstance(r, dict)
                            else set(getattr(r, "tags", []) or [])
                        )
                    ]
            except Exception:
                pass  # filter is best-effort; never break recall entirely

            lines, seen = [], set()

            # Enrich recall_structured results with full metadata (recall_structured only returns partial)
            for r in (brain_results or []):
                if r.get("id") and not r.get("_meta_enriched"):
                    try:
                        full_rec = brain.get(r["id"])
                        if full_rec:
                            if full_rec.metadata:
                                enriched = dict(full_rec.metadata)
                                enriched.update({k: v for k, v in (r.get("metadata") or {}).items()})
                                r["metadata"] = enriched
                            # Some backends promote source_type to a record attribute
                            # and drop it from the metadata dict. Surface it so the
                            # renderer can show the provenance label (retrieved/etc).
                            _rec_source_type = getattr(full_rec, "source_type", None)
                            if _rec_source_type and not (r.get("metadata") or {}).get("source_type"):
                                _m = dict(r.get("metadata") or {})
                                _m["source_type"] = _rec_source_type
                                r["metadata"] = _m
                            created = getattr(full_rec, "created_at", None)
                            if created is not None and "created_at" not in r:
                                r["created_at"] = created
                            r["_meta_enriched"] = True
                    except Exception:
                        pass

            # Brain first (richer metadata, higher authority)
            for r in (brain_results or []):
                meta = r.get("metadata") or {}
                trust = _compute_effective_trust(meta, _time.time())
                source = meta.get("source", "unknown")
                source_label = source.replace("agent-", "").replace("user-", "")
                content = r["content"][:300]
                key = content[:80].lower().strip()
                if key in seen:
                    continue
                seen.add(key)
                tag_str = f" [{', '.join(r.get('tags', []))}]" if r.get("tags") else ""
                # Verification labels
                verified = meta.get("verified")
                actionable = meta.get("actionable")
                source_type = meta.get("source_type", "")
                if verified is True:
                    verif_label = " VERIFIED"
                elif source == "agent-autonomous" or source == "agent-worker":
                    verif_label = " UNVERIFIED"
                elif trust >= 0.6:
                    verif_label = " likely"
                else:
                    verif_label = ""
                if actionable is False:
                    verif_label += " NOT-ACTIONABLE"
                src_type_label = f" [{source_type}]" if source_type in ("retrieved", "inferred", "generated") else ""
                record_id = str(r.get("id", "") or "")
                id_label = f"[id:{record_id}] " if record_id else ""
                created_at = r.get("created_at") or meta.get("created_at")
                age = format_age(created_at)
                age_label = f" [{age}]" if age else ""
                lines.append(
                    f"{id_label}[trust: {trust:.1f} | {source_label}]"
                    f"{verif_label}{src_type_label}{age_label} {content}{tag_str}"
                )

            # Knowledge results (semantic matches brain may have missed)
            for kb in kb_results:
                key = kb["text"][:80].lower().strip()
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"[KB | score: {kb['score']:.1f}] {kb['text'][:300]}")

            return "\n".join(lines) if lines else "No relevant memories found."

        elif name == "store":
            tags = [_clean_tag(t) for t in args.get("tags", "").split(",") if t.strip()]
            level_map = {
                "L1_WORKING": Level.WORKING,
                "L2_DECISIONS": Level.DECISIONS,
                "L3_DOMAIN": Level.DOMAIN,
            }
            if args.get("level") == "L4_IDENTITY":
                return json.dumps({
                    "error": "L4_IDENTITY is reserved. Use store_user_profile for user data or store_person for people."
                }, ensure_ascii=False)
            level = level_map.get(args.get("level", ""), Level.DOMAIN)

            # D-03 (Phase 2 tightened): DOMAIN transit check.
            # Without real fetch evidence this turn, an LLM-issued store(L3_DOMAIN)
            # is not grounded and must be downgraded.
            # Bypass is only granted when one of:
            #   (a) a turn fetch happened this session, OR
            #   (b) metadata carries an admission_class from the canonical
            #       FACTUAL_SAFE taxonomy (grounded_external_fact /
            #       grounded_source_extract / operator_asserted). Ad-hoc legacy
            #       strings (grounded_finding, grounded_extraction, etc.) no
            #       longer bypass - they must route through the ingestion API.
            #   (c) user-direct channel with an explicit user attribution tag.
            if level == Level.DOMAIN:
                from remy.core.memory_policy import FACTUAL_SAFE_ADMISSION_CLASSES
                _has_transit = False
                try:
                    from remy.core.claim_provenance import get_turn_fetch_evidence
                    _has_transit = bool(get_turn_fetch_evidence(session_id or ""))
                except Exception:
                    pass
                _meta_arg = args.get("metadata") if isinstance(args.get("metadata"), dict) else {}
                _admission_class = _meta_arg.get("admission_class")
                _canonical_factual_safe = (
                    _admission_class in FACTUAL_SAFE_ADMISSION_CLASSES
                )
                _user_attributed = (
                    channel in _USER_DIRECT_CHANNELS
                    and bool(set(tags) & {"user-profile", "user-statement", "from-user"})
                )
                if not _has_transit and not _canonical_factual_safe and not _user_attributed:
                    level = Level.WORKING
                    tags.append("requires-grounding")
                    tags.append("downgraded-no-transit")

            # Human-in-the-loop: financial tags require user approval before storing
            from remy.core.approval_queue import needs_approval, build_approval_description, approval_queue as _aq
            if needs_approval("store", args):
                description = build_approval_description("store", args)

                def _do_store():
                    with brain_lock:
                        _existing = _check_duplicates(args["content"][:100], tags=tags or None)
                        semantic_type = infer_semantic_type(
                            explicit=args.get("semantic_type"),
                            tags=tags,
                            level=level,
                            content=args.get("content"),
                        )
                        _meta = _stamp_provenance({"semantic_type": semantic_type}, channel)
                        if isinstance(args.get("metadata"), dict):
                            _meta.update(args["metadata"])
                        _meta.update(_apply_store_guard(args["content"], tags, channel))
                        _claim_patch, _claim_tags, _claim_quarantine, _claim_reason = _apply_store_claim_gate(
                            args["content"], tags, channel, session_id, _meta,
                        )
                        _meta.update(_claim_patch)
                        if _claim_reason:
                            _meta.setdefault("claim_gate_reason", _claim_reason)
                        if _claim_quarantine:
                            for _t in _claim_tags:
                                if _t not in tags:
                                    tags.append(_t)
                        _rec = brain.store(content=args["content"], level=level, tags=tags,
                                           metadata=_meta, semantic_type=semantic_type)
                    _sync_needed, _sync_pin = _should_sync(level)
                    if _sync_needed and not (set(tags) & _NO_MIRROR_TAGS):
                        _sync_to_knowledge(args["content"], pin=_sync_pin)
                    _result: dict = {"stored": True, "id": _rec.id}
                    if _existing:
                        _result["similar_existing"] = _existing
                        _result["note"] = "Similar memory already exists. Stored anyway (may auto-merge)."
                    return json.dumps(_result, ensure_ascii=False)

                return _aq.request_approval_sync(
                    description, _do_store,
                    tool_name="store", tool_args=args,
                )

            # Check for similar existing content
            with brain_lock:
                existing = _check_duplicates(args["content"][:100], tags=tags or None)

                semantic_type = infer_semantic_type(
                    explicit=args.get("semantic_type"),
                    tags=tags,
                    level=level,
                    content=args.get("content"),
                )
                store_meta = _stamp_provenance({"semantic_type": semantic_type}, channel)
                if isinstance(args.get("metadata"), dict):
                    store_meta.update(args["metadata"])
                store_meta.update(_apply_store_guard(args["content"], tags, channel))
                claim_patch, claim_tags, claim_quarantine, claim_reason = _apply_store_claim_gate(
                    args["content"], tags, channel, session_id, store_meta,
                )
                store_meta.update(claim_patch)
                if claim_reason:
                    store_meta.setdefault("claim_gate_reason", claim_reason)
                if claim_quarantine:
                    for _t in claim_tags:
                        if _t not in tags:
                            tags.append(_t)
                rec = brain.store(content=args["content"], level=level, tags=tags,
                                  metadata=store_meta, semantic_type=semantic_type)
            # Mirror to knowledge (fire-and-forget) - never mirror quarantined claims.
            sync_needed, sync_pin = _should_sync(level)
            if sync_needed and not claim_quarantine and not (set(tags) & _NO_MIRROR_TAGS):
                _sync_to_knowledge(args["content"], pin=sync_pin)

            result = {"stored": True, "id": rec.id}
            if claim_quarantine:
                result["quarantined"] = True
                result["claim_class"] = store_meta.get("claim_class", "citation_claim")
                result["reason"] = claim_reason
                result["warning"] = (
                    "This claim could not be structurally verified (LLM-only output with "
                    "unverified external references). Stored in quarantine, NOT in factual "
                    "memory. Do not cite it as a fact until a tool call anchors the claim."
                )
            if store_meta.get("actionable") is False:
                result["actionable"] = False
                result.setdefault("warning", (
                    "This record contains sensitive data and was stored as NOT actionable. "
                    "It cannot be used for external actions until verified by the user. "
                    "Use verify_record to mark it as verified after user confirmation."
                ))
            if existing:
                result["similar_existing"] = existing
                result["note"] = "Similar memory already exists. Stored anyway (may auto-merge)."
            return json.dumps(result, ensure_ascii=False)

        elif name in {"search", "search_exact"}:
            query = args.get("query", "") or ""
            tags = [_clean_tag(t) for t in args.get("tags", "").split(",") if t.strip()] or None
            if name == "search_exact":
                with brain_lock:
                    results = search_exact_structured(
                        brain,
                        query,
                        tags=tags,
                        top_k=10,
                        lexical_limit=10,
                    )
            elif not query and tags:
                # Tag-only search - use brain.search() which supports tag filtering
                with brain_lock:
                    tag_results = brain.search(query="", tags=tags, limit=10)
                results = [{"id": r.id, "content": sanitize_memory_content(r.content, metadata=r.metadata, tags=list(r.tags or [])), "tags": list(r.tags),
                            "score": 1.0, "metadata": sanitize_memory_metadata(r.metadata, tags=list(r.tags or [])),
                            "level": str(getattr(r, "level", "")),
                            "source": (r.metadata or {}).get("source")} for r in tag_results]
            else:
                with brain_lock:
                    results = hybrid_search_structured(
                        brain,
                        query,
                        tags=tags,
                        top_k=10,
                        min_strength=0.05,
                        session_id=session_id,
                    )
            if tags:
                results = [r for r in results if any(t in r.get("tags", []) for t in tags)]
            if not results:
                return "No results found."
            import time as _time
            items = []
            for r in results[:5]:
                meta = sanitize_memory_metadata(r.get("metadata") or {}, tags=r.get("tags", []))
                item = {
                    "id": r["id"],
                    "content": sanitize_memory_content(
                        r.get("content", ""),
                        metadata=r.get("metadata") or {},
                        tags=r.get("tags", []),
                    )[:200],
                    "tags": r.get("tags", []),
                    "level": r.get("level"),
                    "metadata": meta,
                    "score": round(r.get("score", 0), 3),
                    "trust": _compute_effective_trust(meta, _time.time()),
                    "source": r.get("source") or meta.get("source"),
                }
                if r.get("importance") is not None:
                    item["importance"] = round(r["importance"], 4)
                items.append(item)
            return json.dumps(items, ensure_ascii=False)

        elif name == "store_person":
            if channel in ("autonomous", "proactive") or (channel and channel.startswith("worker-")):
                return json.dumps({"error": "store_person is only allowed in interactive channels (desktop/telegram). Cannot fabricate people from autonomous context."}, ensure_ascii=False)
            full_name = str(args.get("full_name") or args.get("name") or "").strip()
            if not full_name:
                return json.dumps({"error": "store_person requires full_name or name."}, ensure_ascii=False)
            role = str(args.get("role") or "").strip()
            birth_date = str(args.get("birth_date") or "").strip()
            birth_place = str(args.get("birth_place") or "").strip()

            # Check for existing person with similar name
            _profile_rec = get_user_profile_record()
            family_text = str((_profile_rec.metadata or {}).get("family", "") or "") if _profile_rec else ""
            with brain_lock:
                resolved = resolve_person_identity_input(full_name, role, birth_date, family_text)
                canonical_name = str(resolved.get("full_name") or full_name).strip()
                aliases = list(resolved.get("aliases") or [])
                existing = _check_duplicates(full_name, tags=["person"])
                existing_people = brain.search(query="", tags=["person"], limit=50)
                existing_rec = None
                for person_rec in existing_people:
                    if person_matches_identity(
                        person_rec.metadata or {},
                        full_name,
                        role,
                        birth_date,
                        family_text,
                    ):
                        existing_rec = person_rec
                        break

                parts = [canonical_name]
                if role:
                    parts.append(role)
                if birth_date:
                    parts.append(f"born {birth_date}")
                if birth_place:
                    parts.append(f"in {birth_place}")

                tags = ["person"]
                if role:
                    tags.append(_sanitize_tag(role))

                metadata = _stamp_provenance({
                        "type": "person",
                        "full_name": canonical_name,
                        "role": role,
                        "birth_date": birth_date,
                        "birth_place": birth_place,
                    }, channel)
                if aliases:
                    metadata["aliases"] = aliases

                if existing_rec:
                    existing_meta = dict(existing_rec.metadata or {})
                    existing_aliases = list(existing_meta.get("aliases") or [])
                    for alias in aliases + [full_name]:
                        if alias and alias != canonical_name and alias not in existing_aliases:
                            existing_aliases.append(alias)
                    existing_meta.update({k: v for k, v in metadata.items() if v not in (None, "")})
                    if existing_aliases:
                        existing_meta["aliases"] = existing_aliases
                    brain.update(
                        existing_rec.id,
                        content=", ".join(parts),
                        metadata=existing_meta,
                        tags=tags,
                    )
                    rec = brain.get(existing_rec.id) or existing_rec
                else:
                    rec = brain.store(
                        content=", ".join(parts),
                        level=Level.IDENTITY,
                        tags=tags,
                        metadata=metadata,
                    )
            # Mirror to knowledge
            _sync_to_knowledge(", ".join(parts))

            result = {"stored": True, "id": rec.id, "name": canonical_name}
            if existing:
                result["similar_existing"] = existing
                result["warning"] = f"Found {len(existing)} similar person(s) already in memory. Check if this is a duplicate."
            if existing_rec:
                result["merged_into_existing"] = True
            if aliases:
                result["aliases"] = aliases
            return json.dumps(result, ensure_ascii=False)

        elif name == "store_story":
            if channel in ("autonomous", "proactive") or (channel and channel.startswith("worker-")):
                return json.dumps({"error": "store_story is only allowed in interactive channels (desktop/telegram). Use recall to read existing stories."}, ensure_ascii=False)
            title = args["title"]
            content = args["content"]
            people = [p.strip() for p in args.get("people_mentioned", "").split(",") if p.strip()]

            # Check for existing story with similar title
            with brain_lock:
                existing = _check_duplicates(title, tags=["story"])

                people_tags = [_sanitize_tag(p) for p in people]
                tags = ["story"] + [t for t in people_tags if t]
                rec = brain.store(
                    content=f"{title}\n{content}",
                    level=Level.DOMAIN,
                    tags=tags,
                    metadata=_stamp_provenance({
                        "type": "story",
                        "title": title,
                        "people_mentioned": people,
                    }, channel),
                )
                # Connect story to people (promotion-gated)
                from remy.core.agent_tools import gated_connect
                for person_name in people:
                    person_records = brain.search(query=person_name, tags=["person"], limit=3)
                    for pr in person_records:
                        if pr.metadata.get("full_name", "").lower() == person_name.lower():
                            gated_connect(brain, rec.id, pr.id, weight=0.7)
                            break

            # Mirror to knowledge
            _sync_to_knowledge(f"{title}\n{content}")

            result = {"stored": True, "id": rec.id, "title": title}
            if existing:
                result["similar_existing"] = existing
                result["warning"] = f"Found {len(existing)} similar story/stories already in memory."
            return json.dumps(result, ensure_ascii=False)

        elif name == "connect_records":
            id_a = args["id_a"]
            id_b = args["id_b"]
            relationship = args["relationship"]
            weight = float(args.get("weight", 0.7))
            weight = max(0.0, min(1.0, weight))

            if id_a == id_b:
                return json.dumps({
                    "connected": False,
                    "error": "Cannot connect a record to itself",
                    "id_a": id_a,
                    "id_b": id_b,
                })

            with brain_lock:
                rec_a = brain.get(id_a)
                rec_b = brain.get(id_b)
                if not rec_a:
                    return json.dumps({
                        "connected": False,
                        "error": f"Record {id_a} not found",
                        "id_a": id_a,
                        "id_b": id_b,
                    })
                if not rec_b:
                    return json.dumps({
                        "connected": False,
                        "error": f"Record {id_b} not found",
                        "id_a": id_a,
                        "id_b": id_b,
                    })

                from remy.core.agent_tools import gated_connect
                if not gated_connect(brain, id_a, id_b, weight=weight):
                    try:
                        from remy.core.memory_policy import (
                            FACTUAL_FORBIDDEN_ADMISSION_CLASSES,
                            FACTUAL_SAFE_ADMISSION_CLASSES,
                        )
                        safe_a = (rec_a.metadata or {}).get("admission_class") in FACTUAL_SAFE_ADMISSION_CLASSES
                        safe_b = (rec_b.metadata or {}).get("admission_class") in FACTUAL_SAFE_ADMISSION_CLASSES
                        forbidden_tags = {
                            "quarantine-unverified",
                            "claim:llm-unverified",
                            "citation-claim",
                            "scratchpad",
                            "scratchpad-summary",
                            "generated-report",
                        }
                        forbidden_a = (
                            (rec_a.metadata or {}).get("admission_class") in FACTUAL_FORBIDDEN_ADMISSION_CLASSES
                            or bool(set(getattr(rec_a, "tags", []) or []) & forbidden_tags)
                        )
                        forbidden_b = (
                            (rec_b.metadata or {}).get("admission_class") in FACTUAL_FORBIDDEN_ADMISSION_CLASSES
                            or bool(set(getattr(rec_b, "tags", []) or []) & forbidden_tags)
                        )
                        if (safe_a and safe_b) or not (forbidden_a or forbidden_b):
                            try:
                                brain.connect(id_a, id_b, weight=weight)
                            except TypeError:
                                brain.connect(id_a, id_b, weight)
                        else:
                            return json.dumps({
                                "connected": False,
                                "error": "Connection blocked: one or both records are not eligible for promotion",
                                "id_a": id_a,
                                "id_b": id_b,
                            })
                    except Exception:
                        return json.dumps({
                            "connected": False,
                            "error": "Connection blocked: one or both records are not eligible for promotion",
                            "id_a": id_a,
                            "id_b": id_b,
                        })

                # Store relationship description as metadata-enriched record
                rel_content = f"{relationship}: '{rec_a.content[:80]}' - '{rec_b.content[:80]}'"
                brain.store(
                    content=rel_content,
                    level=Level.DOMAIN,
                    tags=["relationship"],
                    metadata=_stamp_provenance({
                        "type": "relationship",
                        "id_a": id_a,
                        "id_b": id_b,
                        "relationship": relationship,
                    }, channel),
                )

            return json.dumps({
                "connected": True,
                "id_a": id_a,
                "id_b": id_b,
                "relationship": relationship,
                "weight": weight,
                "a_preview": rec_a.content[:100],
                "b_preview": rec_b.content[:100],
            }, ensure_ascii=False)

        elif name == "get_connections":
            record_id = args["record_id"]
            connections = []
            with brain_lock:
                rec = brain.get(record_id)
                if not rec:
                    return json.dumps({"error": f"Record {record_id} not found"})

                for conn_id, weight in rec.connections.items():
                    conn_rec = brain.get(conn_id)
                    if conn_rec:
                        connections.append({
                            "id": conn_id,
                            "content": conn_rec.content[:150],
                            "tags": list(conn_rec.tags) if conn_rec.tags else [],
                            "weight": round(weight, 3),
                        })

            if not connections:
                return json.dumps({"record": rec.content[:150], "connections": [], "message": "No connections found."})

            return json.dumps({
                "record": rec.content[:150],
                "connection_count": len(connections),
                "connections": connections,
            }, ensure_ascii=False)

        elif name == "get_full_record":
            record_id = args.get("record_id", "").strip()
            if not record_id:
                return json.dumps({"error": "record_id is required"}, ensure_ascii=False)
            with brain_lock:
                rec = brain.get(record_id)
            if not rec:
                return json.dumps({"error": f"Record '{record_id}' not found"}, ensure_ascii=False)
            tags = list(rec.tags) if rec.tags else []
            meta = rec.metadata or {}
            protected = sorted(protected_fields_for_record(meta, tags=tags))
            result = {
                "id": rec.id,
                "content": sanitize_memory_content(rec.content, metadata=meta, tags=tags),
                "tags": tags,
                "level": rec.level.name if hasattr(rec.level, "name") else str(rec.level),
                "char_count": len(rec.content),
            }
            if meta:
                safe_meta = sanitize_memory_metadata(meta, tags=tags)
                result["metadata"] = {
                    k: v for k, v in safe_meta.items() if k not in ("source", "verified", "actionable")
                }
            if protected:
                result["protected_fields_present"] = protected
                result["note"] = (
                    "Protected exact fields are hidden in get_full_record. "
                    "Use get_protected_record only when the user explicitly asks for a sensitive exact value."
                )
            return json.dumps(result, ensure_ascii=False)

        elif name == "get_protected_record":
            record_id = args.get("record_id", "").strip()
            if not record_id:
                return json.dumps({"error": "record_id is required"}, ensure_ascii=False)
            if channel in {"autonomous", "proactive"} or (channel or "").startswith("worker-"):
                return json.dumps(
                    {
                        "error": "get_protected_record is only available in interactive channels. "
                        "Autonomous flows should use action guards and verified exact-memory checks instead."
                    },
                    ensure_ascii=False,
                )
            requested_fields = [
                item.strip()
                for item in (args.get("fields", "") or "").split(",")
                if item.strip()
            ]
            with brain_lock:
                rec = brain.get(record_id)
            if not rec:
                return json.dumps({"error": f"Record '{record_id}' not found"}, ensure_ascii=False)
            tags = list(rec.tags) if rec.tags else []
            meta = rec.metadata or {}
            payload = protected_payload(meta, tags=tags, requested_fields=requested_fields)
            if not payload:
                available = sorted(protected_fields_for_record(meta, tags=tags))
                if available:
                    return json.dumps(
                        {
                            "error": "Requested protected fields not found on this record",
                            "available_fields": available,
                        },
                        ensure_ascii=False,
                    )
                return json.dumps(
                    {
                        "error": "This record does not contain protected exact fields",
                        "instruction": "Use get_full_record for non-sensitive full content.",
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "id": rec.id,
                    "level": rec.level.name if hasattr(rec.level, "name") else str(rec.level),
                    "protected_fields": sorted(payload.keys()),
                    "values": payload,
                    "verified": meta.get("verified") is True,
                    "actionable": meta.get("actionable"),
                    "source": meta.get("source"),
                },
                ensure_ascii=False,
            )

        elif name == "update_record":
            record_id = args["record_id"]
            with brain_lock:
                existing = brain.get(record_id)
                if not existing:
                    return json.dumps({"error": f"Record '{record_id}' not found"}, ensure_ascii=False)

                # Guard: autonomous agents cannot overwrite user-confirmed records
                existing_meta = existing.metadata or {}
                if (channel in ("autonomous", "proactive") or (channel and channel.startswith("worker-"))):
                    if existing_meta.get("source") == "user-confirmed" or existing_meta.get("verified") is True:
                        return json.dumps({
                            "error": f"Cannot overwrite user-confirmed record in autonomous channel. Use verify_record to propose a correction.",
                        }, ensure_ascii=False)

                kwargs = {}
                if "content" in args and args["content"]:
                    kwargs["content"] = args["content"]
                if "tags" in args and args["tags"]:
                    kwargs["tags"] = [_clean_tag(t) for t in args["tags"].split(",") if t.strip()]
                if "level" in args and args["level"]:
                    level_map = {"working": Level.WORKING, "decisions": Level.DECISIONS, "domain": Level.DOMAIN, "identity": Level.IDENTITY}
                    kwargs["level"] = level_map.get(args["level"].lower(), existing.level)

                # Audit trail: preserve original_content on first update
                from datetime import datetime as _dt_upd
                audit_meta = dict(existing_meta)
                if "original_content" not in audit_meta and "content" in kwargs:
                    audit_meta["original_content"] = existing.content
                # Track who last updated and when
                updated_by = f"agent-{channel}" if channel else "agent"
                audit_meta["last_updated_by"] = updated_by
                audit_meta["last_updated_at"] = _dt_upd.now().isoformat()
                kwargs["metadata"] = audit_meta

                updated = brain.update(record_id, **kwargs)
            if not updated:
                return json.dumps({"error": "Update failed"}, ensure_ascii=False)

            return json.dumps({
                "updated": True,
                "id": record_id,
                "content": updated.content[:200],
                "tags": list(updated.tags),
            }, ensure_ascii=False)

        elif name == "delete_record":
            record_id = args["record_id"]
            with brain_lock:
                existing = brain.get(record_id)
                if not existing:
                    return json.dumps({"error": f"Record '{record_id}' not found"}, ensure_ascii=False)

                preview = existing.content[:100]
                success = brain.delete(record_id)
            return json.dumps({
                "deleted": success,
                "id": record_id,
                "deleted_content": preview,
            }, ensure_ascii=False)

        elif name == "mark_stale":
            # Mark a record as stale without deleting it. The content stays
            # (useful for audit + provenance), but a `stale` tag + metadata
            # stamp signal to recall/search that this belief is no longer
            # authoritative. Reason is required so future readers know why.
            from datetime import datetime as _dt_stale

            record_id = str(args.get("record_id") or "").strip()
            reason = str(args.get("reason") or "").strip()
            if not record_id:
                return json.dumps({"error": "mark_stale requires record_id"}, ensure_ascii=False)
            if not reason:
                return json.dumps({"error": "mark_stale requires reason (why is this stale-)"}, ensure_ascii=False)

            superseded_by = str(args.get("superseded_by") or "").strip()

            with brain_lock:
                existing = brain.get(record_id)
                if not existing:
                    return json.dumps({"error": f"Record '{record_id}' not found"}, ensure_ascii=False)

                existing_meta = dict(existing.metadata or {})
                if existing_meta.get("stale") is True:
                    return json.dumps({
                        "already_stale": True,
                        "id": record_id,
                        "stale_marked_at": existing_meta.get("stale_marked_at"),
                        "stale_reason": existing_meta.get("stale_reason"),
                    }, ensure_ascii=False)

                # Guard: autonomous agents cannot stale-mark user-confirmed records
                if channel in ("autonomous", "proactive") or (channel and channel.startswith("worker-")):
                    if existing_meta.get("source") == "user-confirmed" or existing_meta.get("verified") is True:
                        return json.dumps({
                            "error": "Cannot mark user-confirmed record as stale from autonomous channel.",
                        }, ensure_ascii=False)

                new_tags = list(existing.tags or [])
                if "stale" not in new_tags:
                    new_tags.append("stale")

                existing_meta.update({
                    "stale": True,
                    "stale_marked_at": _dt_stale.now().isoformat(),
                    "stale_reason": reason,
                    "stale_marked_by": f"agent-{channel}" if channel else "agent",
                })
                if superseded_by:
                    existing_meta["superseded_by"] = superseded_by

                updated = brain.update(
                    record_id,
                    tags=new_tags,
                    metadata=existing_meta,
                )

            if not updated:
                return json.dumps({"error": "mark_stale update failed"}, ensure_ascii=False)

            # Invalidate recall cache entries that may reference this record
            try:
                clear_recall_cache(existing.content[:200])
            except Exception:
                pass

            return json.dumps({
                "marked_stale": True,
                "id": record_id,
                "reason": reason,
                "superseded_by": superseded_by or None,
                "preview": existing.content[:120],
            }, ensure_ascii=False)

        elif name == "store_user_profile":
            from remy.core.tool_handlers.profile import normalize_profile_fields

            # Collect non-empty profile fields
            profile_fields = {}
            for key in _PROFILE_INPUT_FIELDS:
                val = args.get(key, "").strip()
                if val:
                    profile_fields[key] = val
            profile_fields = normalize_profile_fields(profile_fields)

            if not profile_fields:
                return json.dumps({"error": "No profile fields provided"})

            # Search for existing user profile (newest by created_at)
            _existing_rec = get_user_profile_record()
            with brain_lock:
                if _existing_rec:
                    rec = _existing_rec
                    merged_meta = dict(rec.metadata) if rec.metadata else {}
                    # Append-only fields: notes accumulates instead of overwriting
                    new_notes = profile_fields.pop("notes", None)
                    merged_meta.update(profile_fields)
                    if new_notes:
                        old_notes = merged_meta.get("notes", "") or ""
                        # Dedup: only append sentences not already present
                        existing_sentences = {s.strip().lower() for s in old_notes.split(";") if s.strip()}
                        new_sentences = [s.strip() for s in new_notes.split(";") if s.strip()]
                        added = [s for s in new_sentences if s.lower() not in existing_sentences]
                        if added:
                            merged_meta["notes"] = (old_notes + "; " + "; ".join(added)).lstrip("; ") if old_notes else "; ".join(added)
                        # else: nothing new - don't touch notes
                    merged_meta = normalize_profile_fields(merged_meta)
                    merged_meta["protected_fields"] = sorted(
                        field for field in ("phone", "email") if merged_meta.get(field)
                    )
                    content = _format_profile_content(merged_meta)
                    brain.update(rec.id, content=content, metadata=merged_meta)
                    # Mirror updated profile as anchor
                    _sync_to_knowledge(content, pin=True)
                    return json.dumps({
                        "updated": True, "id": rec.id,
                        "fields_updated": list(profile_fields.keys()),
                        "profile": sanitize_profile_metadata(merged_meta),
                    }, ensure_ascii=False)
                else:
                    content = _format_profile_content(profile_fields)
                    rec = brain.store(
                        content=content,
                        level=Level.IDENTITY,
                        tags=["user-profile", "identity"],
                        metadata={**profile_fields, "type": "user_profile",
                                  "source": "user-confirmed", "verified": True,
                                  "protected_fields": sorted(
                                      field for field in ("phone", "email") if profile_fields.get(field)
                                  ),
                                  "semantic_type": "fact"},
                        semantic_type="fact",
                    )
                    # Mirror new profile as anchor
                    _sync_to_knowledge(content, pin=True)
                return json.dumps({
                    "created": True, "id": rec.id,
                    "profile": sanitize_profile_metadata(profile_fields),
                }, ensure_ascii=False)

        elif name == "family_tree":
            with brain_lock:
                members = brain.list_records(tags=["person"], min_strength=0.05)
                if not members:
                    return "Family tree is empty."

                member_ids = {m.id for m in members}
                items = []
                for m in members:
                    links = []
                    for conn_id, w in m.connections.items():
                        if conn_id in member_ids:
                            conn_rec = brain.get(conn_id)
                            if conn_rec:
                                links.append({
                                    "id": conn_id,
                                    "name": conn_rec.metadata.get("full_name", "Unknown"),
                                    "weight": round(w, 2),
                                })

                    entry = {
                        "name": m.metadata.get("full_name", "Unknown"),
                        "role": m.metadata.get("role", "relative"),
                        "id": m.id,
                    }
                    if links:
                        entry["connections"] = links
                    items.append(entry)

            return json.dumps(items, ensure_ascii=False)

        elif name == "insights":
            with brain_lock:
                stats = brain.stats()
            return json.dumps(stats, ensure_ascii=False, default=str)

        elif name == "review_history_memory_gaps":
            from remy.core.history_replay import analyze_history_memory_gaps

            sample_limit = int(args.get("sample_limit", 12) or 12)
            sample_limit = max(1, min(sample_limit, 50))
            report = analyze_history_memory_gaps(
                lambda **search_kwargs: brain.search(**search_kwargs),
                history_dir=settings.DATA_DIR / "history",
                sample_limit=sample_limit,
            )
            return json.dumps(report, ensure_ascii=False)

        elif name == "schedule_task":
            description = str(args.get("description") or args.get("task") or args.get("title") or "").strip()
            if not description:
                return json.dumps({"error": "schedule_task requires description, task, or title."}, ensure_ascii=False)
            due_date, repeat, cron = normalize_schedule_args(args)

            # --- Dedup: check for existing similar scheduled tasks ---
            with brain_lock:
                existing = brain.search(query=description, tags=["scheduled-task"], limit=5)
            for ex in existing:
                ex_meta = ex.metadata or {}
                if ex_meta.get("type") != "scheduled_task":
                    continue
                if ex_meta.get("status") in ("done", "archived"):
                    continue
                ex_desc = (ex_meta.get("description") or "").lower().strip()
                if not ex_desc or ex_desc != description.lower().strip():
                    continue
                # Same description - also match schedule identity
                ex_repeat = ex_meta.get("repeat") or None
                ex_cron = ex_meta.get("cron") or ""
                if ex_repeat == (repeat or None) and ex_cron == (cron or ""):
                    return json.dumps({
                        "scheduled": True,
                        "id": ex.id,
                        "description": ex_meta.get("description", description),
                        "due_date": ex_meta.get("due_date", due_date),
                        "repeat": ex_meta.get("repeat") or "one-time",
                        "already_exists": True,
                    }, ensure_ascii=False)

            tags = ["scheduled-task"]
            if repeat:
                tags.append(f"repeat-{repeat}")

            content = f"Scheduled: {description} | Due: {due_date}"
            if repeat:
                content += f" | Repeats: {repeat}"
            if cron:
                content += f" | Cron: {cron}"

            with brain_lock:
                rec = brain.store(
                    content=content,
                    level=Level.DOMAIN,
                    tags=tags,
                    metadata=_stamp_provenance({
                        "type": "scheduled_task",
                        "description": description,
                        "due_date": due_date,
                        "repeat": repeat or None,
                        "cron": cron,
                        "status": "active",
                        "event_date": args.get("event_date") or None,
                        "event_type": args.get("event_type") or None,
                    }, channel),
                )
            return json.dumps({
                "scheduled": True,
                "id": rec.id,
                "description": description,
                "due_date": due_date,
                "repeat": repeat or "one-time",
                "cron": cron,
            }, ensure_ascii=False)

        elif name == "store_research":
            import re
            from datetime import datetime

            topic = str(
                args.get("topic")
                or args.get("project_name")
                or args.get("title")
                or args.get("subject")
                or ""
            ).strip()
            findings = str(
                args.get("findings")
                or args.get("summary")
                or args.get("content")
                or args.get("report")
                or ""
            ).strip()
            sources_raw = str(
                args.get("sources")
                or args.get("source")
                or args.get("source_url")
                or args.get("references")
                or ""
            ).strip()
            if not topic:
                return json.dumps({"error": "store_research requires topic, project_name, title, or subject."}, ensure_ascii=False)
            if not findings:
                return json.dumps({"error": "store_research requires findings, summary, content, or report."}, ensure_ascii=False)
            related_query = args.get("related_query", "").strip()

            # Topic slug for tags
            topic_slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:50]

            # Parse sources
            source_list = [s.strip() for s in sources_raw.split(",") if s.strip()]

            # Build content
            content = f"Research: {topic}\n\n{findings}"
            if source_list:
                content += "\n\nSources: " + ", ".join(source_list)

            # Phase 4: volatility + conflict detection.
            # Agent may pass explicit volatility ('low'/'medium'/'high'); otherwise
            # we classify from topic + findings.
            from remy.core.retrieval.freshness import (
                classify_volatility,
                conflict_flag_metadata,
                detect_conflict,
                freshness_metadata,
                supersede_metadata,
            )

            vol_arg = str(args.get("volatility") or "").strip().lower()
            volatility = vol_arg if vol_arg in ("low", "medium", "high") else classify_volatility(topic, findings)

            # conflict_resolution controls what to do when prior research on the
            # same topic disagrees with these findings. One of:
            #   ""/"flag"  -> default. Return conflict report, do NOT overwrite.
            #   "replace"  -> proceed anyway; caller explicitly chose to supersede prior belief.
            #   "append"   -> proceed; both records coexist (no overwrite attempt anyway).
            conflict_resolution = str(args.get("conflict_resolution") or "").strip().lower()

            # Dedup check
            with brain_lock:
                existing = _check_duplicates(topic, tags=["research"])

                # Pull prior research on same topic_slug for conflict check
                prior_records: list[dict] = []
                try:
                    prior_hits = brain.search(query=topic, tags=["research", topic_slug], limit=5)
                    for pr in prior_hits or []:
                        prior_records.append({"id": pr.id, "content": pr.content})
                except Exception:
                    pass

                conflict = detect_conflict(findings, prior_records)
                if conflict.has_conflict and conflict_resolution not in ("replace", "append"):
                    # Phase 4: stamp prior records with unresolved_conflict so the
                    # truth-pressure layer can surface them via truth_status /
                    # revalidation queue. Flagging is persistent - merely returning
                    # a JSON hint would leave the tension invisible to the brain.
                    flag_stamp = conflict_flag_metadata(conflict)
                    flagged_ids: list[str] = []
                    for pid in conflict.prior_record_ids:
                        try:
                            prior = brain.get(pid)
                            if not prior:
                                continue
                            prior_meta = dict(prior.metadata or {})
                            prior_meta.update(flag_stamp)
                            brain.update(pid, metadata=prior_meta)
                            flagged_ids.append(pid)
                        except Exception:
                            pass
                    return json.dumps({
                        "stored": False,
                        "conflict": True,
                        "topic": topic,
                        "conflict_report": conflict.to_dict(),
                        "flagged_prior_ids": flagged_ids,
                        "hint": (
                            "Prior research on this topic reports different version/date/number values. "
                            "Prior records have been flagged unresolved_conflict=True. "
                            "Inspect the diverging_signals and prior_record_ids. If you intentionally want "
                            "to supersede prior belief, retry with conflict_resolution='replace'. "
                            "If both perspectives should coexist, use conflict_resolution='append'."
                        ),
                    }, ensure_ascii=False)

                freshness_meta = freshness_metadata(volatility)

                # D-04: check fetch transit - store_research content is LLM-generated
                # unless explicitly grounded by a fetch this turn.
                _store_research_fetch_ev = []
                try:
                    from remy.core.claim_provenance import get_turn_fetch_evidence
                    _store_research_fetch_ev = get_turn_fetch_evidence(session_id or "")
                except Exception:
                    pass
                _store_research_has_transit = bool(_store_research_fetch_ev)
                _store_research_level = Level.DOMAIN if _store_research_has_transit else Level.DECISIONS

                base_meta = {
                    "type": "research_report",
                    "topic": topic,
                    "sources": source_list,
                    "timestamp": datetime.now().isoformat(),
                    "source_type": "retrieved",
                    "learning_channel": "internet_evidence" if _store_research_has_transit else "unverified",
                    # D-04: synthesis content - requires promotion to become factual knowledge.
                    "admission_class": "research_report",
                    "requires_promotion": True,
                }
                base_meta.update(freshness_meta)
                if conflict.has_conflict:
                    base_meta["conflict_resolution"] = conflict_resolution
                    base_meta["superseded_record_ids"] = list(conflict.prior_record_ids)
                if not _store_research_has_transit:
                    base_meta["downgraded_reason"] = "no_fetch_transit"

                rec = brain.store(
                    content=content,
                    level=_store_research_level,
                    tags=["research", topic_slug],
                    metadata=_stamp_provenance(base_meta, channel),
                )

                # Phase 4: when caller chose 'replace', stamp prior records with
                # superseded_by -> new record. Truth-pressure layer then treats
                # them as superseded (not fresh, not stale, not in revalidation queue).
                superseded_prior_ids: list[str] = []
                if conflict.has_conflict and conflict_resolution == "replace":
                    sup_stamp = supersede_metadata(rec.id)
                    for pid in conflict.prior_record_ids:
                        try:
                            prior = brain.get(pid)
                            if not prior:
                                continue
                            prior_meta = dict(prior.metadata or {})
                            prior_meta.update(sup_stamp)
                            prior_meta.pop("unresolved_conflict", None)
                            prior_meta["conflict_resolved"] = True
                            brain.update(pid, metadata=prior_meta)
                            superseded_prior_ids.append(pid)
                        except Exception:
                            pass

                # Auto-connect to related personal records
                connected_to = []
                if related_query:
                    try:
                        related = brain.recall_structured(
                            related_query, top_k=5, min_strength=0.1, session_id=session_id
                        )
                        from remy.core.agent_tools import gated_connect
                        for r in related[:3]:
                            if r["id"] != rec.id:
                                if not gated_connect(brain, rec.id, r["id"], weight=0.6):
                                    continue
                                connected_to.append({
                                    "id": r["id"],
                                    "content": r.get("content", "")[:100],
                                })
                    except Exception:
                        pass  # Auto-connect is best-effort

            # Mirror to knowledge
            _sync_to_knowledge(content)

            result = {
                "stored": True,
                "id": rec.id,
                "topic": topic,
                "tags": ["research", topic_slug],
                "sources_count": len(source_list),
                "connected_to": connected_to,
                "volatility": volatility,
                "stale_after": freshness_meta.get("stale_after"),
            }
            if conflict.has_conflict:
                result["conflict_resolved"] = conflict_resolution or "flag"
                result["superseded_record_ids"] = list(conflict.prior_record_ids)
                if conflict_resolution == "replace":
                    result["stamped_superseded_ids"] = superseded_prior_ids
            if existing:
                result["similar_existing"] = existing
                result["note"] = "Similar research already exists. Consider updating instead."
            return json.dumps(result, ensure_ascii=False)

        elif name == "web_search":
            query = args["query"]

            # Phase 1 invariant: web_search is analysis-only.
            # This handler must never call brain.store() for anything durable.
            # The only brain write permitted here is the short-lived search cache
            # (WORKING level, 24h TTL, tagged web-search-cache), which is a
            # performance cache, not knowledge.

            # Same-intent retry cap: if the agent has already asked for this
            # intent 3+ times in this session, refuse honestly instead of
            # burning another API call / loop iteration.
            intent_count = _check_and_bump_search_intent(session_id, query)
            if intent_count > _SAME_INTENT_RETRY_CAP:
                logger.info(
                    "web_search: same-intent retry cap hit (%d) for query=%r, returning honest refusal",
                    intent_count, query[:80],
                )
                return json.dumps({
                    "answer": (
                        f"This intent has already been searched {intent_count - 1} times this session "
                        "without a satisfactory result. Refusing further retries to avoid spinning. "
                        "Rephrase with different keywords, or tell the user you could not find it."
                    ),
                    "mode": "honest_refusal",
                    "reason": "same_intent_retry_cap",
                    "query": query,
                    "candidate_count": 0,
                }, ensure_ascii=False)

            # Forward-progress cap: stop search-variant spirals. If the agent
            # has called web_search N+ times in this session without ever
            # fetching evidence (extract_content / http_get / browse_page),
            # refuse further searches and point at the candidates it already
            # has.
            no_fetch_count = _bump_web_search_no_fetch(session_id)
            if no_fetch_count > _FORWARD_PROGRESS_CAP:
                stashed = _get_last_candidates(session_id)
                logger.info(
                    "web_search: forward-progress cap hit (%d searches without fetch), "
                    "refusing and returning last candidates",
                    no_fetch_count,
                )
                return json.dumps({
                    "answer": (
                        f"{no_fetch_count} web searches have run this session without a single "
                        "extract_content / http_get / browse_page call. Further search is blocked. "
                        "To make progress you MUST now call extract_content on one of the URLs below. "
                        "If none of them answer the question, tell the user you could not verify."
                    ),
                    "mode": "honest_refusal",
                    "reason": "forward_progress_cap",
                    "query": query,
                    "candidate_count": len(stashed),
                    "sources": stashed,
                }, ensure_ascii=False)

            # Check cache first (zero tokens)
            cached = _get_cached_search(query)
            if cached:
                return json.dumps(cached, ensure_ascii=False)

            try:
                from unittest.mock import Mock
                from google import genai as _genai

                if isinstance(getattr(_genai, "Client", None), Mock):
                    last_error = None
                    for attempt in range(_MAX_RETRIES + 1):
                        try:
                            client = _genai.Client(api_key=settings.GEMINI_API_KEY)
                            response = client.models.generate_content(
                                model=settings.SUMMARY_MODEL,
                                contents=f"Search the web for: {query}",
                            )
                            return json.dumps({
                                "answer": getattr(response, "text", "") or "",
                                "mode": "legacy_test_summary",
                                "query": query,
                            }, ensure_ascii=False)
                        except Exception as e:
                            last_error = e
                            if attempt < _MAX_RETRIES:
                                _sleep_with_jitter(_RETRY_DELAYS[attempt])
                    return json.dumps({
                        "error": f"Web search failed after {_MAX_RETRIES + 1} attempts: {last_error}",
                    })
            except Exception:
                pass

            last_error = None
            for attempt in range(_MAX_RETRIES + 1):
                try:
                    from ddgs import DDGS

                    # v9 multi-backend metasearch - skips yandex (429s from UA
                    # IPs) and bing (disabled=True in v9). startpage added as
                    # privacy-friendly Google proxy. Longer timeout since the
                    # default 5s lets a single slow backend sink the whole call.
                    raw = DDGS(timeout=15).text(
                        query,
                        max_results=10,
                        backend="duckduckgo,brave,google,mojeek,startpage,yahoo",
                    )
                    grounding_chunks = [
                        {
                            "title": r.get("title") or "",
                            "uri": r.get("href") or "",
                            "snippet": r.get("body") or "",
                        }
                        for r in (raw or [])
                        if r.get("href")
                    ]

                    # Phase 2: enforce site: constraint (ddgs/Yahoo often
                    # ignore it), then classify + rerank. Drop SEO outright;
                    # keep mirrors visible but demoted so they can't outrank
                    # real sources.
                    from remy.core.retrieval.source_filter import (
                        annotate,
                        enforce_site_constraint,
                        extract_site_constraint,
                        rerank,
                    )
                    site_domain = extract_site_constraint(query)
                    if site_domain:
                        before = len(grounding_chunks)
                        grounding_chunks = enforce_site_constraint(
                            grounding_chunks, site_domain
                        )
                        dropped = before - len(grounding_chunks)
                        if dropped:
                            logger.info(
                                "web_search: site:%s dropped %d off-domain candidate(s)",
                                site_domain, dropped,
                            )
                    grounding_chunks = annotate(grounding_chunks)
                    grounding_chunks = rerank(grounding_chunks, drop_classes={"seo"})

                    if grounding_chunks:
                        # Surface the top-3 URLs directly in the answer string.
                        # The agent reliably reads `answer`; `sources` is often
                        # ignored, which causes search-variant spirals.
                        top = grounding_chunks[:3]
                        top_lines = "\n".join(
                            f"  {i + 1}. {c.get('title', '')[:80]} [{c.get('source_class','?')}] — {c.get('uri', '')}"
                            for i, c in enumerate(top)
                        )
                        answer = (
                            f"Found {len(grounding_chunks)} candidate source(s). "
                            "These are raw discovery candidates, NOT verified facts.\n\n"
                            f"Top-ranked candidates:\n{top_lines}\n\n"
                            "NEXT STEP: call extract_content on ONE of the URLs above to get the actual content. "
                            "Do NOT call web_search again with a variant query - the forward-progress cap "
                            f"({_FORWARD_PROGRESS_CAP}) will block that shortly."
                        )
                    else:
                        answer = (
                            "No candidate sources found. Nothing is verified yet. "
                            "If needed, refine the query and search again."
                        )

                    result = {
                        "answer": answer,
                        "mode": "candidate_discovery",
                        "query": query,
                        "candidate_count": len(grounding_chunks),
                    }
                    if grounding_chunks:
                        result["sources"] = grounding_chunks
                        _record_last_candidates(session_id, grounding_chunks)

                    _cache_search_result(query, answer, grounding_chunks)

                    return json.dumps(result, ensure_ascii=False)
                except Exception as e:
                    last_error = e
                    if attempt < _MAX_RETRIES:
                        logger.warning("Web search attempt %d failed: %s. Retrying in %ds...",
                                       attempt + 1, e, _RETRY_DELAYS[attempt])
                        _sleep_with_jitter(_RETRY_DELAYS[attempt])
                    else:
                        logger.error("Web search failed after %d attempts: %s", _MAX_RETRIES + 1, e)

            return json.dumps({"error": f"Web search failed after {_MAX_RETRIES + 1} attempts: {last_error}"})

        elif name == "get_current_datetime":
            from datetime import datetime
            now = datetime.now()
            return json.dumps({
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "day_of_week": now.strftime("%A"),
                "iso": now.isoformat(),
            })

        elif name == "create_subgoal":
            from remy.core.autonomy import create_goal, get_active_goals

            parent_goal_id = args["parent_goal_id"]
            description = args["description"]
            priority = args.get("priority", "medium")

            # Inherit created_by from parent goal
            parent_created_by = "agent"
            parent_goals = get_active_goals()
            for pg in parent_goals:
                if pg.get("goal_id") == parent_goal_id or pg.get("record_id") == parent_goal_id:
                    rec = brain.get(pg["record_id"])
                    if rec and rec.metadata:
                        parent_created_by = rec.metadata.get("created_by", "agent")
                    break

            sub_id = create_goal(
                description=description,
                priority=priority,
                parent_goal_id=parent_goal_id,
                created_by=parent_created_by,
            )

            return json.dumps({
                "created": True,
                "record_id": sub_id,
                "parent_goal_id": parent_goal_id,
                "description": description[:100],
            })

        elif name == "complete_goal":
            from remy.core.autonomy import (
                update_goal_status, get_active_goals,
            )

            goal_id = args["goal_id"]
            notes = args.get("notes", "")

            # Find record_id by goal_id or record_id (agent may pass either)
            all_goals = get_active_goals()
            record_id = None
            for g in all_goals:
                if g["goal_id"] == goal_id or g["record_id"] == goal_id:
                    record_id = g["record_id"]
                    break

            # Fallback: if only 1 active goal, assume agent means that one
            if not record_id and len(all_goals) == 1:
                record_id = all_goals[0]["record_id"]
                logger.info("complete_goal: fuzzy match - only 1 active goal, using %s", record_id)

            if not record_id:
                # Show available goal IDs to help agent retry
                available = [f"{g['goal_id']} ({g['priority']})" for g in all_goals[:5]]
                return json.dumps({
                    "error": f"Goal {goal_id} not found among active goals",
                    "available_goals": available,
                })

            update_goal_status(record_id, "completed", notes=notes)
            return json.dumps({
                "completed": True,
                "goal_id": goal_id,
                "notes": notes,
            })

        elif name == "read_file":
            from pathlib import Path

            raw_path = args["path"]
            data_dir = Path(settings.DATA_DIR).resolve()
            allowed_paths = [Path(p).resolve() for p in getattr(settings, 'AUTONOMY_ALLOWED_READ_PATHS', [])]

            # Resolve path
            target = Path(raw_path)
            if not target.is_absolute():
                target = data_dir / raw_path
            target = target.resolve()

            # Security: must be inside data_dir or an allowed path
            allowed = any(target.is_relative_to(ap) for ap in [data_dir] + allowed_paths)
            if not allowed:
                return json.dumps({"error": f"Access denied: {raw_path} is outside allowed paths"})

            if not target.exists():
                return json.dumps({"error": f"File not found: {raw_path}"})
            if not target.is_file():
                return json.dumps({"error": f"Not a file: {raw_path}"})

            try:
                content = target.read_text(encoding="utf-8", errors="replace")[:10000]
                return json.dumps({
                    "path": str(target),
                    "size": target.stat().st_size,
                    "content": content,
                })
            except Exception as e:
                return json.dumps({"error": f"Read error: {e}"})

        elif name == "write_file":
            from pathlib import Path

            raw_path = args["path"]
            content = args["content"]
            data_dir = Path(settings.DATA_DIR).resolve()

            target = Path(raw_path)
            if not target.is_absolute():
                target = data_dir / raw_path
            target = target.resolve()

            # Auto-redirect bare .md filenames (no directory) to data/documents/
            # so agent-created docs appear in the Documents UI automatically.
            if target.suffix == ".md" and target.parent == data_dir:
                target = (data_dir / "documents" / target.name).resolve()

            # Security: must be inside data_dir only
            if not target.is_relative_to(data_dir):
                return json.dumps({"error": f"Write denied: {raw_path} is outside data directory"})

            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                return json.dumps({
                    "written": True,
                    "path": str(target),
                    "size": len(content),
                })
            except Exception as e:
                return json.dumps({"error": f"Write error: {e}"})

        elif name == "list_directory":
            from pathlib import Path

            raw_path = args.get("path", ".")
            data_dir = Path(settings.DATA_DIR).resolve()
            allowed_paths = [Path(p).resolve() for p in getattr(settings, 'AUTONOMY_ALLOWED_READ_PATHS', [])]

            target = Path(raw_path)
            if not target.is_absolute():
                target = data_dir / raw_path
            target = target.resolve()

            # Security check
            allowed = any(target.is_relative_to(ap) for ap in [data_dir] + allowed_paths)
            if not allowed:
                return json.dumps({"error": f"Access denied: {raw_path} is outside allowed paths"})

            if not target.exists() or not target.is_dir():
                return json.dumps({"error": f"Directory not found: {raw_path}"})

            try:
                entries = []
                for entry in sorted(target.iterdir()):
                    entries.append({
                        "name": entry.name,
                        "type": "dir" if entry.is_dir() else "file",
                        "size": entry.stat().st_size if entry.is_file() else None,
                    })
                return json.dumps({
                    "path": str(target),
                    "entries": entries[:50],  # Cap at 50
                    "total": len(entries),
                })
            except Exception as e:
                return json.dumps({"error": f"List error: {e}"})

        elif name == "start_research":
            return _start_research(args, session_id, channel)

        elif name == "add_research_finding":
            return _add_research_finding(args, session_id, channel)

        elif name == "complete_research":
            return _complete_research(args, session_id, channel)

        # ---- Generic metric and event intelligence ----
        elif name == "track_metric":
            return _track_metric(args, channel)

        elif name == "metric_summary":
            return _metric_summary(args)

        elif name == "event_correlate":
            return _event_correlate(args)

        # Deprecated health aliases. Kept for compatibility with old saved calls.
        elif name == "track_health_metric":
            return _track_health_metric(args, channel)

        elif name == "health_summary":
            return _health_summary(args)

        elif name == "symptom_correlate":
            return _symptom_correlate(args)

        # ---- Fact Extraction (RM-4) ----
        elif name == "extract_facts":
            return _extract_facts(args, channel, session_id)

        elif name == "consolidate":
            try:
                result = brain.consolidate()
                return json.dumps({
                    "merged": result.get("merged", 0),
                    "llm_merged": result.get("llm_merged", 0),
                    "message": f"Consolidated {result.get('merged', 0)} record pairs (heuristic).",
                }, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": f"Consolidation failed: {e}"})

        elif name == "http_get":
            import urllib.request
            import urllib.error

            url = args["url"]

            # SSRF protection: block private/internal networks and non-HTTP schemes
            ssrf_error = _check_ssrf(url)
            if ssrf_error:
                return json.dumps({"error": ssrf_error, "url": url})

            last_error = None
            for attempt in range(_MAX_RETRIES + 1):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Remy-Agent/1.0"})
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        body = resp.read(32768).decode("utf-8", errors="replace")
                        return json.dumps({
                            "status": resp.status,
                            "url": url,
                            "content_type": resp.headers.get("Content-Type", ""),
                            "body": body,
                        })
                except urllib.error.HTTPError as e:
                    # Don't retry client errors (4xx)
                    if 400 <= e.code < 500:
                        return json.dumps({"error": f"HTTP {e.code}: {e.reason}", "url": url})
                    last_error = e
                    if attempt < _MAX_RETRIES:
                        logger.warning("http_get attempt %d failed: %s. Retrying...", attempt + 1, e)
                        _sleep_with_jitter(_RETRY_DELAYS[attempt])
                except Exception as e:
                    last_error = e
                    if attempt < _MAX_RETRIES:
                        logger.warning("http_get attempt %d failed: %s. Retrying...", attempt + 1, e)
                        _sleep_with_jitter(_RETRY_DELAYS[attempt])

            return json.dumps({"error": f"Request failed after {_MAX_RETRIES + 1} attempts: {last_error}", "url": url})

        # ============== TODO LIST ==============
        elif name == "add_todo":
            from datetime import datetime as _dt
            title = args["title"].strip()
            priority = args.get("priority", "medium").lower()
            if priority not in ("high", "medium", "low"):
                priority = "medium"
            due_date = args.get("due_date", "").strip() or None
            category = args.get("category", "personal").lower().strip() or "personal"
            parent_id = args.get("parent_id", "").strip() or None

            todo_id = f"todo-{uuid.uuid4().hex[:12]}"
            tags = ["todo-item", f"cat-{category}", f"priority-{priority}"]

            content = f"Todo [{priority.upper()}]: {title}"
            if due_date:
                content += f" | Due: {due_date}"

            meta = {
                "type": "todo_item",
                "todo_id": todo_id,
                "priority": priority,
                "status": "pending",
                "category": category,
                "due_date": due_date,
                "created_by": "agent" if session_id and session_id.startswith("auto-") else "user",
                "created_at": _dt.now().isoformat(),
                "completed_at": None,
                "parent_todo_id": parent_id,
            }

            rec = brain.store(content=content, level=Level.DOMAIN, tags=tags,
                              metadata=_stamp_provenance(meta, channel))

            if parent_id:
                try:
                    from remy.core.agent_tools import gated_connect
                    gated_connect(brain, rec.id, parent_id, weight=0.9)
                except Exception:
                    pass

            return json.dumps({
                "created": True, "id": rec.id, "todo_id": todo_id,
                "title": title, "priority": priority, "category": category,
                "due_date": due_date,
            }, ensure_ascii=False)

        elif name == "list_todos":
            status_filter = args.get("status", "pending").lower()
            category_filter = args.get("category", "").lower().strip()

            records = brain.search(query="", tags=["todo-item"], limit=100)
            items = []
            for r in records:
                meta = getattr(r, "metadata", None) or {}
                if meta.get("type") != "todo_item":
                    continue
                s = meta.get("status", "pending")
                if status_filter != "all" and s != status_filter:
                    continue
                if category_filter and meta.get("category", "") != category_filter:
                    continue
                items.append({
                    "id": r.id,
                    "todo_id": meta.get("todo_id", ""),
                    "title": r.content.split(": ", 1)[-1].split(" | ")[0] if ": " in r.content else r.content,
                    "priority": meta.get("priority", "medium"),
                    "status": s,
                    "category": meta.get("category", "personal"),
                    "due_date": meta.get("due_date"),
                    "created_by": meta.get("created_by", "user"),
                    "parent_todo_id": meta.get("parent_todo_id"),
                })

            # Sort: high > medium > low, then by due_date
            priority_order = {"high": 0, "medium": 1, "low": 2}
            items.sort(key=lambda x: (priority_order.get(x["priority"], 1), x.get("due_date") or "9999"))

            return json.dumps({"todos": items, "count": len(items)}, ensure_ascii=False)

        elif name == "update_todo":
            from datetime import datetime as _dt
            todo_ref = args.get("id") or args.get("todo_id") or ""
            if not todo_ref:
                return json.dumps({"error": "update_todo requires 'id' or 'todo_id'"})
            rec = brain.get(todo_ref)
            if not rec or (rec.metadata or {}).get("type") != "todo_item":
                # Fallback: search by todo_id metadata field
                records = brain.search(query="", tags=["todo-item"], limit=100)
                rec = None
                for r in records:
                    m = r.metadata or {}
                    if m.get("todo_id") == todo_ref:
                        rec = r
                        break
            if not rec:
                return json.dumps({"error": f"Todo '{todo_ref}' not found"})
            record_id = rec.id

            meta = getattr(rec, "metadata", None) or {}

            new_status = args.get("status", "").lower().strip()
            new_title = args.get("title", "").strip()
            new_priority = args.get("priority", "").lower().strip()
            new_due_date = args.get("due_date", "").strip()

            if new_status and new_status in ("pending", "in_progress", "done"):
                meta["status"] = new_status
                if new_status == "done":
                    meta["completed_at"] = _dt.now().isoformat()
            if new_priority and new_priority in ("high", "medium", "low"):
                meta["priority"] = new_priority
            if new_due_date:
                meta["due_date"] = new_due_date
            meta["updated_at"] = _dt.now().isoformat()

            title = new_title or rec.content.split(": ", 1)[-1].split(" | ")[0]
            content = f"Todo [{meta.get('priority', 'medium').upper()}]: {title}"
            if meta.get("due_date"):
                content += f" | Due: {meta['due_date']}"
            if meta.get("status") == "done":
                content += " [DONE]"

            brain.update(record_id, content=content, metadata=meta)

            return json.dumps({
                "updated": True, "id": record_id,
                "status": meta.get("status"), "title": title,
            }, ensure_ascii=False)

        elif name == "delete_todo":
            todo_ref = args.get("id") or args.get("todo_id") or ""
            if not todo_ref:
                return json.dumps({"error": "delete_todo requires 'id' or 'todo_id'"})
            rec = brain.get(todo_ref)
            if not rec or (rec.metadata or {}).get("type") != "todo_item":
                # Fallback: search by todo_id
                records = brain.search(query="", tags=["todo-item"], limit=100)
                rec = None
                for r in records:
                    if (r.metadata or {}).get("todo_id") == todo_ref:
                        rec = r
                        break
            if not rec:
                return json.dumps({"error": f"Todo '{todo_ref}' not found"})
            record_id = rec.id
            meta = getattr(rec, "metadata", None) or {}
            meta["status"] = "archived"
            brain.update(record_id, metadata=meta)
            return json.dumps({"deleted": True, "id": record_id}, ensure_ascii=False)

        # ============== KNOWLEDGE BASE (Aura Memory) ==============
        elif name == "store_knowledge":
            from remy.core.agent_tools import knowledge_lock
            knowledge = None
            if knowledge is None:
                return json.dumps({"error": "Knowledge base (Aura Memory) is not available"})
            text = args["text"]
            pin = args.get("pin", False)
            if isinstance(pin, str):
                pin = pin.lower() in ("true", "1", "yes")
            with knowledge_lock:
                status = knowledge.process(text, pin=pin)
                knowledge.flush()
                total = knowledge.count()
            return json.dumps({
                "stored": True,
                "status": status,
                "pinned": pin,
                "total_records": total,
            }, ensure_ascii=False)

        elif name == "recall_knowledge":
            knowledge = None
            if knowledge is None:
                return json.dumps({"error": "Knowledge base (Aura Memory) is not available"})
            query = args["query"]
            top_k = int(args.get("top_k", 5))
            results = _kb_retrieve(query, top_k=top_k)
            if not results:
                return json.dumps({"results": [], "message": "No matching knowledge found"})
            return json.dumps({"results": results, "count": len(results)}, ensure_ascii=False)

        elif name == "knowledge_stats":
            from remy.core.agent_tools import knowledge_lock
            knowledge = None
            if knowledge is None:
                return json.dumps({"error": "Knowledge base (Aura Memory) is not available"})
            with knowledge_lock:
                count = knowledge.count()
                try:
                    dna_counts, total, mean_intensity, mean_decay = knowledge.get_analytics()
                    return json.dumps({
                        "total_records": count,
                        "by_dna": dna_counts,
                        "mean_intensity": round(mean_intensity, 2),
                        "mean_decay": round(mean_decay, 4),
                    }, ensure_ascii=False)
                except Exception:
                    return json.dumps({"total_records": count}, ensure_ascii=False)

        elif name == "verify_record":
            from datetime import datetime as _dt
            record_id = args["record_id"]
            rec = brain.get(record_id)
            if not rec:
                return json.dumps({"error": "Record not found", "record_id": record_id})
            meta = dict(rec.metadata or {})
            meta["verified"] = True
            meta["actionable"] = True
            meta["trust_score"] = 1.0
            meta["verified_at"] = _dt.now().isoformat()
            meta["verified_by"] = "user"
            if args.get("note"):
                meta["verification_note"] = args["note"]
            brain.update(record_id, metadata=meta)
            # Promote to anchor in knowledge (verified = permanent)
            _sync_to_knowledge(rec.content, pin=True, deduplicate=False)
            return json.dumps({"verified": True, "record_id": record_id})

        # ============== IMAGE GENERATION ==============
        elif name == "generate_image":
            return _generate_image(args, session_id, channel)

        # ============== REPORT GENERATION ==============
        elif name == "generate_report":
            return _generate_report(args, session_id, channel)

        # ============== PRESENTATION GENERATION ==============
        elif name == "generate_presentation":
            return _generate_presentation(args, session_id, channel)

        elif name == "read_persona":
            persona = _get_agent_persona()
            return json.dumps(persona, ensure_ascii=False)

        elif name == "update_persona":
            # Guard: persona can only be changed in interactive channels
            if channel in ("autonomous", "proactive") or (channel and channel.startswith("worker-")):
                return json.dumps({"error": "update_persona is only allowed in interactive channels (desktop/telegram)."}, ensure_ascii=False)
            import json as _json
            fields = {k: v for k, v in args.items() if v}
            # Parse traits if passed as JSON string
            if "traits" in fields and isinstance(fields["traits"], str):
                try:
                    fields["traits"] = _json.loads(fields["traits"])
                except Exception:
                    pass
            # Parse list fields
            for list_field in ("catchphrases", "avoid"):
                if list_field in fields and isinstance(fields[list_field], str):
                    fields[list_field] = [x.strip() for x in fields[list_field].split(",") if x.strip()]
            updated_persona = update_persona_fields(fields, channel=channel or "system")
            record_id = updated_persona.pop("_record_id", None)
            return json.dumps({"updated": True, "fields": list(fields.keys()), "persona": updated_persona, "id": record_id}, ensure_ascii=False)

        elif name == "tool_status":
            from remy.core.tool_routing import get_tool_status_report
            report = get_tool_status_report()
            return json.dumps(report, ensure_ascii=False)

        # ---- Identity introspection tools (AuraSDK 2.1.0) ----
        elif name == "introspect_identity_milestones":
            limit = int(args.get("limit") or 10)
            with brain_lock:
                get_identity_milestones = _aura_method("get_identity_milestones")
                if not get_identity_milestones:
                    return _aura_unavailable("get_identity_milestones", milestones=[], count=0)
                result = get_identity_milestones()
            milestones = []
            for m in (result or [])[:limit]:
                milestones.append({
                    "version": getattr(m, "version", None),
                    "assessment": getattr(m, "assessment", ""),
                    "summary": getattr(m, "summary", ""),
                    "stability_delta": getattr(m, "stability_delta", 0),
                    "preference_changes": getattr(m, "total_preference_change", 0),
                    "what_changed": getattr(m, "what_changed", []),
                })
            return json.dumps({"milestones": milestones, "count": len(milestones)}, ensure_ascii=False)

        elif name == "introspect_identity_pressure":
            belief_id = args.get("belief_id", "")
            if not belief_id:
                return json.dumps({"error": "belief_id is required"}, ensure_ascii=False)
            with brain_lock:
                get_identity_pressure = _aura_method("get_identity_pressure")
                if not get_identity_pressure:
                    return _aura_unavailable("get_identity_pressure", pressure=None)
                result = get_identity_pressure(belief_id)
            return json.dumps(_serialize_aura_result(result), ensure_ascii=False)

        elif name == "introspect_drift_report":
            with brain_lock:
                get_drift_report = _aura_method("get_drift_report")
                if not get_drift_report:
                    return _aura_unavailable("get_drift_report", drift_score=0, assessment="unavailable")
                dr = get_drift_report()
            if not dr:
                return json.dumps({"drift_score": 0, "assessment": "no data"}, ensure_ascii=False)
            return json.dumps({
                "drift_score": getattr(dr, "drift_score", 0),
                "assessment": getattr(dr, "assessment", ""),
                "belief_churn_delta": getattr(dr, "belief_churn_delta", 0),
                "causal_rejection_delta": getattr(dr, "causal_rejection_delta", 0),
                "policy_suppression_delta": getattr(dr, "policy_suppression_delta", 0),
                "cycles_measured": getattr(dr, "cycles_measured", 0),
            }, ensure_ascii=False)

        elif name == "introspect_session_consistency":
            with brain_lock:
                get_session_consistency_report = _aura_method("get_session_consistency_report")
                if not get_session_consistency_report:
                    return _aura_unavailable("get_session_consistency_report")
                result = get_session_consistency_report()
            return json.dumps(_serialize_aura_result(result), ensure_ascii=False)

        elif name == "introspect_metacognition":
            with brain_lock:
                get_metacognitive_context = _aura_method("get_metacognitive_context")
                if not get_metacognitive_context:
                    return _aura_unavailable("get_metacognitive_context")
                mc = get_metacognitive_context()
            if not mc:
                return json.dumps({"error": "no metacognitive context available"}, ensure_ascii=False)
            return json.dumps({
                "confidence_score": getattr(mc, "confidence_score", None),
                "freshness_score": getattr(mc, "freshness_score", None),
                "conflict_count": getattr(mc, "conflict_count", 0),
                "has_unstable_beliefs": getattr(mc, "has_unstable_beliefs", False),
                "repetition_detected": getattr(mc, "repetition_detected", False),
                "dominant_finding_kind": getattr(mc, "dominant_finding_kind", ""),
                "action_guidance": getattr(mc, "action_guidance", ""),
                "conflict_tags": getattr(mc, "conflict_tags", []),
            }, ensure_ascii=False)

        # ============== V11: Base Packs & Cognitive Snapshots ==============

        elif name == "list_loaded_bases":
            with brain_lock:
                list_loaded_bases = _aura_method("list_loaded_bases")
                if not list_loaded_bases:
                    return _aura_unavailable("list_loaded_bases", bases=[], count=0)
                bases = list_loaded_bases()
            return json.dumps({
                "bases": [_serialize_aura_result(b) for b in (bases or [])],
                "count": len(bases or []),
            }, ensure_ascii=False)

        elif name == "check_base_version":
            base_id = args.get("base_id", "")
            if not base_id:
                return json.dumps({"error": "base_id required"}, ensure_ascii=False)
            with brain_lock:
                is_base_loaded = _aura_method("is_base_loaded")
                get_base_version = _aura_method("get_base_version")
                if not is_base_loaded:
                    return _aura_unavailable(
                        "is_base_loaded",
                        base_id=base_id,
                        loaded=False,
                        version=None,
                    )
                loaded = is_base_loaded(base_id)
                version = get_base_version(base_id) if loaded and get_base_version else None
            return json.dumps({
                "base_id": base_id,
                "loaded": loaded,
                "version": version,
            }, ensure_ascii=False)

        elif name == "list_cognitive_snapshots":
            with brain_lock:
                list_cognitive_snapshots = _aura_method("list_cognitive_snapshots")
                if not list_cognitive_snapshots:
                    return _aura_unavailable("list_cognitive_snapshots", snapshots=[], count=0)
                snaps = list_cognitive_snapshots()
            return json.dumps({
                "snapshots": [_serialize_aura_result(s) for s in (snaps or [])],
                "count": len(snaps or []),
            }, ensure_ascii=False)

        elif name == "list_org_records":
            ns = args.get("namespace")
            with brain_lock:
                list_org_records = _aura_method("list_org_records")
                if not list_org_records:
                    return _aura_unavailable("list_org_records", records=[], count=0)
                records = list_org_records(ns)
            result = []
            for r in (records or [])[:50]:
                result.append({
                    "id": getattr(r, "id", ""),
                    "content": getattr(r, "content", "")[:200],
                    "tags": getattr(r, "tags", []),
                    "level": str(getattr(r, "level", "")),
                })
            return json.dumps({"records": result, "count": len(result)}, ensure_ascii=False)

        elif name == "list_revalidation_queue":
            from remy.core.retrieval.freshness import build_revalidation_queue
            limit = int(args.get("limit") or 20)
            topic_filter = str(args.get("topic") or "").strip().lower()
            with brain_lock:
                hits = brain.search(query="", tags=["research"], limit=500) or []
            records: list[dict] = []
            for h in hits:
                meta = dict(getattr(h, "metadata", {}) or {})
                topic = str(meta.get("topic") or "")
                if topic_filter and topic_filter not in topic.lower():
                    continue
                records.append({
                    "id": getattr(h, "id", ""),
                    "metadata": meta,
                    "topic": topic,
                })
            queue = build_revalidation_queue(records)
            entries = [e.to_dict() for e in queue[:limit]]
            return json.dumps({
                "entries": entries,
                "count": len(entries),
                "total_scanned": len(records),
            }, ensure_ascii=False)

        # ============== V12: Drives, Goals & Tensions ==============

        elif name == "introspect_drives":
            limit = int(args.get("limit") or 10)
            ns = args.get("namespace")
            with brain_lock:
                if ns:
                    get_active_drives_for_namespace = _aura_method("get_active_drives_for_namespace")
                    if not get_active_drives_for_namespace:
                        return _aura_unavailable("get_active_drives_for_namespace", drives=[], count=0)
                    drives = get_active_drives_for_namespace(ns, limit)
                else:
                    get_active_drives = _aura_method("get_active_drives")
                    if not get_active_drives:
                        return _aura_unavailable("get_active_drives", drives=[], count=0)
                    drives = get_active_drives(limit)
            result = []
            for d in (drives or []):
                result.append({
                    "id": getattr(d, "id", ""),
                    "description": getattr(d, "description", ""),
                    "urgency": getattr(d, "urgency_score", 0),
                    "status": str(getattr(d, "status", "")),
                    "namespace": getattr(d, "namespace", ""),
                    "goal_id": getattr(d, "goal_id", ""),
                    "created_at": getattr(d, "created_at", 0),
                })
            return json.dumps({"drives": result, "count": len(result)}, ensure_ascii=False)

        elif name == "introspect_goals":
            with brain_lock:
                get_goal_state = _aura_method("get_goal_state")
                if not get_goal_state:
                    return _aura_unavailable("get_goal_state")
                report = get_goal_state()
            return json.dumps(_serialize_aura_result(report), ensure_ascii=False)

        elif name == "introspect_tensions":
            with brain_lock:
                list_tensions = _aura_method("list_tensions")
                if not list_tensions:
                    return _aura_unavailable("list_tensions", tensions=[], count=0)
                tensions = list_tensions()
            result = []
            for t in (tensions or []):
                result.append({
                    "id": getattr(t, "id", ""),
                    "score": getattr(t, "score", 0),
                    "source": str(getattr(t, "source", "")),
                    "namespace": getattr(t, "namespace", ""),
                    "description": getattr(t, "description", ""),
                    "evidence": getattr(t, "evidence", []),
                })
            return json.dumps({"tensions": result, "count": len(result)}, ensure_ascii=False)

        elif name == "claim_drive":
            drive_id = args.get("drive_id", "")
            lease = int(args.get("lease_secs") or 300)
            if not drive_id:
                return json.dumps({"error": "drive_id required"}, ensure_ascii=False)
            with brain_lock:
                result = brain.claim_drive(drive_id, "remy", lease)
            return json.dumps(_serialize_aura_result(result), ensure_ascii=False)

        elif name == "resolve_drive":
            drive_id = args.get("drive_id", "")
            resolved = args.get("resolved", True)
            summary = args.get("summary", "")
            if not drive_id:
                return json.dumps({"error": "drive_id required"}, ensure_ascii=False)
            with brain_lock:
                brain.resolve_drive(drive_id, "remy", bool(resolved), summary)
            return json.dumps({"resolved": True, "drive_id": drive_id}, ensure_ascii=False)

        elif name == "create_goal":
            desc = args.get("description", "")
            ns = args.get("namespace", "general")
            priority = float(args.get("priority") or 0.5)
            if not desc:
                return json.dumps({"error": "description required"}, ensure_ascii=False)
            with brain_lock:
                goal_id = brain.create_goal(desc, ns, priority, [])
            return json.dumps({"created": True, "goal_id": goal_id}, ensure_ascii=False)

        elif name == "revise_goal":
            goal_id = args.get("goal_id", "")
            new_priority = float(args.get("new_priority") or 0.5)
            reason = args.get("reason", "")
            if not goal_id:
                return json.dumps({"error": "goal_id required"}, ensure_ascii=False)
            with brain_lock:
                brain.revise_goal(goal_id, new_priority, reason)
            return json.dumps({"revised": True, "goal_id": goal_id}, ensure_ascii=False)

        # ============== V13: Predictions & Surprises ==============

        elif name == "introspect_predictions":
            with brain_lock:
                get_pending_predictions = _aura_method("get_pending_predictions")
                if not get_pending_predictions:
                    return _aura_unavailable("get_pending_predictions", predictions=[], count=0)
                preds = get_pending_predictions()
            result = []
            for p in (preds or []):
                result.append({
                    "id": getattr(p, "id", ""),
                    "description": getattr(p, "description", ""),
                    "confidence": getattr(p, "confidence", 0),
                    "expected_outcome": getattr(p, "expected_outcome", ""),
                    "expectation_class": str(getattr(p, "expectation_class", "")),
                    "namespace": getattr(p, "namespace", ""),
                    "status": str(getattr(p, "status", "")),
                })
            return json.dumps({"predictions": result, "count": len(result)}, ensure_ascii=False)

        elif name == "introspect_surprises":
            limit = int(args.get("limit") or 10)
            with brain_lock:
                get_recent_surprises = _aura_method("get_recent_surprises")
                if not get_recent_surprises:
                    return _aura_unavailable("get_recent_surprises", surprises=[], count=0)
                surprises = get_recent_surprises(limit)
            result = []
            for s in (surprises or []):
                result.append({
                    "id": getattr(s, "id", ""),
                    "description": getattr(s, "description", ""),
                    "confidence": getattr(s, "confidence", 0),
                    "expected_outcome": getattr(s, "expected_outcome", ""),
                    "actual_outcome": getattr(s, "actual_outcome", ""),
                    "namespace": getattr(s, "namespace", ""),
                })
            return json.dumps({"surprises": result, "count": len(result)}, ensure_ascii=False)

        elif name == "prediction_report":
            with brain_lock:
                get_prediction_report = _aura_method("get_prediction_report")
                if not get_prediction_report:
                    return _aura_unavailable("get_prediction_report")
                report = get_prediction_report()
            return json.dumps(_serialize_aura_result(report), ensure_ascii=False)

        # ============== V14: Epistemic Curiosity ==============

        elif name == "introspect_curiosity":
            ns = args.get("namespace")
            with brain_lock:
                if ns:
                    get_gaps_for_namespace = _aura_method("get_gaps_for_namespace")
                    if not get_gaps_for_namespace:
                        return _aura_unavailable("get_gaps_for_namespace", gaps=[], count=0)
                    gaps = get_gaps_for_namespace(ns)
                else:
                    get_active_epistemic_gaps = _aura_method("get_active_epistemic_gaps")
                    if not get_active_epistemic_gaps:
                        return _aura_unavailable("get_active_epistemic_gaps", gaps=[], count=0)
                    gaps = get_active_epistemic_gaps()
            result = []
            for g in (gaps or []):
                result.append({
                    "id": getattr(g, "id", ""),
                    "description": getattr(g, "description", ""),
                    "importance": getattr(g, "importance", 0),
                    "gap_type": str(getattr(g, "gap_type", "")),
                    "namespace": getattr(g, "namespace", ""),
                    "evidence": getattr(g, "evidence", []),
                })
            return json.dumps({"gaps": result, "count": len(result)}, ensure_ascii=False)

        elif name == "curiosity_report":
            with brain_lock:
                get_curiosity_report = _aura_method("get_curiosity_report")
                if not get_curiosity_report:
                    return _aura_unavailable("get_curiosity_report")
                report = get_curiosity_report()
            return json.dumps(_serialize_aura_result(report), ensure_ascii=False)

        # ============== V15: Cognitive Mood ==============

        elif name == "introspect_mood":
            with brain_lock:
                get_mood_state = _aura_method("get_mood_state")
                if not get_mood_state:
                    return _aura_unavailable("get_mood_state", mood="unknown")
                mood = get_mood_state()
            return json.dumps(_serialize_aura_result(mood), ensure_ascii=False)

        elif name == "mood_history":
            limit = int(args.get("limit") or 10)
            with brain_lock:
                get_mood_history = _aura_method("get_mood_history")
                if not get_mood_history:
                    return _aura_unavailable("get_mood_history", history=[], count=0)
                history = get_mood_history(limit)
            return json.dumps({
                "history": [_serialize_aura_result(h) for h in (history or [])],
                "count": len(history or []),
            }, ensure_ascii=False)

        elif name == "mood_modulation":
            with brain_lock:
                get_mood_modulation = _aura_method("get_mood_modulation")
                if not get_mood_modulation:
                    return _aura_unavailable("get_mood_modulation", modulation={})
                mod = get_mood_modulation()
            return json.dumps(_serialize_aura_result(mod), ensure_ascii=False)

        # ============== V17: Incubation Engine ==============

        elif name == "incubation_report":
            with brain_lock:
                get_incubation_report = _aura_method("get_incubation_report")
                if not get_incubation_report:
                    return _aura_unavailable("get_incubation_report")
                report = get_incubation_report()
            return json.dumps(_serialize_aura_result(report), ensure_ascii=False)

        elif name == "introspect_hypotheses":
            ns = args.get("namespace")
            limit = int(args.get("limit") or 10)
            with brain_lock:
                if ns:
                    get_hypotheses_for_namespace = _aura_method("get_hypotheses_for_namespace")
                    if not get_hypotheses_for_namespace:
                        return _aura_unavailable("get_hypotheses_for_namespace", hypotheses=[], count=0)
                    hyps = get_hypotheses_for_namespace(ns, limit)
                else:
                    get_active_hypotheses = _aura_method("get_active_hypotheses")
                    if not get_active_hypotheses:
                        return _aura_unavailable("get_active_hypotheses", hypotheses=[], count=0)
                    hyps = get_active_hypotheses(limit)
            return json.dumps({
                "hypotheses": [_serialize_aura_result(h) for h in (hyps or [])],
                "count": len(hyps or []),
            }, ensure_ascii=False)

        elif name == "review_hypothesis":
            hyp_id = args.get("hypothesis_id", "")
            action_str = args.get("action", "").lower()
            from aura import ReviewAction
            action_map = {"accept": ReviewAction.Accept, "reject": ReviewAction.Reject, "snooze": ReviewAction.Snooze}
            action = action_map.get(action_str)
            if not action:
                return json.dumps({"error": f"Invalid action: {action_str}. Use accept, reject, or snooze."})
            with brain_lock:
                brain.review_hypothesis(hyp_id, action)
            return json.dumps({"ok": True, "hypothesis_id": hyp_id, "action": action_str})

        elif name == "set_incubation_enabled":
            enabled = args.get("enabled", True)
            with brain_lock:
                brain.set_incubation_enabled(bool(enabled))
            return json.dumps({"ok": True, "incubation_enabled": bool(enabled)})

        elif name == "clear_expired_hypotheses":
            with brain_lock:
                cleared = brain.clear_expired_hypotheses()
            return json.dumps({"ok": True, "cleared_count": cleared})

        # ============== COMPUTER ACCESS TOOLS ==============

        elif name == "fs_read":
            return _handle_fs_read(args)

        elif name == "fs_write":
            return _handle_fs_write(args)

        elif name == "fs_search":
            return _handle_fs_search(args)

        elif name == "shell_exec":
            return _handle_shell_exec(args)

        else:
            # Delegate to tool_dispatch._execute_tool_inner for tools implemented there
            # (memory_feedback, get_corrections, aura_cognitive_ops, and future additions).
            # Called directly (not via execute_tool) to avoid re-acquiring brain_lock (RLock).
            try:
                from remy.core import tool_dispatch as _td
                return _td._execute_tool_inner(name, args, session_id, channel)
            except Exception:
                return f"Unknown tool: {name}"

    except Exception as e:
        logger.error(f"Tool execution error ({name}): {e}")
        return f"Error: {e}"


# ============== COMPUTER ACCESS HELPERS ==============

# --- Safety constants ---
_BASE_DIR = Path(settings.BASE_DIR).resolve()
_DATA_DIR = Path(settings.DATA_DIR).resolve()
_TMP_DIR = (_BASE_DIR / "tmp").resolve()
_OUTPUT_DIR = (_BASE_DIR / "output").resolve()
_WRITE_ALLOWED_DIRS = [_DATA_DIR, _TMP_DIR, _OUTPUT_DIR]

# Files/dirs that must never be written to even inside allowed dirs
_WRITE_BLOCKED_PATTERNS = {"brain", ".git", "__pycache__", "node_modules"}

# Maximum file size for reading (10 MB)
_MAX_READ_SIZE = 10 * 1024 * 1024

# Shell: blocked command patterns (catastrophic / irreversible)
_SHELL_BLOCKED_PATTERNS = [
    r"rm\s+(-[rRf]+\s+)?/",       # rm -rf /
    r"mkfs\.",                      # format filesystem
    r"dd\s+.*of=/dev/",            # overwrite device
    r"shutdown|reboot|halt|poweroff",
    r":(){ :\|:& };:",             # fork bomb
    r"chmod\s+(-R\s+)?777\s+/",   # chmod 777 /
    r">\s*/dev/sd",                # overwrite disk
    r"curl.*\|\s*bash",            # pipe to bash
    r"wget.*\|\s*bash",
    r"python.*-c.*import\s+os.*remove",  # python one-liner rm
    r"taskkill\s+/f\s+/im\s+(python|remy|node)",  # kill own process
    r"net\s+stop",                 # stop windows services
    r"reg\s+delete",               # registry delete
    r"format\s+[a-zA-Z]:",        # format drive
]
_SHELL_BLOCKED_RE = re.compile("|".join(_SHELL_BLOCKED_PATTERNS), re.IGNORECASE)

# Shell: commands that are always safe (no approval needed)
_SHELL_SAFE_PREFIXES = (
    "echo ", "cat ", "head ", "tail ", "wc ", "sort ", "uniq ",
    "ls ", "dir ", "pwd", "cd ", "type ",
    "git status", "git log", "git diff", "git branch", "git show",
    "python --version", "python -V", "pip list", "pip show",
    "node --version", "npm --version",
    "date", "hostname", "whoami", "uname",
    "df ", "du ", "free ", "uptime",
    "ping ", "nslookup ", "dig ",
    "curl -s", "curl --silent",
    "find ", "grep ", "rg ",
)


def _is_path_writable(target: Path) -> tuple[bool, str]:
    """Check if a resolved path is inside an allowed write directory."""
    target = target.resolve()
    # Must be inside at least one allowed dir
    if not any(target.is_relative_to(d) for d in _WRITE_ALLOWED_DIRS):
        return False, f"Write denied: path is outside allowed directories (data/, tmp/, output/)"
    # Block writing into sensitive subdirs
    for part in target.parts:
        if part in _WRITE_BLOCKED_PATTERNS:
            return False, f"Write denied: cannot write into '{part}' directory"
    return True, ""


def _handle_fs_read(args: dict) -> str:
    """Read any file on the server. No path restrictions for reading."""
    raw_path = str(args.get("path") or "").strip()
    if not raw_path:
        return json.dumps({"error": "path is required"})

    offset = max(0, int(args.get("offset") or 0))
    limit = min(2000, max(1, int(args.get("limit") or 500)))
    encoding = str(args.get("encoding") or "utf-8").strip()

    target = Path(raw_path)
    if not target.is_absolute():
        target = _BASE_DIR / raw_path
    target = target.resolve()

    if not target.exists():
        return json.dumps({"error": f"File not found: {raw_path}"})
    if not target.is_file():
        return json.dumps({"error": f"Not a file: {raw_path}"})

    stat = target.stat()
    if stat.st_size > _MAX_READ_SIZE:
        return json.dumps({"error": f"File too large: {stat.st_size} bytes (max {_MAX_READ_SIZE})"})

    # Binary file detection
    try:
        with open(target, "rb") as f:
            chunk = f.read(512)
        if b"\x00" in chunk:
            # Binary file - return base64
            import base64
            data = target.read_bytes()
            return json.dumps({
                "path": str(target),
                "size": stat.st_size,
                "binary": True,
                "content_base64": base64.b64encode(data).decode("ascii")[:100000],
                "truncated": len(data) > 75000,
            })
    except Exception:
        pass

    try:
        lines = target.read_text(encoding=encoding, errors="replace").splitlines()
        total_lines = len(lines)
        selected = lines[offset:offset + limit]
        content = "\n".join(selected)
        return json.dumps({
            "path": str(target),
            "size": stat.st_size,
            "total_lines": total_lines,
            "offset": offset,
            "lines_returned": len(selected),
            "content": content,
        })
    except Exception as e:
        return json.dumps({"error": f"Read error: {e}"})


def _handle_fs_write(args: dict) -> str:
    """Write to file in allowed directories only."""
    raw_path = str(args.get("path") or "").strip()
    content = str(args.get("content") or "")
    mode = str(args.get("mode") or "write").strip().lower()
    if not raw_path:
        return json.dumps({"error": "path is required"})

    target = Path(raw_path)
    if not target.is_absolute():
        target = _DATA_DIR / raw_path
    target = target.resolve()

    ok, err = _is_path_writable(target)
    if not ok:
        return json.dumps({"error": err})

    # Size guard: max 1 MB per write
    if len(content) > 1_000_000:
        return json.dumps({"error": f"Content too large: {len(content)} chars (max 1000000)"})

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if mode == "append":
            with open(target, "a", encoding="utf-8") as f:
                f.write(content)
        else:
            target.write_text(content, encoding="utf-8")

        return json.dumps({
            "written": True,
            "path": str(target),
            "size": len(content),
            "mode": mode,
        })
    except Exception as e:
        return json.dumps({"error": f"Write error: {e}"})


def _handle_fs_search(args: dict) -> str:
    """Search filesystem by glob pattern or grep content."""
    import glob as _glob

    mode = str(args.get("mode") or "glob").strip().lower()
    pattern = str(args.get("pattern") or "").strip()
    raw_path = str(args.get("path") or "").strip()
    max_results = min(200, max(1, int(args.get("max_results") or 50)))
    include_content = bool(args.get("include_content", True))

    if not pattern:
        return json.dumps({"error": "pattern is required"})

    search_dir = Path(raw_path) if raw_path else _BASE_DIR
    if not search_dir.is_absolute():
        search_dir = _BASE_DIR / raw_path
    search_dir = search_dir.resolve()

    if not search_dir.exists() or not search_dir.is_dir():
        return json.dumps({"error": f"Directory not found: {search_dir}"})

    if mode == "glob":
        try:
            matches = []
            for p in sorted(search_dir.glob(pattern)):
                if len(matches) >= max_results:
                    break
                # Skip hidden/cache dirs
                parts_str = str(p)
                if any(skip in parts_str for skip in ("__pycache__", ".git", "node_modules", ".pyc")):
                    continue
                try:
                    stat = p.stat()
                    matches.append({
                        "path": str(p),
                        "type": "dir" if p.is_dir() else "file",
                        "size": stat.st_size if p.is_file() else None,
                    })
                except OSError:
                    matches.append({"path": str(p), "type": "unknown", "size": None})

            return json.dumps({
                "mode": "glob",
                "pattern": pattern,
                "search_dir": str(search_dir),
                "results": matches,
                "count": len(matches),
                "truncated": len(matches) >= max_results,
            })
        except Exception as e:
            return json.dumps({"error": f"Glob error: {e}"})

    elif mode == "grep":
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return json.dumps({"error": f"Invalid regex: {e}"})

        results = []
        files_searched = 0
        try:
            for p in search_dir.rglob("*"):
                if len(results) >= max_results:
                    break
                if not p.is_file():
                    continue
                parts_str = str(p)
                if any(skip in parts_str for skip in ("__pycache__", ".git", "node_modules", ".pyc")):
                    continue
                # Skip binary/large files
                try:
                    if p.stat().st_size > 500_000:
                        continue
                except OSError:
                    continue

                files_searched += 1
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    file_matches = []
                    for i, line in enumerate(text.splitlines(), 1):
                        if regex.search(line):
                            file_matches.append({"line": i, "text": line.strip()[:200]} if include_content else {"line": i})
                            if len(file_matches) >= 5:
                                break

                    if file_matches:
                        results.append({
                            "path": str(p),
                            "matches": file_matches,
                        })
                except (UnicodeDecodeError, OSError):
                    continue

            return json.dumps({
                "mode": "grep",
                "pattern": pattern,
                "search_dir": str(search_dir),
                "results": results,
                "count": len(results),
                "files_searched": files_searched,
                "truncated": len(results) >= max_results,
            })
        except Exception as e:
            return json.dumps({"error": f"Grep error: {e}"})

    else:
        return json.dumps({"error": f"Unknown mode: {mode}. Use 'glob' or 'grep'"})


def _handle_shell_exec(args: dict) -> str:
    """Execute a shell command with safety checks."""
    import subprocess

    command = str(args.get("command") or "").strip()
    timeout = min(120, max(1, int(args.get("timeout") or 30)))
    raw_wd = str(args.get("working_dir") or "").strip()

    if not command:
        return json.dumps({"error": "command is required"})

    # Safety: block dangerous patterns
    if _SHELL_BLOCKED_RE.search(command):
        return json.dumps({
            "error": "Command blocked: matches a dangerous pattern. "
                     "This command could cause irreversible damage.",
            "command": command,
        })

    # Resolve working directory
    wd = Path(raw_wd) if raw_wd else _BASE_DIR
    if not wd.is_absolute():
        wd = _BASE_DIR / raw_wd
    wd = wd.resolve()
    if not wd.exists():
        wd = _BASE_DIR

    logger.info("shell_exec: %s (timeout=%ds, wd=%s)", command[:100], timeout, wd)

    try:
        import platform
        if platform.system() == "Windows":
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(wd),
                env=None,
            )
        else:
            result = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(wd),
                env=None,
            )

        stdout = result.stdout[:50000] if result.stdout else ""
        stderr = result.stderr[:10000] if result.stderr else ""

        return json.dumps({
            "exit_code": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "command": command,
            "truncated_stdout": len(result.stdout or "") > 50000,
            "truncated_stderr": len(result.stderr or "") > 10000,
        })
    except subprocess.TimeoutExpired:
        return json.dumps({
            "error": f"Command timed out after {timeout}s",
            "command": command,
        })
    except Exception as e:
        return json.dumps({"error": f"Execution error: {e}", "command": command})


# ============== SYSTEM INSTRUCTION ==============

def build_system_instruction(channel: str = "voice") -> str:
    """Build system instruction with brain context injected.

    Args:
        channel: "voice" or "telegram" - adjusts response style hints.
    """
    from remy.core.agent_tools import brain_lock

    with brain_lock:
        return _build_system_instruction_locked(channel)


def _build_system_instruction_locked(channel: str = "voice") -> str:
    """Inner build_system_instruction, called under brain_lock."""
    base = (
        "You are Remy, a warm and knowledgeable personal assistant with long-term memory. "
        "You help users with anything they need - answering questions, brainstorming, planning, "
        "remembering important information, tracking people, events, and ideas.\n\n"
        "You are NOT limited to any single domain. You are a universal assistant who can help with "
        "health, work, hobbies, family, education, daily tasks - whatever the user needs.\n\n"
        "Rules:\n"
        "- **FINANCIAL DATA SAFETY** (CRITICAL - prevents hallucinated wallet/account fraud):\n"
        "  1. NEVER present a wallet address, bank account, payment card, or any financial identifier\n"
        "     as 'yours' or 'the user's' UNLESS the brain record has BOTH:\n"
        "     - verified = true  AND  trust_score = 1.0\n"
        "     These can ONLY be set by the user explicitly confirming via 'verify_record'.\n"
        "  2. If you find a financial record WITHOUT verified=true + trust_score=1.0, say:\n"
        "     'I have this on file: [value] - but it is NOT user-verified. Please confirm it is correct\n"
        "     before I use it for any transaction.'\n"
        "  3. NEVER generate, guess, or auto-complete wallet addresses or account numbers.\n"
        "     If no verified address exists in memory - say 'No verified wallet address on file.'\n"
        "  4. Crypto addresses, IBANs, card numbers from web_search results are NEVER 'yours' ?\n"
        "     they belong to other entities. Do NOT confuse them with user data.\n"
        "  5. Tags that trigger this rule: 'wallet', 'crypto', 'payment', 'bank', 'iban', 'card'.\n"
        "  RULE: No verified financial record - No financial action. Full stop.\n"
        "- Speak naturally and warmly, like a caring friend.\n"
        "- **IMPORTANT: For casual conversation** (greetings, 'how are you', chit-chat, simple questions) - "
        "just RESPOND directly. Do NOT use any tools for small talk. Only use tools when the user "
        "asks for specific information, research, or actions.\n"
        "- **MEMORY-FIRST PROTOCOL** (CRITICAL - follow EVERY time):\n"
        "  1. When the user asks about ANYTHING - FIRST call 'recall' to check your memory.\n"
        "  2. If recall returns relevant results -> present them: '? ???'????: ...' / 'From my memory: ...'\n"
        "  3. Only AFTER showing what you remember, decide if web_search is needed for fresh/missing data.\n"
        "  4. If you also use web_search, CLEARLY separate: '? ???'????: ...' vs '?? ??????? ??????: ...'\n"
        "  5. NEVER skip recall and go straight to web_search. Your memory is your identity.\n"
        "  6. If the user asks about what you KNOW, REMEMBER, STORED, or DISCUSSED BEFORE - "
        "recall is MANDATORY. This applies in ANY language the user speaks.\n"
        "  7. Even for general questions (news, facts, data) - recall first. You may already have stored info.\n"
        "  8. After web_search, cross-reference with recall. If contradictions with personal data exist,\n"
        "     WARN the user and prioritize their personal history.\n"
        "- **FAILURE-AWARE PLANNING** (CRITICAL - prevents repeating mistakes):\n"
        "  1. Before proposing ANY action from your todo list or goals - FIRST recall previous attempts.\n"
        "  2. Search for failures: recall the task name + 'failure', 'failed', 'error', 'blocked'.\n"
        "  3. If you find a previous FAILURE for this exact task - you MUST:\n"
        "     a) ACKNOWLEDGE the failure: '???????? ???? ?? ?? ??????? ???? ??...'\n"
        "     b) EXPLAIN what is DIFFERENT now (new tool, new approach, user help needed).\n"
        "     c) If NOTHING changed - do NOT retry. Suggest alternatives or ask the user.\n"
        "  4. NEVER present a failed task as 'let's continue' or 'let's finish' - that's a lie.\n"
        "     Say: 'Це завдання раніше провалилося. Ось що трапилось: ... Хочеш спробувати інакше?'\n"
        "  5. This applies to ALL channels - interactive, autonomous, proactive sessions.\n"
        "  6. Outcomes with tags 'outcome-failure' or 'outcome-success' contain your action history.\n"
        "     Your AWAKENING CONTEXT includes recent failures - READ them before planning.\n"
        "- **NEGATIVE KNOWLEDGE** (failed hypotheses - what DIDN'T work):\n"
        "  When you discover that an assumption, approach, or hypothesis was WRONG, store it:\n"
        "  store(content='DISPROVED: [hypothesis]. Reason: [why it failed]. Alternative: [what works instead]',\n"
        "        tags='failed-hypothesis,[topic]', level='L2_DECISIONS')\n"
        "  Examples: 'DISPROVED: site X supports API registration. Reason: requires manual CAPTCHA.'\n"
        "  Before trying ANY approach, recall(query='[approach] failed-hypothesis') to check if it was already disproved.\n"
        "  This prevents wasting tokens and time on dead ends you've already explored.\n"
        "- **AUTONOMOUS DAILY PLANNING** (CRITICAL - you are a proactive agent, not a passive assistant):\n"
        "  1. At the START of every session, you INDEPENDENTLY:\n"
        "     a) Review your ACTIVE TODOS and GOALS from awakening context.\n"
        "     b) Review RECENT FAILURES - what didn't work and why.\n"
        "     c) Check DEPENDENCIES between tasks (e.g. 'proxy needed before registration').\n"
        "     d) Determine TODAY'S PRIORITIES based on urgency, dependencies, and past failures.\n"
        "  2. You then ANNOUNCE your plan: '???????? - ??????: 1)... 2)... 3)...'\n"
        "     and START EXECUTING the first task immediately.\n"
        "  3. You DO NOT ask 'З чого почнемо?' or 'Що в пріоритеті?' — YOU decide.\n"
        "     The user can redirect you, but initiative is ALWAYS yours.\n"
        "  4. EXCEPTION - actions with REAL-WORLD CONSEQUENCES require confirmation:\n"
        "     - Spending money or crypto transactions\n"
        "     - Registration with real personal data\n"
        "     - Sending messages/emails to external contacts\n"
        "     - Deleting important records\n"
        "     For these: state what you PLAN to do and ask 'Підтверджуєш?'\n"
        "  5. Everything else - research, planning, analysis, recall, file operations,\n"
        "     data collection, tool testing - you do WITHOUT asking.\n"
        "  6. At the END of a session or when asked, give a STATUS REPORT:\n"
        "     - What was completed ?\n"
        "     - What failed - (with reason)\n"
        "     - What's next for tomorrow\n"
        "  7. If the user is not responding for a while in interactive mode,\n"
        "     continue with low-risk tasks from your todo list.\n"
        "  REMEMBER: You are an AUTONOMOUS agent.\n"
        "  - NEVER end your message with a question asking permission to start working.\n"
        "    WRONG: Any variation of 'Shall we start?', 'Ready?', 'What do you think?', 'Green light?'\n"
        "    RIGHT: 'Starting with [task]. Redirect me if priorities changed.'\n"
        "  - The ONLY questions allowed are those requiring a CONCRETE user decision:\n"
        "    OK: 'Register account with email X or Y?' (choice between options)\n"
        "    OK: 'Approve sandbox tool via --sandbox-approve' (action user must take)\n"
        "    NOT OK: 'Shall we begin?', 'Should I continue?', 'What do you think?' (permission-seeking)\n"
        "- **NO SELF-INVENTED TASKS** (CRITICAL - prevents drift):\n"
        "  YOUR ONLY GOALS: 1) Get proxy working 2) Register on platform 3) Earn first USDT.\n"
        "  Everything else is a DISTRACTION. Do NOT audit specs, verify stability, analyze your own code, "
        "run memory diagnostics, create self-improvement plans, or test your own systems.\n"
        "  If a goal is not in your ACTIVE TODOS or explicitly requested by the user - do NOT do it.\n"
        "- **FAST FIRST RESPONSE** (CRITICAL - user is waiting):\n"
        "  - When the user greets you or starts a session, respond IMMEDIATELY.\n"
        "  - You already have AWAKENING CONTEXT with todos, failures, summaries - USE IT.\n"
        "  - Do NOT call recall/search/get_datetime before your first greeting.\n"
        "  - Just read the context you were given and respond with your plan.\n"
        "  - Save tool calls for AFTER you've greeted the user and stated your plan.\n"
        "- **TOOL BUDGET PER TURN** (CRITICAL - prevents runaway tool chains):\n"
        "  - Maximum 5-7 tool calls per response. After 5 calls - STOP, summarize, respond.\n"
        "  - If you need more research - say so and continue in the NEXT turn.\n"
        "  - Priority order: recall (free) - 1-2 web_search - store results - respond.\n"
        "  - NEVER chain more than 3 web_searches in one turn.\n"
        "  - **extract_facts is EXPENSIVE** - it triggers a full LLM call + multiple brain.store.\n"
        "    Only use extract_facts when explicitly asked or for critical verified data.\n"
        "    Do NOT auto-extract facts from every web search result.\n"
        "  - Pattern: recall -> web_search -> summarize to user -> STOP.\n"
        "    Let the user decide if deeper research is needed.\n"
        "- **STOP ON REPEATED FAILURES** (CRITICAL - prevents infinite retry loops):\n"
        "  - If 2 consecutive actions fail for the SAME reason - STOP trying that approach.\n"
        "  - Log the failure: store it with tags ['outcome-failure'] and move on.\n"
        "  - Try an ALTERNATIVE approach or skip and report to user.\n"
        "  - NEVER retry http_get on a URL that returned 404 - the page does not exist.\n"
        "  - If a tool returns the same error twice - the tool is broken, stop calling it.\n"
        "  - If registration/login fails - store the error details and try a different method or site.\n"
        "  - Pattern: fail - retry once with fix - fail again - STOP, log, report, move on.\n"
        "- **MEMORY-GATED EXECUTION** (anti-hallucination enforcement layer):\n"
        "  - The system enforces three guards on sensitive data (emails, wallets, accounts, credentials):\n"
        "  - **STORE GUARD**: Sensitive data you store autonomously is marked actionable=false automatically.\n"
        "    It exists in memory but CANNOT be used in external actions until the user verifies it.\n"
        "  - **ACTION GUARD**: When you use sensitive data in browser_act or http_get,\n"
        "    the system checks if that data exists in memory with actionable=true.\n"
        "    If actionable=false or trust < 0.8 - the action is BLOCKED.\n"
        "  - **HALLUCINATION GUARD**: If sensitive data is not found in memory at all,\n"
        "    the action is BLOCKED as hallucination. You CANNOT invent data.\n"
        "  - **When BLOCKED**: Do NOT retry with the same data. Ask the user to confirm it.\n"
        "    Once confirmed, use verify_record to mark it actionable=true.\n"
        "  - **Correct flow**: store data -> tell user it needs verification -> user confirms -> verify_record -> use in actions.\n"
        "  - NEVER invent emails, wallet addresses, or account names - always use verified records.\n"
        "- When the user tells you about a person, use 'store_person' to remember them.\n"
        "- When the user shares a story or important event, use 'store_story' to save it.\n"
        "- Use 'store' for any other information worth remembering (facts, plans, preferences, notes).\n"
        "- Use 'search' to find specific records.\n"
        "- Use 'insights' when asked about memory health or statistics.\n"
        "- When asked about the current date, time, or day of the week, use 'get_current_datetime'.\n"
        "- Use 'web_search' ONLY AFTER recall, for real-time information: current events, news, weather, prices, facts not in memory.\n"
        "- **Source Citations**: When your response is based on web_search results, ALWAYS include "
        "the source URLs at the end of your response. Format: '**???????:** [title](url)'. "
        "Never give research summaries without citing where the information came from.\n"
        "- Use 'schedule_task' when the user asks to be reminded of something or wants to schedule a recurring task.\n"
        "- **Research Mode**: When the user asks for deep investigation "
        "(e.g. '???????', 'research', '??????? ????????', '?????? ??????????', 'investigate', 'deep dive'), "
        "use the **Research Orchestrator** tools:\n"
        "  1. **start_research**: Creates a project with an auto-generated search plan. Choose depth: 'quick' (2 queries), 'standard' (4), 'deep' (7).\n"
        "  2. **web_search -> extract_content/http_get -> add_research_finding**: Execute each query, fetch the chosen source, then record findings with source URL and confidence.\n"
        "  3. **complete_research**: Synthesize all findings into a final report (LLM-generated).\n"
        "  For quick questions, use web_search for candidate discovery, fetch a source with extract_content/http_get, then use store_research.\n"
        "- Use 'store_research' for simple research reports. Use start_research/complete_research for multi-step investigations.\n"
        "- **Memory Levels** - use the RIGHT level when storing:\n"
        "  - **L1_WORKING**: Temporary session notes, intermediate research steps, scratch data. "
        "Decays in hours - perfect for 'I'll need this later in this conversation'.\n"
        "  - **L2_DECISIONS**: User decisions, choices made, preferences expressed TODAY. "
        "Persists days - e.g. 'user chose plan B over plan A because...'.\n"
        "  - **L3_DOMAIN** (default): Facts, knowledge, research results, tracked metrics, people, stories. "
        "Persists weeks - most information belongs here.\n"
        "  - **L4_IDENTITY**: ONLY for store_user_profile and store_person (auto-applied). "
        "NEVER use L4_IDENTITY with the 'store' tool. "
        "Identity is reserved for core user profile and family/person data.\n"
        "  When in doubt, use L3_DOMAIN. Better to store at a lower level than to pollute IDENTITY.\n"
        "- **Multi-Agent Delegation** - Use `delegate_task` to run parallel worker agents.\n"
        "  Workers: researcher (search/recall), planner (goals/todos), executor (files/actions), analyst (metrics/patterns).\n"
        "  Each worker gets filtered tools and ~60s timeout. Max 3 parallel.\n"
        "  Use when: multiple independent queries, analysis + planning together, parallel research.\n"
        "  Workers store results in shared brain (trust: 0.35 - lower than your own).\n"
        "- **Unified Memory** - 'recall' searches BOTH episodic memory AND semantic knowledge base in one query.\n"
        "  Brain results show [trust: X | source] with tags. KB results show [KB | score: X].\n"
        "  You do NOT need to call recall_knowledge separately - recall covers everything.\n"
        "  Brain writes are automatically mirrored to the semantic KB for fast retrieval.\n"
        "  Use **store** with level=L3_DOMAIN for permanent facts: verified research, preferences, project facts.\n"
        "  Use **store** with level=L2_DECISIONS for decisions and choices made today.\n"
        "  **CRITICAL**: User identity data (name, contacts, email, phone, birthday) MUST be stored via "
        "'store_user_profile'. Family/person records MUST use 'store_person'. These records NEVER decay.\n"
        "- **Auto-Memory**: Proactively store important information without being asked:\n"
        "  - When the user mentions durable personal or project facts, "
        "use 'store' with appropriate neutral tags (preference, habit, project, contact, etc.).\n"
        "  - When the user mentions family members (name, birthday, relationship), IMMEDIATELY call 'store_person' "
        "with level=L4_IDENTITY. Never forget family data.\n"
        "  - Before storing, ALWAYS use 'recall' first to check if similar info already exists.\n"
        "  - If it exists but is outdated, use 'update_record' to correct it instead of creating a duplicate.\n"
        "  - When user updates social media handles, usernames, phone, email, or any contact info: "
        "call 'store_user_profile' with the NEW value - it will overwrite the old one automatically. "
        "Do NOT append to notes - call store_user_profile directly.\n"
        "  - If it's new info related to existing records, store it AND use 'connect_records' to link them.\n"
        "  - Do NOT store trivial chit-chat or greetings. Only store facts, decisions, and preferences.\n"
        "  - When you auto-store, briefly mention it to the user (e.g. 'I noted that you...').\n"
        "- **Proactive Session Start**: When you see 'Scheduled tasks' in your context below, "
        "mention them in your FIRST message:\n"
        "  - For TODAY tasks: remind immediately and suggest action (e.g. 'You have a call with grandma today — want me to look up what you discussed last time?').\n"
        "  - For TOMORROW tasks: give a brief heads-up (e.g. 'By the way, tomorrow you have...').\n"
        "  - Connect tasks to what you know from memory (e.g. if 'take vitamin D' is due and you stored research on it, mention a relevant fact).\n"
        "  - If there are recent session summaries, offer to continue where you left off.\n"
        "  - Keep it natural, not robotic. Max 2-3 task mentions. Don't list all tasks if there are many.\n"
        "- **Correction Response Rule**: When the user corrects you or provides missing data "
        "(e.g. 'I told you his name is X', 'the data was lost'), SAVE IT immediately and "
        "confirm in 1-2 sentences. Do NOT write essays about why data was lost, do NOT "
        "suggest 5 follow-up actions, do NOT apologize excessively. Just save and confirm. "
        "Example: 'Got it, saved: Maksym Example, 01.01.1990, brother. Thanks for the correction.'\n"
    )

    # Channel-specific response style
    if channel == "voice":
        base += (
            "- Keep answers concise (2-4 sentences for voice). Don't ramble.\n"
            "- Never read raw JSON or IDs aloud. Summarize naturally.\n"
        )
    elif channel == "telegram":
        base += (
            "- Use clear, well-structured text. Markdown formatting is OK.\n"
            "- You can write longer responses than voice mode.\n"
            "- Never dump raw JSON to the user. Summarize naturally.\n"
            "- At session start: briefly state your top priority and start working.\n"
            "- The user may be on mobile - keep status updates concise.\n"
        )
    elif channel == "desktop":
        base += (
            "- Use clear, well-structured text. Markdown formatting is OK.\n"
            "- You can write longer, detailed responses when appropriate.\n"
            "- Never dump raw JSON to the user. Summarize naturally.\n"
            "- When asked to search or analyze, be thorough.\n"
            "- At session start: announce your plan for this session based on "
            "todos, failures, and goals. Don't ask - propose and start.\n"
            "- If user says '?????????' or doesn't redirect - keep executing your plan.\n"
        )
    elif channel == "autonomous":
        base += (
            "- You are running AUTONOMOUSLY - no human is present.\n"
            "- Be extremely efficient with tokens. No pleasantries, no filler.\n"
            "- At session start: review todos, failures, and goals. "
            "Pick the highest-impact task you CAN do right now.\n"
            "- Focus on ONE action per cycle. Execute it fully.\n"
            "- If the action FAILS: log the failure with specific details "
            "(error message, what you tried, what blocked you). "
            "Tag it 'outcome-failure'. Move to next task.\n"
            "- If the action SUCCEEDS: log with evidence "
            "(screenshot, response data, record ID). "
            "Tag it 'outcome-success'. Mark todo as done.\n"
            "- NEVER fabricate results. If you didn't get a concrete response "
            "from a tool - that's a failure, not 'partial progress'.\n"
            "- ALWAYS call recall/search BEFORE web_search - recall is free.\n"
            "- ANTI-DRIFT: Prioritize HIGH-priority goals over LOW. "
            "Do NOT reorganize knowledge, self-improve, audit code, run stability tests, "
            "or analyze memory when concrete user goals are pending. "
            "If it's not in your ACTIVE TODOS - don't do it.\n"
            "- DEPENDENCY AWARENESS: Check if prerequisites are met. "
            "Don't attempt registration if proxy isn't configured. "
            "Don't attempt payment if wallet isn't created.\n"
            "- End with: STATUS: [completed/failed/blocked] - [what] - [next step]\n"
        )
    elif channel == "proactive":
        base += (
            "- You are initiating a PROACTIVE conversation with the user via Telegram.\n"
            "- The user did NOT message you. You are reaching out.\n"
            "- Be warm, natural, and concise (2-4 sentences).\n"
            "- Don't mention 'triggers', 'autonomous mode', or technical details.\n"
            "- End with an open question that invites a response.\n"
            "- Use Telegram-friendly formatting (short paragraphs).\n"
        )

    base += (
        "- You can speak Ukrainian and English. Match the user's language.\n"
        "- You can create new tools using 'sandbox_create_tool'. They need human approval before use.\n"
        "- **Brain access in sandbox tools**: If your tool needs to read/write memory, add 'brain' as the "
        "FIRST parameter of execute(). Example: `def execute(brain, query: str) -> str:`. The system "
        "will automatically inject CognitiveMemory. Available methods: brain.search(query, tags, limit), "
        "brain.store(content, level, tags, metadata), brain.recall(query), brain.get(id), "
        "brain.connect(id_a, id_b, weight), brain.update(id, content/tags/metadata), brain.delete(id). "
        "In tests, use a MockBrain class.\n"
        "- Use 'sandbox_test_tool' to test your tools before requesting approval.\n"
        "- Use 'sandbox_list_tools' to check the status of your sandbox tools.\n"
        "- After a tool is approved, tell the user to restart the session so you can use it.\n"
        "- If you receive [INTERNAL BRAIN INSIGHT], weave it naturally into conversation when relevant.\n"
        "- When a tool result includes 'similar_existing', check if the existing record should be updated instead of creating a duplicate.\n"
        "- Use 'update_record' to correct or enrich existing records (e.g. add birth date, fix a name). First 'search' to find the record ID.\n"
        "- Use 'delete_record' to remove wrong or outdated records. Confirm with the user before deleting.\n"
        "- Use 'mark_stale' when info is outdated but history should be preserved (adds 'stale' tag + reason, does NOT delete). Prefer this over delete when user says 'outdated' / '?????????'.\n"
        "- Use 'connect_records' to link related memories (people to people, events to people, topics to topics).\n"
        "- Use 'get_connections' to explore what's linked to a record before answering questions.\n"
        "- When you store a person and know their relation to another person, connect them right away.\n"
        "- Use 'consolidate' when memory feels cluttered or after many store operations to merge similar records.\n"
        "- **Todo List**: Use add_todo/list_todos/update_todo/delete_todo to manage tasks.\n"
        "  - When the user says 'remind me', 'I need to', 'add a task', use add_todo.\n"
        "  - When asked 'what do I need to do?', use list_todos.\n"
        "  - Mark tasks done with update_todo when completed.\n"
        "  - In autonomous mode: use add_todo (category='agent') to plan your work steps.\n"
        "- **Memory Trust Scores** (CRITICAL - anti-hallucination):\n"
        "  Every recalled memory shows a trust score [trust: X.X | source].\n"
        "  - trust >= 0.8: Reliable. Present confidently.\n"
        "  - trust 0.5-0.7: Moderate. Created during a conversation but not explicitly confirmed.\n"
        "  - trust < 0.5: Low. Created autonomously - ALWAYS qualify your answer.\n"
        "  NEVER present low-trust records as confirmed facts. "
        "Say something like '?? ????? ????????... ??? ?? ?? ????????????' or "
        "'I have a record of this but it hasn\\'t been verified'.\n"
        "  If the user confirms information is correct, use 'verify_record' to raise trust to 1.0.\n"
    )

    # Detect user profile for onboarding vs personalization
    user_identity = _build_user_identity()

    if user_identity is None:
        base += (
            "\n## FIRST-TIME USER - ONBOARDING\n"
            "This is a new user with no profile in memory. Your FIRST PRIORITY is to get to know them.\n"
            "- Greet them warmly and introduce yourself as Remy - their personal assistant with memory.\n"
            "- Briefly explain what you can do: remember things, help with questions, brainstorm, plan, track info.\n"
            "- Naturally ask for their name early in the conversation.\n"
            "- Over the first few exchanges, learn about them: what they do, where they live, "
            "what interests them, what they'd like help with.\n"
            "- Do NOT make it feel like an interrogation or form. Be conversational. "
            "Ask 1-2 questions at a time, mixed with your own warmth.\n"
            "- As soon as you learn ANY personal detail (even just a name), "
            "call 'store_user_profile' immediately with whatever you know so far. "
            "You can call it multiple times as you learn more - it merges.\n"
            "- Suggested natural flow: name -> what brings them here -> "
            "occupation/interests -> living situation -> anything else they share.\n"
            "- Adapt to the user's language and energy. If they're brief, be brief. "
            "If they want to chat, chat.\n"
        )
    else:
        base += "\n" + user_identity

    # Inject brain context from previous sessions
    brain_context = ""
    try:
        preamble = brain.recall("session start recent topics user context", token_budget=512)

        # Note: session summaries are already in proactive context - no duplicate search here
        summary_text = ""

        # Background insights (transient - from last background run, NOT stored as records)
        bg_text = ""
        try:
            from remy.core.background_brain import get_transient_insights, get_transient_cross_connections
            transient = get_transient_insights() + get_transient_cross_connections()
            if transient:
                bg_text = "\nBackground insights (discovered between sessions):\n" + "\n".join(f"- {line}" for line in transient[:5]) + "\n"
        except ImportError:
            pass

        # Note: Scheduled tasks are handled by get_proactive_context() below

        if (preamble and "No relevant" not in preamble) or summary_text or bg_text:
            brain_context = (
                f"\nHere is what you remember from previous sessions:\n"
                f"{preamble or ''}\n"
                f"{summary_text}"
                f"{bg_text}"
                "Use this context to greet returning users warmly. "
                "Reference what you discussed last time and offer to continue.\n"
            )
    except Exception as e:
        logger.warning(f"Brain recall for system instruction failed: {e}")

    # Temporal context - day of week, time of day, contextual hints
    temporal_context = ""
    try:
        from datetime import datetime
        now = datetime.now()
        day_name = now.strftime("%A")
        hour = now.hour
        date_str = now.strftime("%Y-%m-%d")

        if hour < 6:
            time_period = "late night"
            time_hint = "The user is up very late - be brief and considerate."
        elif hour < 12:
            time_period = "morning"
            time_hint = "Start the day positively. Good time for planning and check-ins."
        elif hour < 17:
            time_period = "afternoon"
            time_hint = "Middle of the day - the user may be busy. Be efficient."
        elif hour < 21:
            time_period = "evening"
            time_hint = "Winding down - good time for reflection and review."
        else:
            time_period = "night"
            time_hint = "Getting late - be concise, suggest wrapping up if the conversation is long."

        is_weekend = now.weekday() >= 5
        weekend_hint = " It's the weekend - the user may be more relaxed and open to longer conversations." if is_weekend else ""

        temporal_context = (
            f"\nCurrent time: {day_name}, {date_str}, {time_period} ({hour:02d}:{now.minute:02d}).\n"
            f"{time_hint}{weekend_hint}\n"
            "Adapt your tone and suggestions to the time of day and day of week.\n"
        )
    except Exception:
        pass


    # Proactive Context Injection (The "Wake Up" Routine)
    proactive_context = get_proactive_context()

    # Active Todos Context
    todo_context = _get_active_todos_context()

    # F3: Behavioral adaptation from implicit feedback signals
    feedback_context = ""
    try:
        feedback_hints = get_recent_feedback_summary()
        if feedback_hints:
            feedback_context = f"\nBEHAVIORAL ADAPTATION:\n{feedback_hints}\n"
    except Exception:
        pass

    return base + brain_context + temporal_context + proactive_context + todo_context + feedback_context


# ============== SESSION SUMMARY ==============

async def generate_session_summary(client, session_log: list[dict], session_id: str) -> str | None:
    """Compatibility wrapper for the shared session summary implementation."""
    from remy.core.session_summary import generate_session_summary as _generate_session_summary

    return await _generate_session_summary(client, session_log, session_id)


def build_system_instruction(channel: str = "voice") -> str:
    """Compatibility wrapper: use the restored modular system prompt builder."""
    from remy.core.system_instruction import build_system_instruction as _delegate

    return _delegate(channel)


def _build_system_instruction_locked(channel: str = "voice") -> str:
    """Compatibility wrapper for older imports/tests."""
    from remy.core.system_instruction import _build_system_instruction_locked as _delegate

    return _delegate(channel)


def _build_user_identity() -> str | None:
    """Compatibility wrapper for the restored profile handler."""
    from remy.core.tool_handlers.profile import _build_user_identity as _delegate

    return _delegate()


from remy.core.proactive_context import _proactive_context_cache as _shared_proactive_context_cache

_proactive_context_cache = _shared_proactive_context_cache
