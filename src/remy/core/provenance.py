"""
Provenance & Memory Guards — trust scoring, source tracking, sensitive data protection.

Three-layer memory-gated execution:
  1. STORE GUARD — sensitive data from autonomous agents gets actionable=false
  2. ACTION GUARD — external tools blocked unless data has actionable=true
  3. HALLUCINATION GUARD — data not in memory = blocked
"""

import logging
import re


def _get_brain():
    """Lazy accessor — reads brain from brain_tools (supports test patching)."""
    import remy.core.brain_tools as _bt

    return _bt.brain


logger = logging.getLogger("BrainTools")

# ============== PROVENANCE ==============

_SOURCE_TRUST: dict[str, float] = {
    "user-confirmed": 1.0,
    "agent-interactive": 0.7,
    "system": 0.6,
    "agent-autonomous": 0.4,
    "agent-worker": 0.35,
    "agent": 0.5,
}

# Epistemological source type — HOW the data was obtained.
# This is the key distinction between "I recorded this" vs "I found this via search".
# recorded  = stored during real-time user interaction or observation
# retrieved = fetched from external source (web search, API) and stored
# inferred  = derived by LLM reasoning (fact extraction, insights, synthesis)
# generated = created by agent without external source (goals, plans, summaries)
SOURCE_TYPES = frozenset({"recorded", "retrieved", "inferred", "generated"})


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


def _stamp_provenance(
    metadata: dict | None, channel: str | None, tags: list[str] | None = None
) -> dict:
    """Merge provenance fields into existing metadata dict.

    Uses setdefault so explicit source/verified/trust_score in existing metadata
    won't be overwritten. Also stamps volatility from tags and timestamp.
    Auto-infers source_type from tags if not explicitly set.
    """
    meta = dict(metadata or {})
    prov = _get_provenance(channel)
    meta.setdefault("source", prov["source"])
    meta.setdefault("verified", prov["verified"])
    meta.setdefault("trust_score", prov["trust_score"])
    # P0 prep: stamp volatility + timestamp for scoring at recall time
    if tags is not None:
        meta.setdefault("volatility", _infer_volatility(tags))
    from datetime import datetime as _dt

    meta.setdefault("timestamp", _dt.now().isoformat())
    # Auto-infer source_type if not explicitly set
    if "source_type" not in meta:
        meta["source_type"] = _infer_source_type(meta, tags)
    return meta


def _pop_source_type(meta: dict) -> str:
    """Extract source_type from metadata for use as a top-level SDK field.

    Removes source_type from metadata dict and returns it.
    If not present, defaults to "recorded".
    Used during the migration from metadata-based to SDK-native source_type.
    """
    return meta.pop("source_type", "recorded")


def _infer_source_type(meta: dict, tags: list[str] | None = None) -> str:
    """Infer epistemological source_type from metadata and tags.

    Priority: explicit source_type > tag-based inference > channel-based default.
    """
    _tags = set(tags or [])
    # Research/web-sourced data
    if _tags & {"web-search-cache", "research-finding", "research-project"}:
        return "retrieved"
    # LLM-generated content
    if _tags & {"extracted-fact", "consolidated-meta", "session-reflection", "research-synthesis"}:
        return "inferred"
    # Agent-generated plans/goals
    if _tags & {
        "autonomous-goal",
        "action-plan",
        "autonomous-outcome",
        "scheduled-task",
        "todo-item",
    }:
        return "generated"
    # Source-based fallback
    source = meta.get("source", "")
    if source == "user-confirmed":
        return "recorded"
    if source == "agent-interactive":
        return "recorded"
    if source == "agent-autonomous":
        return "generated"
    if source == "agent-worker":
        return "generated"
    return "recorded"  # default for interactive


# Source authority multipliers — how much we trust each source at recall time.
# Separate from _SOURCE_TRUST (base trust at store time).
_SOURCE_AUTHORITY: dict[str, float] = {
    "user-telegram": 1.2,
    "user-desktop": 1.2,
    "user-voice": 1.2,
    "user-confirmed": 1.2,
    "agent-interactive": 1.0,
    "system": 0.9,
    "agent": 0.85,
    "agent-autonomous": 0.75,
    "agent-worker": 0.7,
    "agent-inference": 0.65,
}


# Volatility inference — used at store time to stamp records.
_STABLE_TAGS = frozenset(
    {"identity", "contact", "credential", "financial", "person", "agent-persona"}
)
_VOLATILE_TAGS = frozenset(
    {
        "market",
        "price",
        "scheduled-task",
        "todo-item",
        "outcome-failure",
        "outcome-success",
        "web-search-cache",
        "autonomous-outcome",
        "session-summary",
        "session-reflection",
        "action-plan",
        "feedback-signal",
        "autonomous-session",
    }
)


def _infer_volatility(tags: list[str] | None) -> str:
    """Classify record volatility from its tags: stable / volatile / moderate."""
    tag_set = set(tags or [])
    if tag_set & _STABLE_TAGS:
        return "stable"
    if tag_set & _VOLATILE_TAGS:
        return "volatile"
    return "moderate"


def _compute_effective_trust(metadata: dict, now: float) -> float:
    """Compute recall-time effective trust with multi-factor scoring.

    P0 factors:
    - Source Authority: user-stated > interactive > autonomous
    - Recency Boost: +0.2 for today, linearly decays to 0 over 7 days
    """
    try:
        trust = float(metadata.get("trust_score", 0.5))
    except (TypeError, ValueError):
        trust = 0.5
    source = metadata.get("source", "")

    # P0-1: Source Authority multiplier
    authority = _SOURCE_AUTHORITY.get(source, 0.85)

    # P0-2: Recency Boost — fresh records get +0.2, decays over 7 days
    timestamp_str = metadata.get("timestamp") or metadata.get("created_at", "")
    try:
        from datetime import datetime

        ts = datetime.fromisoformat(timestamp_str).timestamp()
    except Exception:
        ts = now - 86400 * 14  # assume 14 days old if unknown
    age_days = max(0, (now - ts) / 86400)
    recency_boost = max(0.0, 0.2 * (1 - age_days / 7))

    effective = (trust + recency_boost) * authority
    return round(max(0.05, min(1.0, effective)), 2)


# ============== MEMORY-GATED EXECUTION ==============

# Tools that perform external actions where hallucinated data is dangerous.
_TRUST_ENFORCED_TOOLS = frozenset(
    {
        "browser_act",  # typing into forms — emails, passwords, wallets
        "http_get",  # API calls with user data in URLs
    }
)

# Arg keys that may contain sensitive user data worth validating
_SENSITIVE_KEYS = frozenset(
    {
        "email",
        "wallet",
        "address",
        "account",
        "username",
        "password",
        "token",
        "api_key",
        "phone",
    }
)

# Patterns that detect sensitive data in free-text fields
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_WALLET_RE = re.compile(r"0x[a-fA-F0-9]{40}")  # Ethereum-style

# Tags that indicate sensitive/identity content for store guard
_SENSITIVE_TAGS = frozenset(
    {
        "identity",
        "financial",
        "wallet",
        "credential",
        "account",
        "registration",
        "login",
        "email",
        "proxy",
        "api-key",
    }
)


# --- GUARD 0: AUTO-PROTECT — add consolidation-safe tags to sensitive content ---

_PHONE_RE = re.compile(r"\b\d[\d\s\-]{8,}\d\b")
_PASSWORD_RE = re.compile(r"(?:password|пароль|pass|pwd)\s*[:=]\s*\S+", re.IGNORECASE)
_API_KEY_RE = re.compile(r"(?:api.?key|token|secret)\s*[:=]\s*\S+", re.IGNORECASE)


def _auto_protect_tags(content: str, tags: list) -> list:
    """Auto-add protective tags to content containing sensitive values.

    Records with these tags are excluded from consolidation, preventing
    data loss when LLM summarization drops specific values like phone numbers.
    Mutates and returns the tags list.
    """
    if _PHONE_RE.search(content):
        if "contact" not in tags:
            tags.append("contact")
    if _EMAIL_RE.search(content):
        if "contact" not in tags:
            tags.append("contact")
    if _WALLET_RE.search(content):
        if "financial" not in tags:
            tags.append("financial")
    if _PASSWORD_RE.search(content) or _API_KEY_RE.search(content):
        if "credential" not in tags:
            tags.append("credential")
    return tags


# --- GUARD 1: STORE GUARD ---


def _apply_store_guard(content: str, tags: list, channel: str | None) -> dict:
    """Auto-set actionable=false for sensitive data stored by autonomous agents.

    Interactive channels (desktop, telegram, voice) store with actionable=true
    because the user is present and implicitly approves.
    Autonomous/worker channels store with actionable=false — requires verify_record.

    Returns metadata fields to merge.
    """
    is_interactive = channel in ("desktop", "telegram", "voice")

    # Check if content or tags indicate sensitive data
    has_sensitive_tags = bool(set(tags) & _SENSITIVE_TAGS)
    has_sensitive_content = bool(_EMAIL_RE.search(content) or _WALLET_RE.search(content))

    if not has_sensitive_tags and not has_sensitive_content:
        return {}  # Not sensitive — no guard needed

    if is_interactive:
        # User is present — data is implicitly approved
        return {"actionable": True}
    else:
        # Autonomous — NOT actionable until user verifies
        return {"actionable": False}


# --- GUARD 2 & 3: ACTION GUARD + HALLUCINATION GUARD ---


def _validate_action_data(name: str, args: dict) -> str | None:
    """Memory-gated execution: block actions with unverified or hallucinated data.

    Three checks per sensitive value:
    1. HALLUCINATION GUARD — value not found in memory at all -> blocked
    2. TRUST GUARD — found but actionable=false or trust < 0.8 -> blocked
    3. PASS — found with actionable=true or (trust >= 0.8 and no actionable field)

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
        # For "text" field in browser_act — check if it contains email or wallet
        if key == "text":
            email_match = _EMAIL_RE.search(value)
            if email_match:
                values_to_check.append(("email", email_match.group()))
            wallet_match = _WALLET_RE.search(value)
            if wallet_match:
                values_to_check.append(("wallet", wallet_match.group()))

    if not values_to_check:
        return None  # No sensitive data in args — proceed

    import time as _time

    now = _time.time()

    for key_name, value in values_to_check:
        # Search brain for this exact value
        records = _get_brain().search(query=value, limit=10)

        # Find records that contain this value — prefer verified/actionable ones
        matches = []
        for r in records:
            if value.lower() in (r.content or "").lower():
                matches.append(r)

        # Pick best match: actionable=True > verified=True > highest trust > first found
        match = None
        if matches:
            for r in matches:
                m = r.metadata or {}
                if m.get("actionable") is True:
                    match = r
                    break
            if match is None:
                for r in matches:
                    m = r.metadata or {}
                    if m.get("verified") is True:
                        match = r
                        break
            if match is None:
                match = matches[0]

        # --- GUARD 3: HALLUCINATION GUARD ---
        if match is None:
            logger.warning(
                "HALLUCINATION GUARD: %s='%s' in tool '%s' — not found in memory",
                key_name,
                value[:30],
                name,
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
            # Explicitly marked as actionable — pass
            continue
        elif actionable is False:
            # Explicitly NOT actionable — blocked
            try:
                trust = float(meta.get("trust_score", 0))
            except (TypeError, ValueError):
                trust = 0.0
            logger.warning(
                "TRUST GUARD: %s='%s' in tool '%s' — actionable=false (trust=%.2f)",
                key_name,
                value[:30],
                name,
                trust,
            )
            return (
                f"TRUST GUARD: {key_name}='{value}' exists in memory but is NOT verified "
                f"(actionable=false, trust={trust:.2f}). Cannot use for external actions.\n"
                f"Action: Ask the user to confirm this data is correct, "
                f"then use verify_record to mark it as verified."
            )
        else:
            # Legacy record (no actionable field) — fall back to trust check
            if meta.get("verified") is True:
                continue  # Verified by user — pass
            trust = _compute_effective_trust(meta, now)
            if trust >= 0.8:
                continue  # High trust — pass
            # Low trust, no actionable field — blocked
            logger.warning(
                "TRUST GUARD: %s='%s' in tool '%s' — trust %.2f < 0.8, no actionable flag",
                key_name,
                value[:30],
                name,
                trust,
            )
            return (
                f"TRUST GUARD: {key_name}='{value}' exists but trust is too low "
                f"(trust={trust:.2f}, needs >= 0.8). Cannot use for external actions.\n"
                f"Action: Ask the user to confirm, then use verify_record."
            )

    return None  # All sensitive data verified — proceed
