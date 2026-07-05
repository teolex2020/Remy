"""Shared memory policy helpers for exact, cognitive, and protected memory."""

from __future__ import annotations

from typing import Any

from remy.core.agent_tools import Level

PROFILE_PUBLIC_FIELDS = (
    "name",
    "age",
    "birth_date",
    "location",
    "occupation",
    "languages",
    "family",
    "personal_focus",
    "interests",
    "notes",
    "social",
)

PROFILE_PROTECTED_FIELDS = (
    "phone",
    "email",
)
PROFILE_INPUT_FIELDS = PROFILE_PUBLIC_FIELDS + PROFILE_PROTECTED_FIELDS

PROTECTED_EXACT_FIELDS = frozenset(
    {
        "phone",
        "email",
        "password",
        "api_key",
        "token",
        "wallet",
        "iban",
        "card",
        "account_number",
        "secret",
    }
)

SEMANTIC_TYPES = frozenset(
    {"fact", "decision", "preference", "contradiction", "trend", "serendipity"}
)

# Phase A.9 — Explicit Admission Classes
#
# Every record written to brain should carry one of these in metadata as
# admission_class. The class determines which recall surfaces can use the
# record as authoritative input.
#
# Allowed on factual/citation/verify surfaces:
#   grounded_external_fact, grounded_source_extract, operator_asserted
#
# Allowed on working/planning surfaces only (not factual primary):
#   working_state, plan, reflection, research_artifact, generated_analysis
#
# Never allowed as primary on any LLM-facing recall surface:
#   unverified_claim  (quarantined; inspection only)

ADMISSION_CLASSES = frozenset({
    "grounded_external_fact",   # verified from external source via tool
    "grounded_source_extract",  # verbatim extract from verified source URL/DOI
    "operator_asserted",        # explicitly confirmed by operator/user
    "working_state",            # tasks, todos, scratchpad, ephemeral state
    "plan",                     # agent plans, strategies, goals
    "reflection",               # agent self-reflection, session summaries
    "research_artifact",        # research-project / research-finding scaffolding
    "generated_analysis",       # LLM-synthesised summaries, reports, synthesis
    "unverified_claim",         # stored but quarantined; not factual substrate
})

# Classes that are safe as primary factual substrate
FACTUAL_SAFE_ADMISSION_CLASSES = frozenset({
    "grounded_external_fact",
    "grounded_source_extract",
    "operator_asserted",
})

# Classes that must never be primary factual substrate
FACTUAL_FORBIDDEN_ADMISSION_CLASSES = frozenset({
    "working_state",
    "plan",
    "reflection",
    "research_artifact",
    "generated_analysis",
    "unverified_claim",
})

# Map from record tags → default admission_class when not explicitly set.
# Used by derive_admission_class() to back-fill class for legacy records.
_TAG_TO_ADMISSION_CLASS: dict[str, str] = {
    "generated-report":        "generated_analysis",
    "research-project":        "research_artifact",
    "research-finding":        "research_artifact",
    "research":                "research_artifact",
    "research-summary":        "research_artifact",
    "session-summary":         "reflection",
    "scratchpad":              "working_state",
    "scratchpad-summary":      "working_state",
    "autonomous-outcome":      "reflection",
    "autonomous-plan":         "plan",
    "session-reflection":      "reflection",
    "proactive-session":       "reflection",
    "quarantine-unverified":   "unverified_claim",
    "claim:llm-unverified":    "unverified_claim",
    "citation-claim":          "unverified_claim",
    "todo-item":               "working_state",
    "outcome-failure":         "working_state",
    "autonomous-session-summary": "reflection",
    "background-insights":     "generated_analysis",
    "runtime-directive":       "working_state",
}


def derive_admission_class(
    metadata: dict | None,
    tags: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Return the admission_class for a record.

    Priority:
      1. Explicit admission_class in metadata (already set by writer)
      2. Tag-based derivation via _TAG_TO_ADMISSION_CLASS
      3. Safe default: "grounded_external_fact" only if verified=True,
         else "unverified_claim" for autonomous sources,
         else "working_state" as conservative fallback.
    """
    meta = metadata or {}

    # 1. Explicit
    explicit = meta.get("admission_class", "")
    if explicit and explicit in ADMISSION_CLASSES:
        return explicit

    # 2. Tag-based
    tag_set = set(tags or [])
    for tag in tag_set:
        cls = _TAG_TO_ADMISSION_CLASS.get(str(tag))
        if cls:
            return cls

    # 3. Source-based default
    source = str(meta.get("source", "") or "")
    verified = bool(meta.get("verified", False))
    if verified and source not in ("agent-autonomous", "agent-worker"):
        return "grounded_external_fact"
    if source in ("agent-autonomous", "agent-worker"):
        return "reflection"
    return "working_state"


_PREFERENCE_WORDS = frozenset({
    # Ukrainian
    "перевагу", "переваг", "віддає", "віддають", "подобається", "подобаються",
    "любить", "любить", "надає", "надають", "хоче", "хочу", "хочемо",
    "вважає", "вважаю", "обирає", "обираю", "preferує", "не любить",
    # English
    "prefer", "prefers", "preference", "favourite", "favorite", "like",
    "likes", "wants", "enjoys", "dislikes",
})

_DECISION_WORDS = frozenset({
    # Ukrainian
    "scheduled", "нагадати", "нагадай", "заплановано", "план", "вирішено",
    "вирішили", "виконати", "зробити", "треба", "потрібно", "варто",
    "задача", "завдання", "ціль", "мета", "дедлайн", "строк", "goal",
    # English
    "scheduled", "todo", "task", "action", "decided", "deadline",
    "reminder", "plan", "step", "goal", "milestone",
})

_TREND_WORDS = frozenset({
    # Ukrainian
    "тренд", "тенденція", "зростає", "падає", "збільшується", "зменшується",
    "часто", "регулярно", "постійно", "зазвичай", "як правило",
    "щодня", "щотижня", "щомісяця", "конкурент", "оновлення",
    # English
    "trend", "pattern", "increasing", "decreasing", "growing", "regularly",
    "usually", "typically", "competitor", "update", "weekly", "monthly",
})

_CONTRADICTION_WORDS = frozenset({
    # Ukrainian
    "disproved", "спростовано", "хибно", "помилково", "неправда",
    "насправді", "але насправді", "суперечить", "конфлікт", "виправлення",
    # English
    "disproved", "incorrect", "wrong", "contradicts", "false", "actually",
    "correction", "override", "conflict",
})

_SERENDIPITY_WORDS = frozenset({
    # Ukrainian
    "випадково", "несподівано", "цікаво", "між іншим", "до речі",
    # English
    "serendipity", "unexpected", "surprisingly", "interesting", "discovered",
    "found by accident", "by the way",
})


def infer_semantic_type(
    *,
    explicit: str | None = None,
    tags: list[str] | tuple[str, ...] | None = None,
    level: Any = None,
    content: str | None = None,
) -> str:
    """Infer semantic_type from explicit hint, tags, level, and content keywords."""
    # 1. explicit always wins
    if explicit:
        value = explicit.strip().lower()
        if value in SEMANTIC_TYPES:
            return value

    # 2. tags
    tag_set = {str(tag).strip().lower() for tag in (tags or []) if str(tag).strip()}
    if "contradiction" in tag_set or any(tag.startswith("outcome-failure") for tag in tag_set):
        return "contradiction"
    if "preference" in tag_set or "interests" in tag_set:
        return "preference"
    if "trend" in tag_set or "pattern" in tag_set:
        return "trend"
    if level == Level.DECISIONS or "decision" in tag_set or "todo-item" in tag_set:
        return "decision"

    # 3. content keyword scan
    if content:
        low = content.lower()
        # strip punctuation so "вважає," matches "вважає"
        import re
        words = set(re.sub(r"[^\w\s]", " ", low).split())

        if words & _CONTRADICTION_WORDS or "DISPROVED" in content:
            return "contradiction"

        if words & _PREFERENCE_WORDS:
            return "preference"

        # "Scheduled:" prefix is a strong decision signal
        if low.startswith("scheduled") or low.startswith("[") and "scheduled" in low[:40]:
            return "decision"
        if words & _DECISION_WORDS:
            return "decision"

        if words & _TREND_WORDS:
            return "trend"

        if words & _SERENDIPITY_WORDS:
            return "serendipity"

    return "fact"


def protected_fields_for_record(
    metadata: dict | None,
    tags: list[str] | tuple[str, ...] | None = None,
) -> set[str]:
    meta = dict(metadata or {})
    protected = {
        str(item)
        for item in (meta.get("protected_fields") or meta.get("_protected_fields") or [])
        if str(item)
    }
    if meta.get("type") == "user_profile" or "user-profile" in set(tags or []):
        protected.update(field for field in PROFILE_PROTECTED_FIELDS if meta.get(field))
    protected.update(field for field in PROTECTED_EXACT_FIELDS if meta.get(field))
    return protected


def sanitize_memory_metadata(
    metadata: dict | None,
    *,
    tags: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Hide protected exact fields in broad retrieval outputs."""
    meta = dict(metadata or {})
    protected = protected_fields_for_record(meta, tags=tags)
    if not protected:
        return meta
    for field in protected:
        if field in meta and meta[field]:
            meta[field] = "[protected]"
    meta["protected_fields"] = sorted(protected)
    meta.pop("_protected_fields", None)
    return meta


def sanitize_memory_content(
    content: str | None,
    *,
    metadata: dict | None = None,
    tags: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Redact exact protected values from broad retrieval content."""
    text = str(content or "")
    protected = protected_fields_for_record(metadata or {}, tags=tags)
    if not protected or not text:
        return text

    meta = dict(metadata or {})
    for field in protected:
        raw_value = meta.get(field)
        if raw_value is None:
            continue
        raw_text = str(raw_value).strip()
        if raw_text:
            text = text.replace(raw_text, "[protected]")
    return text


def protected_payload(
    metadata: dict | None,
    *,
    tags: list[str] | tuple[str, ...] | None = None,
    requested_fields: list[str] | tuple[str, ...] | None = None,
) -> dict[str, str]:
    """Return exact protected values for explicit secure retrieval."""
    meta = dict(metadata or {})
    protected = protected_fields_for_record(meta, tags=tags)
    if requested_fields:
        wanted = {str(field).strip() for field in requested_fields if str(field).strip()}
        protected &= wanted
    payload: dict[str, str] = {}
    for field in sorted(protected):
        value = meta.get(field)
        if value is None:
            continue
        value_text = str(value).strip()
        if value_text:
            payload[field] = value_text
    return payload
