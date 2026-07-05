"""
Canonical ingestion entry points for the brain (Phase 2).

Two — and only two — functions exist to admit knowledge-bearing records into
durable brain state:

  ingest_grounded_evidence(...)   for tool-fetched external sources
  ingest_operator_assertion(...)  for explicit user/operator assertions

Every other write path must either route through these functions or must not
produce factual-safe records at all (working_state / plan / reflection /
research_artifact / generated_analysis / unverified_claim live at lower
surfaces and are never primary factual substrate).

Design constraints (Phase 2):
  • admission_class is restricted to the canonical taxonomy in memory_policy.
    Ad-hoc strings are rejected at the API boundary — the module does not
    tolerate them, by design.
  • grounded_evidence requires: source_url + turn fetch evidence this session
    + a canonicalised match between the claimed source and a fetched URL.
  • operator_assertion requires: user-direct channel (desktop/telegram/voice)
    AND explicit user attribution tag (user-profile / user-statement /
    from-user). "Came from chat" alone is not enough.
  • These functions return a typed IngestionResult; the caller performs the
    actual brain.store() using the resolved level/tags/metadata. That keeps
    the policy layer brain-free and unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from remy.core.agent_tools import Level
from remy.core.claim_provenance import get_turn_fetch_evidence
from remy.core.memory_policy import FACTUAL_SAFE_ADMISSION_CLASSES


# ── Public taxonomy ──────────────────────────────────────────────────────────

# Canonical factual-safe classes accepted by ingest_grounded_evidence.
# Intentionally a closed set — adding new classes here is a deliberate act.
GroundedExtractClass = Literal["grounded_external_fact", "grounded_source_extract"]

OPERATOR_ASSERTED_CLASS = "operator_asserted"

# User-direct channels where operator_asserted can originate.
_USER_DIRECT_CHANNELS = frozenset({"desktop", "telegram", "voice"})

# Tags that mark explicit user attribution on a store call.
_USER_ATTRIBUTION_TAGS = frozenset({"user-profile", "user-statement", "from-user"})


# ── Result type ──────────────────────────────────────────────────────────────

IngestionStatus = Literal["admitted", "quarantined", "rejected"]


@dataclass
class IngestionResult:
    """What the ingestion layer decided for a candidate record.

    Admitted:    caller should brain.store(content, level, tags, metadata).
                 level is DOMAIN (knowledge-bearing).
    Quarantined: caller may store for inspection but the record MUST NOT enter
                 factual recall surfaces. level is WORKING with quarantine tags.
    Rejected:    caller must not store anything. Surface 'reason' to the LLM
                 as an error so it can correct the call (e.g. fetch first).
    """
    status: IngestionStatus
    content: str = ""
    level: object = field(default_factory=lambda: Level.WORKING)
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    reason: str = ""

    @property
    def admitted(self) -> bool:
        return self.status == "admitted"

    @property
    def quarantined(self) -> bool:
        return self.status == "quarantined"

    @property
    def rejected(self) -> bool:
        return self.status == "rejected"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _canonicalize_url(url: str) -> str:
    return (url or "").strip().rstrip("/ ").lower()


def _fetch_evidence_matches(session_id: str, source_url: str) -> dict | None:
    """Return the fetch-evidence item whose URL canonicalises to source_url,
    or None if no such fetch happened this turn."""
    target = _canonicalize_url(source_url)
    if not target:
        return None
    for item in get_turn_fetch_evidence(session_id or ""):
        if _canonicalize_url(str(item.get("url") or "")) == target:
            return item
    return None


def _merge_tags(base: list[str], extras: list[str] | None) -> list[str]:
    seen = set()
    out: list[str] = []
    for tag in list(base) + list(extras or []):
        tag = str(tag).strip()
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


# ── Grounded evidence ingestion ──────────────────────────────────────────────

def ingest_grounded_evidence(
    *,
    content: str,
    source_url: str,
    session_id: str,
    channel: str | None,
    extract_class: GroundedExtractClass,
    extra_tags: list[str] | None = None,
    extra_meta: dict | None = None,
    confidence: float = 0.8,
) -> IngestionResult:
    """Canonical entry point for tool-grounded external evidence.

    Admits into DOMAIN only if:
      • content is non-empty
      • source_url is non-empty
      • extract_class is in FACTUAL_SAFE_ADMISSION_CLASSES (enforced here)
      • a turn fetch for source_url was recorded this session

    Otherwise rejects (no silent downgrade — the caller must know to fetch
    first). This is deliberate: grounded ingestion is the load-bearing path,
    and ambiguity here is what Phase 1 closed. Quiet fallbacks re-open it.
    """
    if not content or not content.strip():
        return IngestionResult(
            status="rejected",
            reason="ingest_grounded_evidence requires non-empty content",
        )
    if not source_url or not source_url.strip():
        return IngestionResult(
            status="rejected",
            reason="ingest_grounded_evidence requires source_url; "
                   "LLM synthesis without a source cannot be grounded knowledge",
        )
    if extract_class not in FACTUAL_SAFE_ADMISSION_CLASSES:
        return IngestionResult(
            status="rejected",
            reason=(
                f"extract_class must be one of "
                f"{sorted(FACTUAL_SAFE_ADMISSION_CLASSES)}; got {extract_class!r}"
            ),
        )
    if extract_class == OPERATOR_ASSERTED_CLASS:
        return IngestionResult(
            status="rejected",
            reason="operator_asserted must go through ingest_operator_assertion, "
                   "not ingest_grounded_evidence",
        )

    match = _fetch_evidence_matches(session_id, source_url)
    if match is None:
        return IngestionResult(
            status="rejected",
            reason=(
                "source_url is not anchored by a fetch this turn. "
                "Call extract_content or http_get on the URL first, then retry. "
                "(tools/channel may not call record_turn_fetch_evidence for you.)"
            ),
        )

    fetch_tool = str(match.get("tool") or "")
    site = str(match.get("site") or "")
    title = str(match.get("title") or "")

    meta: dict = {
        "admission_class": extract_class,
        "learning_channel": "internet_evidence",
        "source_url": source_url,
        "source_anchored": True,
        "source_site": site,
        "source_title": title,
        "fetch_tool": fetch_tool,
        "confidence": confidence,
        "ingested_at": datetime.now().isoformat(),
        "ingestion_api": "ingest_grounded_evidence",
    }
    if channel:
        meta["channel"] = channel
    if extra_meta:
        for k, v in extra_meta.items():
            meta.setdefault(k, v)

    tags = _merge_tags(
        ["grounded-evidence", "claim:tool-verified"],
        extra_tags,
    )

    return IngestionResult(
        status="admitted",
        content=content.strip(),
        level=Level.DOMAIN,
        tags=tags,
        metadata=meta,
    )


# ── Operator-asserted ingestion ──────────────────────────────────────────────

def ingest_operator_assertion(
    *,
    content: str,
    channel: str | None,
    session_id: str,
    extra_tags: list[str] | None = None,
    extra_meta: dict | None = None,
    confidence: float = 0.9,
) -> IngestionResult:
    """Canonical entry point for explicit operator/user assertions.

    Admits into DOMAIN only if:
      • content is non-empty
      • channel is a user-direct channel (desktop / telegram / voice)
      • extra_tags carry an explicit user attribution tag
        (user-profile / user-statement / from-user)

    The tag requirement is deliberate: being invoked from desktop alone is
    not enough. The caller must prove the user said 'remember this as true',
    not 'I'm having a conversation and this sentence happened to appear'.

    Unlike grounded ingestion, this path does NOT require fetch evidence —
    the user IS the source. session_id is accepted for symmetry and may be
    used for future audit trails.
    """
    if not content or not content.strip():
        return IngestionResult(
            status="rejected",
            reason="ingest_operator_assertion requires non-empty content",
        )
    if channel not in _USER_DIRECT_CHANNELS:
        return IngestionResult(
            status="rejected",
            reason=(
                f"operator_asserted requires a user-direct channel "
                f"(one of {sorted(_USER_DIRECT_CHANNELS)}); got channel={channel!r}"
            ),
        )
    attribution = _USER_ATTRIBUTION_TAGS & set(extra_tags or [])
    if not attribution:
        return IngestionResult(
            status="rejected",
            reason=(
                f"operator_asserted requires an explicit user attribution tag: "
                f"one of {sorted(_USER_ATTRIBUTION_TAGS)}. "
                "Chat context alone is not assertion."
            ),
        )

    meta: dict = {
        "admission_class": OPERATOR_ASSERTED_CLASS,
        "learning_channel": "operator_direct",
        "operator_channel": channel,
        "operator_attribution": sorted(attribution),
        "confidence": confidence,
        "ingested_at": datetime.now().isoformat(),
        "ingestion_api": "ingest_operator_assertion",
    }
    if extra_meta:
        for k, v in extra_meta.items():
            meta.setdefault(k, v)

    tags = _merge_tags(
        ["operator-asserted", "claim:user"],
        extra_tags,
    )

    return IngestionResult(
        status="admitted",
        content=content.strip(),
        level=Level.DOMAIN,
        tags=tags,
        metadata=meta,
    )
