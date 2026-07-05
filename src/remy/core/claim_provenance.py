"""
Claim-level provenance — the contract for whether an LLM-emitted statement
may enter brain.store() as a fact, or must be quarantined as an unverified
citation claim.

This is a DIFFERENT axis from the existing agent-channel provenance in
remy.core.provenance (which tracks who/which-agent asked for the write).
That layer answers "who typed this?". THIS layer answers "what kind of
knowledge is this?".

Both are stored on a record's metadata; they are orthogonal and both
required to decide whether a record is trustworthy.

Two dimensions:

1. ClaimClass — WHAT kind of statement this is
     - fact           verified external or direct user input
     - inference      derived by the system from existing facts
     - proposal       hypothesis / research lead / idea
     - citation_claim mention of an external entity awaiting verification

2. ClaimProvenance — HOW this specific claim was anchored
     - user                    operator typed it
     - tool_verified           a tool_call this turn confirmed the external entity
     - llm_unverified          only the LLM produced it; no external anchor
     - system_inferred         brain logic derived it from existing records

Golden rule: a record with claim_class="fact" requires provenance ∈
{user, tool_verified, system_inferred}. LLM-only factual claims are
routed to quarantine (claim_class="citation_claim",
provenance="llm_unverified").

This module is deliberately dependency-free so it can be imported from
factuality, brain_tools, and background_brain without cycles.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Literal


ClaimClass = Literal["fact", "inference", "proposal", "citation_claim"]
ClaimProvenanceKind = Literal[
    "user",
    "tool_verified",
    "llm_unverified",
    "system_inferred",
]


# ── Epistemic axes (Phase A.4.2) ───────────────────────────────────────────
#
# Three orthogonal axes. Never collapse them into one field.
#
#   ClaimClass       "what kind of statement is this"         (legacy, kept)
#   EpistemicStatus  "how true is this for the system"        (new)
#   KnowledgeOrigin  "where did it come from"                 (new)
#
# A fourth axis — ClaimEntitlement — is derived at runtime from the three
# above + current evidence signals. It is NOT stored on records.

EpistemicStatus = Literal[
    "Observed",       # directly seen through a trusted channel this turn
    "Supported",      # independently supported by multiple trusted signals
    "Believed",       # stabilized internal belief from the belief layer
    "Hypothesis",     # plausible, not yet verified
    "Contradicted",   # meaningful conflict exists — active signal, not inert
    "Unknown",        # insufficient basis
]

KnowledgeOrigin = Literal[
    "UserReported",    # came from the user directly
    "ToolObserved",    # produced by a tool call in this turn
    "RuntimeObserved", # produced by the runtime itself (timers, system state)
    "Imported",        # loaded from a pack / KB at startup
    "Generated",       # LLM text without tool grounding
    "Inferred",        # derived inside the brain (belief/concept/abstraction)
]

ClaimEntitlement = Literal[
    "Allowed",            # free to speak as-is
    "RequiresEvidence",   # can speak only if evidence is attached
    "RequiresDowngrade",  # must be re-phrased as tentative
    "Forbidden",          # must not appear in the mouth
]


# Mapping from legacy ClaimClass+ClaimProvenanceKind to the new (status, origin)
# pair. Used when promoting old records into the new axes without migrations.
def derive_epistemic_axes(
    claim_class: ClaimClass,
    provenance_kind: ClaimProvenanceKind,
) -> tuple[EpistemicStatus, KnowledgeOrigin]:
    """Map existing (claim_class, provenance) to (EpistemicStatus, KnowledgeOrigin).

    This is lossy in one direction only: new records can carry explicit
    EpistemicStatus/KnowledgeOrigin metadata; old records are interpreted
    through this map when read.
    """
    if provenance_kind == "user":
        return ("Observed", "UserReported")
    if provenance_kind == "tool_verified":
        if claim_class == "fact":
            return ("Supported", "ToolObserved")
        return ("Believed", "ToolObserved")
    if provenance_kind == "system_inferred":
        if claim_class == "inference":
            return ("Believed", "Inferred")
        return ("Hypothesis", "Inferred")
    # llm_unverified
    if claim_class == "citation_claim":
        return ("Hypothesis", "Generated")
    if claim_class == "proposal":
        return ("Hypothesis", "Generated")
    if claim_class == "inference":
        return ("Hypothesis", "Generated")
    # fact + llm_unverified is the dangerous combination — gate should have
    # quarantined it, but if we see it on read, classify as Hypothesis so
    # downstream phrasing is tentative.
    return ("Hypothesis", "Generated")


def resolve_entitlement(
    status: EpistemicStatus,
    origin: KnowledgeOrigin,
    *,
    has_tool_evidence: bool = False,
    has_source_url: bool = False,
    contains_external_reference: bool = False,
) -> ClaimEntitlement:
    """Decide what the mouth is allowed to do with this claim.

    Orthogonal to validity — this answers 'does the system have the RIGHT
    to say this?', not 'is this true?'.
    """
    if status == "Contradicted":
        return "RequiresDowngrade"
    if status == "Unknown":
        return "Forbidden"

    if contains_external_reference:
        # Any external-entity claim must be anchored.
        if status in ("Observed", "Supported") and (has_tool_evidence or has_source_url):
            return "Allowed"
        if status == "Believed" and (has_tool_evidence or has_source_url):
            return "Allowed"
        if has_tool_evidence or has_source_url:
            return "RequiresEvidence"
        return "Forbidden"

    if status == "Observed":
        return "Allowed"
    if status == "Supported":
        return "Allowed"
    if status == "Believed":
        return "Allowed"
    if status == "Hypothesis":
        return "RequiresDowngrade"
    return "Forbidden"


@dataclass
class ClaimProvenance:
    kind: ClaimProvenanceKind
    # tool_verified fields
    tool: str = ""
    locator: str = ""            # URL / DOI / arxiv id / record id that anchors the claim
    verified_at: float = 0.0     # unix seconds; 0 = not verified
    # system_inferred field
    based_on: list[str] = field(default_factory=list)
    # Free-form note, shown in debug / quarantine reasons
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def user(cls, note: str = "") -> "ClaimProvenance":
        return cls(kind="user", note=note)

    @classmethod
    def tool_verified(cls, tool: str, locator: str, note: str = "") -> "ClaimProvenance":
        return cls(
            kind="tool_verified",
            tool=tool,
            locator=locator,
            verified_at=time.time(),
            note=note,
        )

    @classmethod
    def llm_unverified(cls, note: str = "") -> "ClaimProvenance":
        return cls(kind="llm_unverified", note=note)

    @classmethod
    def system_inferred(cls, based_on: list[str], note: str = "") -> "ClaimProvenance":
        return cls(
            kind="system_inferred",
            based_on=list(based_on or []),
            note=note,
        )


# ── Storage policy ─────────────────────────────────────────────────────────


# Sentinel tags the brain uses to recognise quarantined items.
TAG_QUARANTINE = "quarantine-unverified"
TAG_CITATION_CLAIM = "citation-claim"
TAG_CLAIM_USER = "claim:user"
TAG_CLAIM_TOOL = "claim:tool-verified"
TAG_CLAIM_LLM = "claim:llm-unverified"
TAG_CLAIM_SYSTEM = "claim:system-inferred"

_KIND_TAG = {
    "user": TAG_CLAIM_USER,
    "tool_verified": TAG_CLAIM_TOOL,
    "llm_unverified": TAG_CLAIM_LLM,
    "system_inferred": TAG_CLAIM_SYSTEM,
}


def claim_provenance_tag(prov: ClaimProvenance) -> str:
    return _KIND_TAG[prov.kind]


@dataclass
class StorageDecision:
    """What the brain should do with a candidate record."""
    allow_factual_store: bool
    effective_claim_class: ClaimClass
    tags_to_add: list[str] = field(default_factory=list)
    quarantine: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def decide_storage(
    *,
    requested_class: ClaimClass,
    provenance: ClaimProvenance,
    has_phantom_citations: bool = False,
    phantom_ratio: float = 0.0,
) -> StorageDecision:
    """Map (claim_class, provenance, verifier signals) → StorageDecision.

    The policy:
      • requested=fact + prov=user/tool_verified/system_inferred → allow fact
      • requested=fact + prov=llm_unverified                     → quarantine as citation_claim
      • requested=fact + phantom_ratio >= 0.5                    → quarantine as citation_claim
      • requested=inference/proposal                             → allow as-is (tag provenance)
      • requested=citation_claim                                 → always quarantine
    """
    tags = [claim_provenance_tag(provenance)]

    if requested_class == "citation_claim":
        tags.extend([TAG_QUARANTINE, TAG_CITATION_CLAIM])
        return StorageDecision(
            allow_factual_store=False,
            effective_claim_class="citation_claim",
            tags_to_add=tags,
            quarantine=True,
            reason="citation_claim always quarantined until verification",
        )

    if requested_class == "fact":
        if provenance.kind == "llm_unverified":
            tags.extend([TAG_QUARANTINE, TAG_CITATION_CLAIM])
            return StorageDecision(
                allow_factual_store=False,
                effective_claim_class="citation_claim",
                tags_to_add=tags,
                quarantine=True,
                reason="fact requested but provenance=llm_unverified",
            )
        if has_phantom_citations and phantom_ratio >= 0.5:
            tags.extend([TAG_QUARANTINE, TAG_CITATION_CLAIM])
            return StorageDecision(
                allow_factual_store=False,
                effective_claim_class="citation_claim",
                tags_to_add=tags,
                quarantine=True,
                reason=f"phantom-citation ratio {phantom_ratio:.2f} >= 0.5",
            )
        return StorageDecision(
            allow_factual_store=True,
            effective_claim_class="fact",
            tags_to_add=tags,
            quarantine=False,
            reason="fact with trustworthy provenance",
        )

    # D-03 fix: inference/proposal + llm_unverified must not bypass the DOMAIN
    # gate. An LLM can add an "inference" tag to any content; that tag must not
    # silently grant DOMAIN write rights when there is no fetch grounding.
    if requested_class in ("inference", "proposal") and provenance.kind == "llm_unverified":
        tags.extend([TAG_QUARANTINE])
        return StorageDecision(
            allow_factual_store=False,
            effective_claim_class=requested_class,
            tags_to_add=tags,
            quarantine=True,
            reason=f"{requested_class} with provenance=llm_unverified requires fetch grounding for DOMAIN",
        )

    return StorageDecision(
        allow_factual_store=True,
        effective_claim_class=requested_class,
        tags_to_add=tags,
        quarantine=False,
        reason=f"{requested_class} allowed with provenance={provenance.kind}",
    )


def claim_metadata(prov: ClaimProvenance, claim_class: ClaimClass) -> dict:
    """Return a dict to merge into brain.store(metadata=...)."""
    return {
        "claim_class": claim_class,
        "claim_provenance": prov.to_dict(),
    }


# ── Session-scoped phantom-citation signal ────────────────────────────────
#
# enforce_factuality() runs per turn and produces brain_storage_unsafe +
# phantom counts. The store() gate runs inside a tool call *during the same
# turn* but has no direct handle on that report. We bridge them with a small
# thread-safe registry keyed by session_id. The agent layer pushes a snapshot
# after enforce_factuality; the store gate reads it; stale entries expire.

import threading

_TURN_SIGNAL_LOCK = threading.Lock()
_TURN_SIGNALS: dict[str, dict] = {}
_TURN_SIGNAL_TTL = 120.0  # seconds


def record_turn_factuality_signal(
    session_id: str,
    *,
    phantom_count: int = 0,
    external_total: int = 0,
    phantom_text_markers: bool = False,
    brain_storage_unsafe: bool = False,
) -> None:
    """Store the last factuality signal for this session so the store gate
    can refuse LLM-written facts produced in a phantom-heavy turn."""
    if not session_id:
        return
    now = time.time()
    with _TURN_SIGNAL_LOCK:
        _TURN_SIGNALS[session_id] = {
            "phantom_count": int(phantom_count),
            "external_total": int(external_total),
            "phantom_text_markers": bool(phantom_text_markers),
            "brain_storage_unsafe": bool(brain_storage_unsafe),
            "recorded_at": now,
        }
        # Evict stale entries; bounded size.
        expired = [sid for sid, sig in _TURN_SIGNALS.items()
                   if now - sig.get("recorded_at", 0) > _TURN_SIGNAL_TTL]
        for sid in expired:
            _TURN_SIGNALS.pop(sid, None)


def get_turn_factuality_signal(session_id: str) -> dict | None:
    """Return the last recorded signal for session_id, or None if missing/stale."""
    if not session_id:
        return None
    now = time.time()
    with _TURN_SIGNAL_LOCK:
        sig = _TURN_SIGNALS.get(session_id)
        if not sig:
            return None
        if now - sig.get("recorded_at", 0) > _TURN_SIGNAL_TTL:
            _TURN_SIGNALS.pop(session_id, None)
            return None
        return dict(sig)


def clear_turn_factuality_signal(session_id: str) -> None:
    with _TURN_SIGNAL_LOCK:
        _TURN_SIGNALS.pop(session_id, None)



_FETCH_EVIDENCE_LOCK = threading.Lock()
_TURN_FETCH_EVIDENCE: dict[str, list[dict]] = {}
_TURN_FETCH_EVIDENCE_TTL = 120.0


def _prune_turn_fetch_evidence_locked(now: float) -> None:
    expired = [
        sid
        for sid, items in _TURN_FETCH_EVIDENCE.items()
        if not items or now - max(float(item.get("recorded_at", 0) or 0) for item in items) > _TURN_FETCH_EVIDENCE_TTL
    ]
    for sid in expired:
        _TURN_FETCH_EVIDENCE.pop(sid, None)


def record_turn_fetch_evidence(
    session_id: str,
    *,
    tool: str,
    url: str,
    title: str = "",
    site: str = "",
) -> None:
    if not session_id or not url:
        return
    now = time.time()
    item = {
        "tool": str(tool or ""),
        "url": str(url),
        "title": str(title or ""),
        "site": str(site or ""),
        "recorded_at": now,
    }
    with _FETCH_EVIDENCE_LOCK:
        items = [
            existing for existing in _TURN_FETCH_EVIDENCE.get(session_id, [])
            if now - float(existing.get("recorded_at", 0) or 0) <= _TURN_FETCH_EVIDENCE_TTL
        ]
        if not any(str(existing.get("url", "")) == item["url"] for existing in items):
            items.append(item)
        _TURN_FETCH_EVIDENCE[session_id] = items[-20:]
        _prune_turn_fetch_evidence_locked(now)


def get_turn_fetch_evidence(session_id: str) -> list[dict]:
    if not session_id:
        return []
    now = time.time()
    with _FETCH_EVIDENCE_LOCK:
        items = [
            item for item in _TURN_FETCH_EVIDENCE.get(session_id, [])
            if now - float(item.get("recorded_at", 0) or 0) <= _TURN_FETCH_EVIDENCE_TTL
        ]
        if items:
            _TURN_FETCH_EVIDENCE[session_id] = items
        else:
            _TURN_FETCH_EVIDENCE.pop(session_id, None)
        _prune_turn_fetch_evidence_locked(now)
        return [dict(item) for item in items]


def clear_turn_fetch_evidence(session_id: str) -> None:
    with _FETCH_EVIDENCE_LOCK:
        _TURN_FETCH_EVIDENCE.pop(session_id, None)
