"""ClaimSpan — normalized representation of a verifiable claim in a response.

Part of the response-auditor pipeline: detectors emit ClaimSpans, the resolver
looks for supporting evidence, the orchestrator decides warn/downgrade/redact.

Detectors must NOT know how to resolve evidence. They only classify *what*
evidence would be needed. The single seam is evidence_resolver.find_supporting_evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


ClaimType = Literal[
    "arxiv_id",
    "doi",
    "url_authoritative",
    "live_metric",
    "record_count",
    "belief_count",
    "entitlement",
]


class EvidenceRequirement(str, Enum):
    """What kind of evidence would support a claim.

    TOOL_CALL_WITH_ID   — session log must contain a tool result mentioning entity_hint
    TOOL_CALL_IN_TURN   — any tool call of the right kind in this turn
    FRESH_INTROSPECTION — a recent aura_cognitive_ops call (within TTL)
    TOOL_CALL_ANY       — any tool call in session log that could plausibly back this
    """

    TOOL_CALL_WITH_ID = "tool_call_with_id"
    TOOL_CALL_IN_TURN = "tool_call_in_turn"
    FRESH_INTROSPECTION = "fresh_introspection"
    TOOL_CALL_ANY = "tool_call_any"


@dataclass
class ClaimSpan:
    """A single verifiable claim extracted from a response."""

    text: str
    span: tuple[int, int]
    claim_type: ClaimType
    requires_evidence: EvidenceRequirement
    detector: str
    numeric_value: float | None = None
    entity_hint: str | None = None
    context_window: str = ""


@dataclass
class EvidenceMatch:
    """Evidence found in turn context that supports a claim."""

    source: Literal["session_log", "introspection_cache", "brain_stats"]
    tool_name: str | None = None
    matched_text: str = ""
    freshness_sec: float | None = None


AuditMode = Literal["pass", "warn", "downgrade", "redact"]


@dataclass
class AuditAction:
    """What to do about a single claim."""

    mode: AuditMode
    claim: ClaimSpan
    evidence: EvidenceMatch | None = None
    reason: str = ""
    rewrite: str | None = None


@dataclass
class AuditReport:
    """Result of auditing a full response."""

    response_text: str
    actions: list[AuditAction] = field(default_factory=list)
    rewritten_text: str | None = None

    @property
    def violations(self) -> list[AuditAction]:
        return [a for a in self.actions if a.mode != "pass"]

    @property
    def has_violations(self) -> bool:
        return any(a.mode != "pass" for a in self.actions)
