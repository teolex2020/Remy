"""Evidence resolver — single seam between detectors and evidence lookup.

Detectors produce ClaimSpans with a requires_evidence enum. The resolver is
the ONLY component that knows how to translate that enum into a concrete search
over session_log, introspection_cache, or brain state.

This keeps detectors cheap (regex + classification) and centralizes the crosscut
logic that tends to grow hairy — one place to fix bugs, one place to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from remy.core import introspection_cache
from remy.core.retrieval.claim_spans import (
    ClaimSpan,
    EvidenceMatch,
    EvidenceRequirement,
)


# Tool names whose results legitimately back each claim type.
# Detectors set claim_type; resolver maps that to which tools qualify.
_TOOL_BACKING: dict[str, frozenset[str]] = {
    "arxiv_id": frozenset({"web_search", "extract_content", "http_get", "start_research"}),
    "doi": frozenset({"web_search", "extract_content", "http_get", "start_research"}),
    "url_authoritative": frozenset({"web_search", "extract_content", "http_get"}),
    "record_count": frozenset({"aura_cognitive_ops", "stats", "insights", "recall"}),
    "belief_count": frozenset({"aura_cognitive_ops"}),
    "entitlement": frozenset(
        {"web_search", "extract_content", "http_get", "start_research", "recall", "search"}
    ),
}


@dataclass
class TurnContext:
    """Everything the resolver needs to answer "is this claim supported?".

    session_log: list of {type, tool, args, result_full, ...} dicts accumulated
                 during the current turn (agent.py format).
    session_id:  used to look up introspection cache entries.
    """

    session_log: list[dict[str, Any]] = field(default_factory=list)
    session_id: str | None = None

    def tool_calls(self, tool_names: frozenset[str] | None = None) -> list[dict[str, Any]]:
        out = []
        for entry in self.session_log:
            if entry.get("type") != "tool_call":
                continue
            if tool_names is not None and entry.get("tool") not in tool_names:
                continue
            out.append(entry)
        return out


def find_supporting_evidence(
    turn: TurnContext, claim: ClaimSpan
) -> EvidenceMatch | None:
    """Return evidence matching *claim* in *turn*, or None if unsupported.

    This is the only function detectors should rely on. If a new evidence
    source is added (e.g. belief_store), extend this function — not detectors.
    """
    req = claim.requires_evidence

    if req is EvidenceRequirement.FRESH_INTROSPECTION:
        entry = introspection_cache.get_fresh(turn.session_id)
        if entry is None:
            return None
        import time as _time

        return EvidenceMatch(
            source="introspection_cache",
            tool_name=f"aura_cognitive_ops:{entry.op}",
            matched_text=str(entry.result)[:200],
            freshness_sec=_time.time() - entry.ts,
        )

    backing_tools = _TOOL_BACKING.get(claim.claim_type)
    if not backing_tools:
        return None

    calls = turn.tool_calls(backing_tools)
    if not calls:
        return None

    if req is EvidenceRequirement.TOOL_CALL_WITH_ID:
        hint = (claim.entity_hint or "").strip().lower()
        if not hint:
            return None
        for call in calls:
            blob = (str(call.get("result_full", "")) + " " + str(call.get("args", ""))).lower()
            if hint in blob:
                return EvidenceMatch(
                    source="session_log",
                    tool_name=call.get("tool"),
                    matched_text=hint,
                )
        return None

    if req in (EvidenceRequirement.TOOL_CALL_IN_TURN, EvidenceRequirement.TOOL_CALL_ANY):
        call = calls[-1]
        return EvidenceMatch(
            source="session_log",
            tool_name=call.get("tool"),
            matched_text=str(call.get("result_full", ""))[:200],
        )

    return None
