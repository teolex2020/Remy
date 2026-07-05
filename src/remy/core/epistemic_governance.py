"""
Epistemic Governance — pre-mouth guard for the organism.

This is the architectural layer that decides *whether the system has the
right to say something*, not whether the something is true. It runs BEFORE
the final answer reaches the user (and before factuality.enforce_factuality
does its post-rewrite pass).

Three orthogonal axes, never collapsed:
  - ClaimClass       what kind of statement      (legacy, from claim_provenance)
  - EpistemicStatus  how true it is for system   (new, from claim_provenance)
  - KnowledgeOrigin  where it came from          (new, from claim_provenance)

A fourth axis is derived at runtime:
  - ClaimEntitlement Allowed / RequiresEvidence / RequiresDowngrade / Forbidden

The governance flow:
  1. Extract external claims + count phantoms (via external_claim_verifier)
  2. Classify phantom ratio → soft / aggressive / block mode
  3. Decide per-response entitlement
  4. Rewrite or block the response text deterministically
  5. Emit a structured 'epistemic_downgrade' event for ACL renderer
  6. Publish turn signal so brain.store() gate sees it

This module is deliberately LLM-free. Phrasing is rendered by
acl_renderer.render_brain_voice (kind='epistemic_downgrade'), not generated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ── Gradient thresholds (user-confirmed policy) ───────────────────────────
PHANTOM_SOFT_MAX = 0.30        # < 30% → soft downgrade (prefix only)
PHANTOM_AGGRESSIVE_MAX = 0.70  # 30-70% → aggressive rewrite
# >= 70% → hard block

DowngradeMode = Literal["none", "soft", "aggressive", "block"]

# Phase A.7 Step 2 — structured governance output.
# Brain emits epistemic state only; mouth layer (A.7 Step 4) renders text.
# These enums are the brain-language vocabulary; no human-facing strings.
ReasonCode = Literal[
    "none",
    "no_external_claims",
    "all_external_grounded",
    "external_verification_failed",
    "phantom_text_markers_present",
    "verifier_error",
    "draft_too_short",
]

OperatorHint = Literal[
    "none",
    "retry_with_grounded_search",
    "narrow_question_scope",
    "request_evidence",
    "no_action",
]

AllowedAction = Literal[
    "emit_draft",
    "emit_draft_with_caveat",
    "suppress_draft_emit_uncertainty",
    "ask_narrower_question",
]


@dataclass
class GovernanceOutput:
    """Structured pre-mouth governance decision (Phase A.7 contract).

    Brain-native: contains epistemic state and routing hints only.
    Contains NO human-facing text and NO locale awareness. The mouth
    renderer (A.7 Step 4) consumes this and produces final user text.
    """
    mode: DowngradeMode = "none"
    reason_code: ReasonCode = "none"
    reason_detail: str = ""
    phantom_count: int = 0
    external_total: int = 0
    phantom_ratio: float = 0.0
    phantom_text_markers: bool = False
    contains_external_reference: bool = False
    safe_claims: list = field(default_factory=list)
    operator_hint: OperatorHint = "none"
    allowed_action: AllowedAction = "emit_draft"

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "reason_code": self.reason_code,
            "reason_detail": self.reason_detail,
            "phantom_count": self.phantom_count,
            "external_total": self.external_total,
            "phantom_ratio": self.phantom_ratio,
            "phantom_text_markers": self.phantom_text_markers,
            "contains_external_reference": self.contains_external_reference,
            "safe_claims": list(self.safe_claims),
            "operator_hint": self.operator_hint,
            "allowed_action": self.allowed_action,
        }


def _allowed_action_for_mode(mode: DowngradeMode) -> AllowedAction:
    if mode == "block":
        return "suppress_draft_emit_uncertainty"
    if mode == "aggressive":
        return "emit_draft_with_caveat"
    if mode == "soft":
        return "emit_draft_with_caveat"
    return "emit_draft"


def _operator_hint_for_mode(mode: DowngradeMode) -> OperatorHint:
    if mode == "block":
        return "retry_with_grounded_search"
    if mode == "aggressive":
        return "request_evidence"
    return "none"


@dataclass
class GovernanceDecision:
    """What the pre-mouth guard decided about a draft response."""
    mode: DowngradeMode = "none"
    phantom_count: int = 0
    external_total: int = 0
    phantom_ratio: float = 0.0
    phantom_text_markers: bool = False
    contains_external_reference: bool = False
    reason: str = ""
    # Deterministic banner text the final response will carry (may be empty
    # for mode='none'). Rendered via acl_renderer to keep locale split clean.
    banner: str = ""
    # Structured event the UI can pick up separately from the banner.
    event: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "phantom_count": self.phantom_count,
            "external_total": self.external_total,
            "phantom_ratio": self.phantom_ratio,
            "phantom_text_markers": self.phantom_text_markers,
            "contains_external_reference": self.contains_external_reference,
            "reason": self.reason,
            "banner": self.banner,
            "event": dict(self.event),
        }


def _classify_mode(
    phantom_ratio: float,
    phantom_text_markers: bool,
    has_external: bool,
    phantom_count: int,
) -> DowngradeMode:
    """Gradient ladder per locked policy:
         < 0.30  → soft
         0.30-0.70 → aggressive
         >= 0.70 → block
       phantom_text_markers (placeholder strings like 'умовне посилання')
       always bump to at least 'aggressive'.
    """
    if not has_external and not phantom_text_markers:
        return "none"
    if phantom_count == 0 and not phantom_text_markers:
        return "none"

    if phantom_ratio >= PHANTOM_AGGRESSIVE_MAX:
        return "block"
    if phantom_ratio >= PHANTOM_SOFT_MAX:
        return "aggressive"
    if phantom_text_markers:
        return "aggressive"
    return "soft"


def _render_banner(mode: DowngradeMode, decision: GovernanceDecision, locale: str) -> str:
    """Deterministic banner via ACL renderer. Keeps phrasing locale-aware
    without letting the brain emit natural language."""
    if mode == "none":
        return ""
    try:
        from remy.core.acl_renderer import Locale, render_brain_voice
        event = {
            "kind": "epistemic_downgrade",
            "severity": "urgent" if mode == "block" else (
                "notable" if mode == "aggressive" else "info"
            ),
            "payload": {
                "status": "Hypothesis" if mode != "block" else "Unknown",
                "mode": mode,
                "phantom_count": decision.phantom_count,
                "external_total": decision.external_total,
            },
        }
        decision.event = event
        rendered = render_brain_voice(event, Locale.from_str(locale))
        return rendered or ""
    except Exception:
        # Never let phrasing errors break the response pipeline.
        return ""


# ── Main entrypoint ───────────────────────────────────────────────────────


def decide_governance(
    response_text: str,
    session_log: list,
    *,
    session_id: str = "",
) -> GovernanceOutput:
    """Phase A.7 Step 2 — pure structured governance decision.

    Computes epistemic state from a draft + session_log. No locale.
    No text rendering. No human-facing strings. Brain-language only.

    Side effect: publishes turn signal so brain.store() gate sees it
    (same behavior as the legacy govern_response path).

    Mouth renderer (A.7 Step 4) is responsible for turning this output
    into final user text in the turn's surface language.
    """
    out = GovernanceOutput()

    if not response_text or len(response_text.strip()) < 8:
        out.reason_code = "draft_too_short"
        return out

    try:
        from remy.core.external_claim_verifier import verify_external_claims
        ext = verify_external_claims(response_text, session_log, live_check=False)
    except Exception as exc:
        out.reason_code = "verifier_error"
        out.reason_detail = repr(exc)
        return out

    out.phantom_count = int(ext.phantom_count)
    out.external_total = int(ext.total)
    out.phantom_text_markers = bool(ext.phantom_text_markers)
    out.contains_external_reference = out.external_total > 0
    out.phantom_ratio = (
        out.phantom_count / out.external_total
        if out.external_total > 0 else 0.0
    )

    mode = _classify_mode(
        out.phantom_ratio,
        out.phantom_text_markers,
        out.contains_external_reference,
        out.phantom_count,
    )
    out.mode = mode

    if mode == "none":
        if not out.contains_external_reference:
            out.reason_code = "no_external_claims"
        else:
            out.reason_code = "all_external_grounded"
        out.allowed_action = _allowed_action_for_mode(mode)
        out.operator_hint = _operator_hint_for_mode(mode)
        _publish_signal_from_output(session_id, out)
        return out

    if out.phantom_text_markers and out.phantom_count == 0:
        out.reason_code = "phantom_text_markers_present"
    else:
        out.reason_code = "external_verification_failed"
    out.reason_detail = (
        f"phantom {out.phantom_count}/{out.external_total} "
        f"(ratio {out.phantom_ratio:.2f}) markers={out.phantom_text_markers}"
    )

    out.allowed_action = _allowed_action_for_mode(mode)
    out.operator_hint = _operator_hint_for_mode(mode)

    _publish_signal_from_output(session_id, out)
    return out


def _publish_signal_from_output(session_id: str, out: GovernanceOutput) -> None:
    """Same effect as legacy _publish_signal but driven by GovernanceOutput.

    Kept as a thin wrapper so the structured path doesn't depend on the
    legacy GovernanceDecision shape.
    """
    if not session_id:
        return
    try:
        from remy.core.claim_provenance import record_turn_factuality_signal
        record_turn_factuality_signal(
            session_id,
            phantom_count=out.phantom_count,
            external_total=out.external_total,
            phantom_text_markers=out.phantom_text_markers,
            brain_storage_unsafe=out.mode in ("aggressive", "block"),
        )
    except Exception:
        pass


def govern_response(
    response_text: str,
    session_log: list,
    *,
    locale: str = "en",
    session_id: str = "",
) -> tuple[str, GovernanceDecision]:
    """Legacy adapter — soft / aggressive paths only (Phase A.7 Step 3).

    As of Steps 4 & 5 the block path is owned by the brain-native mouth
    (see mouth.render_block_response); agent.py routes mode='block'
    through decide_governance + the mouth and never calls this adapter
    on block turns. Soft / aggressive still use this legacy path for
    banner rendering; that cutover is deferred to a follow-up.

    Contract: delegate to decide_governance() for the epistemic state,
    project back into a legacy GovernanceDecision, prepend the locale-
    aware banner to the draft.

    Defensive behavior: if a caller still invokes this adapter with
    block-inducing text, the banner is prepended and the draft is
    returned unchanged. No hardcoded honest-uncertainty sentences live
    in this module anymore — the mouth owns that rendering. A draft
    reaching this branch with mode='block' is a caller bug; logging it
    loudly is intentional.
    """
    out = decide_governance(response_text, session_log, session_id=session_id)
    decision = _decision_from_output(out)

    if out.mode == "none":
        return response_text, decision

    banner = _render_banner(out.mode, decision, locale)
    decision.banner = banner

    if out.mode == "block":
        # Block must be routed through the mouth. If execution reaches
        # here, a caller bypassed the new contract — surface the banner
        # but do not fabricate a honest-uncertainty sentence from this
        # module. The draft is returned unchanged so the caller (or a
        # downstream audit) can notice the routing bug instead of it
        # being papered over by a locale-table fallback.
        if banner and banner not in response_text:
            return f"{banner}\n\n{response_text}", decision
        return response_text, decision

    # soft / aggressive — banner prepended to the draft.
    if banner and banner not in response_text:
        return f"{banner}\n\n{response_text}", decision
    return response_text, decision


def decision_from_output(out: GovernanceOutput) -> GovernanceDecision:
    """Public alias — Phase A.7 callers need to project structured output
    back into the legacy decision shape for history/log compatibility."""
    return _decision_from_output(out)


def _decision_from_output(out: GovernanceOutput) -> GovernanceDecision:
    """Project structured A.7 output back into legacy GovernanceDecision shape.

    Used only by the govern_response adapter. The reason field is rendered
    from (reason_code, reason_detail) to preserve the legacy 'phantom N/M ...'
    string callers may have logged or asserted on.
    """
    if out.reason_code == "no_external_claims" or out.reason_code == "all_external_grounded":
        legacy_reason = "no external citations or all grounded"
    elif out.reason_code == "verifier_error":
        legacy_reason = f"verifier_error: {out.reason_detail}"
    elif out.reason_detail:
        legacy_reason = out.reason_detail
    else:
        legacy_reason = ""

    return GovernanceDecision(
        mode=out.mode,
        phantom_count=out.phantom_count,
        external_total=out.external_total,
        phantom_ratio=out.phantom_ratio,
        phantom_text_markers=out.phantom_text_markers,
        contains_external_reference=out.contains_external_reference,
        reason=legacy_reason,
    )


