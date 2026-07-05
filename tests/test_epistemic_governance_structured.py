"""
Phase A.7 — structured governance contract tests.

Behavior assertions on decide_governance(). No exact-string checks,
no locale assertions, no English/Ukrainian vocabulary expectations.
Verifies the brain emits structured epistemic state and nothing else.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from remy.core.epistemic_governance import (
    GovernanceOutput,
    decide_governance,
    decision_from_output,
)


class _FakeExt:
    def __init__(
        self,
        *,
        phantom_count: int,
        total: int,
        phantom_text_markers: bool = False,
    ):
        self.phantom_count = phantom_count
        self.total = total
        self.phantom_text_markers = phantom_text_markers


def _patch_verifier(ext: _FakeExt):
    return patch(
        "remy.core.external_claim_verifier.verify_external_claims",
        return_value=ext,
    )


# ── Empty / too-short draft ─────────────────────────────────────────────────


def test_empty_draft_short_circuits_to_none():
    out = decide_governance("", [])
    assert out.mode == "none"
    assert out.reason_code == "draft_too_short"
    # Brain emits no human text and no locale info.
    assert not hasattr(out, "banner")
    assert not hasattr(out, "honest_text")


def test_whitespace_only_draft_is_short_circuit():
    out = decide_governance("   \n\t  ", [])
    assert out.mode == "none"
    assert out.reason_code == "draft_too_short"


# ── No external claims ──────────────────────────────────────────────────────


def test_no_external_claims_yields_none_mode():
    with _patch_verifier(_FakeExt(phantom_count=0, total=0)):
        out = decide_governance("This is a plain reply with no citations.", [])
    assert out.mode == "none"
    assert out.reason_code == "no_external_claims"
    assert out.contains_external_reference is False
    assert out.allowed_action == "emit_draft"


def test_all_external_grounded_yields_none_mode():
    with _patch_verifier(_FakeExt(phantom_count=0, total=4)):
        out = decide_governance("Reply mentioning sources.", [])
    assert out.mode == "none"
    assert out.reason_code == "all_external_grounded"
    assert out.contains_external_reference is True
    assert out.phantom_ratio == 0.0


# ── Block mode (>=70% phantom) ──────────────────────────────────────────────


def test_block_mode_classified_correctly():
    with _patch_verifier(_FakeExt(phantom_count=6, total=6)):
        out = decide_governance("draft with bad citations" * 5, [])
    assert out.mode == "block"
    assert out.reason_code == "external_verification_failed"
    assert out.phantom_count == 6
    assert out.external_total == 6
    assert out.phantom_ratio == pytest.approx(1.0)
    assert out.allowed_action == "suppress_draft_emit_uncertainty"
    assert out.operator_hint == "retry_with_grounded_search"


def test_block_mode_at_threshold_boundary():
    # ratio == 0.70 should land in block per the gradient.
    with _patch_verifier(_FakeExt(phantom_count=7, total=10)):
        out = decide_governance("draft" * 5, [])
    assert out.mode == "block"
    assert out.allowed_action == "suppress_draft_emit_uncertainty"


# ── Aggressive mode (0.30 .. 0.70) ──────────────────────────────────────────


def test_aggressive_mode_classified_correctly():
    with _patch_verifier(_FakeExt(phantom_count=2, total=5)):
        out = decide_governance("draft" * 5, [])
    assert out.mode == "aggressive"
    assert out.reason_code == "external_verification_failed"
    assert out.allowed_action == "emit_draft_with_caveat"
    assert out.operator_hint == "request_evidence"


def test_phantom_text_markers_alone_promote_to_aggressive():
    with _patch_verifier(
        _FakeExt(phantom_count=0, total=1, phantom_text_markers=True)
    ):
        out = decide_governance("draft" * 5, [])
    assert out.mode == "aggressive"
    assert out.phantom_text_markers is True


# ── Soft mode (<0.30) ───────────────────────────────────────────────────────


def test_soft_mode_classified_correctly():
    with _patch_verifier(_FakeExt(phantom_count=1, total=10)):
        out = decide_governance("draft" * 5, [])
    assert out.mode == "soft"
    assert out.allowed_action == "emit_draft_with_caveat"


# ── Brain emits no human-facing text ────────────────────────────────────────


def test_governance_output_has_no_human_text_fields():
    """The structured contract must not carry rendered human strings."""
    out = GovernanceOutput()
    serialized = out.to_dict()
    forbidden = {"banner", "honest_text", "rendered", "locale", "language", "text"}
    assert forbidden.isdisjoint(serialized.keys()), (
        f"GovernanceOutput leaks human-facing fields: "
        f"{forbidden & set(serialized.keys())}"
    )


def test_decide_governance_signature_has_no_locale():
    """The pure brain decision must not accept a locale parameter."""
    import inspect
    sig = inspect.signature(decide_governance)
    assert "locale" not in sig.parameters, (
        "decide_governance must remain language-agnostic; locale is a "
        "mouth concern, not a brain concern."
    )


# ── Verifier failure is handled as structured state ─────────────────────────


def test_verifier_error_becomes_structured_reason():
    boom = RuntimeError("verifier exploded")
    with patch(
        "remy.core.external_claim_verifier.verify_external_claims",
        side_effect=boom,
    ):
        out = decide_governance("draft" * 5, [])
    assert out.mode == "none"
    assert out.reason_code == "verifier_error"
    # Detail captures the error shape; we don't assert on the exact string.
    assert out.reason_detail
    assert "verifier exploded" in out.reason_detail


# ── Legacy projection preserves enough for history compatibility ────────────


def test_decision_from_output_preserves_counts_and_mode():
    out = GovernanceOutput(
        mode="block",
        reason_code="external_verification_failed",
        reason_detail="phantom 6/6 (ratio 1.00) markers=False",
        phantom_count=6,
        external_total=6,
        phantom_ratio=1.0,
        contains_external_reference=True,
    )
    legacy = decision_from_output(out)
    assert legacy.mode == "block"
    assert legacy.phantom_count == 6
    assert legacy.external_total == 6
    assert legacy.phantom_ratio == 1.0
    assert legacy.contains_external_reference is True
    # reason should be derived from structured output, not invented
    assert legacy.reason  # non-empty


def test_decision_from_output_for_none_mode_has_neutral_reason():
    out = GovernanceOutput(
        mode="none",
        reason_code="no_external_claims",
    )
    legacy = decision_from_output(out)
    assert legacy.mode == "none"
    assert legacy.reason  # something meaningful, not empty


# ── Signal publication side effect ──────────────────────────────────────────


def test_block_publishes_unsafe_storage_signal():
    with _patch_verifier(_FakeExt(phantom_count=6, total=6)), patch(
        "remy.core.claim_provenance.record_turn_factuality_signal"
    ) as record:
        decide_governance("draft" * 5, [], session_id="sess-A")
    record.assert_called_once()
    kwargs = record.call_args.kwargs
    assert kwargs["brain_storage_unsafe"] is True
    assert kwargs["phantom_count"] == 6
    assert kwargs["external_total"] == 6


def test_clean_response_publishes_safe_signal():
    with _patch_verifier(_FakeExt(phantom_count=0, total=2)), patch(
        "remy.core.claim_provenance.record_turn_factuality_signal"
    ) as record:
        decide_governance("draft" * 5, [], session_id="sess-B")
    record.assert_called_once()
    assert record.call_args.kwargs["brain_storage_unsafe"] is False


def test_no_signal_published_without_session_id():
    with _patch_verifier(_FakeExt(phantom_count=6, total=6)), patch(
        "remy.core.claim_provenance.record_turn_factuality_signal"
    ) as record:
        decide_governance("draft" * 5, [], session_id="")
    record.assert_not_called()
