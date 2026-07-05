"""
Phase A.7 — brain-native mouth contract tests.

Behavior assertions on render_block_response(). No exact-string checks
against any specific language. Verifies:

  - the SLM path is taken when call_llm returns usable output
  - the universal fallback is taken when call_llm fails or returns empty
  - the fallback surface is structural (reason_code + counts), not prose
  - the mouth never re-introduces hardcoded per-language sentence tables
  - no fabricated external-reference text leaks from the mouth itself
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from remy.core.epistemic_governance import GovernanceOutput
from remy.core.mouth import (
    MouthRenderResult,
    render_block_response,
)


def _block_output(**overrides) -> GovernanceOutput:
    base = dict(
        mode="block",
        reason_code="external_verification_failed",
        reason_detail="phantom 6/6 (ratio 1.00) markers=False",
        phantom_count=6,
        external_total=6,
        phantom_ratio=1.0,
        contains_external_reference=True,
        operator_hint="retry_with_grounded_search",
        allowed_action="suppress_draft_emit_uncertainty",
    )
    base.update(overrides)
    return GovernanceOutput(**base)


# ── SLM path ────────────────────────────────────────────────────────────────


def test_slm_path_returned_when_call_llm_succeeds():
    fake_ai = SimpleNamespace(content="An honest reply in some language.")
    with patch("remy.core.llm.call_llm", return_value=fake_ai) as call:
        result = render_block_response("user msg", _block_output())

    assert isinstance(result, MouthRenderResult)
    assert result.source == "slm"
    assert result.text == "An honest reply in some language."
    assert result.error == ""
    call.assert_called_once()
    # The mouth must call the SLM with a structured purpose tag.
    assert call.call_args.kwargs.get("purpose") == "mouth_block_render"


def test_slm_path_concatenates_list_content_parts():
    """Some providers (Gemini) return content as a list of parts."""
    fake_ai = SimpleNamespace(
        content=[
            "First sentence. ",
            {"text": "Second sentence."},
        ]
    )
    with patch("remy.core.llm.call_llm", return_value=fake_ai):
        result = render_block_response("user msg", _block_output())

    assert result.source == "slm"
    assert "First sentence." in result.text
    assert "Second sentence." in result.text


def test_slm_path_strips_whitespace():
    fake_ai = SimpleNamespace(content="\n\n   reply with padding   \n")
    with patch("remy.core.llm.call_llm", return_value=fake_ai):
        result = render_block_response("user msg", _block_output())

    assert result.source == "slm"
    assert result.text == "reply with padding"


# ── Fallback path ───────────────────────────────────────────────────────────


def test_fallback_when_call_llm_raises():
    with patch(
        "remy.core.llm.call_llm",
        side_effect=RuntimeError("network down"),
    ):
        result = render_block_response("user msg", _block_output())

    assert result.source == "fallback_universal"
    assert "network down" in result.error
    # Fallback surface must encode structural state, not prose.
    assert "external_verification_failed" in result.text
    assert "6/6" in result.text
    assert "retry_with_grounded_search" in result.text


def test_fallback_when_call_llm_returns_empty():
    fake_ai = SimpleNamespace(content="")
    with patch("remy.core.llm.call_llm", return_value=fake_ai):
        result = render_block_response("user msg", _block_output())

    assert result.source == "fallback_universal"
    assert result.error == "empty_slm_output"
    assert "external_verification_failed" in result.text


def test_fallback_when_call_llm_returns_whitespace_only():
    fake_ai = SimpleNamespace(content="   \n\t  ")
    with patch("remy.core.llm.call_llm", return_value=fake_ai):
        result = render_block_response("user msg", _block_output())

    assert result.source == "fallback_universal"


def test_fallback_surface_is_language_neutral():
    """The fallback must not encode any natural-language sentence —
    only a banner symbol and structural tokens (reason, counts, hint)."""
    with patch(
        "remy.core.llm.call_llm",
        side_effect=RuntimeError("offline"),
    ):
        result = render_block_response("user msg", _block_output())

    # Must contain the structural enums.
    assert "external_verification_failed" in result.text
    assert "6/6" in result.text
    assert "retry_with_grounded_search" in result.text

    # Must NOT contain hardcoded English honest-uncertainty sentences from
    # the old _HONEST_BLOCK_TEXT_EN baggage.
    forbidden_phrases = [
        "I can't honestly confirm",
        "did not pass structural verification",
        "plausible-sounding citations",
        # Ukrainian baggage
        "Я не можу чесно підтвердити",
        "не пройшли структурної перевірки",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in result.text, (
            f"Fallback leaked hardcoded sentence: {phrase!r}"
        )


# ── Mouth never invents external references ─────────────────────────────────


def test_mouth_module_has_no_hardcoded_block_text_tables():
    """No remnants of _HONEST_BLOCK_TEXT_* may live in the mouth module."""
    import remy.core.mouth as m
    forbidden_names = {
        "_HONEST_BLOCK_TEXT_EN",
        "_HONEST_BLOCK_TEXT_UA",
        "_honest_block_text",
    }
    actual = set(dir(m))
    leaked = forbidden_names & actual
    assert not leaked, f"mouth.py leaks legacy text tables: {leaked}"


def test_mouth_render_result_carries_provenance():
    """Every render result must declare which path produced the text,
    so harness/tests can attribute the surface correctly."""
    fake_ai = SimpleNamespace(content="something")
    with patch("remy.core.llm.call_llm", return_value=fake_ai):
        result = render_block_response("user msg", _block_output())
    assert result.source in ("slm", "fallback_universal")


# ── Variation in counts is reflected in fallback ────────────────────────────


def test_fallback_reflects_actual_phantom_counts():
    out = _block_output(phantom_count=9, external_total=11)
    with patch(
        "remy.core.llm.call_llm",
        side_effect=RuntimeError("offline"),
    ):
        result = render_block_response("user msg", out)
    assert "9/11" in result.text


def test_fallback_handles_missing_external_total():
    out = _block_output(phantom_count=3, external_total=0)
    with patch(
        "remy.core.llm.call_llm",
        side_effect=RuntimeError("offline"),
    ):
        result = render_block_response("user msg", out)
    # Use ?-marker rather than divide-by-zero or fabricated number.
    assert "3/?" in result.text
