"""Scar-protection gate tests — run against the REAL AuraSDK consequence memory.

These are falsify-discipline tests: a frozen LLM repeating a common-but-wrong
recommendation must NOT bury a lived world refutation. We assert on real
`aura.Aura` storage, not a mock, so a regression in the SDK surface is caught here.
"""

import tempfile

import pytest

aura = pytest.importorskip("aura")
if not hasattr(aura.Aura, "capture_consequence"):
    pytest.skip(
        "installed AuraSDK lacks native capture_consequence; Remy fallback is covered elsewhere",
        allow_module_level=True,
    )

from remy.core.consequence_gate import (
    consult_consequence_memory,
    gate_proposals,
    render_scar_warning,
)


WARFARIN = "patient on warfarin"
IBUPROFEN = "suggest ibuprofen"
ACETAMINOPHEN = "suggest acetaminophen"


def _store():
    return aura.Aura(tempfile.mkdtemp())


def test_unverified_action_abstains():
    store = _store()
    v = consult_consequence_memory(store, WARFARIN, IBUPROFEN)
    assert v.verdict == "inconclusive"
    assert v.abstain is True
    assert v.is_refuted is False
    assert render_scar_warning(v) == ""


def test_world_refutation_is_surfaced():
    store = _store()
    store.capture_consequence(WARFARIN, IBUPROFEN, "GI bleed - harmful", -1)
    v = consult_consequence_memory(store, WARFARIN, IBUPROFEN)
    assert v.is_refuted is True
    assert v.refutes >= 1
    assert "REFUTED" in render_scar_warning(v)


def test_scar_survives_llm_supporting_frequency():
    """THE GASLIGHT GUARD at the agent layer."""
    store = _store()
    store.capture_consequence(WARFARIN, IBUPROFEN, "GI bleed - harmful", -1)
    # Frozen model floods the common-but-wrong recommendation as support.
    for _ in range(40):
        store.capture_consequence(WARFARIN, IBUPROFEN, "commonly recommended", 1)

    v = consult_consequence_memory(store, WARFARIN, IBUPROFEN)
    assert v.is_refuted is True, "supporting frequency must not flip a scar"
    assert v.scar is True
    assert v.supports >= 40 and v.refutes >= 1
    assert v.scar and "do not clear this" in render_scar_warning(v)


def test_gate_blocks_only_refuted_proposal_pair_scoped():
    store = _store()
    store.capture_consequence(WARFARIN, IBUPROFEN, "GI bleed", -1)
    store.capture_consequence(WARFARIN, ACETAMINOPHEN, "safe, pain relieved", 1)

    report = gate_proposals(
        store,
        [(WARFARIN, IBUPROFEN), (WARFARIN, ACETAMINOPHEN)],
    )
    assert report.proposals_checked == 2
    assert report.blocked is True
    assert len(report.refuted) == 1
    assert report.refuted[0].action == IBUPROFEN
    assert report.banners and "REFUTED" in report.banners[0]


def test_gate_never_crashes_on_bad_store():
    class Broken:
        def consequence_verdict(self, *a, **k):
            raise RuntimeError("boom")

    v = consult_consequence_memory(Broken(), WARFARIN, IBUPROFEN)
    assert v.verdict == "inconclusive"
    assert v.abstain is True
