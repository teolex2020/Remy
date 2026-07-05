"""Live wiring test: typed causal edge refines specialist routing pressure.

A `prefer` routing hint is scaled by the typed causal grammar: a specialist that
counterfactually *causes* success gets a stronger boost than one that merely
*precedes* success (correlation). An `avoid` hint is left untouched — it is
already scar-protected upstream.
"""

import pytest

aura = pytest.importorskip("aura")
if not hasattr(aura, "classify_causal_edge"):
    pytest.skip(
        "installed AuraSDK lacks classify_causal_edge",
        allow_module_level=True,
    )

from remy.core_v3.runtime.specialist_runtime import SpecialistRuntime


classify = SpecialistRuntime._routing_causal_edge


def test_specialist_that_causes_success_is_classified_causes():
    assert classify(supports=8, refutes=1) == "causes"


def test_specialist_that_only_correlates_is_precedes():
    # Routing here mostly precedes FAILURE (success is rare) → correlation.
    assert classify(supports=2, refutes=8) == "precedes"


class _FakeRecorder:
    def __init__(self, hint):
        self._hint = hint

    def routing_policy_hint(self, specialist_id):
        return self._hint


def _runtime_with_hint(hint):
    rt = SpecialistRuntime.__new__(SpecialistRuntime)
    rt.recorder = _FakeRecorder(hint)
    return rt


def _quality():
    return {"quality_adjusted_success_rate": 0.50, "unsupported_claims": 0, "success_rate": 0.50}


def test_prefer_causes_gets_stronger_boost_than_prefer_precedes():
    causal_hint = {"hint": "prefer", "supports": 9, "refutes": 0, "reason": "lived"}
    corr_hint = {"hint": "prefer", "supports": 5, "refutes": 5, "reason": "lived"}

    q_causal = _runtime_with_hint(causal_hint)._with_routing_policy("spec", _quality())
    q_corr = _runtime_with_hint(corr_hint)._with_routing_policy("spec", _quality())

    assert q_causal["routing_causal_edge"] == "causes"
    # A causal specialist is boosted strictly more than a merely-correlated one.
    assert q_causal["routing_policy_adjustment"] > q_corr["routing_policy_adjustment"]
    assert q_causal["quality_adjusted_success_rate"] > q_corr["quality_adjusted_success_rate"]


def test_prefer_precedes_is_dampened_toward_neutral():
    # success rare relative to failure under this specialist → precedes →
    # near-neutral boost, not the full prefer boost.
    corr_hint = {"hint": "prefer", "supports": 2, "refutes": 8, "reason": "lived"}
    q = _runtime_with_hint(corr_hint)._with_routing_policy("spec", _quality())
    assert q["routing_causal_edge"] == "precedes"
    assert q["routing_policy_adjustment"] == pytest.approx(0.05)


def test_avoid_is_not_weakened_by_edge_refinement():
    # avoid is scar-protected upstream; the causal refinement must not touch it.
    avoid_hint = {"hint": "avoid", "supports": 0, "refutes": 6, "reason": "refuted"}
    q = _runtime_with_hint(avoid_hint)._with_routing_policy("spec", _quality())
    assert q["routing_policy"] == "avoid"
    assert q["routing_policy_adjustment"] == pytest.approx(-0.40)
    assert "routing_causal_edge" not in q  # edge refinement only applies to prefer
