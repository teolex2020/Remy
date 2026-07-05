"""Live wiring test: typed causal edge refines the execution-gate scar block.

The gate now distinguishes a genuine causal scar (`refutes` — the action
counterfactually leads to a bad outcome) from a mere correlation (`precedes` —
the bad outcome occurs about as often WITHOUT the action). Only the former
hard-blocks. In a medical context: a medication that *causes* a symptom is
blocked; one that merely *precedes* it is not.
"""

import pytest

aura = pytest.importorskip("aura")
if not hasattr(aura, "classify_causal_edge"):
    pytest.skip(
        "installed AuraSDK lacks classify_causal_edge",
        allow_module_level=True,
    )

from remy.core_v3.runtime.execution_gate import ExecutionGateRuntime


classify = ExecutionGateRuntime._classify_causal_edge


def test_refutation_with_no_support_is_a_causal_scar():
    # The action reliably preceded a bad outcome, never a good one → causal scar.
    assert classify(supports=0, refutes=6) == "refutes"


def test_failures_dominating_is_a_causal_scar():
    # Failures far outnumber the times the action was safe → causal scar.
    assert classify(supports=1, refutes=8) == "refutes"


def test_action_mostly_safe_is_correlation_not_scar():
    # The action was usually taken WITHOUT failure → the rare failure is
    # correlation, not causation → precedes (does not hard-block).
    assert classify(supports=8, refutes=2) == "precedes"


def test_classifier_returns_known_label_set():
    label = classify(supports=3, refutes=3)
    assert label in {"precedes", "causes", "enables", "refutes"}


class _FakeVerdict:
    def __init__(self, supports, refutes, is_refuted=True, scar=False):
        self.supports = supports
        self.refutes = refutes
        self.is_refuted = is_refuted
        self.scar = scar


class _FakeRecorder:
    def __init__(self, verdict):
        self._verdict = verdict

    def scar_check(self, situation, action):
        return self._verdict


class _Step:
    instruction = "give ibuprofen"


class _Mission:
    description = "patient on warfarin"


def _gate_with_recorder(verdict):
    # Build a bare gate object without running __init__ (we only exercise the
    # scar-gate method, which depends solely on self.recorder).
    gate = ExecutionGateRuntime.__new__(ExecutionGateRuntime)
    gate.recorder = _FakeRecorder(verdict)
    return gate


def test_gate_hard_blocks_a_causal_scar():
    gate = _gate_with_recorder(_FakeVerdict(supports=0, refutes=5))
    blocked, reason = gate._consequence_scar_gate(
        mission=_Mission(), goal=None, task=None, step=_Step()
    )
    assert blocked is True
    assert "refutes" in reason


def test_gate_does_not_hard_block_a_correlation():
    # is_refuted True (verdict says refuted) but the action was mostly safe
    # (supports >> refutes) → precedes → the gate must NOT hard-block; the
    # softer policy-hint path handles it.
    gate = _gate_with_recorder(_FakeVerdict(supports=8, refutes=2))
    blocked, reason = gate._consequence_scar_gate(
        mission=_Mission(), goal=None, task=None, step=_Step()
    )
    assert blocked is False
    assert reason == ""
