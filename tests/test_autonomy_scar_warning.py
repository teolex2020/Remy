"""Live wiring test: scar warning in the autonomy decision cycle.

A plan step that the world previously REFUTED (stored as lived consequence)
must surface a scar warning that the per-cycle decision prompt sees — so the
autonomous loop does not silently retry a lived failure. Uses the REAL AuraSDK.
"""

import tempfile

import pytest

aura = pytest.importorskip("aura")
if not hasattr(aura.Aura, "consequence_verdict"):
    pytest.skip(
        "installed AuraSDK lacks consequence_verdict",
        allow_module_level=True,
    )

import remy.core.agent_tools as agent_tools
from remy.core.agent_tools import _AuraCompat
from remy.core.autonomy import ActionPlan, AutonomousLoop


GOAL = "reduce patient nighttime blood pressure"
BAD_STEP = "increase diuretic dose without lab check"


@pytest.fixture
def isolated_brain(monkeypatch):
    store = _AuraCompat(tempfile.mkdtemp())
    # Point the module-level `brain` singleton (used by the scar helper) at the
    # isolated temp store for this test.
    monkeypatch.setattr(agent_tools, "brain", store, raising=False)
    monkeypatch.setattr("remy.core.autonomy.brain", store, raising=False)
    return store


def test_refuted_next_step_surfaces_scar_warning(isolated_brain):
    # World refuted this (goal, step) pair before.
    isolated_brain.capture_consequence(GOAL, BAD_STEP, "hypotensive episode", -1)
    # Even if it later "succeeded" a few times, the scar must hold.
    for _ in range(5):
        isolated_brain.capture_consequence(GOAL, BAD_STEP, "looked fine", 1)

    engine = AutonomousLoop()
    plan = ActionPlan(
        plan_id="p1",
        goal_id="g1",
        goal_description=GOAL,
        steps=[BAD_STEP],
        current_step=0,
    )
    note = engine._scar_warning_for_plan({"description": GOAL}, plan)
    assert note, "a refuted next step must produce a scar warning"
    assert "REFUTED" in note or "СПРОСТУВ" in note


def test_unrefuted_step_has_no_scar_warning(isolated_brain):
    engine = AutonomousLoop()
    plan = ActionPlan(
        plan_id="p1",
        goal_id="g1",
        goal_description=GOAL,
        steps=["schedule a routine blood pressure check"],
        current_step=0,
    )
    note = engine._scar_warning_for_plan({"description": GOAL}, plan)
    assert note == ""


def test_scar_helper_is_fail_soft_on_missing_plan(isolated_brain):
    engine = AutonomousLoop()
    assert engine._scar_warning_for_plan({"description": GOAL}, None) == ""
    assert engine._scar_warning_for_plan({}, None) == ""
