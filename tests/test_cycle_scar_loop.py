"""Live consequence-loop test for the v3 execution cycle.

Closes the loop end-to-end at the agent runtime layer:
  a cycle REFUTES a (goal, action) → it is stored in consequence memory →
  a later scar_check on the SAME pair still reports refuted, even after the
  frozen-model-style SUPPORTS frequency tries to bury it.

Runs against the REAL AuraSDK memory (a fresh temp brain), not a mock.
"""

import tempfile
from types import SimpleNamespace

import pytest

aura = pytest.importorskip("aura")

from remy.core.agent_tools import _AuraCompat
from remy.core.model_trace import extract_model_runtime
import remy.core_v3.memory.memory_api as memory_api
from remy.core_v3.memory.memory_api import AuraMemoryBackend
from remy.core_v3.execution.cycle_recorder import CycleRecorder, CycleRecord
from remy.core_v3.execution.execution_runtime import ExecutionRuntime
from remy.core_v3.agents.base_agent import AgentOutput
from remy.core_v3.runtime.context_runtime import ContextRuntime


GOAL = "schedule patient medication reminder"
BAD_ACTION = "double the dose automatically"


@pytest.fixture
def temp_memory(monkeypatch):
    backend = AuraMemoryBackend(_AuraCompat(tempfile.mkdtemp()))
    # Point the v3 global memory at the isolated temp brain for this test.
    monkeypatch.setattr(memory_api, "_memory", backend, raising=False)
    return backend


def _record(recorder, status):
    rec = CycleRecord()
    rec.cycle_num = 1
    rec.goal_id = "g1"
    rec.goal_description = GOAL
    rec.decision = BAD_ACTION
    rec.status = status
    recorder._store_outcome(rec)


def test_memory_backend_exposes_consequence_verdict(temp_memory):
    v = temp_memory.consequence_verdict(GOAL, BAD_ACTION, namespace="remy")
    assert v["verdict"] == "inconclusive"
    assert v["abstain"] is True


def test_cycle_scar_check_surfaces_refuted_action(temp_memory):
    recorder = CycleRecorder()
    _record(recorder, "failure")  # world refuted this (goal, action)

    verdict = recorder.scar_check(GOAL, BAD_ACTION)
    assert verdict.is_refuted is True
    assert verdict.refutes >= 1


def test_cycle_outcome_scars_planned_action_not_only_chief_decision(temp_memory):
    recorder = CycleRecorder()
    rec = CycleRecord()
    rec.cycle_num = 1
    rec.goal_id = "g1"
    rec.goal_description = GOAL
    rec.decision = "pause"
    rec.planned_action = BAD_ACTION
    rec.status = "failure"
    recorder._store_outcome(rec)

    verdict = recorder.scar_check(GOAL, BAD_ACTION)
    assert verdict.is_refuted is True
    assert verdict.refutes >= 1


def test_model_outcome_is_recorded_as_consequence_memory(temp_memory):
    recorder = CycleRecorder()
    rec = CycleRecord()
    rec.cycle_num = 1
    rec.goal_id = "g1"
    rec.goal_description = GOAL
    rec.specialist = "planner"
    rec.model = "local-fast-model"
    rec.fallback_used = True
    rec.planned_action = BAD_ACTION
    rec.status = "failure"
    recorder._store_outcome(rec)

    verdict = temp_memory.consequence_verdict(
        recorder._model_situation(rec),
        BAD_ACTION,
        namespace="remy-models",
    )
    assert str(verdict["verdict"]).startswith("refut")
    assert verdict["refutes"] >= 1


def test_factuality_failure_is_recorded_as_quality_scar(temp_memory):
    recorder = CycleRecorder()
    rec = CycleRecord()
    rec.cycle_num = 1
    rec.goal_id = "g1"
    rec.goal_description = GOAL
    rec.specialist = "researcher"
    rec.model = "fast-model"
    rec.planned_action = BAD_ACTION
    rec.status = "success"
    rec.unsupported_observed_claims = 2
    recorder._store_outcome(rec)

    verdict = temp_memory.consequence_verdict(
        GOAL,
        BAD_ACTION,
        namespace="remy-factuality",
    )
    assert str(verdict["verdict"]).startswith("refut")
    assert verdict["refutes"] >= 1


def test_v3_factuality_claim_type_is_recorded_as_scar(temp_memory):
    recorder = CycleRecorder()
    rec = CycleRecord()
    rec.cycle_num = 1
    rec.goal_id = "g1"
    rec.goal_description = GOAL
    rec.specialist = "researcher"
    rec.model = "fast-model"
    rec.planned_action = BAD_ACTION
    rec.status = "success"
    rec.unsupported_observed_claims = 1
    rec.factuality_report = SimpleNamespace(
        claim_details=[
            SimpleNamespace(
                claim_class="unverified_current_fact",
                supported=False,
                text="The current price is 10.",
            )
        ]
    )
    recorder._store_outcome(rec)

    verdict = temp_memory.consequence_verdict(
        "factuality-claim-type:unverified_current_fact|runtime:v3",
        "answer_claim_type:unverified_current_fact:without_evidence",
        namespace="remy-factuality",
    )
    assert str(verdict["verdict"]).startswith("refut")
    assert verdict["refutes"] >= 1


def test_factuality_scar_becomes_requires_evidence_policy_hint(temp_memory):
    recorder = CycleRecorder()
    rec = CycleRecord()
    rec.cycle_num = 1
    rec.goal_id = "g1"
    rec.goal_description = GOAL
    rec.specialist = "researcher"
    rec.model = "fast-model"
    rec.planned_action = BAD_ACTION
    rec.status = "success"
    rec.unsupported_observed_claims = 2
    recorder._store_outcome(rec)

    hint = recorder.policy_hint(GOAL, BAD_ACTION)
    assert hint.hint == "requires_evidence"
    assert hint.requires_evidence is True
    assert hint.should_block is False


def test_recorder_routing_policy_hint_reads_specialist_consequence_memory(temp_memory):
    from remy.core_v3.memory.memory_api import get_memory

    memory = get_memory()
    memory.capture_consequence(
        situation="specialist:executor",
        action="route_to:executor",
        consequence="REFUTES",
        trust=-1,
        scope=["policy:avoid", "specialist:executor"],
        namespace="remy-routing",
    )

    hint = CycleRecorder().routing_policy_hint("executor")
    assert hint.hint == "avoid"
    assert hint.should_block is True


def test_execution_runtime_extracts_model_route_from_session_log():
    runtime = ExecutionRuntime()
    model, fallback_used = runtime._extract_model_runtime([
        {"type": "tool_call", "tool": "search"},
        {"type": "llm_call", "model": "cheap-model", "fallback_used": False},
        {"type": "llm_call", "model": "strong-model", "fallback_used": True},
    ])
    assert model == "strong-model"
    assert fallback_used is True


def test_extract_model_runtime_returns_safe_defaults():
    assert extract_model_runtime([]) == ("", False)


def test_llm_model_routing_override_reorders_softly():
    from remy.core.llm import _apply_model_routing, model_routing_override

    assert _apply_model_routing(["primary", "fallback"]) == ["primary", "fallback"]
    with model_routing_override(
        preferred_model="memory-good",
        avoid_models=("primary",),
    ):
        assert _apply_model_routing(["primary", "memory-good", "fallback"]) == [
            "memory-good",
            "fallback",
            "primary",
        ]
    assert _apply_model_routing(["primary", "fallback"]) == ["primary", "fallback"]


def test_agent_output_carries_model_runtime_into_execution_result():
    output = AgentOutput(
        status="success",
        session_log=[
            {"type": "llm_call", "model": "memory-good", "fallback_used": True},
        ],
    )
    result = output.to_execution_result()
    assert result.model == "memory-good"
    assert result.fallback_used is True


def test_recorder_stats_expose_model_outcomes():
    recorder = CycleRecorder()
    recorder._records.append(CycleRecord(
        cycle_num=1,
        goal_id="g1",
        goal_description=GOAL,
        status="success",
        model="fast-model",
        cost_usd=0.01,
    ))
    recorder._records.append(CycleRecord(
        cycle_num=2,
        goal_id="g2",
        goal_description=GOAL,
        status="failure",
        model="fast-model",
        fallback_used=True,
        unsupported_observed_claims=2,
        cost_usd=0.02,
    ))

    stats = recorder.stats()
    assert stats["model_outcomes"][0]["model"] == "fast-model"
    assert stats["model_outcomes"][0]["cycles"] == 2
    assert stats["model_outcomes"][0]["successes"] == 1
    assert stats["model_outcomes"][0]["failures"] == 1
    assert stats["model_outcomes"][0]["fallback_uses"] == 1
    assert stats["model_outcomes"][0]["unsupported_observed_claims"] == 2


def test_model_routing_hint_prefers_supported_and_avoids_scarred_models():
    recorder = CycleRecorder()
    recorder._records.append(CycleRecord(
        cycle_num=1,
        goal_description=GOAL,
        specialist="researcher",
        status="success",
        model="strong-model",
    ))
    recorder._records.append(CycleRecord(
        cycle_num=2,
        goal_description=GOAL,
        specialist="researcher",
        status="failure",
        model="weak-model",
        unsupported_observed_claims=2,
    ))

    hint = recorder.model_routing_hint("researcher")
    assert hint["preferred_model"] == "strong-model"
    assert "weak-model" in hint["avoid_models"]
    assert hint["source"] == "specialist"


def test_model_routing_hint_falls_back_to_global_history():
    recorder = CycleRecorder()
    recorder._records.append(CycleRecord(
        cycle_num=1,
        goal_description=GOAL,
        specialist="analyst",
        status="success",
        model="global-good",
    ))

    hint = recorder.model_routing_hint("researcher")
    assert hint["preferred_model"] == "global-good"
    assert hint["source"] == "global"


def test_cycle_model_routing_outcome_stores_model_action(temp_memory):
    recorder = CycleRecorder()
    rec = CycleRecord(
        cycle_num=1,
        goal_id="g1",
        goal_description="research evidence for medication reminder",
        specialist="researcher",
        status="success",
        model="strong-model",
    )

    recorder._store_outcome(rec)

    hint = temp_memory.policy_hint(
        recorder._model_routing_situation("researcher", "research"),
        "model:strong-model",
        namespace="remy-models",
    )
    assert hint["hint"] == "prefer"
    assert hint["supports"] >= 1


def test_model_routing_hint_reads_persistent_model_memory(temp_memory, monkeypatch):
    from remy.config.settings import settings

    recorder = CycleRecorder()
    monkeypatch.setattr(settings, "SUMMARY_MODEL", "weak-model")
    monkeypatch.setattr(settings, "FALLBACK_MODELS", ["strong-model"])
    situation = recorder._model_routing_situation("researcher")
    temp_memory.capture_consequence(
        situation=situation,
        action="model:weak-model",
        consequence="REFUTES",
        trust=-1,
        scope=["model-routing", "model:weak-model"],
        provenance=["test"],
        namespace="remy-models",
    )
    temp_memory.capture_consequence(
        situation=situation,
        action="model:strong-model",
        consequence="SUPPORTS",
        trust=1,
        scope=["model-routing", "model:strong-model"],
        provenance=["test"],
        namespace="remy-models",
    )

    hint = recorder.model_routing_hint("researcher")

    assert hint["source"] == "memory"
    assert hint["preferred_model"] == "strong-model"
    assert "weak-model" in hint["avoid_models"]


def test_context_runtime_injects_model_routing_hint():
    recorder = CycleRecorder()
    recorder._records.append(CycleRecord(
        cycle_num=1,
        goal_description=GOAL,
        specialist="researcher",
        status="success",
        model="strong-model",
    ))
    recorder._records.append(CycleRecord(
        cycle_num=2,
        goal_description=GOAL,
        specialist="researcher",
        status="failure",
        model="weak-model",
    ))

    runtime = ContextRuntime(
        recorder=recorder,
        budget=SimpleNamespace(
            config=SimpleNamespace(daily_usd=1.0),
            state=SimpleNamespace(daily_spent_usd=0.0),
        ),
        goal_context_runtime=SimpleNamespace(build_goal_dict=lambda **_: {"metadata": {}}),
    )
    ctx = runtime.build(
        mission=SimpleNamespace(id="m1", description=GOAL),
        goal=SimpleNamespace(id="g1", description=GOAL),
        step=SimpleNamespace(id="s1", instruction=BAD_ACTION),
        specialist=SimpleNamespace(
            id="researcher",
            step_budget=3,
            timeout_sec=30,
            tools=(),
            guardrails=(),
            approval_mode="none",
        ),
        memory_context=[],
    )
    assert ctx.preferred_model == "strong-model"
    assert "weak-model" in ctx.avoid_models


def test_cycle_scar_survives_later_success_frequency(temp_memory):
    """THE GASLIGHT GUARD inside the live execution loop."""
    recorder = CycleRecorder()
    _record(recorder, "failure")
    # The same action later "succeeds" many times (flaky / lucky / model-pushed).
    for _ in range(20):
        _record(recorder, "success")

    verdict = recorder.scar_check(GOAL, BAD_ACTION)
    assert verdict.is_refuted is True, "a lived failure must not be buried by later successes"
    assert verdict.scar is True
    assert verdict.refutes >= 1 and verdict.supports >= 20


def test_scar_warning_appears_in_decision_summary(temp_memory):
    recorder = CycleRecorder()
    _record(recorder, "failure")
    summary = recorder.recent_outcomes_summary_with_scars(GOAL, BAD_ACTION)
    assert "REFUTED" in summary


def test_scar_check_is_fail_soft_without_memory(monkeypatch):
    # Force get_memory to raise — scar_check must still return a safe verdict.
    def boom():
        raise RuntimeError("no memory")

    monkeypatch.setattr(memory_api, "get_memory", boom)
    recorder = CycleRecorder()
    verdict = recorder.scar_check(GOAL, BAD_ACTION)
    assert verdict.verdict == "inconclusive"
    assert verdict.abstain is True
