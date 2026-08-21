import pytest


@pytest.mark.asyncio
async def test_context_reducer_compares_raw_and_reduced_with_measurements(tmp_path, monkeypatch):
    from remy.core.context_reducer import compare_context_reducer
    from remy.core import llm_optimization_metrics

    monkeypatch.setattr(llm_optimization_metrics.settings, "DATA_DIR", tmp_path)

    session_log = []
    for i in range(24):
        session_log.append({"type": "user_text", "text": f"noise turn {i} unrelated scheduling note"})
        session_log.append({"type": "model_response", "text": f"noise answer {i}"})
    session_log.append({"type": "user_text", "text": "Patient allergy record: penicillin causes rash."})
    session_log.append({"type": "model_response", "text": "Recorded penicillin allergy."})

    def fake_llm(prompt: str):
        class Result:
            content = "penicillin" if "penicillin" in prompt else "unknown"
            response_metadata = {}
        return Result()

    report = await compare_context_reducer(
        user_text="What allergy is recorded for the patient?",
        session_log=session_log,
        llm_func=fake_llm,
    )

    assert report["schema"] == "remy_llm_optimization_compare_v1"
    assert report["raw"]["answer"] == "penicillin"
    assert report["reduced"]["answer"] == "penicillin"
    assert report["delta"]["prompt_tokens_saved_estimate"] > 0
    assert report["delta"]["prompt_token_reduction_ratio"] > 1
    assert report["claims"]["measured_ab_comparison"] is True
    assert report["claims"]["kv_cache_internal_replacement"] is False
    assert report["claims"]["five_block_pipeline_report"] is True
    assert set(report["blocks"]) == {
        "context_reducer",
        "memory_writer",
        "verifier",
        "model_router",
        "cost_latency_tracker",
    }
    assert report["blocks"]["memory_writer"]["event_written"] is True
    assert report["blocks"]["memory_writer"]["persistent_measurement_written"] is True
    assert report["blocks"]["memory_writer"]["answer_memory_written"] is True
    assert report["blocks"]["memory_writer"]["answer_memory_written"] is True
    assert report["blocks"]["cost_latency_tracker"]["tokens_saved_estimate"] > 0
    assert session_log[-1]["type"] == "llm_optimization_measurement"


@pytest.mark.asyncio
async def test_apply_context_reducer_calls_llm_once_and_records_measurement(tmp_path, monkeypatch):
    from remy.core.context_reducer import apply_context_reducer
    from remy.core import llm_optimization_metrics

    monkeypatch.setattr(llm_optimization_metrics.settings, "DATA_DIR", tmp_path)

    session_log = []
    for i in range(20):
        session_log.append({"type": "user_text", "text": f"old unrelated note {i}"})
        session_log.append({"type": "model_response", "text": f"old unrelated answer {i}"})
    session_log.append({"type": "user_text", "text": "Medication context: aspirin was stopped yesterday."})

    calls = []

    def fake_llm(prompt: str):
        calls.append(prompt)

        class Result:
            content = "Aspirin was stopped."
            response_metadata = {"usage_metadata": {"prompt_tokens": 42, "output_tokens": 5}}

        return Result()

    result = await apply_context_reducer(
        user_text="What medication was stopped?",
        session_log=session_log,
        llm_func=fake_llm,
    )

    report = result["report"]
    assert len(calls) == 1
    assert result["answer"] == "Aspirin was stopped."
    assert report["schema"] == "remy_llm_optimization_apply_v1"
    assert report["claims"]["apply_context_reducer"] is True
    assert report["claims"]["raw_llm_call_skipped"] is True
    assert report["claims"]["measured_ab_comparison"] is False
    assert report["raw"]["llm_call_skipped"] is True
    assert report["reduced"]["answer"] == "Aspirin was stopped."
    assert report["delta"]["prompt_tokens_saved_estimate"] > 0
    assert report["blocks"]["memory_writer"]["event_written"] is True
    assert report["blocks"]["memory_writer"]["persistent_measurement_written"] is True
    assert session_log[-1]["type"] == "llm_optimization_measurement"


@pytest.mark.asyncio
async def test_apply_context_reducer_second_exact_question_uses_memory_only(tmp_path, monkeypatch):
    from remy.core.context_reducer import apply_context_reducer
    from remy.core import llm_optimization_metrics

    monkeypatch.setattr(llm_optimization_metrics.settings, "DATA_DIR", tmp_path)
    llm_optimization_metrics.clear_measurements()
    session_log = [{"type": "user_text", "text": "Medication context: aspirin was stopped."}]
    calls = []

    def fake_llm(prompt: str):
        calls.append(prompt)

        class Result:
            content = "Aspirin was stopped."
            response_metadata = {}

        return Result()

    first = await apply_context_reducer(
        user_text="What medication was stopped?",
        session_log=session_log,
        llm_func=fake_llm,
    )
    second = await apply_context_reducer(
        user_text="What medication was stopped?",
        session_log=session_log,
        llm_func=fake_llm,
    )

    assert len(calls) == 1
    assert first["report"]["schema"] == "remy_llm_optimization_apply_v1"
    assert second["answer"] == "Aspirin was stopped."
    assert second["report"]["schema"] == "remy_llm_optimization_memory_only_v1"
    assert second["report"]["blocks"]["model_router"]["decision"] == "memory_only"
    assert second["report"]["delta"]["llm_calls_saved"] == 1
    assert second["report"]["claims"]["memory_only_cache_hit"] is True


@pytest.mark.asyncio
async def test_apply_context_reducer_returns_verifier_corrected_text(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from remy.core.context_reducer import apply_context_reducer
    from remy.core import llm_optimization_metrics

    monkeypatch.setattr(llm_optimization_metrics.settings, "DATA_DIR", tmp_path)
    llm_optimization_metrics.clear_measurements()

    def fake_llm(prompt: str):
        class Result:
            content = "I verified the latest clinic policy today."
            response_metadata = {}

        return Result()

    def fake_enforce(text, session_log, **kwargs):
        report = SimpleNamespace(
            unsupported_claims_total=1,
            unverified_current_claims=1,
            unsupported_observed_claims=0,
            supported_claims_total=0,
            supported_internal=0,
            supported_external_verified=0,
            unverified_external=1,
            unsupported=1,
            brain_storage_unsafe=False,
            modified=True,
        )
        return "I have not verified the latest clinic policy in this turn.", report

    monkeypatch.setattr("remy.core.factuality.enforce_factuality", fake_enforce)

    result = await apply_context_reducer(
        user_text="What is the latest clinic policy?",
        session_log=[{"type": "user_text", "text": "Prior clinic discussion."}],
        llm_func=fake_llm,
    )

    report = result["report"]
    assert result["answer"] == "I have not verified the latest clinic policy in this turn."
    assert report["answer"] == result["answer"]
    assert report["reduced"]["answer"] == result["answer"]
    assert report["reduced"]["original_answer_preview"] == "I verified the latest clinic policy today."
    assert report["reduced"]["verifier_modified_answer"] is True
    assert report["claims"]["verifier_modified_answer"] is True


@pytest.mark.asyncio
async def test_apply_context_reducer_applies_model_router_override(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from remy.core.context_reducer import apply_context_reducer
    from remy.core import llm_optimization_metrics

    monkeypatch.setattr(llm_optimization_metrics.settings, "DATA_DIR", tmp_path)
    llm_optimization_metrics.clear_measurements()

    def fake_routing(**kwargs):
        return {
            "preferred_model": "cheap-model",
            "avoid_models": (),
            "routing_source": "structural_low_cost_prior",
            "complexity_bucket": "simple",
            "complexity_score": 1,
            "routing_reasons": ("low_pressure",),
        }

    captured = {}

    @contextmanager
    def fake_override(*, preferred_model="", avoid_models=None):
        captured["preferred_model"] = preferred_model
        captured["avoid_models"] = tuple(avoid_models or ())
        yield

    def fake_call_llm(prompt, **kwargs):
        captured["purpose"] = kwargs.get("purpose")

        class Result:
            content = "Routed answer."
            response_metadata = {"_served_by": "cheap-model", "_fallback_used": False}

        return Result()

    monkeypatch.setattr("remy.core.adaptive_model_router.build_adaptive_model_routing", fake_routing)
    monkeypatch.setattr("remy.core.llm.model_routing_override", fake_override)
    monkeypatch.setattr("remy.core.llm.call_llm", fake_call_llm)

    result = await apply_context_reducer(
        user_text="Short question?",
        session_log=[],
        session_id="router-test",
    )

    router = result["report"]["blocks"]["model_router"]
    assert result["answer"] == "Routed answer."
    assert captured["preferred_model"] == "cheap-model"
    assert captured["purpose"] == "context_reducer_apply"
    assert router["decision"] == "cheap_llm"
    assert router["preferred_model"] == "cheap-model"
    assert router["actual_model"] == "cheap-model"
    assert router["router_applied_to_llm_call"] is True
    assert result["report"]["claims"]["model_router_applied"] is True


def test_reduce_context_prefers_query_overlap_and_recent_lines():
    from remy.core.context_reducer import reduce_context_lines

    lines = [
        "user: random calendar note",
        "assistant: ok",
        "user: cardiology medication metoprolol dose changed",
        "assistant: dose recorded",
        "user: latest unrelated message",
    ]

    reduced = reduce_context_lines(lines, "What medication dose changed?", max_lines=3, keep_recent=1)

    assert any("metoprolol" in line for line in reduced)
    assert reduced[-1] == "user: latest unrelated message"


@pytest.mark.asyncio
async def test_context_reducer_real_llm_path_uses_pii_shield(tmp_path, monkeypatch):
    from remy.config.settings import settings
    from remy.core import llm_optimization_metrics
    from remy.core.context_reducer import compare_context_reducer

    monkeypatch.setattr(llm_optimization_metrics.settings, "DATA_DIR", tmp_path)
    from remy.core.pii_vault import get_vault

    monkeypatch.setattr(settings, "PII_SHIELD_ENABLED", True)
    vault = get_vault("ctx-pii-test")
    vault._profile_loaded = True
    vault._profile_values["Alice"] = "name"
    captured = {}

    def fake_call_llm(prompt: str, **kwargs):
        captured["prompt"] = prompt

        class Result:
            content = "Hello [PI!name_1]"
            response_metadata = {}

        return Result()

    monkeypatch.setattr("remy.core.llm.call_llm", fake_call_llm)

    report = await compare_context_reducer(
        user_text="Say hello to Alice",
        session_log=[{"type": "user_text", "text": "My name is Alice"}],
        session_id="ctx-pii-test",
    )

    assert "Alice" not in captured["prompt"]
    assert "[PII:name_1]" in captured["prompt"]
    assert report["raw"]["answer"] == "Hello Alice"
    assert report["reduced"]["answer"] == "Hello Alice"
