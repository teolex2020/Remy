from remy.core import llm_optimization_metrics as metrics


def _report(prompt_tokens_raw=100, prompt_tokens_reduced=25):
    return {
        "schema": "remy_llm_optimization_compare_v1",
        "user_text_preview": "What changed?",
        "raw": {"prompt_tokens_estimate": prompt_tokens_raw, "elapsed_seconds": 2.0, "answer": "raw full answer"},
        "reduced": {"prompt_tokens_estimate": prompt_tokens_reduced, "elapsed_seconds": 0.5, "answer": "reduced full answer"},
        "delta": {
            "prompt_tokens_saved_estimate": prompt_tokens_raw - prompt_tokens_reduced,
            "prompt_token_reduction_ratio": prompt_tokens_raw / prompt_tokens_reduced,
            "latency_saved_seconds": 1.5,
            "wrong_answers_avoided_estimate": 1,
        },
        "blocks": {
            "context_reducer": {"enabled": True},
            "memory_writer": {"enabled": True},
            "verifier": {"enabled": True},
            "model_router": {"decision": "cheap_or_configured_llm"},
            "cost_latency_tracker": {"enabled": True},
        },
        "claims": {"five_block_pipeline_report": True},
    }


def test_measurement_store_persists_compact_reports(tmp_path, monkeypatch):
    class Settings:
        DATA_DIR = tmp_path

    monkeypatch.setattr(metrics, "settings", Settings)
    metrics.clear_measurements()

    written = metrics.append_measurement(_report())
    items = metrics.list_measurements()
    summary = metrics.summarize_measurements(items)

    assert written["schema"] == "remy_llm_optimization_measurement_v1"
    assert items[0]["raw"]["answer_preview"] == "raw full answer"
    assert "answer" not in items[0]["raw"]
    assert summary["count"] == 1
    assert summary["five_block_reports"] == 1
    assert summary["token_saving_wins"] == 1
    assert summary["wrong_answers_avoided_estimate"] == 1
    assert summary["compare_runs"] == 0
    assert summary["apply_runs"] == 0
    assert summary["memory_only_hits"] == 0
    assert summary["llm_calls_saved"] == 0
    assert summary["answer_memory_writes"] == 0
    assert summary["router_decisions"] == {"cheap_or_configured_llm": 1}


def test_measurement_store_clear(tmp_path, monkeypatch):
    class Settings:
        DATA_DIR = tmp_path

    monkeypatch.setattr(metrics, "settings", Settings)
    metrics.append_measurement(_report())
    assert metrics.list_measurements()

    metrics.clear_measurements()

    assert metrics.list_measurements() == []



def test_memory_only_lookup_returns_exact_complete_safe_answer(tmp_path, monkeypatch):
    class Settings:
        DATA_DIR = tmp_path

    monkeypatch.setattr(metrics, "settings", Settings)
    metrics.clear_measurements()
    metrics.append_measurement(_report())

    hit = metrics.find_memory_only_answer("What changed?")

    assert hit is not None
    assert hit["answer"] == "reduced full answer"
    assert metrics.find_memory_only_answer("What changed in another file?") is None


def test_memory_only_lookup_rejects_truncated_answers(tmp_path, monkeypatch):
    class Settings:
        DATA_DIR = tmp_path

    monkeypatch.setattr(metrics, "settings", Settings)
    metrics.clear_measurements()
    report = _report()
    report["reduced"]["answer"] = "x" * 700
    metrics.append_measurement(report)

    assert metrics.find_memory_only_answer("What changed?") is None




def test_memory_only_lookup_reads_answer_memory_store_first(tmp_path, monkeypatch):
    class Settings:
        DATA_DIR = tmp_path

    monkeypatch.setattr(metrics, "settings", Settings)
    metrics.clear_measurements()
    report = _report()
    report["user_text_preview"] = "What changed?"
    metrics.append_answer_memory(report)

    hit = metrics.find_memory_only_answer("What changed?")

    assert hit is not None
    assert hit["answer"] == "reduced full answer"
    assert hit["source_store"] == "answer_memory"


def test_memory_only_lookup_rejects_unsafe_answer_memory_store(tmp_path, monkeypatch):
    class Settings:
        DATA_DIR = tmp_path

    monkeypatch.setattr(metrics, "settings", Settings)
    metrics.clear_measurements()
    report = _report()
    report["blocks"]["verifier"] = {
        "blocked": True,
        "reduced": {"brain_storage_unsafe": True},
    }
    metrics.append_answer_memory(report)

    assert metrics.find_memory_only_answer("What changed?") is None


def test_measurement_summary_counts_memory_only_saved_calls(tmp_path, monkeypatch):
    class Settings:
        DATA_DIR = tmp_path

    monkeypatch.setattr(metrics, "settings", Settings)
    metrics.clear_measurements()
    apply_report = _report()
    apply_report["claims"] = {"five_block_pipeline_report": True, "apply_context_reducer": True}
    apply_report["blocks"]["memory_writer"] = {"answer_memory_written": True}
    memory_report = _report(prompt_tokens_raw=80, prompt_tokens_reduced=1)
    memory_report["claims"] = {"five_block_pipeline_report": True, "memory_only_cache_hit": True}
    memory_report["delta"]["llm_calls_saved"] = 1

    metrics.append_measurement(apply_report)
    metrics.append_measurement(memory_report)
    summary = metrics.summarize_measurements(metrics.list_measurements())

    assert summary["apply_runs"] == 1
    assert summary["memory_only_hits"] == 1
    assert summary["llm_calls_saved"] == 1
    assert summary["answer_memory_writes"] == 1
    assert summary["router_decisions"] == {"cheap_or_configured_llm": 2}



def test_answer_memory_writes_safe_answer_with_sources_and_mistakes(tmp_path, monkeypatch):
    class Settings:
        DATA_DIR = tmp_path

    monkeypatch.setattr(metrics, "settings", Settings)
    metrics.clear_measurements()
    report = _report()
    report["claims"] = {"five_block_pipeline_report": True, "apply_context_reducer": True}
    report["blocks"]["verifier"] = {
        "blocked": False,
        "reduced": {
            "brain_storage_unsafe": False,
            "evidence_record_ids": ["rec-1"],
            "external_citations_total": 1,
            "external_citations_grounded": 1,
            "supported_claims_total": 2,
            "supported_internal": 1,
            "unsupported_claims_total": 0,
        },
    }

    item = metrics.append_answer_memory(report)
    items = metrics.list_answer_memory()

    assert item["schema"] == "remy_llm_optimization_answer_memory_v1"
    assert items[0]["answer_preview"] == "reduced full answer"
    assert items[0]["answer_text_stored"] is True
    assert items[0]["sources"]["evidence_record_ids"] == ["rec-1"]
    assert items[0]["sources"]["external_citations_grounded"] == 1
    assert items[0]["confirmed"]["supported_claims_total"] == 2
    assert items[0]["mistakes"]["unsupported_claims_total"] == 0


def test_answer_memory_suppresses_unsafe_answer_text(tmp_path, monkeypatch):
    class Settings:
        DATA_DIR = tmp_path

    monkeypatch.setattr(metrics, "settings", Settings)
    metrics.clear_measurements()
    report = _report()
    report["blocks"]["verifier"] = {
        "blocked": True,
        "reduced": {
            "brain_storage_unsafe": True,
            "unsupported_claims_total": 3,
            "unverified_current_claims": 2,
        },
    }

    item = metrics.append_answer_memory(report)

    assert item["answer_preview"] == ""
    assert item["answer_text_stored"] is False
    assert item["unsafe_answer_text_suppressed"] is True
    assert item["mistakes"]["unsupported_claims_total"] == 3
    assert item["mistakes"]["blocked"] is True
