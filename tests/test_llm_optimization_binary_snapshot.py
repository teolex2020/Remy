import asyncio


def test_binary_snapshot_roundtrip_and_scoped_readout(tmp_path):
    from tools.validation.llm_optimization_binary_snapshot import (
        SnapshotRecord,
        preflight_snapshot,
        read_binary_snapshot,
        resolve_snapshot,
        source_scope_context_key,
        write_binary_snapshot,
    )

    context_key = source_scope_context_key(scope="scope-a", domain="domain-a", source="source-a")
    records = [
        SnapshotRecord(
            query="What is the code?",
            context_key=context_key,
            answer_text="RX-4471",
            source_id="case-a",
        )
    ]

    meta = write_binary_snapshot(records, tmp_path)
    loaded = read_binary_snapshot(tmp_path)
    ok = resolve_snapshot(loaded, "What is the code?", scope="scope-a", domain="domain-a", source="source-a")
    wrong = resolve_snapshot(loaded, "What is the code?", scope="wrong", domain="domain-a", source="source-a")

    assert meta["record_count"] == 1
    assert preflight_snapshot(loaded)["status"] == "Pass"
    assert ok["status"] == "answered"
    assert ok["answer_text"] == "RX-4471"
    assert wrong["status"] == "blocked"
    assert preflight_snapshot(loaded + loaded)["status"] == "Fail"


def test_binary_snapshot_candidate_requires_answer_authority(tmp_path):
    from tools.validation.llm_optimization_binary_snapshot import (
        SnapshotRecord,
        read_binary_snapshot,
        resolve_snapshot,
        review_snapshot_answer_authority,
        source_scope_context_key,
        write_binary_snapshot,
    )

    context_key = source_scope_context_key(scope="scope-a", domain="domain-a", source="source-a")
    write_binary_snapshot(
        [
            SnapshotRecord(
                query="Can this be answered?",
                context_key=context_key,
                answer_text="candidate-only material",
                status="candidate",
                source_id="case-a",
            )
        ],
        tmp_path,
    )

    packet = resolve_snapshot(
        read_binary_snapshot(tmp_path),
        "Can this be answered?",
        scope="scope-a",
        domain="domain-a",
        source="source-a",
    )
    authority = review_snapshot_answer_authority(packet)

    assert packet["candidate_found"] is True
    assert packet["answer_permission_granted"] is False
    assert authority["decision"] == "Stop"
    assert authority["product_answer_use_allowed"] is False


def test_binary_snapshot_eval_has_high_savings_and_negative_probes(tmp_path):
    from tools.validation.llm_optimization_binary_snapshot import run_binary_snapshot_eval
    from tools.validation.llm_optimization_corpus_eval import DEFAULT_CASES, load_cases

    cases = load_cases(DEFAULT_CASES)[:5]
    report = asyncio.run(
        run_binary_snapshot_eval(
            cases,
            modes=["raw", "binary_snapshot_readout"],
            model="dry-model",
            dry_run=True,
            delay_sec=0.0,
            snapshot_dir=tmp_path,
        )
    )

    snapshot = report["summary"]["by_mode"]["binary_snapshot_readout"]
    assert report["schema"] == "remy_binary_snapshot_eval_v1"
    assert snapshot["accuracy"] == 1.0
    assert snapshot["snapshot_hit_rate"] == 1.0
    assert snapshot["provider_total_saved_pct_vs_raw"] >= 95.0
    assert report["summary"]["negative_probes"]["status"] == "Pass"


def test_binary_snapshot_mixed_partial_snapshot_reports_fallbacks(tmp_path):
    from tools.validation.llm_optimization_binary_snapshot import run_binary_snapshot_eval
    from tools.validation.llm_optimization_corpus_eval import DEFAULT_CASES, load_cases

    cases = load_cases(DEFAULT_CASES)[:6]
    report = asyncio.run(
        run_binary_snapshot_eval(
            cases,
            modes=["raw", "binary_snapshot_readout", "binary_snapshot_or_fallback"],
            model="dry-model",
            dry_run=True,
            delay_sec=0.0,
            snapshot_dir=tmp_path,
            traffic_profile="exact",
            snapshot_coverage_pct=50,
        )
    )

    direct = report["summary"]["by_mode"]["binary_snapshot_readout"]
    fallback = report["summary"]["by_mode"]["binary_snapshot_or_fallback"]
    assert report["summary"]["snapshot_build_cases"] == 3
    assert direct["snapshot_hits"] == 3
    assert direct["blocked_count"] == 3
    assert fallback["snapshot_hits"] == 3
    assert fallback["fallback_count"] == 3
    assert fallback["accuracy"] == 1.0
    assert 0 < fallback["provider_total_saved_pct_vs_raw"] < 100


def test_binary_snapshot_paraphrase_profile_misses_exact_snapshot(tmp_path):
    from tools.validation.llm_optimization_binary_snapshot import run_binary_snapshot_eval
    from tools.validation.llm_optimization_corpus_eval import DEFAULT_CASES, load_cases

    cases = load_cases(DEFAULT_CASES)[:4]
    report = asyncio.run(
        run_binary_snapshot_eval(
            cases,
            modes=["raw", "binary_snapshot_readout", "binary_snapshot_or_fallback"],
            model="dry-model",
            dry_run=True,
            delay_sec=0.0,
            snapshot_dir=tmp_path,
            traffic_profile="paraphrase",
            snapshot_coverage_pct=100,
        )
    )

    direct = report["summary"]["by_mode"]["binary_snapshot_readout"]
    fallback = report["summary"]["by_mode"]["binary_snapshot_or_fallback"]
    assert direct["snapshot_hits"] == 0
    assert direct["blocked_count"] == 4
    assert fallback["fallback_count"] == 4
    assert fallback["accuracy"] == 1.0
