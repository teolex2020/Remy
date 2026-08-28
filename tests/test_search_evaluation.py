from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

from remy.core.search_evaluation import (
    aggregate_case_metrics,
    compare_reports,
    evaluate_ranking,
)


def test_evaluate_ranking_uses_graded_relevance_and_canonical_urls():
    candidates = [
        {"uri": "https://irrelevant.test/page"},
        {"uri": "https://docs.python.org/3/library/asyncio-task.html?utm_source=x#tasks"},
        {"uri": "https://discuss.python.org/t/asyncio/"},
    ]
    metrics = evaluate_ranking(
        candidates,
        {
            "https://docs.python.org/3/library/asyncio-task.html": 3,
            "https://discuss.python.org/t/asyncio": 1,
        },
        expected_domains=["docs.python.org"],
        min_independent_domains=3,
    )

    assert metrics["recall_at_10"] == 1.0
    assert metrics["mrr"] == 0.5
    assert 0.5 < metrics["ndcg_at_10"] < 1.0
    assert metrics["target_domain_hit"] is True
    assert metrics["independent_domain_rate"] == 1.0


def test_evaluate_ranking_reports_output_duplicates_and_input_suppression():
    candidates = [
        {"uri": "https://example.test/docs"},
        {"uri": "https://example.test/docs#section"},
    ]
    metrics = evaluate_ranking(
        candidates,
        {"https://example.test/docs": 3},
        raw_candidate_count=4,
        raw_unique_candidate_count=1,
    )

    assert metrics["duplicate_rate"] == 0.5
    assert metrics["duplicate_suppressed"] == 3
    assert metrics["duplicate_suppression_rate"] == 0.75


def test_aggregate_case_metrics_includes_cache_and_latency_rates():
    cases = [
        {
            "duration_ms": 10,
            "external_search_used": False,
            "metrics": {
                "recall_at_10": 1.0,
                "ndcg_at_10": 0.8,
                "mrr": 1.0,
                "target_domain_hit": True,
                "independent_domain_rate": 1.0,
                "duplicate_rate": 0.0,
                "duplicate_suppression_rate": 0.5,
            },
        },
        {
            "duration_ms": 30,
            "external_search_used": True,
            "metrics": {
                "recall_at_10": 0.5,
                "ndcg_at_10": 0.6,
                "mrr": 0.5,
                "target_domain_hit": False,
                "independent_domain_rate": 0.5,
                "duplicate_rate": 0.0,
                "duplicate_suppression_rate": 0.0,
            },
        },
    ]

    summary = aggregate_case_metrics(cases)

    assert summary["recall_at_10"] == 0.75
    assert summary["target_domain_hit_rate"] == 0.5
    assert summary["local_cache_hit_rate"] == 0.5
    assert summary["external_search_rate"] == 0.5
    assert summary["latency_p50_ms"] == 20.0
    assert summary["latency_p95_ms"] == 29.0


def test_compare_reports_detects_quality_drop_and_duplicate_growth():
    baseline = {
        "summary": {
            "recall_at_10": 0.9,
            "ndcg_at_10": 0.9,
            "mrr": 0.9,
            "duplicate_rate": 0.0,
        }
    }
    current = {
        "summary": {
            "recall_at_10": 0.7,
            "ndcg_at_10": 0.89,
            "mrr": 0.9,
            "duplicate_rate": 0.2,
        }
    }

    comparison = compare_reports(current, baseline)

    assert comparison["passed"] is False
    assert {item["metric"] for item in comparison["regressions"]} == {
        "recall_at_10",
        "duplicate_rate",
    }


def test_versioned_snapshot_benchmark_passes_all_contract_cases():
    root = Path(__file__).resolve().parents[1]
    runner_path = root / "benchmarks" / "retrieval" / "run_quality_benchmark.py"
    spec = importlib.util.spec_from_file_location("remy_quality_benchmark", runner_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    benchmark_path = root / "benchmarks" / "retrieval" / "benchmark_v2.yaml"
    document = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
    report = module.build_report(document, document["cases"], live=False)

    failures = {
        result["id"]: result["issues"]
        for result in report["results"]
        if not result["passed"]
    }
    assert failures == {}
    assert report["summary"]["case_count"] == 18
    assert report["summary"]["duplicate_rate"] == 0.0
