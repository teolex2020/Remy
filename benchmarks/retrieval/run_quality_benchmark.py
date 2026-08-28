"""Search quality benchmark with deterministic snapshots and optional live runs.

Snapshot mode is the CI/regression contract. Live mode uses the real gateway
and reports proxy judgments, so it is useful for diagnostics but never updates
the committed baseline.

Examples:
    python benchmarks/retrieval/run_quality_benchmark.py
    python benchmarks/retrieval/run_quality_benchmark.py --case q03_ukrainian_docs
    python benchmarks/retrieval/run_quality_benchmark.py --live --out live_report.json
    python benchmarks/retrieval/run_quality_benchmark.py --write-baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from remy.core.search_evaluation import (  # noqa: E402
    aggregate_case_metrics,
    compare_reports,
    evaluate_ranking,
)
from remy.core.search_gateway import (  # noqa: E402
    SearchGateway,
    SearchRequest,
    canonicalize_url,
    get_search_gateway,
)


class SnapshotProvider:
    def __init__(self, name: str, results: list[dict[str, Any]]) -> None:
        self.name = name
        self.results = results

    def search(self, _request: SearchRequest) -> list[dict[str, Any]]:
        return [dict(result) for result in self.results]


def _providers(items: list[dict[str, Any]] | None) -> list[SnapshotProvider]:
    return [
        SnapshotProvider(
            name=str(item.get("name") or f"snapshot-{position}"),
            results=list(item.get("results") or []),
        )
        for position, item in enumerate(items or [], start=1)
    ]


def _proxy_judgments(
    candidates: list[dict[str, Any]], expected_domains: list[str]
) -> dict[str, int]:
    """Create explicitly labelled live-run proxy grades from current output."""
    judgments: dict[str, int] = {}
    for candidate in candidates:
        url = str(candidate.get("uri") or "")
        host = url.lower()
        target = any(domain.lower() in host for domain in expected_domains)
        relevance = candidate.get("query_relevance") or {}
        if target:
            judgments[url] = 3
        elif relevance.get("relevant"):
            judgments[url] = 1
        else:
            judgments[url] = 0
    return judgments


def _case_passes(metrics: dict[str, Any], case: dict[str, Any], response) -> tuple[bool, list[str]]:
    thresholds = case.get("thresholds") or {}
    issues: list[str] = []
    minimums = {
        "recall_at_10": float(thresholds.get("min_recall_at_10", 0.0)),
        "ndcg_at_10": float(thresholds.get("min_ndcg_at_10", 0.0)),
        "mrr": float(thresholds.get("min_mrr", 0.0)),
        "independent_domain_count": int(thresholds.get("min_independent_domains", 1)),
    }
    for metric, minimum in minimums.items():
        actual = metrics.get(metric, 0)
        if actual < minimum:
            issues.append(f"{metric}={actual} is below {minimum}")
    if thresholds.get("require_target_domain") and not metrics.get("target_domain_hit"):
        issues.append("expected target domain is absent")
    maximum_duplicate_rate = float(thresholds.get("max_duplicate_rate", 1.0))
    if float(metrics.get("duplicate_rate", 0.0)) > maximum_duplicate_rate:
        issues.append(
            f"duplicate_rate={metrics.get('duplicate_rate')} exceeds {maximum_duplicate_rate}"
        )
    if "external_search_used" in thresholds:
        expected_external = bool(thresholds["external_search_used"])
        if bool(response.external_search_used) != expected_external:
            issues.append(
                "external_search_used="
                f"{response.external_search_used} expected {expected_external}"
            )
    return not issues, issues


def run_case(
    case: dict[str, Any],
    *,
    live: bool = False,
    enable_reranker: bool = True,
) -> dict[str, Any]:
    query = str(case["query"])
    max_results = int(case.get("max_results") or 10)
    expected_domains = list(case.get("expected_domains") or [])
    minimum_domains = int(
        (case.get("thresholds") or {}).get("min_independent_domains", 1)
    )

    if live:
        gateway = (
            get_search_gateway()
            if enable_reranker
            else SearchGateway(enable_local_reranker=False)
        )
        raw_count = None
        raw_unique_count = None
    else:
        providers = _providers(case.get("providers"))
        local_items = case.get("local_results")
        local_provider = (
            SnapshotProvider("local:web-index", list(local_items))
            if local_items is not None
            else None
        )
        gateway = SearchGateway(
            providers=providers,
            recovery_providers=[],
            local_provider=local_provider,
            enable_local_reranker=enable_reranker,
        )
        raw_results = [result for provider in providers for result in provider.results]
        if local_provider is not None:
            raw_results.extend(local_provider.results)
        raw_count = len(raw_results)
        raw_urls = {
            canonicalize_url(
                str(result.get("uri") or result.get("href") or result.get("url") or "")
            )
            for result in raw_results
        }
        raw_unique_count = len({url for url in raw_urls if url})

    response = gateway.search(SearchRequest(query=query, max_results=max_results))
    judgments = (
        _proxy_judgments(response.candidates, expected_domains)
        if live
        else case.get("judgments") or {}
    )
    metrics = evaluate_ranking(
        response.candidates,
        judgments,
        k=10,
        expected_domains=expected_domains,
        min_independent_domains=minimum_domains,
        raw_candidate_count=raw_count,
        raw_unique_candidate_count=raw_unique_count,
    )
    passed, issues = _case_passes(metrics, case, response)
    return {
        "id": case["id"],
        "category": case.get("category", "uncategorized"),
        "language": case.get("language", "unknown"),
        "query": query,
        "passed": passed,
        "issues": issues,
        "metrics": metrics,
        "duration_ms": response.duration_ms,
        "external_search_used": response.external_search_used,
        "diagnostics": response.diagnostics(),
        "top_results": [
            {
                "title": candidate.get("title"),
                "uri": candidate.get("uri"),
                "score": candidate.get("retrieval_score"),
                "source_class": candidate.get("source_class"),
                "grade": judgments.get(candidate.get("uri"), 0),
            }
            for candidate in response.candidates[:10]
        ],
    }


def build_report(
    document: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    live: bool,
    enable_reranker: bool = True,
) -> dict[str, Any]:
    defaults = document.get("defaults") or {}
    prepared: list[dict[str, Any]] = []
    for case in cases:
        merged = dict(defaults)
        merged.update(case)
        merged["thresholds"] = {
            **(defaults.get("thresholds") or {}),
            **(case.get("thresholds") or {}),
        }
        prepared.append(merged)
    results = [
        run_case(case, live=live, enable_reranker=enable_reranker)
        for case in prepared
    ]
    summary = aggregate_case_metrics(results)
    summary.update(
        {
            "passed_cases": sum(result["passed"] for result in results),
            "failed_cases": sum(not result["passed"] for result in results),
        }
    )
    categories = {
        category: aggregate_case_metrics(
            [result for result in results if result["category"] == category]
        )
        for category in sorted({result["category"] for result in results})
    }
    return {
        "schema_version": 2,
        "benchmark_version": document.get("version", 2),
        "mode": (
            "live_proxy" if live
            else "snapshot" if enable_reranker
            else "snapshot_without_reranker"
        ),
        "reranker_enabled": enable_reranker,
        "summary": summary,
        "categories": categories,
        "results": results,
    }


def _print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(
        "quality: "
        f"Recall@10={summary['recall_at_10']:.3f}  "
        f"nDCG@10={summary['ndcg_at_10']:.3f}  "
        f"MRR={summary['mrr']:.3f}"
    )
    print(
        "coverage: "
        f"domains={summary['independent_domain_rate']:.3f}  "
        f"target-hit={summary['target_domain_hit_rate']:.3f}  "
        f"duplicates={summary['duplicate_rate']:.3f}"
    )
    print(
        "runtime: "
        f"external={summary['external_search_rate']:.3f}  "
        f"p50={summary['latency_p50_ms']:.2f}ms  "
        f"p95={summary['latency_p95_ms']:.2f}ms"
    )
    print(
        f"cases: {summary['passed_cases']}/{summary['case_count']} passed"
    )


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    directory = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", default=str(directory / "benchmark_v2.yaml"))
    parser.add_argument("--case", help="Run one case id")
    parser.add_argument("--live", action="store_true", help="Use the real search gateway")
    parser.add_argument("--out", help="Write the full report as JSON")
    parser.add_argument(
        "--baseline", default=str(directory / "search_quality_baseline_v2.json")
    )
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--no-regression-gate", action="store_true")
    parser.add_argument(
        "--compare-reranker",
        action="store_true",
        help="Also run the snapshots without the local reranker and print A/B deltas.",
    )
    args = parser.parse_args()

    with Path(args.bench).open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    cases = list(document.get("cases") or [])
    if args.case:
        cases = [case for case in cases if case.get("id") == args.case]
        if not cases:
            print(f"Unknown case: {args.case}", file=sys.stderr)
            return 2

    report = build_report(document, cases, live=args.live, enable_reranker=True)
    for result in report["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['id']}: {result['query']}")
        for issue in result["issues"]:
            print(f"       ! {issue}")
    print()
    _print_summary(report)

    if args.compare_reranker:
        without = build_report(
            document,
            cases,
            live=args.live,
            enable_reranker=False,
        )
        metrics = ("recall_at_10", "ndcg_at_10", "mrr")
        report["ab_comparison"] = {
            "with_reranker": report["summary"],
            "without_reranker": without["summary"],
            "deltas": {
                metric: round(
                    float(report["summary"][metric])
                    - float(without["summary"][metric]),
                    4,
                )
                for metric in metrics
            },
            "category_deltas": {
                category: {
                    metric: round(
                        float(report["categories"][category][metric])
                        - float(without["categories"][category][metric]),
                        4,
                    )
                    for metric in metrics
                }
                for category in report["categories"]
                if category in without["categories"]
            },
        }
        print("reranker A/B:")
        for metric, delta in report["ab_comparison"]["deltas"].items():
            print(
                f"       {metric}: {without['summary'][metric]:.4f} -> "
                f"{report['summary'][metric]:.4f} ({delta:+.4f})"
            )
        multilingual_delta = report["ab_comparison"]["category_deltas"].get(
            "multilingual_reranking"
        )
        if multilingual_delta:
            print(
                "       hard multilingual nDCG@10: "
                f"{without['categories']['multilingual_reranking']['ndcg_at_10']:.4f} -> "
                f"{report['categories']['multilingual_reranking']['ndcg_at_10']:.4f} "
                f"({multilingual_delta['ndcg_at_10']:+.4f})"
            )

    baseline_path = Path(args.baseline)
    comparison = None
    if args.write_baseline:
        if args.live or args.case:
            print("Refusing to write a baseline from --live or a single --case run.", file=sys.stderr)
            return 2
        baseline_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"baseline written: {baseline_path}")
    elif not args.live and not args.case and baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        comparison = compare_reports(
            report,
            baseline,
            tolerances=document.get("regression_tolerances") or {},
        )
        report["comparison"] = comparison
        print(
            "regression gate: " + ("PASS" if comparison["passed"] else "FAIL")
        )
        for regression in comparison["regressions"]:
            print(
                f"       ! {regression['metric']}: "
                f"{regression['baseline']} -> {regression['current']}"
            )

    if args.out:
        Path(args.out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"report written: {args.out}")

    cases_pass = report["summary"]["failed_cases"] == 0
    gate_pass = comparison is None or comparison["passed"] or args.no_regression_gate
    return 0 if cases_pass and gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
