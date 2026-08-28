"""Deterministic regression gate for claim-to-source binding quality."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from remy.core.claim_source_matrix import (  # noqa: E402
    build_claim_source_matrix,
    evaluate_claim_source_matrix,
)


_HIGHER_IS_BETTER = (
    "status_accuracy",
    "support_url_precision",
    "support_url_recall",
    "unsupported_recall",
    "conflict_recall",
    "corroboration_accuracy",
    "false_corroboration_recall",
    "temporal_status_accuracy",
    "stale_claim_recall",
    "supersession_accuracy",
    "publication_ready_accuracy",
)
_LOWER_IS_BETTER = (
    "false_support_rate",
    "unsafe_supersession_rate",
)


def build_report(document: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    for case in cases:
        reference_time = (
            datetime.fromisoformat(str(case["reference_time"]).replace("Z", "+00:00"))
            if case.get("reference_time")
            else None
        )
        matrix = build_claim_source_matrix(
            case.get("claims") or [],
            case.get("sources") or [],
            contradictions=case.get("contradictions") or [],
            now=reference_time,
        )
        metrics = evaluate_claim_source_matrix(matrix, case.get("expected") or [])
        results.append(
            {
                "id": case["id"],
                "passed": False,
                "metrics": metrics,
                "matrix": matrix,
                "issues": [],
            }
        )

    summary = {
        "case_count": len(results),
        **{
            metric: round(mean(result["metrics"][metric] for result in results), 4)
            if results else 0.0
            for metric in (*_HIGHER_IS_BETTER, *_LOWER_IS_BETTER)
        },
    }
    thresholds = document.get("thresholds") or {}
    for result in results:
        metrics = result["metrics"]
        issues = result["issues"]
        for metric in _HIGHER_IS_BETTER:
            minimum = float(thresholds.get(f"min_{metric}", 0.0))
            if float(metrics[metric]) < minimum:
                issues.append(f"{metric}={metrics[metric]} is below {minimum}")
        for metric in _LOWER_IS_BETTER:
            maximum = float(thresholds.get(f"max_{metric}", 1.0))
            if float(metrics[metric]) > maximum:
                issues.append(f"{metric}={metrics[metric]} exceeds {maximum}")
        result["passed"] = not issues
    summary["passed_cases"] = sum(result["passed"] for result in results)
    summary["failed_cases"] = sum(not result["passed"] for result in results)
    return {
        "schema_version": 1,
        "benchmark_version": document.get("version", 1),
        "summary": summary,
        "results": results,
    }


def compare_report(
    current: dict[str, Any], baseline: dict[str, Any], tolerances: dict[str, Any]
) -> dict[str, Any]:
    regressions = []
    for metric in (*_HIGHER_IS_BETTER, *_LOWER_IS_BETTER):
        now = float(current["summary"].get(metric, 0.0))
        before = float(baseline["summary"].get(metric, 0.0))
        tolerance = float(tolerances.get(metric, 0.0))
        delta = now - before
        regressed = delta > tolerance if metric in _LOWER_IS_BETTER else delta < -tolerance
        if regressed:
            regressions.append(
                {
                    "metric": metric,
                    "baseline": before,
                    "current": now,
                    "delta": round(delta, 4),
                    "tolerance": tolerance,
                }
            )
    return {"passed": not regressions, "regressions": regressions}


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    directory = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", default=str(directory / "claim_matrix_v1.yaml"))
    parser.add_argument("--case")
    parser.add_argument("--out")
    parser.add_argument(
        "--baseline", default=str(directory / "claim_matrix_baseline_v1.json")
    )
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()

    document = yaml.safe_load(Path(args.bench).read_text(encoding="utf-8")) or {}
    cases = list(document.get("cases") or [])
    if args.case:
        cases = [case for case in cases if case.get("id") == args.case]
        if not cases:
            print(f"Unknown case: {args.case}", file=sys.stderr)
            return 2
    report = build_report(document, cases)
    for result in report["results"]:
        print(f"[{'PASS' if result['passed'] else 'FAIL'}] {result['id']}")
        for issue in result["issues"]:
            print(f"       ! {issue}")
    summary = report["summary"]
    print(
        "matrix quality: "
        f"status={summary['status_accuracy']:.3f}  "
        f"URL-P={summary['support_url_precision']:.3f}  "
        f"URL-R={summary['support_url_recall']:.3f}  "
        f"unsupported-R={summary['unsupported_recall']:.3f}  "
        f"conflict-R={summary['conflict_recall']:.3f}  "
        f"corroboration={summary['corroboration_accuracy']:.3f}  "
        f"false-corroboration-R={summary['false_corroboration_recall']:.3f}  "
        f"temporal={summary['temporal_status_accuracy']:.3f}  "
        f"stale-R={summary['stale_claim_recall']:.3f}  "
        f"supersession={summary['supersession_accuracy']:.3f}  "
        f"false-support={summary['false_support_rate']:.3f}  "
        f"unsafe-supersession={summary['unsafe_supersession_rate']:.3f}"
    )
    print(f"cases: {summary['passed_cases']}/{summary['case_count']} passed")

    baseline_path = Path(args.baseline)
    comparison = None
    if args.write_baseline:
        if args.case:
            print("Refusing to write a partial baseline.", file=sys.stderr)
            return 2
        baseline_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"baseline written: {baseline_path}")
    elif not args.case and baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        comparison = compare_report(
            report, baseline, document.get("regression_tolerances") or {}
        )
        report["comparison"] = comparison
        print(f"regression gate: {'PASS' if comparison['passed'] else 'FAIL'}")
        for item in comparison["regressions"]:
            print(f"       ! {item['metric']}: {item['baseline']} -> {item['current']}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0 if summary["failed_cases"] == 0 and (
        comparison is None or comparison["passed"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
