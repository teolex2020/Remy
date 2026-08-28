"""Deterministic regression gate for cross-run mutable claim lifecycle quality."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from statistics import mean
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from remy.core.claim_lifecycle import update_claim_lifecycle  # noqa: E402


_HIGHER_IS_BETTER = (
    "tracked_subject_accuracy",
    "current_value_accuracy",
    "transition_accuracy",
    "pending_accuracy",
    "history_preservation_accuracy",
)
_LOWER_IS_BETTER = ("unsafe_transition_rate",)


def _evaluate(ledger: dict[str, Any], expected: dict[str, Any]) -> dict[str, float]:
    subjects = list((ledger.get("subjects") or {}).values())
    summary = dict(ledger.get("summary") or {})
    current_values = {
        str((subject.get("current") or {}).get("value_signature") or "")
        for subject in subjects
    }
    wanted_value = str(expected.get("current_value") or "")
    history = [
        event
        for subject in subjects
        for event in subject.get("history") or []
        if isinstance(event, dict)
    ]
    actual_transitions = int(summary.get("confirmed_transitions") or 0)
    expected_transitions = int(expected.get("confirmed_transitions") or 0)
    expected_history = int(expected.get("history_events", len(history)) or 0)
    return {
        "tracked_subject_accuracy": float(
            int(summary.get("tracked_subjects") or 0)
            == int(expected.get("tracked_subjects") or 0)
        ),
        "current_value_accuracy": float(
            (wanted_value in current_values) if wanted_value else not current_values
        ),
        "transition_accuracy": float(actual_transitions == expected_transitions),
        "pending_accuracy": float(
            int(summary.get("pending_changes") or 0)
            == int(expected.get("pending_changes") or 0)
        ),
        "history_preservation_accuracy": float(
            len(history) == expected_history
            and all(event.get("history_preserved") is True for event in history)
        ),
        "unsafe_transition_rate": float(actual_transitions > expected_transitions),
    }


def build_report(document: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    thresholds = dict(document.get("thresholds") or {})
    for case in cases:
        ledger = None
        for run in case.get("runs") or []:
            ledger = update_claim_lifecycle(
                ledger,
                {"version": 4, "rows": list(run.get("rows") or [])},
                project_id=str(case.get("project_id") or "benchmark"),
                topic=str(case.get("topic") or ""),
                observed_at=datetime.fromisoformat(
                    str(run["observed_at"]).replace("Z", "+00:00")
                ),
            )
        ledger = ledger or {}
        metrics = _evaluate(ledger, dict(case.get("expected") or {}))
        issues = []
        for metric in _HIGHER_IS_BETTER:
            minimum = float(thresholds.get(f"min_{metric}", 0.0))
            if metrics[metric] < minimum:
                issues.append(f"{metric}={metrics[metric]} is below {minimum}")
        for metric in _LOWER_IS_BETTER:
            maximum = float(thresholds.get(f"max_{metric}", 1.0))
            if metrics[metric] > maximum:
                issues.append(f"{metric}={metrics[metric]} exceeds {maximum}")
        results.append(
            {
                "id": case["id"],
                "passed": not issues,
                "metrics": metrics,
                "ledger": ledger,
                "issues": issues,
            }
        )
    summary = {
        "case_count": len(results),
        "passed_cases": sum(result["passed"] for result in results),
        "failed_cases": sum(not result["passed"] for result in results),
        **{
            metric: round(mean(result["metrics"][metric] for result in results), 4)
            if results else 0.0
            for metric in (*_HIGHER_IS_BETTER, *_LOWER_IS_BETTER)
        },
    }
    return {
        "schema_version": 1,
        "benchmark_version": document.get("version", 1),
        "summary": summary,
        "results": results,
    }


def compare_report(current: dict[str, Any], baseline: dict[str, Any], tolerances: dict[str, Any]) -> dict[str, Any]:
    regressions = []
    for metric in (*_HIGHER_IS_BETTER, *_LOWER_IS_BETTER):
        now = float(current["summary"].get(metric, 0.0))
        before = float(baseline["summary"].get(metric, 0.0))
        tolerance = float(tolerances.get(metric, 0.0))
        delta = now - before
        regressed = delta > tolerance if metric in _LOWER_IS_BETTER else delta < -tolerance
        if regressed:
            regressions.append(
                {"metric": metric, "baseline": before, "current": now, "delta": round(delta, 4), "tolerance": tolerance}
            )
    return {"passed": not regressions, "regressions": regressions}


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    directory = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", default=str(directory / "claim_lifecycle_v1.yaml"))
    parser.add_argument("--case")
    parser.add_argument("--out")
    parser.add_argument("--baseline", default=str(directory / "claim_lifecycle_baseline_v1.json"))
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
        "lifecycle quality: "
        f"tracked={summary['tracked_subject_accuracy']:.3f}  "
        f"current={summary['current_value_accuracy']:.3f}  "
        f"transition={summary['transition_accuracy']:.3f}  "
        f"pending={summary['pending_accuracy']:.3f}  "
        f"history={summary['history_preservation_accuracy']:.3f}  "
        f"unsafe={summary['unsafe_transition_rate']:.3f}"
    )
    print(f"cases: {summary['passed_cases']}/{summary['case_count']} passed")
    baseline_path = Path(args.baseline)
    comparison = None
    if args.write_baseline:
        if args.case:
            print("Refusing to write a partial baseline.", file=sys.stderr)
            return 2
        baseline_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"baseline written: {baseline_path}")
    elif not args.case and baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        comparison = compare_report(report, baseline, document.get("regression_tolerances") or {})
        report["comparison"] = comparison
        print(f"regression gate: {'PASS' if comparison['passed'] else 'FAIL'}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if summary["failed_cases"] == 0 and (comparison is None or comparison["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
