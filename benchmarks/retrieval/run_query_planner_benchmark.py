"""Deterministic quality gate for the evidence-aware research query planner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from remy.core.research_query_planner import build_research_query_plan  # noqa: E402


DEFAULT_BENCH = Path(__file__).with_name("query_planner_v1.yaml")
DEFAULT_BASELINE = Path(__file__).with_name("query_planner_baseline_v1.json")


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _duplicate_rate(queries: list[dict[str, Any]]) -> float:
    keys = [" ".join(str(q.get("text") or "").casefold().split()) for q in queries]
    return (len(keys) - len(set(keys))) / len(keys) if keys else 0.0


def evaluate_case(case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = build_research_query_plan(
        str(case.get("topic") or ""),
        mode=str(case.get("mode") or "balanced"),
        source_scope=str(case.get("source_scope") or "web"),
        source_domains=list(case.get("source_domains") or []),
        seed_queries=list(case.get("seed_queries") or []),
        repair_queries=list(case.get("repair_queries") or []),
        max_queries=case.get("max_queries"),
    )
    queries = list(plan.get("queries") or [])
    intents = [str(query.get("intent") or "") for query in queries]
    required = list(case.get("required_intents") or [])
    matched = sum(intent in intents for intent in required)
    required_text = str(case.get("required_text") or "")
    first_intent = str(case.get("first_intent") or "")
    metrics = {
        "intent_recall": matched / len(required) if required else 1.0,
        "budget_compliance": float(len(queries) <= int(plan.get("budget") or 0)),
        "length_compliance": float(all(len(str(q.get("text") or "")) < 120 for q in queries)),
        "priority_accuracy": float(not first_intent or (intents and intents[0] == first_intent)),
        "duplicate_rate": _duplicate_rate(queries),
        "required_text_found": float(
            not required_text
            or any(required_text.casefold() in str(q.get("text") or "").casefold() for q in queries)
        ),
    }
    return metrics, plan


def run(bench_path: Path, case_filter: str = "") -> dict[str, Any]:
    spec = _load(bench_path)
    cases = [
        case for case in spec.get("cases", [])
        if not case_filter or str(case.get("id")) == case_filter
    ]
    if not cases:
        raise SystemExit(f"No benchmark case matched: {case_filter!r}")
    rows = []
    for case in cases:
        metrics, plan = evaluate_case(case)
        passed = (
            metrics["intent_recall"] == 1.0
            and metrics["budget_compliance"] == 1.0
            and metrics["length_compliance"] == 1.0
            and metrics["priority_accuracy"] == 1.0
            and metrics["duplicate_rate"] == 0.0
            and metrics["required_text_found"] == 1.0
        )
        rows.append({"id": case["id"], "passed": passed, "metrics": metrics, "plan": plan})
    aggregate = {
        key: round(sum(row["metrics"][key] for row in rows) / len(rows), 4)
        for key in rows[0]["metrics"]
    }
    return {
        "version": 1,
        "case_count": len(rows),
        "passed_cases": sum(row["passed"] for row in rows),
        "metrics": aggregate,
        "cases": rows,
    }


def _passes_thresholds(report: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    metrics = report["metrics"]
    return (
        metrics["intent_recall"] >= float(thresholds.get("min_intent_recall", 0))
        and metrics["budget_compliance"] >= float(thresholds.get("min_budget_compliance", 0))
        and metrics["length_compliance"] >= float(thresholds.get("min_length_compliance", 0))
        and metrics["priority_accuracy"] >= float(thresholds.get("min_priority_accuracy", 0))
        and metrics["duplicate_rate"] <= float(thresholds.get("max_duplicate_rate", 1))
        and report["passed_cases"] == report["case_count"]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--case", default="")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()

    spec = _load(args.bench)
    report = run(args.bench, args.case)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.write_baseline:
        args.baseline.write_text(
            json.dumps({"version": 1, "metrics": report["metrics"]}, indent=2) + "\n",
            encoding="utf-8",
        )

    for row in report["cases"]:
        print(f'[{"PASS" if row["passed"] else "FAIL"}] {row["id"]}')
    metrics = report["metrics"]
    print(
        "planner quality: "
        f'intent={metrics["intent_recall"]:.3f}  '
        f'budget={metrics["budget_compliance"]:.3f}  '
        f'length={metrics["length_compliance"]:.3f}  '
        f'priority={metrics["priority_accuracy"]:.3f}  '
        f'duplicates={metrics["duplicate_rate"]:.3f}'
    )
    passed = _passes_thresholds(report, dict(spec.get("thresholds") or {}))
    print(f'cases: {report["passed_cases"]}/{report["case_count"]} passed')
    print(f'quality gate: {"PASS" if passed else "FAIL"}')
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
