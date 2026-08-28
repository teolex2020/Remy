"""Deterministic regression gate for research execution accounting."""

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

from remy.core.research_execution_scheduler import (  # noqa: E402
    build_same_run_recovery,
    reconcile_execution_schedule,
)
from remy.core.research_query_planner import build_research_query_plan  # noqa: E402


DEFAULT_BENCH = Path(__file__).with_name("execution_scheduler_v1.yaml")
DEFAULT_BASELINE = Path(__file__).with_name("execution_scheduler_baseline_v1.json")


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _search(query: str, url: str) -> dict[str, Any]:
    return {
        "type": "tool_call",
        "tool": "web_search",
        "args": {"query": query},
        "result": {"results": [{"url": url, "title": "Evidence"}]},
    }


def _fetch(url: str, *, readable: bool = True) -> dict[str, Any]:
    return {
        "type": "tool_call",
        "tool": "extract_content",
        "args": {"url": url},
        "result": {"url": url, "content": ("Evidence material. " * 20) if readable else "short"},
    }


def _scenario_log(plan: dict[str, Any], scenario: str) -> list[dict[str, Any]]:
    queries = list(plan["queries"])
    if scenario == "no_tools":
        return []
    if scenario == "unrelated_search":
        return [_search("weather in Kyiv", "https://weather.example/today")]
    log: list[dict[str, Any]] = []
    selected = queries[:-1] if scenario == "missing_last_lane" else queries
    for index, item in enumerate(selected, start=1):
        query = str(item["text"])
        if scenario == "varied_complete":
            query = query.replace("documentation", "docs")
        host = "same.example" if scenario == "same_domain" else f"source{index}.example"
        url = f"https://{host}/evidence-{index}"
        log.append(_search(query, url))
        if scenario != "search_only":
            log.append(_fetch(url, readable=not (scenario == "unreadable" and index == 1)))
    return log


def run(path: Path, case_filter: str = "") -> dict[str, Any]:
    spec = _load(path)
    cases = [
        case for case in spec.get("cases", [])
        if not case_filter or str(case.get("id")) == case_filter
    ]
    if not cases:
        raise SystemExit(f"No benchmark case matched: {case_filter!r}")
    rows = []
    for case in cases:
        plan = build_research_query_plan("AI agent memory architecture")
        schedule = reconcile_execution_schedule(plan, _scenario_log(plan, case["scenario"]))
        recovery = build_same_run_recovery(
            schedule,
            worker_status=str(case.get("worker_status") or "success"),
        )
        status_ok = schedule["sufficient"] is bool(case["expected_sufficient"])
        action_ok = schedule["next_action"] == case["expected_next_action"]
        repair_ok = len(schedule["repair_queries"]) >= int(case.get("minimum_repairs", 0))
        recovery_ok = recovery["should_retry"] is bool(case.get("expected_retry"))
        recovery_bound_ok = (
            not recovery["should_retry"]
            or (
                int(recovery.get("step_budget") or 0) <= 4
                and int(recovery.get("timeout_sec") or 0) <= 30
                and len(recovery.get("actions") or []) <= 4
            )
        )
        rows.append(
            {
                "id": case["id"],
                "passed": (
                    status_ok and action_ok and repair_ok
                    and recovery_ok and recovery_bound_ok
                ),
                "status_ok": status_ok,
                "action_ok": action_ok,
                "repair_ok": repair_ok,
                "recovery_ok": recovery_ok,
                "recovery_bound_ok": recovery_bound_ok,
                "schedule": schedule,
                "recovery": recovery,
            }
        )
    count = len(rows)
    metrics = {
        "status_accuracy": round(sum(row["status_ok"] for row in rows) / count, 4),
        "next_action_accuracy": round(sum(row["action_ok"] for row in rows) / count, 4),
        "repair_accuracy": round(sum(row["repair_ok"] for row in rows) / count, 4),
        "recovery_decision_accuracy": round(
            sum(row["recovery_ok"] for row in rows) / count, 4
        ),
        "recovery_bound_compliance": round(
            sum(row["recovery_bound_ok"] for row in rows) / count, 4
        ),
        "false_completion_rate": round(
            sum(
                row["schedule"]["sufficient"]
                and not bool(case["expected_sufficient"])
                for row, case in zip(rows, cases)
            ) / count,
            4,
        ),
    }
    return {
        "version": 1,
        "case_count": count,
        "passed_cases": sum(row["passed"] for row in rows),
        "metrics": metrics,
        "cases": rows,
    }


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
        "scheduler quality: "
        f'status={metrics["status_accuracy"]:.3f}  '
        f'action={metrics["next_action_accuracy"]:.3f}  '
        f'repair={metrics["repair_accuracy"]:.3f}  '
        f'recovery={metrics["recovery_decision_accuracy"]:.3f}  '
        f'bounds={metrics["recovery_bound_compliance"]:.3f}  '
        f'false-complete={metrics["false_completion_rate"]:.3f}'
    )
    thresholds = dict(spec.get("thresholds") or {})
    passed = (
        metrics["status_accuracy"] >= float(thresholds.get("min_status_accuracy", 0))
        and metrics["next_action_accuracy"] >= float(thresholds.get("min_next_action_accuracy", 0))
        and metrics["repair_accuracy"] >= float(thresholds.get("min_repair_accuracy", 0))
        and metrics["recovery_decision_accuracy"] >= float(
            thresholds.get("min_recovery_decision_accuracy", 0)
        )
        and metrics["recovery_bound_compliance"] >= float(
            thresholds.get("min_recovery_bound_compliance", 0)
        )
        and metrics["false_completion_rate"] <= float(thresholds.get("max_false_completion_rate", 1))
        and report["passed_cases"] == report["case_count"]
    )
    print(f'cases: {report["passed_cases"]}/{report["case_count"]} passed')
    print(f'quality gate: {"PASS" if passed else "FAIL"}')
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
