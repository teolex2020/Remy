"""Regression gate for pre-fetch selection and post-fetch marginal gain."""

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

from remy.core.marginal_evidence_controller import (  # noqa: E402
    evaluate_marginal_evidence,
    prioritize_fetch_candidates,
)


DEFAULT_BENCH = Path(__file__).with_name("marginal_evidence_v1.yaml")
DEFAULT_BASELINE = Path(__file__).with_name("marginal_evidence_baseline_v1.json")


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _source(url: str, unique: str) -> dict[str, str]:
    return {
        "url": url,
        "title": "Database consistency evidence",
        "content": (
            "Database consistency replication evidence covers quorum and transactions. "
            f"{unique} " * 18
        ),
    }


def _gain_sources(scenario: str) -> list[dict[str, str]]:
    if scenario == "diverse":
        return [
            _source("https://one.example/a", "majority failure tolerance"),
            _source("https://two.example/b", "serializable anomaly prevention"),
            _source("https://three.example/c", "leader election replicated log"),
        ]
    if scenario == "duplicates":
        content = _source("https://one.example/a", "identical material")["content"]
        return [
            {"url": f"https://{host}.example/page", "content": content}
            for host in ("one", "two", "three")
        ]
    if scenario == "subdomains":
        return [
            _source("https://docs.publisher.co.uk/a", "majority failure tolerance"),
            _source("https://blog.publisher.co.uk/b", "serializable anomaly prevention"),
            _source("https://news.publisher.co.uk/c", "leader election replicated log"),
        ]
    if scenario == "irrelevant":
        return [
            {"url": "https://noise.example/a", "content": "cooking recipe ingredients " * 30},
            {"url": "https://ads.example/b", "content": "discount shopping promotion " * 30},
        ]
    return [
        _source("https://one.example/a", "majority failure tolerance"),
        _source("https://two.example/b", "serializable anomaly prevention"),
    ]


def _prefetch_candidates(scenario: str) -> list[dict[str, str]]:
    if scenario == "prefetch_duplicate":
        return [
            {"url": "https://one.example/a", "title": "Database consistency", "snippet": "database consistency quorum"},
            {"url": "https://one.example/a?utm_source=x", "title": "Database consistency", "snippet": "database consistency quorum"},
            {"url": "https://two.example/b", "title": "Database consistency", "snippet": "database consistency consensus"},
        ]
    if scenario == "prefetch_publisher":
        return [
            {"url": f"https://one.example/{suffix}", "title": "Database consistency", "snippet": f"database consistency {suffix}"}
            for suffix in ("a", "b", "c")
        ]
    return [
        {"url": "https://one.example/a", "title": "Database consistency", "snippet": "database consistency quorum"},
        {"url": "https://noise.example/ad", "title": "Sale", "snippet": "cheap shopping discount"},
    ]


def run(path: Path, case_filter: str = "") -> dict[str, Any]:
    spec = _load(path)
    cases = [case for case in spec.get("cases", []) if not case_filter or case["id"] == case_filter]
    if not cases:
        raise SystemExit(f"No benchmark case matched: {case_filter!r}")
    rows = []
    for case in cases:
        scenario = str(case["scenario"])
        if scenario.startswith("prefetch_"):
            result = prioritize_fetch_candidates(
                "database consistency", _prefetch_candidates(scenario), limit=3
            )
            checks = [result["selected_count"] == int(case["expected_selected"])]
            for expected_key, result_key in (
                ("expected_duplicate_suppressed", "duplicate_suppressed"),
                ("expected_publisher_suppressed", "publisher_suppressed"),
            ):
                if expected_key in case:
                    checks.append(result[result_key] == int(case[expected_key]))
            if "expected_irrelevant_rejected" in case:
                checks.append(
                    sum(item["reason"] == "irrelevant" for item in result["rejected"])
                    == int(case["expected_irrelevant_rejected"])
                )
            rows.append({"id": case["id"], "kind": "prefetch", "passed": all(checks), "result": result})
        else:
            result = evaluate_marginal_evidence(
                "database consistency", _gain_sources(scenario)
            )
            decision_ok = result["decision"] == case["expected_decision"]
            sufficient_ok = result["sufficient"] is bool(case["expected_sufficient"])
            accepted_ok = result["accepted_source_count"] == int(case["expected_accepted"])
            duplicate_ok = result["duplicate_rejected"] == int(case.get("expected_duplicates", result["duplicate_rejected"]))
            rows.append(
                {
                    "id": case["id"], "kind": "postfetch",
                    "passed": decision_ok and sufficient_ok and accepted_ok and duplicate_ok,
                    "decision_ok": decision_ok, "sufficient_ok": sufficient_ok,
                    "accepted_ok": accepted_ok, "duplicate_ok": duplicate_ok,
                    "result": result,
                }
            )
    post = [row for row in rows if row["kind"] == "postfetch"]
    pre = [row for row in rows if row["kind"] == "prefetch"]
    metrics = {
        "decision_accuracy": round(sum(row["decision_ok"] for row in post) / max(1, len(post)), 4),
        "sufficiency_accuracy": round(sum(row["sufficient_ok"] for row in post) / max(1, len(post)), 4),
        "duplicate_accuracy": round(sum(row["duplicate_ok"] for row in post) / max(1, len(post)), 4),
        "prefetch_accuracy": round(sum(row["passed"] for row in pre) / max(1, len(pre)), 4),
    }
    return {"version": 1, "case_count": len(rows), "passed_cases": sum(row["passed"] for row in rows), "metrics": metrics, "cases": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--case", default="")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()
    spec, report = _load(args.bench), run(args.bench, args.case)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.write_baseline:
        args.baseline.write_text(json.dumps({"version": 1, "metrics": report["metrics"]}, indent=2) + "\n", encoding="utf-8")
    for row in report["cases"]:
        print(f'[{"PASS" if row["passed"] else "FAIL"}] {row["id"]}')
    metrics = report["metrics"]
    print("marginal quality: " + "  ".join(f"{key}={value:.3f}" for key, value in metrics.items()))
    thresholds = dict(spec.get("thresholds") or {})
    passed = (
        metrics["decision_accuracy"] >= float(thresholds.get("min_decision_accuracy", 0))
        and metrics["sufficiency_accuracy"] >= float(thresholds.get("min_sufficiency_accuracy", 0))
        and metrics["duplicate_accuracy"] >= float(thresholds.get("min_duplicate_accuracy", 0))
        and metrics["prefetch_accuracy"] >= float(thresholds.get("min_prefetch_accuracy", 0))
        and report["passed_cases"] == report["case_count"]
    )
    print(f'cases: {report["passed_cases"]}/{report["case_count"]} passed')
    print(f'quality gate: {"PASS" if passed else "FAIL"}')
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
