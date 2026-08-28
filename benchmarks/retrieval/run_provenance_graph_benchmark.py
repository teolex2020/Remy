"""Regression gate for local source authority and provenance analysis."""

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

from remy.core.source_provenance_graph import (  # noqa: E402
    apply_provenance_to_marginal_evidence,
    build_source_provenance_graph,
)


DEFAULT_BENCH = Path(__file__).with_name("provenance_graph_v1.yaml")
DEFAULT_BASELINE = Path(__file__).with_name("provenance_graph_baseline_v1.json")


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _source(url: str, unique: str, **extra: Any) -> dict[str, Any]:
    return {
        "url": url,
        "title": "Database consistency evidence",
        "content": (
            "Database consistency replication evidence covers quorum and transactions. "
            f"{unique} " * 18
        ),
        **extra,
    }


def _scenario(name: str) -> list[dict[str, Any]]:
    origin = "https://origin.example/study"
    if name == "independent":
        return [
            _source("https://one.example/a", "majority failure tolerance"),
            _source("https://two.example/b", "serializable anomaly prevention"),
            _source("https://three.example/c", "leader election replicated log"),
        ]
    if name == "syndicated":
        return [
            _source(
                f"https://edition{index}.example/report",
                f"originally published at {origin} edition {index}",
            )
            for index in range(1, 4)
        ]
    if name == "according":
        return [
            _source(
                f"https://news{index}.example/report",
                f"according to {origin} commentary {index}",
            )
            for index in range(1, 4)
        ]
    if name == "duplicates":
        content = _source(
            "https://one.example/report", "identical source material"
        )["content"]
        return [
            {"url": f"https://copy{index}.example/report", "content": content}
            for index in range(1, 4)
        ]
    if name == "official":
        return [
            _source(
                "https://docs.python.org/3/library/pathlib.html",
                "pathlib official api reference",
            )
        ]
    if name == "aggregator":
        return [
            _source(
                "https://mirror.example/report",
                "copied report material",
                evidence_packet={"source_class": "mirror", "ok": True},
            )
        ]
    if name == "mismatch":
        return [
            _source(
                "https://example.org/good",
                "verified identity evidence",
                evidence_packet={"ok": True},
            ),
            _source(
                "https://example.net/bad",
                "mismatched identity evidence",
                evidence_packet={"has_mismatch": True},
            ),
        ]
    if name == "independent_distinct":
        return [
            _source("https://alpha.example/a", "formal proof of quorum safety"),
            _source("https://beta.example/b", "failure injection experiment results"),
            _source("https://gamma.example/c", "production incident measurements"),
        ]
    raise ValueError(f"Unknown scenario: {name}")


def _role_counts(graph: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in graph.get("nodes") or []:
        role = str(node.get("authority_role") or "unknown")
        counts[role] = counts.get(role, 0) + 1
    return counts


def run(path: Path, case_filter: str = "") -> dict[str, Any]:
    spec = _load(path)
    cases = [
        case
        for case in spec.get("cases", [])
        if not case_filter or case["id"] == case_filter
    ]
    if not cases:
        raise SystemExit(f"No benchmark case matched: {case_filter!r}")

    rows = []
    for case in cases:
        sources = _scenario(str(case["scenario"]))
        graph = build_source_provenance_graph(sources)
        root_ok = graph["independent_root_count"] == int(case["expected_roots"])
        expected_roles = dict(case.get("expected_roles") or {})
        actual_roles = _role_counts(graph)
        role_ok = all(
            actual_roles.get(role, 0) == int(count)
            for role, count in expected_roles.items()
        )

        gate_ok = True
        gate_sufficient = None
        if "expected_gate_sufficient" in case:
            marginal = {
                "sufficient": True,
                "stop_recommended": True,
                "decision": "stop_sufficient",
                "accepted_urls": [source["url"] for source in sources],
                "reasons": [],
                "repair_queries": [],
            }
            gated = apply_provenance_to_marginal_evidence(marginal, graph)
            gate_sufficient = bool(gated["sufficient"])
            gate_ok = gate_sufficient is bool(case["expected_gate_sufficient"])

        authority_ok = True
        if case.get("expected_authority_order"):
            scores = {
                node["url"]: float(node["authority_score"])
                for node in graph.get("nodes") or []
            }
            authority_ok = scores[sources[0]["url"]] > scores[sources[1]["url"]]

        rows.append(
            {
                "id": case["id"],
                "expected_roots": int(case["expected_roots"]),
                "passed": root_ok and role_ok and gate_ok and authority_ok,
                "root_ok": root_ok,
                "role_ok": role_ok,
                "gate_ok": gate_ok,
                "authority_ok": authority_ok,
                "expected_roles": expected_roles,
                "actual_roles": actual_roles,
                "gate_sufficient": gate_sufficient,
                "graph": graph,
            }
        )

    role_rows = [row for row in rows if row["expected_roles"]]
    gate_rows = [row for row in rows if row["gate_sufficient"] is not None]
    false_consensus_rows = [
        row for row in gate_rows if row["expected_roots"] < 3
    ]
    metrics = {
        "root_accuracy": round(sum(row["root_ok"] for row in rows) / len(rows), 4),
        "role_accuracy": round(
            sum(row["role_ok"] for row in role_rows) / max(1, len(role_rows)), 4
        ),
        "consensus_gate_accuracy": round(
            sum(row["gate_ok"] for row in gate_rows) / max(1, len(gate_rows)), 4
        ),
        "false_consensus_rate": round(
            sum(bool(row["gate_sufficient"]) for row in false_consensus_rows)
            / max(1, len(false_consensus_rows)),
            4,
        ),
    }
    return {
        "version": 1,
        "case_count": len(rows),
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
    spec, report = _load(args.bench), run(args.bench, args.case)
    if args.out:
        args.out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.write_baseline:
        args.baseline.write_text(
            json.dumps({"version": 1, "metrics": report["metrics"]}, indent=2) + "\n",
            encoding="utf-8",
        )
    for row in report["cases"]:
        print(f'[{"PASS" if row["passed"] else "FAIL"}] {row["id"]}')
    metrics = report["metrics"]
    print(
        "provenance quality: "
        + "  ".join(f"{key}={value:.3f}" for key, value in metrics.items())
    )
    thresholds = dict(spec.get("thresholds") or {})
    passed = (
        metrics["root_accuracy"] >= float(thresholds.get("min_root_accuracy", 0))
        and metrics["role_accuracy"] >= float(thresholds.get("min_role_accuracy", 0))
        and metrics["consensus_gate_accuracy"]
        >= float(thresholds.get("min_consensus_gate_accuracy", 0))
        and metrics["false_consensus_rate"]
        <= float(thresholds.get("max_false_consensus_rate", 1))
        and report["passed_cases"] == report["case_count"]
    )
    print(f'cases: {report["passed_cases"]}/{report["case_count"]} passed')
    print(f'quality gate: {"PASS" if passed else "FAIL"}')
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
