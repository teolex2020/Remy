"""Run Remy's synthetic Tool Contract v2 routing evaluation."""

from __future__ import annotations

import argparse
import json

from remy.core.tool_contracts import ROUTING_BENCHMARK_CASES
from remy.core.tool_routing_benchmark import run_live_tool_routing_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate one connected model's tool selection without executing tools."
    )
    parser.add_argument("--model", default="", help="Exact connected model; default uses normal routing")
    parser.add_argument("--start", type=int, default=0, help="Skip the first N synthetic cases")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N synthetic cases")
    args = parser.parse_args()
    start = max(0, args.start)
    cases = ROUTING_BENCHMARK_CASES[start:]
    if args.limit:
        cases = cases[: max(0, args.limit)]
    report = run_live_tool_routing_benchmark(model=args.model, cases=cases)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
