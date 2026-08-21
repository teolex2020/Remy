"""Quality gate for LLM optimization corpus eval reports.

Reads a JSON report produced by ``llm_optimization_corpus_eval.py`` and fails
(non-zero exit) when the selected optimized mode does not meet product-quality
thresholds.

The point: make every future eval run pass/fail automatically instead of reading
dozens of numbers by hand. This unblocks corpus expansion, model matrices and
long-session stress tests; each just runs the eval and then this gate.

Checks against the selected target mode:
  1. accuracy >= --min-accuracy (default 0.95)
  2. effectiveness_rate >= --min-effectiveness (default 0.80)
  3. context_window_saved_pct_vs_raw >= --min-savings-pct (default 20.0)
     Savings are enforced only for long-session runs
     (append_noise_turns >= --long-session-noise, default 3).
  4. No false savings: a wrong answer must not claim saved context.
  5. Short negative cases must not be marked optimization_effective.
  6. Optional provider total-token savings gate when
     --min-provider-total-savings-pct is provided.

Exit code 0 = pass, 1 = fail, 2 = usage/parse error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "summary" not in data:
        raise ValueError("report has no 'summary' block - not a corpus eval report")
    return data


def _latest_report(reports_dir: Path) -> Path:
    candidates = sorted(reports_dir.glob("corpus_eval_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no corpus_eval_*.json in {reports_dir}")
    return candidates[-1]


def evaluate(
    report: dict[str, Any],
    *,
    min_accuracy: float,
    min_effectiveness: float,
    min_savings_pct: float,
    long_session_noise: int,
    min_provider_total_savings_pct: float | None = None,
    target_mode: str = "projected",
) -> tuple[bool, list[str], list[str]]:
    """Return (passed, failures, notes)."""
    summary = report.get("summary", {})
    by_mode = summary.get("by_mode", {})
    target = by_mode.get(target_mode)
    failures: list[str] = []
    notes: list[str] = []

    if target is None:
        return False, [f"report has no {target_mode!r} mode - run with --mode raw,{target_mode}"], notes

    accuracy = float(target.get("accuracy") or 0.0)
    effectiveness = float(target.get("effectiveness_rate") or 0.0)
    savings = float(target.get("context_window_saved_pct_vs_raw_estimate") or 0.0)
    provider_total_savings = target.get("provider_total_saved_pct_vs_raw")
    noise_value = summary.get("append_noise_turns")
    noise = int(noise_value) if noise_value is not None else 0
    is_long = noise >= long_session_noise

    if accuracy < min_accuracy:
        failures.append(f"accuracy {accuracy:.3f} < {min_accuracy:.2f}")
    else:
        notes.append(f"accuracy {accuracy:.3f} OK")

    if effectiveness < min_effectiveness:
        failures.append(f"effectiveness_rate {effectiveness:.3f} < {min_effectiveness:.2f}")
    else:
        notes.append(f"effectiveness_rate {effectiveness:.3f} OK")

    if is_long:
        if savings < min_savings_pct:
            failures.append(
                f"context savings {savings:.2f}% < {min_savings_pct:.1f}% "
                f"(long session, noise={noise})"
            )
        else:
            notes.append(f"context savings {savings:.2f}% OK (noise={noise})")
    else:
        notes.append(
            f"savings gate skipped (short session noise={noise} < {long_session_noise}); "
            f"observed {savings:.2f}%"
        )

    false_savings = 0
    short_negative_effective: list[str] = []
    for row in report.get("rows", []):
        if row.get("mode") != target_mode:
            continue
        correct = bool(row.get("correct"))
        effective = bool(row.get("optimization_effective"))
        saved = float(row.get("context_window_saved_pct_vs_raw_estimate") or 0.0)
        if not correct and (effective or saved > 0):
            false_savings += 1
        if not is_long and row.get("category") == "negative_case" and effective:
            short_negative_effective.append(str(row.get("case_id") or "<unknown>"))

    if false_savings:
        failures.append(
            f"{false_savings} false-saving case(s): wrong answer but claims saved tokens"
        )
    else:
        notes.append("no false savings")

    if short_negative_effective:
        preview = ", ".join(short_negative_effective[:5])
        suffix = "" if len(short_negative_effective) <= 5 else f", +{len(short_negative_effective) - 5} more"
        failures.append(
            "short negative case(s) marked effective despite short-session gate: "
            f"{preview}{suffix}"
        )
    elif not is_long:
        notes.append("short negative cases not marked effective")

    if min_provider_total_savings_pct is not None:
        if provider_total_savings is None:
            notes.append("provider total-token savings gate skipped (no provider tokens)")
        else:
            provider_total_savings = float(provider_total_savings)
            if provider_total_savings < min_provider_total_savings_pct:
                failures.append(
                    f"provider total-token savings {provider_total_savings:.2f}% "
                    f"< {min_provider_total_savings_pct:.1f}%"
                )
            else:
                notes.append(f"provider total-token savings {provider_total_savings:.2f}% OK")

    return (len(failures) == 0), failures, notes


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "report",
        nargs="?",
        type=Path,
        help="Path to a corpus_eval_*.json report. If omitted, uses the latest in data/llm_optimization/.",
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("data") / "llm_optimization")
    parser.add_argument("--min-accuracy", type=float, default=0.95)
    parser.add_argument("--min-effectiveness", type=float, default=0.80)
    parser.add_argument("--min-savings-pct", type=float, default=20.0)
    parser.add_argument(
        "--target-mode",
        default="projected",
        help=(
            "Optimized mode to gate. Use projected_facts for cost-first runs, "
            "projected_hybrid for decision/preference extraction, or projected "
            "for backward-compatible reports."
        ),
    )
    parser.add_argument(
        "--min-provider-total-savings-pct",
        type=float,
        default=None,
        help="Optional minimum total provider-token savings vs raw. Skipped when report has no provider tokens.",
    )
    parser.add_argument(
        "--long-session-noise",
        type=int,
        default=3,
        help="append_noise_turns at or above which the savings gate is enforced",
    )
    args = parser.parse_args()

    try:
        path = args.report or _latest_report(args.reports_dir)
        report = _load_report(path)
    except Exception as exc:  # noqa: BLE001 - surface any load error as usage error
        print(f"GATE ERROR: {exc}", file=sys.stderr)
        return 2

    passed, failures, notes = evaluate(
        report,
        min_accuracy=args.min_accuracy,
        min_effectiveness=args.min_effectiveness,
        min_savings_pct=args.min_savings_pct,
        min_provider_total_savings_pct=args.min_provider_total_savings_pct,
        long_session_noise=args.long_session_noise,
        target_mode=args.target_mode,
    )

    summary = report.get("summary", {})
    print(f"Report : {path}")
    print(
        f"Model  : {summary.get('model')}  dry_run={summary.get('dry_run')}  "
        f"cases={summary.get('cases')}  noise={summary.get('append_noise_turns')}  "
        f"target={args.target_mode}"
    )
    for note in notes:
        print(f"  ok   - {note}")
    for failure in failures:
        print(f"  FAIL - {failure}")
    print("RESULT :", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
