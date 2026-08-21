"""Run a small LLM optimization eval matrix and gate every run.

This is the product-facing runner: it compares models/noise levels using the
same corpus runner and quality gate, then writes one matrix report. It avoids
manual copy/paste runs where a strong savings number can hide a failed accuracy
gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tools.validation.llm_optimization_corpus_eval import (  # noqa: E402
    DEFAULT_CASES,
    _parse_modes,
    _write_report,
    append_noise_turns,
    load_cases,
    run_eval,
)
from tools.validation.llm_optimization_gate import evaluate as evaluate_gate  # noqa: E402


DEFAULT_MODE = ["raw", "projected_hybrid"]


@dataclass(frozen=True)
class MatrixConfig:
    cases_path: Path = DEFAULT_CASES
    models: tuple[str, ...] = ("gemini-flash-lite-latest",)
    modes: tuple[str, ...] = tuple(DEFAULT_MODE)
    noise_levels: tuple[int, ...] = (40,)
    target_mode: str = "projected_hybrid"
    dry_run: bool = False
    delay_sec: float = 0.0
    case_limit: int = 0
    fold_batch_size: int = 10
    context_window_tokens: int = 128000
    min_accuracy: float = 0.95
    min_effectiveness: float = 0.80
    min_savings_pct: float = 20.0
    min_provider_total_savings_pct: float | None = 20.0
    long_session_noise: int = 3


def _parse_csv(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("value must contain at least one item")
    return result


def _parse_int_csv(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("noise levels must be integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("value must contain at least one integer")
    if any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("noise levels must be >= 0")
    return result


def _write_matrix_report(report: dict[str, Any]) -> Path:
    from remy.config.settings import settings

    out_dir = Path(settings.DATA_DIR) / "llm_optimization"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"matrix_eval_{stamp}_{time.time_ns()}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


async def run_matrix(config: MatrixConfig) -> tuple[dict[str, Any], Path]:
    base_cases = load_cases(config.cases_path)
    if config.case_limit:
        base_cases = base_cases[: config.case_limit]

    started = time.perf_counter()
    runs: list[dict[str, Any]] = []
    all_passed = True

    for model in config.models:
        for noise in config.noise_levels:
            cases = append_noise_turns(base_cases, noise)
            eval_report = await run_eval(
                cases,
                modes=list(config.modes),
                model=model,
                dry_run=config.dry_run,
                delay_sec=config.delay_sec,
                fold_batch_size=config.fold_batch_size,
                context_window_tokens=config.context_window_tokens,
                append_noise_turns=noise,
            )
            eval_path = _write_report(eval_report)
            passed, failures, notes = evaluate_gate(
                eval_report,
                min_accuracy=config.min_accuracy,
                min_effectiveness=config.min_effectiveness,
                min_savings_pct=config.min_savings_pct,
                min_provider_total_savings_pct=config.min_provider_total_savings_pct,
                long_session_noise=config.long_session_noise,
                target_mode=config.target_mode,
            )
            all_passed = all_passed and passed
            target_summary = eval_report["summary"]["by_mode"].get(config.target_mode, {})
            runs.append(
                {
                    "model": model,
                    "noise": noise,
                    "dry_run": config.dry_run,
                    "cases": eval_report["summary"]["cases"],
                    "modes": list(config.modes),
                    "target_mode": config.target_mode,
                    "passed": passed,
                    "failures": failures,
                    "notes": notes,
                    "eval_report": str(eval_path),
                    "accuracy": target_summary.get("accuracy"),
                    "effectiveness_rate": target_summary.get("effectiveness_rate"),
                    "context_window_saved_pct_vs_raw_estimate": target_summary.get(
                        "context_window_saved_pct_vs_raw_estimate"
                    ),
                    "provider_total_saved_pct_vs_raw": target_summary.get(
                        "provider_total_saved_pct_vs_raw"
                    ),
                    "provider_total_tokens": target_summary.get("provider_total_tokens"),
                    "provider_calls": target_summary.get("provider_calls"),
                }
            )

    report = {
        "schema": "remy_llm_optimization_matrix_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "passed": all_passed,
        "config": {
            "cases_path": str(config.cases_path),
            "models": list(config.models),
            "modes": list(config.modes),
            "noise_levels": list(config.noise_levels),
            "target_mode": config.target_mode,
            "dry_run": config.dry_run,
            "case_limit": config.case_limit,
            "min_accuracy": config.min_accuracy,
            "min_effectiveness": config.min_effectiveness,
            "min_savings_pct": config.min_savings_pct,
            "min_provider_total_savings_pct": config.min_provider_total_savings_pct,
        },
        "runs": runs,
    }
    path = _write_matrix_report(report)
    return report, path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--models", type=_parse_csv, default=("gemini-flash-lite-latest",))
    parser.add_argument("--mode", type=_parse_modes, default=list(DEFAULT_MODE))
    parser.add_argument("--noise-levels", type=_parse_int_csv, default=(40,))
    parser.add_argument("--target-mode", default="projected_hybrid")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay-sec", type=float, default=0.0)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--fold-batch-size", type=int, default=10)
    parser.add_argument("--context-window-tokens", type=int, default=128000)
    parser.add_argument("--min-accuracy", type=float, default=0.95)
    parser.add_argument("--min-effectiveness", type=float, default=0.80)
    parser.add_argument("--min-savings-pct", type=float, default=20.0)
    parser.add_argument("--min-provider-total-savings-pct", type=float, default=20.0)
    parser.add_argument("--long-session-noise", type=int, default=3)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config = MatrixConfig(
        cases_path=args.cases,
        models=tuple(args.models),
        modes=tuple(args.mode),
        noise_levels=tuple(args.noise_levels),
        target_mode=args.target_mode,
        dry_run=args.dry_run,
        delay_sec=args.delay_sec,
        case_limit=args.case_limit,
        fold_batch_size=args.fold_batch_size,
        context_window_tokens=args.context_window_tokens,
        min_accuracy=args.min_accuracy,
        min_effectiveness=args.min_effectiveness,
        min_savings_pct=args.min_savings_pct,
        min_provider_total_savings_pct=args.min_provider_total_savings_pct,
        long_session_noise=args.long_session_noise,
    )
    report, path = asyncio.run(run_matrix(config))
    print(json.dumps({"passed": report["passed"], "runs": report["runs"]}, ensure_ascii=False, indent=2))
    print(f"Matrix report: {path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
