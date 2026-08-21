"""Evaluate compact state-language formats for LLM prompt cost.

The goal is not generic compression. It is to test whether a deterministic
intermediate representation of session state can be shorter than natural
language while preserving answer quality.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from remy.core.session_state_wrapper import (  # noqa: E402
    SessionState,
    build_projected_state_from_log,
    estimate_tokens,
    recent_raw_turns,
)
from tools.validation.llm_optimization_corpus_eval import (  # noqa: E402
    DEFAULT_CASES,
    SYSTEM_PROMPT,
    CorpusCase,
    DryMeter,
    GeminiMeter,
    append_noise_turns,
    check_answer,
    load_cases,
    to_session_log,
)


STATE_LANGUAGE_MODES = (
    "raw",
    "state_verbose",
    "state_json",
    "state_kv",
    "state_symbolic",
    "state_kv_hybrid",
    "state_symbolic_hybrid",
)


def _parse_state_sections(snapshot: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = "STATE"
    for raw_line in (snapshot or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.fullmatch(r"\[([A-Z0-9_]+)\]", line)
        if match:
            current = match.group(1)
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return sections


def _compact_line(value: str) -> str:
    return " ".join(str(value or "").replace("\n", " ").split())


def render_state_verbose(state: SessionState, session_log: list[dict[str, str]]) -> str:
    return state.render(recent_raw_turns(session_log, 2))


def render_state_json(state: SessionState, session_log: list[dict[str, str]]) -> str:
    payload = {
        "state": _parse_state_sections(state.snapshot),
        "exact": state.pinned_facts,
        "recent": recent_raw_turns(session_log, 2).splitlines(),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def render_state_kv(state: SessionState, session_log: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for section, values in _parse_state_sections(state.snapshot).items():
        key = section.casefold()
        for value in values:
            lines.append(f"{key}={_compact_line(value)}")
    if state.pinned_facts:
        lines.append("exact=" + "|".join(_compact_line(item) for item in state.pinned_facts))
    recent = recent_raw_turns(session_log, 2)
    if recent:
        lines.append("recent=" + _compact_line(recent))
    return "\n".join(lines)


_SECTION_CODES = {
    "USER_PROFILE": "u",
    "CURRENT_GOAL": "g",
    "IMPORTANT_DECISIONS": "d",
    "OPEN_THREADS": "o",
    "CONSTRAINTS": "c",
    "DO_NOT_FORGET": "m",
    "DECISIONS_AND_PREFERENCES": "d",
    "DURABLE_SOURCE_LINES": "src",
    "STATE": "s",
}


def render_state_symbolic(state: SessionState, session_log: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for section, values in _parse_state_sections(state.snapshot).items():
        code = _SECTION_CODES.get(section, section.casefold())
        for value in values:
            parts.append(f"@{code}:{_compact_line(value)}")
    if state.pinned_facts:
        parts.append("@x:" + "|".join(_compact_line(item) for item in state.pinned_facts))
    recent = recent_raw_turns(session_log, 1)
    if recent:
        parts.append("@r:" + _compact_line(recent))
    return " ".join(parts)


def render_state_for_mode(
    state: SessionState,
    session_log: list[dict[str, str]],
    mode: str,
) -> str:
    base_mode = mode.removesuffix("_hybrid")
    if base_mode == "state_verbose":
        return render_state_verbose(state, session_log)
    if base_mode == "state_json":
        return render_state_json(state, session_log)
    if base_mode == "state_kv":
        return render_state_kv(state, session_log)
    if base_mode == "state_symbolic":
        return render_state_symbolic(state, session_log)
    raise ValueError(f"Unsupported state language mode: {mode}")


def build_state_language_prompt(
    case: CorpusCase,
    mode: str,
    answer_func: Any | None = None,
) -> tuple[str, dict[str, Any]]:
    from remy.core.session_state_wrapper import build_raw_prompt

    session_log = to_session_log(case)
    if mode == "raw":
        return build_raw_prompt(session_log, case.question, system_prompt=SYSTEM_PROMPT), {}

    uses_hybrid = mode.endswith("_hybrid")
    before_calls = getattr(answer_func, "calls", 0) if answer_func else 0
    before_prompt = getattr(answer_func, "prompt_tokens", 0) if answer_func else 0
    before_output = getattr(answer_func, "output_tokens", 0) if answer_func else 0
    state = build_projected_state_from_log(
        session_log,
        session_id=case.case_id,
        decision_llm_func=answer_func if uses_hybrid and callable(answer_func) else None,
    )
    decision_calls = (getattr(answer_func, "calls", 0) - before_calls) if answer_func else 0
    decision_prompt_tokens = (
        getattr(answer_func, "prompt_tokens", 0) - before_prompt
    ) if answer_func else 0
    decision_output_tokens = (
        getattr(answer_func, "output_tokens", 0) - before_output
    ) if answer_func else 0
    state_text = render_state_for_mode(state, session_log, mode)
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "You are given compact session state in a deterministic format. "
        "Treat keys as labels, preserve exact values, and answer the current request.\n\n"
        f"[STATE_FORMAT]\n{mode}\n\n"
        f"[SESSION_STATE]\n{state_text}\n\n"
        f"[CURRENT_REQUEST]\n{case.question}"
    )
    return prompt, {
        "state_format": mode,
        "state_build_strategy": "hybrid" if uses_hybrid else "facts",
        "state_tokens_estimate": estimate_tokens(state_text),
        "pinned_count": len(state.pinned_facts),
        "decision_extract_calls": decision_calls,
        "decision_extract_prompt_tokens": decision_prompt_tokens,
        "decision_extract_output_tokens": decision_output_tokens,
        "decision_extract_total_tokens": decision_prompt_tokens + decision_output_tokens,
        "state_preview": state_text[:600],
    }


async def run_state_language_eval(
    cases: list[CorpusCase],
    *,
    modes: list[str],
    model: str,
    dry_run: bool,
    delay_sec: float,
) -> dict[str, Any]:
    answer_func = DryMeter() if dry_run else GeminiMeter(model, delay_sec=delay_sec)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for case in cases:
        for mode in modes:
            before_prompt = getattr(answer_func, "prompt_tokens", 0)
            before_output = getattr(answer_func, "output_tokens", 0)
            before_calls = getattr(answer_func, "calls", 0)
            prompt, extra = build_state_language_prompt(case, mode, answer_func)
            answer_started = time.perf_counter()
            if dry_run:
                answer = " ".join(case.expected_fragments + [group[0] for group in case.expected_any])
            else:
                answer, _meta = answer_func(prompt)
            latency_ms = int((time.perf_counter() - answer_started) * 1000)
            after_prompt = getattr(answer_func, "prompt_tokens", 0)
            after_output = getattr(answer_func, "output_tokens", 0)
            after_calls = getattr(answer_func, "calls", 0)
            correctness = check_answer(
                answer,
                case.expected_fragments,
                case.must_not_contain,
                case.expected_any,
            )
            provider_prompt_tokens = after_prompt - before_prompt
            provider_output_tokens = after_output - before_output
            setup_prompt_tokens = int(extra.get("decision_extract_prompt_tokens") or 0)
            setup_output_tokens = int(extra.get("decision_extract_output_tokens") or 0)
            final_answer_provider_prompt_tokens = max(0, provider_prompt_tokens - setup_prompt_tokens)
            final_answer_provider_output_tokens = max(0, provider_output_tokens - setup_output_tokens)
            rows.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "risk": case.risk,
                    "mode": mode,
                    "correct": correctness["correct"],
                    "missing": correctness["missing"],
                    "missing_any": correctness["missing_any"],
                    "forbidden": correctness["forbidden"],
                    "prompt_tokens_estimate": estimate_tokens(prompt),
                    "provider_prompt_tokens": provider_prompt_tokens,
                    "provider_output_tokens": provider_output_tokens,
                    "provider_total_tokens": provider_prompt_tokens + provider_output_tokens,
                    "provider_setup_tokens": setup_prompt_tokens + setup_output_tokens,
                    "final_answer_provider_prompt_tokens": final_answer_provider_prompt_tokens,
                    "final_answer_provider_output_tokens": final_answer_provider_output_tokens,
                    "final_answer_provider_total_tokens": (
                        final_answer_provider_prompt_tokens + final_answer_provider_output_tokens
                    ),
                    "provider_calls": after_calls - before_calls,
                    "latency_ms": latency_ms,
                    "answer_preview": answer[:500],
                    **extra,
                }
            )

    raw_by_case = {row["case_id"]: row for row in rows if row["mode"] == "raw"}
    for row in rows:
        raw = raw_by_case.get(row["case_id"])
        if not raw:
            continue
        raw_estimate = int(raw["prompt_tokens_estimate"])
        prompt_estimate = int(row["prompt_tokens_estimate"])
        row["prompt_tokens_delta_vs_raw_estimate"] = raw_estimate - prompt_estimate
        row["context_window_saved_pct_vs_raw_estimate"] = round(
            ((raw_estimate - prompt_estimate) / max(1, raw_estimate)) * 100,
            2,
        )
        raw_provider_total = int(raw.get("provider_total_tokens") or 0)
        provider_total = int(row.get("provider_total_tokens") or 0)
        raw_final_prompt = int(raw.get("final_answer_provider_prompt_tokens") or 0)
        final_prompt = int(row.get("final_answer_provider_prompt_tokens") or 0)
        if raw_final_prompt and final_prompt:
            row["final_answer_provider_prompt_saved_pct_vs_raw"] = round(
                ((raw_final_prompt - final_prompt) / raw_final_prompt) * 100,
                2,
            )
        if raw_provider_total and provider_total:
            row["provider_total_delta_vs_raw"] = raw_provider_total - provider_total
            row["provider_total_saved_pct_vs_raw"] = round(
                ((raw_provider_total - provider_total) / raw_provider_total) * 100,
                2,
            )

    by_mode: dict[str, dict[str, Any]] = {}
    raw_provider_total = sum(int(row.get("provider_total_tokens") or 0) for row in rows if row["mode"] == "raw")
    raw_correct = sum(1 for row in rows if row["mode"] == "raw" and row["correct"])
    raw_cases = sum(1 for row in rows if row["mode"] == "raw")
    raw_accuracy = raw_correct / raw_cases if raw_cases else 0.0
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        correct = sum(1 for row in mode_rows if row["correct"])
        accuracy = correct / len(mode_rows) if mode_rows else 0.0
        raw_sum = sum(
            int(raw_by_case[row["case_id"]]["prompt_tokens_estimate"])
            for row in mode_rows
            if row["case_id"] in raw_by_case
        )
        delta = sum(int(row.get("prompt_tokens_delta_vs_raw_estimate") or 0) for row in mode_rows)
        provider_total = sum(int(row.get("provider_total_tokens") or 0) for row in mode_rows)
        provider_setup = sum(int(row.get("provider_setup_tokens") or 0) for row in mode_rows)
        final_prompt = sum(int(row.get("final_answer_provider_prompt_tokens") or 0) for row in mode_rows)
        final_answer_total = sum(int(row.get("final_answer_provider_total_tokens") or 0) for row in mode_rows)
        raw_final_prompt = sum(
            int(raw_by_case[row["case_id"]].get("final_answer_provider_prompt_tokens") or 0)
            for row in mode_rows
            if row["case_id"] in raw_by_case
        )
        raw_total_for_cases = sum(
            int(raw_by_case[row["case_id"]].get("provider_total_tokens") or 0)
            for row in mode_rows
            if row["case_id"] in raw_by_case
        )
        summary = {
            "cases": len(mode_rows),
            "accuracy": accuracy,
            "accuracy_delta_vs_raw": round(accuracy - raw_accuracy, 4) if mode != "raw" else 0.0,
            "correct": correct,
            "prompt_tokens_estimate": sum(int(row["prompt_tokens_estimate"]) for row in mode_rows),
            "context_window_saved_pct_vs_raw_estimate": round((delta / max(1, raw_sum)) * 100, 2),
            "state_tokens_estimate": sum(int(row.get("state_tokens_estimate") or 0) for row in mode_rows),
            "provider_prompt_tokens": sum(int(row.get("provider_prompt_tokens") or 0) for row in mode_rows),
            "provider_output_tokens": sum(int(row.get("provider_output_tokens") or 0) for row in mode_rows),
            "provider_total_tokens": provider_total,
            "provider_setup_tokens": provider_setup,
            "final_answer_provider_prompt_tokens": final_prompt,
            "final_answer_provider_total_tokens": final_answer_total,
            "provider_calls": sum(int(row.get("provider_calls") or 0) for row in mode_rows),
        }
        if raw_final_prompt and final_prompt:
            summary["final_answer_provider_prompt_saved_pct_vs_raw"] = round(
                ((raw_final_prompt - final_prompt) / raw_final_prompt) * 100,
                2,
            )
        if raw_provider_total and provider_total:
            summary["provider_total_saved_pct_vs_raw"] = round(
                ((raw_provider_total - provider_total) / raw_provider_total) * 100,
                2,
            )
        for repeat_count in (2, 3, 5, 10):
            if raw_total_for_cases and final_answer_total:
                raw_repeated_total = raw_total_for_cases * repeat_count
                optimized_repeated_total = provider_setup + final_answer_total * repeat_count
                summary[f"amortized_provider_total_saved_pct_vs_raw_at_{repeat_count}x"] = round(
                    ((raw_repeated_total - optimized_repeated_total) / raw_repeated_total) * 100,
                    2,
                )
        by_mode[mode] = summary

    return {
        "schema": "remy_state_language_eval_v1",
        "summary": {
            "model": model,
            "dry_run": dry_run,
            "cases": len(cases),
            "modes": modes,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "by_mode": by_mode,
        },
        "rows": rows,
    }


def _parse_modes(value: str) -> list[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [mode for mode in modes if mode not in STATE_LANGUAGE_MODES]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown state-language mode(s): {', '.join(unknown)}")
    return modes


def _write_report(report: dict[str, Any]) -> Path:
    from remy.config.settings import settings

    out_dir = Path(settings.DATA_DIR) / "llm_optimization"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"state_language_eval_{stamp}_{time.time_ns()}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


async def _main_async(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)[: args.case_limit if args.case_limit else None]
    cases = append_noise_turns(cases, args.append_noise_turns)
    report = await run_state_language_eval(
        cases,
        modes=args.mode,
        model=args.model,
        dry_run=args.dry_run,
        delay_sec=args.delay_sec,
    )
    path = _write_report(report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {path}")
    target = report["summary"]["by_mode"].get(args.target_mode, {})
    target_accuracy = float(target.get("accuracy") or 0.0)
    accuracy_delta = float(target.get("accuracy_delta_vs_raw") or 0.0)
    provider_saving = float(target.get("provider_total_saved_pct_vs_raw") or 0.0)
    passed = (
        target_accuracy >= args.min_accuracy
        and accuracy_delta >= -abs(args.max_accuracy_drop)
        and provider_saving >= args.min_provider_saving_pct
    )
    gate = {
        "target_mode": args.target_mode,
        "passed": passed,
        "min_accuracy": args.min_accuracy,
        "max_accuracy_drop": args.max_accuracy_drop,
        "min_provider_saving_pct": args.min_provider_saving_pct,
        "target_accuracy": target_accuracy,
        "target_accuracy_delta_vs_raw": accuracy_delta,
        "target_provider_total_saved_pct_vs_raw": provider_saving,
    }
    print("Quality gate:")
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--mode",
        type=_parse_modes,
        default=["raw", "state_verbose", "state_json", "state_kv", "state_symbolic"],
    )
    parser.add_argument("--target-mode", default="state_kv")
    parser.add_argument("--model", default="gemini-flash-lite-latest")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay-sec", type=float, default=0.0)
    parser.add_argument("--case-limit", type=int, default=10)
    parser.add_argument("--append-noise-turns", type=int, default=0)
    parser.add_argument("--min-accuracy", type=float, default=0.0)
    parser.add_argument("--max-accuracy-drop", type=float, default=1.0)
    parser.add_argument("--min-provider-saving-pct", type=float, default=0.0)
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
