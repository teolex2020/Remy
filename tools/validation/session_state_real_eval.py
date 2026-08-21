"""Run real Gemini evals for Remy's session-state prompt.

Two modes are intentionally separate:

* projected: builds a deterministic no-LLM state projection from transcript.
* fixture: uses trusted fixture snapshots to test the final prompt wrapper.
* incremental: asks the model to build the snapshot turn by turn, then answers.

The second mode is the important production proof because it includes snapshot
update cost and snapshot drift risk.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


SYSTEM_PROMPT = (
    "You are Remy. Answer the current request using only the provided context. "
    "If an exact value is present, output it exactly. Do not invent missing values."
)


def _mg() -> str:
    return "\u043c\u0433"


def _uah() -> str:
    return "\u0433\u0440\u043d"


def _noise(prefix: str, count: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in range(count):
        rows.append(
            {
                "type": "user_text",
                "text": (
                    f"{prefix} routine note {index}: scheduling, formatting, "
                    "and unrelated planning details."
                ),
            }
        )
        rows.append({"type": "model_response", "text": f"Acknowledged routine note {index}."})
    return rows


def _cases(noise_turns: int):
    from remy.core.session_state_wrapper import SessionStateEvalCase

    return [
        SessionStateEvalCase(
            case_id="early_authorization_code",
            session_log=[
                {"type": "user_text", "text": "Critical fact: authorization code is RX-4471."},
                {"type": "model_response", "text": "Stored the authorization code."},
                *_noise("code", noise_turns),
            ],
            user_request="What is the authorization code? Answer with only the code.",
            expected_fragments=["RX-4471"],
            snapshot="[DO_NOT_FORGET]\nAuthorization code: RX-4471.",
        ),
        SessionStateEvalCase(
            case_id="ukrainian_dosage",
            session_log=[
                {"type": "user_text", "text": f"Critical fact: morning dosage is 2,5 {_mg()}."},
                {"type": "model_response", "text": "Stored the morning dosage."},
                *_noise("dose", noise_turns),
            ],
            user_request=(
                "\u042f\u043a\u0435 \u0440\u0430\u043d\u043a\u043e\u0432\u0435 "
                "\u0434\u043e\u0437\u0443\u0432\u0430\u043d\u043d\u044f? "
                "\u0412\u0456\u0434\u043f\u043e\u0432\u0456\u0434\u0430\u0439 "
                "\u0442\u0456\u043b\u044c\u043a\u0438 \u0437\u043d\u0430\u0447\u0435\u043d\u043d\u044f\u043c."
            ),
            expected_fragments=[f"2,5 {_mg()}"],
            snapshot=f"[DO_NOT_FORGET]\nMorning dosage: 2,5 {_mg()}.",
        ),
        SessionStateEvalCase(
            case_id="deadline_datetime",
            session_log=[
                {
                    "type": "user_text",
                    "text": "Critical fact: final report deadline is 2026-07-15 14:30.",
                },
                {"type": "model_response", "text": "Stored the report deadline."},
                *_noise("deadline", noise_turns),
            ],
            user_request="What is the final report deadline? Answer only the date and time.",
            expected_fragments=["2026-07-15 14:30"],
            snapshot="[OPEN_THREADS]\nFinal report deadline: 2026-07-15 14:30.",
        ),
        SessionStateEvalCase(
            case_id="support_email",
            session_log=[
                {
                    "type": "user_text",
                    "text": "Critical fact: support email is clinic.ops@example.com.",
                },
                {"type": "model_response", "text": "Stored the support email."},
                *_noise("email", noise_turns),
            ],
            user_request="Which support email was provided? Answer only the email.",
            expected_fragments=["clinic.ops@example.com"],
            snapshot="[DO_NOT_FORGET]\nSupport email: clinic.ops@example.com.",
        ),
        SessionStateEvalCase(
            case_id="windows_file_path",
            session_log=[
                {
                    "type": "user_text",
                    "text": r"Critical fact: workflow file is E:\remy\workflow.txt.",
                },
                {"type": "model_response", "text": "Stored the workflow file path."},
                *_noise("path", noise_turns),
            ],
            user_request="What is the workflow file path? Answer only the path.",
            expected_fragments=[r"E:\remy\workflow.txt"],
            snapshot=r"[DO_NOT_FORGET]\nWorkflow file: E:\remy\workflow.txt.",
        ),
        SessionStateEvalCase(
            case_id="uah_budget",
            session_log=[
                {"type": "user_text", "text": f"Critical fact: approved budget is 1500 {_uah()}."},
                {"type": "model_response", "text": "Stored the approved budget."},
                *_noise("budget", noise_turns),
            ],
            user_request=(
                "\u042f\u043a\u0438\u0439 \u0437\u0430\u0442\u0432\u0435\u0440\u0434\u0436\u0435\u043d\u0438\u0439 "
                "\u0431\u044e\u0434\u0436\u0435\u0442? "
                "\u0412\u0456\u0434\u043f\u043e\u0432\u0456\u0434\u0430\u0439 "
                "\u0442\u0456\u043b\u044c\u043a\u0438 \u0441\u0443\u043c\u043e\u044e."
            ),
            expected_fragments=[f"1500 {_uah()}"],
            snapshot=f"[IMPORTANT_DECISIONS]\nApproved budget: 1500 {_uah()}.",
        ),
    ]


def _usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {}
    return {
        "prompt_token_count": getattr(usage, "prompt_token_count", None),
        "candidates_token_count": getattr(usage, "candidates_token_count", None),
    }


class GeminiMeter:
    def __init__(self, model: str, *, delay_sec: float = 0.0):
        from google import genai
        from remy.config.settings import settings

        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model = model
        self.delay_sec = max(0.0, delay_sec)
        self.calls = 0
        self.prompt_tokens = 0
        self.output_tokens = 0

    def __call__(self, prompt: str):
        time.sleep(self.delay_sec)
        response = self.client.models.generate_content(model=self.model, contents=prompt)
        usage = _usage_dict(response)
        self.calls += 1
        self.prompt_tokens += int(usage.get("prompt_token_count") or 0)
        self.output_tokens += int(usage.get("candidates_token_count") or 0)
        return (getattr(response, "text", "") or "").strip(), {"usage_metadata": usage}


def _answer_contains_expected(answer: str, expected_fragments: list[str]) -> bool:
    normalized = (answer or "").casefold()
    return all(fragment.casefold() in normalized for fragment in expected_fragments)


def _exchange_text(items: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for item in items:
        kind = item.get("type")
        if kind == "user_text":
            lines.append(f"user: {item.get('text', '')}")
        elif kind == "model_response":
            lines.append(f"assistant: {item.get('text', '')}")
    return "\n".join(lines)


def _exchanges(session_log: list[dict]) -> list[str]:
    result: list[str] = []
    pending: list[dict] = []
    for item in session_log:
        pending.append(item)
        if item.get("type") == "model_response":
            text = _exchange_text(pending)
            if text:
                result.append(text)
            pending = []
    if pending:
        text = _exchange_text(pending)
        if text:
            result.append(text)
    return result


async def _run_fixture_eval(model: str, noise_turns: int, delay_sec: float):
    from remy.core.session_state_wrapper import run_session_state_eval

    report = await run_session_state_eval(
        _cases(noise_turns),
        GeminiMeter(model, delay_sec=delay_sec),
        system_prompt=SYSTEM_PROMPT,
    )
    report["mode"] = "fixture"
    report["limitations"] = [
        "Snapshots are fixture summaries, not generated incrementally in this run.",
        "Provider usage covers final answer calls only; snapshot-update costs are not included.",
    ]
    return report


async def _run_projected_eval(model: str, noise_turns: int, delay_sec: float):
    from remy.core.session_state_wrapper import (
        build_optimized_prompt,
        build_projected_state_from_log,
        build_raw_prompt,
        estimate_tokens,
    )

    rows: list[dict[str, Any]] = []
    raw_correct = 0
    optimized_correct = 0
    raw_prompt_tokens = 0
    optimized_prompt_tokens = 0
    raw_output_tokens = 0
    optimized_output_tokens = 0

    for case in _cases(noise_turns):
        raw_meter = GeminiMeter(model, delay_sec=delay_sec)
        opt_meter = GeminiMeter(model, delay_sec=delay_sec)
        state = build_projected_state_from_log(case.session_log, session_id=f"projected-{case.case_id}")
        raw_prompt = build_raw_prompt(case.session_log, case.user_request, system_prompt=SYSTEM_PROMPT)
        opt_prompt = build_optimized_prompt(state, case.session_log, case.user_request, system_prompt=SYSTEM_PROMPT)
        raw_answer, _ = raw_meter(raw_prompt)
        opt_answer, _ = opt_meter(opt_prompt)
        raw_ok = _answer_contains_expected(raw_answer, case.expected_fragments)
        opt_ok = _answer_contains_expected(opt_answer, case.expected_fragments)

        raw_correct += int(raw_ok)
        optimized_correct += int(opt_ok)
        raw_prompt_tokens += raw_meter.prompt_tokens
        optimized_prompt_tokens += opt_meter.prompt_tokens
        raw_output_tokens += raw_meter.output_tokens
        optimized_output_tokens += opt_meter.output_tokens

        rows.append(
            {
                "case_id": case.case_id,
                "raw_correct": raw_ok,
                "optimized_correct": opt_ok,
                "expected_fragments": case.expected_fragments,
                "raw_answer": raw_answer,
                "optimized_answer": opt_answer,
                "pinned_facts": state.pinned_facts,
                "raw_provider_prompt_tokens": raw_meter.prompt_tokens,
                "optimized_provider_prompt_tokens": opt_meter.prompt_tokens,
                "raw_provider_output_tokens": raw_meter.output_tokens,
                "optimized_provider_output_tokens": opt_meter.output_tokens,
                "prompt_reduction_ratio": round(
                    raw_meter.prompt_tokens / max(1, opt_meter.prompt_tokens),
                    2,
                ),
                "raw_prompt_tokens_estimate": estimate_tokens(raw_prompt),
                "optimized_prompt_tokens_estimate": estimate_tokens(opt_prompt),
            }
        )

    case_count = max(1, len(rows))
    raw_total = raw_prompt_tokens + raw_output_tokens
    optimized_total = optimized_prompt_tokens + optimized_output_tokens
    return {
        "schema": "remy_session_state_projected_eval_v1",
        "mode": "projected",
        "cases": rows,
        "summary": {
            "case_count": len(rows),
            "raw_accuracy": round(raw_correct / case_count, 4),
            "optimized_accuracy": round(optimized_correct / case_count, 4),
            "raw_correct": raw_correct,
            "optimized_correct": optimized_correct,
            "raw_provider_prompt_tokens": raw_prompt_tokens,
            "optimized_provider_prompt_tokens": optimized_prompt_tokens,
            "provider_prompt_tokens_saved": raw_prompt_tokens - optimized_prompt_tokens,
            "provider_prompt_token_reduction_ratio": round(
                raw_prompt_tokens / max(1, optimized_prompt_tokens),
                2,
            ),
            "raw_total_tokens": raw_total,
            "optimized_total_tokens": optimized_total,
            "end_to_end_token_ratio": round(raw_total / max(1, optimized_total), 2),
            "snapshot_update_total_tokens": 0,
            "production_ready": optimized_correct == len(rows)
            and raw_correct == len(rows)
            and optimized_total < raw_total,
        },
        "limitations": [
            "No LLM snapshot update is used; this only preserves exact pinned facts and recent raw turns.",
            "This path is best for exact facts/codes/dates/amounts, not broad prose summarization.",
        ],
    }


async def _run_incremental_eval(
    model: str,
    noise_turns: int,
    delay_sec: float,
    case_limit: int,
    fold_batch_size: int,
):
    from remy.core.session_state_wrapper import (
        SessionState,
        build_optimized_prompt,
        build_raw_prompt,
        estimate_tokens,
        update_state_incremental,
    )

    rows: list[dict[str, Any]] = []
    raw_correct = 0
    optimized_correct = 0
    raw_final_prompt_tokens = 0
    optimized_final_prompt_tokens = 0
    raw_final_output_tokens = 0
    optimized_final_output_tokens = 0
    update_prompt_tokens = 0
    update_output_tokens = 0

    cases = _cases(noise_turns)[:case_limit]
    for case in cases:
        update_meter = GeminiMeter(model, delay_sec=delay_sec)
        raw_answer_meter = GeminiMeter(model, delay_sec=delay_sec)
        optimized_answer_meter = GeminiMeter(model, delay_sec=delay_sec)
        state = SessionState(session_id=f"incremental-{case.case_id}")

        exchanges = _exchanges(case.session_log)
        batch_size = max(1, fold_batch_size)
        for start in range(0, len(exchanges), batch_size):
            exchange = "\n\n".join(exchanges[start:start + batch_size])
            state = await update_state_incremental(
                state,
                exchange,
                update_meter,
                persist=False,
            )

        raw_prompt = build_raw_prompt(case.session_log, case.user_request, system_prompt=SYSTEM_PROMPT)
        optimized_prompt = build_optimized_prompt(
            state,
            case.session_log,
            case.user_request,
            system_prompt=SYSTEM_PROMPT,
        )
        raw_answer, _raw_meta = raw_answer_meter(raw_prompt)
        optimized_answer, _opt_meta = optimized_answer_meter(optimized_prompt)

        raw_ok = _answer_contains_expected(raw_answer, case.expected_fragments)
        optimized_ok = _answer_contains_expected(optimized_answer, case.expected_fragments)
        raw_correct += int(raw_ok)
        optimized_correct += int(optimized_ok)

        raw_final_prompt_tokens += raw_answer_meter.prompt_tokens
        optimized_final_prompt_tokens += optimized_answer_meter.prompt_tokens
        raw_final_output_tokens += raw_answer_meter.output_tokens
        optimized_final_output_tokens += optimized_answer_meter.output_tokens
        update_prompt_tokens += update_meter.prompt_tokens
        update_output_tokens += update_meter.output_tokens

        optimized_total_tokens = (
            update_meter.prompt_tokens
            + update_meter.output_tokens
            + optimized_answer_meter.prompt_tokens
            + optimized_answer_meter.output_tokens
        )
        raw_answer_total_tokens = raw_answer_meter.prompt_tokens + raw_answer_meter.output_tokens
        rows.append(
            {
                "case_id": case.case_id,
                "fold_calls": update_meter.calls,
                "fold_batch_size": batch_size,
                "source_exchanges": len(exchanges),
                "raw_correct": raw_ok,
                "optimized_correct": optimized_ok,
                "expected_fragments": case.expected_fragments,
                "raw_answer": raw_answer,
                "optimized_answer": optimized_answer,
                "snapshot": state.snapshot,
                "pinned_facts": state.pinned_facts,
                "raw_final_prompt_tokens": raw_answer_meter.prompt_tokens,
                "optimized_final_prompt_tokens": optimized_answer_meter.prompt_tokens,
                "snapshot_update_prompt_tokens": update_meter.prompt_tokens,
                "snapshot_update_output_tokens": update_meter.output_tokens,
                "raw_answer_output_tokens": raw_answer_meter.output_tokens,
                "optimized_answer_output_tokens": optimized_answer_meter.output_tokens,
                "final_prompt_reduction_ratio": round(
                    raw_answer_meter.prompt_tokens / max(1, optimized_answer_meter.prompt_tokens),
                    2,
                ),
                "raw_answer_total_tokens": raw_answer_total_tokens,
                "optimized_total_tokens_with_updates": optimized_total_tokens,
                "end_to_end_token_ratio_raw_answer_vs_optimized_with_updates": round(
                    raw_answer_total_tokens / max(1, optimized_total_tokens),
                    4,
                ),
                "raw_prompt_tokens_estimate": estimate_tokens(raw_prompt),
                "optimized_prompt_tokens_estimate": estimate_tokens(optimized_prompt),
            }
        )

    case_count = max(1, len(cases))
    raw_final_total = raw_final_prompt_tokens + raw_final_output_tokens
    optimized_final_total = optimized_final_prompt_tokens + optimized_final_output_tokens
    update_total = update_prompt_tokens + update_output_tokens
    optimized_total = (
        update_total
        + optimized_final_total
    )
    cost_effective_single_final_answer = optimized_total < raw_final_total
    per_answer_savings_after_state = raw_final_total - optimized_final_total
    if per_answer_savings_after_state > 0:
        break_even_final_answer_count = (update_total + per_answer_savings_after_state - 1) // per_answer_savings_after_state
    else:
        break_even_final_answer_count = None
    amortized_three_answers_total = update_total + (3 * optimized_final_total)
    raw_three_answers_total = 3 * raw_final_total
    return {
        "schema": "remy_session_state_incremental_eval_v1",
        "mode": "incremental",
        "cases": rows,
        "summary": {
            "case_count": len(cases),
            "raw_accuracy": round(raw_correct / case_count, 4),
            "optimized_accuracy": round(optimized_correct / case_count, 4),
            "raw_correct": raw_correct,
            "optimized_correct": optimized_correct,
            "raw_final_prompt_tokens": raw_final_prompt_tokens,
            "optimized_final_prompt_tokens": optimized_final_prompt_tokens,
            "raw_final_output_tokens": raw_final_output_tokens,
            "optimized_final_output_tokens": optimized_final_output_tokens,
            "raw_final_total_tokens": raw_final_total,
            "optimized_final_total_tokens": optimized_final_total,
            "final_prompt_tokens_saved": raw_final_prompt_tokens - optimized_final_prompt_tokens,
            "final_prompt_reduction_ratio": round(
                raw_final_prompt_tokens / max(1, optimized_final_prompt_tokens),
                2,
            ),
            "snapshot_update_prompt_tokens": update_prompt_tokens,
            "snapshot_update_output_tokens": update_output_tokens,
            "snapshot_update_total_tokens": update_total,
            "optimized_total_tokens_with_updates": optimized_total,
            "end_to_end_token_ratio_raw_answer_vs_optimized_with_updates": round(
                raw_final_total / max(1, optimized_total),
                4,
            ),
            "per_answer_savings_after_state_tokens": per_answer_savings_after_state,
            "break_even_final_answer_count": break_even_final_answer_count,
            "amortized_three_answers_total_tokens": amortized_three_answers_total,
            "raw_three_answers_total_tokens": raw_three_answers_total,
            "cost_effective_after_three_final_answers": (
                amortized_three_answers_total < raw_three_answers_total
            ),
            "cost_effective_single_final_answer": cost_effective_single_final_answer,
            "accuracy_ready": optimized_correct == len(cases) and raw_correct == len(cases),
            "answer_prompt_smaller": optimized_final_prompt_tokens < raw_final_prompt_tokens,
            "production_ready": optimized_correct == len(cases)
            and raw_correct == len(cases)
            and optimized_final_prompt_tokens < raw_final_prompt_tokens
            and cost_effective_single_final_answer,
        },
        "limitations": [
            "This run includes batched snapshot update calls and final answer calls.",
            "The end-to-end ratio compares one final raw answer call against all snapshot update calls plus the final optimized answer call.",
            "For a real chat product, snapshot updates may be amortized over many future requests in the same session.",
        ],
    }


async def _main() -> int:
    from remy.config.settings import settings

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-flash-lite-latest")
    parser.add_argument("--noise-turns", type=int, default=40)
    parser.add_argument("--delay-sec", type=float, default=0.05)
    parser.add_argument("--mode", choices=["projected", "fixture", "incremental"], default="fixture")
    parser.add_argument("--case-limit", type=int, default=3)
    parser.add_argument("--fold-batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.mode == "projected":
        report = await _run_projected_eval(args.model, args.noise_turns, args.delay_sec)
    elif args.mode == "fixture":
        report = await _run_fixture_eval(args.model, args.noise_turns, args.delay_sec)
    else:
        report = await _run_incremental_eval(
            args.model,
            args.noise_turns,
            args.delay_sec,
            max(1, args.case_limit),
            max(1, args.fold_batch_size),
        )

    report["model"] = args.model
    report["noise_turns_per_case"] = args.noise_turns
    report["fold_batch_size"] = args.fold_batch_size if args.mode == "incremental" else None
    report["generated_at"] = time.time()

    out_dir = settings.DATA_DIR / "llm_optimization"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / (
        f"session_state_{args.mode}_eval_{stamp}_{time.time_ns() % 1_000_000_000}.json"
    )
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report["summary"]
    print(json.dumps({"summary": summary, "report_path": str(out_path)}, ensure_ascii=False, indent=2))
    return 0 if summary["production_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
