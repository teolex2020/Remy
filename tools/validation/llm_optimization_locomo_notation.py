"""Evaluate recode-to-notation on LoCoMo with semantic judging.

This runner tests a different optimization shape from prompt cleaning or
retrieval: recode a whole conversation once into dense notation, then answer
many questions from that notation. Product cost excludes the judge calls used
only for evaluation.
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

from remy.core.session_state_wrapper import estimate_tokens  # noqa: E402
from tools.validation.llm_optimization_corpus_eval import GeminiMeter, DryMeter  # noqa: E402
from tools.validation.llm_optimization_locomo import (  # noqa: E402
    DEFAULT_LOCOMO,
    LocomoItem,
    _conversation_log,
    answer_matches,
)


@dataclass(frozen=True)
class LocomoConversation:
    sample_id: str
    session_log: list[dict[str, str]]
    qa_items: list[LocomoItem]


def _safe_pct(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round((float(numerator) / float(denominator)) * 100, 2)


def build_raw_context(session_log: list[dict[str, str]]) -> str:
    return "\n".join(str(item.get("text") or "").strip() for item in session_log if item.get("text"))


def build_notation_prompt(raw_context: str) -> str:
    return (
        "Recode the conversation into dense machine-readable memory notation.\n"
        "Goal: produce a lossless event ledger that is cheaper than the transcript.\n"
        "Do NOT summarize. Do NOT merge different events. Do NOT replace the source verb "
        "with a broader interpretation. If a line says researched, keep researched; if it "
        "says applied, keep applied. If a detail might answer a future question, keep it.\n"
        "Keep people, relationships, identity, locations, exact dates, relative date clues, "
        "events, plans, preferences, causes, outcomes, objects, quantities, and constraints.\n"
        "For every factual dialogue line, write at least one compact record. If unsure, keep "
        "the fact rather than dropping it.\n"
        "Use compact one-line records only. Prefer these forms:\n"
        "Person| attr=value; relation=value\n"
        "Person.event| action=object @date_or_time_anchor; detail=value\n"
        "Group.event| action=object @date_or_time_anchor; detail=value\n"
        "Person.plan| action=object @date_or_time_anchor; reason=value\n"
        "Person.pref| item=value; reason=value\n"
        "Use underscores for multiword values. Preserve exact dates, relative date clues, "
        "and original event verbs.\n"
        "Return notation only, no prose.\n\n"
        f"CONVERSATION:\n{raw_context}"
    )


def build_answer_prompt(context: str, question: str, *, context_kind: str) -> str:
    return (
        "Answer using only the supplied context. Be concise. If the answer is not present, say UNKNOWN.\n\n"
        f"CONTEXT_KIND: {context_kind}\n"
        "If CONTEXT_KIND is notation, treat it as a compact lossless event ledger. "
        "Underscores are spaces. Event labels and attribute keys are meaningful. "
        "Answer from any direct or clearly equivalent notation record before saying UNKNOWN.\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n"
        "ANSWER:"
    )


def build_judge_prompt(question: str, gold: str, answer: str) -> str:
    return (
        "Judge whether the candidate answer is semantically correct for the question.\n"
        "Accept paraphrases, partial wording differences, and date-equivalent answers "
        "(for example, a concrete date may match a relative date clue if they refer to the same day).\n"
        "Reject answers that miss the requested fact, name a different entity, or invent unsupported details.\n"
        "Reply with exactly YES or NO.\n\n"
        f"QUESTION: {question}\n"
        f"REFERENCE ANSWER: {gold}\n"
        f"CANDIDATE ANSWER: {answer}\n"
        "JUDGMENT:"
    )


def parse_yes_no_judgment(text: str) -> bool:
    first = (text or "").strip().split(maxsplit=1)[0].upper() if (text or "").strip() else ""
    return first.startswith("YES")


def answer_is_unknown(text: str) -> bool:
    normalized = " ".join((text or "").upper().split())
    return normalized in {"UNKNOWN", "NOT PRESENT", "NOT FOUND"} or normalized.startswith("UNKNOWN")


def load_locomo_conversations(
    path: Path,
    *,
    conversation_limit: int = 2,
    session_limit: int = 0,
    qa_limit_per_conversation: int = 10,
    skip_adversarial: bool = True,
) -> list[LocomoConversation]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("LoCoMo file must contain a top-level list")

    conversations: list[LocomoConversation] = []
    for sample_index, sample in enumerate(data[:conversation_limit] if conversation_limit else data):
        if not isinstance(sample, dict):
            continue
        sample_id = str(sample.get("sample_id") or f"sample_{sample_index}")
        conversation = dict(sample.get("conversation") or {})
        if session_limit:
            keep: dict[str, Any] = {}
            for key, value in conversation.items():
                if not key.startswith("session_"):
                    keep[key] = value
                    continue
                try:
                    number = int(key.split("_")[1])
                except Exception:
                    keep[key] = value
                    continue
                if number <= session_limit:
                    keep[key] = value
            conversation = keep
        session_log, by_dia_id = _conversation_log(conversation)

        qa_items: list[LocomoItem] = []
        for qa_index, qa in enumerate(sample.get("qa") or []):
            if not isinstance(qa, dict):
                continue
            category = qa.get("category")
            if skip_adversarial and str(category) == "5":
                continue
            question = str(qa.get("question") or "").strip()
            answer = qa.get("answer")
            if not question or answer is None:
                continue
            evidence_ids = [str(item) for item in qa.get("evidence") or []]
            evidence_log = [
                {"type": "user_text", "text": by_dia_id[evidence_id]}
                for evidence_id in evidence_ids
                if evidence_id in by_dia_id
            ]
            qa_items.append(
                LocomoItem(
                    sample_id=sample_id,
                    qa_index=qa_index,
                    question=question,
                    answer=str(answer),
                    category=category if category is not None else "",
                    evidence=evidence_ids,
                    session_log=session_log,
                    evidence_log=evidence_log,
                )
            )
            if qa_limit_per_conversation and len(qa_items) >= qa_limit_per_conversation:
                break
        if qa_items:
            conversations.append(
                LocomoConversation(
                    sample_id=sample_id,
                    session_log=session_log,
                    qa_items=qa_items,
                )
            )
    if not conversations:
        raise ValueError(f"No LoCoMo conversations loaded from {path}")
    return conversations


def amortized_saving_pct(
    *,
    raw_total_tokens: int,
    notation_answer_tokens: int,
    notation_setup_tokens: int,
    repeat_multiplier: int,
) -> float:
    if repeat_multiplier <= 0:
        raise ValueError("repeat_multiplier must be positive")
    raw = raw_total_tokens * repeat_multiplier
    optimized = notation_setup_tokens + (notation_answer_tokens * repeat_multiplier)
    return _safe_pct(raw - optimized, raw)


async def run_locomo_notation_eval(
    conversations: list[LocomoConversation],
    *,
    model: str,
    judge: str,
    dry_run: bool,
    delay_sec: float,
) -> dict[str, Any]:
    product_meter = DryMeter() if dry_run else GeminiMeter(model, delay_sec=delay_sec)
    judge_meter = DryMeter() if dry_run else GeminiMeter(model, delay_sec=delay_sec)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    conversation_rows: list[dict[str, Any]] = []

    for conversation in conversations:
        raw_context = build_raw_context(conversation.session_log)
        before_prompt = product_meter.prompt_tokens
        before_output = product_meter.output_tokens
        before_calls = product_meter.calls
        if dry_run:
            notation = raw_context
        else:
            notation, _meta = product_meter(build_notation_prompt(raw_context))
        encode_prompt = product_meter.prompt_tokens - before_prompt
        encode_output = product_meter.output_tokens - before_output
        encode_calls = product_meter.calls - before_calls
        conversation_rows.append(
            {
                "sample_id": conversation.sample_id,
                "qa_count": len(conversation.qa_items),
                "raw_context_tokens_estimate": estimate_tokens(raw_context),
                "notation_tokens_estimate": estimate_tokens(notation),
                "estimated_context_saved_pct": _safe_pct(
                    estimate_tokens(raw_context) - estimate_tokens(notation),
                    estimate_tokens(raw_context),
                ),
                "encode_provider_prompt_tokens": encode_prompt,
                "encode_provider_output_tokens": encode_output,
                "encode_provider_total_tokens": encode_prompt + encode_output,
                "encode_provider_calls": encode_calls,
                "notation": notation,
                "notation_preview": notation[:1000],
            }
        )

        for item in conversation.qa_items:
            raw_prompt = build_answer_prompt(raw_context, item.question, context_kind="raw")
            notation_prompt = build_answer_prompt(notation, item.question, context_kind="notation")

            before_prompt = product_meter.prompt_tokens
            before_output = product_meter.output_tokens
            before_calls = product_meter.calls
            if dry_run:
                raw_answer = item.answer
            else:
                raw_answer, _meta = product_meter(raw_prompt)
            raw_prompt_tokens = product_meter.prompt_tokens - before_prompt
            raw_output_tokens = product_meter.output_tokens - before_output
            raw_calls = product_meter.calls - before_calls

            before_prompt = product_meter.prompt_tokens
            before_output = product_meter.output_tokens
            before_calls = product_meter.calls
            if dry_run:
                notation_answer = item.answer
            else:
                notation_answer, _meta = product_meter(notation_prompt)
            notation_prompt_tokens = product_meter.prompt_tokens - before_prompt
            notation_output_tokens = product_meter.output_tokens - before_output
            notation_calls = product_meter.calls - before_calls

            if judge == "exact":
                raw_correct = answer_matches(raw_answer, item.answer)
                notation_correct = answer_matches(notation_answer, item.answer)
                judge_prompt_tokens = 0
                judge_output_tokens = 0
                judge_calls = 0
            else:
                before_prompt = judge_meter.prompt_tokens
                before_output = judge_meter.output_tokens
                before_calls = judge_meter.calls
                if dry_run:
                    raw_correct = True
                    notation_correct = True
                else:
                    raw_judgment, _meta = judge_meter(
                        build_judge_prompt(item.question, item.answer, raw_answer)
                    )
                    notation_judgment, _meta = judge_meter(
                        build_judge_prompt(item.question, item.answer, notation_answer)
                    )
                    raw_correct = parse_yes_no_judgment(raw_judgment)
                    notation_correct = parse_yes_no_judgment(notation_judgment)
                judge_prompt_tokens = judge_meter.prompt_tokens - before_prompt
                judge_output_tokens = judge_meter.output_tokens - before_output
                judge_calls = judge_meter.calls - before_calls

            rows.append(
                {
                    "sample_id": conversation.sample_id,
                    "qa_index": item.qa_index,
                    "case_id": f"{conversation.sample_id}:{item.qa_index}",
                    "category": item.category,
                    "question": item.question,
                    "gold": item.answer,
                    "raw_correct": raw_correct,
                    "notation_correct": notation_correct,
                    "notation_answer_unknown": answer_is_unknown(notation_answer),
                    "raw_answer_preview": raw_answer[:500],
                    "notation_answer_preview": notation_answer[:500],
                    "raw_prompt_tokens_estimate": estimate_tokens(raw_prompt),
                    "notation_prompt_tokens_estimate": estimate_tokens(notation_prompt),
                    "estimated_context_saved_pct": _safe_pct(
                        estimate_tokens(raw_prompt) - estimate_tokens(notation_prompt),
                        estimate_tokens(raw_prompt),
                    ),
                    "raw_provider_prompt_tokens": raw_prompt_tokens,
                    "raw_provider_output_tokens": raw_output_tokens,
                    "raw_provider_total_tokens": raw_prompt_tokens + raw_output_tokens,
                    "raw_provider_calls": raw_calls,
                    "notation_provider_prompt_tokens": notation_prompt_tokens,
                    "notation_provider_output_tokens": notation_output_tokens,
                    "notation_provider_total_tokens": notation_prompt_tokens + notation_output_tokens,
                    "notation_provider_calls": notation_calls,
                    "notation_query_saved_pct_vs_raw": _safe_pct(
                        (raw_prompt_tokens + raw_output_tokens)
                        - (notation_prompt_tokens + notation_output_tokens),
                        raw_prompt_tokens + raw_output_tokens,
                    ),
                    "judge_provider_prompt_tokens": judge_prompt_tokens,
                    "judge_provider_output_tokens": judge_output_tokens,
                    "judge_provider_total_tokens": judge_prompt_tokens + judge_output_tokens,
                    "judge_provider_calls": judge_calls,
                }
            )

    raw_correct = sum(1 for row in rows if row["raw_correct"])
    notation_correct = sum(1 for row in rows if row["notation_correct"])
    raw_total = sum(int(row["raw_provider_total_tokens"]) for row in rows)
    notation_answer_total = sum(int(row["notation_provider_total_tokens"]) for row in rows)
    notation_setup_total = sum(int(row["encode_provider_total_tokens"]) for row in conversation_rows)
    notation_product_total = notation_setup_total + notation_answer_total
    unknown_rows = [row for row in rows if row["notation_answer_unknown"]]
    fallback_raw_total = sum(int(row["raw_provider_total_tokens"]) for row in unknown_rows)
    fallback_product_total = notation_product_total + fallback_raw_total
    fallback_correct = sum(
        1
        for row in rows
        if row["notation_correct"] or (row["notation_answer_unknown"] and row["raw_correct"])
    )

    summary = {
        "model": model,
        "judge": judge,
        "dry_run": dry_run,
        "conversations": len(conversations),
        "cases": len(rows),
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "raw_accuracy": raw_correct / len(rows) if rows else 0.0,
        "notation_accuracy": notation_correct / len(rows) if rows else 0.0,
        "raw_correct": raw_correct,
        "notation_correct": notation_correct,
        "retained_vs_raw_correct": notation_correct / raw_correct if raw_correct else 0.0,
        "raw_provider_total_tokens": raw_total,
        "notation_answer_provider_total_tokens": notation_answer_total,
        "notation_setup_provider_total_tokens": notation_setup_total,
        "notation_product_provider_total_tokens": notation_product_total,
        "notation_query_saved_pct_vs_raw_mean": round(
            sum(float(row["notation_query_saved_pct_vs_raw"]) for row in rows) / max(1, len(rows)),
            2,
        ),
        "notation_product_saved_pct_vs_raw": _safe_pct(raw_total - notation_product_total, raw_total),
        "fallback_on_unknown_cases": len(unknown_rows),
        "fallback_on_unknown_accuracy": fallback_correct / len(rows) if rows else 0.0,
        "fallback_on_unknown_correct": fallback_correct,
        "fallback_on_unknown_raw_provider_total_tokens": fallback_raw_total,
        "fallback_on_unknown_product_provider_total_tokens": fallback_product_total,
        "fallback_on_unknown_product_saved_pct_vs_raw": _safe_pct(
            raw_total - fallback_product_total,
            raw_total,
        ),
        "estimated_context_saved_pct_mean": round(
            sum(float(row["estimated_context_saved_pct"]) for row in rows) / max(1, len(rows)),
            2,
        ),
        "amortized_product_saved_pct_vs_raw": {
            str(multiplier): amortized_saving_pct(
                raw_total_tokens=raw_total,
                notation_answer_tokens=notation_answer_total,
                notation_setup_tokens=notation_setup_total,
                repeat_multiplier=multiplier,
            )
            for multiplier in (1, 2, 3, 5, 10)
        },
        "judge_provider_total_tokens": sum(int(row["judge_provider_total_tokens"]) for row in rows),
        "judge_provider_calls": sum(int(row["judge_provider_calls"]) for row in rows),
    }

    return {
        "schema": "remy_locomo_notation_eval_v1",
        "summary": summary,
        "conversations": conversation_rows,
        "rows": rows,
    }


def _write_report(report: dict[str, Any]) -> Path:
    from remy.config.settings import settings

    out_dir = Path(settings.DATA_DIR) / "llm_optimization"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"locomo_notation_eval_{stamp}_{time.time_ns()}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


async def _main_async(args: argparse.Namespace) -> int:
    conversations = load_locomo_conversations(
        args.locomo,
        conversation_limit=args.conversation_limit,
        session_limit=args.session_limit,
        qa_limit_per_conversation=args.qa_limit_per_conversation,
        skip_adversarial=not args.include_adversarial,
    )
    report = await run_locomo_notation_eval(
        conversations,
        model=args.model,
        judge=args.judge,
        dry_run=args.dry_run,
        delay_sec=args.delay_sec,
    )
    path = _write_report(report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {path}")

    summary = report["summary"]
    accuracy_drop = float(summary["raw_accuracy"]) - float(summary["notation_accuracy"])
    passed = (
        float(summary["notation_accuracy"]) >= args.min_accuracy
        and accuracy_drop <= args.max_accuracy_drop
        and float(summary["notation_product_saved_pct_vs_raw"]) >= args.min_product_saving_pct
    )
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locomo", type=Path, default=DEFAULT_LOCOMO)
    parser.add_argument("--model", default="gemini-flash-lite-latest")
    parser.add_argument("--judge", choices=["exact", "llm"], default="llm")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay-sec", type=float, default=0.0)
    parser.add_argument("--conversation-limit", type=int, default=2)
    parser.add_argument("--session-limit", type=int, default=0)
    parser.add_argument("--qa-limit-per-conversation", type=int, default=10)
    parser.add_argument("--include-adversarial", action="store_true")
    parser.add_argument("--min-accuracy", type=float, default=0.65)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.08)
    parser.add_argument("--min-product-saving-pct", type=float, default=40.0)
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
