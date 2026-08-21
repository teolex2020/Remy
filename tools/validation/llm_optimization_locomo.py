"""Evaluate Remy context optimization on the external LoCoMo benchmark.

LoCoMo is not shaped like our synthetic corpus: it asks episodic memory
questions over long multi-session dialogues. This runner keeps it separate so a
good synthetic score cannot hide an external benchmark failure.
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
    CorpusCase,
    CorpusMessage,
    DryMeter,
    GeminiMeter,
    SYSTEM_PROMPT,
    _parse_modes,
    build_prompt_for_mode,
)
from remy.core.session_state_wrapper import estimate_tokens  # noqa: E402


DEFAULT_LOCOMO = ROOT / "data" / "evals" / "external" / "locomo10.json"
LOCOMO_MODES = ("raw", "projected_facts", "projected_hybrid", "evidence_oracle")


@dataclass(frozen=True)
class LocomoItem:
    sample_id: str
    qa_index: int
    question: str
    answer: str
    category: int | str
    evidence: list[str]
    session_log: list[dict[str, str]]
    evidence_log: list[dict[str, str]]


def _session_sort_key(key: str) -> int:
    try:
        return int(key.split("_")[1])
    except Exception:
        return 0


def _conversation_log(conversation: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, str]]:
    session_keys = sorted(
        [
            key
            for key in conversation
            if key.startswith("session_") and not key.endswith("date_time")
        ],
        key=_session_sort_key,
    )
    log: list[dict[str, str]] = []
    by_dia_id: dict[str, str] = {}
    for session_key in session_keys:
        session_date = str(conversation.get(f"{session_key}_date_time") or "").strip()
        for turn in conversation.get(session_key) or []:
            if not isinstance(turn, dict):
                continue
            speaker = str(turn.get("speaker") or "speaker").strip()
            text = str(turn.get("text") or "").strip()
            dia_id = str(turn.get("dia_id") or "").strip()
            if not text:
                continue
            prefix = f"[{session_date}] " if session_date else ""
            line = f"{prefix}{speaker}: {text}"
            log.append({"type": "user_text", "text": line})
            if dia_id:
                by_dia_id[dia_id] = line
    return log, by_dia_id


def load_locomo_items(
    path: Path,
    *,
    sample_limit: int = 0,
    qa_limit: int = 0,
    skip_adversarial: bool = True,
) -> list[LocomoItem]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("LoCoMo file must contain a top-level list")

    items: list[LocomoItem] = []
    samples = data[:sample_limit] if sample_limit else data
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            continue
        sample_id = str(sample.get("sample_id") or f"sample_{sample_index}")
        session_log, by_dia_id = _conversation_log(sample.get("conversation") or {})
        qa_items = sample.get("qa") or []
        for qa_index, qa in enumerate(qa_items):
            if not isinstance(qa, dict):
                continue
            category = qa.get("category")
            if skip_adversarial and str(category) == "5":
                continue
            answer = qa.get("answer")
            question = str(qa.get("question") or "").strip()
            if answer is None or not question:
                continue
            evidence_ids = [str(item) for item in qa.get("evidence") or []]
            evidence_log = [
                {"type": "user_text", "text": by_dia_id[evidence_id]}
                for evidence_id in evidence_ids
                if evidence_id in by_dia_id
            ]
            items.append(
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
            if qa_limit and len(items) >= qa_limit:
                return items
    if not items:
        raise ValueError(f"No LoCoMo QA items loaded from {path}")
    return items


def _gold_fragments(answer: str) -> list[str]:
    if not answer:
        return []
    raw = str(answer).strip()
    if "," in raw:
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if len(parts) > 1:
            return parts
    return [raw]


def answer_matches(answer: str, gold: str) -> bool:
    answer_cf = (answer or "").casefold()
    gold_cf = str(gold or "").casefold()
    if not gold_cf:
        return False
    if gold_cf in answer_cf:
        return True
    fragments = _gold_fragments(gold)
    if len(fragments) > 1:
        return sum(1 for fragment in fragments if fragment.casefold() in answer_cf) >= max(
            1,
            int(len(fragments) * 0.6),
        )
    tokens = [
        token
        for token in gold_cf.replace(",", " ").replace("/", " ").split()
        if len(token) > 2
    ]
    if not tokens:
        return False
    return sum(1 for token in tokens if token in answer_cf) >= max(1, int(len(tokens) * 0.6))


def _case_for_item(item: LocomoItem, *, evidence_only: bool = False) -> CorpusCase:
    log = item.evidence_log if evidence_only else item.session_log
    messages = [CorpusMessage("user", turn["text"]) for turn in log]
    return CorpusCase(
        case_id=f"{item.sample_id}:{item.qa_index}",
        category=f"locomo_{item.category}",
        risk="external",
        messages=messages,
        question=item.question,
        expected_fragments=_gold_fragments(item.answer),
        notes="LoCoMo external benchmark",
    )


async def build_prompt_for_locomo_mode(
    item: LocomoItem,
    mode: str,
    answer_func: Any,
) -> tuple[str, dict[str, Any]]:
    if mode == "evidence_oracle":
        return await build_prompt_for_mode(
            _case_for_item(item, evidence_only=True),
            "raw",
            answer_func,
        )
    return await build_prompt_for_mode(_case_for_item(item), mode, answer_func)


async def run_locomo_eval(
    items: list[LocomoItem],
    *,
    modes: list[str],
    model: str,
    dry_run: bool,
    delay_sec: float,
) -> dict[str, Any]:
    answer_func = DryMeter() if dry_run else GeminiMeter(model, delay_sec=delay_sec)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for item in items:
        for mode in modes:
            before_prompt = getattr(answer_func, "prompt_tokens", 0)
            before_output = getattr(answer_func, "output_tokens", 0)
            before_calls = getattr(answer_func, "calls", 0)
            prompt, extra = await build_prompt_for_locomo_mode(item, mode, answer_func)
            if dry_run:
                answer = item.answer
            else:
                answer, _meta = answer_func(prompt)
            after_prompt = getattr(answer_func, "prompt_tokens", 0)
            after_output = getattr(answer_func, "output_tokens", 0)
            after_calls = getattr(answer_func, "calls", 0)
            rows.append(
                {
                    "sample_id": item.sample_id,
                    "qa_index": item.qa_index,
                    "case_id": f"{item.sample_id}:{item.qa_index}",
                    "category": item.category,
                    "mode": mode,
                    "question": item.question,
                    "gold": item.answer,
                    "evidence": item.evidence,
                    "correct": answer_matches(answer, item.answer),
                    "prompt_tokens_estimate": estimate_tokens(prompt),
                    "provider_prompt_tokens": after_prompt - before_prompt,
                    "provider_output_tokens": after_output - before_output,
                    "provider_total_tokens": (after_prompt - before_prompt)
                    + (after_output - before_output),
                    "provider_calls": after_calls - before_calls,
                    "answer_preview": answer[:500],
                    **extra,
                }
            )

    raw_by_case = {row["case_id"]: row for row in rows if row["mode"] == "raw"}
    for row in rows:
        raw_row = raw_by_case.get(row["case_id"])
        if not raw_row:
            continue
        raw_tokens = int(raw_row["prompt_tokens_estimate"])
        prompt_tokens = int(row["prompt_tokens_estimate"])
        raw_provider_total = int(raw_row.get("provider_total_tokens") or 0)
        provider_total = int(row.get("provider_total_tokens") or 0)
        row["prompt_tokens_delta_vs_raw_estimate"] = raw_tokens - prompt_tokens
        row["context_window_saved_pct_vs_raw_estimate"] = round(
            ((raw_tokens - prompt_tokens) / max(1, raw_tokens)) * 100,
            2,
        )
        if raw_provider_total and provider_total:
            row["provider_total_delta_vs_raw"] = raw_provider_total - provider_total
            row["provider_total_saved_pct_vs_raw"] = round(
                ((raw_provider_total - provider_total) / raw_provider_total) * 100,
                2,
            )

    by_mode: dict[str, dict[str, Any]] = {}
    raw_correct = sum(1 for row in rows if row["mode"] == "raw" and row["correct"])
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        correct = sum(1 for row in mode_rows if row["correct"])
        context_saved = sum(row.get("prompt_tokens_delta_vs_raw_estimate", 0) for row in mode_rows)
        raw_prompt_sum = sum(
            row.get("prompt_tokens_estimate", 0)
            for row in rows
            if row["mode"] == "raw" and row["case_id"] in {item["case_id"] for item in mode_rows}
        )
        by_mode[mode] = {
            "cases": len(mode_rows),
            "accuracy": correct / len(mode_rows) if mode_rows else 0.0,
            "correct": correct,
            "retained_vs_raw_correct": correct / raw_correct if raw_correct else 0.0,
            "prompt_tokens_estimate": sum(row["prompt_tokens_estimate"] for row in mode_rows),
            "context_window_saved_pct_vs_raw_estimate": round(
                (context_saved / max(1, raw_prompt_sum)) * 100,
                2,
            ),
            "provider_prompt_tokens": sum(row["provider_prompt_tokens"] for row in mode_rows),
            "provider_output_tokens": sum(row["provider_output_tokens"] for row in mode_rows),
            "provider_total_tokens": sum(row["provider_total_tokens"] for row in mode_rows),
            "provider_calls": sum(row["provider_calls"] for row in mode_rows),
        }
        raw_provider_total = by_mode.get("raw", {}).get("provider_total_tokens")
        provider_total = by_mode[mode]["provider_total_tokens"]
        if raw_provider_total and provider_total:
            by_mode[mode]["provider_total_saved_pct_vs_raw"] = round(
                ((raw_provider_total - provider_total) / raw_provider_total) * 100,
                2,
            )

    return {
        "schema": "remy_locomo_eval_v1",
        "summary": {
            "model": model,
            "dry_run": dry_run,
            "cases": len(items),
            "modes": modes,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "by_mode": by_mode,
        },
        "rows": rows,
    }


def _parse_locomo_modes(value: str) -> list[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [mode for mode in modes if mode not in LOCOMO_MODES]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown LoCoMo mode(s): {', '.join(unknown)}")
    if not modes:
        raise argparse.ArgumentTypeError("At least one mode is required")
    return modes


def _write_report(report: dict[str, Any]) -> Path:
    from remy.config.settings import settings

    out_dir = Path(settings.DATA_DIR) / "llm_optimization"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"locomo_eval_{stamp}_{time.time_ns()}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


async def _main_async(args: argparse.Namespace) -> int:
    items = load_locomo_items(
        args.locomo,
        sample_limit=args.sample_limit,
        qa_limit=args.qa_limit,
        skip_adversarial=not args.include_adversarial,
    )
    report = await run_locomo_eval(
        items,
        modes=args.mode,
        model=args.model,
        dry_run=args.dry_run,
        delay_sec=args.delay_sec,
    )
    path = _write_report(report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {path}")
    target = report["summary"]["by_mode"].get(args.target_mode, {})
    passed = float(target.get("accuracy") or 0.0) >= args.min_accuracy
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locomo", type=Path, default=DEFAULT_LOCOMO)
    parser.add_argument("--mode", type=_parse_locomo_modes, default=["raw", "projected_hybrid"])
    parser.add_argument("--target-mode", default="projected_hybrid")
    parser.add_argument("--model", default="gemini-flash-lite-latest")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay-sec", type=float, default=0.0)
    parser.add_argument("--sample-limit", type=int, default=1)
    parser.add_argument("--qa-limit", type=int, default=15)
    parser.add_argument("--include-adversarial", action="store_true")
    parser.add_argument("--min-accuracy", type=float, default=0.50)
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
