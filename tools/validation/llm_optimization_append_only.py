"""Evaluate append-only incremental state for LLM cost reduction.

The design under test:

1. Encode each new exchange exactly once.
2. Freeze the encoded chunk forever.
3. Answer later requests from the appended frozen chunks, optionally selecting
   only relevant chunks locally.

This is intentionally different from re-summarising the whole state. Old facts
are never sent through the encoder again, so compression loss cannot accumulate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
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
from tools.validation.llm_optimization_corpus_eval import GeminiMeter  # noqa: E402
from tools.validation.llm_optimization_locomo import DEFAULT_LOCOMO  # noqa: E402
from tools.validation.llm_optimization_locomo_notation import (  # noqa: E402
    build_judge_prompt,
    parse_yes_no_judgment,
)


APPEND_MODES = ("append_full", "append_retrieved")
ENCODERS = ("llm", "deterministic", "abbreviated")
AMORTIZATION_POINTS = (5, 10, 20, 50, 100, 200)


@dataclass(frozen=True)
class AppendChunk:
    chunk_id: str
    raw_text: str
    compressed_text: str
    source_tokens_estimate: int
    compressed_tokens_estimate: int
    provider_total_tokens: int


@dataclass(frozen=True)
class AppendQuestion:
    qa_index: int
    question: str
    answer: str
    category: int | str
    evidence: list[str]


@dataclass(frozen=True)
class AppendConversation:
    sample_id: str
    raw_context: str
    chunks: list[AppendChunk]
    questions: list[AppendQuestion]


def _safe_pct(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round((float(numerator) / float(denominator)) * 100, 2)


def _session_sort_key(key: str) -> int:
    try:
        return int(key.split("_")[1])
    except Exception:
        return 0


def _included_evidence_prefixes(session_limit: int) -> tuple[str, ...]:
    if session_limit <= 0:
        return ()
    return tuple(f"D{index}:" for index in range(1, session_limit + 1))


def _tokenize_for_selection(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[\w'-]+", text or "", flags=re.UNICODE)
        if len(token) > 2
    }


def select_relevant_chunks(
    chunks: list[AppendChunk],
    question: str,
    *,
    top_k: int,
    include_recent: int = 1,
) -> list[AppendChunk]:
    if top_k <= 0 or top_k >= len(chunks):
        return chunks
    query_tokens = _tokenize_for_selection(question)
    scored: list[tuple[int, int, AppendChunk]] = []
    for index, chunk in enumerate(chunks):
        chunk_tokens = _tokenize_for_selection(chunk.compressed_text)
        overlap = len(query_tokens & chunk_tokens)
        scored.append((overlap, index, chunk))
    selected_indexes = {
        index
        for _score, index, _chunk in sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)[
            :top_k
        ]
    }
    if include_recent > 0:
        selected_indexes.update(range(max(0, len(chunks) - include_recent), len(chunks)))
    return [chunk for index, chunk in enumerate(chunks) if index in selected_indexes]


def build_exchange_compress_prompt(exchange: str) -> str:
    return (
        "Encode this single new exchange into compact model-readable state.\n"
        "Keep every recoverable fact: names, dates, events, places, statuses, preferences, "
        "decisions, causes, outcomes, object details, and temporal anchors.\n"
        "Use short factual phrases. Do not include filler. Do not infer beyond the exchange.\n"
        "Output only compressed line(s).\n\n"
        f"EXCHANGE:\n{exchange}\n\n"
        "COMPRESSED:"
    )


def compact_exchange_dry(exchange: str) -> str:
    cleaned = " ".join((exchange or "").split())
    replacements = {
        "Speaker ": "S",
        "Caroline": "Caroline",
        "Melanie": "Melanie",
        "question": "q",
        "answer": "a",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned


def compact_exchange_deterministic(exchange: str) -> str:
    """Cheap tokenizer-aware turn codec v0.

    This intentionally avoids model calls. It preserves speaker/date/dia ids and
    removes only obvious conversational filler. It is conservative: if a phrase
    may carry facts, it stays.
    """
    lines: list[str] = []
    for raw_line in (exchange or "").splitlines():
        line = " ".join(raw_line.split())
        if not line:
            continue
        line = re.sub(r"^\[([^\]]+)\]\s+", r"@\1 ", line)
        line = re.sub(r"\bD(\d+):(\d+)\b", r"d\1.\2", line)
        line = line.replace(": ", ":")
        line = re.sub(r"\bI am\b", "I'm", line, flags=re.IGNORECASE)
        line = re.sub(r"\bdo not\b", "don't", line, flags=re.IGNORECASE)
        line = re.sub(r"\bgoing to\b", "gonna", line, flags=re.IGNORECASE)
        filler_patterns = (
            r"\bthat sounds (great|good|important|interesting)\.?\b",
            r"\bthanks? for sharing\.?\b",
            r"\bokay\.?\b",
            r"\bok\.?\b",
        )
        if any(re.fullmatch(pattern, line.casefold()) for pattern in filler_patterns):
            continue
        lines.append(line)
    return "\n".join(lines) or compact_exchange_dry(exchange)


_ABBREVIATIONS = {
    "researched": "rsch",
    "researching": "rsch",
    "research": "rsch",
    "adoption": "adpt",
    "agencies": "agnc",
    "agency": "agnc",
    "painted": "pnt",
    "painting": "pnt",
    "visited": "vst",
    "attended": "att",
    "joined": "jn",
    "support": "sup",
    "group": "grp",
    "conference": "conf",
    "workshop": "wkshp",
    "presentation": "prsn",
    "transition": "trans",
    "transgender": "transg",
    "counseling": "cnsl",
    "psychology": "psych",
    "certification": "cert",
    "planning": "pln",
    "planned": "pln",
    "family": "fam",
    "friends": "frnds",
    "mentor": "mntr",
    "mentors": "mntrs",
    "children": "kids",
    "business": "biz",
    "studio": "std",
    "dancing": "dnc",
    "dance": "dnc",
    "destress": "dstrs",
    "martial": "mart",
    "kickboxing": "kickbox",
    "taekwondo": "taekw",
    "screenplay": "scrnpl",
    "finished": "fin",
    "launched": "lnch",
    "campaign": "camp",
    "remember": "rem",
    "because": "bc",
    "before": "bef",
    "after": "aft",
    "yesterday": "yday",
    "tomorrow": "tmrw",
    "January": "Jan",
    "February": "Feb",
    "March": "Mar",
    "April": "Apr",
    "June": "Jun",
    "July": "Jul",
    "August": "Aug",
    "September": "Sep",
    "October": "Oct",
    "November": "Nov",
    "December": "Dec",
}

_STOPWORDS = {
    "a",
    "an",
    "the",
    "to",
    "of",
    "for",
    "with",
    "about",
    "that",
    "this",
    "these",
    "those",
    "very",
    "really",
    "just",
}


def _abbreviate_text(text: str) -> str:
    parts: list[str] = []
    for token in re.findall(r"@\S+|d\d+\.\d+|[A-Za-z][A-Za-z'-]*|\d+|[^\s]", text):
        if token in {"@", ":", ";", ",", "."}:
            continue
        lower = token.casefold()
        if lower in _STOPWORDS:
            continue
        replacement = _ABBREVIATIONS.get(token) or _ABBREVIATIONS.get(lower)
        if replacement:
            parts.append(replacement)
            continue
        if len(token) > 9 and token.isalpha():
            parts.append(token[:6])
        else:
            parts.append(token)
    return " ".join(parts)


def compact_exchange_abbreviated(exchange: str) -> str:
    """Abbreviative codec: preserve sense while shrinking token surface."""
    lines: list[str] = []
    for raw_line in (exchange or "").splitlines():
        line = " ".join(raw_line.split())
        if not line:
            continue
        line = re.sub(r"^\[([^\]]+)\]\s+", r"@\1 ", line)
        line = re.sub(r"\bD(\d+):(\d+)\b", r"d\1.\2", line)
        if re.fullmatch(r".*:\s*(ok|okay|thanks?|thank you|that sounds (great|good|important|interesting))\.?", line.casefold()):
            continue
        line = line.replace(": ", ":")
        lines.append(_abbreviate_text(line))
    return "\n".join(line for line in lines if line).strip() or compact_exchange_deterministic(exchange)


def build_answer_prompt(context: str, question: str, *, context_kind: str) -> str:
    return (
        "Answer using only the supplied context. Be concise. If the answer is absent, say UNKNOWN.\n\n"
        f"CONTEXT_KIND: {context_kind}\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n"
        "ANSWER:"
    )


def _iter_session_exchanges(conversation: dict[str, Any], *, session_limit: int) -> list[tuple[str, str]]:
    session_keys = sorted(
        [
            key
            for key in conversation
            if key.startswith("session_") and not key.endswith("date_time")
        ],
        key=_session_sort_key,
    )
    if session_limit:
        session_keys = [key for key in session_keys if _session_sort_key(key) <= session_limit]

    exchanges: list[tuple[str, str]] = []
    for session_key in session_keys:
        session_date = str(conversation.get(f"{session_key}_date_time") or "").strip()
        turns = [turn for turn in conversation.get(session_key) or [] if isinstance(turn, dict)]
        for index in range(0, len(turns), 2):
            lines: list[str] = []
            for turn in turns[index : index + 2]:
                speaker = str(turn.get("speaker") or "speaker").strip()
                text = str(turn.get("text") or "").strip()
                dia_id = str(turn.get("dia_id") or "").strip()
                if not text:
                    continue
                dia = f"{dia_id} " if dia_id else ""
                date = f"[{session_date}] " if session_date else ""
                lines.append(f"{date}{dia}{speaker}: {text}")
            if lines:
                exchanges.append((f"{session_key}:{index // 2}", "\n".join(lines)))
    return exchanges


def _qa_included(qa: dict[str, Any], *, session_limit: int, skip_adversarial: bool) -> bool:
    if skip_adversarial and str(qa.get("category")) == "5":
        return False
    if qa.get("answer") is None or not str(qa.get("question") or "").strip():
        return False
    prefixes = _included_evidence_prefixes(session_limit)
    if not prefixes:
        return True
    evidence = [str(item) for item in qa.get("evidence") or []]
    return bool(evidence) and any(any(item.startswith(prefix) for prefix in prefixes) for item in evidence)


def _limit_questions(
    sample: dict[str, Any],
    *,
    session_limit: int,
    qa_limit_per_conversation: int,
    skip_adversarial: bool,
) -> list[AppendQuestion]:
    questions: list[AppendQuestion] = []
    for qa_index, qa in enumerate(sample.get("qa") or []):
        if not isinstance(qa, dict):
            continue
        if not _qa_included(qa, session_limit=session_limit, skip_adversarial=skip_adversarial):
            continue
        questions.append(
            AppendQuestion(
                qa_index=qa_index,
                question=str(qa.get("question") or "").strip(),
                answer=str(qa.get("answer")),
                category=qa.get("category") if qa.get("category") is not None else "",
                evidence=[str(item) for item in qa.get("evidence") or []],
            )
        )
        if qa_limit_per_conversation and len(questions) >= qa_limit_per_conversation:
            break
    return questions


def _make_dry_meter():
    class Meter:
        calls = 0
        prompt_tokens = 0
        output_tokens = 0

        def __call__(self, prompt: str):
            self.calls += 1
            self.prompt_tokens += estimate_tokens(prompt)
            output = compact_exchange_dry(prompt)
            self.output_tokens += estimate_tokens(output)
            return output, {"usage_metadata": {}}

    return Meter()


async def load_append_conversations(
    path: Path,
    *,
    conversation_limit: int,
    session_limit: int,
    qa_limit_per_conversation: int,
    skip_adversarial: bool,
    model: str,
    dry_run: bool,
    delay_sec: float,
    encoder_kind: str = "llm",
) -> tuple[list[AppendConversation], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("LoCoMo file must contain a top-level list")
    if encoder_kind not in ENCODERS:
        raise ValueError(f"Unknown encoder kind: {encoder_kind}")
    # Deterministic/abbreviated encoders never call a provider. Constructing a
    # Gemini meter here would still require an API key despite making zero
    # remote calls, which breaks offline evaluation and CI.
    uses_provider = encoder_kind == "llm" and not dry_run
    encoder = GeminiMeter(model, delay_sec=delay_sec) if uses_provider else _make_dry_meter()
    conversations: list[AppendConversation] = []
    samples = data[:conversation_limit] if conversation_limit else data

    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            continue
        sample_id = str(sample.get("sample_id") or f"sample_{sample_index}")
        raw_exchanges = _iter_session_exchanges(sample.get("conversation") or {}, session_limit=session_limit)
        if not raw_exchanges:
            continue
        questions = _limit_questions(
            sample,
            session_limit=session_limit,
            qa_limit_per_conversation=qa_limit_per_conversation,
            skip_adversarial=skip_adversarial,
        )
        if not questions:
            continue

        chunks: list[AppendChunk] = []
        for chunk_id, exchange in raw_exchanges:
            before_prompt = encoder.prompt_tokens
            before_output = encoder.output_tokens
            if encoder_kind == "deterministic":
                compressed = compact_exchange_deterministic(exchange)
                provider_total = 0
            elif encoder_kind == "abbreviated":
                compressed = compact_exchange_abbreviated(exchange)
                provider_total = 0
            elif dry_run:
                compressed = compact_exchange_dry(exchange)
                encoder.calls += 1
                encoder.prompt_tokens += estimate_tokens(exchange)
                encoder.output_tokens += estimate_tokens(compressed)
                provider_total = (encoder.prompt_tokens - before_prompt) + (
                    encoder.output_tokens - before_output
                )
            else:
                compressed, _meta = encoder(build_exchange_compress_prompt(exchange))
                provider_total = (encoder.prompt_tokens - before_prompt) + (
                    encoder.output_tokens - before_output
                )
            chunks.append(
                AppendChunk(
                    chunk_id=chunk_id,
                    raw_text=exchange,
                    compressed_text=compressed.strip() or compact_exchange_dry(exchange),
                    source_tokens_estimate=estimate_tokens(exchange),
                    compressed_tokens_estimate=estimate_tokens(compressed),
                    provider_total_tokens=provider_total,
                )
            )

        conversations.append(
            AppendConversation(
                sample_id=sample_id,
                raw_context="\n".join(exchange for _chunk_id, exchange in raw_exchanges),
                chunks=chunks,
                questions=questions,
            )
        )

    if not conversations:
        raise ValueError(f"No append-only conversations loaded from {path}")
    encoder_usage: dict[str, Any] = {
        "kind": encoder_kind,
        "provider_prompt_tokens": int(encoder.prompt_tokens),
        "provider_output_tokens": int(encoder.output_tokens),
        "provider_total_tokens": int(encoder.prompt_tokens + encoder.output_tokens),
        "provider_calls": int(encoder.calls),
    }
    return conversations, encoder_usage


def _chunks_context(chunks: list[AppendChunk]) -> str:
    return "\n".join(f"{chunk.chunk_id}: {chunk.compressed_text}" for chunk in chunks)


def _economic_curve(
    *,
    raw_total: int,
    optimized_query_total: int,
    setup_total: int,
    case_count: int,
) -> dict[str, float | None]:
    if case_count <= 0:
        return {}
    raw_per_query = raw_total / case_count
    optimized_per_query = optimized_query_total / case_count
    saving_per_query = raw_per_query - optimized_per_query
    break_even = None if saving_per_query <= 0 else round(setup_total / saving_per_query, 2)
    curve: dict[str, float | None] = {"break_even_queries": break_even}
    for queries in AMORTIZATION_POINTS:
        raw_cost = raw_per_query * queries
        optimized_cost = setup_total + (optimized_per_query * queries)
        curve[f"net_saved_pct_at_{queries}_queries"] = _safe_pct(raw_cost - optimized_cost, raw_cost)
    return curve


async def run_append_only_eval(
    conversations: list[AppendConversation],
    *,
    encoder_usage: dict[str, Any],
    modes: list[str],
    model: str,
    judge: str,
    dry_run: bool,
    delay_sec: float,
    retrieval_top_k: int,
) -> dict[str, Any]:
    answer_meter = _make_dry_meter() if dry_run else GeminiMeter(model, delay_sec=delay_sec)
    judge_meter = _make_dry_meter() if dry_run else GeminiMeter(model, delay_sec=delay_sec)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for conversation in conversations:
        for question in conversation.questions:
            before_prompt = answer_meter.prompt_tokens
            before_output = answer_meter.output_tokens
            if dry_run:
                raw_answer = question.answer
                answer_meter.calls += 1
                answer_meter.prompt_tokens += estimate_tokens(conversation.raw_context)
                answer_meter.output_tokens += estimate_tokens(raw_answer)
            else:
                raw_answer, _meta = answer_meter(
                    build_answer_prompt(conversation.raw_context, question.question, context_kind="raw")
                )
            raw_total = (answer_meter.prompt_tokens - before_prompt) + (
                answer_meter.output_tokens - before_output
            )

            mode_answers: dict[str, dict[str, Any]] = {}
            for mode in modes:
                chunks = conversation.chunks
                if mode == "append_retrieved":
                    chunks = select_relevant_chunks(
                        conversation.chunks,
                        question.question,
                        top_k=retrieval_top_k,
                    )
                context = _chunks_context(chunks)
                before_prompt = answer_meter.prompt_tokens
                before_output = answer_meter.output_tokens
                if dry_run:
                    answer = question.answer
                    answer_meter.calls += 1
                    answer_meter.prompt_tokens += estimate_tokens(context)
                    answer_meter.output_tokens += estimate_tokens(answer)
                else:
                    answer, _meta = answer_meter(
                        build_answer_prompt(context, question.question, context_kind=mode)
                    )
                mode_total = (answer_meter.prompt_tokens - before_prompt) + (
                    answer_meter.output_tokens - before_output
                )
                mode_answers[mode] = {
                    "answer": answer,
                    "provider_total_tokens": mode_total,
                    "selected_chunks": len(chunks),
                    "context_tokens_estimate": estimate_tokens(context),
                }

            judgments: dict[str, bool] = {}
            if judge == "exact":
                from tools.validation.llm_optimization_locomo import answer_matches

                judgments["raw"] = answer_matches(raw_answer, question.answer)
                for mode, payload in mode_answers.items():
                    judgments[mode] = answer_matches(str(payload["answer"]), question.answer)
            else:
                all_answers = {"raw": raw_answer, **{mode: str(data["answer"]) for mode, data in mode_answers.items()}}
                for key, answer in all_answers.items():
                    if dry_run:
                        judgments[key] = True
                        continue
                    judgment, _meta = judge_meter(build_judge_prompt(question.question, question.answer, answer))
                    judgments[key] = parse_yes_no_judgment(judgment)

            base = {
                "sample_id": conversation.sample_id,
                "qa_index": question.qa_index,
                "case_id": f"{conversation.sample_id}:{question.qa_index}",
                "question": question.question,
                "gold": question.answer,
                "raw_correct": judgments["raw"],
                "raw_provider_total_tokens": raw_total,
                "raw_answer_preview": raw_answer[:500],
            }
            for mode, payload in mode_answers.items():
                rows.append(
                    {
                        **base,
                        "mode": mode,
                        "optimized_correct": judgments[mode],
                        "optimized_provider_total_tokens": payload["provider_total_tokens"],
                        "optimized_answer_preview": str(payload["answer"])[:500],
                        "selected_chunks": payload["selected_chunks"],
                        "optimized_context_tokens_estimate": payload["context_tokens_estimate"],
                        "query_saved_pct_vs_raw": _safe_pct(
                            raw_total - int(payload["provider_total_tokens"]),
                            raw_total,
                        ),
                    }
                )

    by_mode: dict[str, dict[str, Any]] = {}
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        raw_correct = sum(1 for row in mode_rows if row["raw_correct"])
        optimized_correct = sum(1 for row in mode_rows if row["optimized_correct"])
        raw_total = sum(int(row["raw_provider_total_tokens"]) for row in mode_rows)
        optimized_total = sum(int(row["optimized_provider_total_tokens"]) for row in mode_rows)
        setup_total = int(encoder_usage["provider_total_tokens"])
        by_mode[mode] = {
            "cases": len(mode_rows),
            "raw_accuracy": raw_correct / len(mode_rows) if mode_rows else 0.0,
            "optimized_accuracy": optimized_correct / len(mode_rows) if mode_rows else 0.0,
            "raw_correct": raw_correct,
            "optimized_correct": optimized_correct,
            "retained_vs_raw_correct": optimized_correct / raw_correct if raw_correct else 0.0,
            "raw_provider_total_tokens": raw_total,
            "optimized_query_provider_total_tokens": optimized_total,
            "setup_provider_total_tokens": setup_total,
            "optimized_product_provider_total_tokens": optimized_total + setup_total,
            "query_saved_pct_vs_raw": _safe_pct(raw_total - optimized_total, raw_total),
            "product_saved_pct_vs_raw": _safe_pct(raw_total - optimized_total - setup_total, raw_total),
            "selected_chunks_mean": round(
                sum(int(row["selected_chunks"]) for row in mode_rows) / max(1, len(mode_rows)),
                2,
            ),
            **_economic_curve(
                raw_total=raw_total,
                optimized_query_total=optimized_total,
                setup_total=setup_total,
                case_count=len(mode_rows),
            ),
        }

    return {
        "schema": "remy_append_only_eval_v1",
        "summary": {
            "model": model,
            "judge": judge,
            "dry_run": dry_run,
            "conversations": len(conversations),
            "modes": modes,
            "encoder": encoder_usage,
            "retrieval_top_k": retrieval_top_k,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "by_mode": by_mode,
        },
        "rows": rows,
        "conversations": [
            {
                "sample_id": conversation.sample_id,
                "chunks": len(conversation.chunks),
                "questions": len(conversation.questions),
                "raw_context_tokens_estimate": estimate_tokens(conversation.raw_context),
                "append_state_tokens_estimate": estimate_tokens(_chunks_context(conversation.chunks)),
            }
            for conversation in conversations
        ],
    }


def _parse_modes(value: str) -> list[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [mode for mode in modes if mode not in APPEND_MODES]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown append-only mode(s): {', '.join(unknown)}")
    if not modes:
        raise argparse.ArgumentTypeError("At least one mode is required")
    return modes


def _write_report(report: dict[str, Any]) -> Path:
    from remy.config.settings import settings

    out_dir = Path(settings.DATA_DIR) / "llm_optimization"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"append_only_eval_{stamp}_{time.time_ns()}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


async def _main_async(args: argparse.Namespace) -> int:
    conversations, encoder_usage = await load_append_conversations(
        args.locomo,
        conversation_limit=args.conversation_limit,
        session_limit=args.session_limit,
        qa_limit_per_conversation=args.qa_limit_per_conversation,
        skip_adversarial=not args.include_adversarial,
        model=args.model,
        dry_run=args.dry_run,
        delay_sec=args.delay_sec,
        encoder_kind=args.encoder,
    )
    report = await run_append_only_eval(
        conversations,
        encoder_usage=encoder_usage,
        modes=args.mode,
        model=args.model,
        judge=args.judge,
        dry_run=args.dry_run,
        delay_sec=args.delay_sec,
        retrieval_top_k=args.retrieval_top_k,
    )
    path = _write_report(report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {path}")

    target = report["summary"]["by_mode"].get(args.target_mode, {})
    accuracy_drop = float(target.get("raw_accuracy") or 0.0) - float(target.get("optimized_accuracy") or 0.0)
    passed = (
        float(target.get("optimized_accuracy") or 0.0) >= args.min_accuracy
        and accuracy_drop <= args.max_accuracy_drop
        and float(target.get("product_saved_pct_vs_raw") or 0.0) >= args.min_product_saving_pct
    )
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locomo", type=Path, default=DEFAULT_LOCOMO)
    parser.add_argument("--model", default="gemini-flash-lite-latest")
    parser.add_argument("--mode", type=_parse_modes, default=["append_full"])
    parser.add_argument("--target-mode", default="append_full")
    parser.add_argument("--judge", choices=["exact", "llm"], default="llm")
    parser.add_argument("--encoder", choices=ENCODERS, default="llm")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay-sec", type=float, default=0.0)
    parser.add_argument("--conversation-limit", type=int, default=2)
    parser.add_argument("--session-limit", type=int, default=5)
    parser.add_argument("--qa-limit-per-conversation", type=int, default=10)
    parser.add_argument("--retrieval-top-k", type=int, default=8)
    parser.add_argument("--include-adversarial", action="store_true")
    parser.add_argument("--min-accuracy", type=float, default=0.60)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.02)
    parser.add_argument("--min-product-saving-pct", type=float, default=30.0)
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
