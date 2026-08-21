"""Run repeatable corpus evals for Remy's LLM optimization experiments."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_CASES = ROOT / "data" / "evals" / "llm_optimization" / "cases.jsonl"
SYSTEM_PROMPT = (
    "You are Remy. Answer the current request using only the provided context. "
    "If an exact value is present, output it exactly. Do not invent missing values. "
    "When the saved fact, rule or preference is written in another language or in "
    "transliteration, quote the original phrase verbatim (a translation may be "
    "added in parentheses), never replace it with only a translation. "
    "When answering about saved rules or preferences, include all saved "
    "constraints and style modifiers; do not omit words like terse, brief, "
    "concise, strict, first, urgent, or high-risk when they are present."
)
CANONICAL_MODES = ("raw", "projected_facts", "projected_hybrid", "incremental")
COMPATIBILITY_MODES = ("projected",)
MODES = CANONICAL_MODES + COMPATIBILITY_MODES


@dataclass(frozen=True)
class CorpusMessage:
    role: str
    content: str


@dataclass(frozen=True)
class CorpusCase:
    case_id: str
    category: str
    risk: str
    messages: list[CorpusMessage]
    question: str
    expected_fragments: list[str]
    expected_any: list[list[str]] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    notes: str = ""


def _require_text(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"case {data.get('id') or '<unknown>'}: {key} must be a non-empty string")
    return value.strip()


def _require_text_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"case {data.get('id') or '<unknown>'}: {key} must be a non-empty list")
    result = [str(item).strip() for item in value if str(item).strip()]
    if not result:
        raise ValueError(f"case {data.get('id') or '<unknown>'}: {key} must contain text")
    return result


def parse_case(data: dict[str, Any]) -> CorpusCase:
    messages_raw = data.get("messages")
    if not isinstance(messages_raw, list) or not messages_raw:
        raise ValueError(f"case {data.get('id') or '<unknown>'}: messages must be a non-empty list")

    messages: list[CorpusMessage] = []
    for index, message in enumerate(messages_raw):
        if not isinstance(message, dict):
            raise ValueError(f"case {data.get('id')}: message {index} must be an object")
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if role not in {"user", "assistant"}:
            raise ValueError(f"case {data.get('id')}: message {index} has unsupported role {role!r}")
        if not content:
            raise ValueError(f"case {data.get('id')}: message {index} content is empty")
        messages.append(CorpusMessage(role=role, content=content))

    must_not_raw = data.get("must_not_contain") or []
    if not isinstance(must_not_raw, list):
        raise ValueError(f"case {data.get('id')}: must_not_contain must be a list")
    expected_any_raw = data.get("expected_any") or []
    if not isinstance(expected_any_raw, list):
        raise ValueError(f"case {data.get('id')}: expected_any must be a list")
    expected_any: list[list[str]] = []
    for index, group in enumerate(expected_any_raw):
        if not isinstance(group, list) or not group:
            raise ValueError(f"case {data.get('id')}: expected_any[{index}] must be a non-empty list")
        cleaned = [str(item).strip() for item in group if str(item).strip()]
        if not cleaned:
            raise ValueError(f"case {data.get('id')}: expected_any[{index}] must contain text")
        expected_any.append(cleaned)

    return CorpusCase(
        case_id=_require_text(data, "id"),
        category=_require_text(data, "category"),
        risk=_require_text(data, "risk"),
        messages=messages,
        question=_require_text(data, "question"),
        expected_fragments=_require_text_list(data, "expected_fragments"),
        expected_any=expected_any,
        must_not_contain=[str(item).strip() for item in must_not_raw if str(item).strip()],
        notes=str(data.get("notes") or "").strip(),
    )


def load_cases(path: Path) -> list[CorpusCase]:
    cases: list[CorpusCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                cases.append(parse_case(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if not cases:
        raise ValueError(f"No cases loaded from {path}")
    return cases


def to_session_log(case: CorpusCase) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for message in case.messages:
        if message.role == "user":
            result.append({"type": "user_text", "text": message.content})
        else:
            result.append({"type": "model_response", "text": message.content})
    return result


def append_noise_turns(cases: list[CorpusCase], count: int) -> list[CorpusCase]:
    if count <= 0:
        return cases

    expanded: list[CorpusCase] = []
    for case in cases:
        messages = list(case.messages)
        for index in range(count):
            messages.append(
                CorpusMessage(
                    role="user",
                    content=(
                        f"Routine unrelated continuation {index}: planning, formatting, "
                        "status notes, and non-critical operational context."
                    ),
                )
            )
            messages.append(CorpusMessage(role="assistant", content=f"Acknowledged routine note {index}."))
        expanded.append(
            CorpusCase(
                case_id=case.case_id,
                category=case.category,
                risk=case.risk,
                messages=messages,
                question=case.question,
                expected_fragments=case.expected_fragments,
                expected_any=case.expected_any,
                must_not_contain=case.must_not_contain,
                notes=case.notes,
            )
        )
    return expanded


def conversation_request_count(case: CorpusCase) -> int:
    return sum(1 for message in case.messages if message.role == "user") + 1


# Words that, when they immediately precede a forbidden phrase, mean the answer
# is *prohibiting* it, not using it - e.g. "never use marketing language",
# "without marketing", "not perfect prose". Such a match is not a violation.
_NEGATION_CUES = (
    "never", "not", "no", "without", "avoid", "don't", "do not", "dont",
    "instead of", "rather than", "more than", "over ",
    "\u043d\u0435 ", "\u043d\u0456\u043a\u043e\u043b\u0438", "\u0431\u0435\u0437 ",
    "\u0443\u043d\u0438\u043a\u0430\u0439", "\u0437\u0430\u043c\u0456\u0441\u0442\u044c",
)


def _forbidden_used(answer_cf: str, phrase_cf: str) -> bool:
    """True only when the forbidden phrase is used affirmatively.

    A ``must_not_contain`` phrase is a real violation only if the answer asserts
    it. When it appears right after a negation cue (the answer is stating a rule
    *against* it, or contrasting), it is not a violation. This removes the
    false-fails where a correct answer quotes a prohibition ("never use
    marketing language") or a contrast ("throughput over perfect prose").
    """
    start = 0
    while True:
        idx = answer_cf.find(phrase_cf, start)
        if idx == -1:
            return False
        # look at the up-to-40 chars of context immediately before the match
        window = answer_cf[max(0, idx - 40):idx]
        if not any(cue in window for cue in _NEGATION_CUES):
            return True  # affirmative use - real violation
        start = idx + len(phrase_cf)


def check_answer(
    answer: str,
    expected: list[str],
    must_not: list[str] | None = None,
    expected_any: list[list[str]] | None = None,
) -> dict[str, Any]:
    normalized = (answer or "").casefold()
    missing = [fragment for fragment in expected if fragment.casefold() not in normalized]
    missing_any = [
        group
        for group in expected_any or []
        if not any(fragment.casefold() in normalized for fragment in group)
    ]
    forbidden = [
        fragment
        for fragment in must_not or []
        if _forbidden_used(normalized, fragment.casefold())
    ]
    return {
        "correct": not missing and not missing_any and not forbidden,
        "missing": missing,
        "missing_any": missing_any,
        "forbidden": forbidden,
    }


def _safe_pct(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round((float(numerator) / float(denominator)) * 100, 2)


def _break_even_request_count(
    *,
    raw_prompt_tokens: int,
    optimized_prompt_tokens: int,
    one_time_setup_tokens: int = 0,
) -> int | None:
    """Return first final-answer request where optimized cumulative cost wins."""
    per_request_saving = raw_prompt_tokens - optimized_prompt_tokens
    if per_request_saving <= 0:
        return None
    if one_time_setup_tokens <= 0:
        return 1
    return (one_time_setup_tokens + per_request_saving - 1) // per_request_saving


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


class DryMeter:
    """Local stand-in for snapshot folding; never calls an external model."""

    def __init__(self):
        self.calls = 0
        self.prompt_tokens = 0
        self.output_tokens = 0

    def __call__(self, prompt: str):
        from remy.core.session_state_wrapper import estimate_tokens, extract_pinned_facts

        pins = extract_pinned_facts(prompt)
        snapshot = "[DO_NOT_FORGET]\n" + ", ".join(pins) if pins else "[DO_NOT_FORGET]\nDry run."
        self.calls += 1
        self.prompt_tokens += estimate_tokens(prompt)
        self.output_tokens += estimate_tokens(snapshot)
        return (
            snapshot,
            {
                "usage_metadata": {
                    "prompt_token_count": estimate_tokens(prompt),
                    "candidates_token_count": estimate_tokens(snapshot),
                }
            },
        )


def _exchange_text(items: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for item in items:
        kind = item.get("type")
        if kind == "user_text":
            lines.append(f"user: {item.get('text', '')}")
        elif kind == "model_response":
            lines.append(f"assistant: {item.get('text', '')}")
    return "\n".join(lines)


def _exchanges(session_log: list[dict[str, str]]) -> list[str]:
    result: list[str] = []
    pending: list[dict[str, str]] = []
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


async def _build_incremental_prompt(
    case: CorpusCase,
    answer_func: Any,
    *,
    fold_batch_size: int,
) -> tuple[str, dict[str, Any]]:
    from remy.core.session_state_wrapper import (
        SessionState,
        build_optimized_prompt,
        update_state_incremental,
    )

    session_log = to_session_log(case)
    state = SessionState(session_id=case.case_id)
    exchanges = _exchanges(session_log)
    batches = [
        exchanges[index : index + fold_batch_size]
        for index in range(0, len(exchanges), fold_batch_size)
    ]

    before_prompt = getattr(answer_func, "prompt_tokens", 0)
    before_output = getattr(answer_func, "output_tokens", 0)
    before_calls = getattr(answer_func, "calls", 0)

    for batch in batches:
        state = await update_state_incremental(
            state,
            "\n\n".join(batch),
            answer_func,
            persist=False,
        )

    update_prompt_tokens = getattr(answer_func, "prompt_tokens", 0) - before_prompt
    update_output_tokens = getattr(answer_func, "output_tokens", 0) - before_output
    update_calls = getattr(answer_func, "calls", 0) - before_calls
    prompt = build_optimized_prompt(state, session_log, case.question, system_prompt=SYSTEM_PROMPT)
    return prompt, {
        "state_update_calls": update_calls,
        "state_update_prompt_tokens": update_prompt_tokens,
        "state_update_output_tokens": update_output_tokens,
        "state_update_total_tokens": update_prompt_tokens + update_output_tokens,
    }


async def build_prompt_for_mode(
    case: CorpusCase,
    mode: str,
    answer_func: Any | None = None,
    *,
    fold_batch_size: int = 10,
) -> tuple[str, dict[str, Any]]:
    from remy.core.session_state_wrapper import (
        build_optimized_prompt,
        build_projected_state_from_log,
        build_raw_prompt,
    )

    session_log = to_session_log(case)
    if mode == "raw":
        return build_raw_prompt(session_log, case.question, system_prompt=SYSTEM_PROMPT), {}
    if mode in {"projected", "projected_facts", "projected_hybrid"}:
        # projected_facts is the cost-first baseline: no setup LLM calls.
        # projected_hybrid can recover decisions/preferences, but must prove
        # that its extra provider calls are paid back by later prompt savings.
        # projected is kept as the old compatibility alias for hybrid.
        uses_decision_llm = mode in {"projected", "projected_hybrid"} and callable(answer_func)
        decision_func = answer_func if uses_decision_llm else None
        before_calls = getattr(answer_func, "calls", 0) if answer_func else 0
        before_prompt = getattr(answer_func, "prompt_tokens", 0) if answer_func else 0
        before_output = getattr(answer_func, "output_tokens", 0) if answer_func else 0
        state = build_projected_state_from_log(
            session_log,
            session_id=case.case_id,
            decision_llm_func=decision_func,
        )
        decision_calls = (getattr(answer_func, "calls", 0) - before_calls) if answer_func else 0
        decision_prompt_tokens = (
            getattr(answer_func, "prompt_tokens", 0) - before_prompt
        ) if answer_func else 0
        decision_output_tokens = (
            getattr(answer_func, "output_tokens", 0) - before_output
        ) if answer_func else 0
        prompt = build_optimized_prompt(state, session_log, case.question, system_prompt=SYSTEM_PROMPT)
        return prompt, {
            "projection_strategy": "hybrid" if mode in {"projected", "projected_hybrid"} else "facts",
            "pinned_count": len(state.pinned_facts),
            "decision_extract_calls": decision_calls,
            "decision_extract_prompt_tokens": decision_prompt_tokens,
            "decision_extract_output_tokens": decision_output_tokens,
            "decision_extract_total_tokens": decision_prompt_tokens + decision_output_tokens,
        }
    if mode == "incremental":
        if answer_func is None:
            raise ValueError("incremental mode requires answer_func")
        return await _build_incremental_prompt(case, answer_func, fold_batch_size=fold_batch_size)
    raise ValueError(f"Unsupported mode: {mode}")


async def run_eval(
    cases: list[CorpusCase],
    *,
    modes: list[str],
    model: str,
    dry_run: bool,
    delay_sec: float,
    fold_batch_size: int,
    context_window_tokens: int,
    append_noise_turns: int = 0,
) -> dict[str, Any]:
    from remy.core.session_state_wrapper import estimate_tokens

    answer_func = DryMeter() if dry_run else GeminiMeter(model, delay_sec=delay_sec)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for case in cases:
        for mode in modes:
            before_prompt = getattr(answer_func, "prompt_tokens", 0) if answer_func else 0
            before_output = getattr(answer_func, "output_tokens", 0) if answer_func else 0
            before_calls = getattr(answer_func, "calls", 0) if answer_func else 0

            prompt, extra = await build_prompt_for_mode(
                case,
                mode,
                answer_func,
                fold_batch_size=fold_batch_size,
            )

            answer_started = time.perf_counter()
            if dry_run:
                answer = " ".join(
                    case.expected_fragments + [group[0] for group in case.expected_any]
                )
            else:
                answer, _meta = answer_func(prompt)
            latency_ms = int((time.perf_counter() - answer_started) * 1000)

            after_prompt = getattr(answer_func, "prompt_tokens", 0) if answer_func else 0
            after_output = getattr(answer_func, "output_tokens", 0) if answer_func else 0
            after_calls = getattr(answer_func, "calls", 0) if answer_func else 0
            correctness = check_answer(
                answer,
                case.expected_fragments,
                case.must_not_contain,
                case.expected_any,
            )
            provider_prompt_tokens = after_prompt - before_prompt
            provider_output_tokens = after_output - before_output
            provider_total_tokens = provider_prompt_tokens + provider_output_tokens
            state_update_prompt_tokens = int(extra.get("state_update_prompt_tokens") or 0)
            state_update_output_tokens = int(extra.get("state_update_output_tokens") or 0)
            decision_extract_prompt_tokens = int(extra.get("decision_extract_prompt_tokens") or 0)
            final_answer_provider_prompt_tokens = max(
                0,
                provider_prompt_tokens - state_update_prompt_tokens - decision_extract_prompt_tokens,
            )

            rows.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "risk": case.risk,
                    "mode": mode,
                    "conversation_request_count": conversation_request_count(case),
                    "correct": correctness["correct"],
                    "missing": correctness["missing"],
                    "missing_any": correctness["missing_any"],
                    "forbidden": correctness["forbidden"],
                    "prompt_tokens_estimate": estimate_tokens(prompt),
                    "provider_prompt_tokens": provider_prompt_tokens,
                    "provider_output_tokens": provider_output_tokens,
                    "provider_total_tokens": provider_total_tokens,
                    "final_answer_provider_prompt_tokens": final_answer_provider_prompt_tokens,
                    "provider_calls": after_calls - before_calls,
                    "latency_ms": latency_ms,
                    "answer_preview": answer[:500],
                    **extra,
                }
            )

    raw_by_case = {row["case_id"]: row for row in rows if row["mode"] == "raw"}
    for row in rows:
        raw_row = raw_by_case.get(row["case_id"])
        if not raw_row:
            continue
        raw_estimate = int(raw_row["prompt_tokens_estimate"])
        prompt_estimate = int(row["prompt_tokens_estimate"])
        delta_estimate = raw_estimate - prompt_estimate
        setup_estimate = int(row.get("state_update_total_tokens") or 0) + int(
            row.get("decision_extract_total_tokens") or 0
        )
        row["raw_prompt_tokens_estimate"] = raw_estimate
        row["prompt_tokens_delta_vs_raw_estimate"] = delta_estimate
        row["context_window_saved_tokens_estimate"] = delta_estimate
        row["context_window_tokens"] = context_window_tokens
        row["raw_context_window_used_pct_estimate"] = _safe_pct(raw_estimate, context_window_tokens)
        row["context_window_used_pct_estimate"] = _safe_pct(prompt_estimate, context_window_tokens)
        row["context_window_freed_pct_points_estimate"] = round(
            row["raw_context_window_used_pct_estimate"] - row["context_window_used_pct_estimate"],
            2,
        )
        row["context_window_saved_pct_vs_raw_estimate"] = _safe_pct(delta_estimate, raw_estimate)
        row["optimization_effective"] = bool(row.get("correct")) and delta_estimate > 0
        row["efficiency_score_estimate"] = (
            max(0.0, float(row["context_window_saved_pct_vs_raw_estimate"]))
            if row["optimization_effective"]
            else 0.0
        )
        row["break_even_request_count_estimate"] = _break_even_request_count(
            raw_prompt_tokens=raw_estimate,
            optimized_prompt_tokens=prompt_estimate,
            one_time_setup_tokens=setup_estimate,
        )

        raw_provider_prompt = int(raw_row.get("final_answer_provider_prompt_tokens") or 0)
        provider_prompt = int(row.get("final_answer_provider_prompt_tokens") or 0)
        raw_provider_total = int(raw_row.get("provider_total_tokens") or 0)
        provider_total = int(row.get("provider_total_tokens") or 0)
        provider_setup = (
            int(row.get("state_update_prompt_tokens") or 0)
            + int(row.get("state_update_output_tokens") or 0)
            + int(row.get("decision_extract_prompt_tokens") or 0)
            + int(row.get("decision_extract_output_tokens") or 0)
        )
        if raw_provider_prompt and provider_prompt:
            row["provider_prompt_delta_vs_raw"] = raw_provider_prompt - provider_prompt
            row["provider_prompt_saved_pct_vs_raw"] = _safe_pct(
                raw_provider_prompt - provider_prompt,
                raw_provider_prompt,
            )
            row["provider_break_even_request_count"] = _break_even_request_count(
                raw_prompt_tokens=raw_provider_prompt,
                optimized_prompt_tokens=provider_prompt,
                one_time_setup_tokens=provider_setup,
            )
        if raw_provider_total and provider_total:
            row["provider_total_delta_vs_raw"] = raw_provider_total - provider_total
            row["provider_total_saved_pct_vs_raw"] = _safe_pct(
                raw_provider_total - provider_total,
                raw_provider_total,
            )

    by_mode: dict[str, dict[str, Any]] = {}
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        correct = sum(1 for row in mode_rows if row["correct"])
        effective = sum(1 for row in mode_rows if row.get("optimization_effective"))
        break_even_values = [
            int(row["break_even_request_count_estimate"])
            for row in mode_rows
            if row.get("break_even_request_count_estimate") is not None
        ]
        context_saved_tokens = sum(
            row.get("context_window_saved_tokens_estimate", 0) for row in mode_rows
        )
        by_mode[mode] = {
            "cases": len(mode_rows),
            "accuracy": correct / len(mode_rows) if mode_rows else 0.0,
            "correct": correct,
            "effective_cases": effective,
            "effectiveness_rate": effective / len(mode_rows) if mode_rows else 0.0,
            "prompt_tokens_estimate": sum(row["prompt_tokens_estimate"] for row in mode_rows),
            "prompt_tokens_delta_vs_raw_estimate": sum(
                row.get("prompt_tokens_delta_vs_raw_estimate", 0) for row in mode_rows
            ),
            "context_window_saved_tokens_estimate": context_saved_tokens,
            "context_window_saved_pct_vs_raw_estimate": _safe_pct(
                context_saved_tokens,
                sum(row.get("raw_prompt_tokens_estimate", 0) for row in mode_rows),
            ),
            "accuracy_weighted_context_saving_pct": round(
                (correct / len(mode_rows) if mode_rows else 0.0)
                * max(
                    0.0,
                    _safe_pct(
                        context_saved_tokens,
                        sum(row.get("raw_prompt_tokens_estimate", 0) for row in mode_rows),
                    ),
                ),
                2,
            ),
            "mean_efficiency_score_estimate": round(
                sum(float(row.get("efficiency_score_estimate") or 0.0) for row in mode_rows)
                / len(mode_rows),
                2,
            )
            if mode_rows
            else 0.0,
            "context_window_freed_pct_points_estimate": round(
                sum(row.get("context_window_freed_pct_points_estimate", 0.0) for row in mode_rows),
                2,
            ),
            "profitable_cases_estimate": sum(
                1 for row in mode_rows if row.get("break_even_request_count_estimate") is not None
            ),
            "first_profitable_request_estimate": min(break_even_values)
            if break_even_values
            else None,
            "median_break_even_request_estimate": sorted(break_even_values)[
                len(break_even_values) // 2
            ]
            if break_even_values
            else None,
            "provider_prompt_tokens": sum(row["provider_prompt_tokens"] for row in mode_rows),
            "provider_output_tokens": sum(row["provider_output_tokens"] for row in mode_rows),
            "provider_total_tokens": sum(row["provider_total_tokens"] for row in mode_rows),
            "provider_calls": sum(row["provider_calls"] for row in mode_rows),
        }

    if "raw" in by_mode:
        raw_estimate = max(1, by_mode["raw"]["prompt_tokens_estimate"])
        raw_provider_total = int(by_mode["raw"].get("provider_total_tokens") or 0)
        raw_provider_prompt = int(by_mode["raw"].get("provider_prompt_tokens") or 0)
        for mode in modes:
            by_mode[mode]["estimated_prompt_ratio_vs_raw"] = round(
                raw_estimate / max(1, by_mode[mode]["prompt_tokens_estimate"]),
                2
            )
            provider_total = int(by_mode[mode].get("provider_total_tokens") or 0)
            provider_prompt = int(by_mode[mode].get("provider_prompt_tokens") or 0)
            if raw_provider_total and provider_total:
                by_mode[mode]["provider_total_delta_vs_raw"] = raw_provider_total - provider_total
                by_mode[mode]["provider_total_saved_pct_vs_raw"] = _safe_pct(
                    raw_provider_total - provider_total,
                    raw_provider_total,
                )
            if raw_provider_prompt and provider_prompt:
                by_mode[mode]["provider_prompt_delta_vs_raw"] = raw_provider_prompt - provider_prompt
                by_mode[mode]["provider_prompt_saved_pct_vs_raw"] = _safe_pct(
                    raw_provider_prompt - provider_prompt,
                    raw_provider_prompt,
                )

    return {
        "summary": {
            "model": model,
            "dry_run": dry_run,
            "cases": len(cases),
            "modes": modes,
            "context_window_tokens": context_window_tokens,
            "append_noise_turns": append_noise_turns,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "by_mode": by_mode,
        },
        "rows": rows,
    }


def _parse_modes(value: str) -> list[str]:
    if value == "all":
        return list(CANONICAL_MODES)
    modes = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [mode for mode in modes if mode not in MODES]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown mode(s): {', '.join(unknown)}")
    return modes


def _write_report(report: dict[str, Any]) -> Path:
    from remy.config.settings import settings

    out_dir = Path(settings.DATA_DIR) / "llm_optimization"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"corpus_eval_{stamp}_{time.time_ns()}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


async def _main_async(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case.case_id in wanted]
        missing = sorted(wanted - {case.case_id for case in cases})
        if missing:
            raise ValueError(f"Unknown case id(s): {', '.join(missing)}")
    if args.case_limit:
        cases = cases[: args.case_limit]
    cases = append_noise_turns(cases, args.append_noise_turns)
    report = await run_eval(
        cases,
        modes=args.mode,
        model=args.model,
        dry_run=args.dry_run,
        delay_sec=args.delay_sec,
        fold_batch_size=args.fold_batch_size,
        context_window_tokens=args.context_window_tokens,
        append_noise_turns=args.append_noise_turns,
    )
    path = _write_report(report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--mode", type=_parse_modes, default=list(CANONICAL_MODES))
    parser.add_argument("--model", default="gemini-flash-lite-latest")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay-sec", type=float, default=0.0)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run only the named case id. Can be passed multiple times.",
    )
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--fold-batch-size", type=int, default=10)
    parser.add_argument("--context-window-tokens", type=int, default=128000)
    parser.add_argument(
        "--append-noise-turns",
        type=int,
        default=0,
        help="Append N unrelated user/assistant exchanges to each case to simulate later requests.",
    )
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
