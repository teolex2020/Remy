"""Evaluate Remy context optimization on RepoBench-style code completion data.

RepoBench is useful for this project because it tests repository-level code
context rather than human episodic memory. This runner accepts local parquet,
JSON, or JSONL files shaped like ``tianyang/repobench_python_v1.1`` and keeps
the code benchmark separate from the synthetic Remy corpus.
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
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from remy.core.session_state_wrapper import (  # noqa: E402
    build_optimized_prompt,
    build_projected_state_from_log,
    estimate_tokens,
)
from tools.validation.llm_optimization_corpus_eval import (  # noqa: E402
    DryMeter,
    GeminiMeter,
)


DEFAULT_REPOBENCH = ROOT / "data" / "evals" / "external" / "repobench_python_cross_file_first.parquet"
REPOBENCH_MODES = (
    "raw",
    "projected_facts",
    "projected_hybrid",
    "cropped_only",
    "retrieved_snippets",
    "retrieved_snippets_or_fallback",
    "gold_snippet_oracle",
)
CODE_SYSTEM_PROMPT = (
    "You are evaluating repository-level Python code completion. "
    "Return exactly the next line of code and nothing else. "
    "Do not use markdown fences. Preserve indentation if needed."
)


@dataclass(frozen=True)
class RepoSnippet:
    identifier: str
    path: str
    snippet: str


@dataclass(frozen=True)
class RepoBenchItem:
    case_id: str
    repo_name: str
    file_path: str
    context: list[RepoSnippet]
    import_statement: str
    cropped_code: str
    next_line: str
    gold_snippet_index: int | None
    level: str


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _context_from_any(value: Any) -> list[RepoSnippet]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return [RepoSnippet(identifier="", path="", snippet=value)]
    if not isinstance(value, Iterable):
        return []

    snippets: list[RepoSnippet] = []
    for entry in value:
        if isinstance(entry, dict):
            identifier = _clean_text(entry.get("identifier"))
            path = _clean_text(entry.get("path"))
            snippet = _clean_text(entry.get("snippet"))
        else:
            identifier = ""
            path = ""
            snippet = _clean_text(entry)
        if snippet:
            snippets.append(RepoSnippet(identifier=identifier, path=path, snippet=snippet))
    return snippets


def _item_from_row(row: dict[str, Any], index: int) -> RepoBenchItem | None:
    next_line = _clean_text(row.get("next_line"))
    cropped_code = _clean_text(row.get("cropped_code"))
    if not next_line or not cropped_code:
        return None
    repo_name = _clean_text(row.get("repo_name")) or "unknown_repo"
    file_path = _clean_text(row.get("file_path")) or "unknown.py"
    return RepoBenchItem(
        case_id=f"{repo_name}:{file_path}:{index}",
        repo_name=repo_name,
        file_path=file_path,
        context=_context_from_any(row.get("context")),
        import_statement=_clean_text(row.get("import_statement")),
        cropped_code=cropped_code,
        next_line=next_line,
        gold_snippet_index=_safe_int(row.get("gold_snippet_index")),
        level=_clean_text(row.get("level")),
    )


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise ValueError(f"{path}:{line_no}: JSONL row must be an object")
                rows.append(data)
        return rows
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("rows") or data.get("data") or [data]
    if not isinstance(data, list):
        raise ValueError(f"{path}: JSON file must contain a list or a row object")
    return [row for row in data if isinstance(row, dict)]


def load_repobench_items(path: Path, *, case_limit: int = 0) -> list[RepoBenchItem]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pandas as pd

        frame = pd.read_parquet(path)
        rows = frame.to_dict(orient="records")
    elif suffix in {".json", ".jsonl"}:
        rows = _load_json_rows(path)
    else:
        raise ValueError(f"Unsupported RepoBench file type: {path.suffix}")

    items: list[RepoBenchItem] = []
    for index, row in enumerate(rows):
        item = _item_from_row(row, index)
        if item is None:
            continue
        items.append(item)
        if case_limit and len(items) >= case_limit:
            break
    if not items:
        raise ValueError(f"No RepoBench items loaded from {path}")
    return items


def _snippet_block(snippet: RepoSnippet) -> str:
    header_parts = []
    if snippet.path:
        header_parts.append(f"Path: {snippet.path}")
    if snippet.identifier:
        header_parts.append(f"Identifier: {snippet.identifier}")
    header = "# " + " | ".join(header_parts) if header_parts else "# Context snippet"
    return f"{header}\n{snippet.snippet}"


def _repo_context_text(item: RepoBenchItem, snippets: list[RepoSnippet]) -> str:
    cross_file = "\n\n".join(_snippet_block(snippet) for snippet in snippets)
    target_parts = [
        f"# Path: {item.file_path}",
        item.import_statement,
        item.cropped_code,
    ]
    target = "\n".join(part for part in target_parts if part)
    if cross_file:
        return f"# Repo Name: {item.repo_name}\n{cross_file}\n\n{target}"
    return f"# Repo Name: {item.repo_name}\n{target}"


def _gold_snippets(item: RepoBenchItem) -> list[RepoSnippet]:
    if item.gold_snippet_index is None:
        return []
    if item.gold_snippet_index < 0 or item.gold_snippet_index >= len(item.context):
        return []
    return [item.context[item.gold_snippet_index]]


_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PYTHON_STOPWORDS = {
    "and",
    "as",
    "assert",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "false",
    "finally",
    "for",
    "from",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "none",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "self",
    "true",
    "try",
    "while",
    "with",
    "yield",
}


def _identifier_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in _IDENTIFIER_RE.findall(text or ""):
        token = match.casefold()
        if len(token) < 3 or token in _PYTHON_STOPWORDS:
            continue
        tokens.add(token)
        for part in token.split("_"):
            if len(part) >= 3 and part not in _PYTHON_STOPWORDS:
                tokens.add(part)
    return tokens


def _path_tokens(path: str) -> set[str]:
    return _identifier_tokens((path or "").replace("/", " ").replace("\\", " ").replace(".", " "))


def _last_target_tokens(item: RepoBenchItem, *, max_lines: int = 40) -> set[str]:
    lines = item.cropped_code.splitlines()
    tail = "\n".join(lines[-max_lines:])
    return _identifier_tokens(f"{item.import_statement}\n{tail}\n{item.file_path}")


def score_snippet_for_item(item: RepoBenchItem, snippet: RepoSnippet) -> float:
    """Cheap lexical score for choosing code context before the LLM call."""
    target_tokens = _last_target_tokens(item)
    snippet_tokens = _identifier_tokens(f"{snippet.identifier}\n{snippet.snippet}")
    snippet_path_tokens = _path_tokens(snippet.path)
    target_path_tokens = _path_tokens(item.file_path)

    overlap = target_tokens & snippet_tokens
    path_overlap = target_path_tokens & snippet_path_tokens
    import_overlap = _identifier_tokens(item.import_statement) & snippet_tokens

    score = 0.0
    score += len(overlap) * 4.0
    score += len(import_overlap) * 6.0
    score += len(path_overlap) * 2.0
    if snippet.identifier and snippet.identifier.casefold() in target_tokens:
        score += 12.0
    if snippet.path and item.file_path:
        if Path(snippet.path).suffix == Path(item.file_path).suffix:
            score += 0.5
        if Path(snippet.path).parent == Path(item.file_path).parent:
            score += 1.5
    # Slightly prefer compact snippets when lexical evidence is tied.
    score -= min(2.0, estimate_tokens(snippet.snippet) / 1000)
    return score


def scored_snippets_for_item(item: RepoBenchItem) -> list[tuple[float, int, RepoSnippet]]:
    scored = [
        (score_snippet_for_item(item, snippet), index, snippet)
        for index, snippet in enumerate(item.context)
    ]
    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    return scored


def retrieved_snippets(item: RepoBenchItem, *, limit: int = 1) -> list[RepoSnippet]:
    if limit <= 0 or not item.context:
        return []
    scored = scored_snippets_for_item(item)
    selected = [snippet for score, _index, snippet in scored if score > 0][:limit]
    if selected:
        return selected
    return [snippet for _score, _index, snippet in scored[:1]]


def review_retrieved_snippet_authority(
    item: RepoBenchItem,
    selected: list[RepoSnippet],
    *,
    min_score: float = 4.0,
    ambiguity_margin: float = 1.0,
) -> dict[str, Any]:
    """Fail-closed gate for using retrieved code snippets as cheap context."""
    scored = scored_snippets_for_item(item)
    selected_indexes = [
        item.context.index(snippet)
        for snippet in selected
        if snippet in item.context
    ]
    top_score = float(scored[0][0]) if scored else 0.0
    second_score = float(scored[1][0]) if len(scored) > 1 else 0.0
    candidate_found = bool(selected)
    candidate_ambiguous = (
        candidate_found
        and len(scored) > 1
        and top_score > 0
        and (top_score - second_score) < ambiguity_margin
    )

    block_reasons: list[str] = []
    if not candidate_found:
        block_reasons.append("RetrievedCandidateMissing")
    if top_score <= 0:
        block_reasons.append("RetrievedCandidateHasNoPositiveEvidence")
    if top_score < min_score:
        block_reasons.append("RetrievedCandidateBelowAuthorityThreshold")
    if candidate_ambiguous:
        block_reasons.append("RetrievedCandidateAmbiguous")

    decision = "Stop" if block_reasons else "Go"
    return {
        "schema": "remy_repobench_retrieval_authority_v1",
        "decision": decision,
        "status": "PASS" if decision == "Go" else "FAIL",
        "block_reasons": list(dict.fromkeys(block_reasons)),
        "candidate_found": candidate_found,
        "candidate_ambiguous": candidate_ambiguous,
        "candidate_authorized": decision == "Go",
        "product_context_use_allowed": decision == "Go",
        "selected_indexes": selected_indexes,
        "top_score": round(top_score, 3),
        "second_score": round(second_score, 3),
        "min_score": min_score,
        "ambiguity_margin": ambiguity_margin,
        "answer_permission_granted": False,
        "truth_asserted": False,
        "llm_called_by_gate": False,
    }


def _session_log_for_code(item: RepoBenchItem) -> list[dict[str, str]]:
    log: list[dict[str, str]] = []
    for snippet in item.context:
        log.append({"type": "user_text", "text": _snippet_block(snippet)})
    target = _repo_context_text(item, [])
    log.append({"type": "user_text", "text": target})
    return log


def _completion_question(item: RepoBenchItem) -> str:
    return (
        "Complete the next Python line for the target file. "
        "Return exactly one line.\n"
        f"Target file: {item.file_path}"
    )


def _raw_prompt(item: RepoBenchItem, snippets: list[RepoSnippet] | None = None) -> str:
    selected = item.context if snippets is None else snippets
    return (
        f"{CODE_SYSTEM_PROMPT}\n\n"
        "[REPOSITORY_CONTEXT]\n"
        f"{_repo_context_text(item, selected)}\n\n"
        "[TASK]\n"
        f"{_completion_question(item)}"
    )


async def build_prompt_for_repobench_mode(
    item: RepoBenchItem,
    mode: str,
    answer_func: Any,
) -> tuple[str, dict[str, Any]]:
    if mode == "raw":
        return _raw_prompt(item), {}
    if mode == "cropped_only":
        return _raw_prompt(item, []), {}
    if mode in {"retrieved_snippets", "retrieved_snippets_or_fallback"}:
        selected = retrieved_snippets(item)
        authority = review_retrieved_snippet_authority(item, selected)
        use_raw_fallback = mode == "retrieved_snippets_or_fallback" and authority["decision"] != "Go"
        prompt = _raw_prompt(item) if use_raw_fallback else _raw_prompt(item, selected)
        return prompt, {
            "retrieved_snippet_count": len(selected),
            "retrieved_snippet_indexes": authority["selected_indexes"],
            "retrieved_hit_gold": item.gold_snippet_index in [
                item.context.index(snippet)
                for snippet in selected
                if snippet in item.context
            ],
            "candidate_found": authority["candidate_found"],
            "candidate_ambiguous": authority["candidate_ambiguous"],
            "candidate_authorized": authority["candidate_authorized"],
            "authority_decision": authority["decision"],
            "fallback_raw": use_raw_fallback,
            "fallback_reason": ";".join(authority["block_reasons"]) if use_raw_fallback else "",
            "retrieval_top_score": authority["top_score"],
            "retrieval_second_score": authority["second_score"],
        }
    if mode == "gold_snippet_oracle":
        return _raw_prompt(item, _gold_snippets(item)), {
            "oracle_gold_snippet_index": item.gold_snippet_index,
        }
    if mode in {"projected_facts", "projected_hybrid"}:
        uses_decision_llm = mode == "projected_hybrid" and callable(answer_func)
        decision_func = answer_func if uses_decision_llm else None
        before_calls = getattr(answer_func, "calls", 0) if answer_func else 0
        before_prompt = getattr(answer_func, "prompt_tokens", 0) if answer_func else 0
        before_output = getattr(answer_func, "output_tokens", 0) if answer_func else 0
        state = build_projected_state_from_log(
            _session_log_for_code(item),
            session_id=item.case_id,
            decision_llm_func=decision_func,
        )
        prompt = build_optimized_prompt(
            state,
            _session_log_for_code(item),
            _completion_question(item),
            system_prompt=CODE_SYSTEM_PROMPT,
        )
        return prompt, {
            "projection_strategy": "hybrid" if mode == "projected_hybrid" else "facts",
            "pinned_count": len(state.pinned_facts),
            "decision_extract_calls": getattr(answer_func, "calls", 0) - before_calls,
            "decision_extract_prompt_tokens": getattr(answer_func, "prompt_tokens", 0)
            - before_prompt,
            "decision_extract_output_tokens": getattr(answer_func, "output_tokens", 0)
            - before_output,
        }
    raise ValueError(f"Unsupported RepoBench mode: {mode}")


def normalize_code_line(value: str) -> str:
    line = (value or "").strip()
    if line.startswith("```"):
        lines = [item.strip() for item in line.splitlines()]
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        if lines and lines[0].casefold() in {"python", "py"}:
            lines = lines[1:]
        line = "\n".join(lines).strip()
    line = line.splitlines()[0].strip() if line.splitlines() else line
    return " ".join(line.split())


def code_line_matches(answer: str, gold: str) -> bool:
    answer_norm = normalize_code_line(answer)
    gold_norm = normalize_code_line(gold)
    if not answer_norm or not gold_norm:
        return False
    return answer_norm == gold_norm or gold_norm in answer_norm


def compute_retrieval_gate_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure whether Go/Stop correlates with retrieval correctness.

    Savings are commercially useful only if the gate stops bad cheap context
    more often than it stops good cheap context. This diagnostic is intentionally
    computed after evaluation, so it is a falsification metric, not an oracle
    used during routing.
    """
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        case_id = str(row.get("case_id") or "")
        mode = str(row.get("mode") or "")
        if not case_id or not mode:
            continue
        by_case.setdefault(case_id, {})[mode] = row

    paired: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for modes in by_case.values():
        retrieved = modes.get("retrieved_snippets")
        fallback = modes.get("retrieved_snippets_or_fallback")
        if retrieved and fallback:
            paired.append((retrieved, fallback))

    wrong_retrieval = 0
    wrong_retrieval_stopped = 0
    correct_retrieval = 0
    correct_retrieval_stopped = 0
    stop_count = 0
    stop_on_wrong = 0
    stop_on_correct = 0
    fallback_fixed_wrong = 0
    fallback_broke_correct = 0

    for retrieved, fallback in paired:
        retrieved_correct = bool(retrieved.get("correct"))
        stopped = str(fallback.get("authority_decision") or "") == "Stop"
        fallback_correct = bool(fallback.get("correct"))

        if stopped:
            stop_count += 1
        if retrieved_correct:
            correct_retrieval += 1
            if stopped:
                correct_retrieval_stopped += 1
                stop_on_correct += 1
            if not fallback_correct:
                fallback_broke_correct += 1
        else:
            wrong_retrieval += 1
            if stopped:
                wrong_retrieval_stopped += 1
                stop_on_wrong += 1
            if fallback_correct:
                fallback_fixed_wrong += 1

    wrong_stop_rate = wrong_retrieval_stopped / wrong_retrieval if wrong_retrieval else 0.0
    false_stop_rate = correct_retrieval_stopped / correct_retrieval if correct_retrieval else 0.0
    stop_precision = stop_on_wrong / stop_count if stop_count else 0.0
    return {
        "schema": "remy_repobench_retrieval_gate_diagnostics_v1",
        "paired_cases": len(paired),
        "wrong_retrieval_count": wrong_retrieval,
        "wrong_retrieval_stopped_count": wrong_retrieval_stopped,
        "wrong_retrieval_go_count": wrong_retrieval - wrong_retrieval_stopped,
        "wrong_retrieval_stop_rate": round(wrong_stop_rate, 4),
        "correct_retrieval_count": correct_retrieval,
        "correct_retrieval_stopped_count": correct_retrieval_stopped,
        "correct_retrieval_go_count": correct_retrieval - correct_retrieval_stopped,
        "false_stop_rate_on_correct_retrieval": round(false_stop_rate, 4),
        "stop_count": stop_count,
        "stop_precision_wrong_retrieval": round(stop_precision, 4),
        "fallback_fixed_wrong_retrieval_count": fallback_fixed_wrong,
        "fallback_broke_correct_retrieval_count": fallback_broke_correct,
        "gate_discriminates": wrong_stop_rate > false_stop_rate and wrong_retrieval_stopped > 0,
    }


async def run_repobench_eval(
    items: list[RepoBenchItem],
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
            prompt, extra = await build_prompt_for_repobench_mode(item, mode, answer_func)
            answer_started = time.perf_counter()
            if dry_run:
                answer = item.next_line
            else:
                answer, _meta = answer_func(prompt)
            latency_ms = int((time.perf_counter() - answer_started) * 1000)
            after_prompt = getattr(answer_func, "prompt_tokens", 0)
            after_output = getattr(answer_func, "output_tokens", 0)
            after_calls = getattr(answer_func, "calls", 0)
            provider_prompt = after_prompt - before_prompt
            provider_output = after_output - before_output
            rows.append(
                {
                    "case_id": item.case_id,
                    "repo_name": item.repo_name,
                    "file_path": item.file_path,
                    "level": item.level,
                    "mode": mode,
                    "correct": code_line_matches(answer, item.next_line),
                    "gold": item.next_line,
                    "answer_preview": answer[:500],
                    "prompt_tokens_estimate": estimate_tokens(prompt),
                    "provider_prompt_tokens": provider_prompt,
                    "provider_output_tokens": provider_output,
                    "provider_total_tokens": provider_prompt + provider_output,
                    "provider_calls": after_calls - before_calls,
                    "latency_ms": latency_ms,
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
        row["prompt_tokens_delta_vs_raw_estimate"] = raw_estimate - prompt_estimate
        row["context_window_saved_pct_vs_raw_estimate"] = round(
            ((raw_estimate - prompt_estimate) / max(1, raw_estimate)) * 100,
            2,
        )
        raw_provider_total = int(raw_row.get("provider_total_tokens") or 0)
        provider_total = int(row.get("provider_total_tokens") or 0)
        if raw_provider_total and provider_total:
            row["provider_total_delta_vs_raw"] = raw_provider_total - provider_total
            row["provider_total_saved_pct_vs_raw"] = round(
                ((raw_provider_total - provider_total) / raw_provider_total) * 100,
                2,
            )

    by_mode: dict[str, dict[str, Any]] = {}
    raw_provider_total_by_mode = 0
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        correct = sum(1 for row in mode_rows if row["correct"])
        raw_prompt_sum = sum(
            int(raw_by_case[row["case_id"]]["prompt_tokens_estimate"])
            for row in mode_rows
            if row["case_id"] in raw_by_case
        )
        context_delta = sum(int(row.get("prompt_tokens_delta_vs_raw_estimate") or 0) for row in mode_rows)
        provider_total = sum(int(row.get("provider_total_tokens") or 0) for row in mode_rows)
        if mode == "raw":
            raw_provider_total_by_mode = provider_total
        summary_row = {
            "cases": len(mode_rows),
            "accuracy": correct / len(mode_rows) if mode_rows else 0.0,
            "correct": correct,
            "prompt_tokens_estimate": sum(int(row["prompt_tokens_estimate"]) for row in mode_rows),
            "context_window_saved_pct_vs_raw_estimate": round(
                (context_delta / max(1, raw_prompt_sum)) * 100,
                2,
            ),
            "provider_prompt_tokens": sum(int(row.get("provider_prompt_tokens") or 0) for row in mode_rows),
            "provider_output_tokens": sum(int(row.get("provider_output_tokens") or 0) for row in mode_rows),
            "provider_total_tokens": provider_total,
            "provider_calls": sum(int(row.get("provider_calls") or 0) for row in mode_rows),
        }
        if mode in {"retrieved_snippets", "retrieved_snippets_or_fallback"}:
            retrieved_rows = [row for row in mode_rows if "retrieved_hit_gold" in row]
            hits = sum(1 for row in retrieved_rows if row.get("retrieved_hit_gold"))
            candidates = sum(1 for row in retrieved_rows if row.get("candidate_found"))
            ambiguous = sum(1 for row in retrieved_rows if row.get("candidate_ambiguous"))
            authorized = sum(1 for row in retrieved_rows if row.get("candidate_authorized"))
            fallbacks = sum(1 for row in retrieved_rows if row.get("fallback_raw"))
            summary_row["retrieved_hit_gold"] = hits
            summary_row["retrieved_hit_rate"] = hits / len(retrieved_rows) if retrieved_rows else 0.0
            summary_row["candidate_found_count"] = candidates
            summary_row["candidate_found_rate"] = candidates / len(retrieved_rows) if retrieved_rows else 0.0
            summary_row["candidate_ambiguous_count"] = ambiguous
            summary_row["candidate_authorized_count"] = authorized
            summary_row["candidate_authorized_rate"] = (
                authorized / len(retrieved_rows) if retrieved_rows else 0.0
            )
            summary_row["fallback_count"] = fallbacks
            summary_row["fallback_rate"] = fallbacks / len(retrieved_rows) if retrieved_rows else 0.0
        by_mode[mode] = summary_row

    if raw_provider_total_by_mode:
        for mode in modes:
            provider_total = by_mode[mode]["provider_total_tokens"]
            if provider_total:
                by_mode[mode]["provider_total_saved_pct_vs_raw"] = round(
                    ((raw_provider_total_by_mode - provider_total) / raw_provider_total_by_mode)
                    * 100,
                    2,
                )

    return {
        "schema": "remy_repobench_eval_v1",
        "summary": {
            "model": model,
            "dry_run": dry_run,
            "cases": len(items),
            "modes": modes,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "by_mode": by_mode,
            "retrieval_gate_diagnostics": compute_retrieval_gate_diagnostics(rows),
        },
        "rows": rows,
    }


def _parse_repobench_modes(value: str) -> list[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [mode for mode in modes if mode not in REPOBENCH_MODES]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown RepoBench mode(s): {', '.join(unknown)}")
    if not modes:
        raise argparse.ArgumentTypeError("At least one mode is required")
    return modes


def _write_report(report: dict[str, Any]) -> Path:
    from remy.config.settings import settings

    out_dir = Path(settings.DATA_DIR) / "llm_optimization"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"repobench_eval_{stamp}_{time.time_ns()}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


async def _main_async(args: argparse.Namespace) -> int:
    items = load_repobench_items(args.repobench, case_limit=args.case_limit)
    target_mode = args.target_mode if args.target_mode in args.mode else args.mode[-1]
    report = await run_repobench_eval(
        items,
        modes=args.mode,
        model=args.model,
        dry_run=args.dry_run,
        delay_sec=args.delay_sec,
    )
    path = _write_report(report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {path}")
    if target_mode != args.target_mode:
        print(f"Target mode {args.target_mode!r} was not run; using {target_mode!r}.")
    target = report["summary"]["by_mode"].get(target_mode, {})
    passed = float(target.get("accuracy") or 0.0) >= args.min_accuracy
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repobench", type=Path, default=DEFAULT_REPOBENCH)
    parser.add_argument("--mode", type=_parse_repobench_modes, default=["raw", "projected_hybrid"])
    parser.add_argument("--target-mode", default="projected_hybrid")
    parser.add_argument("--model", default="gemini-flash-lite-latest")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay-sec", type=float, default=0.0)
    parser.add_argument("--case-limit", type=int, default=10)
    parser.add_argument("--min-accuracy", type=float, default=0.50)
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
