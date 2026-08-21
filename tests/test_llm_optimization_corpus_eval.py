import asyncio
import json

import pytest


def test_load_cases_and_validate_schema(tmp_path):
    from tools.validation.llm_optimization_corpus_eval import load_cases

    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "case-1",
                "category": "exact_facts",
                "risk": "low",
                "messages": [
                    {"role": "user", "content": "Critical code RX-4471."},
                    {"role": "assistant", "content": "Stored."},
                ],
                "question": "What code?",
                "expected_fragments": ["RX-4471"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_cases(path)

    assert len(cases) == 1
    assert cases[0].case_id == "case-1"
    assert cases[0].messages[0].role == "user"


def test_default_corpus_is_versionable_and_product_relevant():
    from tools.validation.llm_optimization_corpus_eval import DEFAULT_CASES, load_cases

    cases = load_cases(DEFAULT_CASES)
    ids = [case.case_id for case in cases]
    categories = {case.category for case in cases}
    risks = {case.risk for case in cases}
    category_counts = {
        category: sum(1 for case in cases if case.category == category)
        for category in categories
    }

    assert len(cases) >= 30
    assert len(ids) == len(set(ids))
    assert {
        "exact_facts",
        "support_chat",
        "health_notes",
        "coding_agent_session",
        "reasoning",
        "negative_case",
    } <= categories
    assert category_counts["reasoning"] >= 8
    assert category_counts["negative_case"] >= 3
    assert "high" in risks
    assert all(case.expected_fragments for case in cases)
    assert all(not any("@" in fragment and "example.com" not in fragment for fragment in case.expected_fragments) for case in cases)


def test_load_cases_reports_line_number_for_bad_json(tmp_path):
    from tools.validation.llm_optimization_corpus_eval import load_cases

    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "ok"}\n{"broken": \n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"bad\.jsonl:1:|bad\.jsonl:2:"):
        load_cases(path)


def test_parse_case_rejects_empty_expected_fragments():
    from tools.validation.llm_optimization_corpus_eval import parse_case

    with pytest.raises(ValueError, match="expected_fragments"):
        parse_case(
            {
                "id": "bad",
                "category": "exact_facts",
                "risk": "low",
                "messages": [{"role": "user", "content": "hello"}],
                "question": "What?",
                "expected_fragments": [],
            }
        )


def test_parse_case_rejects_unknown_message_role():
    from tools.validation.llm_optimization_corpus_eval import parse_case

    with pytest.raises(ValueError, match="unsupported role"):
        parse_case(
            {
                "id": "bad-role",
                "category": "exact_facts",
                "risk": "low",
                "messages": [{"role": "tool", "content": "hidden"}],
                "question": "What?",
                "expected_fragments": ["hidden"],
            }
        )


def test_check_answer_requires_expected_and_rejects_forbidden():
    from tools.validation.llm_optimization_corpus_eval import check_answer

    ok = check_answer("The code is RX-4471.", ["RX-4471"], ["RX-0000"])
    ok_any = check_answer(
        "Prioritize brevity and clarity.",
        [],
        [],
        [["terse", "brevity", "brief", "concise"]],
    )
    bad_any = check_answer(
        "Use an operational tone.",
        [],
        [],
        [["terse", "brevity", "brief", "concise"]],
    )
    bad_missing = check_answer("The code is unknown.", ["RX-4471"], [])
    bad_forbidden = check_answer("The code is RX-4471 and RX-0000.", ["RX-4471"], ["RX-0000"])
    negated_forbidden = check_answer("The code is RX-4471, not RX-0000.", ["RX-4471"], ["RX-0000"])

    assert ok["correct"] is True
    assert ok_any["correct"] is True
    assert bad_any["correct"] is False
    assert bad_any["missing_any"] == [["terse", "brevity", "brief", "concise"]]
    assert bad_missing["correct"] is False
    assert bad_missing["missing"] == ["RX-4471"]
    assert bad_forbidden["correct"] is False
    assert bad_forbidden["forbidden"] == ["RX-0000"]
    assert negated_forbidden["correct"] is True


def test_append_noise_turns_tracks_conversation_request_count():
    from tools.validation.llm_optimization_corpus_eval import (
        CorpusCase,
        CorpusMessage,
        append_noise_turns,
        conversation_request_count,
    )

    case = CorpusCase(
        case_id="noise",
        category="exact_facts",
        risk="low",
        messages=[
            CorpusMessage("user", "Critical fact: code is RX-4471."),
            CorpusMessage("assistant", "Stored."),
        ],
        question="What code?",
        expected_fragments=["RX-4471"],
    )

    expanded = append_noise_turns([case], 3)[0]

    assert conversation_request_count(case) == 2
    assert conversation_request_count(expanded) == 5
    assert len(expanded.messages) == len(case.messages) + 6


def test_append_noise_turns_does_not_mutate_original_case():
    from tools.validation.llm_optimization_corpus_eval import (
        CorpusCase,
        CorpusMessage,
        append_noise_turns,
    )

    case = CorpusCase(
        case_id="immutable",
        category="exact_facts",
        risk="low",
        messages=[CorpusMessage("user", "Critical fact: code is RX-4471.")],
        question="What code?",
        expected_fragments=["RX-4471"],
    )

    expanded = append_noise_turns([case], 2)[0]

    assert len(case.messages) == 1
    assert len(expanded.messages) == 5
    assert expanded is not case


def test_parse_modes_accepts_all_and_rejects_unknown():
    from tools.validation.llm_optimization_corpus_eval import CANONICAL_MODES, _parse_modes

    assert _parse_modes("all") == list(CANONICAL_MODES)
    assert _parse_modes("raw,projected") == ["raw", "projected"]
    assert _parse_modes("raw,projected_facts,projected_hybrid") == [
        "raw",
        "projected_facts",
        "projected_hybrid",
    ]
    with pytest.raises(Exception, match="Unknown mode"):
        _parse_modes("raw,unknown")


def test_projected_prompt_keeps_exact_facts_and_can_be_smaller():
    from tools.validation.llm_optimization_corpus_eval import (
        CorpusCase,
        CorpusMessage,
        build_prompt_for_mode,
    )
    from remy.core.session_state_wrapper import estimate_tokens

    messages = [
        CorpusMessage("user", "Critical fact: authorization code is RX-4471."),
        CorpusMessage("assistant", "Stored."),
    ]
    for index in range(80):
        messages.append(CorpusMessage("user", f"Routine unrelated planning note {index}."))
        messages.append(CorpusMessage("assistant", "Acknowledged."))

    case = CorpusCase(
        case_id="long",
        category="exact_facts",
        risk="medium",
        messages=messages,
        question="What is the authorization code?",
        expected_fragments=["RX-4471"],
    )

    raw_prompt, _ = asyncio.run(build_prompt_for_mode(case, "raw"))
    projected_prompt, extra = asyncio.run(build_prompt_for_mode(case, "projected"))

    assert "RX-4471" in projected_prompt
    assert extra["pinned_count"] >= 1
    assert estimate_tokens(projected_prompt) < estimate_tokens(raw_prompt)


def test_projected_facts_and_hybrid_are_separate_cost_modes():
    from tools.validation.llm_optimization_corpus_eval import (
        CorpusCase,
        CorpusMessage,
        build_prompt_for_mode,
    )

    class FakeDecisionMeter:
        def __init__(self):
            self.calls = 0
            self.prompt_tokens = 0
            self.output_tokens = 0

        def __call__(self, prompt: str):
            self.calls += 1
            self.prompt_tokens += 100
            self.output_tokens += 20
            return "- prefer terse status reports\n", {"usage_metadata": {}}

    case = CorpusCase(
        case_id="decision-split",
        category="reasoning",
        risk="medium",
        messages=[
            CorpusMessage("user", "Decision: for this client prefer terse status reports."),
            CorpusMessage("assistant", "Stored."),
        ],
        question="What reporting style should be used?",
        expected_fragments=["terse status reports"],
    )
    meter = FakeDecisionMeter()

    facts_prompt, facts_extra = asyncio.run(
        build_prompt_for_mode(case, "projected_facts", meter)
    )
    hybrid_prompt, hybrid_extra = asyncio.run(
        build_prompt_for_mode(case, "projected_hybrid", meter)
    )

    assert facts_extra["projection_strategy"] == "facts"
    assert facts_extra["decision_extract_calls"] == 0
    assert "[DECISIONS_AND_PREFERENCES]" not in facts_prompt
    assert hybrid_extra["projection_strategy"] == "hybrid"
    assert hybrid_extra["decision_extract_calls"] == 1
    assert hybrid_extra["decision_extract_total_tokens"] == 120
    assert "[DECISIONS_AND_PREFERENCES]" in hybrid_prompt


def test_dry_run_reports_all_modes_without_provider_calls():
    from tools.validation.llm_optimization_corpus_eval import (
        CorpusCase,
        CorpusMessage,
        run_eval,
    )

    case = CorpusCase(
        case_id="dry",
        category="exact_facts",
        risk="low",
        messages=[
            CorpusMessage("user", "Critical fact: code is RX-4471."),
            CorpusMessage("assistant", "Stored."),
        ],
        question="What code?",
        expected_fragments=["RX-4471"],
    )

    report = asyncio.run(
        run_eval(
            [case],
            modes=["raw", "projected"],
            model="dry",
            dry_run=True,
            delay_sec=0,
            fold_batch_size=10,
            context_window_tokens=128000,
        )
    )

    assert report["summary"]["by_mode"]["raw"]["accuracy"] == 1.0
    assert report["summary"]["by_mode"]["projected"]["accuracy"] == 1.0
    assert report["summary"]["by_mode"]["raw"]["provider_calls"] == 0
    assert "provider_total_tokens" in report["summary"]["by_mode"]["projected"]
    assert report["summary"]["context_window_tokens"] == 128000
    assert "context_window_saved_pct_vs_raw_estimate" in report["summary"]["by_mode"]["projected"]
    assert "effectiveness_rate" in report["summary"]["by_mode"]["projected"]
    assert "accuracy_weighted_context_saving_pct" in report["summary"]["by_mode"]["projected"]


def test_dry_run_all_modes_includes_incremental_setup_cost():
    from tools.validation.llm_optimization_corpus_eval import (
        CorpusCase,
        CorpusMessage,
        run_eval,
    )

    messages = [
        CorpusMessage("user", "Critical fact: authorization code is RX-4471."),
        CorpusMessage("assistant", "Stored."),
    ]
    for index in range(30):
        messages.append(CorpusMessage("user", f"Routine unrelated planning note {index}."))
        messages.append(CorpusMessage("assistant", "Acknowledged."))

    case = CorpusCase(
        case_id="incremental",
        category="exact_facts",
        risk="medium",
        messages=messages,
        question="What is the authorization code?",
        expected_fragments=["RX-4471"],
    )

    report = asyncio.run(
        run_eval(
            [case],
            modes=["raw", "incremental"],
            model="dry",
            dry_run=True,
            delay_sec=0,
            fold_batch_size=5,
            context_window_tokens=128000,
        )
    )
    incremental_row = next(row for row in report["rows"] if row["mode"] == "incremental")

    assert incremental_row["state_update_calls"] > 1
    assert incremental_row["state_update_total_tokens"] > 0
    assert incremental_row["break_even_request_count_estimate"] is not None
    assert incremental_row["break_even_request_count_estimate"] >= 1


def test_context_window_metrics_do_not_crash_with_zero_limit():
    from tools.validation.llm_optimization_corpus_eval import (
        CorpusCase,
        CorpusMessage,
        run_eval,
    )

    case = CorpusCase(
        case_id="zero-window",
        category="exact_facts",
        risk="low",
        messages=[CorpusMessage("user", "Critical fact: code is RX-4471.")],
        question="What code?",
        expected_fragments=["RX-4471"],
    )

    report = asyncio.run(
        run_eval(
            [case],
            modes=["raw", "projected"],
            model="dry",
            dry_run=True,
            delay_sec=0,
            fold_batch_size=10,
            context_window_tokens=0,
        )
    )
    raw_row = next(row for row in report["rows"] if row["mode"] == "raw")

    assert raw_row["raw_context_window_used_pct_estimate"] == 0.0
    assert raw_row["context_window_used_pct_estimate"] == 0.0


def test_break_even_and_context_window_metrics_are_reported():
    from tools.validation.llm_optimization_corpus_eval import (
        CorpusCase,
        CorpusMessage,
        run_eval,
    )

    messages = [
        CorpusMessage("user", "Critical fact: authorization code is RX-4471."),
        CorpusMessage("assistant", "Stored."),
    ]
    for index in range(80):
        messages.append(CorpusMessage("user", f"Routine unrelated planning note {index}."))
        messages.append(CorpusMessage("assistant", "Acknowledged."))

    case = CorpusCase(
        case_id="long",
        category="exact_facts",
        risk="medium",
        messages=messages,
        question="What is the authorization code?",
        expected_fragments=["RX-4471"],
    )

    report = asyncio.run(
        run_eval(
            [case],
            modes=["raw", "projected"],
            model="dry",
            dry_run=True,
            delay_sec=0,
            fold_batch_size=10,
            context_window_tokens=1000,
        )
    )
    projected_row = next(row for row in report["rows"] if row["mode"] == "projected")

    assert projected_row["prompt_tokens_delta_vs_raw_estimate"] > 0
    assert projected_row["context_window_saved_tokens_estimate"] > 0
    assert projected_row["context_window_saved_pct_vs_raw_estimate"] > 0
    assert projected_row["optimization_effective"] is True
    assert projected_row["efficiency_score_estimate"] > 0
    assert projected_row["break_even_request_count_estimate"] == 1
    assert report["summary"]["by_mode"]["projected"]["profitable_cases_estimate"] == 1
    assert report["summary"]["by_mode"]["projected"]["effective_cases"] == 1
    assert report["summary"]["by_mode"]["projected"]["effectiveness_rate"] == 1.0
    assert report["summary"]["by_mode"]["projected"]["accuracy_weighted_context_saving_pct"] > 0


def test_short_context_is_not_marked_effective_even_when_correct():
    from tools.validation.llm_optimization_corpus_eval import (
        CorpusCase,
        CorpusMessage,
        run_eval,
    )

    case = CorpusCase(
        case_id="short",
        category="negative_case",
        risk="low",
        messages=[
            CorpusMessage("user", "Critical fact: code is RX-4471."),
            CorpusMessage("assistant", "Stored."),
        ],
        question="What code?",
        expected_fragments=["RX-4471"],
    )

    report = asyncio.run(
        run_eval(
            [case],
            modes=["raw", "projected"],
            model="dry",
            dry_run=True,
            delay_sec=0,
            fold_batch_size=10,
            context_window_tokens=128000,
        )
    )
    projected_row = next(row for row in report["rows"] if row["mode"] == "projected")

    assert projected_row["correct"] is True
    assert projected_row["optimization_effective"] is False
    assert projected_row["efficiency_score_estimate"] == 0.0
    assert report["summary"]["by_mode"]["projected"]["effectiveness_rate"] == 0.0
