import asyncio
import json


def _sample_locomo_payload():
    return [
        {
            "sample_id": "sample-a",
            "conversation": {
                "session_1_date_time": "2026-07-01",
                "session_1": [
                    {"speaker": "Caroline", "dia_id": "D1:1", "text": "I moved from Sweden."},
                    {"speaker": "Melanie", "dia_id": "D1:2", "text": "I painted a sunrise in 2022."},
                ],
            },
            "qa": [
                {
                    "question": "Where did Caroline move from?",
                    "answer": "Sweden",
                    "evidence": ["D1:1"],
                    "category": 1,
                },
                {
                    "question": "What did Melanie paint?",
                    "answer": "sunrise",
                    "evidence": ["D1:2"],
                    "category": 1,
                },
            ],
        }
    ]


def test_load_locomo_conversations_groups_qa_by_sample(tmp_path):
    from tools.validation.llm_optimization_locomo_notation import load_locomo_conversations

    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(_sample_locomo_payload()), encoding="utf-8")

    conversations = load_locomo_conversations(path, qa_limit_per_conversation=1)

    assert len(conversations) == 1
    assert conversations[0].sample_id == "sample-a"
    assert len(conversations[0].qa_items) == 1
    assert conversations[0].session_log[0]["text"] == "[2026-07-01] Caroline: I moved from Sweden."


def test_notation_prompt_and_judge_parser_are_strict():
    from tools.validation.llm_optimization_locomo_notation import (
        answer_is_unknown,
        build_answer_prompt,
        build_notation_prompt,
        parse_yes_no_judgment,
    )

    notation_prompt = build_notation_prompt("Caroline: I moved from Sweden.")
    answer_prompt = build_answer_prompt("Caroline| origin=Sweden", "Where from?", context_kind="notation")

    assert "Return notation only" in notation_prompt
    assert "Person.event|" in notation_prompt
    assert "CONTEXT_KIND: notation" in answer_prompt
    assert parse_yes_no_judgment("YES\nBecause equivalent") is True
    assert parse_yes_no_judgment("NO, missing date") is False
    assert parse_yes_no_judgment("") is False
    assert answer_is_unknown("UNKNOWN") is True
    assert answer_is_unknown("Unknown - not in context") is True
    assert answer_is_unknown("The answer is Sweden") is False


def test_amortized_saving_counts_setup_once():
    from tools.validation.llm_optimization_locomo_notation import amortized_saving_pct

    assert amortized_saving_pct(
        raw_total_tokens=1000,
        notation_answer_tokens=200,
        notation_setup_tokens=300,
        repeat_multiplier=1,
    ) == 50.0
    assert amortized_saving_pct(
        raw_total_tokens=1000,
        notation_answer_tokens=200,
        notation_setup_tokens=300,
        repeat_multiplier=2,
    ) == 65.0


def test_locomo_notation_dry_run_report(tmp_path):
    from tools.validation.llm_optimization_locomo_notation import (
        load_locomo_conversations,
        run_locomo_notation_eval,
    )

    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(_sample_locomo_payload()), encoding="utf-8")
    conversations = load_locomo_conversations(path)

    report = asyncio.run(
        run_locomo_notation_eval(
            conversations,
            model="dry-model",
            judge="exact",
            dry_run=True,
            delay_sec=0.0,
        )
    )

    summary = report["summary"]
    assert report["schema"] == "remy_locomo_notation_eval_v1"
    assert summary["raw_accuracy"] == 1.0
    assert summary["notation_accuracy"] == 1.0
    assert summary["notation_setup_provider_total_tokens"] == 0
    assert summary["fallback_on_unknown_cases"] == 0
    assert summary["fallback_on_unknown_accuracy"] == 1.0
    assert "1" in summary["amortized_product_saved_pct_vs_raw"]
