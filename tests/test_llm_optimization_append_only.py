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
                    {"speaker": "Caroline", "dia_id": "D1:3", "text": "I researched adoption agencies."},
                    {"speaker": "Melanie", "dia_id": "D1:4", "text": "That sounds important."},
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
                    "question": "What did Caroline research?",
                    "answer": "adoption agencies",
                    "evidence": ["D1:3"],
                    "category": 1,
                },
            ],
        }
    ]


def test_select_relevant_chunks_keeps_matching_and_recent():
    from tools.validation.llm_optimization_append_only import AppendChunk, select_relevant_chunks

    chunks = [
        AppendChunk("a", "raw a", "Caroline origin Sweden", 4, 3, 10),
        AppendChunk("b", "raw b", "Melanie painted sunrise", 4, 3, 10),
        AppendChunk("c", "raw c", "Recent unrelated update", 4, 3, 10),
    ]

    selected = select_relevant_chunks(chunks, "Where is Caroline from?", top_k=1, include_recent=1)

    assert [chunk.chunk_id for chunk in selected] == ["a", "c"]


def test_economic_curve_reports_break_even_and_amortized_savings():
    from tools.validation.llm_optimization_append_only import _economic_curve

    curve = _economic_curve(
        raw_total=1000,
        optimized_query_total=500,
        setup_total=250,
        case_count=10,
    )

    assert curve["break_even_queries"] == 5.0
    assert curve["net_saved_pct_at_5_queries"] == 0.0
    assert curve["net_saved_pct_at_10_queries"] == 25.0


def test_append_only_dry_run_report(tmp_path):
    from tools.validation.llm_optimization_append_only import (
        load_append_conversations,
        run_append_only_eval,
    )

    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(_sample_locomo_payload()), encoding="utf-8")

    conversations, encoder_usage = asyncio.run(
        load_append_conversations(
            path,
            conversation_limit=1,
            session_limit=1,
            qa_limit_per_conversation=2,
            skip_adversarial=True,
            model="dry-model",
            dry_run=True,
            delay_sec=0.0,
        )
    )
    report = asyncio.run(
        run_append_only_eval(
            conversations,
            encoder_usage=encoder_usage,
            modes=["append_full", "append_retrieved"],
            model="dry-model",
            judge="exact",
            dry_run=True,
            delay_sec=0.0,
            retrieval_top_k=1,
        )
    )

    summary = report["summary"]
    assert report["schema"] == "remy_append_only_eval_v1"
    assert summary["conversations"] == 1
    assert summary["encoder"]["provider_total_tokens"] > 0
    assert summary["by_mode"]["append_full"]["optimized_accuracy"] == 1.0
    assert summary["by_mode"]["append_retrieved"]["selected_chunks_mean"] >= 1


def test_deterministic_encoder_has_no_provider_setup_cost(tmp_path):
    from tools.validation.llm_optimization_append_only import load_append_conversations

    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(_sample_locomo_payload()), encoding="utf-8")

    conversations, encoder_usage = asyncio.run(
        load_append_conversations(
            path,
            conversation_limit=1,
            session_limit=1,
            qa_limit_per_conversation=2,
            skip_adversarial=True,
            model="dry-model",
            dry_run=False,
            delay_sec=0.0,
            encoder_kind="deterministic",
        )
    )

    assert conversations
    assert encoder_usage["kind"] == "deterministic"
    assert encoder_usage["provider_total_tokens"] == 0
    assert all(chunk.provider_total_tokens == 0 for chunk in conversations[0].chunks)


def test_abbreviated_encoder_shortens_but_keeps_key_markers():
    from tools.validation.llm_optimization_append_only import (
        compact_exchange_abbreviated,
        compact_exchange_deterministic,
    )

    exchange = (
        "[7 May 2023] D1:3 Caroline: I researched adoption agencies before the workshop.\n"
        "[7 May 2023] D1:4 Melanie: That sounds important."
    )

    deterministic = compact_exchange_deterministic(exchange)
    abbreviated = compact_exchange_abbreviated(exchange)

    assert len(abbreviated) < len(deterministic)
    assert "Caroline" in abbreviated
    assert "rsch" in abbreviated
    assert "adpt" in abbreviated
    assert "d1.3" in abbreviated
