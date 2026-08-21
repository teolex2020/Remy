import asyncio
import json


def test_locomo_loader_and_dry_eval(tmp_path):
    from remy.config.settings import settings
    from tools.validation.llm_optimization_locomo import (
        load_locomo_items,
        run_locomo_eval,
    )

    locomo = tmp_path / "locomo.json"
    locomo.write_text(
        json.dumps(
            [
                {
                    "sample_id": "sample-a",
                    "conversation": {
                        "speaker_a": "A",
                        "speaker_b": "B",
                        "session_1_date_time": "2026-07-01",
                        "session_1": [
                            {"speaker": "A", "dia_id": "D1:1", "text": "I moved from Sweden."},
                            {"speaker": "B", "dia_id": "D1:2", "text": "Noted."},
                        ],
                    },
                    "qa": [
                        {
                            "question": "Where did A move from?",
                            "answer": "Sweden",
                            "evidence": ["D1:1"],
                            "category": 1,
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    items = load_locomo_items(locomo)
    assert len(items) == 1
    assert items[0].evidence_log[0]["text"] == "[2026-07-01] A: I moved from Sweden."

    report = asyncio.run(
        run_locomo_eval(
            items,
            modes=["raw", "evidence_oracle"],
            model="dry-model",
            dry_run=True,
            delay_sec=0.0,
        )
    )

    assert report["schema"] == "remy_locomo_eval_v1"
    assert report["summary"]["by_mode"]["raw"]["accuracy"] == 1.0
    assert report["summary"]["by_mode"]["evidence_oracle"]["accuracy"] == 1.0


def test_locomo_answer_match_allows_partial_list_answers():
    from tools.validation.llm_optimization_locomo import answer_matches

    assert answer_matches("She likes pottery, painting, and camping.", "pottery, camping, painting, swimming")
    assert not answer_matches("She likes pottery.", "pottery, camping, painting, swimming")
