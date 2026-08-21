import asyncio
import json


def _fixture_row():
    return {
        "repo_name": "demo/repo",
        "file_path": "app/main.py",
        "context": [
            {
                "identifier": "build_message",
                "path": "app/helpers.py",
                "snippet": "def build_message(name):\n    return f'Hello {name}'",
            },
            {
                "identifier": "unused",
                "path": "app/unused.py",
                "snippet": "def unused():\n    return None",
            },
        ],
        "import_statement": "from app.helpers import build_message",
        "cropped_code": "def greet(name):\n    ",
        "next_line": "return build_message(name)",
        "gold_snippet_index": 0,
        "level": "2k",
    }


def test_repobench_loader_and_dry_eval(tmp_path):
    from tools.validation.llm_optimization_repobench import (
        load_repobench_items,
        run_repobench_eval,
    )

    path = tmp_path / "repobench.jsonl"
    path.write_text(json.dumps(_fixture_row()) + "\n", encoding="utf-8")

    items = load_repobench_items(path)
    assert len(items) == 1
    assert items[0].context[0].path == "app/helpers.py"
    assert items[0].next_line == "return build_message(name)"

    report = asyncio.run(
        run_repobench_eval(
            items,
            modes=[
                "raw",
                "cropped_only",
                "retrieved_snippets",
                "retrieved_snippets_or_fallback",
                "gold_snippet_oracle",
            ],
            model="dry-model",
            dry_run=True,
            delay_sec=0.0,
        )
    )

    assert report["schema"] == "remy_repobench_eval_v1"
    assert report["summary"]["by_mode"]["raw"]["accuracy"] == 1.0
    assert report["summary"]["by_mode"]["retrieved_snippets"]["accuracy"] == 1.0
    assert report["summary"]["by_mode"]["gold_snippet_oracle"]["accuracy"] == 1.0
    retrieved_row = next(row for row in report["rows"] if row["mode"] == "retrieved_snippets")
    assert retrieved_row["retrieved_hit_gold"] is True
    assert retrieved_row["retrieved_snippet_indexes"] == [0]
    assert retrieved_row["authority_decision"] == "Go"
    assert retrieved_row["candidate_authorized"] is True
    fallback_row = next(row for row in report["rows"] if row["mode"] == "retrieved_snippets_or_fallback")
    assert fallback_row["fallback_raw"] is False
    assert report["summary"]["by_mode"]["retrieved_snippets_or_fallback"]["candidate_authorized_rate"] == 1.0
    assert (
        report["summary"]["by_mode"]["gold_snippet_oracle"][
            "context_window_saved_pct_vs_raw_estimate"
        ]
        > 0
    )


def test_repobench_code_line_matcher_normalizes_markdown_and_spacing():
    from tools.validation.llm_optimization_repobench import code_line_matches

    assert code_line_matches("```python\nreturn  build_message(name)\n```", "return build_message(name)")
    assert not code_line_matches("return other(name)", "return build_message(name)")


def test_repobench_retrieval_scores_identifier_and_import_overlap():
    from tools.validation.llm_optimization_repobench import (
        load_repobench_items,
        retrieved_snippets,
    )

    row = _fixture_row()
    item = load_repobench_items_from_rows_for_test([row])[0]
    selected = retrieved_snippets(item, limit=1)
    assert selected[0].identifier == "build_message"


def test_repobench_retrieval_authority_falls_back_on_ambiguous_candidate():
    from tools.validation.llm_optimization_repobench import (
        retrieved_snippets,
        review_retrieved_snippet_authority,
    )

    row = _fixture_row()
    row["context"] = [
        {
            "identifier": "build_message",
            "path": "app/helpers_a.py",
            "snippet": "def build_message(name):\n    return name",
        },
        {
            "identifier": "build_message",
            "path": "app/helpers_b.py",
            "snippet": "def build_message(name):\n    return name.upper()",
        },
    ]
    item = load_repobench_items_from_rows_for_test([row])[0]
    selected = retrieved_snippets(item, limit=1)
    authority = review_retrieved_snippet_authority(item, selected)

    assert authority["decision"] == "Stop"
    assert authority["candidate_found"] is True
    assert authority["candidate_ambiguous"] is True
    assert "RetrievedCandidateAmbiguous" in authority["block_reasons"]


def test_repobench_gate_diagnostics_measure_wrong_retrieval_stops():
    from tools.validation.llm_optimization_repobench import compute_retrieval_gate_diagnostics

    rows = [
        {
            "case_id": "wrong-allowed",
            "mode": "retrieved_snippets",
            "correct": False,
        },
        {
            "case_id": "wrong-allowed",
            "mode": "retrieved_snippets_or_fallback",
            "correct": False,
            "authority_decision": "Go",
        },
        {
            "case_id": "wrong-stopped",
            "mode": "retrieved_snippets",
            "correct": False,
        },
        {
            "case_id": "wrong-stopped",
            "mode": "retrieved_snippets_or_fallback",
            "correct": True,
            "authority_decision": "Stop",
        },
        {
            "case_id": "correct-stopped",
            "mode": "retrieved_snippets",
            "correct": True,
        },
        {
            "case_id": "correct-stopped",
            "mode": "retrieved_snippets_or_fallback",
            "correct": True,
            "authority_decision": "Stop",
        },
    ]

    diagnostics = compute_retrieval_gate_diagnostics(rows)

    assert diagnostics["paired_cases"] == 3
    assert diagnostics["wrong_retrieval_count"] == 2
    assert diagnostics["wrong_retrieval_stopped_count"] == 1
    assert diagnostics["wrong_retrieval_go_count"] == 1
    assert diagnostics["correct_retrieval_stopped_count"] == 1
    assert diagnostics["stop_precision_wrong_retrieval"] == 0.5
    assert diagnostics["fallback_fixed_wrong_retrieval_count"] == 1


def load_repobench_items_from_rows_for_test(rows):
    from tools.validation.llm_optimization_repobench import _item_from_row

    return [_item_from_row(row, index) for index, row in enumerate(rows)]
