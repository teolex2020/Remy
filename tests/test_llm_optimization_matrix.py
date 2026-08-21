import asyncio
import json


def test_matrix_runner_writes_report_and_gates_dry_run(tmp_path, monkeypatch):
    from remy.config.settings import settings
    from tools.validation.llm_optimization_matrix import MatrixConfig, run_matrix

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    cases = tmp_path / "cases.jsonl"
    messages = [
        {"role": "user", "content": "Critical fact: authorization code is RX-4471."},
        {"role": "assistant", "content": "Stored."},
    ]
    cases.write_text(
        json.dumps(
            {
                "id": "matrix-case",
                "category": "exact_facts",
                "risk": "low",
                "messages": messages,
                "question": "What is the authorization code?",
                "expected_fragments": ["RX-4471"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report, path = asyncio.run(
        run_matrix(
            MatrixConfig(
                cases_path=cases,
                models=("dry-model",),
                modes=("raw", "projected_hybrid"),
                noise_levels=(3,),
                dry_run=True,
                min_savings_pct=0.0,
                min_provider_total_savings_pct=None,
            )
        )
    )

    assert report["schema"] == "remy_llm_optimization_matrix_v1"
    assert report["passed"] is True
    assert path.exists()
    assert len(report["runs"]) == 1
    run = report["runs"][0]
    assert run["model"] == "dry-model"
    assert run["passed"] is True
    assert run["eval_report"]
