import pytest


def _report(
    *,
    accuracy=1.0,
    effectiveness=1.0,
    savings=25.0,
    provider_total_savings=None,
    noise=3,
    rows=None,
    mode="projected",
):
    target = {
        "accuracy": accuracy,
        "effectiveness_rate": effectiveness,
        "context_window_saved_pct_vs_raw_estimate": savings,
    }
    if provider_total_savings is not None:
        target["provider_total_saved_pct_vs_raw"] = provider_total_savings
    return {
        "summary": {
            "append_noise_turns": noise,
            "by_mode": {mode: target},
        },
        "rows": rows
        if rows is not None
        else [
            {
                "mode": mode,
                "case_id": "ok",
                "category": "exact_facts",
                "correct": True,
                "optimization_effective": True,
                "context_window_saved_pct_vs_raw_estimate": savings,
            }
        ],
    }


def _evaluate(report, **overrides):
    from tools.validation.llm_optimization_gate import evaluate

    params = {
        "min_accuracy": 0.95,
        "min_effectiveness": 0.80,
        "min_savings_pct": 20.0,
        "long_session_noise": 3,
        "min_provider_total_savings_pct": None,
        "target_mode": "projected",
    }
    params.update(overrides)
    return evaluate(report, **params)


def test_gate_passes_good_long_session_report():
    passed, failures, notes = _evaluate(_report())

    assert passed is True
    assert failures == []
    assert any("accuracy" in note for note in notes)
    assert any("context savings" in note for note in notes)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("accuracy", 0.90, "accuracy"),
        ("effectiveness", 0.50, "effectiveness_rate"),
        ("savings", 10.0, "context savings"),
    ],
)
def test_gate_fails_thresholds(field, value, expected):
    kwargs = {field: value}

    passed, failures, _notes = _evaluate(_report(**kwargs))

    assert passed is False
    assert any(expected in failure for failure in failures)


def test_gate_skips_savings_threshold_for_short_run():
    report = _report(savings=-5.0, noise=0)

    passed, failures, notes = _evaluate(report)

    assert passed is True
    assert failures == []
    assert any("savings gate skipped" in note for note in notes)


def test_gate_fails_false_savings_on_wrong_answer():
    report = _report(
        rows=[
            {
                "mode": "projected",
                "case_id": "wrong",
                "category": "reasoning",
                "correct": False,
                "optimization_effective": True,
                "context_window_saved_pct_vs_raw_estimate": 30.0,
            }
        ]
    )

    passed, failures, _notes = _evaluate(report)

    assert passed is False
    assert any("false-saving" in failure for failure in failures)


def test_gate_fails_short_negative_case_marked_effective():
    report = _report(
        noise=0,
        savings=0.0,
        rows=[
            {
                "mode": "projected",
                "case_id": "negative_short_exact_email",
                "category": "negative_case",
                "correct": True,
                "optimization_effective": True,
                "context_window_saved_pct_vs_raw_estimate": 12.0,
            }
        ],
    )

    passed, failures, _notes = _evaluate(report)

    assert passed is False
    assert any("short negative" in failure for failure in failures)


def test_gate_fails_missing_projected_mode():
    passed, failures, _notes = _evaluate({"summary": {"by_mode": {}}, "rows": []})

    assert passed is False
    assert any("projected" in failure for failure in failures)


def test_gate_can_target_projected_facts_mode():
    report = _report(mode="projected_facts", provider_total_savings=7.0)

    passed, failures, notes = _evaluate(
        report,
        target_mode="projected_facts",
        min_provider_total_savings_pct=0.0,
    )

    assert passed is True
    assert failures == []
    assert any("provider total-token savings" in note for note in notes)


def test_gate_passes_provider_total_token_savings_when_requested():
    report = _report(provider_total_savings=12.5)

    passed, failures, notes = _evaluate(report, min_provider_total_savings_pct=5.0)

    assert passed is True
    assert failures == []
    assert any("provider total-token savings" in note for note in notes)


def test_gate_fails_provider_total_token_savings_when_requested():
    report = _report(provider_total_savings=-8.0)

    passed, failures, _notes = _evaluate(report, min_provider_total_savings_pct=0.0)

    assert passed is False
    assert any("provider total-token savings" in failure for failure in failures)


def test_gate_skips_provider_total_token_savings_without_provider_tokens():
    report = _report(provider_total_savings=None)

    passed, failures, notes = _evaluate(report, min_provider_total_savings_pct=0.0)

    assert passed is True
    assert failures == []
    assert any("provider total-token savings gate skipped" in note for note in notes)
