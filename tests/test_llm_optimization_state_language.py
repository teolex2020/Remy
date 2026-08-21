import asyncio


def test_state_language_renderers_preserve_exact_facts_and_reduce_tokens():
    from remy.core.session_state_wrapper import SessionState, estimate_tokens
    from tools.validation.llm_optimization_state_language import (
        render_state_json,
        render_state_kv,
        render_state_symbolic,
        render_state_verbose,
    )

    state = SessionState(
        session_id="demo",
        snapshot=(
            "[CURRENT_GOAL]\n"
            "Prepare the customer support workflow.\n"
            "[DO_NOT_FORGET]\n"
            "Order ORD-9342 payout date 2026-07-12."
        ),
        pinned_facts=["ORD-9342", "2026-07-12"],
    )
    log = [{"type": "user_text", "text": "Remember order ORD-9342."}]

    verbose = render_state_verbose(state, log)
    as_json = render_state_json(state, log)
    kv = render_state_kv(state, log)
    symbolic = render_state_symbolic(state, log)

    for rendered in (as_json, kv, symbolic):
        assert "ORD-9342" in rendered
        assert "2026-07-12" in rendered

    assert estimate_tokens(kv) <= estimate_tokens(verbose)
    assert estimate_tokens(symbolic) <= estimate_tokens(verbose)


def test_state_language_eval_dry_run_reports_savings():
    from tools.validation.llm_optimization_corpus_eval import (
        DEFAULT_CASES,
        append_noise_turns,
        load_cases,
    )
    from tools.validation.llm_optimization_state_language import run_state_language_eval

    cases = append_noise_turns(load_cases(DEFAULT_CASES)[:3], 12)
    report = asyncio.run(
        run_state_language_eval(
            cases,
            modes=["raw", "state_verbose", "state_kv", "state_symbolic"],
            model="dry-model",
            dry_run=True,
            delay_sec=0.0,
        )
    )

    assert report["schema"] == "remy_state_language_eval_v1"
    assert report["summary"]["by_mode"]["raw"]["accuracy"] == 1.0
    assert report["summary"]["by_mode"]["state_kv"]["accuracy"] == 1.0
    assert report["summary"]["by_mode"]["state_kv"]["context_window_saved_pct_vs_raw_estimate"] > 0


def test_state_language_hybrid_records_decision_setup_cost():
    from tools.validation.llm_optimization_corpus_eval import DEFAULT_CASES, load_cases
    from tools.validation.llm_optimization_state_language import build_state_language_prompt

    class FakeMeter:
        def __init__(self):
            self.calls = 0
            self.prompt_tokens = 0
            self.output_tokens = 0
            self.prompts = []

        def __call__(self, prompt):
            self.prompts.append(prompt)
            self.calls += 1
            self.prompt_tokens += 11
            self.output_tokens += 7
            return (
                "- decided we go with Postgres because data is highly relational",
                {"usage_metadata": {"prompt_tokens": 11, "output_tokens": 7}},
            )

    fake_llm = FakeMeter()

    case = next(item for item in load_cases(DEFAULT_CASES) if item.case_id == "reasoning_decision_rationale")
    prompt, extra = build_state_language_prompt(case, "state_symbolic_hybrid", fake_llm)

    assert fake_llm.prompts
    assert "Postgres" in prompt
    assert "relational" in prompt
    assert extra["state_build_strategy"] == "hybrid"
    assert extra["decision_extract_calls"] >= 1
    assert extra["decision_extract_total_tokens"] > 0


def test_state_language_summary_reports_amortized_savings():
    from tools.validation.llm_optimization_state_language import run_state_language_eval
    from tools.validation.llm_optimization_corpus_eval import DEFAULT_CASES, append_noise_turns, load_cases

    cases = append_noise_turns(load_cases(DEFAULT_CASES)[:2], 8)
    report = asyncio.run(
        run_state_language_eval(
            cases,
            modes=["raw", "state_symbolic_hybrid"],
            model="dry-model",
            dry_run=True,
            delay_sec=0.0,
        )
    )

    summary = report["summary"]["by_mode"]["state_symbolic_hybrid"]
    assert "final_answer_provider_total_tokens" in summary
    assert "provider_setup_tokens" in summary
