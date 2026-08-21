import pytest


def test_extract_pinned_facts_supports_unicode_and_deduplicates():
    from remy.core.session_state_wrapper import extract_pinned_facts

    text = (
        "Patient code RX-4471, dose 2,5 \u043c\u0433 at 09:30, invoice "
        "1500 \u0433\u0440\u043d, "
        "email test@example.com, file E:\\remy\\workflow.txt, "
        "date 2026-07-01 14:30."
    )

    pins = extract_pinned_facts(text)

    assert "RX-4471" in pins
    assert "2,5 \u043c\u0433" in pins
    assert "09:30" in pins
    assert "1500 \u0433\u0440\u043d" in pins
    assert "test@example.com" in pins
    assert "E:\\remy\\workflow.txt" in pins
    assert "2026-07-01 14:30" in pins
    assert "2026-07-01" not in pins


def test_projected_state_preserves_evidence_lines_for_pin_meaning():
    from remy.core.session_state_wrapper import build_projected_state_from_log

    session_log = [
        {
            "type": "user_text",
            "text": (
                "Support case: order id ORD-9342. Refund status is approved, "
                "payout date 2026-07-12."
            ),
        },
        {"type": "model_response", "text": "Stored."},
        {
            "type": "user_text",
            "text": "Coding note: failing test is tests/test_pipeline_runner.py::test_router_fallback.",
        },
    ]

    state = build_projected_state_from_log(session_log)

    assert "ORD-9342" in state.pinned_facts
    assert "2026-07-12" in state.pinned_facts
    assert "tests/test_pipeline_runner.py::test_router_fallback" in state.pinned_facts
    assert "Refund status is approved" in state.snapshot
    assert "failing test is tests/test_pipeline_runner.py::test_router_fallback" in state.snapshot


def test_decision_extractor_is_marker_gated():
    from remy.core.session_state_wrapper import extract_decisions

    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        return "- should not be called"

    result = extract_decisions("Routine note: update the wiki index page.", fake_llm)

    assert result == ""
    assert calls == []

    result = extract_decisions("Routine status notes and non-critical operational context.", fake_llm)

    assert result == ""
    assert calls == []


def test_projected_state_can_include_decisions_and_preferences():
    from remy.core.session_state_wrapper import build_projected_state_from_log

    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        assert "Do NOT summarize" in prompt
        assert "Do not drop modifiers" in prompt
        return "- decided we go with Postgres because data is highly relational"

    session_log = [
        {
            "type": "user_text",
            "text": (
                "We compared Postgres and MongoDB. I decided we go with Postgres "
                "because our data is highly relational."
            ),
        },
        {"type": "model_response", "text": "Understood."},
        {"type": "user_text", "text": "Routine note: update the wiki index."},
    ]

    state = build_projected_state_from_log(session_log, decision_llm_func=fake_llm)

    assert len(calls) == 1
    assert "[DECISIONS_AND_PREFERENCES]" in state.snapshot
    assert "Postgres" in state.snapshot
    assert "highly relational" in state.snapshot


def test_projected_state_can_include_durable_status_reasons():
    from remy.core.session_state_wrapper import build_projected_state_from_log

    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        assert "customer/status/risk labels" in prompt
        return (
            "- customer Harbor Inc is marked churn-risk because the admin team "
            "lost reporting access after the SSO migration"
        )

    session_log = [
        {
            "type": "user_text",
            "text": (
                "CRM note: customer Harbor Inc is marked churn-risk because "
                "the admin team lost reporting access after the SSO migration."
            ),
        },
        {"type": "model_response", "text": "Stored the churn risk note."},
    ]

    state = build_projected_state_from_log(session_log, decision_llm_func=fake_llm)

    assert len(calls) == 1
    assert "Harbor Inc" in state.snapshot
    assert "lost reporting access" in state.snapshot
    assert "SSO migration" in state.snapshot


def test_projected_state_keeps_marked_source_lines_for_style_modifiers():
    from remy.core.session_state_wrapper import build_projected_state_from_log

    def fake_llm(_prompt):
        return "- Never use marketing language; start with impact, then current status"

    session_log = [
        {
            "type": "user_text",
            "text": (
                "I prefer terse operational updates. Never use marketing language "
                "in internal incident reports; start with impact, then current status."
            ),
        },
        {"type": "model_response", "text": "Stored the incident report style preference."},
        {
            "type": "user_text",
            "text": "Routine status notes and non-critical operational context.",
        },
    ]

    state = build_projected_state_from_log(session_log, decision_llm_func=fake_llm)

    assert "[DURABLE_SOURCE_LINES]" in state.snapshot
    assert "terse operational updates" in state.snapshot
    assert "Routine status notes" not in state.snapshot


def test_projected_state_keeps_facts_even_when_decision_extractor_returns_none():
    from remy.core.session_state_wrapper import build_projected_state_from_log

    def fake_llm(_prompt):
        return "NONE"

    session_log = [
        {
            "type": "user_text",
            "text": "I decided the release code remains RX-4471 because support already has it.",
        }
    ]

    state = build_projected_state_from_log(session_log, decision_llm_func=fake_llm)

    assert "RX-4471" in state.pinned_facts
    assert "RX-4471" in state.snapshot
    assert "[DECISIONS_AND_PREFERENCES]" not in state.snapshot


def test_extract_pinned_facts_keeps_multi_dash_codes_whole():
    from remy.core.session_state_wrapper import extract_pinned_facts

    pins = extract_pinned_facts("Invoice INV-2026-44 must be sent by 2026-07-20.")

    assert "INV-2026-44" in pins
    assert "INV-2026" not in pins


def test_state_persistence_uses_settings_data_dir(tmp_path, monkeypatch):
    from remy.config.settings import settings
    from remy.core.session_state_wrapper import SessionState, load_state, save_state

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

    state = SessionState(
        session_id="chat/session 1",
        snapshot="[CURRENT_GOAL]\nTest persistence.",
        pinned_facts=["RX-4471"],
        turns_folded=3,
    )
    save_state(state)

    loaded = load_state("chat/session 1")

    assert loaded.session_id == "chat/session 1"
    assert loaded.snapshot == state.snapshot
    assert loaded.pinned_facts == ["RX-4471"]
    assert loaded.turns_folded == 3
    assert (tmp_path / "session_state" / "chat_session_1.json").exists()


@pytest.mark.asyncio
async def test_update_state_incremental_skips_failed_exchange(tmp_path, monkeypatch):
    from remy.config.settings import settings
    from remy.core.session_state_wrapper import SessionState, update_state_incremental

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        return "should not be used"

    state = SessionState(session_id="failed")
    result = await update_state_incremental(
        state,
        "assistant: partial broken answer RX-4471",
        fake_llm,
        exchange_failed=True,
    )

    assert result.failure_count == 1
    assert result.snapshot == ""
    assert result.pinned_facts == []
    assert calls == []


@pytest.mark.asyncio
async def test_update_state_incremental_records_pins_and_metrics(tmp_path, monkeypatch):
    from remy.config.settings import settings
    from remy.core.session_state_wrapper import SessionState, update_state_incremental

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

    def fake_llm(prompt):
        assert "NEW EXCHANGE:" in prompt
        return (
            "[CURRENT_GOAL]\nRemember exact medical note.",
            {"usage_metadata": {"prompt_tokens": 20, "output_tokens": 7}},
        )

    state = SessionState(session_id="ok")
    result = await update_state_incremental(
        state,
        "user: Save dose 5 \u043c\u0433 and code RX-4471.",
        fake_llm,
    )

    assert result.turns_folded == 1
    assert "5 \u043c\u0433" in result.pinned_facts
    assert "RX-4471" in result.pinned_facts
    metrics = (tmp_path / "llm_optimization" / "session_state_metrics.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"snapshot_update_total_tokens": 27' in metrics


def test_build_optimized_prompt_keeps_recent_raw_turns_and_current_request(tmp_path, monkeypatch):
    from remy.config.settings import settings
    from remy.core.session_state_wrapper import SessionState, build_optimized_prompt

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

    session_log = []
    for index in range(8):
        session_log.append({"type": "user_text", "text": f"user turn {index}"})
        session_log.append({"type": "model_response", "text": f"assistant turn {index}"})

    state = SessionState(
        session_id="prompt",
        snapshot="[CURRENT_GOAL]\nKeep project context.",
        pinned_facts=["RX-4471"],
        turns_folded=12,
    )

    prompt = build_optimized_prompt(
        state,
        session_log,
        "What is the current code?",
        system_prompt="You are Remy.",
    )

    assert "You are Remy." in prompt
    assert "=== SESSION STATE" in prompt
    assert "RX-4471" in prompt
    assert "user turn 7" in prompt
    assert "assistant turn 7" in prompt
    assert "user turn 0" not in prompt
    assert "=== CURRENT REQUEST ===\nWhat is the current code?" in prompt


def test_projected_state_gate_rejects_short_sessions_and_accepts_long_ones(tmp_path, monkeypatch):
    from remy.config.settings import settings
    from remy.core.session_state_wrapper import should_use_projected_state

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

    short_log = [{"type": "user_text", "text": "Critical code RX-4471."}]
    short_decision = should_use_projected_state(
        short_log,
        "What code?",
        min_raw_tokens=50,
    )
    assert short_decision["use_projected"] is False

    long_log = [{"type": "user_text", "text": "Critical code RX-4471."}]
    for index in range(80):
        long_log.append(
            {
                "type": "user_text",
                "text": f"routine planning note {index} with verbose unrelated context",
            }
        )
        long_log.append({"type": "model_response", "text": f"ack {index}"})

    long_decision = should_use_projected_state(
        long_log,
        "What code?",
        min_raw_tokens=50,
    )
    assert long_decision["use_projected"] is True
    assert long_decision["tokens_saved_estimate"] > 0
    assert long_decision["pinned_count"] >= 1


@pytest.mark.asyncio
async def test_session_state_eval_requires_correctness_not_only_token_savings(tmp_path, monkeypatch):
    from remy.config.settings import settings
    from remy.core.session_state_wrapper import SessionStateEvalCase, run_session_state_eval

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

    session_log = [
        {
            "type": "user_text",
            "text": "Critical intake note: authorization code RX-4471 must be preserved.",
        },
        {"type": "model_response", "text": "Stored the authorization code."},
    ]
    for index in range(30):
        session_log.append({"type": "user_text", "text": f"routine unrelated planning note {index}"})
        session_log.append({"type": "model_response", "text": f"acknowledged planning note {index}"})

    def evidence_sensitive_answer(prompt: str):
        return "The code is RX-4471." if "RX-4471" in prompt else "I do not know."

    report = await run_session_state_eval(
        [
            SessionStateEvalCase(
                case_id="early-code",
                session_log=session_log,
                user_request="What was the authorization code?",
                expected_fragments=["RX-4471"],
                snapshot="[CURRENT_GOAL]\nAnswer questions using preserved exact facts.",
            )
        ],
        evidence_sensitive_answer,
    )

    summary = report["summary"]
    assert summary["raw_accuracy"] == 1.0
    assert summary["optimized_accuracy"] == 1.0
    assert summary["token_reduction_ratio"] > 1
    assert summary["production_ready"] is True


@pytest.mark.asyncio
async def test_session_state_eval_marks_wrong_fast_answers_not_production_ready(tmp_path, monkeypatch):
    from remy.config.settings import settings
    from remy.core.session_state_wrapper import SessionStateEvalCase, run_session_state_eval

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

    session_log = [{"type": "user_text", "text": "Critical code RX-4471."}]

    def bad_answer(_prompt: str):
        return "The code is RX-0000."

    report = await run_session_state_eval(
        [
            SessionStateEvalCase(
                case_id="wrong",
                session_log=session_log,
                user_request="What code?",
                expected_fragments=["RX-4471"],
                snapshot="[DO_NOT_FORGET]\nCritical code RX-4471.",
            )
        ],
        bad_answer,
    )

    assert report["summary"]["optimized_accuracy"] == 0.0
    assert report["summary"]["production_ready"] is False
