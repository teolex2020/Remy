from remy.core.eval_metrics import compute_response_metrics
from remy.core.factuality import classify_response_claims, enforce_factuality, summarize_claim_details


def test_enforce_factuality_downgrades_unverified_observed_claim():
    text, report = enforce_factuality(
        "I just reviewed your repository and it looks strong.",
        session_log=[],
    )

    assert report.unsupported_observed_claims == 1
    assert report.claim_counts["observed_fact"] == 1
    assert report.modified is True
    assert "reviewed your repository" not in text.lower()
    assert "stored context" in text.lower() or "conversation" in text.lower()


def test_enforce_factuality_flags_unverified_current_fact_without_evidence():
    text, report = enforce_factuality(
        "The latest GitHub repo has 42k stars right now.",
        session_log=[],
    )

    assert report.unverified_current_claims == 1
    assert report.claim_counts["unverified_current_fact"] == 1
    assert report.modified is True
    assert "hypoth" in text.lower() or "provisional" in text.lower()
    assert "42k stars" in text


def test_classify_response_claims_distinguishes_claim_classes():
    claims = classify_response_claims(
        "Based on our conversation, you prefer tea. It seems you are optimizing for speed. "
        "I checked the latest release notes.",
        session_log=[],
    )

    assert [claim.claim_class for claim in claims] == [
        "memory_fact",
        "inference",
        "observed_fact",
    ]
    assert claims[-1].supported is False


def test_enforce_factuality_allows_observed_claim_with_external_evidence():
    session_log = [
        {"type": "tool_call", "tool": "extract_content", "result_full": "{\"url\": \"https://example.com/source\", \"title\": \"Trusted Source\", \"content\": \"Found matching sources\"}"},
    ]
    text, report = enforce_factuality(
        "I checked the latest information and found two sources.",
        session_log=session_log,
    )

    assert report.had_external_evidence is True
    assert report.unsupported_observed_claims == 0
    assert report.unverified_current_claims == 0
    assert "I checked the latest information and found two sources." in text


def test_enforce_factuality_appends_source_links_for_external_facts():
    session_log = [
        {
            "type": "tool_call",
            "tool": "extract_content",
            "args_full": {"url": "https://github.com/teolex2020/AuraSDK/releases/tag/v1.4.0"},
            "result_full": "{\"url\": \"https://github.com/teolex2020/AuraSDK/releases/tag/v1.4.0\", \"title\": \"AuraSDK v1.4.0\", \"content\": \"release notes\"}",
        },
    ]
    text, report = enforce_factuality(
        "I checked the project page and the latest version is v1.4.0.",
        session_log=session_log,
    )

    assert report.had_external_evidence is True
    assert report.citations_added == 1
    assert "Sources:" in text
    assert "https://github.com/teolex2020/AuraSDK/releases/tag/v1.4.0" in text


def test_enforce_factuality_marks_missing_source_link_when_no_url_captured():
    session_log = [
        {"type": "tool_call", "tool": "browse_page", "result": "Loaded successfully"},
    ]
    text, report = enforce_factuality(
        "I checked the page and the competitor just shipped a new feature.",
        session_log=session_log,
    )

    assert report.had_external_evidence is True
    assert report.missing_source_links is True
    assert "Source note:" in text


def test_enforce_factuality_dedupes_repeated_source_note():
    session_log = [
        {"type": "tool_call", "tool": "browse_page", "result": "Loaded successfully"},
    ]
    original = (
        "Here is the answer.\n\n"
        "Source note: I used external tools in this turn, but no stable source link "
        "was captured in the tool output, so treat specific external facts as provisional "
        "until I provide a direct link.\n\n"
        "Source note: I used external tools in this turn, but no stable source link "
        "was captured in the tool output, so treat specific external facts as provisional "
        "until I provide a direct link."
    )

    text, report = enforce_factuality(original, session_log=session_log)

    assert report.missing_source_links is True
    assert text.count("Source note:") == 1


def test_enforce_factuality_flags_market_research_numbers_without_evidence():
    text, report = enforce_factuality(
        "Code Archaeology is a $30-40B market and teams pay $50k+ contracts according to deep research in 2026.",
        session_log=[],
    )

    assert report.unverified_current_claims == 1
    assert report.claim_counts["unverified_current_fact"] == 1
    assert report.modified is True
    assert "hypoth" in text.lower()
    assert "$30-40B" in text or "$30-40b" in text.lower()


def test_enforce_factuality_downgrades_substantive_answer_without_evidence_anchor():
    text, report = enforce_factuality(
        (
            "For programming work, Claude Sonnet is the strongest overall choice. "
            "DeepSeek Coder is the better budget option for teams that need low cost. "
            "GPT-4o is a reliable fallback when context quality matters."
        ),
        session_log=[],
    )

    assert report.ungrounded_answer_claims == 1
    assert report.unsupported_claims_total == 1
    assert report.claim_counts["ungrounded_answer"] == 1
    assert report.modified is True
    assert "evidence anchor" in text.lower()


def test_enforce_factuality_allows_market_research_claim_with_source_links():
    session_log = [
        {
            "type": "tool_call",
            "tool": "extract_content",
            "args_full": {"url": "https://example.com/report"},
            "result_full": "{\"url\": \"https://example.com/report\", \"title\": \"Market Report\", \"content\": \"benchmark https://example.com/bench\"}",
        },
    ]

    text, report = enforce_factuality(
        "Software maintenance spending is large in 2026 based on recent research.",
        session_log=session_log,
    )

    assert report.had_external_evidence is True
    assert report.unverified_current_claims == 0
    assert "Sources:" in text
    assert "https://example.com/report" in text


def test_classify_response_claims_binds_memory_fact_to_recall_record():
    session_log = [
        {
            "type": "tool_call",
            "tool": "recall",
            "result": "[id:rec-123] [trust: 0.9 | interactive] User prefers tea over coffee [preference]",
        },
    ]

    claims = classify_response_claims(
        "Based on our conversation, you prefer tea.",
        session_log=session_log,
    )
    text, report = enforce_factuality(
        "Based on our conversation, you prefer tea.",
        session_log=session_log,
    )

    assert len(claims) == 1
    assert claims[0].claim_class == "memory_fact"
    assert report.supported_claims_total == 1
    assert report.unsupported_claims_total == 0
    assert report.evidence_record_ids == ["rec-123"]
    assert "not backed by the recalled evidence" not in text.lower()


def test_enforce_factuality_rewrites_when_most_claims_lack_recall_support():
    session_log = [
        {
            "type": "tool_call",
            "tool": "recall",
            "result": "[id:rec-1] [trust: 0.9 | interactive] User prefers tea over coffee [preference]",
        },
    ]

    text, report = enforce_factuality(
        "Based on our conversation, you prefer tea. It seems you are focused on speed. "
        "From our conversation, you usually deploy on Fridays.",
        session_log=session_log,
    )

    assert report.unsupported_claims_total >= 2
    assert report.supported_claims_total >= 1
    assert report.modified is True
    assert "not backed by the recalled evidence" in text.lower()


def test_enforce_factuality_structures_mixed_claims_into_sections():
    session_log = [
        {
            "type": "tool_call",
            "tool": "recall",
            "result": "[id:rec-1] [trust: 0.9 | interactive] User prefers tea over coffee [preference]",
        },
    ]

    text, report = enforce_factuality(
        "Based on our conversation, you prefer tea. It seems you are focused on speed.",
        session_log=session_log,
    )

    assert report.supported_claims_total == 1
    assert report.unsupported_claims_total == 1
    assert "Structured answer:" in text
    assert "Facts:" in text
    assert "Unknowns:" in text or "Inferences:" in text
    assert "rec-1" in text


def test_enforce_factuality_puts_unverified_current_claims_in_needs_verification():
    session_log = [
        {
            "type": "tool_call",
            "tool": "web_search",
            "result": "Results found",
        },
    ]

    text, report = enforce_factuality(
        "The latest GitHub repo has 42k stars right now.",
        session_log=session_log,
    )

    assert report.had_external_evidence is False
    assert "Needs verification:" in text or "Unverified draft:" in text


def test_eval_metrics_store_unsupported_claim_count():
    metrics = compute_response_metrics(
        session_id="s1",
        channel="desktop",
        messages=[],
        session_log=[],
        response_text="Based on context only.",
        unsupported_observed_claims=2,
    )

    assert metrics.unsupported_observed_claims == 2


def test_summarize_claim_details_returns_serializable_claims():
    _text, report = enforce_factuality(
        "Based on our conversation, you prefer tea. It seems you are focused on speed.",
        session_log=[
            {
                "type": "tool_call",
                "tool": "recall",
                "result": "[id:rec-1] [trust: 0.9 | interactive] User prefers tea over coffee [preference]",
            },
        ],
    )

    claims = summarize_claim_details(report)

    assert len(claims) == 2
    assert claims[0]["claim_class"] == "memory_fact"
    assert claims[0]["supporting_record_ids"] == ["rec-1"]
    assert isinstance(claims[1]["supported"], bool)



def test_enforce_factuality_does_not_treat_web_search_as_external_evidence():
    session_log = [
        {
            "type": "tool_call",
            "tool": "web_search",
            "result_full": '{"mode": "candidate_discovery", "sources": [{"title": "Candidate", "uri": "https://example.com/candidate"}]}',
        },
    ]
    text, report = enforce_factuality(
        "I checked the latest information and found two sources.",
        session_log=session_log,
    )

    assert report.had_external_evidence is False
    assert report.unsupported_observed_claims == 1
    assert report.modified is True
