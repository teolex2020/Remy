"""Tests for P3.3 research memory — contradiction tracking, prior findings, prompt formatting."""

from unittest.mock import patch

from remy.core.research_memory import (
    check_finding_contradictions,
    extract_known_companies,
    format_existing_research_for_prompt,
    get_contradictions,
    get_unresolved_contradictions,
    record_contradiction,
    resolve_contradiction,
)

# ============== Contradiction Index ==============


class TestContradictionIndex:
    def test_record_and_retrieve(self, tmp_path):
        with patch("remy.core.research_memory.settings") as mock:
            mock.DATA_DIR = tmp_path
            record_contradiction(
                topic="AI market",
                finding_a="Company X leads the market",
                finding_b="Company Y leads the market",
                source_a="https://a.com",
                source_b="https://b.com",
                finding_id_a="f1",
                finding_id_b="f2",
                project_id="p1",
            )
            records = get_contradictions()

        assert len(records) == 1
        assert records[0]["topic"] == "AI market"
        assert records[0]["status"] == "unresolved"
        assert records[0]["count"] == 1

    def test_dedup_by_finding_ids(self, tmp_path):
        with patch("remy.core.research_memory.settings") as mock:
            mock.DATA_DIR = tmp_path
            for _ in range(3):
                record_contradiction(
                    topic="AI market",
                    finding_a="A leads",
                    finding_b="B leads",
                    finding_id_a="f1",
                    finding_id_b="f2",
                )
            records = get_contradictions()

        assert len(records) == 1
        assert records[0]["count"] == 3

    def test_dedup_order_independent(self, tmp_path):
        with patch("remy.core.research_memory.settings") as mock:
            mock.DATA_DIR = tmp_path
            record_contradiction(
                topic="t",
                finding_a="a",
                finding_b="b",
                finding_id_a="f1",
                finding_id_b="f2",
            )
            record_contradiction(
                topic="t",
                finding_a="a",
                finding_b="b",
                finding_id_a="f2",
                finding_id_b="f1",
            )
            records = get_contradictions()

        assert len(records) == 1
        assert records[0]["count"] == 2

    def test_no_dedup_without_finding_ids(self, tmp_path):
        with patch("remy.core.research_memory.settings") as mock:
            mock.DATA_DIR = tmp_path
            record_contradiction(topic="t", finding_a="a", finding_b="b")
            record_contradiction(topic="t", finding_a="a", finding_b="b")
            records = get_contradictions()

        assert len(records) == 2

    def test_filter_by_topic(self, tmp_path):
        with patch("remy.core.research_memory.settings") as mock:
            mock.DATA_DIR = tmp_path
            record_contradiction(topic="AI market", finding_a="a", finding_b="b")
            record_contradiction(topic="Cloud infra", finding_a="c", finding_b="d")
            ai = get_contradictions(topic="AI")
            cloud = get_contradictions(topic="cloud")

        assert len(ai) == 1
        assert len(cloud) == 1

    def test_limit(self, tmp_path):
        with patch("remy.core.research_memory.settings") as mock:
            mock.DATA_DIR = tmp_path
            for i in range(10):
                record_contradiction(
                    topic="t",
                    finding_a=f"a{i}",
                    finding_b=f"b{i}",
                    finding_id_a=f"fa{i}",
                    finding_id_b=f"fb{i}",
                )
            records = get_contradictions(limit=3)

        assert len(records) == 3

    def test_get_unresolved_only(self, tmp_path):
        with patch("remy.core.research_memory.settings") as mock:
            mock.DATA_DIR = tmp_path
            record_contradiction(
                topic="t",
                finding_a="a",
                finding_b="b",
                finding_id_a="f1",
                finding_id_b="f2",
            )
            record_contradiction(
                topic="t",
                finding_a="c",
                finding_b="d",
                finding_id_a="f3",
                finding_id_b="f4",
            )
            resolve_contradiction("f1", "f2", "resolved")
            unresolved = get_unresolved_contradictions(topic="t")

        assert len(unresolved) == 1
        assert unresolved[0]["finding_id_a"] == "f3"


class TestResolveContradiction:
    def test_resolve_existing(self, tmp_path):
        with patch("remy.core.research_memory.settings") as mock:
            mock.DATA_DIR = tmp_path
            record_contradiction(
                topic="t",
                finding_a="a",
                finding_b="b",
                finding_id_a="f1",
                finding_id_b="f2",
            )
            result = resolve_contradiction("f1", "f2", "superseded")

        assert result is True

    def test_resolve_returns_false_for_missing(self, tmp_path):
        with patch("remy.core.research_memory.settings") as mock:
            mock.DATA_DIR = tmp_path
            result = resolve_contradiction("x", "y")

        assert result is False

    def test_resolved_has_timestamp(self, tmp_path):
        with patch("remy.core.research_memory.settings") as mock:
            mock.DATA_DIR = tmp_path
            record_contradiction(
                topic="t",
                finding_a="a",
                finding_b="b",
                finding_id_a="f1",
                finding_id_b="f2",
            )
            resolve_contradiction("f1", "f2")
            records = get_contradictions()

        assert records[0]["status"] == "resolved"
        assert "resolved_at" in records[0]


# ============== Entity Extraction ==============


class TestExtractKnownCompanies:
    def test_extracts_capitalized_phrases(self):
        findings = [
            {"content": "OpenAI launched GPT-5. Google Cloud expanded services."},
            {"content": "Microsoft Azure leads in enterprise cloud."},
        ]
        companies = extract_known_companies(findings)
        # Should find multi-word capitalized phrases
        assert any("Microsoft Azure" in c for c in companies) or any(
            "Google Cloud" in c for c in companies
        )

    def test_filters_common_words(self):
        findings = [{"content": "The Research Report shows important Analysis Data."}]
        companies = extract_known_companies(findings)
        # "Research", "Report", "Analysis", "Data" should be filtered
        for c in companies:
            assert c.lower() not in ("research", "report", "analysis", "data")

    def test_empty_findings(self):
        assert extract_known_companies([]) == []

    def test_caps_at_20(self):
        content = " ".join(f"Company{i} Name{i}" for i in range(30))
        findings = [{"content": content}]
        companies = extract_known_companies(findings)
        assert len(companies) <= 20


# ============== Prompt Formatting ==============


class TestFormatExistingResearchForPrompt:
    def test_empty_when_no_data(self, tmp_path):
        with (
            patch("remy.core.research_memory.get_prior_findings", return_value=[]),
            patch("remy.core.research_memory.get_completed_reports", return_value=[]),
            patch("remy.core.research_memory.get_unresolved_contradictions", return_value=[]),
        ):
            text = format_existing_research_for_prompt("anything")

        assert text == ""

    def test_includes_findings(self):
        findings = [
            {
                "id": "f1",
                "content": "Company X has 40% market share",
                "source_url": "https://a.com",
                "confidence": 0.8,
                "project_id": "p1",
                "timestamp": "",
            },
        ]
        with (
            patch("remy.core.research_memory.get_prior_findings", return_value=findings),
            patch("remy.core.research_memory.get_completed_reports", return_value=[]),
            patch("remy.core.research_memory.get_unresolved_contradictions", return_value=[]),
        ):
            text = format_existing_research_for_prompt("market share")

        assert "EXISTING RESEARCH" in text
        assert "40% market share" in text
        assert "[0.8]" in text
        assert "Build on existing" in text

    def test_includes_reports(self):
        reports = [
            {
                "id": "r1",
                "topic": "Cloud market",
                "content": "Report content...",
                "sources": [],
                "findings_count": 5,
                "confidence_avg": 0.75,
                "project_id": "p1",
                "timestamp": "",
            },
        ]
        with (
            patch("remy.core.research_memory.get_prior_findings", return_value=[]),
            patch("remy.core.research_memory.get_completed_reports", return_value=reports),
            patch("remy.core.research_memory.get_unresolved_contradictions", return_value=[]),
        ):
            text = format_existing_research_for_prompt("cloud")

        assert "PRIOR REPORT" in text
        assert "Cloud market" in text
        assert "5 findings" in text

    def test_includes_contradictions(self):
        contradictions = [
            {"finding_a": "X leads", "finding_b": "Y leads", "status": "unresolved"},
        ]
        with (
            patch("remy.core.research_memory.get_prior_findings", return_value=[]),
            patch("remy.core.research_memory.get_completed_reports", return_value=[]),
            patch(
                "remy.core.research_memory.get_unresolved_contradictions",
                return_value=contradictions,
            ),
        ):
            text = format_existing_research_for_prompt("market")

        assert "UNRESOLVED CONTRADICTIONS" in text
        assert "X leads" in text
        assert "Y leads" in text

    def test_includes_known_entities(self):
        findings = [
            {
                "id": "f1",
                "content": "Amazon Web Services dominates",
                "source_url": "",
                "confidence": 0.9,
                "project_id": "",
                "timestamp": "",
            },
        ]
        with (
            patch("remy.core.research_memory.get_prior_findings", return_value=findings),
            patch("remy.core.research_memory.get_completed_reports", return_value=[]),
            patch("remy.core.research_memory.get_unresolved_contradictions", return_value=[]),
        ):
            text = format_existing_research_for_prompt("cloud")

        assert "Known entities" in text
        assert "Amazon Web Services" in text


# ============== Contradiction Detection ==============


class TestCheckFindingContradictions:
    def test_detects_opposing_sentiment(self):
        prior = [
            {
                "id": "f1",
                "content": "Company X increased revenue and improved market position significantly",
                "source_url": "",
                "confidence": 0.8,
                "project_id": "",
                "timestamp": "",
            },
        ]
        with (
            patch("remy.core.research_memory.get_prior_findings", return_value=prior),
            patch("remy.core.research_memory.record_contradiction"),
        ):
            result = check_finding_contradictions(
                new_content="Company X declined in revenue and failed to maintain position",
                new_source="https://b.com",
                new_finding_id="f2",
                topic="Company X",
                project_id="p1",
            )

        assert len(result) >= 1
        assert result[0]["finding_id"] == "f1"

    def test_no_contradiction_for_similar_sentiment(self):
        prior = [
            {
                "id": "f1",
                "content": "Company X increased revenue and grew market share",
                "source_url": "",
                "confidence": 0.8,
                "project_id": "",
                "timestamp": "",
            },
        ]
        with (
            patch("remy.core.research_memory.get_prior_findings", return_value=prior),
            patch("remy.core.research_memory.record_contradiction"),
        ):
            result = check_finding_contradictions(
                new_content="Company X increased profits and grew its customer base",
                new_source="https://b.com",
                new_finding_id="f2",
                topic="Company X",
            )

        assert len(result) == 0

    def test_skips_self(self):
        prior = [
            {
                "id": "f1",
                "content": "Company X not performing well, declined sharply",
                "source_url": "",
                "confidence": 0.8,
                "project_id": "",
                "timestamp": "",
            },
        ]
        with (
            patch("remy.core.research_memory.get_prior_findings", return_value=prior),
            patch("remy.core.research_memory.record_contradiction"),
        ):
            result = check_finding_contradictions(
                new_content="Company X not performing well",
                new_source="",
                new_finding_id="f1",  # same ID
                topic="Company X",
            )

        assert len(result) == 0

    def test_records_contradictions(self):
        prior = [
            {
                "id": "f1",
                "content": "Company X is the largest and best in the industry",
                "source_url": "https://a.com",
                "confidence": 0.8,
                "project_id": "",
                "timestamp": "",
            },
        ]
        with (
            patch("remy.core.research_memory.get_prior_findings", return_value=prior) as mock_prior,
            patch("remy.core.research_memory.record_contradiction") as mock_record,
        ):
            check_finding_contradictions(
                new_content="Company X declined and is no longer the top player, failed completely",
                new_source="https://b.com",
                new_finding_id="f2",
                topic="Company X",
                project_id="p1",
            )

        if mock_record.called:
            call_kwargs = mock_record.call_args[1]
            assert call_kwargs["topic"] == "Company X"
            assert call_kwargs["finding_id_a"] == "f2"

    def test_empty_prior_findings(self):
        with patch("remy.core.research_memory.get_prior_findings", return_value=[]):
            result = check_finding_contradictions(
                new_content="anything",
                new_source="",
                new_finding_id="f1",
                topic="topic",
            )

        assert result == []

    def test_caps_at_3_recorded(self):
        prior = [
            {
                "id": f"f{i}",
                "content": f"Company increased and grew and improved and succeeded and is leading finding {i}",
                "source_url": "",
                "confidence": 0.8,
                "project_id": "",
                "timestamp": "",
            }
            for i in range(10)
        ]
        with (
            patch("remy.core.research_memory.get_prior_findings", return_value=prior) as mock_prior,
            patch("remy.core.research_memory.record_contradiction") as mock_record,
        ):
            check_finding_contradictions(
                new_content="Company declined and dropped and failed and is no longer leading",
                new_source="",
                new_finding_id="f99",
                topic="topic",
            )

        # Should record at most 3
        assert mock_record.call_count <= 3


# ============== Research Worker Prompt Integration ==============


class TestResearchWorkerPromptIntegration:
    def test_prompt_includes_research_memory(self):
        from remy.core.workers.research_worker import build_research_worker_prompt

        with patch(
            "remy.core.research_memory.format_existing_research_for_prompt",
            return_value="\nEXISTING RESEARCH (from prior sessions):\n- 3 prior findings\n",
        ):
            prompt = build_research_worker_prompt(
                {"description": "Analyze AI market", "goal_template": "market_research"},
            )

        assert "EXISTING RESEARCH" in prompt
        assert "prior findings" in prompt

    def test_prompt_works_without_research_memory(self):
        from remy.core.workers.research_worker import build_research_worker_prompt

        with patch(
            "remy.core.research_memory.format_existing_research_for_prompt",
            side_effect=ImportError("no module"),
        ):
            prompt = build_research_worker_prompt(
                {"description": "Analyze AI market", "goal_template": "market_research"},
            )

        assert "RESEARCH_WORKER" in prompt
        # Should not crash


# ============== Metrics Integration ==============


class TestResearchMemoryMetrics:
    def test_detect_memory_signals_from_auto_contradictions(self):
        from remy.core.task_metrics import detect_memory_signals

        log = [
            {
                "type": "tool_call",
                "tool": "add_research_finding",
                "result": '{"stored": true, "auto_contradictions": [{"finding_id": "f1", "score": 3}]}',
            },
        ]
        mem_assisted, retry_shaped = detect_memory_signals(log)
        assert mem_assisted is True

    def test_detect_memory_signals_no_contradictions(self):
        from remy.core.task_metrics import detect_memory_signals

        log = [
            {"type": "tool_call", "tool": "add_research_finding", "result": '{"stored": true}'},
        ]
        mem_assisted, retry_shaped = detect_memory_signals(log)
        assert mem_assisted is False

    def test_detect_memory_signals_dict_result(self):
        from remy.core.task_metrics import detect_memory_signals

        log = [
            {
                "type": "tool_call",
                "tool": "add_research_finding",
                "result": {"stored": True, "auto_contradictions": [{"finding_id": "f1"}]},
            },
        ]
        mem_assisted, retry_shaped = detect_memory_signals(log)
        assert mem_assisted is True
