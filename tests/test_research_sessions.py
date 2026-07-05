"""Tests for ResearchSessions — persisted research session tracking."""

import json
from unittest.mock import patch

import pytest

from remy.core.research_sessions import (
    ResearchSession,
    append_queries,
    get_or_create_research_session,
    get_research_session_summary,
    get_research_session_trace,
    load_research_session,
    mark_session_completed,
    reconcile_session_sources,
    record_contradictions,
    record_finding,
    record_source_decision,
    record_source_fetch,
    save_research_session,
)


@pytest.fixture
def sessions_dir(tmp_path):
    """Patch SESSIONS_DIR to a temp directory."""
    with patch("remy.core.research_sessions.SESSIONS_DIR", tmp_path):
        yield tmp_path


def _session(
    goal_id: str = "goal-1",
    pack_id: str = "market_research",
    topic: str = "test topic",
    research_mode: str = "balanced",
    source_scope: str = "web",
) -> ResearchSession:
    return ResearchSession(
        session_id="rs-test123",
        goal_id=goal_id,
        pack_id=pack_id,
        topic=topic,
        research_mode=research_mode,
        source_scope=source_scope,
    )


# ============== ResearchSession dataclass ==============


class TestResearchSession:
    def test_defaults(self):
        s = _session()
        assert s.status == "active"
        assert s.citation_required is True
        assert s.source_domains == []
        assert s.resumed_runs == 0
        assert s.findings == []
        assert s.contradictions == []

    def test_summary_empty(self):
        s = _session()
        summary = s.summary()
        assert summary["session_id"] == "rs-test123"
        assert summary["status"] == "active"
        assert summary["findings_count"] == 0
        assert summary["accepted_sources_count"] == 0
        assert summary["citation_coverage_rate"] == 0.0

    def test_summary_with_data(self):
        s = _session()
        s.accepted_sources = [{"url": "a"}, {"url": "b"}]
        s.rejected_sources = [{"url": "c"}]
        s.findings = [{"text": "f1"}, {"text": "f2"}, {"text": "f3"}]
        s.contradictions = [{"id": "c1"}]
        summary = s.summary()
        assert summary["accepted_sources_count"] == 2
        assert summary["rejected_sources_count"] == 1
        assert summary["findings_count"] == 3
        assert summary["contradictions_count"] == 1
        # coverage = accepted / findings = 2/3
        assert summary["citation_coverage_rate"] == round(2 / 3, 3)


# ============== save / load ==============


class TestSaveLoad:
    def test_save_and_load(self, sessions_dir):
        s = _session()
        save_research_session(s)

        loaded = load_research_session("goal-1")
        assert loaded is not None
        assert loaded.session_id == "rs-test123"
        assert loaded.pack_id == "market_research"
        assert loaded.topic == "test topic"

    def test_load_missing_returns_none(self, sessions_dir):
        assert load_research_session("nonexistent") is None

    def test_load_corrupt_returns_none(self, sessions_dir):
        bad_file = sessions_dir / "bad.json"
        bad_file.write_text("not json")
        assert load_research_session("bad") is None

    def test_save_updates_timestamp(self, sessions_dir):
        s = _session()
        old_time = s.updated_at
        save_research_session(s)
        loaded = load_research_session("goal-1")
        assert loaded.updated_at >= old_time

    def test_file_written_as_json(self, sessions_dir):
        s = _session()
        save_research_session(s)
        raw = json.loads((sessions_dir / "goal-1.json").read_text(encoding="utf-8"))
        assert raw["goal_id"] == "goal-1"
        assert raw["pack_id"] == "market_research"


# ============== get_or_create_research_session ==============


class TestGetOrCreate:
    def test_creates_new(self, sessions_dir):
        session, resumed = get_or_create_research_session(
            goal_id="g1",
            pack_id="market_research",
            topic="AI market",
            research_mode="deep",
            source_scope="papers",
        )
        assert resumed is False
        assert session.goal_id == "g1"
        assert session.research_mode == "deep"
        assert session.session_id.startswith("rs-")
        assert session.resumed_runs == 0

    def test_resumes_existing(self, sessions_dir):
        # Create first
        get_or_create_research_session(
            goal_id="g1", pack_id="mr", topic="AI", research_mode="balanced", source_scope="web"
        )
        # Resume
        session, resumed = get_or_create_research_session(
            goal_id="g1",
            pack_id="mr",
            topic="AI updated",
            research_mode="deep",
            source_scope="papers",
        )
        assert resumed is True
        assert session.resumed_runs == 1
        assert session.topic == "AI updated"
        assert session.research_mode == "deep"

    def test_resume_increments_counter(self, sessions_dir):
        get_or_create_research_session(
            goal_id="g1", pack_id="mr", topic="t", research_mode="balanced", source_scope="web"
        )
        for _ in range(3):
            get_or_create_research_session(
                goal_id="g1", pack_id="mr", topic="t", research_mode="balanced", source_scope="web"
            )
        loaded = load_research_session("g1")
        assert loaded.resumed_runs == 3

    def test_warnings_deduplicated(self, sessions_dir):
        get_or_create_research_session(
            goal_id="g1",
            pack_id="mr",
            topic="t",
            research_mode="balanced",
            source_scope="web",
            warnings=["warn1"],
        )
        session, _ = get_or_create_research_session(
            goal_id="g1",
            pack_id="mr",
            topic="t",
            research_mode="balanced",
            source_scope="web",
            warnings=["warn1", "warn2"],
        )
        assert session.warnings == ["warn1", "warn2"]

    def test_source_domains_forwarded(self, sessions_dir):
        session, _ = get_or_create_research_session(
            goal_id="g1",
            pack_id="mr",
            topic="t",
            research_mode="balanced",
            source_scope="domain",
            source_domains=["arxiv.org", "scholar.google.com"],
        )
        assert session.source_domains == ["arxiv.org", "scholar.google.com"]


# ============== append_queries ==============


class TestAppendQueries:
    def test_appends_new_queries(self, sessions_dir):
        save_research_session(_session())
        result = append_queries("goal-1", ["q1", "q2"])
        assert result is not None
        assert result.generated_queries == ["q1", "q2"]

    def test_deduplicates_queries(self, sessions_dir):
        save_research_session(_session())
        append_queries("goal-1", ["q1", "q2"])
        result = append_queries("goal-1", ["q2", "q3"])
        assert result.generated_queries == ["q1", "q2", "q3"]

    def test_skips_empty(self, sessions_dir):
        save_research_session(_session())
        result = append_queries("goal-1", ["", "q1", ""])
        assert result.generated_queries == ["q1"]

    def test_returns_none_for_missing(self, sessions_dir):
        assert append_queries("nonexistent", ["q1"]) is None


# ============== record_source_fetch ==============


class TestRecordSourceFetch:
    def test_records_source(self, sessions_dir):
        save_research_session(_session())
        result = record_source_fetch("goal-1", {"url": "https://example.com", "title": "Test"})
        assert len(result.fetched_sources) == 1

    def test_deduplicates_by_url(self, sessions_dir):
        save_research_session(_session())
        record_source_fetch("goal-1", {"url": "https://example.com"})
        result = record_source_fetch("goal-1", {"url": "https://example.com"})
        assert len(result.fetched_sources) == 1

    def test_returns_none_for_missing(self, sessions_dir):
        assert record_source_fetch("nonexistent", {"url": "x"}) is None


# ============== record_source_decision ==============


class TestRecordSourceDecision:
    def test_accepted(self, sessions_dir):
        save_research_session(_session())
        result = record_source_decision(
            "goal-1", {"url": "https://good.com"}, accepted=True, reason="relevant"
        )
        assert len(result.accepted_sources) == 1
        assert result.accepted_sources[0]["reason"] == "relevant"

    def test_rejected(self, sessions_dir):
        save_research_session(_session())
        result = record_source_decision(
            "goal-1", {"url": "https://bad.com"}, accepted=False, reason="spam"
        )
        assert len(result.rejected_sources) == 1
        assert result.rejected_sources[0]["reason"] == "spam"

    def test_deduplicates_by_url(self, sessions_dir):
        save_research_session(_session())
        record_source_decision("goal-1", {"url": "https://a.com"}, accepted=True)
        result = record_source_decision("goal-1", {"url": "https://a.com"}, accepted=True)
        assert len(result.accepted_sources) == 1

    def test_returns_none_for_missing(self, sessions_dir):
        assert record_source_decision("x", {"url": "y"}, accepted=True) is None


# ============== record_finding ==============


class TestRecordFinding:
    def test_records_finding(self, sessions_dir):
        save_research_session(_session())
        result = record_finding("goal-1", {"text": "important insight", "source": "https://a.com"})
        assert len(result.findings) == 1
        assert result.findings[0]["text"] == "important insight"

    def test_allows_duplicates(self, sessions_dir):
        """Findings are appended without dedup (unlike sources)."""
        save_research_session(_session())
        record_finding("goal-1", {"text": "f1"})
        result = record_finding("goal-1", {"text": "f1"})
        assert len(result.findings) == 2

    def test_returns_none_for_missing(self, sessions_dir):
        assert record_finding("x", {"text": "y"}) is None

    def test_reconcile_infers_profile_url_and_accepted_source(self, sessions_dir):
        save_research_session(_session())
        record_finding(
            "goal-1",
            {
                "summary": "Harrison Chase (@hwchase17) discusses agent memory and LangGraph.",
                "source_url": "",
            },
        )
        session = reconcile_session_sources("goal-1")
        assert session is not None
        assert session.findings[0]["source_url"] == "https://x.com/hwchase17"
        assert any(item["url"] == "https://x.com/hwchase17" for item in session.fetched_sources)
        assert any(item["url"] == "https://x.com/hwchase17" for item in session.accepted_sources)


# ============== record_contradictions ==============


class TestRecordContradictions:
    def test_records_contradiction(self, sessions_dir):
        save_research_session(_session())
        result = record_contradictions("goal-1", [{"id": "c1", "claim_a": "X", "claim_b": "Y"}])
        assert len(result.contradictions) == 1

    def test_deduplicates_by_id(self, sessions_dir):
        save_research_session(_session())
        record_contradictions("goal-1", [{"id": "c1", "detail": "first"}])
        result = record_contradictions("goal-1", [{"id": "c1", "detail": "second"}])
        assert len(result.contradictions) == 1

    def test_returns_none_for_missing(self, sessions_dir):
        assert record_contradictions("x", [{"id": "c1"}]) is None


# ============== mark_session_completed ==============


class TestMarkCompleted:
    def test_marks_completed(self, sessions_dir):
        save_research_session(_session())
        result = mark_session_completed("goal-1", final_artifact_id="art-123")
        assert result.status == "completed"
        assert result.final_artifact_id == "art-123"

    def test_without_artifact(self, sessions_dir):
        save_research_session(_session())
        result = mark_session_completed("goal-1")
        assert result.status == "completed"
        assert result.final_artifact_id == ""

    def test_returns_none_for_missing(self, sessions_dir):
        assert mark_session_completed("x") is None


# ============== get_research_session_summary ==============


class TestGetSummary:
    def test_returns_summary(self, sessions_dir):
        save_research_session(_session())
        summary = get_research_session_summary("goal-1")
        assert summary is not None
        assert summary["session_id"] == "rs-test123"

    def test_returns_none_for_missing(self, sessions_dir):
        assert get_research_session_summary("x") is None


class TestGetTrace:
    def test_returns_trace_with_queries_domains_and_gaps(self, sessions_dir):
        session = _session()
        session.generated_queries = [
            "vat filing deadlines 2026 ukraine",
            "vat invoice reconciliation checklist",
            "primary source for VAT reporting guidance",
        ]
        session.accepted_sources = [
            {"url": "https://tax.gov.ua/article", "title": "Tax guidance"},
            {"url": "https://tax.gov.ua/checklist", "title": "Checklist"},
        ]
        session.fetched_sources = [
            {"url": "https://minfin.com.ua/news", "title": "Minfin"},
        ]
        session.findings = [
            {"text": "Need primary citation"},
            {"text": "Need another corroborating source"},
            {"text": "Cross-check deadline variance"},
            {"text": "Confirm filing portal update"},
        ]
        session.contradictions = [{"id": "c1", "summary": "Conflicting deadline"}]
        session.warnings = ["Need a primary source before publishing."]
        session.final_artifact_id = "report-1"
        save_research_session(session)

        with patch(
            "remy.core.research_sessions._load_artifact_info",
            return_value={
                "record_id": "report-1",
                "artifact_format": "markdown",
                "viewer_url": "/api/autonomy/research-artifacts/report-1/view",
                "markdown_url": "/api/autonomy/research-artifacts/report-1/markdown",
                "pdf_url": "/api/reports/vat.pdf",
                "pdf_filename": "vat.pdf",
                "markdown_available": True,
                "markdown_preview": "# VAT reporting",
            },
        ):
            trace = get_research_session_trace("goal-1")

        assert trace is not None
        assert trace["recent_queries"][-1] == "primary source for VAT reporting guidance"
        assert trace["top_source_domains"][0]["domain"] == "tax.gov.ua"
        assert trace["accepted_source_preview"][0]["title"] == "Tax guidance"
        assert trace["artifact"]["pdf_url"] == "/api/reports/vat.pdf"
        assert trace["artifact"]["viewer_url"].endswith("/report-1/view")
        assert trace["artifact"]["markdown_url"].endswith("/report-1/markdown")
        assert trace["artifact"]["markdown_available"] is True
        assert any("citation coverage" in gap.lower() for gap in trace["knowledge_gaps"])
        assert any("contradictory" in gap.lower() for gap in trace["knowledge_gaps"])
        assert trace["warnings"][0] == "Need a primary source before publishing."

    def test_returns_none_for_missing(self, sessions_dir):
        assert get_research_session_trace("x") is None
