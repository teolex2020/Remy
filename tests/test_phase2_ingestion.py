"""
Phase 2 Ingestion Tests

Behavioral invariants for the canonical ingestion entry points.

ingest_grounded_evidence:
  INV-G1: empty content → rejected
  INV-G2: missing source_url → rejected
  INV-G3: non-canonical extract_class → rejected
  INV-G4: operator_asserted via grounded path → rejected
  INV-G5: source_url not anchored by turn fetch → rejected
  INV-G6: different URL fetched than claimed → rejected
  INV-G7: grounded_external_fact with matching fetch → admitted at DOMAIN
  INV-G8: grounded_source_extract with matching fetch → admitted at DOMAIN
  INV-G9: URL canonicalisation handles trailing slash + case
  INV-G10: admitted record has learning_channel=internet_evidence
  INV-G11: admitted record has source_anchored=True
  INV-G12: admitted record has fetch_tool from the evidence item

ingest_operator_assertion:
  INV-O1: empty content → rejected
  INV-O2: non-user-direct channel → rejected
  INV-O3: user-direct channel without attribution tag → rejected
  INV-O4: user-direct channel + user-profile tag → admitted at DOMAIN
  INV-O5: user-direct channel + from-user tag → admitted at DOMAIN
  INV-O6: admitted record has admission_class=operator_asserted
  INV-O7: admitted record has learning_channel=operator_direct
"""
from unittest.mock import patch


_FETCH_EV = [{"url": "https://example.com/study", "title": "Study",
              "site": "example.com", "tool": "extract_content"}]


# ── ingest_grounded_evidence ─────────────────────────────────────────────────

class TestGroundedEvidenceRejection:
    def test_empty_content_rejected(self):
        """INV-G1: empty content → rejected."""
        from remy.core.ingestion import ingest_grounded_evidence
        result = ingest_grounded_evidence(
            content="", source_url="https://example.com/study",
            session_id="s1", channel="worker",
            extract_class="grounded_external_fact",
        )
        assert result.rejected
        assert "content" in result.reason

    def test_missing_source_url_rejected(self):
        """INV-G2: missing source_url → rejected."""
        from remy.core.ingestion import ingest_grounded_evidence
        result = ingest_grounded_evidence(
            content="Vitamin C supports immunity.", source_url="",
            session_id="s1", channel="worker",
            extract_class="grounded_external_fact",
        )
        assert result.rejected
        assert "source_url" in result.reason

    def test_non_canonical_class_rejected(self):
        """INV-G3: non-canonical extract_class → rejected."""
        from remy.core.ingestion import ingest_grounded_evidence
        result = ingest_grounded_evidence(
            content="X", source_url="https://example.com/study",
            session_id="s1", channel="worker",
            extract_class="grounded_extraction",  # ad-hoc legacy name
        )
        assert result.rejected
        assert "extract_class" in result.reason

    def test_operator_asserted_rejected_via_grounded_path(self):
        """INV-G4: operator_asserted via grounded path → rejected."""
        from remy.core.ingestion import ingest_grounded_evidence
        result = ingest_grounded_evidence(
            content="X", source_url="https://example.com/study",
            session_id="s1", channel="worker",
            extract_class="operator_asserted",
        )
        assert result.rejected
        assert "operator_asserted" in result.reason

    def test_unanchored_source_url_rejected(self):
        """INV-G5: source_url not anchored by turn fetch → rejected."""
        from remy.core.ingestion import ingest_grounded_evidence
        with patch("remy.core.ingestion.get_turn_fetch_evidence", return_value=[]):
            result = ingest_grounded_evidence(
                content="Vitamin C supports immunity.",
                source_url="https://example.com/study",
                session_id="s1", channel="worker",
                extract_class="grounded_external_fact",
            )
        assert result.rejected
        assert "anchored" in result.reason or "fetch" in result.reason

    def test_different_url_fetched_rejected(self):
        """INV-G6: different URL fetched than claimed → rejected."""
        from remy.core.ingestion import ingest_grounded_evidence
        other = [{"url": "https://other.com/page", "title": "", "site": "", "tool": "extract_content"}]
        with patch("remy.core.ingestion.get_turn_fetch_evidence", return_value=other):
            result = ingest_grounded_evidence(
                content="X", source_url="https://example.com/study",
                session_id="s1", channel="worker",
                extract_class="grounded_external_fact",
            )
        assert result.rejected


class TestGroundedEvidenceAdmission:
    def test_external_fact_admitted_at_domain(self):
        """INV-G7: grounded_external_fact with matching fetch → admitted at DOMAIN."""
        from remy.core.agent_tools import Level
        from remy.core.ingestion import ingest_grounded_evidence
        with patch("remy.core.ingestion.get_turn_fetch_evidence", return_value=_FETCH_EV):
            result = ingest_grounded_evidence(
                content="Vitamin C supports immune function.",
                source_url="https://example.com/study",
                session_id="s1", channel="worker",
                extract_class="grounded_external_fact",
            )
        assert result.admitted
        assert result.level == Level.DOMAIN

    def test_source_extract_admitted_at_domain(self):
        """INV-G8: grounded_source_extract with matching fetch → admitted at DOMAIN."""
        from remy.core.agent_tools import Level
        from remy.core.ingestion import ingest_grounded_evidence
        with patch("remy.core.ingestion.get_turn_fetch_evidence", return_value=_FETCH_EV):
            result = ingest_grounded_evidence(
                content="Verbatim extract from source.",
                source_url="https://example.com/study",
                session_id="s1", channel="worker",
                extract_class="grounded_source_extract",
            )
        assert result.admitted
        assert result.level == Level.DOMAIN

    def test_url_canonicalisation_trailing_slash_case(self):
        """INV-G9: URL canonicalisation handles trailing slash + case."""
        from remy.core.ingestion import ingest_grounded_evidence
        fetch_ev = [{"url": "HTTPS://Example.com/Study/", "title": "", "site": "", "tool": "extract_content"}]
        with patch("remy.core.ingestion.get_turn_fetch_evidence", return_value=fetch_ev):
            result = ingest_grounded_evidence(
                content="X", source_url="https://example.com/study",
                session_id="s1", channel="worker",
                extract_class="grounded_external_fact",
            )
        assert result.admitted

    def test_admitted_has_internet_evidence_channel(self):
        """INV-G10: admitted record has learning_channel=internet_evidence."""
        from remy.core.ingestion import ingest_grounded_evidence
        with patch("remy.core.ingestion.get_turn_fetch_evidence", return_value=_FETCH_EV):
            result = ingest_grounded_evidence(
                content="X", source_url="https://example.com/study",
                session_id="s1", channel="worker",
                extract_class="grounded_external_fact",
            )
        assert result.metadata["learning_channel"] == "internet_evidence"

    def test_admitted_has_source_anchored_true(self):
        """INV-G11: admitted record has source_anchored=True."""
        from remy.core.ingestion import ingest_grounded_evidence
        with patch("remy.core.ingestion.get_turn_fetch_evidence", return_value=_FETCH_EV):
            result = ingest_grounded_evidence(
                content="X", source_url="https://example.com/study",
                session_id="s1", channel="worker",
                extract_class="grounded_external_fact",
            )
        assert result.metadata["source_anchored"] is True

    def test_admitted_has_fetch_tool_recorded(self):
        """INV-G12: admitted record has fetch_tool from the evidence item."""
        from remy.core.ingestion import ingest_grounded_evidence
        with patch("remy.core.ingestion.get_turn_fetch_evidence", return_value=_FETCH_EV):
            result = ingest_grounded_evidence(
                content="X", source_url="https://example.com/study",
                session_id="s1", channel="worker",
                extract_class="grounded_external_fact",
            )
        assert result.metadata["fetch_tool"] == "extract_content"


# ── ingest_operator_assertion ────────────────────────────────────────────────

class TestOperatorAssertionRejection:
    def test_empty_content_rejected(self):
        """INV-O1: empty content → rejected."""
        from remy.core.ingestion import ingest_operator_assertion
        result = ingest_operator_assertion(
            content="", channel="desktop", session_id="s1",
            extra_tags=["user-profile"],
        )
        assert result.rejected

    def test_non_user_direct_channel_rejected(self):
        """INV-O2: non-user-direct channel → rejected."""
        from remy.core.ingestion import ingest_operator_assertion
        result = ingest_operator_assertion(
            content="User prefers vegetarian diet.",
            channel="desktop_worker",  # worker, not user-direct
            session_id="s1",
            extra_tags=["user-profile"],
        )
        assert result.rejected
        assert "user-direct" in result.reason

    def test_missing_attribution_tag_rejected(self):
        """INV-O3: user-direct channel without attribution tag → rejected."""
        from remy.core.ingestion import ingest_operator_assertion
        result = ingest_operator_assertion(
            content="Some assertion.",
            channel="desktop",
            session_id="s1",
            extra_tags=["note"],  # no attribution tag
        )
        assert result.rejected
        assert "attribution" in result.reason


class TestOperatorAssertionAdmission:
    def test_desktop_user_profile_admitted(self):
        """INV-O4: desktop + user-profile → admitted at DOMAIN."""
        from remy.core.agent_tools import Level
        from remy.core.ingestion import ingest_operator_assertion
        result = ingest_operator_assertion(
            content="User prefers vegetarian diet.",
            channel="desktop", session_id="s1",
            extra_tags=["user-profile"],
        )
        assert result.admitted
        assert result.level == Level.DOMAIN

    def test_telegram_from_user_admitted(self):
        """INV-O5: telegram + from-user → admitted at DOMAIN."""
        from remy.core.agent_tools import Level
        from remy.core.ingestion import ingest_operator_assertion
        result = ingest_operator_assertion(
            content="Remember this as true.",
            channel="telegram", session_id="s1",
            extra_tags=["from-user"],
        )
        assert result.admitted
        assert result.level == Level.DOMAIN

    def test_admitted_has_operator_asserted_class(self):
        """INV-O6: admitted record has admission_class=operator_asserted."""
        from remy.core.ingestion import ingest_operator_assertion
        result = ingest_operator_assertion(
            content="X", channel="desktop", session_id="s1",
            extra_tags=["user-profile"],
        )
        assert result.metadata["admission_class"] == "operator_asserted"

    def test_admitted_has_operator_direct_channel(self):
        """INV-O7: admitted record has learning_channel=operator_direct."""
        from remy.core.ingestion import ingest_operator_assertion
        result = ingest_operator_assertion(
            content="X", channel="desktop", session_id="s1",
            extra_tags=["user-profile"],
        )
        assert result.metadata["learning_channel"] == "operator_direct"
