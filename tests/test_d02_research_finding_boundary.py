"""
D-02 Learning Boundary Tests: add_research_finding

Behavioral invariants:
  INV-1: No source_url → error. LLM synthesis without source cannot become a finding.
  INV-2: source_url present but not fetched this turn → error (not anchored).
  INV-3: source_url present AND fetched this turn → stored at Level.DOMAIN.
  INV-4: Stored grounded finding has learning_channel=internet_evidence.
  INV-5: Stored grounded finding has admission_class=grounded_external_fact (canonical, Phase 2).
  INV-6: Stored grounded finding has source_anchored=True.
  INV-7: research_summary artifact stores at Level.WORKING, not DOMAIN.
  INV-8: research_summary artifact has admission_class=research_artifact.
"""
import json
import pytest
from unittest.mock import MagicMock, patch


# ── shared fixtures ───────────────────────────────────────────────────────────

def _make_project_rec(project_id: str = "rp-test-d02"):
    rec = MagicMock()
    rec.id = "proj-rec-1"
    rec.metadata = {
        "type": "research_project",
        "project_id": project_id,
        "topic": "test topic",
        "status": "researching",
        "query_plan": ["q1"],
        "queries_done": 0,
        "findings_count": 0,
        "finding_ids": [],
    }
    return rec


@pytest.fixture()
def brain_mock():
    stored_rec = MagicMock()
    stored_rec.id = "finding-rec-1"
    mock = MagicMock()
    mock.store.return_value = stored_rec
    mock.get.return_value = None
    mock.update.return_value = None
    mock.connect.return_value = None
    return mock


@pytest.fixture()
def project_rec():
    return _make_project_rec()


# ── helper: run _add_research_finding via tool_handlers path ─────────────────

def _call_finding(brain_mock, project_rec, args: dict, session_id: str = "sess-d02"):
    from remy.core.tool_handlers.research import _add_research_finding
    import threading

    brain_lock = threading.Lock()

    with patch("remy.core.tool_handlers.research._get_brain", return_value=brain_mock), \
         patch("remy.core.tool_handlers.research._get_brain_lock", return_value=brain_lock), \
         patch("remy.core.tool_handlers.research._get_research_project", return_value=project_rec), \
         patch("remy.core.tool_utils._check_duplicates", return_value=[]), \
         patch("remy.core.research_memory.check_finding_contradictions", return_value=[]):
        return json.loads(_add_research_finding(args, session_id=session_id))


# ── INV-1: No source_url → error ─────────────────────────────────────────────

class TestNoSourceUrl:
    def test_no_source_url_returns_error(self, brain_mock, project_rec):
        """INV-1: Missing source_url must return an error, not store anything."""
        result = _call_finding(brain_mock, project_rec, {
            "project_id": "rp-test-d02",
            "content": "Some finding from LLM synthesis",
        })
        assert "error" in result
        brain_mock.store.assert_not_called()

    def test_no_source_url_error_mentions_fetch(self, brain_mock, project_rec):
        """INV-1: Error message must explain that fetch is required."""
        result = _call_finding(brain_mock, project_rec, {
            "project_id": "rp-test-d02",
            "content": "Some finding",
        })
        msg = result.get("error", "")
        assert "source_url" in msg or "fetch" in msg.lower()

    def test_empty_source_url_returns_error(self, brain_mock, project_rec):
        """INV-1: Empty source_url is treated same as missing."""
        result = _call_finding(brain_mock, project_rec, {
            "project_id": "rp-test-d02",
            "content": "Some finding",
            "source_url": "",
        })
        assert "error" in result
        brain_mock.store.assert_not_called()


# ── INV-2: source_url present but not fetched → error ────────────────────────

class TestSourceUrlNotAnchored:
    def test_unanchored_source_url_returns_error(self, brain_mock, project_rec):
        """INV-2: source_url without fetch evidence must return an error."""
        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=[]), \
             patch("remy.core.ingestion.get_turn_fetch_evidence", return_value=[]):
            result = _call_finding(brain_mock, project_rec, {
                "project_id": "rp-test-d02",
                "content": "Finding from candidate snippet",
                "source_url": "https://example.com/article",
            })
        assert "error" in result
        brain_mock.store.assert_not_called()

    def test_unanchored_error_mentions_extract_content(self, brain_mock, project_rec):
        """INV-2: Error must guide caller to fetch first."""
        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=[]), \
             patch("remy.core.ingestion.get_turn_fetch_evidence", return_value=[]):
            result = _call_finding(brain_mock, project_rec, {
                "project_id": "rp-test-d02",
                "content": "Finding",
                "source_url": "https://example.com/article",
            })
        msg = result.get("error", "")
        assert "extract_content" in msg or "fetch" in msg.lower() or "http_get" in msg

    def test_different_url_fetched_not_sufficient(self, brain_mock, project_rec):
        """INV-2: Fetching a different URL does not anchor the source_url."""
        fetch_ev = [{"url": "https://other.com/page", "title": "", "site": "", "tool": "extract_content"}]
        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=fetch_ev), \
             patch("remy.core.ingestion.get_turn_fetch_evidence", return_value=fetch_ev):
            result = _call_finding(brain_mock, project_rec, {
                "project_id": "rp-test-d02",
                "content": "Finding",
                "source_url": "https://example.com/article",
            })
        assert "error" in result
        brain_mock.store.assert_not_called()


# ── INV-3–6: Grounded finding → DOMAIN with full provenance ──────────────────

class TestGroundedFinding:
    """source_url present AND fetched this turn → allowed."""

    _SOURCE_URL = "https://example.com/nutrition-study"
    _FETCH_EV = [{"url": "https://example.com/nutrition-study", "title": "Nutrition", "site": "example.com", "tool": "extract_content"}]

    def _call_grounded(self, brain_mock, project_rec):
        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=self._FETCH_EV), \
             patch("remy.core.ingestion.get_turn_fetch_evidence", return_value=self._FETCH_EV):
            return _call_finding(brain_mock, project_rec, {
                "project_id": "rp-test-d02",
                "content": "Vitamin C supports immune function",
                "source_url": self._SOURCE_URL,
                "confidence": "0.85",
            })

    def test_stores_successfully(self, brain_mock, project_rec):
        """INV-3: Grounded finding must be stored."""
        result = self._call_grounded(brain_mock, project_rec)
        assert result.get("stored") is True
        brain_mock.store.assert_called_once()

    def test_stores_at_domain_level(self, brain_mock, project_rec):
        """INV-3: Must store at Level.DOMAIN."""
        from remy.core.agent_tools import Level
        self._call_grounded(brain_mock, project_rec)
        call = brain_mock.store.call_args
        assert call.kwargs["level"] == Level.DOMAIN

    def test_learning_channel_internet_evidence(self, brain_mock, project_rec):
        """INV-4: learning_channel must be internet_evidence."""
        self._call_grounded(brain_mock, project_rec)
        meta = brain_mock.store.call_args.kwargs["metadata"]
        assert meta.get("learning_channel") == "internet_evidence"

    def test_admission_class_grounded_external_fact(self, brain_mock, project_rec):
        """INV-5: admission_class must be grounded_external_fact (Phase 2 canonical name)."""
        self._call_grounded(brain_mock, project_rec)
        meta = brain_mock.store.call_args.kwargs["metadata"]
        assert meta.get("admission_class") == "grounded_external_fact"

    def test_source_anchored_true(self, brain_mock, project_rec):
        """INV-6: source_anchored must be True."""
        self._call_grounded(brain_mock, project_rec)
        meta = brain_mock.store.call_args.kwargs["metadata"]
        assert meta.get("source_anchored") is True

    def test_source_url_in_metadata(self, brain_mock, project_rec):
        """INV-3: source_url must be stored in metadata."""
        self._call_grounded(brain_mock, project_rec)
        meta = brain_mock.store.call_args.kwargs["metadata"]
        assert meta.get("source_url") == self._SOURCE_URL

    def test_research_finding_tag_present(self, brain_mock, project_rec):
        """INV-3: research-finding tag must be present."""
        self._call_grounded(brain_mock, project_rec)
        tags = brain_mock.store.call_args.kwargs["tags"]
        assert "research-finding" in tags


# ── INV-7–8: research_summary artifact → WORKING ─────────────────────────────

class TestResearchSummaryArtifact:
    """_store_research_summary_artifact must store at WORKING, not DOMAIN."""

    def test_stores_at_working_level(self):
        """INV-7: Summary artifact must be at Level.WORKING / L1_WORKING."""
        brain_mock = MagicMock()
        brain_mock.store.return_value = MagicMock(id="summary-rec-1")

        with patch("remy.core.agent_tools.brain", brain_mock):
            from remy.core.workers.research_worker import _store_research_summary_artifact
            _store_research_summary_artifact(
                goal={"goal_id": "g1", "goal_template": "research"},
                session_summary={"findings_count": 3, "accepted_sources_count": 2,
                                 "research_mode": "web", "source_scope": "public"},
                findings=[{"summary": "Finding 1", "source_url": "https://example.com/1"}],
            )

        assert brain_mock.store.called
        call = brain_mock.store.call_args
        level_arg = call.args[1] if len(call.args) > 1 else call.kwargs.get("level")
        # Stored as L1_WORKING string level
        assert "WORKING" in str(level_arg).upper() or level_arg == "L1_WORKING"

    def test_admission_class_research_artifact(self):
        """INV-8: admission_class must be research_artifact."""
        brain_mock = MagicMock()
        brain_mock.store.return_value = MagicMock(id="summary-rec-1")

        with patch("remy.core.agent_tools.brain", brain_mock):
            from remy.core.workers.research_worker import _store_research_summary_artifact
            _store_research_summary_artifact(
                goal={"goal_id": "g1", "goal_template": "research"},
                session_summary={"findings_count": 1, "accepted_sources_count": 1,
                                 "research_mode": "web", "source_scope": "public"},
                findings=[],
            )

        call = brain_mock.store.call_args
        meta = call.args[2] if len(call.args) > 2 else call.kwargs.get("metadata", {})
        assert meta.get("admission_class") == "research_artifact"

    def test_requires_promotion_flag_set(self):
        """INV-8: requires_promotion=True must be set on summary artifact."""
        brain_mock = MagicMock()
        brain_mock.store.return_value = MagicMock(id="summary-rec-1")

        with patch("remy.core.agent_tools.brain", brain_mock):
            from remy.core.workers.research_worker import _store_research_summary_artifact
            _store_research_summary_artifact(
                goal={"goal_id": "g1", "goal_template": "research"},
                session_summary={"findings_count": 1, "accepted_sources_count": 1,
                                 "research_mode": "web", "source_scope": "public"},
                findings=[],
            )

        call = brain_mock.store.call_args
        meta = call.args[2] if len(call.args) > 2 else call.kwargs.get("metadata", {})
        assert meta.get("requires_promotion") is True

    def test_no_domain_level(self):
        """INV-7: Level must NOT be DOMAIN."""
        brain_mock = MagicMock()
        brain_mock.store.return_value = MagicMock(id="summary-rec-1")

        with patch("remy.core.agent_tools.brain", brain_mock):
            from remy.core.workers.research_worker import _store_research_summary_artifact
            _store_research_summary_artifact(
                goal={"goal_id": "g1", "goal_template": "research"},
                session_summary={"findings_count": 1, "accepted_sources_count": 0,
                                 "research_mode": "web", "source_scope": "public"},
                findings=[],
            )

        call = brain_mock.store.call_args
        level_arg = call.args[1] if len(call.args) > 1 else call.kwargs.get("level")
        assert "DOMAIN" not in str(level_arg).upper()


# ── brain_tools.py duplicate path ────────────────────────────────────────────

class TestBrainToolsDuplicatePath:
    """brain_tools._add_research_finding must also enforce the same boundary."""

    def test_no_source_url_returns_error(self):
        """D-02: brain_tools path must reject missing source_url."""
        from remy.core.brain_tools import _add_research_finding as bt_finding
        result = json.loads(bt_finding({
            "project_id": "rp-x",
            "content": "Some synthesized finding",
        }))
        assert "error" in result

    def test_unanchored_source_url_returns_error(self):
        """D-02: brain_tools path must reject unanchored source_url."""
        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=[]), \
             patch("remy.core.ingestion.get_turn_fetch_evidence", return_value=[]):
            from remy.core.brain_tools import _add_research_finding as bt_finding
            result = json.loads(bt_finding({
                "project_id": "rp-x",
                "content": "Some finding",
                "source_url": "https://example.com/page",
            }, session_id="sess-bt"))
        assert "error" in result
