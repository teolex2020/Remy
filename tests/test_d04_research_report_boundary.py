"""
D-04 Research Report Boundary Tests

Behavioral invariants:
  INV-1: complete_research report has admission_class=research_report.
  INV-2: complete_research report has requires_promotion=True.
  INV-3: complete_research report is NOT stored at Level.DOMAIN (stored at DECISIONS).
  INV-4: start_research project record has admission_class=research_project.
  INV-5: store_research without fetch → stored at DECISIONS, not DOMAIN.
  INV-6: store_research without fetch → requires_promotion=True.
  INV-7: store_research with fetch → stored at DOMAIN.
  INV-8: _store_research_summary_artifact (worker) → Level.WORKING, not DOMAIN.
  INV-9: _store_research_summary_artifact → admission_class=research_artifact.
  INV-10: _store_research_summary_artifact → requires_promotion=True.
"""
import json
import threading
from unittest.mock import MagicMock, patch


_FETCH_EV = [{"url": "https://example.com/study", "title": "Study",
              "site": "example.com", "tool": "extract_content"}]

_REAL_LOCK = threading.Lock()


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_project_rec(project_id: str = "rp-d04"):
    rec = MagicMock()
    rec.id = "proj-rec-d04"
    rec.metadata = {
        "type": "research_project",
        "project_id": project_id,
        "topic": "test topic",
        "status": "researching",
        "query_plan": ["q1"],
        "queries_done": 1,
        "findings_count": 1,
        "finding_ids": ["finding-1"],
        "research_mode": "web",
        "source_scope": "public",
        "source_domains": [],
    }
    return rec


def _make_finding_rec():
    rec = MagicMock()
    rec.id = "finding-1"
    rec.content = "Vitamin C supports immune function"
    rec.metadata = {
        "source_url": "https://example.com/study",
        "source_anchored": True,
        "confidence": 0.85,
        "admission_class": "grounded_external_fact",
    }
    return rec


def _make_brain_mock_with_finding():
    stored_rec = MagicMock()
    stored_rec.id = "report-rec-d04"
    finding_rec = _make_finding_rec()
    mock = MagicMock()
    mock.store.return_value = stored_rec
    mock.get.side_effect = lambda record_id: finding_rec if record_id == "finding-1" else stored_rec
    mock.search.return_value = []
    mock.update.return_value = None
    mock.connect.return_value = None
    return mock


def _level_from_call(call):
    return call.kwargs.get("level") or (call.args[1] if len(call.args) > 1 else None)


def _meta_from_call(call):
    return call.kwargs.get("metadata") or (call.args[2] if len(call.args) > 2 else {}) or {}


# ── INV-1–3: complete_research report ────────────────────────────────────────

class TestCompleteResearchReport:
    def _run_complete(self, brain_mock):
        from remy.core.tool_handlers.research import _complete_research
        project_rec = _make_project_rec()

        _dummy_verification = MagicMock()
        _dummy_verification.verified = True
        _dummy_verification.repair_required = False
        _dummy_verification.to_dict.return_value = {"verified": True}

        with patch("remy.core.tool_handlers.research._get_brain", return_value=brain_mock), \
             patch("remy.core.tool_handlers.research._get_brain_lock", return_value=_REAL_LOCK), \
             patch("remy.core.tool_handlers.research._get_research_project", return_value=project_rec), \
             patch("remy.core.verification_gate.run_research_completion_verification_gate",
                   return_value=_dummy_verification), \
             patch("remy.core.verification_gate.emit_verification_incident"), \
             patch("remy.core.verification_gate.resolve_verification_incident"), \
             patch("remy.core.llm.call_llm",
                   return_value=MagicMock(content="Synthesized report text.")):
            return json.loads(_complete_research(
                {"project_id": "rp-d04"}, session_id="sess-d04"
            ))

    def test_report_has_admission_class(self):
        """INV-1: complete_research report must have admission_class=research_report."""
        brain_mock = _make_brain_mock_with_finding()
        self._run_complete(brain_mock)
        meta = _meta_from_call(brain_mock.store.call_args)
        assert meta.get("admission_class") == "research_report"

    def test_report_has_requires_promotion(self):
        """INV-2: complete_research report must have requires_promotion=True."""
        brain_mock = _make_brain_mock_with_finding()
        self._run_complete(brain_mock)
        meta = _meta_from_call(brain_mock.store.call_args)
        assert meta.get("requires_promotion") is True

    def test_report_not_at_domain(self):
        """INV-3: complete_research report must NOT be stored at Level.DOMAIN."""
        from remy.core.agent_tools import Level
        brain_mock = _make_brain_mock_with_finding()
        self._run_complete(brain_mock)
        level = _level_from_call(brain_mock.store.call_args)
        assert level != Level.DOMAIN, f"Expected not DOMAIN, got {level}"

    def test_report_at_decisions(self):
        """INV-3: complete_research report must be stored at Level.DECISIONS."""
        from remy.core.agent_tools import Level
        brain_mock = _make_brain_mock_with_finding()
        self._run_complete(brain_mock)
        level = _level_from_call(brain_mock.store.call_args)
        assert level == Level.DECISIONS


# ── INV-4: start_research project record ─────────────────────────────────────

class TestStartResearchProject:
    def test_project_record_has_admission_class(self):
        """INV-4: start_research project record must have admission_class=research_project."""
        from remy.core.tool_handlers.research import _start_research
        brain_mock = MagicMock()
        stored_rec = MagicMock()
        stored_rec.id = "proj-rec-new"
        brain_mock.store.return_value = stored_rec
        brain_mock.search.return_value = []

        with patch("remy.core.tool_handlers.research._get_brain", return_value=brain_mock), \
             patch("remy.core.tool_handlers.research._get_brain_lock", return_value=_REAL_LOCK), \
             patch("remy.core.llm.call_llm",
                   return_value=MagicMock(content='["query 1", "query 2"]')):
            result = json.loads(_start_research(
                {"project_id": "rp-new", "topic": "vitamin c research", "depth": "quick"},
                session_id="sess-d04"
            ))

        assert result.get("created") is True
        meta = _meta_from_call(brain_mock.store.call_args)
        assert meta.get("admission_class") == "research_project"


# ── INV-5–7: store_research tool ─────────────────────────────────────────────

class TestStoreResearch:
    def _call_store_research(self, brain_mock, fetch_ev=None, args_override=None):
        from remy.core.brain_tools import _execute_tool_inner
        args = {
            "topic": "vitamin c benefits",
            "findings": "Vitamin C supports immunity and wound healing.",
            "sources": "https://example.com/study",
        }
        if args_override:
            args.update(args_override)

        fetch_ev = fetch_ev if fetch_ev is not None else []

        with patch("remy.core.brain_tools.brain", brain_mock), \
             patch("remy.core.brain_tools.brain_lock", _REAL_LOCK), \
             patch("remy.core.tool_utils._check_duplicates", return_value=[]), \
             patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=fetch_ev):
            return json.loads(_execute_tool_inner(
                "store_research", args, session_id="sess-d04", channel="desktop_worker"
            ))

    def test_no_fetch_stored_at_decisions(self):
        """INV-5: store_research without fetch → Level.DECISIONS."""
        from remy.core.agent_tools import Level
        brain_mock = MagicMock()
        brain_mock.store.return_value = MagicMock(id="sr-1")
        brain_mock.search.return_value = []
        self._call_store_research(brain_mock, fetch_ev=[])
        level = _level_from_call(brain_mock.store.call_args)
        assert level == Level.DECISIONS

    def test_no_fetch_not_domain(self):
        """INV-5: store_research without fetch → NOT Level.DOMAIN."""
        from remy.core.agent_tools import Level
        brain_mock = MagicMock()
        brain_mock.store.return_value = MagicMock(id="sr-1")
        brain_mock.search.return_value = []
        self._call_store_research(brain_mock, fetch_ev=[])
        level = _level_from_call(brain_mock.store.call_args)
        assert level != Level.DOMAIN

    def test_no_fetch_requires_promotion(self):
        """INV-6: store_research without fetch → requires_promotion=True."""
        brain_mock = MagicMock()
        brain_mock.store.return_value = MagicMock(id="sr-1")
        brain_mock.search.return_value = []
        self._call_store_research(brain_mock, fetch_ev=[])
        meta = _meta_from_call(brain_mock.store.call_args)
        assert meta.get("requires_promotion") is True

    def test_no_fetch_has_admission_class(self):
        """INV-6: store_research without fetch → admission_class=research_report."""
        brain_mock = MagicMock()
        brain_mock.store.return_value = MagicMock(id="sr-1")
        brain_mock.search.return_value = []
        self._call_store_research(brain_mock, fetch_ev=[])
        meta = _meta_from_call(brain_mock.store.call_args)
        assert meta.get("admission_class") == "research_report"

    def test_with_fetch_stored_at_domain(self):
        """INV-7: store_research with fetch evidence → Level.DOMAIN."""
        from remy.core.agent_tools import Level
        brain_mock = MagicMock()
        brain_mock.store.return_value = MagicMock(id="sr-2")
        brain_mock.search.return_value = []
        self._call_store_research(brain_mock, fetch_ev=_FETCH_EV)
        level = _level_from_call(brain_mock.store.call_args)
        assert level == Level.DOMAIN


# ── INV-8–10: research_worker summary artifact ────────────────────────────────

class TestResearchWorkerArtifact:
    def _run_summary_artifact(self, brain_mock):
        with patch("remy.core.agent_tools.brain", brain_mock):
            from remy.core.workers.research_worker import _store_research_summary_artifact
            _store_research_summary_artifact(
                goal={"goal_id": "g-d04", "goal_template": "research"},
                session_summary={"findings_count": 2, "accepted_sources_count": 2,
                                 "research_mode": "web", "source_scope": "public"},
                findings=[{"summary": "Finding 1", "source_url": "https://example.com/1"}],
            )

    def test_stored_at_working_not_domain(self):
        """INV-8: summary artifact must NOT be at Level.DOMAIN."""
        brain_mock = MagicMock()
        brain_mock.store.return_value = MagicMock(id="sa-1")
        self._run_summary_artifact(brain_mock)
        call = brain_mock.store.call_args
        level_arg = call.args[1] if len(call.args) > 1 else call.kwargs.get("level")
        assert "DOMAIN" not in str(level_arg).upper()

    def test_stored_at_working_level(self):
        """INV-8: summary artifact must be at WORKING level."""
        brain_mock = MagicMock()
        brain_mock.store.return_value = MagicMock(id="sa-1")
        self._run_summary_artifact(brain_mock)
        call = brain_mock.store.call_args
        level_arg = call.args[1] if len(call.args) > 1 else call.kwargs.get("level")
        assert "WORKING" in str(level_arg).upper() or level_arg == "L1_WORKING"

    def test_admission_class_research_artifact(self):
        """INV-9: summary artifact must have admission_class=research_artifact."""
        brain_mock = MagicMock()
        brain_mock.store.return_value = MagicMock(id="sa-1")
        self._run_summary_artifact(brain_mock)
        call = brain_mock.store.call_args
        meta = call.args[2] if len(call.args) > 2 else call.kwargs.get("metadata", {})
        assert meta.get("admission_class") == "research_artifact"

    def test_requires_promotion_true(self):
        """INV-10: summary artifact must have requires_promotion=True."""
        brain_mock = MagicMock()
        brain_mock.store.return_value = MagicMock(id="sa-1")
        self._run_summary_artifact(brain_mock)
        call = brain_mock.store.call_args
        meta = call.args[2] if len(call.args) > 2 else call.kwargs.get("metadata", {})
        assert meta.get("requires_promotion") is True
