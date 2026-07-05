"""Tests for RM-1: Research Orchestrator — start_research, add_research_finding, complete_research."""

import json
from unittest.mock import patch, MagicMock

import pytest
from aura import Aura as CognitiveMemory, Level


# ============== start_research ==============

class TestStartResearch:

    def test_creates_project_with_llm_plan(self, tmp_path):
        """start_research creates a project with LLM-generated query plan."""
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.brain_tools.brain", b), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.tool_registry.settings") as ms:
            ms.SANDBOX_DIR = tmp_path / "sandbox"
            ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"
            ms.GEMINI_API_KEY = "fake-key"
            ms.SUMMARY_MODEL = "test-model"

            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(
                content='["What is vitamin D?", "Vitamin D deficiency symptoms", "Vitamin D sources"]'
            )

            with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_llm):
                from remy.core.brain_tools import _start_research

                result = json.loads(_start_research({"topic": "Vitamin D benefits", "depth": "standard"}))

                assert result["created"] is True
                assert result["topic"] == "Vitamin D benefits"
                assert result["depth"] == "standard"
                assert len(result["query_plan"]) >= 1
                assert result["project_id"].startswith("rp-")
        b.close()

    def test_creates_project_with_fallback_on_llm_failure(self, tmp_path):
        """start_research falls back to topic as query when LLM fails."""
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.brain_tools.brain", b), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.tool_registry.settings") as ms:
            ms.SANDBOX_DIR = tmp_path / "sandbox"
            ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"
            ms.GEMINI_API_KEY = "fake-key"
            ms.SUMMARY_MODEL = "test-model"

            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = Exception("LLM unavailable")

            with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_llm):
                from remy.core.brain_tools import _start_research

                result = json.loads(_start_research({"topic": "Sleep quality"}))

                assert result["created"] is True
                assert "Sleep quality" in result["query_plan"]
        b.close()

    def test_depth_quick_generates_2_queries(self, tmp_path):
        """Quick depth generates 2 queries."""
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.brain_tools.brain", b), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.tool_registry.settings") as ms:
            ms.SANDBOX_DIR = tmp_path / "sandbox"
            ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"
            ms.GEMINI_API_KEY = "fake-key"
            ms.SUMMARY_MODEL = "test-model"

            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(
                content='["query one", "query two"]'
            )

            with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_llm):
                from remy.core.brain_tools import _start_research

                result = json.loads(_start_research({"topic": "test", "depth": "quick"}))
                assert result["depth"] == "quick"
                assert result["queries_total"] == 2
        b.close()

    def test_project_stored_in_brain(self, tmp_path):
        """Project is stored as brain record with correct tags."""
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.brain_tools.brain", b), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.tool_registry.settings") as ms:
            ms.SANDBOX_DIR = tmp_path / "sandbox"
            ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"
            ms.GEMINI_API_KEY = "fake-key"
            ms.SUMMARY_MODEL = "test-model"

            with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_cls:
                mock_cls.return_value.invoke.side_effect = Exception("skip LLM")

                from remy.core.brain_tools import _start_research, _RESEARCH_PROJECT_TAG

                result = json.loads(_start_research({"topic": "test project"}))
                assert result["created"] is True

                records = b.search(query="", tags=[_RESEARCH_PROJECT_TAG], limit=5)
                assert len(records) >= 1
                meta = records[0].metadata
                assert meta["project_id"] == result["project_id"]
                assert meta["status"] == "researching"
        b.close()


# ============== add_research_finding ==============

class TestAddResearchFinding:

    def _create_project(self, brain_instance):
        """Helper: create a minimal research project."""
        rec = brain_instance.store(
            content="Research Project: test topic",
            level=Level.DOMAIN,
            tags=["research-project", "test-topic"],
            metadata={
                "type": "research_project",
                "project_id": "rp-test123",
                "topic": "test topic",
                "status": "researching",
                "query_plan": ["query1"],
                "queries_done": 0,
                "findings_count": 0,
                "finding_ids": [],
            },
        )
        return rec

    def test_adds_finding_to_project(self, tmp_path):
        """Finding is stored and connected to project when source_url is anchored."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        self._create_project(b)

        fetch_ev = [{"url": "https://example.com/article", "title": "", "site": "example.com", "tool": "extract_content"}]
        with patch("remy.core.brain_tools.brain", b), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.tool_registry.settings") as ms, \
             patch("remy.core.ingestion.get_turn_fetch_evidence", return_value=fetch_ev):
            ms.SANDBOX_DIR = tmp_path / "sandbox"
            ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"

            from remy.core.brain_tools import _add_research_finding

            result = json.loads(_add_research_finding({
                "project_id": "rp-test123",
                "content": "Vitamin D helps with calcium absorption",
                "source_url": "https://example.com/article",
                "confidence": "0.9",
            }, session_id="sess-test"))

            assert result["stored"] is True
            assert result["findings_count"] == 1
            assert result["project_id"] == "rp-test123"
        b.close()

    def test_increments_findings_count(self, tmp_path):
        """Without source_url, finding is rejected (D-02 boundary enforced)."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        self._create_project(b)

        with patch("remy.core.brain_tools.brain", b), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.tool_registry.settings") as ms:
            ms.SANDBOX_DIR = tmp_path / "sandbox"
            ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"

            from remy.core.brain_tools import _add_research_finding

            # Without source_url → error (D-02: source_url is required)
            result1 = json.loads(_add_research_finding({"project_id": "rp-test123", "content": "Finding 1"}))
            assert "error" in result1

            result2 = json.loads(_add_research_finding({"project_id": "rp-test123", "content": "Finding 2"}))
            assert "error" in result2
        b.close()

    def test_error_on_unknown_project(self, tmp_path):
        """Returns error for nonexistent project."""
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.brain_tools.brain", b), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.tool_registry.settings") as ms:
            ms.SANDBOX_DIR = tmp_path / "sandbox"
            ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"

            from remy.core.brain_tools import _add_research_finding

            result = json.loads(_add_research_finding({
                "project_id": "rp-nonexistent", "content": "something"
            }))

            assert "error" in result
        b.close()

    def test_contradiction_creates_connection(self, tmp_path):
        """Contradicting finding creates connection to contradicted finding (source anchored)."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        self._create_project(b)

        fetch_ev = [
            {"url": "https://who.int/vitd", "title": "", "site": "who.int", "tool": "extract_content"},
            {"url": "https://pubmed.ncbi.nlm.nih.gov/trial2024", "title": "", "site": "pubmed.ncbi.nlm.nih.gov", "tool": "extract_content"},
        ]
        with patch("remy.core.brain_tools.brain", b), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.tool_registry.settings") as ms, \
             patch("remy.core.ingestion.get_turn_fetch_evidence", return_value=fetch_ev):
            ms.SANDBOX_DIR = tmp_path / "sandbox"
            ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"

            from remy.core.brain_tools import _add_research_finding

            first = json.loads(_add_research_finding({
                "project_id": "rp-test123",
                "content": "Vitamin D optimal dose is 1000 IU daily for bone health according to WHO guidelines",
                "source_url": "https://who.int/vitd",
            }, session_id="sess-test"))
            first_id = first["finding_id"]

            second = json.loads(_add_research_finding({
                "project_id": "rp-test123",
                "content": "New clinical trial suggests 4000 IU of Vitamin D is required for immune system support in northern latitudes",
                "source_url": "https://pubmed.ncbi.nlm.nih.gov/trial2024",
                "contradicts_finding_id": first_id,
            }, session_id="sess-test"))

            assert second["contradicts"] == first_id
            # Check connection exists
            second_rec = b.get(second["finding_id"])
            assert first_id in second_rec.connections
        b.close()


# ============== complete_research ==============

class TestCompleteResearch:

    def _setup_project_with_findings(self, brain_instance):
        """Helper: create a project and add findings."""
        proj = brain_instance.store(
            content="Research Project: sleep quality",
            level=Level.DOMAIN,
            tags=["research-project"],
            metadata={
                "type": "research_project",
                "project_id": "rp-sleep",
                "topic": "sleep quality",
                "status": "researching",
                "query_plan": ["sleep tips"],
                "queries_done": 1,
                "findings_count": 0,
                "finding_ids": [],
            },
        )

        findings = []
        for text in [
            "Adults need 7-9 hours of sleep",
            "Blue light before bed disrupts melatonin",
        ]:
            f = brain_instance.store(
                content=f"Research finding (sleep quality): {text}",
                level=Level.DOMAIN,
                tags=["research-finding"],
                metadata={
                    "type": "research_finding",
                    "project_id": "rp-sleep",
                    "source_url": "https://example.com",
                    "confidence": 0.8,
                },
            )
            findings.append(f)
            brain_instance.connect(f.id, proj.id, weight=0.8)

        # Update project metadata
        meta = dict(proj.metadata)
        meta["finding_ids"] = [f.id for f in findings]
        meta["findings_count"] = len(findings)
        brain_instance.update(proj.id, metadata=meta)

        return proj, findings

    def test_synthesizes_report(self, tmp_path):
        """complete_research generates a report from findings."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        proj, findings = self._setup_project_with_findings(b)

        with patch("remy.core.brain_tools.brain", b), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.tool_registry.settings") as ms:
            ms.SANDBOX_DIR = tmp_path / "sandbox"
            ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"
            ms.GEMINI_API_KEY = "fake-key"
            ms.SUMMARY_MODEL = "test-model"

            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(
                content="Sleep quality depends on 7-9h of sleep and avoiding blue light before bed."
            )

            with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_llm):
                from remy.core.brain_tools import _complete_research

                result = json.loads(_complete_research({"project_id": "rp-sleep"}))

                assert result["completed"] is True
                assert result["verification"]["status"] == "verified"
                assert result["verification"]["verified"] is True
                assert result["findings_count"] == 2
                assert result["source_count"] >= 1
                assert "sleep" in result["report"].lower() or "Sleep" in result["report"]
                assert result["confidence_avg"] == 0.8
        b.close()

    def test_marks_project_complete(self, tmp_path):
        """Project status is set to 'complete' after synthesis."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        proj, findings = self._setup_project_with_findings(b)

        with patch("remy.core.brain_tools.brain", b), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.tool_registry.settings") as ms:
            ms.SANDBOX_DIR = tmp_path / "sandbox"
            ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"
            ms.GEMINI_API_KEY = "fake-key"
            ms.SUMMARY_MODEL = "test-model"

            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(content="Summary report.")

            with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_llm):
                from remy.core.brain_tools import _complete_research

                _complete_research({"project_id": "rp-sleep"})

                updated = b.get(proj.id)
                assert updated.metadata["status"] == "complete"
                assert "completed_at" in updated.metadata
        b.close()

    def test_research_verification_gate_blocks_completion(self, tmp_path):
        """Project should not flip to complete if research verification fails."""
        from remy.core.verification_gate import VerificationResult, VerificationStatus

        b = CognitiveMemory(str(tmp_path / "brain"))
        proj, findings = self._setup_project_with_findings(b)

        with patch("remy.core.brain_tools.brain", b), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.tool_registry.settings") as ms:
            ms.SANDBOX_DIR = tmp_path / "sandbox"
            ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"
            ms.GEMINI_API_KEY = "fake-key"
            ms.SUMMARY_MODEL = "test-model"

            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(content="Summary report.")

            with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_llm), \
                 patch(
                     "remy.core.verification_gate.run_research_completion_verification_gate",
                     return_value=VerificationResult(
                         status=VerificationStatus.REPAIR_REQUIRED.value,
                         verified=False,
                         failure_code="verification_failed",
                         reason="Research artifact did not pass verification.",
                         artifact_ids=["broken-report"],
                         repair_required=True,
                     ),
                 ):
                from remy.core.brain_tools import _complete_research

                result = json.loads(_complete_research({"project_id": "rp-sleep"}))

                assert result["completed"] is False
                assert result["verification"]["status"] == "repair_required"
                assert result["verification"]["failure_code"] == "verification_failed"

                updated = b.get(proj.id)
                assert updated.metadata["status"] == "researching"
                assert "completed_at" not in updated.metadata
        b.close()

    def test_error_on_unknown_project(self, tmp_path):
        """Returns error for nonexistent project."""
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.brain_tools.brain", b), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.tool_registry.settings") as ms:
            ms.SANDBOX_DIR = tmp_path / "sandbox"
            ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"

            from remy.core.brain_tools import _complete_research

            result = json.loads(_complete_research({"project_id": "rp-nonexistent"}))
            assert "error" in result
        b.close()

    def test_error_on_no_findings(self, tmp_path):
        """Returns error when project has no findings."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        b.store(
            content="Research Project: empty",
            level=Level.DOMAIN,
            tags=["research-project"],
            metadata={
                "type": "research_project",
                "project_id": "rp-empty",
                "topic": "empty project",
                "status": "researching",
                "finding_ids": [],
                "findings_count": 0,
            },
        )

        with patch("remy.core.brain_tools.brain", b), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.tool_registry.settings") as ms:
            ms.SANDBOX_DIR = tmp_path / "sandbox"
            ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"

            from remy.core.brain_tools import _complete_research

            result = json.loads(_complete_research({"project_id": "rp-empty"}))
            assert "error" in result
        b.close()

    def test_fallback_on_llm_failure(self, tmp_path):
        """Falls back to concatenating findings when LLM fails."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        proj, findings = self._setup_project_with_findings(b)

        with patch("remy.core.brain_tools.brain", b), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.tool_registry.settings") as ms:
            ms.SANDBOX_DIR = tmp_path / "sandbox"
            ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"
            ms.GEMINI_API_KEY = "fake-key"
            ms.SUMMARY_MODEL = "test-model"

            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = Exception("LLM down")

            with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_llm):
                from remy.core.brain_tools import _complete_research

                result = json.loads(_complete_research({"project_id": "rp-sleep"}))

                assert result["completed"] is True
                assert result["findings_count"] == 2
        b.close()

    def test_stores_report_in_knowledge_base(self, tmp_path):
        """Report is stored in Knowledge Base (Aura Memory) if available."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        proj, findings = self._setup_project_with_findings(b)

        with patch("remy.core.brain_tools.brain", b), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.tool_registry.settings") as ms:
            ms.SANDBOX_DIR = tmp_path / "sandbox"
            ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"
            ms.GEMINI_API_KEY = "fake-key"
            ms.SUMMARY_MODEL = "test-model"

            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(content="Summary report.")

            # Mock Aura Memory (knowledge)
            mock_knowledge = MagicMock()
            
            with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_llm), \
                 patch("remy.core.agent_tools.knowledge", mock_knowledge):
                
                from remy.core.brain_tools import _complete_research

                _complete_research({"project_id": "rp-sleep"})

                # Verify knowledge.process was called with pin=True
                assert mock_knowledge.process.call_count == 1
                args, kwargs = mock_knowledge.process.call_args
                assert "Summary report" in args[0]
                assert kwargs.get("pin") is True
                assert mock_knowledge.flush.called
        b.close()


# ============== get_active_research_projects ==============

class TestGetActiveResearchProjects:

    def test_returns_active_projects(self, tmp_path):
        """Lists only non-complete research projects."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        b.store(
            content="Research Project: active one",
            level=Level.DOMAIN,
            tags=["research-project"],
            metadata={
                "project_id": "rp-active",
                "topic": "active topic",
                "status": "researching",
                "depth": "standard",
                "query_plan": ["q1", "q2"],
                "queries_done": 1,
                "findings_count": 2,
            },
        )
        b.store(
            content="Research Project: done one",
            level=Level.DOMAIN,
            tags=["research-project"],
            metadata={
                "project_id": "rp-done",
                "topic": "done topic",
                "status": "complete",
            },
        )

        with patch("remy.core.brain_tools.brain", b):
            from remy.core.brain_tools import get_active_research_projects

            projects = get_active_research_projects()
            assert len(projects) == 1
            assert projects[0]["project_id"] == "rp-active"
            assert projects[0]["queries_done"] == 1
            assert projects[0]["findings_count"] == 2
        b.close()

    def test_returns_empty_when_no_projects(self, tmp_path):
        """Returns empty list when no research projects exist."""
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.brain_tools.brain", b):
            from remy.core.brain_tools import get_active_research_projects

            projects = get_active_research_projects()
            assert projects == []
        b.close()


# ============== Consolidation skips research tags ==============

class TestResearchTagsSkipped:

    def test_research_project_tag_in_skip_list(self):
        """research-project tag should be in _CONSOLIDATION_SKIP_TAGS."""
        from remy.core.background_brain import _CONSOLIDATION_SKIP_TAGS
        assert "research-project" in _CONSOLIDATION_SKIP_TAGS

    def test_research_finding_tag_in_skip_list(self):
        """research-finding tag should be in _CONSOLIDATION_SKIP_TAGS."""
        from remy.core.background_brain import _CONSOLIDATION_SKIP_TAGS
        assert "research-finding" in _CONSOLIDATION_SKIP_TAGS


# ============== execute_tool integration ==============

class TestExecuteToolResearch:

    def test_start_research_via_execute_tool(self, tmp_path):
        """start_research is reachable through execute_tool."""
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.brain_tools.brain", b), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.brain_tools.tool_health") as mh, \
             patch("remy.core.tool_registry.settings") as ms:
            ms.SANDBOX_DIR = tmp_path / "sandbox"
            ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"
            ms.GEMINI_API_KEY = "fake-key"
            ms.SUMMARY_MODEL = "test-model"
            mh.is_available.return_value = True

            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = Exception("skip")

            with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_llm):
                from remy.core.brain_tools import execute_tool

                result = json.loads(execute_tool("start_research", {"topic": "test integration"}))
                assert result["created"] is True
        b.close()

    def test_start_research_accepts_description_alias(self, tmp_path):
        """start_research accepts common alias fields instead of topic."""
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.brain_tools.brain", b), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.brain_tools.tool_health") as mh, \
             patch("remy.core.tool_registry.settings") as ms:
            ms.SANDBOX_DIR = tmp_path / "sandbox"
            ms.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"
            ms.GEMINI_API_KEY = "fake-key"
            ms.SUMMARY_MODEL = "test-model"
            mh.is_available.return_value = True

            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = Exception("skip")

            with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_llm):
                from remy.core.brain_tools import execute_tool

                result = json.loads(
                    execute_tool(
                        "start_research",
                        {"description": "enterprise compliance requirements for AuraSDK"},
                    )
                )
                assert result["created"] is True
                assert result["topic"] == "enterprise compliance requirements for AuraSDK"
        b.close()
