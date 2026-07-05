"""Tests for Anti-Hallucination Storage Guards — code-level protection against
autonomous agent fabricating people, stories, tracked metrics, or overwriting
verified facts."""

import json
import threading
from unittest.mock import patch

import pytest


@pytest.fixture
def mock_brain(tmp_path):
    """Real CognitiveMemory for integration testing."""
    from remy.core.agent_tools import _AuraCompat as Aura

    b = Aura(str(tmp_path / "test_brain"))
    yield b
    b.close()


@pytest.fixture
def execute_tool(mock_brain, tmp_path):
    """Provide execute_tool with mocked brain."""
    with (
        patch("remy.core.brain_tools.brain", mock_brain),
        patch("remy.core.brain_tools.brain_lock", threading.RLock()),
        patch("remy.core.brain_tools._registry", None),
        patch("remy.core.tool_registry.settings") as mock_settings,
    ):
        mock_settings.SANDBOX_DIR = tmp_path / "sandbox"
        mock_settings.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"

        from remy.core.brain_tools import execute_tool

        yield execute_tool


# ============== store_person guards ==============


class TestStorePersonGuard:
    def test_blocked_in_autonomous(self, execute_tool):
        result = json.loads(
            execute_tool(
                "store_person",
                {"full_name": "Fake Person", "role": "uncle"},
                channel="autonomous",
            )
        )
        assert "error" in result
        assert "interactive" in result["error"].lower()

    def test_blocked_in_worker(self, execute_tool):
        result = json.loads(
            execute_tool(
                "store_person",
                {"full_name": "Fake Person"},
                channel="worker-researcher",
            )
        )
        assert "error" in result

    def test_blocked_in_proactive(self, execute_tool):
        result = json.loads(
            execute_tool(
                "store_person",
                {"full_name": "Fake Person"},
                channel="proactive",
            )
        )
        assert "error" in result

    def test_allowed_in_desktop(self, execute_tool):
        result = json.loads(
            execute_tool(
                "store_person",
                {"full_name": "Real Person", "role": "friend"},
                channel="desktop",
            )
        )
        assert result.get("stored") is True
        assert result["name"] == "Real Person"

    def test_allowed_in_telegram(self, execute_tool):
        result = json.loads(
            execute_tool(
                "store_person",
                {"full_name": "Real Person"},
                channel="telegram",
            )
        )
        assert result.get("stored") is True

    def test_allowed_with_no_channel(self, execute_tool):
        """Default channel=None (tests, interactive) should work."""
        result = json.loads(
            execute_tool(
                "store_person",
                {"full_name": "Test Person"},
            )
        )
        assert result.get("stored") is True


# ============== store_story guards ==============


class TestStoreStoryGuard:
    def test_blocked_in_autonomous(self, execute_tool):
        result = json.loads(
            execute_tool(
                "store_story",
                {"title": "Fake Event", "content": "Never happened"},
                channel="autonomous",
            )
        )
        assert "error" in result

    def test_blocked_in_proactive(self, execute_tool):
        result = json.loads(
            execute_tool(
                "store_story",
                {"title": "Fake Event", "content": "Never happened"},
                channel="proactive",
            )
        )
        assert "error" in result

    def test_allowed_in_desktop(self, execute_tool):
        result = json.loads(
            execute_tool(
                "store_story",
                {"title": "Real Story", "content": "User told me this"},
                channel="desktop",
            )
        )
        assert result.get("stored") is True


# ============== track_metric guards ==============


class TestMetricGuard:
    def test_blocked_in_autonomous(self, execute_tool):
        result = json.loads(
            execute_tool(
                "track_metric",
                {"metric_type": "focus_minutes", "value": 120, "unit": "min"},
                channel="autonomous",
            )
        )
        assert "error" in result
        assert "metric" in result["error"].lower()

    def test_blocked_in_worker(self, execute_tool):
        result = json.loads(
            execute_tool(
                "track_metric",
                {"metric_type": "project_score", "value": 80, "unit": "points"},
                channel="worker-analyst",
            )
        )
        assert "error" in result

    def test_allowed_in_telegram(self, execute_tool):
        result = execute_tool(
            "track_metric",
            {"metric_type": "project_score", "value": 79.6, "unit": "points"},
            channel="telegram",
        )
        assert "Recorded" in result or "project_score" in result

    def test_allowed_in_desktop(self, execute_tool):
        result = execute_tool(
            "track_metric",
            {"metric_type": "focus_minutes", "value": 85, "unit": "min"},
            channel="desktop",
        )
        assert "Recorded" in result or "focus_minutes" in result


# ============== update_persona guards ==============


class TestUpdatePersonaGuard:
    def test_blocked_in_autonomous(self, execute_tool):
        result = json.loads(
            execute_tool(
                "update_persona",
                {"tone": "aggressive"},
                channel="autonomous",
            )
        )
        assert "error" in result
        assert "interactive" in result["error"].lower()

    def test_blocked_in_worker(self, execute_tool):
        result = json.loads(
            execute_tool(
                "update_persona",
                {"name": "Evil Agent"},
                channel="worker-planner",
            )
        )
        assert "error" in result

    def test_allowed_in_desktop(self, execute_tool):
        result = json.loads(
            execute_tool(
                "update_persona",
                {"tone": "friendly"},
                channel="desktop",
            )
        )
        assert result.get("updated") is True


# ============== update_record guards ==============


class TestUpdateRecordGuard:
    def _store_verified_record(self, execute_tool, mock_brain):
        """Helper: store a record and mark it as user-confirmed."""
        result = json.loads(
            execute_tool(
                "store",
                {"content": "User lives in Kyiv", "tags": "location"},
                channel="desktop",
            )
        )
        rec_id = result["id"]
        # Mark as user-confirmed
        meta = (mock_brain.get(rec_id).metadata or {}).copy()
        meta["source"] = "user-confirmed"
        meta["verified"] = True
        meta["trust_score"] = 1.0
        mock_brain.update(rec_id, metadata=meta)
        return rec_id

    def test_autonomous_cannot_overwrite_verified(self, execute_tool, mock_brain):
        rec_id = self._store_verified_record(execute_tool, mock_brain)
        result = json.loads(
            execute_tool(
                "update_record",
                {"record_id": rec_id, "content": "User lives in London"},
                channel="autonomous",
            )
        )
        assert "error" in result
        assert (
            "user-confirmed" in result["error"].lower() or "autonomous" in result["error"].lower()
        )
        # Verify content was NOT changed
        rec = mock_brain.get(rec_id)
        assert "Kyiv" in rec.content

    def test_autonomous_can_update_unverified(self, execute_tool):
        # Store with autonomous channel (low trust, not verified)
        result = json.loads(
            execute_tool(
                "store",
                {"content": "Agent noted something", "tags": "note"},
                channel="autonomous",
            )
        )
        rec_id = result["id"]
        update_result = json.loads(
            execute_tool(
                "update_record",
                {"record_id": rec_id, "content": "Agent corrected note"},
                channel="autonomous",
            )
        )
        assert update_result.get("updated") is True

    def test_desktop_can_update_verified(self, execute_tool, mock_brain):
        rec_id = self._store_verified_record(execute_tool, mock_brain)
        result = json.loads(
            execute_tool(
                "update_record",
                {"record_id": rec_id, "content": "User moved to Lviv"},
                channel="desktop",
            )
        )
        assert result.get("updated") is True

    def test_audit_trail_original_content(self, execute_tool, mock_brain):
        """First update should preserve original_content."""
        result = json.loads(
            execute_tool(
                "store",
                {"content": "Original fact", "tags": "test"},
                channel="desktop",
            )
        )
        rec_id = result["id"]
        execute_tool(
            "update_record",
            {"record_id": rec_id, "content": "Updated fact"},
            channel="desktop",
        )
        rec = mock_brain.get(rec_id)
        meta = rec.metadata or {}
        assert "Original fact" in meta.get("original_content", "")
        assert meta.get("last_updated_by") == "agent-desktop"
        assert "last_updated_at" in meta

    def test_audit_trail_preserves_original_on_second_update(self, execute_tool, mock_brain):
        """Second update should NOT overwrite original_content."""
        result = json.loads(
            execute_tool(
                "store",
                {"content": "First version", "tags": "test"},
                channel="desktop",
            )
        )
        rec_id = result["id"]
        execute_tool(
            "update_record",
            {"record_id": rec_id, "content": "Second version"},
            channel="desktop",
        )
        execute_tool(
            "update_record",
            {"record_id": rec_id, "content": "Third version"},
            channel="desktop",
        )
        rec = mock_brain.get(rec_id)
        meta = rec.metadata or {}
        assert "First version" in meta["original_content"]


# ============== recall verification labels ==============


class TestRecallVerificationLabels:
    def test_verified_label(self, execute_tool, mock_brain):
        """User-confirmed records show VERIFIED."""
        result = json.loads(
            execute_tool(
                "store",
                {"content": "Verified fact about user", "tags": "test"},
                channel="desktop",
            )
        )
        rec_id = result["id"]
        meta = (mock_brain.get(rec_id).metadata or {}).copy()
        meta["verified"] = True
        meta["trust_score"] = 1.0
        mock_brain.update(rec_id, metadata=meta)

        recall_result = execute_tool("recall", {"query": "verified fact"})
        assert "VERIFIED" in recall_result

    def test_unverified_label_for_autonomous(self, execute_tool):
        """Autonomous-stored records show UNVERIFIED."""
        execute_tool(
            "store",
            {"content": "Autonomous stored data xyz123", "tags": "test"},
            channel="autonomous",
        )
        recall_result = execute_tool("recall", {"query": "xyz123"})
        assert "UNVERIFIED" in recall_result

    def test_likely_label_for_interactive(self, execute_tool):
        """Interactive-stored records with trust >= 0.6 show 'likely'."""
        execute_tool(
            "store",
            {"content": "Desktop stored info abc456", "tags": "test"},
            channel="desktop",
        )
        recall_result = execute_tool("recall", {"query": "abc456"})
        assert "likely" in recall_result or "VERIFIED" in recall_result

    def test_not_actionable_flag(self, execute_tool, mock_brain):
        """Records with actionable=False show NOT-ACTIONABLE."""
        result = json.loads(
            execute_tool(
                "store",
                {"content": "Sensitive data notact789", "tags": "test"},
                channel="autonomous",
            )
        )
        rec_id = result["id"]
        meta = (mock_brain.get(rec_id).metadata or {}).copy()
        meta["actionable"] = False
        mock_brain.update(rec_id, metadata=meta)

        recall_result = execute_tool("recall", {"query": "notact789"})
        assert "NOT-ACTIONABLE" in recall_result


# ============== extract_facts provenance ==============


class TestExtractFactsProvenance:
    @patch("remy.core.llm.call_llm")
    def test_extracted_facts_marked_unverified(self, mock_llm, execute_tool, mock_brain):
        mock_llm.return_value.content = json.dumps(
            [
                {
                    "subject": "Vitamin C",
                    "predicate": "boosts",
                    "object": "immunity",
                    "context": "health",
                }
            ]
        )
        execute_tool("extract_facts", {"text": "Vitamin C boosts immunity", "source": "web"})
        records = mock_brain.search(query="Vitamin C", tags=["extracted-fact"], limit=5)
        assert len(records) >= 1
        meta = records[0].metadata or {}
        assert meta.get("verified") is False
        assert meta.get("extraction_method") == "llm"


# ============== source_type epistemological tagging ==============


class TestSourceType:
    def test_interactive_store_is_recorded(self, execute_tool, mock_brain):
        """Interactive channel stores are source_type='recorded'."""
        result = json.loads(
            execute_tool(
                "store",
                {"content": "User told me about Kyiv", "tags": "fact"},
                channel="desktop",
            )
        )
        rec = mock_brain.get(result["id"])
        # SDK 1.2.0: source_type is a top-level field on the record
        assert rec.source_type == "recorded"

    def test_autonomous_store_is_generated(self, execute_tool, mock_brain):
        """Autonomous channel stores default to source_type='generated'."""
        result = json.loads(
            execute_tool(
                "store",
                {"content": "Agent decided to organize files", "tags": "plan"},
                channel="autonomous",
            )
        )
        rec = mock_brain.get(result["id"])
        assert rec.source_type == "generated"

    def test_research_store_is_retrieved(self, execute_tool, mock_brain):
        """store_research records are source_type='retrieved'."""
        result = json.loads(
            execute_tool(
                "store_research",
                {
                    "topic": "Bitcoin price analysis",
                    "findings": "BTC reached $67k on March 1",
                    "sources": "coingecko.com",
                },
                channel="desktop",
            )
        )
        rec = mock_brain.get(result["id"])
        assert rec.source_type == "retrieved"

    @patch("remy.core.llm.call_llm")
    def test_extracted_facts_are_inferred(self, mock_llm, execute_tool, mock_brain):
        """LLM-extracted facts are source_type='inferred'."""
        mock_llm.return_value.content = json.dumps(
            [{"subject": "Iron", "predicate": "prevents", "object": "anemia", "context": "health"}]
        )
        execute_tool("extract_facts", {"text": "Iron prevents anemia", "source": "web"})
        records = mock_brain.search(query="Iron", tags=["extracted-fact"], limit=5)
        assert len(records) >= 1
        assert records[0].source_type == "inferred"

    def test_recall_shows_source_type_for_retrieved(self, execute_tool, mock_brain):
        """Recall output includes 'retrieved' label for web-sourced data."""
        result = json.loads(
            execute_tool(
                "store_research",
                {
                    "topic": "ETH price srctype test",
                    "findings": "ETH at $2000 srctype123",
                    "sources": "coingecko.com",
                },
                channel="desktop",
            )
        )
        recall_result = execute_tool("recall", {"query": "srctype123"})
        assert "retrieved" in recall_result

    def test_recall_hides_source_type_for_recorded(self, execute_tool):
        """Recall output does NOT show 'recorded' label (it's the default)."""
        execute_tool(
            "store",
            {"content": "Normal interactive fact stype456", "tags": "test"},
            channel="desktop",
        )
        recall_result = execute_tool("recall", {"query": "stype456"})
        # 'recorded' should not appear as a label (it's the default, not shown)
        assert "recorded" not in recall_result

    def test_explicit_source_type_not_overwritten(self, execute_tool, mock_brain):
        """Explicit source_type in metadata is preserved by _stamp_provenance."""
        from remy.core.provenance import _stamp_provenance

        meta = _stamp_provenance({"source_type": "retrieved"}, channel="desktop", tags=["test"])
        assert meta["source_type"] == "retrieved"
