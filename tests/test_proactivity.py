"""Tests for Phase 1: Brain-driven proactivity."""

import json
import uuid
from unittest.mock import patch, MagicMock

import pytest
from aura import Aura as CognitiveMemory


# ============== System Instruction Tests ==============

class TestSystemInstruction:

    def test_includes_brain_context(self, tmp_path):
        """When brain has memories, system instruction includes them."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        # Store content that matches the recall query "session start family history recent topics"
        b.store(content="session start family history recent topics about grandmother Maria in Kyiv", tags=["person"])

        with patch("remy.core.brain_tools.brain", b):
            from remy.core.brain_tools import build_system_instruction
            instruction = build_system_instruction()

        assert "remember from previous sessions" in instruction
        assert "Maria" in instruction or "COGNITIVE CONTEXT" in instruction
        b.close()

    def test_handles_empty_brain(self, tmp_path):
        """Empty brain → instruction still valid, no crash."""
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.brain_tools.brain", b):
            from remy.core.brain_tools import build_system_instruction
            instruction = build_system_instruction()

        assert "Remy" in instruction
        assert "remember from previous sessions" not in instruction
        b.close()

    def test_handles_brain_error(self, tmp_path):
        """Brain error → falls back to base instruction."""
        mock_brain = MagicMock()
        mock_brain.recall.side_effect = RuntimeError("DB corrupt")

        with patch("remy.core.brain_tools.brain", mock_brain):
            from remy.core.brain_tools import build_system_instruction
            instruction = build_system_instruction()

        assert "Remy" in instruction
        # Should not crash, just warn
        assert "remember from previous sessions" not in instruction

    def test_contains_insight_instruction(self, tmp_path):
        """System instruction tells Remy how to handle brain insights."""
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.brain_tools.brain", b):
            from remy.core.brain_tools import build_system_instruction
            instruction = build_system_instruction()

        assert "INTERNAL BRAIN INSIGHT" in instruction
        b.close()


# ============== Session ID Tests ==============

class TestSessionId:

    def test_session_id_generated(self):
        """GeminiLiveSession gets a UUID session_id."""
        with patch("remy.core.gemini_live.pyaudio") as mock_pya, \
             patch("remy.core.gemini_live.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = "test-key"
            mock_settings.GEMINI_MODEL = "test-model"
            mock_settings.GEMINI_VOICE = "Zephyr"
            mock_pya.PyAudio.return_value = MagicMock()

            with patch("remy.core.gemini_live.genai"):
                from remy.core.gemini_live import GeminiLiveSession
                session = GeminiLiveSession()

            assert session.session_id is not None
            # Should be a valid UUID
            uuid.UUID(session.session_id)

    def test_recall_passes_session_id(self, tmp_path):
        """execute_tool('recall') passes session_id to brain.recall_structured()."""
        mock_brain = MagicMock()
        mock_brain.recall_structured.return_value = []

        with patch("remy.core.brain_tools.brain", mock_brain), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.tool_registry.settings") as mock_settings:
            mock_settings.SANDBOX_DIR = tmp_path / "sandbox"
            mock_settings.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"

            from remy.core.brain_tools import execute_tool
            execute_tool("recall", {"query": "grandmother"}, session_id="test-session-123")

        mock_brain.recall_structured.assert_called_once_with(
            "grandmother",
            top_k=15,
            session_id="test-session-123",
        )

    def test_search_passes_session_id(self, tmp_path):
        """execute_tool('search') passes session_id to brain.recall_structured()."""
        mock_brain = MagicMock()
        mock_brain.recall_structured.return_value = []

        with patch("remy.core.brain_tools.brain", mock_brain), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.tool_registry.settings") as mock_settings:
            mock_settings.SANDBOX_DIR = tmp_path / "sandbox"
            mock_settings.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"

            from remy.core.brain_tools import execute_tool
            execute_tool("search", {"query": "uncle"}, session_id="test-session-456")

        mock_brain.recall_structured.assert_called_once_with(
            "uncle",
            top_k=20,
            min_strength=0.05,
            session_id="test-session-456",
        )


# ============== Format Insights Tests ==============

class TestFormatInsights:

    @pytest.fixture
    def session_cls(self):
        """Get GeminiLiveSession class for testing static method."""
        from remy.core.gemini_live import GeminiLiveSession
        return GeminiLiveSession

    def test_decay_risk(self, session_cls):
        insights = [{
            "type": "decay_risk",
            "summary": "Important records fading",
            "record_ids": ["r1"],
            "details": {
                "records": [
                    {"id": "r1", "content": "Grandmother Maria lived in Kyiv", "strength": 0.3, "activation_count": 5}
                ]
            }
        }]
        result = session_cls._format_insights(insights)
        assert "fading" in result
        assert "Grandmother Maria" in result

    def test_conflict(self, session_cls):
        insights = [{
            "type": "conflict",
            "summary": "Contradiction detected",
            "record_ids": ["r1", "r2"],
            "details": {
                "pairs": [{
                    "id_a": "r1", "id_b": "r2",
                    "content_a": "Maria was born in 1935",
                    "content_b": "Maria was born in 1937",
                }]
            }
        }]
        result = session_cls._format_insights(insights)
        assert "contradiction" in result.lower() or "Possible" in result
        assert "Maria" in result

    def test_hot_topic(self, session_cls):
        insights = [{
            "type": "hot_topic",
            "summary": "Active topics",
            "record_ids": [],
            "details": {
                "topics": [
                    {"tag": "grandmother", "record_count": 5, "avg_activations": 4.2},
                    {"tag": "war", "record_count": 3, "avg_activations": 3.5},
                ]
            }
        }]
        result = session_cls._format_insights(insights)
        assert "grandmother" in result
        assert "war" in result

    def test_empty_insights(self, session_cls):
        result = session_cls._format_insights([])
        assert result == ""

    def test_filters_unimportant(self, session_cls):
        """Only decay_risk, conflict, hot_topic produce output."""
        insights = [{
            "type": "stale_topic",
            "summary": "Stale topics",
            "record_ids": [],
            "details": {"topics": [{"tag": "old_stuff", "record_count": 2, "avg_strength": 0.1}]}
        }]
        result = session_cls._format_insights(insights)
        assert result == ""

    def test_multiple_insights(self, session_cls):
        insights = [
            {
                "type": "decay_risk",
                "summary": "Fading",
                "record_ids": ["r1"],
                "details": {"records": [{"id": "r1", "content": "Uncle Petro", "strength": 0.2, "activation_count": 3}]}
            },
            {
                "type": "hot_topic",
                "summary": "Hot",
                "record_ids": [],
                "details": {"topics": [{"tag": "family", "record_count": 10, "avg_activations": 5}]}
            },
        ]
        result = session_cls._format_insights(insights)
        assert "Uncle Petro" in result
        assert "family" in result
        assert ";" in result  # joined with semicolons


# ============== Settings Tests ==============

class TestSettings:

    def test_proactive_interval_default(self):
        from remy.config.settings import Settings
        s = Settings(GEMINI_API_KEY="test")
        assert s.PROACTIVE_INTERVAL_SEC == 300
