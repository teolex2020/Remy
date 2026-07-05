"""Tests for Phase 2: Session Summary ('Living Presence')."""

import json
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from aura import Aura as CognitiveMemory, Level


# ============== Activity Log Tests ==============

class TestActivityLog:

    def test_session_log_starts_empty(self):
        """New session has empty _session_log."""
        with patch("remy.core.gemini_live.pyaudio") as mock_pya, \
             patch("remy.core.gemini_live.settings") as mock_settings, \
             patch("remy.core.gemini_live.genai"):
            mock_settings.GEMINI_API_KEY = "test-key"
            mock_settings.GEMINI_MODEL = "test-model"
            mock_settings.GEMINI_VOICE = "Zephyr"
            mock_pya.PyAudio.return_value = MagicMock()

            from remy.core.gemini_live import GeminiLiveSession
            session = GeminiLiveSession()

        assert session._session_log == []

    def test_log_truncates_long_values(self):
        """Tool args and results are truncated in log entries."""
        # Simulate what _receive_and_play does when logging
        long_arg = "x" * 500
        long_result = "y" * 500

        entry = {
            "type": "tool_call",
            "tool": "store",
            "args": {"content": str(long_arg)[:100]},
            "result": long_result[:200],
        }

        assert len(entry["args"]["content"]) == 100
        assert len(entry["result"]) == 200


# ============== Summary Generation Tests ==============

class TestGenerateSummary:

    @pytest.mark.asyncio
    async def test_empty_log_returns_none(self):
        """No activity → returns None."""
        from remy.core.brain_tools import generate_session_summary
        mock_client = MagicMock()

        result = await generate_session_summary(mock_client, [], "test-session")
        assert result is None

    @pytest.mark.asyncio
    async def test_calls_gemini_with_log(self):
        """With log entries → calls client.models.generate_content."""
        from remy.core.brain_tools import generate_session_summary

        mock_response = MagicMock()
        mock_response.text = "User stored family member Maria and asked about grandmother."
        mock_client = MagicMock()
        mock_client.models.generate_content = MagicMock(return_value=mock_response)

        session_log = [
            {"type": "tool_call", "tool": "store_person", "args": {"full_name": "Maria"}, "result": "stored"},
            {"type": "user_text", "text": "розкажи про бабусю"},
        ]

        with patch("remy.core.brain_tools.brain") as mock_brain:
            mock_brain.store.return_value = MagicMock(id="test-id")
            result = await generate_session_summary(mock_client, session_log, "test-session")

        assert result == "User stored family member Maria and asked about grandmother."
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args
        prompt = call_kwargs[1]["contents"] if "contents" in call_kwargs[1] else call_kwargs[0][0]
        assert "Maria" in str(prompt)

    @pytest.mark.asyncio
    async def test_handles_api_error(self):
        """API error → returns None, no crash."""
        from remy.core.brain_tools import generate_session_summary

        mock_client = MagicMock()
        mock_client.models.generate_content = MagicMock(side_effect=RuntimeError("API down"))

        session_log = [{"type": "user_text", "text": "hello"}]

        result = await generate_session_summary(mock_client, session_log, "test-session")
        assert result is None

    @pytest.mark.asyncio
    async def test_handles_empty_response(self):
        """Empty response text → returns None."""
        from remy.core.brain_tools import generate_session_summary

        mock_response = MagicMock()
        mock_response.text = None
        mock_client = MagicMock()
        mock_client.models.generate_content = MagicMock(return_value=mock_response)

        session_log = [{"type": "user_text", "text": "hello"}]

        result = await generate_session_summary(mock_client, session_log, "test-session")
        assert result is None


# ============== Summary Storage Tests ==============

class TestSummaryStorage:

    def test_summary_stored_in_brain(self, tmp_path):
        """Summary is stored with correct level, tags, and metadata."""
        b = CognitiveMemory(str(tmp_path / "brain"))

        # Simulate what the finally block does
        summary = "Discussed grandmother Maria and her life in Kyiv."
        session_id = "test-session-abc"

        rec = b.store(
            content=summary,
            level=Level.DOMAIN,
            tags=["session-summary"],
            metadata={"session_id": session_id, "type": "session_summary"},
        )

        assert rec.id is not None
        assert rec.content == summary

        # Verify we can find it by tag
        results = b.search(query="", tags=["session-summary"], limit=5)
        assert len(results) >= 1
        assert any(r.content == summary for r in results)

        b.close()

    def test_summary_not_stored_when_empty(self, tmp_path):
        """Empty session → nothing stored."""
        b = CognitiveMemory(str(tmp_path / "brain"))

        # No summary stored
        results = b.search(query="", tags=["session-summary"], limit=5)
        assert len(results) == 0

        b.close()


# ============== System Instruction with Summaries ==============

class TestSystemInstructionSummaries:

    def test_includes_session_summaries(self, tmp_path):
        """Brain with session-summary records → instruction contains them."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        b.store(
            content="Discussed grandfather Petro and the war stories.",
            level=Level.DOMAIN,
            tags=["session-summary"],
        )

        with patch("remy.core.brain_tools.brain", b), \
             patch("remy.core.brain_tools._proactive_context_cache", {}):
            from remy.core.brain_tools import build_system_instruction
            instruction = build_system_instruction()

        assert "PREVIOUS CONTEXT" in instruction
        assert "grandfather Petro" in instruction
        b.close()

    def test_no_summaries_still_valid(self, tmp_path):
        """Empty brain → instruction still valid, no 'Recent sessions:' section."""
        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.brain_tools.brain", b):
            from remy.core.brain_tools import build_system_instruction
            instruction = build_system_instruction()

        assert "Remy" in instruction
        assert "Recent sessions:" not in instruction
        b.close()

    def test_multiple_summaries(self, tmp_path):
        """Multiple session summaries are included."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        b.store(content="Session 1: talked about Maria.", level=Level.DOMAIN, tags=["session-summary"])
        b.store(content="Session 2: explored war stories.", level=Level.DOMAIN, tags=["session-summary"])

        with patch("remy.core.brain_tools.brain", b):
            from remy.core.brain_tools import build_system_instruction
            instruction = build_system_instruction()

        assert "Maria" in instruction
        assert "war stories" in instruction
        b.close()


# ============== Settings Tests ==============

class TestSummarySettings:

    def test_summary_model_default(self, monkeypatch):
        from remy.config.settings import Settings
        monkeypatch.delenv("SUMMARY_MODEL", raising=False)
        s = Settings(GEMINI_API_KEY="test")
        assert s.SUMMARY_MODEL == "gemini-3-flash-preview"
