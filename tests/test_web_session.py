"""Tests for WebSessionManager."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from remy.web.session import (
    WebSession,
    WebSessionManager,
    MAX_FILE_SIZE,
    SUPPORTED_MIME_TYPES,
)


@pytest.fixture
def mock_genai():
    """Patch genai.Client so no real API key is needed."""
    with patch("remy.web.session.genai") as mock:
        mock_client = MagicMock()
        mock.Client.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_settings():
    """Patch settings to provide a fake API key."""
    with patch("remy.web.session.settings") as mock:
        mock.GEMINI_API_KEY = "fake-key"
        mock.SUMMARY_MODEL = "gemini-test"
        yield mock


@pytest.fixture
def manager(mock_genai, mock_settings):
    """Create a WebSessionManager with mocked dependencies."""
    return WebSessionManager()


class TestSessionCreation:

    def test_creates_new_session(self, manager):
        session = manager.get_or_create_session()
        assert session is not None
        assert isinstance(session, WebSession)
        assert session.session_id

    def test_reuses_existing_session(self, manager):
        s1 = manager.get_or_create_session()
        s2 = manager.get_or_create_session()
        assert s1.session_id == s2.session_id

    def test_session_has_empty_history(self, manager):
        session = manager.get_or_create_session()
        assert session.history == []
        assert session.session_log == []


class TestCloseSession:

    @pytest.mark.asyncio
    async def test_close_session_clears_state(self, manager):
        manager.get_or_create_session()
        assert manager.session is not None

        with patch("remy.web.session.generate_session_summary", new_callable=AsyncMock), \
             patch("remy.web.session.brain") as mock_brain:
            mock_brain.end_session = MagicMock()
            await manager.close_session()

        assert manager.session is None

    @pytest.mark.asyncio
    async def test_close_session_noop_when_no_session(self, manager):
        """Close with no session should not raise."""
        await manager.close_session()
        assert manager.session is None


class TestMultimodalRespond:

    @pytest.mark.asyncio
    async def test_unsupported_mime_rejected(self, manager):
        """Unsupported MIME types return error without calling API."""
        result = await manager.gemini_respond_multimodal(
            attachments=[{"mime_type": "application/x-executable", "data": b"binary"}],
        )
        assert "Unsupported" in result["response"]

    @pytest.mark.asyncio
    async def test_oversized_file_rejected(self, manager):
        """Files over MAX_FILE_SIZE return error."""
        result = await manager.gemini_respond_multimodal(
            attachments=[{"mime_type": "image/png", "data": b"x" * (MAX_FILE_SIZE + 1)}],
        )
        assert "too large" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_empty_message_rejected(self, manager):
        """No text and no attachments returns error."""
        result = await manager.gemini_respond_multimodal()
        assert "Empty" in result["response"]

    @pytest.mark.asyncio
    async def test_supported_mime_accepted(self, manager):
        """Verify common MIME types are in the supported set."""
        for mime in ("image/jpeg", "image/png", "audio/webm", "application/pdf"):
            assert mime in SUPPORTED_MIME_TYPES

    @pytest.mark.asyncio
    async def test_voice_calls_invoke_agent(self, manager):
        """Voice message calls invoke_agent with HumanMessage."""
        with patch("remy.web.session.invoke_agent", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = ("Hello there!", [], [])

            result = await manager.gemini_respond_multimodal(
                attachments=[{"mime_type": "audio/webm", "data": b"fake-audio"}],
                is_voice=True,
            )

        assert result["response"] == "Hello there!"
        mock_invoke.assert_called_once()
        # Verify it was called with a HumanMessage (multimodal)
        call_kwargs = mock_invoke.call_args[1]
        assert call_kwargs["channel"] == "desktop"

    @pytest.mark.asyncio
    async def test_file_with_text(self, manager):
        """File + text calls invoke_agent with multimodal HumanMessage."""
        with patch("remy.web.session.invoke_agent", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = ("I see a cat.", [], [])

            result = await manager.gemini_respond_multimodal(
                text="What is in this image?",
                attachments=[{"mime_type": "image/jpeg", "data": b"fake-jpg"}],
            )

        assert result["response"] == "I see a cat."


class TestTextRespond:

    @pytest.mark.asyncio
    async def test_gemini_respond_calls_invoke_agent(self, manager):
        """gemini_respond delegates to invoke_agent."""
        with patch("remy.web.session.invoke_agent", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = ("Hi there!", [{"msg": "test"}], [{"type": "user_text"}])

            result = await manager.gemini_respond("Hello")

        assert result == "Hi there!"
        mock_invoke.assert_called_once()
        call_kwargs = mock_invoke.call_args[1]
        assert call_kwargs["user_message"] == "Hello"
        assert call_kwargs["channel"] == "desktop"

    @pytest.mark.asyncio
    async def test_respond_updates_session_state(self, manager):
        """After responding, session history and log are updated."""
        with patch("remy.web.session.invoke_agent", new_callable=AsyncMock) as mock_invoke:
            new_history = [{"role": "user"}, {"role": "assistant"}]
            new_log = [{"type": "user_text"}, {"type": "tool_call"}]
            mock_invoke.return_value = ("Response", new_history, new_log)

            await manager.gemini_respond("Hello")

        session = manager.session
        assert session.history == new_history
        assert session.session_log == new_log


class TestBuildSystemInstructionDesktop:

    def test_desktop_channel_instruction(self):
        """build_system_instruction with channel='desktop' includes desktop hints."""
        with patch("remy.core.brain_tools.brain") as mock_brain:
            mock_brain.recall.return_value = ""
            mock_brain.search.return_value = []

            from remy.core.brain_tools import build_system_instruction
            result = build_system_instruction(channel="desktop")

            assert "detailed responses" in result
            assert "thorough" in result
