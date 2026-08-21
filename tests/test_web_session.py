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


def test_finish_trajectory_turn_observes_active_self_modification_canary():
    session = WebSession(
        session_id="session-canary",
        project_id="project-canary",
        session_log=[{"type": "factuality_analysis", "unsupported_claims_total": 0}],
    )
    store = MagicMock()
    lab = MagicMock()
    with patch(
        "remy.core.trajectory_store.get_trajectory_store",
        return_value=store,
    ), patch(
        "remy.core.self_modification_lab.get_self_modification_lab",
        return_value=lab,
    ):
        WebSessionManager._finish_trajectory_turn(session, 0)

    store.record_diagnostics.assert_called_once()
    store.finish_turn.assert_called_once_with(session_id="session-canary", error=None)
    lab.observe_active_canary_telemetry.assert_called_once_with(
        project_id="project-canary",
        trajectory_store=store,
    )


@pytest.fixture
def mock_genai():
    """Patch genai.Client so no real API key is needed."""
    with patch("remy.web.session.genai") as mock:
        mock_client = MagicMock()
        mock.Client.return_value = mock_client
        yield mock_client


@pytest.fixture(autouse=True)
def isolate_web_session_storage(tmp_path, monkeypatch):
    from remy.config.settings import settings as real_settings
    from remy.core.project_store import reset_project_store_for_tests

    monkeypatch.setattr(real_settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(real_settings, "AURA_BRAIN_PATH", tmp_path / "brain")
    reset_project_store_for_tests()
    yield
    reset_project_store_for_tests()


@pytest.fixture
def mock_settings(tmp_path):
    """Patch settings to provide a fake API key."""
    with patch("remy.web.session.settings") as mock:
        mock.GEMINI_API_KEY = "fake-key"
        mock.SUMMARY_MODEL = "gemini-test"
        mock.DATA_DIR = tmp_path
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
        """Transcript restoration stays lazy until the first model turn."""
        session = manager.get_or_create_session()
        assert session.history == []
        assert session.history_loaded is False
        assert session.session_log == []

    def test_restores_short_term_history_and_collapses_adjacent_retries(
        self,
        manager,
    ):
        from langchain_core.messages import AIMessage, HumanMessage

        session = WebSession(
            session_id="conversation-test",
            project_id="project-test",
        )
        project = MagicMock(
            project_id="project-test",
            brain_id="brain-test",
        )
        transcript = MagicMock()
        transcript.list_session.return_value = [
            {"role": "user", "content": "Привіт"},
            {"role": "user", "content": "  Привіт  "},
            {"role": "assistant", "content": "Вітаю"},
            {"role": "user", "content": "Пам'ятаєш попереднє питання?"},
        ]
        project_store = MagicMock()
        project_store.require_project.return_value = project

        with (
            patch("remy.web.session.get_project_store", return_value=project_store),
            patch(
                "remy.core.transcript_store.get_transcript_store",
                return_value=transcript,
            ),
        ):
            restored = manager._restore_short_term_history(session)
            restored_again = manager._restore_short_term_history(session)

        assert restored == 3
        assert restored_again == 3
        assert [type(message) for message in session.history] == [
            HumanMessage,
            AIMessage,
            HumanMessage,
        ]
        assert [message.content for message in session.history] == [
            "Привіт",
            "Вітаю",
            "Пам'ятаєш попереднє питання?",
        ]
        transcript.list_session.assert_called_once_with(
            "conversation-test",
            owner_project_id="project-test",
            include_legacy_unscoped=False,
            limit=80,
        )

    def test_restores_stable_session_id_from_marker(
        self, tmp_path, mock_genai, mock_settings
    ):
        mock_settings.DATA_DIR = tmp_path
        first = WebSessionManager()
        first_id = first.get_or_create_session().session_id

        second = WebSessionManager()
        assert second.get_or_create_session().session_id == first_id

    def test_background_result_is_attached_to_context_once(self, manager):
        session = manager.get_or_create_session()
        ledger = MagicMock()
        ledger.consume_continuations.return_value = [{
            "continuation_id": "continuation-1",
            "kind": "research_result",
            "source_id": "rp-1",
            "content": "Grounded report",
            "created_at": "2026-07-19T10:00:00Z",
            "metadata": {"delivery_target": "web", "status": "completed"},
        }]
        with patch("remy.core.execution_ledger.get_execution_ledger", return_value=ledger):
            attached = manager._attach_pending_continuations(session)

        assert attached == 1
        assert "Grounded report" in session.history[0].content
        assert session.session_log[0]["type"] == "background_result"
        assert session.session_log[0]["text"] == "Grounded report"
        ledger.consume_continuations.assert_called_once()


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

    @pytest.mark.asyncio
    async def test_shutdown_preserves_session_for_restart(
        self, tmp_path, mock_genai, mock_settings
    ):
        mock_settings.DATA_DIR = tmp_path
        first = WebSessionManager()
        session_id = first.get_or_create_session().session_id
        with patch("remy.web.session.brain") as mock_brain:
            mock_brain.end_session = MagicMock()
            await first.close_session(
                generate_summary=False,
                preserve_for_resume=True,
            )

        restarted = WebSessionManager()
        assert restarted.get_or_create_session().session_id == session_id

    @pytest.mark.asyncio
    async def test_explicit_close_clears_resumable_session(
        self, tmp_path, mock_genai, mock_settings
    ):
        mock_settings.DATA_DIR = tmp_path
        first = WebSessionManager()
        first_id = first.get_or_create_session().session_id
        with patch("remy.web.session.brain") as mock_brain:
            mock_brain.end_session = MagicMock()
            await first.close_session(generate_summary=False)

        restarted = WebSessionManager()
        assert restarted.get_or_create_session().session_id != first_id

    @pytest.mark.asyncio
    async def test_close_session_skips_summary_during_server_shutdown(self, manager):
        manager.get_or_create_session()
        manager.session.session_log.append({"type": "user_text", "text": "hello"})

        with patch("remy.web.session.generate_session_summary", new_callable=AsyncMock) as mock_summary, \
             patch("remy.web.session.brain") as mock_brain:
            mock_brain.end_session = MagicMock()
            await manager.close_session(generate_summary=False)

        mock_summary.assert_not_awaited()
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

    @pytest.mark.asyncio
    async def test_followup_turn_receives_previous_short_term_history(self, manager):
        from langchain_core.messages import AIMessage, HumanMessage

        first_history = [
            HumanMessage(content="My launch is in October"),
            AIMessage(content="I will keep that in this chat."),
        ]
        second_history = first_history + [
            HumanMessage(content="When is it?"),
            AIMessage(content="October."),
        ]
        with patch(
            "remy.web.session.invoke_agent",
            new_callable=AsyncMock,
        ) as mock_invoke:
            mock_invoke.side_effect = [
                ("I will keep that in this chat.", first_history, []),
                ("October.", second_history, []),
            ]

            await manager.gemini_respond("My launch is in October")
            await manager.gemini_respond("When is it?")

        assert mock_invoke.await_count == 2
        assert mock_invoke.await_args_list[1].kwargs["history"] == first_history


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
