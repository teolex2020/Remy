"""TEST-2: Integration Tests — combined_runner, WebSocket chat, autonomous cycle."""

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aura import Aura as CognitiveMemory
from fastapi import FastAPI
from fastapi.testclient import TestClient

from remy.web.api import router, set_session_manager
from remy.web.session import WebSession


# ============== SECTION 1: COMBINED RUNNER ==============


def _combined_mocks():
    """Common mock context for combined_runner tests."""
    return {
        "brain": patch("remy.core.combined_runner.brain"),
        "registry": patch("remy.core.combined_runner.get_registry"),
        "settings": patch("remy.core.combined_runner.settings"),
    }


def _configure_settings(mock_s):
    mock_s.SUMMARY_MODEL = "test-model"
    mock_s.AURA_BRAIN_PATH = "/tmp/brain"
    mock_s.WEB_HOST = "127.0.0.1"
    mock_s.WEB_PORT = 8080
    mock_s.AUTONOMY_CYCLE_INTERVAL_SEC = 1
    mock_s.AUTONOMY_DAILY_TOKEN_LIMIT = 100000


class TestCombinedRunnerIntegration:

    @pytest.mark.asyncio
    async def test_all_channels_start_and_shutdown(self):
        """All three channels start, run briefly, and shut down cleanly."""
        mocks = _combined_mocks()
        with mocks["brain"] as mb, mocks["registry"] as mr, mocks["settings"] as ms:
            mb.count.return_value = 0
            mr.return_value.get_all_declarations.return_value = []
            _configure_settings(ms)

            shutdown_ref = {}
            orig_event = asyncio.Event

            def patched_event():
                ev = orig_event()
                shutdown_ref["event"] = ev
                return ev

            async def auto_start():
                # Signal shutdown so the while-loop exits instead of restarting
                if "event" in shutdown_ref:
                    shutdown_ref["event"].set()
                await asyncio.sleep(3600)

            mock_auto = MagicMock()
            mock_auto.session_id = "test-auto-1"
            mock_auto.start = AsyncMock(side_effect=auto_start)
            mock_auto.stop = MagicMock()

            mock_tg_app = AsyncMock()

            mock_server = MagicMock()
            mock_server.serve = AsyncMock(side_effect=asyncio.CancelledError)
            mock_server.shutdown = AsyncMock()

            with patch("remy.core.autonomy.AutonomousLoop",
                        return_value=mock_auto) as auto_cls, \
                 patch("remy.core.combined_runner._start_telegram_async",
                        new_callable=AsyncMock, return_value=mock_tg_app), \
                 patch("remy.core.combined_runner._create_uvicorn_server",
                        return_value=mock_server), \
                 patch("remy.core.combined_runner._stop_telegram_async",
                        new_callable=AsyncMock), \
                 patch("remy.core.combined_runner.asyncio.Event",
                        side_effect=patched_event):

                from remy.core.combined_runner import run_combined
                await asyncio.wait_for(
                    run_combined(autonomous=True, telegram=True, web=True),
                    timeout=10,
                )

            auto_cls.assert_called_once()
            mock_auto.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_autonomous_only_restart(self):
        """Autonomous loop completes → restart logic creates a new loop."""
        mocks = _combined_mocks()
        with mocks["brain"] as mb, mocks["registry"] as mr, mocks["settings"] as ms:
            mb.count.return_value = 0
            mr.return_value.get_all_declarations.return_value = []
            _configure_settings(ms)

            call_count = 0
            shutdown_ref = {}

            orig_event = asyncio.Event

            def patched_event():
                ev = orig_event()
                shutdown_ref["event"] = ev
                return ev

            async def mock_start():
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    # Second call: signal shutdown so run_combined exits
                    if "event" in shutdown_ref:
                        shutdown_ref["event"].set()
                    await asyncio.sleep(3600)  # Wait until cancelled
                return  # First call: complete normally (triggers restart)

            mock_auto = MagicMock()
            mock_auto.session_id = "test-auto-1"
            mock_auto.start = AsyncMock(side_effect=mock_start)
            mock_auto.stop = MagicMock()

            with patch("remy.core.autonomy.AutonomousLoop",
                        return_value=mock_auto), \
                 patch("remy.core.combined_runner.asyncio.sleep",
                        new_callable=AsyncMock), \
                 patch("remy.core.combined_runner.asyncio.Event",
                        side_effect=patched_event):

                from remy.core.combined_runner import run_combined
                await asyncio.wait_for(
                    run_combined(autonomous=True, telegram=False, web=False),
                    timeout=10,
                )

            assert call_count == 2

    @pytest.mark.asyncio
    async def test_autonomous_restart_after_crash(self):
        """Autonomous loop crashes → restart after 30s delay."""
        mocks = _combined_mocks()
        with mocks["brain"] as mb, mocks["registry"] as mr, mocks["settings"] as ms:
            mb.count.return_value = 0
            mr.return_value.get_all_declarations.return_value = []
            _configure_settings(ms)

            call_count = 0
            shutdown_ref = {}

            orig_event = asyncio.Event

            def patched_event():
                ev = orig_event()
                shutdown_ref["event"] = ev
                return ev

            async def mock_start():
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    if "event" in shutdown_ref:
                        shutdown_ref["event"].set()
                    await asyncio.sleep(3600)
                raise RuntimeError("Crash!")

            mock_auto = MagicMock()
            mock_auto.session_id = "test-auto-1"
            mock_auto.start = AsyncMock(side_effect=mock_start)
            mock_auto.stop = MagicMock()

            sleep_calls = []

            async def mock_sleep(seconds):
                sleep_calls.append(seconds)

            with patch("remy.core.autonomy.AutonomousLoop",
                        return_value=mock_auto), \
                 patch("remy.core.combined_runner.asyncio.sleep",
                        side_effect=mock_sleep), \
                 patch("remy.core.combined_runner.asyncio.Event",
                        side_effect=patched_event):

                from remy.core.combined_runner import run_combined
                await asyncio.wait_for(
                    run_combined(autonomous=True, telegram=False, web=False),
                    timeout=10,
                )

            assert call_count == 2
            assert 30 in sleep_calls

    @pytest.mark.asyncio
    async def test_signal_triggers_shutdown(self):
        """Signal handler sets shutdown event → cleanup runs."""
        mocks = _combined_mocks()
        with mocks["brain"] as mb, mocks["registry"] as mr, mocks["settings"] as ms:
            mb.count.return_value = 0
            mr.return_value.get_all_declarations.return_value = []
            _configure_settings(ms)

            handlers = {}

            def capture_handler(sig, handler):
                handlers[sig] = handler

            mock_auto = MagicMock()
            mock_auto.session_id = "test-auto-1"

            async def wait_forever():
                await asyncio.sleep(3600)

            mock_auto.start = AsyncMock(side_effect=wait_forever)
            mock_auto.stop = MagicMock()

            async def trigger_signal():
                await asyncio.sleep(0.05)
                import signal
                if signal.SIGINT in handlers:
                    handlers[signal.SIGINT]()

            loop = asyncio.get_event_loop()

            with patch("remy.core.autonomy.AutonomousLoop",
                        return_value=mock_auto), \
                 patch.object(loop, "add_signal_handler", side_effect=capture_handler):

                from remy.core.combined_runner import run_combined
                task = asyncio.create_task(
                    run_combined(autonomous=True, telegram=False, web=False)
                )
                trigger = asyncio.create_task(trigger_signal())
                await asyncio.wait_for(
                    asyncio.gather(task, trigger),
                    timeout=10,
                )

            mock_auto.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_windows_signal_fallback(self):
        """NotImplementedError on add_signal_handler → loop still runs."""
        mocks = _combined_mocks()
        with mocks["brain"] as mb, mocks["registry"] as mr, mocks["settings"] as ms:
            mb.count.return_value = 0
            mr.return_value.get_all_declarations.return_value = []
            _configure_settings(ms)

            call_count = 0
            shutdown_ref = {}
            orig_event = asyncio.Event

            def patched_event():
                ev = orig_event()
                shutdown_ref["event"] = ev
                return ev

            async def mock_start():
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    if "event" in shutdown_ref:
                        shutdown_ref["event"].set()
                    await asyncio.sleep(3600)
                # First call: complete normally

            mock_auto = MagicMock()
            mock_auto.session_id = "test-auto-1"
            mock_auto.start = AsyncMock(side_effect=mock_start)
            mock_auto.stop = MagicMock()

            loop = asyncio.get_event_loop()

            with patch("remy.core.autonomy.AutonomousLoop",
                        return_value=mock_auto), \
                 patch.object(loop, "add_signal_handler",
                              side_effect=NotImplementedError), \
                 patch("remy.core.combined_runner.asyncio.sleep",
                        new_callable=AsyncMock), \
                 patch("remy.core.combined_runner.asyncio.Event",
                        side_effect=patched_event):

                from remy.core.combined_runner import run_combined
                await asyncio.wait_for(
                    run_combined(autonomous=True, telegram=False, web=False),
                    timeout=10,
                )

            # Verify it ran despite NotImplementedError on signal handlers
            assert call_count >= 1

    @pytest.mark.asyncio
    async def test_cleanup_continues_on_error(self):
        """Cleanup function error doesn't prevent other cleanups."""
        mocks = _combined_mocks()
        with mocks["brain"] as mb, mocks["registry"] as mr, mocks["settings"] as ms:
            mb.count.return_value = 0
            mr.return_value.get_all_declarations.return_value = []
            _configure_settings(ms)

            cleanup_results = []

            mock_auto = MagicMock()
            mock_auto.session_id = "test-auto-1"
            mock_auto.start = AsyncMock(side_effect=asyncio.CancelledError)

            def bad_cleanup():
                cleanup_results.append("bad_called")
                raise RuntimeError("cleanup fail")

            mock_auto.stop = bad_cleanup

            mock_server = MagicMock()
            mock_server.serve = AsyncMock(side_effect=asyncio.CancelledError)

            def good_cleanup():
                cleanup_results.append("good_called")

            mock_server.shutdown = good_cleanup

            with patch("remy.core.autonomy.AutonomousLoop",
                        return_value=mock_auto), \
                 patch("remy.core.combined_runner._create_uvicorn_server",
                        return_value=mock_server):

                from remy.core.combined_runner import run_combined
                await run_combined(autonomous=True, telegram=False, web=True)

            assert "bad_called" in cleanup_results
            assert "good_called" in cleanup_results


# ============== SECTION 2: WEBSOCKET CHAT FLOW ==============


@pytest.fixture
def ws_app():
    """FastAPI app + TestClient for WebSocket integration tests."""
    app = FastAPI()
    app.include_router(router)

    mock_manager = MagicMock()
    session = WebSession(session_id="ws-test-session")
    mock_manager.get_or_create_session.return_value = session
    mock_manager.close_session = AsyncMock()

    set_session_manager(mock_manager)

    with patch("remy.web.api.metrics_collector") as mock_metrics:
        yield {
            "client": TestClient(app),
            "manager": mock_manager,
            "metrics": mock_metrics,
        }


class TestWebSocketChatFlow:

    def test_ws_text_message_streaming(self, ws_app):
        """Text message → typing + token stream + done."""
        client = ws_app["client"]
        manager = ws_app["manager"]

        async def mock_stream(text):
            yield {"type": "token", "content": "Hello"}
            yield {"type": "token", "content": " world"}
            yield {"type": "final", "text": "Hello world"}

        manager.gemini_respond_stream = mock_stream

        with client.websocket_connect("/api/ws/chat") as ws:
            ws.send_json({"type": "message", "text": "Hi"})

            msg1 = ws.receive_json()
            assert msg1["type"] == "typing"

            msg2 = ws.receive_json()
            assert msg2 == {"type": "token", "content": "Hello"}

            msg3 = ws.receive_json()
            assert msg3 == {"type": "token", "content": " world"}

            # No "text" event because streamed_any=True
            msg4 = ws.receive_json()
            assert msg4["type"] == "done"

    def test_ws_text_no_stream_sends_text(self, ws_app):
        """No tokens yielded → falls back to text event."""
        client = ws_app["client"]
        manager = ws_app["manager"]

        async def mock_stream(text):
            yield {"type": "final", "text": "Full response"}

        manager.gemini_respond_stream = mock_stream

        with client.websocket_connect("/api/ws/chat") as ws:
            ws.send_json({"type": "message", "text": "Hi"})

            msg1 = ws.receive_json()
            assert msg1["type"] == "typing"

            msg2 = ws.receive_json()
            assert msg2["type"] == "text"
            assert msg2["content"] == "Full response"

            msg3 = ws.receive_json()
            assert msg3["type"] == "done"

    def test_ws_tool_events(self, ws_app):
        """Tool start/end events forwarded to client."""
        client = ws_app["client"]
        manager = ws_app["manager"]

        async def mock_stream(text):
            yield {"type": "tool_start", "tool": "web_search"}
            yield {"type": "tool_end", "tool": "web_search"}
            yield {"type": "token", "content": "Result"}
            yield {"type": "final", "text": "Result"}

        manager.gemini_respond_stream = mock_stream

        with client.websocket_connect("/api/ws/chat") as ws:
            ws.send_json({"type": "message", "text": "search"})

            ws.receive_json()  # typing

            msg = ws.receive_json()
            assert msg == {"type": "tool_start", "content": "web_search"}

            msg = ws.receive_json()
            assert msg == {"type": "tool_end", "content": "web_search"}

            msg = ws.receive_json()
            assert msg == {"type": "token", "content": "Result"}

            msg = ws.receive_json()
            assert msg["type"] == "done"

    def test_ws_error_sends_friendly_message(self, ws_app):
        """LLM error → friendly error message with retryable flag."""
        client = ws_app["client"]
        manager = ws_app["manager"]

        async def mock_stream(text):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
            yield  # noqa: unreachable — makes this an async generator

        manager.gemini_respond_stream = mock_stream

        with client.websocket_connect("/api/ws/chat") as ws:
            ws.send_json({"type": "message", "text": "test"})

            ws.receive_json()  # typing

            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "rate limit" in msg["content"].lower()
            assert msg["retryable"] is True

            msg = ws.receive_json()
            assert msg["type"] == "done"

    def test_ws_voice_message(self, ws_app):
        """Voice message → multimodal response with speak=True."""
        client = ws_app["client"]
        manager = ws_app["manager"]

        manager.gemini_respond_multimodal = AsyncMock(
            return_value={"response": "Voice response", "input_transcript": "test"}
        )

        audio_b64 = base64.b64encode(b"fake-audio-data").decode()

        with client.websocket_connect("/api/ws/chat") as ws:
            ws.send_json({
                "type": "voice",
                "audio": audio_b64,
                "mime_type": "audio/webm",
            })

            ws.receive_json()  # typing

            msg = ws.receive_json()
            assert msg["type"] == "text"
            assert msg["content"] == "Voice response"
            assert msg["speak"] is True

            msg = ws.receive_json()
            assert msg["type"] == "done"

        manager.gemini_respond_multimodal.assert_called_once()
        call_kwargs = manager.gemini_respond_multimodal.call_args
        assert call_kwargs.kwargs.get("is_voice") is True

    def test_ws_file_upload(self, ws_app):
        """File upload → text response."""
        client = ws_app["client"]
        manager = ws_app["manager"]

        manager.gemini_respond_multimodal = AsyncMock(
            return_value={"response": "File analysis"}
        )

        file_b64 = base64.b64encode(b"file-content").decode()

        with client.websocket_connect("/api/ws/chat") as ws:
            ws.send_json({
                "type": "file",
                "data": file_b64,
                "name": "test.txt",
                "mime_type": "text/plain",
                "text": "Analyze this",
            })

            ws.receive_json()  # typing

            msg = ws.receive_json()
            assert msg["type"] == "text"
            assert msg["content"] == "File analysis"

            msg = ws.receive_json()
            assert msg["type"] == "done"

        call_kwargs = manager.gemini_respond_multimodal.call_args
        assert call_kwargs.kwargs.get("text") == "Analyze this"

    def test_ws_new_session(self, ws_app):
        """New session → session_reset event."""
        client = ws_app["client"]
        manager = ws_app["manager"]

        with client.websocket_connect("/api/ws/chat") as ws:
            ws.send_json({"type": "new_session"})

            msg = ws.receive_json()
            assert msg["type"] == "session_reset"

        # Called twice: once for new_session message, once in finally block on WS disconnect.
        # Second call is idempotent (session already None).
        assert manager.close_session.call_count >= 1
        manager.get_or_create_session.assert_called()

    def test_ws_empty_message_ignored(self, ws_app):
        """Empty text message is silently ignored."""
        client = ws_app["client"]
        manager = ws_app["manager"]

        async def mock_stream(text):
            yield {"type": "token", "content": "OK"}
            yield {"type": "final", "text": "OK"}

        manager.gemini_respond_stream = mock_stream

        with client.websocket_connect("/api/ws/chat") as ws:
            ws.send_json({"type": "message", "text": ""})
            ws.send_json({"type": "message", "text": "real"})

            msg = ws.receive_json()
            assert msg["type"] == "typing"

            msg = ws.receive_json()
            assert msg == {"type": "token", "content": "OK"}

            msg = ws.receive_json()
            assert msg["type"] == "done"

    def test_ws_accepts_local_connection_without_auth_cookie(self):
        """The local desktop websocket accepts connections without auth cookies."""
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        with patch("remy.web.api.metrics_collector"):
            with client.websocket_connect("/api/ws/chat") as ws:
                assert ws is not None

    def test_ws_metrics_tracked(self, ws_app):
        """WebSocket connect/disconnect tracked in metrics."""
        client = ws_app["client"]
        metrics = ws_app["metrics"]
        manager = ws_app["manager"]

        async def mock_stream(text):
            yield {"type": "final", "text": "OK"}

        manager.gemini_respond_stream = mock_stream

        with client.websocket_connect("/api/ws/chat") as ws:
            ws.send_json({"type": "message", "text": "Hi"})
            ws.receive_json()  # typing
            ws.receive_json()  # text
            ws.receive_json()  # done

        metrics.ws_connected.assert_called_with("chat")
        metrics.ws_disconnected.assert_called_with("chat")


# ============== SECTION 3: AUTONOMOUS CYCLE ==============


@pytest.fixture
def auto_brain(tmp_path):
    """Real CognitiveMemory for autonomous cycle tests."""
    b = CognitiveMemory(str(tmp_path / "auto_brain"))
    yield b
    b.close()


def _make_eval_response(success=True, confidence=0.9, goal_completed=False):
    """Build a mock AIMessage with evaluation JSON in content."""
    import json
    mock_msg = MagicMock()
    mock_msg.content = json.dumps({
        "success": success,
        "confidence": confidence,
        "reason": "Test evaluation",
        "goal_completed": goal_completed,
    })
    return mock_msg


def _auto_patches(brain_instance):
    """Common patches for AutonomousLoop tests."""
    return {
        "brain": patch("remy.core.autonomy.brain", brain_instance),
        "invoke": patch(
            "remy.core.agent.invoke_agent",
            new_callable=AsyncMock,
            return_value=("Agent response text", [], []),
        ),
        "call_llm": patch(
            "remy.core.llm.call_llm_async",
            new_callable=AsyncMock,
            return_value=_make_eval_response(),
        ),
        "bg": patch(
            "remy.core.background_brain.run_background",
            return_value=None,
        ),
        "settings": patch("remy.core.autonomy.settings"),
        "sleep": patch("remy.core.autonomy.asyncio.sleep", new_callable=AsyncMock),
        "usage": patch("remy.core.usage_stats.usage_tracker"),
        "save_budget": patch("remy.core.autonomy.save_budget"),
    }


def _configure_auto_settings(mock_s):
    mock_s.AUTONOMY_DAILY_TOKEN_LIMIT = 100000
    mock_s.AUTONOMY_HOURLY_TOKEN_LIMIT = 50000
    mock_s.AUTONOMY_SESSION_TOKEN_LIMIT = 25000
    mock_s.AUTONOMY_CYCLE_INTERVAL_SEC = 1
    mock_s.AUTONOMY_MAX_SESSION_MINUTES = 30
    mock_s.AUTONOMY_QUIET_HOURS_START = 23
    mock_s.AUTONOMY_QUIET_HOURS_END = 7
    mock_s.AUTONOMY_MAX_ACTIONS_PER_HOUR = 20
    mock_s.AUTONOMY_TELEGRAM_NOTIFICATIONS = False
    mock_s.AUTONOMY_ALLOWED_READ_PATHS = []
    mock_s.DATA_DIR = "/tmp/data"
    mock_s.PROACTIVE_CHAT_ID = None
    mock_s.TELEGRAM_BOT_TOKEN = None


class TestAutonomousCycleIntegration:

    @pytest.mark.asyncio
    async def test_full_success_cycle(self, auto_brain):
        """Full cycle: goal → invoke → evaluate → record outcome."""
        patches = _auto_patches(auto_brain)
        with patches["brain"], patches["invoke"] as mock_invoke, \
             patches["call_llm"], patches["bg"], patches["settings"] as ms, \
             patches["sleep"], patches["usage"], patches["save_budget"]:
            _configure_auto_settings(ms)

            from remy.core.autonomy import AutonomousLoop, create_goal
            create_goal("Test integration goal", priority="high")

            loop = AutonomousLoop()
            loop._is_quiet_hours = MagicMock(return_value=False)
            loop._should_start_proactive_session = MagicMock(return_value=None)

            await loop._cycle()

            mock_invoke.assert_called_once()
            assert len(loop.action_log) == 1
            assert loop.action_log[0].success is True
            assert loop.consecutive_failures == 0
            assert loop.budget.tokens_today > 0

            outcomes = auto_brain.search(query="", tags=["autonomous-outcome"], limit=10)
            assert len(outcomes) >= 1

    @pytest.mark.asyncio
    async def test_cycle_failure_increments_counter(self, auto_brain):
        """Failed evaluation → consecutive_failures increments."""
        patches = _auto_patches(auto_brain)
        with patches["brain"], patches["invoke"], \
             patches["call_llm"] as mock_llm, patches["bg"], \
             patches["settings"] as ms, patches["sleep"], \
             patches["usage"], patches["save_budget"]:
            _configure_auto_settings(ms)
            mock_llm.return_value = _make_eval_response(success=False, confidence=0.7)

            from remy.core.autonomy import AutonomousLoop, create_goal
            create_goal("Failing goal", priority="high")

            loop = AutonomousLoop()
            loop._is_quiet_hours = MagicMock(return_value=False)
            loop._should_start_proactive_session = MagicMock(return_value=None)

            await loop._cycle()

            assert loop.consecutive_failures == 1
            assert loop.action_log[0].success is False

            outcomes = auto_brain.search(query="", tags=["outcome-failure"], limit=10)
            assert len(outcomes) >= 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_after_three_failures(self, auto_brain):
        """3 consecutive failures → skip action, sleep 600s."""
        patches = _auto_patches(auto_brain)
        with patches["brain"], patches["invoke"] as mock_invoke, \
             patches["call_llm"], patches["bg"], patches["settings"] as ms, \
             patches["sleep"] as mock_sleep, patches["usage"], patches["save_budget"]:
            _configure_auto_settings(ms)

            from remy.core.autonomy import AutonomousLoop, create_goal
            create_goal("Goal", priority="high")

            loop = AutonomousLoop()
            loop._is_quiet_hours = MagicMock(return_value=False)
            loop._should_start_proactive_session = MagicMock(return_value=None)
            loop.consecutive_failures = 3

            await loop._cycle()

            mock_invoke.assert_not_called()
            mock_sleep.assert_any_call(600)
            assert loop.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_budget_exhaustion_skips_action(self, auto_brain):
        """Budget exceeded → no action taken."""
        patches = _auto_patches(auto_brain)
        with patches["brain"], patches["invoke"] as mock_invoke, \
             patches["call_llm"], patches["bg"], patches["settings"] as ms, \
             patches["sleep"], patches["usage"], patches["save_budget"]:
            _configure_auto_settings(ms)

            from remy.core.autonomy import AutonomousLoop, create_goal
            create_goal("Goal", priority="high")

            loop = AutonomousLoop()
            loop._is_quiet_hours = MagicMock(return_value=False)
            loop._should_start_proactive_session = MagicMock(return_value=None)
            loop.budget.tokens_this_session = loop.budget.session_limit

            await loop._cycle()

            mock_invoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_quiet_hours_skips_cycle(self, auto_brain):
        """Quiet hours → cycle returns immediately."""
        patches = _auto_patches(auto_brain)
        with patches["brain"], patches["invoke"] as mock_invoke, \
             patches["call_llm"], patches["bg"], patches["settings"] as ms, \
             patches["sleep"] as mock_sleep, patches["usage"], patches["save_budget"]:
            _configure_auto_settings(ms)

            from remy.core.autonomy import AutonomousLoop
            loop = AutonomousLoop()
            loop._is_quiet_hours = MagicMock(return_value=True)

            await loop._cycle()

            mock_invoke.assert_not_called()
            mock_sleep.assert_any_call(300)

    @pytest.mark.asyncio
    async def test_goal_auto_completion(self, auto_brain):
        """Evaluation says goal_completed=True → goal marked completed."""
        patches = _auto_patches(auto_brain)
        with patches["brain"], patches["invoke"], \
             patches["call_llm"] as mock_llm, patches["bg"], \
             patches["settings"] as ms, patches["sleep"], \
             patches["usage"], patches["save_budget"]:
            _configure_auto_settings(ms)
            mock_llm.return_value = _make_eval_response(
                success=True, confidence=0.95, goal_completed=True,
            )

            from remy.core.autonomy import (
                AutonomousLoop, create_goal, get_active_goals,
            )
            create_goal("Complete me", priority="high")

            loop = AutonomousLoop()
            loop._is_quiet_hours = MagicMock(return_value=False)
            loop._should_start_proactive_session = MagicMock(return_value=None)

            await loop._cycle()

            active = get_active_goals()
            assert len(active) == 0

    @pytest.mark.asyncio
    async def test_maintenance_mode_skips_action(self, auto_brain):
        """maintenance_only=True + LLM still down → only background tasks."""
        patches = _auto_patches(auto_brain)
        with patches["brain"], patches["invoke"] as mock_invoke, \
             patches["call_llm"], patches["bg"] as mock_bg, \
             patches["settings"] as ms, patches["sleep"], \
             patches["usage"], patches["save_budget"]:
            _configure_auto_settings(ms)

            from remy.core.autonomy import AutonomousLoop
            loop = AutonomousLoop()
            loop._is_quiet_hours = MagicMock(return_value=False)
            loop._should_start_proactive_session = MagicMock(return_value=None)
            loop.maintenance_only = True
            loop._test_llm_health = AsyncMock(return_value=False)

            await loop._cycle()

            mock_invoke.assert_not_called()
            mock_bg.assert_called_once()

    @pytest.mark.asyncio
    async def test_maintenance_mode_recovery(self, auto_brain):
        """maintenance_only=True + LLM recovers → normal action."""
        patches = _auto_patches(auto_brain)
        with patches["brain"], patches["invoke"] as mock_invoke, \
             patches["call_llm"], patches["bg"], patches["settings"] as ms, \
             patches["sleep"], patches["usage"], patches["save_budget"]:
            _configure_auto_settings(ms)

            from remy.core.autonomy import AutonomousLoop, create_goal
            create_goal("Recovery goal", priority="high")

            loop = AutonomousLoop()
            loop._is_quiet_hours = MagicMock(return_value=False)
            loop._should_start_proactive_session = MagicMock(return_value=None)
            loop.maintenance_only = True
            loop.consecutive_llm_failures = 5
            loop._test_llm_health = AsyncMock(return_value=True)

            await loop._cycle()

            assert loop.maintenance_only is False
            assert loop.consecutive_llm_failures == 0
            mock_invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_failures_enter_maintenance(self, auto_brain):
        """5 consecutive LLM failures → enter maintenance-only mode."""
        patches = _auto_patches(auto_brain)
        with patches["brain"], patches["invoke"] as mock_invoke, \
             patches["call_llm"], patches["bg"], patches["settings"] as ms, \
             patches["sleep"], patches["usage"], patches["save_budget"]:
            _configure_auto_settings(ms)
            mock_invoke.side_effect = RuntimeError("LLM down")

            from remy.core.autonomy import AutonomousLoop, create_goal
            create_goal("Doomed goal", priority="high")

            loop = AutonomousLoop()
            loop._is_quiet_hours = MagicMock(return_value=False)
            loop._should_start_proactive_session = MagicMock(return_value=None)
            loop.consecutive_llm_failures = 4

            await loop._cycle()

            assert loop.maintenance_only is True
            assert loop.consecutive_llm_failures == 5
