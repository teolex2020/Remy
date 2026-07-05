"""Tests for PinchTab sidecar autostart integration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCombinedRunnerPinchTab:
    @pytest.mark.asyncio
    async def test_run_combined_starts_and_stops_pinchtab(self):
        shutdown_ref = {}
        original_event = asyncio.Event

        def patched_event():
            event = original_event()
            shutdown_ref["event"] = event
            return event

        async def run_uvicorn_until_test_shutdown(_server):
            shutdown_ref["event"].set()

        with patch("remy.core.combined_runner.ensure_pinchtab_running", new=AsyncMock()) as mock_ensure, \
             patch("remy.core.combined_runner.shutdown_pinchtab", new=AsyncMock()) as mock_shutdown, \
             patch("remy.core.combined_runner._print_combined_banner"), \
             patch("remy.core.combined_runner._create_uvicorn_server", return_value=MagicMock()), \
             patch("remy.core.combined_runner._run_uvicorn_safe", side_effect=run_uvicorn_until_test_shutdown), \
             patch("remy.core.combined_runner._stop_uvicorn", new=AsyncMock()), \
             patch("remy.core.combined_runner.set_web_runtime_enabled"), \
             patch("remy.core.combined_runner.asyncio.Event", side_effect=patched_event), \
             patch("remy.core.combined_runner.settings") as mock_settings:
            mock_settings.WEB_HOST = "127.0.0.1"
            mock_settings.WEB_PORT = 8080

            from remy.core.combined_runner import run_combined
            await run_combined(autonomous=False, telegram=False, web=True)

            mock_ensure.assert_awaited_once()
            mock_shutdown.assert_awaited_once()


class TestDesktopGuiPinchTab:
    def test_run_web_only_starts_and_stops_sidecar(self):
        with patch("remy.core.desktop_gui.ensure_pinchtab_running_sync") as mock_ensure, \
             patch("remy.core.desktop_gui.shutdown_pinchtab_sync") as mock_shutdown, \
             patch("remy.core.desktop_gui.set_web_runtime_enabled"), \
             patch("remy.core.desktop_gui.DesktopGUI._print_banner"), \
             patch("remy.core.desktop_gui.DesktopGUI._run_server"):
            from remy.core.desktop_gui import DesktopGUI

            gui = DesktopGUI()
            gui.run_web_only()

            mock_ensure.assert_called_once_with()
            mock_shutdown.assert_called_once_with()

    def test_quiet_asyncio_handler_suppresses_proactor_cleanup_noise(self):
        from remy.core.desktop_gui import _install_quiet_asyncio_handler

        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        default_handler = MagicMock()
        loop.get_exception_handler.return_value = None
        loop.default_exception_handler = default_handler

        _install_quiet_asyncio_handler(loop)

        handler = loop.set_exception_handler.call_args[0][0]
        handler(loop, {"message": "Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)"})
        default_handler.assert_not_called()

        handler(loop, {"message": "Other asyncio error"})
        default_handler.assert_called_once()


class TestAutonomyEntryPointPinchTab:
    @pytest.mark.asyncio
    async def test_autonomy_entrypoint_wraps_sidecar_lifecycle(self):
        marker = {"ran": False}

        async def _job():
            marker["ran"] = True

        with patch("remy.core.pinchtab_service.ensure_pinchtab_running", new=AsyncMock()) as mock_ensure, \
             patch("remy.core.pinchtab_service.shutdown_pinchtab", new=AsyncMock()) as mock_shutdown:
            from remy.main import _run_autonomy_entrypoint

            await _run_autonomy_entrypoint(_job)

            assert marker["ran"] is True
            mock_ensure.assert_awaited_once()
            mock_shutdown.assert_awaited_once()
