"""Tests for Setup Wizard and Settings/Export/Diagnostics API endpoints."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from aura import Level
from fastapi.testclient import TestClient

from remy.web.api import router, set_session_manager
from remy.web.session import WebSession, WebSessionManager


# ============== SETUP WIZARD TESTS ==============


class TestSetupModule:
    """Tests for core/setup.py functions."""

    def test_needs_setup_true_when_no_key(self):
        with patch("remy.core.setup.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = None
            with patch.dict(os.environ, {}, clear=True):
                from remy.core.setup import needs_setup
                assert needs_setup() is True

    def test_needs_setup_false_when_key_set(self):
        with patch("remy.core.setup.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = "test-key-123"
            from remy.core.setup import needs_setup
            assert needs_setup() is False

    def test_ensure_directories(self, tmp_path):
        with patch("remy.core.setup.settings") as mock_settings:
            mock_settings.DATA_DIR = tmp_path / "data"
            from remy.core.setup import ensure_directories
            ensure_directories()
            assert (tmp_path / "data").exists()
            assert (tmp_path / "data" / "logs").exists()
            assert (tmp_path / "data" / "sandbox").exists()

    def test_update_env_value_creates_file(self, tmp_path):
        env_file = tmp_path / ".env"
        with patch("remy.core.setup.ENV_FILE", env_file):
            from remy.core.setup import update_env_value
            update_env_value("TEST_KEY", "test_value")
            content = env_file.read_text()
            assert "TEST_KEY=test_value" in content

    def test_update_env_value_replaces_existing(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_KEY=old_value\nOTHER=keep\n")
        with patch("remy.core.setup.ENV_FILE", env_file):
            from remy.core.setup import update_env_value
            update_env_value("TEST_KEY", "new_value")
            content = env_file.read_text()
            assert "TEST_KEY=new_value" in content
            assert "OTHER=keep" in content
            assert "old_value" not in content

    def test_get_env_value_reads(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MY_KEY=my_val\n")
        with patch("remy.core.setup.ENV_FILE", env_file):
            from remy.core.setup import get_env_value
            assert get_env_value("MY_KEY") == "my_val"
            assert get_env_value("MISSING") is None

    def test_get_env_value_no_file(self, tmp_path):
        env_file = tmp_path / ".env.nonexistent"
        with patch("remy.core.setup.ENV_FILE", env_file):
            from remy.core.setup import get_env_value
            assert get_env_value("ANYTHING") is None

    def test_set_runtime_setting_writes_data_file(self, tmp_path):
        runtime_file = tmp_path / "runtime_settings.json"
        with patch("remy.config.settings.RUNTIME_SETTINGS_FILE", runtime_file):
            from remy.config.settings import set_runtime_setting, settings

            set_runtime_setting("SUMMARY_MODEL", "gemini-2.0-flash", target=settings)

            data = json.loads(runtime_file.read_text(encoding="utf-8"))
            assert data["SUMMARY_MODEL"] == "gemini-2.0-flash"


# ============== READONLY MODE TESTS ==============


class TestReadonlyMode:
    """Tests for WebSessionManager in readonly mode (no API key)."""

    def test_readonly_when_no_api_key(self):
        with patch("remy.web.session.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = None
            with patch.dict(os.environ, {}, clear=True):
                manager = WebSessionManager()
                assert manager.readonly is True
                assert manager.client is None

    def test_not_readonly_when_api_key_set(self):
        with patch("remy.web.session.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = "test-key"
            manager = WebSessionManager()
            assert manager.readonly is False

    def test_refresh_credentials_enables_chat_after_key_is_added(self):
        with patch("remy.web.session.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = None
            with patch.dict(os.environ, {}, clear=True):
                manager = WebSessionManager()
                assert manager.readonly is True
                assert manager.client is None

                mock_settings.GEMINI_API_KEY = "test-key"
                with patch("remy.web.session.genai.Client", return_value=object()) as client_cls:
                    manager.refresh_credentials()

                assert manager.readonly is False
                assert manager.client is not None
                client_cls.assert_called_once_with(api_key="test-key")

    @pytest.mark.asyncio
    async def test_readonly_respond_returns_message(self):
        with patch("remy.web.session.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = None
            with patch.dict(os.environ, {}, clear=True):
                manager = WebSessionManager()
                result = await manager.gemini_respond("hello")
                assert "API key" in result
                assert "Settings" in result

    @pytest.mark.asyncio
    async def test_readonly_multimodal_returns_message(self):
        with patch("remy.web.session.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = None
            with patch.dict(os.environ, {}, clear=True):
                manager = WebSessionManager()
                result = await manager.gemini_respond_multimodal(text="hello")
                assert "API key" in result["response"]

    @pytest.mark.asyncio
    async def test_readonly_close_session_no_crash(self):
        with patch("remy.web.session.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = None
            with patch.dict(os.environ, {}, clear=True):
                manager = WebSessionManager()
                manager.get_or_create_session()
                await manager.close_session()
                assert manager.session is None


# ============== API ENDPOINT TESTS ==============


@pytest.fixture
def mock_brain(tmp_path):
    from aura import Aura as CognitiveMemory
    b = CognitiveMemory(str(tmp_path / "test_brain"))
    yield b
    b.close()


@pytest.fixture
def mock_session_manager():
    manager = MagicMock()
    session = WebSession(session_id="test-session-123")
    manager.get_or_create_session.return_value = session
    return manager


@pytest.fixture
def client(mock_brain, mock_session_manager):
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    set_session_manager(mock_session_manager)

    with patch("remy.web.api.brain", mock_brain):
        yield TestClient(app)


class TestSettingsEndpoint:

    def test_get_settings(self, client):
        with patch("remy.web.api.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = "test-key-123456789"
            mock_settings.SUMMARY_MODEL = "gemini-3-flash-preview"
            mock_settings.GEMINI_VOICE = "Zephyr"
            mock_settings.TELEGRAM_BOT_TOKEN = None
            mock_settings.PROACTIVE_CHAT_ID = None
            mock_settings.WEB_HOST = "127.0.0.1"
            mock_settings.WEB_PORT = 8080
            res = client.get("/api/settings")

        assert res.status_code == 200
        data = res.json()
        assert data["has_api_key"] is True
        assert "..." in data["gemini_api_key_masked"]
        assert data["summary_model"] == "gemini-3-flash-preview"
        assert data["proactive_chat_id"] is None

    def test_get_settings_no_key(self, client):
        with patch("remy.web.api.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = None
            mock_settings.SUMMARY_MODEL = "gemini-3-flash-preview"
            mock_settings.GEMINI_VOICE = "Zephyr"
            mock_settings.TELEGRAM_BOT_TOKEN = None
            mock_settings.PROACTIVE_CHAT_ID = None
            mock_settings.WEB_HOST = "127.0.0.1"
            mock_settings.WEB_PORT = 8080
            with patch.dict(os.environ, {}, clear=True):
                res = client.get("/api/settings")

        data = res.json()
        assert data["has_api_key"] is False

    def test_update_settings(self, client, tmp_path):
        runtime_file = tmp_path / "runtime_settings.json"
        with patch("remy.config.settings.RUNTIME_SETTINGS_FILE", runtime_file):
            res = client.put("/api/settings", json={"summary_model": "gemini-2.0-flash"})

        assert res.status_code == 200
        data = res.json()
        assert "SUMMARY_MODEL" in data["updated"]
        assert "apply immediately" in data["note"]
        saved = json.loads(runtime_file.read_text(encoding="utf-8"))
        assert saved["SUMMARY_MODEL"] == "gemini-2.0-flash"

    def test_update_api_key_refreshes_session_credentials(
        self, client, tmp_path, mock_session_manager
    ):
        runtime_file = tmp_path / "runtime_settings.json"
        mock_session_manager.refresh_credentials = MagicMock()

        with patch("remy.config.settings.RUNTIME_SETTINGS_FILE", runtime_file):
            res = client.put("/api/settings", json={"gemini_api_key": "test-key"})

        assert res.status_code == 200
        assert "GEMINI_API_KEY" in res.json()["updated"]
        mock_session_manager.refresh_credentials.assert_called_once()


class TestExportEndpoint:

    def test_export_empty_brain(self, client):
        res = client.get("/api/export")
        assert res.status_code == 200
        data = res.json()
        assert data["count"] == 0
        assert data["records"] == []
        assert "exported_at" in data

    def test_export_with_records(self, client, mock_brain):
        mock_brain.store("Record 1", tags=["test"], level=Level.DOMAIN)
        mock_brain.store("Record 2", tags=["test"], level=Level.DOMAIN)

        res = client.get("/api/export")
        data = res.json()
        assert data["count"] >= 2
        assert len(data["records"]) >= 2
        assert data["records"][0]["content"] is not None

    def test_export_has_download_header(self, client):
        res = client.get("/api/export")
        assert "content-disposition" in res.headers
        assert "remy-brain-export" in res.headers["content-disposition"]


class TestDiagnosticsEndpoint:

    def test_diagnostics_basic(self, client):
        with patch("remy.web.api.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = "test-key"
            mock_settings.SUMMARY_MODEL = "gemini-3-flash-preview"
            mock_settings.AURA_BRAIN_PATH = Path("/tmp/brain")
            mock_settings.TELEGRAM_BOT_TOKEN = None
            mock_settings.SANDBOX_DIR = Path("/tmp/sandbox")
            res = client.get("/api/diagnostics")

        assert res.status_code == 200
        data = res.json()
        assert data["status"] in ("ok", "degraded")
        assert "uptime" in data
        assert "platform" in data
        assert "brain" in data
        assert "records" in data["brain"]

    def test_diagnostics_degraded_without_key(self, client):
        with patch("remy.web.api.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = None
            mock_settings.SUMMARY_MODEL = "gemini-3-flash-preview"
            mock_settings.AURA_BRAIN_PATH = Path("/tmp/brain")
            mock_settings.TELEGRAM_BOT_TOKEN = None
            mock_settings.SANDBOX_DIR = Path("/tmp/sandbox")
            with patch.dict(os.environ, {}, clear=True):
                res = client.get("/api/diagnostics")

        data = res.json()
        assert data["status"] == "degraded"
        assert data["api_key_configured"] is False
