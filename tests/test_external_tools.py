"""Tests for External Tools — read_file, write_file, list_directory, http_get."""

import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from aura import Aura as CognitiveMemory


@pytest.fixture
def tools_env(tmp_path):
    """Isolated environment for external tool tests."""
    brain = CognitiveMemory(str(tmp_path / "tools_brain"))
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    with patch("remy.core.brain_tools.settings") as mock_settings:
        mock_settings.DATA_DIR = data_dir
        mock_settings.AUTONOMY_ALLOWED_READ_PATHS = []
        mock_settings.GEMINI_API_KEY = "test-key"
        mock_settings.SUMMARY_MODEL = "test-model"

        with patch("remy.core.brain_tools.brain", brain):
            yield {
                "brain": brain,
                "data_dir": data_dir,
                "settings": mock_settings,
            }

    brain.close()


class TestReadFile:
    """Tests for read_file tool."""

    def test_read_file_in_data_dir(self, tools_env):
        from remy.core.brain_tools import execute_tool

        # Create a test file
        test_file = tools_env["data_dir"] / "test.txt"
        test_file.write_text("Hello, World!", encoding="utf-8")

        result = execute_tool("read_file", {"path": "test.txt"})
        data = json.loads(result)

        assert "content" in data
        assert data["content"] == "Hello, World!"
        assert data["size"] == 13

    def test_read_file_outside_data_dir_denied(self, tools_env):
        from remy.core.brain_tools import execute_tool

        result = execute_tool("read_file", {"path": "C:/Windows/System32/drivers/etc/hosts"})
        data = json.loads(result)

        assert "error" in data
        assert "denied" in data["error"].lower() or "outside" in data["error"].lower()

    def test_read_nonexistent_file(self, tools_env):
        from remy.core.brain_tools import execute_tool

        result = execute_tool("read_file", {"path": "nonexistent.txt"})
        data = json.loads(result)

        assert "error" in data
        assert "not found" in data["error"].lower()

    def test_read_file_in_allowed_path(self, tools_env):
        from remy.core.brain_tools import execute_tool

        # Add an allowed path
        allowed_dir = tools_env["data_dir"].parent / "allowed"
        allowed_dir.mkdir()
        (allowed_dir / "ok.txt").write_text("Allowed content", encoding="utf-8")
        tools_env["settings"].AUTONOMY_ALLOWED_READ_PATHS = [str(allowed_dir)]

        result = execute_tool("read_file", {"path": str(allowed_dir / "ok.txt")})
        data = json.loads(result)

        assert data["content"] == "Allowed content"


class TestWriteFile:
    """Tests for write_file tool."""

    def test_write_file_in_data_dir(self, tools_env):
        from remy.core.brain_tools import execute_tool

        result = execute_tool("write_file", {
            "path": "output.txt",
            "content": "Generated content",
        })
        data = json.loads(result)

        assert data["written"] is True
        assert (tools_env["data_dir"] / "output.txt").read_text() == "Generated content"

    def test_write_file_creates_subdirs(self, tools_env):
        from remy.core.brain_tools import execute_tool

        result = execute_tool("write_file", {
            "path": "subdir/deep/file.txt",
            "content": "Deep file",
        })
        data = json.loads(result)

        assert data["written"] is True
        assert (tools_env["data_dir"] / "subdir" / "deep" / "file.txt").exists()

    def test_write_file_outside_data_dir_denied(self, tools_env):
        from remy.core.brain_tools import execute_tool

        result = execute_tool("write_file", {
            "path": "C:/Windows/malicious.txt",
            "content": "evil",
        })
        data = json.loads(result)

        assert "error" in data
        assert "denied" in data["error"].lower() or "outside" in data["error"].lower()


class TestListDirectory:
    """Tests for list_directory tool."""

    def test_list_data_dir(self, tools_env):
        from remy.core.brain_tools import execute_tool

        # Create some files
        (tools_env["data_dir"] / "a.txt").write_text("a")
        (tools_env["data_dir"] / "b.txt").write_text("b")
        (tools_env["data_dir"] / "subdir").mkdir()

        result = execute_tool("list_directory", {"path": "."})
        data = json.loads(result)

        assert "entries" in data
        names = [e["name"] for e in data["entries"]]
        assert "a.txt" in names
        assert "b.txt" in names
        assert "subdir" in names

    def test_list_outside_data_dir_denied(self, tools_env):
        from remy.core.brain_tools import execute_tool

        result = execute_tool("list_directory", {"path": "C:/Windows/System32"})
        data = json.loads(result)

        assert "error" in data


class TestHttpGet:
    """Tests for http_get tool."""

    def test_http_get_success(self, tools_env):
        from remy.core.brain_tools import execute_tool

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"result": "ok"}'
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = execute_tool("http_get", {"url": "https://api.example.com/data"})

        data = json.loads(result)
        assert data["status"] == 200
        assert "result" in data["body"]

    def test_http_get_error(self, tools_env):
        from remy.core.brain_tools import execute_tool

        with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
            result = execute_tool("http_get", {"url": "https://fail.example.com"})

        data = json.loads(result)
        assert "error" in data
