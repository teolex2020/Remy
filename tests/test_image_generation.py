"""Tests for image generation tool."""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ============== TOOL DECLARATION ==============


def test_generate_image_tool_exists():
    """generate_image should be declared in BRAIN_TOOLS."""
    from remy.core.brain_tools import BRAIN_TOOLS

    names = [t.name for t in BRAIN_TOOLS]
    assert "generate_image" in names


def test_generate_image_tool_has_prompt_param():
    """generate_image should require a 'prompt' parameter."""
    from remy.core.brain_tools import BRAIN_TOOLS

    decl = next(t for t in BRAIN_TOOLS if t.name == "generate_image")
    assert "prompt" in decl.parameters.properties
    assert "prompt" in decl.parameters.required


# ============== IMAGE GENERATION HELPER ==============


class TestGenerateImage:

    @patch("remy.core.brain_tools.brain")
    @patch("remy.core.brain_tools.settings")
    def test_success(self, mock_settings, mock_brain):
        """Successful image generation saves file and brain record."""
        from remy.core.brain_tools import _generate_image

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.DATA_DIR = Path(tmpdir)
            mock_settings.GEMINI_API_KEY = "test-key"

            # Mock brain.store
            mock_rec = MagicMock()
            mock_rec.id = "img-001"
            mock_brain.store.return_value = mock_rec

            # Mock streaming chunk with inline image
            mock_part = MagicMock()
            mock_part.inline_data = MagicMock()
            mock_part.inline_data.data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
            mock_part.inline_data.mime_type = "image/png"
            mock_part.text = None

            mock_chunk = MagicMock()
            mock_chunk.parts = [mock_part]

            import google.genai

            with patch.object(google.genai, "Client", return_value=MagicMock()) as mock_client_cls:
                client_inst = mock_client_cls.return_value
                # generate_content_stream returns an iterable of chunks
                client_inst.models.generate_content_stream.return_value = [mock_chunk]

                result = _generate_image(
                    {"prompt": "a sunset over mountains"},
                    session_id="test-session",
                    channel="desktop",
                )

            parsed = json.loads(result)
            assert parsed["generated"] is True
            assert parsed["prompt"] == "a sunset over mountains"
            assert "filename" in parsed
            assert parsed["url"].startswith("/api/generated_images/")
            assert parsed["record_id"] == "img-001"

            # File should exist on disk
            image_dir = Path(tmpdir) / "generated_images"
            files = list(image_dir.glob("gen_*.png"))
            assert len(files) == 1

            # Brain record should be stored
            mock_brain.store.assert_called_once()
            call_kwargs = mock_brain.store.call_args
            assert "generated-image" in call_kwargs.kwargs.get("tags", call_kwargs[1].get("tags", []))

    @patch("remy.core.brain_tools.brain")
    @patch("remy.core.brain_tools.settings")
    def test_no_image_returned(self, mock_settings, mock_brain):
        """When API returns text but no image, return error."""
        from remy.core.brain_tools import _generate_image

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.DATA_DIR = Path(tmpdir)
            mock_settings.GEMINI_API_KEY = "test-key"

            # Mock streaming chunk with text only (no inline_data)
            mock_part = MagicMock()
            mock_part.inline_data = None
            mock_part.text = "I cannot generate that image"

            mock_chunk = MagicMock()
            mock_chunk.parts = [mock_part]

            import google.genai

            with patch.object(google.genai, "Client", return_value=MagicMock()) as mock_client_cls:
                client_inst = mock_client_cls.return_value
                client_inst.models.generate_content_stream.return_value = [mock_chunk]

                result = _generate_image(
                    {"prompt": "something"},
                    session_id=None,
                    channel=None,
                )

            parsed = json.loads(result)
            assert parsed["generated"] is False
            assert "error" in parsed
            assert "cannot generate" in parsed["error"]

    @patch("remy.core.brain_tools.brain")
    @patch("remy.core.brain_tools.settings")
    def test_no_candidates(self, mock_settings, mock_brain):
        """When API returns empty stream, return error."""
        from remy.core.brain_tools import _generate_image

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.DATA_DIR = Path(tmpdir)
            mock_settings.GEMINI_API_KEY = "test-key"

            # Empty chunk with no parts
            mock_chunk = MagicMock()
            mock_chunk.parts = None

            import google.genai

            with patch.object(google.genai, "Client", return_value=MagicMock()) as mock_client_cls:
                client_inst = mock_client_cls.return_value
                client_inst.models.generate_content_stream.return_value = [mock_chunk]

                result = _generate_image(
                    {"prompt": "test"},
                    session_id=None,
                    channel=None,
                )

            parsed = json.loads(result)
            assert parsed["generated"] is False

    @patch("remy.core.brain_tools.brain")
    @patch("remy.core.brain_tools.settings")
    def test_api_error(self, mock_settings, mock_brain):
        """When Gemini API raises, execute_tool returns Error string."""
        from remy.core.brain_tools import _generate_image

        mock_settings.DATA_DIR = Path(tempfile.gettempdir())
        mock_settings.GEMINI_API_KEY = "test-key"

        import google.genai

        with patch.object(google.genai, "Client", return_value=MagicMock()) as mock_client_cls:
            client_inst = mock_client_cls.return_value
            client_inst.models.generate_content_stream.side_effect = Exception("Rate limit exceeded")

            with pytest.raises(Exception, match="Rate limit"):
                _generate_image({"prompt": "test"}, None, None)


# ============== API ENDPOINT ==============


class TestServeGeneratedImage:

    def _make_client(self):
        from fastapi.testclient import TestClient
        from remy.web.api import router
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)  # router already has prefix="/api"
        return TestClient(app)

    def test_serve_image(self):
        """Endpoint serves existing image file."""
        from remy.config.settings import settings

        image_dir = Path(settings.DATA_DIR) / "generated_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        test_image = image_dir / "gen_test1234.png"
        test_image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
        try:
            client = self._make_client()
            resp = client.get("/api/generated_images/gen_test1234.png")
            assert resp.status_code == 200
            assert "image/png" in resp.headers["content-type"]
        finally:
            test_image.unlink(missing_ok=True)

    def test_not_found(self):
        """Endpoint returns 404 for missing file."""
        client = self._make_client()
        resp = client.get("/api/generated_images/nonexistent_xyz.png")
        assert resp.status_code == 404

    def test_path_traversal_blocked(self):
        """Endpoint blocks path traversal attempts."""
        client = self._make_client()
        resp = client.get("/api/generated_images/..%2F..%2Fetc%2Fpasswd")
        assert resp.status_code in (404, 422)


# ============== TELEGRAM IMAGE DETECTION ==============


def test_telegram_image_regex():
    """Regex correctly finds image URL in response text."""
    import re

    response = "Here's your image! ![sunset](/api/generated_images/gen_abc12345.png)"
    match = re.search(r'/api/generated_images/([\w.]+)', response)
    assert match is not None
    assert match.group(1) == "gen_abc12345.png"


def test_telegram_no_image_in_text():
    """Regex returns None when no image URL present."""
    import re

    response = "Hello! How can I help you today?"
    match = re.search(r'/api/generated_images/([\w.]+)', response)
    assert match is None
