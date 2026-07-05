"""Tests for Playwright browser tools: BrowserManager, browser_vision, brain_tools dispatch."""

import json
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============== BROWSER MANAGER TESTS ==============


class TestBrowserManagerSSRF:
    """SSRF protection tests — no Playwright needed."""

    def test_ssrf_blocks_localhost(self):
        from remy.core.browser import BrowserManager, SSRFError
        with pytest.raises(SSRFError, match="localhost"):
            BrowserManager.check_ssrf("http://localhost:8080/admin")

    def test_ssrf_blocks_127(self):
        from remy.core.browser import BrowserManager, SSRFError
        with pytest.raises(SSRFError, match="127.0.0.1"):
            BrowserManager.check_ssrf("http://127.0.0.1/secret")

    def test_ssrf_blocks_private_10(self):
        from remy.core.browser import BrowserManager, SSRFError
        with pytest.raises(SSRFError, match="private"):
            BrowserManager.check_ssrf("http://10.0.0.1/internal")

    def test_ssrf_blocks_private_192(self):
        from remy.core.browser import BrowserManager, SSRFError
        with pytest.raises(SSRFError, match="private"):
            BrowserManager.check_ssrf("http://192.168.1.1/router")

    def test_ssrf_blocks_link_local(self):
        from remy.core.browser import BrowserManager, SSRFError
        with pytest.raises(SSRFError, match="private"):
            BrowserManager.check_ssrf("http://169.254.169.254/metadata")

    def test_ssrf_blocks_file_protocol(self):
        from remy.core.browser import BrowserManager, SSRFError
        with pytest.raises(SSRFError, match="protocol"):
            BrowserManager.check_ssrf("file:///etc/passwd")

    def test_ssrf_blocks_ftp_protocol(self):
        from remy.core.browser import BrowserManager, SSRFError
        with pytest.raises(SSRFError, match="protocol"):
            BrowserManager.check_ssrf("ftp://internal.server/data")

    def test_ssrf_allows_public_https(self):
        from remy.core.browser import BrowserManager
        # Should not raise
        BrowserManager.check_ssrf("https://example.com/page")

    def test_ssrf_allows_public_http(self):
        from remy.core.browser import BrowserManager
        BrowserManager.check_ssrf("http://example.com/page")

    def test_ssrf_allows_hostname(self):
        from remy.core.browser import BrowserManager
        BrowserManager.check_ssrf("https://www.google.com/search?q=test")


class TestBrowserManagerDailyLimit:
    """Daily action limit tests."""

    def test_daily_limit_enforced(self):
        from remy.core.browser import BrowserManager, DailyLimitError

        mgr = BrowserManager()
        mgr._action_count_date = date.today()
        mgr._action_count_today = 200  # At limit

        with patch("remy.core.browser.settings") as mock_settings:
            mock_settings.BROWSER_DAILY_ACTION_LIMIT = 200
            with pytest.raises(DailyLimitError, match="Daily browser action limit"):
                mgr._check_daily_limit()

    def test_daily_limit_resets_on_new_day(self):
        from remy.core.browser import BrowserManager

        mgr = BrowserManager()
        mgr._action_count_date = date(2024, 1, 1)  # Old date
        mgr._action_count_today = 999

        with patch("remy.core.browser.settings") as mock_settings:
            mock_settings.BROWSER_DAILY_ACTION_LIMIT = 200
            # Should not raise — counter resets on new day
            mgr._check_daily_limit()
            assert mgr._action_count_today == 0

    def test_record_action_increments(self):
        from remy.core.browser import BrowserManager

        mgr = BrowserManager()
        mgr._action_count_date = date.today()
        mgr._action_count_today = 5

        mgr._record_action()
        assert mgr._action_count_today == 6
        assert mgr._last_activity > 0


class TestBrowserManagerSingleton:
    """Singleton pattern tests."""

    def test_singleton_returns_same_instance(self):
        from remy.core.browser import BrowserManager
        BrowserManager.reset()
        a = BrowserManager.get()
        b = BrowserManager.get()
        assert a is b
        BrowserManager.reset()

    def test_reset_clears_singleton(self):
        from remy.core.browser import BrowserManager
        BrowserManager.reset()
        a = BrowserManager.get()
        BrowserManager.reset()
        b = BrowserManager.get()
        assert a is not b
        BrowserManager.reset()

    def test_save_screenshot(self, tmp_path):
        from remy.core.browser import BrowserManager

        mgr = BrowserManager()
        png_data = b"\x89PNG\r\n\x1a\nfakeimage"

        with patch("remy.core.browser.settings") as mock_settings:
            mock_settings.DATA_DIR = tmp_path
            filename = mgr.save_screenshot(png_data)

        assert filename.startswith("ss_")
        assert filename.endswith(".png")
        saved = (tmp_path / "browser_screenshots" / filename).read_bytes()
        assert saved == png_data


class TestNormalizeSelector:
    """Selector normalization: jQuery → Playwright."""

    def test_contains_double_quotes(self):
        from remy.core.browser import BrowserManager
        assert BrowserManager._normalize_selector('button:contains("Accept all")') == 'button:has-text("Accept all")'

    def test_contains_single_quotes(self):
        from remy.core.browser import BrowserManager
        assert BrowserManager._normalize_selector("button:contains('OK')") == "button:has-text('OK')"

    def test_contains_no_quotes(self):
        from remy.core.browser import BrowserManager
        assert BrowserManager._normalize_selector("div:contains(Sign in)") == 'div:has-text("Sign in")'

    def test_css_selector_unchanged(self):
        from remy.core.browser import BrowserManager
        assert BrowserManager._normalize_selector("#submit-btn") == "#submit-btn"

    def test_playwright_has_text_unchanged(self):
        from remy.core.browser import BrowserManager
        assert BrowserManager._normalize_selector('button:has-text("OK")') == 'button:has-text("OK")'

    def test_nested_contains(self):
        from remy.core.browser import BrowserManager
        result = BrowserManager._normalize_selector('div[role="dialog"] button:contains("Accept")')
        assert ':has-text("Accept")' in result
        assert ':contains' not in result


# ============== BROWSER VISION TESTS ==============


class TestBrowserVision:

    @pytest.mark.asyncio
    async def test_analyze_screenshot_success(self):
        from remy.core.browser_vision import analyze_screenshot

        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "description": "A login page",
            "elements": [{"type": "input", "text": "Email", "selector": "#email", "purpose": "email field"}],
            "forms": [],
            "answer": None,
            "suggested_actions": ["Type email"],
        })

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch("remy.core.browser_vision.settings") as mock_settings, \
             patch("google.genai.Client", return_value=mock_client):
            mock_settings.GEMINI_API_KEY = "test-key"
            mock_settings.BROWSER_VISION_MODEL = "gemini-2.0-flash"

            result = await analyze_screenshot(
                screenshot_png=b"\x89PNGfake",
                page_url="https://example.com/login",
            )

        assert result["description"] == "A login page"
        assert len(result["elements"]) == 1
        assert result["elements"][0]["selector"] == "#email"

    @pytest.mark.asyncio
    async def test_analyze_screenshot_with_question(self):
        from remy.core.browser_vision import analyze_screenshot

        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "description": "A pricing page",
            "elements": [],
            "forms": [],
            "answer": "The basic plan costs $9.99/month",
            "suggested_actions": [],
        })

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch("remy.core.browser_vision.settings") as mock_settings, \
             patch("google.genai.Client", return_value=mock_client):
            mock_settings.GEMINI_API_KEY = "test-key"
            mock_settings.BROWSER_VISION_MODEL = "gemini-2.0-flash"

            result = await analyze_screenshot(
                screenshot_png=b"\x89PNGfake",
                question="How much does the basic plan cost?",
                page_url="https://example.com/pricing",
            )

        assert result["answer"] == "The basic plan costs $9.99/month"

    @pytest.mark.asyncio
    async def test_analyze_screenshot_json_parse_error(self):
        from remy.core.browser_vision import analyze_screenshot

        mock_response = MagicMock()
        mock_response.text = "This is not JSON, just a plain text description of the page."

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch("remy.core.browser_vision.settings") as mock_settings, \
             patch("google.genai.Client", return_value=mock_client):
            mock_settings.GEMINI_API_KEY = "test-key"
            mock_settings.BROWSER_VISION_MODEL = "gemini-2.0-flash"

            result = await analyze_screenshot(
                screenshot_png=b"\x89PNGfake",
                page_url="https://example.com",
            )

        # Fallback: description = raw text, empty elements/forms
        assert "This is not JSON" in result["description"]
        assert result["elements"] == []

    @pytest.mark.asyncio
    async def test_analyze_screenshot_api_error(self):
        from remy.core.browser_vision import analyze_screenshot

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API error")

        with patch("remy.core.browser_vision.settings") as mock_settings, \
             patch("google.genai.Client", return_value=mock_client):
            mock_settings.GEMINI_API_KEY = "test-key"
            mock_settings.BROWSER_VISION_MODEL = "gemini-2.0-flash"

            result = await analyze_screenshot(
                screenshot_png=b"\x89PNGfake",
                page_url="https://example.com",
            )

        assert "error" in result
        assert "API error" in result["error"]

    @pytest.mark.asyncio
    async def test_analyze_screenshot_strips_code_fences(self):
        from remy.core.browser_vision import analyze_screenshot

        mock_response = MagicMock()
        mock_response.text = '```json\n{"description": "test page", "elements": [], "forms": [], "answer": null, "suggested_actions": []}\n```'

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch("remy.core.browser_vision.settings") as mock_settings, \
             patch("google.genai.Client", return_value=mock_client):
            mock_settings.GEMINI_API_KEY = "test-key"
            mock_settings.BROWSER_VISION_MODEL = "gemini-2.0-flash"

            result = await analyze_screenshot(
                screenshot_png=b"\x89PNGfake",
                page_url="https://example.com",
            )

        assert result["description"] == "test page"


# ============== BRAIN_TOOLS DECLARATIONS ==============


class TestBrowserToolDeclarations:
    """Verify browser tools are declared in BRAIN_TOOLS."""

    def test_browse_page_in_brain_tools(self):
        from remy.core.brain_tools import BRAIN_TOOLS
        names = [t.name for t in BRAIN_TOOLS]
        assert "browse_page" in names

    def test_browser_act_in_brain_tools(self):
        from remy.core.brain_tools import BRAIN_TOOLS
        names = [t.name for t in BRAIN_TOOLS]
        assert "browser_act" in names

    def test_browser_close_in_brain_tools(self):
        from remy.core.brain_tools import BRAIN_TOOLS
        names = [t.name for t in BRAIN_TOOLS]
        assert "browser_close" in names


# ============== BROWSER TOOL DISPATCH ==============


class TestBrowserToolDispatch:
    """Test tool dispatch via execute_tool."""

    @pytest.fixture(autouse=True)
    def _patch_browser_settings(self, tmp_path):
        """Ensure settings.DATA_DIR is a real Path so mkdir() doesn't create MagicMock folders."""
        with patch("remy.core.browser.settings") as mock_browser_settings:
            mock_browser_settings.DATA_DIR = tmp_path
            mock_browser_settings.BROWSER_HEADLESS = True
            mock_browser_settings.BROWSER_IDLE_TIMEOUT_SEC = 300
            mock_browser_settings.BROWSER_DAILY_ACTION_LIMIT = 200
            mock_browser_settings.BROWSER_PAGE_TIMEOUT_MS = 30000
            yield mock_browser_settings

    def test_browse_page_disabled(self):
        """When BROWSER_ENABLED=False, returns error."""
        from remy.core.brain_tools import _handle_browser_tool

        with patch("remy.core.brain_tools.settings") as mock_settings:
            mock_settings.BROWSER_ENABLED = False
            result = _handle_browser_tool("browse_page", {"url": "https://example.com"}, None, None)

        data = json.loads(result)
        assert "disabled" in data["error"].lower() or "BROWSER_ENABLED" in data["error"]

    def test_browse_page_success(self):
        """Mock playwright + vision, verify JSON response."""
        from remy.core.brain_tools import _handle_browser_tool

        mock_mgr = MagicMock()
        mock_mgr.navigate = AsyncMock(return_value=b"\x89PNGfake")
        mock_mgr.get_page_url = AsyncMock(return_value="https://example.com")
        mock_mgr.get_page_text = AsyncMock(return_value="Hello World")
        mock_mgr.save_screenshot.return_value = "ss_abc123.png"

        vision_result = {
            "description": "Example page",
            "elements": [],
            "forms": [],
            "answer": None,
            "suggested_actions": [],
        }

        with patch("remy.core.brain_tools.settings") as mock_settings, \
             patch("remy.core.browser.BrowserManager.get", return_value=mock_mgr), \
             patch("remy.core.browser_vision.analyze_screenshot",
                   new_callable=lambda: AsyncMock(return_value=vision_result)):
            mock_settings.BROWSER_ENABLED = True
            result = _handle_browser_tool("browse_page", {"url": "https://example.com"}, "s1", "desktop")

        data = json.loads(result)
        assert data["url"] == "https://example.com"
        assert data["description"] == "Example page"
        assert "ss_abc123.png" in data["screenshot"]

    def test_browser_act_click(self):
        """Mock page.click(), verify screenshot taken."""
        from remy.core.brain_tools import _handle_browser_tool

        mock_mgr = MagicMock()
        mock_mgr.act = AsyncMock(return_value=b"\x89PNGfake")
        mock_mgr.get_page_url = AsyncMock(return_value="https://example.com/next")
        mock_mgr.get_page_text = AsyncMock(return_value="Next page")
        mock_mgr.save_screenshot.return_value = "ss_def456.png"

        vision_result = {
            "description": "Next page after click",
            "elements": [],
            "forms": [],
            "answer": None,
            "suggested_actions": [],
        }

        with patch("remy.core.brain_tools.settings") as mock_settings, \
             patch("remy.core.browser.BrowserManager.get", return_value=mock_mgr), \
             patch("remy.core.browser_vision.analyze_screenshot",
                   new_callable=lambda: AsyncMock(return_value=vision_result)):
            mock_settings.BROWSER_ENABLED = True
            result = _handle_browser_tool(
                "browser_act",
                {"action": "click", "selector": "#submit-btn"},
                "s1", "desktop",
            )

        data = json.loads(result)
        assert data["action"] == "click"
        mock_mgr.act.assert_called_once()

    def test_browser_act_type(self):
        """Mock page.fill(), verify text entered."""
        from remy.core.brain_tools import _handle_browser_tool

        mock_mgr = MagicMock()
        mock_mgr.act = AsyncMock(return_value=b"\x89PNGfake")
        mock_mgr.get_page_url = AsyncMock(return_value="https://example.com")
        mock_mgr.get_page_text = AsyncMock(return_value="Form page")
        mock_mgr.save_screenshot.return_value = "ss_ghi789.png"

        vision_result = {
            "description": "Form with text entered",
            "elements": [],
            "forms": [],
            "answer": None,
            "suggested_actions": [],
        }

        with patch("remy.core.brain_tools.settings") as mock_settings, \
             patch("remy.core.browser.BrowserManager.get", return_value=mock_mgr), \
             patch("remy.core.browser_vision.analyze_screenshot",
                   new_callable=lambda: AsyncMock(return_value=vision_result)):
            mock_settings.BROWSER_ENABLED = True
            result = _handle_browser_tool(
                "browser_act",
                {"action": "type", "selector": "#email", "text": "user@test.com"},
                "s1", "desktop",
            )

        data = json.loads(result)
        assert data["action"] == "type"
        call_kwargs = mock_mgr.act.call_args
        assert call_kwargs.kwargs["text"] == "user@test.com"

    def test_browser_close(self):
        """Verify cleanup is called."""
        from remy.core.brain_tools import _handle_browser_tool

        mock_mgr = MagicMock()
        mock_mgr.close = AsyncMock()

        with patch("remy.core.brain_tools.settings") as mock_settings, \
             patch("remy.core.browser.BrowserManager.get", return_value=mock_mgr):
            mock_settings.BROWSER_ENABLED = True
            result = _handle_browser_tool("browser_close", {}, "s1", "desktop")

        data = json.loads(result)
        assert data["closed"] is True
        mock_mgr.close.assert_called_once()

    def test_browse_page_ssrf_blocked(self):
        """SSRF error returns clean JSON error."""
        from remy.core.browser import SSRFError
        from remy.core.brain_tools import _handle_browser_tool

        mock_mgr = MagicMock()
        mock_mgr.navigate = AsyncMock(side_effect=SSRFError("Blocked localhost"))

        with patch("remy.core.brain_tools.settings") as mock_settings, \
             patch("remy.core.browser.BrowserManager.get", return_value=mock_mgr):
            mock_settings.BROWSER_ENABLED = True
            result = _handle_browser_tool(
                "browse_page", {"url": "http://127.0.0.1"}, "s1", "desktop"
            )

        data = json.loads(result)
        assert "Blocked" in data["error"]


# ============== SCREENSHOT API ENDPOINT ==============


class TestScreenshotEndpoint:

    def _make_client(self):
        from fastapi.testclient import TestClient
        from remy.web.api import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_serve_screenshot(self):
        """Screenshot file served as PNG."""
        from remy.config.settings import settings
        from pathlib import Path

        ss_dir = Path(settings.DATA_DIR) / "browser_screenshots"
        ss_dir.mkdir(parents=True, exist_ok=True)
        test_file = ss_dir / "test_browser_ss.png"
        test_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
        try:
            client = self._make_client()
            resp = client.get("/api/browser_screenshots/test_browser_ss.png")
            assert resp.status_code == 200
            assert "image/png" in resp.headers["content-type"]
        finally:
            test_file.unlink(missing_ok=True)

    def test_serve_screenshot_not_found(self):
        """Missing screenshot returns 404."""
        client = self._make_client()
        resp = client.get("/api/browser_screenshots/nonexistent_xyz.png")
        assert resp.status_code == 404

    def test_serve_screenshot_path_traversal(self):
        """Path traversal attempt blocked."""
        client = self._make_client()
        resp = client.get("/api/browser_screenshots/../../../etc/passwd")
        assert resp.status_code in (404, 400, 422)


# ============== OUTSIDE BRAIN_LOCK ==============


class TestBrowserOutsideLock:

    def test_browser_tools_bypass_brain_lock(self):
        """Verify browser tools are dispatched BEFORE brain_lock acquisition."""
        from remy.core.brain_tools import execute_tool

        # Mock the browser handler to track it was called
        with patch("remy.core.brain_tools._handle_browser_tool") as mock_handler, \
             patch("remy.core.brain_tools.settings") as mock_settings:
            mock_handler.return_value = '{"ok": true}'
            mock_settings.BROWSER_ENABLED = True

            result = execute_tool("browse_page", {"url": "https://example.com"})
            mock_handler.assert_called_once_with(
                "browse_page", {"url": "https://example.com"}, None, None
            )
            assert json.loads(result)["ok"] is True


# ============== SMART CIRCUIT BREAKER TESTS ==============


class TestSmartCircuitBreaker:
    """Smart browser circuit breaker: analyze errors → suggest pivot → hard stop."""

    @staticmethod
    def _fake_run_async(result: str):
        def _run(coro):
            close = getattr(coro, "close", None)
            if callable(close):
                close()
            return result

        return _run

    @pytest.fixture(autouse=True)
    def _reset_circuit_breaker(self):
        """Reset circuit breaker state before each test."""
        import remy.core.brain_tools as bt
        bt._consecutive_browser_failures = 0
        bt._browser_error_history.clear()
        yield
        bt._consecutive_browser_failures = 0
        bt._browser_error_history.clear()

    @pytest.fixture(autouse=True)
    def _patch_browser_settings(self, tmp_path):
        with patch("remy.core.browser.settings") as mock_browser_settings:
            mock_browser_settings.DATA_DIR = tmp_path
            mock_browser_settings.BROWSER_HEADLESS = True
            mock_browser_settings.BROWSER_IDLE_TIMEOUT_SEC = 300
            mock_browser_settings.BROWSER_DAILY_ACTION_LIMIT = 200
            mock_browser_settings.BROWSER_PAGE_TIMEOUT_MS = 30000
            yield mock_browser_settings

    def _make_failure_result(self, error_msg="Action failed: Timeout 10000ms exceeded"):
        return json.dumps({"error": error_msg})

    def _make_success_result(self):
        return json.dumps({"action": "click", "url": "https://example.com", "description": "OK"})

    def test_no_trigger_under_threshold(self):
        """2 failures should NOT trigger circuit breaker."""
        import remy.core.brain_tools as bt
        from remy.core.brain_tools import _handle_browser_tool

        bt._consecutive_browser_failures = 2
        bt._browser_error_history = ["err1", "err2"]

        with patch("remy.core.brain_tools.settings") as ms, \
             patch("remy.core.brain_tools._run_async", side_effect=self._fake_run_async(self._make_failure_result())):
            ms.BROWSER_ENABLED = True
            result = _handle_browser_tool("browser_act", {"action": "click", "selector": "#btn"}, None, None)

        data = json.loads(result)
        # Should get the actual error, not circuit breaker message
        assert "CIRCUIT BREAKER" not in data.get("error", "")
        assert "BROWSER ERROR PATTERN" not in data.get("error", "")
        assert bt._consecutive_browser_failures == 3

    def test_analyze_threshold_returns_diagnosis(self):
        """At 3 failures, browser_act returns error analysis instead of executing."""
        import remy.core.brain_tools as bt
        from remy.core.brain_tools import _handle_browser_tool

        bt._consecutive_browser_failures = 3
        bt._browser_error_history = [
            "Locator.click: Timeout 10000ms exceeded selector='#btn1'",
            "Locator.click: Timeout 10000ms exceeded selector='#btn2'",
            "Locator.click: Timeout 10000ms exceeded selector='#btn3'",
        ]

        with patch("remy.core.brain_tools.settings") as ms:
            ms.BROWSER_ENABLED = True
            # Should NOT even call _run_async — returns analysis immediately
            result = _handle_browser_tool("browser_act", {"action": "click", "selector": "#btn4"}, None, None)

        data = json.loads(result)
        assert "BROWSER ERROR PATTERN" in data["error"]
        assert "failed-hypothesis" in data["error"]
        assert "DISPROVED" in data["error"]
        # Should mention timeout diagnosis
        assert "NOT interactive" in data["error"] or "timeout" in data["error"].lower()
        # Should list failed selectors
        assert "#btn1" in data["error"] or "#btn2" in data["error"]

    def test_analyze_allows_browse_page(self):
        """browse_page should still work at 3 failures (only browser_act blocked)."""
        import remy.core.brain_tools as bt
        from remy.core.brain_tools import _handle_browser_tool

        bt._consecutive_browser_failures = 3
        bt._browser_error_history = ["err1", "err2", "err3"]

        with patch("remy.core.brain_tools.settings") as ms, \
             patch("remy.core.brain_tools._run_async", side_effect=self._fake_run_async('{"url":"https://example.com","description":"OK"}')):
            ms.BROWSER_ENABLED = True
            result = _handle_browser_tool("browse_page", {"url": "https://example.com"}, None, None)

        data = json.loads(result)
        assert "error" not in data
        assert data["url"] == "https://example.com"

    def test_hard_stop_at_six_failures(self):
        """At 6 failures, hard stop for both browse_page and browser_act."""
        import remy.core.brain_tools as bt
        from remy.core.brain_tools import _handle_browser_tool

        bt._consecutive_browser_failures = 6
        bt._browser_error_history = ["err"] * 6

        with patch("remy.core.brain_tools.settings") as ms:
            ms.BROWSER_ENABLED = True
            result = _handle_browser_tool("browser_act", {"action": "click", "selector": "#x"}, None, None)

        data = json.loads(result)
        assert "HARD STOP" in data["error"]
        # Should reset after hard stop
        assert bt._consecutive_browser_failures == 0
        assert len(bt._browser_error_history) == 0

    def test_hard_stop_blocks_browse_page_too(self):
        """At 6 failures, even browse_page is blocked."""
        import remy.core.brain_tools as bt
        from remy.core.brain_tools import _handle_browser_tool

        bt._consecutive_browser_failures = 6
        bt._browser_error_history = ["err"] * 6

        with patch("remy.core.brain_tools.settings") as ms:
            ms.BROWSER_ENABLED = True
            result = _handle_browser_tool("browse_page", {"url": "https://example.com"}, None, None)

        data = json.loads(result)
        assert "HARD STOP" in data["error"]

    def test_browser_close_always_allowed(self):
        """browser_close works even at max failures and resets state."""
        import remy.core.brain_tools as bt
        from remy.core.brain_tools import _handle_browser_tool

        bt._consecutive_browser_failures = 10
        bt._browser_error_history = ["err"] * 10

        mock_mgr = MagicMock()
        mock_mgr.close = AsyncMock()

        with patch("remy.core.brain_tools.settings") as ms, \
             patch("remy.core.browser.BrowserManager.get", return_value=mock_mgr):
            ms.BROWSER_ENABLED = True
            result = _handle_browser_tool("browser_close", {}, None, None)

        data = json.loads(result)
        assert data["closed"] is True
        assert bt._consecutive_browser_failures == 0
        assert len(bt._browser_error_history) == 0

    def test_success_resets_counter(self):
        """A successful browser_act resets failures and error history."""
        import remy.core.brain_tools as bt
        from remy.core.brain_tools import _handle_browser_tool

        bt._consecutive_browser_failures = 2
        bt._browser_error_history = ["err1", "err2"]

        with patch("remy.core.brain_tools.settings") as ms, \
             patch("remy.core.brain_tools._run_async", side_effect=self._fake_run_async(self._make_success_result())):
            ms.BROWSER_ENABLED = True
            result = _handle_browser_tool("browser_act", {"action": "click", "selector": "#btn"}, None, None)

        assert bt._consecutive_browser_failures == 0
        assert len(bt._browser_error_history) == 0

    def test_error_history_tracks_messages(self):
        """Error messages are accumulated in _browser_error_history."""
        import remy.core.brain_tools as bt
        from remy.core.brain_tools import _handle_browser_tool

        errors = [
            "Locator.click: Timeout selector='#a'",
            "Locator.click: Timeout selector='#b'",
        ]

        for err in errors:
            with patch("remy.core.brain_tools.settings") as ms, \
                 patch("remy.core.brain_tools._run_async", side_effect=self._fake_run_async(self._make_failure_result(err))):
                ms.BROWSER_ENABLED = True
                _handle_browser_tool("browser_act", {"action": "click", "selector": "#x"}, None, None)

        assert len(bt._browser_error_history) == 2
        assert "selector='#a'" in bt._browser_error_history[0]
        assert "selector='#b'" in bt._browser_error_history[1]


class TestAnalyzeBrowserErrors:
    """Unit tests for _analyze_browser_errors() pattern detection."""

    def test_timeout_diagnosis(self):
        from remy.core.brain_tools import _analyze_browser_errors
        errors = [
            "Locator.click: Timeout 10000ms exceeded selector='#btn1'",
            "Locator.click: Timeout 10000ms exceeded selector='#btn2'",
            "Locator.click: Timeout 15000ms exceeded selector='.nav-link'",
        ]
        result = _analyze_browser_errors(errors)
        assert "NOT interactive" in result
        assert "failed-hypothesis" in result
        assert "#btn1" in result
        assert "#btn2" in result

    def test_not_found_diagnosis(self):
        from remy.core.brain_tools import _analyze_browser_errors
        errors = [
            "No element found for selector '#missing'",
            "Element not found: .does-not-exist",
            "No element matching locator('.gone')",
        ]
        result = _analyze_browser_errors(errors)
        assert "don't match" in result or "not found" in result.lower()
        assert "failed-hypothesis" in result

    def test_detached_diagnosis(self):
        from remy.core.brain_tools import _analyze_browser_errors
        errors = [
            "Element is detached from DOM",
            "Navigation interrupted the action",
            "Page navigated during wait",
        ]
        result = _analyze_browser_errors(errors)
        assert "detach" in result.lower() or "SPA" in result
        assert "failed-hypothesis" in result

    def test_generic_diagnosis(self):
        from remy.core.brain_tools import _analyze_browser_errors
        errors = ["Some unknown error happened", "Another weird failure"]
        result = _analyze_browser_errors(errors)
        assert "BROWSER ERROR PATTERN" in result
        assert "failed-hypothesis" in result
        # Should include error text in diagnosis
        assert "unknown error" in result.lower() or "weird failure" in result.lower()

    def test_selector_extraction(self):
        from remy.core.brain_tools import _analyze_browser_errors
        errors = [
            "Locator.click: Timeout selector='button.submit'",
            "Locator.click: Timeout selector='a.nav-link'",
            "Locator.click: Timeout selector='button.submit'",  # duplicate
        ]
        result = _analyze_browser_errors(errors)
        assert "button.submit" in result
        assert "a.nav-link" in result


class TestHybridBrowserBackend:

    @pytest.mark.asyncio
    async def test_browse_page_uses_pinchtab_backend(self):
        from remy.core.tool_handlers.browser_dispatch import _handle_browse_page

        mock_pinch = MagicMock()
        mock_pinch.browse_page = AsyncMock(return_value={
            "url": "https://example.com",
            "requested_url": "https://example.com",
            "page_text": "Example text body",
            "dom_elements": [{"selector": "ref:e1", "tag": "link", "text": "Docs"}],
            "dom_form_fields": [],
        })

        with patch("remy.core.tool_handlers.browser_dispatch._should_try_pinchtab_browse", return_value=True), \
             patch("remy.core.pinchtab.PinchTabManager.get", return_value=mock_pinch):
            result = json.loads(await _handle_browse_page({"url": "example.com"}, None, None))

        assert result["backend"] == "pinchtab"
        assert result["url"] == "https://example.com"
        assert result["dom_elements"][0]["selector"] == "ref:e1"

    def test_pinchtab_action_requires_ref_for_clicks(self):
        from remy.core.tool_handlers.browser_dispatch import _should_try_pinchtab_act

        with patch("remy.core.tool_handlers.browser_dispatch._pinchtab_enabled", return_value=True):
            assert _should_try_pinchtab_act("click", "ref:e7") is True
            assert _should_try_pinchtab_act("click", "#submit") is False
            assert _should_try_pinchtab_act("wait", None) is True

    @pytest.mark.asyncio
    async def test_browser_close_closes_pinchtab_when_enabled(self):
        from remy.core.tool_handlers.browser_dispatch import _handle_browser_close

        mock_browser = MagicMock()
        mock_browser.close = AsyncMock()
        mock_pinch = MagicMock()
        mock_pinch.close = AsyncMock()

        with patch("remy.core.browser.BrowserManager.get", return_value=mock_browser), \
             patch("remy.core.tool_handlers.browser_dispatch._pinchtab_enabled", return_value=True), \
             patch("remy.core.pinchtab.PinchTabManager.get", return_value=mock_pinch):
            result = json.loads(await _handle_browser_close({}, None, None))

        assert result["closed"] is True
        mock_pinch.close.assert_awaited_once()
