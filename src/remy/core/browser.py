"""
Browser Manager — Playwright-based browser automation for the agent.

Provides a persistent Chromium browser context with cookie/localStorage persistence,
SSRF protection, daily action limits, and idle timeout.
"""

import ipaddress
import logging
import threading
import time
import uuid
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

from remy.config.settings import settings

logger = logging.getLogger("Browser")


class SSRFError(Exception):
    """Raised when a URL targets a private/local address."""


class DailyLimitError(Exception):
    """Raised when daily browser action limit is exceeded."""


class BrowserDisabledError(Exception):
    """Raised when browser tools are disabled."""


class BrowserManager:
    """Singleton managing persistent Playwright browser context.

    Thread-safe. Uses persistent context so cookies/localStorage survive
    browser restarts. Single page reused across calls.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._playwright = None
        self._context = None
        self._page = None
        self._loop_id: int | None = None  # id() of the event loop that owns the browser
        self._last_activity: float = 0
        self._action_count_today: int = 0
        self._action_count_date: date | None = None

    @classmethod
    def get(cls) -> "BrowserManager":
        """Thread-safe singleton accessor."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton (for tests)."""
        with cls._lock:
            cls._instance = None

    # ============== SSRF PROTECTION ==============

    @staticmethod
    def check_ssrf(url: str) -> None:
        """Block requests to private/local addresses and non-HTTP protocols."""
        parsed = urlparse(url)

        # Only allow http/https
        if parsed.scheme not in ("http", "https"):
            raise SSRFError(f"Blocked protocol: {parsed.scheme}://")

        hostname = parsed.hostname or ""

        # Block localhost variants
        if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            raise SSRFError(f"Blocked localhost address: {hostname}")

        # Block private IP ranges
        try:
            addr = ipaddress.ip_address(hostname)
            if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
                raise SSRFError(f"Blocked private/reserved IP: {hostname}")
        except ValueError:
            # Not an IP literal — hostname is fine (DNS resolution happens in browser)
            pass

    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalize user-provided URLs for browser navigation."""
        text = (url or "").strip()
        if not text:
            return ""
        parsed = urlparse(text)
        if not parsed.scheme:
            return f"https://{text}"
        return text

    # ============== DAILY LIMIT ==============

    def _check_daily_limit(self) -> None:
        """Enforce daily action limit, reset counter on new day."""
        today = date.today()
        if self._action_count_date != today:
            self._action_count_today = 0
            self._action_count_date = today

        if self._action_count_today >= settings.BROWSER_DAILY_ACTION_LIMIT:
            raise DailyLimitError(
                f"Daily browser action limit reached ({settings.BROWSER_DAILY_ACTION_LIMIT}). "
                "Try again tomorrow."
            )

    def _record_action(self) -> None:
        """Record a browser action for daily limit tracking."""
        today = date.today()
        if self._action_count_date != today:
            self._action_count_today = 0
            self._action_count_date = today
        self._action_count_today += 1
        self._last_activity = time.time()

    # ============== BROWSER LIFECYCLE ==============

    async def ensure_browser(self):
        """Launch browser if not running, return the page.

        Uses persistent context for cookie/localStorage persistence.
        Detects event-loop changes (e.g. _run_async creates a new loop each call)
        and re-launches the browser when the loop has changed.
        """
        import asyncio
        import os
        from playwright.async_api import async_playwright

        current_loop_id = id(asyncio.get_running_loop())

        # If page exists and belongs to the SAME event loop, reuse it
        if (self._page and not self._page.is_closed()
                and self._loop_id == current_loop_id):
            return self._page

        # Event loop changed or page is stale — cleanup old resources.
        # _cleanup may fail on cross-loop objects; that's OK, we just null them out.
        await self._cleanup()

        # Windows fix: Playwright expects HOME env var (Linux/macOS convention).
        # On Windows only USERPROFILE exists — set HOME so Playwright can find its binaries.
        if not os.environ.get("HOME") and os.environ.get("USERPROFILE"):
            os.environ["HOME"] = os.environ["USERPROFILE"]

        profile_dir = str(Path(settings.DATA_DIR) / "browser_profile")
        Path(profile_dir).mkdir(parents=True, exist_ok=True)

        self._playwright = await async_playwright().start()
        self._context = await self._playwright.chromium.launch_persistent_context(
            profile_dir,
            headless=settings.BROWSER_HEADLESS,
            viewport={"width": 1280, "height": 900},
            locale="en-US",
            timezone_id="Europe/Kyiv",
            args=["--disable-blink-features=AutomationControlled"],
        )

        # Use first page or create new one
        if self._context.pages:
            self._page = self._context.pages[0]
        else:
            self._page = await self._context.new_page()

        self._loop_id = current_loop_id
        self._last_activity = time.time()
        logger.info("Browser launched (headless=%s)", settings.BROWSER_HEADLESS)
        return self._page

    async def navigate(self, url: str) -> bytes:
        """Navigate to URL and return screenshot PNG bytes.

        Checks SSRF and daily limits before navigating.
        """
        self.check_ssrf(url)
        self._check_daily_limit()

        page = await self.ensure_browser()
        await page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=settings.BROWSER_PAGE_TIMEOUT_MS,
        )
        # Extra wait for SPA frameworks (React/Next.js/Nuxt) to render
        try:
            await page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass  # SPA sites may never reach networkidle — that's fine
        # Brief pause for JS hydration (React attaches event handlers after render)
        import asyncio as _aio
        await _aio.sleep(1)
        self._record_action()

        return await page.screenshot(type="png")

    @staticmethod
    def _normalize_selector(selector: str) -> str:
        """Convert invalid/jQuery selectors to Playwright-compatible format.

        Handles:
        - :contains("text") → :has-text("text")  (jQuery → Playwright)
        - .Class__* wildcard → [class*="Class"] (CSS Modules partial match)
        - text=... and role=... pass through (native Playwright)
        """
        import re
        # jQuery :contains("text") → Playwright :has-text("text")
        selector = re.sub(
            r':contains\((["\'])(.+?)\1\)',
            r':has-text(\1\2\1)',
            selector,
        )
        # :contains(text) without quotes
        selector = re.sub(
            r':contains\(([^)]+)\)',
            r':has-text("\1")',
            selector,
        )
        # CSS Modules wildcard: .ClassName__* or .ClassName__abc123
        # → [class*="ClassName"] (partial attribute match)
        selector = re.sub(
            r'\.([A-Za-z][A-Za-z0-9_]*)__\*',
            r'[class*="\1"]',
            selector,
        )
        return selector

    async def act(self, action: str, selector: str | None = None,
                  text: str | None = None, url: str | None = None) -> bytes:
        """Perform an action on the current page and return screenshot.

        Supported actions: click, type, scroll_down, scroll_up, select,
        wait, goto, back, forward.
        """
        self._check_daily_limit()
        page = await self.ensure_browser()

        # Normalize selector (convert jQuery :contains etc.)
        if selector:
            selector = self._normalize_selector(selector)

        if action == "click":
            if not selector:
                raise ValueError("click requires a selector")
            loc = page.locator(selector).first
            try:
                await loc.wait_for(state="visible", timeout=15000)
                await loc.click(timeout=10000)
            except Exception as primary_err:
                # Fallback: only for plain text selectors (e.g. 'text="Sign In"')
                # Do NOT fallback for CSS selectors — they'd produce garbage role names
                import re as _re
                m = _re.search(r'^text="(.+)"$', selector)
                if not m:
                    m = _re.search(r':has-text\("(.+?)"\)$', selector)
                if m:
                    text_content = m.group(1)
                    fallback = page.get_by_role("link", name=text_content).or_(
                        page.get_by_role("button", name=text_content)
                    )
                    await fallback.first.click(timeout=10000)
                else:
                    raise primary_err
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=5000)
            except Exception:
                pass  # Some clicks don't trigger navigation

        elif action == "type":
            if not selector or text is None:
                raise ValueError("type requires selector and text")
            locator = page.locator(selector)
            count = await locator.count()
            if count > 1:
                logger.warning(
                    "Selector '%s' matches %d elements — using first visible/enabled",
                    selector, count,
                )
                # Try to find visible & enabled element among matches
                for i in range(min(count, 5)):
                    el = locator.nth(i)
                    if await el.is_visible() and await el.is_enabled():
                        await el.fill(text)
                        break
                else:
                    await locator.first.fill(text)
            else:
                await locator.first.fill(text)

        elif action == "fill_form":
            # Fill multiple form fields at once.
            # text must be a JSON string: [{"selector": "...", "value": "..."}]
            if not text:
                raise ValueError("fill_form requires text as JSON: [{\"selector\":\"...\",\"value\":\"...\"}]")
            import json as _json
            try:
                fields = _json.loads(text)
            except _json.JSONDecodeError:
                raise ValueError("fill_form text must be valid JSON array")
            filled = []
            errors = []
            for fld in fields:
                sel = fld.get("selector", "")
                val = fld.get("value", "")
                fld_type = fld.get("type", "text")
                if not sel:
                    errors.append("missing selector in field entry")
                    continue
                sel = self._normalize_selector(sel)
                try:
                    loc = page.locator(sel)
                    if await loc.count() == 0:
                        errors.append(f"selector not found: {sel}")
                        continue
                    if fld_type == "select":
                        await loc.first.select_option(val)
                    elif fld_type in ("checkbox", "radio"):
                        if val.lower() in ("true", "1", "yes", "on"):
                            await loc.first.check()
                        else:
                            await loc.first.uncheck()
                    else:
                        await loc.first.fill(val)
                    filled.append(sel)
                except Exception as e:
                    errors.append(f"{sel}: {e}")
            logger.info("fill_form: filled %d/%d fields, errors: %s",
                        len(filled), len(fields), errors or "none")

        elif action == "scroll_down":
            await page.evaluate("window.scrollBy(0, 600)")

        elif action == "scroll_up":
            await page.evaluate("window.scrollBy(0, -600)")

        elif action == "select":
            if not selector or not text:
                raise ValueError("select requires selector and text (option value)")
            await page.locator(selector).first.select_option(text)

        elif action == "wait":
            timeout = int(text) if text and text.isdigit() else 2000
            timeout = min(timeout, 10000)  # Cap at 10s
            await page.wait_for_timeout(timeout)

        elif action == "goto":
            if not url:
                raise ValueError("goto requires url")
            self.check_ssrf(url)
            await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=settings.BROWSER_PAGE_TIMEOUT_MS,
            )
            try:
                await page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass

        elif action == "back":
            await page.go_back(wait_until="domcontentloaded", timeout=10000)

        elif action == "forward":
            await page.go_forward(wait_until="domcontentloaded", timeout=10000)

        else:
            raise ValueError(f"Unknown action: {action}")

        self._record_action()
        return await page.screenshot(type="png")

    async def screenshot(self) -> bytes:
        """Take screenshot of current page."""
        page = await self.ensure_browser()
        return await page.screenshot(type="png")

    async def get_page_text(self) -> str:
        """Extract visible text from current page (truncated for token economy)."""
        page = await self.ensure_browser()
        try:
            text = await page.inner_text("body")
            return text[:5000]
        except Exception:
            return ""

    async def get_page_url(self) -> str:
        """Get current page URL."""
        if self._page and not self._page.is_closed():
            return self._page.url
        return ""

    async def close(self) -> None:
        """Close browser and free all resources."""
        await self._cleanup()
        logger.info("Browser closed")

    async def _cleanup(self) -> None:
        """Internal cleanup of browser resources.

        Safe to call even if objects belong to a dead event loop —
        all exceptions are silently swallowed.
        """
        try:
            if self._page and not self._page.is_closed():
                await self._page.close()
        except Exception:
            pass
        self._page = None

        try:
            if self._context:
                await self._context.close()
        except Exception:
            pass
        self._context = None

        try:
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass
        self._playwright = None
        self._loop_id = None

    def is_idle(self) -> bool:
        """Check if browser has been idle too long."""
        if self._last_activity == 0:
            return False
        return (time.time() - self._last_activity) > settings.BROWSER_IDLE_TIMEOUT_SEC

    def save_screenshot(self, png_bytes: bytes) -> str:
        """Save screenshot to disk and return filename."""
        ss_dir = Path(settings.DATA_DIR) / "browser_screenshots"
        ss_dir.mkdir(parents=True, exist_ok=True)
        filename = f"ss_{uuid.uuid4().hex[:8]}.png"
        (ss_dir / filename).write_bytes(png_bytes)
        return filename
