"""Browser contract for the persistent, resizable application sidebar."""

import pytest
from playwright.sync_api import sync_playwright


@pytest.mark.e2e
def test_sidebar_resizes_with_pointer_and_keyboard_and_persists(server_url):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        startup_requests = []
        page.on("request", lambda request: startup_requests.append(request.url))
        page.add_init_script("""
            localStorage.setItem('remy_first_run_done_v1', '1');
            if (!sessionStorage.getItem('sidebar-resize-test-initialized')) {
                localStorage.removeItem('remy_sidebar_width_v1');
                sessionStorage.setItem('sidebar-resize-test-initialized', '1');
            }
        """)

        page.goto(server_url, wait_until="domcontentloaded")
        page.wait_for_function(
            "document.querySelector('#startup-splash')?.classList.contains('is-hidden')",
            timeout=10_000,
        )
        page.wait_for_selector(".conversation-load-status", state="detached", timeout=10_000)

        assert any("/api/system/operator-alerts?" in url for url in startup_requests)
        assert not any(url.endswith("/api/system/status") for url in startup_requests)

        sidebar = page.locator(".sidebar")
        handle = page.locator("#sidebar-resize-handle")

        def wait_for_sidebar_width(expected):
            page.wait_for_function(
                "expected => Math.abs(document.querySelector('.sidebar').getBoundingClientRect().width - expected) <= 0.05",
                arg=expected,
            )

        assert handle.is_visible()
        assert handle.get_attribute("role") == "separator"
        assert round(sidebar.bounding_box()["width"]) == 224

        handle_box = handle.bounding_box()
        page.mouse.move(handle_box["x"] + handle_box["width"] / 2, 300)
        page.mouse.down()
        page.mouse.move(336, 300, steps=6)
        page.mouse.up()

        resized_width = round(sidebar.bounding_box()["width"])
        assert abs(resized_width - 336) <= 2
        assert handle.get_attribute("aria-valuenow") == str(resized_width)
        assert page.evaluate("localStorage.getItem('remy_sidebar_width_v1')") == str(resized_width)

        page.reload(wait_until="domcontentloaded")
        page.wait_for_function(
            "document.querySelector('#startup-splash')?.classList.contains('is-hidden')",
            timeout=10_000,
        )
        sidebar = page.locator(".sidebar")
        handle = page.locator("#sidebar-resize-handle")
        assert round(sidebar.bounding_box()["width"]) == resized_width

        handle.focus()
        page.keyboard.press("Home")
        wait_for_sidebar_width(184)
        assert round(sidebar.bounding_box()["width"]) == 184
        page.keyboard.press("End")
        wait_for_sidebar_width(420)
        assert round(sidebar.bounding_box()["width"]) == 420
        page.keyboard.press("Shift+ArrowLeft")
        wait_for_sidebar_width(396)
        assert round(sidebar.bounding_box()["width"]) == 396

        handle.dblclick()
        wait_for_sidebar_width(224)
        assert round(sidebar.bounding_box()["width"]) == 224
        assert page.evaluate("localStorage.getItem('remy_sidebar_width_v1')") == "224"

        page.set_viewport_size({"width": 760, "height": 900})
        assert not handle.is_visible()
        page.click("#btn-menu")
        assert sidebar.evaluate(
            "element => element.getBoundingClientRect().width <= window.innerWidth - 44"
        )
        browser.close()
