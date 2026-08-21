"""Browser contract for category-based Settings navigation."""

import pytest
from playwright.sync_api import sync_playwright


@pytest.mark.e2e
def test_settings_hub_switches_panels_and_persists_category(server_url):
    settings_payload = {
        "summary_model": "test-model",
        "gemini_model": "gemini-3.1-flash-live-preview",
        "gemini_voice": "Zephyr",
        "custom_system_prompt": "Prefer concise answers.",
        "has_telegram": False,
        "telegram_bot_masked": "",
        "proactive_chat_id": "",
        "has_smtp": False,
        "smtp_user": "",
    }
    diagnostics = {
        "status": "ok",
        "uptime": "5m",
        "model": "test-model",
        "brain": {"records": 12},
    }

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.add_init_script("""
            localStorage.setItem('remy_first_run_done_v1', '1');
            if (!sessionStorage.getItem('settings-hub-test-initialized')) {
                localStorage.removeItem('remy.settings.category');
                sessionStorage.setItem('settings-hub-test-initialized', '1');
            }
        """)
        page.route("**/api/settings", lambda route: route.fulfill(json=settings_payload))
        page.route("**/api/diagnostics", lambda route: route.fulfill(json=diagnostics))
        page.route("**/api/secrets", lambda route: route.fulfill(json={"secrets": []}))
        page.route("**/api/workspaces", lambda route: route.fulfill(json={"workspaces": []}))
        page.route("**/api/model-registry", lambda route: route.fulfill(json={"models": []}))
        page.route("**/api/models", lambda route: route.fulfill(json={"models": ["test-model"]}))
        page.route("**/api/push/status", lambda route: route.fulfill(json={"supported": False}))
        page.route("**/api/aura/status", lambda route: route.fulfill(json={"installed": True, "version": "test"}))
        page.route("**/api/llamacpp/status", lambda route: route.fulfill(json={"installed": False, "models": []}))

        page.goto(server_url, wait_until="domcontentloaded")
        page.wait_for_selector(".sidebar", timeout=10_000)
        page.wait_for_selector("#startup-splash.is-hidden", timeout=10_000)
        page.click('[data-view="settings"]')
        page.wait_for_selector(".settings-hub", timeout=10_000)

        assert page.locator("[data-settings-category]").count() == 5
        assert page.locator('[data-settings-panel]:visible').count() == 1
        assert page.locator('[data-settings-category="overview"]').get_attribute("aria-selected") == "true"
        assert page.locator('[data-settings-panel="overview"] .diag-grid').is_visible()
        assert not page.locator('[data-settings-panel="models"]').is_visible()

        page.click('[data-settings-category="models"]')
        assert page.locator('[data-settings-category="models"]').get_attribute("aria-selected") == "true"
        assert page.locator('[data-settings-panel="models"] #set-model').is_visible()
        assert page.locator('[data-settings-panel="models"] #local-models-section').is_visible()
        assert not page.locator('[data-settings-panel="overview"]').is_visible()

        page.locator('[data-settings-category="models"]').focus()
        page.keyboard.press("ArrowRight")
        assert page.locator('[data-settings-category="personalization"]').get_attribute("aria-selected") == "true"
        assert page.locator("#set-theme").is_visible()

        page.click('[data-settings-category="workspace"]')
        assert page.locator("#btn-workspace-choose").is_visible()
        assert page.locator('[data-settings-panel]:visible').count() == 1

        page.set_viewport_size({"width": 760, "height": 900})
        page.wait_for_timeout(50)
        assert page.locator(".settings-hub").evaluate(
            "element => getComputedStyle(element).gridTemplateColumns.split(' ').length === 1"
        )
        assert page.locator(".settings-hub-tabs").evaluate(
            "element => getComputedStyle(element).flexDirection === 'row'"
        )
        assert page.locator(".settings-hub").evaluate(
            "element => element.scrollWidth <= element.clientWidth"
        )

        page.set_viewport_size({"width": 1440, "height": 900})
        page.reload(wait_until="domcontentloaded")
        page.wait_for_selector(".sidebar", timeout=10_000)
        page.wait_for_selector("#startup-splash.is-hidden", timeout=10_000)
        page.click('[data-view="settings"]')
        page.wait_for_selector(".settings-hub", timeout=10_000)
        assert page.locator('[data-settings-category="workspace"]').get_attribute("aria-selected") == "true"
        assert page.locator("#btn-workspace-choose").is_visible()
        browser.close()
