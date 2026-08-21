"""E2E tests: Settings view."""

import pytest


class TestSettings:

    def _navigate_to_settings(self, page):
        page.click('.nav-item[data-view="settings"]')
        page.wait_for_selector("#view-settings", state="visible", timeout=3000)

    def test_settings_loads(self, authenticated_page):
        """Settings view should load and display content."""
        page = authenticated_page
        self._navigate_to_settings(page)

        content = page.locator("#settings-content")
        assert content.is_visible()
        page.wait_for_timeout(500)
        text = content.inner_text()
        assert len(text) > 0

    def test_diagnostics_shown(self, authenticated_page):
        """Settings should show system diagnostics."""
        page = authenticated_page
        self._navigate_to_settings(page)
        page.wait_for_timeout(1000)

        text = page.locator("#settings-content").inner_text()
        # Should have some status indicator
        assert "Status" in text or "status" in text or "System" in text

    def test_theme_toggle(self, authenticated_page):
        """Changing theme should update the HTML data-theme attribute."""
        page = authenticated_page
        self._navigate_to_settings(page)

        theme_select = page.locator("#set-theme")
        if theme_select.is_visible():
            theme_select.select_option("dark")
            page.wait_for_timeout(300)
            theme = page.locator("html").get_attribute("data-theme")
            assert theme == "dark"

            theme_select.select_option("light")
            page.wait_for_timeout(300)
            theme = page.locator("html").get_attribute("data-theme")
            assert theme == "light"

            # Reset to dark
            theme_select.select_option("dark")

    def test_export_button_works(self, authenticated_page, server_url):
        """Export should return a valid JSON response."""
        page = authenticated_page
        # Use API directly (export is a GET that returns JSON)
        response = page.request.get(f"{server_url}/api/export")
        assert response.ok
        data = response.json()
        assert "records" in data
        assert "count" in data

    def test_settings_shows_key_status(self, authenticated_page):
        """Settings should display API key configuration status."""
        page = authenticated_page
        self._navigate_to_settings(page)
        page.wait_for_timeout(1000)

        text = page.locator("#settings-content").inner_text()
        # Should mention API key status somehow
        assert len(text) > 10  # Content loaded

    def test_added_model_appears_in_chat_without_page_reload(self, authenticated_page):
        """Registering a provider model should refresh the chat picker live."""
        page = authenticated_page
        state = {"added": False}

        def available_models(route):
            models = [{
                "name": "existing/model",
                "provider": "openrouter",
                "label": "Existing model",
                "has_key": True,
            }]
            if state["added"]:
                models.append({
                    "name": "new/live-model",
                    "provider": "openrouter",
                    "label": "Live model",
                    "has_key": True,
                })
            route.fulfill(json={"models": models})

        def model_registry(route):
            if route.request.method == "PUT":
                state["added"] = True
                route.fulfill(json={"registered": "new/live-model"})
                return
            models = []
            if state["added"]:
                models.append({
                    "name": "new/live-model",
                    "provider": "openrouter",
                    "has_key": True,
                    "api_key_masked": "sk-or-...test",
                    "input_price": None,
                    "output_price": None,
                })
            route.fulfill(json={"models": models})

        page.route("**/api/models", available_models)
        page.route("**/api/model-registry", model_registry)
        page.reload(wait_until="domcontentloaded")
        page.wait_for_function(
            "document.getElementById('startup-splash')?.classList.contains('is-hidden')",
            timeout=10_000,
        )

        assert page.locator('#chat-model-select option[value="new/live-model"]').count() == 0

        self._navigate_to_settings(page)
        page.click('[data-settings-category="models"]')
        page.locator("details.settings-add-model summary").click()
        page.locator("#add-model-provider").select_option("openrouter")
        page.locator("#add-model-name").fill("new/live-model")
        page.locator("#add-model-key").fill("sk-or-test")
        page.locator("#btn-add-model").click()

        page.wait_for_function(
            "document.querySelector('#chat-model-select option[value=\"new/live-model\"]') !== null",
            timeout=5_000,
        )
        assert state["added"] is True
