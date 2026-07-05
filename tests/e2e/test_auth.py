"""E2E tests for the local desktop app access model."""


class TestAuth:

    def test_root_loads_app_directly(self, page, server_url):
        """The local desktop app serves the main UI directly."""
        page.goto(server_url, wait_until="domcontentloaded")
        page.wait_for_selector(".sidebar", timeout=10000)
        assert page.is_visible(".sidebar")

    def test_api_accessible_without_cookie(self, page, server_url):
        """Local API endpoints are accessible without auth cookies."""
        response = page.request.get(f"{server_url}/api/stats")
        assert response.ok

    def test_settings_view_loads(self, authenticated_page):
        """Settings view loads without a login/logout flow."""
        page = authenticated_page
        page.click('.nav-item[data-view="settings"]')
        page.wait_for_selector("#view-settings", state="visible", timeout=3000)
        page.wait_for_timeout(500)
        assert page.is_visible("#view-settings")
