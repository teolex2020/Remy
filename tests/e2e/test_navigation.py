"""E2E tests: Navigation between views."""

import pytest


class TestNavigation:

    def test_default_view_is_chat(self, authenticated_page):
        """After login, the chat view should be active by default."""
        page = authenticated_page
        nav = page.locator('.nav-item[data-view="chat"]')
        assert "active" in (nav.get_attribute("class") or "")

    @pytest.mark.parametrize("view", [
        "memory", "tasks", "knowledge", "graph",
        "activity", "history", "stats", "settings",
    ])
    def test_switch_to_view(self, authenticated_page, view):
        """Clicking a nav item shows the corresponding view."""
        page = authenticated_page
        page.click(f'.nav-item[data-view="{view}"]')
        page.wait_for_selector(f"#view-{view}", state="visible", timeout=3000)
        assert page.is_visible(f"#view-{view}")

    def test_nav_active_class_toggles(self, authenticated_page):
        """Only the clicked nav item should have the active class."""
        page = authenticated_page
        # Click memory
        page.click('.nav-item[data-view="memory"]')
        page.wait_for_timeout(300)
        assert "active" in (page.locator('.nav-item[data-view="memory"]').get_attribute("class") or "")
        assert "active" not in (page.locator('.nav-item[data-view="chat"]').get_attribute("class") or "")

        # Click tasks
        page.click('.nav-item[data-view="tasks"]')
        page.wait_for_timeout(300)
        assert "active" in (page.locator('.nav-item[data-view="tasks"]').get_attribute("class") or "")
        assert "active" not in (page.locator('.nav-item[data-view="memory"]').get_attribute("class") or "")

    def test_mobile_sidebar_toggle(self, authenticated_page):
        """On mobile viewport, hamburger menu opens and closes sidebar."""
        page = authenticated_page
        page.set_viewport_size({"width": 375, "height": 667})
        page.wait_for_timeout(300)

        sidebar = page.locator(".sidebar")

        # Click hamburger menu to open
        menu_btn = page.locator("#btn-menu")
        if menu_btn.is_visible():
            menu_btn.click()
            page.wait_for_timeout(300)
            assert "open" in (sidebar.get_attribute("class") or "")

            # Click overlay to close (force=True because sidebar overlaps part of it)
            overlay = page.locator("#sidebar-overlay")
            if overlay.is_visible():
                overlay.click(force=True)
                page.wait_for_timeout(300)
                assert "open" not in (sidebar.get_attribute("class") or "")

        # Reset viewport
        page.set_viewport_size({"width": 1280, "height": 720})
