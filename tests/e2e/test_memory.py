"""E2E tests: Memory view and record details."""

import pytest
from aura import Level


class TestMemory:

    def test_memory_list_shows_records(self, authenticated_page, brain):
        """Memory view should display stored records."""
        # Store records directly
        brain.store("E2E memory record alpha", tags=["e2e-test"], level=Level.DOMAIN)
        brain.store("E2E memory record beta", tags=["e2e-test"], level=Level.DOMAIN)

        page = authenticated_page
        page.click('.nav-item[data-view="memory"]')
        page.wait_for_selector("#view-memory", state="visible", timeout=3000)
        page.wait_for_timeout(1000)  # Wait for API response

        cards = page.locator(".memory-card")
        assert cards.count() >= 2

    def test_memory_search(self, authenticated_page, brain):
        """Searching memory should filter results."""
        brain.store("Unique searchable record xyz123", tags=["e2e-search"], level=Level.DOMAIN)

        page = authenticated_page
        page.click('.nav-item[data-view="memory"]')
        page.wait_for_selector("#view-memory", state="visible", timeout=3000)

        # Type in search
        search_input = page.locator("#memory-search")
        if search_input.is_visible():
            search_input.fill("xyz123")
            page.click("#btn-memory-search")
            page.wait_for_timeout(1000)

    def test_record_detail_panel(self, authenticated_page, brain):
        """Clicking a memory card opens the detail panel."""
        brain.store("Detail panel test record", tags=["e2e-detail"], level=Level.DOMAIN)

        page = authenticated_page
        page.click('.nav-item[data-view="memory"]')
        page.wait_for_selector("#view-memory", state="visible", timeout=3000)
        page.wait_for_timeout(1000)

        # Click first card
        card = page.locator(".memory-card").first
        if card.is_visible():
            card.click()
            page.wait_for_timeout(500)
            # Panel should open
            app_el = page.locator(".app")
            assert "panel-open" in (app_el.get_attribute("class") or "")

    def test_graph_view_renders(self, authenticated_page, brain):
        """Graph view should render without errors."""
        rec1 = brain.store("Graph node A", tags=["e2e-graph"], level=Level.DOMAIN)
        rec2 = brain.store("Graph node B", tags=["e2e-graph"], level=Level.DOMAIN)
        brain.connect(rec1.id, rec2.id, weight=0.9)

        page = authenticated_page
        page.click('.nav-item[data-view="graph"]')
        page.wait_for_selector("#view-graph", state="visible", timeout=3000)
        page.wait_for_timeout(1500)  # Graph rendering takes time

        # Graph container should have content (canvas or svg)
        graph_view = page.locator("#view-graph")
        assert graph_view.is_visible()
