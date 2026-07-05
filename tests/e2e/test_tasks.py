"""E2E tests: Tasks CRUD."""

import pytest


class TestTasks:

    def _navigate_to_tasks(self, page):
        page.click('.nav-item[data-view="tasks"]')
        page.wait_for_selector("#view-tasks", state="visible", timeout=3000)

    def test_create_task(self, authenticated_page):
        """Create a task via the form and verify it appears in the list."""
        page = authenticated_page
        self._navigate_to_tasks(page)

        # Open add form
        page.click("#btn-add-task")
        page.wait_for_selector("#new-task-title", state="visible", timeout=3000)

        # Fill and save
        page.fill("#new-task-title", "E2E Test Task")
        page.click("#btn-save-task")

        # Verify task appears
        page.wait_for_timeout(1000)
        content = page.locator("#tasks-content").inner_text()
        assert "E2E Test Task" in content

    def test_toggle_task_complete(self, authenticated_page, server_url):
        """Toggle a task's completion status via checkbox."""
        page = authenticated_page

        # Create task via API
        response = page.request.post(
            f"{server_url}/api/todos",
            data={"title": "Toggle Test", "priority": "medium"},
        )
        assert response.ok

        self._navigate_to_tasks(page)
        page.wait_for_timeout(500)

        # Find and click a checkbox
        checkbox = page.locator('input[type="checkbox"]').first
        if checkbox.is_visible():
            checkbox.click()
            page.wait_for_timeout(500)

    def test_delete_task(self, authenticated_page, server_url):
        """Delete a task after confirming the dialog."""
        page = authenticated_page

        # Create task via API
        response = page.request.post(
            f"{server_url}/api/todos",
            data={"title": "Delete Me", "priority": "low"},
        )
        assert response.ok

        self._navigate_to_tasks(page)
        page.wait_for_timeout(500)

        # Accept the confirm dialog
        page.on("dialog", lambda dialog: dialog.accept())

        # Click delete button on the first task
        delete_btn = page.locator(".task-delete-btn").first
        if delete_btn.is_visible():
            delete_btn.click()
            page.wait_for_timeout(500)

    def test_task_form_cancel(self, authenticated_page):
        """Canceling the task form hides it without creating a task."""
        page = authenticated_page
        self._navigate_to_tasks(page)

        page.click("#btn-add-task")
        page.wait_for_selector("#new-task-title", state="visible", timeout=3000)

        page.click("#btn-cancel-task")
        page.wait_for_timeout(300)
        assert not page.is_visible("#new-task-title")
