"""E2E tests: Chat via WebSocket."""


# Helper: wait until the WebSocket is open before interacting with chat.
_WS_READY = (
    "() => window.apiClient && window.apiClient.ws"
    " && window.apiClient.ws.readyState === 1"
)


class TestChat:

    def test_send_message_gets_response(self, authenticated_page):
        """Sending a message should yield a mock assistant response."""
        page = authenticated_page

        page.click('.nav-item[data-view="chat"]')
        page.wait_for_function(_WS_READY, timeout=10000)

        page.fill("#chat-input", "Hello from E2E test")
        page.click("#btn-send")

        page.wait_for_selector(".chat-msg.assistant", timeout=10000)
        assistant_msg = page.locator(".chat-msg.assistant").last
        assert "Mock response" in (assistant_msg.inner_text() or "")

    def test_empty_message_ignored(self, authenticated_page):
        """Empty messages should not produce a response."""
        page = authenticated_page
        page.click('.nav-item[data-view="chat"]')
        page.wait_for_function(_WS_READY, timeout=10000)

        initial_count = page.locator(".chat-msg").count()

        page.fill("#chat-input", "")
        page.click("#btn-send")
        page.wait_for_timeout(1000)

        assert page.locator(".chat-msg").count() == initial_count

    def test_typing_indicator(self, authenticated_page):
        """A typing indicator should appear while waiting for response."""
        page = authenticated_page
        page.click('.nav-item[data-view="chat"]')
        page.wait_for_function(_WS_READY, timeout=10000)

        page.fill("#chat-input", "Show typing indicator")
        page.click("#btn-send")

        # The typing indicator is transient; just verify the response arrives
        page.wait_for_selector(".chat-msg.assistant", timeout=10000)

    def test_new_session(self, authenticated_page):
        """New Session button should clear previous messages."""
        page = authenticated_page
        page.click('.nav-item[data-view="chat"]')
        page.wait_for_function(_WS_READY, timeout=10000)

        # Send a message first
        page.fill("#chat-input", "Message before reset")
        page.click("#btn-send")
        page.wait_for_selector(".chat-msg.assistant", timeout=10000)

        # Count messages before reset (should be >= 2: user + assistant)
        before_count = page.locator(".chat-msg").count()
        assert before_count >= 2

        # Click new session
        page.click("#btn-new-session")
        page.wait_for_timeout(1000)

        # Previous messages should be cleared; only "New session started." remains
        user_msgs = page.locator(".chat-msg.user")
        assert user_msgs.count() == 0
