"""E2E test fixtures — real uvicorn server + Playwright browser."""

import socket
import threading
import time

import pytest
import uvicorn
from aura import Aura as CognitiveMemory

from remy.web.session import WebSessionManager


# ============== Mock Session Manager ==============


class MockWebSessionManager(WebSessionManager):
    """WebSessionManager that returns mock responses without a real LLM."""

    def __init__(self):
        self.readonly = False
        self.client = None
        self.session = None

    async def gemini_respond_stream(self, user_text):
        yield {"type": "token", "content": "Mock response"}
        yield {"type": "final", "text": "Mock response", "messages": []}

    async def gemini_respond_multimodal(self, text=None, attachments=None, is_voice=False):
        return {"response": "Mock multimodal response", "input_transcript": None}

    async def close_session(self):
        self.session = None


# ============== Mock Scheduler ==============


class MockScheduler:
    """No-op scheduler that doesn't start background tasks."""

    def __init__(self):
        self.running = False

    async def start(self):
        self.running = True

    async def stop(self):
        self.running = False


# ============== Free Port ==============


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ============== Server Fixture ==============


@pytest.fixture(scope="session")
def _brain(tmp_path_factory):
    """Session-scoped real CognitiveMemory."""
    path = tmp_path_factory.mktemp("e2e_brain")
    b = CognitiveMemory(str(path))
    yield b
    b.close()


@pytest.fixture(scope="session")
def server_url(_brain):
    """Start a real uvicorn server in a daemon thread, yield its URL."""
    import remy.web.api as api_mod
    import remy.core.desktop_gui as gui_mod

    port = _free_port()

    # Save originals for restore
    orig_brain_api = api_mod.brain
    orig_brain_gui = gui_mod.brain
    orig_scheduler = api_mod._scheduler

    # Swap in test doubles (direct assignment, not unittest.mock.patch)
    api_mod.brain = _brain
    gui_mod.brain = _brain
    api_mod._scheduler = MockScheduler()
    api_mod.set_session_manager(MockWebSessionManager())

    app = gui_mod.create_app()

    config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready
    url = f"http://127.0.0.1:{port}"
    import urllib.request
    for _ in range(50):
        try:
            urllib.request.urlopen(f"{url}/api/stats", timeout=1)
            break
        except Exception:
            time.sleep(0.1)

    # Disable rate limiting: after first request, middleware stack is built.
    # Walk it and raise the cap so E2E tests aren't throttled.
    # Use class name check (not isinstance) because unit tests may cause
    # module reloads that break class identity.
    middleware = app.middleware_stack
    while middleware is not None:
        if type(middleware).__name__ == "RateLimitMiddleware":
            middleware.max_requests = 999_999
            break
        middleware = getattr(middleware, "app", None)

    yield url

    server.should_exit = True
    thread.join(timeout=10)

    # Restore originals so other tests are not affected
    api_mod.brain = orig_brain_api
    gui_mod.brain = orig_brain_gui
    api_mod._scheduler = orig_scheduler

    # Allow the daemon thread's event loop to fully close
    time.sleep(0.5)


@pytest.fixture(scope="session")
def brain(_brain):
    """Expose the session-scoped brain for tests that need to store records."""
    return _brain


# ============== App Page ==============


@pytest.fixture
def authenticated_page(page, server_url):
    """Open the local desktop app directly."""
    page.goto(server_url, wait_until="domcontentloaded")
    page.wait_for_selector(".sidebar", timeout=10000)
    yield page
