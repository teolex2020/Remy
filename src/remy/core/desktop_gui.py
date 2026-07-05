"""
Desktop GUI launcher — FastAPI backend + PyWebView native window.

Usage:
    remy --desktop   # native window
    remy --web       # browser only
"""

import asyncio
import importlib
import logging
import socket
import threading
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from remy.config.settings import settings
from remy.core.agent_tools import brain
from remy.core.brain_tools import get_registry
from remy.core.notification_router import set_web_runtime_enabled
from remy.core.pinchtab_service import (
    ensure_pinchtab_running_sync,
    shutdown_pinchtab_sync,
)
from remy.web.api import (
    load_push_subscription,
    router,
    set_session_manager,
    shutdown_cleanup,
    start_scheduler,
)
from remy.web.session import WebSessionManager

logger = logging.getLogger("DesktopGUI")


@asynccontextmanager
async def _app_lifespan(app: FastAPI):
    await start_scheduler()
    await load_push_subscription()
    try:
        yield
    finally:
        await shutdown_cleanup()


def _static_dir() -> Path:
    """Resolve frontend static assets for source checkouts and packaged builds."""
    try:
        static = resources.files("remy.web").joinpath("static")
        path = Path(str(static))
    except Exception:
        path = Path(__file__).parent.parent / "web" / "static"

    if not (path / "index.html").exists():
        fallback = Path(__file__).parent.parent / "web" / "static"
        if (fallback / "index.html").exists():
            return fallback
    return path


STATIC_DIR = _static_dir()


ROUTE_MODULES = (
    "remy.web.routes.autonomy_routes",
    "remy.web.routes.diagnostics",
    "remy.web.routes.documents_routes",
    "remy.web.routes.knowledge_routes",
    "remy.web.routes.media_routes",
    "remy.web.routes.memory",
    "remy.web.routes.pricing_routes",
    "remy.web.routes.push_routes",
    "remy.web.routes.settings_routes",
    "remy.web.routes.system_routes",
    "remy.web.routes.todos_routes",
    "remy.web.routes.glass_brain_routes",
    "remy.web.routes.ollama_routes",
    "remy.web.routes.pipeline_routes",
    "remy.web.routes.scheduled_pipeline_routes",
    "remy.web.routes.automation_routes",
    "remy.web.routes.websocket",
)


def _is_port_available(host: str, port: int) -> bool:
    """Return True if the local web server can bind to host:port."""
    family = socket.AF_INET6 if host == "::1" else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        if probe.connect_ex((host, port)) == 0:
            return False
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _choose_web_port(host: str, preferred_port: int, attempts: int = 20) -> int:
    """Use the configured port when possible, otherwise try nearby ports."""
    for offset in range(attempts):
        port = preferred_port + offset
        if _is_port_available(host, port):
            if port != preferred_port:
                logger.warning(
                    "Configured web port %s is busy; using %s for this run",
                    preferred_port,
                    port,
                )
            return port
    return preferred_port


def _dedupe_routes(app: FastAPI) -> None:
    """Keep split route modules authoritative while preserving legacy fallbacks."""
    seen = set()
    unique = []
    for route in app.router.routes:
        path = getattr(route, "path", None) or getattr(route, "path_format", None)
        if not path:
            # FastAPI may keep included routers as wrapper routes. Those wrappers
            # intentionally have no direct path; collapsing them by "" drops whole
            # route modules and turns frontend API calls into 404s.
            unique.append(route)
            continue
        methods = tuple(sorted(getattr(route, "methods", []) or []))
        key = (type(route).__name__, path, methods)
        if key in seen:
            continue
        seen.add(key)
        unique.append(route)
    app.router.routes = unique


def _install_quiet_asyncio_handler(loop: asyncio.AbstractEventLoop):
    """Suppress harmless Windows Proactor transport cleanup noise."""
    original_handler = loop.get_exception_handler()

    def _quiet_exception_handler(loop, context):
        message = context.get("message", "")
        exception = context.get("exception")
        if "_call_connection_lost" in message or "Event loop is closed" in str(exception or ""):
            return
        if original_handler:
            original_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(_quiet_exception_handler)


def create_app() -> FastAPI:
    """Create FastAPI app with routes and static files."""
    app = FastAPI(title="Remy - Local AI Workflow Automation", lifespan=_app_lifespan)

    # Prefer the newer split route modules first. Keep the legacy web/api.py
    # router as a fallback for endpoints that have not yet been migrated.
    for module_name in ROUTE_MODULES:
        module = importlib.import_module(module_name)
        app.include_router(module.router, prefix="/api")
    app.include_router(router)
    _dedupe_routes(app)

    from remy.web.api import (
        RateLimitMiddleware, RequestLoggingMiddleware,
        NoCacheStaticMiddleware,
    )
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(NoCacheStaticMiddleware)

    # Static files (HTML/CSS/JS frontend).
    # Wrap in a guard so WebSocket requests that miss all routes don't crash
    # StaticFiles (which only handles HTTP).
    _static = StaticFiles(directory=str(STATIC_DIR), html=True)

    async def _safe_static(scope, receive, send):
        if scope["type"] != "http":
            # WebSocket or lifespan — not for StaticFiles
            if scope["type"] == "websocket":
                await receive()          # consume the WS connect
                await send({"type": "websocket.close", "code": 1000})
            return
        await _static(scope, receive, send)

    app.mount("/", _safe_static, name="static")

    return app


class DesktopGUI:
    """Launches the desktop GUI: FastAPI server + optional PyWebView window."""

    def __init__(self):
        self.manager = WebSessionManager()
        set_session_manager(self.manager)
        self.app = create_app()
        self.host = settings.WEB_HOST
        self.port = _choose_web_port(self.host, settings.WEB_PORT)
        settings.WEB_PORT = self.port

    def _print_banner(self, mode: str):
        from remy.core.agent_tools import brain_lock
        with brain_lock:
            brain_count = brain.count()
        registry = get_registry()
        tool_count = len(registry.get_all_declarations())

        print("=" * 50)
        print(f"REMY — {mode}")
        if self.manager.readonly:
            print("  ** READONLY MODE — no API key, chat disabled **")
            print("  ** Go to Settings page to configure API key **")
        print(f"Model: {settings.SUMMARY_MODEL}")
        print(f"Brain: {settings.AURA_BRAIN_PATH} ({brain_count} records)")
        print(f"Tools: {tool_count}")
        print(f"URL: http://{self.host}:{self.port}")
        print("Press Ctrl+C to stop.")
        print("=" * 50)

    def _run_server(self):
        """Run uvicorn in the current thread (blocking)."""
        async def _serve():
            loop = asyncio.get_running_loop()
            _install_quiet_asyncio_handler(loop)

            config = uvicorn.Config(
                app=self.app,
                host=self.host,
                port=self.port,
                log_level="warning",
                ws_max_size=30 * 1024 * 1024,  # 30MB for base64-encoded files
            )
            server = uvicorn.Server(config)
            try:
                await server.serve()
            except asyncio.CancelledError:
                pass  # Normal shutdown — suppress noisy traceback

        asyncio.run(_serve())

    def run_desktop(self):
        """Start FastAPI in a background thread, then open PyWebView native window."""
        import webview

        self._print_banner("DESKTOP GUI")
        set_web_runtime_enabled(True)

        # Start Pinchtab in background — don't block window open
        threading.Thread(target=ensure_pinchtab_running_sync, daemon=True, name="pinchtab-init").start()

        # Start uvicorn in a daemon thread
        server_thread = threading.Thread(target=self._run_server, daemon=True)
        server_thread.start()

        url = f"http://{self.host}:{self.port}"
        logger.info(f"Opening native window: {url}")

        window = webview.create_window(
            "Remy",
            url,
            width=1200,
            height=800,
            min_size=(800, 500),
        )
        webview.start()

        # Window closed — cleanup
        set_web_runtime_enabled(False)
        shutdown_pinchtab_sync()
        logger.info("Desktop window closed, shutting down...")

    def run_web_only(self):
        """Start FastAPI server for browser access (no PyWebView)."""
        self._print_banner("WEB GUI (browser)")
        set_web_runtime_enabled(True)
        threading.Thread(target=ensure_pinchtab_running_sync, daemon=True, name="pinchtab-init").start()
        try:
            self._run_server()
        finally:
            set_web_runtime_enabled(False)
            shutdown_pinchtab_sync()
