"""
Structured Logging Configuration — JSON for files, colored text for console.

Provides:
- ContextJsonFormatter: JSON log formatter with contextvars (session_id, channel, request_id)
- ColoredConsoleFormatter: Human-readable colored console output
- setup_logging(): Main setup function (replaces main.py's version)
- setup_autonomy_file_handler(): Shared autonomy.log handler
- log_context(): Context manager for setting log context per-request
- generate_request_id(): Short unique ID for HTTP request tracing
"""

import contextvars
import logging
import uuid
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from pythonjsonlogger.json import JsonFormatter

# ============== CONTEXT VARIABLES ==============
# Set per-request, automatically included in JSON logs.
# Thread-safe, asyncio-compatible, propagate through asyncio.to_thread().

ctx_session_id = contextvars.ContextVar("session_id", default=None)
ctx_channel = contextvars.ContextVar("channel", default=None)
ctx_request_id = contextvars.ContextVar("request_id", default=None)


# ============== JSON FORMATTER (files) ==============


class ContextJsonFormatter(JsonFormatter):
    """JSON formatter that auto-injects contextvars fields."""

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record["session_id"] = ctx_session_id.get()
        log_record["channel"] = ctx_channel.get()
        log_record["request_id"] = ctx_request_id.get()
        log_record["level"] = record.levelname
        log_record["logger"] = record.name


# ============== CONSOLE FORMATTER (human-readable) ==============


class ColoredConsoleFormatter(logging.Formatter):
    """Human-readable colored console output. Falls back gracefully on non-TTY."""

    COLORS = {
        "DEBUG": "\033[36m",      # cyan
        "INFO": "\033[32m",       # green
        "WARNING": "\033[33m",    # yellow
        "ERROR": "\033[31m",      # red
        "CRITICAL": "\033[1;31m", # bold red
    }
    RESET = "\033[0m"

    def format(self, record):
        ctx_parts = []
        sid = ctx_session_id.get()
        if sid:
            ctx_parts.append(f"sid={sid[:8]}")
        ch = ctx_channel.get()
        if ch:
            ctx_parts.append(f"ch={ch}")
        ctx_str = f" [{' '.join(ctx_parts)}]" if ctx_parts else ""

        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET if color else ""

        timestamp = self.formatTime(record, "%H:%M:%S")
        return (
            f"{timestamp} {color}{record.levelname:<7}{reset} "
            f"{record.name}{ctx_str} {record.getMessage()}"
        )


# ============== SETUP FUNCTIONS ==============


def setup_logging(
    log_to_file: bool = True,
    log_level: str = "INFO",
    log_dir: Path | None = None,
) -> logging.Logger:
    """Configure logging: JSON for files, colored text for console.

    Args:
        log_to_file: Whether to write to assistant.log
        log_level: Root log level (DEBUG/INFO/WARNING/ERROR)
        log_dir: Directory for log files. If None, uses settings.DATA_DIR / "logs"

    Returns:
        The "Remy" logger for backward compatibility.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    # 1. Console handler — human-readable, colored
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(ColoredConsoleFormatter())
    root.addHandler(console)

    # 2. File handler — JSON, machine-parseable
    if log_to_file:
        if log_dir is None:
            from remy.config.settings import settings
            log_dir = settings.DATA_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_dir / "assistant.log",
            maxBytes=1 * 1024 * 1024,  # 1 MB
            backupCount=2,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(ContextJsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s"
        ))
        root.addHandler(file_handler)

    return logging.getLogger("Remy")


def setup_autonomy_file_handler(logger_instance: logging.Logger, log_dir: Path):
    """Add a JSON-formatted RotatingFileHandler for autonomy.log.

    Keeps the same file path, rotation, and size as the original setup.
    Includes dedup guard to prevent duplicate handlers.
    """
    log_path = log_dir / "autonomy.log"

    # Guard: don't add duplicate handlers
    for h in logger_instance.handlers:
        if isinstance(h, RotatingFileHandler) and hasattr(h, "baseFilename"):
            if Path(h.baseFilename).name == "autonomy.log":
                return

    handler = RotatingFileHandler(
        log_path,
        maxBytes=1 * 1024 * 1024,  # 1 MB
        backupCount=2,
        encoding="utf-8",
        delay=True,
    )
    handler.setFormatter(ContextJsonFormatter(
        "%(asctime)s %(levelname)s %(message)s"
    ))
    logger_instance.addHandler(handler)


# ============== CONTEXT HELPERS ==============


@contextmanager
def log_context(session_id=None, channel=None, request_id=None):
    """Context manager to set logging context for a block of code.

    Usage:
        with log_context(session_id="abc-123", channel="telegram"):
            logger.info("Processing message")  # auto-enriched
    """
    tokens = []
    if session_id is not None:
        tokens.append(ctx_session_id.set(session_id))
    if channel is not None:
        tokens.append(ctx_channel.set(channel))
    if request_id is not None:
        tokens.append(ctx_request_id.set(request_id))
    try:
        yield
    finally:
        for token in tokens:
            token.var.reset(token)


def generate_request_id() -> str:
    """Generate a short request ID for HTTP request tracing."""
    return uuid.uuid4().hex[:12]
