"""Tests for structured logging configuration — JSON formatter, contextvars, setup."""

import json
import logging
import re

import pytest

from remy.core.logging_config import (
    ColoredConsoleFormatter,
    ContextJsonFormatter,
    ctx_channel,
    ctx_request_id,
    ctx_session_id,
    generate_request_id,
    log_context,
    setup_autonomy_file_handler,
    setup_logging,
)


# ============== FIXTURES ==============


@pytest.fixture(autouse=True)
def _reset_contextvars():
    """Ensure contextvars are clean before and after each test."""
    # Reset to defaults
    for var in (ctx_session_id, ctx_channel, ctx_request_id):
        tok = var.set(None)
    yield
    for var in (ctx_session_id, ctx_channel, ctx_request_id):
        tok = var.set(None)


@pytest.fixture
def json_formatter():
    return ContextJsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s")


@pytest.fixture
def colored_formatter():
    return ColoredConsoleFormatter()


def _make_record(msg="test message", name="TestLogger", level=logging.INFO):
    """Create a log record for testing formatters."""
    logger = logging.getLogger(name)
    return logger.makeRecord(name, level, "test.py", 42, msg, (), None)


# ============== JSON FORMATTER ==============


def test_json_formatter_valid_json(json_formatter):
    """JSON formatter output is valid JSON."""
    record = _make_record("hello world")
    output = json_formatter.format(record)
    parsed = json.loads(output)
    assert parsed["message"] == "hello world"
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "TestLogger"


def test_context_vars_in_json(json_formatter):
    """Contextvars appear in JSON output when set."""
    ctx_session_id.set("sess-abc-123")
    ctx_channel.set("telegram")
    ctx_request_id.set("req-xyz")

    record = _make_record("with context")
    output = json_formatter.format(record)
    parsed = json.loads(output)

    assert parsed["session_id"] == "sess-abc-123"
    assert parsed["channel"] == "telegram"
    assert parsed["request_id"] == "req-xyz"


def test_context_vars_default_none(json_formatter):
    """Contextvars default to None when not set."""
    record = _make_record("no context")
    output = json_formatter.format(record)
    parsed = json.loads(output)

    assert parsed["session_id"] is None
    assert parsed["channel"] is None
    assert parsed["request_id"] is None


# ============== LOG CONTEXT ==============


def test_log_context_sets_and_resets():
    """log_context sets vars inside block and resets after."""
    assert ctx_session_id.get() is None
    assert ctx_channel.get() is None

    with log_context(session_id="abc-123", channel="desktop"):
        assert ctx_session_id.get() == "abc-123"
        assert ctx_channel.get() == "desktop"

    # After block — reset to None
    assert ctx_session_id.get() is None
    assert ctx_channel.get() is None


def test_log_context_resets_on_exception():
    """log_context resets vars even when exception occurs."""
    with pytest.raises(ValueError):
        with log_context(session_id="fail-sess", channel="web"):
            assert ctx_session_id.get() == "fail-sess"
            raise ValueError("boom")

    assert ctx_session_id.get() is None
    assert ctx_channel.get() is None


def test_log_context_partial_vars():
    """log_context only sets provided vars, leaves others untouched."""
    ctx_channel.set("pre-existing")

    with log_context(session_id="partial-test"):
        assert ctx_session_id.get() == "partial-test"
        assert ctx_channel.get() == "pre-existing"  # not touched

    assert ctx_session_id.get() is None
    assert ctx_channel.get() == "pre-existing"  # still untouched

    # Cleanup
    ctx_channel.set(None)


# ============== COLORED FORMATTER ==============


def test_colored_formatter_smoke(colored_formatter):
    """Colored formatter doesn't crash and produces output."""
    record = _make_record("smoke test")
    output = colored_formatter.format(record)
    assert "smoke test" in output
    assert "INFO" in output


def test_colored_formatter_with_context(colored_formatter):
    """Colored formatter includes session_id and channel when set."""
    ctx_session_id.set("abcdef1234567890")
    ctx_channel.set("telegram")

    record = _make_record("ctx test")
    output = colored_formatter.format(record)

    assert "sid=abcdef12" in output  # truncated to 8 chars
    assert "ch=telegram" in output


def test_colored_formatter_no_context(colored_formatter):
    """Colored formatter works without context (no brackets)."""
    record = _make_record("no ctx")
    output = colored_formatter.format(record)

    assert "sid=" not in output
    assert "ch=" not in output
    assert "[" not in output or "INFO" in output  # no context brackets


# ============== SETUP LOGGING ==============


def test_setup_logging_creates_handlers(tmp_path):
    """setup_logging creates console + file handlers."""
    logger = setup_logging(log_to_file=True, log_dir=tmp_path)

    root = logging.getLogger()
    handler_types = [type(h).__name__ for h in root.handlers]

    assert "StreamHandler" in handler_types
    assert "RotatingFileHandler" in handler_types
    assert logger.name == "Remy"


def test_setup_logging_clears_duplicates(tmp_path):
    """Calling setup_logging twice doesn't create duplicate handlers."""
    setup_logging(log_to_file=True, log_dir=tmp_path)
    handler_count_1 = len(logging.getLogger().handlers)

    setup_logging(log_to_file=True, log_dir=tmp_path)
    handler_count_2 = len(logging.getLogger().handlers)

    assert handler_count_1 == handler_count_2


def test_setup_logging_no_file(tmp_path):
    """setup_logging with log_to_file=False creates only console handler."""
    setup_logging(log_to_file=False)

    root = logging.getLogger()
    handler_types = [type(h).__name__ for h in root.handlers]

    assert "StreamHandler" in handler_types
    assert "RotatingFileHandler" not in handler_types


# ============== AUTONOMY HANDLER ==============


def test_autonomy_handler_dedup(tmp_path):
    """setup_autonomy_file_handler doesn't add duplicate handlers."""
    test_logger = logging.getLogger("test.autonomy.dedup")
    test_logger.handlers.clear()

    setup_autonomy_file_handler(test_logger, tmp_path)
    count_1 = len(test_logger.handlers)

    setup_autonomy_file_handler(test_logger, tmp_path)
    count_2 = len(test_logger.handlers)

    assert count_1 == 1
    assert count_2 == 1  # no duplicate


# ============== GENERATE REQUEST ID ==============


def test_generate_request_id_format():
    """generate_request_id returns 12-char hex string."""
    rid = generate_request_id()
    assert len(rid) == 12
    assert re.fullmatch(r"[0-9a-f]{12}", rid)


def test_generate_request_id_unique():
    """generate_request_id returns unique values."""
    ids = {generate_request_id() for _ in range(100)}
    assert len(ids) == 100
