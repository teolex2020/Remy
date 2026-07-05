"""Tests for Critical Action Audit Trail."""

import json
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def audit_dir(tmp_path):
    """Temporary audit log directory."""
    d = tmp_path / "audit_logs"
    d.mkdir()
    return d


@pytest.fixture
def logger(audit_dir):
    """Fresh AuditLogger instance."""
    from remy.core.audit_trail import AuditLogger
    return AuditLogger(log_dir=audit_dir)


# ============== AuditLogger ==============

class TestAuditLogger:

    def test_log_action_creates_jsonl_file(self, logger, audit_dir):
        logger.log_action(
            tool_name="cdp_agent_manager",
            tool_input={"action": "create_wallet"},
            raw_output='{"wallet": "0xABC"}',
            status="success",
            execution_time_ms=150.0,
        )
        files = list(audit_dir.glob("audit_*.jsonl"))
        assert len(files) == 1
        with open(files[0]) as f:
            entry = json.loads(f.readline())
        assert entry["tool_name"] == "cdp_agent_manager"
        assert entry["status"] == "success"
        assert "timestamp" in entry
        assert "checksum" in entry

    def test_log_action_checksum_valid(self, logger, audit_dir):
        from remy.core.audit_trail import AuditLogger
        entry = logger.log_action(
            tool_name="test_tool",
            tool_input={"x": 1},
            raw_output="ok",
            status="success",
            execution_time_ms=10.0,
        )
        expected = AuditLogger._compute_checksum(entry)
        assert entry["checksum"] == expected

    def test_log_action_appends(self, logger, audit_dir):
        logger.log_action("tool1", {}, "out1", "success", 10.0)
        logger.log_action("tool2", {}, "out2", "error", 20.0, error_message="fail")
        logger.log_action("tool3", {}, "out3", "success", 30.0)
        files = list(audit_dir.glob("audit_*.jsonl"))
        assert len(files) == 1
        with open(files[0]) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 3

    def test_sanitize_input_redacts_secrets(self, logger):
        sanitized = logger._sanitize_input({
            "action": "create",
            "password": "secret123",
            "api_key": "sk-abc",
            "network": "base",
        })
        assert sanitized["action"] == "create"
        assert sanitized["network"] == "base"
        assert sanitized["password"] == "***REDACTED***"
        assert sanitized["api_key"] == "***REDACTED***"

    def test_sanitize_input_redacts_partial_match(self, logger):
        sanitized = logger._sanitize_input({
            "user_api_key_v2": "my-key",
            "seed_phrase_backup": "word1 word2",
        })
        assert sanitized["user_api_key_v2"] == "***REDACTED***"
        assert sanitized["seed_phrase_backup"] == "***REDACTED***"

    def test_truncate_output(self, logger):
        long_output = "x" * 3000
        truncated = logger._truncate_output(long_output)
        assert len(truncated) < 3000
        assert truncated.endswith("...[truncated]")

    def test_truncate_output_short(self, logger):
        short = "short output"
        assert logger._truncate_output(short) == short

    def test_get_recent_logs(self, logger):
        for i in range(5):
            logger.log_action(f"tool_{i}", {}, f"out_{i}", "success", 10.0)
        logs = logger.get_recent_logs(n=3)
        assert len(logs) == 3

    def test_get_recent_logs_filter_by_tool(self, logger):
        logger.log_action("tool_a", {}, "out_a", "success", 10.0)
        logger.log_action("tool_b", {}, "out_b", "success", 10.0)
        logger.log_action("tool_a", {}, "out_a2", "error", 10.0)
        logs = logger.get_recent_logs(n=10, tool_name="tool_a")
        assert len(logs) == 2
        assert all(l["tool_name"] == "tool_a" for l in logs)

    def test_verify_integrity_clean(self, logger):
        logger.log_action("tool", {}, "out", "success", 10.0)
        logger.log_action("tool", {}, "out2", "error", 20.0)
        result = logger.verify_integrity()
        assert result["integrity"] == "OK"
        assert result["total_entries"] == 2
        assert result["corrupted_entries"] == 0

    def test_verify_integrity_corrupted(self, logger, audit_dir):
        logger.log_action("tool", {}, "out", "success", 10.0)
        # Tamper with the log file
        log_file = list(audit_dir.glob("audit_*.jsonl"))[0]
        with open(log_file, "r") as f:
            entry = json.loads(f.readline())
        entry["raw_output"] = "TAMPERED"
        with open(log_file, "w") as f:
            f.write(json.dumps(entry) + "\n")
        result = logger.verify_integrity()
        assert result["integrity"] == "COMPROMISED"
        assert result["corrupted_entries"] == 1

    def test_channel_recorded(self, logger, audit_dir):
        logger.log_action("tool", {}, "out", "success", 10.0, channel="autonomous")
        files = list(audit_dir.glob("audit_*.jsonl"))
        with open(files[0]) as f:
            entry = json.loads(f.readline())
        assert entry["channel"] == "autonomous"

    def test_channel_omitted_when_none(self, logger, audit_dir):
        logger.log_action("tool", {}, "out", "success", 10.0)
        files = list(audit_dir.glob("audit_*.jsonl"))
        with open(files[0]) as f:
            entry = json.loads(f.readline())
        assert "channel" not in entry

    def test_get_summary(self, logger):
        logger.log_action("tool_a", {}, "out", "success", 10.0)
        logger.log_action("tool_a", {}, "out", "error", 10.0, error_message="fail")
        logger.log_action("tool_b", {}, "out", "success", 10.0)
        summary = logger.get_summary()
        assert summary["total"] == 3
        assert summary["success"] == 2
        assert summary["error"] == 1
        assert "tool_a" in summary["by_tool"]
        assert summary["by_tool"]["tool_a"]["total"] == 2


# ============== is_critical ==============

class TestIsCritical:

    def test_default_critical_tool(self):
        from remy.core.audit_trail import (
            _DEFAULT_CRITICAL_TOOLS, is_critical, reset_audit_logger,
        )
        reset_audit_logger()
        with patch("remy.config.settings.settings") as mock_s:
            mock_s.CRITICAL_TOOLS = []
            tool = next(iter(_DEFAULT_CRITICAL_TOOLS))
            assert is_critical(tool) is True
        reset_audit_logger()

    def test_non_critical_tool(self):
        from remy.core.audit_trail import is_critical, reset_audit_logger
        reset_audit_logger()
        with patch("remy.config.settings.settings") as mock_s:
            mock_s.CRITICAL_TOOLS = []
            assert is_critical("recall") is False
            assert is_critical("store") is False
        reset_audit_logger()

    def test_custom_critical_tools_from_settings(self):
        from remy.core.audit_trail import is_critical, reset_audit_logger
        reset_audit_logger()
        with patch("remy.config.settings.settings") as mock_s:
            mock_s.CRITICAL_TOOLS = ["my_custom_tool", "another_tool"]
            assert is_critical("my_custom_tool") is True
            assert is_critical("cdp_agent_manager") is False  # defaults replaced
        reset_audit_logger()


# ============== execute_tool integration ==============

class TestExecuteToolAudit:

    def test_critical_tool_audited(self, tmp_path):
        """execute_tool logs when a critical sandbox tool is called."""
        audit_dir = tmp_path / "audit"
        audit_dir.mkdir()

        mock_brain = MagicMock()
        mock_registry = MagicMock()
        mock_registry.is_sandbox_tool.return_value = True
        mock_registry.execute_sandbox_tool.return_value = '{"wallet": "0xABC"}'

        from remy.core.audit_trail import AuditLogger, reset_audit_logger
        reset_audit_logger()

        with patch("remy.core.brain_tools.brain", mock_brain), \
             patch("remy.core.brain_tools._registry", mock_registry), \
             patch("remy.core.brain_tools.tool_health") as mock_health, \
             patch("remy.core.audit_trail.is_critical", return_value=True), \
             patch("remy.core.audit_trail.get_audit_logger") as mock_get_logger:
            mock_health.is_available.return_value = True
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            from remy.core.brain_tools import execute_tool
            result = execute_tool("cdp_agent_manager", {"action": "create_wallet"},
                                  channel="autonomous")

        assert result == '{"wallet": "0xABC"}'
        mock_logger.log_action.assert_called_once()
        call_kwargs = mock_logger.log_action.call_args
        assert call_kwargs.kwargs["tool_name"] == "cdp_agent_manager"
        assert call_kwargs.kwargs["status"] == "success"
        assert call_kwargs.kwargs["channel"] == "autonomous"
        reset_audit_logger()

    def test_non_critical_tool_not_audited(self, tmp_path):
        """Regular tools like 'recall' are not audited."""
        mock_brain = MagicMock()
        mock_brain.recall_structured.return_value = []

        from remy.core.audit_trail import reset_audit_logger
        reset_audit_logger()

        with patch("remy.core.brain_tools.brain", mock_brain), \
             patch("remy.core.brain_tools._registry", None), \
             patch("remy.core.brain_tools.tool_health") as mock_health, \
             patch("remy.core.audit_trail.is_critical", return_value=False), \
             patch("remy.core.audit_trail.get_audit_logger") as mock_get_logger, \
             patch("remy.core.tool_registry.settings") as mock_settings:
            mock_health.is_available.return_value = True
            mock_settings.SANDBOX_DIR = tmp_path / "sandbox"
            mock_settings.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"

            from remy.core.brain_tools import execute_tool
            execute_tool("recall", {"query": "test"})

        mock_get_logger.assert_not_called()
        reset_audit_logger()

    def test_error_result_logged_as_error(self, tmp_path):
        """When tool returns error JSON, audit status is 'error'."""
        mock_brain = MagicMock()
        mock_registry = MagicMock()
        mock_registry.is_sandbox_tool.return_value = True
        mock_registry.execute_sandbox_tool.return_value = '{"error": "InvalidAPIKey"}'

        from remy.core.audit_trail import reset_audit_logger
        reset_audit_logger()

        with patch("remy.core.brain_tools.brain", mock_brain), \
             patch("remy.core.brain_tools._registry", mock_registry), \
             patch("remy.core.brain_tools.tool_health") as mock_health, \
             patch("remy.core.audit_trail.is_critical", return_value=True), \
             patch("remy.core.audit_trail.get_audit_logger") as mock_get_logger:
            mock_health.is_available.return_value = True
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            from remy.core.brain_tools import execute_tool
            execute_tool("cdp_agent_manager", {"action": "create_wallet"})

        call_kwargs = mock_logger.log_action.call_args
        assert call_kwargs.kwargs["status"] == "error"
        assert call_kwargs.kwargs["error_message"] == "InvalidAPIKey"
        reset_audit_logger()
