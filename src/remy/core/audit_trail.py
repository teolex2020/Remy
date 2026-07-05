"""
Audit Trail — append-only log for critical tool executions.

Captures raw input/output of critical tools (finance, registration, identity)
so the user can verify what actually happened regardless of agent interpretation.

The agent cannot edit this log — only append. Each entry has a SHA-256 checksum
for tamper detection.
"""

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ============================================================
# Default critical tools — audited when no CRITICAL_TOOLS env set
# ============================================================

_DEFAULT_CRITICAL_TOOLS: set[str] = {
    # Finance / crypto
    "cdp_agent_manager",
    "cdp_wallet_manager",
    # Registration / identity
    "identity_manager",
    "mail_tm_tool",
    "sms_man_manager",
    "smspool_tool",
    # Anti-fraud
    "capsolver_manager",
    "proxy_manager",
    # External platforms
    "openwork_client",
    # HTTP (registrations, forms)
    "http_request",
    "http_poster",
}

# Sensitive keys in tool input — values will be redacted
_SENSITIVE_KEYS = {
    "password", "api_key", "secret", "token",
    "card_number", "cvv", "private_key", "seed_phrase",
}


# ============================================================
# AuditLogger
# ============================================================

class AuditLogger:
    """
    Append-only JSONL logger for critical tool executions.

    Each entry contains: timestamp, tool_name, tool_input (sanitized),
    raw_output, status, execution_time_ms, channel, a SHA-256 checksum,
    and a prev_hash linking to the previous entry (Merkle hash-chain).

    The chain guarantees: if any entry is modified or deleted, all subsequent
    checksums break — tamper-evident by design.
    """

    # Genesis hash for the first entry in each daily file
    GENESIS_HASH = "0" * 16

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._prev_hash: str = self.GENESIS_HASH
        self._prev_hash_loaded: bool = False
        self._lock = threading.Lock()

    def _get_log_path(self) -> Path:
        """One JSONL file per day."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_dir / f"audit_{date_str}.jsonl"

    def _load_prev_hash(self) -> str:
        """Load the last checksum from today's file to continue the chain."""
        log_path = self._get_log_path()
        if not log_path.exists():
            return self.GENESIS_HASH
        last_checksum = self.GENESIS_HASH
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        last_checksum = entry.get("checksum", self.GENESIS_HASH)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
        return last_checksum

    @staticmethod
    def _compute_checksum(entry: dict) -> str:
        """SHA-256 of entry (excluding checksum field), truncated to 16 hex chars.

        Because the entry includes prev_hash, this creates a Merkle chain:
        checksum_n = SHA256(prev_hash + entry_data)
        """
        clean = {k: v for k, v in entry.items() if k != "checksum"}
        raw = json.dumps(clean, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _sanitize_input(tool_input: dict) -> dict:
        """Redact sensitive values (passwords, API keys, etc.)."""
        sanitized = {}
        for key, value in tool_input.items():
            if any(s in key.lower() for s in _SENSITIVE_KEYS):
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, str) and len(value) > 500:
                sanitized[key] = value[:500] + "...[truncated]"
            else:
                sanitized[key] = value
        return sanitized

    @staticmethod
    def _truncate_output(output: Any, max_length: int = 2000) -> Any:
        """Cap output size for the log."""
        if isinstance(output, str) and len(output) > max_length:
            return output[:max_length] + "...[truncated]"
        return output

    def log_action(
        self,
        tool_name: str,
        tool_input: dict,
        raw_output: Any,
        status: str,
        execution_time_ms: float,
        channel: str | None = None,
        error_message: str | None = None,
    ) -> dict:
        """Append a critical action entry to the audit log. Returns the entry.

        Each entry includes prev_hash from the previous entry, creating a
        Merkle hash-chain. Tampering with any entry breaks all subsequent checksums.
        """
        with self._lock:
            # Lazy-load chain head on first call
            if not self._prev_hash_loaded:
                self._prev_hash = self._load_prev_hash()
                self._prev_hash_loaded = True

            entry: dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool_name": tool_name,
                "tool_input": self._sanitize_input(tool_input),
                "raw_output": self._truncate_output(raw_output),
                "status": status,
                "execution_time_ms": round(execution_time_ms, 2),
                "prev_hash": self._prev_hash,
            }
            if channel:
                entry["channel"] = channel
            if error_message:
                entry["error"] = error_message

            entry["checksum"] = self._compute_checksum(entry)

            # Advance the chain
            self._prev_hash = entry["checksum"]

            try:
                with open(self._get_log_path(), "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except OSError as e:
                logger.error("Failed to write audit log: %s", e)

        return entry

    def get_recent_logs(self, n: int = 20, tool_name: str | None = None) -> list[dict]:
        """Return the last N audit entries, newest first. Optionally filter by tool."""
        entries: list[dict] = []
        log_files = sorted(self.log_dir.glob("audit_*.jsonl"), reverse=True)

        for log_file in log_files:
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            if tool_name and entry.get("tool_name") != tool_name:
                                continue
                            entries.append(entry)
                        except json.JSONDecodeError:
                            continue
            except OSError:
                continue
            if len(entries) >= n * 2:
                break

        entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return entries[:n]

    def verify_integrity(self) -> dict:
        """Verify both per-entry checksums AND Merkle chain continuity.

        Returns:
            {
                "total_entries": int,
                "corrupted_entries": int,  # bad checksums
                "chain_breaks": int,       # prev_hash mismatches
                "integrity": "OK" | "COMPROMISED",
                "files_checked": int,
            }
        """
        total = 0
        corrupted = 0
        chain_breaks = 0
        files_checked = 0

        for log_file in sorted(self.log_dir.glob("audit_*.jsonl")):
            files_checked += 1
            prev_hash = self.GENESIS_HASH
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        total += 1
                        try:
                            entry = json.loads(line)
                            # Check entry checksum
                            expected = self._compute_checksum(entry)
                            if entry.get("checksum") != expected:
                                corrupted += 1
                            # Check Merkle chain link
                            entry_prev = entry.get("prev_hash")
                            if entry_prev is not None and entry_prev != prev_hash:
                                chain_breaks += 1
                            prev_hash = entry.get("checksum", prev_hash)
                        except json.JSONDecodeError:
                            corrupted += 1
            except OSError:
                continue

        ok = corrupted == 0 and chain_breaks == 0
        return {
            "total_entries": total,
            "corrupted_entries": corrupted,
            "chain_breaks": chain_breaks,
            "integrity": "OK" if ok else "COMPROMISED",
            "files_checked": files_checked,
        }

    def get_summary(self) -> dict:
        """Aggregate stats by tool and status."""
        logs = self.get_recent_logs(n=1000)
        by_status = {"total": len(logs), "success": 0, "error": 0, "timeout": 0}
        by_tool: dict[str, dict] = {}

        for entry in logs:
            status = entry.get("status", "unknown")
            if status in by_status:
                by_status[status] += 1

            tn = entry.get("tool_name", "unknown")
            if tn not in by_tool:
                by_tool[tn] = {"total": 0, "success": 0, "error": 0, "timeout": 0}
            by_tool[tn]["total"] += 1
            if status in by_tool[tn]:
                by_tool[tn][status] += 1

        by_status["by_tool"] = by_tool
        return by_status


# ============================================================
# Module-level helpers
# ============================================================

_audit_logger: AuditLogger | None = None
_critical_tools: set[str] | None = None


def _load_critical_tools() -> set[str]:
    """Load critical tools from settings (if configured) or use defaults."""
    global _critical_tools
    if _critical_tools is not None:
        return _critical_tools
    try:
        from remy.config.settings import settings
        configured = settings.CRITICAL_TOOLS
        if configured:
            _critical_tools = set(configured)
        else:
            _critical_tools = _DEFAULT_CRITICAL_TOOLS.copy()
    except Exception:
        _critical_tools = _DEFAULT_CRITICAL_TOOLS.copy()
    return _critical_tools


def is_critical(tool_name: str) -> bool:
    """Check if a tool is in the critical tools list."""
    return tool_name in _load_critical_tools()


def get_audit_logger() -> AuditLogger:
    """Module singleton for AuditLogger."""
    global _audit_logger
    if _audit_logger is None:
        try:
            from remy.config.settings import settings
            _audit_logger = AuditLogger(settings.AUDIT_LOG_DIR)
        except Exception:
            _audit_logger = AuditLogger(Path("data/audit_logs"))
    return _audit_logger


def reset_audit_logger() -> None:
    """Reset singleton (for testing)."""
    global _audit_logger, _critical_tools
    _audit_logger = None
    _critical_tools = None
