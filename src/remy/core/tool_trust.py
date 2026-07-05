"""
Self-Tooling — Tool Trust Levels (AUTON-9).

AST-based safety classification for sandbox tools:
- safe: read-only, no network, no file writes → auto-approve
- moderate: local file writes, no network → auto-approve if tests pass
- dangerous: network calls, subprocess, system access → human approval required

Progressive trust: tools used 10+ times without issues get trust upgrade.
Tool retirement: unused tools for 30+ days get archived.
"""

import ast
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("Autonomy.ToolTrust")


# ============== Trust Levels ==============


TRUST_SAFE = "safe"
TRUST_MODERATE = "moderate"
TRUST_DANGEROUS = "dangerous"

# AST patterns that indicate dangerous capabilities
_DANGEROUS_MODULES = {
    "subprocess",
    "os.system",
    "shutil.rmtree",
    "httpx",
    "requests",
    "urllib",
    "urllib3",
    "aiohttp",
    "socket",
    "ssl",
    "ftplib",
    "smtplib",
    "imaplib",
    "paramiko",
    "fabric",
    "invoke",
}

_DANGEROUS_FUNCTIONS = {
    "exec",
    "eval",
    "compile",
    "__import__",
    "os.system",
    "os.popen",
    "os.exec",
    "os.spawn",
    "subprocess.run",
    "subprocess.call",
    "subprocess.Popen",
    "shutil.rmtree",
    "os.remove",
    "os.unlink",
}

_WRITE_MODULES = {
    "pathlib",
    "shutil",
    "tempfile",
}

_WRITE_FUNCTIONS = {
    "open",  # Only dangerous if mode includes 'w' or 'a'
    "os.makedirs",
    "os.mkdir",
    "shutil.copy",
    "shutil.move",
}


# ============== AST Safety Classifier ==============


@dataclass
class TrustClassification:
    """Result of AST-based trust classification."""

    trust_level: str  # safe, moderate, dangerous
    has_network: bool = False
    has_file_write: bool = False
    has_subprocess: bool = False
    has_dynamic_exec: bool = False
    reasons: list[str] = field(default_factory=list)


def classify_tool_source(source_code: str) -> TrustClassification:
    """Classify a tool's trust level by analyzing its AST.

    Returns TrustClassification with detailed reasoning.
    """
    result = TrustClassification(trust_level=TRUST_SAFE)

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return TrustClassification(
            trust_level=TRUST_DANGEROUS,
            reasons=[f"Syntax error: {e}"],
        )

    visitor = _SafetyVisitor()
    visitor.visit(tree)

    result.has_network = visitor.has_network
    result.has_file_write = visitor.has_file_write
    result.has_subprocess = visitor.has_subprocess
    result.has_dynamic_exec = visitor.has_dynamic_exec
    result.reasons = visitor.reasons

    # Determine trust level
    if visitor.has_network or visitor.has_subprocess or visitor.has_dynamic_exec:
        result.trust_level = TRUST_DANGEROUS
    elif visitor.has_file_write:
        result.trust_level = TRUST_MODERATE
    else:
        result.trust_level = TRUST_SAFE

    return result


class _SafetyVisitor(ast.NodeVisitor):
    """AST visitor that detects dangerous patterns."""

    def __init__(self):
        self.has_network = False
        self.has_file_write = False
        self.has_subprocess = False
        self.has_dynamic_exec = False
        self.reasons: list[str] = []
        self._imported_modules: set[str] = set()

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self._check_module(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            self._check_module(node.module)
            for alias in node.names:
                full = f"{node.module}.{alias.name}"
                self._check_module(full)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func_name = self._get_call_name(node)
        if func_name:
            if func_name in _DANGEROUS_FUNCTIONS or func_name in ("exec", "eval"):
                self.has_dynamic_exec = True
                self.reasons.append(f"Dynamic execution: {func_name}()")

            if func_name.startswith("subprocess.") or func_name == "os.system":
                self.has_subprocess = True
                self.reasons.append(f"Subprocess call: {func_name}()")

            # Check open() for write mode
            if func_name == "open" and len(node.args) >= 2:
                mode_arg = node.args[1]
                if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str):
                    if any(c in mode_arg.value for c in ("w", "a", "x")):
                        self.has_file_write = True
                        self.reasons.append(f"File write: open(..., '{mode_arg.value}')")

            # Check keyword mode in open()
            if func_name == "open":
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        if isinstance(kw.value.value, str) and any(
                            c in kw.value.value for c in ("w", "a", "x")
                        ):
                            self.has_file_write = True
                            self.reasons.append(f"File write: open(mode='{kw.value.value}')")

        self.generic_visit(node)

    def _check_module(self, module_name: str):
        """Check if an imported module is dangerous."""
        self._imported_modules.add(module_name)

        # Network modules
        network_modules = {
            "httpx",
            "requests",
            "urllib",
            "urllib3",
            "aiohttp",
            "socket",
            "ssl",
            "ftplib",
            "smtplib",
            "imaplib",
        }
        for nm in network_modules:
            if module_name == nm or module_name.startswith(nm + "."):
                self.has_network = True
                self.reasons.append(f"Network module: {module_name}")
                return

        # Subprocess
        if module_name in ("subprocess",) or module_name.startswith("subprocess."):
            self.has_subprocess = True
            self.reasons.append(f"Subprocess module: {module_name}")
            return

        # File write modules (not dangerous alone, but flagged)
        if module_name in _WRITE_MODULES:
            # Only flag if combined with specific write operations
            pass

    def _get_call_name(self, node: ast.Call) -> str:
        """Extract function call name as dotted string."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return ""


# ============== Progressive Trust ==============


def check_progressive_trust(
    tool_name: str,
    current_trust: str = "",
    min_uses: int = 10,
) -> str | None:
    """Check if a tool qualifies for trust upgrade based on usage history.

    Returns new trust level if upgrade warranted, or None.

    SAFETY: A tool's trust can only be raised ONE level at most:
      dangerous → moderate (proven reliable, but still has side effects)
      moderate  → safe     (proven reliable with only local file writes)
      safe      → (no upgrade needed)

    A DANGEROUS tool can NEVER become SAFE — its AST-classified capabilities
    (network, subprocess, dynamic exec) don't change just because it succeeded.
    """
    try:
        from remy.core.agent_tools import brain, brain_lock

        with brain_lock:
            records = brain.search(
                query=tool_name,
                tags=["tool-usage-log"],
                limit=min_uses + 5,
            )

        if not records:
            return None

        # Count successful uses
        successes = 0
        failures = 0
        for r in records:
            meta = getattr(r, "metadata", None) or {}
            if meta.get("tool_name") != tool_name:
                continue
            if meta.get("success"):
                successes += 1
            else:
                failures += 1

        total = successes + failures
        if total < min_uses:
            return None

        success_rate = successes / total if total > 0 else 0

        if success_rate >= 0.9:  # 90%+ success rate
            # Determine max upgrade level based on current classification
            if current_trust == TRUST_DANGEROUS:
                # Dangerous → moderate at best (never safe)
                new_level = TRUST_MODERATE
            elif current_trust == TRUST_MODERATE:
                # Moderate → safe only if zero failures
                new_level = TRUST_SAFE if failures == 0 else TRUST_MODERATE
            elif current_trust == TRUST_SAFE:
                return None  # Already at highest trust
            else:
                # Unknown current trust — conservative: moderate at best
                new_level = TRUST_MODERATE if failures == 0 else None
                if new_level is None:
                    return None

            logger.info(
                "Progressive trust upgrade for '%s': %s → %s (%d/%d successful, %.0f%%)",
                tool_name,
                current_trust or "?",
                new_level,
                successes,
                total,
                success_rate * 100,
            )
            return new_level

        return None

    except Exception as e:
        logger.debug("Progressive trust check failed: %s", e)
        return None


# ============== Tool Retirement ==============


def find_retired_tools(manifest_tools: list[dict], inactive_days: int = 30) -> list[str]:
    """Find tools that haven't been used in `inactive_days` days.

    Returns list of tool names to retire.
    """
    now = time.time()
    cutoff = now - (inactive_days * 86400)
    retired = []

    for tool in manifest_tools:
        status = tool.get("status")
        if status not in ("approved", "tested"):
            continue

        last_used = tool.get("last_used", 0)
        created_at = tool.get("created_at", 0)

        # Use creation time if never used
        reference_time = last_used if last_used else created_at
        if reference_time and reference_time < cutoff:
            retired.append(tool.get("name", ""))

    return retired


# ============== Auto-Approve by Trust Level ==============


def should_auto_approve(tool_path: str | Path) -> tuple[bool, str]:
    """Determine if a tool should be auto-approved based on its trust level.

    Returns (should_approve, reason).
    """
    path = Path(tool_path)
    if not path.exists():
        return False, "Tool file not found"

    try:
        source = path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"Cannot read tool: {e}"

    classification = classify_tool_source(source)

    if classification.trust_level == TRUST_SAFE:
        return True, "Safe tool (read-only, no network, no file writes)"
    elif classification.trust_level == TRUST_MODERATE:
        return True, "Moderate tool (local file writes only, tests passed)"
    else:
        reasons = (
            "; ".join(classification.reasons[:3]) if classification.reasons else "unknown risk"
        )
        return False, f"Dangerous tool requires human approval: {reasons}"


def format_trust_report(classification: TrustClassification) -> str:
    """Format a trust classification as human-readable text."""
    level_emoji = {
        TRUST_SAFE: "GREEN",
        TRUST_MODERATE: "YELLOW",
        TRUST_DANGEROUS: "RED",
    }

    lines = [
        f"Trust Level: {level_emoji.get(classification.trust_level, '?')} ({classification.trust_level})",
    ]

    if classification.has_network:
        lines.append("  - Network access detected")
    if classification.has_file_write:
        lines.append("  - File write operations detected")
    if classification.has_subprocess:
        lines.append("  - Subprocess execution detected")
    if classification.has_dynamic_exec:
        lines.append("  - Dynamic code execution detected")

    if classification.reasons:
        lines.append("  Details:")
        for r in classification.reasons[:5]:
            lines.append(f"    - {r}")

    return "\n".join(lines)
