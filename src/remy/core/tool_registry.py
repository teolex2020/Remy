"""
Tool Registry — bridges core tools + approved sandbox tools into Gemini FunctionDeclarations.

Loads approved sandbox tools from manifest at session start.
Delegates sandbox tool execution to SandboxRunner (subprocess).
"""

import logging
from pathlib import Path
import time

from google.genai import types

from remy.config.settings import settings
from remy.sandbox.manifest import SandboxManifest
from remy.sandbox.runner import execute_tool

logger = logging.getLogger(__name__)

# Type mapping from manifest JSON to Gemini Schema types
_TYPE_MAP = {
    "STRING": "STRING",
    "INTEGER": "INTEGER",
    "NUMBER": "NUMBER",
    "BOOLEAN": "BOOLEAN",
    "ARRAY": "ARRAY",
}


class ToolRegistry:
    """Manages core + sandbox tools for Gemini sessions."""

    def __init__(self, core_tools: list[types.FunctionDeclaration]):
        self._core_tools = core_tools
        self._manifest = SandboxManifest(settings.SANDBOX_DIR / "manifest.json")
        self._sandbox_names: set[str] = set()
        self._sandbox_decls_cache: list[types.FunctionDeclaration] | None = None
        # Pre-populate sandbox names so is_sandbox_tool works immediately
        for tool in self._manifest.get_approved_tools():
            self._sandbox_names.add(tool["name"])

    def invalidate_cache(self) -> None:
        """Clear cached sandbox declarations (call after approval/rejection)."""
        self._sandbox_decls_cache = None

    def get_all_declarations(self) -> list[types.FunctionDeclaration]:
        """Return core tools + approved sandbox tools as FunctionDeclarations."""
        if self._sandbox_decls_cache is None:
            self._sandbox_decls_cache = self._load_sandbox_declarations()
            logger.info("Tools loaded: %d core + %d sandbox",
                        len(self._core_tools), len(self._sandbox_decls_cache))
        return self._core_tools + self._sandbox_decls_cache

    @property
    def tool_count(self) -> int:
        """Total number of tools (core + sandbox) without triggering full reload."""
        sandbox_count = len(self._manifest.get_approved_tools())
        return len(self._core_tools) + sandbox_count

    def _load_sandbox_declarations(self) -> list[types.FunctionDeclaration]:
        """Convert approved sandbox tools to Gemini FunctionDeclarations."""
        approved = self._manifest.get_approved_tools()
        declarations = []

        for tool in approved:
            try:
                properties = {}
                params = tool.get("parameters", {})
                
                # Robustness: Handle if parameters are stored as JSON string
                if isinstance(params, str):
                    try:
                        import json
                        params = json.loads(params)
                    except Exception:
                        logger.error("Invalid parameters JSON for tool %s", tool["name"])
                        continue

                # Handle full JSON schema (properties wrapper)
                if "properties" in params and isinstance(params["properties"], dict):
                    params = params["properties"]

                for param_name, param_info in params.items():
                    # Skip non-dict items if any garbage remains
                    if not isinstance(param_info, dict):
                        continue
                        
                    schema_type = _TYPE_MAP.get(param_info.get("type", "STRING"), "STRING")
                    properties[param_name] = types.Schema(
                        type=schema_type,
                        description=param_info.get("description", ""),
                    )

                decl = types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool["description"],
                    parameters=types.Schema(
                        type="OBJECT",
                        properties=properties,
                        required=tool.get("required", []),
                    ),
                )
                declarations.append(decl)
                self._sandbox_names.add(tool["name"])
                logger.debug("Sandbox tool loaded: %s", tool["name"])
            except Exception as e:
                logger.error("Failed to load sandbox tool %s: %s", tool["name"], e)

        return declarations

    def get_tools_config(self) -> list[types.Tool]:
        """Return complete tools config: function declarations + grounding tools.

        Combines brain/sandbox FunctionDeclarations with native Gemini tools
        (Google Search grounding, code execution).
        """
        all_decls = self.get_all_declarations()
        tools = [types.Tool(function_declarations=all_decls)]

        # Google Search grounding — real-time web access
        tools.append(types.Tool(google_search=types.GoogleSearch()))

        # Code execution — calculator, data processing
        tools.append(types.Tool(code_execution=types.ToolCodeExecution()))

        return tools

    def is_sandbox_tool(self, name: str) -> bool:
        return name in self._sandbox_names

    def execute_sandbox_tool(self, name: str, args: dict) -> str:
        """Execute a sandbox tool via subprocess."""
        tool = self._manifest.get_tool(name)
        if not tool:
            return f"Sandbox tool '{name}' not found."

        tool_path = Path(settings.SANDBOX_TOOLS_DIR) / tool["file"]
        if not tool_path.exists():
            return f"Tool file missing: {tool['file']}"

        start = time.perf_counter()
        success, result = execute_tool(tool_path, args, brain_path=str(settings.AURA_BRAIN_PATH))
        duration_ms = int((time.perf_counter() - start) * 1000)

        # Update the in-memory tool entry directly so callers/tests holding the
        # manifest record see telemetry immediately, then persist via manifest helper.
        telemetry = tool.setdefault("telemetry", {})
        telemetry["call_count"] = int(telemetry.get("call_count", 0)) + 1
        telemetry["error_count"] = int(telemetry.get("error_count", 0))
        telemetry["total_ms"] = int(telemetry.get("total_ms", 0)) + max(0, duration_ms)
        if not success:
            telemetry["error_count"] += 1

        if hasattr(self._manifest, "record_telemetry"):
            self._manifest.record_telemetry(name, success=success, duration_ms=duration_ms)
        if hasattr(self._manifest, "save"):
            self._manifest.save()

        if success:
            return result
        return f"Sandbox tool error: {result}"

    @property
    def manifest(self) -> SandboxManifest:
        return self._manifest
