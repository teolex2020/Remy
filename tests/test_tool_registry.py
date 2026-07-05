"""Tests for ToolRegistry — bridges core + sandbox tools to Gemini FunctionDeclarations."""

import json
from pathlib import Path
from unittest.mock import patch

from google.genai import types

from conftest import GOOD_TOOL_CODE, GOOD_TOOL_WITH_BRAIN
from remy.core.tool_registry import ToolRegistry
from remy.sandbox.manifest import SandboxManifest


def _make_core_tools():
    """Minimal core tool list for testing."""
    return [
        types.FunctionDeclaration(
            name="recall",
            description="Recall memories",
            parameters=types.Schema(type="OBJECT", properties={
                "query": types.Schema(type="STRING", description="query"),
            }, required=["query"]),
        ),
    ]


class TestRegistryBasic:

    def test_core_tools_only(self, tmp_path):
        manifest_path = tmp_path / "manifest.json"
        core = _make_core_tools()

        with patch("remy.core.tool_registry.settings") as mock_settings:
            mock_settings.SANDBOX_DIR = tmp_path
            mock_settings.SANDBOX_TOOLS_DIR = tmp_path / "tools"
            reg = ToolRegistry(core)

        decls = reg.get_all_declarations()
        assert len(decls) == 1
        assert decls[0].name == "recall"

    def test_is_sandbox_tool_false_for_core(self, tmp_path):
        with patch("remy.core.tool_registry.settings") as mock_settings:
            mock_settings.SANDBOX_DIR = tmp_path
            mock_settings.SANDBOX_TOOLS_DIR = tmp_path / "tools"
            reg = ToolRegistry(_make_core_tools())

        assert not reg.is_sandbox_tool("recall")
        assert not reg.is_sandbox_tool("nonexistent")


class TestRegistryWithSandbox:

    def _setup_approved_tool(self, tmp_path):
        """Create an approved tool in manifest + file on disk."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir(exist_ok=True)
        tool_file = tools_dir / "calc_bmi.py"
        tool_file.write_text(GOOD_TOOL_CODE, encoding="utf-8")

        manifest = SandboxManifest(tmp_path / "manifest.json")
        manifest.add_tool(
            name="calc_bmi", file="calc_bmi.py",
            description="Calculate BMI",
            parameters={"height_cm": {"type": "NUMBER"}, "weight_kg": {"type": "NUMBER"}},
            required=["height_cm", "weight_kg"],
        )
        manifest.set_test_result("calc_bmi", passed=1, failed=0)
        manifest.submit_for_approval("calc_bmi")
        manifest.update_status("calc_bmi", "approved")
        return manifest

    def test_loads_approved_sandbox_tools(self, tmp_path):
        self._setup_approved_tool(tmp_path)
        core = _make_core_tools()

        with patch("remy.core.tool_registry.settings") as mock_settings:
            mock_settings.SANDBOX_DIR = tmp_path
            mock_settings.SANDBOX_TOOLS_DIR = tmp_path / "tools"
            reg = ToolRegistry(core)

        decls = reg.get_all_declarations()
        names = [d.name for d in decls]
        assert "recall" in names
        assert "calc_bmi" in names
        assert len(decls) == 2

    def test_is_sandbox_tool_true_for_approved(self, tmp_path):
        self._setup_approved_tool(tmp_path)

        with patch("remy.core.tool_registry.settings") as mock_settings:
            mock_settings.SANDBOX_DIR = tmp_path
            mock_settings.SANDBOX_TOOLS_DIR = tmp_path / "tools"
            reg = ToolRegistry(_make_core_tools())

        assert reg.is_sandbox_tool("calc_bmi")
        assert not reg.is_sandbox_tool("recall")

    def test_execute_sandbox_tool(self, tmp_path):
        self._setup_approved_tool(tmp_path)

        with patch("remy.core.tool_registry.settings") as mock_settings:
            mock_settings.SANDBOX_DIR = tmp_path
            mock_settings.SANDBOX_TOOLS_DIR = tmp_path / "tools"
            reg = ToolRegistry(_make_core_tools())

            result = reg.execute_sandbox_tool("calc_bmi", {"height_cm": 180, "weight_kg": 75})
            data = json.loads(result)
            assert data["bmi"] == 23.1

    def test_execute_missing_tool(self, tmp_path):
        with patch("remy.core.tool_registry.settings") as mock_settings:
            mock_settings.SANDBOX_DIR = tmp_path
            mock_settings.SANDBOX_TOOLS_DIR = tmp_path / "tools"
            reg = ToolRegistry(_make_core_tools())

        result = reg.execute_sandbox_tool("nonexistent", {})
        assert "not found" in result

    def test_pending_tools_not_loaded(self, tmp_path):
        """Tools that haven't been approved should not appear in declarations."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir(exist_ok=True)
        (tools_dir / "pending_tool.py").write_text(GOOD_TOOL_CODE, encoding="utf-8")

        manifest = SandboxManifest(tmp_path / "manifest.json")
        manifest.add_tool("pending_tool", "pending_tool.py", "Pending", {}, [])
        manifest.set_test_result("pending_tool", passed=1, failed=0)
        manifest.submit_for_approval("pending_tool")
        # Not approved!

        with patch("remy.core.tool_registry.settings") as mock_settings:
            mock_settings.SANDBOX_DIR = tmp_path
            mock_settings.SANDBOX_TOOLS_DIR = tools_dir
            reg = ToolRegistry(_make_core_tools())

        decls = reg.get_all_declarations()
        names = [d.name for d in decls]
        assert "pending_tool" not in names


class TestToolsConfig:

    def test_includes_function_declarations(self, tmp_path):
        """get_tools_config() includes brain tool FunctionDeclarations."""
        with patch("remy.core.tool_registry.settings") as mock_settings:
            mock_settings.SANDBOX_DIR = tmp_path
            mock_settings.SANDBOX_TOOLS_DIR = tmp_path / "tools"
            reg = ToolRegistry(_make_core_tools())

        config = reg.get_tools_config()
        # First tool should have function_declarations
        assert config[0].function_declarations is not None
        assert len(config[0].function_declarations) >= 1

    def test_includes_google_search(self, tmp_path):
        """get_tools_config() includes Google Search grounding tool."""
        with patch("remy.core.tool_registry.settings") as mock_settings:
            mock_settings.SANDBOX_DIR = tmp_path
            mock_settings.SANDBOX_TOOLS_DIR = tmp_path / "tools"
            reg = ToolRegistry(_make_core_tools())

        config = reg.get_tools_config()
        has_search = any(t.google_search is not None for t in config)
        assert has_search

    def test_includes_code_execution(self, tmp_path):
        """get_tools_config() includes code execution tool."""
        with patch("remy.core.tool_registry.settings") as mock_settings:
            mock_settings.SANDBOX_DIR = tmp_path
            mock_settings.SANDBOX_TOOLS_DIR = tmp_path / "tools"
            reg = ToolRegistry(_make_core_tools())

        config = reg.get_tools_config()
        has_code_exec = any(t.code_execution is not None for t in config)
        assert has_code_exec

    def test_config_has_three_tools(self, tmp_path):
        """get_tools_config() returns exactly 3 Tool objects."""
        with patch("remy.core.tool_registry.settings") as mock_settings:
            mock_settings.SANDBOX_DIR = tmp_path
            mock_settings.SANDBOX_TOOLS_DIR = tmp_path / "tools"
            reg = ToolRegistry(_make_core_tools())

        config = reg.get_tools_config()
        assert len(config) == 3


class TestBrainAccessInSandbox:
    """Tests for sandbox tools with brain access."""

    def _setup_approved_brain_tool(self, tmp_path):
        """Create an approved brain-aware tool."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir(exist_ok=True)
        tool_file = tools_dir / "scan_notes.py"
        tool_file.write_text(GOOD_TOOL_WITH_BRAIN, encoding="utf-8")

        manifest = SandboxManifest(tmp_path / "manifest.json")
        manifest.add_tool(
            name="scan_notes", file="scan_notes.py",
            description="Scan brain for notes",
            parameters={"tag": {"type": "STRING"}},
            required=["tag"],
        )
        manifest.set_test_result("scan_notes", passed=1, failed=0)
        manifest.submit_for_approval("scan_notes")
        manifest.update_status("scan_notes", "approved")
        return manifest

    def test_execute_sandbox_tool_with_brain(self, tmp_path):
        """Brain-aware sandbox tool gets CognitiveMemory via registry."""
        self._setup_approved_brain_tool(tmp_path)

        # Create a brain with data
        from aura import Aura as CognitiveMemory, Level
        brain_path = tmp_path / "brain"
        b = CognitiveMemory(str(brain_path))
        b.store(content="Note about sleep quality", level=Level.DOMAIN, tags=["health"])
        b.close()

        with patch("remy.core.tool_registry.settings") as mock_settings:
            mock_settings.SANDBOX_DIR = tmp_path
            mock_settings.SANDBOX_TOOLS_DIR = tmp_path / "tools"
            mock_settings.AURA_BRAIN_PATH = brain_path
            reg = ToolRegistry(_make_core_tools())

            result = reg.execute_sandbox_tool("scan_notes", {"tag": "health"})
            data = json.loads(result)
            assert data["count"] == 1
            assert "sleep" in data["notes"][0]

    def test_execute_sandbox_tool_no_brain_still_works(self, tmp_path):
        """Regular tool (no brain param) still works with brain_path passed."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir(exist_ok=True)
        (tools_dir / "calc_bmi.py").write_text(GOOD_TOOL_CODE, encoding="utf-8")

        manifest = SandboxManifest(tmp_path / "manifest.json")
        manifest.add_tool("calc_bmi", "calc_bmi.py", "BMI",
                          {"height_cm": {"type": "NUMBER"}, "weight_kg": {"type": "NUMBER"}},
                          ["height_cm", "weight_kg"])
        manifest.set_test_result("calc_bmi", passed=1, failed=0)
        manifest.submit_for_approval("calc_bmi")
        manifest.update_status("calc_bmi", "approved")

        with patch("remy.core.tool_registry.settings") as mock_settings:
            mock_settings.SANDBOX_DIR = tmp_path
            mock_settings.SANDBOX_TOOLS_DIR = tools_dir
            mock_settings.AURA_BRAIN_PATH = tmp_path / "brain"
            reg = ToolRegistry(_make_core_tools())

            result = reg.execute_sandbox_tool("calc_bmi", {"height_cm": 180, "weight_kg": 75})
            data = json.loads(result)
            assert data["bmi"] == 23.1
