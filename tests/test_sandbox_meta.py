"""Tests for sandbox meta-tools (create/test/list) via execute_tool dispatch."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from conftest import GOOD_TOOL_CODE, BAD_TOOL_NO_EXECUTE


@pytest.fixture
def sandbox_env(tmp_path):
    """Set up isolated sandbox environment and provide execute_tool."""
    from aura import Aura as CognitiveMemory

    brain = CognitiveMemory(str(tmp_path / "brain"))
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()

    with patch("remy.core.brain_tools.brain", brain), \
         patch("remy.core.brain_tools._registry", None), \
         patch("remy.core.tool_registry.settings") as mock_reg_settings, \
         patch("remy.core.brain_tools.settings") as mock_bt_settings:

        for ms in [mock_reg_settings, mock_bt_settings]:
            ms.SANDBOX_DIR = sandbox_dir
            ms.SANDBOX_TOOLS_DIR = tools_dir
            ms.AUTONOMY_AUTO_APPROVE_SANDBOX = False

        from remy.core.brain_tools import execute_tool
        yield {
            "execute": execute_tool,
            "brain": brain,
            "tools_dir": tools_dir,
            "sandbox_dir": sandbox_dir,
        }

    brain.close()


class TestSandboxCreate:

    def test_create_valid_tool(self, sandbox_env):
        result = sandbox_env["execute"]("sandbox_create_tool", {
            "name": "calc_bmi",
            "code": GOOD_TOOL_CODE,
        })
        data = json.loads(result)
        assert data["created"] is True
        assert data["name"] == "calc_bmi"
        assert data["status"] == "draft"

        # File written to disk
        assert (sandbox_env["tools_dir"] / "calc_bmi.py").exists()

    def test_create_invalid_tool_rejected(self, sandbox_env):
        result = sandbox_env["execute"]("sandbox_create_tool", {
            "name": "broken",
            "code": BAD_TOOL_NO_EXECUTE,
        })
        data = json.loads(result)
        assert data["created"] is False
        assert "execute()" in data["error"]

        # File should be cleaned up
        assert not (sandbox_env["tools_dir"] / "broken.py").exists()

    def test_create_stores_in_brain(self, sandbox_env):
        sandbox_env["execute"]("sandbox_create_tool", {
            "name": "calc_bmi",
            "code": GOOD_TOOL_CODE,
        })
        # Check brain learned about the creation
        result = sandbox_env["brain"].recall("sandbox calc_bmi", token_budget=1024)
        assert result and "calc_bmi" in result


class TestSandboxTest:

    def test_test_passing_tool(self, sandbox_env):
        # First create
        sandbox_env["execute"]("sandbox_create_tool", {
            "name": "calc_bmi",
            "code": GOOD_TOOL_CODE,
        })
        # Then test
        result = sandbox_env["execute"]("sandbox_test_tool", {"name": "calc_bmi"})
        data = json.loads(result)
        assert data["tested"] is True
        assert data["passed"] >= 1
        assert data["failed"] == 0
        assert data["status"] == "pending"

    def test_test_nonexistent_tool(self, sandbox_env):
        result = sandbox_env["execute"]("sandbox_test_tool", {"name": "ghost"})
        data = json.loads(result)
        assert data["tested"] is False

    def test_test_failing_tool(self, sandbox_env):
        bad_code = '''\
import json

TOOL_NAME = "bad"
TOOL_DESCRIPTION = "Fails tests"
TOOL_PARAMETERS = {}
TOOL_REQUIRED = []

def execute() -> str:
    return "ok"

def test_execute():
    assert 1 == 2
'''
        sandbox_env["execute"]("sandbox_create_tool", {"name": "bad", "code": bad_code})
        result = sandbox_env["execute"]("sandbox_test_tool", {"name": "bad"})
        data = json.loads(result)
        assert data["tested"] is False
        assert data["failed"] >= 1


class TestSandboxList:

    def test_list_empty(self, sandbox_env):
        result = sandbox_env["execute"]("sandbox_list_tools", {})
        assert "No sandbox tools" in result

    def test_list_after_create(self, sandbox_env):
        sandbox_env["execute"]("sandbox_create_tool", {
            "name": "calc_bmi",
            "code": GOOD_TOOL_CODE,
        })
        result = sandbox_env["execute"]("sandbox_list_tools", {})
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["name"] == "calc_bmi"
        assert data[0]["status"] == "draft"


class TestFullWorkflow:
    """End-to-end: create → test → list → (approve manually) → verify."""

    def test_create_test_list_flow(self, sandbox_env):
        ex = sandbox_env["execute"]

        # 1. Create
        r = json.loads(ex("sandbox_create_tool", {"name": "calc_bmi", "code": GOOD_TOOL_CODE}))
        assert r["created"] is True

        # 2. Test
        r = json.loads(ex("sandbox_test_tool", {"name": "calc_bmi"}))
        assert r["tested"] is True
        assert r["status"] == "pending"

        # 3. List — should show pending
        r = json.loads(ex("sandbox_list_tools", {}))
        assert r[0]["status"] == "pending"

        # 4. Manual approval (simulates CLI — uses same manifest file)
        #    The registry's manifest is an in-memory object, so we update it
        #    via the file (like the real CLI does) and reload
        from remy.sandbox.manifest import SandboxManifest
        cli_manifest = SandboxManifest(sandbox_env["sandbox_dir"] / "manifest.json")
        assert cli_manifest.update_status("calc_bmi", "approved")

        # 5. Verify: the registry's manifest reads from same file,
        #    so we need to reload it (like a new session would)
        from remy.core.brain_tools import get_registry
        reg_manifest = get_registry().manifest
        reg_manifest._data = reg_manifest._load()
        r = json.loads(ex("sandbox_list_tools", {}))
        assert r[0]["status"] == "approved"
