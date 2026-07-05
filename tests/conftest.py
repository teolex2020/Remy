"""
Shared fixtures for Remy tests.

Uses temporary directories for brain and sandbox — no side effects on real data.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest
from remy.core.agent_tools import _AuraCompat as CognitiveMemory

from remy.sandbox.manifest import SandboxManifest


def pytest_collection_modifyitems(config, items):
    """Skip E2E tests unless explicitly requested.

    E2E tests start a real uvicorn server and require process isolation
    from unit tests.  Run them separately: ``pytest tests/e2e/``
    """
    if any("e2e" in str(a) for a in config.args):
        # User explicitly asked for E2E tests — run them
        return
    e2e_skip = pytest.mark.skip(reason="E2E tests run separately: pytest tests/e2e/")
    for item in items:
        if "/e2e/" in item.nodeid or "\\e2e\\" in item.nodeid:
            item.add_marker(e2e_skip)


@pytest.fixture(autouse=True)
def _no_knowledge_sync(request, monkeypatch):
    """Prevent tests from writing to real Aura Memory (knowledge base).

    _sync_to_knowledge fires inside many brain_tools functions.
    Without this mock, test data leaks into the production knowledge store.

    Tests that need the real _sync_to_knowledge should use:
        @pytest.mark.real_sync
    """
    if "real_sync" in {m.name for m in request.node.iter_markers()}:
        return
    monkeypatch.setattr(
        "remy.core.brain_tools._sync_to_knowledge", lambda *a, **kw: False
    )


@pytest.fixture(autouse=True)
def _disable_approval_queue():
    """Disable the approval queue singleton for ALL tests.

    Without this, any execute_tool("store", tags="crypto|wallet|...") or
    browser_act on financial URLs will call request_approval_sync() which
    blocks for APPROVAL_TIMEOUT_SEC (default 120s) waiting for a Telegram
    or Web GUI reply that never comes in test environment.

    Tests that specifically test approval queue behaviour create their own
    ApprovalQueue instances and set _enabled = True explicitly.
    """
    from remy.core.approval_queue import approval_queue as _aq
    original = _aq._enabled
    _aq._enabled = False
    yield
    _aq._enabled = original


@pytest.fixture
def tmp_dir(tmp_path):
    """Temporary directory that auto-cleans."""
    return tmp_path


@pytest.fixture
def brain(tmp_path):
    """Fresh CognitiveMemory instance in temp dir."""
    b = CognitiveMemory(str(tmp_path / "brain"))
    yield b
    b.close()


@pytest.fixture
def manifest(tmp_path):
    """Fresh SandboxManifest in temp dir."""
    return SandboxManifest(tmp_path / "manifest.json")


@pytest.fixture
def sandbox_tools_dir(tmp_path):
    """Temp directory for sandbox tool files."""
    d = tmp_path / "tools"
    d.mkdir()
    return d


GOOD_TOOL_CODE = '''\
import json

TOOL_NAME = "calc_bmi"
TOOL_DESCRIPTION = "Calculate Body Mass Index"
TOOL_PARAMETERS = {
    "height_cm": {"type": "NUMBER", "description": "Height in cm"},
    "weight_kg": {"type": "NUMBER", "description": "Weight in kg"},
}
TOOL_REQUIRED = ["height_cm", "weight_kg"]
DEPENDENCIES = []

def execute(height_cm: float, weight_kg: float) -> str:
    bmi = weight_kg / (height_cm / 100) ** 2
    return json.dumps({"bmi": round(bmi, 1)})

def test_execute():
    result = json.loads(execute(180, 75))
    assert result["bmi"] == 23.1
'''

GOOD_TOOL_WITH_BRAIN = '''\
import json

TOOL_NAME = "scan_notes"
TOOL_DESCRIPTION = "Scan brain for notes matching a tag and return summary"
TOOL_PARAMETERS = {
    "tag": {"type": "STRING", "description": "Tag to search for"},
}
TOOL_REQUIRED = ["tag"]
DEPENDENCIES = []

def execute(brain, tag: str) -> str:
    results = brain.search(query="", tags=[tag], limit=10)
    notes = [r.content for r in results]
    return json.dumps({"count": len(notes), "notes": notes})

def test_execute():
    class MockBrain:
        def search(self, query="", tags=None, limit=10):
            return []
    result = json.loads(execute(MockBrain(), "test"))
    assert result["count"] == 0
'''

BAD_TOOL_NO_EXECUTE = '''\
TOOL_NAME = "broken"
TOOL_DESCRIPTION = "Missing execute"
TOOL_PARAMETERS = {}

def test_something():
    pass
'''

BAD_TOOL_SYNTAX = '''\
TOOL_NAME = "syntax_err"
def execute(
    # missing closing paren
'''

BAD_TOOL_NO_TESTS = '''\
TOOL_NAME = "no_tests"
TOOL_DESCRIPTION = "No tests"
TOOL_PARAMETERS = {}

def execute() -> str:
    return "ok"
'''
