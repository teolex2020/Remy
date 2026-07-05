"""Tests for SandboxRunner — AST validation, venv, subprocess execution."""

from pathlib import Path

from conftest import GOOD_TOOL_CODE, GOOD_TOOL_WITH_BRAIN, BAD_TOOL_NO_EXECUTE, BAD_TOOL_SYNTAX, BAD_TOOL_NO_TESTS
from remy.sandbox.runner import validate_tool_file, run_tests, execute_tool, ensure_venv


class TestValidateToolFile:
    """AST-based validation without executing code."""

    def test_valid_tool(self, sandbox_tools_dir):
        path = sandbox_tools_dir / "calc_bmi.py"
        path.write_text(GOOD_TOOL_CODE, encoding="utf-8")

        valid, msg = validate_tool_file(path)
        assert valid is True
        assert "Valid" in msg

    def test_missing_execute(self, sandbox_tools_dir):
        path = sandbox_tools_dir / "broken.py"
        path.write_text(BAD_TOOL_NO_EXECUTE, encoding="utf-8")

        valid, msg = validate_tool_file(path)
        assert valid is False
        assert "execute()" in msg

    def test_syntax_error(self, sandbox_tools_dir):
        path = sandbox_tools_dir / "syntax_err.py"
        path.write_text(BAD_TOOL_SYNTAX, encoding="utf-8")

        valid, msg = validate_tool_file(path)
        assert valid is False
        assert "Syntax error" in msg

    def test_missing_tests(self, sandbox_tools_dir):
        path = sandbox_tools_dir / "no_tests.py"
        path.write_text(BAD_TOOL_NO_TESTS, encoding="utf-8")

        valid, msg = validate_tool_file(path)
        assert valid is False
        assert "test_" in msg

    def test_missing_constants(self, sandbox_tools_dir):
        code = '''\
TOOL_NAME = "partial"

def execute():
    return "ok"

def test_execute():
    assert execute() == "ok"
'''
        path = sandbox_tools_dir / "partial.py"
        path.write_text(code, encoding="utf-8")

        valid, msg = validate_tool_file(path)
        assert valid is False
        assert "TOOL_DESCRIPTION" in msg or "TOOL_PARAMETERS" in msg


class TestRunTests:
    """Run pytest in sandbox subprocess."""

    def test_passing_tests(self, sandbox_tools_dir):
        path = sandbox_tools_dir / "calc_bmi.py"
        path.write_text(GOOD_TOOL_CODE, encoding="utf-8")

        success, passed, failed, output = run_tests(path)
        assert success is True
        assert passed >= 1
        assert failed == 0

    def test_failing_tests(self, sandbox_tools_dir):
        code = '''\
import json

TOOL_NAME = "bad_math"
TOOL_DESCRIPTION = "Bad math"
TOOL_PARAMETERS = {}

def execute() -> str:
    return json.dumps({"result": 42})

def test_execute():
    assert 1 == 2, "intentional failure"
'''
        path = sandbox_tools_dir / "bad_math.py"
        path.write_text(code, encoding="utf-8")

        success, passed, failed, output = run_tests(path)
        assert success is False
        assert failed >= 1


class TestExecuteTool:
    """Execute sandbox tool in subprocess."""

    def test_execute_good_tool(self, sandbox_tools_dir):
        path = sandbox_tools_dir / "calc_bmi.py"
        path.write_text(GOOD_TOOL_CODE, encoding="utf-8")

        success, result = execute_tool(path, {"height_cm": 180, "weight_kg": 75})
        assert success is True
        import json
        data = json.loads(result)
        assert data["bmi"] == 23.1

    def test_execute_with_error(self, sandbox_tools_dir):
        code = '''\
import json

TOOL_NAME = "crasher"
TOOL_DESCRIPTION = "Crashes on purpose"
TOOL_PARAMETERS = {}

def execute() -> str:
    raise ValueError("boom")

def test_execute():
    pass
'''
        path = sandbox_tools_dir / "crasher.py"
        path.write_text(code, encoding="utf-8")

        success, result = execute_tool(path, {})
        assert success is False
        assert "boom" in result


class TestExecuteToolWithBrain:
    """Execute sandbox tool with brain injection."""

    def test_execute_tool_with_brain(self, sandbox_tools_dir, tmp_path):
        """Tool with 'brain' param receives CognitiveMemory instance."""
        path = sandbox_tools_dir / "scan_notes.py"
        path.write_text(GOOD_TOOL_WITH_BRAIN, encoding="utf-8")

        # Create a temp brain and store something
        from aura import Aura as CognitiveMemory, Level
        brain_path = str(tmp_path / "test_brain")
        b = CognitiveMemory(brain_path)
        b.store(content="Test note about vitamins", level=Level.DOMAIN, tags=["health"])
        b.close()

        success, result = execute_tool(path, {"tag": "health"}, brain_path=brain_path)
        assert success is True
        import json
        data = json.loads(result)
        assert data["count"] == 1
        assert "vitamins" in data["notes"][0]

    def test_execute_tool_without_brain_param(self, sandbox_tools_dir, tmp_path):
        """Regular tool (no brain param) works fine even when brain_path is passed."""
        path = sandbox_tools_dir / "calc_bmi.py"
        path.write_text(GOOD_TOOL_CODE, encoding="utf-8")

        # brain_path is passed but tool doesn't need it — should be ignored
        success, result = execute_tool(path, {"height_cm": 180, "weight_kg": 75},
                                       brain_path=str(tmp_path / "unused_brain"))
        assert success is True
        import json
        data = json.loads(result)
        assert data["bmi"] == 23.1

    def test_execute_tool_brain_missing_path(self, sandbox_tools_dir):
        """Tool requiring brain but brain_path=None → error."""
        path = sandbox_tools_dir / "scan_notes.py"
        path.write_text(GOOD_TOOL_WITH_BRAIN, encoding="utf-8")

        success, result = execute_tool(path, {"tag": "test"}, brain_path=None)
        assert success is False
        assert "brain" in result.lower()


class TestEnsureVenv:
    """Venv creation (slow — only runs once, cached after)."""

    def test_venv_exists_after_ensure(self):
        ensure_venv()
        from remy.sandbox.runner import _sandbox_python
        assert Path(_sandbox_python()).exists()
