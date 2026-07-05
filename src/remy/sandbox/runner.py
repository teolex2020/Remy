"""
Sandbox Runner — isolated execution of agent-created tools.

All sandbox code runs in a subprocess with its own venv.
The main process NEVER imports sandbox tool code directly.
"""

import ast
import json
import logging
import subprocess
import sys
import textwrap
from pathlib import Path

from remy.config.settings import settings

logger = logging.getLogger(__name__)

SANDBOX_VENV = settings.SANDBOX_DIR / "venv"
TOOL_REQUIRED_CONSTANTS = {"TOOL_NAME", "TOOL_DESCRIPTION", "TOOL_PARAMETERS"}


def _sandbox_python() -> str:
    """Return path to sandbox venv Python executable."""
    if sys.platform == "win32":
        return str(SANDBOX_VENV / "Scripts" / "python.exe")
    return str(SANDBOX_VENV / "bin" / "python")


def ensure_venv():
    """Create sandbox venv if it doesn't exist."""
    if Path(_sandbox_python()).exists():
        return
    logger.info("Creating sandbox venv at %s", SANDBOX_VENV)
    SANDBOX_VENV.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "venv", str(SANDBOX_VENV)],
        check=True,
        timeout=120,
    )
    # Install pytest in sandbox
    subprocess.run(
        [_sandbox_python(), "-m", "pip", "install", "pytest", "-q"],
        check=True,
        timeout=120,
        capture_output=True,
    )
    # Install aura for brain access in sandbox tools: prefer the vendored
    # wheel (offline/dev), otherwise fall back to the published PyPI package
    # so a fresh checkout without vendor/ still gets a working sandbox.
    wheel_path = settings.AURA_WHEEL_PATH
    aura_source = str(wheel_path) if wheel_path.exists() else "aura-memory"
    subprocess.run(
        [_sandbox_python(), "-m", "pip", "install", aura_source, "-q"],
        check=False,  # Don't fail if aura install has issues
        timeout=120,
        capture_output=True,
    )
    logger.info(
        "aura installed in sandbox venv from %s",
        "vendored wheel" if wheel_path.exists() else "PyPI (aura-memory)",
    )
    logger.info("Sandbox venv ready")


def install_dependencies(deps: list[str]) -> tuple[bool, str]:
    """Install dependencies into sandbox venv. Returns (success, output)."""
    if not deps:
        return True, "No dependencies to install."
    ensure_venv()
    result = subprocess.run(
        [_sandbox_python(), "-m", "pip", "install"] + deps + ["-q"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        return False, result.stderr[:500]
    return True, f"Installed: {', '.join(deps)}"


def validate_tool_file(path: Path) -> tuple[bool, str]:
    """Validate tool file structure via AST (no execution).

    Checks:
    - File parses as valid Python
    - TOOL_NAME, TOOL_DESCRIPTION, TOOL_PARAMETERS constants exist
    - execute() function exists
    - No dangerous operations that could harm the host system
    """
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    found_constants = set()
    found_execute = False
    found_test = False

    # Dangerous imports/calls that could modify the host system
    _DANGEROUS_IMPORTS = {"os", "shutil", "subprocess", "sys", "importlib", "ctypes", "socket"}
    _DANGEROUS_BUILTINS = {"exec", "eval", "compile", "__import__", "open"}
    _DANGEROUS_PATTERNS: list[str] = []

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _DANGEROUS_IMPORTS:
                    _DANGEROUS_PATTERNS.append(f"dangerous import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in _DANGEROUS_IMPORTS:
                    _DANGEROUS_PATTERNS.append(f"dangerous import: from {node.module}")

        # Check dangerous builtin calls
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _DANGEROUS_BUILTINS:
                # open() for write modes is dangerous; read-only is ok
                if func.id == "open":
                    # Check if mode arg suggests writing
                    mode_arg = None
                    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                        mode_arg = node.args[1].value
                    for kw in node.keywords:
                        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                            mode_arg = kw.value.value
                    if mode_arg and any(c in str(mode_arg) for c in ("w", "a", "x")):
                        _DANGEROUS_PATTERNS.append(f"file write: open(..., '{mode_arg}')")
                else:
                    _DANGEROUS_PATTERNS.append(f"dangerous builtin: {func.id}()")

    if _DANGEROUS_PATTERNS:
        return False, "Security violation — sandbox tools cannot modify the host system: " + "; ".join(_DANGEROUS_PATTERNS[:3])

    for node in ast.iter_child_nodes(tree):
        # Check module-level assignments
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in TOOL_REQUIRED_CONSTANTS:
                    found_constants.add(target.id)
        # Check functions
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "execute":
                found_execute = True
            elif node.name.startswith("test_"):
                found_test = True

    missing = TOOL_REQUIRED_CONSTANTS - found_constants
    errors = []
    if missing:
        errors.append(f"Missing constants: {', '.join(sorted(missing))}")
    if not found_execute:
        errors.append("Missing execute() function")
    if not found_test:
        errors.append("Missing test_*() function(s)")

    if errors:
        return False, "; ".join(errors)
    return True, "Valid tool file"


_PYTEST_SUCCESS_MARKERS = ["passed", "PASSED", "test result: ok"]
_PYTEST_FAILURE_MARKERS = [
    "FAILED",
    " ERROR",
    "AssertionError",
    "Traceback (most recent call last)",
    "errors during collection",
]


def _world_fact_from_pytest(returncode: int, stdout: str, stderr: str) -> str:
    """3-state world fact for a pytest run, via the SDK executable judge.

    Carries the proven discipline: the exit code alone is not trusted (pytest
    exits 5 when NO tests were collected — that is `inconclusive`, not a failure;
    and a run can fail in stderr while stdout looks clean). The fact is read from
    what pytest actually says. Fail-soft to a conservative label on older wheels.
    """
    # pytest exit code 5 (and its "no tests ran" / "collected 0 items" banner)
    # means nothing was exercised — that is `inconclusive`, NOT a failure. This
    # is pytest-specific knowledge the generic SDK judge cannot infer from a
    # non-zero exit, so normalize it here before handing off.
    combined = (stdout + stderr).lower()
    no_tests_ran = (
        returncode == 5
        or "no tests ran" in combined
        or "collected 0 items" in combined
    )
    has_failure = any(m.lower() in combined for m in _PYTEST_FAILURE_MARKERS)
    if no_tests_ran and not has_failure:
        return "inconclusive"

    try:
        from aura import world_fact_from_output

        # Hand the judge an exit code it can trust: keep 0, otherwise 1 (a real
        # failure). The pytest-only "nothing collected" case was handled above.
        normalized_exit = 0 if returncode == 0 else 1
        return world_fact_from_output(
            exit_code=normalized_exit,
            stdout=stdout,
            stderr=stderr,
            success_markers=_PYTEST_SUCCESS_MARKERS,
            failure_markers=_PYTEST_FAILURE_MARKERS,
        )
    except Exception:
        if returncode == 0:
            return "supports"
        # Non-zero with no observable failure marker (e.g. pytest exit 5,
        # nothing collected) is treated as inconclusive, not a false refutation.
        if "failed" in (stdout + stderr).lower() or "error" in (stdout + stderr).lower():
            return "refutes"
        return "inconclusive"


def run_tests(tool_path: Path) -> tuple[bool, int, int, str]:
    """Run tests in sandbox subprocess. Returns (success, passed, failed, output)."""
    success, passed, failed, output, _fact = run_tests_world_fact(tool_path)
    return success, passed, failed, output


def run_tests_world_fact(tool_path: Path) -> tuple[bool, int, int, str, str]:
    """Like `run_tests` but also returns a 3-state world fact.

    Returns `(success, passed, failed, output, world_fact)` where `world_fact` is
    "supports" | "refutes" | "inconclusive". `inconclusive` means the tests RAN
    but produced no observable pass/fail (e.g. no tests were collected) — the
    caller should NOT treat that as a validated tool, leaving any evidence debt
    open rather than learning a false positive.
    """
    ensure_venv()
    result = subprocess.run(
        [
            _sandbox_python(),
            "-m",
            "pytest",
            tool_path.name,
            "-v",
            "--tb=short",
            "--rootdir",
            str(tool_path.parent),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        cwd=str(tool_path.parent),
    )
    output = result.stdout + result.stderr

    # Parse pytest output for pass/fail counts
    passed = output.count(" PASSED")
    failed = output.count(" FAILED") + output.count(" ERROR")

    success = result.returncode == 0
    world_fact = _world_fact_from_pytest(result.returncode, result.stdout, result.stderr)
    return success, passed, failed, output[:1000], world_fact


def execute_tool(tool_path: Path, args: dict, brain_path: str | None = None) -> tuple[bool, str]:
    """Execute a sandbox tool in subprocess. Returns (success, result_json).

    If brain_path is provided AND tool's execute() accepts a 'brain' parameter,
    CognitiveMemory is opened and passed automatically.
    """
    ensure_venv()

    # Allowed write roots: only sandbox data dir and system temp
    import tempfile
    _allowed_write_roots = [
        str(tool_path.parent),           # sandbox tools dir itself
        str(settings.SANDBOX_DIR),       # full sandbox dir
        tempfile.gettempdir(),           # system temp
    ]
    _blocked_roots = [
        str(settings.BASE_DIR / "src"),  # agent core — absolutely forbidden
    ]

    # Build a runner script that imports tool, optionally injects brain
    # Filesystem guard is injected first — intercepts open() and blocks writes outside allowed zones
    runner_code = textwrap.dedent(f"""\
        import sys, json, inspect, asyncio
        import builtins as _builtins
        import pathlib as _pathlib
        import os as _os

        _ALLOWED_WRITE_ROOTS = {_allowed_write_roots!r}
        _BLOCKED_ROOTS = {_blocked_roots!r}

        def _check_path_write(path_str):
            p = _os.path.abspath(str(path_str))
            for blocked in _BLOCKED_ROOTS:
                if p.startswith(_os.path.abspath(blocked)):
                    raise PermissionError(
                        f"Sandbox security: write access to agent core is forbidden: {{p}}"
                    )
            for allowed in _ALLOWED_WRITE_ROOTS:
                if p.startswith(_os.path.abspath(allowed)):
                    return  # ok
            raise PermissionError(
                f"Sandbox security: write outside allowed zone is forbidden: {{p}}"
            )

        _original_open = _builtins.open
        def _safe_open(file, mode="r", *args, **kwargs):
            if any(c in str(mode) for c in ("w", "a", "x")):
                _check_path_write(file)
            return _original_open(file, mode, *args, **kwargs)
        _builtins.open = _safe_open

        sys.path.insert(0, {str(tool_path.parent)!r})
        import {tool_path.stem} as tool

        brain = None
        try:
            sig = inspect.signature(tool.execute)
            needs_brain = "brain" in sig.parameters

            if needs_brain:
                brain_path = {brain_path!r}
                if brain_path:
                    from aura import Aura as CognitiveMemory
                    brain = CognitiveMemory(brain_path)
                else:
                    print(json.dumps({{"ok": False, "error": "Tool requires brain but no brain_path provided"}}))
                    sys.exit(0)

            kwargs = json.loads(sys.argv[1])
            if needs_brain:
                kwargs["brain"] = brain

            if inspect.iscoroutinefunction(tool.execute):
                result = asyncio.run(tool.execute(**kwargs))
            else:
                result = tool.execute(**kwargs)
                
            print(json.dumps({{"ok": True, "result": result}}))
        except Exception as e:
            print(json.dumps({{"ok": False, "error": str(e)}}))
        finally:
            if brain is not None:
                brain.close()
    """)

    result = subprocess.run(
        [_sandbox_python(), "-c", runner_code, json.dumps(args)],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        return False, result.stderr[:500] or "Subprocess failed"

    try:
        data = json.loads(result.stdout.strip())
        if data.get("ok"):
            return True, data["result"]
        return False, data.get("error", "Unknown error")
    except (json.JSONDecodeError, KeyError):
        return False, result.stdout[:500]
