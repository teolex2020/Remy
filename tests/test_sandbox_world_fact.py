"""Live wiring test: executable-judge 3-state fact gates sandbox tool approval.

A sandbox tool is only auto-approvable when its tests genuinely ran and passed
(`supports`). A run that produced no observable pass/fail (`inconclusive` — e.g.
no tests collected, or a hidden stderr error under a clean exit) must NOT be
treated as a validated tool. Uses the REAL SDK executable judge.
"""

import pytest

aura = pytest.importorskip("aura")
if not hasattr(aura, "world_fact_from_output"):
    pytest.skip(
        "installed AuraSDK lacks world_fact_from_output",
        allow_module_level=True,
    )

from remy.sandbox.runner import _world_fact_from_pytest


def test_passing_pytest_run_is_supports():
    stdout = "test_tool.py::test_basic PASSED\n\n1 passed in 0.01s"
    assert _world_fact_from_pytest(0, stdout, "") == "supports"


def test_failing_pytest_run_is_refutes():
    stdout = "test_tool.py::test_basic FAILED\n\nE   AssertionError: nope"
    assert _world_fact_from_pytest(1, stdout, "") == "refutes"


def test_no_tests_collected_is_inconclusive_not_failure():
    # pytest exits 5 when no tests were collected. That is NOT a failure — the
    # tool simply was not exercised. The world fact must be inconclusive so the
    # tool is not auto-approved (and not falsely refuted either).
    stdout = "collected 0 items\n\nno tests ran in 0.00s"
    assert _world_fact_from_pytest(5, stdout, "") == "inconclusive"


def test_hidden_stderr_error_under_clean_exit_is_refutes():
    # The proven discipline: exit code lies. A clean exit (0) with an error in
    # stderr must still be read as a refutation.
    assert _world_fact_from_pytest(
        0, "", "Traceback (most recent call last):\n  ImportError: boom"
    ) == "refutes"


def test_collection_error_is_refutes():
    stdout = "===== ERRORS =====\n2 errors during collection"
    assert _world_fact_from_pytest(2, stdout, "") == "refutes"
