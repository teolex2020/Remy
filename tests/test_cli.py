"""Tests for the public Remy command-line interface and package extras."""

import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from remy.main import _build_parser, _parse_args, main
from remy.optional_dependencies import require_extra


@pytest.mark.parametrize(
    ("command", "attribute"),
    [
        ("web", "web"),
        ("desktop", "desktop"),
        ("setup", "setup"),
        ("doctor", "doctor"),
    ],
)
def test_friendly_subcommands_enable_existing_execution_paths(command, attribute):
    args = _parse_args([command])

    assert args.command == command
    assert getattr(args, attribute) is True


@pytest.mark.parametrize("legacy_flag", ["--web", "--desktop", "--setup", "--doctor"])
def test_legacy_mode_flags_remain_supported(legacy_flag):
    args = _parse_args([legacy_flag])

    assert getattr(args, legacy_flag.removeprefix("--").replace("-", "_")) is True
    assert args.command is None


def test_global_options_work_after_subcommand():
    args = _parse_args(["web", "--log-level", "DEBUG"])

    assert args.web is True
    assert args.log_level == "DEBUG"


def test_help_presents_quick_start_commands():
    help_text = _build_parser().format_help()

    assert "{web,desktop,setup,doctor}" in help_text


def test_missing_optional_feature_has_actionable_install_hint():
    with patch("remy.optional_dependencies.find_spec", return_value=None):
        with pytest.raises(SystemExit) as exc_info:
            require_extra("webview", "desktop", "desktop mode")

    assert 'pip install "remy[desktop]"' in str(exc_info.value)


def test_available_optional_feature_is_accepted():
    with patch("remy.optional_dependencies.find_spec", return_value=object()):
        require_extra("webview", "desktop", "desktop mode")


def test_desktop_command_checks_extra_before_runtime_imports():
    with patch("remy.main.require_extra", side_effect=SystemExit("missing")) as check:
        with pytest.raises(SystemExit, match="missing"):
            main(["desktop"])

    check.assert_called_once_with("webview", "desktop", "desktop mode")


def _dependency_name(requirement: str) -> str:
    return requirement.split("[", 1)[0].split("~", 1)[0].split("=", 1)[0].lower()


def _project_metadata() -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
        import tomli as tomllib

    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_default_package_excludes_optional_feature_stacks():
    project = _project_metadata()
    base_names = {_dependency_name(item) for item in project["dependencies"]}
    extras = project["optional-dependencies"]

    assert {"fastapi", "uvicorn", "google-genai", "aura-memory"} <= base_names
    assert {
        "pywebview",
        "pyaudio",
        "playwright",
        "python-telegram-bot",
        "reportlab",
        "langchain-openai",
    }.isdisjoint(base_names)
    assert {
        "desktop",
        "voice",
        "browser",
        "telegram",
        "push",
        "documents",
        "openai",
        "anthropic",
        "nvidia",
        "providers",
        "all",
    } <= set(extras)


def test_all_extra_contains_every_feature_dependency():
    extras = _project_metadata()["optional-dependencies"]
    all_names = {_dependency_name(item) for item in extras["all"]}

    for extra in ("desktop", "voice", "browser", "telegram", "push", "documents", "providers"):
        assert {_dependency_name(item) for item in extras[extra]} <= all_names


def test_web_runtime_imports_without_optional_feature_packages():
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        blocked = {
            "webview", "pyaudio", "playwright", "telegram", "pywebpush",
            "reportlab", "pptx", "fitz", "PIL", "xlsxwriter", "openpyxl",
            "langchain_openai", "langchain_anthropic",
            "langchain_nvidia_ai_endpoints", "langchain_community",
            "openai", "anthropic",
        }

        class OptionalPackageBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in blocked:
                    raise ModuleNotFoundError(f"blocked optional package: {fullname}")
                return None

        sys.meta_path.insert(0, OptionalPackageBlocker())

        from remy.core.desktop_gui import create_app

        app = create_app()
        assert app.title.startswith("Remy")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
