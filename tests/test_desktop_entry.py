"""Tests for the installed desktop launcher entrypoint."""

from unittest.mock import MagicMock, patch
from pathlib import Path


def test_desktop_entry_opens_gui_without_setup_wizard():
    from remy import desktop_entry

    gui_instance = MagicMock()

    with (
        patch("remy.desktop_entry.load_dotenv") as load_dotenv,
        patch("remy.core.logging_config.setup_logging") as setup_logging,
        patch("remy.core.setup.ensure_directories") as ensure_directories,
        patch("remy.core.desktop_gui.DesktopGUI", return_value=gui_instance) as desktop_gui,
        patch("remy.core.setup.run_setup_wizard") as run_setup_wizard,
    ):
        desktop_entry.main()

    load_dotenv.assert_called_once()
    setup_logging.assert_called_once_with(log_to_file=True)
    ensure_directories.assert_called_once()
    desktop_gui.assert_called_once()
    gui_instance.run_desktop.assert_called_once()
    run_setup_wizard.assert_not_called()


def test_pyproject_declares_desktop_gui_script():
    try:
        import tomllib
    except ModuleNotFoundError:
        import pytest

        pytest.skip("tomllib is available on Python 3.11+")

    pyproject = tomllib.loads(open("pyproject.toml", "rb").read().decode("utf-8"))

    assert pyproject["project"]["gui-scripts"]["remy-app"] == "remy.desktop_entry:main"


def test_package_module_entrypoint_delegates_to_main():
    import remy.__main__ as module_entry
    import remy.main as cli_main

    assert module_entry.main is cli_main.main


def test_desktop_entry_keeps_remy_imports_inside_main():
    source = Path("src/remy/desktop_entry.py").read_text(encoding="utf-8")

    assert "from remy.core." not in source.split("def main", 1)[0]
