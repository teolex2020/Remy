"""Desktop launcher entrypoint for installed Windows builds.

This module is intentionally separate from ``remy.main``. The command-line
entrypoint keeps developer/server/voice modes, while this entrypoint is what a
desktop shortcut or future installer should launch.
"""

from dotenv import load_dotenv


def main() -> None:
    """Open Remy as a local desktop app without requiring a terminal wizard."""
    load_dotenv()

    from remy.core.logging_config import setup_logging
    from remy.core.setup import ensure_directories

    setup_logging(log_to_file=True)
    ensure_directories()

    from remy.core.desktop_gui import DesktopGUI

    gui = DesktopGUI()
    gui.run_desktop()


if __name__ == "__main__":
    main()
