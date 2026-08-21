"""Desktop launcher entrypoint for installed Windows builds.

This module is intentionally separate from ``remy.main``. The command-line
entrypoint keeps developer/server/voice modes, while this entrypoint is what a
desktop shortcut or future installer should launch.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv


_INTERNAL_AURA_PROBE_ARG = "--remy-internal-aura-probe"
_SINGLE_INSTANCE_MUTEX_NAME = r"Local\AuroraSeed.Remy.Desktop"
_single_instance_mutex = None


def _run_internal_aura_probe(argv: list[str]) -> bool:
    """Handle the isolated Aura health check used by packaged builds.

    In a PyInstaller build ``sys.executable`` points at ``Remy.exe`` rather
    than at a Python interpreter. The regular ``python -c`` probe therefore
    has to re-enter this executable through a private command-line argument.
    """
    if not argv or argv[0] != _INTERNAL_AURA_PROBE_ARG:
        return False
    if len(argv) != 2:
        raise SystemExit(f"{_INTERNAL_AURA_PROBE_ARG} requires one store path")

    from aura import Aura

    store = Aura(str(Path(argv[1])))
    close = getattr(store, "close", None)
    if callable(close):
        close()
    return True


def _acquire_single_instance() -> bool:
    """Allow only one interactive Remy desktop process per Windows session."""
    global _single_instance_mutex
    if sys.platform != "win32":
        return True
    if _single_instance_mutex is not None:
        return False

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_mutex = kernel32.CreateMutexW
    create_mutex.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    create_mutex.restype = wintypes.HANDLE
    handle = create_mutex(None, False, _SINGLE_INSTANCE_MUTEX_NAME)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        kernel32.CloseHandle(handle)
        return False

    # Holding the handle for the process lifetime holds the named mutex.
    _single_instance_mutex = (kernel32, handle)
    return True


def main() -> None:
    """Open Remy as a local desktop app without requiring a terminal wizard."""
    if _run_internal_aura_probe(sys.argv[1:]):
        return
    if not _acquire_single_instance():
        return

    load_dotenv()

    from remy.core.logging_config import setup_logging
    from remy.core.setup import ensure_directories

    setup_logging(log_to_file=True)
    ensure_directories()

    from remy.optional_dependencies import require_extra

    require_extra("webview", "desktop", "desktop mode")

    from remy.core.desktop_gui import DesktopGUI

    gui = DesktopGUI()
    gui.run_desktop()


if __name__ == "__main__":
    main()
