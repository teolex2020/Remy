"""Point frozen Playwright at the Chromium bundled with Remy."""

import os
import sys
from pathlib import Path


if getattr(sys, "frozen", False):
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    browser_root = bundle_root / "playwright" / "driver" / "package" / ".local-browsers"
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(browser_root))
