# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir build for the consumer-facing Remy desktop app."""

import os
import importlib.util
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


PROJECT_ROOT = Path(SPECPATH).parents[1]
ENTRYPOINT = PROJECT_ROOT / "src" / "remy" / "desktop_entry.py"

datas = collect_data_files("remy")
datas += [
    (str(PROJECT_ROOT / ".env.example"), "."),
    (str(PROJECT_ROOT / "LICENSE"), "."),
    (str(PROJECT_ROOT / "packaging" / "agent-lab-runtime" / "Dockerfile"), "agent-lab-runtime"),
    (str(PROJECT_ROOT / "packaging" / "agent-lab-runtime" / ".dockerignore"), "agent-lab-runtime"),
]

# PLAYWRIGHT_BROWSERS_PATH=0 stores Chromium below the Playwright package.
# Collecting its data makes browser automation work without a post-install
# download on the user's machine.
datas += collect_data_files("playwright")
datas += collect_data_files("webview")

playwright_spec = importlib.util.find_spec("playwright")
if playwright_spec and playwright_spec.submodule_search_locations:
    playwright_root = Path(next(iter(playwright_spec.submodule_search_locations)))
    bundled_browsers = playwright_root / "driver" / "package" / ".local-browsers"
    if bundled_browsers.exists():
        datas.append(
            (
                str(bundled_browsers),
                "playwright/driver/package/.local-browsers",
            )
        )

binaries = collect_dynamic_libs("aura")

hiddenimports = collect_submodules("remy")
hiddenimports += [
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
]

a = Analysis(
    [str(ENTRYPOINT)],
    pathex=[str(PROJECT_ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(PROJECT_ROOT / "packaging" / "windows" / "playwright_runtime.py")],
    excludes=[
        "datasets",
        "pandas",
        "pyarrow",
        "pytest",
        "tensorflow",
        "tkinter",
        "torch",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Remy",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version=os.environ.get("REMY_VERSION_FILE"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Remy",
)
