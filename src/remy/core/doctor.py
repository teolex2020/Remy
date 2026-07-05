"""
remy doctor — startup diagnostics.

Checks all critical dependencies and configurations before launch.
Prints a clear status report and returns True if system is healthy.
"""

import os
import socket
from pathlib import Path


def _ok(label: str, detail: str = "") -> dict:
    return {"status": "ok", "label": label, "detail": detail}


def _warn(label: str, detail: str = "", fix: str = "") -> dict:
    return {"status": "warn", "label": label, "detail": detail, "fix": fix}


def _fail(label: str, detail: str = "", fix: str = "") -> dict:
    return {"status": "fail", "label": label, "detail": detail, "fix": fix}


# ============== INDIVIDUAL CHECKS ==============

def check_env_file() -> dict:
    """Check if .env file exists."""
    from remy.config.settings import settings
    env_path = settings.BASE_DIR / ".env"
    if env_path.exists():
        return _ok(".env file", str(env_path))
    return _fail(".env file", "Not found", "Run `remy --setup` to create it")


def check_gemini_api_key() -> dict:
    """Check if GEMINI_API_KEY is set."""
    from remy.config.settings import settings
    key = settings.GEMINI_API_KEY
    if not key:
        return _fail("GEMINI_API_KEY", "Not set", "Add GEMINI_API_KEY=... to your .env file")
    if len(key) < 10:
        return _fail("GEMINI_API_KEY", "Too short — likely invalid", "Check your API key at aistudio.google.com")
    masked = key[:6] + "..." + key[-4:]
    return _ok("GEMINI_API_KEY", masked)


def check_web_port() -> dict:
    """Check if WEB_PORT is available."""
    from remy.config.settings import settings
    if not settings.WEB_ENABLED:
        return _ok("Web port", "WEB_ENABLED=false, skipped")
    port = settings.WEB_PORT
    host = settings.WEB_HOST
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex((host, port))
            if result == 0:
                return _warn(
                    f"Web port {port}",
                    f"Port {port} already in use on {host}",
                    f"Stop existing process or change WEB_PORT in .env"
                )
    except Exception:
        pass
    return _ok(f"Web port {port}", f"{host}:{port} is available")


def check_duplicate_process() -> dict:
    """Check if another remy process is already running."""
    current_pid = os.getpid()
    try:
        import psutil
        remy_procs = []
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = " ".join(proc.info.get("cmdline") or [])
                if proc.info["pid"] != current_pid and "remy" in cmdline.lower():
                    remy_procs.append(proc.info["pid"])
            except Exception:
                pass
        if remy_procs:
            pids = ", ".join(str(p) for p in remy_procs)
            return _warn(
                "Duplicate process",
                f"Remy already running (PID: {pids})",
                "Stop the existing instance before starting a new one"
            )
        return _ok("Duplicate process", "No other remy process detected")
    except ImportError:
        return _ok("Duplicate process", "psutil not installed — check skipped")


def check_playwright() -> dict:
    """Check if Playwright and Chromium are installed."""
    try:
        import importlib
        importlib.import_module("playwright.sync_api")
    except ImportError:
        return _fail(
            "Playwright",
            "Not installed",
            "Run: pip install playwright && python -m playwright install chromium"
        )
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            exec_path = p.chromium.executable_path
            if exec_path and Path(exec_path).exists():
                return _ok("Playwright + Chromium", str(exec_path))
            return _warn(
                "Playwright",
                "Installed but Chromium binary not found",
                "Run: python -m playwright install chromium"
            )
    except Exception as e:
        return _warn("Playwright", f"Installed but check failed: {e}", "Run: python -m playwright install chromium")


def check_telegram() -> dict:
    """Check Telegram bot token and security mode."""
    from remy.config.settings import settings
    token = settings.TELEGRAM_BOT_TOKEN
    if not token:
        return _ok("Telegram", "Not configured (TELEGRAM_BOT_TOKEN not set)")

    # Check allowlist
    allowed = settings.TELEGRAM_ALLOWED_CHAT_IDS
    if not allowed:
        return _warn(
            "Telegram security",
            "Bot is in OPEN MODE -- any user can send messages",
            "Add TELEGRAM_ALLOWED_CHAT_IDS=<your_chat_id> to .env\n"
            "  Get your ID by messaging @userinfobot on Telegram"
        )

    masked_token = token[:10] + "..." + token[-4:]
    ids_str = ", ".join(str(i) for i in allowed)
    return _ok("Telegram", f"Token: {masked_token} | Allowlist: {ids_str}")


def check_aura_memory() -> dict:
    """Check Aura memory health."""
    from remy.config.settings import settings
    brain_path = settings.AURA_BRAIN_PATH
    if not brain_path.exists():
        return _warn(
            "Aura memory",
            "Brain directory does not exist yet",
            "It will be created automatically on first run"
        )
    try:
        from aura import Aura

        brain = Aura(str(brain_path))
        try:
            if hasattr(brain, "count"):
                count = brain.count()
            elif hasattr(brain, "list_records"):
                count = len(brain.list_records())
            else:
                count = len(brain.search(query="", limit=5000))
        finally:
            close = getattr(brain, "close", None)
            if callable(close):
                close()
        if count == 0:
            return _warn("Aura memory", "Brain exists but has 0 records", "Brain will populate as you interact with Remy")
        return _ok("Aura memory", f"{count} records in {brain_path}")
    except Exception as e:
        return _fail("Aura memory", f"Failed to load: {e}", "Check if aura-memory is installed correctly")


def check_data_dirs() -> dict:
    """Check that required data directories exist."""
    from remy.config.settings import settings
    required = [
        settings.DATA_DIR,
        settings.AURA_BRAIN_PATH,
        settings.AURA_MEMORY_PATH,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return _warn(
            "Data directories",
            f"Missing: {', '.join(missing)}",
            "Run `remy --setup` or start remy once to auto-create them"
        )
    return _ok("Data directories", str(settings.DATA_DIR))


# ============== REPORT ==============

def run_doctor(verbose: bool = False) -> bool:
    """Run all checks and print a formatted report. Returns True if no failures."""
    # Use ASCII symbols — safe on all terminals including Windows cp1251
    ICON_OK   = "[OK]"
    ICON_WARN = "[!!]"
    ICON_FAIL = "[XX]"

    checks = [
        check_env_file,
        check_gemini_api_key,
        check_data_dirs,
        check_aura_memory,
        check_web_port,
        check_duplicate_process,
        check_playwright,
        check_telegram,
    ]

    print()
    print("=" * 56)
    print("  Remy Doctor -- System Diagnostics")
    print("=" * 56)
    print()

    results = []
    for check_fn in checks:
        try:
            result = check_fn()
        except Exception as e:
            result = _fail(check_fn.__name__, f"Check crashed: {e}")
        results.append(result)

        status = result["status"]
        label = result["label"]
        detail = result.get("detail", "")
        fix = result.get("fix", "")

        if status == "ok":
            icon = ICON_OK
        elif status == "warn":
            icon = ICON_WARN
        else:
            icon = ICON_FAIL

        detail_str = f" -- {detail}" if detail else ""
        print(f"  {icon}  {label}{detail_str}")
        if fix and status != "ok":
            for line in fix.split("\n"):
                print(f"         -> {line}")

    failures = [r for r in results if r["status"] == "fail"]
    warnings = [r for r in results if r["status"] == "warn"]

    print()
    print("=" * 56)
    if failures:
        print(f"  FAILED: {len(failures)} error(s) -- Remy may not start correctly")
    elif warnings:
        print(f"  WARNING: {len(warnings)} warning(s) -- Remy can start but review above")
    else:
        print("  OK: All checks passed -- Remy is ready")
    print("=" * 56)
    print()

    return len(failures) == 0
