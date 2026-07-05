"""
Setup Wizard — first-run detection and interactive configuration.

Handles:
- Auto-detect missing .env file
- Interactive API key setup
- Directory creation
- Validation of configuration
"""

import os
import sys
from pathlib import Path

from remy.config.settings import settings


ENV_FILE = settings.BASE_DIR / ".env"

ENV_TEMPLATE = """\
# Remy — Configuration
# Get your Gemini API key at: https://aistudio.google.com/apikey

GEMINI_API_KEY={api_key}

# Model settings (defaults are fine for most users)
# GEMINI_MODEL=gemini-2.5-flash-native-audio-preview-12-2025
# SUMMARY_MODEL=gemini-3-flash-preview
# GEMINI_VOICE=Zephyr

# Telegram bot (optional)
# TELEGRAM_BOT_TOKEN=
# PROACTIVE_CHAT_ID=

# Web GUI
# WEB_HOST=127.0.0.1
# WEB_PORT=8080
"""


def needs_setup() -> bool:
    """Check if first-run setup is needed."""
    api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
    return not api_key


def ensure_directories():
    """Create required data directories."""
    dirs = [
        settings.DATA_DIR,
        settings.AURA_BRAIN_PATH,
        settings.AURA_MEMORY_PATH,
        settings.DATA_DIR / "logs",
        settings.DATA_DIR / "history",
        settings.DATA_DIR / "sandbox",
        settings.DATA_DIR / "browser_screenshots",
        settings.DATA_DIR / "generated_images",
        settings.DATA_DIR / "reports",
        settings.DATA_DIR / "presentations",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def run_setup_wizard() -> bool:
    """Interactive setup wizard. Returns True if setup completed successfully."""
    import sys
    if not sys.stdin.isatty():
        print()
        print("=" * 50)
        print("  REMY — Setup")
        print("=" * 50)
        print()
        print("  Non-interactive environment detected.")
        print(f"  Edit your .env file directly: {ENV_FILE}")
        print()
        print("  Required: GEMINI_API_KEY=<your key>")
        print("  Optional: TELEGRAM_BOT_TOKEN=<token>")
        print()
        print("  Get Gemini key at: https://aistudio.google.com/apikey")
        print("=" * 50)
        return False

    print()
    print("=" * 50)
    print("  REMY — First-Time Setup")
    print("=" * 50)
    print()
    print("Welcome! Let's configure your AI assistant.")
    print()

    # Step 1: API Key
    print("Step 1: Gemini API Key")
    print("  Get yours at: https://aistudio.google.com/apikey")
    print()

    existing_key = get_env_value("GEMINI_API_KEY")
    if existing_key:
        masked = existing_key[:8] + "..." + existing_key[-4:]
        api_key = input(f"  Current key: {masked} (press Enter to keep): ").strip()
        if not api_key:
            api_key = existing_key
    else:
        api_key = input("  Enter your Gemini API key: ").strip()
        if not api_key:
            print("\n  No API key provided. You can run with --setup later.")
            return False

    # Step 2: Optional Telegram
    print()
    print("Step 2: Telegram Bot (optional, press Enter to skip)")
    existing_telegram = get_env_value("TELEGRAM_BOT_TOKEN")
    if existing_telegram:
        masked_tg = existing_telegram[:6] + "..." + existing_telegram[-4:]
        telegram_token = input(f"  Current token: {masked_tg} (press Enter to keep): ").strip()
        if not telegram_token:
            telegram_token = existing_telegram
    else:
        telegram_token = input("  Telegram bot token: ").strip()

    # Step 3: Create .env
    env_content = ENV_TEMPLATE.format(api_key=api_key)
    if telegram_token:
        env_content = env_content.replace(
            "# TELEGRAM_BOT_TOKEN=",
            f"TELEGRAM_BOT_TOKEN={telegram_token}",
        )

    ENV_FILE.write_text(env_content, encoding="utf-8")
    print()
    print(f"  Configuration saved to: {ENV_FILE}")

    # Step 3: Directories
    ensure_directories()
    print("  Data directories created.")

    print()
    print("=" * 50)
    print("  Setup complete! Start with:")
    print("    remy --web     (browser)")
    print("    remy --desktop  (native window)")
    print("    remy --telegram (Telegram bot)")
    print("=" * 50)
    print()

    return True


def update_env_value(key: str, value: str):
    """Update a single key in the .env file, creating if needed."""
    env_path = ENV_FILE
    lines = []

    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    # Find and replace existing key
    found = False
    for i, line in enumerate(lines):
        stripped = line.lstrip("# ").split("=", 1)
        if stripped[0].strip() == key:
            lines[i] = f"{key}={value}"
            found = True
            break

    if not found:
        lines.append(f"{key}={value}")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_env_value(key: str) -> str | None:
    """Read a value from the .env file."""
    if not ENV_FILE.exists():
        return None

    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            return v.strip()

    return None
