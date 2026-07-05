"""
Remy
Local-first AI workflow automation with governed memory and human-approved execution.
"""

import asyncio
import argparse

from dotenv import load_dotenv

load_dotenv()

from .config.settings import settings
from .core.logging_config import setup_logging

logger = setup_logging(log_to_file=True)


async def _run_autonomy_entrypoint(coro_factory):
    """Best-effort browser sidecar startup for autonomous modes."""
    from .core.pinchtab_service import ensure_pinchtab_running, shutdown_pinchtab

    await ensure_pinchtab_running()
    try:
        await coro_factory()
    finally:
        await shutdown_pinchtab()


# ============== MAIN ==============

def _sandbox_list():
    """Show all sandbox tools and their status."""
    from .sandbox.manifest import SandboxManifest
    manifest = SandboxManifest(settings.SANDBOX_DIR / "manifest.json")
    tools = manifest.summary()
    if not tools:
        print("No sandbox tools found.")
        return
    print(f"\n{'Name':<25} {'Status':<12} {'Tests':<15} Description")
    print("-" * 80)
    for t in tools:
        test = t.get("test_result")
        test_str = f"{test['passed']}P/{test['failed']}F" if test else "not run"
        print(f"{t['name']:<25} {t['status']:<12} {test_str:<15} {t['description']}")
    print()


def _sandbox_approve():
    """Interactive approval of pending sandbox tools."""
    from .sandbox.manifest import SandboxManifest
    from pathlib import Path

    manifest = SandboxManifest(settings.SANDBOX_DIR / "manifest.json")
    pending = manifest.get_pending_tools()
    if not pending:
        print("No tools pending approval.")
        return

    for tool in pending:
        print(f"\n{'=' * 60}")
        print(f"Tool: {tool['name']}")
        print(f"Description: {tool['description']}")
        print(f"Dependencies: {', '.join(tool.get('dependencies', [])) or 'none'}")
        test = tool.get("test_result")
        if test:
            print(f"Tests: {test['passed']} passed, {test['failed']} failed")

        # Show source code
        tool_path = Path(settings.SANDBOX_TOOLS_DIR) / tool["file"]
        if tool_path.exists():
            print(f"\n--- Source: {tool_path.name} ---")
            print(tool_path.read_text(encoding="utf-8"))
            print("--- End ---")

        choice = input("\n[A]pprove / [R]eject / [S]kip? ").strip().lower()
        if choice == "a":
            manifest.update_status(tool["name"], "approved")
            print(f"  -> {tool['name']} APPROVED. Will be available on next agent request.")
        elif choice == "r":
            manifest.update_status(tool["name"], "rejected")
            print(f"  -> {tool['name']} REJECTED.")
        else:
            print(f"  -> Skipped.")


def main():
    parser_arg = argparse.ArgumentParser(
        description="Remy - local-first AI workflow automation"
    )
    parser_arg.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Log level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser_arg.add_argument("--sandbox-list", action="store_true", help="List all sandbox tools")
    parser_arg.add_argument("--sandbox-approve", action="store_true", help="Review and approve pending sandbox tools")
    parser_arg.add_argument("--background", action="store_true", help="Run background brain processing (decay, reflect, insights)")
    parser_arg.add_argument("--telegram", action="store_true", help="Start Telegram bot")
    parser_arg.add_argument("--desktop", action="store_true", help="Start desktop GUI (PyWebView native window)")
    parser_arg.add_argument("--web", action="store_true", help="Start web GUI only (no autonomy, no telegram)")
    parser_arg.add_argument("--autonomous", action="store_true", help="Start autonomous agent mode (self-directed, goal-driven)")
    parser_arg.add_argument("--autonomous-v3", action="store_true", help="Start opt-in autonomous v3 runtime (research-first)")
    parser_arg.add_argument("--serve", action="store_true", help="Start web GUI only (safe default)")
    parser_arg.add_argument("--serve-all", action="store_true", help="Start all channels enabled in .env (web + telegram + autonomy)")
    parser_arg.add_argument("--setup", action="store_true", help="Run first-time setup wizard")
    parser_arg.add_argument("--doctor", action="store_true", help="Run system diagnostics and check configuration")
    args = parser_arg.parse_args()

    # Re-initialize logging with CLI-specified level
    setup_logging(log_to_file=True, log_level=args.log_level)

    # Setup wizard
    from .core.setup import needs_setup, run_setup_wizard, ensure_directories
    ensure_directories()

    if args.setup:
        run_setup_wizard()
        return

    if args.doctor:
        from .core.doctor import run_doctor
        run_doctor()
        return

    # Auto-detect first run for interactive modes
    if needs_setup() and (args.web or args.desktop or args.telegram or args.autonomous or not any([
        args.sandbox_list, args.sandbox_approve, args.background
    ])):
        print("No API key found. Running first-time setup...")
        if not run_setup_wizard():
            return
        # Reload settings after .env creation
        load_dotenv(override=True)

    if args.sandbox_list:
        _sandbox_list()
        return
    if args.sandbox_approve:
        _sandbox_approve()
        return
    if args.background:
        from .core.background_brain import run_background, print_report, send_notifications
        report = run_background()
        print_report(report)
        asyncio.run(send_notifications(report))
        return
    # --serve: web only (safe default)
    if args.serve:
        from .core.combined_runner import run_combined
        try:
            asyncio.run(run_combined(autonomous=False, telegram=False, web=True))
        except KeyboardInterrupt:
            logger.info("Serve mode ended")
        return

    # --serve-all: all channels from .env
    if args.serve_all:
        from .core.combined_runner import run_combined
        try:
            asyncio.run(run_combined(
                autonomous=settings.AUTONOMY_ENABLED,
                telegram=bool(settings.TELEGRAM_BOT_TOKEN),
                web=settings.WEB_ENABLED,
            ))
        except KeyboardInterrupt:
            logger.info("Serve-all mode ended")
        return

    # Combined mode: multiple channels in one process
    combined_count = sum([args.autonomous, args.autonomous_v3, args.telegram, args.web])
    if combined_count > 1:
        from .core.combined_runner import run_combined
        try:
            asyncio.run(run_combined(
                autonomous=(args.autonomous or args.autonomous_v3),
                telegram=args.telegram,
                web=args.web,
            ))
        except KeyboardInterrupt:
            logger.info("Combined mode ended")
        return

    # Single-channel modes
    if args.telegram:
        from .core.telegram_bot import TelegramBot
        bot = TelegramBot()
        bot.run()
        return
    if args.desktop:
        from .core.desktop_gui import DesktopGUI
        gui = DesktopGUI()
        gui.run_desktop()
        return
    if args.autonomous:
        from .core.combined_runner import run_autonomy_standalone
        try:
            asyncio.run(run_autonomy_standalone())
        except KeyboardInterrupt:
            logger.info("Autonomous mode ended")
        return
    if args.autonomous_v3:
        from .core.combined_runner import run_autonomy_standalone
        try:
            asyncio.run(run_autonomy_standalone(version_override="v3"))
        except KeyboardInterrupt:
            logger.info("Autonomous v3 mode ended")
        return
    if args.web:
        from .core.desktop_gui import DesktopGUI
        gui = DesktopGUI()
        gui.run_web_only()
        return

    from .core.gemini_live import run_gemini_live
    try:
        asyncio.run(run_gemini_live())
    except KeyboardInterrupt:
        logger.info("Session ended")


if __name__ == "__main__":
    main()
