"""
Standalone maintenance cron — runs background brain without the web server.

This script is the missing piece that makes V17 incubation, TCS thermal routing,
and V16 drives work between sessions. Without it, all cognitive loops only run
during active user conversation (30 min/day).

Usage:
  python -m tools.maintenance.maintenance_cron              # run once
  python -m tools.maintenance.maintenance_cron --loop 7200   # run every 2 hours
  python -m tools.maintenance.maintenance_cron --quiet        # suppress report output

Schedule with Windows Task Scheduler or cron:
  # Every 3 hours
  schtasks /create /tn "Remi Maintenance" /tr "python -m tools.maintenance.maintenance_cron" /sc HOURLY /mo 3
  # Or crontab:
  0 */3 * * * cd /path/to/remy/app && python -m tools.maintenance.maintenance_cron
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

if not getattr(sys, "frozen", False):
    _src = str(Path(__file__).parents[2] / "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

# Reconfigure stdout for Windows Unicode
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MaintenanceCron")


def _init_brain():
    """Initialize Aura brain for standalone maintenance."""
    from remy.config.settings import settings
    from aura import Aura

    brain_path = Path(settings.AURA_BRAIN_PATH)
    brain_path.mkdir(parents=True, exist_ok=True)

    # Probe store compatibility before opening
    try:
        brain = Aura(str(brain_path))
        count = brain.count()
        logger.info("Brain opened: %s (%d records)", brain_path, count)
        return brain
    except Exception as e:
        logger.error("Failed to open brain at %s: %s", brain_path, e)
        return None


def run_once(brain, *, quiet: bool = False, notify: bool = False) -> dict:
    """Run one full maintenance cycle.

    This calls the same run_background() that the web scheduler uses,
    which triggers:
      - AuraSDK run_maintenance() (decay, reflect, epistemic, belief,
        concept, causal, policy, V17 incubation, V16 drives, mood)
      - TCS thermal map + cycle classification + plasticity
      - LLM gating (cold graph = skip)
      - Consolidation with thermal routing
      - Archival with cold-zone acceleration
      - Observation logging
    """
    from remy.core.background_brain import run_background, print_report

    logger.info("Starting maintenance cycle...")
    start = time.time()

    report = run_background(brain)

    elapsed = time.time() - start
    report["cron_elapsed_sec"] = round(elapsed, 2)

    if not quiet:
        print_report(report)

        # Extra TCS info — ACL rendered
        thermal_report_obj = report.get("_thermal_report_obj")
        if thermal_report_obj:
            from remy.core.acl_renderer import (
                Locale,
                thermal_summary_from_report,
                render_thermal_summary,
            )
            expr = thermal_summary_from_report(thermal_report_obj)
            print(f"\n{render_thermal_summary(expr, Locale.EN)}")
        elif report.get("thermal"):
            thermal = report["thermal"]
            print(f"\nThermal: energy={thermal.get('total_energy', 0):.1f} "
                  f"hot={thermal.get('hot_zone_count', 0)} "
                  f"cold={thermal.get('cold_mass_count', 0)} "
                  f"clusters={thermal.get('cluster_count', 0)}")

        cycle_cls = report.get("cycle_classification")
        if cycle_cls:
            print(f"Cycle: {cycle_cls['type']} ({cycle_cls['reason']})")
            if cycle_cls.get("skip_diagnostics"):
                print("  Diagnostics: SKIPPED (cold cycle)")
            if cycle_cls.get("skip_llm"):
                print("  LLM: SKIPPED (cold cycle)")

        plast = report.get("plasticity")
        if plast and plast.get("total_tracked", 0) > 0:
            print(f"Plasticity: {plast['total_tracked']} tracked, "
                  f"{plast.get('weakened', 0)} weakened, "
                  f"{plast.get('pruned', 0)} pruned")

        print(f"\nCompleted in {elapsed:.1f}s")

    if notify:
        try:
            import asyncio
            from remy.core.background_brain import send_notifications
            asyncio.run(send_notifications(report, brain))
        except Exception as e:
            logger.debug("Notification skipped: %s", e)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Remi standalone maintenance cron",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--loop", type=int, default=0, metavar="SECONDS",
        help="Run continuously with this interval between cycles (0 = run once)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress report output (still logs to stderr)",
    )
    parser.add_argument(
        "--notify", action="store_true",
        help="Send notifications via Telegram/WebPush after each cycle",
    )
    parser.add_argument(
        "--log-json", type=str, default="", metavar="PATH",
        help="Append each report as JSON line to this file",
    )
    args = parser.parse_args()

    brain = _init_brain()
    if brain is None:
        logger.error("Cannot start: brain initialization failed")
        return 1

    try:
        if args.loop <= 0:
            # Run once
            report = run_once(brain, quiet=args.quiet, notify=args.notify)
            if args.log_json:
                _append_json_log(args.log_json, report)
            return 0 if "error" not in report else 1
        else:
            # Loop mode
            logger.info("Entering loop mode: cycle every %d seconds", args.loop)
            cycle = 0
            while True:
                cycle += 1
                logger.info("=== Cycle %d at %s ===", cycle, datetime.now().isoformat())
                try:
                    report = run_once(brain, quiet=args.quiet, notify=args.notify)
                    if args.log_json:
                        _append_json_log(args.log_json, report)
                except Exception as e:
                    logger.error("Cycle %d failed: %s", cycle, e)
                logger.info("Sleeping %d seconds...", args.loop)
                time.sleep(args.loop)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    finally:
        try:
            brain.close()
            logger.info("Brain closed")
        except Exception:
            pass


def _append_json_log(path: str, report: dict) -> None:
    """Append report as one JSON line."""
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        logger.debug("Failed to write JSON log: %s", e)


if __name__ == "__main__":
    sys.exit(main() or 0)
