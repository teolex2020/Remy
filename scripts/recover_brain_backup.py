#!/usr/bin/env python3
"""
Recover missing records from a startup quarantine backup into the active brain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from remy.core.startup_recovery import apply_backup_recovery, inspect_backup_recovery


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover missing records from a startup backup JSON file.")
    parser.add_argument("--backup", required=True, help="Path to brain_backup_*.json")
    parser.add_argument("--apply", action="store_true", help="Write recovered records into the active brain")
    parser.add_argument("--dry-run", action="store_true", help="Only report what would be recovered")
    args = parser.parse_args()

    backup_path = Path(args.backup).resolve()
    if not backup_path.exists():
        raise FileNotFoundError(backup_path)

    summary = apply_backup_recovery(backup_path) if args.apply and not args.dry_run else inspect_backup_recovery(backup_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
