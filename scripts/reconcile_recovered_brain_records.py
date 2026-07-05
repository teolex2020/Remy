#!/usr/bin/env python3
"""
Reconcile records imported by startup backup recovery.
"""

from __future__ import annotations

import argparse
import json

from remy.core.startup_recovery import reconcile_recovered_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile startup backup recovery records.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = reconcile_recovered_records(apply=args.apply and not args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
