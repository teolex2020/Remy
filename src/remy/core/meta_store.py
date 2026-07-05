"""
Meta Store — organized .meta/ directory for system state files.

Separates system-managed state from user data:
- data/.meta/metrics/     — task_metrics.json, execution_log.jsonl
- data/.meta/monitoring/  — monitoring_snapshots.json
- data/.meta/browser/     — browser_failure_memory.json, browser_success_playbooks.json
- data/.meta/budget/      — autonomy_budget.json, survival_state.json, token_usage.json

User data stays in data/:
- data/brain/             — semantic memory
- data/missions.json      — user-defined missions
- data/history/           — session history
- data/browser_profile/   — Playwright profile
- data/browser_screenshots/

On first run, auto-migrates existing files from data/ to data/.meta/.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from remy.config.settings import settings

logger = logging.getLogger("MetaStore")

# Subdirectories within .meta/
_SUBDIRS = ("metrics", "monitoring", "browser", "budget")

# Files to migrate: (old_relative_path, new_subdirectory)
_MIGRATIONS: list[tuple[str, str]] = [
    ("task_metrics.json", "metrics"),
    ("execution_log.jsonl", "metrics"),
    ("eval_metrics.jsonl", "metrics"),
    ("monitoring_snapshots.json", "monitoring"),
    ("browser_failure_memory.json", "browser"),
    ("browser_success_playbooks.json", "browser"),
    ("autonomy_budget.json", "budget"),
    ("survival_state.json", "budget"),
    ("token_usage.json", "budget"),
    ("autonomy_benchmarks.json", "metrics"),
    ("autonomy_live_validation_report.json", "metrics"),
    ("autonomy_live_validation_scenarios.json", "metrics"),
    ("model_registry.json", "budget"),
    ("pricing.json", "budget"),
]


def meta_dir() -> Path:
    """Root .meta/ directory."""
    return settings.DATA_DIR / ".meta"


def meta_path(subdirectory: str, filename: str) -> Path:
    """Resolve a file path within .meta/. Creates subdirectory if needed."""
    path = meta_dir() / subdirectory
    path.mkdir(parents=True, exist_ok=True)
    return path / filename


def ensure_meta_dirs():
    """Create .meta/ subdirectories."""
    root = meta_dir()
    root.mkdir(parents=True, exist_ok=True)
    for sub in _SUBDIRS:
        (root / sub).mkdir(exist_ok=True)


def auto_migrate():
    """Move existing data/ files to data/.meta/ if they haven't been moved yet.

    Safe to call multiple times — only moves files that exist at old location
    and don't exist at new location.
    """
    ensure_meta_dirs()
    moved = 0

    for old_name, sub in _MIGRATIONS:
        old_path = settings.DATA_DIR / old_name
        new_path = meta_dir() / sub / old_name

        if old_path.exists() and not new_path.exists():
            try:
                shutil.move(str(old_path), str(new_path))
                moved += 1
                logger.info("Migrated %s → .meta/%s/%s", old_name, sub, old_name)
            except Exception as e:
                logger.warning("Failed to migrate %s: %s", old_name, e)

    if moved:
        logger.info("Meta migration complete: %d files moved", moved)


def resolve_path(filename: str, subdirectory: str | None = None) -> Path:
    """Resolve a data file path — checks .meta/ first, falls back to data/.

    This is the main entry point for code that needs to read/write state files.
    During the migration period, it checks both locations.
    """
    if subdirectory:
        meta = meta_dir() / subdirectory / filename
        if meta.exists():
            return meta
        # Check old location
        old = settings.DATA_DIR / filename
        if old.exists():
            return old
        # Default to new location
        (meta_dir() / subdirectory).mkdir(parents=True, exist_ok=True)
        return meta
    # No subdirectory — return data/ path
    return settings.DATA_DIR / filename
