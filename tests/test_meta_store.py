"""Tests for MetaStore — .meta/ directory pattern for system state."""

import json
from unittest.mock import patch

import pytest

from remy.core.meta_store import (
    auto_migrate,
    ensure_meta_dirs,
    meta_dir,
    meta_path,
    resolve_path,
)


@pytest.fixture
def fake_data_dir(tmp_path):
    """Patch settings.DATA_DIR to a temp directory."""
    with patch("remy.core.meta_store.settings") as mock_settings:
        mock_settings.DATA_DIR = tmp_path
        yield tmp_path


# ============== meta_dir / meta_path ==============


class TestMetaDir:
    def test_returns_meta_subdir(self, fake_data_dir):
        result = meta_dir()
        assert result == fake_data_dir / ".meta"

    def test_meta_path_creates_subdir(self, fake_data_dir):
        result = meta_path("metrics", "test.json")
        assert result == fake_data_dir / ".meta" / "metrics" / "test.json"
        assert (fake_data_dir / ".meta" / "metrics").is_dir()


# ============== ensure_meta_dirs ==============


class TestEnsureMetaDirs:
    def test_creates_all_subdirs(self, fake_data_dir):
        ensure_meta_dirs()
        root = fake_data_dir / ".meta"
        assert root.is_dir()
        for sub in ("metrics", "monitoring", "browser", "budget"):
            assert (root / sub).is_dir()

    def test_idempotent(self, fake_data_dir):
        ensure_meta_dirs()
        ensure_meta_dirs()  # Should not raise
        assert (fake_data_dir / ".meta").is_dir()


# ============== auto_migrate ==============


class TestAutoMigrate:
    def test_migrates_existing_file(self, fake_data_dir):
        # Create a file at old location
        old_file = fake_data_dir / "task_metrics.json"
        old_file.write_text('{"test": true}')

        auto_migrate()

        # Should be at new location
        new_file = fake_data_dir / ".meta" / "metrics" / "task_metrics.json"
        assert new_file.exists()
        assert json.loads(new_file.read_text()) == {"test": True}
        # Old file should be gone
        assert not old_file.exists()

    def test_does_not_overwrite_existing_new(self, fake_data_dir):
        # Create files at both locations
        old_file = fake_data_dir / "task_metrics.json"
        old_file.write_text('{"old": true}')

        new_dir = fake_data_dir / ".meta" / "metrics"
        new_dir.mkdir(parents=True)
        new_file = new_dir / "task_metrics.json"
        new_file.write_text('{"new": true}')

        auto_migrate()

        # New file should be untouched
        assert json.loads(new_file.read_text()) == {"new": True}
        # Old file should still exist (not migrated since new exists)
        assert old_file.exists()

    def test_skips_missing_files(self, fake_data_dir):
        # No files to migrate — should not raise
        auto_migrate()
        assert (fake_data_dir / ".meta").is_dir()

    def test_migrates_multiple_files(self, fake_data_dir):
        files_to_create = [
            ("task_metrics.json", "metrics"),
            ("execution_log.jsonl", "metrics"),
            ("autonomy_budget.json", "budget"),
            ("monitoring_snapshots.json", "monitoring"),
        ]
        for name, _ in files_to_create:
            (fake_data_dir / name).write_text("{}")

        auto_migrate()

        for name, subdir in files_to_create:
            assert (fake_data_dir / ".meta" / subdir / name).exists()
            assert not (fake_data_dir / name).exists()

    def test_idempotent(self, fake_data_dir):
        (fake_data_dir / "task_metrics.json").write_text("{}")
        auto_migrate()
        auto_migrate()  # Second call should be a no-op
        assert (fake_data_dir / ".meta" / "metrics" / "task_metrics.json").exists()


# ============== resolve_path ==============


class TestResolvePath:
    def test_returns_meta_path_when_exists(self, fake_data_dir):
        meta = fake_data_dir / ".meta" / "metrics"
        meta.mkdir(parents=True)
        target = meta / "test.json"
        target.write_text("{}")

        result = resolve_path("test.json", "metrics")
        assert result == target

    def test_falls_back_to_data_dir(self, fake_data_dir):
        old = fake_data_dir / "test.json"
        old.write_text("{}")

        result = resolve_path("test.json", "metrics")
        assert result == old

    def test_defaults_to_meta_when_neither_exists(self, fake_data_dir):
        result = resolve_path("new_file.json", "metrics")
        assert result == fake_data_dir / ".meta" / "metrics" / "new_file.json"
        # Should have created the subdirectory
        assert (fake_data_dir / ".meta" / "metrics").is_dir()

    def test_no_subdirectory_returns_data_dir(self, fake_data_dir):
        result = resolve_path("missions.json")
        assert result == fake_data_dir / "missions.json"

    def test_prefers_meta_over_old(self, fake_data_dir):
        """When file exists in both locations, .meta/ wins."""
        old = fake_data_dir / "test.json"
        old.write_text('{"old": true}')

        meta = fake_data_dir / ".meta" / "budget"
        meta.mkdir(parents=True)
        new = meta / "test.json"
        new.write_text('{"new": true}')

        result = resolve_path("test.json", "budget")
        assert result == new
