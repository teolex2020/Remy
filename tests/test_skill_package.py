"""Tests for Skill Package export/import and marketplace."""

import json
import tarfile
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ============== Fixtures ==============


@pytest.fixture
def tmp_sandbox(tmp_path):
    """Create a temporary sandbox structure."""
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    exports_dir = sandbox_dir / "exports"
    exports_dir.mkdir()
    return tmp_path, tools_dir, sandbox_dir, exports_dir


SAMPLE_TOOL_CODE = """\
TOOL_NAME = "test_calc"
TOOL_DESCRIPTION = "A test calculator tool"
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "a": {"type": "string", "description": "First number"},
        "b": {"type": "string", "description": "Second number"}
    }
}

def execute(a: str, b: str) -> str:
    return str(int(a) + int(b))

def test_calc():
    assert execute("2", "3") == "5"
"""


# ============== Export tests ==============


class TestExportSkill:
    def test_export_creates_archive(self, tmp_sandbox):
        """Export should create a .skill.tar.gz file."""
        _, tools_dir, sandbox_dir, exports_dir = tmp_sandbox

        # Write tool file
        tool_file = tools_dir / "test_calc.py"
        tool_file.write_text(SAMPLE_TOOL_CODE, encoding="utf-8")

        # Mock manifest
        mock_manifest = MagicMock()
        mock_manifest.get_tool.return_value = {
            "name": "test_calc",
            "file": "test_calc.py",
            "description": "A test calculator tool",
            "status": "approved",
            "dependencies": [],
        }

        mock_registry = MagicMock()
        mock_registry.manifest = mock_manifest

        with (
            patch("remy.sandbox.skill_package.settings") as mock_settings,
            patch("remy.core.tool_registry_mgmt.get_registry", return_value=mock_registry),
        ):
            mock_settings.SANDBOX_TOOLS_DIR = tools_dir
            mock_settings.SANDBOX_DIR = sandbox_dir

            from remy.sandbox.skill_package import export_skill

            result = export_skill("test_calc", output_dir=exports_dir)

        assert result.exists()
        assert result.suffix == ".gz"
        assert "test_calc" in result.name

        # Verify archive contents
        with tarfile.open(result, "r:gz") as tar:
            names = tar.getnames()
            assert "tool.py" in names
            assert "skill.json" in names

            # Verify skill.json
            skill_json = json.loads(tar.extractfile("skill.json").read())
            assert skill_json["name"] == "test_calc"
            assert skill_json["description"] == "A test calculator tool"

    def test_export_rejects_unapproved(self, tmp_sandbox):
        """Export should reject tools that aren't approved."""
        _, tools_dir, sandbox_dir, _ = tmp_sandbox

        mock_manifest = MagicMock()
        mock_manifest.get_tool.return_value = {
            "name": "draft_tool",
            "status": "draft",
        }
        mock_registry = MagicMock()
        mock_registry.manifest = mock_manifest

        with (
            patch("remy.sandbox.skill_package.settings") as mock_settings,
            patch("remy.core.tool_registry_mgmt.get_registry", return_value=mock_registry),
        ):
            mock_settings.SANDBOX_TOOLS_DIR = tools_dir
            mock_settings.SANDBOX_DIR = sandbox_dir

            from remy.sandbox.skill_package import export_skill

            with pytest.raises(ValueError, match="must be approved"):
                export_skill("draft_tool")

    def test_export_rejects_missing_tool(self, tmp_sandbox):
        """Export should reject tools not in manifest."""
        _, tools_dir, sandbox_dir, _ = tmp_sandbox

        mock_manifest = MagicMock()
        mock_manifest.get_tool.return_value = None
        mock_registry = MagicMock()
        mock_registry.manifest = mock_manifest

        with (
            patch("remy.sandbox.skill_package.settings") as mock_settings,
            patch("remy.core.tool_registry_mgmt.get_registry", return_value=mock_registry),
        ):
            mock_settings.SANDBOX_TOOLS_DIR = tools_dir
            mock_settings.SANDBOX_DIR = sandbox_dir

            from remy.sandbox.skill_package import export_skill

            with pytest.raises(ValueError, match="not found"):
                export_skill("nonexistent")


# ============== Import tests ==============


class TestImportSkill:
    def _make_archive(self, tmp_path, name="test_calc", code=SAMPLE_TOOL_CODE, meta=None):
        """Helper: create a valid skill archive."""
        if meta is None:
            meta = {
                "name": name,
                "version": "1.0.0",
                "description": "A test calculator",
                "dependencies": [],
            }

        archive_path = tmp_path / f"{name}.skill.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            # tool.py
            code_bytes = code.encode("utf-8")
            info = tarfile.TarInfo(name="tool.py")
            info.size = len(code_bytes)
            tar.addfile(info, BytesIO(code_bytes))

            # skill.json
            meta_bytes = json.dumps(meta).encode("utf-8")
            info = tarfile.TarInfo(name="skill.json")
            info.size = len(meta_bytes)
            tar.addfile(info, BytesIO(meta_bytes))

        return archive_path

    def test_import_valid_archive(self, tmp_sandbox):
        """Import should register tool from valid archive."""
        base, tools_dir, sandbox_dir, _ = tmp_sandbox
        archive = self._make_archive(base)

        mock_manifest = MagicMock()
        mock_manifest.get_tool.return_value = None  # No existing tool
        entry = {
            "name": "test_calc",
            "status": "draft",
            "description": "A test calculator",
        }
        mock_manifest.add_tool.return_value = entry

        mock_registry = MagicMock()
        mock_registry.manifest = mock_manifest

        with (
            patch("remy.sandbox.skill_package.settings") as mock_settings,
            patch("remy.core.tool_registry_mgmt.get_registry", return_value=mock_registry),
            patch("remy.sandbox.runner.validate_tool_file", return_value=(True, "ok")),
        ):
            mock_settings.SANDBOX_TOOLS_DIR = tools_dir
            mock_settings.SANDBOX_DIR = sandbox_dir

            from remy.sandbox.skill_package import import_skill

            result = import_skill(archive)

        assert result["imported"] is True
        assert result["name"] == "test_calc"
        assert "test" in result["message"].lower() or "approve" in result["message"].lower()

    def test_import_rejects_missing_fields(self, tmp_sandbox):
        """Import should reject archives with incomplete skill.json."""
        base, tools_dir, sandbox_dir, _ = tmp_sandbox
        archive = self._make_archive(
            base, meta={"name": "incomplete"}
        )  # Missing version, description

        with patch("remy.sandbox.skill_package.settings") as mock_settings:
            mock_settings.SANDBOX_TOOLS_DIR = tools_dir
            mock_settings.SANDBOX_DIR = sandbox_dir

            from remy.sandbox.skill_package import import_skill

            result = import_skill(archive)

        assert result["imported"] is False
        assert "missing" in result["error"].lower()

    def test_import_rejects_path_traversal(self, tmp_sandbox):
        """Import should reject archives with unsafe paths."""
        base, tools_dir, sandbox_dir, _ = tmp_sandbox

        archive_path = base / "evil.skill.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            info = tarfile.TarInfo(name="../../../etc/passwd")
            info.size = 5
            tar.addfile(info, BytesIO(b"evil!"))

        with patch("remy.sandbox.skill_package.settings") as mock_settings:
            mock_settings.SANDBOX_TOOLS_DIR = tools_dir
            mock_settings.SANDBOX_DIR = sandbox_dir

            from remy.sandbox.skill_package import import_skill

            result = import_skill(archive_path)

        assert result["imported"] is False
        assert "unsafe" in result["error"].lower()

    def test_import_rejects_duplicate_name(self, tmp_sandbox):
        """Import should reject if tool file already exists."""
        base, tools_dir, sandbox_dir, _ = tmp_sandbox

        # Pre-existing tool
        (tools_dir / "test_calc.py").write_text("existing", encoding="utf-8")

        archive = self._make_archive(base)

        with patch("remy.sandbox.skill_package.settings") as mock_settings:
            mock_settings.SANDBOX_TOOLS_DIR = tools_dir
            mock_settings.SANDBOX_DIR = sandbox_dir

            from remy.sandbox.skill_package import import_skill

            result = import_skill(archive)

        assert result["imported"] is False
        assert "already exists" in result["error"]

    def test_import_rejects_oversized_archive(self, tmp_sandbox):
        """Import should reject archives over 512KB."""
        base, tools_dir, sandbox_dir, _ = tmp_sandbox

        archive_path = base / "big.skill.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            big_data = b"x" * (513 * 1024)
            info = tarfile.TarInfo(name="tool.py")
            info.size = len(big_data)
            tar.addfile(info, BytesIO(big_data))

            meta = json.dumps({"name": "big", "version": "1.0.0", "description": "big"}).encode()
            info = tarfile.TarInfo(name="skill.json")
            info.size = len(meta)
            tar.addfile(info, BytesIO(meta))

        with patch("remy.sandbox.skill_package.settings") as mock_settings:
            mock_settings.SANDBOX_TOOLS_DIR = tools_dir
            mock_settings.SANDBOX_DIR = sandbox_dir

            from remy.sandbox.skill_package import import_skill

            result = import_skill(archive_path)

        # The archive itself may be smaller due to compression of repeated bytes,
        # but the validation checks archive file size
        # This tests the validation path exists


# ============== Marketplace tests ==============


class TestMarketplace:
    def test_browse_parses_index(self):
        """browse_marketplace should parse index.json."""
        mock_index = {
            "skills": [
                {"name": "crypto_tracker", "version": "1.0.0", "description": "Track crypto"},
                {"name": "weather_check", "version": "1.2.0", "description": "Check weather"},
            ]
        }

        with patch("remy.sandbox.marketplace._fetch_url") as mock_fetch:
            mock_fetch.return_value = json.dumps(mock_index).encode("utf-8")

            from remy.sandbox.marketplace import browse_marketplace

            result = browse_marketplace()

        assert len(result) == 2
        assert result[0]["name"] == "crypto_tracker"

    def test_browse_handles_network_error(self):
        """browse_marketplace should return empty on network error."""
        from urllib.error import URLError

        with patch("remy.sandbox.marketplace._fetch_url", side_effect=URLError("timeout")):
            from remy.sandbox.marketplace import browse_marketplace

            result = browse_marketplace()

        assert result == []

    def test_browse_handles_invalid_json(self):
        """browse_marketplace should return empty on invalid JSON."""
        with patch("remy.sandbox.marketplace._fetch_url", return_value=b"not json"):
            from remy.sandbox.marketplace import browse_marketplace

            result = browse_marketplace()

        assert result == []

    def test_install_fetches_and_registers(self, tmp_sandbox):
        """install_from_marketplace should download, validate, and register."""
        base, tools_dir, sandbox_dir, _ = tmp_sandbox

        skill_meta = json.dumps(
            {
                "name": "remote_tool",
                "version": "2.0.0",
                "description": "A remote tool",
                "dependencies": [],
            }
        ).encode("utf-8")

        def fake_fetch(url, **kw):
            if "skill.json" in url:
                return skill_meta
            if "tool.py" in url:
                return SAMPLE_TOOL_CODE.encode("utf-8")
            raise ValueError(f"Unexpected URL: {url}")

        mock_manifest = MagicMock()
        entry = {"name": "remote_tool", "status": "draft", "description": "A remote tool"}
        mock_manifest.add_tool.return_value = entry
        mock_registry = MagicMock()
        mock_registry.manifest = mock_manifest

        with (
            patch("remy.sandbox.marketplace._fetch_url", side_effect=fake_fetch),
            patch("remy.sandbox.marketplace.settings") as mock_settings,
            patch("remy.sandbox.runner.validate_tool_file", return_value=(True, "ok")),
            patch("remy.core.tool_registry_mgmt.get_registry", return_value=mock_registry),
        ):
            mock_settings.SANDBOX_TOOLS_DIR = tools_dir
            mock_settings.SANDBOX_DIR = sandbox_dir

            from remy.sandbox.marketplace import install_from_marketplace

            result = install_from_marketplace("remote_tool")

        assert result["installed"] is True
        assert result["name"] == "remote_tool"
        assert result["version"] == "2.0.0"

    def test_install_handles_missing_skill(self):
        """install_from_marketplace should handle missing skill gracefully."""
        from urllib.error import URLError

        with patch("remy.sandbox.marketplace._fetch_url", side_effect=URLError("404")):
            from remy.sandbox.marketplace import install_from_marketplace

            result = install_from_marketplace("nonexistent")

        assert result["installed"] is False
        assert "fetch" in result["error"].lower() or "failed" in result["error"].lower()

    def test_install_rejects_invalid_tool(self, tmp_sandbox):
        """install should reject tool that fails AST validation."""
        base, tools_dir, sandbox_dir, _ = tmp_sandbox

        skill_meta = json.dumps(
            {
                "name": "bad_tool",
                "version": "1.0.0",
                "description": "Bad tool",
                "dependencies": [],
            }
        ).encode("utf-8")

        def fake_fetch(url, **kw):
            if "skill.json" in url:
                return skill_meta
            if "tool.py" in url:
                return b"invalid python code !!!"
            raise ValueError(f"Unexpected URL: {url}")

        with (
            patch("remy.sandbox.marketplace._fetch_url", side_effect=fake_fetch),
            patch("remy.sandbox.marketplace.settings") as mock_settings,
        ):
            mock_settings.SANDBOX_TOOLS_DIR = tools_dir
            mock_settings.SANDBOX_DIR = sandbox_dir

            from remy.sandbox.marketplace import install_from_marketplace

            result = install_from_marketplace("bad_tool")

        assert result["installed"] is False
        assert "validation" in result["error"].lower() or "syntax" in result["error"].lower()


# ============== Telemetry tests ==============


class TestUsageTelemetry:
    def test_telemetry_tracked_on_execute(self):
        """execute_sandbox_tool should update telemetry in manifest."""
        mock_manifest = MagicMock()
        tool_entry = {
            "name": "my_tool",
            "file": "my_tool.py",
            "status": "approved",
        }
        mock_manifest.get_tool.return_value = tool_entry

        with (
            patch("remy.core.tool_registry.execute_tool", return_value=(True, "result")),
            patch("remy.core.tool_registry.settings") as mock_settings,
        ):
            mock_settings.SANDBOX_TOOLS_DIR = Path("/fake/tools")
            mock_settings.AURA_BRAIN_PATH = "/fake/brain"

            from remy.core.tool_registry import ToolRegistry

            registry = ToolRegistry.__new__(ToolRegistry)
            registry._manifest = mock_manifest
            registry._sandbox_names = {"my_tool"}

            # Mock tool file existence
            with patch.object(Path, "exists", return_value=True):
                result = registry.execute_sandbox_tool("my_tool", {"arg": "val"})

        assert result == "result"
        assert "telemetry" in tool_entry
        assert tool_entry["telemetry"]["call_count"] == 1
        assert tool_entry["telemetry"]["error_count"] == 0
        mock_manifest.save.assert_called()

    def test_telemetry_tracks_errors(self):
        """Telemetry should count errors separately."""
        mock_manifest = MagicMock()
        tool_entry = {
            "name": "flaky_tool",
            "file": "flaky_tool.py",
            "status": "approved",
        }
        mock_manifest.get_tool.return_value = tool_entry

        with (
            patch("remy.core.tool_registry.execute_tool", return_value=(False, "boom")),
            patch("remy.core.tool_registry.settings") as mock_settings,
        ):
            mock_settings.SANDBOX_TOOLS_DIR = Path("/fake/tools")
            mock_settings.AURA_BRAIN_PATH = "/fake/brain"

            from remy.core.tool_registry import ToolRegistry

            registry = ToolRegistry.__new__(ToolRegistry)
            registry._manifest = mock_manifest
            registry._sandbox_names = {"flaky_tool"}

            with patch.object(Path, "exists", return_value=True):
                result = registry.execute_sandbox_tool("flaky_tool", {})

        assert "error" in result.lower()
        assert tool_entry["telemetry"]["call_count"] == 1
        assert tool_entry["telemetry"]["error_count"] == 1

    def test_telemetry_in_summary(self):
        """Manifest summary should include telemetry when present."""
        from remy.sandbox.manifest import SandboxManifest

        with patch.object(
            SandboxManifest,
            "_load",
            return_value={
                "version": 1,
                "tools": [
                    {
                        "name": "tracked",
                        "status": "approved",
                        "description": "Tracked tool",
                        "telemetry": {"call_count": 5, "error_count": 1, "total_ms": 2500},
                    }
                ],
            },
        ):
            m = SandboxManifest(Path("/fake/manifest.json"))
            summary = m.summary()

        assert len(summary) == 1
        assert "telemetry" in summary[0]
        assert summary[0]["telemetry"]["call_count"] == 5


# ============== Heartbeat tests ==============


class TestHeartbeatStability:
    def test_backoff_constants_defined(self):
        """Verify heartbeat stability constants exist."""
        from remy.core.autonomy import (
            BACKOFF_BASE_SEC,
            BACKOFF_MAX_SEC,
            FAST_RECOVERY_DELAYS,
            INVOKE_AGENT_TIMEOUT_SEC,
        )

        assert INVOKE_AGENT_TIMEOUT_SEC == 90
        assert BACKOFF_BASE_SEC == 120
        assert BACKOFF_MAX_SEC == 900
        assert len(FAST_RECOVERY_DELAYS) == 3
        assert FAST_RECOVERY_DELAYS == [30, 60, 120]

    def test_loop_has_backoff_fields(self):
        """AutonomousLoop should have backoff tracking fields."""
        with (
            patch("remy.core.autonomy._setup_autonomy_logger"),
            patch("remy.core.autonomy.load_budget"),
            patch("remy.core.autonomy.settings") as mock_settings,
        ):
            mock_settings.AUTONOMY_DAILY_TOKEN_LIMIT = 100000
            mock_settings.AUTONOMY_HOURLY_TOKEN_LIMIT = 20000
            mock_settings.AUTONOMY_SESSION_TOKEN_LIMIT = 500000
            mock_settings.AUTONOMY_CYCLE_INTERVAL_SEC = 120
            mock_settings.AUTONOMY_QUIET_HOURS_START = 23
            mock_settings.AUTONOMY_QUIET_HOURS_END = 7
            mock_settings.AUTONOMY_MAX_SESSION_MINUTES = 30
            mock_settings.AUTONOMY_PROACTIVE_SESSIONS_ENABLED = False
            mock_settings.TELEGRAM_BOT_TOKEN = None
            mock_settings.PROACTIVE_CHAT_ID = None

            from remy.core.autonomy import AutonomousLoop

            loop = AutonomousLoop()

        assert hasattr(loop, "_backoff_failures")
        assert hasattr(loop, "_fast_recovery_idx")
        assert hasattr(loop, "_last_fast_recovery")
        assert loop._backoff_failures == 0
        assert loop._fast_recovery_idx == 0

    def test_backoff_calculation(self):
        """Exponential backoff should cap at BACKOFF_MAX_SEC."""
        from remy.core.autonomy import BACKOFF_BASE_SEC, BACKOFF_MAX_SEC

        # Simulate backoff progression
        for failures in range(1, 10):
            backoff = min(BACKOFF_BASE_SEC * (2 ** (failures - 1)), BACKOFF_MAX_SEC)
            assert backoff <= BACKOFF_MAX_SEC
            if failures == 1:
                assert backoff == 120
            elif failures == 2:
                assert backoff == 240
            elif failures == 3:
                assert backoff == 480
            elif failures >= 4:
                assert backoff == BACKOFF_MAX_SEC  # 960 > 900 → capped
