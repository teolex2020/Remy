"""Tests for SandboxManifest — JSON registry with status flow."""

import json
from pathlib import Path

from remy.sandbox.manifest import SandboxManifest


class TestManifestBasic:
    """Core CRUD operations."""

    def test_empty_manifest(self, manifest):
        assert manifest.tools == []
        assert manifest.get_approved_tools() == []
        assert manifest.get_pending_tools() == []
        assert manifest.summary() == []

    def test_add_tool(self, manifest):
        entry = manifest.add_tool(
            name="my_tool", file="my_tool.py",
            description="Test tool", parameters={"x": {"type": "STRING"}},
            required=["x"],
        )
        assert entry["name"] == "my_tool"
        assert entry["status"] == "draft"
        assert entry["test_result"] is None
        assert len(manifest.tools) == 1

    def test_get_tool(self, manifest):
        manifest.add_tool("t1", "t1.py", "Tool 1", {}, [])
        assert manifest.get_tool("t1") is not None
        assert manifest.get_tool("nonexistent") is None

    def test_add_duplicate_resets_to_draft(self, manifest):
        manifest.add_tool("t1", "t1.py", "v1", {}, [])
        manifest.set_test_result("t1", passed=1, failed=0)
        assert manifest.get_tool("t1")["status"] == "tested"

        # Re-add same name → resets to draft
        manifest.add_tool("t1", "t1_v2.py", "v2", {}, [])
        assert manifest.get_tool("t1")["status"] == "draft"
        assert manifest.get_tool("t1")["file"] == "t1_v2.py"

    def test_summary(self, manifest):
        manifest.add_tool("t1", "t1.py", "First tool with a very long description that should be truncated", {}, [])
        s = manifest.summary()
        assert len(s) == 1
        assert s[0]["name"] == "t1"
        assert len(s[0]["description"]) <= 80


class TestManifestStatusFlow:
    """Verify draft → tested → pending → approved | rejected."""

    def test_happy_path_to_approved(self, manifest):
        manifest.add_tool("t", "t.py", "desc", {}, [])
        assert manifest.get_tool("t")["status"] == "draft"

        # draft → tested (via set_test_result)
        manifest.set_test_result("t", passed=2, failed=0)
        assert manifest.get_tool("t")["status"] == "tested"

        # tested → pending
        assert manifest.submit_for_approval("t")
        assert manifest.get_tool("t")["status"] == "pending"

        # pending → approved
        assert manifest.update_status("t", "approved")
        assert manifest.get_tool("t")["status"] == "approved"
        assert len(manifest.get_approved_tools()) == 1

    def test_happy_path_to_rejected(self, manifest):
        manifest.add_tool("t", "t.py", "desc", {}, [])
        manifest.set_test_result("t", passed=1, failed=0)
        manifest.submit_for_approval("t")

        assert manifest.update_status("t", "rejected")
        assert manifest.get_tool("t")["status"] == "rejected"
        assert manifest.get_approved_tools() == []

    def test_invalid_transition_draft_to_approved(self, manifest):
        manifest.add_tool("t", "t.py", "desc", {}, [])
        assert not manifest.update_status("t", "approved")
        assert manifest.get_tool("t")["status"] == "draft"

    def test_invalid_transition_draft_to_pending(self, manifest):
        manifest.add_tool("t", "t.py", "desc", {}, [])
        assert not manifest.update_status("t", "pending")

    def test_submit_only_from_tested(self, manifest):
        manifest.add_tool("t", "t.py", "desc", {}, [])
        # Can't submit from draft
        assert not manifest.submit_for_approval("t")

    def test_rejected_can_go_back_to_draft(self, manifest):
        manifest.add_tool("t", "t.py", "desc", {}, [])
        manifest.set_test_result("t", passed=1, failed=0)
        manifest.submit_for_approval("t")
        manifest.update_status("t", "rejected")

        assert manifest.update_status("t", "draft")
        assert manifest.get_tool("t")["status"] == "draft"

    def test_failed_tests_stay_draft(self, manifest):
        manifest.add_tool("t", "t.py", "desc", {}, [])
        manifest.set_test_result("t", passed=0, failed=2)
        # Status doesn't advance to tested if failures
        assert manifest.get_tool("t")["status"] == "draft"

    def test_update_nonexistent_tool(self, manifest):
        assert not manifest.update_status("ghost", "approved")
        assert not manifest.submit_for_approval("ghost")


class TestManifestPersistence:
    """Verify manifest survives save/load cycle."""

    def test_persist_and_reload(self, tmp_path):
        path = tmp_path / "manifest.json"

        m1 = SandboxManifest(path)
        m1.add_tool("t1", "t1.py", "Tool 1", {"x": {"type": "STRING"}}, ["x"], ["requests"])
        m1.set_test_result("t1", passed=3, failed=0)
        m1.submit_for_approval("t1")

        # Load fresh instance from same file
        m2 = SandboxManifest(path)
        t = m2.get_tool("t1")
        assert t is not None
        assert t["status"] == "pending"
        assert t["dependencies"] == ["requests"]
        assert t["test_result"]["passed"] == 3

    def test_manifest_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "manifest.json"
        m = SandboxManifest(path)
        m.add_tool("t", "t.py", "desc", {}, [])
        assert path.exists()

    def test_corrupted_manifest_resets(self, tmp_path):
        path = tmp_path / "manifest.json"
        path.write_text("not valid json!!!", encoding="utf-8")

        m = SandboxManifest(path)
        assert m.tools == []  # graceful reset
