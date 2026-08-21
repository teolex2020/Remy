import json
from pathlib import Path

import pytest

from remy.core.workspace_permissions import (
    WorkspaceAccessError,
    WorkspacePermissionManager,
    workspace_list,
    workspace_read,
    workspace_search,
    workspace_write,
    build_code_workspace_context,
)


def test_builtin_data_workspace_supports_relative_read_write(tmp_path, monkeypatch):
    from remy.config.settings import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    written = json.loads(workspace_write({"path": "sandbox/item.txt", "content": "hello"}))
    assert written["written"] is True
    read = json.loads(workspace_read({"path": "sandbox/item.txt"}))
    assert read["content"] == "hello"
    assert read["workspace_id"] == "data"


def test_builtin_data_workspace_denies_internal_files(tmp_path, monkeypatch):
    from remy.config.settings import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "secret.txt").write_text("secret", encoding="utf-8")

    denied = json.loads(
        workspace_read({"path": "workspace://data/runtime/secret.txt"})
    )

    assert "internal project data" in denied["error"].lower()


def test_external_folder_is_denied_until_explicitly_granted(tmp_path):
    data = tmp_path / "data"
    outside = tmp_path / "project"
    data.mkdir()
    outside.mkdir()
    target = outside / "README.md"
    target.write_text("project", encoding="utf-8")
    manager = WorkspacePermissionManager(data)

    with pytest.raises(WorkspaceAccessError):
        manager.resolve(str(target), "read")

    grant = manager.add_grant(str(outside), permissions={"read"})
    resolved, matched = manager.resolve(f"workspace://{grant['workspace_id']}/README.md", "read")
    assert resolved == target.resolve()
    assert matched["workspace_id"] == grant["workspace_id"]


def test_read_grant_does_not_imply_write_or_execute(tmp_path):
    data = tmp_path / "data"
    project = tmp_path / "project"
    data.mkdir()
    project.mkdir()
    manager = WorkspacePermissionManager(data)
    grant = manager.add_grant(str(project), permissions={"read"})

    for capability in ("write", "execute"):
        with pytest.raises(WorkspaceAccessError, match=f"no {capability} permission"):
            manager.resolve(grant["uri"], capability)


def test_update_and_revoke_take_effect_immediately(tmp_path):
    data = tmp_path / "data"
    project = tmp_path / "project"
    data.mkdir()
    project.mkdir()
    manager = WorkspacePermissionManager(data)
    grant = manager.add_grant(str(project), permissions={"read"})
    manager.update_grant(grant["workspace_id"], permissions={"read", "write"})
    _, updated = manager.resolve(grant["uri"], "write")
    assert "write" in updated["permissions"]

    manager.revoke_grant(grant["workspace_id"])
    with pytest.raises(WorkspaceAccessError, match="Unknown or revoked"):
        manager.resolve(grant["uri"], "read")


def test_drive_or_filesystem_root_grant_is_rejected(tmp_path):
    manager = WorkspacePermissionManager(tmp_path / "data")
    root = Path(tmp_path.anchor)
    with pytest.raises(ValueError, match="entire filesystem or drive"):
        manager.add_grant(str(root), permissions={"read"})


def test_workspace_uri_cannot_escape_with_parent_segments(tmp_path):
    data = tmp_path / "data"
    project = tmp_path / "project"
    data.mkdir()
    project.mkdir()
    manager = WorkspacePermissionManager(data)
    grant = manager.add_grant(str(project), permissions={"read"})
    with pytest.raises(WorkspaceAccessError, match="escapes"):
        manager.resolve(f"{grant['uri']}../secret.txt", "read")


def test_list_and_search_are_scoped_to_granted_workspace(tmp_path, monkeypatch):
    from remy.config.settings import settings

    data = tmp_path / "data"
    project = tmp_path / "project"
    data.mkdir()
    project.mkdir()
    (project / "app.py").write_text("needle = True", encoding="utf-8")
    monkeypatch.setattr(settings, "DATA_DIR", data)
    grant = WorkspacePermissionManager(data).add_grant(str(project), permissions={"read"})

    listed = json.loads(workspace_list({"path": grant["uri"]}))
    assert [entry["name"] for entry in listed["entries"]] == ["app.py"]
    searched = json.loads(workspace_search({"path": grant["uri"], "mode": "grep", "pattern": "needle"}))
    assert searched["count"] == 1


def test_access_audit_contains_allow_and_deny_events(tmp_path):
    data = tmp_path / "data"
    project = tmp_path / "project"
    data.mkdir()
    project.mkdir()
    manager = WorkspacePermissionManager(data)
    grant = manager.add_grant(str(project), permissions={"read"})
    manager.resolve(grant["uri"], "read")
    with pytest.raises(WorkspaceAccessError):
        manager.resolve(grant["uri"], "write")
    events = manager.read_audit()
    assert any(event["allowed"] is False and "missing write" in event["reason"] for event in events)
    assert any(event["action"] == "grant" for event in events)


def test_code_workspace_context_maps_repo_without_copying_file_contents(tmp_path, monkeypatch):
    from remy.config.settings import settings

    data = tmp_path / "data"
    project = tmp_path / "project"
    data.mkdir()
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='demo'", encoding="utf-8")
    (project / "src").mkdir()
    (project / "src" / "app.py").write_text("TOP_SECRET_SOURCE = True", encoding="utf-8")
    (project / "node_modules").mkdir()
    (project / "node_modules" / "ignored.js").write_text("ignored", encoding="utf-8")
    monkeypatch.setattr(settings, "DATA_DIR", data)
    grant = WorkspacePermissionManager(data).add_grant(str(project), permissions={"read"})

    context = build_code_workspace_context(grant["workspace_id"])

    assert "ACTIVE CODE WORKSPACE" in context
    assert grant["uri"] in context
    assert "pyproject.toml" in context
    assert ".py:1" in context
    assert "TOP_SECRET_SOURCE" not in context
    assert "ignored.js" not in context
