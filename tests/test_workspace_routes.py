import pytest
from unittest.mock import AsyncMock

from remy.web.routes import settings_routes


@pytest.mark.asyncio
async def test_workspace_routes_manage_capabilities(tmp_path, monkeypatch):
    from remy.config.settings import settings

    data = tmp_path / "data"
    project = tmp_path / "project"
    data.mkdir()
    project.mkdir()
    monkeypatch.setattr(settings, "DATA_DIR", data)

    created = await settings_routes.add_local_workspace(
        settings_routes.WorkspaceGrantPayload(path=str(project), read=True, write=False, execute=False)
    )
    workspace = created["workspace"]
    assert workspace["permissions"] == ["read"]

    listed = await settings_routes.list_local_workspaces()
    assert {item["workspace_id"] for item in listed["workspaces"]} == {"data", workspace["workspace_id"]}

    updated = await settings_routes.update_local_workspace(
        workspace["workspace_id"],
        settings_routes.WorkspaceUpdatePayload(read=True, write=True, execute=True),
    )
    assert updated["workspace"]["permissions"] == ["execute", "read", "write"]

    revoked = await settings_routes.revoke_local_workspace(workspace["workspace_id"])
    assert revoked["revoked"] is True
    listed_after = await settings_routes.list_local_workspaces()
    assert [item["workspace_id"] for item in listed_after["workspaces"]] == ["data"]


@pytest.mark.asyncio
async def test_workspace_route_rejects_drive_root(tmp_path, monkeypatch):
    from fastapi import HTTPException
    from remy.config.settings import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "data")
    with pytest.raises(HTTPException) as exc:
        await settings_routes.add_local_workspace(
            settings_routes.WorkspaceGrantPayload(path=tmp_path.anchor, read=True)
        )
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_native_workspace_picker_has_no_operation_timeout(tmp_path, monkeypatch):
    from remy.config.settings import settings

    data = tmp_path / "data"
    project = tmp_path / "project"
    data.mkdir()
    project.mkdir()
    monkeypatch.setattr(settings, "DATA_DIR", data)
    run_mock = AsyncMock(return_value=str(project))
    monkeypatch.setattr(settings_routes, "run_in_thread", run_mock)

    result = await settings_routes.select_local_workspace(
        settings_routes.WorkspaceGrantPayload(read=True, write=False, execute=False)
    )

    assert result["cancelled"] is False
    assert result["workspace"]["root_path"] == str(project.resolve())
    assert run_mock.await_args.kwargs["timeout"] is None
