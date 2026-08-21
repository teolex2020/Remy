from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from remy.core.project_store import (
    LEGACY_PROJECT_ID,
    get_project_store,
    project_artifact_dir,
    reset_project_store_for_tests,
)


@pytest.fixture
def project_surfaces(tmp_path, monkeypatch):
    from remy.config.settings import settings

    data_dir = tmp_path / "data"
    monkeypatch.setattr(settings, "DATA_DIR", data_dir)
    monkeypatch.setattr(settings, "AURA_BRAIN_PATH", data_dir / "brain")
    reset_project_store_for_tests()
    store = get_project_store()
    first = store.create_project("First")
    second = store.create_project("Second")
    yield store, first, second, data_dir
    reset_project_store_for_tests()


def test_artifact_directories_follow_project_boundaries(project_surfaces):
    _, first, second, data_dir = project_surfaces

    first_docs = project_artifact_dir("documents", first.project_id, create=True)
    second_docs = project_artifact_dir("documents", second.project_id, create=True)
    legacy_docs = project_artifact_dir(
        "documents",
        LEGACY_PROJECT_ID,
        legacy_data_dir=data_dir,
    )

    assert first_docs == Path(first.brain_path).parent / "documents"
    assert second_docs == Path(second.brain_path).parent / "documents"
    assert first_docs != second_docs
    assert legacy_docs == data_dir / "documents"
    with pytest.raises(ValueError, match="Unsupported"):
        project_artifact_dir("../escape", first.project_id)


@pytest.mark.asyncio
async def test_documents_and_reports_are_invisible_across_projects(
    project_surfaces,
):
    from remy.web.routes import documents_routes

    store, first, second, _ = project_surfaces
    store.set_active_project(first.project_id)
    await documents_routes.update_document(
        "private.md",
        documents_routes.DocumentUpdate(content="first project only"),
    )
    first_report_dir = project_artifact_dir("reports", first.project_id, create=True)
    (first_report_dir / "private.pdf").write_bytes(b"%PDF-first")

    first_documents = await documents_routes.list_documents()
    first_reports = await documents_routes.list_reports()
    store.set_active_project(second.project_id)
    second_documents = await documents_routes.list_documents()
    second_reports = await documents_routes.list_reports()

    assert first_documents["project_id"] == first.project_id
    assert [item["name"] for item in first_documents["documents"]] == ["private.md"]
    assert first_documents["documents"][0]["agent_access"] is False
    assert [item["name"] for item in first_reports["reports"]] == ["private.pdf"]
    assert first_reports["reports"][0]["download_url"].endswith(
        "/private.pdf?download=1"
    )
    assert second_documents == {"project_id": second.project_id, "documents": []}
    assert second_reports == {"project_id": second.project_id, "reports": []}
    with pytest.raises(HTTPException) as exc:
        await documents_routes.get_document("private.md")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_media_endpoint_serves_only_the_active_project_copy(project_surfaces):
    from remy.web.routes import media_routes

    store, first, second, _ = project_surfaces
    first_dir = project_artifact_dir("reports", first.project_id, create=True)
    second_dir = project_artifact_dir("reports", second.project_id, create=True)
    (first_dir / "same.pdf").write_bytes(b"%PDF-first")
    (second_dir / "same.pdf").write_bytes(b"%PDF-second")

    store.set_active_project(first.project_id)
    first_response = await media_routes.serve_report("same.pdf")
    store.set_active_project(second.project_id)
    second_response = await media_routes.serve_report("same.pdf")
    store.set_active_project(first.project_id)
    download_response = await media_routes.serve_report("same.pdf", download=True)

    assert Path(first_response.path).read_bytes() == b"%PDF-first"
    assert Path(second_response.path).read_bytes() == b"%PDF-second"
    assert download_response.headers["content-disposition"].startswith("attachment;")


@pytest.mark.asyncio
async def test_document_agent_access_is_explicit_and_project_scoped(
    project_surfaces,
):
    from remy.core.workspace_permissions import (
        workspace_list,
        workspace_read,
        workspace_search,
    )
    from remy.web.routes import documents_routes

    store, first, second, _ = project_surfaces
    store.set_active_project(first.project_id)
    await documents_routes.update_document(
        "brief.md",
        documents_routes.DocumentUpdate(content="private plan"),
    )

    denied = json.loads(
        workspace_read({"path": "workspace://data/documents/brief.md"})
    )
    assert "private" in denied["error"].lower()
    hidden = json.loads(
        workspace_list({"path": "workspace://data/documents/"})
    )
    hidden_search = json.loads(
        workspace_search(
            {
                "path": "workspace://data/",
                "mode": "grep",
                "pattern": "private plan",
            }
        )
    )
    assert hidden["entries"] == []
    assert hidden_search["results"] == []

    changed = await documents_routes.update_document_access(
        "brief.md",
        documents_routes.DocumentAccessUpdate(agent_access=True),
    )
    allowed = json.loads(
        workspace_read({"path": "workspace://data/documents/brief.md"})
    )
    visible_search = json.loads(
        workspace_search(
            {
                "path": "workspace://data/",
                "mode": "grep",
                "pattern": "private plan",
            }
        )
    )
    assert changed["agent_access"] is True
    assert allowed["content"] == "private plan"
    assert visible_search["count"] == 1

    store.set_active_project(second.project_id)
    missing = json.loads(
        workspace_read({"path": "workspace://data/documents/brief.md"})
    )
    assert "not found" in missing["error"].lower()


def test_builtin_agent_workspace_hides_internal_project_state(project_surfaces):
    from remy.core.workspace_permissions import workspace_list, workspace_read

    store, first, _, _ = project_surfaces
    store.set_active_project(first.project_id)
    project_root = Path(first.brain_path).parent
    (project_root / "runtime").mkdir()
    (project_root / "runtime" / "secret.txt").write_text(
        "not model context",
        encoding="utf-8",
    )

    listing = json.loads(workspace_list({"path": "workspace://data/"}))
    denied = json.loads(
        workspace_read({"path": "workspace://data/runtime/secret.txt"})
    )

    assert all(item["name"] != "runtime" for item in listing["entries"])
    assert "internal project data" in denied["error"].lower()


def test_browser_screenshots_are_written_to_the_active_project(project_surfaces):
    from remy.core.browser import BrowserManager

    store, first, second, _ = project_surfaces
    manager = BrowserManager()

    store.set_active_project(first.project_id)
    first_name = manager.save_screenshot(b"first")
    store.set_active_project(second.project_id)
    second_name = manager.save_screenshot(b"second")

    assert (
        project_artifact_dir("browser_screenshots", first.project_id) / first_name
    ).read_bytes() == b"first"
    assert (
        project_artifact_dir("browser_screenshots", second.project_id) / second_name
    ).read_bytes() == b"second"


def test_glass_brain_reads_the_active_microbrain(project_surfaces):
    from remy.web.routes import glass_brain_routes

    store, first, second, _ = project_surfaces
    store.set_active_project(first.project_id)
    assert glass_brain_routes._data_dir() == first.brain_path
    store.set_active_project(second.project_id)
    assert glass_brain_routes._data_dir() == second.brain_path


def test_artifact_generators_use_project_directory_resolver():
    from pathlib import Path

    brain_tools = Path("src/remy/core/brain_tools.py").read_text(encoding="utf-8")
    dispatch = Path("src/remy/core/tool_dispatch.py").read_text(encoding="utf-8")
    for source in (brain_tools, dispatch):
        assert 'project_artifact_dir(\n        "generated_images"' in source
        assert 'project_artifact_dir(\n            "reports"' in source
        assert 'project_artifact_dir(\n            "presentations"' in source


def test_report_generator_writes_inside_bound_project(
    project_surfaces,
    monkeypatch,
):
    from remy.core import brain_tools
    from remy.core.microbrain import bind_project

    _, first, _, _ = project_surfaces
    fake_brain = MagicMock()
    fake_brain.store.return_value = MagicMock(id="report-record")
    monkeypatch.setattr(brain_tools, "brain", fake_brain)
    monkeypatch.setattr(brain_tools, "brain_lock", threading.RLock())

    with bind_project(first.project_id):
        result = json.loads(
            brain_tools._generate_report(
                {
                    "title": "Private report",
                    "sections": [
                        {
                            "type": "section",
                            "title": "Result",
                            "body": "Visible only in the first project.",
                        }
                    ],
                },
                session_id="conversation-1",
                channel="desktop",
            )
        )

    report_path = (
        project_artifact_dir("reports", first.project_id) / result["filename"]
    )
    assert result["generated"] is True
    assert report_path.is_file()
    assert report_path.parent == Path(first.brain_path).parent / "reports"
