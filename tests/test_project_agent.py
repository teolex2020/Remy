from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest

from remy.core.microbrain import bind_project
from remy.core.project_agent import (
    ProjectAgentStore,
    build_project_agent_instruction,
)
from remy.core.project_store import get_project_store, reset_project_store_for_tests


@pytest.fixture
def projects(tmp_path, monkeypatch):
    from remy.config.settings import settings

    data_dir = tmp_path / "data"
    monkeypatch.setattr(settings, "DATA_DIR", data_dir)
    monkeypatch.setattr(settings, "AURA_BRAIN_PATH", data_dir / "brain")
    reset_project_store_for_tests()
    store = get_project_store()
    first = store.create_project("Biology")
    second = store.create_project("Marketing")
    yield first, second
    reset_project_store_for_tests()


def test_project_agent_profile_is_scoped_and_enters_system_instruction(projects):
    first, second = projects
    ProjectAgentStore(first.project_id).update_profile(
        {
            "instruction": (
                "Act as a careful molecular biology research partner. "
                "State the assay and its controls, and label every unsupported "
                "mechanism as a hypothesis."
            ),
        }
    )

    with bind_project(first.project_id):
        first_context = build_project_agent_instruction()
    with bind_project(second.project_id):
        second_context = build_project_agent_instruction()

    assert "molecular biology" in first_context
    assert "unsupported mechanism" in first_context
    assert second_context == ""


def test_enabled_knowledge_pack_retrieves_relevant_source_only(projects):
    first, second = projects
    biology = ProjectAgentStore(first.project_id)
    pack = biology.create_pack("Cell protocols", "Laboratory reference material")
    source = biology.add_source(
        pack["pack_id"],
        "assays.md",
        (
            "# Organoid assay\n\n"
            "Measure cell regeneration with blinded controls and three replicates.\n\n"
            "Unrelated administrative note about room booking."
        ).encode(),
    )

    context = biology.retrieve("How should we measure cell regeneration?")

    assert "Knowledge Pack: Cell protocols" in context
    assert "Source: assays.md" in context
    assert "blinded controls" in context
    assert source["name"] == "assays.md"
    assert "stored_name" not in source
    assert ProjectAgentStore(second.project_id).retrieve("cell regeneration") == ""


def test_legacy_multi_field_profile_is_combined_into_one_instruction(projects):
    first, _ = projects
    agent = ProjectAgentStore(first.project_id)
    agent.meta_path.parent.mkdir(parents=True, exist_ok=True)
    agent.meta_path.write_text(
        json.dumps(
            {
                "version": 1,
                "project_id": first.project_id,
                "profile": {
                    "display_name": "Cell Partner",
                    "specialization": "Molecular biology",
                    "personality": "Careful",
                    "operating_instructions": "Cite protocols.",
                    "evidence_policy": (
                        "Separate sourced facts, project memory, and model inference."
                    ),
                },
                "knowledge_packs": [],
            }
        ),
        encoding="utf-8",
    )

    instruction = agent.get()["profile"]["instruction"]

    assert "Agent name: Cell Partner" in instruction
    assert "Professional specialization: Molecular biology" in instruction
    assert "Project working instructions: Cite protocols." in instruction
    assert "Evidence standard" not in instruction


def test_disabled_pack_is_not_model_context_and_can_be_reenabled(projects):
    first, _ = projects
    agent = ProjectAgentStore(first.project_id)
    pack = agent.create_pack("Private methods")
    agent.add_source(
        pack["pack_id"], "method.txt", b"The zircon assay uses protocol delta."
    )

    assert "zircon assay" in agent.retrieve("zircon assay")
    agent.update_pack(pack["pack_id"], {"enabled": False})
    assert agent.retrieve("zircon assay") == ""
    agent.update_pack(pack["pack_id"], {"enabled": True})
    assert "zircon assay" in agent.retrieve("zircon assay")


def test_pack_deletion_removes_known_sources_without_touching_memory(projects):
    first, _ = projects
    agent = ProjectAgentStore(first.project_id)
    pack = agent.create_pack("Temporary")
    source = agent.add_source(pack["pack_id"], "notes.md", b"temporary knowledge")
    source_dir = agent.sources_root / pack["pack_id"]

    assert source_dir.is_dir()
    assert source["source_id"].startswith("source-")
    agent.delete_pack(pack["pack_id"])

    assert not source_dir.exists()
    assert agent.get()["knowledge_packs"] == []
    assert Path(first.brain_path).is_dir()


def test_source_guards_reject_traversal_and_binary_content(projects):
    first, _ = projects
    agent = ProjectAgentStore(first.project_id)
    pack = agent.create_pack("Guarded")

    with pytest.raises(ValueError, match="plain filename"):
        agent.add_source(pack["pack_id"], "../escape.md", b"bad")
    with pytest.raises(ValueError, match="Unsupported"):
        agent.add_source(pack["pack_id"], "payload.exe", b"bad")
    with pytest.raises(ValueError, match="UTF-8"):
        agent.add_source(pack["pack_id"], "binary.txt", b"\xff\xfe\xfa")


def test_project_agent_ui_contract_is_present():
    html = Path("src/remy/web/static/index.html").read_text(encoding="utf-8")
    js = Path("src/remy/web/static/js/app.js").read_text(encoding="utf-8")

    assert 'id="project-agent-panel"' in html
    assert 'id="knowledge-pack-create-form"' in html
    assert 'id="knowledge-pack-files"' in html
    assert "Create and upload" in html
    assert "knowledgePackDescription" not in js
    assert "Agent instruction and Knowledge Packs" in js
    assert "uploadKnowledgePackSource" in js


@pytest.mark.asyncio
async def test_project_agent_routes_manage_profile_pack_and_source(projects, monkeypatch):
    from starlette.datastructures import UploadFile

    from remy.web.routes import project_routes

    first, _ = projects
    monkeypatch.setattr(project_routes, "_invalidate_agent_prompt_cache", lambda: None)
    profile = await project_routes.update_project_agent(
        first.project_id,
        project_routes.ProjectAgentProfileUpdate(
            instruction="Act as a marketing researcher and verify sources."
        ),
    )
    created = await project_routes.create_knowledge_pack(
        first.project_id,
        project_routes.KnowledgePackCreate(
            name="Market references", description="Approved market briefs"
        ),
    )
    pack_id = created["pack"]["pack_id"]
    uploaded = await project_routes.upload_knowledge_pack_source(
        first.project_id,
        pack_id,
        UploadFile(
            filename="brief.md",
            file=BytesIO(b"Retention is the primary launch metric."),
        ),
    )
    current = await project_routes.get_project_agent(first.project_id)

    assert profile["profile"]["instruction"].startswith("Act as a marketing")
    assert uploaded["source"]["name"] == "brief.md"
    assert current["knowledge_packs"][0]["sources"][0]["name"] == "brief.md"
