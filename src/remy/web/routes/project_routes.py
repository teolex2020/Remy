"""Project and MicroBrain lifecycle API."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from remy.core.microbrain import bind_project
from remy.core.project_store import (
    LEGACY_PROJECT_ID,
    LOCAL_BRAIN_PROVIDER,
    ProjectRecord,
    get_project_store,
)
from remy.web.routes._helpers import _get_api

router = APIRouter()


class ProjectCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    domain: str = Field(default="", max_length=80)
    description: str = Field(default="", max_length=2000)
    workspace_id: str = Field(default="", max_length=160)
    activate: bool = True


class ProjectUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=120)
    domain: str | None = Field(default=None, max_length=80)
    description: str | None = Field(default=None, max_length=2000)
    workspace_id: str | None = Field(default=None, max_length=160)


class ProjectAgentProfileUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str | None = Field(default=None, max_length=6000)


class KnowledgePackCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)


class KnowledgePackUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=500)
    enabled: bool | None = None


def _invalidate_agent_prompt_cache() -> None:
    try:
        from remy.core.agent import invalidate_system_instruction_cache

        invalidate_system_instruction_cache()
    except Exception:
        pass


def _serialize(project: ProjectRecord, *, active_project_id: str) -> dict:
    return {
        "project_id": project.project_id,
        "brain_id": project.brain_id,
        "name": project.name,
        "domain": getattr(project, "domain", ""),
        "description": getattr(project, "description", ""),
        "brain_uri": project.brain_uri,
        "brain_provider": project.brain_provider,
        "brain_location": (
            "local"
            if project.brain_provider == LOCAL_BRAIN_PROVIDER
            else "server"
        ),
        "workspace_id": project.workspace_id,
        "created_at": project.created_at,
        "updated_at": project.updated_at,
        "archived_at": project.archived_at,
        "legacy": project.legacy,
        "active": project.project_id == active_project_id,
    }


async def _activate_project(project_id: str) -> tuple[ProjectRecord, str]:
    store = get_project_store()
    target = store.require_project(project_id, include_archived=False)
    new_session_id = ""

    try:
        manager = _get_api().get_session_manager()
    except RuntimeError:
        manager = None

    if manager is not None and manager.session is not None:
        active_session = manager.session
        if active_session.project_id != target.project_id:
            # Preserve the old project's active conversation exactly as-is.
            with bind_project(active_session.project_id):
                await manager.unload_session()

    store.set_active_project(target.project_id)
    if manager is not None:
        new_session_id = manager.get_or_create_session(target.project_id).session_id
    return target, new_session_id


@router.get("/projects")
async def list_projects(include_archived: bool = False):
    store = get_project_store()
    active_id = store.get_active_project().project_id
    projects = store.list_projects(include_archived=include_archived)
    return {
        "active_project_id": active_id,
        "projects": [
            _serialize(project, active_project_id=active_id)
            for project in projects
        ],
    }


@router.post("/projects")
async def create_project(payload: ProjectCreate):
    store = get_project_store()
    try:
        project = store.create_project(
            payload.name,
            domain=payload.domain,
            description=payload.description,
            workspace_id=payload.workspace_id,
        )
        new_session_id = ""
        if payload.activate:
            project, new_session_id = await _activate_project(project.project_id)
        active_id = store.get_active_project().project_id
        return {
            "project": _serialize(project, active_project_id=active_id),
            "session_id": new_session_id,
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/projects/{project_id}")
async def get_project(project_id: str):
    store = get_project_store()
    project = store.get_project(project_id, include_archived=True)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    active_id = store.get_active_project().project_id
    return {"project": _serialize(project, active_project_id=active_id)}


@router.patch("/projects/{project_id}")
async def update_project(project_id: str, payload: ProjectUpdate):
    try:
        store = get_project_store()
        project = store.update_project(
            project_id,
            name=payload.name,
            domain=payload.domain,
            description=payload.description,
            workspace_id=payload.workspace_id,
        )
        try:
            from remy.core.agent import invalidate_system_instruction_cache

            invalidate_system_instruction_cache()
        except Exception:
            # Project metadata is already persisted. A cache miss on the next
            # process start still restores the correct workspace context.
            pass
        active_id = store.get_active_project().project_id
        return {"project": _serialize(project, active_project_id=active_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/projects/{project_id}/agent")
async def get_project_agent(project_id: str):
    from remy.core.project_agent import get_project_agent_store

    try:
        return get_project_agent_store(project_id).get()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.put("/projects/{project_id}/agent")
async def update_project_agent(
    project_id: str,
    payload: ProjectAgentProfileUpdate,
):
    from remy.core.project_agent import get_project_agent_store

    try:
        result = get_project_agent_store(project_id).update_profile(
            payload.model_dump(exclude_unset=True)
        )
        _invalidate_agent_prompt_cache()
        return result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/projects/{project_id}/knowledge-packs")
async def create_knowledge_pack(project_id: str, payload: KnowledgePackCreate):
    from remy.core.project_agent import get_project_agent_store

    try:
        pack = get_project_agent_store(project_id).create_pack(
            payload.name, payload.description
        )
        return {"pack": pack}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.patch("/projects/{project_id}/knowledge-packs/{pack_id}")
async def update_knowledge_pack(
    project_id: str,
    pack_id: str,
    payload: KnowledgePackUpdate,
):
    from remy.core.project_agent import get_project_agent_store

    try:
        pack = get_project_agent_store(project_id).update_pack(
            pack_id, payload.model_dump(exclude_unset=True)
        )
        return {"pack": pack}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Knowledge Pack not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete("/projects/{project_id}/knowledge-packs/{pack_id}")
async def delete_knowledge_pack(project_id: str, pack_id: str):
    from remy.core.project_agent import get_project_agent_store

    try:
        get_project_agent_store(project_id).delete_pack(pack_id)
        return {"ok": True, "deleted": pack_id}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Knowledge Pack not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/projects/{project_id}/knowledge-packs/{pack_id}/sources")
async def upload_knowledge_pack_source(
    project_id: str,
    pack_id: str,
    file: UploadFile = File(...),
):
    from remy.core.project_agent import (
        ALLOWED_SOURCE_EXTENSIONS,
        MAX_SOURCE_BYTES,
        PLAIN_TEXT_SOURCE_EXTENSIONS,
        get_project_agent_store,
    )

    data = await file.read(MAX_SOURCE_BYTES + 1)
    if len(data) > MAX_SOURCE_BYTES:
        raise HTTPException(status_code=413, detail="Knowledge source is larger than 5 MB")
    if not data:
        raise HTTPException(status_code=422, detail="Knowledge source is empty")
    try:
        filename = Path(file.filename or "").name
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SOURCE_EXTENSIONS:
            raise ValueError(
                "Unsupported knowledge source. Allowed: "
                + ", ".join(sorted(ALLOWED_SOURCE_EXTENSIONS))
            )
        store = get_project_agent_store(project_id)
        if suffix in PLAIN_TEXT_SOURCE_EXTENSIONS:
            source = store.add_source(pack_id, filename, data)
        else:
            from remy.core.corpus_preprocessor import extract_clean_text

            with tempfile.TemporaryDirectory(prefix="remy-knowledge-") as directory:
                path = Path(directory) / f"source{suffix}"
                path.write_bytes(data)
                content, _warnings = extract_clean_text(
                    path,
                    max_bytes_per_file=MAX_SOURCE_BYTES,
                    extractor_provider="built_in",
                )
            source = store.add_extracted_source(
                pack_id,
                filename,
                str(content or "")[:500_000],
                original_size=len(data),
            )
        return {"source": source}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Knowledge Pack not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete(
    "/projects/{project_id}/knowledge-packs/{pack_id}/sources/{source_id}"
)
async def delete_knowledge_pack_source(
    project_id: str,
    pack_id: str,
    source_id: str,
):
    from remy.core.project_agent import get_project_agent_store

    try:
        get_project_agent_store(project_id).delete_source(pack_id, source_id)
        return {"ok": True, "deleted": source_id}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Knowledge source not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/projects/{project_id}/activate")
async def activate_project(project_id: str):
    try:
        project, session_id = await _activate_project(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    active_id = get_project_store().get_active_project().project_id
    return {
        "project": _serialize(project, active_project_id=active_id),
        "session_id": session_id,
        "chat_reset": True,
    }


@router.delete("/projects/{project_id}")
async def archive_project(project_id: str):
    store = get_project_store()
    try:
        target = store.require_project(project_id, include_archived=True)
        new_session_id = ""
        chat_reset = False
        if not target.archived_at and store.get_active_project().project_id == project_id:
            # Move the live session away before making its project unavailable.
            # This flushes the conversation while its original MicroBrain is
            # still mounted and prevents a stale WebSession from crossing scope.
            _, new_session_id = await _activate_project(LEGACY_PROJECT_ID)
            chat_reset = True
        project = store.archive_project(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    active_id = store.get_active_project().project_id
    return {
        "project": _serialize(project, active_project_id=active_id),
        "archived": True,
        "brain_deleted": False,
        "active_project_id": active_id,
        "session_id": new_session_id,
        "chat_reset": chat_reset,
    }


@router.post("/projects/{project_id}/restore")
async def restore_project(project_id: str):
    store = get_project_store()
    try:
        project = store.restore_project(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    active_id = store.get_active_project().project_id
    return {
        "project": _serialize(project, active_project_id=active_id),
        "restored": True,
        "brain_restored": True,
        "active_project_id": active_id,
    }
