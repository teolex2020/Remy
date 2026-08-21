"""Project-owned conversation lifecycle and transcript API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from remy.core.conversation_store import ConversationRecord, get_conversation_store
from remy.core.microbrain import current_project_id
from remy.core.project_store import LEGACY_PROJECT_ID
from remy.core.transcript_store import get_transcript_store
from remy.web.routes._helpers import _TIMEOUT_FAST, _get_api, run_in_thread

router = APIRouter()


class ConversationCreate(BaseModel):
    title: str = Field(default="New conversation", max_length=120)


class ConversationUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=120)


def _serialize(record: ConversationRecord, *, active_id: str) -> dict:
    return {
        **record.to_dict(),
        "active": record.conversation_id == active_id,
    }


def _current_store():
    return get_conversation_store(current_project_id())


@router.get("/conversations")
async def list_conversations(include_archived: bool = False):
    project_id = current_project_id()
    manager = _get_api().get_session_manager()
    session = manager.get_or_create_session(project_id)
    store = get_conversation_store(project_id)
    active_id = store.get_active_id() or session.session_id
    return {
        "project_id": project_id,
        "active_conversation_id": active_id,
        "conversations": [
            _serialize(record, active_id=active_id)
            for record in store.list(include_archived=include_archived)
        ],
    }


@router.post("/conversations")
async def create_conversation(payload: ConversationCreate):
    project_id = current_project_id()
    store = get_conversation_store(project_id)
    record = store.create(payload.title)
    session = await _get_api().get_session_manager().switch_conversation(
        project_id,
        record.conversation_id,
    )
    return {
        "conversation": _serialize(record, active_id=session.session_id),
        "active_conversation_id": session.session_id,
        "chat_reset": True,
    }


@router.post("/conversations/{conversation_id}/activate")
async def activate_conversation(conversation_id: str):
    project_id = current_project_id()
    store = get_conversation_store(project_id)
    try:
        record = store.require(conversation_id)
        session = await _get_api().get_session_manager().switch_conversation(
            project_id,
            record.conversation_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    return {
        "conversation": _serialize(record, active_id=session.session_id),
        "active_conversation_id": session.session_id,
        "chat_reset": True,
    }


@router.patch("/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    payload: ConversationUpdate,
):
    store = _current_store()
    try:
        record = store.update(conversation_id, title=payload.title)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "conversation": _serialize(record, active_id=store.get_active_id()),
    }


@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str, limit: int = 500):
    project_id = current_project_id()

    def _load_transcript():
        store = get_conversation_store(project_id)
        record = store.require(conversation_id)
        # Read one extra row so the UI can progressively reveal older history
        # instead of eagerly shipping and rendering hundreds of messages on
        # every conversation switch.
        safe_limit = max(1, min(int(limit), 1999))
        items = get_transcript_store().list_session(
            record.conversation_id,
            owner_project_id=record.project_id,
            include_legacy_unscoped=record.project_id == LEGACY_PROJECT_ID,
            limit=safe_limit + 1,
        )
        active_id = store.get_active_id()
        return record, active_id, safe_limit, items

    try:
        record, active_id, safe_limit, items = await run_in_thread(
            _load_transcript,
            timeout=_TIMEOUT_FAST,
            error_msg="Conversation history timed out",
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    has_more = len(items) > safe_limit
    if has_more:
        items = items[-safe_limit:]
    return {
        "conversation": _serialize(record, active_id=active_id),
        "messages": items,
        "has_more": has_more,
        "limit": safe_limit,
    }


@router.delete("/conversations/{conversation_id}")
async def archive_conversation(conversation_id: str):
    project_id = current_project_id()
    store = get_conversation_store(project_id)
    manager = _get_api().get_session_manager()
    try:
        record = store.require(conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc

    was_active = store.get_active_id() == record.conversation_id
    if (
        manager.session is not None
        and manager.session.project_id == project_id
        and manager.session.session_id == record.conversation_id
    ):
        await manager.unload_session()
    archived = store.archive(record.conversation_id)

    active_id = store.get_active_id()
    if was_active or not active_id:
        remaining = store.list()
        replacement = remaining[0] if remaining else store.create()
        session = await manager.switch_conversation(
            project_id,
            replacement.conversation_id,
        )
        active_id = session.session_id

    return {
        "conversation": _serialize(archived, active_id=active_id),
        "active_conversation_id": active_id,
        "archived": True,
        "chat_reset": was_active,
    }
