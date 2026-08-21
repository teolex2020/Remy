from __future__ import annotations

import asyncio

import pytest

from remy.core.conversation_store import ConversationStore
from remy.core.project_store import get_project_store, reset_project_store_for_tests
from remy.core.transcript_store import TranscriptStore


@pytest.fixture
def isolated_projects(tmp_path, monkeypatch):
    from remy.config.settings import settings

    data_dir = tmp_path / "data"
    monkeypatch.setattr(settings, "DATA_DIR", data_dir)
    monkeypatch.setattr(settings, "AURA_BRAIN_PATH", data_dir / "brain")
    reset_project_store_for_tests()
    store = get_project_store()
    first = store.create_project("First")
    second = store.create_project("Second")
    yield store, first, second
    reset_project_store_for_tests()


def test_conversation_catalogs_are_physically_project_scoped(isolated_projects):
    _, first, second = isolated_projects
    first_store = ConversationStore(first.project_id)
    second_store = ConversationStore(second.project_id)

    created = first_store.create("Private planning")

    assert first_store.root == first_store.root.parent / "conversations"
    assert first.project_id in str(first_store.catalog_path)
    assert second.project_id in str(second_store.catalog_path)
    assert first_store.require(created.conversation_id).brain_id == first.brain_id
    assert second_store.get(created.conversation_id) is None


def test_conversation_can_be_created_as_inactive_fork(isolated_projects):
    _, first, _ = isolated_projects
    store = ConversationStore(first.project_id)
    source = store.create("Source")

    fork = store.create(
        "Fork",
        activate=False,
        metadata={"fork": {"source_conversation_id": source.conversation_id}},
    )

    assert store.get_active_id() == source.conversation_id
    assert fork.metadata == {
        "fork": {"source_conversation_id": source.conversation_id}
    }


def test_transcript_reads_require_the_owning_project(tmp_path):
    store = TranscriptStore(tmp_path / "transcripts.sqlite3")
    conversation_id = "9ec6ed69-95f5-4a46-a7d5-4ad05bd3dbcf"
    store.append(
        session_id=conversation_id,
        owner_project_id="project-one",
        brain_id="brain-one",
        role="user",
        content="first project secret",
    )
    store.append(
        session_id=conversation_id,
        owner_project_id="project-two",
        brain_id="brain-two",
        role="assistant",
        content="second project answer",
    )

    first = store.list_session(
        conversation_id,
        owner_project_id="project-one",
    )
    second = store.list_session(
        conversation_id,
        owner_project_id="project-two",
    )

    assert [item["content"] for item in first] == ["first project secret"]
    assert [item["content"] for item in second] == ["second project answer"]
    assert store.search(
        "second project answer",
        owner_project_id="project-one",
    ) == []


def test_transcript_limit_returns_latest_messages_in_chat_order(tmp_path):
    store = TranscriptStore(tmp_path / "transcripts.sqlite3")
    conversation_id = "f5cf13d4-7f59-48cb-8ff2-c132ecab18ee"
    for index in range(5):
        store.append(
            session_id=conversation_id,
            owner_project_id="project-one",
            brain_id="brain-one",
            role="user" if index % 2 == 0 else "assistant",
            content=f"message {index}",
        )

    latest = store.list_session(
        conversation_id,
        owner_project_id="project-one",
        limit=2,
    )

    assert [item["content"] for item in latest] == ["message 3", "message 4"]


def test_first_user_message_titles_the_conversation(isolated_projects):
    _, first, _ = isolated_projects
    store = ConversationStore(first.project_id)
    created = store.create()

    store.touch_from_user_message(
        created.conversation_id,
        "  Design   a project-scoped memory architecture  ",
    )

    assert (
        store.require(created.conversation_id).title
        == "Design a project-scoped memory architecture"
    )


def test_archiving_active_chat_keeps_transcript_and_clears_pointer(
    isolated_projects,
):
    _, first, _ = isolated_projects
    store = ConversationStore(first.project_id)
    created = store.create("Keep this history")

    archived = store.archive(created.conversation_id)

    assert archived.archived_at
    assert store.get_active_id() == ""
    assert store.get(created.conversation_id, include_archived=True) is not None
    assert store.list() == []


def test_session_switch_preserves_both_project_conversations(
    isolated_projects,
    monkeypatch,
):
    from remy.web.session import WebSessionManager

    project_store, first, _ = isolated_projects
    project_store.set_active_project(first.project_id)
    manager = WebSessionManager()
    original = manager.get_or_create_session(first.project_id)
    other = ConversationStore(first.project_id).create("Second chat")

    mounted = asyncio.run(
        manager.switch_conversation(first.project_id, other.conversation_id)
    )

    catalog = ConversationStore(first.project_id)
    assert mounted.session_id == other.conversation_id
    assert catalog.get(original.session_id) is not None
    assert catalog.get(other.conversation_id) is not None
    assert catalog.get_active_id() == other.conversation_id


@pytest.mark.asyncio
async def test_conversation_api_lifecycle_stays_inside_active_project(
    isolated_projects,
    monkeypatch,
):
    from remy.web.routes import conversation_routes
    from remy.web.session import WebSessionManager

    project_store, first, second = isolated_projects
    project_store.set_active_project(first.project_id)
    manager = WebSessionManager()

    class _Api:
        @staticmethod
        def get_session_manager():
            return manager

    monkeypatch.setattr(conversation_routes, "_get_api", lambda: _Api())

    initial = await conversation_routes.list_conversations()
    created = await conversation_routes.create_conversation(
        conversation_routes.ConversationCreate(title="Architecture")
    )
    conversation_id = created["conversation"]["conversation_id"]
    get_transcript_store = conversation_routes.get_transcript_store
    get_transcript_store().append(
        session_id=conversation_id,
        owner_project_id=first.project_id,
        brain_id=first.brain_id,
        role="user",
        content="Only the first project may read this",
    )

    messages = await conversation_routes.get_conversation_messages(conversation_id)
    get_transcript_store().append(
        session_id=conversation_id,
        owner_project_id=first.project_id,
        brain_id=first.brain_id,
        role="assistant",
        content="A newer reply",
    )
    limited_messages = await conversation_routes.get_conversation_messages(
        conversation_id,
        limit=1,
    )
    renamed = await conversation_routes.update_conversation(
        conversation_id,
        conversation_routes.ConversationUpdate(title="Architecture review"),
    )

    assert initial["project_id"] == first.project_id
    assert created["active_conversation_id"] == conversation_id
    assert [item["content"] for item in messages["messages"]] == [
        "Only the first project may read this"
    ]
    assert [item["content"] for item in limited_messages["messages"]] == [
        "A newer reply"
    ]
    assert limited_messages["has_more"] is True
    assert limited_messages["limit"] == 1
    assert renamed["conversation"]["title"] == "Architecture review"
    assert ConversationStore(second.project_id).get(conversation_id) is None

    archived = await conversation_routes.archive_conversation(conversation_id)
    assert archived["archived"] is True
    assert archived["active_conversation_id"] != conversation_id


def test_frontend_exposes_project_chat_controls():
    from pathlib import Path

    html = Path("src/remy/web/static/index.html").read_text(encoding="utf-8")
    app_js = Path("src/remy/web/static/js/app.js").read_text(encoding="utf-8")
    chat_js = Path("src/remy/web/static/js/chat.js").read_text(encoding="utf-8")
    api_js = Path("src/remy/web/static/js/api-client.js").read_text(encoding="utf-8")

    assert 'id="conversation-list"' in html
    assert 'id="chat-conversation-title"' in html
    assert 'id="project-create-domain"' in html
    assert 'id="project-create-description"' in html
    assert "shared memory for all its chats" in html
    assert "activateConversation" in app_js
    assert "archiveConversation" in app_js
    assert "shared project memory" in app_js
    assert "loadConversationTranscript" in chat_js
    assert "document.createDocumentFragment()" in chat_js
    assert "CONVERSATION_INITIAL_LIMIT = 120" in chat_js
    assert "controller.abort(), 8000" in chat_js
    assert 'error?.name === "AbortError"' in chat_js
    assert "skipTranscriptReload" in app_js
    assert "await loadConversations({ emitActive: false });\n                await switchView" not in app_js
    assert "limit = 120" in api_js
    assert '"/api/conversations"' in api_js
