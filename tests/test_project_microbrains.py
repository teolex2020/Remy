from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from remy.core.conversation_store import ConversationStore
from remy.core.execution_ledger import ExecutionLedger
from remy.core.microbrain import MicroBrainRegistry, bind_project
from remy.core.project_store import (
    LEGACY_PROJECT_ID,
    LOCAL_BRAIN_PROVIDER,
    SERVER_BRAIN_PROVIDER,
    ProjectStore,
    get_project_store,
    local_brain_path,
    project_data_root,
    reset_project_store_for_tests,
)
from remy.core.transcript_store import TranscriptStore


@pytest.fixture(autouse=True)
def _reset_project_singleton():
    reset_project_store_for_tests()
    yield
    reset_project_store_for_tests()


def test_legacy_brain_becomes_a_project_without_moving_data(tmp_path):
    data_dir = tmp_path / "data"
    legacy_brain = data_dir / "brain"
    legacy_brain.mkdir(parents=True)
    marker = legacy_brain / "existing-memory.bin"
    marker.write_bytes(b"keep-me")

    store = ProjectStore(data_dir=data_dir, legacy_brain_path=legacy_brain)
    legacy = store.require_project(LEGACY_PROJECT_ID)

    assert legacy.legacy is True
    assert legacy.brain_path == str(legacy_brain.resolve())
    assert store.get_active_project().project_id == LEGACY_PROJECT_ID
    assert marker.read_bytes() == b"keep-me"


def test_each_project_gets_a_unique_microbrain_boundary(tmp_path):
    store = ProjectStore(
        data_dir=tmp_path / "data",
        legacy_brain_path=tmp_path / "data" / "brain",
    )

    first = store.create_project("First")
    second = store.create_project("Second")

    assert first.project_id != second.project_id
    assert first.brain_id != second.brain_id
    assert first.brain_path != second.brain_path
    assert (tmp_path / "data" / "projects" / first.project_id / "brain").is_dir()
    assert (tmp_path / "data" / "projects" / second.project_id / "brain").is_dir()


def test_old_catalog_records_default_to_local_provider(tmp_path):
    data_dir = tmp_path / "data"
    store = ProjectStore(
        data_dir=data_dir,
        legacy_brain_path=data_dir / "brain",
    )
    project = store.create_project("Existing")
    payload = json.loads(store.catalog_path.read_text(encoding="utf-8"))
    for item in payload["projects"]:
        item.pop("brain_provider", None)
        item.pop("brain_locator", None)
        item.pop("domain", None)
        item.pop("description", None)
    store.catalog_path.write_text(json.dumps(payload), encoding="utf-8")

    reopened = ProjectStore(
        data_dir=data_dir,
        legacy_brain_path=data_dir / "brain",
    )
    migrated = reopened.require_project(project.project_id)

    assert migrated.brain_provider == LOCAL_BRAIN_PROVIDER
    assert migrated.brain_locator == migrated.brain_path
    assert migrated.brain_uri == f"microbrain://{project.brain_id}/"
    assert migrated.domain == ""
    assert migrated.description == ""


def test_project_profile_is_shared_workspace_context_not_chat_metadata(
    tmp_path,
    monkeypatch,
):
    from remy.config.settings import settings

    data_dir = tmp_path / "data"
    monkeypatch.setattr(settings, "DATA_DIR", data_dir)
    monkeypatch.setattr(settings, "AURA_BRAIN_PATH", data_dir / "brain")
    reset_project_store_for_tests()
    store = get_project_store()
    project = store.create_project(
        "Cell research",
        domain="Biology",
        description="Study cell regeneration and keep source-backed findings.",
    )
    conversation = ConversationStore(project.project_id).create("Literature review")

    reopened = ProjectStore(
        data_dir=data_dir,
        legacy_brain_path=data_dir / "brain",
    ).require_project(project.project_id)

    assert reopened.domain == "Biology"
    assert reopened.description == (
        "Study cell regeneration and keep source-backed findings."
    )
    assert conversation.project_id == project.project_id
    assert conversation.brain_id == project.brain_id
    assert "domain" not in (conversation.metadata or {})


def test_catalog_rejects_microbrain_path_escape(tmp_path):
    data_dir = tmp_path / "data"
    store = ProjectStore(
        data_dir=data_dir,
        legacy_brain_path=data_dir / "brain",
    )
    project = store.create_project("Contained")

    payload = json.loads(store.catalog_path.read_text(encoding="utf-8"))
    for item in payload["projects"]:
        if item["project_id"] == project.project_id:
            item["brain_path"] = str((tmp_path / "escaped-brain").resolve())
    store.catalog_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="escapes its project boundary"):
        store.list_projects()


@pytest.mark.parametrize(
    ("marker_payload", "message"),
    [
        ("{not-json", "unreadable"),
        ("{}", "has no project_id"),
        ('{"project_id":"project-does-not-exist"}', "unavailable"),
    ],
)
def test_active_project_marker_fails_closed(tmp_path, marker_payload, message):
    data_dir = tmp_path / "data"
    store = ProjectStore(
        data_dir=data_dir,
        legacy_brain_path=data_dir / "brain",
    )
    store.active_path.write_text(marker_payload, encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        store.get_active_project()


def test_missing_active_project_marker_does_not_fall_back_to_legacy(tmp_path):
    data_dir = tmp_path / "data"
    store = ProjectStore(
        data_dir=data_dir,
        legacy_brain_path=data_dir / "brain",
    )
    store.active_path.unlink()

    with pytest.raises(RuntimeError, match="marker is missing"):
        store.get_active_project()


def test_brain_status_reports_broken_project_context_without_claiming_legacy():
    from remy.core.agent_tools import get_brain_startup_status

    with patch(
        "remy.core.agent_tools.current_project_id",
        side_effect=RuntimeError("active project marker is broken"),
    ):
        status = get_brain_startup_status()

    assert status["active_project_id"] == ""
    assert status["project_context_error"] == "active project marker is broken"
    assert status["microbrain_host"]["ownership_known"] is False


def test_bound_project_profile_is_injected_without_cross_project_context(
    tmp_path,
    monkeypatch,
):
    from remy.config.settings import settings
    from remy.core.system_instruction import _build_project_workspace_context

    data_dir = tmp_path / "data"
    monkeypatch.setattr(settings, "DATA_DIR", data_dir)
    monkeypatch.setattr(settings, "AURA_BRAIN_PATH", data_dir / "brain")
    reset_project_store_for_tests()
    store = get_project_store()
    marketing = store.create_project(
        "Launch",
        domain="Marketing",
        description="Prepare the product launch.",
    )
    biology = store.create_project(
        "Cells",
        domain="Biology",
        description="Analyze regeneration experiments.",
    )

    with bind_project(marketing.project_id):
        context = _build_project_workspace_context()

    assert "Project: Launch" in context
    assert "Area: Marketing" in context
    assert "Purpose: Prepare the product launch." in context
    assert "chats share one project memory" in context
    assert "Biology" not in context
    assert biology.name not in context


def test_legacy_in_memory_project_without_profile_does_not_break_chat_context(
    monkeypatch,
):
    from types import SimpleNamespace

    from remy.core import project_store
    from remy.core.system_instruction import _build_project_workspace_context

    legacy_record = SimpleNamespace(name="Legacy Workspace")
    legacy_store = SimpleNamespace(
        require_project=lambda _project_id: legacy_record,
    )
    monkeypatch.setattr(project_store, "get_project_store", lambda: legacy_store)

    context = _build_project_workspace_context()

    assert "Project: Legacy Workspace" in context
    assert "Area:" not in context
    assert "Purpose:" not in context


def test_durable_runtime_records_reject_ownerless_access(tmp_path):
    ledger = ExecutionLedger(tmp_path / "execution-ledger.sqlite3")
    transcripts = TranscriptStore(tmp_path / "transcripts.sqlite3")

    with pytest.raises(ValueError, match="owner_project_id and brain_id"):
        ledger.claim(kind="worker", job_id="ownerless")
    with pytest.raises(ValueError, match="owner_project_id and brain_id"):
        ledger.enqueue_continuation(
            session_id="session-test",
            kind="result",
            source_id="source-test",
            content="must not be stored",
        )
    with pytest.raises(ValueError, match="owner_project_id and brain_id"):
        ledger.consume_continuations("session-test")
    with pytest.raises(ValueError, match="owner_project_id and brain_id"):
        transcripts.append(
            session_id="session-test",
            role="user",
            content="must not be stored",
        )
    with pytest.raises(ValueError, match="requires owner_project_id"):
        transcripts.list_session("session-test", owner_project_id="")
    with pytest.raises(ValueError, match="requires owner_project_id"):
        transcripts.search("anything")


class _FakeBrain:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.records: list[str] = []
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _CloseFailingBrain(_FakeBrain):
    def close(self) -> None:
        raise RuntimeError("flush failed")


def test_registry_mounts_only_the_bound_project(tmp_path, monkeypatch):
    from remy.config.settings import settings

    data_dir = tmp_path / "data"
    legacy_brain = data_dir / "brain"
    monkeypatch.setattr(settings, "DATA_DIR", data_dir)
    monkeypatch.setattr(settings, "AURA_BRAIN_PATH", legacy_brain)
    reset_project_store_for_tests()

    store = ProjectStore(data_dir=data_dir, legacy_brain_path=legacy_brain)
    first = store.create_project("First")
    second = store.create_project("Second")
    registry = MicroBrainRegistry(
        lambda project: _FakeBrain(project.project_id),
        project_store=store,
    )

    with bind_project(first.project_id):
        first_brain = registry.get()
        first_brain.records.append("first-only")
    with bind_project(second.project_id):
        second_brain = registry.get()
        second_brain.records.append("second-only")

    assert first_brain is not second_brain
    assert first_brain.records == ["first-only"]
    assert second_brain.records == ["second-only"]

    assert registry.close_all() == []
    assert first_brain.closed is True
    assert second_brain.closed is True


def test_registry_evicts_the_least_recently_used_idle_microbrain(tmp_path):
    store = ProjectStore(
        data_dir=tmp_path / "data",
        legacy_brain_path=tmp_path / "data" / "brain",
    )
    first = store.create_project("First")
    second = store.create_project("Second")
    third = store.create_project("Third")
    registry = MicroBrainRegistry(
        lambda project: _FakeBrain(project.project_id),
        project_store=store,
        max_open=2,
    )

    first_brain = registry.get(first.project_id)
    second_brain = registry.get(second.project_id)
    assert registry.get(first.project_id) is first_brain
    third_brain = registry.get(third.project_id)

    status = registry.host_status()
    assert second_brain.closed is True
    assert first_brain.closed is False
    assert third_brain.closed is False
    assert status["mounted_projects"] == [first.project_id, third.project_id]
    assert status["mounted_count"] == 2
    assert status["over_capacity"] is False

    assert registry.close_all() == []


def test_registry_pins_active_and_bound_projects_until_scope_exits(
    tmp_path,
    monkeypatch,
):
    from remy.config.settings import settings

    data_dir = tmp_path / "data"
    legacy_brain = data_dir / "brain"
    monkeypatch.setattr(settings, "DATA_DIR", data_dir)
    monkeypatch.setattr(settings, "AURA_BRAIN_PATH", legacy_brain)
    reset_project_store_for_tests()

    store = get_project_store()
    first = store.create_project("First")
    second = store.create_project("Second")
    store.set_active_project(first.project_id)
    registry = MicroBrainRegistry(
        lambda project: _FakeBrain(project.project_id),
        project_store=store,
        max_open=1,
    )

    first_brain = registry.get(first.project_id)
    with bind_project(second.project_id):
        second_brain = registry.get()
        status = registry.host_status()
        assert status["over_capacity"] is True
        assert set(status["pinned_projects"]) == {
            first.project_id,
            second.project_id,
        }
        assert first_brain.closed is False
        assert second_brain.closed is False

    status = registry.host_status()
    assert status["mounted_projects"] == [first.project_id]
    assert status["over_capacity"] is False
    assert second_brain.closed is True
    assert first_brain.closed is False

    assert registry.close_all() == []


def test_registry_rejects_closing_a_microbrain_that_is_in_use(
    tmp_path,
    monkeypatch,
):
    from remy.config.settings import settings

    data_dir = tmp_path / "data"
    legacy_brain = data_dir / "brain"
    monkeypatch.setattr(settings, "DATA_DIR", data_dir)
    monkeypatch.setattr(settings, "AURA_BRAIN_PATH", legacy_brain)
    reset_project_store_for_tests()

    store = get_project_store()
    project = store.create_project("In use")
    registry = MicroBrainRegistry(
        lambda item: _FakeBrain(item.project_id),
        project_store=store,
    )
    brain = registry.get(project.project_id)

    with bind_project(project.project_id):
        with pytest.raises(RuntimeError, match="still in use"):
            registry.close_project(project.project_id)

    assert brain.closed is False
    assert registry.close_project(project.project_id) is True
    assert brain.closed is True


def test_registry_does_not_evict_when_active_project_ownership_is_unknown(
    tmp_path,
):
    store = ProjectStore(
        data_dir=tmp_path / "data",
        legacy_brain_path=tmp_path / "data" / "brain",
    )
    first = store.create_project("First")
    second = store.create_project("Second")
    third = store.create_project("Third")
    registry = MicroBrainRegistry(
        lambda project: _FakeBrain(project.project_id),
        project_store=store,
        max_open=2,
    )

    brains = [
        registry.get(first.project_id),
        registry.get(second.project_id),
    ]
    store.active_path.write_text("{not-json", encoding="utf-8")
    brains.append(registry.get(third.project_id))

    status = registry.host_status()
    assert status["ownership_known"] is False
    assert status["mounted_count"] == 3
    assert status["over_capacity"] is True
    assert all(brain.closed is False for brain in brains)

    assert registry.close_all() == []


def test_registry_retains_and_reports_a_microbrain_that_failed_to_close(
    tmp_path,
):
    store = ProjectStore(
        data_dir=tmp_path / "data",
        legacy_brain_path=tmp_path / "data" / "brain",
    )
    first = store.create_project("First")
    second = store.create_project("Second")

    def opener(project):
        if project.project_id == first.project_id:
            return _CloseFailingBrain(project.project_id)
        return _FakeBrain(project.project_id)

    registry = MicroBrainRegistry(
        opener,
        project_store=store,
        max_open=1,
    )
    first_brain = registry.get(first.project_id)
    registry.get(second.project_id)

    status = registry.host_status()
    assert first_brain.closed is False
    assert status["mounted_count"] == 2
    assert status["over_capacity"] is True
    assert status["eviction_errors"][-1] == {
        "project_id": first.project_id,
        "error": "flush failed",
    }

    errors = registry.close_all()
    assert len(errors) == 1
    assert errors[0][0] == first.project_id
    assert str(errors[0][1]) == "flush failed"


def test_registry_opens_remote_microbrain_through_registered_provider(tmp_path):
    from remy.web.routes.project_routes import _serialize

    store = ProjectStore(
        data_dir=tmp_path / "data",
        legacy_brain_path=tmp_path / "data" / "brain",
    )
    remote = store.create_project(
        "Server brain",
        brain_provider=SERVER_BRAIN_PROVIDER,
        brain_locator="tenant-user/project-brain",
    )
    opened = []

    def open_remote(project):
        opened.append((project.brain_provider, project.brain_locator))
        return _FakeBrain(project.project_id)

    registry = MicroBrainRegistry(
        providers={SERVER_BRAIN_PROVIDER: open_remote},
        project_store=store,
    )

    instance = registry.get(remote.project_id)

    assert instance.project_id == remote.project_id
    assert opened == [(SERVER_BRAIN_PROVIDER, "tenant-user/project-brain")]
    assert remote.brain_path == ""
    assert registry.available_providers() == [SERVER_BRAIN_PROVIDER]
    public = _serialize(remote, active_project_id=remote.project_id)
    assert public["brain_provider"] == SERVER_BRAIN_PROVIDER
    assert public["brain_location"] == "server"
    assert "tenant-user" not in json.dumps(public)


def test_registry_rejects_unconfigured_remote_provider(tmp_path):
    store = ProjectStore(
        data_dir=tmp_path / "data",
        legacy_brain_path=tmp_path / "data" / "brain",
    )
    remote = store.create_project(
        "Server brain",
        brain_provider=SERVER_BRAIN_PROVIDER,
        brain_locator="tenant-user/project-brain",
    )
    registry = MicroBrainRegistry(
        lambda project: _FakeBrain(project.project_id),
        project_store=store,
    )

    with pytest.raises(RuntimeError, match="provider is not configured"):
        registry.get(remote.project_id)


def test_registry_can_refresh_only_one_provider(tmp_path):
    store = ProjectStore(
        data_dir=tmp_path / "data",
        legacy_brain_path=tmp_path / "data" / "brain",
    )
    local = store.create_project("Local")
    remote = store.create_project(
        "Server",
        brain_provider=SERVER_BRAIN_PROVIDER,
        brain_locator="https://memory.example.test",
    )
    registry = MicroBrainRegistry(
        providers={
            LOCAL_BRAIN_PROVIDER: lambda project: _FakeBrain(project.project_id),
            SERVER_BRAIN_PROVIDER: lambda project: _FakeBrain(project.project_id),
        },
        project_store=store,
    )
    local_brain = registry.get(local.project_id)
    remote_brain = registry.get(remote.project_id)

    assert registry.close_provider(SERVER_BRAIN_PROVIDER) == []
    assert remote_brain.closed is True
    assert local_brain.closed is False
    assert registry.is_initialized(remote.project_id) is False
    assert registry.is_initialized(local.project_id) is True


def test_remote_memory_keeps_local_project_artifacts_without_fake_brain_path(
    tmp_path,
    monkeypatch,
):
    from remy.config.settings import settings

    data_dir = tmp_path / "data"
    monkeypatch.setattr(settings, "DATA_DIR", data_dir)
    monkeypatch.setattr(settings, "AURA_BRAIN_PATH", data_dir / "brain")
    reset_project_store_for_tests()
    remote = get_project_store().create_project(
        "Server brain",
        brain_provider=SERVER_BRAIN_PROVIDER,
        brain_locator="tenant-user/project-brain",
    )

    assert project_data_root(remote.project_id) == (
        data_dir / "projects" / remote.project_id
    ).resolve()
    with pytest.raises(RuntimeError, match="has no local brain path"):
        local_brain_path(remote.project_id)


def test_real_aura_stores_are_physically_isolated(tmp_path):
    from aura import Aura

    store = ProjectStore(
        data_dir=tmp_path / "data",
        legacy_brain_path=tmp_path / "data" / "brain",
    )
    first = store.create_project("First")
    second = store.create_project("Second")
    registry = MicroBrainRegistry(
        lambda project: Aura(project.brain_path),
        project_store=store,
    )

    first_brain = registry.get(first.project_id)
    second_brain = registry.get(second.project_id)
    first_brain.store("secret belonging only to project one", tags=["isolation-test"])

    assert len(first_brain.search(query="", tags=["isolation-test"], limit=10)) == 1
    assert second_brain.search(query="", tags=["isolation-test"], limit=10) == []
    assert registry.close_all() == []


def test_experiments_are_stored_inside_their_project_boundary(tmp_path, monkeypatch):
    import remy.core.experiment_lab as experiment_lab
    from remy.config.settings import settings

    data_dir = tmp_path / "data"
    monkeypatch.setattr(settings, "DATA_DIR", data_dir)
    monkeypatch.setattr(settings, "AURA_BRAIN_PATH", data_dir / "brain")
    reset_project_store_for_tests()
    experiment_lab._engines.clear()

    store = ProjectStore(data_dir=data_dir, legacy_brain_path=data_dir / "brain")
    first = store.create_project("First")
    second = store.create_project("Second")
    first_engine = experiment_lab.get_experiment_engine(first.project_id)
    second_engine = experiment_lab.get_experiment_engine(second.project_id)
    created = first_engine.store.create(
        title="Private experiment",
        problem="Only project one can see this",
        success_criteria="",
        models=["model-a"],
    )

    assert first_engine is not second_engine
    assert first_engine.store.root == Path(first.brain_path).parent / "experiments"
    assert second_engine.store.root == Path(second.brain_path).parent / "experiments"
    assert first_engine.store.get(created["experiment_id"]) is not None
    assert second_engine.store.get(created["experiment_id"]) is None
    assert created["owner_project_id"] == first.project_id
    assert created["brain_id"] == first.brain_id


def test_web_session_never_resumes_under_another_project(tmp_path, monkeypatch):
    from remy.config.settings import settings
    from remy.web.session import WebSessionManager

    data_dir = tmp_path / "data"
    legacy_brain = data_dir / "brain"
    monkeypatch.setattr(settings, "DATA_DIR", data_dir)
    monkeypatch.setattr(settings, "AURA_BRAIN_PATH", legacy_brain)
    reset_project_store_for_tests()

    store = ProjectStore(data_dir=data_dir, legacy_brain_path=legacy_brain)
    first = store.create_project("First")
    second = store.create_project("Second")
    store.set_active_project(first.project_id)

    first_manager = WebSessionManager()
    first_session = first_manager.get_or_create_session(first.project_id)

    restored_manager = WebSessionManager()
    restored = restored_manager.get_or_create_session(first.project_id)
    assert restored.session_id == first_session.session_id
    assert restored.project_id == first.project_id

    store.set_active_project(second.project_id)
    second_manager = WebSessionManager()
    second_session = second_manager.get_or_create_session(second.project_id)
    assert second_session.session_id != first_session.session_id
    assert second_session.project_id == second.project_id

    with pytest.raises(RuntimeError, match="another project"):
        first_manager.get_or_create_session(second.project_id)


@pytest.mark.asyncio
async def test_project_api_creates_and_activates_microbrain(tmp_path, monkeypatch):
    from remy.config.settings import settings
    from remy.web.routes import project_routes

    data_dir = tmp_path / "data"
    monkeypatch.setattr(settings, "DATA_DIR", data_dir)
    monkeypatch.setattr(settings, "AURA_BRAIN_PATH", data_dir / "brain")
    reset_project_store_for_tests()

    class _ApiWithoutSessionManager:
        @staticmethod
        def get_session_manager():
            raise RuntimeError("not initialized")

    monkeypatch.setattr(project_routes, "_get_api", lambda: _ApiWithoutSessionManager())
    created = await project_routes.create_project(
        project_routes.ProjectCreate(
            name="Remy",
            domain="Marketing",
            description="Coordinate launch research and documents.",
            activate=True,
        )
    )
    project = created["project"]

    assert project["active"] is True
    assert project["brain_uri"].startswith("microbrain://brain-")
    assert project["brain_provider"] == LOCAL_BRAIN_PROVIDER
    assert project["brain_location"] == "local"
    assert project["domain"] == "Marketing"
    assert project["description"] == "Coordinate launch research and documents."
    assert created["session_id"] == ""

    listed = await project_routes.list_projects()
    assert listed["active_project_id"] == project["project_id"]
    assert {item["name"] for item in listed["projects"]} == {
        "Legacy Workspace",
        "Remy",
    }


@pytest.mark.asyncio
async def test_project_api_creates_from_title_without_optional_profile(
    tmp_path,
    monkeypatch,
):
    from remy.config.settings import settings
    from remy.web.routes import project_routes

    data_dir = tmp_path / "data"
    monkeypatch.setattr(settings, "DATA_DIR", data_dir)
    monkeypatch.setattr(settings, "AURA_BRAIN_PATH", data_dir / "brain")
    reset_project_store_for_tests()

    created = await project_routes.create_project(
        project_routes.ProjectCreate.model_validate(
            {"name": "Marketing", "activate": False}
        )
    )

    assert created["project"]["name"] == "Marketing"
    assert created["project"]["domain"] == ""
    assert created["project"]["description"] == ""
    assert created["project"]["active"] is False


def test_public_project_create_rejects_memory_provider_overrides():
    from pydantic import ValidationError

    from remy.web.routes.project_routes import ProjectCreate, ProjectUpdate

    with pytest.raises(ValidationError, match="brain_provider"):
        ProjectCreate.model_validate(
            {
                "name": "Remote work",
                "brain_provider": SERVER_BRAIN_PROVIDER,
                "brain_locator": "https://memory.example.test",
            }
        )
    with pytest.raises(ValidationError, match="brain_locator"):
        ProjectUpdate.model_validate(
            {"name": "Renamed", "brain_locator": "https://memory.example.test"}
        )


@pytest.mark.asyncio
async def test_active_project_archive_flushes_session_and_restore_keeps_brain(
    tmp_path,
    monkeypatch,
):
    from remy.config.settings import settings
    from remy.web.routes import project_routes

    data_dir = tmp_path / "data"
    monkeypatch.setattr(settings, "DATA_DIR", data_dir)
    monkeypatch.setattr(settings, "AURA_BRAIN_PATH", data_dir / "brain")
    reset_project_store_for_tests()

    store = get_project_store()
    project = store.create_project("Client work")
    store.set_active_project(project.project_id)

    class _SessionManager:
        def __init__(self):
            self.session = SimpleNamespace(project_id=project.project_id)
            self.unloaded_project_id = ""

        async def unload_session(self):
            self.unloaded_project_id = self.session.project_id
            self.session = None

        def get_or_create_session(self, project_id):
            self.session = SimpleNamespace(
                project_id=project_id,
                session_id=f"session-{project_id}",
            )
            return self.session

    manager = _SessionManager()
    api = SimpleNamespace(get_session_manager=lambda: manager)
    monkeypatch.setattr(project_routes, "_get_api", lambda: api)

    archived = await project_routes.archive_project(project.project_id)

    assert archived["archived"] is True
    assert archived["brain_deleted"] is False
    assert archived["chat_reset"] is True
    assert archived["active_project_id"] == LEGACY_PROJECT_ID
    assert manager.unloaded_project_id == project.project_id
    assert manager.session.project_id == LEGACY_PROJECT_ID

    restored = await project_routes.restore_project(project.project_id)
    restored_project = restored["project"]

    assert restored["restored"] is True
    assert restored["brain_restored"] is True
    assert restored_project["archived_at"] == ""
    assert restored_project["brain_id"] == project.brain_id
    assert store.require_project(project.project_id).brain_path == project.brain_path
