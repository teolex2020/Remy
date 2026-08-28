import pytest

from remy.core.agent_lab import AgentLabStore
from remy.core.agent_lab_workspace import AgentLabWorkspaceManager
from remy.web.routes import agent_lab_routes


class FakeTrajectoryStore:
    def __init__(self):
        self.begins = []
        self.events = []
        self.completions = []

    def begin_execution_run(self, **payload):
        self.begins.append(payload)
        return "agent-lab-trajectory-root"

    def record_execution_event(self, **payload):
        self.events.append(payload)
        return f"event-{len(self.events)}"

    def complete_execution_run(self, **payload):
        self.completions.append(payload)
        return "agent-lab-trajectory-result"


class InactiveCoordinator:
    def is_active(self, _run_id):
        return False


@pytest.mark.asyncio
async def test_agent_lab_route_lifecycle_emits_trajectory(tmp_path, monkeypatch):
    store = AgentLabStore(tmp_path, owner_project_id="project-1", brain_id="brain-1")
    trajectory = FakeTrajectoryStore()
    monkeypatch.setattr(agent_lab_routes, "_store", lambda: store)
    monkeypatch.setattr(agent_lab_routes, "current_project_id", lambda: "project-1")
    monkeypatch.setattr(agent_lab_routes, "get_trajectory_store", lambda: trajectory)

    created = await agent_lab_routes.create_agent_lab_run(agent_lab_routes.AgentLabCreatePayload(
        title="Autonomous build",
        goal="Build and independently verify a local micro-site",
    ))
    run_id = created["run"]["run_id"]
    assert trajectory.begins[0]["scope"] == "agent_lab"
    assert created["run"]["trajectory_run_event_id"] == "agent-lab-trajectory-root"

    prepared = await agent_lab_routes.prepare_agent_lab_run(run_id)
    assert prepared["run"]["status"] == "prepared"
    assert [event["event_kind"] for event in trajectory.events] == [
        "AGENT_LAB_PHASE", "AGENT_LAB_TEAM", "AGENT_LAB_DECISION", "AGENT_LAB_PHASE"
    ]

    await agent_lab_routes.stage_agent_lab_file(run_id, agent_lab_routes.AgentLabFilePayload(
        path="src/main.py",
        content='from pathlib import Path\nPath("artifacts/result.txt").write_text("ready", encoding="utf-8")\nprint("built")\n',
    ))
    await agent_lab_routes.stage_agent_lab_file(run_id, agent_lab_routes.AgentLabFilePayload(
        path="tests/verify.py",
        content='from pathlib import Path\nassert Path("artifacts/result.txt").read_text(encoding="utf-8") == "ready"\nprint("verified")\n',
    ))
    started = await agent_lab_routes.start_agent_lab_run(
        run_id, agent_lab_routes.AgentLabExecutePayload(timeout_seconds=5)
    )
    assert started["execution"]["status"] == "passed"
    assert started["run"]["status"] == "paused"
    assert started["run"]["artifacts"][0]["name"] == "result.txt"
    artifact_response = await agent_lab_routes.download_agent_lab_artifact(
        run_id, started["run"]["artifacts"][0]["artifact_id"]
    )
    assert artifact_response.filename == "result.txt"

    verified = await agent_lab_routes.verify_agent_lab_run(
        run_id, agent_lab_routes.AgentLabVerifyPayload(timeout_seconds=5)
    )
    assert verified["verified"] is True
    assert verified["run"]["status"] == "completed"
    assert verified["run"]["proof_pack"]["decision"] == "accepted"
    assert verified["verification"]["read_only"] is True
    assert verified["run"]["trajectory_result_event_id"] == "agent-lab-trajectory-result"
    assert trajectory.completions[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_agent_lab_archive_route_removes_closed_run_from_history(tmp_path, monkeypatch):
    store = AgentLabStore(tmp_path)
    record = store.create(goal="Clear the visible laboratory history")
    monkeypatch.setattr(agent_lab_routes, "_store", lambda: store)
    monkeypatch.setattr(agent_lab_routes, "_coordinator", lambda: InactiveCoordinator())

    result = await agent_lab_routes.archive_agent_lab_run(record["run_id"])

    assert result == {
        "archived": True,
        "run_id": record["run_id"],
        "evidence_preserved": True,
    }
    assert store.list() == []
    assert store.require(record["run_id"])["archived_at"]


@pytest.mark.asyncio
async def test_snapshot_retention_routes_are_project_scoped_and_audited(tmp_path, monkeypatch):
    store = AgentLabStore(tmp_path, owner_project_id="project-1")
    trajectory = FakeTrajectoryStore()
    manager = AgentLabWorkspaceManager(store)
    monkeypatch.setattr(agent_lab_routes, "_store", lambda: store)
    monkeypatch.setattr(agent_lab_routes, "_workspace_manager", lambda: manager)
    monkeypatch.setattr(agent_lab_routes, "current_project_id", lambda: "project-1")
    monkeypatch.setattr(agent_lab_routes, "get_trajectory_store", lambda: trajectory)
    created = await agent_lab_routes.create_agent_lab_run(
        agent_lab_routes.AgentLabCreatePayload(goal="Build then clean private copies")
    )
    run_id = created["run"]["run_id"]
    await agent_lab_routes.prepare_agent_lab_run(run_id)
    snapshot = manager.create_snapshot(run_id, node_id="build")
    manager.stage_file(
        run_id,
        snapshot["workspace_id"],
        path="src/main.py",
        content="print('canonical')\n",
    )
    manager.merge(run_id, snapshot["workspace_id"])

    inspected = await agent_lab_routes.get_agent_lab_snapshot_retention(run_id)
    assert inspected["retention"]["cleanup_eligible_count"] == 1
    cleaned = await agent_lab_routes.cleanup_agent_lab_snapshots(
        run_id,
        agent_lab_routes.AgentLabSnapshotCleanupPayload(),
    )

    assert cleaned["cleaned"] is True
    assert cleaned["retention"]["snapshot_count"] == 0
    assert cleaned["run"]["snapshot_cleanup_receipts"][-1]["removed_count"] == 1
    assert trajectory.events[-1]["event_kind"] == "AGENT_LAB_RETENTION"


@pytest.mark.asyncio
async def test_failed_execution_preserves_repair_checkpoint(tmp_path, monkeypatch):
    store = AgentLabStore(tmp_path, owner_project_id="project-1")
    trajectory = FakeTrajectoryStore()
    monkeypatch.setattr(agent_lab_routes, "_store", lambda: store)
    monkeypatch.setattr(agent_lab_routes, "current_project_id", lambda: "project-1")
    monkeypatch.setattr(agent_lab_routes, "get_trajectory_store", lambda: trajectory)
    created = await agent_lab_routes.create_agent_lab_run(
        agent_lab_routes.AgentLabCreatePayload(goal="Build a bounded program")
    )
    run_id = created["run"]["run_id"]
    await agent_lab_routes.prepare_agent_lab_run(run_id)
    await agent_lab_routes.stage_agent_lab_file(
        run_id,
        agent_lab_routes.AgentLabFilePayload(
            path="src/main.py", content='raise RuntimeError("repair me")\n'
        ),
    )

    result = await agent_lab_routes.start_agent_lab_run(
        run_id, agent_lab_routes.AgentLabExecutePayload(timeout_seconds=5)
    )

    assert result["execution"]["status"] == "failed"
    assert result["run"]["status"] == "paused"
    assert result["run"]["phase"] == "repair_needed"
    assert next(step for step in result["run"]["plan"] if step["step_id"] == "build")["status"] == "needs_repair"
    assert not trajectory.completions


@pytest.mark.asyncio
async def test_agent_lab_route_rejects_start_before_prepare(tmp_path, monkeypatch):
    store = AgentLabStore(tmp_path)
    monkeypatch.setattr(agent_lab_routes, "_store", lambda: store)
    record = store.create(goal="A bounded task")

    with pytest.raises(agent_lab_routes.HTTPException) as exc:
        await agent_lab_routes.start_agent_lab_run(record["run_id"])
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_agent_lab_autonomous_route_starts_bounded_coordinator(tmp_path, monkeypatch):
    store = AgentLabStore(tmp_path)
    record = store.create(goal="Autonomous bounded build")
    store.prepare(record["run_id"])

    class FakeCoordinator:
        def __init__(self):
            self.calls = []

        def start(self, run_id, **options):
            self.calls.append((run_id, options))
            return store.transition(run_id, "running", message="fake coordinator")

        def is_active(self, run_id):
            return True

        def cancel(self, run_id):
            return True

    coordinator = FakeCoordinator()
    monkeypatch.setattr(agent_lab_routes, "_store", lambda: store)
    monkeypatch.setattr(agent_lab_routes, "_coordinator", lambda: coordinator)
    monkeypatch.setattr(
        agent_lab_routes,
        "list_agent_lab_models",
        lambda: _async_value({"models": [{"name": "model-a"}], "default": "model-a"}),
    )

    response = await agent_lab_routes.start_autonomous_agent_lab_run(
        record["run_id"],
        agent_lab_routes.AgentLabAutonomousPayload(model="model-a", max_repair_rounds=1),
    )

    assert response["autonomous"] is True
    assert coordinator.calls == [(
        record["run_id"],
        {
            "model": "model-a",
            "verifier_model": "",
            "max_repair_rounds": 1,
            "isolation_mode": "bounded_process",
        },
    )]


@pytest.mark.asyncio
async def test_paused_lab_accepts_clarification_and_continues(tmp_path, monkeypatch):
    store = AgentLabStore(tmp_path)
    record = store.create(goal="Prepare a report")
    store.prepare(record["run_id"])
    store.transition(record["run_id"], "running", message="started")
    store.mutate(record["run_id"], lambda item: item.update({
        "phase": "repair_needed",
        "error": "Expected output format is unclear",
        "trajectory_run_event_id": "agent-lab-trajectory-root",
        "autonomous": {
            "model": "model-a",
            "verifier_model": "model-b",
            "max_repair_rounds": 1,
            "isolation_mode": "automatic",
        },
    }))
    store.transition(record["run_id"], "paused", message="needs input")

    class FakeCoordinator:
        def __init__(self):
            self.calls = []

        def is_active(self, run_id):
            return False

        def start(self, run_id, **options):
            self.calls.append((run_id, options))
            return store.transition(run_id, "running", message="continued")

    coordinator = FakeCoordinator()
    trajectory = FakeTrajectoryStore()
    monkeypatch.setattr(agent_lab_routes, "_store", lambda: store)
    monkeypatch.setattr(agent_lab_routes, "_coordinator", lambda: coordinator)
    monkeypatch.setattr(agent_lab_routes, "get_trajectory_store", lambda: trajectory)

    response = await agent_lab_routes.clarify_agent_lab_run(
        record["run_id"],
        agent_lab_routes.AgentLabClarificationPayload(
            message="Deliver an HTML report with a comparison table."
        ),
    )

    saved = store.require(record["run_id"])
    assert response["continued"] is True
    assert saved["status"] == "running"
    assert "Additional user clarification" in saved["goal"]
    assert saved["user_clarifications"][-1]["message"].startswith("Deliver an HTML")
    assert coordinator.calls[0][1] == {
        "model": "model-a",
        "verifier_model": "model-b",
        "max_repair_rounds": 1,
        "isolation_mode": "automatic",
    }
    assert trajectory.events[-1]["event_kind"] == "AGENT_LAB_DECISION"


async def _async_value(value):
    return value


@pytest.mark.asyncio
async def test_agent_lab_isolation_status_does_not_expose_local_executable(monkeypatch):
    monkeypatch.setattr(
        agent_lab_routes,
        "probe_agent_lab_container_runtime",
        lambda: {
            "available": True,
            "engine": "docker",
            "executable": "C:/private/tools/docker.exe",
            "image": "remy-agent-lab-runtime:py312",
            "image_present": True,
            "reason": "",
            "security_contract": ["network_none"],
        },
    )
    result = await agent_lab_routes.get_agent_lab_isolation_status()
    assert result["container"]["available"] is True
    assert "executable" not in result["container"]
    assert result["modes"] == ["bounded_process", "container_required"]
    assert result["selection_modes"] == ["automatic", "bounded_process", "container_required"]
    assert result["automatic_supported"] is True
    assert [item["mode"] for item in result["backends"]] == result["modes"]
    assert result["backends"][0]["available"] is True
    assert result["backends"][1]["capabilities"]["isolation_rank"] == 20


@pytest.mark.asyncio
async def test_agent_lab_runtime_preparation_does_not_expose_local_executable(monkeypatch):
    monkeypatch.setattr(
        agent_lab_routes,
        "prepare_agent_lab_container_runtime",
        lambda: {
            "available": True,
            "prepared": True,
            "reason_code": "ready",
            "engine": "docker",
            "executable": "C:/private/tools/docker.exe",
            "image": "remy-agent-lab-runtime:py312",
            "image_id": "sha256:" + "a" * 64,
        },
    )

    result = await agent_lab_routes.prepare_agent_lab_isolation_runtime()

    assert result["container"]["prepared"] is True
    assert "executable" not in result["container"]
    assert result["backend"]["mode"] == "container_required"


@pytest.mark.asyncio
async def test_agent_lab_autonomous_route_rejects_unconnected_model(tmp_path, monkeypatch):
    store = AgentLabStore(tmp_path)
    record = store.create(goal="Autonomous bounded build")
    store.prepare(record["run_id"])
    monkeypatch.setattr(agent_lab_routes, "_store", lambda: store)
    monkeypatch.setattr(
        agent_lab_routes,
        "list_agent_lab_models",
        lambda: _async_value({"models": [{"name": "model-a"}], "default": "model-a"}),
    )

    with pytest.raises(agent_lab_routes.HTTPException) as exc:
        await agent_lab_routes.start_autonomous_agent_lab_run(
            record["run_id"],
            agent_lab_routes.AgentLabAutonomousPayload(model="not-connected"),
        )
    assert exc.value.status_code == 422

    with pytest.raises(agent_lab_routes.HTTPException) as verifier_exc:
        await agent_lab_routes.start_autonomous_agent_lab_run(
            record["run_id"],
            agent_lab_routes.AgentLabAutonomousPayload(
                model="model-a",
                verifier_model="not-connected",
            ),
        )
    assert verifier_exc.value.status_code == 422


def test_agent_lab_ui_is_separate_lazy_and_observable():
    from pathlib import Path

    html = Path("src/remy/web/static/index.html").read_text(encoding="utf-8")
    app_js = Path("src/remy/web/static/js/app.js").read_text(encoding="utf-8")
    lab_js = Path("src/remy/web/static/js/agent-lab.js").read_text(encoding="utf-8")

    assert 'data-view="agent-lab"' in html
    assert 'id="view-agent-lab"' in html
    assert 'import("./agent-lab.js?v=3.4")' in app_js
    assert "Builder fan-out & file claims" in lab_js
    assert "renderBuilderFanout" in lab_js
    assert "/snapshots/retention" in lab_js
    assert "/snapshots/cleanup" in lab_js
    assert "Clean conflict snapshots" in lab_js
    assert "/api/agent-lab/isolation" in lab_js
    assert "Array.isArray(data.backends)" in lab_js
    assert "item.security_tier" in lab_js
    assert "Install Docker Desktop" in lab_js
    assert "data-agent-lab-container-recheck" in lab_js
    assert "Prepare secure runtime" in lab_js
    assert "/api/agent-lab/isolation/prepare" in lab_js
    assert '/api/agent-lab/runs' in lab_js
    assert 'Central assignment & specialist fan-in' in lab_js
    assert 'Agent-owned plan' in lab_js
    assert 'Task Ledger' in lab_js
    assert 'Progress Ledger' in lab_js
    assert 'Workspace & artifacts' in lab_js
    assert 'Merge gate' in lab_js
    assert 'Private snapshot' in lab_js
    assert 'Execution console' in lab_js
    assert '/files`' in lab_js
    assert '/autonomous`' in lab_js
    assert 'Closed Agent Laboratory' in lab_js
    assert 'Laboratory history' in lab_js
    assert 'interactive: true' in lab_js
    assert 'Start' in lab_js
    assert 'Creating a safe workspace' in lab_js
    assert 'Technical details' in lab_js
    assert 'Remy needs your help' in lab_js
    assert 'This is an internal laboratory issue—not a problem with your request.' in lab_js
    assert 'Retry automatically' in lab_js
    assert 'isAwaitingUser' in lab_js
    assert 'refreshSelectedRun' in lab_js
    assert 'POLL_BASE_MS = 4000' in lab_js
    assert 'response.headers.get("Retry-After")' in lab_js
    assert 'setTimeout(() => loadAgentLab(), 1400)' not in lab_js
    assert '/clarify`' in lab_js
    assert 'Remove from history' in lab_js
    assert '/archive`' in lab_js
    assert 'Proof Pack' in lab_js
    assert 'execution-trajectory-open' in lab_js
