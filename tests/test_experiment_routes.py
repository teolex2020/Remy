import pytest
from io import BytesIO
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from remy.web.routes import experiment_routes


@pytest.mark.asyncio
async def test_self_modification_canary_policy_route_is_project_scoped(tmp_path, monkeypatch):
    from remy.config.settings import settings
    import remy.core.self_modification_lab as module

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(module, "_DEFAULT_LAB", None)
    monkeypatch.setattr(experiment_routes, "_active_project_id", lambda: "project-route")

    listed = await experiment_routes.list_self_modification_proposals()
    assert listed["policy"]["source"] == "default"
    assert listed["constraints"]["canary_target_requests_per_cohort"] == 20

    updated = await experiment_routes.update_self_modification_canary_policy(
        experiment_routes.SelfModificationCanaryPolicyPayload(
            minimum_requests_per_cohort=7,
            target_requests_per_cohort=30,
            minimum_observation_seconds=120,
            inconclusive_alert_seconds=900,
        )
    )
    assert updated["policy"]["source"] == "project"
    assert updated["policy"]["minimum_requests_per_cohort"] == 7
    assert (
        await experiment_routes.list_self_modification_proposals()
    )["constraints"]["canary_target_requests_per_cohort"] == 30


@pytest.mark.asyncio
async def test_role_catalog_route_is_user_explanatory():
    result = await experiment_routes.experiment_role_catalog()
    roles = result["roles"]
    investigator = next(item for item in roles if item["preset_id"] == "investigator")
    assert investigator["description"]
    assert investigator["best_for"]
    assert investigator["expected_output"]
    assert investigator["instruction"]


@pytest.mark.asyncio
async def test_experiment_route_lifecycle_without_model_call(tmp_path, monkeypatch):
    from remy.config.settings import settings
    import remy.core.experiment_lab as module

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "SUMMARY_MODEL", "test-model")
    monkeypatch.setattr(module, "_engine", None)
    monkeypatch.setattr(module, "_engine_data_dir", None)
    monkeypatch.setattr("remy.core.model_registry.list_registered_models", lambda: [])

    created = await experiment_routes.create_experiment(experiment_routes.ExperimentCreate(
        title="Route test", problem="Test the API", models=["test-model"], rounds=1,
    ))
    experiment_id = created["experiment"]["experiment_id"]
    await experiment_routes.add_experiment_data(
        experiment_id, experiment_routes.ExperimentData(name="facts", content="A fact")
    )
    detail = await experiment_routes.get_experiment(experiment_id)
    assert detail["experiment"]["datasets"][0]["name"] == "facts"
    listed = await experiment_routes.list_experiments()
    assert listed["experiments"][0]["experiment_id"] == experiment_id


@pytest.mark.asyncio
async def test_create_rejects_unconnected_model(tmp_path, monkeypatch):
    from remy.config.settings import settings
    import remy.core.experiment_lab as module

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "SUMMARY_MODEL", "connected-model")
    monkeypatch.setattr(module, "_engine", None)
    monkeypatch.setattr(module, "_engine_data_dir", None)
    monkeypatch.setattr("remy.core.model_registry.list_registered_models", lambda: [])
    with pytest.raises(HTTPException) as exc:
        await experiment_routes.create_experiment(experiment_routes.ExperimentCreate(
            title="Bad", problem="No key", models=["unknown-model"], rounds=1,
        ))
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_canvas_validate_and_create_draft(tmp_path, monkeypatch):
    from remy.config.settings import settings
    from remy.core.experiment_canvas import build_template
    import remy.core.experiment_lab as module

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "SUMMARY_MODEL", "test-model")
    monkeypatch.setattr(module, "_engine", None)
    monkeypatch.setattr(module, "_engine_data_dir", None)
    monkeypatch.setattr("remy.core.model_registry.list_registered_models", lambda: [])
    canvas = build_template("blank", ["test-model"])
    next(node for node in canvas["nodes"] if node["type"] == "problem")["data"].update({
        "title": "Canvas route", "problem": "Compile this graph",
    })
    valid = await experiment_routes.validate_experiment_canvas(experiment_routes.CanvasPayload(canvas=canvas))
    assert valid["valid"] is True
    created = await experiment_routes.create_canvas_experiment(experiment_routes.CanvasPayload(canvas=canvas))
    assert created["experiment"]["mode"] == "canvas"
    assert created["experiment"]["participants"][0]["model"] == "test-model"

    problem = next(node for node in canvas["nodes"] if node["type"] == "problem")
    problem["data"]["title"] = "Edited canvas"
    updated = await experiment_routes.update_canvas_experiment(
        created["experiment"]["experiment_id"], experiment_routes.CanvasPayload(canvas=canvas)
    )
    assert updated["experiment"]["title"] == "Edited canvas"


@pytest.mark.asyncio
async def test_quick_update_and_recoverable_delete_routes(tmp_path, monkeypatch):
    from remy.config.settings import settings
    import remy.core.experiment_lab as module

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "SUMMARY_MODEL", "test-model")
    monkeypatch.setattr(module, "_engine", None)
    monkeypatch.setattr(module, "_engine_data_dir", None)
    monkeypatch.setattr("remy.core.model_registry.list_registered_models", lambda: [])
    created = await experiment_routes.create_experiment(experiment_routes.ExperimentCreate(
        title="Delete me", problem="Temporary", models=["test-model"], rounds=1,
    ))
    experiment_id = created["experiment"]["experiment_id"]
    updated = await experiment_routes.update_experiment(experiment_id, experiment_routes.ExperimentUpdate(
        title="Edited test", problem="Still temporary", models=["test-model"], rounds=2,
    ))
    assert updated["experiment"]["title"] == "Edited test"
    deleted = await experiment_routes.delete_experiment(experiment_id)
    assert deleted["deleted"] is True and deleted["recoverable"] is True
    assert not (tmp_path / "experiments" / experiment_id).exists()


@pytest.mark.asyncio
async def test_continue_route_queues_followup(tmp_path, monkeypatch):
    from remy.config.settings import settings
    import remy.core.experiment_lab as module

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(module, "_engine", None)
    monkeypatch.setattr(module, "_engine_data_dir", None)
    engine = module.get_experiment_engine()
    record = engine.store.create(
        title="Completed", problem="Initial problem", success_criteria="",
        models=["test-model"], rounds=1, max_calls=2,
    )
    engine.store.mutate(record["experiment_id"], lambda item: item.update({
        "status": "completed", "current_round": 1, "calls_used": 2,
    }))
    monkeypatch.setattr(engine, "_run", lambda experiment_id, token: __import__("asyncio").sleep(0))

    response = await experiment_routes.continue_experiment(
        record["experiment_id"],
        experiment_routes.ExperimentContinue(question="Follow up", rounds=1),
    )
    assert response["continued"] is True
    assert response["experiment"]["status"] == "queued"
    assert response["experiment"]["rounds"] == 2


@pytest.mark.asyncio
async def test_scenario_intervention_route(tmp_path, monkeypatch):
    from remy.config.settings import settings
    import remy.core.experiment_lab as module

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(module, "_engine", None)
    monkeypatch.setattr(module, "_engine_data_dir", None)
    engine = module.get_experiment_engine()
    record = engine.store.create(
        title="Scenario", problem="What if?", success_criteria="",
        models=["test-model"], rounds=2, max_calls=3,
    )
    engine.store.mutate(record["experiment_id"], lambda item: item.update({
        "mode": "canvas",
        "experiment_plan": {"scenario": {"enabled": True, "replicas": 1}},
    }))
    response = await experiment_routes.add_scenario_intervention(
        record["experiment_id"],
        experiment_routes.ScenarioIntervention(content="A shock occurs", round=2),
    )
    assert response["scheduled"] is True
    assert response["experiment"]["scenario_runtime_interventions"][0]["content"] == "A shock occurs"


@pytest.mark.asyncio
async def test_canvas_source_file_is_extracted_before_draft_creation(tmp_path, monkeypatch):
    from remy.config.settings import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    upload = UploadFile(filename="evidence.txt", file=BytesIO("Important observation".encode("utf-8")))
    result = await experiment_routes.extract_canvas_source_file(upload)
    assert result["name"] == "evidence.txt"
    assert result["content"] == "Important observation"
    assert result["characters"] == len("Important observation")
