import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from remy.web.routes import pipeline_routes


def _pipeline_body(**overrides):
    payload = {
        "id": "pipe-test",
        "name": "Test pipeline",
        "description": "",
        "steps": [{"id": "s1", "type": "template", "label": "Text", "config": {"text": "ok"}}],
        "drawflow_data": None,
    }
    payload.update(overrides)
    return pipeline_routes.PipelineSaveRequest(**payload)


@pytest.mark.asyncio
async def test_chat_pipeline_stream_records_and_links_trajectory(monkeypatch):
    from remy.core import conversation_store, microbrain, pipeline_runner, trajectory_store
    from remy.core import transcript_store, workflow_runs

    calls = []
    transcript = []

    class FakeConversations:
        def require(self, conversation_id):
            assert conversation_id == "conversation-1"
            return SimpleNamespace(brain_id="brain-1")

        def touch_from_user_message(self, conversation_id, text):
            calls.append(("touch", conversation_id, text))

    class FakeTrajectory:
        def begin_turn(self, **kwargs):
            calls.append(("begin_turn", kwargs))
            return "turn-1"

        def begin_pipeline_run(self, **kwargs):
            calls.append(("begin_pipeline_run", kwargs))
            return "pipeline-run-event"

        def begin_pipeline_step(self, **kwargs):
            calls.append(("begin_pipeline_step", kwargs))
            return "pipeline-step-event"

        def complete_pipeline_step(self, **kwargs):
            calls.append(("complete_pipeline_step", kwargs))

        def record_pipeline_route(self, **kwargs):
            calls.append(("record_pipeline_route", kwargs))
            return "pipeline-route-event"

        def complete_pipeline_run(self, **kwargs):
            calls.append(("complete_pipeline_run", kwargs))
            return "pipeline-result-event"

        def finish_turn(self, **kwargs):
            calls.append(("finish_turn", kwargs))

    class FakeTranscript:
        def append(self, **kwargs):
            transcript.append(kwargs)

    @contextmanager
    def fake_bind_project(_project_id):
        yield

    async def fake_run_steps(_steps, _input):
        yield {"type": "start", "total": 1}
        yield {
            "type": "step_start", "index": 0, "id": "s1",
            "step_type": "router", "label": "Choose", "input": "hello",
        }
        yield {
            "type": "step_done", "index": 0, "id": "s1",
            "step_type": "router", "label": "Choose",
            "output": "Selected routes: output_2", "route_outputs": ["output_2"],
        }
        yield {"type": "done", "output": "final answer"}

    monkeypatch.setattr(microbrain, "current_project_id", lambda: "project-1")
    monkeypatch.setattr(microbrain, "bind_project", fake_bind_project)
    monkeypatch.setattr(
        pipeline_routes,
        "_load_pipeline",
        lambda _pipeline_id: {
            "id": "pipe-test", "name": "Test pipeline",
            "steps": [{
                "id": "s1", "type": "router", "label": "Choose",
                "config": {
                    "_data_ref": "{{input}}",
                    "routes": [{"operator": "fallback", "value": ""}],
                },
            }],
            "drawflow_data": None,
        },
    )
    monkeypatch.setattr(conversation_store, "get_conversation_store", lambda _project_id: FakeConversations())
    monkeypatch.setattr(trajectory_store, "get_trajectory_store", lambda: FakeTrajectory())
    monkeypatch.setattr(transcript_store, "get_transcript_store", lambda: FakeTranscript())
    monkeypatch.setattr(pipeline_runner, "run_pipeline_steps", fake_run_steps)
    monkeypatch.setattr(
        workflow_runs,
        "start_workflow_run",
        lambda **_kwargs: {
            "run_id": "run-1", "execution_attempt_id": "attempt-1",
            "run_envelope": {"status": "running"},
        },
    )
    monkeypatch.setattr(
        workflow_runs, "update_workflow_run_progress",
        lambda *_args, **_kwargs: {"status": "running"},
    )
    monkeypatch.setattr(
        workflow_runs, "finish_workflow_run",
        lambda record, **_kwargs: {**record, "run_envelope": {"status": "completed"}},
    )

    response = await pipeline_routes.run_pipeline(
        pipeline_routes.PipelineRunRequest(
            pipeline_id="pipe-test",
            input_text="hello",
            conversation_id="conversation-1",
        )
    )
    payloads = []
    async for chunk in response.body_iterator:
        text = chunk.decode() if isinstance(chunk, bytes) else chunk
        payloads.extend(
            json.loads(line[5:].strip())
            for line in text.splitlines()
            if line.startswith("data:")
        )

    started = next(item for item in payloads if item["type"] == "run_started")
    link = next(item for item in payloads if item["type"] == "trajectory_link")
    assert started["conversation_id"] == "conversation-1"
    assert started["trajectory_event_id"] == "pipeline-run-event"
    assert link["event_id"] == "pipeline-result-event"
    assert [item[0] for item in calls] == [
        "touch", "begin_turn", "begin_pipeline_run", "begin_pipeline_step",
        "complete_pipeline_step", "record_pipeline_route", "complete_pipeline_run",
        "finish_turn",
    ]
    assert [item["role"] for item in transcript] == ["user", "assistant"]
    assert transcript[-1]["content"] == "final answer"


@pytest.mark.asyncio
async def test_save_pipeline_rejects_invalid_utility_block(tmp_path, monkeypatch):
    from remy.config.settings import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

    with pytest.raises(HTTPException) as exc:
        await pipeline_routes.save_pipeline(_pipeline_body(
            steps=[
                {"id": "s1", "type": "file_write", "label": "File", "config": {"filename": "../bad.txt", "text": "x"}},
            ],
        ))

    assert exc.value.status_code == 400
    assert "file name must be a simple file name inside workflow_files" in exc.value.detail["errors"][0]
    assert not (tmp_path / "pipelines" / "pipe-test.json").exists()


@pytest.mark.asyncio
async def test_save_pipeline_accepts_valid_utility_blocks(tmp_path, monkeypatch):
    from remy.config.settings import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

    saved = await pipeline_routes.save_pipeline(_pipeline_body(
        steps=[
            {"id": "s1", "type": "set_variable", "label": "Set", "config": {"name": "topic", "value": "demo"}},
            {"id": "s2", "type": "page_scrape", "label": "Scrape", "config": {"url": "https://example.test", "mode": "text"}},
            {"id": "s3", "type": "condition", "label": "Condition", "config": {"condition": "has scraped text"}},
            {"id": "s4", "type": "loop", "label": "Loop", "config": {"max_iterations": 3}},
            {"id": "s5", "type": "template", "label": "Text", "config": {"text": "{{topic}}"}},
        ],
    ))

    assert saved["id"] == "pipe-test"
    assert (tmp_path / "pipelines" / "pipe-test.json").exists()


@pytest.mark.asyncio
async def test_save_pipeline_uses_atomic_write(tmp_path, monkeypatch):
    from remy.config.settings import settings
    from remy.core.file_utils import atomic_write as real_atomic_write

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    calls = []

    def _spy_atomic_write(path, content, encoding="utf-8"):
        calls.append(path)
        return real_atomic_write(path, content, encoding)

    monkeypatch.setattr(pipeline_routes, "atomic_write", _spy_atomic_write)

    await pipeline_routes.save_pipeline(_pipeline_body())

    assert len(calls) == 1
    assert str(calls[0]).endswith("pipe-test.json")


@pytest.mark.asyncio
async def test_home_template_dry_run_records_backend_history(tmp_path, monkeypatch):
    from remy.config.settings import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

    result = await pipeline_routes.run_home_template(
        pipeline_routes.HomeTemplateRunRequest(
            template_id="daily-brief",
            title="Create Daily Brief",
            pack="Personal Admin Pack",
            mode="dry_run",
            inputs={"time": "09:00", "scope": "tasks"},
            steps=["Search tasks", "Draft brief", "Require approval"],
        )
    )

    run = result["run"]
    assert run["template_id"] == "daily-brief"
    assert run["status"] == "dry_run_ready"
    assert run["cost"] == "$0.00 local"
    assert "Search tasks" in run["preview"]

    listed = await pipeline_routes.list_home_template_runs()
    assert listed["runs"][0]["template_id"] == "daily-brief"
    assert listed["runs"][0]["status"] == "dry_run_ready"


@pytest.mark.asyncio
async def test_pipeline_memory_report_endpoint(tmp_path, monkeypatch):
    from remy.config.settings import settings
    from remy.core import workflow_runs

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    record = workflow_runs.start_workflow_run(kind="pipeline", workflow_id="pipe-test")
    workflow_runs.finish_workflow_run(
        record,
        status="ok",
        trace=[{"id": "s1", "type": "memory_search", "output": "[Nothing found in memory]"}],
    )

    report = await pipeline_routes.get_pipeline_memory_report("pipe-test")

    assert report["workflow_id"] == "pipe-test"
    assert report["evaluated_run_count"] == 1
    assert report["totals"]["empty_search_count"] == 1


@pytest.mark.asyncio
async def test_pipeline_preflight_flags_scraper_without_limit(tmp_path, monkeypatch):
    from remy.config.settings import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    await pipeline_routes.save_pipeline(_pipeline_body(
        steps=[
            {"id": "s1", "type": "page_scrape", "label": "Scrape", "config": {"url": "https://example.test"}},
        ],
    ))

    report = await pipeline_routes.get_pipeline_preflight("pipe-test")

    assert report["ok"] is False
    assert report["blockers"][0]["code"] == "page_scrape_without_limit"


@pytest.mark.asyncio
async def test_pipeline_preflight_blocks_missing_http_auth_secret(tmp_path, monkeypatch):
    from remy.config.settings import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", None)
    await pipeline_routes.save_pipeline(_pipeline_body(
        steps=[
            {
                "id": "s1",
                "type": "http_request",
                "label": "Private API",
                "config": {
                    "url": "https://api.example.test",
                    "auth_secret_key": "openrouter_api_key",
                },
            },
        ],
    ))

    report = await pipeline_routes.get_pipeline_preflight("pipe-test")

    assert report["ok"] is False
    assert "missing_http_auth_secret" in {item["code"] for item in report["blockers"]}


@pytest.mark.asyncio
async def test_http_connection_test_uses_vault_secret_without_echoing(monkeypatch):
    import httpx
    from remy.config.settings import settings

    captured = {}

    class FakeResponse:
        status_code = 200
        text = "connected ok"

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def request(self, method, url, headers=None, content=None):
            captured.update({
                "method": method,
                "url": url,
                "headers": headers or {},
                "content": content,
            })
            return FakeResponse()

    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "sk-test-secret")
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    result = await pipeline_routes.test_http_connection(
        pipeline_routes.HttpConnectionTestRequest(
            url="https://api.example.test/status",
            auth_secret_key="openrouter_api_key",
            auth_scheme="Bearer",
        )
    )

    assert result["ok"] is True
    assert result["status_code"] == 200
    assert captured["headers"]["Authorization"] == "Bearer sk-test-secret"
    assert "sk-test-secret" not in str(result)


@pytest.mark.asyncio
async def test_http_connection_test_blocks_missing_secret(monkeypatch):
    from remy.config.settings import settings

    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", None)

    with pytest.raises(HTTPException) as exc:
        await pipeline_routes.test_http_connection(
            pipeline_routes.HttpConnectionTestRequest(
                url="https://api.example.test/status",
                auth_secret_key="openrouter_api_key",
            )
        )

    assert exc.value.status_code == 400
    assert "Authorization secret" in exc.value.detail


@pytest.mark.asyncio
async def test_scrape_test_endpoint_uses_workflow_scraper(monkeypatch):
    from remy.core import pipeline_runner

    captured = {}

    async def fake_scrape(config):
        captured.update(config)
        return "PAGE SCRAPE\nURL: https://example.test\n\nProduct title\n\nPrice: 1200 UAH"

    monkeypatch.setattr(pipeline_runner, "_run_page_scrape", fake_scrape)

    result = await pipeline_routes.test_page_scrape(
        pipeline_routes.ScrapeTestRequest(
            url="https://example.test",
            mode="text",
            max_chars=900,
        )
    )

    assert result["ok"] is True
    assert captured == {"url": "https://example.test", "mode": "text", "max_chars": 900}
    assert "Product title" in result["preview"]


@pytest.mark.asyncio
async def test_home_template_run_is_approval_gated(tmp_path, monkeypatch):
    from remy.config.settings import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

    result = await pipeline_routes.run_home_template(
        pipeline_routes.HomeTemplateRunRequest(
            template_id="monitor-website",
            title="Monitor Website",
            pack="Research Pack",
            mode="run",
            inputs={"url": "https://example.com", "cadence": "daily"},
            steps=["Fetch page", "Summarize change", "Auto-pause on failure"],
        )
    )

    assert result["run"]["status"] == "queued_for_human_approval"
    assert result["run"]["mode"] == "run"


@pytest.mark.asyncio
async def test_clear_home_template_runs(tmp_path, monkeypatch):
    from remy.config.settings import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

    await pipeline_routes.run_home_template(
        pipeline_routes.HomeTemplateRunRequest(
            template_id="save-memory",
            title="Save Memory",
            mode="dry_run",
            inputs={"memory": "decision", "source": "operator note"},
            steps=["Create candidate"],
        )
    )

    cleared = await pipeline_routes.clear_home_template_runs()
    assert cleared["deleted"] == 1
    listed = await pipeline_routes.list_home_template_runs()
    assert listed["runs"] == []


@pytest.mark.asyncio
async def test_pipeline_and_home_templates_share_catalog():
    pipeline = await pipeline_routes.list_templates()
    home = await pipeline_routes.list_home_templates()

    pipeline_ids = {item["id"] for item in pipeline["templates"]}
    home_ids = {item["id"] for item in home["templates"]}

    assert {"summarize-document", "daily-brief", "monitor-website", "save-memory"} <= pipeline_ids
    assert pipeline_ids == home_ids


@pytest.mark.asyncio
async def test_builtin_pipeline_templates_pass_backend_validation():
    result = await pipeline_routes.list_templates()
    builtins = [item for item in result["templates"] if item.get("source") == "built-in"]

    assert builtins
    for template in builtins:
        errors = pipeline_routes._validate_pipeline_payload(
            name=template["name"],
            steps=template["steps"],
            drawflow_data=template.get("drawflow_data"),
        )
        assert errors == [], f"{template['id']} failed validation: {errors}"


@pytest.mark.asyncio
async def test_instantiate_pipeline_template_creates_saved_pipeline(tmp_path, monkeypatch):
    from remy.config.settings import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

    result = await pipeline_routes.instantiate_pipeline_template(
        "monitor-website",
        pipeline_routes.PipelineTemplateInstantiateRequest(
            name="Website monitor pipeline",
            inputs={"url": "https://example.test/changelog"},
        ),
    )

    pipeline = result["pipeline"]
    assert result["template_id"] == "monitor-website"
    assert pipeline["name"] == "Website monitor pipeline"
    assert pipeline["source_template_id"] == "monitor-website"
    assert pipeline["source_template_name"] == "Monitor Website"
    assert pipeline["steps"][0]["type"] == "page_scrape"
    assert pipeline["steps"][0]["config"]["url"] == "https://example.test/changelog"
    assert (tmp_path / "pipelines" / f"{pipeline['id']}.json").exists()
    loaded = await pipeline_routes.get_pipeline(pipeline["id"])
    assert loaded["source_template_id"] == "monitor-website"
    listed = await pipeline_routes.list_pipelines()
    listed_pipeline = next(item for item in listed["pipelines"] if item["id"] == pipeline["id"])
    assert listed_pipeline["source_template_name"] == "Monitor Website"

    updated = await pipeline_routes.save_pipeline(
        pipeline_routes.PipelineSaveRequest(
            id=pipeline["id"],
            name="Renamed monitor pipeline",
            description="",
            steps=pipeline["steps"],
            drawflow_data=pipeline.get("drawflow_data"),
        )
    )
    assert updated["name"] == "Renamed monitor pipeline"
    assert updated["source_template_id"] == "monitor-website"
    assert updated["source_template_name"] == "Monitor Website"


@pytest.mark.asyncio
async def test_instantiate_pipeline_template_applies_document_source(tmp_path, monkeypatch):
    from remy.config.settings import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

    result = await pipeline_routes.instantiate_pipeline_template(
        "summarize-document",
        pipeline_routes.PipelineTemplateInstantiateRequest(inputs={"source": "notes.md", "goal": "risks"}),
    )

    pipeline = result["pipeline"]
    assert pipeline["steps"][0]["type"] == "file_read"
    assert pipeline["steps"][0]["config"]["filename"] == "notes.md"


@pytest.mark.asyncio
async def test_save_pipeline_template_adds_custom_template(tmp_path, monkeypatch):
    from remy.config.settings import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

    result = await pipeline_routes.save_pipeline_template(
        pipeline_routes.PipelineTemplateSaveRequest(
            name="My scraper",
            steps=[
                {"id": "s1", "type": "page_scrape", "label": "Scrape", "config": {"url": "https://example.test", "mode": "text", "max_chars": 12000}},
                {"id": "s2", "type": "llm_call", "label": "Summarize", "config": {"prompt": "{{s1.output}}"}},
            ],
            drawflow_data=None,
        )
    )

    assert result["template"]["source"] == "custom"
    listed = await pipeline_routes.list_templates()
    custom = [item for item in listed["templates"] if item.get("source") == "custom"]
    assert custom[0]["name"] == "My scraper"
    assert custom[0]["steps"][0]["type"] == "page_scrape"

    deleted = await pipeline_routes.delete_pipeline_template(custom[0]["id"])
    assert deleted["deleted"] is True
    listed_after = await pipeline_routes.list_templates()
    assert not [item for item in listed_after["templates"] if item.get("source") == "custom"]

    with pytest.raises(HTTPException) as exc:
        await pipeline_routes.delete_pipeline_template("daily-brief")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_custom_pipeline_templates_are_capped(tmp_path, monkeypatch):
    from remy.config.settings import settings
    from remy.core import workflow_templates

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(workflow_templates, "MAX_CUSTOM_TEMPLATES_PER_KIND", 2)

    for index in range(3):
        await pipeline_routes.save_pipeline_template(
            pipeline_routes.PipelineTemplateSaveRequest(
                name=f"Template {index}",
                steps=[
                    {
                        "id": "s1",
                        "type": "template",
                        "label": "Text",
                        "config": {"text": f"template {index}"},
                    },
                ],
                drawflow_data=None,
            )
        )

    listed = await pipeline_routes.list_templates()
    custom = [item for item in listed["templates"] if item.get("source") == "custom"]
    assert len(custom) == 2
    assert [item["name"] for item in custom] == ["Template 2", "Template 1"]


@pytest.mark.asyncio
async def test_custom_pipeline_template_storage_uses_atomic_write(tmp_path, monkeypatch):
    from remy.config.settings import settings
    from remy.core import workflow_templates
    from remy.core.file_utils import atomic_write as real_atomic_write

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    calls = []

    def _spy_atomic_write(path, content, encoding="utf-8"):
        calls.append(path)
        return real_atomic_write(path, content, encoding)

    monkeypatch.setattr(workflow_templates, "atomic_write", _spy_atomic_write)

    await pipeline_routes.save_pipeline_template(
        pipeline_routes.PipelineTemplateSaveRequest(
            name="Atomic template",
            steps=[
                {
                    "id": "s1",
                    "type": "template",
                    "label": "Text",
                    "config": {"text": "atomic"},
                },
            ],
            drawflow_data=None,
        )
    )

    assert len(calls) == 1
    assert str(calls[0]).endswith("custom.json")
