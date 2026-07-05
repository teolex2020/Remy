from types import SimpleNamespace
from datetime import datetime

import pytest
from fastapi import HTTPException

from remy.web.routes import automation_routes


@pytest.mark.asyncio
async def test_automation_templates_expose_product_catalog():
    result = await automation_routes.list_automation_templates()
    ids = {item["id"] for item in result["templates"]}

    assert {"daily-brief", "monitor-website", "extract-deadlines", "save-memory"} <= ids
    daily = next(item for item in result["templates"] if item["id"] == "daily-brief")
    assert daily["trigger"]["type"] == "schedule"
    assert daily["steps"]
    monitor = next(item for item in result["templates"] if item["id"] == "monitor-website")
    assert monitor["steps"][0]["type"] == "page_scrape"
    assert daily["output_destination"]["type"] == "chat"


@pytest.mark.asyncio
async def test_builtin_automation_templates_pass_backend_validation():
    result = await automation_routes.list_automation_templates()
    builtins = [item for item in result["templates"] if item.get("source") == "built-in"]

    assert builtins
    for template in builtins:
        errors = automation_routes._validate_automation_payload(
            name=template["name"],
            trigger=template["trigger"],
            steps=template["steps"],
            output_destination=template["output_destination"],
            drawflow_data=template.get("drawflow_data"),
        )
        assert errors == [], f"{template['id']} failed validation: {errors}"


@pytest.mark.asyncio
async def test_instantiate_automation_template_creates_disabled_automation(monkeypatch):
    fake_api = SimpleNamespace(brain=_Brain(), brain_lock=_Lock())
    monkeypatch.setattr(automation_routes, "_get_api", lambda: fake_api)

    result = await automation_routes.instantiate_automation_template(
        "daily-brief",
        automation_routes.AutomationTemplateInstantiate(
            name="My daily brief",
            inputs={"time": "08:30", "scope": "tasks launch decisions"},
        ),
    )

    automation = result["automation"]
    assert result["template_id"] == "daily-brief"
    assert automation["name"] == "My daily brief"
    assert automation["source_template_id"] == "daily-brief"
    assert automation["source_template_name"] == "Create Daily Brief"
    assert automation["enabled"] is False
    assert automation["trigger"]["type"] == "schedule"
    assert automation["trigger"]["time_of_day"] == "08:30"
    assert automation["steps"][0]["type"] == "memory_search"
    assert automation["steps"][0]["config"]["query"] == "tasks launch decisions"
    assert fake_api.brain.records
    assert fake_api.brain.records[0].metadata["source_template_id"] == "daily-brief"


@pytest.mark.asyncio
async def test_instantiate_automation_template_applies_scraper_url(monkeypatch):
    fake_api = SimpleNamespace(brain=_Brain(), brain_lock=_Lock())
    monkeypatch.setattr(automation_routes, "_get_api", lambda: fake_api)

    result = await automation_routes.instantiate_automation_template(
        "monitor-website",
        automation_routes.AutomationTemplateInstantiate(inputs={"url": "https://example.test/status"}),
    )

    automation = result["automation"]
    assert automation["steps"][0]["type"] == "page_scrape"
    assert automation["steps"][0]["config"]["url"] == "https://example.test/status"


def test_automation_error_prefixes_include_scrape_failures():
    assert "[Scrape error:" in automation_routes.ERROR_PREFIXES


@pytest.mark.asyncio
async def test_save_automation_template_adds_custom_template(tmp_path, monkeypatch):
    from remy.config.settings import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

    result = await automation_routes.save_automation_template(
        automation_routes.AutomationTemplateSave(
            name="My daily monitor",
            trigger={"type": "manual"},
            steps=[{"id": "s1", "type": "template", "label": "Text", "config": {"text": "ok"}}],
            output_destination={"type": "chat"},
            drawflow_data=None,
        )
    )

    assert result["template"]["source"] == "custom"
    listed = await automation_routes.list_automation_templates()
    custom = [item for item in listed["templates"] if item.get("source") == "custom"]
    assert custom[0]["name"] == "My daily monitor"
    assert custom[0]["output_destination"]["type"] == "chat"

    deleted = await automation_routes.delete_automation_template(custom[0]["id"])
    assert deleted["deleted"] is True
    listed_after = await automation_routes.list_automation_templates()
    assert not [item for item in listed_after["templates"] if item.get("source") == "custom"]

    with pytest.raises(HTTPException) as exc:
        await automation_routes.delete_automation_template("daily-brief")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_custom_automation_templates_are_capped(tmp_path, monkeypatch):
    from remy.config.settings import settings
    from remy.core import workflow_templates

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(workflow_templates, "MAX_CUSTOM_TEMPLATES_PER_KIND", 2)

    for index in range(3):
        await automation_routes.save_automation_template(
            automation_routes.AutomationTemplateSave(
                name=f"Automation Template {index}",
                trigger={"type": "manual"},
                steps=[
                    {
                        "id": "s1",
                        "type": "template",
                        "label": "Text",
                        "config": {"text": f"automation template {index}"},
                    },
                ],
                output_destination={"type": "chat"},
                drawflow_data=None,
            )
        )

    listed = await automation_routes.list_automation_templates()
    custom = [item for item in listed["templates"] if item.get("source") == "custom"]
    assert len(custom) == 2
    assert [item["name"] for item in custom] == ["Automation Template 2", "Automation Template 1"]


class _Lock:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _Brain:
    def __init__(self):
        self.records = []
        self._next_id = 1

    def store(self, content, tags=None, metadata=None):
        rec = SimpleNamespace(
            id=f"rec-{self._next_id}",
            content=content,
            tags=list(tags or []),
            metadata=dict(metadata or {}),
        )
        self._next_id += 1
        self.records.append(rec)
        return rec

    def search(self, query="", tags=None, limit=500, **_kwargs):
        records = self.records
        if tags:
            wanted = set(tags)
            records = [r for r in records if wanted.issubset(set(getattr(r, "tags", [])))]
        return records[:limit]

    def delete(self, record_id):
        self.records = [r for r in self.records if r.id != record_id]


def _valid_body(**overrides):
    payload = {
        "name": "Morning summary",
        "enabled": True,
        "trigger": {"type": "manual"},
        "steps": [{"id": "s1", "type": "template", "label": "Text", "config": {"text": "ok"}}],
        "output_destination": {"type": "chat"},
        "drawflow_data": None,
    }
    payload.update(overrides)
    return automation_routes.AutomationSave(**payload)


def test_steps_in_execution_order_follows_canvas_connections():
    meta = {
        "steps": [
            {"id": "s4", "type": "llm_call"},
            {"id": "s3", "type": "web_search"},
        ],
        "drawflow_data": {
            "drawflow": {
                "Home": {
                    "data": {
                        "1": {"name": "trigger", "outputs": {"output_1": {"connections": [{"node": "3"}]}}},
                        "2": {"name": "output", "outputs": {}},
                        "3": {"name": "web_search", "outputs": {"output_1": {"connections": [{"node": "4"}]}}},
                        "4": {"name": "llm_call", "outputs": {"output_1": {"connections": [{"node": "2"}]}}},
                    }
                }
            }
        },
    }

    ordered = automation_routes._steps_in_execution_order(meta)

    assert [step["id"] for step in ordered] == ["s3", "s4"]


def test_latest_missed_scheduled_run_detects_daily_slot_after_startup():
    due = automation_routes.latest_missed_scheduled_run(
        {
            "enabled": True,
            "created_at": "2026-06-28T10:00:00",
            "last_run_at": "2026-06-28T09:00:00",
            "trigger": {"type": "schedule", "schedule_type": "daily", "time_of_day": "09:00"},
        },
        now=datetime(2026, 6, 29, 10, 30),
    )

    assert due == datetime(2026, 6, 29, 9, 0)


def test_latest_missed_scheduled_run_skips_when_slot_already_ran():
    due = automation_routes.latest_missed_scheduled_run(
        {
            "enabled": True,
            "created_at": "2026-06-28T10:00:00",
            "last_run_at": "2026-06-29T09:05:00",
            "trigger": {"type": "schedule", "schedule_type": "daily", "time_of_day": "09:00"},
        },
        now=datetime(2026, 6, 29, 10, 30),
    )

    assert due is None


def test_latest_missed_scheduled_run_uses_previous_hourly_slot():
    due = automation_routes.latest_missed_scheduled_run(
        {
            "enabled": True,
            "created_at": "2026-06-28T10:00:00",
            "last_run_at": "2026-06-29T12:20:00",
            "trigger": {"type": "schedule", "schedule_type": "hourly", "time_of_day": "09:15"},
        },
        now=datetime(2026, 6, 29, 14, 10),
    )

    assert due == datetime(2026, 6, 29, 13, 15)


def test_latest_missed_scheduled_run_respects_disabled_catch_up_and_custom_cron():
    base = {
        "enabled": True,
        "created_at": "2026-06-28T10:00:00",
        "last_run_at": "2026-06-28T09:00:00",
    }

    assert automation_routes.latest_missed_scheduled_run(
        {**base, "trigger": {"type": "schedule", "schedule_type": "daily", "time_of_day": "09:00", "catch_up": False}},
        now=datetime(2026, 6, 29, 10, 30),
    ) is None
    assert automation_routes.latest_missed_scheduled_run(
        {**base, "trigger": {"type": "schedule", "schedule_type": "custom", "cron": "0 9 * * *"}},
        now=datetime(2026, 6, 29, 10, 30),
    ) is None


@pytest.mark.asyncio
async def test_create_automation_rejects_empty_workflow(monkeypatch):
    fake_api = SimpleNamespace(brain=_Brain(), brain_lock=_Lock())
    monkeypatch.setattr(automation_routes, "_get_api", lambda: fake_api)

    with pytest.raises(HTTPException) as exc:
        await automation_routes.create_automation(_valid_body(steps=[]))

    assert exc.value.status_code == 400
    assert "Add at least one action block" in exc.value.detail["errors"][0]
    assert fake_api.brain.records == []


@pytest.mark.asyncio
async def test_create_automation_rejects_broken_visual_graph(monkeypatch):
    fake_api = SimpleNamespace(brain=_Brain(), brain_lock=_Lock())
    monkeypatch.setattr(automation_routes, "_get_api", lambda: fake_api)

    with pytest.raises(HTTPException) as exc:
        await automation_routes.create_automation(_valid_body(
            steps=[
                {"id": "s3", "type": "merge", "label": "Merge", "config": {"mode": "combine_text"}},
            ],
            drawflow_data={
                "drawflow": {
                    "Home": {
                        "data": {
                            "1": {"name": "trigger", "outputs": {"output_1": {"connections": [{"node": "3"}]}}},
                            "2": {"name": "output", "inputs": {"input_1": {"connections": []}}},
                            "3": {
                                "name": "merge",
                                "inputs": {"input_1": {"connections": [{"node": "1"}]}},
                                "outputs": {},
                            },
                        }
                    }
                }
            },
        ))

    assert exc.value.status_code == 400
    assert "Automation has no connected path from trigger to output." in exc.value.detail["errors"]
    assert "Automation Merge block 3 needs at least two connected inputs." in exc.value.detail["errors"]
    assert fake_api.brain.records == []


@pytest.mark.asyncio
async def test_run_automation_records_success(monkeypatch):
    fake_api = SimpleNamespace(brain=_Brain(), brain_lock=_Lock())
    monkeypatch.setattr(automation_routes, "_get_api", lambda: fake_api)
    monkeypatch.setattr(automation_routes, "_execute_automation", lambda _meta: _async_result(("final output", 1)))

    created = await automation_routes.create_automation(_valid_body())
    result = await automation_routes.run_automation_now(created["automation_id"])

    assert result["ok"] is True
    assert result["steps_run"] == 1
    stored = fake_api.brain.search(tags=["automation"])[0].metadata
    assert stored["last_run_status"] == "ok"
    assert stored["run_count"] == 1
    assert stored["consecutive_failures"] == 0
    assert stored["last_output_preview"] == "final output"


@pytest.mark.asyncio
async def test_automation_memory_report_endpoint(tmp_path, monkeypatch):
    from remy.config.settings import settings
    from remy.core import workflow_runs

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    record = workflow_runs.start_workflow_run(kind="automation", workflow_id="auto-test")
    workflow_runs.finish_workflow_run(
        record,
        status="ok",
        trace=[{"id": "s1", "type": "memory_save", "output": "Saved to memory"}],
    )

    report = await automation_routes.get_automation_memory_report("auto-test")

    assert report["workflow_id"] == "auto-test"
    assert report["evaluated_run_count"] == 1
    assert report["totals"]["memory_save_count"] == 1


@pytest.mark.asyncio
async def test_automation_endpoints_reject_path_like_ids():
    bad_id = "../automation"

    for call in [
        lambda: automation_routes.list_automation_runs(bad_id),
        lambda: automation_routes.get_automation_memory_report(bad_id),
        lambda: automation_routes.get_automation_run("auto-test", "../run"),
    ]:
        with pytest.raises(HTTPException) as exc:
            await call()
        assert exc.value.status_code == 400
        assert exc.value.detail == "Invalid id"


@pytest.mark.asyncio
async def test_automation_preflight_warns_for_scheduled_without_baseline(monkeypatch, tmp_path):
    from remy.config.settings import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    fake_api = SimpleNamespace(brain=_Brain(), brain_lock=_Lock())
    monkeypatch.setattr(automation_routes, "_get_api", lambda: fake_api)

    created = await automation_routes.create_automation(_valid_body(
        trigger={"type": "schedule", "schedule_type": "daily", "time_of_day": "09:00"},
        steps=[
            {"id": "s1", "type": "memory_save", "label": "Save", "config": {"text": "{{prev}}"}},
        ],
    ))

    report = await automation_routes.get_automation_preflight(created["automation_id"])

    assert report["ok"] is True
    assert {item["code"] for item in report["warnings"]} >= {
        "scheduled_without_baseline",
        "memory_save_without_dedup",
    }


@pytest.mark.asyncio
async def test_automation_preflight_blocks_scheduled_external_execution(monkeypatch, tmp_path):
    from remy.config.settings import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    fake_api = SimpleNamespace(brain=_Brain(), brain_lock=_Lock())
    monkeypatch.setattr(automation_routes, "_get_api", lambda: fake_api)

    created = await automation_routes.create_automation(_valid_body(
        trigger={"type": "schedule", "schedule_type": "daily", "time_of_day": "09:00"},
        steps=[
            {"id": "s1", "type": "http_request", "label": "Webhook", "config": {"url": "https://example.test"}},
        ],
        output_destination={"type": "webhook", "url": "https://example.test/hook"},
    ))

    report = await automation_routes.get_automation_preflight(created["automation_id"], mode="scheduled_execution")

    assert report["ok"] is False
    assert "scheduled_without_baseline" in {item["code"] for item in report["blockers"]}
    assert "scheduled_external_output_without_approval" in {item["code"] for item in report["blockers"]}


@pytest.mark.asyncio
async def test_automation_preflight_blocks_missing_http_auth_secret(monkeypatch, tmp_path):
    from remy.config.settings import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", None)
    fake_api = SimpleNamespace(brain=_Brain(), brain_lock=_Lock())
    monkeypatch.setattr(automation_routes, "_get_api", lambda: fake_api)

    created = await automation_routes.create_automation(_valid_body(
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

    report = await automation_routes.get_automation_preflight(created["automation_id"])

    assert report["ok"] is False
    assert "missing_http_auth_secret" in {item["code"] for item in report["blockers"]}


@pytest.mark.asyncio
async def test_automation_preflight_blocks_missing_telegram_output_secret(monkeypatch, tmp_path):
    from remy.config.settings import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", None)
    fake_api = SimpleNamespace(brain=_Brain(), brain_lock=_Lock())
    monkeypatch.setattr(automation_routes, "_get_api", lambda: fake_api)

    created = await automation_routes.create_automation(_valid_body(
        output_destination={"type": "telegram", "chat_id": "123"},
    ))

    report = await automation_routes.get_automation_preflight(created["automation_id"])

    assert report["ok"] is False
    assert "missing_telegram_secret" in {item["code"] for item in report["blockers"]}


@pytest.mark.asyncio
async def test_automation_preflight_blocks_missing_email_output_secret(monkeypatch, tmp_path):
    from remy.config.settings import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "SMTP_USER", "")
    monkeypatch.setattr(settings, "SMTP_PASSWORD", None)
    fake_api = SimpleNamespace(brain=_Brain(), brain_lock=_Lock())
    monkeypatch.setattr(automation_routes, "_get_api", lambda: fake_api)

    created = await automation_routes.create_automation(_valid_body(
        output_destination={"type": "email", "to": "user@example.test"},
    ))

    report = await automation_routes.get_automation_preflight(created["automation_id"])
    blocker_codes = {item["code"] for item in report["blockers"]}

    assert report["ok"] is False
    assert "missing_email_account" in blocker_codes
    assert "missing_email_secret" in blocker_codes


@pytest.mark.asyncio
async def test_run_automation_returns_output_beyond_preview_limit(monkeypatch):
    fake_api = SimpleNamespace(brain=_Brain(), brain_lock=_Lock())
    long_output = "x" * 1200
    monkeypatch.setattr(automation_routes, "_get_api", lambda: fake_api)
    monkeypatch.setattr(automation_routes, "_execute_automation", lambda _meta: _async_result((long_output, 1)))

    created = await automation_routes.create_automation(_valid_body())
    result = await automation_routes.run_automation_now(created["automation_id"])

    assert result["output"] == long_output
    assert result["output_length"] == 1200
    assert result["output_truncated"] is False


@pytest.mark.asyncio
async def test_execute_automation_returns_per_step_trace(monkeypatch):
    from remy.core import pipeline_runner

    outputs = ["search output", "ai summary"]

    async def _fake_step(_step, _ctx):
        return outputs.pop(0)

    async def _fake_deliver(*_args, **_kwargs):
        return None

    monkeypatch.setattr(pipeline_runner, "_execute_step", _fake_step)
    monkeypatch.setattr(automation_routes, "_deliver", _fake_deliver)

    output, steps_run, trace = await automation_routes._execute_automation({
        "name": "Trace test",
        "trigger": {"type": "manual"},
        "steps": [
            {"id": "s1", "type": "web_search", "label": "Web Search", "config": {"query": "x"}},
            {"id": "s2", "type": "llm_call", "label": "AI Response", "config": {"prompt": "{{prev}}"}},
        ],
        "output_destination": {"type": "chat"},
    })

    assert output == "ai summary"
    assert steps_run == 2
    assert [(item["label"], item["status"], item["output"]) for item in trace] == [
        ("Web Search", "ok", "search output"),
        ("AI Response", "ok", "ai summary"),
    ]


@pytest.mark.asyncio
async def test_execute_automation_retries_step_before_failing(monkeypatch):
    from remy.core import pipeline_runner

    calls = {"count": 0}

    async def _flaky_step(_step, _ctx):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary error")
        return "recovered"

    async def _fake_deliver(*_args, **_kwargs):
        return None

    monkeypatch.setattr(pipeline_runner, "_execute_step", _flaky_step)
    monkeypatch.setattr(automation_routes, "_deliver", _fake_deliver)

    output, steps_run, trace = await automation_routes._execute_automation({
        "name": "Retry test",
        "trigger": {"type": "manual"},
        "steps": [
            {
                "id": "s1",
                "type": "template",
                "label": "Flaky",
                "config": {"text": "x", "_retry_enabled": True, "_retry_count": 1},
            },
        ],
        "output_destination": {"type": "chat"},
    })

    assert output == "recovered"
    assert steps_run == 1
    assert calls["count"] == 2
    assert trace[0]["retry_attempts"] == 2
    assert trace[0]["recovered_from_error"] == "temporary error"


@pytest.mark.asyncio
async def test_execute_automation_router_follows_selected_branch(monkeypatch):
    from remy.core import pipeline_runner

    async def _fake_step(step, _ctx):
        if step["type"] == "router":
            return "2"
        return step["label"].lower()

    async def _fake_deliver(*_args, **_kwargs):
        return None

    monkeypatch.setattr(pipeline_runner, "_execute_step", _fake_step)
    monkeypatch.setattr(automation_routes, "_deliver", _fake_deliver)

    output, steps_run, trace = await automation_routes._execute_automation({
        "name": "Router automation",
        "trigger": {"type": "manual"},
        "steps": [
            {"id": "s3", "type": "router", "label": "Router", "config": {"_data_ref": "{{prev}}"}},
            {"id": "s4", "type": "template", "label": "Chosen", "config": {"text": "chosen"}},
            {"id": "s5", "type": "template", "label": "Skipped", "config": {"text": "skipped"}},
        ],
        "output_destination": {"type": "chat"},
        "drawflow_data": {
            "drawflow": {
                "Home": {
                    "data": {
                        "1": {"name": "trigger", "outputs": {"output_1": {"connections": [{"node": "3"}]}}},
                        "2": {"name": "output", "outputs": {}},
                        "3": {"name": "router", "outputs": {
                            "output_1": {"connections": [{"node": "5"}]},
                            "output_2": {"connections": [{"node": "4"}]},
                        }},
                        "4": {"name": "template", "outputs": {"output_1": {"connections": [{"node": "2"}]}}},
                        "5": {"name": "template", "outputs": {"output_1": {"connections": [{"node": "2"}]}}},
                    }
                }
            }
        },
    })

    assert output == "chosen"
    assert steps_run == 2
    assert [item["id"] for item in trace] == ["s3", "s4"]
    assert trace[0]["output"] == "Selected routes: output_2"


@pytest.mark.asyncio
async def test_execute_automation_router_can_deliver_multiple_branches(monkeypatch):
    from remy.core import pipeline_runner

    async def _fake_step(step, _ctx):
        if step["type"] == "router":
            return "1,2"
        return step["label"].lower()

    delivered = {}

    async def _fake_deliver(output, *_args, **_kwargs):
        delivered["output"] = output

    monkeypatch.setattr(pipeline_runner, "_execute_step", _fake_step)
    monkeypatch.setattr(automation_routes, "_deliver", _fake_deliver)

    output, steps_run, trace = await automation_routes._execute_automation({
        "name": "Router automation",
        "trigger": {"type": "manual"},
        "steps": [
            {"id": "s3", "type": "router", "label": "Router", "config": {"_data_ref": "{{prev}}"}},
            {"id": "s4", "type": "template", "label": "One", "config": {"text": "one"}},
            {"id": "s5", "type": "template", "label": "Two", "config": {"text": "two"}},
        ],
        "output_destination": {"type": "chat"},
        "drawflow_data": {
            "drawflow": {
                "Home": {
                    "data": {
                        "1": {"name": "trigger", "outputs": {"output_1": {"connections": [{"node": "3"}]}}},
                        "2": {"name": "output", "outputs": {}},
                        "3": {"name": "router", "outputs": {
                            "output_1": {"connections": [{"node": "4"}]},
                            "output_2": {"connections": [{"node": "5"}]},
                        }},
                        "4": {"name": "template", "outputs": {"output_1": {"connections": [{"node": "2"}]}}},
                        "5": {"name": "template", "outputs": {"output_1": {"connections": [{"node": "2"}]}}},
                    }
                }
            }
        },
    })

    assert output == "one\n\n---\n\ntwo"
    assert delivered["output"] == output
    assert steps_run == 3
    assert [item["id"] for item in trace] == ["s3", "s4", "s5"]


@pytest.mark.asyncio
async def test_execute_automation_merge_waits_for_router_branches(monkeypatch):
    delivered = {}

    async def _fake_deliver(output, *_args, **_kwargs):
        delivered["output"] = output

    monkeypatch.setattr(automation_routes, "_deliver", _fake_deliver)

    output, steps_run, trace = await automation_routes._execute_automation({
        "name": "Merge automation",
        "trigger": {"type": "manual"},
        "steps": [
            {
                "id": "s3",
                "type": "router",
                "label": "Router",
                "config": {
                    "_data_ref": "{{prev}}",
                    "mode": "all",
                    "routes": [
                        {"label": "A", "operator": "always", "value": ""},
                        {"label": "B", "operator": "always", "value": ""},
                    ],
                },
            },
            {"id": "s4", "type": "template", "label": "A", "config": {"text": "branch A"}},
            {"id": "s5", "type": "template", "label": "B", "config": {"text": "branch B"}},
            {"id": "s6", "type": "merge", "label": "Merge", "config": {"mode": "combine_text", "separator": " | ", "input_count": 2}},
        ],
        "output_destination": {"type": "chat"},
        "drawflow_data": {
            "drawflow": {
                "Home": {
                    "data": {
                        "1": {"name": "trigger", "outputs": {"output_1": {"connections": [{"node": "3", "output": "input_1"}]}}},
                        "2": {"name": "output", "outputs": {}},
                        "3": {"name": "router", "outputs": {
                            "output_1": {"connections": [{"node": "4", "output": "input_1"}]},
                            "output_2": {"connections": [{"node": "5", "output": "input_1"}]},
                        }},
                        "4": {"name": "template", "outputs": {"output_1": {"connections": [{"node": "6", "output": "input_1"}]}}},
                        "5": {"name": "template", "outputs": {"output_1": {"connections": [{"node": "6", "output": "input_2"}]}}},
                        "6": {
                            "name": "merge",
                            "inputs": {
                                "input_1": {"connections": [{"node": "4", "input": "output_1"}]},
                                "input_2": {"connections": [{"node": "5", "input": "output_1"}]},
                            },
                            "outputs": {"output_1": {"connections": [{"node": "2", "output": "input_1"}]}},
                        },
                    }
                }
            }
        },
    })

    assert output == "branch A | branch B"
    assert delivered["output"] == output
    assert steps_run == 4
    assert [item["id"] for item in trace] == ["s3", "s4", "s5", "s6"]


@pytest.mark.asyncio
async def test_execute_automation_error_output_routes_to_error_handler(monkeypatch):
    delivered = {}

    async def _fake_deliver(output, *_args, **_kwargs):
        delivered["output"] = output

    monkeypatch.setattr(automation_routes, "_deliver", _fake_deliver)

    output, steps_run, trace = await automation_routes._execute_automation({
        "name": "Error handler automation",
        "trigger": {"type": "manual"},
        "steps": [
            {
                "id": "s3",
                "type": "parse_json",
                "label": "Parse",
                "config": {"text": "not json", "path": "$"},
            },
            {
                "id": "s4",
                "type": "error_handler",
                "label": "Recover",
                "config": {"fallback_text": "fallback {{error}}"},
            },
        ],
        "output_destination": {"type": "chat"},
        "drawflow_data": {
            "drawflow": {
                "Home": {
                    "data": {
                        "1": {"name": "trigger", "outputs": {"output_1": {"connections": [{"node": "3", "output": "input_1"}]}}},
                        "2": {"name": "output", "outputs": {}},
                        "3": {
                            "name": "parse_json",
                            "outputs": {
                                "output_1": {"connections": []},
                                "output_2": {"connections": [{"node": "4", "output": "input_1"}]},
                            },
                        },
                        "4": {
                            "name": "error_handler",
                            "outputs": {"output_1": {"connections": [{"node": "2", "output": "input_1"}]}},
                        },
                    }
                }
            }
        },
    })

    assert output.startswith("fallback Step 1 failed: [JSON parse error:")
    assert delivered["output"] == output
    assert steps_run == 1
    assert [(item["id"], item["status"]) for item in trace] == [("s3", "error"), ("s4", "ok")]


@pytest.mark.asyncio
async def test_run_automation_records_and_auto_pauses_repeated_failures(monkeypatch):
    fake_api = SimpleNamespace(brain=_Brain(), brain_lock=_Lock())
    monkeypatch.setattr(automation_routes, "_get_api", lambda: fake_api)

    async def _fail(_meta):
        raise automation_routes.AutomationExecutionError("broken step")

    monkeypatch.setattr(automation_routes, "_execute_automation", _fail)

    created = await automation_routes.create_automation(_valid_body())
    for _ in range(3):
        with pytest.raises(HTTPException):
            await automation_routes.run_automation_now(created["automation_id"])

    stored = fake_api.brain.search(tags=["automation"])[0].metadata
    assert stored["last_run_status"] == "error"
    assert stored["failure_count"] == 3
    assert stored["consecutive_failures"] == 3
    assert stored["enabled"] is False
    assert "Automatically paused" in stored["disabled_reason"]


async def _async_result(value):
    return value
