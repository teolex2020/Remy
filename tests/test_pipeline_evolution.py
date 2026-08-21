import json

import pytest


def _session_log(query="AI agent news"):
    return [
        {"type": "user_text", "text": "check the latest AI agent news"},
        {
            "type": "tool_call",
            "tool": "web_search",
            "args_full": {"query": query},
            "result": "source result",
        },
    ]


def test_repeated_successful_turn_becomes_draft(tmp_path, monkeypatch):
    from remy.config.settings import settings
    from remy.core.pipeline_evolution import observe_successful_turn

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    first = observe_successful_turn(
        session_id="s1",
        user_text="Check the latest AI agent news and summarize it",
        session_log=_session_log(),
    )
    second = observe_successful_turn(
        session_id="s1",
        user_text="Please check latest AI agent news and summarize it",
        session_log=_session_log(),
    )

    assert first["status"] == "observing"
    assert second["status"] == "draft"
    assert second["occurrence_count"] == 2
    assert [step["type"] for step in second["pipeline"]["steps"]] == ["web_search", "llm_call"]
    assert second["risk"]["approval_required"] is True


def test_explicit_recurring_request_immediately_drafts_schedule(tmp_path, monkeypatch):
    from remy.config.settings import settings
    from remy.core.pipeline_evolution import observe_successful_turn

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    item = observe_successful_turn(
        session_id="s2",
        user_text="Every day at 08:30 check the project website and summarize changes",
        session_log=_session_log(),
    )

    assert item["status"] == "draft"
    assert item["trigger"] == {"type": "schedule", "schedule_type": "daily", "time_of_day": "08:30"}


def test_repeated_greeting_never_becomes_pipeline(tmp_path, monkeypatch):
    from remy.config.settings import settings
    from remy.core.pipeline_evolution import observe_successful_turn

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    for _ in range(3):
        item = observe_successful_turn(
            session_id="small-talk",
            user_text="Привіт, як справи",
            session_log=[
                {"type": "user_text", "text": "Привіт, як справи"},
                {"type": "model_response", "text": "Усе добре"},
            ],
        )
        assert item is None

    assert not (tmp_path / "pipeline_evolution.sqlite3").exists()


def test_compiler_marks_side_effect_risk_and_deduplicates_steps():
    from remy.core.pipeline_evolution import compile_pipeline

    compiled = compile_pipeline("Save a recurring digest", [
        {"tool": "web_search", "args": {"query": "digest"}},
        {"tool": "web_search", "args": {"query": "digest"}},
        {"tool": "store", "args": {"content": "x"}},
    ])

    assert [step["type"] for step in compiled["steps"]] == ["web_search", "memory_save", "llm_call"]
    assert compiled["risk"]["level"] == "medium"


def test_dry_run_replaces_mutations():
    from remy.web.routes.pipeline_routes import _safe_dry_run_steps

    safe = _safe_dry_run_steps([
        {"id": "s1", "type": "memory_save", "label": "Save", "config": {"text": "x"}},
        {"id": "s2", "type": "http_request", "label": "Post", "config": {"method": "POST", "url": "https://example.com"}},
        {"id": "s3", "type": "page_scrape", "label": "Read", "config": {"url": "https://example.com"}},
    ])

    assert [step["type"] for step in safe] == ["template", "template", "page_scrape"]


def test_agent_can_explicitly_propose_and_list_candidate(tmp_path, monkeypatch):
    from remy.config.settings import settings
    from remy.core.tool_dispatch import execute_tool

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    proposed = json.loads(execute_tool(
        "propose_pipeline_candidate",
        {"task": "Prepare a reusable weekly project digest"},
        session_id="agent-session",
        channel="desktop",
    ))
    listed = json.loads(execute_tool("list_pipeline_candidates", {"status": "draft"}))

    assert proposed["candidate"]["status"] == "draft"
    assert listed["items"][0]["candidate_id"] == proposed["candidate"]["candidate_id"]


@pytest.mark.asyncio
async def test_approval_requires_successful_dry_run(tmp_path, monkeypatch):
    from fastapi import HTTPException
    from remy.config.settings import settings
    from remy.core.pipeline_evolution import observe_successful_turn
    from remy.web.routes import pipeline_routes

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    item = observe_successful_turn(
        session_id="s3",
        user_text="Automatically summarize these notes whenever I ask",
        session_log=[],
        force_draft=True,
    )
    with pytest.raises(HTTPException) as exc:
        await pipeline_routes.decide_pipeline_candidate(
            item["candidate_id"], pipeline_routes.PipelineCandidateDecisionRequest(decision="approve")
        )
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_dry_run_then_approval_materializes_manual_pipeline(tmp_path, monkeypatch):
    from remy.config.settings import settings
    from remy.core.pipeline_evolution import observe_successful_turn
    from remy.web.routes import pipeline_routes
    import remy.core.pipeline_runner as runner

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    item = observe_successful_turn(
        session_id="s4",
        user_text="Summarize my project notes in a repeatable way",
        session_log=[],
        force_draft=True,
    )

    async def fake_run(_steps, _input):
        yield {"type": "step_done", "id": "s1", "step_type": "llm_call", "label": "Result", "output": "ok"}
        yield {"type": "done", "output": "ok"}

    monkeypatch.setattr(runner, "run_pipeline_steps", fake_run)
    dry = await pipeline_routes.dry_run_pipeline_candidate(
        item["candidate_id"], pipeline_routes.PipelineCandidateDryRunRequest()
    )
    assert dry["dry_run"]["passed"] is True

    approved = await pipeline_routes.decide_pipeline_candidate(
        item["candidate_id"], pipeline_routes.PipelineCandidateDecisionRequest(decision="approve")
    )
    assert approved["status"] == "activated"
    assert approved["pipeline_id"]
    saved = json.loads((tmp_path / "pipelines" / f"{approved['pipeline_id']}.json").read_text(encoding="utf-8"))
    assert saved["generated_from_candidate"] == item["candidate_id"]
    assert saved["success_criteria"]


@pytest.mark.asyncio
async def test_scheduled_candidate_approval_activates_automation(tmp_path, monkeypatch):
    from remy.config.settings import settings
    from remy.core.pipeline_evolution import observe_successful_turn, record_dry_run
    from remy.web.routes import automation_routes, pipeline_routes

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    item = observe_successful_turn(
        session_id="s5",
        user_text="Every day at 07:45 prepare a project digest",
        session_log=[],
        force_draft=True,
    )
    record_dry_run(item["candidate_id"], {"passed": True, "output_preview": "ok"}, passed=True)
    captured = {}

    async def fake_create(body):
        captured["body"] = body
        return {"ok": True, "automation_id": "auto-generated"}

    monkeypatch.setattr(automation_routes, "create_automation", fake_create)
    approved = await pipeline_routes.decide_pipeline_candidate(
        item["candidate_id"], pipeline_routes.PipelineCandidateDecisionRequest(decision="approve")
    )

    assert approved["automation_id"] == "auto-generated"
    assert captured["body"].enabled is True
    assert captured["body"].trigger["time_of_day"] == "07:45"
    assert captured["body"].generated_from_candidate == item["candidate_id"]
