import asyncio


def test_build_autonomy_status_payload_includes_runtime_snapshot_fields(monkeypatch):
    from remy.core.event_bus import event_bus
    from remy.web.routes import autonomy_routes as routes

    runtime_status = {
        "running": True,
        "current_goal": {"id": "goal-1", "description": "Verify source chain", "priority": "high"},
        "current_mission": {
            "id": "m1",
            "description": "Research target",
            "active_tasks": 2,
            "focus_stale_cycles": 1,
            "completed_tasks": 3,
            "pending_tasks": 2,
            "blocked_tasks": 1,
            "failed_tasks": 1,
            "pending_task_labels": ["Collect filings", "Compare source trail"],
            "pending_task_items": [
                {"goal_id": "goal-a", "label": "Collect filings", "status": "active", "detail": ""},
                {"goal_id": "goal-b", "label": "Compare source trail", "status": "pending", "detail": ""},
                {"goal_id": "goal-c", "label": "Resolve login blocker", "status": "blocked_external", "detail": "Waiting for captcha bypass"},
            ],
            "total_tasks": 5,
        },
        "current_task": {"action": "web_search(latest filings)"},
        "current_step": {"instruction": "Collect primary sources", "step_num": 2, "total_steps": 4},
        "last_cycle_result": {"decision": "success", "reason": "Collected evidence"},
        "current_role": "researcher",
        "last_agent_response": {
            "response": "Collected the primary sources and summarized them.",
            "duration_ms": 2400,
            "tokens_estimated": 700,
        },
        "last_research_activity": {
            "tool": "web_search",
            "calls": 3,
            "successful_calls": 3,
            "gathered_sources": 2,
            "stored_findings": 1,
            "summary": "Research: 3 calls, 2 source steps, 1 storage steps, last web_search",
        },
        "research_session": {
            "session_id": "rs-1",
            "topic": "Source chain verification",
            "generated_queries_count": 3,
            "accepted_sources_count": 2,
            "findings_count": 1,
            "contradictions_count": 0,
            "citation_coverage_rate": 1.0,
            "recent_queries": ["source chain sec filing"],
            "knowledge_gaps": ["Synthesize accepted sources into explicit findings."],
            "artifact": {
                "record_id": "report-1",
                "artifact_format": "markdown",
                "viewer_url": "/api/autonomy/research-artifacts/report-1/view",
                "markdown_url": "/api/autonomy/research-artifacts/report-1/markdown",
                "pdf_url": "/api/reports/source-chain.pdf",
                "pdf_filename": "source-chain.pdf",
                "markdown_available": True,
                "markdown_preview": "# Source chain verification",
            },
        },
        "scheduler_reason": "active",
        "scheduler_selection": {
            "mission_id": "m1",
            "score": 3.4,
            "reason": "runnable_task,routing_prefer=researcher:0.20",
            "runnable_count": 2,
            "details": {
                "routing_reason": "routing_prefer=researcher",
                "routing_factor": 0.2,
            },
        },
        "specialist_resolution": {
            "specialist_id": "researcher",
            "reason": "routing_pressure_override:task_specialist:executor->researcher",
            "quality_factor": 0.85,
            "sensitive": True,
        },
    }

    monkeypatch.setattr(
        "remy.core.combined_runner.get_activity_runtime_snapshot",
        lambda goal_limit=3, approval_limit=10, transport_connected=False: {
            **runtime_status,
            "running": True,
            "session_id": "sess-123",
            "version": "v2",
            "transport_connected": transport_connected,
            "pending_approvals": 2,
            "approval_queue": [
                {"id": "appr-1", "description": "Review payout request", "age_sec": 12},
                {"id": "appr-2", "description": "Approve publish action", "age_sec": 6},
            ],
            "budget": {"daily_limit_usd": 5.0, "daily_spent_usd": 1.25},
            "evaluation": {"failure_history_size": 2},
            "factuality": {"unsupported_observed_claims_total": 3},
            "quality_debt_by_specialist": [{"id": "researcher", "quality_debt": 0.2, "unsupported_claims": 3}],
            "evidence_debt_queue": [{"id": "debt-1", "action": "Verify evidence before relying on action: answer_without_evidence."}],
            "scheduler_decisions_recent": [{"specialist": "researcher", "reason": "fallback after low evidence"}],
            "routing_pressure": {
                "top_candidate": {"id": "analyst", "quality_adjusted_success_rate": 0.91},
                "highest_pressure": {"id": "researcher", "quality_debt": 0.2},
            },
        },
    )
    monkeypatch.setattr(type(event_bus), "subscriber_count", property(lambda self: 2))

    payload = asyncio.run(routes.build_autonomy_status_payload())

    assert payload["running"] is True
    assert payload["version"] == "v2"
    assert payload["session_id"] == "sess-123"
    assert payload["transport_connected"] is True
    assert payload["current_goal"]["id"] == "goal-1"
    assert payload["current_mission"]["active_tasks"] == 2
    assert payload["current_mission"]["completed_tasks"] == 3
    assert payload["current_mission"]["pending_tasks"] == 2
    assert payload["current_mission"]["blocked_tasks"] == 1
    assert payload["current_mission"]["failed_tasks"] == 1
    assert payload["current_mission"]["pending_task_labels"][0] == "Collect filings"
    assert payload["current_mission"]["pending_task_items"][0]["status"] == "active"
    assert payload["current_mission"]["pending_task_items"][0]["goal_id"] == "goal-a"
    assert payload["current_mission"]["pending_task_items"][2]["detail"] == "Waiting for captcha bypass"
    assert payload["current_role"] == "researcher"
    assert payload["last_agent_response"]["duration_ms"] == 2400
    assert payload["last_research_activity"]["gathered_sources"] == 2
    assert payload["research_session"]["session_id"] == "rs-1"
    assert payload["research_session"]["recent_queries"][0] == "source chain sec filing"
    assert payload["research_session"]["artifact"]["viewer_url"].endswith("/report-1/view")
    assert payload["research_session"]["artifact"]["markdown_url"].endswith("/report-1/markdown")
    assert payload["research_session"]["artifact"]["pdf_filename"] == "source-chain.pdf"
    assert payload["budget"]["daily_spent_usd"] == 1.25
    assert payload["pending_approvals"] == 2
    assert payload["approval_queue"][0]["id"] == "appr-1"
    assert payload["evaluation"]["failure_history_size"] == 2
    assert payload["factuality"]["unsupported_observed_claims_total"] == 3
    assert payload["quality_debt_by_specialist"][0]["id"] == "researcher"
    assert payload["evidence_debt_queue"][0]["id"] == "debt-1"
    assert payload["scheduler_decisions_recent"][0]["specialist"] == "researcher"
    assert payload["routing_pressure"]["top_candidate"]["id"] == "analyst"
    assert payload["scheduler_selection"]["mission_id"] == "m1"
    assert payload["scheduler_selection"]["details"]["routing_reason"] == "routing_prefer=researcher"
    assert payload["specialist_resolution"]["specialist_id"] == "researcher"
    assert payload["specialist_resolution"]["reason"].startswith("routing_pressure_override:")


def test_build_autonomy_status_payload_returns_explicit_defaults_without_loop():
    from remy.web.routes import autonomy_routes as routes
    from remy.core.event_bus import event_bus

    from remy.core import combined_runner

    original_get_snapshot = combined_runner.get_activity_runtime_snapshot
    original_subscriber_count = type(event_bus).subscriber_count
    combined_runner.get_activity_runtime_snapshot = lambda goal_limit=3, approval_limit=10, transport_connected=False: {
        "running": False,
        "session_id": None,
        "version": "v2",
        "transport_connected": transport_connected,
        "pending_approvals": 1,
        "approval_queue": [{"id": "appr-9", "description": "Need approval", "age_sec": 2}],
        "budget": {"alert_level": "green", "llm_cost_today": 0.1},
        "evaluation": {},
        "factuality": {},
        "quality_debt_by_specialist": [],
        "evidence_debt_queue": [],
        "scheduler_decisions_recent": [],
        "routing_pressure": {},
        "scheduler_selection": {},
        "specialist_resolution": {},
    }
    type(event_bus).subscriber_count = property(lambda self: 0)
    try:
        payload = asyncio.run(routes.build_autonomy_status_payload())
    finally:
        combined_runner.get_activity_runtime_snapshot = original_get_snapshot
        type(event_bus).subscriber_count = original_subscriber_count

    assert payload["running"] is False
    assert payload["version"] == "v2"
    assert payload["transport_connected"] is False
    assert "current_goal" in payload
    assert "current_role" in payload
    assert "last_agent_response" in payload
    assert "last_research_activity" in payload
    assert "research_session" in payload
    assert "quality_debt_by_specialist" in payload
    assert "evidence_debt_queue" in payload
    assert "evaluation" in payload
    assert "factuality" in payload
    assert "routing_pressure" in payload
    assert "scheduler_selection" in payload
    assert "specialist_resolution" in payload
    assert payload["pending_approvals"] == 1
    assert payload["approval_queue"][0]["id"] == "appr-9"
    assert payload["budget"]["llm_cost_today"] == 0.1


def test_get_survival_status_uses_shared_budget_snapshot(monkeypatch):
    from remy.web.routes import autonomy_routes as routes

    monkeypatch.setattr(
        "remy.core.combined_runner.get_budget_runtime_snapshot",
        lambda goal_limit=3, approval_limit=10: {
            "balance_usd": 12.5,
            "usdt": 5.0,
            "trx": 20.0,
            "runway_days": 14,
            "alert_level": "yellow",
            "llm_cost_today": 0.82,
            "llm_cost_lifetime_usd": 12.34,
            "llm_tokens_today": 321,
            "llm_tokens_lifetime": 6543,
            "last_check": "2026-03-19T10:00:00",
        },
    )
    monkeypatch.setattr(
        "remy.core.survival.check_wallet_balance",
        lambda: {"address": "wallet-1", "trx": 20.0, "usdt": 5.0, "error": None},
    )
    monkeypatch.setattr("remy.core.survival.estimate_runway", lambda total_usd, llm_cost_today=0.0: 14.0)

    payload = asyncio.run(routes.get_survival_status())

    assert payload["wallet"]["address"] == "wallet-1"
    assert payload["runway"]["days"] == 14.0
    assert payload["runway"]["daily_burn_usd"] == 1.32
    assert payload["spending"]["llm_cost_today_usd"] == 0.82
    assert payload["spending"]["llm_cost_lifetime_usd"] == 12.34
    assert payload["spending"]["llm_tokens_today"] == 321
    assert payload["spending"]["llm_tokens_lifetime"] == 6543
    assert payload["last_check"]["alert_level"] == "yellow"


def test_get_research_artifact_markdown_returns_canonical_markdown(monkeypatch):
    from types import SimpleNamespace
    from remy.web.routes import autonomy_routes as routes

    class _FakeBrain:
        def get(self, record_id):
            assert record_id == "report-1"
            return SimpleNamespace(
                id="report-1",
                content="fallback body",
                metadata={"markdown_body": "# Research Report\n\nCanonical body"},
            )

    class _FakeLock:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        routes,
        "_get_api",
        lambda: SimpleNamespace(brain=_FakeBrain(), brain_lock=_FakeLock()),
    )

    response = asyncio.run(routes.get_research_artifact_markdown("report-1"))

    assert response.status_code == 200
    assert response.body.decode("utf-8").startswith("# Research Report")


def test_get_research_artifact_viewer_returns_shell(monkeypatch):
    from types import SimpleNamespace
    from remy.web.routes import autonomy_routes as routes

    class _FakeBrain:
        def get(self, record_id):
            assert record_id == "report-1"
            return SimpleNamespace(
                id="report-1",
                content="# Canonical Report\n\nBody",
                metadata={
                    "title": "Canonical Report",
                    "pdf_url": "/api/reports/report-1.pdf",
                    "pdf_filename": "report-1.pdf",
                    "citation_complete": True,
                    "citation_count": 4,
                    "findings_count": 3,
                    "confidence_avg": 0.82,
                    "evidence_note": "All core findings are source-backed.",
                    "sources": [
                        "https://sec.gov/report",
                        "https://www.nasdaq.com/article",
                    ],
                    "contradictions_count": 1,
                },
            )

    class _FakeLock:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        routes,
        "_get_api",
        lambda: SimpleNamespace(brain=_FakeBrain(), brain_lock=_FakeLock()),
    )

    response = asyncio.run(routes.get_research_artifact_viewer("report-1"))

    body = response.body.decode("utf-8")
    assert response.status_code == 200
    assert "/js/research-viewer.js" in body
    assert "/api/autonomy/research-artifacts/report-1/markdown" in body
    assert 'data-record-id="report-1"' in body
    assert "Download Markdown" in body
    assert "/api/reports/report-1.pdf" in body
    assert "Sections" in body
    assert "Citation Status" in body
    assert "All core findings are source-backed." in body
    assert "sec.gov, www.nasdaq.com" in body
    assert "1 contradictions" in body
