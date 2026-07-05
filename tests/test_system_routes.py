import json
import asyncio
from contextlib import nullcontext
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_client():
    from remy.web.routes.system_routes import router

    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def test_system_packs_marks_disabled(monkeypatch):
    from remy.config.settings import settings

    monkeypatch.setattr(settings, "PACKS_DISABLED", ["publisher"])
    client = _make_client()

    res = client.get("/api/system/packs")
    assert res.status_code == 200
    packs = {pack["id"]: pack for pack in res.json()["packs"]}
    assert packs["publisher"]["enabled"] is False
    assert packs["general"]["enabled"] is True


def test_system_pack_toggle_updates_runtime_settings(tmp_path, monkeypatch):
    import importlib
    from remy.config.settings import settings

    settings_module = importlib.import_module("remy.config.settings")
    runtime_file = tmp_path / "runtime_settings.json"
    monkeypatch.setattr(settings, "PACKS_DISABLED", [])
    monkeypatch.setattr(settings_module, "RUNTIME_SETTINGS_FILE", runtime_file)

    client = _make_client()
    res = client.post("/api/system/packs/publisher", json={"enabled": False})
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["enabled"] is False
    assert "publisher" in data["disabled_packs"]

    saved = json.loads(runtime_file.read_text(encoding="utf-8"))
    assert saved["PACKS_DISABLED"] == ["publisher"]


def test_resolve_pack_falls_back_when_pack_disabled(monkeypatch):
    from remy.config.settings import settings
    from remy.core.capability_packs import GENERAL, resolve_pack

    monkeypatch.setattr(settings, "PACKS_DISABLED", ["publisher"])

    pack = resolve_pack({"description": "write article draft for dev.to"})
    assert pack.id == GENERAL.id


def test_system_status_includes_recent_operator_alerts(monkeypatch):
    import remy.web.routes.system_routes as system_routes

    fake_api = type("Api", (), {})()
    fake_api._start_time = 0
    fake_api.brain = type("Brain", (), {"count": lambda self: 3, "search": lambda self, **kwargs: []})()
    from contextlib import nullcontext
    fake_api.brain_lock = nullcontext()
    fake_api.settings = type("Settings", (), {"AURA_BRAIN_PATH": "data/brain"})()

    monkeypatch.setattr(system_routes, "_get_api", lambda: fake_api)

    client = _make_client()
    with monkeypatch.context() as m:
        m.setattr(
            "remy.core.notification_router.get_recent_notifications",
            lambda **kwargs: [
                {
                    "id": "alert-1",
                    "type": "operator_alert",
                    "level": "warning",
                    "message": "Gateway degraded",
                    "timestamp": 123.0,
                    "acknowledged": False,
                    "gateway_health": "degraded",
                    "health_level": "YELLOW",
                    "source": "harness_eval",
                    "scenario_id": "verify_gate_ablation",
                    "action_target": "open_memory_verification",
                    "artifact_ids": ["report-1"],
                    "failure_code": "verification_failed",
                    "verification_status": "repair_required",
                    "verification_reason": "Artifact needs repair.",
                    "requested": 1,
                    "applied": 0,
                    "skipped": 1,
                }
            ],
        )
        res = client.get("/api/system/status")

    assert res.status_code == 200
    data = res.json()
    assert data["operator_alerts"]["count"] == 1
    assert data["operator_alerts"]["unacknowledged_count"] == 1
    assert data["operator_alerts"]["items"][0]["message"] == "Gateway degraded"
    assert data["operator_alerts"]["items"][0]["source"] == "harness_eval"
    assert data["operator_alerts"]["items"][0]["scenario_id"] == "verify_gate_ablation"
    assert data["operator_alerts"]["items"][0]["action_target"] == "open_memory_verification"
    assert data["operator_alerts"]["items"][0]["artifact_ids"] == ["report-1"]
    assert data["operator_alerts"]["items"][0]["failure_code"] == "verification_failed"
    assert data["operator_alerts"]["items"][0]["verification_status"] == "repair_required"
    assert data["operator_alerts"]["items"][0]["requested"] == 1
    assert data["operator_alerts"]["items"][0]["applied"] == 0
    assert data["operator_alerts"]["items"][0]["skipped"] == 1


def test_acknowledge_operator_alert_route(monkeypatch):
    client = _make_client()
    with monkeypatch.context() as m:
        m.setattr(
            "remy.core.notification_router.acknowledge_notification",
            lambda alert_id: alert_id == "alert-1",
        )
        res = client.post("/api/system/operator-alerts/alert-1/ack")

    assert res.status_code == 200
    assert res.json()["ok"] is True


def test_reconstruct_missing_memory_route(monkeypatch):
    import remy.web.routes.system_routes as system_routes
    import remy.core.brain_tools as brain_tools

    stored = []

    class FakeBrain:
        def search(self, **kwargs):
            return []

        def store(self, content, level=None, tags=None, metadata=None):
            stored.append(
                {
                    "content": content,
                    "level": level,
                    "tags": list(tags or []),
                    "metadata": metadata or {},
                }
            )
            return type("Record", (), {"id": f"rec-{len(stored)}"})()

    fake_api = type("Api", (), {})()
    fake_api._start_time = 0
    fake_api.brain = FakeBrain()
    fake_api.settings = type("Settings", (), {"AURA_BRAIN_PATH": "data/brain", "DATA_DIR": "data"})()
    emitted = []

    monkeypatch.setattr(system_routes, "_get_api", lambda: fake_api)
    monkeypatch.setattr(brain_tools, "brain", fake_api.brain, raising=False)
    monkeypatch.setattr(brain_tools, "brain_lock", nullcontext(), raising=False)
    monkeypatch.setattr(
        "remy.core.history_replay.analyze_history_memory_gaps",
        lambda *args, **kwargs: {
            "recent_missing": [{"candidate_id": "session.json:1:schedule_task", "label": "Schedule task"}],
        },
    )
    monkeypatch.setattr(
        "remy.core.history_replay.reconstruct_history_candidates",
        lambda execute_fn, **kwargs: {
            "requested": 1,
            "applied": 1,
            "skipped": 0,
            "missing_candidate_ids": [],
            "tool_errors": [],
            "verification": {"status": "verified", "verified": True, "reason": "ok"},
        },
    )
    monkeypatch.setattr(
        "remy.core.notification_router.notify",
        lambda message, **kwargs: emitted.append({"message": message, **kwargs}),
    )

    client = _make_client()
    res = client.post("/api/system/memory/reconstruct-missing", json={"candidate_ids": []})

    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["stats"]["applied"] == 1
    assert data["stats"]["verification"]["status"] == "verified"
    assert emitted
    assert emitted[0]["event_type"] == "verification.resolved"
    assert stored
    assert "reconstruction_review" in stored[0]["tags"]
    assert stored[0]["metadata"]["applied"] == 1
    assert stored[0]["metadata"]["verification"]["status"] == "verified"


def test_reconstruct_missing_memory_emits_operator_alert_on_verification_failure(monkeypatch):
    import remy.web.routes.system_routes as system_routes
    import remy.core.brain_tools as brain_tools

    stored = []

    class FakeBrain:
        def search(self, **kwargs):
            return []

        def store(self, content, level=None, tags=None, metadata=None):
            stored.append(
                {
                    "content": content,
                    "level": level,
                    "tags": list(tags or []),
                    "metadata": metadata or {},
                }
            )
            return type("Record", (), {"id": f"rec-{len(stored)}"})()

    fake_api = type("Api", (), {})()
    fake_api._start_time = 0
    fake_api.brain = FakeBrain()
    fake_api.settings = type("Settings", (), {"AURA_BRAIN_PATH": "data/brain", "DATA_DIR": "data"})()

    emitted = []

    monkeypatch.setattr(system_routes, "_get_api", lambda: fake_api)
    monkeypatch.setattr(brain_tools, "brain", fake_api.brain, raising=False)
    monkeypatch.setattr(brain_tools, "brain_lock", nullcontext(), raising=False)
    monkeypatch.setattr(
        "remy.core.history_replay.analyze_history_memory_gaps",
        lambda *args, **kwargs: {
            "recent_missing": [{"candidate_id": "session.json:1:schedule_task", "label": "Schedule task"}],
        },
    )
    monkeypatch.setattr(
        "remy.core.history_replay.reconstruct_history_candidates",
        lambda execute_fn, **kwargs: {
            "requested": 1,
            "applied": 0,
            "skipped": 1,
            "missing_candidate_ids": [],
            "tool_errors": [],
            "verification": {
                "status": "repair_required",
                "verified": False,
                "failure_code": "verification_failed",
                "reason": "Reconstruction did not restore any selected candidates.",
                "artifact_ids": [],
                "repair_required": True,
            },
        },
    )
    monkeypatch.setattr(
        "remy.core.notification_router.notify",
        lambda message, **kwargs: emitted.append({"message": message, **kwargs}),
    )

    client = _make_client()
    res = client.post("/api/system/memory/reconstruct-missing", json={"candidate_ids": []})

    assert res.status_code == 200
    assert emitted
    assert emitted[0]["event_type"] == "operator_alert"
    assert emitted[0]["event_data"]["source"] == "reconstruct_missing_memory"
    assert emitted[0]["event_data"]["action_target"] == "open_missing_memory_review"
    assert emitted[0]["event_data"]["requested"] == 1
    assert len(stored) == 2
    assert "reconstruction_review" in stored[0]["tags"]
    assert "incident_snapshot" in stored[1]["tags"]
    assert stored[1]["metadata"]["verification_status"] == "repair_required"


def test_run_verify_gate_eval_route(monkeypatch):
    import remy.web.routes.system_routes as system_routes

    fake_api = type("Api", (), {})()
    fake_api._start_time = 0
    fake_api.brain = type("Brain", (), {"count": lambda self: 3, "search": lambda self, **kwargs: []})()
    fake_api.settings = type("Settings", (), {"AURA_BRAIN_PATH": "data/brain", "DATA_DIR": "data"})()

    async def fake_cached_memory_status(api):
        return {
            "verification": {
                "verified_count": 2,
                "repair_required_count": 1,
                "recent": [],
                "last_reconstruction": {},
            }
        }

    monkeypatch.setattr(system_routes, "_get_api", lambda: fake_api)
    monkeypatch.setattr(system_routes, "_get_cached_memory_status", fake_cached_memory_status)
    emitted = []
    monkeypatch.setattr(
        "remy.core.notification_router.notify",
        lambda message, **kwargs: emitted.append({"message": message, **kwargs}),
    )

    client = _make_client()
    res = client.post("/api/system/harness/evals/verify-gate/run")

    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["result"]["id"] == "verify_gate_ablation"
    assert data["result"]["status"] == "completed"
    assert data["result"]["delta"]["prevented_false_successes"] == 1
    assert data["result"]["ablation"]["false_success_rate"] == 0.3333
    assert emitted
    assert emitted[0]["event_type"] == "operator_alert"
    assert emitted[0]["event_data"]["scenario_id"] == "verify_gate_ablation"
    assert emitted[0]["event_data"]["action_target"] == "open_memory_verification"


def test_run_verify_gate_eval_resolves_prior_alert_when_clean(monkeypatch):
    import remy.web.routes.system_routes as system_routes

    fake_api = type("Api", (), {})()
    fake_api._start_time = 0
    fake_api.brain = type("Brain", (), {"count": lambda self: 3, "search": lambda self, **kwargs: []})()
    fake_api.settings = type("Settings", (), {"AURA_BRAIN_PATH": "data/brain", "DATA_DIR": "data"})()

    async def fake_cached_memory_status(api):
        return {
            "verification": {
                "verified_count": 2,
                "repair_required_count": 0,
                "recent": [],
                "last_reconstruction": {},
            }
        }

    emitted = []
    monkeypatch.setattr(system_routes, "_get_api", lambda: fake_api)
    monkeypatch.setattr(system_routes, "_get_cached_memory_status", fake_cached_memory_status)
    monkeypatch.setattr(
        "remy.core.notification_router.notify",
        lambda message, **kwargs: emitted.append({"message": message, **kwargs}),
    )

    client = _make_client()
    res = client.post("/api/system/harness/evals/verify-gate/run")

    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["result"]["status"] == "completed"
    assert emitted
    assert emitted[0]["event_type"] == "harness_eval.resolved"
    assert emitted[0]["event_data"]["resolves"] == ["harness_eval|verify_gate_ablation"]
    assert emitted[0]["event_data"]["action_target"] == "open_memory_verification"


def test_run_recovery_replay_eval_route(monkeypatch):
    import remy.web.routes.system_routes as system_routes

    fake_api = type("Api", (), {})()
    fake_api._start_time = 0
    fake_api.brain = type("Brain", (), {"count": lambda self: 3, "search": lambda self, **kwargs: []})()
    fake_api.settings = type("Settings", (), {"AURA_BRAIN_PATH": "data/brain", "DATA_DIR": "data"})()

    async def fake_cached_memory_status(api):
        return {
            "history_review": {
                "missing_candidates_count": 2,
                "review_candidates_count": 3,
            },
            "verification": {
                "last_reconstruction": {"status": "verified"},
            },
        }

    monkeypatch.setattr(system_routes, "_get_api", lambda: fake_api)
    monkeypatch.setattr(system_routes, "_get_cached_memory_status", fake_cached_memory_status)

    client = _make_client()
    res = client.post("/api/system/harness/evals/recovery-replay/run")

    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["result"]["id"] == "recovery_replay_ablation"
    assert data["result"]["status"] == "completed"
    assert data["result"]["ablation"]["assumed_unresolved_missing"] == 2
    assert data["result"]["delta"]["recovery_visibility_gain"] == 5


def test_run_correction_loop_eval_route(monkeypatch):
    import remy.web.routes.system_routes as system_routes

    fake_api = type("Api", (), {})()
    fake_api._start_time = 0
    fake_api.brain = type("Brain", (), {"count": lambda self: 3, "search": lambda self, **kwargs: []})()
    fake_api.settings = type("Settings", (), {"AURA_BRAIN_PATH": "data/brain", "DATA_DIR": "data"})()

    async def fake_cached_memory_status(api):
        return {
            "corrections": {
                "suggestions_count": 2,
                "review_queue_count": 1,
                "recently_corrected_count": 3,
            }
        }

    monkeypatch.setattr(system_routes, "_get_api", lambda: fake_api)
    monkeypatch.setattr(system_routes, "_get_cached_memory_status", fake_cached_memory_status)

    client = _make_client()
    res = client.post("/api/system/harness/evals/correction-loop/run")

    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["result"]["id"] == "correction_loop_ablation"
    assert data["result"]["status"] == "completed"
    assert data["result"]["ablation"]["assumed_missed_corrections"] == 3
    assert data["result"]["delta"]["repair_events_captured"] == 3


def test_run_decision_dossier_eval_route(monkeypatch):
    import remy.web.routes.system_routes as system_routes

    fake_api = type("Api", (), {})()
    fake_api._start_time = 0
    fake_api.settings = type("Settings", (), {"AURA_BRAIN_PATH": "data/brain", "DATA_DIR": "data"})()

    class Record:
        def __init__(self, tags):
            self.tags = tags

    fake_api.brain = type(
        "Brain",
        (),
        {
            "count": lambda self: 3,
            "search": lambda self, **kwargs: [
                Record(["decision_dossier"]),
                Record(["decision_dossier", "pinned_snapshot"]),
            ],
        },
    )()

    monkeypatch.setattr(system_routes, "_get_api", lambda: fake_api)
    monkeypatch.setattr(
        "remy.core.combined_runner.get_operator_console_snapshot",
        lambda: {"autonomy": {"goals": {"active": 2, "active_list": [{"id": "g1"}, {"id": "g2"}]}}},
    )

    client = _make_client()
    res = client.post("/api/system/harness/evals/decision-dossier/run")

    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["result"]["id"] == "decision_dossier_ablation"
    assert data["result"]["status"] == "completed"
    assert data["result"]["baseline"]["decision_snapshots"] == 2
    assert data["result"]["baseline"]["pinned_snapshots"] == 1
    assert data["result"]["ablation"]["assumed_untraceable_goals"] == 2


def test_build_system_status_payload_includes_snapshot_contract_and_packs(monkeypatch):
    import remy.web.routes.system_routes as system_routes
    from remy.config.settings import settings

    fake_api = type("Api", (), {})()
    fake_api._start_time = 10
    class Brain:
        def count(self):
            return 7

        def salience_summary(self):
            return {"high_salience_count": 2, "avg_salience": 0.42, "max_salience": 0.9}

        def high_salience_records(self, limit=5):
            return [{"id": "rec-1"}, {"id": "rec-2"}]

        def latest_reflection_digest(self):
            return {"summary_count": 3, "high_severity_count": 1, "top_findings": [{"kind": "blocker"}]}

        def contradiction_review_queue(self, limit=5):
            return [{"cluster_id": "c-1"}, {"cluster_id": "c-2"}]

        def contradiction_clusters(self, limit=5):
            return [{"id": "cluster-1"}]

        def belief_instability_summary(self):
            return {"contradiction_cluster_count": 4}

        def get_correction_review_queue(self, limit=5):
            return [{"target_id": "rec-5", "reason_detail": "conflicts with corrected birthday"}]

        def get_suggested_corrections(self, limit=5):
            return [{"target_id": "rec-5", "suggested_action": "review_and_update"}]

        def get_recently_corrected_beliefs(self, limit=5):
            return [{"target_id": "belief-1", "reason_detail": "user correction applied"}]

        def get_suggested_corrections_report(self, limit=5):
            return {"entry_count": 1, "entries": [{"target_id": "rec-5"}]}

        def search(self, **kwargs):
            tags = kwargs.get("tags") or []
            if "generated-report" in tags:
                return [
                    type(
                        "Report",
                        (),
                        {
                            "id": "report-1",
                            "content": "Generated PDF report: VAT",
                            "metadata": {
                                "title": "VAT report",
                                "verification": {"status": "verified", "reason": "PDF verified cleanly."},
                            },
                        },
                    )()
                ]
            if "research" in tags:
                return [
                    type(
                        "Research",
                        (),
                        {
                            "id": "research-1",
                            "content": "AuraSDK research summary",
                            "metadata": {
                                "type": "research_report",
                                "topic": "AuraSDK v6",
                                "verification": {"status": "repair_required", "reason": "Research artifact needs review."},
                            },
                        },
                    )()
                ]
            return [
                type(
                    "Goal",
                    (),
                    {
                        "id": "goal-1",
                        "content": "Investigate primary filings",
                        "metadata": {"status": "active", "priority": "high"},
                    },
                )()
            ]

    fake_api.brain = Brain()
    from contextlib import nullcontext
    fake_api.brain_lock = nullcontext()
    fake_api.settings = type("Settings", (), {"AURA_BRAIN_PATH": "data/brain"})()

    class DummyRegistry:
        def all(self):
            return {"web": {"status": "ok"}, "telegram": {"status": "degraded"}, "autonomy": {"status": "ok"}}

        def summary(self):
            return {"health": "degraded", "running": 2, "degraded": 1}

    class DummyLoop:
        chief = type(
            "Chief",
            (),
            {
                "dashboard_runtime": type(
                    "DashboardRuntime",
                    (),
                    {
                        "improvement_summary": staticmethod(
                            lambda: {
                                "learning": {
                                    "outcomes_observed": 4,
                                    "insights_total": 2,
                                },
                                "reviewable_insights": [
                                    {
                                        "id": "fit_executor_low",
                                        "category": "specialist_fit",
                                        "description": "executor has low success rate",
                                        "confidence": 0.8,
                                        "proposal": "Route fewer cycles through this specialist until the strategy or domain fit improves.",
                                    }
                                ],
                                "top_playbooks": [
                                    {"id": "pb-1", "name": "Execution path", "success_rate": 0.75}
                                ],
                            }
                        )
                    },
                )()
            },
        )()

    async def fake_build_packs():
        return {"packs": [{"id": "general", "enabled": True}]}

    monkeypatch.setattr(system_routes, "_get_api", lambda: fake_api)
    monkeypatch.setattr(
        "remy.core.agent_tools.get_brain_startup_status",
        lambda: {
            "quarantined_at_startup": True,
            "quarantine_reason": "failed to fill whole buffer",
            "startup_blocked": False,
            "startup_incident": "",
            "quarantine_path": "data/brain_incompatible_20260331_223805",
            "backup_path": "data/brain_backup_20260331_223805.json",
            "startup_artifact_id": "startup-artifact-1",
            "recovery": {
                "status": "history_replayed",
                "files": 45,
                "entries": 1177,
                "tool_calls_replayed": 129,
                "tool_calls_skipped": 17,
            },
        },
    )
    monkeypatch.setattr(
        "remy.core.history_replay.analyze_history_memory_gaps",
        lambda *args, **kwargs: {
            "missing_candidates_count": 2,
            "review_candidates_count": 3,
            "missing_by_tool": {"store_person": 1, "store_research": 1},
            "recent_missing": [{"tool": "store_person", "label": "Ганна", "reason": "missing from active memory"}],
            "review_candidates": [{"tool": "store_research", "label": "AuraSDK v6", "summary": "reflection + salience"}],
            "recommended_actions": ["Replay or selectively reconstruct missing memory from history."],
        },
    )
    system_routes._last_reconstruction_verification = {
        "status": "repair_required",
        "reason": "Partial reconstruction needs review.",
    }
    system_routes._last_harness_eval_runs = {
        "verify_gate_ablation": {
            "id": "verify_gate_ablation",
            "status": "completed",
            "summary": "Verify gate prevented 1 false success claim.",
            "ablation": {"false_success_rate": 0.5},
        },
        "recovery_replay_ablation": {
            "id": "recovery_replay_ablation",
            "status": "completed",
            "summary": "Recovery/replay surfaced hidden missing memory.",
            "ablation": {"assumed_unresolved_missing": 2},
        },
        "correction_loop_ablation": {
            "id": "correction_loop_ablation",
            "status": "completed",
            "summary": "Correction loop surfaced recent repair pressure.",
            "ablation": {"assumed_missed_corrections": 3},
        },
        "decision_dossier_ablation": {
            "id": "decision_dossier_ablation",
            "status": "completed",
            "summary": "Decision dossier kept review snapshots available.",
            "ablation": {"assumed_missing_snapshots": 2},
        },
    }
    monkeypatch.setattr(
        "remy.core.harness_eval_history.get_harness_eval_history_summary",
        lambda limit=20: {
            "total_runs": 6,
            "scenario_counts": {
                "verify_gate_ablation": 2,
                "recovery_replay_ablation": 2,
                "correction_loop_ablation": 1,
                "decision_dossier_ablation": 1,
            },
            "latest_entries": [
                {"id": "decision_dossier_ablation", "status": "completed", "summary": "Decision dossier kept snapshots.", "executed_at": 1},
                {"id": "correction_loop_ablation", "status": "completed", "summary": "Correction loop surfaced repair pressure.", "executed_at": 1},
            ],
        },
    )
    monkeypatch.setattr("remy.core.notification_router.get_recent_notifications", lambda **kwargs: [])
    monkeypatch.setattr(
        "remy.core.combined_runner.get_operator_console_snapshot",
        lambda goal_limit=5, approval_limit=10: {
            "channels": {
                "web": {
                    "enabled": True,
                    "url": "http://127.0.0.1:8080",
                    "health": {"status": "ok"},
                },
                "telegram": {
                    "configured": True,
                    "secure": False,
                    "allowed_ids": [],
                    "authorization_hint": "Set TELEGRAM_ALLOWED_CHAT_IDS=<your_chat_id> to move Telegram out of open mode.",
                    "health": {"status": "degraded"},
                },
                "autonomy": {
                    "enabled": True,
                    "version": "v3",
                    "configured_version": "v3",
                    "maintenance_only": True,
                    "cycle_sec": 120,
                    "health": {"status": "ok"},
                },
                "registry_summary": {"health": "degraded", "running": 2, "degraded": 1},
            },
            "gateway": {
                "name": "Remy Gateway",
                "primary_remote_surface": "telegram",
                "status": "degraded",
            },
            "control": {
                "running": True,
                "session_id": "sess-321",
                "active_version": "v3",
                "configured_version": "v3",
                "runtime_loaded": True,
                "maintenance_only": True,
            },
            "autonomy": {
                "running": True,
                "session_id": "sess-321",
                "version": "v3",
                "current_role": "researcher",
                "research_session": {"topic": "AuraSDK v6"},
                "goals": {
                    "total": 1,
                    "active": 1,
                    "blocked": 0,
                    "active_list": [{"id": "goal-1", "content": "Investigate primary filings", "priority": "high"}],
                },
            },
            "approvals": {"pending_count": 1, "pending": [{"id": "appr-1"}]},
            "budget": {
                "balance_usd": 12.5,
                "usdt": 5.0,
                "trx": 20.0,
                "runway_days": 14,
                "alert_level": "yellow",
                "llm_cost_today": 0.82,
                "last_check": "2026-03-19T10:00:00",
            },
            "evaluation": {
                "failure_history_size": 2,
                "specialist_scores": {"researcher": {"unsupported_claims": 2}},
                "routing_pressure": {
                    "top_candidate": {"id": "analyst", "quality_adjusted_success_rate": 0.91},
                    "highest_pressure": {"id": "researcher", "quality_debt": 0.2},
                },
            },
            "factuality": {
                "unsupported_observed_claims_total": 2,
                "quality_debt_by_specialist": [
                    {"id": "researcher", "quality_debt": 0.2, "unsupported_claims": 2}
                ],
                "scheduler_decisions_recent": [
                    {"specialist": "researcher", "reason": "fallback after low evidence"}
                ],
            },
            "improvement": DummyLoop().chief.dashboard_runtime.improvement_summary(),
        },
    )
    monkeypatch.setattr(system_routes, "build_capability_packs_payload", fake_build_packs)
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setattr(settings, "TELEGRAM_ALLOWED_CHAT_IDS", [])
    monkeypatch.setattr(settings, "PRIMARY_REMOTE_SURFACE", "telegram")

    data = asyncio.run(system_routes.build_system_status_payload(include_packs=True))

    assert data["gateway"]["name"] == "Remy Gateway"
    assert data["harness"]["contract"]["version"] == "2026-03-30.v1"
    assert data["harness"]["charter"]["version"] == "2026-03-30.v1"
    assert data["harness"]["modules"]["version"] == "2026-03-30.v1"
    assert data["harness"]["modules"]["count"] >= 5
    assert data["harness"]["modules"]["active_count"] >= 5
    assert data["harness"]["active_module"]["id"] == "research"
    assert data["harness"]["migration"]["version"] == "2026-03-30.v1"
    assert data["harness"]["migration"]["artifact_count"] >= 5
    assert len(data["harness"]["migration"]["runtime_behavior_changes"]) >= 2
    assert len(data["harness"]["charter"]["approval_semantics"]) >= 2
    assert len(data["harness"]["charter"]["shutdown_semantics"]) >= 2
    assert data["harness"]["role_contracts"]["count"] >= 5
    assert data["harness"]["role_contracts"]["current_role"]["id"] == "researcher"
    assert "complete_research" in data["harness"]["role_contracts"]["current_role"]["allowed_tools"]
    assert data["harness"]["script_adapter_registry"]["count"] >= 6
    assert data["harness"]["script_adapter_registry"]["critical_count"] >= 4
    assert data["harness"]["script_adapter_registry"]["categories"]["system_action"] >= 2
    assert data["harness"]["state_semantics"]["count"] >= 5
    assert data["harness"]["state_semantics"]["durable_count"] >= 2
    assert data["harness"]["state_semantics"]["states"][0]["id"] == "active_runtime_state"
    assert data["harness"]["ablation"]["count"] >= 4
    assert data["harness"]["ablation"]["enabled_count"] >= 3
    assert data["harness"]["ablation"]["modules"][0]["id"] == "verify_gate"
    assert data["harness"]["eval_matrix"]["version"] == "2026-03-30.v1"
    assert data["harness"]["eval_matrix"]["count"] >= 4
    assert data["harness"]["eval_matrix"]["planned_count"] >= 4
    assert "false_success_rate" in data["harness"]["eval_matrix"]["tracked_metrics"]
    assert data["harness"]["eval_matrix"]["scenarios"][0]["id"] == "verify_gate_ablation"
    assert data["harness"]["eval_matrix"]["latest_runs"]["verify_gate_ablation"]["status"] == "completed"
    assert data["harness"]["eval_matrix"]["latest_runs"]["recovery_replay_ablation"]["status"] == "completed"
    assert data["harness"]["eval_matrix"]["latest_runs"]["correction_loop_ablation"]["status"] == "completed"
    assert data["harness"]["eval_matrix"]["latest_runs"]["decision_dossier_ablation"]["status"] == "completed"
    assert data["harness"]["eval_matrix"]["history"]["total_runs"] == 6
    assert data["harness"]["eval_matrix"]["history"]["scenario_counts"]["verify_gate_ablation"] == 2
    assert "store_integrity_incident" in data["harness"]["failure_taxonomy"]["codes"]
    assert data["harness"]["incident"]["code"] == "memory_recovery_applied"
    assert data["harness"]["startup"]["quarantined_at_startup"] is True
    assert data["harness"]["startup"]["startup_artifact_id"] == "startup-artifact-1"
    assert data["harness"]["startup"]["quarantine_path"] == "data/brain_incompatible_20260331_223805"
    assert data["harness"]["startup"]["backup_path"] == "data/brain_backup_20260331_223805.json"
    assert data["harness"]["startup"]["recovery"]["tool_calls_replayed"] == 129
    assert data["gateway"]["status"] == "degraded"
    assert data["evaluation"]["failure_history_size"] == 2
    assert data["evaluation"]["routing_pressure"]["top_candidate"]["id"] == "analyst"
    assert data["factuality"]["unsupported_observed_claims_total"] == 2
    assert data["factuality"]["quality_debt_by_specialist"][0]["id"] == "researcher"
    assert data["channels"]["telegram"]["authorization_hint"]
    assert data["autonomy"]["running"] is True
    assert data["autonomy"]["session_id"] == "sess-321"
    assert data["channels"]["autonomy"]["version"] == "v3"
    assert data["channels"]["autonomy"]["configured_version"] == "v3"
    assert data["channels"]["autonomy"]["maintenance_only"] is True
    assert data["autonomy"]["goals"]["active"] == 1
    assert data["memory"]["records"] == 7
    assert data["memory"]["salience"]["high_count"] == 2
    assert data["memory"]["reflection"]["summary_count"] == 3
    assert data["memory"]["contradictions"]["review_queue_count"] == 2
    assert data["memory"]["contradictions"]["cluster_count"] == 4
    assert data["memory"]["corrections"]["suggestions_count"] == 1
    assert data["memory"]["corrections"]["review_queue_count"] == 1
    assert data["memory"]["corrections"]["recently_corrected_count"] == 1
    assert data["memory"]["corrections"]["top_suggestions"][0]["target_id"] == "rec-5"
    assert data["memory"]["history_review"]["missing_candidates_count"] == 2
    assert data["memory"]["history_review"]["review_candidates_count"] == 3
    assert data["memory"]["history_review"]["recent_missing"][0]["label"] == "Ганна"
    assert data["memory"]["verification"]["verified_count"] == 1
    assert data["memory"]["verification"]["repair_required_count"] == 1
    assert data["memory"]["verification"]["recent"][0]["status"] == "verified"
    assert data["memory"]["verification"]["recent"][0]["artifact_ids"] == []
    assert data["memory"]["verification"]["last_reconstruction"]["status"] == "repair_required"
    assert data["budget"]["llm_cost_today"] == 0.82
    assert data["improvement"]["learning"]["insights_total"] == 2
    assert data["improvement"]["reviewable_insights"][0]["id"] == "fit_executor_low"
    assert data["improvement"]["top_playbooks"][0]["id"] == "pb-1"
    assert data["packs"][0]["id"] == "general"


def test_system_status_caches_memory_block(monkeypatch):
    import remy.web.routes.system_routes as system_routes

    system_routes._invalidate_memory_status_cache()
    calls = {"count": 0}

    fake_api = type("Api", (), {})()
    fake_api._start_time = 10
    fake_api.brain = type(
        "Brain",
        (),
        {
            "count": lambda self: calls.__setitem__("count", calls["count"] + 1) or 5,
            "search": lambda self, **kwargs: [],
        },
    )()
    from contextlib import nullcontext
    fake_api.brain_lock = nullcontext()
    fake_api.settings = type("Settings", (), {"AURA_BRAIN_PATH": "data/brain", "DATA_DIR": "data"})()

    monkeypatch.setattr(system_routes, "_get_api", lambda: fake_api)
    monkeypatch.setattr("remy.core.combined_runner.get_operator_console_snapshot", lambda: {})
    monkeypatch.setattr("remy.core.notification_router.get_recent_notifications", lambda **kwargs: [])
    monkeypatch.setattr("remy.core.agent_tools.get_brain_startup_status", lambda: {})
    monkeypatch.setattr(
        "remy.core.history_replay.analyze_history_memory_gaps",
        lambda *args, **kwargs: {"missing_candidates_count": 0, "review_candidates_count": 0},
    )

    first = asyncio.run(system_routes.build_system_status_payload())
    second = asyncio.run(system_routes.build_system_status_payload())

    assert first["memory"]["records"] == 5
    assert second["memory"]["records"] == 5
    assert calls["count"] == 1


def test_startup_recovery_status_route(monkeypatch):
    client = _make_client()
    monkeypatch.setattr(
        "remy.core.agent_tools.get_brain_startup_status",
        lambda: {
            "quarantined_at_startup": True,
            "backup_path": "data/brain_backup_20260331_223805.json",
        },
    )
    monkeypatch.setattr(
        "remy.web.routes.system_routes._build_startup_recovery_status",
        lambda startup_status: {
            "available": True,
            "backup_path": startup_status["backup_path"],
            "preview": {"missing_records": 483, "backup_records": 536},
            "last_apply": {},
            "last_reconcile": {},
        },
    )
    res = client.get("/api/system/startup-recovery/status")
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["recovery"]["preview"]["missing_records"] == 483


def test_startup_recovery_apply_route(monkeypatch):
    import remy.web.routes.system_routes as system_routes

    stored = []
    emitted = []

    class FakeBrain:
        def search(self, **kwargs):
            return []

        def store(self, content, level=None, tags=None, metadata=None):
            stored.append({"content": content, "tags": list(tags or []), "metadata": metadata or {}})
            return type("Record", (), {"id": f"rec-{len(stored)}"})()

    fake_api = type("Api", (), {})()
    fake_api.brain = FakeBrain()
    fake_api.brain_lock = nullcontext()

    monkeypatch.setattr(system_routes, "_get_api", lambda: fake_api)
    monkeypatch.setattr(
        "remy.core.agent_tools.get_brain_startup_status",
        lambda: {"backup_path": "data/brain_backup_20260331_223805.json"},
    )
    monkeypatch.setattr(
        "remy.core.startup_recovery.apply_backup_recovery",
        lambda path: {
            "backup_path": str(path),
            "imported_count": 483,
            "imported_by_level": {"IDENTITY": 13, "DECISIONS": 58, "DOMAIN": 412},
            "recovery_artifact_id": "artifact-1",
        },
    )
    monkeypatch.setattr(
        "remy.core.notification_router.notify",
        lambda message, **kwargs: emitted.append({"message": message, **kwargs}),
    )

    res = _make_client().post("/api/system/startup-recovery/recover", json={})
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["result"]["imported_count"] == 483
    assert data["result"]["operator_artifact_id"] == "rec-1"
    assert stored
    assert "startup_backup_recovery" in stored[0]["tags"]
    assert emitted
    assert emitted[0]["event_type"] == "operator_alert"
    assert emitted[0]["event_data"]["verification_status"] == "backup_recovery_applied"


def test_startup_recovery_reconcile_route(monkeypatch):
    import remy.web.routes.system_routes as system_routes

    stored = []
    emitted = []

    class FakeBrain:
        def search(self, **kwargs):
            return []

        def store(self, content, level=None, tags=None, metadata=None):
            stored.append({"content": content, "tags": list(tags or []), "metadata": metadata or {}})
            return type("Record", (), {"id": f"rec-{len(stored)}"})()

    fake_api = type("Api", (), {})()
    fake_api.brain = FakeBrain()
    fake_api.brain_lock = nullcontext()

    monkeypatch.setattr(system_routes, "_get_api", lambda: fake_api)
    monkeypatch.setattr(
        "remy.core.startup_recovery.reconcile_recovered_records",
        lambda apply=True: {
            "recovered_records": 483,
            "changes_needed": 36,
            "applied": True,
            "changes": [],
        },
    )
    monkeypatch.setattr(
        "remy.core.notification_router.notify",
        lambda message, **kwargs: emitted.append({"message": message, **kwargs}),
    )

    res = _make_client().post("/api/system/startup-recovery/reconcile")
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["result"]["changes_needed"] == 36
    assert data["result"]["operator_artifact_id"] == "rec-1"
    assert stored
    assert "startup_backup_reconciliation" in stored[0]["tags"]
    assert emitted
    assert emitted[0]["event_data"]["verification_status"] == "backup_reconciliation_applied"


def test_startup_recovery_cleanup_route(monkeypatch):
    import remy.web.routes.system_routes as system_routes

    stored = []
    emitted = []

    class FakeBrain:
        def search(self, **kwargs):
            return []

        def store(self, content, level=None, tags=None, metadata=None):
            stored.append({"content": content, "tags": list(tags or []), "metadata": metadata or {}})
            return type("Record", (), {"id": f"rec-{len(stored)}"})()

    fake_api = type("Api", (), {})()
    fake_api.brain = FakeBrain()
    fake_api.brain_lock = nullcontext()

    monkeypatch.setattr(system_routes, "_get_api", lambda: fake_api)
    monkeypatch.setattr(
        "remy.core.startup_recovery.cleanup_recovered_records",
        lambda apply=True: {
            "recovered_records": 483,
            "cleanup_candidates": 483,
            "applied": True,
            "changes": [],
        },
    )
    monkeypatch.setattr(
        "remy.core.notification_router.notify",
        lambda message, **kwargs: emitted.append({"message": message, **kwargs}),
    )

    res = _make_client().post("/api/system/startup-recovery/cleanup")
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["result"]["cleanup_candidates"] == 483
    assert data["result"]["operator_artifact_id"] == "rec-1"
    assert stored
    assert "startup_backup_recovery_cleanup" in stored[0]["tags"]
    assert emitted
    assert emitted[0]["event_data"]["verification_status"] == "backup_recovery_cleanup_applied"
