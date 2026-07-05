"""Tests for Combined Runner — parallel channel coordination."""

import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock

import pytest


# ============== BANNER TESTS ==============


class TestCombinedBanner:

    def _print_banner(self, **kwargs):
        """Call _print_combined_banner with mocked brain/registry."""
        with patch("remy.core.combined_runner.brain") as mock_brain, \
             patch("remy.core.combined_runner.get_registry") as mock_reg, \
             patch("remy.core.combined_runner.settings") as mock_s:
            mock_brain.count.return_value = 42
            mock_reg.return_value.get_all_declarations.return_value = [1, 2, 3]
            mock_s.SUMMARY_MODEL = "test-model"
            mock_s.AURA_BRAIN_PATH = "/tmp/brain"
            mock_s.WEB_HOST = "127.0.0.1"
            mock_s.WEB_PORT = 8080
            mock_s.AUTONOMY_CYCLE_INTERVAL_SEC = 120
            mock_s.AUTONOMY_DAILY_TOKEN_LIMIT = 100_000

            from remy.core.combined_runner import _print_combined_banner
            _print_combined_banner(**kwargs)

    def test_banner_all_channels(self, capsys):
        self._print_banner(autonomous=True, telegram=True, web=True)
        output = capsys.readouterr().out
        assert "Remy" in output
        assert "Autonomy" in output
        assert "Telegram" in output
        assert "Web" in output
        assert "Memory ready" in output
        assert "readyry" not in output
        assert "Tool registry deferred" in output

    def test_console_line_overwrite_clears_longer_progress_text(self, capsys):
        from remy.core.combined_runner import _overwrite_console_line

        _overwrite_console_line("short", "longer text")

        output = capsys.readouterr().out
        assert output == "\rshort      \n"

    def test_banner_two_channels(self, capsys):
        self._print_banner(autonomous=True, telegram=True, web=False)
        output = capsys.readouterr().out
        assert "Autonomy" in output
        assert "Telegram" in output
        assert "disabled" in output

    def test_banner_autonomous_web(self, capsys):
        self._print_banner(autonomous=True, telegram=False, web=True)
        output = capsys.readouterr().out
        assert "Autonomy" in output
        assert "Web server" in output
        assert "cycle" in output

    def test_banner_shows_model(self, capsys):
        self._print_banner(autonomous=True, telegram=True, web=False)
        output = capsys.readouterr().out
        assert "test-model" in output

    def test_banner_eager_mode_shows_counts(self, capsys):
        with patch.dict("os.environ", {"REMY_EAGER_STARTUP_BANNER": "1"}):
            self._print_banner(autonomous=True, telegram=True, web=True)
        output = capsys.readouterr().out
        assert "42 records" in output
        assert "3 tools" in output


# ============== TELEGRAM ASYNC TESTS ==============


class TestTelegramAsync:

    @pytest.mark.asyncio
    async def test_stop_telegram_handles_errors(self):
        """_stop_telegram_async should not raise on errors."""
        mock_app = AsyncMock()
        mock_app.updater.stop.side_effect = RuntimeError("already stopped")

        from remy.core.combined_runner import _stop_telegram_async
        # Should not raise
        await _stop_telegram_async(mock_app)

    @pytest.mark.asyncio
    async def test_stop_telegram_calls_sequence(self):
        """Stop should call updater.stop, app.stop, app.shutdown in order."""
        mock_app = AsyncMock()
        call_order = []
        mock_app.updater.stop = AsyncMock(side_effect=lambda: call_order.append("updater"))
        mock_app.stop = AsyncMock(side_effect=lambda: call_order.append("stop"))
        mock_app.shutdown = AsyncMock(side_effect=lambda: call_order.append("shutdown"))

        from remy.core.combined_runner import _stop_telegram_async
        await _stop_telegram_async(mock_app)

        assert call_order == ["updater", "stop", "shutdown"]


# ============== UVICORN SETUP TESTS ==============


class TestUvicornSetup:

    def test_create_uvicorn_server(self):
        """_create_uvicorn_server returns a uvicorn.Server."""
        with patch("remy.core.combined_runner.settings") as mock_s, \
             patch("remy.core.desktop_gui.create_app") as mock_create, \
             patch("remy.web.api.set_session_manager"), \
             patch("remy.web.session.WebSessionManager"):
            mock_s.WEB_HOST = "127.0.0.1"
            mock_s.WEB_PORT = 8080
            mock_create.return_value = MagicMock()

            from remy.core.combined_runner import _create_uvicorn_server
            server = _create_uvicorn_server()

            import uvicorn
            assert isinstance(server, uvicorn.Server)
            assert server.config.install_signal_handlers is False


# ============== RUN COMBINED TESTS ==============


class TestRunCombined:

    @pytest.mark.asyncio
    async def test_no_channels_returns(self):
        """run_combined with no channels should return without error."""
        with patch("remy.core.combined_runner.brain") as mock_brain, \
             patch("remy.core.combined_runner.get_registry") as mock_reg, \
             patch("remy.core.combined_runner.settings") as mock_s:
            mock_brain.count.return_value = 0
            mock_reg.return_value.get_all_declarations.return_value = []
            mock_s.SUMMARY_MODEL = "test"
            mock_s.AURA_BRAIN_PATH = "/tmp"
            mock_s.WEB_HOST = "127.0.0.1"
            mock_s.WEB_PORT = 8080
            mock_s.AUTONOMY_CYCLE_INTERVAL_SEC = 1
            mock_s.AUTONOMY_DAILY_TOKEN_LIMIT = 1000

            from remy.core.combined_runner import run_combined
            await run_combined(autonomous=False, telegram=False, web=False)


class TestRuntimeAccessors:

    @pytest.mark.asyncio
    async def test_run_autonomy_standalone_delegates_to_run_combined(self):
        import remy.core.combined_runner as combined_runner

        with patch.object(combined_runner, "run_combined", new=AsyncMock()) as mock_run_combined:
            await combined_runner.run_autonomy_standalone(version_override="v3")

        mock_run_combined.assert_awaited_once_with(
            autonomous=True,
            telegram=False,
            web=False,
            autonomy_version_override="v3",
        )

    def test_configured_autonomy_version_prefers_runtime_override(self):
        import remy.core.combined_runner as combined_runner

        original_override = combined_runner._autonomy_version_override
        original_autonomy_v3 = combined_runner.settings.AUTONOMY_V3
        try:
            combined_runner.settings.AUTONOMY_V3 = False
            combined_runner._autonomy_version_override = "v3"

            assert combined_runner._configured_autonomy_version() == "v3"
        finally:
            combined_runner._autonomy_version_override = original_override
            combined_runner.settings.AUTONOMY_V3 = original_autonomy_v3

    def test_launch_autonomy_task_uses_version_override(self):
        import remy.core.combined_runner as combined_runner

        original_runtime = combined_runner._autonomy_runtime
        original_loop = combined_runner._auto_loop
        original_task = combined_runner._auto_task
        original_autonomy_v3 = combined_runner.settings.AUTONOMY_V3
        fake_task = MagicMock()
        fake_loop = type(
            "Loop",
            (),
            {
                "start": lambda self: None,
                "session_id": "sess-v3",
            },
        )()
        try:
            combined_runner.settings.AUTONOMY_V3 = False
            with patch.object(combined_runner, "_instantiate_autonomy_runtime", return_value=({"loop": fake_loop, "chief": object()}, fake_loop, "v3")) as mock_instantiate, \
                 patch("asyncio.create_task", return_value=fake_task):
                runtime, loop, task, version = combined_runner._launch_autonomy_task(
                    "autonomous-test",
                    version_override="v3",
                )

            mock_instantiate.assert_called_once_with(version_override="v3")
            assert runtime["loop"] is fake_loop
            assert loop is fake_loop
            assert task is fake_task
            assert version == "v3"
            assert combined_runner._autonomy_runtime == runtime
            assert combined_runner._auto_loop is fake_loop
            assert combined_runner._auto_task is fake_task
        finally:
            combined_runner._autonomy_runtime = original_runtime
            combined_runner._auto_loop = original_loop
            combined_runner._auto_task = original_task
            combined_runner.settings.AUTONOMY_V3 = original_autonomy_v3

    def test_get_autonomy_control_state_exposes_maintenance_flag(self):
        import remy.core.combined_runner as combined_runner

        original_runtime = combined_runner._autonomy_runtime
        original_loop = combined_runner._auto_loop
        original_autonomy_v3 = combined_runner.settings.AUTONOMY_V3
        try:
            combined_runner.settings.AUTONOMY_V3 = True
            combined_runner._autonomy_runtime = {"chief": object()}
            combined_runner._auto_loop = type(
                "Loop",
                (),
                {
                    "running": False,
                    "session_id": "sess-maint",
                    "status": lambda self: {
                        "version": "v3",
                        "maintenance_only": True,
                    },
                },
            )()

            state = combined_runner.get_autonomy_control_state()

            assert state["configured_version"] == "v3"
            assert state["active_version"] == "v3"
            assert state["runtime_loaded"] is True
            assert state["maintenance_only"] is True
        finally:
            combined_runner.settings.AUTONOMY_V3 = original_autonomy_v3
            combined_runner._autonomy_runtime = original_runtime
            combined_runner._auto_loop = original_loop

    def test_get_autonomy_runtime_component_prefers_runtime_dict(self):
        import remy.core.combined_runner as combined_runner

        original_runtime = combined_runner._autonomy_runtime
        original_loop = combined_runner._auto_loop
        try:
            combined_runner._autonomy_runtime = {"dashboard_runtime": "dash-v3"}
            combined_runner._auto_loop = MagicMock()
            assert combined_runner.get_autonomy_runtime_component("dashboard_runtime") == "dash-v3"
        finally:
            combined_runner._autonomy_runtime = original_runtime
            combined_runner._auto_loop = original_loop

    def test_get_autonomy_runtime_component_falls_back_to_loop_chief(self):
        import remy.core.combined_runner as combined_runner

        original_runtime = combined_runner._autonomy_runtime
        original_loop = combined_runner._auto_loop
        try:
            chief = type("Chief", (), {"dashboard_runtime": "dash-from-chief"})()
            loop = type("Loop", (), {"chief": chief})()
            combined_runner._autonomy_runtime = None
            combined_runner._auto_loop = loop
            assert combined_runner.get_autonomy_runtime_component("dashboard_runtime") == "dash-from-chief"
        finally:
            combined_runner._autonomy_runtime = original_runtime
            combined_runner._auto_loop = original_loop

    def test_get_operator_runtime_snapshot_returns_shared_operator_state(self, tmp_path):
        import remy.core.combined_runner as combined_runner

        original_runtime = combined_runner._autonomy_runtime
        original_loop = combined_runner._auto_loop
        fake_goal = type(
            "Goal",
            (),
            {
                "id": "goal-1",
                "content": "Investigate primary filings",
                "metadata": {"status": "active", "priority": "high"},
            },
        )()
        fake_state = type(
            "State",
            (),
            {
                "last_total_usd": 12.5,
                "last_usdt": 5.0,
                "last_trx": 20.0,
                "last_runway_days": 14,
                "last_status": "yellow",
                "llm_cost_today": 0.82,
                "last_balance_check": "2026-03-19T10:00:00",
            },
        )()
        fake_evaluator = type(
            "Evaluator",
            (),
            {
                "summary": lambda self: {
                    "failure_history_size": 2,
                    "specialist_scores": {
                        "researcher": {
                            "success_rate": 0.7,
                            "quality_adjusted_success_rate": 0.5,
                            "unsupported_claims": 2,
                            "factuality_penalty": 0.2,
                        }
                    },
                }
            },
        )()
        fake_ops_query_runtime = type(
            "OpsQueryRuntime",
            (),
            {
                "pending_approval_items": lambda self, limit=5: [
                    {
                        "id": "approval-v3-1",
                        "action_id": "approval-v3-1",
                        "action": "routing_pressure:researcher:Research counterparty profile",
                        "description": "Routing pressure approval: specialist 'researcher' is degraded (quality=0.49, unsupported_claims=3) for medium-risk work",
                        "specialist": "researcher",
                        "risk_category": "medium",
                        "created_at": time.time(),
                        "context": {"quality_debt": 0.23, "target": "Research counterparty profile"},
                        "source": "v3_governance",
                    }
                ],
                "recent_approvals": lambda self, limit=5: [
                    {
                        "id": "approval-v3-1",
                        "action": "routing_pressure:researcher:Research counterparty profile",
                        "description": "Routing pressure approval: specialist 'researcher' is degraded (quality=0.49, unsupported_claims=3) for medium-risk work",
                        "status": "approved",
                        "decided_by": "telegram",
                        "wait_sec": 9.0,
                        "specialist": "researcher",
                        "context": {"quality_debt": 0.23, "target": "Research counterparty profile"},
                        "routing_pressure": True,
                    }
                ],
                "factuality_summary": lambda self: {
                    "unsupported_observed_claims_total": 2,
                    "top_offenders": [{"id": "researcher", "unsupported_claims": 2}],
                },
                "quality_debt_by_specialist": lambda self: [
                    {"id": "researcher", "quality_debt": 0.2, "unsupported_claims": 2}
                ],
                "scheduler_decisions_recent": lambda self, limit=5: [
                    {"specialist": "researcher", "reason": "fallback after low evidence"}
                ],
                "routing_pressure_summary": lambda self: {
                    "preferred": [{"id": "analyst", "quality_adjusted_success_rate": 0.9, "quality_debt": 0.0, "unsupported_claims": 0}],
                    "degraded": [{"id": "researcher", "quality_debt": 0.2, "unsupported_claims": 2, "quality_adjusted_success_rate": 0.5}],
                    "top_candidate": {"id": "analyst", "quality_adjusted_success_rate": 0.9},
                    "highest_pressure": {"id": "researcher", "quality_debt": 0.2},
                },
            },
        )()
        try:
            combined_runner._autonomy_runtime = {
                "chief": object(),
                "evaluator": fake_evaluator,
                "ops_query_runtime": fake_ops_query_runtime,
            }
            combined_runner._auto_loop = type(
                "Loop",
                (),
                {
                    "running": True,
                    "session_id": "sess-321",
                    "status": lambda self: {"current_goal": "Investigate primary filings"},
                },
            )()
            budget_path = tmp_path / "autonomy_budget.json"
            budget_path.write_text(
                '{"cost_today_usd": 0.82, "total_cost_lifetime_usd": 12.34, "tokens_today": 321, "tokens_this_hour": 45, "total_tokens_lifetime": 6543, "daily_limit": 100000, "hourly_limit": 20000}',
                encoding="utf-8",
            )
            with patch("remy.core.approval_queue.approval_queue") as mock_queue, \
                 patch("remy.core.combined_runner.brain") as mock_brain, \
                 patch("remy.core.survival.load_state", return_value=fake_state), \
                 patch.object(combined_runner.settings, "DATA_DIR", tmp_path):
                mock_queue.snapshot_pending.return_value = [
                    {
                        "id": "appr-1",
                        "action_id": "appr-1",
                        "description": "Review payout request",
                        "timeout_sec": 60,
                        "created_at": time.time() - 12,
                        "expires_at": time.time() + 48,
                        "age_sec": 12,
                    }
                ]
                mock_brain.search.return_value = [fake_goal]

                snapshot = combined_runner.get_operator_runtime_snapshot(goal_limit=3, approval_limit=5)

            assert snapshot["autonomy"]["running"] is True
            assert snapshot["autonomy"]["version"] == "v3"
            assert snapshot["approvals"]["pending_count"] == 2
            approval_ids = {item["action_id"] for item in snapshot["approvals"]["pending"]}
            assert "appr-1" in approval_ids
            assert "approval-v3-1" in approval_ids
            v3_item = next(item for item in snapshot["approvals"]["pending"] if item["action_id"] == "approval-v3-1")
            legacy_item = next(item for item in snapshot["approvals"]["pending"] if item["action_id"] == "appr-1")
            assert legacy_item["timeout_sec"] is not None
            assert legacy_item["created_at"] is not None
            assert legacy_item["expires_at"] is not None
            assert v3_item["routing_pressure"] is True
            assert v3_item["specialist"] == "researcher"
            assert v3_item["context"]["target"] == "Research counterparty profile"
            assert snapshot["approvals"]["recent"][0]["routing_pressure"] is True
            assert snapshot["approvals"]["recent"][0]["decided_by"] == "telegram"
            assert snapshot["goals"]["active"] == 1
            assert snapshot["goals"]["active_list"][0]["id"] == "goal-1"
            assert snapshot["budget"]["llm_cost_today"] == 0.82
            assert snapshot["budget"]["cost_today_usd"] == 0.82
            assert snapshot["budget"]["llm_cost_lifetime_usd"] is not None
            assert snapshot["budget"]["daily_limit"] == 100000
            assert snapshot["budget"]["hourly_limit"] == 20000
            assert snapshot["budget"]["tokens_today"] == 321
            assert snapshot["budget"]["tokens_this_hour"] == 45
            assert snapshot["budget"]["llm_tokens_today"] is not None
            assert snapshot["budget"]["llm_tokens_this_hour"] == 45
            assert snapshot["budget"]["llm_tokens_lifetime"] is not None
            assert snapshot["evaluation"]["specialist_scores"]["researcher"]["unsupported_claims"] == 2
            assert snapshot["factuality"]["unsupported_observed_claims_total"] == 2
            assert snapshot["quality_debt_by_specialist"][0]["id"] == "researcher"
            assert snapshot["scheduler_decisions_recent"][0]["specialist"] == "researcher"
            assert snapshot["routing_pressure"]["top_candidate"]["id"] == "analyst"
            assert snapshot["routing_pressure"]["highest_pressure"]["id"] == "researcher"
        finally:
            combined_runner._autonomy_runtime = original_runtime
            combined_runner._auto_loop = original_loop

    def test_get_autonomy_operator_snapshot_merges_runtime_and_operator_state(self):
        import remy.core.combined_runner as combined_runner

        with patch.object(combined_runner, "get_autonomy_status_snapshot", return_value={
            "running": True,
            "session_id": "sess-123",
            "version": "v3",
            "current_goal": {"id": "goal-1"},
        }), patch.object(combined_runner, "get_operator_runtime_snapshot", return_value={
            "approvals": {
                "pending_count": 2,
                "pending": [{"id": "appr-1"}],
            },
            "budget": {"llm_cost_today": 0.82},
            "evaluation": {"failure_history_size": 2},
            "factuality": {"unsupported_observed_claims_total": 3},
            "quality_debt_by_specialist": [{"id": "researcher"}],
            "scheduler_decisions_recent": [{"specialist": "researcher"}],
            "routing_pressure": {"top_candidate": {"id": "analyst"}},
        }), patch(
            "remy.core.research_sessions.get_research_session_trace",
            return_value={
                "session_id": "rs-1",
                "topic": "VAT reporting",
                "recent_queries": ["vat deadlines 2026"],
                "knowledge_gaps": ["Improve citation coverage for current findings."],
            },
        ):
            snapshot = combined_runner.get_autonomy_operator_snapshot(goal_limit=3, approval_limit=10)

        assert snapshot["running"] is True
        assert snapshot["session_id"] == "sess-123"
        assert snapshot["version"] == "v3"
        assert snapshot["current_goal"] == {"id": "goal-1"}
        assert snapshot["pending_approvals"] == 2
        assert snapshot["approval_queue"] == [{"id": "appr-1"}]
        assert snapshot["budget"] == {"llm_cost_today": 0.82}
        assert snapshot["evaluation"]["failure_history_size"] == 2
        assert snapshot["factuality"]["unsupported_observed_claims_total"] == 3
        assert snapshot["quality_debt_by_specialist"][0]["id"] == "researcher"
        assert snapshot["scheduler_decisions_recent"][0]["specialist"] == "researcher"
        assert snapshot["routing_pressure"]["top_candidate"]["id"] == "analyst"
        assert snapshot["research_session"]["session_id"] == "rs-1"
        assert snapshot["research_session"]["topic"] == "VAT reporting"

    def test_get_activity_runtime_snapshot_adds_transport_state(self):
        import remy.core.combined_runner as combined_runner

        with patch.object(combined_runner, "get_autonomy_operator_snapshot", return_value={
            "running": True,
            "session_id": "sess-123",
            "version": "v3",
        }):
            snapshot = combined_runner.get_activity_runtime_snapshot(
                goal_limit=3,
                approval_limit=10,
                transport_connected=True,
            )

        assert snapshot["running"] is True
        assert snapshot["session_id"] == "sess-123"
        assert snapshot["version"] == "v3"
        assert snapshot["transport_connected"] is True

    def test_get_runtime_transport_snapshot_returns_connected_and_count(self):
        import remy.core.combined_runner as combined_runner

        class _Bus:
            subscriber_count = 3

        with patch("remy.core.event_bus.event_bus", _Bus()):
            snapshot = combined_runner.get_runtime_transport_snapshot()

        assert snapshot == {"subscribers": 3, "connected": True}

    def test_get_system_runtime_snapshot_merges_control_operator_and_improvement(self):
        import remy.core.combined_runner as combined_runner

        dashboard_runtime = type(
            "DashboardRuntime",
            (),
            {
                "improvement_summary": lambda self: {
                    "learning": {"insights_total": 2},
                    "reviewable_insights": [{"id": "insight-1"}],
                    "top_playbooks": [{"id": "pb-1"}],
                }
            },
        )()

        with patch.object(combined_runner, "get_autonomy_control_state", return_value={
            "running": True,
            "session_id": "sess-123",
            "active_version": "v3",
            "configured_version": "v3",
            "maintenance_only": True,
        }), patch.object(combined_runner, "get_operator_runtime_snapshot", return_value={
            "autonomy": {"running": True, "session_id": "sess-123", "version": "v3"},
            "goals": {"active": 1},
            "approvals": {"pending_count": 2, "pending": [{"id": "appr-1"}]},
            "budget": {"llm_cost_today": 0.82},
            "evaluation": {"failure_history_size": 2},
            "factuality": {"unsupported_observed_claims_total": 3},
            "quality_debt_by_specialist": [{"id": "researcher"}],
            "scheduler_decisions_recent": [{"specialist": "researcher"}],
            "routing_pressure": {"top_candidate": {"id": "analyst"}},
        }), patch.object(combined_runner, "get_autonomy_runtime_component", return_value=dashboard_runtime):
            snapshot = combined_runner.get_system_runtime_snapshot(goal_limit=5, approval_limit=10)

        assert snapshot["control"]["active_version"] == "v3"
        assert snapshot["autonomy"]["running"] is True
        assert snapshot["autonomy"]["goals"]["active"] == 1
        assert snapshot["approvals"]["pending_count"] == 2
        assert snapshot["budget"]["llm_cost_today"] == 0.82
        assert snapshot["evaluation"]["failure_history_size"] == 2
        assert snapshot["evaluation"]["routing_pressure"]["top_candidate"]["id"] == "analyst"
        assert snapshot["factuality"]["unsupported_observed_claims_total"] == 3
        assert snapshot["factuality"]["quality_debt_by_specialist"][0]["id"] == "researcher"
        assert snapshot["improvement"]["learning"]["insights_total"] == 2

    def test_get_channel_status_snapshot_merges_gateway_and_control_state(self):
        import remy.core.combined_runner as combined_runner

        registry = type(
            "Registry",
            (),
            {
                "all": lambda self: {
                    "web": {"status": "ok"},
                    "telegram": {"status": "degraded"},
                    "autonomy": {"status": "ok"},
                },
                "summary": lambda self: {"health": "degraded", "running": 2},
            },
        )()

        with patch("remy.core.gateway.get_registry", return_value=registry), \
             patch("remy.core.notification_router.is_web_runtime_enabled", return_value=True), \
             patch.object(combined_runner, "get_autonomy_control_state", return_value={
                 "running": True,
                 "session_id": "sess-123",
                 "active_version": "v3",
                 "configured_version": "v3",
                 "maintenance_only": True,
             }):
            snapshot = combined_runner.get_channel_status_snapshot()

        assert snapshot["channels"]["registry_summary"]["health"] == "degraded"
        assert snapshot["channels"]["web"]["enabled"] is True
        assert snapshot["channels"]["autonomy"]["version"] == "v3"
        assert snapshot["channels"]["autonomy"]["maintenance_only"] is True
        assert snapshot["gateway"]["status"] == "degraded"
        assert snapshot["control"]["session_id"] == "sess-123"

    def test_get_channel_status_snapshot_does_not_mark_inactive_telegram_as_open_mode(self):
        import remy.core.combined_runner as combined_runner

        registry = type(
            "Registry",
            (),
            {
                "all": lambda self: {
                    "web": {"status": "running"},
                },
                "summary": lambda self: {"health": "ok", "running": 1},
            },
        )()

        with patch("remy.core.gateway.get_registry", return_value=registry), \
             patch("remy.core.notification_router.is_web_runtime_enabled", return_value=True), \
             patch.object(combined_runner.settings, "TELEGRAM_BOT_TOKEN", "test-token"), \
             patch.object(combined_runner.settings, "TELEGRAM_ALLOWED_CHAT_IDS", []), \
             patch.object(combined_runner.settings, "PRIMARY_REMOTE_SURFACE", "telegram"), \
             patch.object(combined_runner, "get_autonomy_control_state", return_value={
                 "running": False,
                 "session_id": None,
                 "active_version": "v3",
                 "configured_version": "v3",
                 "maintenance_only": False,
             }):
            snapshot = combined_runner.get_channel_status_snapshot()

        assert snapshot["channels"]["telegram"]["configured"] is True
        assert snapshot["channels"]["telegram"]["enabled"] is False
        assert snapshot["channels"]["telegram"]["authorization_hint"] == ""
        assert snapshot["gateway"]["primary_remote_surface"] == "web"

    def test_get_operator_console_snapshot_merges_channel_and_runtime_views(self):
        import remy.core.combined_runner as combined_runner

        with patch.object(combined_runner, "get_channel_status_snapshot", return_value={
            "channels": {"web": {"enabled": True}},
            "gateway": {"status": "ok"},
            "control": {"session_id": "sess-console", "active_version": "v3"},
        }), patch.object(combined_runner, "get_system_runtime_snapshot", return_value={
            "control": {"session_id": "sess-console", "active_version": "v3"},
            "autonomy": {"running": True, "version": "v3"},
            "approvals": {"pending_count": 1, "pending": [{"id": "appr-1"}]},
            "budget": {"llm_cost_today": 0.5},
            "evaluation": {"failure_history_size": 1},
            "factuality": {"unsupported_observed_claims_total": 0},
            "improvement": {"learning": {"insights_total": 2}},
        }):
            snapshot = combined_runner.get_operator_console_snapshot(goal_limit=5, approval_limit=10)

        assert snapshot["channels"]["web"]["enabled"] is True
        assert snapshot["gateway"]["status"] == "ok"
        assert snapshot["control"]["session_id"] == "sess-console"
        assert snapshot["autonomy"]["running"] is True
        assert snapshot["approvals"]["pending_count"] == 1
        assert snapshot["budget"]["llm_cost_today"] == 0.5
        assert snapshot["evaluation"]["failure_history_size"] == 1
        assert snapshot["improvement"]["learning"]["insights_total"] == 2

    def test_get_activity_feed_snapshot_builds_shared_activity_payload(self):
        import remy.core.combined_runner as combined_runner
        from contextlib import nullcontext

        goals = [type("Goal", (), {"id": "goal-1", "content": "Goal", "metadata": {"status": "active"}})()]
        outcomes = [type("Outcome", (), {"id": "out-1", "content": "Outcome", "metadata": {"success": True, "tokens_used": 5}})()]
        reflections = [type("Reflection", (), {"id": "ref-1", "content": "Reflection", "metadata": {"session_id": "sess-1"}})()]
        proactive = [type("Proactive", (), {"id": "pro-1", "content": "Ping", "metadata": {"trigger_reason": "idle"}})()]
        brain = type(
            "Brain",
            (),
            {
                "search": lambda self, query, tags, limit=50: (
                    goals if "autonomous-goal" in tags else
                    outcomes if "autonomous-outcome" in tags else
                    reflections if "session-reflection" in tags else
                    proactive if "proactive-session" in tags else
                    []
                )
            },
        )()

        payload = combined_runner.get_activity_feed_snapshot(brain, nullcontext())

        assert payload["summary"]["total_actions"] == 1
        assert payload["summary"]["success"] == 1
        assert payload["summary"]["active_goals"] == 1
        assert payload["goals"][0]["id"] == "goal-1"
        assert payload["outcomes"][0]["id"] == "out-1"

    def test_get_budget_runtime_snapshot_normalizes_budget_shape(self):
        import remy.core.combined_runner as combined_runner

        with patch.object(combined_runner, "get_operator_runtime_snapshot", return_value={
            "budget": {
                "llm_cost_today": 0.82,
                "llm_cost_lifetime_usd": 12.34,
            }
        }), patch.object(combined_runner.settings, "AUTONOMY_DAILY_COST_LIMIT_USD", 3.5):
            snapshot = combined_runner.get_budget_runtime_snapshot(goal_limit=5, approval_limit=10)

        assert snapshot["daily_cost_limit_usd"] == 3.5
        assert snapshot["llm_cost_today"] == 0.82
        assert snapshot["cost_today_usd"] == 0.82
        assert snapshot["llm_cost_lifetime_usd"] == 12.34

    def test_get_goal_runtime_snapshot_returns_goal_slice(self):
        import remy.core.combined_runner as combined_runner

        with patch.object(combined_runner, "get_operator_runtime_snapshot", return_value={
            "goals": {
                "total": 4,
                "active": 2,
                "blocked": 1,
                "active_list": [{"id": "goal-1"}],
            }
        }):
            snapshot = combined_runner.get_goal_runtime_snapshot(goal_limit=5, approval_limit=10)

        assert snapshot["total"] == 4
        assert snapshot["active"] == 2
        assert snapshot["blocked"] == 1
        assert snapshot["active_list"] == [{"id": "goal-1"}]

    def test_get_approval_runtime_snapshot_returns_approval_slice(self):
        import remy.core.combined_runner as combined_runner

        with patch.object(combined_runner, "get_operator_runtime_snapshot", return_value={
            "approvals": {
                "pending_count": 2,
                "pending": [{"action_id": "approve-1"}],
            }
        }):
            snapshot = combined_runner.get_approval_runtime_snapshot(goal_limit=5, approval_limit=10)

        assert snapshot["pending_count"] == 2
        assert snapshot["pending"] == [{"action_id": "approve-1"}]

    def test_resolve_operator_approval_prefers_v3_governance(self):
        import remy.core.combined_runner as combined_runner

        approval = type(
            "ApprovalEngine",
            (),
            {
                "pending": lambda self: [type("Req", (), {"id": "approval-v3-1"})()],
                "approve": lambda self, request_id, decided_by="operator": request_id == "approval-v3-1" and decided_by == "telegram",
                "deny": lambda self, request_id, decided_by="operator", reason="": False,
            },
        )()
        with patch.object(combined_runner, "get_autonomy_runtime_component", return_value=approval):
            result = combined_runner.resolve_operator_approval("approval-v3", approved=True, decided_by="telegram")

        assert result["ok"] is True
        assert result["source"] == "v3_governance"
        assert result["action_id"] == "approval-v3-1"

    def test_resolve_operator_approval_falls_back_to_legacy_queue(self):
        import remy.core.combined_runner as combined_runner

        with patch.object(combined_runner, "get_autonomy_runtime_component", return_value=None), \
             patch("remy.core.approval_queue.approval_queue") as mock_queue:
            mock_queue.resolve_by_id.return_value = True
            result = combined_runner.resolve_operator_approval("appr-1", approved=False, decided_by="web")

        mock_queue.resolve_by_id.assert_called_once_with("appr-1", approved=False)
        assert result["ok"] is True
        assert result["source"] == "legacy_queue"

    def test_resolve_operator_approval_reply_uses_oldest_pending_item(self):
        import remy.core.combined_runner as combined_runner

        with patch.object(combined_runner, "get_approval_runtime_snapshot", return_value={
            "pending": [
                {"action_id": "newer-1", "created_at": 200.0},
                {"action_id": "older-1", "created_at": 100.0},
            ]
        }), patch.object(combined_runner, "resolve_operator_approval", return_value={"ok": True, "source": "v3_governance"}) as mock_resolve:
            result = combined_runner.resolve_operator_approval_reply("yes", decided_by="telegram-reply")

        mock_resolve.assert_called_once_with("older-1", approved=True, decided_by="telegram-reply")
        assert result["consumed"] is True
        assert result["approved"] is True
        assert result["action_id"] == "older-1"

    def test_resolve_operator_guidance_reply_uses_oldest_pending_item(self):
        import remy.core.combined_runner as combined_runner

        with patch.object(combined_runner, "get_guidance_runtime_snapshot", return_value={
            "pending": [
                {"request_id": "newer-guidance", "created_at": 200.0},
                {"request_id": "older-guidance", "created_at": 100.0},
            ]
        }), patch.object(combined_runner, "resolve_operator_guidance", return_value={"ok": True, "request_id": "older-guidance"}) as mock_resolve:
            result = combined_runner.resolve_operator_guidance_reply("Use the cached report")

        mock_resolve.assert_called_once_with("older-guidance", "Use the cached report")
        assert result["consumed"] is True
        assert result["request_id"] == "older-guidance"


# ============== TELEGRAM REGISTER HANDLERS TESTS ==============


class TestRegisterHandlers:

    def _make_bot(self):
        with patch("remy.core.telegram_bot.settings") as mock_settings, \
             patch("remy.core.telegram_bot.genai"), \
             patch("remy.core.telegram_bot.brain"):
            mock_settings.GEMINI_API_KEY = "test-key"
            mock_settings.TELEGRAM_BOT_TOKEN = "test-token"
            mock_settings.SUMMARY_MODEL = "test-model"
            mock_settings.AURA_BRAIN_PATH = "/tmp/brain"
            from remy.core.telegram_bot import TelegramBot
            return TelegramBot()

    def test_register_handlers_adds_expected_set(self):
        """register_handlers should register the full operator command set."""
        bot = self._make_bot()
        mock_app = MagicMock()
        mock_app.add_handler = MagicMock()

        bot.register_handlers(mock_app)

        assert mock_app.add_handler.call_count == 16

    def test_no_duplicate_voice_handler(self):
        """Only one voice handler should be registered (bugfix)."""
        bot = self._make_bot()
        mock_app = MagicMock()
        handlers_added = []
        mock_app.add_handler = lambda h: handlers_added.append(h)

        bot.register_handlers(mock_app)

        from telegram.ext import MessageHandler
        voice_handlers = [
            h for h in handlers_added
            if isinstance(h, MessageHandler) and "VOICE" in str(h.filters)
        ]
        assert len(voice_handlers) == 1

    def test_run_uses_register_handlers(self):
        """run() should call register_handlers internally."""
        bot = self._make_bot()

        with patch.object(bot, "register_handlers") as mock_reg, \
             patch("remy.core.telegram_bot.ApplicationBuilder") as mock_builder, \
             patch("remy.core.telegram_bot.brain") as mock_brain, \
             patch("remy.core.telegram_bot.get_registry") as mock_registry:
            mock_brain.count.return_value = 0
            mock_registry.return_value.get_all_declarations.return_value = []
            mock_app = MagicMock()
            mock_builder.return_value.token.return_value.build.return_value = mock_app

            # run_polling is blocking, so mock it
            mock_app.run_polling = MagicMock()

            bot.run()
            mock_reg.assert_called_once_with(mock_app)

    @pytest.mark.asyncio
    async def test_start_telegram_async_configures_command_menu(self):
        mock_app = AsyncMock()
        mock_app.bot.set_my_commands = AsyncMock()
        mock_app.initialize = AsyncMock()
        mock_app.start = AsyncMock()
        mock_app.add_error_handler = MagicMock()
        mock_app.updater = AsyncMock()
        mock_app.updater.start_polling = AsyncMock()
        mock_builder = MagicMock()
        mock_builder.return_value.token.return_value.build.return_value = mock_app

        with patch("telegram.ext.ApplicationBuilder", mock_builder), \
             patch("remy.core.telegram_bot.TelegramBot") as mock_bot_cls:
            mock_bot = MagicMock()
            mock_bot.token = "test-token"
            mock_bot.register_handlers = MagicMock()
            mock_bot.configure_app = AsyncMock()
            mock_bot_cls.return_value = mock_bot

            from remy.core.combined_runner import _start_telegram_async
            app = await _start_telegram_async()

            assert app is mock_app
            mock_bot.register_handlers.assert_called_once_with(mock_app)
            mock_bot.configure_app.assert_awaited_once_with(mock_app)
