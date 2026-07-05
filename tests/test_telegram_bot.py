"""Tests for Phase 4: Telegram Bot (Omni-Channel)."""

import time
from unittest.mock import patch, MagicMock, AsyncMock
from types import SimpleNamespace

import pytest


# ============== Chat Session Tests ==============

class TestChatSession:

    def _make_bot(self):
        """Create a TelegramBot with mocked externals."""
        with patch("remy.core.telegram_bot.settings") as mock_settings, \
             patch("remy.core.telegram_bot.genai"), \
             patch("remy.core.telegram_bot.brain"):
            mock_settings.GEMINI_API_KEY = "test-key"
            mock_settings.TELEGRAM_BOT_TOKEN = "test-token"
            mock_settings.SUMMARY_MODEL = "test-model"
            mock_settings.AURA_BRAIN_PATH = "/tmp/brain"

            from remy.core.telegram_bot import TelegramBot
            return TelegramBot()

    def test_create_new_session(self):
        """First access to a chat_id creates a new session."""
        bot = self._make_bot()
        session = bot._get_or_create_session(12345)
        assert session.session_id is not None
        assert session.history == []
        assert session.session_log == []
        assert 12345 in bot._sessions

    def test_reuse_existing_session(self):
        """Second access to same chat_id returns existing session."""
        bot = self._make_bot()
        s1 = bot._get_or_create_session(12345)
        s2 = bot._get_or_create_session(12345)
        assert s1.session_id == s2.session_id

    def test_different_chats_different_sessions(self):
        """Different chat_ids get different sessions."""
        bot = self._make_bot()
        s1 = bot._get_or_create_session(111)
        s2 = bot._get_or_create_session(222)
        assert s1.session_id != s2.session_id

    def test_timeout_creates_new_session(self):
        """Stale session (past timeout) creates a new session."""
        from remy.core.telegram_bot import SESSION_TIMEOUT_SEC

        bot = self._make_bot()
        s1 = bot._get_or_create_session(12345)
        old_id = s1.session_id

        # Simulate timeout
        s1.last_activity = time.time() - SESSION_TIMEOUT_SEC - 1

        with patch.object(bot, '_close_session_sync'):
            s2 = bot._get_or_create_session(12345)

        assert s2.session_id != old_id


# ============== LangGraph Agent Integration Tests ==============

class TestGeminiRespond:

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

    @pytest.mark.asyncio
    async def test_respond_calls_invoke_agent(self):
        """_gemini_respond delegates to invoke_agent."""
        bot = self._make_bot()

        with patch("remy.core.telegram_bot.invoke_agent", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = ("Hello! How can I help?", [], [])

            result = await bot._gemini_respond(12345, "Hello")

        assert result == "Hello! How can I help?"
        mock_invoke.assert_called_once()
        call_kwargs = mock_invoke.call_args[1]
        assert call_kwargs["user_message"] == "Hello"
        assert call_kwargs["channel"] == "telegram"

    @pytest.mark.asyncio
    async def test_respond_updates_session_state(self):
        """After responding, session history and log are updated."""
        bot = self._make_bot()

        new_history = [{"role": "user"}, {"role": "assistant"}]
        new_log = [{"type": "user_text"}, {"type": "tool_call"}]

        with patch("remy.core.telegram_bot.invoke_agent", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = ("Response", new_history, new_log)

            await bot._gemini_respond(12345, "Hello")

        session = bot._sessions[12345]
        assert session.history == new_history
        assert session.session_log == new_log

    @pytest.mark.asyncio
    async def test_respond_logs_user_text(self):
        """User text is logged in session_log before calling agent."""
        bot = self._make_bot()

        with patch("remy.core.telegram_bot.invoke_agent", new_callable=AsyncMock) as mock_invoke:
            # Return the session_log that was passed (includes user text)
            def capture_log(**kwargs):
                return ("Response", [], kwargs["session_log"])
            mock_invoke.side_effect = capture_log

            await bot._gemini_respond(12345, "Hello there")

        session = bot._sessions[12345]
        assert any(e.get("type") == "user_text" for e in session.session_log)

    @pytest.mark.asyncio
    async def test_respond_passes_session_id(self):
        """invoke_agent receives the correct session_id."""
        bot = self._make_bot()

        with patch("remy.core.telegram_bot.invoke_agent", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = ("OK", [], [])

            await bot._gemini_respond(12345, "test")

        call_kwargs = mock_invoke.call_args[1]
        session = bot._sessions[12345]
        assert call_kwargs["session_id"] == session.session_id


# ============== Session Management Tests ==============

class TestSessionManagement:

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

    @pytest.mark.asyncio
    async def test_close_session_removes_from_cache(self):
        """Closing a session removes it from _sessions."""
        bot = self._make_bot()
        bot._get_or_create_session(12345)
        assert 12345 in bot._sessions

        with patch("remy.core.telegram_bot.generate_session_summary", new_callable=AsyncMock), \
             patch("remy.core.telegram_bot.brain") as mock_brain:
            mock_brain.end_session = MagicMock()
            await bot._close_session(12345)

        assert 12345 not in bot._sessions

    @pytest.mark.asyncio
    async def test_close_nonexistent_session_no_error(self):
        """Closing a non-existent session doesn't crash."""
        bot = self._make_bot()

        # Should not raise
        await bot._close_session(99999)


# ============== Build System Instruction Channel Tests ==============

class TestBuildSystemInstruction:

    def test_voice_channel_instructions(self, tmp_path):
        """Voice channel includes 'concise' and 'aloud' hints."""
        from aura import Aura as CognitiveMemory

        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.brain_tools.brain", b):
            from remy.core.brain_tools import build_system_instruction
            instruction = build_system_instruction(channel="voice")

        assert "concise" in instruction.lower()
        assert "aloud" in instruction.lower()
        b.close()

    def test_telegram_channel_instructions(self, tmp_path):
        """Telegram channel includes 'markdown' and 'structured' hints."""
        from aura import Aura as CognitiveMemory

        b = CognitiveMemory(str(tmp_path / "brain"))

        with patch("remy.core.brain_tools.brain", b):
            from remy.core.brain_tools import build_system_instruction
            instruction = build_system_instruction(channel="telegram")

        assert "markdown" in instruction.lower()
        assert "aloud" not in instruction.lower()
        b.close()


# ============== Settings Tests ==============

class TestTelegramBotSettings:

    def test_telegram_token_default_none(self):
        from remy.config.settings import Settings
        with patch.dict("os.environ", {}, clear=False):
            # Remove any TELEGRAM_BOT_TOKEN from env so default applies
            import os
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)
            os.environ.pop("PACKS_DISABLED", None)
            s = Settings(GEMINI_API_KEY="test", _env_file=None)
            assert s.TELEGRAM_BOT_TOKEN is None


class TestTelegramAuthorization:

    def _make_bot(self, allowed_ids=None):
        with patch("remy.core.telegram_bot.settings") as mock_settings, \
             patch("remy.core.telegram_bot.genai"), \
             patch("remy.core.telegram_bot.brain"):
            mock_settings.GEMINI_API_KEY = "test-key"
            mock_settings.TELEGRAM_BOT_TOKEN = "test-token"
            mock_settings.SUMMARY_MODEL = "test-model"
            mock_settings.AURA_BRAIN_PATH = "/tmp/brain"
            mock_settings.TELEGRAM_ALLOWED_CHAT_IDS = allowed_ids or []

            from remy.core.telegram_bot import TelegramBot
            return TelegramBot()

    def test_authorization_help_mentions_chat_id(self):
        bot = self._make_bot([123])
        text = bot._authorization_help_text(777)
        assert "777" in text
        assert "TELEGRAM_ALLOWED_CHAT_IDS" in text

    @pytest.mark.asyncio
    async def test_whoami_command_reports_status(self):
        bot = self._make_bot([123])
        update = MagicMock()
        update.effective_chat.id = 123
        update.message.reply_text = AsyncMock()

        await bot.whoami_command(update, MagicMock())

        update.message.reply_text.assert_awaited_once()
        sent = update.message.reply_text.await_args.args[0]
        assert "Chat ID" in sent
        assert "authorized" in sent

    @pytest.mark.asyncio
    async def test_help_command_lists_operator_controls(self):
        bot = self._make_bot([123])
        update = MagicMock()
        update.effective_chat.id = 123
        update.message.reply_text = AsyncMock()

        await bot.help_command(update, MagicMock())

        sent = update.message.reply_text.await_args.args[0]
        assert "/status" in sent
        assert "/approve" in sent
        assert "Telegram Operator Mode" in sent

    @pytest.mark.asyncio
    async def test_unauthorized_status_command_returns_auth_help(self):
        bot = self._make_bot([123])
        update = MagicMock()
        update.effective_chat.id = 999
        update.message.reply_text = AsyncMock()

        await bot.status_command(update, MagicMock())

        sent = update.message.reply_text.await_args.args[0]
        assert "TELEGRAM_ALLOWED_CHAT_IDS" in sent
        assert "999" in sent

    @pytest.mark.asyncio
    async def test_status_command_uses_shared_control_state(self):
        bot = self._make_bot([123])
        update = MagicMock()
        update.effective_chat.id = 123
        update.message.reply_text = AsyncMock()

        with patch("remy.core.combined_runner.get_channel_status_snapshot") as mock_snapshot:
            mock_snapshot.return_value = {
                "channels": {
                    "registry_summary": {"health": "ok"},
                    "web": {"health": {"status": "ok"}},
                    "telegram": {"health": {}},
                    "autonomy": {"health": {}},
                },
                "control": {
                    "running": False,
                    "session_id": None,
                    "active_version": "v3",
                    "configured_version": "v3",
                    "runtime_loaded": False,
                    "maintenance_only": False,
                },
            }

            await bot.status_command(update, MagicMock())

        sent = update.message.reply_text.await_args.args[0]
        assert "Autonomy configured: v3 (inactive)" in sent

    @pytest.mark.asyncio
    async def test_status_command_shows_maintenance_mode(self):
        bot = self._make_bot([123])
        update = MagicMock()
        update.effective_chat.id = 123
        update.message.reply_text = AsyncMock()

        with patch("remy.core.combined_runner.get_channel_status_snapshot") as mock_snapshot:
            mock_snapshot.return_value = {
                "channels": {
                    "registry_summary": {"health": "ok"},
                    "web": {"health": {"status": "ok"}},
                    "telegram": {"health": {}},
                    "autonomy": {"health": {}},
                },
                "control": {
                    "running": True,
                    "session_id": "sess-321",
                    "active_version": "v3",
                    "configured_version": "v3",
                    "runtime_loaded": True,
                    "maintenance_only": True,
                },
            }

            await bot.status_command(update, MagicMock())

        sent = update.message.reply_text.await_args.args[0]
        assert "Autonomy loop: running (v3)" in sent
        assert "Mode: maintenance only" in sent

    def test_telegram_commands_include_operator_menu(self):
        bot = self._make_bot([123])
        commands = bot._telegram_commands()
        command_names = [cmd.command for cmd in commands]
        assert "help" in command_names
        assert "whoami" in command_names
        assert "ops" in command_names
        assert "approvals" in command_names
        assert "approve" in command_names

    @pytest.mark.asyncio
    async def test_configure_app_sets_command_menu(self):
        bot = self._make_bot([123])
        app = MagicMock()
        app.bot.set_my_commands = AsyncMock()

        await bot.configure_app(app)

        app.bot.set_my_commands.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_approvals_command_lists_pending_actions(self):
        bot = self._make_bot([123])
        update = MagicMock()
        update.effective_chat.id = 123
        update.message.reply_text = AsyncMock()
        with patch("remy.core.combined_runner.get_approval_runtime_snapshot") as mock_snapshot:
            mock_snapshot.return_value = {
                "pending": [
                    {
                        "id": "abcd1234-0000",
                        "action_id": "abcd1234-0000",
                        "age_sec": 12,
                        "description": "Review payout request",
                    }
                ]
            }
            await bot.approvals_command(update, MagicMock())

        sent = update.message.reply_text.await_args.args[0]
        assert "Pending approvals" in sent
        assert "abcd1234" in sent
        assert "Review payout request" in sent

    @pytest.mark.asyncio
    async def test_approvals_command_shows_routing_pressure_context(self):
        bot = self._make_bot([123])
        update = MagicMock()
        update.effective_chat.id = 123
        update.message.reply_text = AsyncMock()
        with patch("remy.core.combined_runner.get_approval_runtime_snapshot") as mock_snapshot:
            mock_snapshot.return_value = {
                "pending": [
                    {
                        "id": "approval-v3-1",
                        "action_id": "approval-v3-1",
                        "age_sec": 12,
                        "description": "Routing pressure approval: specialist 'researcher' is degraded",
                        "specialist": "researcher",
                        "routing_pressure": True,
                        "context": {"target": "Research counterparty profile"},
                    }
                ]
            }
            await bot.approvals_command(update, MagicMock())

        sent = update.message.reply_text.await_args.args[0]
        assert "routing pressure" in sent
        assert "specialist researcher" in sent
        assert "Research counterparty profile" in sent

    @pytest.mark.asyncio
    async def test_approvals_command_empty(self):
        bot = self._make_bot([123])
        update = MagicMock()
        update.effective_chat.id = 123
        update.message.reply_text = AsyncMock()

        with patch("remy.core.combined_runner.get_approval_runtime_snapshot") as mock_snapshot:
            mock_snapshot.return_value = {"pending": []}
            await bot.approvals_command(update, MagicMock())

        sent = update.message.reply_text.await_args.args[0]
        assert "No pending approvals" in sent

    @pytest.mark.asyncio
    async def test_approve_command_uses_shared_snapshot_for_oldest_pending(self):
        bot = self._make_bot([123])
        update = MagicMock()
        update.effective_chat.id = 123
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.args = []

        with patch("remy.core.combined_runner.get_approval_runtime_snapshot") as mock_snapshot, \
             patch("remy.core.combined_runner.resolve_operator_approval") as mock_resolve:
            mock_snapshot.return_value = {
                "pending": [
                    {
                        "id": "abcd1234-0000",
                        "action_id": "abcd1234-0000",
                        "description": "Review payout request",
                    }
                ]
            }
            mock_resolve.return_value = {"ok": True, "action_id": "abcd1234-0000", "source": "legacy_queue"}

            await bot.approve_command(update, context)

        mock_resolve.assert_called_once_with("abcd1234-0000", approved=True, decided_by="telegram")
        sent = update.message.reply_text.await_args.args[0]
        assert "Approved" in sent
        assert "Review payout request" in sent

    @pytest.mark.asyncio
    async def test_approve_command_includes_routing_pressure_context(self):
        bot = self._make_bot([123])
        update = MagicMock()
        update.effective_chat.id = 123
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.args = []

        with patch("remy.core.combined_runner.get_approval_runtime_snapshot") as mock_snapshot, \
             patch("remy.core.combined_runner.resolve_operator_approval") as mock_resolve:
            mock_snapshot.return_value = {
                "pending": [
                    {
                        "id": "approval-v3-1",
                        "action_id": "approval-v3-1",
                        "description": "Routing pressure approval: specialist 'researcher' is degraded",
                        "specialist": "researcher",
                        "routing_pressure": True,
                        "context": {"target": "Research counterparty profile"},
                    }
                ]
            }
            mock_resolve.return_value = {"ok": True, "action_id": "approval-v3-1", "source": "v3_governance"}

            await bot.approve_command(update, context)

        sent = update.message.reply_text.await_args.args[0]
        assert "Approved" in sent
        assert "routing pressure" in sent
        assert "specialist researcher" in sent
        assert "Research counterparty profile" in sent

    @pytest.mark.asyncio
    async def test_deny_command_uses_shared_snapshot_for_oldest_pending(self):
        bot = self._make_bot([123])
        update = MagicMock()
        update.effective_chat.id = 123
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.args = []

        with patch("remy.core.combined_runner.get_approval_runtime_snapshot") as mock_snapshot, \
             patch("remy.core.combined_runner.resolve_operator_approval") as mock_resolve:
            mock_snapshot.return_value = {
                "pending": [
                    {
                        "id": "abcd1234-0000",
                        "action_id": "abcd1234-0000",
                        "description": "Review payout request",
                    }
                ]
            }
            mock_resolve.return_value = {"ok": True, "action_id": "abcd1234-0000", "source": "legacy_queue"}

            await bot.deny_command(update, context)

        mock_resolve.assert_called_once_with("abcd1234-0000", approved=False, decided_by="telegram")
        sent = update.message.reply_text.await_args.args[0]
        assert "Denied" in sent
        assert "Review payout request" in sent

    @pytest.mark.asyncio
    async def test_handle_message_consumes_free_text_approval_reply_via_shared_seam(self):
        bot = self._make_bot([123])
        update = MagicMock()
        update.effective_chat.id = 123
        update.message.text = "yes"
        update.message.reply_text = AsyncMock()
        update.effective_chat.send_action = AsyncMock()

        with patch("remy.core.combined_runner.resolve_operator_approval_reply", return_value={"consumed": True, "approved": True, "action_id": "approval-v3-1"}), \
             patch.object(bot, "_gemini_respond", new=AsyncMock()) as mock_respond:
            await bot.handle_message(update, MagicMock())

        mock_respond.assert_not_awaited()
        update.effective_chat.send_action.assert_not_awaited()
        update.message.reply_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handle_message_consumes_free_text_guidance_reply_via_shared_seam(self):
        bot = self._make_bot([123])
        update = MagicMock()
        update.effective_chat.id = 123
        update.message.text = "Use the cached report"
        update.message.reply_text = AsyncMock()
        update.effective_chat.send_action = AsyncMock()

        with patch("remy.core.combined_runner.resolve_operator_approval_reply", return_value={"consumed": False}), \
             patch("remy.core.combined_runner.resolve_operator_guidance_reply", return_value={"consumed": True, "request_id": "guidance-1"}), \
             patch.object(bot, "_gemini_respond", new=AsyncMock()) as mock_respond:
            await bot.handle_message(update, MagicMock())

        mock_respond.assert_not_awaited()
        update.effective_chat.send_action.assert_not_awaited()
        update.message.reply_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ops_command_sends_digest(self):
        bot = self._make_bot([123])
        update = MagicMock()
        update.effective_chat.id = 123
        update.message.reply_text = AsyncMock()

        with patch.object(bot, "_build_operator_digest", new=AsyncMock(return_value="*Remy Ops Digest*\nApprovals: 0 pending")):
            await bot.ops_command(update, MagicMock())

        sent = update.message.reply_text.await_args.args[0]
        assert "Remy Ops Digest" in sent
        assert "Approvals" in sent

    @pytest.mark.asyncio
    async def test_build_operator_digest_uses_shared_snapshot(self):
        bot = self._make_bot([123])

        with patch("remy.core.gateway.get_registry") as mock_registry, \
             patch("remy.core.combined_runner.get_operator_console_snapshot") as mock_snapshot:
            mock_registry.return_value.summary.return_value = {"health": "degraded"}
            mock_snapshot.return_value = {
                "autonomy": {"running": True, "version": "v3"},
                "approvals": {"pending_count": 2},
                "goals": {"active": 1, "blocked": 1},
                "budget": {"alert_level": "yellow", "llm_cost_today": 0.82},
                "factuality": {"unsupported_observed_claims_total": 2},
                "quality_debt_by_specialist": [{"id": "researcher", "quality_debt": 0.2}],
                "routing_pressure": {
                    "top_candidate": {"id": "analyst", "quality_adjusted_success_rate": 0.91},
                    "highest_pressure": {"id": "researcher", "quality_debt": 0.2},
                },
            }

            digest = await bot._build_operator_digest()

        assert "Runtime: degraded" in digest
        assert "Autonomy: running (v3)" in digest
        assert "Approvals: 2 pending" in digest
        assert "Goals: 1 active, 1 blocked" in digest
        assert "Budget: yellow, LLM today $0.8200" in digest
        assert "Factuality: 2 unsupported observed claims" in digest
        assert "Quality debt: researcher (0.20)" in digest
        assert "Routing prefer: analyst (0.91)" in digest
        assert "Routing avoid: researcher (0.20 debt)" in digest

    @pytest.mark.asyncio
    async def test_goals_command_uses_shared_goal_snapshot(self):
        bot = self._make_bot([123])
        update = MagicMock()
        update.effective_chat.id = 123
        update.message.reply_text = AsyncMock()

        with patch("remy.core.combined_runner.get_goal_runtime_snapshot") as mock_snapshot:
            mock_snapshot.return_value = {
                "total": 4,
                "active": 2,
                "blocked": 1,
                "active_list": [
                    {"id": "goal-1", "content": "Investigate primary filings", "priority": "high"},
                    {"id": "goal-2", "content": "Draft summary", "priority": "medium"},
                ],
            }
            await bot.goals_command(update, MagicMock())

        sent = update.message.reply_text.await_args.args[0]
        assert "*Goals* — 4 total" in sent
        assert "🟢 Active (2):" in sent
        assert "[high] Investigate primary filings" in sent
        assert "🔴 Blocked: 1" in sent
        assert "⏳ Pending: 1" in sent

    @pytest.mark.asyncio
    async def test_budget_command_uses_shared_budget_snapshot(self):
        bot = self._make_bot([123])
        update = MagicMock()
        update.effective_chat.id = 123
        update.message.reply_text = AsyncMock()

        with patch("remy.core.combined_runner.get_budget_runtime_snapshot") as mock_snapshot:
            mock_snapshot.return_value = {
                "balance_usd": 12.5,
                "runway_days": 14,
                "llm_cost_today": 0.82,
                "alert_level": "yellow",
            }
            await bot.budget_command(update, MagicMock())

        sent = update.message.reply_text.await_args.args[0]
        assert "*Budget*" in sent
        assert "Balance: $12.500" in sent
        assert "Runway: 14 days" in sent
        assert "LLM today: $0.8200" in sent
        assert "Status: yellow" in sent
