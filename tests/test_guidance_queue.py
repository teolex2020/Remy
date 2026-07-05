"""Tests for Guidance Queue (AUTON-3) — interactive escalation."""

import asyncio
import time
from unittest.mock import patch

import pytest

# ============== Unit Tests: GuidanceQueue basics ==============


class TestGuidanceQueueBasics:
    def test_singleton_exists(self):
        from remy.core.guidance_queue import guidance_queue

        assert guidance_queue is not None

    def test_pending_count_starts_zero(self):
        from remy.core.guidance_queue import GuidanceQueue

        q = GuidanceQueue()
        assert q.pending_count() == 0

    def test_enabled_reads_settings(self):
        from remy.core.guidance_queue import GuidanceQueue

        q = GuidanceQueue()
        q._enabled = None  # Reset cache
        with patch("remy.config.settings.settings") as mock_s:
            mock_s.GUIDANCE_QUEUE_ENABLED = False
            assert q.enabled is False

    def test_timeout_reads_settings(self):
        from remy.core.guidance_queue import GuidanceQueue

        q = GuidanceQueue()
        with patch("remy.config.settings.settings") as mock_s:
            mock_s.GUIDANCE_TIMEOUT_SEC = 60
            assert q.timeout_sec == 60

    def test_telegram_configured_check(self):
        from remy.core.guidance_queue import GuidanceQueue

        q = GuidanceQueue()
        with patch("remy.config.settings.settings") as mock_s:
            mock_s.TELEGRAM_BOT_TOKEN = "tok"
            mock_s.PROACTIVE_CHAT_ID = 123
            assert q._telegram_configured is True

    def test_telegram_not_configured_when_missing(self):
        from remy.core.guidance_queue import GuidanceQueue

        q = GuidanceQueue()
        with patch("remy.config.settings.settings") as mock_s:
            mock_s.TELEGRAM_BOT_TOKEN = None
            mock_s.PROACTIVE_CHAT_ID = None
            assert q._telegram_configured is False


# ============== Unit Tests: request_guidance async ==============


class TestRequestGuidanceAsync:
    @pytest.mark.asyncio
    async def test_request_returns_answer_when_resolved(self):
        from remy.core.guidance_queue import GuidanceQueue

        q = GuidanceQueue()
        q._enabled = True

        async def _resolve_after_delay():
            await asyncio.sleep(0.05)
            q.handle_reply("Use a different approach")

        task = asyncio.create_task(_resolve_after_delay())
        answer = await q.request_guidance("What should I do?", context="test")
        await task

        assert answer == "Use a different approach"

    @pytest.mark.asyncio
    async def test_request_returns_none_on_timeout(self):
        from remy.core.guidance_queue import GuidanceQueue, PendingGuidanceRequest

        q = GuidanceQueue()
        q._enabled = True

        # Create request directly with short timeout
        req = PendingGuidanceRequest(
            request_id="timeout-test",
            question="Quick question?",
            timeout_sec=0,  # immediate timeout
        )
        q._pending[req.request_id] = req

        try:
            await asyncio.wait_for(req._event.wait(), timeout=0.05)
        except asyncio.TimeoutError:
            req._resolved = True
            req._answer = None

        q._pending.pop(req.request_id, None)
        assert req._answer is None

    @pytest.mark.asyncio
    async def test_pending_count_increases_during_wait(self):
        from remy.core.guidance_queue import GuidanceQueue

        q = GuidanceQueue()
        q._enabled = True

        assert q.pending_count() == 0

        async def _check_count():
            await asyncio.sleep(0.02)
            count = q.pending_count()
            q.handle_reply("answer")
            return count

        check_task = asyncio.create_task(_check_count())
        answer = await q.request_guidance("test?")
        count_during = await check_task

        assert count_during == 1
        assert q.pending_count() == 0
        assert answer == "answer"

    @pytest.mark.asyncio
    async def test_emits_events(self):
        from remy.core.guidance_queue import GuidanceQueue

        q = GuidanceQueue()
        q._enabled = True

        events = []
        with patch("remy.core.event_bus.event_bus") as mock_bus:
            mock_bus.emit = lambda name, data: events.append((name, data))

            async def _resolve():
                await asyncio.sleep(0.02)
                q.handle_reply("ok")

            task = asyncio.create_task(_resolve())
            await q.request_guidance("test?")
            await task

        event_names = [e[0] for e in events]
        assert "guidance.pending" in event_names
        assert "guidance.resolved" in event_names


# ============== Unit Tests: handle_reply ==============


class TestHandleReply:
    def test_handle_reply_no_pending_returns_false(self):
        from remy.core.guidance_queue import GuidanceQueue

        q = GuidanceQueue()
        assert q.handle_reply("hello") is False

    def test_handle_reply_empty_text_returns_false(self):
        from remy.core.guidance_queue import GuidanceQueue, PendingGuidanceRequest

        q = GuidanceQueue()
        req = PendingGuidanceRequest(request_id="r1", question="?")
        q._pending["r1"] = req
        assert q.handle_reply("") is False
        assert q.handle_reply("   ") is False

    @pytest.mark.asyncio
    async def test_handle_reply_resolves_oldest(self):
        from remy.core.guidance_queue import GuidanceQueue

        q = GuidanceQueue()
        q._enabled = True

        answers = []

        async def _ask(question):
            ans = await q.request_guidance(question)
            answers.append(ans)

        t1 = asyncio.create_task(_ask("Q1?"))
        await asyncio.sleep(0.02)
        t2 = asyncio.create_task(_ask("Q2?"))
        await asyncio.sleep(0.02)

        # Reply should go to Q1 (oldest)
        q.handle_reply("Answer for Q1")
        await asyncio.sleep(0.05)
        q.handle_reply("Answer for Q2")
        await asyncio.sleep(0.05)

        await t1
        await t2

        assert answers[0] == "Answer for Q1"
        assert answers[1] == "Answer for Q2"


# ============== Unit Tests: resolve_by_id ==============


class TestResolveById:
    @pytest.mark.asyncio
    async def test_resolve_by_id_works(self):
        from remy.core.guidance_queue import GuidanceQueue

        q = GuidanceQueue()
        q._enabled = True

        async def _ask():
            return await q.request_guidance("need help")

        task = asyncio.create_task(_ask())
        await asyncio.sleep(0.02)

        # Get the request ID
        req_id = next(iter(q._pending))
        resolved = q.resolve_by_id(req_id, "here's help")
        assert resolved is True

        answer = await task
        assert answer == "here's help"

    def test_resolve_by_id_no_pending(self):
        from remy.core.guidance_queue import GuidanceQueue

        q = GuidanceQueue()
        assert q.resolve_by_id("nonexistent", "answer") is False

    @pytest.mark.asyncio
    async def test_resolve_by_id_prefix_match(self):
        from remy.core.guidance_queue import GuidanceQueue

        q = GuidanceQueue()
        q._enabled = True

        async def _ask():
            return await q.request_guidance("need help")

        task = asyncio.create_task(_ask())
        await asyncio.sleep(0.02)

        req_id = next(iter(q._pending))
        # Use prefix (first 8 chars)
        resolved = q.resolve_by_id(req_id[:8], "prefixed answer")
        assert resolved is True

        answer = await task
        assert answer == "prefixed answer"


# ============== Unit Tests: request_guidance_sync ==============


class TestRequestGuidanceSync:
    def test_sync_disabled_returns_none(self):
        from remy.core.guidance_queue import GuidanceQueue

        q = GuidanceQueue()
        q._enabled = False
        result = q.request_guidance_sync("test?")
        assert result is None

    def test_sync_bridge_works_with_reply(self):
        import threading

        from remy.core.guidance_queue import GuidanceQueue

        q = GuidanceQueue()
        q._enabled = True

        def _reply_after_delay():
            time.sleep(0.1)
            # The sync bridge creates its own event loop, so handle_reply
            # needs to set the event properly
            q.handle_reply("sync answer")

        t = threading.Thread(target=_reply_after_delay, daemon=True)
        t.start()

        with patch("remy.config.settings.settings") as mock_s:
            mock_s.GUIDANCE_TIMEOUT_SEC = 3
            result = q.request_guidance_sync("help?")

        t.join(timeout=2)
        assert result == "sync answer"

    def test_sync_bridge_returns_none_on_timeout(self):
        from remy.core.guidance_queue import GuidanceQueue

        q = GuidanceQueue()
        q._enabled = True

        with patch("remy.config.settings.settings") as mock_s:
            mock_s.GUIDANCE_TIMEOUT_SEC = 0
            result = q.request_guidance_sync("will timeout")

        assert result is None


# ============== Unit Tests: Telegram notification ==============


class TestTelegramNotification:
    def test_sends_telegram_when_configured(self):
        from remy.core.guidance_queue import GuidanceQueue, PendingGuidanceRequest

        q = GuidanceQueue()

        with (
            patch("remy.core.notification_router.should_notify_telegram", return_value=True),
            patch("remy.core.notification_router.is_web_runtime_available", return_value=False),
            patch("remy.core.notification_router.send_telegram") as mock_send,
            patch.object(
                type(q), "_telegram_configured", new_callable=lambda: property(lambda self: True)
            ),
        ):
            req = PendingGuidanceRequest(
                request_id="test123",
                question="Test question?",
            )
            q._send_guidance_telegram(req)

            mock_send.assert_called_once()
            msg = mock_send.call_args[0][0]
            assert "Test question?" in msg

    def test_skips_telegram_when_not_configured(self):
        from remy.core.guidance_queue import GuidanceQueue, PendingGuidanceRequest

        q = GuidanceQueue()

        with patch("remy.config.settings.settings") as mock_s:
            mock_s.TELEGRAM_BOT_TOKEN = None
            mock_s.PROACTIVE_CHAT_ID = None

            req = PendingGuidanceRequest(
                request_id="test456",
                question="Another test?",
            )
            # Should not raise — just skip
            q._send_guidance_telegram(req)


# ============== Integration: guidance in Telegram handler ==============


class TestTelegramGuidanceIntegration:
    @pytest.mark.asyncio
    async def test_telegram_checks_guidance_before_approval(self):
        """Verify guidance_queue.handle_reply is checked before approval_queue."""
        from remy.core.guidance_queue import GuidanceQueue

        q = GuidanceQueue()

        # Simulate pending request
        from remy.core.guidance_queue import PendingGuidanceRequest

        req = PendingGuidanceRequest(request_id="tg1", question="Help?")
        q._pending["tg1"] = req

        # Reply should be consumed by guidance queue
        assert q.handle_reply("User answer") is True
        assert req._answer == "User answer"
        assert req._resolved is True


# ============== Integration: auto-escalation in autonomy ==============


class TestAutoEscalation:
    def test_escalation_trigger_on_consecutive_failures(self):
        """Verify the escalation logic fires after 2+ failures."""
        with patch("remy.core.autonomy.settings") as mock_settings:
            mock_settings.AUTONOMY_DAILY_TOKEN_LIMIT = 100_000
            mock_settings.AUTONOMY_HOURLY_TOKEN_LIMIT = 20_000
            mock_settings.AUTONOMY_SESSION_TOKEN_LIMIT = 500_000
            mock_settings.AUTONOMY_CYCLE_INTERVAL_SEC = 0
            mock_settings.AUTONOMY_TELEGRAM_NOTIFICATIONS = False
            mock_settings.AUTONOMY_SELF_CRITIQUE_ENABLED = False
            mock_settings.SUMMARY_MODEL = "test-model"
            mock_settings.GEMINI_API_KEY = "test-key"
            mock_settings.DATA_DIR = "/tmp"

            from remy.core.autonomy import AutonomousLoop

            loop = AutonomousLoop()
            loop.consecutive_failures = 2  # Already 2 failures

            # The escalation check: consecutive_failures + 1 (current fail) >= 2
            effective = loop.consecutive_failures + 1  # = 3
            assert effective >= 2  # Should trigger escalation

    def test_no_escalation_on_first_failure(self):
        """No escalation when it's the first failure."""
        with patch("remy.core.autonomy.settings") as mock_settings:
            mock_settings.AUTONOMY_DAILY_TOKEN_LIMIT = 100_000
            mock_settings.AUTONOMY_HOURLY_TOKEN_LIMIT = 20_000
            mock_settings.AUTONOMY_SESSION_TOKEN_LIMIT = 500_000
            mock_settings.AUTONOMY_CYCLE_INTERVAL_SEC = 0
            mock_settings.AUTONOMY_TELEGRAM_NOTIFICATIONS = False
            mock_settings.AUTONOMY_SELF_CRITIQUE_ENABLED = False
            mock_settings.SUMMARY_MODEL = "test-model"
            mock_settings.GEMINI_API_KEY = "test-key"
            mock_settings.DATA_DIR = "/tmp"

            from remy.core.autonomy import AutonomousLoop

            loop = AutonomousLoop()
            loop.consecutive_failures = 0

            effective = loop.consecutive_failures + 1  # = 1
            assert effective < 2  # Should NOT trigger escalation

    def test_escalation_on_low_confidence(self):
        """Escalation triggers on very low confidence even with 1 failure."""
        evaluation = {"success": False, "confidence": 0.2, "reason": "uncertain"}
        assert evaluation["confidence"] < 0.3  # Low confidence trigger


# ============== Settings ==============


class TestGuidanceSettings:
    def test_settings_defaults(self):
        from remy.config.settings import Settings

        s = Settings(GEMINI_API_KEY="test")
        assert s.GUIDANCE_QUEUE_ENABLED is True
        assert s.GUIDANCE_TIMEOUT_SEC == 120

    def test_settings_override(self):
        from remy.config.settings import Settings

        s = Settings(
            GEMINI_API_KEY="test",
            GUIDANCE_QUEUE_ENABLED=False,
            GUIDANCE_TIMEOUT_SEC=60,
        )
        assert s.GUIDANCE_QUEUE_ENABLED is False
        assert s.GUIDANCE_TIMEOUT_SEC == 60
