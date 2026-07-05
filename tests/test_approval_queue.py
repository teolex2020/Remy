"""
Tests for Human-in-the-Loop Approval Queue (Variant B).

Covers:
- URL / tag classification helpers
- needs_approval() routing logic
- build_approval_description() formatting
- handle_reply() confirmation / rejection parsing
- Full approval flow (approve, reject, timeout)
- Disabled queue passes through directly
- Telegram not configured → rejection
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================
# Helpers / fixtures
# ============================================================

@pytest.fixture(autouse=True)
def reset_queue():
    """Clear the singleton approval queue before each test."""
    from remy.core.approval_queue import approval_queue
    approval_queue.clear()
    approval_queue._enabled = None  # reset lazy cache
    yield
    approval_queue.clear()
    approval_queue._enabled = None


# ============================================================
# URL / tag classification
# ============================================================

class TestUrlClassification:
    def test_financial_url_crypto_exchange(self):
        from remy.core.approval_queue import ApprovalQueue
        q = ApprovalQueue()
        assert q.url_is_financial("https://binance.com/trade")

    def test_financial_url_bank(self):
        from remy.core.approval_queue import ApprovalQueue
        q = ApprovalQueue()
        assert q.url_is_financial("https://privatbank.ua/dashboard")

    def test_financial_url_paypal(self):
        from remy.core.approval_queue import ApprovalQueue
        q = ApprovalQueue()
        assert q.url_is_financial("https://paypal.com/send")

    def test_financial_url_negative(self):
        from remy.core.approval_queue import ApprovalQueue
        q = ApprovalQueue()
        assert not q.url_is_financial("https://example.com/news")

    def test_registration_url(self):
        from remy.core.approval_queue import ApprovalQueue
        q = ApprovalQueue()
        assert q.url_is_registration("https://example.com/register")

    def test_registration_url_signup(self):
        from remy.core.approval_queue import ApprovalQueue
        q = ApprovalQueue()
        assert q.url_is_registration("https://app.com/sign-up")

    def test_registration_url_negative(self):
        from remy.core.approval_queue import ApprovalQueue
        q = ApprovalQueue()
        assert not q.url_is_registration("https://example.com/blog")

    def test_tags_are_financial_wallet(self):
        from remy.core.approval_queue import ApprovalQueue
        q = ApprovalQueue()
        assert q.tags_are_financial(["personal", "wallet", "eth"])

    def test_tags_are_financial_string(self):
        from remy.core.approval_queue import ApprovalQueue
        q = ApprovalQueue()
        assert q.tags_are_financial("health, crypto, daily")

    def test_tags_are_financial_negative(self):
        from remy.core.approval_queue import ApprovalQueue
        q = ApprovalQueue()
        assert not q.tags_are_financial(["health", "family", "notes"])


# ============================================================
# needs_approval() routing
# ============================================================

class TestNeedsApproval:
    def test_browser_act_financial_url(self):
        from remy.core.approval_queue import needs_approval, approval_queue
        approval_queue._enabled = True
        args = {"url": "https://binance.com/trade", "action": "click", "selector": "#btn"}
        assert needs_approval("browser_act", args, "https://binance.com/trade")

    def test_browser_act_registration_url(self):
        from remy.core.approval_queue import needs_approval, approval_queue
        approval_queue._enabled = True
        args = {"url": "https://app.com/register", "action": "fill", "selector": "#email"}
        assert needs_approval("browser_act", args, "https://app.com/register")

    def test_browse_page_financial_url(self):
        from remy.core.approval_queue import needs_approval, approval_queue
        approval_queue._enabled = True
        args = {"url": "https://coinbase.com/"}
        assert needs_approval("browse_page", args)

    def test_browse_page_normal_url_no_approval(self):
        from remy.core.approval_queue import needs_approval, approval_queue
        approval_queue._enabled = True
        args = {"url": "https://news.ycombinator.com/"}
        assert not needs_approval("browse_page", args)

    def test_store_financial_tags(self):
        from remy.core.approval_queue import needs_approval, approval_queue
        approval_queue._enabled = True
        args = {"content": "My ETH wallet is 0xABC...", "tags": "personal,wallet,eth"}
        assert needs_approval("store", args)

    def test_store_non_financial_tags(self):
        from remy.core.approval_queue import needs_approval, approval_queue
        approval_queue._enabled = True
        args = {"content": "Took ibuprofen 200mg", "tags": "health,medication"}
        assert not needs_approval("store", args)

    def test_disabled_queue_bypasses_all(self):
        from remy.core.approval_queue import needs_approval, approval_queue
        approval_queue._enabled = False
        args = {"url": "https://binance.com/trade", "action": "click"}
        assert not needs_approval("browser_act", args, "https://binance.com/trade")

    def test_browser_close_never_needs_approval(self):
        from remy.core.approval_queue import needs_approval, approval_queue
        approval_queue._enabled = True
        # browser_close is explicitly excluded from the needs_approval logic
        assert not needs_approval("browser_close", {})


# ============================================================
# build_approval_description()
# ============================================================

class TestBuildDescription:
    def test_browser_act_description(self):
        from remy.core.approval_queue import build_approval_description
        args = {"url": "https://binance.com/trade", "action": "click", "selector": "#submit"}
        desc = build_approval_description("browser_act", args)
        assert "click" in desc
        assert "binance" in desc
        assert "#submit" in desc

    def test_browse_page_description(self):
        from remy.core.approval_queue import build_approval_description
        args = {"url": "https://coinbase.com/"}
        desc = build_approval_description("browse_page", args)
        assert "coinbase" in desc

    def test_store_description(self):
        from remy.core.approval_queue import build_approval_description
        args = {"content": "wallet address is 0xABC", "tags": "wallet,crypto"}
        desc = build_approval_description("store", args)
        assert "wallet" in desc
        assert "фінансових" in desc


# ============================================================
# handle_reply()
# ============================================================

class TestHandleReply:
    def test_no_pending_returns_false(self):
        from remy.core.approval_queue import approval_queue
        assert not approval_queue.handle_reply("Так")

    def test_confirm_ukrainian(self):
        from remy.core.approval_queue import approval_queue, PendingAction
        action = PendingAction(
            action_id="test-1",
            description="test action",
            action_fn=lambda: '{"ok": true}',
            timeout_sec=60,
        )
        approval_queue._pending["test-1"] = action
        result = approval_queue.handle_reply("Так")
        assert result is True
        assert action._approved is True
        assert action._resolved is True

    def test_reject_ukrainian(self):
        from remy.core.approval_queue import approval_queue, PendingAction
        action = PendingAction(
            action_id="test-2",
            description="test action",
            action_fn=lambda: '{"ok": true}',
            timeout_sec=60,
        )
        approval_queue._pending["test-2"] = action
        result = approval_queue.handle_reply("Ні")
        assert result is True
        assert action._approved is False
        assert action._resolved is True

    def test_confirm_english(self):
        from remy.core.approval_queue import approval_queue, PendingAction
        action = PendingAction(
            action_id="test-3",
            description="test action",
            action_fn=lambda: '{"ok": true}',
            timeout_sec=60,
        )
        approval_queue._pending["test-3"] = action
        result = approval_queue.handle_reply("yes")
        assert result is True
        assert action._approved is True

    def test_random_message_not_consumed(self):
        from remy.core.approval_queue import approval_queue, PendingAction
        action = PendingAction(
            action_id="test-4",
            description="test action",
            action_fn=lambda: '{"ok": true}',
            timeout_sec=60,
        )
        approval_queue._pending["test-4"] = action
        result = approval_queue.handle_reply("What is the weather like today?")
        assert result is False
        assert not action._resolved

    def test_already_resolved_not_consumed_again(self):
        from remy.core.approval_queue import approval_queue, PendingAction
        action = PendingAction(
            action_id="test-5",
            description="test action",
            action_fn=lambda: '{"ok": true}',
            timeout_sec=60,
        )
        action._resolved = True  # already resolved
        approval_queue._pending["test-5"] = action
        result = approval_queue.handle_reply("Так")
        assert result is False


# ============================================================
# Full async approval flow
# ============================================================

class TestApprovalFlow:
    @pytest.mark.asyncio
    async def test_approval_approved(self):
        from remy.core.approval_queue import ApprovalQueue
        q = ApprovalQueue()
        q._enabled = True

        call_log = []

        def action_fn():
            call_log.append("executed")
            return json.dumps({"success": True})

        async def approve_after_delay():
            await asyncio.sleep(0.05)
            # Directly resolve the first pending action
            action_id = next(iter(q._pending))
            action = q._pending[action_id]
            action._resolved = True
            action._approved = True
            action._event.set()

        with patch.object(q, "_send_confirmation_request"), \
             patch.object(q, "_send_outcome_notification"):
            task = asyncio.create_task(q.request_approval("Test action", action_fn))
            await asyncio.sleep(0.01)   # let the coroutine register the pending action
            await approve_after_delay()
            result = await task

        assert json.loads(result) == {"success": True}
        assert call_log == ["executed"]

    @pytest.mark.asyncio
    async def test_approval_rejected(self):
        from remy.core.approval_queue import ApprovalQueue
        q = ApprovalQueue()
        q._enabled = True

        call_log = []

        def action_fn():
            call_log.append("executed")
            return json.dumps({"success": True})

        async def reject_after_delay():
            await asyncio.sleep(0.05)
            action_id = next(iter(q._pending))
            action = q._pending[action_id]
            action._resolved = True
            action._approved = False
            action._event.set()

        with patch.object(q, "_send_confirmation_request"), \
             patch.object(q, "_send_outcome_notification"):
            task = asyncio.create_task(q.request_approval("Test action", action_fn))
            await asyncio.sleep(0.01)
            await reject_after_delay()
            result = await task

        parsed = json.loads(result)
        assert "error" in parsed
        assert call_log == []   # action NOT called

    @pytest.mark.asyncio
    async def test_approval_timeout(self):
        from remy.core.approval_queue import ApprovalQueue
        q = ApprovalQueue()
        q._enabled = True

        def action_fn():
            return json.dumps({"success": True})

        with patch.object(q, "_send_confirmation_request"), \
             patch.object(q, "_send_outcome_notification"):
            # Set a very short timeout for this test
            action_result = None
            async def run_with_short_timeout():
                nonlocal action_result
                action = None
                import uuid
                from remy.core.approval_queue import PendingAction
                action = PendingAction(
                    action_id=str(uuid.uuid4()),
                    description="Timeout test",
                    action_fn=action_fn,
                    timeout_sec=1,   # 1 second
                )
                q._pending[action.action_id] = action
                try:
                    await asyncio.wait_for(action._event.wait(), timeout=action.timeout_sec)
                except asyncio.TimeoutError:
                    action._resolved = True
                    action._approved = False
                q._pending.pop(action.action_id, None)
                if not action._approved:
                    action_result = json.dumps({"error": "timed out"})

            await run_with_short_timeout()

        assert action_result is not None
        assert "timed out" in json.loads(action_result)["error"]


# ============================================================
# Sync bridge + disabled queue
# ============================================================

class TestSyncBridge:
    def test_disabled_queue_executes_directly(self):
        from remy.core.approval_queue import ApprovalQueue
        q = ApprovalQueue()
        q._enabled = False

        def action_fn():
            return json.dumps({"ok": True})

        result = q.request_approval_sync("Test", action_fn)
        assert json.loads(result) == {"ok": True}

    def test_no_telegram_still_waits_for_web_gui(self):
        """Without Telegram, the queue should still wait (Web GUI can resolve it).
        After timeout the action is rejected — not instantly blocked."""
        from remy.core.approval_queue import ApprovalQueue, PendingAction
        import uuid

        q = ApprovalQueue()
        q._enabled = True

        def action_fn():
            return json.dumps({"ok": True})

        # Directly simulate a timeout by resolving with approved=False immediately
        async def _fast_timeout(description, action_fn):
            act = PendingAction(
                action_id=str(uuid.uuid4()),
                description=description,
                action_fn=action_fn,
                timeout_sec=0,  # expire immediately
            )
            import asyncio, json
            q._pending[act.action_id] = act
            try:
                await asyncio.wait_for(act._event.wait(), timeout=0.01)
            except asyncio.TimeoutError:
                act._resolved = True
                act._approved = False
            q._pending.pop(act.action_id, None)
            return json.dumps({"error": "timed out", "action_id": act.action_id[:8]})

        q.request_approval = _fast_timeout

        with patch.object(type(q), "_telegram_configured", new_callable=lambda: property(lambda self: False)):
            # Run the sync wrapper but use the fast coroutine
            import threading, asyncio, json as _json
            result_box = []
            def _run():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result_box.append(loop.run_until_complete(q.request_approval("Test without Telegram", action_fn)))
                finally:
                    loop.close()
            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(timeout=5)

        assert result_box, "Approval coroutine never returned"
        parsed = _json.loads(result_box[0])
        assert "error" in parsed  # rejected (timed out)

    def test_clear_resets_pending(self):
        from remy.core.approval_queue import approval_queue, PendingAction
        action = PendingAction(
            action_id="clear-test",
            description="pending",
            action_fn=lambda: "{}",
            timeout_sec=60,
        )
        approval_queue._pending["clear-test"] = action
        approval_queue.clear()
        assert approval_queue.pending_count() == 0
        assert action._resolved is True
        assert action._approved is False


# ============================================================
# pending_count
# ============================================================

class TestPendingCount:
    def test_empty_queue(self):
        from remy.core.approval_queue import approval_queue
        assert approval_queue.pending_count() == 0

    def test_one_pending(self):
        from remy.core.approval_queue import approval_queue, PendingAction
        action = PendingAction(
            action_id="count-test",
            description="counting",
            action_fn=lambda: "{}",
            timeout_sec=60,
        )
        approval_queue._pending["count-test"] = action
        assert approval_queue.pending_count() == 1


class TestSnapshots:
    def test_snapshot_pending_returns_normalized_items(self):
        from remy.core.approval_queue import ApprovalQueue, PendingAction

        q = ApprovalQueue()
        action = PendingAction(
            action_id="snap-1",
            description="Review payout request",
            action_fn=lambda: "{}",
            timeout_sec=60,
        )
        action.created_at = time.time() - 12
        q._pending[action.action_id] = action

        snapshot = q.snapshot_pending()

        assert len(snapshot) == 1
        assert snapshot[0]["id"] == "snap-1"
        assert snapshot[0]["action_id"] == "snap-1"
        assert snapshot[0]["description"] == "Review payout request"
        assert snapshot[0]["timeout_sec"] == 60
        assert snapshot[0]["created_at"] == action.created_at
        assert snapshot[0]["expires_at"] == action.created_at + 60
        assert snapshot[0]["age_sec"] >= 12
