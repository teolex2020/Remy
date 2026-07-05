"""
Human-in-the-Loop Approval Queue — Variant B.

When the autonomous agent tries to perform a critical action (browser on a
financial/registration URL, or storing a financial record), execution is
paused and the user receives a Telegram confirmation request.

The action only proceeds when the user replies "Так", "так", "yes", or "YES"
within APPROVAL_TIMEOUT_SEC (default 120 s).  Any other reply (or timeout)
cancels the action and logs it as rejected.

Usage (from brain_tools.py):
    from remy.core.approval_queue import approval_queue, needs_approval

    if needs_approval(tool_name, args, url):
        result = await approval_queue.request_approval(description, action_fn)
        return result

    # Or synchronous wrapper (for use inside execute_tool):
    result = approval_queue.request_approval_sync(description, action_fn)
    return result

Reply handling (from telegram_bot.py):
    from remy.core.approval_queue import approval_queue
    if approval_queue.handle_reply(text):
        return  # message was an approval reply, not a normal message
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================
# URL patterns that trigger approval
# ============================================================

# Financial / crypto service keywords in URL hostnames
_FINANCIAL_URL_KEYWORDS = {
    "bank", "crypto", "wallet", "coin", "exchange", "binance", "coinbase",
    "kraken", "bybit", "okx", "huobi", "kucoin", "bitfinex", "bitget",
    "metamask", "etherscan", "bscscan", "blockchain", "ledger",
    "paypal", "stripe", "wise", "revolut", "monobank", "privatbank",
    "visa", "mastercard", "swift", "iban",
    "invest", "trading", "forex", "stock", "brokerage",
}

# Registration / identity service keywords in URL hostnames
_REGISTRATION_URL_KEYWORDS = {
    "register", "signup", "sign-up", "create-account", "join",
    "identity", "kyc", "verification", "verify",
}

# Memory tags that require approval before storing
_FINANCIAL_TAGS = {"wallet", "crypto", "payment", "bank", "iban", "card", "seed_phrase", "private_key"}

# ============================================================
# Wallet / crypto transaction gates (mandatory approval)
# ============================================================

# Tool names that involve real money movement — ALWAYS require approval
_WALLET_TOOLS = {
    "cdp_agent_manager",
    "cdp_wallet_manager",
    "tron_transfer",
    "send_usdt",
    "send_trx",
    "swap_token",
    "approve_token",
}

# Args keys that indicate money movement
_MONEY_ARG_KEYS = {
    "amount", "value", "transfer_amount", "send_amount",
    "recipient", "to_address", "destination",
}

# Sensitive action types within generic tools (e.g. http_post to exchange APIs)
_SENSITIVE_ACTION_PATTERNS = {
    "transfer", "withdraw", "send", "swap", "approve",
    "buy", "sell", "trade", "stake", "unstake",
}

# User confirmation phrases (case-insensitive)
_CONFIRM_PHRASES = {"так", "yes", "підтверджую", "confirm", "approve", "ok", "ок", "✓", "✅"}
_REJECT_PHRASES = {"ні", "no", "скасувати", "cancel", "reject", "відхилити", "стоп", "stop"}


# ============================================================
# Multi-model review (second-opinion LLM gate)
# ============================================================

def review_action(
    tool_name: str,
    args: dict,
    description: str,
    url: str | None = None,
) -> dict | None:
    """Quick second-opinion LLM review of an action pending approval.

    Returns {"safe": bool, "concerns": str} or None on error (fail-open).
    Uses REVIEW_MODEL (cheap, fast model).
    """
    try:
        from remy.config.settings import settings
        if not settings.REVIEW_ENABLED:
            return None
    except Exception:
        return None

    import json as _json

    args_str = _json.dumps(args, ensure_ascii=False, default=str)[:500]
    prompt = (
        "Review this autonomous agent action for safety risks.\n"
        f"Tool: {tool_name}\n"
        f"Args: {args_str}\n"
        f"URL: {url or 'N/A'}\n"
        f"Description: {description}\n\n"
        "Check for: suspicious URLs, wallet/crypto addresses, credential exposure, "
        "unauthorized financial operations, identity theft patterns.\n\n"
        'Reply ONLY with JSON: {"safe": true/false, "concerns": "brief explanation if not safe"}'
    )

    try:
        from remy.core.llm import get_llm
        llm = get_llm(settings.REVIEW_MODEL)
        result = llm.invoke(prompt)
        # Parse JSON from response (handle markdown-wrapped JSON)
        text = result.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = _json.loads(text)
        return {
            "safe": bool(data.get("safe", True)),
            "concerns": str(data.get("concerns", "")),
        }
    except Exception as e:
        logger.warning("Multi-model review failed (fail-open): %s", e)
        return None


# ============================================================
# Pending action record
# ============================================================

@dataclass
class PendingAction:
    """An action waiting for user approval."""
    action_id: str
    description: str
    action_fn: Callable[[], str]          # Callable returning JSON str result
    created_at: float = field(default_factory=time.time)
    timeout_sec: int = 120
    # Set by the queue when resolved
    _event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _approved: bool = False
    _resolved: bool = False


# ============================================================
# Approval Queue (singleton)
# ============================================================

class ApprovalQueue:
    """Async queue that holds actions pending human approval via Telegram."""

    def __init__(self):
        self._pending: dict[str, PendingAction] = {}   # action_id → PendingAction
        self._enabled: bool | None = None              # lazy-loaded from settings

    # ----------------------------------------------------------
    # Configuration
    # ----------------------------------------------------------

    @property
    def enabled(self) -> bool:
        if self._enabled is None:
            try:
                from remy.config.settings import settings
                self._enabled = getattr(settings, "APPROVAL_QUEUE_ENABLED", True)
            except Exception:
                self._enabled = True
        return self._enabled

    @property
    def timeout_sec(self) -> int:
        try:
            from remy.config.settings import settings
            return getattr(settings, "APPROVAL_TIMEOUT_SEC", 120)
        except Exception:
            return 120

    @property
    def _telegram_configured(self) -> bool:
        try:
            from remy.config.settings import settings
            return bool(settings.TELEGRAM_BOT_TOKEN and settings.PROACTIVE_CHAT_ID)
        except Exception:
            return False

    # ----------------------------------------------------------
    # URL / tag classification helpers
    # ----------------------------------------------------------

    @staticmethod
    def url_is_financial(url: str) -> bool:
        """Return True if the URL looks like a financial or crypto service."""
        if not url:
            return False
        url_lower = url.lower()
        return any(kw in url_lower for kw in _FINANCIAL_URL_KEYWORDS)

    @staticmethod
    def url_is_registration(url: str) -> bool:
        """Return True if the URL looks like a registration/identity flow."""
        if not url:
            return False
        url_lower = url.lower()
        return any(kw in url_lower for kw in _REGISTRATION_URL_KEYWORDS)

    @staticmethod
    def tags_are_financial(tags: list[str] | str) -> bool:
        """Return True if any tag is in the financial tags set."""
        if isinstance(tags, str):
            tag_set = {t.strip().lower() for t in tags.split(",") if t.strip()}
        else:
            tag_set = {t.strip().lower() for t in tags if t.strip()}
        return bool(tag_set & _FINANCIAL_TAGS)

    # ----------------------------------------------------------
    # Telegram helpers
    # ----------------------------------------------------------

    def _send_confirmation_request(self, action: PendingAction) -> None:
        """Fire-and-forget: send Telegram message asking for approval."""
        if not self._telegram_configured:
            logger.warning("Approval queue: Telegram not configured — auto-rejecting action '%s'", action.action_id[:8])
            return

        msg = (
            f"⚠️ *Підтвердження дії* (ID: `{action.action_id[:8]}`)\n\n"
            f"{action.description}\n\n"
            f"Відповідайте *Так* для підтвердження або *Ні* для скасування.\n"
            f"Тайм-аут: {action.timeout_sec} с."
        )

        async def _send():
            try:
                from remy.config.settings import settings
                from telegram import Bot
                bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
                await bot.send_message(
                    chat_id=settings.PROACTIVE_CHAT_ID,
                    text=msg,
                    parse_mode="Markdown",
                )
                logger.info("Approval request sent for action %s", action.action_id[:8])
            except Exception as e:
                logger.warning("Failed to send approval request: %s", e)

        # Schedule on the running event loop if available
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_send())
        except RuntimeError:
            # No running loop (sync context) — use asyncio.run in thread
            import threading
            threading.Thread(target=lambda: asyncio.run(_send()), daemon=True).start()

    def _send_outcome_notification(self, action_id: str, approved: bool, description: str) -> None:
        """Notify user of the final outcome (approved / rejected / timed out)."""
        if not self._telegram_configured:
            return

        status = "✅ Підтверджено" if approved else "❌ Скасовано"
        msg = f"{status} (ID: `{action_id[:8]}`)\n_{description[:120]}_"

        async def _send():
            try:
                from remy.config.settings import settings
                from telegram import Bot
                bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
                await bot.send_message(
                    chat_id=settings.PROACTIVE_CHAT_ID,
                    text=msg,
                    parse_mode="Markdown",
                )
            except Exception as e:
                logger.warning("Failed to send approval outcome: %s", e)

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_send())
        except RuntimeError:
            import threading
            threading.Thread(target=lambda: asyncio.run(_send()), daemon=True).start()

    # ----------------------------------------------------------
    # EventBus helpers (Web GUI notifications)
    # ----------------------------------------------------------

    def _emit_pending(self, action: "PendingAction") -> None:
        """Emit approval.pending event so the Web GUI shows a notification card."""
        try:
            from remy.core.event_bus import event_bus
            from remy.core.runtime_event_contract import build_runtime_event
            event_bus.emit("approval.pending", build_runtime_event(
                "approval.pending",
                event_domain="approval",
                payload={
                    "action_id": action.action_id,
                    "description": action.description,
                    "timeout_sec": action.timeout_sec,
                    "created_at": action.created_at,
                },
                legacy_fields={
                    "action_id": action.action_id,
                    "description": action.description,
                    "timeout_sec": action.timeout_sec,
                    "created_at": action.created_at,
                },
            ))
        except Exception as e:
            logger.debug("Could not emit approval.pending event: %s", e)

    def _emit_resolved(self, action: "PendingAction") -> None:
        """Emit approval.resolved event so the Web GUI removes the card."""
        try:
            from remy.core.event_bus import event_bus
            from remy.core.runtime_event_contract import build_runtime_event
            decision = "approved" if action._approved else "denied"
            routing_pressure = "routing pressure" in (action.description or "").lower()
            event_bus.emit("approval.resolved", build_runtime_event(
                "approval.resolved",
                event_domain="approval",
                payload={
                    "action_id": action.action_id,
                    "approved": action._approved,
                    "decision": decision,
                    "description": action.description,
                    "routing_pressure": routing_pressure,
                },
                legacy_fields={
                    "action_id": action.action_id,
                    "approved": action._approved,
                    "decision": decision,
                    "description": action.description,
                    "routing_pressure": routing_pressure,
                },
            ))
        except Exception as e:
            logger.debug("Could not emit approval.resolved event: %s", e)

        if action._approved:
            try:
                from remy.core.autonomy_goals import unblock_goal_by_action_id
                unblock_goal_by_action_id(action.action_id)
            except Exception as e:
                logger.debug("Could not unblock goal for action %s: %s", action.action_id[:8], e)

    # ----------------------------------------------------------
    # Core approval flow (async)
    # ----------------------------------------------------------

    async def request_approval(self, description: str, action_fn: Callable[[], str]) -> str:
        """
        Pause, ask user via Telegram, wait for reply, execute if approved.

        Returns the action result JSON string on approval, or a rejection JSON
        string on denial / timeout.
        """
        import json

        action = PendingAction(
            action_id=str(uuid.uuid4()),
            description=description,
            action_fn=action_fn,
            timeout_sec=self.timeout_sec,
        )
        self._pending[action.action_id] = action

        logger.info("Approval required for action %s: %s", action.action_id[:8], description[:80])
        self._send_confirmation_request(action)

        # Emit to event_bus so Web GUI can show a notification card
        self._emit_pending(action)

        try:
            await asyncio.wait_for(action._event.wait(), timeout=action.timeout_sec)
        except asyncio.TimeoutError:
            action._resolved = True
            action._approved = False
            logger.warning("Approval timed out for action %s", action.action_id[:8])
            self._send_outcome_notification(action.action_id, approved=False, description=description)

        self._pending.pop(action.action_id, None)

        # Emit resolved event so Web GUI removes the card
        self._emit_resolved(action)

        if action._approved:
            logger.info("Action %s approved — executing", action.action_id[:8])
            self._send_outcome_notification(action.action_id, approved=True, description=description)
            try:
                return action.action_fn()
            except Exception as e:
                logger.error("Approved action %s raised: %s", action.action_id[:8], e)
                return json.dumps({"error": f"Action failed after approval: {e}"})
        else:
            reason = "timed out" if not action._resolved else "rejected by user"
            logger.info("Action %s %s", action.action_id[:8], reason)
            return json.dumps({
                "error": f"Action was {reason} by user. Do not retry automatically.",
                "action_id": action.action_id[:8],
            })

    # ----------------------------------------------------------
    # Synchronous bridge (for use inside execute_tool)
    # ----------------------------------------------------------

    def request_approval_sync(
        self,
        description: str,
        action_fn: Callable[[], str],
        *,
        tool_name: str | None = None,
        tool_args: dict | None = None,
        url: str | None = None,
    ) -> str:
        """
        Synchronous wrapper around request_approval.

        If there is a running event loop (autonomous mode, combined runner),
        we schedule the coroutine and block the current thread with run_until_complete
        on a NEW loop — avoiding deadlock.  If no loop is running, we use asyncio.run().

        Optional tool_name/tool_args/url enable multi-model review: a second LLM
        reviews the action before presenting it to the human. Advisory only.
        """
        import json

        if not self.enabled:
            # Queue disabled — execute directly (useful in tests / dev)
            logger.debug("Approval queue disabled — executing action directly")
            return action_fn()

        if not self._telegram_configured:
            # No Telegram — log warning, but continue: Web GUI can still approve via WS/REST.
            logger.warning(
                "Approval queue: Telegram not configured — waiting for Web GUI approval: %s",
                description[:80],
            )

        # Multi-model review: ask a second LLM for a quick safety check
        if tool_name:
            review = review_action(tool_name, tool_args or {}, description, url)
            if review and not review["safe"]:
                description += f"\n\n⚠️ AI Review: {review['concerns']}"
            elif review and review["safe"]:
                description += "\n\n✅ AI Review: OK"
            # If review is None (error/disabled) — no annotation

        # Run the async approval flow in a dedicated thread-safe event loop
        import threading

        result_container: list[str] = []
        exception_container: list[Exception] = []

        def _run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                coro = self.request_approval(description, action_fn)
                result_container.append(new_loop.run_until_complete(coro))
            except Exception as exc:
                exception_container.append(exc)
            finally:
                new_loop.close()

        thread = threading.Thread(target=_run_in_thread, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout_sec + 10)  # extra buffer

        if exception_container:
            raise exception_container[0]

        if result_container:
            return result_container[0]

        # Thread timed out (shouldn't happen, but defensive)
        return json.dumps({"error": "Approval thread timed out unexpectedly."})

    # ----------------------------------------------------------
    # Reply handler (called from telegram_bot.handle_message)
    # ----------------------------------------------------------

    def handle_reply(self, text: str) -> bool:
        """
        Check if a Telegram reply resolves a pending approval.

        Returns True if the message was consumed as an approval/rejection reply.
        Returns False if it's a normal message (caller should process normally).
        """
        if not self._pending:
            return False

        text_lower = text.strip().lower()

        # Check confirm
        is_confirm = any(phrase in text_lower for phrase in _CONFIRM_PHRASES)
        is_reject = any(phrase in text_lower for phrase in _REJECT_PHRASES)

        if not (is_confirm or is_reject):
            return False

        # Resolve the oldest pending action (FIFO)
        oldest_id = next(iter(self._pending))
        action = self._pending.get(oldest_id)
        if action is None or action._resolved:
            return False

        action._resolved = True
        action._approved = is_confirm

        # Signal the waiting coroutine
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(action._event.set)
        except RuntimeError:
            # No running loop — set directly (happens in sync tests)
            action._event.set()

        logger.info(
            "Approval reply '%s' resolved action %s → %s",
            text[:20], action.action_id[:8], "APPROVED" if is_confirm else "REJECTED",
        )
        return True

    def resolve_by_id(self, action_id: str, approved: bool) -> bool:
        """
        REST-based resolution: find action by ID (full UUID or first-8 prefix), resolve it.
        Returns True if found and successfully resolved, False otherwise.
        Called by web API approve/reject endpoints.
        """
        target = None
        for aid, action in self._pending.items():
            if aid == action_id or aid.startswith(action_id):
                target = action
                break
        if target is None or target._resolved:
            return False

        target._resolved = True
        target._approved = approved

        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(target._event.set)
        except RuntimeError:
            target._event.set()

        logger.info(
            "Web GUI resolved action %s → %s",
            target.action_id[:8], "APPROVED" if approved else "REJECTED",
        )
        return True

    def pending_count(self) -> int:
        """Number of actions currently waiting for approval."""
        return len(self._pending)

    def snapshot_pending(self, limit: int | None = None) -> list[dict]:
        """Return a normalized snapshot of currently pending approval actions."""
        now = time.time()
        items: list[dict] = []
        for action in self._pending.values():
            if action._resolved:
                continue
            items.append(
                {
                    "id": action.action_id,
                    "action_id": action.action_id,
                    "description": action.description[:150],
                    "timeout_sec": action.timeout_sec,
                    "created_at": action.created_at,
                    "expires_at": action.created_at + action.timeout_sec,
                    "age_sec": int(max(0, now - action.created_at)),
                }
            )
            if limit is not None and len(items) >= limit:
                break
        return items

    def clear(self) -> None:
        """Reject all pending actions (used during shutdown or tests)."""
        import json
        for action in list(self._pending.values()):
            if not action._resolved:
                action._resolved = True
                action._approved = False
                try:
                    loop = asyncio.get_running_loop()
                    loop.call_soon_threadsafe(action._event.set)
                except RuntimeError:
                    action._event.set()
        self._pending.clear()


# ============================================================
# Module-level singleton
# ============================================================

approval_queue = ApprovalQueue()


# ============================================================
# Public helper: does this tool call need approval?
# ============================================================

def needs_approval(tool_name: str, args: dict, url: str | None = None, channel: str | None = None) -> bool:
    """
    Return True if this tool invocation requires human approval.

    Rules (checked in order — first match wins):
    1. Wallet/crypto transaction tools — ALWAYS require approval (no exceptions)
    2. Tools with money-movement args (amount, recipient, etc.)
    3. browser_act on a financial or registration URL
    4. browse_page on a financial URL
    5. store with financial tags

    Exception: Registration URLs skip approval when the channel is interactive
    (desktop, telegram, voice) because the user explicitly asked for the action.
    Financial URLs and wallet tools always require approval regardless of channel.
    """
    if not approval_queue.enabled:
        return False

    # ── Gate 1: Wallet/crypto tools — MANDATORY, no exceptions ──
    if tool_name in _WALLET_TOOLS:
        return True

    # ── Gate 2: Money-movement args in any tool ──
    if _has_money_args(tool_name, args):
        return True

    # Interactive channels: user is present and explicitly asked — skip registration approval
    _interactive_channels = {"desktop", "telegram", "voice", "proactive"}
    is_interactive = channel in _interactive_channels

    # ── Gate 3: browser_act on financial / registration URLs ──
    if tool_name == "browser_act":
        if url and approval_queue.url_is_financial(url):
            return True
        if url and approval_queue.url_is_registration(url) and not is_interactive:
            return True
        action_url = args.get("url", url or "")
        if action_url and approval_queue.url_is_financial(action_url):
            return True
        if action_url and approval_queue.url_is_registration(action_url) and not is_interactive:
            return True

    # ── Gate 4: browse_page on financial URLs ──
    if tool_name == "browse_page":
        nav_url = args.get("url", url or "")
        if nav_url and approval_queue.url_is_financial(nav_url):
            return True

    # ── Gate 5: store with financial tags ──
    if tool_name == "store":
        tags = args.get("tags", "")
        if approval_queue.tags_are_financial(tags):
            return True

    return False


def _has_money_args(tool_name: str, args: dict) -> bool:
    """Check if tool args contain money-movement parameters.

    Returns True if:
    - Any arg key matches _MONEY_ARG_KEYS AND has a non-empty value
    - Any arg value string contains sensitive action patterns (transfer, withdraw, etc.)
    """
    if not args:
        return False

    # Check for money-related argument keys with non-empty values
    for key in _MONEY_ARG_KEYS:
        val = args.get(key)
        if val is not None and val != "" and val != 0:
            return True

    # Check for sensitive action patterns in string arg values
    for key, val in args.items():
        if not isinstance(val, str):
            continue
        val_lower = val.lower()
        for pattern in _SENSITIVE_ACTION_PATTERNS:
            if pattern in val_lower:
                # Only flag if the tool could actually execute the action
                if tool_name in ("http_get", "http_post", "http_request", "http_poster",
                                 "browser_act", "cdp_agent_manager"):
                    return True

    return False


def build_approval_description(tool_name: str, args: dict, url: str | None = None) -> str:
    """Build a human-readable description of the action that needs approval."""
    # Wallet / crypto transaction tools — prominent warning
    if tool_name in _WALLET_TOOLS:
        amount = args.get("amount", args.get("value", args.get("transfer_amount", "?")))
        recipient = args.get("recipient", args.get("to_address", args.get("destination", "?")))
        return (
            f"💰 ТРАНЗАКЦІЯ: {tool_name}\n"
            f"Сума: {amount}\n"
            f"Отримувач: {recipient}\n"
            f"Аргументи: {str(args)[:300]}"
        )

    # Money-movement args in generic tools
    money_keys = [k for k in _MONEY_ARG_KEYS if args.get(k)]
    if money_keys:
        details = ", ".join(f"{k}={args[k]}" for k in money_keys)
        return (
            f"💰 Фінансова дія: {tool_name}\n"
            f"Параметри: {details}\n"
            f"Повні аргументи: {str(args)[:200]}"
        )

    if tool_name == "browser_act":
        action = args.get("action", "невідома дія")
        target_url = args.get("url", url or "невідомий URL")
        selector = args.get("selector", "")
        text = args.get("text", "")
        parts = [f"Браузер: {action}"]
        if selector:
            parts.append(f"елемент: `{selector}`")
        if text:
            parts.append(f"текст: `{text[:50]}`")
        parts.append(f"сторінка: {target_url}")
        return " | ".join(parts)

    if tool_name == "browse_page":
        nav_url = args.get("url", url or "")
        return f"Навігація браузера: {nav_url}"

    if tool_name == "store":
        tags = args.get("tags", "")
        content_preview = args.get("content", "")[:100]
        return f"Збереження фінансових даних\nТеги: {tags}\nВміст: {content_preview}…"

    return f"Виклик інструменту: {tool_name}\nАргументи: {str(args)[:200]}"
