"""
Telegram Bot — text interface to Remy with LangGraph agent + brain tools.

Same brain, same tools, same personality as the voice/web interface.
Uses LangGraph StateGraph for conversation with function calling.

Usage:
    remy --telegram
"""

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from google import genai
from google.genai import types
from telegram import BotCommand, Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from remy.config.settings import settings
from remy.core.agent_tools import brain
from remy.core.brain_tools import (
    generate_session_summary,
    get_registry,
)
from remy.core.agent import invoke_agent

logger = logging.getLogger("TelegramBot")

SESSION_TIMEOUT_SEC = 1800  # 30 minutes


@dataclass
class ChatSession:
    """Per-chat conversation state."""

    session_id: str
    history: list = field(default_factory=list)
    session_log: list = field(default_factory=list)
    last_activity: float = field(default_factory=time.time)


class TelegramBot:
    """Telegram bot with LangGraph agent and brain tools."""

    def __init__(self):
        api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env or environment")

        token = settings.TELEGRAM_BOT_TOKEN or os.environ.get("TELEGRAM_BOT_TOKEN")
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN not set in .env or environment")

        # Client kept only for session summary generation
        self.client = genai.Client(api_key=api_key)
        self.token = token
        self._sessions: dict[int, ChatSession] = {}
        self._allowed_chat_ids: set[int] = set(settings.TELEGRAM_ALLOWED_CHAT_IDS)

    def _is_authorized(self, chat_id: int) -> bool:
        """Check if a chat ID is authorized. Empty whitelist = allow all (open mode)."""
        if not self._allowed_chat_ids:
            return True
        return chat_id in self._allowed_chat_ids

    def _authorization_help_text(self, chat_id: int) -> str:
        """Explain how to authorize a Telegram chat safely."""
        if not self._allowed_chat_ids:
            return (
                "Telegram is currently in open mode.\n"
                f"Your chat ID: `{chat_id}`\n"
                "For safer operator access, add this ID to `TELEGRAM_ALLOWED_CHAT_IDS` in `.env`."
            )
        return (
            "Access denied.\n"
            f"Your chat ID: `{chat_id}`\n"
            "Ask the operator to add this ID to `TELEGRAM_ALLOWED_CHAT_IDS` in `.env`, "
            "then restart Remy or reload runtime settings."
        )

    def _operator_help_text(self) -> str:
        """Return a compact operator command reference."""
        return (
            "*Remy Telegram Operator Mode*\n"
            "`/whoami` — show your chat ID and authorization status\n"
            "`/ops` — compact operator digest\n"
            "`/status` — channel and autonomy runtime health\n"
            "`/approvals` — list pending approval requests\n"
            "`/goals` — active autonomous goals\n"
            "`/budget` — wallet and LLM spend summary\n"
            "`/approve [id]` — approve the oldest or specific pending action\n"
            "`/deny [id]` — deny the oldest or specific pending action\n"
            "`/pause` — pause autonomy loop\n"
            "`/resume` — resume autonomy loop\n"
            "`/new` — start a fresh conversation session\n"
            "`/stats` — memory statistics\n"
            "\n"
            "Primary remote surface: Telegram operator mode."
        )

    def _telegram_commands(self) -> list[BotCommand]:
        """Native Telegram command menu for operator mode."""
        return [
            BotCommand("help", "Operator commands"),
            BotCommand("whoami", "Show chat ID and access status"),
            BotCommand("ops", "Compact operator digest"),
            BotCommand("status", "Runtime channel health"),
            BotCommand("approvals", "List pending approvals"),
            BotCommand("goals", "Active autonomous goals"),
            BotCommand("budget", "Budget and LLM spend"),
            BotCommand("approve", "Approve pending action"),
            BotCommand("deny", "Deny pending action"),
            BotCommand("pause", "Pause autonomy loop"),
            BotCommand("resume", "Resume autonomy loop"),
            BotCommand("new", "Start a fresh session"),
            BotCommand("stats", "Memory statistics"),
        ]

    async def configure_app(self, app) -> None:
        """Configure Telegram app metadata after initialization."""
        try:
            await app.bot.set_my_commands(self._telegram_commands())
        except Exception as e:
            logger.warning("Failed to set Telegram command menu: %s", e)

    async def _build_operator_digest(self) -> str:
        """Build a compact remote-operator digest for Telegram."""
        lines = ["*Remy Ops Digest*"]

        try:
            from remy.core.gateway import get_registry as get_channel_registry

            summary = get_channel_registry().summary()
            lines.append(f"Runtime: {summary.get('health', 'unknown')}")
        except Exception as e:
            lines.append(f"Runtime: error ({e})")

        try:
            from remy.core.combined_runner import get_operator_console_snapshot

            snapshot = await asyncio.to_thread(get_operator_console_snapshot, goal_limit=3, approval_limit=5)
            autonomy = snapshot.get("autonomy", {})
            lines.append(
                f"Autonomy: {'running' if autonomy.get('running') else 'stopped'} ({autonomy.get('version', 'v2')})"
            )
            approvals = snapshot.get("approvals", {})
            lines.append(f"Approvals: {approvals.get('pending_count', 0)} pending")
            goals = snapshot.get("goals", {})
            lines.append(f"Goals: {goals.get('active', 0)} active, {goals.get('blocked', 0)} blocked")
            budget = snapshot.get("budget", {})
            cost = budget.get("llm_cost_today")
            status = budget.get("alert_level", "unknown")
            cost_str = f"${cost:.4f}" if isinstance(cost, (int, float)) else "—"
            lines.append(f"Budget: {status}, LLM today {cost_str}")
            factuality = snapshot.get("factuality", {})
            unsupported = factuality.get("unsupported_observed_claims_total")
            if unsupported is not None:
                lines.append(f"Factuality: {int(unsupported)} unsupported observed claims")
            quality_debt = snapshot.get("quality_debt_by_specialist", []) or []
            if quality_debt:
                top = quality_debt[0]
                lines.append(
                    f"Quality debt: {top.get('id', 'unknown')} ({float(top.get('quality_debt', 0.0)):.2f})"
                )
            routing = snapshot.get("routing_pressure", {}) or {}
            preferred = routing.get("top_candidate") or {}
            degraded = routing.get("highest_pressure") or {}
            if preferred:
                lines.append(
                    f"Routing prefer: {preferred.get('id', 'unknown')} ({float(preferred.get('quality_adjusted_success_rate', 0.0)):.2f})"
                )
            if degraded:
                lines.append(
                    f"Routing avoid: {degraded.get('id', 'unknown')} ({float(degraded.get('quality_debt', 0.0)):.2f} debt)"
                )
        except Exception as e:
            lines.append(f"Ops snapshot: error ({e})")

        lines.append("")
        lines.append("Use `/status`, `/approvals`, `/goals`, `/budget` for detail.")
        return "\n".join(lines)

    def _get_or_create_session(self, chat_id: int) -> ChatSession:
        """Get existing session or create a new one. Auto-closes stale sessions."""
        now = time.time()

        if chat_id in self._sessions:
            session = self._sessions[chat_id]
            if now - session.last_activity > SESSION_TIMEOUT_SEC:
                # Stale session — close it and create a new one
                self._close_session_sync(chat_id)
            else:
                session.last_activity = now
                return session

        session = ChatSession(
            session_id=str(uuid.uuid4()),
            last_activity=now,
        )
        self._sessions[chat_id] = session
        logger.info(f"New session for chat {chat_id}: {session.session_id[:8]}...")
        return session

    def gc_stale_sessions(self):
        """Remove all sessions inactive for longer than SESSION_TIMEOUT_SEC.

        Call periodically in long-running server deployments to prevent
        unbounded memory growth from abandoned sessions.
        """
        now = time.time()
        stale = [
            cid for cid, s in self._sessions.items()
            if now - s.last_activity > SESSION_TIMEOUT_SEC
        ]
        for cid in stale:
            self._close_session_sync(cid)
        if stale:
            logger.info("GC: cleaned %d stale session(s)", len(stale))

    def _close_session_sync(self, chat_id: int):
        """Close a session: end brain session + remove from cache."""
        session = self._sessions.pop(chat_id, None)
        if not session:
            return

        try:
            from remy.core.agent_tools import brain_lock
            with brain_lock:
                brain.end_session(session.session_id)
        except Exception as e:
            logger.warning(f"end_session failed: {e}")

        logger.info(f"Session closed for chat {chat_id}: {session.session_id[:8]}...")

    async def _close_session(self, chat_id: int):
        """Close a session: generate summary, end brain session, remove from cache."""
        session = self._sessions.get(chat_id)
        if not session:
            return

        # Generate session summary before closing
        try:
            await generate_session_summary(
                self.client, session.session_log, session.session_id
            )
        except Exception as e:
            logger.warning(f"Session summary failed: {e}")

        self._sessions.pop(chat_id, None)

        try:
            from remy.core.agent_tools import brain_lock

            def _end_session_locked():
                with brain_lock:
                    brain.end_session(session.session_id)

            await asyncio.to_thread(_end_session_locked)
        except Exception as e:
            logger.warning(f"end_session failed: {e}")

        logger.info(f"Session closed for chat {chat_id}: {session.session_id[:8]}...")

    async def _gemini_respond(self, chat_id: int, user_message: str) -> str:
        """Send user message through LangGraph agent."""
        from langchain_core.messages import HumanMessage
        from remy.core.logging_config import log_context

        session = self._get_or_create_session(chat_id)

        with log_context(session_id=session.session_id, channel="telegram"):
            # Log user text (if string) or placeholder
            if isinstance(user_message, str):
                session.session_log.append({"type": "user_text", "text": user_message[:200]})
            else:
                session.session_log.append({"type": "user_voice", "text": "[Audio Message]"})

            response_text, new_history, new_log = await invoke_agent(
                user_message=user_message,
                session_id=session.session_id,
                channel="telegram",
                session_log=session.session_log,
                history=session.history,
            )

            session.history = new_history
            session.session_log = new_log
            return response_text

    # ============== TELEGRAM HANDLERS ==============

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start — greeting."""
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            logger.warning("Unauthorized /start from chat_id=%d", chat_id)
            await update.message.reply_text(self._authorization_help_text(chat_id), parse_mode="Markdown")
            return
        session = self._get_or_create_session(chat_id)

        # Try to get brain context for a personalized greeting
        greeting = await self._gemini_respond(
            chat_id, "The user just started a new conversation. Greet them warmly."
        )
        await update.message.reply_text(greeting)

    async def new_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /new — reset session, generate summary, start fresh."""
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            await update.message.reply_text(self._authorization_help_text(chat_id), parse_mode="Markdown")
            return
        await self._close_session(chat_id)
        session = self._get_or_create_session(chat_id)
        await update.message.reply_text(
            "New session started. Previous conversation has been summarized and saved."
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help — show operator command reference."""
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            await update.message.reply_text(self._authorization_help_text(chat_id), parse_mode="Markdown")
            return
        await update.message.reply_text(self._operator_help_text(), parse_mode="Markdown")

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats — brain statistics."""
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            await update.message.reply_text(self._authorization_help_text(chat_id), parse_mode="Markdown")
            return
        try:
            count = await asyncio.to_thread(brain.count)
            stats = await asyncio.to_thread(brain.stats)
            text = f"Brain: {count} records\n"
            if isinstance(stats, dict):
                for key, val in stats.items():
                    text += f"  {key}: {val}\n"
            await update.message.reply_text(text)
        except Exception as e:
            await update.message.reply_text(f"Error fetching stats: {e}")

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status — runtime channel health from gateway registry."""
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            await update.message.reply_text(self._authorization_help_text(chat_id), parse_mode="Markdown")
            return
        try:
            from remy.core.combined_runner import get_channel_status_snapshot

            channel_snapshot = await asyncio.to_thread(get_channel_status_snapshot)
            summary = channel_snapshot.get("channels", {}).get("registry_summary", {})
            all_ch = {
                "web": channel_snapshot.get("channels", {}).get("web", {}).get("health") or {},
                "telegram": channel_snapshot.get("channels", {}).get("telegram", {}).get("health") or {},
                "autonomy": channel_snapshot.get("channels", {}).get("autonomy", {}).get("health") or {},
            }

            lines = [f"*Status* — {summary.get('health', '?').upper()}"]
            for name, ch in all_ch.items():
                st = ch.get("status", "?")
                uptime = ch.get("uptime_sec")
                uptime_str = f" ({int(uptime)}s up)" if uptime else ""
                err = ch.get("error")
                err_str = f"\n  ⚠ {err}" if err else ""
                lines.append(f"• {name}: {st}{uptime_str}{err_str}")

            control = channel_snapshot.get("control", {})
            if control.get("runtime_loaded") or control.get("running"):
                lines.append(
                    f"\nAutonomy loop: {'running' if control.get('running') else 'stopped'} ({control.get('active_version', 'v2')})"
                )
                sid = control.get("session_id")
                if sid is not None:
                    lines.append(f"Session: {str(sid)[:8]}...")
                if control.get("maintenance_only"):
                    lines.append("Mode: maintenance only")
            else:
                lines.append(
                    f"\nAutonomy configured: {control.get('configured_version', 'v2')} (inactive)"
                )

            await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def whoami_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /whoami вЂ” show chat ID and authorization guidance."""
        chat_id = update.effective_chat.id
        status = "authorized" if self._is_authorized(chat_id) else "not authorized"
        lines = [
            f"Chat ID: `{chat_id}`",
            f"Status: *{status}*",
        ]
        if self._allowed_chat_ids:
            allowed = ", ".join(str(cid) for cid in sorted(self._allowed_chat_ids))
            lines.append(f"Allowlist: `{allowed}`")
        else:
            lines.append("Allowlist: open mode")
        lines.append("")
        lines.append(self._authorization_help_text(chat_id))
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def ops_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /ops — compact operator digest."""
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            await update.message.reply_text(self._authorization_help_text(chat_id), parse_mode="Markdown")
            return
        try:
            await update.message.reply_text(await self._build_operator_digest(), parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def approvals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /approvals — list pending approval requests for remote operators."""
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            await update.message.reply_text(self._authorization_help_text(chat_id), parse_mode="Markdown")
            return
        try:
            from remy.core.combined_runner import get_approval_runtime_snapshot

            approvals = await asyncio.to_thread(get_approval_runtime_snapshot, goal_limit=3, approval_limit=10)
            pending = list(approvals.get("pending", []) or [])
            pending = [
                SimpleNamespace(
                    action_id=str(action.get("action_id") or action.get("id") or ""),
                    created_at=time.time() - int(action.get("age_sec") or 0),
                    description=str(action.get("description") or ""),
                    specialist=str(action.get("specialist") or ""),
                    routing_pressure=bool(action.get("routing_pressure")),
                    target=str((action.get("context") or {}).get("target") or ""),
                )
                for action in pending
            ]
            if not pending:
                await update.message.reply_text("No pending approvals.")
                return

            now = time.time()
            lines = [f"*Pending approvals* — {len(pending)}"]
            for action in pending[:10]:
                age_sec = int(now - action.created_at)
                meta = []
                if action.specialist:
                    meta.append(f"specialist {action.specialist}")
                if action.routing_pressure:
                    meta.append("routing pressure")
                if action.target:
                    meta.append(action.target[:80])
                lines.append(
                    f"\n• `{action.action_id[:8]}` — {age_sec}s old\n"
                    f"  {action.description[:120]}"
                )
                if meta:
                    lines.append(f"  _{' • '.join(meta)}_")
            if len(pending) > 10:
                lines.append(f"\n…and {len(pending) - 10} more")
            lines.append("\nUse `/approve <id>` or `/deny <id>`.")
            await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def goals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /goals — list active autonomous goals."""
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            await update.message.reply_text(self._authorization_help_text(chat_id), parse_mode="Markdown")
            return
        try:
            from remy.core.combined_runner import get_goal_runtime_snapshot

            goals = await asyncio.to_thread(get_goal_runtime_snapshot, goal_limit=5, approval_limit=5)
            active = goals.get("active_list", []) or []

            lines = [f"*Goals* — {goals.get('total', 0)} total"]
            if active:
                lines.append(f"\n🟢 Active ({goals.get('active', 0)}):")
                for g in active[:5]:
                    pri = g.get("priority", "medium")
                    lines.append(f"  [{pri}] {str(g.get('content', ''))[:60]}")
            if goals.get("blocked", 0):
                lines.append(f"\n🔴 Blocked: {goals.get('blocked', 0)}")
            pending_count = max(
                int(goals.get("total", 0) or 0)
                - int(goals.get("active", 0) or 0)
                - int(goals.get("blocked", 0) or 0),
                0,
            )
            if pending_count:
                lines.append(f"\n⏳ Pending: {pending_count}")

            await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def budget_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /budget — wallet balance and LLM spend."""
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            await update.message.reply_text(self._authorization_help_text(chat_id), parse_mode="Markdown")
            return
        try:
            from remy.core.combined_runner import get_budget_runtime_snapshot

            budget = await asyncio.to_thread(get_budget_runtime_snapshot, goal_limit=3, approval_limit=5)
            bal = budget.get("balance_usd")
            runway = budget.get("runway_days")
            cost = budget.get("llm_cost_today")
            level = budget.get("alert_level", "unknown")

            bal_str = f"${bal:.3f}" if bal is not None else "—"
            runway_str = f"{runway} days" if runway is not None else "—"
            cost_str = f"${cost:.4f}" if cost is not None else "—"

            text = (
                f"*Budget*\n"
                f"Balance: {bal_str}\n"
                f"Runway: {runway_str}\n"
                f"LLM today: {cost_str}\n"
                f"Status: {level}"
            )
            await update.message.reply_text(text, parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def approve_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /approve [id] — approve a pending action."""
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            await update.message.reply_text(self._authorization_help_text(chat_id), parse_mode="Markdown")
            return
        try:
            from remy.core.combined_runner import get_approval_runtime_snapshot, resolve_operator_approval

            approvals = await asyncio.to_thread(get_approval_runtime_snapshot, goal_limit=3, approval_limit=10)
            pending = list(approvals.get("pending", []) or [])
            if not pending:
                await update.message.reply_text("No pending approvals.")
                return

            args = context.args
            if args:
                action_id = args[0]
                chosen = next(
                    (item for item in pending if str(item.get("action_id") or item.get("id") or "") == action_id),
                    None,
                )
                result = await asyncio.to_thread(resolve_operator_approval, action_id, approved=True, decided_by="telegram")
                if result.get("ok"):
                    await update.message.reply_text(self._format_approval_resolution_reply("Approved", chosen, str(result.get("action_id") or action_id)), parse_mode="Markdown")
                else:
                    await update.message.reply_text(f"Not found: {action_id}")
            else:
                # Approve the oldest pending action
                oldest = pending[0]
                oldest_id = str(oldest.get("action_id") or oldest.get("id") or "")
                result = await asyncio.to_thread(resolve_operator_approval, oldest_id, approved=True, decided_by="telegram")
                if result.get("ok"):
                    await update.message.reply_text(self._format_approval_resolution_reply("Approved", oldest, str(result.get("action_id") or oldest_id)), parse_mode="Markdown")
                else:
                    await update.message.reply_text("Failed to approve.")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def deny_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /deny [id] — deny a pending action."""
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            await update.message.reply_text(self._authorization_help_text(chat_id), parse_mode="Markdown")
            return
        try:
            from remy.core.combined_runner import get_approval_runtime_snapshot, resolve_operator_approval

            approvals = await asyncio.to_thread(get_approval_runtime_snapshot, goal_limit=3, approval_limit=10)
            pending = list(approvals.get("pending", []) or [])
            if not pending:
                await update.message.reply_text("No pending approvals.")
                return

            args = context.args
            if args:
                action_id = args[0]
                chosen = next(
                    (item for item in pending if str(item.get("action_id") or item.get("id") or "") == action_id),
                    None,
                )
                result = await asyncio.to_thread(resolve_operator_approval, action_id, approved=False, decided_by="telegram")
                if result.get("ok"):
                    await update.message.reply_text(self._format_approval_resolution_reply("Denied", chosen, str(result.get("action_id") or action_id)), parse_mode="Markdown")
                else:
                    await update.message.reply_text(f"Not found: {action_id}")
            else:
                oldest = pending[0]
                oldest_id = str(oldest.get("action_id") or oldest.get("id") or "")
                result = await asyncio.to_thread(resolve_operator_approval, oldest_id, approved=False, decided_by="telegram")
                if result.get("ok"):
                    await update.message.reply_text(self._format_approval_resolution_reply("Denied", oldest, str(result.get("action_id") or oldest_id)), parse_mode="Markdown")
                else:
                    await update.message.reply_text("Failed to deny.")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    def _format_approval_resolution_reply(self, decision: str, item: dict | None, fallback_id: str) -> str:
        prefix = "✅" if decision == "Approved" else "🚫"
        if not item:
            return f"{prefix} {decision}: {fallback_id}"
        details = []
        specialist = str(item.get("specialist") or "")
        if specialist:
            details.append(f"specialist {specialist}")
        if item.get("routing_pressure"):
            details.append("routing pressure")
        target = str((item.get("context") or {}).get("target") or "")
        if target:
            details.append(target[:80])
        body = str(item.get("description") or item.get("action") or fallback_id)[:120]
        if not details:
            return f"{prefix} {decision}: {body}"
        return f"{prefix} {decision}: {body}\n_{' • '.join(details)}_"

    async def pause_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pause — stop the autonomy loop."""
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            await update.message.reply_text(self._authorization_help_text(chat_id), parse_mode="Markdown")
            return
        try:
            from remy.core.combined_runner import stop_autonomy
            await stop_autonomy()
            await update.message.reply_text("⏸ Autonomy loop paused.")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def resume_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /resume — restart the autonomy loop."""
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            return
        try:
            from remy.core.combined_runner import start_autonomy
            await start_autonomy()
            await update.message.reply_text("▶️ Autonomy loop resumed.")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    _GC_INTERVAL = 100  # Run GC every N messages

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle any text message — main conversation loop."""
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            logger.warning("Unauthorized message from chat_id=%d", chat_id)
            await update.message.reply_text(self._authorization_help_text(chat_id), parse_mode="Markdown")
            return

        user_text = update.message.text

        if not user_text:
            return

        # Periodic GC of stale sessions
        self._gc_counter = getattr(self, "_gc_counter", 0) + 1
        if self._gc_counter >= self._GC_INTERVAL:
            self._gc_counter = 0
            self.gc_stale_sessions()

        # Human-in-the-loop: check if this message is an approval/rejection reply
        from remy.core.combined_runner import resolve_operator_approval_reply, resolve_operator_guidance_reply
        reply_resolution = resolve_operator_approval_reply(user_text, decided_by="telegram-reply")
        if reply_resolution.get("consumed"):
            # Message consumed as approval reply — acknowledge and stop
            await update.message.reply_text("✅ Відповідь отримана. Дія буде виконана/скасована.")
            return

        guidance_resolution = resolve_operator_guidance_reply(user_text)
        if guidance_resolution.get("consumed"):
            await update.message.reply_text("вњ… Р’С–РґРїРѕРІС–РґСЊ РѕС‚СЂРёРјР°РЅР°. РђРіРµРЅС‚ РїСЂРѕРґРѕРІР¶РёС‚СЊ РІРёРєРѕРЅР°РЅРЅСЏ.")
            return

        # Typing indicator
        await update.effective_chat.send_action("typing")

        response_text = await self._gemini_respond(chat_id, user_text)

        if not response_text:
            response_text = "I didn't have anything to say. Could you rephrase?"

        # Send generated image if present in response
        import re
        image_match = re.search(r'/api/generated_images/([\w.]+)', response_text)
        if image_match:
            filename = image_match.group(1)
            image_path = Path(settings.DATA_DIR) / "generated_images" / filename
            if image_path.exists():
                # Strip the image URL from text for cleaner caption
                caption = re.sub(r'!\[[^\]]*\]\(/api/generated_images/[\w.]+\)', '', response_text).strip()
                try:
                    with open(image_path, 'rb') as photo:
                        await update.message.reply_photo(
                            photo=photo,
                            caption=caption[:1024] if caption else None,
                            parse_mode="Markdown" if caption else None,
                        )
                    return
                except Exception as e:
                    logger.warning("Failed to send photo via Telegram: %s", e)

        # Send browser screenshot if present in response
        screenshot_match = re.search(r'/api/browser_screenshots/([\w.]+\.png)', response_text)
        if screenshot_match:
            filename = screenshot_match.group(1)
            ss_path = Path(settings.DATA_DIR) / "browser_screenshots" / filename
            if ss_path.exists():
                caption = re.sub(r'!\[[^\]]*\]\(/api/browser_screenshots/[\w.]+\.png\)', '', response_text).strip()
                try:
                    with open(ss_path, 'rb') as photo:
                        await update.message.reply_photo(
                            photo=photo,
                            caption=caption[:1024] if caption else None,
                        )
                    return
                except Exception as e:
                    logger.warning("Failed to send browser screenshot via Telegram: %s", e)

        # Send generated PDF report if present in response
        report_match = re.search(r'/api/reports/([^\s\)]+\.pdf)', response_text)
        if report_match:
            from urllib.parse import unquote
            filename = unquote(report_match.group(1))
            report_path = Path(settings.DATA_DIR) / "reports" / filename
            if report_path.exists():
                caption = re.sub(r'\[[^\]]*\]\(/api/reports/[^\)]+\.pdf\)', '', response_text).strip()
                try:
                    with open(report_path, 'rb') as doc:
                        await update.message.reply_document(
                            document=doc,
                            filename=filename,
                            caption=caption[:1024] if caption else None,
                            parse_mode="Markdown" if caption else None,
                        )
                    return
                except Exception as e:
                    logger.warning("Failed to send report via Telegram: %s", e)

        # Send generated PPTX presentation if present in response
        pptx_match = re.search(r'/api/presentations/([^\s\)]+\.pptx)', response_text)
        if pptx_match:
            from urllib.parse import unquote
            filename = unquote(pptx_match.group(1))
            pptx_path = Path(settings.DATA_DIR) / "presentations" / filename
            if pptx_path.exists():
                caption = re.sub(r'\[[^\]]*\]\(/api/presentations/[^\)]+\.pptx\)', '', response_text).strip()
                try:
                    with open(pptx_path, 'rb') as doc:
                        await update.message.reply_document(
                            document=doc,
                            filename=filename,
                            caption=caption[:1024] if caption else None,
                            parse_mode="Markdown" if caption else None,
                        )
                    return
                except Exception as e:
                    logger.warning("Failed to send presentation via Telegram: %s", e)

        # Send response, falling back to plain text if markdown fails
        try:
            await update.message.reply_text(response_text, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(response_text)


    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle voice messages — transcribing via Gemini."""
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            await update.message.reply_text(self._authorization_help_text(chat_id), parse_mode="Markdown")
            return
        voice = update.message.voice

        if not voice:
            return

        # 1. Download voice file (OGG)
        try:
            file_info = await context.bot.get_file(voice.file_id)
            file_bytes = await file_info.download_as_bytearray()
        except Exception as e:
            logger.error(f"Failed to download voice: {e}")
            await update.message.reply_text("Failed to download voice message.")
            return

        # 2. Transcribe via Gemini (multimodal)
        # Instead of manual transcription, we send audio directly to agent (LangGraph)
        # This matches web/desktop behavior for consistency.
        try:
            from langchain_core.messages import HumanMessage
            import base64

            # Convert to base64 for LangChain (it handles it internally for Google GenAI)
            # Actually LangChain Google GenAI expects image_url or media blocks
            # For audio, we use a "media" block with base64 data
            
            # The agent.py invoke_agent expects HumanMessage content to be a list for multimodal
            
            b64_data = base64.b64encode(file_bytes).decode("utf-8")
            
            content_parts = [
                {
                    "type": "text", 
                    "text": "The user sent a voice message. Listen to it carefully and respond naturally. "
                            "Do NOT transcribe it unless asked."
                },
                {
                    "type": "media",
                    "mime_type": "audio/ogg",
                    "data": b64_data
                }
            ]

            user_msg = HumanMessage(content=content_parts)

            # 3. Process through agent
            await update.effective_chat.send_action("typing")
            
            # We need to update _gemini_respond to accept HumanMessage
            # It currently takes str. Let's update it or call invoke_agent directly here.
            # Updating _gemini_respond is better for DRY.
            
            response_text = await self._gemini_respond(chat_id, user_msg)

            if not response_text:
                response_text = "I didn't have anything to say."

            try:
                await update.message.reply_text(response_text, parse_mode="Markdown")
            except Exception:
                await update.message.reply_text(response_text)

        except Exception as e:
            logger.error(f"Voice processing error: {e}")
            await update.message.reply_text(f"Error processing voice: {e}")

    def register_handlers(self, app):
        """Register all message handlers on the given Application."""
        app.add_handler(CommandHandler("start", self.start_command))
        app.add_handler(CommandHandler("help", self.help_command))
        app.add_handler(CommandHandler("new", self.new_command))
        app.add_handler(CommandHandler("stats", self.stats_command))
        app.add_handler(CommandHandler("ops", self.ops_command))
        app.add_handler(CommandHandler("status", self.status_command))
        app.add_handler(CommandHandler("whoami", self.whoami_command))
        app.add_handler(CommandHandler("approvals", self.approvals_command))
        app.add_handler(CommandHandler("goals", self.goals_command))
        app.add_handler(CommandHandler("budget", self.budget_command))
        app.add_handler(CommandHandler("approve", self.approve_command))
        app.add_handler(CommandHandler("deny", self.deny_command))
        app.add_handler(CommandHandler("pause", self.pause_command))
        app.add_handler(CommandHandler("resume", self.resume_command))
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )
        app.add_handler(
            MessageHandler(filters.VOICE, self.handle_voice)
        )

    def run(self):
        """Start the bot with long-polling."""
        app = ApplicationBuilder().token(self.token).build()
        self.register_handlers(app)
        app.post_init = self.configure_app

        from remy.core.agent_tools import brain_lock
        with brain_lock:
            brain_count = brain.count()
        registry = get_registry()
        tool_count = len(registry.get_all_declarations())

        print("=" * 50)
        print("REMY — TELEGRAM BOT")
        print(f"Model: {settings.SUMMARY_MODEL}")
        print(f"Brain: {settings.AURA_BRAIN_PATH} ({brain_count} records)")
        print(f"Tools: {tool_count}")
        print(f"Session timeout: {SESSION_TIMEOUT_SEC}s")
        print("Bot is running. Press Ctrl+C to stop.")
        print("=" * 50)

        logger.info("Starting Telegram bot...")
        app.run_polling()
