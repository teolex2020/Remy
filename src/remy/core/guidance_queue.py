"""
Guidance Queue (AUTON-3) — interactive escalation for autonomous mode.

When the agent is stuck, it can ask the user a free-text question via
Telegram or Web GUI. The autonomous cycle pauses until the user responds
or the timeout expires.

Architecture mirrors approval_queue.py:
- PendingGuidanceRequest dataclass with asyncio.Event
- Telegram: sends question, any reply text is the answer
- Web GUI: REST endpoint + WebSocket events
- Sync bridge for tool execution context
"""

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger("Autonomy.Guidance")


@dataclass
class PendingGuidanceRequest:
    """A question waiting for user response."""

    request_id: str
    question: str
    context: str = ""  # Goal, attempts, last error — for user context
    created_at: float = field(default_factory=time.time)
    timeout_sec: int = 120
    # Set by the queue when resolved
    _event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _answer: str | None = None
    _resolved: bool = False


class GuidanceQueue:
    """Async queue that holds questions pending user response via Telegram/Web."""

    def __init__(self):
        self._pending: dict[str, PendingGuidanceRequest] = {}
        self._enabled: bool | None = None

    @property
    def enabled(self) -> bool:
        if self._enabled is None:
            try:
                from remy.config.settings import settings

                self._enabled = getattr(settings, "GUIDANCE_QUEUE_ENABLED", True)
            except Exception:
                self._enabled = True
        return self._enabled

    @property
    def timeout_sec(self) -> int:
        try:
            from remy.config.settings import settings

            return getattr(settings, "GUIDANCE_TIMEOUT_SEC", 120)
        except Exception:
            return 120

    @property
    def _telegram_configured(self) -> bool:
        try:
            from remy.config.settings import settings

            return bool(settings.TELEGRAM_BOT_TOKEN and settings.PROACTIVE_CHAT_ID)
        except Exception:
            return False

    def pending_count(self) -> int:
        return sum(1 for r in self._pending.values() if not r._resolved)

    def snapshot_pending(self, limit: int | None = None) -> list[dict]:
        """Return a normalized snapshot of unresolved guidance requests."""
        items: list[dict] = []
        for req in self._pending.values():
            if req._resolved:
                continue
            items.append(
                {
                    "request_id": req.request_id,
                    "question": req.question,
                    "context": req.context,
                    "timeout_sec": req.timeout_sec,
                    "created_at": req.created_at,
                    "expires_at": req.created_at + req.timeout_sec,
                }
            )
            if limit is not None and len(items) >= limit:
                break
        return items

    # ============== Core async flow ==============

    async def request_guidance(self, question: str, context: str = "") -> str | None:
        """Ask user a question. Returns answer text or None on timeout."""
        req = PendingGuidanceRequest(
            request_id=str(uuid.uuid4()),
            question=question,
            context=context,
            timeout_sec=self.timeout_sec,
        )
        self._pending[req.request_id] = req

        logger.info(
            "Guidance requested %s: %s",
            req.request_id[:8],
            question[:80],
        )
        self._send_guidance_telegram(req)
        self._emit_pending(req)

        try:
            await asyncio.wait_for(req._event.wait(), timeout=req.timeout_sec)
        except asyncio.TimeoutError:
            req._resolved = True
            req._answer = None
            logger.warning("Guidance timed out for %s", req.request_id[:8])

        self._pending.pop(req.request_id, None)
        self._emit_resolved(req)

        if req._answer is not None:
            logger.info(
                "Guidance %s answered: %s",
                req.request_id[:8],
                req._answer[:50],
            )
        return req._answer

    def request_guidance_sync(self, question: str, context: str = "") -> str | None:
        """Sync bridge — runs async guidance in a daemon thread with its own event loop."""
        if not self.enabled:
            logger.debug("Guidance queue disabled — returning None")
            return None

        result_container: list[str | None] = []
        exception_container: list[Exception] = []

        def _run():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                answer = new_loop.run_until_complete(
                    self.request_guidance(question, context),
                )
                result_container.append(answer)
            except Exception as exc:
                exception_container.append(exc)
            finally:
                new_loop.close()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout_sec + 10)

        if exception_container:
            logger.warning("Guidance thread raised: %s", exception_container[0])
            return None

        return result_container[0] if result_container else None

    # ============== Reply handling ==============

    def handle_reply(self, text: str) -> bool:
        """Check if a Telegram message resolves a pending guidance request.

        Any non-empty text is accepted as the answer. Returns True if consumed.
        """
        if not self._pending:
            return False

        text = text.strip()
        if not text:
            return False

        # Resolve the oldest pending request (FIFO)
        oldest_id = next(iter(self._pending))
        req = self._pending.get(oldest_id)
        if req is None or req._resolved:
            return False

        req._resolved = True
        req._answer = text

        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(req._event.set)
        except RuntimeError:
            req._event.set()

        logger.info(
            "Guidance reply resolved %s: %s",
            req.request_id[:8],
            text[:50],
        )
        return True

    def resolve_by_id(self, request_id: str, answer: str) -> bool:
        """REST-based answer submission via Web GUI."""
        target = None
        for rid, req in self._pending.items():
            if rid == request_id or rid.startswith(request_id):
                target = req
                break

        if target is None or target._resolved:
            return False

        target._resolved = True
        target._answer = answer

        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(target._event.set)
        except RuntimeError:
            target._event.set()

        logger.info(
            "Web GUI answered guidance %s: %s",
            target.request_id[:8],
            answer[:50],
        )
        return True

    # ============== Telegram notifications ==============

    def _send_guidance_telegram(self, req: PendingGuidanceRequest) -> None:
        """Send guidance request — Telegram only if user is NOT on web."""
        try:
            from remy.core.notification_router import (
                is_web_runtime_available,
                send_telegram,
                should_notify_telegram,
            )
        except ImportError:
            return

        try:
            from remy.config.settings import settings

            if (
                getattr(settings, "TELEGRAM_SUPPRESS_WHEN_WEB_ENABLED", True)
                and is_web_runtime_available()
            ):
                logger.info(
                    "Guidance %s: web runtime available, suppressing Telegram",
                    req.request_id[:8],
                )
                return
        except Exception as e:
            logger.debug("Could not evaluate web-runtime suppression for guidance: %s", e)

        if not should_notify_telegram():
            # User is on web — guidance.pending event already emitted,
            # web UI will show the interactive card.
            logger.info("Guidance %s: user on web, skipping Telegram", req.request_id[:8])
            return

        if not self._telegram_configured:
            return

        context_text = f"\n\n📋 Контекст: {req.context}" if req.context else ""
        msg = (
            f"❓ *Потрібна допомога* (ID: `{req.request_id[:8]}`)\n\n"
            f"{req.question}{context_text}\n\n"
            f"Будь ласка, відповідайте текстом у цьому чаті.\n"
            f"Тайм-аут: {req.timeout_sec} с."
        )

        send_telegram(msg)
        logger.info("Guidance request sent to Telegram %s", req.request_id[:8])

    # ============== Event bus ==============

    def _emit_pending(self, req: PendingGuidanceRequest) -> None:
        try:
            from remy.core.event_bus import event_bus
            from remy.core.runtime_event_contract import build_runtime_event

            event_bus.emit(
                "guidance.pending",
                build_runtime_event(
                    "guidance.pending",
                    event_domain="guidance",
                    payload={
                        "request_id": req.request_id,
                        "question": req.question,
                        "context": req.context,
                        "timeout_sec": req.timeout_sec,
                        "created_at": req.created_at,
                    },
                    legacy_fields={
                        "request_id": req.request_id,
                        "question": req.question,
                        "context": req.context,
                        "timeout_sec": req.timeout_sec,
                        "created_at": req.created_at,
                    },
                ),
            )
        except Exception:
            pass

    def _emit_resolved(self, req: PendingGuidanceRequest) -> None:
        try:
            from remy.core.event_bus import event_bus
            from remy.core.runtime_event_contract import build_runtime_event

            event_bus.emit(
                "guidance.resolved",
                build_runtime_event(
                    "guidance.resolved",
                    event_domain="guidance",
                    payload={
                        "request_id": req.request_id,
                        "answer": req._answer,
                        "timed_out": req._answer is None,
                    },
                    legacy_fields={
                        "request_id": req.request_id,
                        "answer": req._answer,
                        "timed_out": req._answer is None,
                    },
                ),
            )
        except Exception:
            pass


# Module-level singleton
guidance_queue = GuidanceQueue()
