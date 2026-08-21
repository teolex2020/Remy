"""Local durable execution support for Remy's LangGraph agents.

LangGraph checkpoints are operational state: they let an interrupted graph
continue from a node boundary.  They intentionally do not replace Aura's
long-term memory or Remy's execution ledger for side-effect receipts.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from pathlib import Path
from typing import Any

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from remy.config.settings import settings

logger = logging.getLogger(__name__)

_runtime: "DurableGraphRuntime | None" = None


def _safe_checkpoint_serializer() -> JsonPlusSerializer:
    """Build a non-pickle serializer across supported LangGraph releases."""
    kwargs: dict[str, Any] = {"pickle_fallback": False}
    parameters = inspect.signature(JsonPlusSerializer).parameters
    if "allowed_msgpack_modules" in parameters:
        kwargs["allowed_msgpack_modules"] = None
    return JsonPlusSerializer(**kwargs)


def durable_thread_id(session_id: str, channel: str) -> str:
    """Return the stable, namespaced checkpoint thread for a Remy session."""
    safe_session = "".join(
        char for char in str(session_id or "") if char.isalnum() or char in {"-", "_"}
    )[:160]
    safe_channel = "".join(
        char for char in str(channel or "") if char.isalnum() or char in {"-", "_"}
    )[:40]
    return f"remy:{safe_channel or 'unknown'}:{safe_session or 'anonymous'}"


def durable_config(
    session_id: str,
    channel: str,
    *,
    recursion_limit: int | None = None,
    checkpoint_ns: str = "",
) -> dict[str, Any]:
    """Build the config shared by invoke, stream, inspection, and resume."""
    config: dict[str, Any] = {
        "configurable": {
            "thread_id": durable_thread_id(session_id, channel),
            "checkpoint_ns": checkpoint_ns,
        }
    }
    if recursion_limit is not None:
        config["recursion_limit"] = int(recursion_limit)
    return config


class DurableGraphRuntime:
    """Own one async SQLite checkpointer for the lifetime of the web server."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path or settings.DATA_DIR / "langgraph" / "checkpoints.sqlite3")
        self._connection: aiosqlite.Connection | None = None
        self._saver: AsyncSqliteSaver | None = None
        self._lock = asyncio.Lock()

    async def get_saver(self) -> AsyncSqliteSaver:
        if self._saver is not None:
            return self._saver
        async with self._lock:
            if self._saver is not None:
                return self._saver
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            connection = await aiosqlite.connect(str(self.db_path))
            try:
                await connection.execute("PRAGMA journal_mode=WAL")
                await connection.execute("PRAGMA synchronous=NORMAL")
                await connection.execute("PRAGMA busy_timeout=5000")
                await connection.commit()
                # Explicit safe allowlist mode.  Never enable pickle fallback:
                # checkpoint files are local, but may still contain model-shaped data.
                serializer = _safe_checkpoint_serializer()
                saver = AsyncSqliteSaver(connection, serde=serializer)
                await saver.setup()
            except BaseException:
                await connection.close()
                raise
            self._connection = connection
            self._saver = saver
            logger.info("LangGraph durable checkpoints ready: %s", self.db_path)
            return saver

    async def delete_thread(self, session_id: str, channel: str) -> None:
        saver = await self.get_saver()
        await saver.adelete_thread(durable_thread_id(session_id, channel))
        await self.record_checkpoint_status(
            session_id=session_id,
            channel=channel,
            status="empty",
            event_type="checkpoint.cleared",
        )

    async def record_checkpoint_status(
        self,
        *,
        session_id: str,
        channel: str,
        status: str,
        event_type: str = "checkpoint.observed",
        project_id: str = "",
        next_nodes: list[str] | tuple[str, ...] = (),
        pending_tasks: list[dict[str, Any]] | None = None,
        checkpoint_created_at: str | None = None,
    ) -> dict[str, Any]:
        """Mirror a bounded checkpoint status into the common session journal."""
        try:
            if not project_id:
                from remy.core.microbrain import current_project_id

                project_id = current_project_id()
            if not project_id or not session_id:
                return {}
            from remy.core.session_event_store import get_session_event_store

            details = {
                "thread_id": durable_thread_id(session_id, channel),
                "channel": channel,
                "next": list(next_nodes or ()),
                "pending_tasks": list(pending_tasks or []),
                "checkpoint_created_at": checkpoint_created_at,
            }
            return await asyncio.to_thread(
                get_session_event_store().append_event,
                subject_event_id=f"checkpoint:{durable_thread_id(session_id, channel)}",
                project_id=project_id,
                session_id=session_id,
                event_type=event_type,
                kind="CHECKPOINT",
                status=status,
                payload={"details": details},
                changed_fields=("status", "next", "pending_tasks"),
            )
        except Exception as exc:
            logger.warning("Could not mirror checkpoint status for %s: %s", session_id, exc)
            return {}

    async def close(self) -> None:
        async with self._lock:
            connection = self._connection
            self._connection = None
            self._saver = None
            if connection is not None:
                await connection.close()
                logger.info("LangGraph durable checkpoint connection closed")


def get_durable_graph_runtime() -> DurableGraphRuntime:
    global _runtime
    if _runtime is None:
        _runtime = DurableGraphRuntime()
    return _runtime


async def close_durable_graph_runtime() -> None:
    global _runtime
    runtime = _runtime
    _runtime = None
    if runtime is not None:
        await runtime.close()
