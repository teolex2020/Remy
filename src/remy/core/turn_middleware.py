"""Small native middleware pipeline for Remy model turns."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from remy.core.file_utils import atomic_write

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TurnContext:
    session_id: str = ""
    channel: str = ""
    project_id: str = ""
    cancellation_reason: str = "the previous execution ended before a tool result was recorded"


@dataclass(slots=True)
class TurnMiddlewareResult:
    messages: list[BaseMessage]
    state_updates: list[BaseMessage] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)


class BeforeModelMiddleware(Protocol):
    name: str

    def __call__(
        self,
        result: TurnMiddlewareResult,
        context: TurnContext,
    ) -> TurnMiddlewareResult: ...


def _safe_segment(value: str, fallback: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "")).strip("-._")
    return clean[:80] or fallback


class ToolResultArtifactMiddleware:
    """Persist large tool results inside the active project before truncation."""

    name = "tool-result-artifacts"

    def __init__(self, *, threshold: int = 4000, preview_chars: int = 1200):
        self.threshold = max(500, int(threshold))
        self.preview_chars = max(200, int(preview_chars))

    def __call__(
        self,
        result: TurnMiddlewareResult,
        context: TurnContext,
    ) -> TurnMiddlewareResult:
        changed: list[BaseMessage] = []
        for message in result.messages:
            content = message.content if isinstance(message, ToolMessage) else None
            if not isinstance(content, str) or len(content) <= self.threshold:
                changed.append(message)
                continue
            if context.channel == "trajectory-replay":
                changed.append(message.model_copy(update={
                    "content": (
                        content[: self.preview_chars]
                        + f"\n\n...[sandbox fixture truncated: {len(content)} chars]"
                    )
                }))
                continue
            try:
                from remy.core.microbrain import current_project_id
                from remy.core.project_store import project_artifact_dir, project_data_root

                project_id = context.project_id or current_project_id()
                root = project_artifact_dir("artifacts", project_id, create=True)
                session_dir = root / "tool-results" / _safe_segment(
                    context.session_id,
                    "anonymous",
                )
                full_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                tool_name = _safe_segment(message.name or "tool", "tool")
                target = session_dir / f"{tool_name}-{full_digest[:20]}.txt"
                if not target.exists():
                    atomic_write(target, content)
                relative = target.relative_to(project_data_root(project_id)).as_posix()
                marker = f"project://{relative}"
                preview = (
                    content[: self.preview_chars]
                    + f"\n\n...[full result offloaded: {len(content)} chars]\n"
                    + f"Full result: {marker}"
                )
                artifact = {
                    "kind": "tool_result",
                    "project_id": project_id,
                    "path": relative,
                    "uri": marker,
                    "sha256": full_digest,
                    "chars": len(content),
                }
                replacement = message.model_copy(
                    update={"content": preview, "artifact": artifact}
                )
                changed.append(replacement)
                # Same message ID replaces the large checkpoint value instead
                # of appending a duplicate history entry.
                if replacement.id:
                    result.state_updates.append(replacement)
                result.artifacts.append(artifact)
            except Exception as exc:  # Artifact failure must not block an answer.
                logger.warning("Could not offload large tool result: %s", exc)
                changed.append(message)
        result.messages = changed
        return result


class IncompleteToolCallMiddleware:
    """Turn dangling tool requests into explicit cancellation receipts."""

    name = "incomplete-tool-call-receipts"

    def __call__(
        self,
        result: TurnMiddlewareResult,
        context: TurnContext,
    ) -> TurnMiddlewareResult:
        answered_ids = {
            str(message.tool_call_id)
            for message in result.messages
            if isinstance(message, ToolMessage)
        }
        patched: list[BaseMessage] = []
        for message in result.messages:
            patched.append(message)
            if not isinstance(message, AIMessage) or not message.tool_calls:
                continue
            for tool_call in message.tool_calls:
                call_id = str(tool_call.get("id") or "").strip()
                if not call_id or call_id in answered_ids:
                    continue
                tool_name = str(tool_call.get("name") or "tool")
                receipt_id = "remy-cancel-" + hashlib.sha256(
                    call_id.encode("utf-8")
                ).hexdigest()[:24]
                receipt = ToolMessage(
                    content=(
                        f"Execution receipt: tool call '{tool_name}' was cancelled because "
                        f"{context.cancellation_reason}. No successful result exists. "
                        "Retry only if the action is still required."
                    ),
                    tool_call_id=call_id,
                    name=tool_name,
                    id=receipt_id,
                    status="error",
                    additional_kwargs={
                        "remy_receipt": "cancelled",
                        "reason": context.cancellation_reason,
                    },
                )
                patched.append(receipt)
                result.state_updates.append(receipt)
                answered_ids.add(call_id)
        result.messages = patched
        return result


class IntermediateToolTextMiddleware:
    """Hide draft text emitted together with tool calls from the next turn."""

    name = "intermediate-tool-text"

    def __call__(
        self,
        result: TurnMiddlewareResult,
        context: TurnContext,
    ) -> TurnMiddlewareResult:
        del context
        result.messages = [
            message.model_copy(update={"content": ""})
            if isinstance(message, AIMessage) and message.tool_calls and message.content
            else message
            for message in result.messages
        ]
        return result


DEFAULT_BEFORE_MODEL_MIDDLEWARE: tuple[BeforeModelMiddleware, ...] = (
    ToolResultArtifactMiddleware(),
    IncompleteToolCallMiddleware(),
    IntermediateToolTextMiddleware(),
)


def run_before_model_middleware(
    messages: Sequence[BaseMessage],
    context: TurnContext,
    *,
    middleware: Sequence[BeforeModelMiddleware] | None = None,
) -> TurnMiddlewareResult:
    """Run deterministic request-time middleware in declared order."""
    result = TurnMiddlewareResult(messages=list(messages))
    for step in middleware or DEFAULT_BEFORE_MODEL_MIDDLEWARE:
        result = step(result, context)
    return result
