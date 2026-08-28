"""Deterministic control-plane middleware around canonical tool execution.

This is intentionally separate from ``event_bus``. UI events are lossy and
fire-and-forget; control middleware is ordered, synchronous, privacy-audited,
and runs inside ToolPipeline before policy/approval and after execution.
"""

from __future__ import annotations

import inspect
import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping


MiddlewarePhase = Literal["before", "after"]


@dataclass(frozen=True, slots=True)
class ToolMiddlewareEvent:
    phase: MiddlewarePhase
    invocation_id: str
    tool_name: str
    args: dict[str, Any]
    result: str = ""
    session_id: str = ""
    channel: str = ""


@dataclass(frozen=True, slots=True)
class ToolMiddlewareOutcome:
    action: Literal["pass", "modify", "block"] = "pass"
    reason: str = ""
    args_patch: Mapping[str, Any] = field(default_factory=dict)
    result: str | None = None

    @classmethod
    def modify_args(cls, patch: Mapping[str, Any], *, reason: str = "") -> "ToolMiddlewareOutcome":
        return cls(action="modify", reason=reason, args_patch=dict(patch))

    @classmethod
    def modify_result(cls, result: str, *, reason: str = "") -> "ToolMiddlewareOutcome":
        return cls(action="modify", reason=reason, result=str(result))

    @classmethod
    def block(cls, reason: str) -> "ToolMiddlewareOutcome":
        return cls(action="block", reason=str(reason or "Tool call blocked by middleware"))


ToolMiddlewareHandler = Callable[[ToolMiddlewareEvent], ToolMiddlewareOutcome | Mapping[str, Any] | None]


@dataclass(frozen=True, slots=True)
class _Registration:
    name: str
    handler: ToolMiddlewareHandler
    phases: frozenset[MiddlewarePhase]
    fail_closed_before: bool


def _coerce_outcome(value: Any) -> ToolMiddlewareOutcome:
    if value is None:
        return ToolMiddlewareOutcome()
    if isinstance(value, ToolMiddlewareOutcome):
        return value
    if isinstance(value, Mapping):
        action = str(value.get("action") or "pass").strip().lower()
        if action not in {"pass", "modify", "block"}:
            raise ValueError(f"Unsupported middleware action: {action}")
        patch = value.get("args_patch") or {}
        if not isinstance(patch, Mapping):
            raise ValueError("Middleware args_patch must be an object")
        return ToolMiddlewareOutcome(
            action=action,
            reason=str(value.get("reason") or ""),
            args_patch=dict(patch),
            result=None if value.get("result") is None else str(value.get("result")),
        )
    raise TypeError("Tool middleware must return ToolMiddlewareOutcome, mapping, or None")


class ToolMiddlewareChain:
    """Thread-safe ordered pass/modify/block handler chain."""

    def __init__(self) -> None:
        self._registrations: list[_Registration] = []
        self._lock = threading.RLock()

    def register(
        self,
        name: str,
        handler: ToolMiddlewareHandler,
        *,
        phases: Iterable[MiddlewarePhase] = ("before", "after"),
        fail_closed_before: bool = True,
    ) -> None:
        normalized = str(name or "").strip()
        selected = frozenset(phases)
        if not normalized or not re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", normalized):
            raise ValueError("Middleware name is invalid")
        if not selected or not selected.issubset({"before", "after"}):
            raise ValueError("Middleware phases must contain before and/or after")
        if not callable(handler):
            raise TypeError("Middleware handler must be callable")
        with self._lock:
            if any(item.name == normalized for item in self._registrations):
                raise ValueError(f"Middleware already registered: {normalized}")
            self._registrations.append(
                _Registration(normalized, handler, selected, bool(fail_closed_before))
            )

    def unregister(self, name: str) -> bool:
        with self._lock:
            for index, item in enumerate(self._registrations):
                if item.name == name:
                    self._registrations.pop(index)
                    return True
        return False

    def registrations(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(item.name for item in self._registrations)

    def _snapshot(self, phase: MiddlewarePhase) -> tuple[_Registration, ...]:
        with self._lock:
            return tuple(item for item in self._registrations if phase in item.phases)

    def run_before(self, ctx: Any) -> dict[str, Any]:
        receipts: list[dict[str, Any]] = []
        for item in self._snapshot("before"):
            event = ToolMiddlewareEvent(
                phase="before",
                invocation_id=str(ctx.invocation_id),
                tool_name=str(ctx.resolved_name),
                args=dict(ctx.args),
                session_id=str(ctx.session_id),
                channel=str(ctx.channel),
            )
            try:
                raw = item.handler(event)
                if inspect.isawaitable(raw):
                    raise TypeError("Async middleware is not supported in synchronous tool dispatch")
                outcome = _coerce_outcome(raw)
                if outcome.action == "modify" and outcome.result is not None:
                    raise ValueError("Before middleware cannot modify a tool result")
            except Exception as exc:
                receipts.append(
                    {
                        "name": item.name,
                        "action": "error",
                        "phase": "before",
                        "fail_closed": item.fail_closed_before,
                        "error_type": type(exc).__name__,
                    }
                )
                if item.fail_closed_before:
                    return {
                        "blocked": True,
                        "reason": f"Tool middleware {item.name} failed closed",
                        "handlers": receipts,
                    }
                continue
            receipt: dict[str, Any] = {
                "name": item.name,
                "action": outcome.action,
                "phase": "before",
            }
            if outcome.action == "modify":
                patch = dict(outcome.args_patch)
                ctx.args.update(patch)
                receipt["modified_keys"] = sorted(str(key) for key in patch)
            receipts.append(receipt)
            if outcome.action == "block":
                return {
                    "blocked": True,
                    "reason": outcome.reason or f"Blocked by {item.name}",
                    "handlers": receipts,
                }
        return {"blocked": False, "handlers": receipts}

    def run_after(self, ctx: Any) -> dict[str, Any]:
        receipts: list[dict[str, Any]] = []
        for item in self._snapshot("after"):
            event = ToolMiddlewareEvent(
                phase="after",
                invocation_id=str(ctx.invocation_id),
                tool_name=str(ctx.resolved_name),
                args=dict(ctx.args),
                result=str(ctx.result),
                session_id=str(ctx.session_id),
                channel=str(ctx.channel),
            )
            try:
                raw = item.handler(event)
                if inspect.isawaitable(raw):
                    raise TypeError("Async middleware is not supported in synchronous tool dispatch")
                outcome = _coerce_outcome(raw)
                if outcome.action == "block":
                    raise ValueError("After middleware cannot claim an executed side effect was blocked")
                if outcome.args_patch:
                    raise ValueError("After middleware cannot modify tool arguments")
                receipt: dict[str, Any] = {
                    "name": item.name,
                    "action": outcome.action,
                    "phase": "after",
                }
                if outcome.action == "modify":
                    if outcome.result is None:
                        raise ValueError("After middleware modify requires a result")
                    ctx.result = outcome.result
                    receipt["result_modified"] = True
                receipts.append(receipt)
            except Exception as exc:
                # The handler already ran. Hiding success with an error can make
                # the model retry a side effect, so after failures are fail-open.
                receipts.append(
                    {
                        "name": item.name,
                        "action": "error",
                        "phase": "after",
                        "fail_closed": False,
                        "error_type": type(exc).__name__,
                    }
                )
        return {"handlers": receipts}


_SIMPLE_SHELL_GRAVITY = re.compile(
    r"^\s*(?:cat|type|get-content|gc|grep|rg|findstr|ls|dir)\b",
    flags=re.IGNORECASE,
)
_SHELL_COMPOSITION = re.compile(r"(?:&&|\|\||[|;<>])")


def contract_shell_gravity_guard(event: ToolMiddlewareEvent) -> ToolMiddlewareOutcome | None:
    """Redirect simple read/search/list commands to narrower workspace tools."""
    if event.phase != "before" or event.tool_name != "shell_exec":
        return None
    command = str(event.args.get("command") or "")
    if _SHELL_COMPOSITION.search(command) or not _SIMPLE_SHELL_GRAVITY.search(command):
        return None
    return ToolMiddlewareOutcome.block(
        "A dedicated workspace tool must be used for simple file reads, searches, or listings "
        "(fs_read, fs_search, or list_directory) instead of shell_exec"
    )


def normalize_query_whitespace(event: ToolMiddlewareEvent) -> ToolMiddlewareOutcome | None:
    """Canonicalize harmless query whitespace before caches and policy hashes."""
    if event.phase != "before" or event.tool_name not in {"recall", "search", "web_search"}:
        return None
    query = event.args.get("query")
    if not isinstance(query, str):
        return None
    normalized = " ".join(query.split())
    if normalized == query:
        return None
    return ToolMiddlewareOutcome.modify_args(
        {"query": normalized},
        reason="normalized query whitespace",
    )


def build_default_tool_middleware_chain() -> ToolMiddlewareChain:
    chain = ToolMiddlewareChain()
    chain.register(
        "contract.query-whitespace",
        normalize_query_whitespace,
        phases=("before",),
        fail_closed_before=True,
    )
    chain.register(
        "contract.shell-gravity",
        contract_shell_gravity_guard,
        phases=("before",),
        fail_closed_before=True,
    )
    return chain


_global_chain: ToolMiddlewareChain | None = None
_global_lock = threading.Lock()


def get_tool_middleware_chain() -> ToolMiddlewareChain:
    global _global_chain
    with _global_lock:
        if _global_chain is None:
            _global_chain = build_default_tool_middleware_chain()
        return _global_chain


def middleware_block_result(tool_name: str, reason: str) -> str:
    return json.dumps(
        {
            "error": str(reason or "Tool call blocked by middleware"),
            "tool": str(tool_name or ""),
            "middleware": {"blocked": True},
        },
        ensure_ascii=False,
    )
