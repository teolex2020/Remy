"""Ordered, observable execution pipeline for every Remy tool call.

The pipeline deliberately owns orchestration only.  Existing handlers remain the
execution backend, while policy, provenance, approval and observation get stable
stage boundaries that can be inspected in Trajectory and tested independently.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Mapping


class ToolDecision(IntEnum):
    """Restriction level.  A pipeline decision may only move upward."""

    ALLOW = 0
    REQUIRE_APPROVAL = 1
    DENY = 2


STAGE_ORDER = (
    "resolve",
    "validate",
    "provenance",
    "pre_policy",
    "monotonic_guards",
    "approval",
    "execute",
    "post_policy",
    "artifact_spill",
    "durable_observation",
)


@dataclass(slots=True)
class ToolPipelineContext:
    name: str
    args: dict[str, Any]
    session_id: str = ""
    channel: str = ""
    invocation_id: str = field(default_factory=lambda: f"tool-run-{uuid.uuid4().hex}")
    resolved_name: str = ""
    decision: ToolDecision = ToolDecision.ALLOW
    decision_reason: str = ""
    approval_required: bool = False
    approval_mode: str = "not-required"
    provenance: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    result: str = ""
    error: str = ""
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    stage_trace: list[dict[str, Any]] = field(default_factory=list)

    def tighten(self, decision: ToolDecision, reason: str = "") -> bool:
        """Apply a guard decision without permitting later policy weakening."""

        proposed = ToolDecision(decision)
        if proposed < self.decision:
            return False
        if proposed > self.decision:
            self.decision = proposed
            self.decision_reason = str(reason or self.decision_reason)
        elif reason and not self.decision_reason:
            self.decision_reason = str(reason)
        return True

    def snapshot(self) -> dict[str, Any]:
        return {
            "invocation_id": self.invocation_id,
            "tool": self.resolved_name or self.name,
            "session_id": self.session_id,
            "channel": self.channel,
            "decision": self.decision.name.lower(),
            "decision_reason": self.decision_reason,
            "approval_required": self.approval_required,
            "approval_mode": self.approval_mode,
            "provenance": dict(self.provenance),
            "policy": dict(self.policy),
            "artifacts": list(self.artifacts),
            "result": {
                "status": "error" if self.error else "completed",
                "chars": len(self.result),
                "sha256": hashlib.sha256(self.result.encode("utf-8")).hexdigest()
                if self.result
                else "",
                "error": self.error,
            },
            "duration_ms": max(
                0,
                int(((self.completed_at or time.time()) - self.started_at) * 1000),
            ),
            "stages": list(self.stage_trace),
        }


_last_pipeline_snapshot: ContextVar[dict[str, Any] | None] = ContextVar(
    "remy_last_tool_pipeline_snapshot",
    default=None,
)


def get_last_tool_pipeline_snapshot(*, clear: bool = False) -> dict[str, Any] | None:
    """Return the last pipeline receipt in the current execution context."""

    value = _last_pipeline_snapshot.get()
    if clear:
        _last_pipeline_snapshot.set(None)
    return dict(value) if value else None


ResolveHook = Callable[[ToolPipelineContext], str]
ContextHook = Callable[[ToolPipelineContext], Any]
ExecuteHook = Callable[[ToolPipelineContext], str]


class ToolPipeline:
    """Run a tool through a fixed sequence of policy and observation stages."""

    def __init__(
        self,
        *,
        executor: ExecuteHook,
        resolver: ResolveHook | None = None,
        provenance: ContextHook | None = None,
        pre_policy: ContextHook | None = None,
        monotonic_guards: ContextHook | None = None,
        approval: ContextHook | None = None,
        post_policy: ContextHook | None = None,
        artifact_spill: ContextHook | None = None,
        durable_observation: ContextHook | None = None,
    ):
        self.executor = executor
        self.resolver = resolver
        self.provenance_hook = provenance
        self.pre_policy_hook = pre_policy
        self.monotonic_guards_hook = monotonic_guards
        self.approval_hook = approval
        self.post_policy_hook = post_policy
        self.artifact_spill_hook = artifact_spill
        self.durable_observation_hook = durable_observation

    @staticmethod
    def _apply_guard(ctx: ToolPipelineContext, value: Any) -> None:
        if not value:
            return
        values = value if isinstance(value, (list, tuple)) else (value,)
        for item in values:
            if not isinstance(item, Mapping):
                continue
            raw_decision = str(item.get("decision") or "allow").upper()
            try:
                decision = ToolDecision[raw_decision]
            except KeyError:
                continue
            ctx.tighten(decision, str(item.get("reason") or ""))
            if item.get("policy") and isinstance(item["policy"], Mapping):
                ctx.policy.update(dict(item["policy"]))
            if item.get("result") and decision == ToolDecision.DENY:
                ctx.result = str(item["result"])

    @staticmethod
    def _trace(ctx: ToolPipelineContext, stage: str, started: float, status: str) -> None:
        ctx.stage_trace.append(
            {
                "stage": stage,
                "status": status,
                "duration_ms": max(0, int((time.time() - started) * 1000)),
                "decision": ctx.decision.name.lower(),
            }
        )

    def run(
        self,
        name: str,
        args: Mapping[str, Any] | None,
        *,
        session_id: str | None = None,
        channel: str | None = None,
    ) -> str:
        safe_args = dict(args) if isinstance(args, Mapping) else {}
        ctx = ToolPipelineContext(
            name=str(name or ""),
            args=safe_args,
            session_id=str(session_id or ""),
            channel=str(channel or ""),
        )

        for stage in STAGE_ORDER:
            started = time.time()
            status = "completed"
            try:
                if stage == "resolve":
                    ctx.resolved_name = str(
                        self.resolver(ctx) if self.resolver else ctx.name.strip()
                    )
                elif stage == "validate":
                    if not ctx.resolved_name:
                        ctx.result = json.dumps({"error": "Tool name is required"})
                        ctx.error = "Tool name is required"
                        ctx.tighten(ToolDecision.DENY, ctx.error)
                    elif args is not None and not isinstance(args, Mapping):
                        ctx.result = json.dumps({"error": "Tool arguments must be an object"})
                        ctx.error = "Tool arguments must be an object"
                        ctx.tighten(ToolDecision.DENY, ctx.error)
                elif stage == "provenance" and self.provenance_hook:
                    value = self.provenance_hook(ctx)
                    if isinstance(value, Mapping):
                        ctx.provenance.update(dict(value))
                elif stage == "pre_policy" and self.pre_policy_hook:
                    self._apply_guard(ctx, self.pre_policy_hook(ctx))
                elif stage == "monotonic_guards" and self.monotonic_guards_hook:
                    if ctx.decision == ToolDecision.DENY:
                        status = "skipped"
                    else:
                        self._apply_guard(ctx, self.monotonic_guards_hook(ctx))
                elif stage == "approval" and self.approval_hook:
                    if ctx.decision == ToolDecision.DENY:
                        status = "skipped"
                    else:
                        value = self.approval_hook(ctx)
                        if isinstance(value, Mapping):
                            required = bool(value.get("required"))
                            ctx.approval_required = required
                            ctx.approval_mode = str(
                                value.get("mode") or ("required" if required else "not-required")
                            )
                            if required:
                                ctx.tighten(
                                    ToolDecision.REQUIRE_APPROVAL,
                                    str(value.get("reason") or "handler approval required"),
                                )
                            if value.get("approved") is False:
                                ctx.tighten(
                                    ToolDecision.DENY,
                                    str(value.get("reason") or "Approval denied"),
                                )
                                if value.get("result"):
                                    ctx.result = str(value["result"])
                elif stage == "execute":
                    if ctx.decision == ToolDecision.DENY:
                        status = "skipped"
                        if not ctx.result:
                            ctx.result = json.dumps(
                                {"error": ctx.decision_reason or "Tool execution denied"},
                                ensure_ascii=False,
                            )
                    else:
                        ctx.result = str(self.executor(ctx))
                elif stage == "post_policy" and self.post_policy_hook:
                    self.post_policy_hook(ctx)
                elif stage == "artifact_spill" and self.artifact_spill_hook:
                    artifacts = self.artifact_spill_hook(ctx)
                    if isinstance(artifacts, Mapping):
                        ctx.artifacts.append(dict(artifacts))
                    elif isinstance(artifacts, (list, tuple)):
                        ctx.artifacts.extend(dict(item) for item in artifacts if isinstance(item, Mapping))
                elif stage == "durable_observation":
                    ctx.completed_at = time.time()
                    if self.durable_observation_hook:
                        self.durable_observation_hook(ctx)
            except Exception as exc:
                status = "failed"
                if stage == "execute":
                    ctx.error = f"{type(exc).__name__}: {exc}"
                    ctx.result = f"Error: {exc}"
                elif stage in {"resolve", "validate", "pre_policy", "monotonic_guards", "approval"}:
                    ctx.error = f"{stage}: {type(exc).__name__}: {exc}"
                    ctx.tighten(ToolDecision.DENY, ctx.error)
                # Observation and artifact failures are intentionally fail-open.
            finally:
                self._trace(ctx, stage, started, status)

        ctx.completed_at = ctx.completed_at or time.time()
        snapshot = ctx.snapshot()
        _last_pipeline_snapshot.set(snapshot)
        return ctx.result


def spill_large_tool_result(
    ctx: ToolPipelineContext,
    *,
    threshold: int = 4000,
) -> list[dict[str, Any]]:
    """Persist a large result idempotently while leaving handler output intact."""

    content = str(ctx.result or "")
    if len(content) <= max(500, int(threshold)) or ctx.channel == "trajectory-replay":
        return []
    from remy.core.file_utils import atomic_write
    from remy.core.microbrain import current_project_id
    from remy.core.project_store import project_artifact_dir, project_data_root

    safe_session = re.sub(r"[^a-zA-Z0-9._-]+", "-", ctx.session_id).strip("-._")[:80]
    safe_tool = re.sub(r"[^a-zA-Z0-9._-]+", "-", ctx.resolved_name).strip("-._")[:80]
    project_id = current_project_id()
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    root = project_artifact_dir("artifacts", project_id, create=True)
    target = root / "tool-results" / (safe_session or "anonymous") / f"{safe_tool or 'tool'}-{digest[:20]}.txt"
    if not target.exists():
        atomic_write(target, content)
    relative = target.relative_to(project_data_root(project_id)).as_posix()
    return [
        {
            "kind": "tool_result",
            "project_id": project_id,
            "path": relative,
            "uri": f"project://{relative}",
            "sha256": digest,
            "chars": len(content),
        }
    ]
