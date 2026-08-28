from __future__ import annotations

import json

from remy.core.tool_pipeline import (
    STAGE_ORDER,
    ToolDecision,
    ToolPipeline,
    get_last_tool_pipeline_snapshot,
)
from remy.core.trajectory_store import TrajectoryStore


def test_tool_pipeline_has_stable_stage_order_and_receipt():
    observed = []

    pipeline = ToolPipeline(
        resolver=lambda ctx: observed.append("resolve-hook") or f"resolved_{ctx.name}",
        provenance=lambda ctx: {"source": "test", "trust_score": 1.0},
        before_middleware=lambda ctx: observed.append("middleware-before") or None,
        pre_policy=lambda ctx: observed.append("pre-policy") or None,
        monotonic_guards=lambda ctx: observed.append("guards") or None,
        approval=lambda ctx: {"required": False, "mode": "not-required"},
        executor=lambda ctx: observed.append("execute") or json.dumps({"ok": True}),
        after_middleware=lambda ctx: observed.append("middleware-after"),
        post_policy=lambda ctx: observed.append("post-policy"),
        artifact_spill=lambda ctx: observed.append("artifact-spill") or [],
        durable_observation=lambda ctx: observed.append("durable-observation"),
    )

    result = pipeline.run("demo", {"value": 7}, session_id="session-1", channel="desktop")
    receipt = get_last_tool_pipeline_snapshot()

    assert json.loads(result) == {"ok": True}
    assert receipt is not None
    assert receipt["tool"] == "resolved_demo"
    assert receipt["provenance"]["source"] == "test"
    assert [stage["stage"] for stage in receipt["stages"]] == list(STAGE_ORDER)
    assert observed == [
        "resolve-hook",
        "middleware-before",
        "pre-policy",
        "guards",
        "execute",
        "middleware-after",
        "post-policy",
        "artifact-spill",
        "durable-observation",
    ]


def test_tool_pipeline_guard_decision_cannot_be_weakened():
    executed = []
    pipeline = ToolPipeline(
        executor=lambda ctx: executed.append(ctx.resolved_name) or "should-not-run",
        pre_policy=lambda ctx: {
            "decision": "deny",
            "reason": "untrusted source",
            "result": json.dumps({"error": "blocked"}),
        },
        # A later guard may not relax an earlier denial.
        monotonic_guards=lambda ctx: {"decision": "allow", "reason": "late allow"},
    )

    result = pipeline.run("dangerous_tool", {})
    receipt = get_last_tool_pipeline_snapshot()

    assert executed == []
    assert json.loads(result) == {"error": "blocked"}
    assert receipt is not None
    assert receipt["decision"] == ToolDecision.DENY.name.lower()
    assert receipt["decision_reason"] == "untrusted source"
    execute_stage = next(stage for stage in receipt["stages"] if stage["stage"] == "execute")
    assert execute_stage["status"] == "skipped"


def test_later_equal_denial_cannot_replace_first_block_reason_or_result():
    pipeline = ToolPipeline(
        executor=lambda ctx: "should-not-run",
        before_middleware=lambda ctx: {
            "decision": "deny",
            "reason": "first causal block",
            "result": json.dumps({"error": "first"}),
        },
        pre_policy=lambda ctx: {
            "decision": "deny",
            "reason": "later duplicate block",
            "result": json.dumps({"error": "later"}),
        },
    )

    result = pipeline.run("demo", {})
    receipt = get_last_tool_pipeline_snapshot(clear=True)

    assert json.loads(result)["error"] == "first"
    assert receipt["decision_reason"] == "first causal block"


def test_tool_pipeline_validation_denies_non_mapping_arguments():
    pipeline = ToolPipeline(executor=lambda ctx: "should-not-run")

    result = pipeline.run("demo", ["not", "an", "object"])
    receipt = get_last_tool_pipeline_snapshot(clear=True)

    assert json.loads(result)["error"] == "Tool arguments must be an object"
    assert receipt is not None
    assert receipt["decision"] == "deny"
    assert get_last_tool_pipeline_snapshot() is None


def test_required_approval_is_visible_without_weakening_prior_guards():
    pipeline = ToolPipeline(
        executor=lambda ctx: json.dumps({"approved": True}),
        approval=lambda ctx: {"required": True, "mode": "handler-managed"},
    )

    result = pipeline.run("financial_action", {"amount": 10})
    receipt = get_last_tool_pipeline_snapshot(clear=True)

    assert json.loads(result)["approved"] is True
    assert receipt["decision"] == "require_approval"
    assert receipt["approval_required"] is True
    assert receipt["approval_mode"] == "handler-managed"


def test_pipeline_managed_approval_wraps_exact_executor_once():
    calls = []

    def approval_executor(ctx, execute):
        calls.append((ctx.resolved_name, dict(ctx.args)))
        ctx.approval_outcome = "approved"
        return execute(ctx)

    pipeline = ToolPipeline(
        executor=lambda ctx: calls.append("executed") or json.dumps({"ok": True}),
        approval=lambda ctx: {
            "required": True,
            "mode": "pipeline-managed",
            "target": "https://bank.example/action",
            "description": "Sensitive action",
        },
        approval_executor=approval_executor,
    )

    result = pipeline.run("financial_action", {"amount": 10})
    receipt = get_last_tool_pipeline_snapshot(clear=True)

    assert json.loads(result)["ok"] is True
    assert calls == [("financial_action", {"amount": 10}), "executed"]
    assert receipt["decision"] == "require_approval"
    assert receipt["approval_mode"] == "pipeline-managed"
    assert receipt["approval_outcome"] == "approved"
    assert len(receipt["approval_target_sha256"]) == 64
    assert "bank.example" not in json.dumps(receipt)
    assert "Sensitive action" not in json.dumps(receipt)


def test_pipeline_managed_approval_without_executor_fails_closed():
    executed = []
    pipeline = ToolPipeline(
        executor=lambda ctx: executed.append(True) or "unsafe",
        approval=lambda ctx: {"required": True, "mode": "pipeline-managed"},
    )

    result = pipeline.run("financial_action", {"amount": 10})
    receipt = get_last_tool_pipeline_snapshot(clear=True)

    assert executed == []
    assert "approval executor is unavailable" in json.loads(result)["error"]
    assert receipt["decision"] == "deny"
    assert receipt["approval_outcome"] == "denied"


def test_pipeline_approval_grant_is_exact_and_context_local(monkeypatch):
    from remy.core import approval_queue as approvals

    monkeypatch.setattr(approvals.approval_queue, "_enabled", True)
    args = {"amount": 10, "recipient": "wallet-a"}
    assert approvals.needs_approval("send_usdt", args) is True
    with approvals.pipeline_approval_grant("send_usdt", args):
        assert approvals.needs_approval("send_usdt", args) is False
        assert approvals.needs_approval("send_usdt", {**args, "amount": 11}) is True
        assert approvals.needs_approval("send_trx", args) is True
    assert approvals.needs_approval("send_usdt", args) is True


def test_dispatch_pipeline_owns_sensitive_approval_once(monkeypatch):
    from remy.core import approval_queue as approvals
    from remy.core import brain_tools, tool_dispatch

    calls = []
    monkeypatch.setattr(approvals.approval_queue, "_enabled", True)
    monkeypatch.setattr(brain_tools.tool_health, "is_available", lambda name: True)
    monkeypatch.setattr(tool_dispatch, "_pipeline_pre_policy_receipt", lambda ctx: None)
    monkeypatch.setattr(tool_dispatch, "_pipeline_durable_observation", lambda ctx: None)
    monkeypatch.setattr(
        approvals.approval_queue,
        "request_approval_sync",
        lambda description, action_fn, **kwargs: calls.append(kwargs["tool_name"]) or action_fn(),
    )
    monkeypatch.setattr(
        tool_dispatch,
        "_execute_tool_backend",
        lambda name, args, session_id, channel: (
            calls.append("handler")
            or json.dumps({"grant": approvals.has_pipeline_approval_grant(name, args)})
        ),
    )

    result = tool_dispatch.execute_tool(
        "send_usdt",
        {"amount": 10, "recipient": "wallet-a"},
        channel="desktop",
    )
    receipt = get_last_tool_pipeline_snapshot(clear=True)

    assert json.loads(result)["grant"] is True
    assert calls == ["send_usdt", "handler"]
    assert receipt["approval_mode"] == "pipeline-managed"
    assert receipt["approval_outcome"] == "approved"


def test_dispatch_pipeline_appends_privacy_safe_durable_receipt(tmp_path, monkeypatch):
    from remy.core import tool_dispatch, trajectory_store

    trajectory = TrajectoryStore(tmp_path / "trajectory.sqlite3")
    trajectory.begin_turn(
        session_id="session-1",
        project_id="project-1",
        content="Run the tool",
    )
    monkeypatch.setattr(trajectory_store, "get_trajectory_store", lambda: trajectory)
    monkeypatch.setattr(
        tool_dispatch,
        "_execute_tool_backend",
        lambda name, args, session_id, channel: json.dumps({"ok": True}),
    )

    result = tool_dispatch.execute_tool(
        "demo",
        {"secret": "must-not-be-copied", "count": 2},
        session_id="session-1",
        channel="desktop",
    )

    assert json.loads(result) == {"ok": True}
    policy_events = [
        event
        for event in trajectory.list_events(project_id="project-1", session_id="session-1")
        if event["kind"] == "POLICY"
    ]
    receipt = policy_events[-1]["input"]
    assert receipt["subtype"] == "tool_pipeline"
    assert receipt["tool"] == "demo"
    assert receipt["argument_keys"] == ["count", "secret"]
    assert len(receipt["argument_sha256"]) == 64
    assert receipt["policy"]["circuit_breaker"]["status"] == "allow"
    assert "must-not-be-copied" not in json.dumps(receipt)
    assert [stage["stage"] for stage in receipt["stages"]] == list(STAGE_ORDER)


def test_circuit_breaker_denial_happens_before_execute(monkeypatch):
    from remy.core import brain_tools, tool_dispatch

    class ClosedHealth:
        @staticmethod
        def is_available(name):
            return False

        @staticmethod
        def get_health_report():
            return {"demo": "open"}

    executed = []
    monkeypatch.setattr(brain_tools, "tool_health", ClosedHealth())
    monkeypatch.setattr(tool_dispatch, "_pipeline_pre_policy_receipt", lambda ctx: None)
    monkeypatch.setattr(tool_dispatch, "_pipeline_durable_observation", lambda ctx: None)
    monkeypatch.setattr(
        tool_dispatch,
        "_execute_tool_backend",
        lambda *args: executed.append(args) or "should-not-run",
    )

    result = tool_dispatch.execute_tool("demo", {})
    receipt = get_last_tool_pipeline_snapshot(clear=True)

    assert executed == []
    assert "temporarily unavailable" in json.loads(result)["error"]
    assert receipt["decision"] == "deny"
    assert receipt["policy"]["circuit_breaker"]["status"] == "deny"


def test_provenance_denial_is_monotonic_and_redacted_in_receipt(monkeypatch):
    from remy.core import brain_tools, provenance, tool_dispatch

    executed = []
    monkeypatch.setattr(brain_tools.tool_health, "is_available", lambda name: True)
    monkeypatch.setattr(provenance, "_TRUST_ENFORCED_TOOLS", frozenset({"sensitive_demo"}))
    monkeypatch.setattr(
        provenance,
        "_validate_action_data",
        lambda name, args: "TRUST GUARD: email='private@example.com' is unverified",
    )
    monkeypatch.setattr(tool_dispatch, "_pipeline_pre_policy_receipt", lambda ctx: None)
    monkeypatch.setattr(tool_dispatch, "_pipeline_durable_observation", lambda ctx: None)
    monkeypatch.setattr(
        tool_dispatch,
        "_execute_tool_backend",
        lambda *args: executed.append(args) or "should-not-run",
    )

    result = tool_dispatch.execute_tool(
        "sensitive_demo",
        {"email": "private@example.com"},
        channel="autonomous",
    )
    receipt = get_last_tool_pipeline_snapshot(clear=True)

    assert executed == []
    assert "private@example.com" in json.loads(result)["error"]
    assert receipt["decision"] == "deny"
    assert "private@example.com" not in json.dumps(receipt)
    assert receipt["policy"]["provenance_guard"]["status"] == "deny"


def test_workspace_permission_denial_happens_before_handler(monkeypatch):
    from remy.core import brain_tools, tool_dispatch, workspace_permissions

    class DeniedManager:
        @staticmethod
        def resolve(raw, permission, must_exist=False):
            raise workspace_permissions.WorkspaceAccessError("outside approved workspaces")

    executed = []
    monkeypatch.setattr(brain_tools.tool_health, "is_available", lambda name: True)
    monkeypatch.setattr(workspace_permissions, "get_workspace_manager", lambda: DeniedManager())
    monkeypatch.setattr(tool_dispatch, "_pipeline_pre_policy_receipt", lambda ctx: None)
    monkeypatch.setattr(tool_dispatch, "_pipeline_durable_observation", lambda ctx: None)
    monkeypatch.setattr(
        tool_dispatch,
        "_execute_tool_backend",
        lambda *args: executed.append(args) or "should-not-run",
    )

    result = tool_dispatch.execute_tool("read_file", {"path": "C:/private.txt"})
    receipt = get_last_tool_pipeline_snapshot(clear=True)

    assert executed == []
    assert "outside approved workspaces" in json.loads(result)["error"]
    assert receipt["decision"] == "deny"
    assert receipt["policy"]["workspace_guard"]["allowed"] is False
