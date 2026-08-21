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
        pre_policy=lambda ctx: observed.append("pre-policy") or None,
        monotonic_guards=lambda ctx: observed.append("guards") or None,
        approval=lambda ctx: {"required": False, "mode": "not-required"},
        executor=lambda ctx: observed.append("execute") or json.dumps({"ok": True}),
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
        "pre-policy",
        "guards",
        "execute",
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
