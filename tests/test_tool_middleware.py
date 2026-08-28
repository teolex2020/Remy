from __future__ import annotations

import json

from remy.core.tool_middleware import (
    ToolMiddlewareChain,
    ToolMiddlewareOutcome,
    build_default_tool_middleware_chain,
)
from remy.core.tool_pipeline import ToolPipeline, get_last_tool_pipeline_snapshot


def _pipeline(chain, executed):
    def before(ctx):
        receipt = chain.run_before(ctx)
        ctx.policy["middleware_before"] = receipt
        if receipt.get("blocked"):
            return {
                "decision": "deny",
                "reason": receipt["reason"],
                "result": json.dumps({"error": receipt["reason"]}),
            }
        return None

    def after(ctx):
        ctx.policy["middleware_after"] = chain.run_after(ctx)

    return ToolPipeline(
        executor=lambda ctx: executed.append(dict(ctx.args)) or "raw-result",
        before_middleware=before,
        after_middleware=after,
    )


def test_middleware_runs_in_registration_order_and_chains_modifications():
    chain = ToolMiddlewareChain()
    observed = []
    chain.register(
        "first",
        lambda event: observed.append(("first", dict(event.args)))
        or ToolMiddlewareOutcome.modify_args({"query": "normalized"}),
        phases=("before",),
    )
    chain.register(
        "second",
        lambda event: observed.append(("second", dict(event.args))),
        phases=("before",),
    )
    executed = []

    result = _pipeline(chain, executed).run("web_search", {"query": "  raw  "})

    assert result == "raw-result"
    assert observed == [("first", {"query": "  raw  "}), ("second", {"query": "normalized"})]
    assert executed == [{"query": "normalized"}]


def test_before_block_is_monotonic_and_prevents_execution():
    chain = ToolMiddlewareChain()
    chain.register("deny", lambda _event: ToolMiddlewareOutcome.block("policy says no"), phases=("before",))
    executed = []

    result = _pipeline(chain, executed).run("demo", {"secret": "not-in-receipt"})
    receipt = get_last_tool_pipeline_snapshot(clear=True)

    assert executed == []
    assert json.loads(result)["error"] == "policy says no"
    assert receipt["decision"] == "deny"
    assert "not-in-receipt" not in json.dumps(receipt)


def test_before_handler_failure_is_closed_but_after_failure_is_open():
    before = ToolMiddlewareChain()
    before.register("broken", lambda _event: 1 / 0, phases=("before",), fail_closed_before=True)
    executed = []
    blocked = _pipeline(before, executed).run("demo", {})
    assert executed == []
    assert "failed closed" in json.loads(blocked)["error"]

    after = ToolMiddlewareChain()
    after.register("broken-after", lambda _event: 1 / 0, phases=("after",))
    executed = []
    preserved = _pipeline(after, executed).run("demo", {})
    receipt = get_last_tool_pipeline_snapshot(clear=True)
    assert preserved == "raw-result"
    assert executed == [{}]
    assert receipt["policy"]["middleware_after"]["handlers"][0]["action"] == "error"


def test_after_middleware_can_modify_result_but_cannot_fake_a_block():
    chain = ToolMiddlewareChain()
    chain.register(
        "redactor",
        lambda _event: ToolMiddlewareOutcome.modify_result("safe-result"),
        phases=("after",),
    )
    chain.register(
        "late-block",
        lambda _event: ToolMiddlewareOutcome.block("too late"),
        phases=("after",),
    )

    result = _pipeline(chain, []).run("demo", {})
    receipt = get_last_tool_pipeline_snapshot(clear=True)

    assert result == "safe-result"
    handlers = receipt["policy"]["middleware_after"]["handlers"]
    assert handlers[0]["result_modified"] is True
    assert handlers[1]["action"] == "error"


def test_default_chain_blocks_simple_shell_gravity_only():
    chain = build_default_tool_middleware_chain()
    blocked_exec = []
    blocked = _pipeline(chain, blocked_exec).run(
        "shell_exec",
        {"command": "Get-Content README.md", "working_dir": "workspace://project/"},
    )
    assert blocked_exec == []
    assert "fs_read" in json.loads(blocked)["error"]

    allowed_exec = []
    allowed = _pipeline(chain, allowed_exec).run(
        "shell_exec",
        {"command": "python -m pytest", "working_dir": "workspace://project/"},
    )
    assert allowed == "raw-result"
    assert len(allowed_exec) == 1


def test_default_chain_normalizes_query_before_execution():
    chain = build_default_tool_middleware_chain()
    executed = []

    result = _pipeline(chain, executed).run("web_search", {"query": "  local   agent\n sandbox "})

    assert result == "raw-result"
    assert executed == [{"query": "local agent sandbox"}]


def test_canonical_dispatch_uses_default_middleware_before_backend(monkeypatch):
    from remy.core import tool_dispatch

    executed = []
    monkeypatch.setattr(tool_dispatch, "_pipeline_pre_policy_receipt", lambda ctx: None)
    monkeypatch.setattr(tool_dispatch, "_pipeline_durable_observation", lambda ctx: None)
    monkeypatch.setattr(
        tool_dispatch,
        "_execute_tool_backend",
        lambda *args: executed.append(args) or "should-not-run",
    )

    result = tool_dispatch.execute_tool(
        "shell_exec",
        {"command": "Get-Content README.md", "working_dir": "workspace://project/"},
        session_id="middleware-integration",
        channel="desktop",
    )
    receipt = get_last_tool_pipeline_snapshot(clear=True)

    assert executed == []
    assert "fs_read" in json.loads(result)["error"]
    assert receipt["policy"]["middleware_before"]["status"] == "deny"
    assert receipt["policy"]["middleware_before"]["handlers"][-1]["name"] == "contract.shell-gravity"
