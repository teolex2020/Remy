from __future__ import annotations

import pytest

from remy.core.event_bus import EventBus
from remy.core_v3.integrations import (
    BaseIntegrationPlugin,
    IntegrationRegistry,
    PluginContext,
    PluginContextDisposed,
    PluginRequest,
    PluginResourceConflict,
    PluginResourceKind,
    PluginResult,
)


class LifecyclePlugin(BaseIntegrationPlugin):
    plugin_id = "lifecycle"

    def __init__(self, setup):
        self._setup = setup

    def setup(self, ctx):
        return self._setup(ctx)

    def teardown(self, ctx):
        teardown = ctx.metadata.get("teardown")
        if teardown:
            teardown()

    def supports(self, action):
        return action == "run"

    def estimate_cost(self, action, payload=None):
        return 0.0

    def execute(self, request: PluginRequest, ctx: PluginContext) -> PluginResult:
        return PluginResult(ok=True, data={"service": ctx.service("client")})


def test_unregister_disposes_every_resource_in_reverse_registration_order():
    events = []

    class Service:
        def close(self):
            events.append("service")

    def setup(ctx):
        ctx.metadata["teardown"] = lambda: events.append("teardown")
        ctx.register_tool("lookup", object(), dispose=lambda: events.append("tool"))
        ctx.register_listener("updates", object(), dispose=lambda: events.append("listener"))
        ctx.register_service("client", Service())
        ctx.register_prompt_section(
            "rules",
            "Use verified sources.",
            dispose=lambda: events.append("prompt"),
        )

    registry = IntegrationRegistry()
    ctx = registry.register(LifecyclePlugin(setup))

    assert registry.resources.get(PluginResourceKind.TOOL, "lookup") is not None
    report = registry.unregister("lifecycle")

    assert report is not None and report.ok
    assert report.order == (
        "prompt_section:rules",
        "service:client",
        "listener:updates",
        "tool:lookup",
        "custom:plugin.teardown",
    )
    assert events == ["prompt", "service", "listener", "tool", "teardown"]
    assert all(not resources for resources in registry.resources.snapshot().values())
    assert ctx.disposed is True
    assert ctx.dispose().already_disposed is True


def test_dispose_continues_after_cleanup_failure_and_hides_all_capabilities():
    events = []

    def fail_listener():
        events.append("listener-failed")
        raise RuntimeError("listener cleanup exploded")

    def setup(ctx):
        ctx.metadata["teardown"] = lambda: events.append("teardown")
        ctx.register_tool("lookup", object(), dispose=lambda: events.append("tool"))
        ctx.register_listener("updates", object(), dispose=fail_listener)
        ctx.register_prompt_section("rules", "rules", dispose=lambda: events.append("prompt"))

    registry = IntegrationRegistry()
    registry.register(LifecyclePlugin(setup))

    report = registry.unregister("lifecycle")

    assert report is not None and report.ok is False
    assert [failure.name for failure in report.failures] == ["updates"]
    assert "listener cleanup exploded" in report.failures[0].error
    assert events == ["prompt", "listener-failed", "tool", "teardown"]
    assert all(not resources for resources in registry.resources.snapshot().values())


def test_setup_failure_rolls_back_partial_registration_before_publish():
    events = []

    class Service:
        def close(self):
            events.append("service")

    def setup(ctx):
        ctx.metadata["teardown"] = lambda: events.append("teardown")
        ctx.register_tool("partial-tool", object(), dispose=lambda: events.append("tool"))
        ctx.register_service("partial-service", Service())
        raise RuntimeError("setup injection")

    registry = IntegrationRegistry()

    with pytest.raises(RuntimeError, match="setup injection"):
        registry.register(LifecyclePlugin(setup))

    assert registry.get("lifecycle") is None
    assert all(not resources for resources in registry.resources.snapshot().values())
    assert events == ["service", "tool", "teardown"]


def test_resource_collision_rolls_back_only_the_new_plugin():
    class FirstPlugin(LifecyclePlugin):
        plugin_id = "first"

    class SecondPlugin(LifecyclePlugin):
        plugin_id = "second"

    events = []
    registry = IntegrationRegistry()

    def first_setup(ctx):
        ctx.register_tool("shared", "first")

    registry.register(FirstPlugin(first_setup))

    def second_setup(ctx):
        ctx.metadata["teardown"] = lambda: events.append("second-teardown")
        ctx.register_listener("temporary", object(), dispose=lambda: events.append("listener"))
        ctx.register_tool("shared", "second")

    with pytest.raises(PluginResourceConflict, match="already owned"):
        registry.register(SecondPlugin(second_setup))

    assert registry.get("first") is not None
    assert registry.get("second") is None
    assert registry.resources.get(PluginResourceKind.TOOL, "shared") == "first"
    assert registry.resources.get(PluginResourceKind.LISTENER, "temporary") is None
    assert events == ["listener", "second-teardown"]


def test_execution_context_can_read_but_not_register_plugin_resources():
    registry = IntegrationRegistry()

    def setup(ctx):
        ctx.register_service("client", {"ready": True})
        ctx.register_prompt_section("rules", "Plugin rules")

    registry.register(LifecyclePlugin(setup))
    execution = registry.bind_execution_context(
        "lifecycle",
        PluginContext(mission_id="mission-1"),
    )

    assert execution.mission_id == "mission-1"
    assert execution.service("client") == {"ready": True}
    assert execution.prompt_section("rules") == "Plugin rules"
    with pytest.raises(PluginContextDisposed, match="active setup scope"):
        execution.register_tool("late", object())


def test_event_bus_subscription_is_removed_on_unload():
    bus = EventBus()
    registry = IntegrationRegistry()

    def setup(ctx):
        ctx.subscribe_event_bus("events", bus)

    registry.register(LifecyclePlugin(setup))

    assert bus.subscriber_count == 1
    registry.unregister("lifecycle")
    assert bus.subscriber_count == 0


def test_runtime_exposes_idempotent_plugin_shutdown_hook():
    from remy.core_v3.runtime.bootstrap import create_v3_runtime

    runtime = create_v3_runtime()
    loaded = [plugin.plugin_id for plugin in runtime["integration_registry"].all()]

    reports = runtime["dispose"]()

    assert [report.plugin_id for report in reports] == list(reversed(loaded))
    assert runtime["integration_registry"].all() == []
    assert runtime["dispose"]() == []
