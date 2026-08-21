"""
Registry of integration plugins.
"""

from __future__ import annotations

import threading

from .contracts import BaseIntegrationPlugin, PluginCapability, PluginContext
from .plugin_context import DisposeReport, PluginResourceRegistry


class IntegrationRegistry:
    def __init__(self, *, resources: PluginResourceRegistry | None = None):
        self._plugins: dict[str, BaseIntegrationPlugin] = {}
        self._contexts: dict[str, PluginContext] = {}
        self._registration_order: list[str] = []
        self._resources = resources or PluginResourceRegistry()
        self._lock = threading.RLock()

    @property
    def resources(self) -> PluginResourceRegistry:
        return self._resources

    def register(self, plugin: BaseIntegrationPlugin) -> PluginContext:
        plugin_id = str(plugin.plugin_id or "").strip()
        if not plugin_id:
            raise ValueError("plugin_id is required")
        with self._lock:
            if plugin_id in self._plugins:
                raise ValueError(f"Plugin '{plugin_id}' is already registered")
            ctx = PluginContext.scoped(plugin_id, self._resources)
            # Teardown is registered first, so it runs last after every resource.
            ctx.defer("plugin.teardown", lambda: plugin.teardown(ctx))
            try:
                setup_cleanup = plugin.setup(ctx)
                if setup_cleanup is not None:
                    if not callable(setup_cleanup):
                        raise TypeError("plugin.setup() must return a cleanup callback or None")
                    ctx.defer("plugin.setup_cleanup", setup_cleanup)
            except Exception as exc:
                report = ctx.dispose()
                if report.failures and hasattr(exc, "add_note"):
                    exc.add_note(
                        "Plugin setup rollback failures: "
                        + "; ".join(failure.error for failure in report.failures)
                    )
                raise
            self._plugins[plugin_id] = plugin
            self._contexts[plugin_id] = ctx
            self._registration_order.append(plugin_id)
            return ctx

    def unregister(self, plugin_id: str) -> DisposeReport | None:
        with self._lock:
            plugin = self._plugins.pop(plugin_id, None)
            ctx = self._contexts.pop(plugin_id, None)
            if plugin is None or ctx is None:
                return None
            try:
                self._registration_order.remove(plugin_id)
            except ValueError:
                pass
        return ctx.dispose()

    def dispose_all(self) -> list[DisposeReport]:
        with self._lock:
            plugin_ids = list(reversed(self._registration_order))
        reports: list[DisposeReport] = []
        for plugin_id in plugin_ids:
            report = self.unregister(plugin_id)
            if report is not None:
                reports.append(report)
        return reports

    def get(self, plugin_id: str) -> BaseIntegrationPlugin | None:
        with self._lock:
            return self._plugins.get(plugin_id)

    def context(self, plugin_id: str) -> PluginContext | None:
        with self._lock:
            return self._contexts.get(plugin_id)

    def bind_execution_context(
        self,
        plugin_id: str,
        ctx: PluginContext,
    ) -> PluginContext:
        with self._lock:
            if plugin_id not in self._plugins:
                raise KeyError(plugin_id)
        return ctx.bind_resource_view(plugin_id, self._resources)

    def all(self) -> list[BaseIntegrationPlugin]:
        with self._lock:
            return list(self._plugins.values())

    def by_capability(self, capability_name: str) -> list[BaseIntegrationPlugin]:
        return [
            plugin for plugin in self.all()
            if any(cap.name == capability_name for cap in plugin.capabilities)
        ]

    def capabilities(self) -> list[PluginCapability]:
        caps: list[PluginCapability] = []
        for plugin in self.all():
            caps.extend(plugin.capabilities)
        return caps
