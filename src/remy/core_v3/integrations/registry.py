"""
Registry of integration plugins.
"""

from __future__ import annotations

from .contracts import BaseIntegrationPlugin, PluginCapability


class IntegrationRegistry:
    def __init__(self):
        self._plugins: dict[str, BaseIntegrationPlugin] = {}

    def register(self, plugin: BaseIntegrationPlugin) -> None:
        self._plugins[plugin.plugin_id] = plugin

    def get(self, plugin_id: str) -> BaseIntegrationPlugin | None:
        return self._plugins.get(plugin_id)

    def all(self) -> list[BaseIntegrationPlugin]:
        return list(self._plugins.values())

    def by_capability(self, capability_name: str) -> list[BaseIntegrationPlugin]:
        return [
            plugin for plugin in self._plugins.values()
            if any(cap.name == capability_name for cap in plugin.capabilities)
        ]

    def capabilities(self) -> list[PluginCapability]:
        caps: list[PluginCapability] = []
        for plugin in self._plugins.values():
            caps.extend(plugin.capabilities)
        return caps

