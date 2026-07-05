from __future__ import annotations

from ..contracts import (
    AuthMode,
    BaseIntegrationPlugin,
    ExecutionMode,
    PluginCapability,
    PluginContext,
    PluginRequest,
    PluginResult,
)


class TelegramPlugin(BaseIntegrationPlugin):
    plugin_id = "telegram"
    label = "Telegram"
    auth_mode = AuthMode.API_KEY
    default_mode = ExecutionMode.API
    risk_level = "medium"
    capabilities = (
        PluginCapability("telegram.send", "Send Telegram message"),
        PluginCapability("telegram.send_digest", "Send digest"),
        PluginCapability("telegram.request_approval", "Request operator approval"),
    )

    def supports(self, action: str) -> bool:
        return action in {cap.name for cap in self.capabilities}

    def estimate_cost(self, action: str, payload: dict | None = None) -> float:
        return 0.0002

    def execute(self, request: PluginRequest, ctx: PluginContext) -> PluginResult:
        return PluginResult(
            ok=True,
            message=f"{request.action} queued via Telegram plugin.",
            data={"status": "stub", "chat_id": request.payload.get("chat_id")},
        )

