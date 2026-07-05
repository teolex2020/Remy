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


class EmailSendPlugin(BaseIntegrationPlugin):
    plugin_id = "email_send"
    label = "Email Send"
    auth_mode = AuthMode.API_KEY
    default_mode = ExecutionMode.API
    risk_level = "medium"
    capabilities = (
        PluginCapability("email.send", "Send outbound email"),
        PluginCapability("email.send_template", "Send templated email"),
    )

    def supports(self, action: str) -> bool:
        return action in {cap.name for cap in self.capabilities}

    def estimate_cost(self, action: str, payload: dict | None = None) -> float:
        return 0.001

    def execute(self, request: PluginRequest, ctx: PluginContext) -> PluginResult:
        return PluginResult(
            ok=True,
            message=f"{request.action} queued via email send plugin.",
            data={"status": "stub", "to": request.payload.get("to", "")},
        )

