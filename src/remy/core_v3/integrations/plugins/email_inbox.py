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


class EmailInboxPlugin(BaseIntegrationPlugin):
    plugin_id = "email_inbox"
    label = "Email Inbox"
    auth_mode = AuthMode.APP_PASSWORD
    default_mode = ExecutionMode.PROTOCOL
    capabilities = (
        PluginCapability("email.fetch_latest", "Fetch latest inbound email"),
        PluginCapability("email.search", "Search inbox"),
        PluginCapability("email.extract_verification_link", "Extract verification links/codes"),
    )

    def supports(self, action: str) -> bool:
        return action in {cap.name for cap in self.capabilities}

    def estimate_cost(self, action: str, payload: dict | None = None) -> float:
        return 0.0005

    def execute(self, request: PluginRequest, ctx: PluginContext) -> PluginResult:
        if request.action == "email.extract_verification_link":
            return PluginResult(
                ok=True,
                message="Email verification extraction delegated to inbox backend.",
                data={"status": "stub", "query": request.payload.get("query", "")},
            )
        return PluginResult(
            ok=True,
            message=f"{request.action} accepted by inbox plugin.",
            data={"status": "stub", "payload": request.payload},
        )

