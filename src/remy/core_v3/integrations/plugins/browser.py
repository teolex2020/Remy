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
from ..router import requires_manual_login


class BrowserPlugin(BaseIntegrationPlugin):
    plugin_id = "browser"
    label = "Browser"
    auth_mode = AuthMode.BROWSER_SESSION
    default_mode = ExecutionMode.BROWSER
    risk_level = "medium"
    capabilities = (
        PluginCapability("browser.open", "Open page"),
        PluginCapability("browser.act", "Interact with page"),
        PluginCapability("browser.login_assisted", "Login with human-assisted flow"),
    )

    def supports(self, action: str) -> bool:
        return action in {cap.name for cap in self.capabilities}

    def estimate_cost(self, action: str, payload: dict | None = None) -> float:
        return 0.003 if action == "browser.act" else 0.001

    def requires_manual_assist(self, request: PluginRequest) -> str | None:
        site = request.payload.get("site", "") or request.payload.get("url", "")
        if requires_manual_login(site, request.action):
            return "Strict login provider detected. Reuse an existing session or perform manual login."
        return None

    def execute(self, request: PluginRequest, ctx: PluginContext) -> PluginResult:
        manual_reason = self.requires_manual_assist(request)
        if manual_reason:
            return PluginResult(
                ok=False,
                status="manual_assist_required",
                message=manual_reason,
                requires_human=True,
                manual_reason=manual_reason,
                data={"site": request.payload.get("site") or request.payload.get("url", "")},
            )
        return PluginResult(
            ok=True,
            message=f"{request.action} accepted by browser plugin.",
            data={"status": "stub", "url": request.payload.get("url", "")},
        )

