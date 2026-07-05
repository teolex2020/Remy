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


class GitHubPlugin(BaseIntegrationPlugin):
    plugin_id = "github"
    label = "GitHub"
    auth_mode = AuthMode.OAUTH
    default_mode = ExecutionMode.API
    risk_level = "medium"
    capabilities = (
        PluginCapability("github.create_repo", "Create repository"),
        PluginCapability("github.update_profile_readme", "Update profile README"),
        PluginCapability("github.prepare_signup", "Prepare account signup details"),
    )

    def supports(self, action: str) -> bool:
        return action in {cap.name for cap in self.capabilities}

    def estimate_cost(self, action: str, payload: dict | None = None) -> float:
        return 0.002

    def execute(self, request: PluginRequest, ctx: PluginContext) -> PluginResult:
        if request.action == "github.prepare_signup":
            email = request.payload.get("email", "")
            username = request.payload.get("username", "")
            return PluginResult(
                ok=True,
                message="GitHub signup draft prepared.",
                data={
                    "email": email,
                    "username": username,
                    "next_step": "browser_signup_or_api_import",
                },
            )
        return PluginResult(
            ok=True,
            message=f"{request.action} accepted by GitHub plugin.",
            data={"status": "stub"},
        )

