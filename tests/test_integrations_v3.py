import tempfile


def _build_gateway():
    from remy.core_v3.governance.approval_engine import ApprovalEngine
    from remy.core_v3.governance.audit_engine import AuditEngine
    from remy.core_v3.governance.budget_engine import BudgetConfig, BudgetEngine
    from remy.core_v3.governance.policy_engine import PolicyEngine
    from remy.core_v3.integrations import AuthStore, IntegrationGateway, IntegrationRegistry
    from remy.core_v3.integrations.plugins.browser import BrowserPlugin
    from remy.core_v3.integrations.plugins.email_inbox import EmailInboxPlugin
    from remy.core_v3.integrations.plugins.email_send import EmailSendPlugin
    from remy.core_v3.integrations.plugins.github import GitHubPlugin
    from remy.core_v3.integrations.plugins.telegram import TelegramPlugin

    registry = IntegrationRegistry()
    registry.register(EmailInboxPlugin())
    registry.register(EmailSendPlugin())
    registry.register(GitHubPlugin())
    registry.register(TelegramPlugin())
    registry.register(BrowserPlugin())
    audit = AuditEngine(log_path=tempfile.mktemp(suffix=".jsonl"))
    gateway = IntegrationGateway(
        registry=registry,
        policy=PolicyEngine(),
        budget=BudgetEngine(config=BudgetConfig(daily_usd=1.0, per_cycle_usd=1.0)),
        approval=ApprovalEngine(),
        audit=audit,
        auth_store=AuthStore(),
    )
    return gateway, registry, audit


class TestIntegrationRegistry:
    def test_registry_by_capability(self):
        _, registry, _ = _build_gateway()
        plugins = registry.by_capability("email.fetch_latest")
        assert len(plugins) == 1
        assert plugins[0].plugin_id == "email_inbox"


class TestIntegrationGateway:
    def test_email_send_requires_approval(self):
        from remy.core_v3.integrations import IntegrationDecision, PluginRequest

        gateway, _, audit = _build_gateway()
        outcome = gateway.execute(
            "email_send",
            PluginRequest(action="email.send", payload={"to": "x@example.com"}),
        )
        assert outcome.decision == IntegrationDecision.APPROVAL_REQUIRED
        assert outcome.approval_id
        assert audit.recent(1)[0].event_type != "policy_violation"

    def test_browser_google_login_requires_manual_assist(self):
        from remy.core_v3.integrations import IntegrationDecision, PluginRequest

        gateway, _, audit = _build_gateway()
        outcome = gateway.execute(
            "browser",
            PluginRequest(
                action="browser.login_assisted",
                payload={"site": "https://accounts.google.com/signin"},
            ),
        )
        assert outcome.decision == IntegrationDecision.MANUAL_ASSIST_REQUIRED
        assert outcome.result is not None
        assert outcome.result.requires_human is True
        assert audit.recent(1)[0].event_type == "policy_violation"

    def test_github_prepare_signup_runs(self):
        from remy.core_v3.integrations import IntegrationDecision, PluginRequest

        gateway, _, _ = _build_gateway()
        outcome = gateway.execute(
            "github",
            PluginRequest(
                action="github.prepare_signup",
                payload={"email": "remy@example.com", "username": "remy-ai"},
            ),
        )
        assert outcome.decision == IntegrationDecision.ALLOW
        assert outcome.result is not None
        assert outcome.result.data["username"] == "remy-ai"

    def test_budget_records_successful_integration(self):
        from remy.core_v3.integrations import PluginContext, PluginRequest

        gateway, _, _ = _build_gateway()
        start_spend = gateway.budget.state.daily_spent_usd
        outcome = gateway.execute(
            "email_inbox",
            PluginRequest(action="email.fetch_latest"),
            ctx=PluginContext(mission_id="m1"),
        )
        assert outcome.result is not None and outcome.result.ok
        assert gateway.budget.state.daily_spent_usd > start_spend
        assert gateway.budget.state.mission_spent["m1"] > 0


class TestBootstrapIntegration:
    def test_bootstrap_exposes_integrations(self):
        from remy.core_v3.runtime.bootstrap import create_v3_runtime

        runtime = create_v3_runtime()
        assert "integrations" in runtime
        assert "integration_registry" in runtime
        assert runtime["integration_registry"].get("browser") is not None
