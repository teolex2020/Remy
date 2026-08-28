"""Contract tests for the local desktop app production model."""

import importlib
import json
import re
import socket
from pathlib import Path
from unittest.mock import patch


def _route_keys(app):
    keys = []
    for route in app.router.routes:
        methods = tuple(sorted(getattr(route, "methods", []) or []))
        keys.append((type(route).__name__, getattr(route, "path", ""), methods))
    return keys


def _ui_block_types(path: str) -> set[str]:
    js = Path(path).read_text(encoding="utf-8")
    catalog = js.split("const BLOCKS = [", 1)[1].split("];", 1)[0]
    return set(re.findall(r'type:\s*"([^"]+)"', catalog))


def test_create_app_has_no_web_auth_surface():
    from remy.core.desktop_gui import create_app

    app = create_app()
    paths = {getattr(route, "path", "") for route in app.router.routes}

    assert "/api/login" not in paths
    assert "/api/logout" not in paths
    assert "/api/check-auth" not in paths
    assert "/login.html" not in paths

    middleware_names = {mw.cls.__name__ for mw in app.user_middleware}
    assert "AuthMiddleware" not in middleware_names


def test_create_app_keeps_routes_deduplicated():
    from remy.core.desktop_gui import create_app

    app = create_app()
    keys = [key for key in _route_keys(app) if key[1]]

    assert len(keys) == len(set(keys))


def test_project_api_exposes_reversible_lifecycle():
    from fastapi.testclient import TestClient

    from remy.core.desktop_gui import create_app

    client = TestClient(create_app())

    assert client.get("/api/projects").status_code == 200
    assert client.post("/api/projects/project-does-not-exist/restore").status_code == 404


def test_create_app_exposes_frontend_required_api_endpoints():
    from fastapi.testclient import TestClient

    from remy.core.desktop_gui import create_app

    client = TestClient(create_app())

    assert client.get("/api/ping").status_code == 200
    assert client.get("/api/settings").status_code == 200
    secrets = client.get("/api/secrets")
    assert secrets.status_code == 200
    assert "secrets" in secrets.json()
    assert client.get("/api/model-registry").status_code == 200
    assert client.get("/api/chat/brain-voice").status_code == 200
    assert client.get("/api/pipelines/home-templates/runs").status_code == 200
    home_run = client.post(
        "/api/pipelines/home-templates/run",
        json={
            "template_id": "daily-brief",
            "title": "Create Daily Brief",
            "pack": "Personal Admin Pack",
            "mode": "dry_run",
            "inputs": {"time": "09:00", "scope": "tasks"},
            "steps": ["Search tasks", "Draft brief"],
        },
    )
    assert home_run.status_code == 200
    assert home_run.json()["run"]["status"] == "dry_run_ready"
    assert client.post("/api/workflows/http-test", json={"url": "not-a-url"}).status_code == 400
    assert client.post("/api/workflows/scrape-test", json={"url": "not-a-url"}).status_code == 400
    assert client.delete("/api/pipelines/home-templates/runs").status_code == 200
    assert client.post("/api/end-session").status_code == 200


def test_web_ui_exposes_graceful_stop_control():
    html = Path("src/remy/web/static/index.html").read_text(encoding="utf-8")
    app_js = Path("src/remy/web/static/js/app.js").read_text(encoding="utf-8")

    assert 'id="btn-stop-remy"' in html
    assert 'window.apiClient.shutdownServer()' in app_js
    assert 'server-shutdown-started' in app_js


def test_chat_header_omits_unused_optimization_controls():
    html = Path("src/remy/web/static/index.html").read_text(encoding="utf-8")

    assert 'id="context-reducer-compare"' not in html
    assert 'id="context-reducer-apply"' not in html
    assert 'id="btn-context-reducer-lab"' not in html
    assert 'id="btn-compare"' in html
    assert 'id="tts-enabled"' in html


def test_chat_composer_groups_context_tools_model_and_send_action():
    html = Path("src/remy/web/static/index.html").read_text(encoding="utf-8")
    css = Path("src/remy/web/static/css/main.css").read_text(encoding="utf-8")
    chat_js = Path("src/remy/web/static/js/chat.js").read_text(encoding="utf-8")

    assert 'class="chat-composer-context"' in html
    assert 'class="chat-composer-shell"' in html
    assert 'class="input-row chat-composer-toolbar"' in html
    assert 'class="chat-composer-tools"' in html
    assert 'class="chat-composer-routing"' in html
    assert 'placeholder="Describe what you want Remy to do…"' in html
    assert html.index('id="project-context-selector"') < html.index('id="chat-input"')
    assert html.index('id="chat-input"') < html.index('id="chat-model-select"')
    assert html.index('id="chat-model-select"') < html.index('id="btn-send"')
    assert 'class="chat-send-button"' in html
    assert ".chat-composer-shell:focus-within" in css
    assert ".chat-model-select option" in css
    assert "color-scheme: dark" in css
    assert '[data-theme="light"] .chat-model-select option' in css
    assert "@media (max-width: 760px)" in css
    assert 'sendBtn.textContent = "Queue"' not in chat_js
    assert 'sendBtn.setAttribute("aria-label", "Queue message")' in chat_js


def test_server_shutdown_endpoint_requests_combined_runner_shutdown():
    from fastapi.testclient import TestClient

    from remy.core.desktop_gui import create_app

    with patch("remy.core.combined_runner.request_graceful_shutdown", return_value=True) as request_shutdown:
        response = TestClient(create_app()).post("/api/server/shutdown")

    assert response.status_code == 200
    assert response.json()["ok"] is True
    request_shutdown.assert_called_once_with()


def test_local_secret_vault_saves_and_clears_runtime_secret(tmp_path):
    from fastapi.testclient import TestClient

    from remy.core.desktop_gui import create_app

    runtime_file = tmp_path / "runtime_settings.json"
    client = TestClient(create_app())

    with patch("remy.config.settings.RUNTIME_SETTINGS_FILE", runtime_file):
        saved = client.put("/api/secrets/openrouter_api_key", json={"value": "sk-test-secret"})
        assert saved.status_code == 200
        assert saved.json()["secret"]["configured"] is True
        assert "sk-test-secret" not in saved.text
        assert json.loads(runtime_file.read_text(encoding="utf-8"))["OPENROUTER_API_KEY"] == "sk-test-secret"

        listed = client.get("/api/secrets")
        assert listed.status_code == 200
        assert "sk-test-secret" not in listed.text

        unknown_test = client.post("/api/secrets/not_a_secret/test")
        assert unknown_test.status_code == 404

        cleared = client.put("/api/secrets/openrouter_api_key", json={"value": ""})
        assert cleared.status_code == 200
        assert cleared.json()["secret"]["configured"] is False
        assert "OPENROUTER_API_KEY" not in json.loads(runtime_file.read_text(encoding="utf-8"))


def test_desktop_app_wires_all_split_route_modules():
    from remy.core.desktop_gui import ROUTE_MODULES

    routes_dir = Path("src/remy/web/routes")
    expected_modules = set()
    for path in routes_dir.glob("*.py"):
        if path.name.startswith("_") or path.name == "__init__.py":
            continue

        module_name = f"remy.web.routes.{path.stem}"
        module = importlib.import_module(module_name)
        if hasattr(module, "router"):
            expected_modules.add(module_name)

    assert set(ROUTE_MODULES) == expected_modules


def test_create_app_registers_runtime_lifecycle_once():
    from fastapi.testclient import TestClient

    from remy.core.desktop_gui import create_app

    with patch("remy.core.desktop_gui.start_scheduler") as start_scheduler, \
         patch("remy.core.desktop_gui.load_push_subscription") as load_push_subscription, \
         patch("remy.core.desktop_gui.shutdown_cleanup") as shutdown_cleanup:
        with TestClient(create_app()):
            start_scheduler.assert_awaited_once()
            load_push_subscription.assert_awaited_once()
            shutdown_cleanup.assert_not_awaited()

        shutdown_cleanup.assert_awaited_once()


def test_web_host_is_forced_to_localhost_for_desktop_security():
    from remy.config.settings import Settings

    assert Settings(WEB_HOST="0.0.0.0").WEB_HOST == "127.0.0.1"
    assert Settings(WEB_HOST="192.168.1.5").WEB_HOST == "127.0.0.1"
    assert Settings(WEB_HOST="").WEB_HOST == "127.0.0.1"
    assert Settings(WEB_HOST="localhost").WEB_HOST == "localhost"
    assert Settings(WEB_HOST="::1").WEB_HOST == "::1"


def test_settings_ui_uses_put_for_settings_updates():
    from pathlib import Path

    js = Path("src/remy/web/static/js/settings.js").read_text(encoding="utf-8")
    app_js = Path("src/remy/web/static/js/app.js").read_text(encoding="utf-8")
    css = Path("src/remy/web/static/css/main.css").read_text(encoding="utf-8")

    assert 'fetch("/api/settings", { method: "POST"' not in js
    assert 'fetch("/api/settings", {\n            method: "PUT"' in js
    assert 'fetch("/api/secrets")' in js
    assert 'fetch(`/api/secrets/${encodeURIComponent(key)}`' in js
    assert 'fetch(`/api/secrets/${encodeURIComponent(key)}/test`' in js
    assert "settings-secret-test" in js
    assert "settings-secret-test-status" in js
    assert "settings-secret-row" in js
    assert "SETTINGS_CATEGORIES" in js
    assert "buildSettingsNavigation" in js
    assert "data-settings-category" in js
    assert "data-settings-panel" in js
    assert 'localStorage.getItem("remy.settings.category")' in js
    assert 'import("./settings.js?v=1.33")' in app_js
    assert ".settings-hub-tabs" in css
    assert ".settings-hub-panel[hidden]" in css


def test_index_keeps_heavy_views_lazy_loaded():
    from pathlib import Path

    html = Path("src/remy/web/static/index.html").read_text(encoding="utf-8")
    reliability_js = Path("src/remy/web/static/js/reliability.js").read_text(encoding="utf-8")

    assert '/js/api-client.js' in html
    assert '/js/chat.js' in html
    assert '/js/app.js?v=1.76' in html
    assert 'id="first-run-wizard"' in html
    for module in [
        "memory.js",
        "tasks.js",
        "stats.js",
        "settings.js",
        "history.js",
        "activity.js",
        "reliability.js",
        "approval.js",
        "guidance.js",
        "knowledge.js",
        "experiments.js",
    ]:
        assert f'/js/{module}' not in html
    assert 'import("./pipelines.js?v=3.0")' in reliability_js
    assert 'import("./automations.js?v=2.9")' in reliability_js


def test_automations_canvas_exposes_per_block_run_results():
    js = Path("src/remy/web/static/js/automations.js").read_text(encoding="utf-8")

    assert "_applyRunTraceToCanvas" in js
    assert "_lastRunTraceByStepId" in js
    assert "pf-node-run-badge" in js
    assert "at-step-modal" in js
    assert 'Completed - ${data.steps_run} step(s)</div>${_renderRunTrace(data.trace || [])}' not in js


def test_experiment_canvas_is_lazy_and_exposes_required_research_blocks():
    html = Path("src/remy/web/static/index.html").read_text(encoding="utf-8")
    app_js = Path("src/remy/web/static/js/app.js").read_text(encoding="utf-8")
    experiments_js = Path("src/remy/web/static/js/experiments.js").read_text(encoding="utf-8")
    css = Path("src/remy/web/static/css/main.css").read_text(encoding="utf-8")

    assert 'data-view="experiments"' in html
    assert 'id="view-experiments"' in html
    assert 'import("./experiments.js?v=1.15")' in app_js
    for node_type in ("problem", "role_model", "discussion_group", "shared_board", "success_gate", "synthesis"):
        assert f'type: "{node_type}"' in experiments_js
    assert "/api/experiments/canvas/validate" in experiments_js
    assert 'id="exp-edit"' in experiments_js
    assert 'id="exp-delete"' in experiments_js
    assert 'method: "DELETE"' in experiments_js
    assert 'method: existingId ? "PUT" : "POST"' in experiments_js
    assert 'id="exp-canvas-workspace"' in experiments_js
    assert "grid-template-columns:220px minmax(0,1fr) 0px" in experiments_js
    assert "closeCanvasConfigPanel" in experiments_js
    assert "/continue`" in experiments_js
    assert "Continue this experiment" in experiments_js
    assert "Durable checkpoint:" in experiments_js
    assert "/pause`" in experiments_js
    assert "/resume`" in experiments_js
    assert "Require my approval before final synthesis" in experiments_js
    assert "Choose responsibility" in experiments_js
    assert "What this role contributes" in experiments_js
    assert "Custom role" in experiments_js
    assert "/api/experiments/roles" in experiments_js
    assert "proposed next task · not executed" in experiments_js
    assert "Self-improvement Lab" in experiments_js
    assert "/api/experiments/self-modifications" in experiments_js
    assert "/run-evaluation`" in experiments_js
    assert "Approve exact hash" in experiments_js
    assert "Refresh & enforce gate" in experiments_js
    assert "/canary-telemetry/observe" in experiments_js
    assert "aggregate-only; prompts and responses excluded" in experiments_js
    assert "statistical confidence" in experiments_js
    assert "minimum_observation_window" in experiments_js
    assert "statisticalChecks.failure_rate_non_inferior" in experiments_js
    assert "/api/experiments/self-modifications/policy" in experiments_js
    assert "Save project policy" in experiments_js
    assert "self-mod-live-alerts" in experiments_js
    assert "open_self_modification_lab" in app_js
    assert "/api/trajectory/self-modifications/" in experiments_js
    assert ".self-mod-progress" in css
    assert "#experiments-content {" in css
    assert "overflow-y: auto;" in css.split("#experiments-content {", 1)[1].split("}", 1)[0]
    assert '/css/main.css?v=1.64' in html


def test_sidebar_exposes_project_microbrain_switcher():
    html = Path("src/remy/web/static/index.html").read_text(encoding="utf-8")
    app_js = Path("src/remy/web/static/js/app.js").read_text(encoding="utf-8")
    api_js = Path("src/remy/web/static/js/api-client.js").read_text(encoding="utf-8")

    assert 'id="project-select"' in html
    assert 'id="btn-new-project"' in html
    assert 'id="btn-manage-projects"' in html
    assert 'id="project-manager-modal"' in html
    assert 'id="project-manager-active-list"' in html
    assert 'id="project-manager-archived-list"' in html
    assert "Only the project name is required." in html
    assert "shared project memory" in app_js
    assert "activateProject(nextProjectId)" in app_js
    assert "updateProject(project.project_id" in app_js
    assert "archiveProject(project.project_id)" in app_js
    assert "restoreProject(project.project_id)" in app_js
    assert '"/api/projects"' in api_js
    assert "/activate`" in api_js
    assert "/restore`" in api_js
    assert "formatApiErrorDetail" in api_js
    assert 'const body = { name, activate };' in api_js
    assert "if (domain) body.domain = domain;" in api_js
    assert "hasLegacyProjectProfileValidation" in api_js


def test_activity_exposes_durable_background_research_controls():
    app_js = Path("src/remy/web/static/js/app.js").read_text(encoding="utf-8")
    activity_js = Path("src/remy/web/static/js/activity.js").read_text(encoding="utf-8")

    assert 'import("./activity.js?v=1.23")' in app_js
    assert "refreshBackgroundResearch" in activity_js
    assert "getKnowledgeResearch" in activity_js
    assert "background_research" in activity_js
    assert "pauseResearch(projectId)" in activity_js
    assert "resumeResearch(projectId)" in activity_js
    assert "checkpoint ${checkpoint.node}" in activity_js


def test_sidebar_is_compact_and_grouped_by_workflow():
    html = Path("src/remy/web/static/index.html").read_text(encoding="utf-8")
    css = Path("src/remy/web/static/css/main.css").read_text(encoding="utf-8")
    app_js = Path("src/remy/web/static/js/app.js").read_text(encoding="utf-8")

    positions = [
        html.index('<li class="nav-group-label">Work</li>'),
        html.index('<li class="nav-group-label">Build</li>'),
        html.index('<li class="nav-group-label">Knowledge</li>'),
        html.index('<details id="sidebar-insights">'),
        html.index('<li class="nav-group-label">System</li>'),
    ]
    assert positions == sorted(positions)
    assert 'data-view="home"' not in html
    assert 'id="view-home"' not in html
    assert "HOME_TEMPLATES" not in app_js
    assert "initHomeSurface" not in app_js
    assert '<li class="nav-item active" data-view="chat">' in html
    assert 'switchView("chat")' in app_js
    assert '<span class="nav-label">Memory Map</span>' in html
    assert '<span class="nav-label">Analytics</span>' in html
    assert "SIDEBAR_INSIGHTS_OPEN_KEY" in app_js
    assert 'item.setAttribute("aria-current", "page")' in app_js
    assert 'item.setAttribute("tabindex", "0")' in app_js
    assert "overflow-y: auto;" in css.split(".nav-list {", 1)[1].split("}", 1)[0]
    assert "min-height: 31px;" in css.split(".nav-item {", 1)[1].split("}", 1)[0]
    assert "--sidebar-width-default: 224px;" in css
    assert 'id="sidebar-resize-handle"' in html
    assert 'role="separator"' in html
    assert "SIDEBAR_WIDTH_KEY" in app_js
    assert "_initSidebarResize" in app_js
    assert 'window.localStorage.setItem(SIDEBAR_WIDTH_KEY' in app_js
    assert "cursor: col-resize;" in css
    assert "getOperatorAlerts" in app_js
    assert "getSystemStatus" not in app_js.split("async function _refreshOperatorAlerts", 1)[1].split("}", 1)[0]
    assert 'const CACHE_NAME = "remy-v1.86"' in Path(
        "src/remy/web/static/sw.js"
    ).read_text(encoding="utf-8")


def test_glass_brain_exposes_search_details_and_coverage_audit():
    js = Path("src/remy/web/static/js/glass_brain.js").read_text(encoding="utf-8")
    css = Path("src/remy/web/static/css/main.css").read_text(encoding="utf-8")
    app_js = Path("src/remy/web/static/js/app.js").read_text(encoding="utf-8")

    assert 'fetch("/api/graph?mode=full")' in js
    assert "Coverage audit" in js
    assert "gb-search" in js
    assert "gb-level-filter" in js
    assert "gb-edge-limit" in js
    assert "_showNodeDetail" in js
    assert "Thermal mapping" in js
    assert "_localNodeIds" in js
    assert "Local Brain" in js
    assert "_setGraphHighlight" in js
    assert "_LAYOUT_STORAGE_KEY" in js
    assert "_saveCurrentLayout" in js
    assert "Reset layout" in js
    assert ".gb-graph-tools" in css
    assert ".gb-node-detail-overlay" in css
    assert ".gb-local-mode" in css
    assert ".gb-local-actions" in css
    assert 'import("./glass_brain.js?v=1.6")' in app_js


def test_automations_editor_uses_full_height_resizable_workspace():
    js = Path("src/remy/web/static/js/automations.js").read_text(encoding="utf-8")
    css = Path("src/remy/web/static/css/main.css").read_text(encoding="utf-8")

    assert "at-config-resize" in js
    assert "_bindConfigResize" in js
    assert "#view-automations" in css
    assert "#automations-content" in css
    assert ".pf-config-panel { position: absolute;" in css
    assert "#at-palette-panel { flex: 1; min-height: 0; overflow-y: auto;" in css


def test_automations_config_opens_from_node_button_not_selection():
    js = Path("src/remy/web/static/js/automations.js").read_text(encoding="utf-8")
    app_js = Path("src/remy/web/static/js/app.js").read_text(encoding="utf-8")
    css = Path("src/remy/web/static/css/main.css").read_text(encoding="utf-8")

    assert "pf-node-config-btn" in js
    assert "_openNodeConfig" in js
    assert "_ensureNodeConfigButtons" in js
    assert "_editor.on(\"nodeSelected\",   id => { _selectedNodeId = String(id); });" in js
    assert "_editor.on(\"nodeSelected\",   id => _openNodeConfig(id));" not in js
    assert 'import("./automations.js?v=2.9")' in app_js
    assert "remy_first_run_done_v1" in app_js
    assert "first-run-save-key" in app_js
    assert "_instantiateHomeTemplate" not in app_js
    assert "pf-node-config-btn" in css
    assert "_deleteNodeFromCanvas" in js
    assert 'deleteNodeButton?.addEventListener("pointerdown"' in js
    assert "data-safety-report" in js
    assert "_confirmAutomationPreflight" in js
    assert "auth_secret_key" in js
    assert "Authorization secret" in js
    assert "Authorization Not set" in js
    assert "_httpAuthReadinessHtml" in js
    assert "_testHttpConnection" in js
    assert "Test Connection" in js
    assert "/api/workflows/http-test" in js
    assert "_testPageScrape" in js
    assert "Test Scrape" in js
    assert "/api/workflows/scrape-test" in js
    assert "at-save-template-btn" in js
    assert "/api/automations/templates" in js
    assert "at-template-del-btn" in js
    assert "_openPendingAutomationFromHome" in js
    assert "remy_pending_automation_open" in js
    assert "source_template_name" in js
    assert "From ${_esc(a.source_template_name)}" in js
    assert "_renderAutomationTemplateChip" in js
    assert "pf-source-template-chip" in js
    assert 'method: "DELETE"' in js
    assert "Custom" in js
    assert 'fetch("/api/secrets")' in js
    assert "_outputReadinessHtml" in js
    assert "Telegram Not set" in js
    assert "Email Not set" in js
    assert "pf-output-readiness" in css


def test_pipelines_editor_matches_canvas_debugging_contract():
    js = Path("src/remy/web/static/js/pipelines.js").read_text(encoding="utf-8")
    app_js = Path("src/remy/web/static/js/app.js").read_text(encoding="utf-8")
    css = Path("src/remy/web/static/css/main.css").read_text(encoding="utf-8")

    assert "pf-node-config-btn" in js
    assert "_openNodeConfig" in js
    assert "_editor.on(\"nodeSelected\", id => { _selectedNodeId = String(id); });" in js
    assert "_editor.on(\"nodeSelected\", id => _openNodeConfig(id));" not in js
    assert "_applyPipelineStepEventToCanvas" in js
    assert "_applyResultBadge" in js
    assert "pf-step-modal" in js
    assert "_deleteNodeFromCanvas" in js
    assert 'deleteNodeButton?.addEventListener("pointerdown"' in js
    assert 'import("./pipelines.js?v=3.0")' in app_js
    assert "data-safety-report" in js
    assert "_confirmPipelinePreflight" in js
    assert "#pipelines-content" in css
    assert "pf-node-run-running" in css
    assert "auth_secret_key" in js
    assert "Authorization secret" in js
    assert "Authorization Not set" in js
    assert "_httpAuthReadinessHtml" in js
    assert "_testHttpConnection" in js
    assert "Test Connection" in js
    assert "/api/workflows/http-test" in js
    assert "_testPageScrape" in js
    assert "Test Scrape" in js
    assert "/api/workflows/scrape-test" in js
    assert "pf-save-template-btn" in js
    assert "/api/pipelines/templates" in js
    assert "pf-template-del-btn" in js
    assert "_openPendingPipelineFromHome" in js
    assert "remy_pending_pipeline_open" in js
    assert "source_template_name" in js
    assert "From ${_esc(p.source_template_name)}" in js
    assert "_renderPipelineTemplateChip" in js
    assert "pf-source-template-chip" in js
    assert 'method: "DELETE"' in js
    assert "Custom" in js
    assert 'fetch("/api/secrets")' in js


def test_router_block_is_available_in_pipelines_and_automations():
    pipelines_js = Path("src/remy/web/static/js/pipelines.js").read_text(encoding="utf-8")
    automations_js = Path("src/remy/web/static/js/automations.js").read_text(encoding="utf-8")
    app_js = Path("src/remy/web/static/js/app.js").read_text(encoding="utf-8")

    assert 'type: "router"' in pipelines_js
    assert 'type: "router"' in automations_js
    assert 'type: "merge"' in pipelines_js
    assert 'type: "merge"' in automations_js
    for block_type in [
        "delay",
        "filter",
        "set_variable",
        "parse_json",
        "transform",
        "notification",
        "file_read",
        "file_write",
        "code",
        "error_handler",
    ]:
        assert f'type: "{block_type}"' in pipelines_js
        assert f'type: "{block_type}"' in automations_js
    assert "function _routerRoutes" in pipelines_js
    assert "function _routerRoutes" in automations_js
    assert "function _routerOperatorOptions" in pipelines_js
    assert "function _routerOperatorOptions" in automations_js
    assert "function _syncMergeInputs" in pipelines_js
    assert "function _syncMergeInputs" in automations_js
    assert "function _blockHelpHtml" in pipelines_js
    assert "function _blockHelpHtml" in automations_js
    assert "pf-block-help" in Path("src/remy/web/static/css/main.css").read_text(encoding="utf-8")
    assert 'data-route-key="operator"' in pipelines_js
    assert 'data-route-key="operator"' in automations_js
    assert 'data-route-key="value"' in pipelines_js
    assert 'data-route-key="value"' in automations_js
    assert "pf-router-add-route" in pipelines_js
    assert "pf-router-add-route" in automations_js
    assert "Selected routes:" in Path("src/remy/core/pipeline_runner.py").read_text(encoding="utf-8")
    assert 'BLOCKS.filter(b => b.type !== "condition")' in pipelines_js
    assert "pf-history-btn" in pipelines_js
    assert "at-history-btn" in automations_js
    assert "pf-step-pin" in pipelines_js
    assert "at-step-pin" in automations_js
    assert "pf-history-rerun" in pipelines_js
    assert "at-history-rerun" in automations_js
    assert "pf-history-copy-output" in pipelines_js
    assert "at-history-copy-output" in automations_js
    assert "Copy output" in pipelines_js
    assert "Copy output" in automations_js
    assert "_retry_enabled" in automations_js
    assert "function _blockOutputCount" in pipelines_js
    assert "function _blockOutputCount" in automations_js
    assert "function _ensureErrorOutputs" in pipelines_js
    assert "function _ensureErrorOutputs" in automations_js
    assert "fallback_text" in pipelines_js
    assert "fallback_text" in automations_js
    assert "allow_local_execution" in pipelines_js
    assert "allow_local_execution" in automations_js
    assert "safe_expression" in pipelines_js
    assert "safe_expression" in automations_js
    assert "Local script on this computer" in pipelines_js
    assert "Local script on this computer" in automations_js
    assert '"merge",' in Path("src/remy/core/workflow_validation.py").read_text(encoding="utf-8")
    assert '"merge": _run_merge' in Path("src/remy/core/pipeline_runner.py").read_text(encoding="utf-8")
    assert '"parse_json": _run_parse_json' in Path("src/remy/core/pipeline_runner.py").read_text(encoding="utf-8")
    assert '"file_write": _run_file_write' in Path("src/remy/core/pipeline_runner.py").read_text(encoding="utf-8")
    assert "_pinned_enabled" in pipelines_js
    assert "_pinned_enabled" in automations_js
    assert "/runs" in pipelines_js
    assert "/runs" in automations_js
    assert 'import("./pipelines.js?v=3.0")' in app_js
    assert 'import("./automations.js?v=2.9")' in app_js
    assert "data-safety-report" in pipelines_js
    assert "data-safety-report" in automations_js


def test_canvas_block_catalogues_match_backend_execution_contract():
    from remy.core import pipeline_runner
    from remy.core.workflow_validation import SUPPORTED_WORKFLOW_STEP_TYPES
    from remy.web.routes.automation_routes import ALLOWED_STEP_TYPES, ERROR_PREFIXES as AUTOMATION_ERROR_PREFIXES
    from remy.web.routes.pipeline_routes import ALLOWED_PIPELINE_STEP_TYPES

    pipeline_blocks = _ui_block_types("src/remy/web/static/js/pipelines.js")
    automation_blocks = _ui_block_types("src/remy/web/static/js/automations.js")
    runner_blocks = set(pipeline_runner._RUNNERS)

    assert pipeline_blocks <= ALLOWED_PIPELINE_STEP_TYPES
    assert automation_blocks <= ALLOWED_STEP_TYPES
    assert pipeline_blocks <= runner_blocks
    assert automation_blocks <= runner_blocks
    assert pipeline_blocks <= SUPPORTED_WORKFLOW_STEP_TYPES
    assert automation_blocks <= SUPPORTED_WORKFLOW_STEP_TYPES
    assert set(pipeline_runner.ERROR_PREFIXES) <= set(AUTOMATION_ERROR_PREFIXES)


def test_desktop_static_assets_resolve_to_index_html():
    from remy.core.desktop_gui import _static_dir

    static_dir = _static_dir()

    assert static_dir.exists()
    assert (static_dir / "index.html").exists()


def test_desktop_port_falls_back_when_preferred_port_is_busy():
    from remy.core.desktop_gui import _choose_web_port

    host = "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        sock.listen(1)
        busy_port = sock.getsockname()[1]

        chosen = _choose_web_port(host, busy_port, attempts=5)

    assert chosen != busy_port
    assert busy_port < chosen <= busy_port + 4
