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

    assert 'fetch("/api/settings", { method: "POST"' not in js
    assert 'fetch("/api/settings", {\n            method: "PUT"' in js
    assert 'fetch("/api/secrets")' in js
    assert 'fetch(`/api/secrets/${encodeURIComponent(key)}`' in js
    assert 'fetch(`/api/secrets/${encodeURIComponent(key)}/test`' in js
    assert "settings-secret-test" in js
    assert "settings-secret-test-status" in js
    assert "settings-secret-row" in js


def test_index_keeps_heavy_views_lazy_loaded():
    from pathlib import Path

    html = Path("src/remy/web/static/index.html").read_text(encoding="utf-8")
    reliability_js = Path("src/remy/web/static/js/reliability.js").read_text(encoding="utf-8")

    assert '/js/api-client.js' in html
    assert '/js/chat.js' in html
    assert '/js/app.js?v=1.26' in html
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
    ]:
        assert f'/js/{module}' not in html
    assert 'import("./pipelines.js?v=3.0")' in reliability_js
    assert 'import("./automations.js?v=2.8")' in reliability_js


def test_automations_canvas_exposes_per_block_run_results():
    js = Path("src/remy/web/static/js/automations.js").read_text(encoding="utf-8")

    assert "_applyRunTraceToCanvas" in js
    assert "_lastRunTraceByStepId" in js
    assert "pf-node-run-badge" in js
    assert "at-step-modal" in js
    assert 'Completed - ${data.steps_run} step(s)</div>${_renderRunTrace(data.trace || [])}' not in js


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
    assert 'import("./automations.js?v=2.8")' in app_js
    assert "remy_first_run_done_v1" in app_js
    assert "first-run-save-key" in app_js
    assert "_instantiateHomeTemplate" in app_js
    assert "/api/automations/templates/" in app_js
    assert "/api/pipelines/templates/" in app_js
    assert "inputs," in app_js
    assert "remy_pending_automation_open" in app_js
    assert "remy_pending_pipeline_open" in app_js
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
    assert 'import("./automations.js?v=2.8")' in app_js
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
