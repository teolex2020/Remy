import asyncio


def test_get_live_validation_scenarios_route(tmp_path):
    from remy.core import autonomy_live_validation as live
    from remy.web.routes import autonomy_routes as routes

    original_data_dir = live.settings.DATA_DIR
    live.settings.DATA_DIR = tmp_path
    try:
        data = asyncio.run(routes.get_autonomy_live_validation_scenarios())
    finally:
        live.settings.DATA_DIR = original_data_dir

    assert "scenarios" in data
    assert len(data["scenarios"]) >= 3


def test_save_live_validation_scenarios_route_updates_pack(tmp_path):
    from remy.core import autonomy_live_validation as live
    from remy.core import browser_failure_memory as mem
    from remy.web.routes import autonomy_routes as routes

    original_live_dir = live.settings.DATA_DIR
    original_mem_dir = mem.settings.DATA_DIR
    live.settings.DATA_DIR = tmp_path
    mem.settings.DATA_DIR = tmp_path
    try:
        mem.record_browser_success(
            tool="browser_act",
            action="click",
            url="https://example.com/dashboard",
            text="Welcome back. Dashboard is ready.",
            selector="button[type=submit]",
            status="verified",
        )
        payload = {
            "scenarios": [
                {
                    "name": "signup_realish",
                    "category": "signup",
                    "goal_template": "signup_operator",
                    "goal": "Register at https://example.com/signup",
                    "target_url": "https://example.com/signup",
                    "action": "click",
                    "expected_artifact": "dashboard URL",
                }
            ]
        }
        data = asyncio.run(routes.save_autonomy_live_validation_scenarios(payload))
    finally:
        live.settings.DATA_DIR = original_live_dir
        mem.settings.DATA_DIR = original_mem_dir

    assert data["scenarios"][0]["name"] == "signup_realish"
    assert data["report"]["summary"]["total"] == 1
    assert data["report"]["results"][0]["status"] == "ready"


def test_save_live_validation_scenarios_route_deduplicates_names(tmp_path):
    from remy.core import autonomy_live_validation as live
    from remy.web.routes import autonomy_routes as routes

    original_data_dir = live.settings.DATA_DIR
    live.settings.DATA_DIR = tmp_path
    try:
        payload = {
            "scenarios": [
                {"name": "dup", "goal": "One"},
                {"name": "dup", "goal": "Two"},
            ]
        }
        data = asyncio.run(routes.save_autonomy_live_validation_scenarios(payload))
    finally:
        live.settings.DATA_DIR = original_data_dir

    assert [item["name"] for item in data["scenarios"]] == ["dup", "dup_2"]
