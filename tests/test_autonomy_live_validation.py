import json


def test_load_live_validation_scenarios_seeds_defaults(tmp_path):
    from remy.core import autonomy_live_validation as live

    original_data_dir = live.settings.DATA_DIR
    live.settings.DATA_DIR = tmp_path
    try:
        scenarios = live.load_live_validation_scenarios()
    finally:
        live.settings.DATA_DIR = original_data_dir

    assert len(scenarios) >= 3
    assert any(item["goal_template"] == "signup_operator" for item in scenarios)
    saved = json.loads(
        (tmp_path / "autonomy_live_validation_scenarios.json").read_text(encoding="utf-8")
    )
    assert saved[0]["name"] == scenarios[0]["name"]


def test_run_live_validation_pack_uses_browser_memory(tmp_path):
    from remy.core import autonomy_live_validation as live
    from remy.core import browser_failure_memory as mem

    original_live_dir = live.settings.DATA_DIR
    original_mem_dir = mem.settings.DATA_DIR
    live.settings.DATA_DIR = tmp_path
    mem.settings.DATA_DIR = tmp_path
    try:
        live.save_live_validation_scenarios(
            [
                {
                    "name": "signup_realish",
                    "category": "signup",
                    "goal_template": "signup_operator",
                    "goal": "Register at https://example.com/signup and verify dashboard access.",
                    "target_url": "https://example.com/signup",
                    "action": "click",
                    "expected_artifact": "dashboard URL",
                    "notes": "",
                }
            ]
        )
        mem.record_browser_success(
            tool="browser_act",
            action="click",
            url="https://example.com/dashboard",
            text="Welcome back. Dashboard is ready.",
            selector="button[type=submit]",
            status="verified",
        )
        report = live.run_live_validation_pack()
    finally:
        live.settings.DATA_DIR = original_live_dir
        mem.settings.DATA_DIR = original_mem_dir

    assert report["summary"]["total"] == 1
    assert report["summary"]["ready"] == 1
    assert report["results"][0]["preferred_selectors"][0]["selector"] == "button[type=submit]"
    assert "Prefer selectors such as button[type=submit]" in report["results"][0]["checklist"][-1]


def test_load_live_validation_report_returns_none_for_invalid_json(tmp_path):
    from remy.core import autonomy_live_validation as live

    original_data_dir = live.settings.DATA_DIR
    live.settings.DATA_DIR = tmp_path
    try:
        (tmp_path / "autonomy_live_validation_report.json").write_text("{invalid", encoding="utf-8")
        report = live.load_live_validation_report()
    finally:
        live.settings.DATA_DIR = original_data_dir

    assert report is None


def test_save_live_validation_scenarios_normalizes_fields(tmp_path):
    from remy.core import autonomy_live_validation as live

    original_data_dir = live.settings.DATA_DIR
    live.settings.DATA_DIR = tmp_path
    try:
        live.save_live_validation_scenarios(
            [
                {
                    "name": "  signup_real  ",
                    "goal_template": "signup_operator",
                    "goal": "  Register account  ",
                    "target_url": " example.com/signup ",
                }
            ]
        )
        scenarios = live.load_live_validation_scenarios()
    finally:
        live.settings.DATA_DIR = original_data_dir

    assert scenarios[0]["name"] == "signup_real"
    assert scenarios[0]["goal"] == "Register account"
    assert scenarios[0]["target_url"] == "example.com/signup"
    assert scenarios[0]["category"] == "other"


def test_save_live_validation_scenarios_deduplicates_names(tmp_path):
    from remy.core import autonomy_live_validation as live

    original_data_dir = live.settings.DATA_DIR
    live.settings.DATA_DIR = tmp_path
    try:
        live.save_live_validation_scenarios(
            [
                {"name": "repeat_name", "goal": "One"},
                {"name": "repeat_name", "goal": "Two"},
                {"goal": "Three"},
                {"goal": "Four"},
            ]
        )
        scenarios = live.load_live_validation_scenarios()
    finally:
        live.settings.DATA_DIR = original_data_dir

    assert [item["name"] for item in scenarios] == [
        "repeat_name",
        "repeat_name_2",
        "scenario_3",
        "scenario_4",
    ]
