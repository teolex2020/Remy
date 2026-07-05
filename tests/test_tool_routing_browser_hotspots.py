def test_format_tool_health_for_prompt_includes_browser_hotspots(tmp_path):
    from remy.core import browser_failure_memory as mem
    from remy.core.tool_routing import format_tool_health_for_prompt

    original_data_dir = mem.settings.DATA_DIR
    mem.settings.DATA_DIR = tmp_path
    try:
        mem.record_browser_failure(
            tool="browser_act",
            action="click",
            url="https://example.com/signup",
            text="captcha challenge",
            status="attempted",
        )
        text = format_tool_health_for_prompt(
            {
                "healthy": ["web_search"],
                "degraded": [],
                "unavailable": [],
                "alternatives": {},
            }
        )
    finally:
        mem.settings.DATA_DIR = original_data_dir

    assert "RECENT BROWSER HOTSPOTS" in text
    assert "example.com" in text
    assert "captcha" in text


def test_format_tool_health_for_prompt_includes_success_playbooks(tmp_path):
    from remy.core import browser_failure_memory as mem
    from remy.core.tool_routing import format_tool_health_for_prompt

    original_data_dir = mem.settings.DATA_DIR
    mem.settings.DATA_DIR = tmp_path
    try:
        mem.record_browser_success(
            tool="browser_act",
            action="click",
            url="https://example.com/dashboard",
            text="Welcome back. Dashboard is ready.",
            status="verified",
        )
        text = format_tool_health_for_prompt(
            {
                "healthy": ["web_search"],
                "degraded": [],
                "unavailable": [],
                "alternatives": {},
            }
        )
    finally:
        mem.settings.DATA_DIR = original_data_dir

    assert "REUSABLE BROWSER PLAYBOOKS" in text
    assert "example.com" in text
    assert "signup" in text
