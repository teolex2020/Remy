import json


def test_record_browser_failure_clusters_by_domain_and_signature(tmp_path):
    from remy.core import browser_failure_memory as mem

    original_data_dir = mem.settings.DATA_DIR
    mem.settings.DATA_DIR = tmp_path
    try:
        mem.record_browser_failure(
            tool="browser_act",
            action="click",
            url="https://example.com/signup",
            text="Please correct the required fields",
            status="attempted",
        )
        mem.record_browser_failure(
            tool="browser_act",
            action="click",
            url="https://example.com/signup",
            text="Please correct the required fields",
            status="attempted",
        )
        report = mem.get_browser_failure_report(limit=5)
    finally:
        mem.settings.DATA_DIR = original_data_dir

    assert report["total_clusters"] == 1
    assert report["total_failures"] == 2
    assert report["hottest_domain"] == "example.com"
    assert report["top_clusters"][0]["signature"] == "validation_error"
    saved = json.loads((tmp_path / "browser_failure_memory.json").read_text(encoding="utf-8"))
    assert saved[0]["count"] == 2


def test_browser_failure_report_summarizes_multiple_signatures(tmp_path):
    from remy.core import browser_failure_memory as mem

    original_data_dir = mem.settings.DATA_DIR
    mem.settings.DATA_DIR = tmp_path
    try:
        mem.record_browser_failure(
            tool="browse_page",
            url="https://example.com/signup",
            text="captcha challenge",
            status="attempted",
        )
        mem.record_browser_failure(
            tool="browser_act",
            action="click",
            url="https://another.com/publish",
            text="Check your email to verify your account",
            status="attempted",
        )
        report = mem.get_browser_failure_report(limit=5)
    finally:
        mem.settings.DATA_DIR = original_data_dir

    assert report["total_clusters"] == 2
    assert report["total_failures"] == 2
    assert {item["signature"] for item in report["top_clusters"]} >= {
        "captcha",
        "email_verification",
    }


def test_record_browser_success_builds_playbook_report(tmp_path):
    from remy.core import browser_failure_memory as mem

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
        mem.record_browser_success(
            tool="browser_act",
            action="click",
            url="https://example.com/dashboard",
            text="Welcome back. Dashboard is ready.",
            status="verified",
        )
        report = mem.get_browser_success_report(limit=5)
    finally:
        mem.settings.DATA_DIR = original_data_dir

    assert report["total_playbooks"] == 1
    assert report["total_successes"] == 2
    assert report["top_playbooks"][0]["flow"] == "signup"


def test_get_browser_execution_hints_prefers_matching_domain_and_flow(tmp_path):
    from remy.core import browser_failure_memory as mem

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
        mem.record_browser_success(
            tool="browser_act",
            action="click",
            url="https://example.com/dashboard",
            text="Welcome back. Dashboard is ready.",
            status="verified",
        )
        mem.record_browser_success(
            tool="browser_act",
            action="click",
            url="https://other.com/posts/123",
            text="Post is live and public.",
            status="verified",
        )
        hints = mem.get_browser_execution_hints(
            url="https://example.com/signup",
            text="Register a new account on example.com",
            action="click",
            limit=2,
        )
    finally:
        mem.settings.DATA_DIR = original_data_dir

    assert hints["domain"] == "example.com"
    assert hints["flow"] == "signup"
    assert hints["failure_hints"][0]["signature"] == "captcha"
    assert hints["success_hints"][0]["domain"] == "example.com"


def test_get_browser_execution_hints_merges_avoided_and_preferred_selectors(tmp_path):
    from remy.core import browser_failure_memory as mem

    original_data_dir = mem.settings.DATA_DIR
    mem.settings.DATA_DIR = tmp_path
    try:
        mem.record_browser_failure(
            tool="browser_act",
            action="click",
            url="https://example.com/signup",
            text="Timeout selecting submit button",
            selector="#submit",
            status="attempted",
        )
        mem.record_browser_failure(
            tool="browser_act",
            action="click",
            url="https://example.com/signup",
            text="Timeout selecting submit button",
            selector="#submit",
            status="attempted",
        )
        mem.record_browser_success(
            tool="browser_act",
            action="click",
            url="https://example.com/dashboard",
            text="Welcome back. Dashboard is ready.",
            selector="button[type=submit]",
            status="verified",
        )
        hints = mem.get_browser_execution_hints(
            url="https://example.com/signup",
            text="Register a new account on example.com",
            action="click",
            limit=3,
        )
    finally:
        mem.settings.DATA_DIR = original_data_dir

    assert hints["avoided_selectors"][0]["selector"] == "#submit"
    assert hints["avoided_selectors"][0]["count"] == 2
    assert hints["preferred_selectors"][0]["selector"] == "button[type=submit]"
