from remy.core.workers.browser_worker import (
    _derive_browser_worker_status,
    build_browser_worker_prompt,
)
from remy.core.workers.contracts import WorkerExecutionResult
from remy.core.workers.reporter import format_worker_report


def test_browser_worker_prompt_is_operational():
    prompt = build_browser_worker_prompt(
        goal={
            "description": "Register at https://example.com/signup",
            "goal_template": "signup_operator",
            "resume_context": "Resume from verification page",
            "blocked_reason": "email verification required",
        },
        session_log=[
            {
                "type": "tool_call",
                "tool": "browse_page",
                "status": "attempted",
                "page_state": "login",
                "visible_error_text": "Enter a valid email or phone number",
                "evidence": {"page_url": "https://accounts.google.com/signin"},
            }
        ],
    )

    assert "You are BROWSER_WORKER." in prompt
    assert "Use browse_page/browser_act/browser_close only." in prompt
    assert "Enter a valid email or phone number" in prompt
    assert "Resume from verification page" in prompt


def test_browser_worker_prompt_includes_publisher_playbook():
    prompt = build_browser_worker_prompt(
        goal={
            "description": "Draft a Dev.to article comparing AuraSDK and Mem0",
            "goal_template": "publisher",
            "target_url": "https://dev.to/new",
        },
        session_log=[],
    )

    assert "PUBLISHER MODE" in prompt
    assert "Mode: article" in prompt
    assert "Channel: devto" in prompt
    assert "draft" in prompt.lower()


def test_browser_worker_status_prefers_blocked_external():
    status, evidence = _derive_browser_worker_status(
        [
            {
                "type": "tool_call",
                "tool": "browser_act",
                "status": "attempted",
                "external_blocker_likely": True,
                "blocker_reason": "email verification required",
                "visible_error_text": "Check your email to continue",
                "evidence": {"page_url": "https://app.example.com/welcome"},
            }
        ],
        "Status: attempted",
    )

    assert status == "blocked_external"
    assert evidence["current_url"] == "https://app.example.com/welcome"


def test_reporter_formats_browser_worker_result():
    text = format_worker_report(
        WorkerExecutionResult(
            worker="browser_worker",
            status="blocked_external",
            response_text="raw long worker text",
            evidence={
                "current_url": "https://accounts.google.com/signin",
                "page_state": "login",
                "visible_error_text": "Enter a valid email or phone number",
            },
        )
    )

    assert "Status: blocked_external" in text
    assert "https://accounts.google.com/signin" in text
    assert "Enter a valid email or phone number" in text


def test_reporter_formats_publisher_worker_result():
    text = format_worker_report(
        WorkerExecutionResult(
            worker="browser_worker",
            status="attempted",
            response_text="raw long worker text",
            evidence={
                "current_url": "https://dev.to/new",
                "page_state": "draft",
                "visible_error_text": "Draft saved",
                "capability_pack": "publisher",
                "publisher_mode": "article",
                "publisher_channel": "devto",
                "approval_mode": "all_clicks",
            },
        )
    )

    assert "Draft target: mode=article | channel=devto" in text
    assert "draft mode" in text.lower() or "approval" in text.lower()
