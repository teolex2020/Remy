"""Deterministic benchmark scenarios for autonomous task reliability.

These benchmarks validate the evidence/recovery logic without requiring live
network/browser sessions. The latest report is stored in DATA_DIR so the web UI
can surface benchmark health alongside runtime activity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from remy.config.settings import settings
from remy.core.autonomy import _detect_external_blocker, _detect_resume_state_reset
from remy.core.success_criteria import verify_criterion


@dataclass
class BenchmarkResult:
    name: str
    passed: bool
    detail: str
    category: str


def _benchmark_path() -> Path:
    return settings.DATA_DIR / "autonomy_benchmarks.json"


def run_autonomy_benchmarks() -> dict:
    """Run deterministic task-flow benchmarks and persist a JSON report."""
    results = [
        _signup_success_benchmark(),
        _signup_external_blocker_benchmark(),
        _signup_resume_reset_benchmark(),
        _publish_success_benchmark(),
        _publish_resume_reset_benchmark(),
        _market_research_completion_benchmark(),
    ]

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    category_summary: dict[str, dict[str, int | float]] = {}
    for category in sorted({r.category for r in results}):
        category_results = [r for r in results if r.category == category]
        category_passed = sum(1 for r in category_results if r.passed)
        category_total = len(category_results)
        category_summary[category] = {
            "passed": category_passed,
            "failed": category_total - category_passed,
            "total": category_total,
            "pass_rate": round((category_passed / category_total) * 100, 1)
            if category_total
            else 0.0,
        }
    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "passed": passed,
            "failed": total - passed,
            "total": total,
            "pass_rate": round((passed / total) * 100, 1) if total else 0.0,
        },
        "category_summary": category_summary,
        "results": [
            {
                "name": r.name,
                "passed": r.passed,
                "detail": r.detail,
                "category": r.category,
            }
            for r in results
        ],
    }
    path = _benchmark_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def load_benchmark_report() -> dict | None:
    """Load the latest benchmark report if present."""
    path = _benchmark_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _signup_success_benchmark() -> BenchmarkResult:
    ok, reason = verify_criterion(
        {"type": "signup_completed"},
        session_log=[
            {
                "type": "tool_call",
                "tool": "browser_act",
                "verified": True,
                "status": "verified",
                "evidence": {
                    "page_url": "https://app.example.com/dashboard",
                    "page_text_snippet": "Welcome back. Dashboard is ready. Sign out",
                },
            }
        ],
    )
    return BenchmarkResult("signup_completed", ok, reason, "execution")


def _signup_external_blocker_benchmark() -> BenchmarkResult:
    blocker = _detect_external_blocker(
        {"goal_template": "signup_operator"},
        [
            {
                "type": "tool_call",
                "tool": "browser_act",
                "answer": "Check your email to verify your account",
                "evidence": {
                    "page_url": "https://app.example.com/welcome",
                    "page_text_snippet": "Check your email to verify your account",
                },
            }
        ],
    )
    passed = bool(blocker and blocker.get("reason") == "email verification required")
    detail = blocker["evidence"] if blocker else "No blocker detected"
    return BenchmarkResult("signup_external_blocker", passed, detail, "recovery")


def _signup_resume_reset_benchmark() -> BenchmarkResult:
    warning = _detect_resume_state_reset(
        {"goal_template": "signup_operator", "resume_context": "Resume from dashboard"},
        [
            {
                "type": "tool_call",
                "tool": "browse_page",
                "evidence": {
                    "page_url": "https://app.example.com/signup",
                    "page_text_snippet": "Create account to continue",
                },
            }
        ],
    )
    return BenchmarkResult(
        "signup_resume_reset",
        "State reset detected" in warning,
        warning or "No reset warning emitted",
        "recovery",
    )


def _publish_success_benchmark() -> BenchmarkResult:
    ok, reason = verify_criterion(
        {"type": "post_published"},
        session_log=[
            {
                "type": "tool_call",
                "tool": "browser_act",
                "verified": True,
                "status": "verified",
                "answer": "Your post is live now",
                "evidence": {
                    "page_url": "https://x.com/test/status/12345",
                    "page_text_snippet": "Your post is live",
                },
            }
        ],
    )
    return BenchmarkResult("post_published", ok, reason, "execution")


def _publish_resume_reset_benchmark() -> BenchmarkResult:
    warning = _detect_resume_state_reset(
        {"goal_template": "publisher", "resume_context": "Resume from publish verification"},
        [
            {
                "type": "tool_call",
                "tool": "browse_page",
                "evidence": {
                    "page_url": "https://x.com/compose/post",
                    "page_text_snippet": "Draft your post here",
                },
            }
        ],
    )
    return BenchmarkResult(
        "publish_resume_reset",
        "State reset detected" in warning,
        warning or "No reset warning emitted",
        "recovery",
    )


def _market_research_completion_benchmark() -> BenchmarkResult:
    session_log = [
        {
            "type": "tool_call",
            "tool": "add_research_finding",
            "stored": True,
            "findings_count": 3,
        },
        {
            "type": "tool_call",
            "tool": "generate_report",
            "generated": True,
            "url": "/api/reports/market.pdf",
            "record_id": "rec-123",
        },
    ]
    ok_count, reason_count = verify_criterion(
        {
            "type": "numeric_result",
            "tool": "add_research_finding",
            "fields": ["findings_count"],
            "min_value": 3,
        },
        session_log=session_log,
    )
    ok_artifact, reason_artifact = verify_criterion(
        {
            "type": "artifact_created",
            "tool": "generate_report",
            "artifact_fields": ["url", "record_id"],
        },
        session_log=session_log,
    )
    passed = ok_count and ok_artifact
    detail = f"{reason_count}; {reason_artifact}"
    return BenchmarkResult("market_research_completion", passed, detail, "task")
