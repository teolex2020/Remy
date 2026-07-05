"""
Tool Health Visibility & Adaptive Routing (AUTON-11).

Provides:
- tool_status tool: agent sees tool health states
- Alternative tool routing: fallback when primary tools fail
- Health info in decision prompt
- Proactive tool testing on session start
"""

import logging

from remy.core.event_bus import event_bus

logger = logging.getLogger("Autonomy.ToolRouting")


_ALTERNATIVE_ROUTES: dict[str, list[dict]] = {
    "web_search": [
        {"tool": "http_get", "hint": "Use http_get with a search URL for manual results"},
        {"tool": "recall", "hint": "Try recall to find previously stored info"},
    ],
    "browse_page": [
        {"tool": "extract_content", "hint": "Use extract_content for clean text (no JS rendering)"},
        {"tool": "http_get", "hint": "Use http_get to fetch page content directly"},
    ],
    "http_get": [
        {"tool": "extract_content", "hint": "Use extract_content for cleaner article/page text"},
        {"tool": "web_search", "hint": "Use web_search as fallback for web content"},
    ],
    "extract_content": [
        {"tool": "http_get", "hint": "Use http_get for raw HTML if extraction fails"},
        {"tool": "browse_page", "hint": "Use browse_page for JS-rendered pages"},
    ],
}


def get_alternatives(tool_name: str) -> list[dict]:
    return _ALTERNATIVE_ROUTES.get(tool_name, [])


def get_best_alternative(tool_name: str) -> dict | None:
    alternatives = get_alternatives(tool_name)
    if not alternatives:
        return None

    try:
        from remy.core.tool_health import tool_health

        for alt in alternatives:
            if tool_health.is_available(alt["tool"]):
                return alt
    except Exception:
        if alternatives:
            return alternatives[0]

    return None


def get_tool_status_report() -> dict:
    try:
        from remy.core.tool_health import tool_health

        report = tool_health.get_health_report()
    except Exception:
        report = {}

    healthy = []
    degraded = []
    unavailable = []
    alternatives = {}

    external_tools = {"web_search", "http_get", "browse_page", "browser_act", "browser_close"}

    for tool_name in external_tools:
        if tool_name in report:
            status = report[tool_name]
            if "UNAVAILABLE" in status:
                unavailable.append({"tool": tool_name, "status": status})
                alt = get_best_alternative(tool_name)
                if alt:
                    alternatives[tool_name] = alt
            else:
                degraded.append({"tool": tool_name, "status": status})
        else:
            healthy.append(tool_name)

    return {
        "healthy": sorted(healthy),
        "degraded": degraded,
        "unavailable": unavailable,
        "alternatives": alternatives,
    }


def format_tool_health_for_prompt(report: dict | None = None) -> str:
    """Format tool health plus browser execution memory for decision prompts."""
    if report is None:
        report = get_tool_status_report()

    browser_failures = None
    browser_successes = None
    try:
        from remy.core.browser_failure_memory import (
            get_browser_failure_report,
            get_browser_success_report,
        )

        browser_failures = get_browser_failure_report(limit=3)
        browser_successes = get_browser_success_report(limit=3)
    except Exception:
        browser_failures = None
        browser_successes = None

    if (
        not report["degraded"]
        and not report["unavailable"]
        and not (browser_failures and browser_failures["top_clusters"])
        and not (browser_successes and browser_successes["top_playbooks"])
    ):
        return ""

    lines = ["TOOL HEALTH STATUS:"]

    for item in report["unavailable"]:
        line = f"  [UNAVAILABLE] {item['tool']}: {item['status']}"
        alt = report.get("alternatives", {}).get(item["tool"])
        if alt:
            line += f" -> Use {alt['tool']} instead ({alt['hint']})"
        lines.append(line)

    for item in report["degraded"]:
        lines.append(f"  [DEGRADED] {item['tool']}: {item['status']}")

    if browser_failures and browser_failures["top_clusters"]:
        lines.append("RECENT BROWSER HOTSPOTS:")
        for rec in browser_failures["top_clusters"]:
            lines.append(
                f"  - {rec['domain']}: {rec['signature']} x{rec['count']} "
                f"({rec['tool']}{'/' + rec['action'] if rec.get('action') else ''})"
            )

    if browser_successes and browser_successes["top_playbooks"]:
        lines.append("REUSABLE BROWSER PLAYBOOKS:")
        for rec in browser_successes["top_playbooks"]:
            lines.append(
                f"  - {rec['domain']}: {rec['flow']} x{rec['count']} "
                f"({rec['tool']}{'/' + rec['action'] if rec.get('action') else ''})"
            )

    lines.append("Avoid unavailable tools. Prefer healthy alternatives.")
    return "\n".join(lines)


async def test_tools_on_startup() -> dict:
    """Quick health check of external tools at session startup."""
    results = {}

    try:
        from remy.core.tool_dispatch import execute_tool

        result = execute_tool("get_current_datetime", {})
        results["get_current_datetime"] = (
            "ok" if result and "error" not in result.lower() else f"error: {result[:100]}"
        )
    except Exception as e:
        results["get_current_datetime"] = f"error: {e}"

    event_bus.emit("tool_startup_test", {"results": results})
    return results
