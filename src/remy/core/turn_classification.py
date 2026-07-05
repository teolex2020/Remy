"""Turn Classification — rule-based classifier for autonomy cycles.

Classifies each cycle as productive/maintenance/idle based on which tools
were called. Used to filter idle cycles from context (saves tokens) and
for metrics/dashboard.

Cost: zero LLM calls — pure set intersection.
"""

from __future__ import annotations

from enum import Enum


class TurnClass(str, Enum):
    PRODUCTIVE = "productive"
    MAINTENANCE = "maintenance"
    IDLE = "idle"


# If ANY tool in the cycle matches PRODUCTIVE_TOOLS → classify as productive.
# Priority: productive > maintenance > idle.
PRODUCTIVE_TOOLS: frozenset[str] = frozenset(
    {
        "recall",
        "store",
        "web_search",
        "browse_page",
        "browser_act",
        "delegate_task",
        "generate_report",
        "generate_image",
        "extract_content",
        "extract_facts",
        "create_subgoal",
        "complete_goal",
        "track_metric",
        "event_correlate",
        "write_file",
        "http_get",
        "start_research",
        "add_research_finding",
        "complete_research",
        "store_research",
        "scratchpad",
        "add_todo",
        "update_todo",
        "store_person",
        "store_story",
        "sandbox_create_tool",
        "sandbox_test_tool",
        "sandbox_approve_tool",
        "export_skill",
        "import_skill",
        "install_marketplace_skill",
    }
)

MAINTENANCE_TOOLS: frozenset[str] = frozenset(
    {
        "consolidate",
        "insights",
        "verify_record",
        "update_record",
        "delete_record",
        "mark_stale",
        "connect_records",
        "tool_status",
        "list_available_tools",
        "enable_tools",
        "update_persona",
        "store_user_profile",
        "add_runtime_directive",
        "remove_runtime_directive",
    }
)

# Everything else (or zero-tool cycles) is IDLE.
# Explicitly listed for documentation, not used in logic.
IDLE_TOOLS: frozenset[str] = frozenset(
    {
        "get_current_datetime",
        "read_persona",
        "read_file",
        "list_todos",
        "list_directory",
        "metric_summary",
        "people_list",
        "get_full_record",
        "get_protected_record",
        "get_connections",
        "search",
        "search_exact",
        "recall_structured",
        "browse_marketplace",
        "sandbox_list_tools",
        "request_guidance",
    }
)


def classify_turn(session_log: list[dict]) -> TurnClass:
    """Classify an autonomy cycle based on tools called.

    Args:
        session_log: List of dicts with at least {"type": "tool_call", "tool": str}.

    Returns:
        TurnClass.PRODUCTIVE if any productive tool was called,
        TurnClass.MAINTENANCE if any maintenance tool was called,
        TurnClass.IDLE otherwise (including zero-tool cycles).
    """
    tool_names = {e["tool"] for e in session_log if e.get("type") == "tool_call"}

    if not tool_names:
        return TurnClass.IDLE

    if tool_names & PRODUCTIVE_TOOLS:
        return TurnClass.PRODUCTIVE

    if tool_names & MAINTENANCE_TOOLS:
        return TurnClass.MAINTENANCE

    return TurnClass.IDLE
