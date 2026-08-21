"""Progressive capability bundles built on Remy's selective tool loading."""

from __future__ import annotations


SKILL_CATALOG: dict[str, dict] = {
    "deep_research": {
        "description": "Grounded multi-source research and durable reports",
        "tools": ["start_research", "research_status", "research_report", "web_search", "browse_page", "extract_content"],
    },
    "project_work": {
        "description": "Plan, delegate, run workflows, and inspect project files",
        "tools": [
            "delegate_task",
            "list_child_sessions",
            "get_child_report",
            "follow_up_child_session",
            "interrupt_child_session",
            "resume_child_session",
            "list_pipeline_candidates",
            "propose_pipeline_candidate",
            "list_ptc_tools",
            "validate_ptc_program",
            "run_ptc_program",
            "list_files",
            "read_file",
            "write_file",
        ],
    },
    "documents": {
        "description": "Create reports, presentations, and structured artifacts",
        "tools": ["generate_report", "generate_presentation", "extract_content"],
    },
    "personal_organizer": {
        "description": "Todos, reminders, calendar, and tracked metrics",
        "tools": ["schedule_task", "add_todo", "list_todos", "update_todo", "track_metric", "metric_summary"],
    },
    "memory_audit": {
        "description": "Inspect exact transcripts and memory gaps",
        "tools": ["search_transcript_history", "review_history_memory_gaps", "insights", "explain_memory_recall"],
    },
}


def list_skills() -> list[dict]:
    return [
        {"name": name, "description": data["description"], "tools": list(data["tools"])}
        for name, data in SKILL_CATALOG.items()
    ]


def tools_for_skill(name: str, available: set[str]) -> list[str]:
    skill = SKILL_CATALOG.get(name)
    if not skill:
        return []
    return [tool for tool in skill["tools"] if tool in available]
