"""Shared serializers for research project payloads."""


def serialize_completed_research_project(rec) -> dict:
    meta = rec.metadata or {}
    return {
        "project_id": meta.get("project_id"),
        "topic": rec.content.replace("Completed Research Project: ", ""),
        "status": "completed",
        "completed_at": meta.get("completed_at"),
        "report_preview": meta.get("report_preview"),
    }
