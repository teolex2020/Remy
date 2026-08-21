"""Shared serializers for research project payloads."""


def serialize_completed_research_project(rec) -> dict:
    meta = rec.metadata or {}
    payload = {
        "project_id": meta.get("project_id"),
        "topic": meta.get("topic") or rec.content.replace("Completed Research Project: ", ""),
        "status": "completed",
        "completed_at": meta.get("completed_at"),
        "report_preview": meta.get("report_preview"),
    }
    for key in ("execution_attempt_id", "job_state", "report_id", "pdf_url"):
        if meta.get(key) not in (None, ""):
            payload[key] = meta[key]
    return payload
