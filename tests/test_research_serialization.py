from types import SimpleNamespace


def test_serialize_completed_research_project_maps_completed_record():
    from remy.web.routes._research_serialization import serialize_completed_research_project

    rec = SimpleNamespace(
        content="Completed Research Project: Vitamin D",
        metadata={
            "project_id": "rp-vitd",
            "completed_at": "2026-03-19T12:00:00",
            "report_preview": "Key findings...",
        },
    )

    payload = serialize_completed_research_project(rec)

    assert payload == {
        "project_id": "rp-vitd",
        "topic": "Vitamin D",
        "status": "completed",
        "completed_at": "2026-03-19T12:00:00",
        "report_preview": "Key findings...",
    }
