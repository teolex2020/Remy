from types import SimpleNamespace


def test_serialize_goal_record_returns_shared_goal_shape():
    from remy.web.routes._goal_serialization import serialize_goal_record

    rec = SimpleNamespace(
        id="goal-1",
        content="Goal [HIGH]: Investigate primary source trail",
        metadata={
            "status": "blocked_external",
            "priority": "high",
            "goal_type": "research",
            "blocked_reason": "Need verification",
            "blocked_evidence": "Email code required",
            "resume_context": "Resume after verification",
            "attempts": 2,
            "created_at": "2026-03-19T10:00:00",
            "updated_at": "2026-03-19T11:00:00",
        },
    )

    payload = serialize_goal_record(rec)

    assert payload["id"] == "goal-1"
    assert payload["status"] == "blocked_external"
    assert payload["priority"] == "high"
    assert payload["blocked_reason"] == "Need verification"
    assert payload["blocked_evidence"] == "Email code required"
    assert payload["resume_context"] == "Resume after verification"
    assert payload["attempts"] == 2
    assert payload["timestamp"] == "2026-03-19T11:00:00"


def test_serialize_goal_as_todo_maps_goal_to_agent_todo_shape():
    from remy.web.routes._goal_serialization import serialize_goal_as_todo

    rec = SimpleNamespace(
        id="goal-1",
        content="Goal [HIGH]: Register at example.com",
        metadata={
            "type": "autonomous_goal",
            "status": "blocked_external",
            "priority": "high",
            "goal_type": "signup",
            "blocked_action_id": "act-123",
            "blocked_reason": "email verification required",
            "resume_context": "Continue from dashboard after verification",
            "attempts": 2,
            "created_at": "2026-03-07T10:00:00",
        },
    )

    payload = serialize_goal_as_todo(rec)

    assert payload is not None
    assert payload["id"] == "goal-1"
    assert payload["status"] == "in_progress"
    assert payload["source"] == "goal"
    assert payload["category"] == "agent"
    assert payload["raw_status"] == "blocked_external"
    assert payload["blocked_action_id"] == "act-123"
    assert payload["blocked_reason"] == "email verification required"
    assert payload["resume_context"] == "Continue from dashboard after verification"
