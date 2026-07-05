def test_build_mission_activity_state_uses_shared_mission_hydration(monkeypatch):
    from remy.core.activity_state import build_mission_activity_state

    fake_records = [
        type(
            "Record",
            (),
            {
                "id": "goal-a",
                "content": "Task [task-a]: Collect filings",
                "metadata": {"mission_task_id": "task-a", "status": "active"},
            },
        )(),
        type(
            "Record",
            (),
            {
                "id": "goal-b",
                "content": "Task [task-b]: Compare source trail",
                "metadata": {
                    "mission_task_id": "task-b",
                    "status": "blocked_external",
                    "blocked_reason": "Waiting for captcha bypass",
                },
            },
        )(),
        type(
            "Record",
            (),
            {
                "id": "goal-c",
                "content": "Task [task-c]: Draft summary",
                "metadata": {"mission_task_id": "task-c", "status": "completed"},
            },
        )(),
    ]

    monkeypatch.setattr(
        "remy.core.autonomy_goals._load_missions",
        lambda: [{"id": "mission-alpha", "description": "Alpha mission", "tasks": [{"action": "Collect filings"}]}],
    )
    monkeypatch.setattr("remy.core.agent_tools.brain.search", lambda **kwargs: fake_records)
    monkeypatch.setattr("remy.core.orchestrator.get_focus_stale_cycles", lambda mission_id: 2)

    state = build_mission_activity_state(
        "mission-alpha",
        active_task_goal_records=[{"mission_task_id": "task-a"}],
    )

    assert state["id"] == "mission-alpha"
    assert state["description"] == "Alpha mission"
    assert state["active_tasks"] == 1
    assert state["pending_tasks"] == 1
    assert state["blocked_tasks"] == 1
    assert state["failed_tasks"] == 0
    assert state["completed_tasks"] == 1
    assert state["focus_stale_cycles"] == 2
    assert state["pending_task_labels"][0] == "Collect filings"
    assert state["pending_task_items"][1]["status"] == "blocked_external"
    assert state["pending_task_items"][1]["detail"] == "Waiting for captcha bypass"
