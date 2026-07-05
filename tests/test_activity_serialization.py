import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch


def test_build_activity_payload_returns_shared_summary_and_lists():
    from remy.web.routes._activity_serialization import build_activity_payload

    goals = [
        SimpleNamespace(id="goal-1", content="Goal A", metadata={"status": "active"}),
        SimpleNamespace(id="goal-2", content="Goal B", metadata={"status": "blocked_external"}),
        SimpleNamespace(id="goal-3", content="Goal C", metadata={"status": "completed"}),
    ]
    outcomes = [
        SimpleNamespace(id="out-1", content="Success result", metadata={"success": True, "tokens_used": 100}),
        SimpleNamespace(id="out-2", content="Failure result", metadata={"success": False, "tokens_used": 50}),
    ]
    reflections = [SimpleNamespace(id="ref-1", content="Reflection", metadata={"session_id": "sess-1"})]
    proactive = [SimpleNamespace(id="pro-1", content="Ping", metadata={"trigger_reason": "inactivity"})]

    payload = build_activity_payload(goals, outcomes, reflections, proactive)

    assert payload["summary"]["total_actions"] == 2
    assert payload["summary"]["success"] == 1
    assert payload["summary"]["failure"] == 1
    assert payload["summary"]["success_rate"] == 50
    assert payload["summary"]["total_tokens"] == 150
    assert payload["summary"]["active_goals"] == 1
    assert payload["summary"]["blocked_external_goals"] == 1
    assert payload["summary"]["completed_goals"] == 1
    assert len(payload["goals"]) == 3
    assert len(payload["outcomes"]) == 2
    assert len(payload["reflections"]) == 1
    assert len(payload["proactive"]) == 1


def test_get_activity_uses_shared_activity_builder():
    from remy.web.routes import autonomy_routes as routes

    fake_api = SimpleNamespace(
        brain_lock=nullcontext(),
        brain=SimpleNamespace(),
    )

    with patch("remy.web.routes.autonomy_routes._get_api", return_value=fake_api), \
         patch(
             "remy.core.combined_runner.get_activity_feed_snapshot",
             return_value={
                 "summary": {"total_actions": 1, "success": 1, "active_goals": 1},
                 "goals": [{"id": "goal-1"}],
                 "outcomes": [{"id": "out-1"}],
                 "reflections": [],
                 "proactive": [],
             },
         ):
        payload = asyncio.run(routes.get_activity())

    assert payload["summary"]["total_actions"] == 1
    assert payload["summary"]["success"] == 1
    assert payload["summary"]["active_goals"] == 1
    assert payload["goals"][0]["id"] == "goal-1"
    assert payload["outcomes"][0]["id"] == "out-1"
