import json
from unittest.mock import patch

import pytest


@pytest.fixture
def replay_brain(tmp_path):
    from aura import Aura

    brain = Aura(str(tmp_path / "replay_brain"))
    yield brain
    brain.close()


def _write_history(tmp_path, entries):
    history_dir = tmp_path / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    (history_dir / "session.json").write_text(
        json.dumps({"log": entries}, ensure_ascii=False),
        encoding="utf-8",
    )
    return history_dir


def test_analyze_history_memory_gaps_detects_missing_items(replay_brain, tmp_path):
    from remy.core.history_replay import analyze_history_memory_gaps

    history_dir = _write_history(
        tmp_path,
        [
            {
                "type": "tool_call",
                "tool": "store_user_profile",
                "timestamp": "2026-03-23T10:00:00",
                "args": {"name": "Богдан", "location": "Київ"},
                "result": '{"stored": true}',
            },
            {
                "type": "tool_call",
                "tool": "store_person",
                "timestamp": "2026-03-23T10:01:00",
                "args": {"full_name": "Ганна", "birth_date": "1940-01-01"},
                "result": '{"stored": true}',
            },
            {
                "type": "tool_call",
                "tool": "schedule_task",
                "timestamp": "2026-03-23T10:02:00",
                "args": {"description": "Подзвонити мамі"},
                "result": '{"scheduled": true}',
            },
        ],
    )

    report = analyze_history_memory_gaps(
        lambda **search_kwargs: replay_brain.search(**search_kwargs),
        history_dir=history_dir,
        sample_limit=10,
    )

    assert report["missing_candidates_count"] >= 2
    assert any(tool in report["missing_by_tool"] for tool in ("store_user_profile", "store_person"))
    assert report["missing_by_tool"].get("schedule_task", 0) >= 1
    assert all(item["candidate_id"] for item in report["recent_missing"])


def test_review_history_memory_gaps_tool_returns_review_candidates(tmp_path):
    from aura import Aura

    brain = Aura(str(tmp_path / "tool_brain"))
    _write_history(
        tmp_path,
        [
            {
                "type": "tool_call",
                "tool": "store_research",
                "timestamp": "2026-03-23T11:00:00",
                "args": {
                    "project_name": "AuraSDK v6",
                    "summary": "Salience, reflection, contradiction governance.",
                },
                "result": '{"stored": true}',
            },
            {
                "type": "tool_call",
                "tool": "store_story",
                "timestamp": "2026-03-23T11:05:00",
                "args": {
                    "title": "Початок проекту",
                    "content": "Проєкт почався з Remembrance Hub.",
                },
                "result": '{"stored": true}',
            },
        ],
    )

    with patch("remy.core.brain_tools.brain", brain), patch("remy.core.brain_tools.settings") as mock_settings:
        mock_settings.DATA_DIR = tmp_path
        from remy.core.brain_tools import execute_tool

        payload = json.loads(execute_tool("review_history_memory_gaps", {"sample_limit": 5}))

    brain.close()

    assert payload["review_candidates_count"] >= 2
    assert any(item["tool"] == "store_research" for item in payload["review_candidates"])
    assert any(item["tool"] == "store_story" for item in payload["review_candidates"])


def test_reconstruct_history_candidates_selectively_applies(tmp_path):
    from remy.core.history_replay import reconstruct_history_candidates

    history_dir = _write_history(
        tmp_path,
        [
            {
                "type": "tool_call",
                "tool": "store_person",
                "timestamp": "2026-03-23T10:01:00",
                "args": {"full_name": "Ганна", "birth_date": "1940-01-01"},
                "result": '{"stored": true}',
            },
            {
                "type": "tool_call",
                "tool": "schedule_task",
                "timestamp": "2026-03-23T10:02:00",
                "args": {"description": "Подзвонити мамі"},
                "result": '{"scheduled": true}',
            },
        ],
    )

    calls = []

    def _execute(tool, tool_args):
        calls.append((tool, tool_args))
        return '{"ok": true}'

    stats = reconstruct_history_candidates(
        _execute,
        candidate_ids=["session.json:1:schedule_task"],
        history_dir=history_dir,
    )

    assert stats["requested"] == 1
    assert stats["applied"] == 1
    assert stats["applied_candidate_ids"] == ["session.json:1:schedule_task"]
    assert stats["skipped_candidate_ids"] == []
    assert stats["verification"]["status"] == "verified"
    assert stats["verification"]["verified"] is True
    assert calls == [("schedule_task", {"description": "Подзвонити мамі"})]


def test_reconstruct_history_candidates_marks_partial_apply_for_review(tmp_path):
    from remy.core.history_replay import reconstruct_history_candidates

    history_dir = _write_history(
        tmp_path,
        [
            {
                "type": "tool_call",
                "tool": "store_person",
                "timestamp": "2026-03-23T10:01:00",
                "args": {"full_name": "Ганна", "birth_date": "1940-01-01"},
                "result": '{"stored": true}',
            },
            {
                "type": "tool_call",
                "tool": "schedule_task",
                "timestamp": "2026-03-23T10:02:00",
                "args": {"description": "Подзвонити мамі"},
                "result": '{"scheduled": true}',
            },
        ],
    )

    def _execute(tool, tool_args):
        if tool == "schedule_task":
            return '{"ok": true}'
        return '{"error": "failed"}'

    stats = reconstruct_history_candidates(
        _execute,
        candidate_ids=["session.json:0:store_person", "session.json:1:schedule_task"],
        history_dir=history_dir,
    )

    assert stats["applied"] == 1
    assert stats["skipped"] == 1
    assert stats["verification"]["status"] == "repair_required"
    assert stats["verification"]["verified"] is False
    assert stats["verification"]["failure_code"] == "verification_failed"
