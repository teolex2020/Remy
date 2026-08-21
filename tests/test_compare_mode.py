from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from remy.web.routes import websocket as websocket_routes


class _CompareManager:
    def __init__(self):
        self.calls = []

    async def compare_model_stream(self, text, model):
        self.calls.append((text, model))
        if model == "final-only":
            yield {"type": "final", "text": "Complete final response"}
        elif model == "streamed":
            yield {"type": "token", "content": "Streamed response"}
            yield {"type": "final", "text": "Streamed response"}
        else:
            yield {"type": "final", "text": ""}


def _compare_client(monkeypatch, manager):
    api = SimpleNamespace(
        get_session_manager=lambda: manager,
        metrics_collector=SimpleNamespace(),
    )
    monkeypatch.setattr(websocket_routes, "_get_api", lambda: api)
    app = FastAPI()
    app.include_router(websocket_routes.router, prefix="/api")
    return TestClient(app)


def _receive_until_all_done(ws):
    events = []
    while True:
        event = ws.receive_json()
        events.append(event)
        if event["type"] == "all_done":
            return events


def test_compare_forwards_final_only_responses_and_deduplicates_models(monkeypatch):
    manager = _CompareManager()
    client = _compare_client(monkeypatch, manager)

    with client.websocket_connect("/api/ws/compare") as ws:
        ws.send_json(
            {
                "text": "Compare these answers",
                "models": ["final-only", "streamed", "final-only"],
            }
        )
        events = _receive_until_all_done(ws)

    tokens = [event for event in events if event["type"] == "token"]
    assert {event["model"]: event["content"] for event in tokens} == {
        "final-only": "Complete final response",
        "streamed": "Streamed response",
    }
    assert sum(event["model"] == "streamed" for event in tokens) == 1
    assert {event["model"] for event in events if event["type"] == "done"} == {
        "final-only",
        "streamed",
    }
    assert sorted(manager.calls) == [
        ("Compare these answers", "final-only"),
        ("Compare these answers", "streamed"),
    ]


def test_compare_returns_visible_error_for_empty_model_completion(monkeypatch):
    manager = _CompareManager()
    client = _compare_client(monkeypatch, manager)

    with client.websocket_connect("/api/ws/compare") as ws:
        ws.send_json({"text": "Compare", "models": ["empty"]})
        events = _receive_until_all_done(ws)

    error = next(event for event in events if event["type"] == "error")
    assert error["model"] == "empty"
    assert "without response text" in error["content"]
    assert not any(event["type"] == "done" for event in events)


@pytest.mark.asyncio
async def test_compare_model_stream_uses_isolated_session_and_model_routing(monkeypatch):
    from remy.web.session import WebSessionManager

    manager = object.__new__(WebSessionManager)
    manager.readonly = False
    active_session = SimpleNamespace(
        session_id="conversation-main",
        project_id="project-main",
        history=["history-entry"],
        session_log=[{"type": "existing"}],
    )
    manager.get_or_create_session = lambda: active_session
    manager._restore_short_term_history = lambda session: 0
    captured = {}

    @contextmanager
    def fake_bind(project_id):
        captured["project_id"] = project_id
        yield SimpleNamespace(project_id=project_id)

    async def fake_agent_stream(**kwargs):
        captured.update(kwargs)
        yield {"type": "final", "text": "isolated"}

    monkeypatch.setattr("remy.web.session.bind_project", fake_bind)
    monkeypatch.setattr("remy.core.agent.invoke_agent_stream", fake_agent_stream)

    events = [
        event
        async for event in manager.compare_model_stream(
            "Question",
            "provider/model-a",
        )
    ]

    assert events == [{"type": "final", "text": "isolated"}]
    assert captured["project_id"] == "project-main"
    assert captured["session_id"].startswith("compare-")
    assert captured["session_id"] != active_session.session_id
    assert captured["replay_preferred_model"] == "provider/model-a"
    assert captured["capability_profile"] == "read_only"
    assert captured["history"] == ["history-entry"]
    assert active_session.history == ["history-entry"]
    assert active_session.session_log == [{"type": "existing"}]


def test_compare_ui_handles_terminal_and_early_close_states():
    from pathlib import Path

    chat = Path("src/remy/web/static/js/chat.js").read_text(encoding="utf-8")
    assert 'data.type === "all_done"' in chat
    assert "terminalModels.add(data.model)" in chat
    assert "Connection closed before this model returned a response." in chat
