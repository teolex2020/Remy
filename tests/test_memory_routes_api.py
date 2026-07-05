from contextlib import nullcontext
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from remy.web.routes import memory as memory_routes


def _build_client(fake_api):
    app = FastAPI()
    app.include_router(memory_routes.router, prefix="/api")
    memory_routes._get_api = lambda: fake_api
    return TestClient(app)


def test_tier_stats_route_normalizes_nested_stats():
    fake_api = SimpleNamespace(
        brain=SimpleNamespace(
            tier_stats=lambda: {
                "cognitive": {"total": 8, "working": 3, "decisions": 5},
                "core": {"total": 12, "domain": 7, "identity": 5},
                "total": 20,
            }
        ),
        brain_lock=nullcontext(),
    )
    client = _build_client(fake_api)

    res = client.get("/api/memory/tier-stats")
    assert res.status_code == 200
    data = res.json()

    assert data["levels"]["identity"] == 5
    assert data["levels"]["domain"] == 7
    assert data["levels"]["decisions"] == 5
    assert data["levels"]["working"] == 3
    assert data["core"]["total"] == 12
    assert data["cognitive"]["total"] == 8
    assert data["total"] == 20


def test_tier_stats_route_preserves_strength_distribution():
    fake_api = SimpleNamespace(
        brain=SimpleNamespace(
            tier_stats=lambda: {
                "levels": {"IDENTITY": 1, "DOMAIN": 2, "DECISIONS": 3, "WORKING": 4},
                "strength_distribution": {"strong": 2, "medium": 5, "weak": 3},
            }
        ),
        brain_lock=nullcontext(),
    )
    client = _build_client(fake_api)

    res = client.get("/api/memory/tier-stats")
    assert res.status_code == 200
    data = res.json()

    assert data["levels"]["IDENTITY"] == 1
    assert data["levels"]["DOMAIN"] == 2
    assert data["levels"]["DECISIONS"] == 3
    assert data["levels"]["WORKING"] == 4
    assert data["strength_distribution"] == {"strong": 2, "medium": 5, "weak": 3}


def test_search_route_uses_exact_mode_tool():
    called = {}

    def _execute_tool(name, args, session_id):
        called["name"] = name
        called["args"] = args
        called["session_id"] = session_id
        return "[]"

    fake_api = SimpleNamespace(
        execute_tool=_execute_tool,
        get_session_manager=lambda: SimpleNamespace(
            get_or_create_session=lambda: SimpleNamespace(session_id="sess-1")
        ),
    )
    client = _build_client(fake_api)

    res = client.post(
        "/api/search",
        json={"query": "Oleksandr", "tags": "user-profile", "mode": "exact"},
    )
    assert res.status_code == 200
    assert called["name"] == "search_exact"
    assert called["args"] == {"query": "Oleksandr", "tags": "user-profile"}


def test_record_feedback_route_uses_memory_feedback_tool():
    called = {}

    def _execute_tool(name, args, session_id):
        called["name"] = name
        called["args"] = args
        called["session_id"] = session_id
        return '{"record_id":"rec-1","useful":false,"net_score":-1,"positive":0,"negative":1,"reason_stored":true}'

    fake_api = SimpleNamespace(
        execute_tool=_execute_tool,
        get_session_manager=lambda: SimpleNamespace(
            get_or_create_session=lambda: SimpleNamespace(session_id="sess-1")
        ),
    )
    client = _build_client(fake_api)

    res = client.post(
        "/api/records/rec-1/feedback",
        json={"useful": False, "reason": "Wrong evidence"},
    )
    assert res.status_code == 200
    data = res.json()

    assert called["name"] == "memory_feedback"
    assert called["args"] == {
        "record_id": "rec-1",
        "useful": False,
        "reason": "Wrong evidence",
    }
    assert data["record_id"] == "rec-1"
    assert data["negative"] == 1
