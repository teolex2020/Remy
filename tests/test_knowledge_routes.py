from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_client():
    from remy.web.routes.knowledge_routes import router

    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def test_serialize_goal_as_calendar_task_maps_goal():
    from remy.web.routes._goal_serialization import serialize_goal_as_calendar_task

    rec = SimpleNamespace(
        id="goal-1",
        content="Goal [HIGH]: Register at example.com",
        metadata={
            "type": "autonomous_goal",
            "status": "active",
            "deadline": "2026-03-21",
            "task_repeat": "weekly",
        },
    )

    payload = serialize_goal_as_calendar_task(rec)

    assert payload == {
        "id": "goal-1",
        "description": "Register at example.com",
        "due_date": "2026-03-21",
        "repeat": "weekly",
        "status": "active",
        "source": "goal",
        "content": "Goal [HIGH]: Register at example.com",
    }


def test_get_calendar_tasks_uses_shared_goal_calendar_mapper():
    client = _make_client()
    goal = SimpleNamespace(
        id="goal-1",
        content="Goal [HIGH]: Register at example.com",
        metadata={
            "type": "autonomous_goal",
            "status": "active",
            "deadline": "2026-03-21",
        },
    )
    fake_api = SimpleNamespace(
        brain_lock=nullcontext(),
        brain=SimpleNamespace(
            search=lambda query, tags, limit=200: (
                [goal] if "autonomous-goal" in tags else []
            )
        ),
    )

    with patch("remy.web.routes.knowledge_routes._get_api", return_value=fake_api):
        response = client.get("/api/knowledge/calendar")

    assert response.status_code == 200
    payload = response.json()
    assert payload["tasks"] == [
        {
            "id": "goal-1",
            "description": "Register at example.com",
            "due_date": "2026-03-21",
            "repeat": None,
            "status": "active",
            "source": "goal",
            "content": "Goal [HIGH]: Register at example.com",
        }
    ]


def test_get_identity_sanitizes_profile_and_filters_invalid_people():
    client = _make_client()
    updates = []
    profile = SimpleNamespace(
        id="profile-1",
        metadata={
            "name": "Богдан",
            "occupation": "user@example.com",
            "notes": "Email: user@example.com; Номер телефону: +380000000000; Project: AuraSDK",
            "verified": True,
        },
    )
    valid_person = SimpleNamespace(
        id="person-1",
        content="Марія, mother",
        metadata={
            "type": "person",
            "full_name": "Марія",
            "role": "мати",
            "verified": False,
            "trust_score": 0.5,
        },
    )
    invalid_person = SimpleNamespace(
        id="person-2",
        content="[2026-03-22 Sunday] {'type': 'text', 'text': 'bad payload'}",
        metadata={
            "full_name": "[2026-03-22 Sunday] {'type': 'text', 'text': 'bad payload'}",
            "verified": False,
            "trust_score": 0.5,
        },
    )
    fake_api = SimpleNamespace(
        brain_lock=nullcontext(),
        brain=SimpleNamespace(
            update=lambda record_id, **kwargs: updates.append((record_id, kwargs)),
            search=lambda query, tags, limit=50: (
                [profile] if "user-profile" in tags else [valid_person, invalid_person]
            )
        ),
    )

    with patch("remy.web.routes.knowledge_routes._get_api", return_value=fake_api):
        response = client.get("/api/knowledge/identity")

    assert response.status_code == 200
    payload = response.json()
    assert payload["profile"]["occupation"] == ""
    assert payload["profile"]["email"] == "user@example.com"
    assert payload["profile"]["phone"] == "+380000000000"
    assert "user@example.com" not in payload["profile"]["notes"]
    assert [person["full_name"] for person in payload["people"]] == ["Марія"]
    assert updates[0][0] == "profile-1"
    assert updates[1][0] == "person-2"


def test_get_identity_merges_generic_family_reference_into_named_person():
    client = _make_client()
    profile = SimpleNamespace(
        id="profile-1",
        metadata={
            "name": "Богдан",
            "family": "бабуся Ганна (01.01.1940), брат Тарас (01.01.1990)",
            "verified": True,
        },
    )
    generic_grandmother = SimpleNamespace(
        id="person-1",
        content="Бабуся Богдана",
        metadata={
            "type": "person",
            "full_name": "Бабуся Богдана",
            "role": "бабуся",
            "verified": False,
            "trust_score": 0.5,
        },
    )
    named_grandmother = SimpleNamespace(
        id="person-2",
        content="Ганна",
        metadata={
            "type": "person",
            "full_name": "Ганна",
            "birth_date": "01.01.1940",
            "verified": False,
            "trust_score": 0.5,
        },
    )
    fake_api = SimpleNamespace(
        brain_lock=nullcontext(),
        brain=SimpleNamespace(
            update=lambda *args, **kwargs: None,
            search=lambda query, tags, limit=50: (
                [profile]
                if "user-profile" in tags
                else [generic_grandmother, named_grandmother]
            ),
        ),
    )

    with patch("remy.web.routes.knowledge_routes._get_api", return_value=fake_api):
        response = client.get("/api/knowledge/identity")

    assert response.status_code == 200
    payload = response.json()
    assert [person["full_name"] for person in payload["people"]] == ["Ганна"]
    assert payload["people"][0]["birth_date"] == "01.01.1940"
    assert "Бабуся Богдана" in payload["people"][0]["aliases"]
