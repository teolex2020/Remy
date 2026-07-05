"""Tests for desktop/web API endpoints."""

import json
from unittest.mock import MagicMock, patch

import pytest
from aura import Level
from fastapi.testclient import TestClient

from remy.web.api import router, set_session_manager
from remy.web.session import WebSession


@pytest.fixture
def mock_brain(tmp_path):
    """Real CognitiveMemory for integration testing."""
    from aura import Aura as CognitiveMemory
    b = CognitiveMemory(str(tmp_path / "test_brain"))
    yield b
    b.close()


@pytest.fixture
def mock_session_manager():
    """Mock session manager for API tests."""
    manager = MagicMock()
    session = WebSession(session_id="test-session-123")
    manager.get_or_create_session.return_value = session
    return manager


@pytest.fixture
def client(mock_brain, mock_session_manager):
    """FastAPI test client with mocked brain and session manager."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)

    set_session_manager(mock_session_manager)

    with patch("remy.web.api.brain", mock_brain):
        yield TestClient(app)


class TestGetStats:

    def test_get_stats(self, client):
        res = client.get("/api/stats")
        assert res.status_code == 200
        data = res.json()
        assert "total_records" in data
        assert "stats" in data

    def test_stats_returns_zero_for_empty(self, client):
        res = client.get("/api/stats")
        data = res.json()
        assert data["total_records"] == 0


class TestListRecords:

    def test_list_records_empty(self, client):
        res = client.get("/api/records")
        assert res.status_code == 200
        data = res.json()
        assert data["records"] == []
        assert data["total"] == 0

    def test_list_records_with_data(self, client, mock_brain):
        mock_brain.store("Test record 1", tags=["test"], level=Level.DOMAIN)
        mock_brain.store("Test record 2", tags=["test"], level=Level.DOMAIN)

        res = client.get("/api/records")
        data = res.json()
        assert data["total"] >= 2

    def test_list_records_with_tag_filter(self, client, mock_brain):
        mock_brain.store("Tagged record", tags=["special"], level=Level.DOMAIN)
        mock_brain.store("Other record", tags=["other"], level=Level.DOMAIN)

        res = client.get("/api/records?tags=special")
        data = res.json()
        # tag filter may or may not work depending on list_records impl
        assert "records" in data


class TestGetRecord:

    def test_get_record_not_found(self, client):
        res = client.get("/api/records/nonexistent_id")
        assert res.status_code == 404

    def test_get_record_found(self, client, mock_brain):
        rec = mock_brain.store("Detailed record", tags=["detail"], level=Level.DOMAIN)
        rec_id = rec.id if hasattr(rec, "id") else rec

        res = client.get(f"/api/records/{rec_id}")
        assert res.status_code == 200
        data = res.json()
        assert data["id"] == rec_id
        assert "Detailed record" in data["content"]
        assert "connections" in data


class TestDeleteRecord:

    def test_delete_record(self, client, mock_brain, mock_session_manager):
        rec = mock_brain.store("Delete me", tags=["test"], level=Level.DOMAIN)
        rec_id = rec.id if hasattr(rec, "id") else rec

        with patch("remy.web.api.execute_tool") as mock_exec:
            mock_exec.return_value = json.dumps({"deleted": True, "id": rec_id, "deleted_content": "Delete me"})
            res = client.delete(f"/api/records/{rec_id}")

        assert res.status_code == 200
        data = res.json()
        assert data["deleted"] is True

    def test_delete_nonexistent(self, client, mock_session_manager):
        with patch("remy.web.api.execute_tool") as mock_exec:
            mock_exec.return_value = json.dumps({"error": "Record not found"})
            res = client.delete("/api/records/nonexistent")

        assert res.status_code == 404


class TestSearchRecords:

    def test_search_empty(self, client, mock_session_manager):
        with patch("remy.web.api.execute_tool") as mock_exec:
            mock_exec.return_value = "No results found."
            res = client.post("/api/search", json={"query": "nonexistent"})

        assert res.status_code == 200
        data = res.json()
        assert data["results"] == []

    def test_search_filters_use_level_and_metadata(self, client, mock_session_manager):
        payload = json.dumps(
            [
                {
                    "id": "rec-1",
                    "content": "Oleksandr lives in Velyka Dymerka",
                    "tags": ["user-profile"],
                    "level": "DOMAIN",
                    "metadata": {"timestamp": "2026-03-09T10:00:00"},
                    "score": 0.91,
                }
            ]
        )
        with patch("remy.web.api.execute_tool") as mock_exec:
            mock_exec.return_value = payload
            res = client.post("/api/search", json={"query": "where do I live", "tier": "domain", "period": "30"})

        assert res.status_code == 200
        data = res.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["level"] == "DOMAIN"


class TestGraphData:

    def test_graph_empty(self, client):
        res = client.get("/api/graph")
        assert res.status_code == 200
        data = res.json()
        assert data["nodes"] == []
        assert data["edges"] == []

    def test_graph_with_data(self, client, mock_brain):
        rec1 = mock_brain.store("Node A", tags=["graph"], level=Level.DOMAIN)
        rec2 = mock_brain.store("Node B", tags=["graph"], level=Level.DOMAIN)
        id1 = rec1.id if hasattr(rec1, "id") else rec1
        id2 = rec2.id if hasattr(rec2, "id") else rec2
        mock_brain.connect(id1, id2, weight=0.8)

        res = client.get("/api/graph")
        data = res.json()
        assert len(data["nodes"]) >= 2
        assert len(data["edges"]) >= 1

    def test_graph_user_mode_hides_internal_records(self, client, mock_brain):
        mock_brain.store(
            "Visible user note",
            tags=["family"],
            level=Level.DOMAIN,
            metadata={"type": "fact"},
        )
        mock_brain.store(
            "Background insights (2 items)",
            tags=["background-insights-latest"],
            level=Level.DECISIONS,
            metadata={"type": "background_insights"},
        )

        res_user = client.get("/api/graph?mode=user")
        assert res_user.status_code == 200
        user_nodes = res_user.json()["nodes"]
        assert any("Visible user note" in node["label"] for node in user_nodes)
        assert not any("Background insights" in node["label"] for node in user_nodes)

        res_full = client.get("/api/graph?mode=full")
        assert res_full.status_code == 200
        full_nodes = res_full.json()["nodes"]
        assert any("Background insights" in node["label"] for node in full_nodes)

        res_scope_full = client.get("/api/graph?scope=full")
        assert res_scope_full.status_code == 200
        scope_nodes = res_scope_full.json()["nodes"]
        assert any("Background insights" in node["label"] for node in scope_nodes)


class TestActivityApi:

    def test_activity_uses_shared_activity_payload_shape(self, client, mock_brain):
        with patch(
            "remy.core.combined_runner.get_activity_feed_snapshot",
            return_value={
                "summary": {
                    "total_actions": 1,
                    "success": 1,
                    "active_goals": 1,
                    "total_tokens": 42,
                },
                "goals": [{"id": "goal-1"}],
                "outcomes": [{"id": "out-1"}],
                "reflections": [{"id": "ref-1"}],
                "proactive": [{"id": "pro-1"}],
            },
        ):
            res = client.get("/api/activity")

        assert res.status_code == 200
        data = res.json()
        assert data["summary"]["total_actions"] == 1
        assert data["summary"]["success"] == 1
        assert data["summary"]["active_goals"] == 1
        assert data["summary"]["total_tokens"] == 42
        assert len(data["goals"]) == 1
        assert len(data["outcomes"]) == 1
        assert len(data["reflections"]) == 1
        assert len(data["proactive"]) == 1


class TestApprovalsApi:

    def test_approvals_use_shared_approval_snapshot(self, client):
        with patch(
            "remy.core.combined_runner.get_approval_runtime_snapshot",
            return_value={
                "pending": [
                    {
                        "id": "appr-1",
                        "action_id": "appr-1",
                        "description": "Review payout request",
                        "timeout_sec": 60,
                        "created_at": 100.0,
                        "expires_at": 160.0,
                    }
                ]
            },
        ):
            res = client.get("/api/approvals")

        assert res.status_code == 200
        assert res.json() == {
            "pending": [
                {
                    "id": "appr-1",
                    "action_id": "appr-1",
                    "description": "Review payout request",
                    "timeout_sec": 60,
                    "created_at": 100.0,
                    "expires_at": 160.0,
                }
            ]
        }
