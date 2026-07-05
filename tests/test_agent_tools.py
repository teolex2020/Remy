"""
Tests for _AuraCompat — the thin wrapper around AuraSDK.

Critical compatibility layer: all metadata goes through str serialization
(Rust core requires string values) and deserialization on read.
"""

import pytest

from remy.core.agent_tools import _AuraCompat
from remy.core.agent_tools import _LevelCompat as Level


@pytest.fixture
def brain(tmp_path):
    """Fresh _AuraCompat instance in temp dir."""
    b = _AuraCompat(str(tmp_path / "brain"))
    yield b
    b.close()


# ============== Metadata roundtrip (most critical tests) ==============


class TestMetadataRoundtrip:
    """store → get → verify metadata values survive str serialization."""

    def test_bool_true_roundtrip(self, brain):
        rec = brain.store("test content", metadata={"verified": True})
        fetched = brain.get(rec.id)
        assert fetched.metadata["verified"] is True

    def test_bool_false_roundtrip(self, brain):
        rec = brain.store("test content", metadata={"actionable": False})
        fetched = brain.get(rec.id)
        assert fetched.metadata["actionable"] is False

    def test_none_roundtrip(self, brain):
        rec = brain.store("test content", metadata={"deadline": None})
        fetched = brain.get(rec.id)
        assert fetched.metadata["deadline"] is None

    def test_int_roundtrip(self, brain):
        rec = brain.store("test", metadata={"attempts": 5})
        fetched = brain.get(rec.id)
        assert fetched.metadata["attempts"] == 5
        assert isinstance(fetched.metadata["attempts"], int)

    def test_float_roundtrip(self, brain):
        rec = brain.store("test", metadata={"trust_score": 0.85})
        fetched = brain.get(rec.id)
        assert abs(fetched.metadata["trust_score"] - 0.85) < 0.001
        assert isinstance(fetched.metadata["trust_score"], float)

    def test_list_roundtrip(self, brain):
        rec = brain.store("test", metadata={"sources": ["url1", "url2"]})
        fetched = brain.get(rec.id)
        assert fetched.metadata["sources"] == ["url1", "url2"]
        assert isinstance(fetched.metadata["sources"], list)

    def test_dict_roundtrip(self, brain):
        rec = brain.store("test", metadata={"extra": {"key": "value"}})
        fetched = brain.get(rec.id)
        assert fetched.metadata["extra"] == {"key": "value"}

    def test_string_stays_string(self, brain):
        rec = brain.store("test", metadata={"source": "user-confirmed"})
        fetched = brain.get(rec.id)
        assert fetched.metadata["source"] == "user-confirmed"
        assert isinstance(fetched.metadata["source"], str)

    def test_known_string_key_not_parsed_as_number(self, brain):
        """timestamp, created_at, session_id etc. must stay strings even if they look numeric."""
        rec = brain.store(
            "test",
            metadata={
                "timestamp": "2026-02-25T19:00:00",
                "session_id": "abc-123",
                "goal_id": "goal-abc123",
            },
        )
        fetched = brain.get(rec.id)
        assert isinstance(fetched.metadata["timestamp"], str)
        assert isinstance(fetched.metadata["session_id"], str)
        assert isinstance(fetched.metadata["goal_id"], str)

    def test_mixed_metadata_at_once(self, brain):
        """Real-world metadata used by brain_tools (trust guard scenario)."""
        meta = {
            "source": "agent-autonomous",
            "verified": False,
            "actionable": False,
            "trust_score": 0.4,
            "attempts": 0,
            "deadline": None,
            "outcome_ids": [],
        }
        rec = brain.store("Goal: research topic X", metadata=meta)
        fetched = brain.get(rec.id)
        m = fetched.metadata

        assert m["source"] == "agent-autonomous"
        assert m["verified"] is False
        assert m["actionable"] is False
        assert abs(m["trust_score"] - 0.4) < 0.001
        assert m["attempts"] == 0
        assert m["deadline"] is None
        assert m["outcome_ids"] == []

    def test_update_metadata_roundtrip(self, brain):
        """update() preserves deserialization after round-trip."""
        rec = brain.store("test", metadata={"verified": False, "attempts": 0})
        brain.update(rec.id, metadata={"verified": True, "attempts": 3})
        fetched = brain.get(rec.id)
        assert fetched.metadata["verified"] is True
        assert fetched.metadata["attempts"] == 3


# ============== search() returns _RecordProxy ==============


class TestSearchReturnsProxy:
    def test_search_metadata_deserialized(self, brain):
        brain.store(
            "recall-able content", tags=["test"], metadata={"verified": True, "trust_score": 0.9}
        )
        results = brain.search(query="recall-able", limit=5)
        assert len(results) >= 1
        m = results[0].metadata
        assert m["verified"] is True
        assert abs(m["trust_score"] - 0.9) < 0.001

    def test_list_records_metadata_deserialized(self, brain):
        brain.store("list test", tags=["search-test"], metadata={"actionable": True, "attempts": 2})
        results = brain.list_records(tags=["search-test"])
        assert len(results) >= 1
        m = results[0].metadata
        assert m["actionable"] is True
        assert m["attempts"] == 2


# ============== Level compat ==============


class TestLevelCompat:
    def test_uppercase_working(self):
        assert Level.WORKING is not None

    def test_capitalized_working(self):
        assert Level.Working is not None

    def test_same_value(self):
        assert Level.WORKING == Level.Working
        assert Level.DOMAIN == Level.Domain
        assert Level.IDENTITY == Level.Identity
        assert Level.DECISIONS == Level.Decisions


# ============== store() returns _StoreResult ==============


class TestStoreResult:
    def test_store_returns_id_attribute(self, brain):
        result = brain.store("test content")
        assert hasattr(result, "id")
        assert isinstance(result.id, str)
        assert len(result.id) > 0

    def test_store_result_str(self, brain):
        result = brain.store("test")
        assert str(result) == result.id


# ============== insights() backward compat ==============


class TestInsightsCompat:
    def test_insights_returns_list(self, brain):
        result = brain.insights()
        assert isinstance(result, list)

    def test_insights_can_surface_hot_topic(self, brain):
        brain.store("User tracks blood pressure daily", tags=["health", "blood-pressure"])
        brain.store("Blood pressure improved this week", tags=["health", "blood-pressure"])

        result = brain.insights()

        assert any(ins.get("type") == "hot_topic" for ins in result)


# ============== brain_lock is a no-op ==============


class TestNoopLock:
    def test_brain_lock_noop(self):
        from remy.core.agent_tools import brain_lock

        with brain_lock:
            pass  # Should not raise

    def test_knowledge_removed(self):
        """knowledge is kept as None stub for backward compat (unified into brain)."""
        import remy.core.agent_tools as at

        # knowledge = None (stub for backward compat with test patches)
        assert not hasattr(at, "knowledge") or at.knowledge is None
