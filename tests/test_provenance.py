"""Tests for Memory Provenance — anti-hallucination trust system."""

import json
from unittest.mock import patch

import pytest


@pytest.fixture
def mock_brain(tmp_path):
    """Real CognitiveMemory for integration testing."""
    from aura import Aura as CognitiveMemory
    b = CognitiveMemory(str(tmp_path / "test_brain"))
    yield b
    b.close()


@pytest.fixture
def execute_tool(mock_brain, tmp_path):
    """Provide execute_tool with mocked brain and isolated knowledge."""
    import threading
    from unittest.mock import MagicMock
    mock_kb = MagicMock()
    mock_kb.retrieve_matrix.return_value = ([], [], [])
    mock_kb.list_memories.return_value = []

    with patch("remy.core.brain_tools.brain", mock_brain), \
         patch("remy.core.brain_tools._registry", None), \
         patch("remy.core.agent_tools.knowledge", mock_kb), \
         patch("remy.core.agent_tools.knowledge_lock", threading.Lock()), \
         patch("remy.core.tool_registry.settings") as mock_settings:
        mock_settings.SANDBOX_DIR = tmp_path / "sandbox"
        mock_settings.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"

        from remy.core.brain_tools import execute_tool
        yield execute_tool


# ============== _get_provenance ==============

class TestGetProvenance:

    def test_autonomous_channel(self):
        from remy.core.brain_tools import _get_provenance
        p = _get_provenance("autonomous")
        assert p["source"] == "agent-autonomous"
        assert p["verified"] is False

    def test_desktop_channel(self):
        from remy.core.brain_tools import _get_provenance
        p = _get_provenance("desktop")
        assert p["source"] == "agent-interactive"
        assert p["verified"] is False

    def test_telegram_channel(self):
        from remy.core.brain_tools import _get_provenance
        p = _get_provenance("telegram")
        assert p["source"] == "agent-interactive"
        assert p["verified"] is False

    def test_voice_channel(self):
        from remy.core.brain_tools import _get_provenance
        p = _get_provenance("voice")
        assert p["source"] == "agent-interactive"
        assert p["verified"] is False

    def test_system_channel(self):
        from remy.core.brain_tools import _get_provenance
        p = _get_provenance("system")
        assert p["source"] == "system"
        assert p["verified"] is False

    def test_none_channel(self):
        from remy.core.brain_tools import _get_provenance
        p = _get_provenance(None)
        assert p["source"] == "agent"
        assert p["verified"] is False

    def test_proactive_channel(self):
        from remy.core.brain_tools import _get_provenance
        p = _get_provenance("proactive")
        assert p["source"] == "agent-interactive"
        assert p["verified"] is False


# ============== _stamp_provenance ==============

class TestStampProvenance:

    def test_stamps_empty_metadata(self):
        from remy.core.brain_tools import _stamp_provenance
        result = _stamp_provenance(None, "autonomous")
        assert result["source"] == "agent-autonomous"
        assert result["verified"] is False

    def test_stamps_existing_metadata(self):
        from remy.core.brain_tools import _stamp_provenance
        result = _stamp_provenance({"type": "test"}, "desktop")
        assert result["type"] == "test"
        assert result["source"] == "agent-interactive"
        assert result["verified"] is False

    def test_preserves_explicit_source(self):
        from remy.core.brain_tools import _stamp_provenance
        result = _stamp_provenance({"source": "user-confirmed", "verified": True}, "autonomous")
        assert result["source"] == "user-confirmed"
        assert result["verified"] is True

    def test_does_not_mutate_input(self):
        from remy.core.brain_tools import _stamp_provenance
        original = {"type": "test"}
        result = _stamp_provenance(original, "autonomous")
        assert "source" not in original
        assert "source" in result


# ============== Store tool provenance ==============

class TestStoreProvenance:

    def test_store_with_autonomous_channel(self, execute_tool, mock_brain):
        result = execute_tool("store", {"content": "Test fact", "tags": "test"},
                              channel="autonomous")
        data = json.loads(result)
        rec = mock_brain.get(data["id"])
        assert rec.metadata["source"] == "agent-autonomous"
        assert rec.metadata["verified"] is False

    def test_store_with_desktop_channel(self, execute_tool, mock_brain):
        result = execute_tool("store", {"content": "Test fact", "tags": "test"},
                              channel="desktop")
        data = json.loads(result)
        rec = mock_brain.get(data["id"])
        assert rec.metadata["source"] == "agent-interactive"
        assert rec.metadata["verified"] is False

    def test_store_with_no_channel(self, execute_tool, mock_brain):
        result = execute_tool("store", {"content": "Test fact", "tags": "test"})
        data = json.loads(result)
        rec = mock_brain.get(data["id"])
        assert rec.metadata["source"] == "agent"
        assert rec.metadata["verified"] is False


class TestStorePersonProvenance:

    def test_store_person_stamps_provenance(self, execute_tool, mock_brain):
        result = execute_tool("store_person", {
            "full_name": "Maria",
            "relation": "grandmother",
        }, channel="desktop")
        data = json.loads(result)
        rec = mock_brain.get(data["id"])
        assert rec.metadata["source"] == "agent-interactive"
        assert "verified" in rec.metadata


class TestStoreStoryProvenance:

    def test_store_story_stamps_provenance(self, execute_tool, mock_brain):
        result = execute_tool("store_story", {
            "title": "Test Story",
            "content": "A story about testing provenance.",
        }, channel="telegram")
        data = json.loads(result)
        rec = mock_brain.get(data["id"])
        assert rec.metadata["source"] == "agent-interactive"
        assert rec.metadata["verified"] is False


# ============== verify_record ==============

class TestVerifyRecord:

    def test_verify_sets_verified_true(self, execute_tool, mock_brain):
        store_result = execute_tool("store", {"content": "Unverified fact"},
                                    channel="autonomous")
        record_id = json.loads(store_result)["id"]

        result = execute_tool("verify_record", {"record_id": record_id})
        data = json.loads(result)
        assert data["verified"] is True
        assert data["record_id"] == record_id

        rec = mock_brain.get(record_id)
        assert rec.metadata["verified"] is True
        assert rec.metadata["verified_by"] == "user"
        assert "verified_at" in rec.metadata

    def test_verify_with_note(self, execute_tool, mock_brain):
        store_result = execute_tool("store", {"content": "Some fact"},
                                    channel="autonomous")
        record_id = json.loads(store_result)["id"]

        execute_tool("verify_record", {
            "record_id": record_id,
            "note": "User confirmed via chat",
        })

        rec = mock_brain.get(record_id)
        assert rec.metadata["verification_note"] == "User confirmed via chat"

    def test_verify_not_found(self, execute_tool):
        result = execute_tool("verify_record", {"record_id": "nonexistent-id"})
        data = json.loads(result)
        assert "error" in data


class TestProtectedRecordRetrieval:

    def test_autonomous_cannot_read_protected_exact_fields(self, execute_tool):
        store_result = execute_tool(
            "store_user_profile",
            {"name": "Taras", "email": "taras@example.com"},
            channel="desktop",
        )
        record_id = json.loads(store_result)["id"]

        result = execute_tool(
            "get_protected_record",
            {"record_id": record_id, "fields": "email"},
            channel="autonomous",
        )
        data = json.loads(result)

        assert "error" in data
        assert "interactive channels" in data["error"]


# ============== Channel ContextVar ==============

class TestChannelContextVar:

    def test_set_and_get_channel(self):
        from remy.core.langgraph_tools import set_channel, get_channel
        set_channel("autonomous")
        assert get_channel() == "autonomous"
        set_channel("desktop")
        assert get_channel() == "desktop"
        set_channel(None)
        assert get_channel() is None


# ============== System instruction ==============

class TestSystemInstructionProvenance:

    def test_trust_rules_in_instruction(self):
        with patch("remy.core.brain_tools.brain") as mock_b:
            mock_b.search.return_value = []
            mock_b.recall.return_value = ""
            from remy.core.brain_tools import build_system_instruction
            instruction = build_system_instruction(channel="desktop")
            assert "Trust Scores" in instruction
            assert "verify_record" in instruction
            assert "trust" in instruction.lower()


# ============== Autonomy provenance ==============

class TestAutonomyProvenance:

    def test_create_goal_has_provenance(self, mock_brain):
        with patch("remy.core.autonomy.brain", mock_brain):
            from remy.core.autonomy import create_goal
            record_id = create_goal("Test goal", priority="medium")
            rec = mock_brain.get(record_id)
            assert rec.metadata["source"] == "agent-autonomous"
            assert rec.metadata["verified"] is False
            assert rec.metadata["trust_score"] == 0.4


# ============== Trust Scores ==============

class TestTrustScoreMapping:

    def test_get_provenance_includes_trust_score(self):
        from remy.core.brain_tools import _get_provenance
        assert _get_provenance("autonomous")["trust_score"] == 0.4
        assert _get_provenance("desktop")["trust_score"] == 0.7
        assert _get_provenance("system")["trust_score"] == 0.6
        assert _get_provenance(None)["trust_score"] == 0.5

    def test_stamp_provenance_adds_trust_score(self):
        from remy.core.brain_tools import _stamp_provenance
        result = _stamp_provenance(None, "autonomous")
        assert result["trust_score"] == 0.4

    def test_stamp_preserves_explicit_trust(self):
        from remy.core.brain_tools import _stamp_provenance
        result = _stamp_provenance({"trust_score": 1.0}, "autonomous")
        assert result["trust_score"] == 1.0


class TestComputeEffectiveTrust:

    def test_no_decay_for_interactive(self):
        import time
        from remy.core.brain_tools import _compute_effective_trust
        meta = {"source": "agent-interactive", "trust_score": 0.7}
        # Even with old created_at, interactive records don't decay
        old_time = time.time() - (60 * 86400)  # 60 days ago
        assert _compute_effective_trust(meta, old_time) == 0.7

    def test_decay_for_autonomous(self):
        import time
        from remy.core.brain_tools import _compute_effective_trust
        meta = {"source": "agent-autonomous", "trust_score": 0.4}
        # 15 days old → age_factor = max(0.5, 1.0 - 15/30) = 0.5
        old_time = time.time() - (15 * 86400)
        result = _compute_effective_trust(meta, old_time)
        assert result == 0.2  # 0.4 * 0.5

    def test_floor_at_half(self):
        import time
        from remy.core.brain_tools import _compute_effective_trust
        meta = {"source": "agent-autonomous", "trust_score": 0.4}
        # 60 days old → age_factor = max(0.5, 1.0 - 60/30) = 0.5 (floored)
        old_time = time.time() - (60 * 86400)
        result = _compute_effective_trust(meta, old_time)
        assert result == 0.2  # 0.4 * 0.5 (floor)

    def test_verified_no_decay(self):
        import time
        from remy.core.brain_tools import _compute_effective_trust
        meta = {"source": "agent-autonomous", "trust_score": 1.0}
        # trust >= 1.0 so no decay even for autonomous
        old_time = time.time() - (60 * 86400)
        assert _compute_effective_trust(meta, old_time) == 1.0

    def test_default_trust_for_legacy_records(self):
        import time
        from remy.core.brain_tools import _compute_effective_trust
        # Legacy record without trust_score → defaults to 0.5
        meta = {}
        assert _compute_effective_trust(meta, time.time()) == 0.5


class TestRecallTrustAnnotations:

    def test_recall_shows_trust_format(self, execute_tool, mock_brain):
        # Store a record with autonomous channel
        execute_tool("store", {"content": "Wallet created on Base", "tags": "crypto"},
                     channel="autonomous")
        result = execute_tool("recall", {"query": "Wallet created on Base"}, channel="desktop")
        assert "[trust:" in result
        assert "Wallet created on Base" in result

    def test_recall_empty(self, execute_tool):
        result = execute_tool("recall", {"query": "nonexistent_xyz_query_42"})
        assert "No relevant memories" in result


class TestVerifyRecordTrust:

    def test_verify_sets_trust_1(self, execute_tool, mock_brain):
        store_result = execute_tool("store", {"content": "Unverified claim"},
                                    channel="autonomous")
        record_id = json.loads(store_result)["id"]

        rec_before = mock_brain.get(record_id)
        assert rec_before.metadata["trust_score"] == 0.4

        execute_tool("verify_record", {"record_id": record_id})

        rec_after = mock_brain.get(record_id)
        assert rec_after.metadata["trust_score"] == 1.0
        assert rec_after.metadata["verified"] is True


class TestSearchTrust:

    def test_search_includes_trust_field(self, execute_tool):
        hit = {
            "id": "rec-trust",
            "content": "Searchable trust test item",
            "tags": ["trust-test"],
            "level": "DOMAIN",
            "metadata": {"source": "agent-autonomous", "trust_score": 0.4},
            "score": 0.9,
        }
        with patch("remy.core.brain_tools.hybrid_search_structured", return_value=[hit]):
            result = execute_tool("search", {"query": "Searchable trust test", "tags": "trust-test"})
        data = json.loads(result)
        assert len(data) > 0
        assert "trust" in data[0]
        assert "source" in data[0]
        assert data[0]["trust"] <= 0.4  # autonomous trust
        assert data[0]["source"] == "agent-autonomous"
