"""Tests for unified dual-layer memory (Cognitive ↔ Knowledge sync)."""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest
from aura import Level


# --------------- helpers ---------------

@pytest.fixture
def mock_brain(tmp_path):
    from aura import Aura as CognitiveMemory
    b = CognitiveMemory(str(tmp_path / "test_brain"))
    yield b
    b.close()


@pytest.fixture
def mock_knowledge():
    """Mock AuraMemory with the methods we use."""
    kb = MagicMock()
    kb.process.return_value = "stored"
    kb.retrieve.return_value = []
    kb.retrieve_full.return_value = []
    kb.retrieve_matrix.return_value = ([], [], [])  # (scores, ids, matches)
    kb.list_memories.return_value = []
    kb.flush.return_value = None
    kb.count.return_value = 0
    return kb


@pytest.fixture
def knowledge_lock():
    return threading.Lock()


@pytest.fixture
def execute_tool(mock_brain, mock_knowledge, knowledge_lock, tmp_path):
    """execute_tool with both brain and knowledge mocked."""
    with patch("remy.core.brain_tools.brain", mock_brain), \
         patch("remy.core.brain_tools.brain_lock", threading.RLock()), \
         patch("remy.core.brain_tools._registry", None), \
         patch("remy.core.agent_tools.knowledge", mock_knowledge), \
         patch("remy.core.agent_tools.knowledge_lock", knowledge_lock), \
         patch("remy.core.tool_registry.settings") as mock_settings:
        mock_settings.SANDBOX_DIR = tmp_path / "sandbox"
        mock_settings.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"

        from remy.core.brain_tools import execute_tool
        yield execute_tool


# --------------- _should_sync ---------------

class TestShouldSync:

    def test_identity_syncs_as_anchor(self):
        from remy.core.brain_tools import _should_sync

        should, pin = _should_sync(Level.IDENTITY)
        assert should is True
        assert pin is True

    def test_domain_syncs_without_pin(self):
        from remy.core.brain_tools import _should_sync

        should, pin = _should_sync(Level.DOMAIN)
        assert should is True
        assert pin is False

    def test_working_does_not_sync(self):
        from remy.core.brain_tools import _should_sync

        should, pin = _should_sync(Level.WORKING)
        assert should is False
        assert pin is False

    def test_string_level(self):
        from remy.core.brain_tools import _should_sync
        should, pin = _should_sync("IDENTITY")
        assert should is True
        assert pin is True


# --------------- _sync_to_knowledge ---------------

@pytest.mark.real_sync
class TestSyncToKnowledge:

    def test_sync_succeeds(self, mock_knowledge, knowledge_lock):
        with patch("remy.core.agent_tools.knowledge", mock_knowledge), \
             patch("remy.core.agent_tools.knowledge_lock", knowledge_lock):
            from remy.core.brain_tools import _sync_to_knowledge
            result = _sync_to_knowledge("This is important content to store")
            assert result is True
            mock_knowledge.process.assert_called_once()

    def test_sync_noop_when_knowledge_none(self, knowledge_lock):
        with patch("remy.core.agent_tools.knowledge", None), \
             patch("remy.core.agent_tools.knowledge_lock", knowledge_lock):
            from remy.core.brain_tools import _sync_to_knowledge
            result = _sync_to_knowledge("Some content here for testing")
            assert result is False

    def test_sync_skips_short_content(self, mock_knowledge, knowledge_lock):
        with patch("remy.core.agent_tools.knowledge", mock_knowledge), \
             patch("remy.core.agent_tools.knowledge_lock", knowledge_lock):
            from remy.core.brain_tools import _sync_to_knowledge
            result = _sync_to_knowledge("short")
            assert result is False
            mock_knowledge.process.assert_not_called()

    def test_sync_ignores_errors(self, mock_knowledge, knowledge_lock):
        mock_knowledge.process.side_effect = RuntimeError("DB error")
        with patch("remy.core.agent_tools.knowledge", mock_knowledge), \
             patch("remy.core.agent_tools.knowledge_lock", knowledge_lock):
            from remy.core.brain_tools import _sync_to_knowledge
            result = _sync_to_knowledge("This should not raise an exception")
            assert result is False  # Failed, but no exception raised

    def test_sync_dedup_skips_existing(self, mock_knowledge, knowledge_lock):
        # _sync_to_knowledge uses retrieve_matrix for dedup (not retrieve_full)
        # retrieve_matrix returns (scores, ids, texts) tuple
        mock_knowledge.retrieve_matrix.return_value = ([0.85], ["id1"], ["Already exists in memory"])
        with patch("remy.core.agent_tools.knowledge", mock_knowledge), \
             patch("remy.core.agent_tools.knowledge_lock", knowledge_lock):
            from remy.core.brain_tools import _sync_to_knowledge
            result = _sync_to_knowledge("Already exists in memory exactly")
            assert result is False
            mock_knowledge.process.assert_not_called()

    def test_sync_no_dedup_when_disabled(self, mock_knowledge, knowledge_lock):
        mock_knowledge.retrieve.return_value = ["Already exists"]
        with patch("remy.core.agent_tools.knowledge", mock_knowledge), \
             patch("remy.core.agent_tools.knowledge_lock", knowledge_lock):
            from remy.core.brain_tools import _sync_to_knowledge
            result = _sync_to_knowledge("Already exists in memory base", deduplicate=False)
            assert result is True
            mock_knowledge.process.assert_called_once()


# --------------- Unified recall ---------------

class TestUnifiedRecall:

    def test_recall_brain_only(self, execute_tool, mock_brain, mock_knowledge):
        """When KB returns nothing, only brain results appear."""
        mock_knowledge.retrieve_full.return_value = []

        mock_brain.store(content="Grandmother Maria was born in Kyiv",
                         level=Level.DOMAIN, tags=["person"])
        result = execute_tool("recall", {"query": "grandmother Maria"})
        assert "Maria" in result or "Kyiv" in result

    def test_recall_kb_only(self, execute_tool, mock_knowledge):
        """When brain returns nothing, KB results appear."""
        mock_knowledge.retrieve_matrix.return_value = (
            [5.0],
            ["kb1"],
            [{"text": "Vitamin D aids calcium absorption", "id": "kb1", "intensity": 1.0, "dna": "general"}]
        )
        result = execute_tool("recall", {"query": "vitamin D"})
        assert "KB" in result
        assert "Vitamin D" in result

    def test_recall_merges_both(self, execute_tool, mock_brain, mock_knowledge):
        """Both brain and KB results appear."""
        mock_brain.store(content="My doctor recommended vitamin D supplements",
                         level=Level.DOMAIN, tags=["health"])
        mock_knowledge.retrieve_matrix.return_value = (
            [3.5],
            ["kb2"],
            [{"text": "Vitamin D is essential for bone health", "id": "kb2", "intensity": 1.0, "dna": "general"}]
        )
        result = execute_tool("recall", {"query": "vitamin D"})
        assert "trust:" in result  # brain result
        assert "KB" in result      # knowledge result

    def test_recall_deduplicates(self, execute_tool, mock_brain, mock_knowledge):
        """Same content from both systems appears only once."""
        text = "Paris is the capital of France and a major European city"
        mock_brain.store(content=text, level=Level.DOMAIN, tags=["fact"])
        mock_knowledge.retrieve_matrix.return_value = (
            [8.0], ["kb3"],
            [{"text": text, "id": "kb3", "intensity": 1.0, "dna": "general"}]
        )
        result = execute_tool("recall", {"query": "Paris capital"})
        # Should appear only once (brain version takes priority)
        assert result.count("Paris") == 1

    def test_recall_empty_both(self, execute_tool, mock_knowledge):
        """Both empty returns 'No relevant memories'."""
        # retrieve_matrix default already returns empty via fixture
        result = execute_tool("recall", {"query": "nonexistent topic xyz"})
        assert "No relevant memories" in result

    def test_recall_kb_failure_graceful(self, execute_tool, mock_brain, mock_knowledge):
        """KB failure doesn't break recall — falls back to brain only."""
        mock_knowledge.retrieve_matrix.side_effect = RuntimeError("KB crashed")
        mock_knowledge.list_memories.side_effect = RuntimeError("KB crashed")
        mock_brain.store(content="Important brain memory about health",
                         level=Level.DOMAIN, tags=["health"])
        result = execute_tool("recall", {"query": "health"})
        assert "health" in result.lower()
        assert "KB" not in result  # No KB results due to error


# --------------- Store mirrors to knowledge ---------------

@pytest.mark.real_sync
class TestStoreMirrors:

    def test_store_mirrors(self, execute_tool, mock_knowledge):
        result = execute_tool("store", {
            "content": "Important fact about family history details",
            "tags": "family,history",
            "level": "L3_DOMAIN",
            "metadata": {"admission_class": "operator_asserted"},
        })
        data = json.loads(result)
        assert data["stored"] is True
        mock_knowledge.process.assert_called()

    def test_store_person_mirrors(self, execute_tool, mock_knowledge):
        result = execute_tool("store_person", {
            "full_name": "Maria Ivanova Testova",
            "role": "grandmother",
        })
        data = json.loads(result)
        assert data["stored"] is True
        mock_knowledge.process.assert_called()

    def test_store_no_mirror_for_system_tags(self, execute_tool, mock_knowledge):
        """Records with system tags should not mirror."""
        result = execute_tool("store", {
            "content": "This is a cache entry that should not sync to knowledge",
            "tags": "web-search-cache",
            "level": "L3_DOMAIN",
        })
        data = json.loads(result)
        assert data["stored"] is True
        mock_knowledge.process.assert_not_called()

    def test_store_working_level_no_mirror(self, execute_tool, mock_knowledge):
        """WORKING level should not sync."""
        result = execute_tool("store", {
            "content": "Temporary working memory that is ephemeral",
            "tags": "temp",
            "level": "L1_WORKING",
        })
        data = json.loads(result)
        assert data["stored"] is True
        mock_knowledge.process.assert_not_called()

    def test_store_identity_mirrors_as_anchor(self, execute_tool, mock_knowledge):
        """Profile records should mirror with pin=True."""
        result = execute_tool("store_user_profile", {
            "name": "Test User",
            "notes": "Core identity information about user profile",
        })
        data = json.loads(result)
        assert data.get("created") or data.get("updated")
        # pin=True for IDENTITY
        call_args = mock_knowledge.process.call_args
        assert call_args[1].get("pin") is True or call_args[0][1] is True

    def test_store_user_profile_mirrors_as_anchor(self, execute_tool, mock_knowledge):
        result = execute_tool("store_user_profile", {"name": "Test User"})
        data = json.loads(result)
        assert data.get("created") or data.get("updated")
        # Should be pinned
        mock_knowledge.process.assert_called()
        call_args = mock_knowledge.process.call_args
        assert call_args[1].get("pin") is True or (len(call_args[0]) > 1 and call_args[0][1] is True)


# --------------- _check_duplicates with KB ---------------

class TestCheckDuplicatesKB:

    def test_includes_kb_match(self, mock_brain, mock_knowledge, knowledge_lock):
        mock_knowledge.retrieve_matrix.return_value = (
            [0.9], ["kb10"],
            [{"text": "Very similar content found", "id": "kb10", "intensity": 1.0, "dna": "general"}]
        )
        with patch("remy.core.brain_tools.brain", mock_brain), \
             patch("remy.core.agent_tools.knowledge", mock_knowledge), \
             patch("remy.core.agent_tools.knowledge_lock", knowledge_lock):
            from remy.core.brain_tools import _check_duplicates
            results = _check_duplicates("similar content query here")
            kb_matches = [r for r in results if "kb_match" in r]
            assert len(kb_matches) >= 1
            assert kb_matches[0]["kb_score"] > 0.3

    def test_no_kb_match_below_threshold(self, mock_brain, mock_knowledge, knowledge_lock):
        mock_knowledge.retrieve_matrix.return_value = (
            [0.1], ["kb11"],
            [{"text": "Loosely related content", "id": "kb11", "intensity": 0.5, "dna": "general"}]
        )
        with patch("remy.core.brain_tools.brain", mock_brain), \
             patch("remy.core.agent_tools.knowledge", mock_knowledge), \
             patch("remy.core.agent_tools.knowledge_lock", knowledge_lock):
            from remy.core.brain_tools import _check_duplicates
            results = _check_duplicates("some query that is different")
            kb_matches = [r for r in results if "kb_match" in r]
            assert len(kb_matches) == 0


# --------------- Phase 9: _sync_knowledge ---------------

class TestSyncKnowledgePhase9:

    def test_syncs_unmirrored_records(self, mock_brain, mock_knowledge, knowledge_lock):

        mock_brain.store(content="Record that needs sync to knowledge base",
                         level=Level.DOMAIN, tags=["fact"])
        with patch("remy.core.agent_tools.knowledge", mock_knowledge), \
             patch("remy.core.agent_tools.knowledge_lock", knowledge_lock):
            from remy.core.background_brain import _sync_knowledge
            synced = _sync_knowledge(mock_brain)
            assert synced >= 1
            mock_knowledge.process.assert_called()
            mock_knowledge.flush.assert_called()

    def test_skips_already_mirrored(self, mock_brain, mock_knowledge, knowledge_lock):

        rec = mock_brain.store(content="Already mirrored record to knowledge",
                               level=Level.DOMAIN, tags=["fact"])
        mock_brain.update(rec.id, metadata={"mirrored_to_kb": True})
        with patch("remy.core.agent_tools.knowledge", mock_knowledge), \
             patch("remy.core.agent_tools.knowledge_lock", knowledge_lock):
            from remy.core.background_brain import _sync_knowledge
            synced = _sync_knowledge(mock_brain)
            assert synced == 0

    def test_skips_system_tags(self, mock_brain, mock_knowledge, knowledge_lock):

        mock_brain.store(content="Web search cache entry should not sync",
                         level=Level.DOMAIN, tags=["web-search-cache"])
        with patch("remy.core.agent_tools.knowledge", mock_knowledge), \
             patch("remy.core.agent_tools.knowledge_lock", knowledge_lock):
            from remy.core.background_brain import _sync_knowledge
            synced = _sync_knowledge(mock_brain)
            assert synced == 0

    def test_respects_cap(self, mock_brain, mock_knowledge, knowledge_lock):

        for i in range(60):
            mock_brain.store(content=f"Record number {i} content for sync test",
                             level=Level.DOMAIN, tags=["test"])
        with patch("remy.core.agent_tools.knowledge", mock_knowledge), \
             patch("remy.core.agent_tools.knowledge_lock", knowledge_lock):
            from remy.core.background_brain import _sync_knowledge
            synced = _sync_knowledge(mock_brain)
            assert synced <= 50  # _KB_SYNC_MAX_PER_RUN

    def test_noop_when_knowledge_none(self, mock_brain, knowledge_lock):
        with patch("remy.core.agent_tools.knowledge", None), \
             patch("remy.core.agent_tools.knowledge_lock", knowledge_lock):
            from remy.core.background_brain import _sync_knowledge
            synced = _sync_knowledge(mock_brain)
            assert synced == 0

    def test_pins_identity_records(self, mock_brain, mock_knowledge, knowledge_lock):

        mock_brain.store(content="User core identity profile information",
                         level=Level.IDENTITY, tags=["user-profile"])
        with patch("remy.core.agent_tools.knowledge", mock_knowledge), \
             patch("remy.core.agent_tools.knowledge_lock", knowledge_lock):
            from remy.core.background_brain import _sync_knowledge
            synced = _sync_knowledge(mock_brain)
            assert synced >= 1
            # Verify pin=True was used for IDENTITY
            call_args = mock_knowledge.process.call_args
            assert call_args[1].get("pin") is True or (len(call_args[0]) > 1 and call_args[0][1] is True)
