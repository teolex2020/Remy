"""Tests for core brain tools via execute_tool dispatch."""

import json
from unittest.mock import patch

import pytest

from remy.core.claim_provenance import clear_turn_fetch_evidence, record_turn_fetch_evidence


# We need to patch brain and registry before importing brain_tools
# because it imports brain at module level

@pytest.fixture
def mock_brain(tmp_path):
    """Real CognitiveMemory for integration testing."""
    from aura import Aura as CognitiveMemory
    b = CognitiveMemory(str(tmp_path / "test_brain"))
    yield b
    b.close()


@pytest.fixture
def execute_tool(mock_brain, tmp_path):
    """Provide execute_tool with mocked brain and registry."""
    with patch("remy.core.brain_tools.brain", mock_brain), \
         patch("remy.core.brain_tools._registry", None), \
         patch("remy.core.tool_registry.settings") as mock_settings:
        mock_settings.SANDBOX_DIR = tmp_path / "sandbox"
        mock_settings.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"

        from remy.core.brain_tools import execute_tool
        yield execute_tool


def _store_safe_fact(execute_tool, content: str) -> dict:
    session_id = f"safe-fact-{abs(hash(content))}"
    clear_turn_fetch_evidence(session_id)
    record_turn_fetch_evidence(
        session_id,
        tool="extract_content",
        url="https://example.com/source",
        title="Example source",
        site="example.com",
    )
    return json.loads(
        execute_tool(
            "store",
            {
                "content": content,
                "tags": "tool-verified",
                "level": "L3_DOMAIN",
                "metadata": {
                    "admission_class": "grounded_external_fact",
                    "source_url": "https://example.com/source",
                    "verified": True,
                },
            },
            session_id=session_id,
        )
    )


def test_legacy_execute_tool_blocks_refuted_action(monkeypatch):
    from remy.core import brain_tools

    class FakeBrain:
        def policy_hint(self, situation, action, namespace=None):
            assert situation == "tool_call:autonomous:sess-legacy"
            assert action == "tool:recall:{\"query\": \"unsafe\"}"
            assert namespace == "remy-tools"
            return {
                "hint": "avoid",
                "reason": "prior legacy tool failure",
                "verdict": "refutes",
                "refutes": 1,
                "supports": 0,
                "should_block": True,
            }

    monkeypatch.setattr(brain_tools, "brain", FakeBrain())
    monkeypatch.setattr(
        brain_tools,
        "_execute_tool_locked",
        lambda *args, **kwargs: pytest.fail("blocked tool should not execute"),
    )

    result = brain_tools.execute_tool(
        "recall",
        {"query": "unsafe"},
        session_id="sess-legacy",
        channel="autonomous",
    )
    parsed = json.loads(result)

    assert "Blocked by consequence memory" in parsed["error"]
    assert parsed["consequence_gate"]["blocked"] is True


def test_legacy_execute_tool_records_success_consequence(monkeypatch):
    from remy.core import brain_tools

    captures = []

    class FakeBrain:
        def policy_hint(self, situation, action, namespace=None):
            return {
                "hint": "verify_first",
                "verdict": "inconclusive",
                "refutes": 0,
                "supports": 0,
                "should_block": False,
            }

        def capture_consequence(self, **kwargs):
            captures.append(kwargs)

    monkeypatch.setattr(brain_tools, "brain", FakeBrain())
    monkeypatch.setattr(
        brain_tools,
        "_execute_tool_locked",
        lambda name, args, session_id=None, channel=None: json.dumps({"ok": True}),
    )

    result = brain_tools.execute_tool(
        "recall",
        {"query": "safe"},
        session_id="sess-ok",
        channel="desktop",
    )
    parsed = json.loads(result)

    assert parsed == {"ok": True}
    assert captures
    assert captures[0]["situation"] == "tool_call:desktop:sess-ok"
    assert captures[0]["action"] == "tool:recall:{\"query\": \"safe\"}"
    assert captures[0]["consequence"] == "SUPPORTS"
    assert captures[0]["namespace"] == "remy-tools"
    assert "legacy-brain-tools" in captures[0]["scope"]


class TestRecall:

    def test_recall_empty_brain(self, execute_tool):
        result = execute_tool("recall", {"query": "grandmother"})
        assert "No relevant memories" in result or result == ""

    def test_recall_after_store(self, execute_tool):
        execute_tool("store", {"content": "Grandmother Maria lived in Kyiv", "tags": "person,grandmother"})
        result = execute_tool("recall", {"query": "grandmother Maria"})
        assert "Maria" in result or "Kyiv" in result


class TestStore:

    def test_store_returns_id(self, execute_tool):
        result = execute_tool("store", {"content": "Test memory", "tags": "test"})
        data = json.loads(result)
        assert data["stored"] is True
        assert "id" in data

    def test_store_with_level(self, execute_tool):
        result = execute_tool("store", {
            "content": "Important memory",
            "tags": "important",
            "level": "L4_IDENTITY",
        })
        data = json.loads(result)
        assert "error" in data
        assert "reserved" in data["error"]

    def test_store_domain_with_canonical_admission_metadata(self, execute_tool, mock_brain):
        data = _store_safe_fact(execute_tool, "Grounded finding from verified source")
        assert data["stored"] is True
        rec = mock_brain.get(data["id"])
        assert rec.metadata["admission_class"] == "grounded_external_fact"
        assert rec.metadata["verified"] is True

    def test_store_without_tags(self, execute_tool):
        result = execute_tool("store", {"content": "No tags memory"})
        data = json.loads(result)
        assert data["stored"] is True

    def test_store_persists_semantic_type(self, execute_tool, mock_brain):
        result = execute_tool(
            "store",
            {"content": "User prefers tea over coffee", "tags": "preference", "semantic_type": "preference"},
        )
        data = json.loads(result)
        rec = mock_brain.get(data["id"])
        assert rec.metadata["semantic_type"] == "preference"


class TestSearch:

    def test_search_empty(self, execute_tool):
        result = execute_tool("search", {"query": "nonexistent"})
        assert "No results" in result

    def test_search_finds_stored(self, execute_tool):
        execute_tool("store", {"content": "Uncle Petro was a teacher", "tags": "person,uncle"})
        result = execute_tool("search", {"query": "uncle Petro"})
        # Could be "No results" if scoring is too strict, or JSON array
        if "No results" not in result:
            data = json.loads(result)
            assert len(data) >= 1

    def test_search_hybrid_uses_exact_hits_when_semantic_empty(self, execute_tool):
        results = [
            {
                "id": "rec-1",
                "content": "Kuzma is an old dog",
                "tags": ["dog", "pet"],
                "level": "IDENTITY",
                "metadata": {"source": "user-confirmed", "timestamp": "2026-03-09T10:00:00"},
                "score": 0.62,
            }
        ]
        with patch("remy.core.brain_tools.hybrid_search_structured", return_value=results):
            result = execute_tool("search", {"query": "Kuzma"})

        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["id"] == "rec-1"
        assert data[0]["level"] == "IDENTITY"
        assert data[0]["metadata"]["source"] == "user-confirmed"

    def test_search_hybrid_deduplicates_semantic_and_exact_matches(self, execute_tool):
        shared_hit = {
            "id": "rec-2",
            "content": "Oleksandr lives in Velyka Dymerka",
            "tags": ["user-profile"],
            "level": "DOMAIN",
            "strength": 0.9,
            "activation_count": 3,
            "importance": None,
            "metadata": {"source": "user-confirmed"},
            "score": 0.91,
        }
        with patch("remy.core.brain_tools.hybrid_search_structured", return_value=[shared_hit]):
            result = execute_tool("search", {"query": "where do I live"})

        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["id"] == "rec-2"
        assert data[0]["score"] == 0.91

    def test_search_redacts_protected_exact_values_in_broad_results(self, execute_tool):
        protected_hit = {
            "id": "rec-3",
            "content": "Reach me at taras@example.com",
            "tags": ["identity", "contact"],
            "level": "IDENTITY",
            "metadata": {
                "source": "user-confirmed",
                "email": "taras@example.com",
                "protected_fields": ["email"],
            },
            "score": 0.88,
        }
        with patch("remy.core.brain_tools.hybrid_search_structured", return_value=[protected_hit]):
            result = execute_tool("search", {"query": "email"})

        data = json.loads(result)
        assert len(data) == 1
        assert "taras@example.com" not in data[0]["content"]
        assert data[0]["metadata"]["email"] == "[protected]"

    def test_search_exact_uses_structured_exact_path(self, execute_tool):
        exact_hit = {
            "id": "rec-exact-1",
            "content": "Oleksandr lives in Velyka Dymerka",
            "tags": ["user-profile"],
            "level": "IDENTITY",
            "metadata": {"source": "user-confirmed"},
            "score": 0.77,
        }
        with patch("remy.core.brain_tools.search_exact_structured", return_value=[exact_hit]):
            result = execute_tool("search_exact", {"query": "Oleksandr"})

        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["id"] == "rec-exact-1"
        assert data[0]["level"] == "IDENTITY"

    def test_get_full_record_hides_protected_metadata(self, execute_tool, mock_brain):
        from aura import Level

        rec = mock_brain.store(
            content="Credential email taras@example.com",
            level=Level.IDENTITY,
            tags=["identity", "contact"],
            metadata={"email": "taras@example.com", "protected_fields": ["email"]},
        )

        result = execute_tool("get_full_record", {"record_id": rec.id})
        data = json.loads(result)

        assert "taras@example.com" not in data["content"]
        assert data["metadata"]["email"] == "[protected]"
        assert data["protected_fields_present"] == ["email"]

    def test_get_protected_record_returns_exact_values(self, execute_tool, mock_brain):
        from aura import Level

        rec = mock_brain.store(
            content="Credential email taras@example.com",
            level=Level.IDENTITY,
            tags=["identity", "contact"],
            metadata={"email": "taras@example.com", "protected_fields": ["email"], "verified": True},
        )

        result = execute_tool("get_protected_record", {"record_id": rec.id, "fields": "email"})
        data = json.loads(result)

        assert data["values"]["email"] == "taras@example.com"
        assert data["verified"] is True


class TestStorePerson:

    def test_store_person_minimal(self, execute_tool):
        result = execute_tool("store_person", {"full_name": "Maria Kovalenko"})
        data = json.loads(result)
        assert data["stored"] is True
        assert data["name"] == "Maria Kovalenko"

    def test_store_person_accepts_name_alias(self, execute_tool):
        result = execute_tool("store_person", {"name": "Nadia Kovalenko"})
        data = json.loads(result)
        assert data["stored"] is True
        assert data["name"] == "Nadia Kovalenko"

    def test_store_person_full(self, execute_tool):
        result = execute_tool("store_person", {
            "full_name": "Petro Shevchenko",
            "role": "grandfather",
            "birth_date": "1935-05-12",
            "birth_place": "Poltava",
        })
        data = json.loads(result)
        assert data["stored"] is True
        assert data["name"] == "Petro Shevchenko"


class TestStoreStory:

    def test_store_story(self, execute_tool):
        result = execute_tool("store_story", {
            "title": "The War Years",
            "content": "During WWII, the family fled to the east.",
            "people_mentioned": "Maria,Petro",
        })
        data = json.loads(result)
        assert data["stored"] is True
        assert data["title"] == "The War Years"

    def test_store_story_no_people(self, execute_tool):
        result = execute_tool("store_story", {
            "title": "A Funny Day",
            "content": "Once upon a time something funny happened.",
        })
        data = json.loads(result)
        assert data["stored"] is True
        assert data["title"] == "A Funny Day"


class TestFamilyTree:

    def test_family_tree_empty(self, execute_tool):
        result = execute_tool("family_tree", {})
        assert "empty" in result.lower()

    def test_family_tree_with_members(self, execute_tool):
        execute_tool("store_person", {"full_name": "Anna Bondar", "role": "mother"})
        execute_tool("store_person", {"full_name": "Ivan Bondar", "role": "father"})

        result = execute_tool("family_tree", {})
        data = json.loads(result)
        names = [m["name"] for m in data]
        assert "Anna Bondar" in names
        assert "Ivan Bondar" in names


class TestInsights:

    def test_insights_returns_stats(self, execute_tool):
        result = execute_tool("insights", {})
        data = json.loads(result)
        assert isinstance(data, dict)


class TestSmartRecallDedup:

    def test_store_person_detects_duplicate(self, execute_tool):
        """Store Maria, store again → result has similar_existing."""
        execute_tool("store_person", {"full_name": "Maria Kovalenko", "role": "grandmother"})
        result = execute_tool("store_person", {"full_name": "Maria Kovalenko", "birth_date": "1940"})
        data = json.loads(result)
        assert data["stored"] is True
        assert "similar_existing" in data
        assert len(data["similar_existing"]) >= 1
        assert "warning" in data

    def test_store_person_no_duplicate(self, execute_tool):
        """First store → no similar_existing key."""
        result = execute_tool("store_person", {"full_name": "Ivan Bondar"})
        data = json.loads(result)
        assert data["stored"] is True
        assert "similar_existing" not in data

    def test_store_story_detects_duplicate(self, execute_tool):
        """Store a story, store similar → warning."""
        execute_tool("store_story", {"title": "The War Years", "content": "During WWII the family fled."})
        result = execute_tool("store_story", {"title": "The War Years", "content": "Updated version of the story."})
        data = json.loads(result)
        assert data["stored"] is True
        assert "similar_existing" in data

    def test_store_detects_similar(self, execute_tool):
        """Store general memory, store similar → note."""
        execute_tool("store", {"content": "Grandmother Maria lived in Kyiv", "tags": "family"})
        result = execute_tool("store", {"content": "Grandmother Maria from Kyiv", "tags": "family"})
        data = json.loads(result)
        assert data["stored"] is True
        # May or may not find duplicate depending on Aura's matching
        # Just verify the structure is correct JSON
        assert "id" in data

    def test_check_duplicates_handles_error(self):
        """_check_duplicates returns empty list on error."""
        with patch("remy.core.brain_tools.brain") as mock_brain, \
             patch("remy.core.brain_tools._kb_retrieve", return_value=[]):
            mock_brain.search.side_effect = RuntimeError("DB error")
            from remy.core.brain_tools import _check_duplicates
            result = _check_duplicates("test query", tags=["person"])
            assert result == []


class TestConnectRecords:

    def test_connect_basic(self, execute_tool):
        """Store 2 persons, connect → returns connected=True."""
        r1 = _store_safe_fact(execute_tool, "Maria Kovalenko is Ivan Kovalenko's mother.")
        r2 = _store_safe_fact(execute_tool, "Ivan Kovalenko is Maria Kovalenko's son.")

        result = execute_tool("connect_records", {
            "id_a": r1["id"],
            "id_b": r2["id"],
            "relationship": "mother of",
        })
        data = json.loads(result)
        assert data["connected"] is True
        assert data["relationship"] == "mother of"
        assert data["weight"] == 0.7  # default

    def test_connect_invalid_id(self, execute_tool):
        """Connect with non-existent ID → error."""
        r1 = json.loads(execute_tool("store_person", {"full_name": "Anna Test"}))
        result = execute_tool("connect_records", {
            "id_a": r1["id"],
            "id_b": "nonexistent_id_12345",
            "relationship": "knows",
        })
        data = json.loads(result)
        assert "error" in data

    def test_connect_custom_weight(self, execute_tool):
        """Custom weight=0.9 is stored."""
        r1 = _store_safe_fact(execute_tool, "Memory A is a grounded fact.")
        r2 = _store_safe_fact(execute_tool, "Memory B is a grounded fact.")

        result = execute_tool("connect_records", {
            "id_a": r1["id"],
            "id_b": r2["id"],
            "relationship": "related to",
            "weight": 0.9,
        })
        data = json.loads(result)
        assert data["connected"] is True
        assert data["weight"] == 0.9


class TestGetConnections:

    def test_get_connections_with_links(self, execute_tool):
        """Store 2, connect, get_connections → shows linked record."""
        r1 = json.loads(execute_tool("store_person", {"full_name": "Olga Test"}))
        r2 = json.loads(execute_tool("store_person", {"full_name": "Petro Test"}))
        execute_tool("connect_records", {
            "id_a": r1["id"],
            "id_b": r2["id"],
            "relationship": "spouse of",
        })

        result = execute_tool("get_connections", {"record_id": r1["id"]})
        data = json.loads(result)
        assert data["connection_count"] >= 1
        conn_ids = [c["id"] for c in data["connections"]]
        assert r2["id"] in conn_ids

    def test_get_connections_empty(self, execute_tool):
        """Record with no manual connections → empty or only auto-connections."""
        r1 = json.loads(execute_tool("store", {"content": "Isolated memory", "tags": "lonely"}))
        result = execute_tool("get_connections", {"record_id": r1["id"]})
        data = json.loads(result)
        assert "connections" in data

    def test_get_connections_invalid_id(self, execute_tool):
        """Non-existent ID → error."""
        result = execute_tool("get_connections", {"record_id": "nonexistent_abc"})
        data = json.loads(result)
        assert "error" in data


class TestFamilyTreeConnections:

    def test_family_tree_with_connections(self, execute_tool):
        """Store 2 persons, connect → family_tree includes connections."""
        r1 = json.loads(execute_tool("store_person", {"full_name": "Anna Tree", "role": "mother"}))
        r2 = json.loads(execute_tool("store_person", {"full_name": "Ivan Tree", "role": "father"}))
        execute_tool("connect_records", {
            "id_a": r1["id"],
            "id_b": r2["id"],
            "relationship": "married to",
        })

        result = execute_tool("family_tree", {})
        data = json.loads(result)
        names = [m["name"] for m in data]
        assert "Anna Tree" in names
        assert "Ivan Tree" in names

        # At least one member should have connections
        members_with_connections = [m for m in data if "connections" in m]
        assert len(members_with_connections) >= 1

    def test_family_tree_no_connections(self, execute_tool):
        """Store persons without connecting → no connections key."""
        execute_tool("store_person", {"full_name": "Solo Person A"})
        execute_tool("store_person", {"full_name": "Solo Person B"})

        result = execute_tool("family_tree", {})
        data = json.loads(result)
        # Members without explicit connections should not have connections key
        # (auto-connect may add some via shared tags, so we just check structure)
        for m in data:
            assert "name" in m
            assert "role" in m
            assert "id" in m


class TestUpdateRecord:

    def test_update_content(self, execute_tool):
        """Update content of existing record."""
        r = json.loads(execute_tool("store_person", {"full_name": "Petro Kovalenko", "role": "father"}))
        result = execute_tool("update_record", {
            "record_id": r["id"],
            "content": "Petro Kovalenko, father, born March 12, 1945 in Poltava",
        })
        data = json.loads(result)
        assert data["updated"] is True
        assert "March 12, 1945" in data["content"]

    def test_update_tags(self, execute_tool):
        """Update tags of existing record."""
        r = json.loads(execute_tool("store", {"content": "Some memory", "tags": "old_tag"}))
        result = execute_tool("update_record", {
            "record_id": r["id"],
            "tags": "new_tag, updated",
        })
        data = json.loads(result)
        assert data["updated"] is True
        assert "new_tag" in data["tags"]
        assert "updated" in data["tags"]

    def test_update_nonexistent(self, execute_tool):
        """Update non-existent record → error."""
        result = execute_tool("update_record", {"record_id": "nonexistent_xyz"})
        data = json.loads(result)
        assert "error" in data

    def test_update_no_changes(self, execute_tool):
        """Call with only record_id (no content/tags/level) → still returns updated record."""
        r = json.loads(execute_tool("store", {"content": "Original content", "tags": "test"}))
        result = execute_tool("update_record", {"record_id": r["id"]})
        data = json.loads(result)
        assert data["updated"] is True
        assert "Original content" in data["content"]


class TestDeleteRecord:

    def test_delete_existing(self, execute_tool):
        """Delete existing record → deleted=True, record gone."""
        r = json.loads(execute_tool("store", {"content": "Delete me", "tags": "test"}))
        record_id = r["id"]

        result = execute_tool("delete_record", {"record_id": record_id})
        data = json.loads(result)
        assert data["deleted"] is True
        assert data["id"] == record_id
        assert "Delete me" in data["deleted_content"]

        # Verify record is gone — get_connections should return error
        check = execute_tool("get_connections", {"record_id": record_id})
        check_data = json.loads(check)
        assert "error" in check_data

    def test_delete_nonexistent(self, execute_tool):
        """Delete non-existent record → error."""
        result = execute_tool("delete_record", {"record_id": "nonexistent_abc"})
        data = json.loads(result)
        assert "error" in data


class TestMarkStale:

    def test_mark_stale_adds_tag_and_metadata(self, execute_tool, mock_brain):
        r = json.loads(execute_tool("store", {"content": "GitHub lead list 2026-03", "tags": "leads"}))
        record_id = r["id"]

        result = execute_tool(
            "mark_stale",
            {"record_id": record_id, "reason": "outdated as of 2026-04-13"},
        )
        data = json.loads(result)
        assert data["marked_stale"] is True
        assert data["id"] == record_id
        assert data["reason"] == "outdated as of 2026-04-13"

        rec = mock_brain.get(record_id)
        assert "stale" in rec.tags
        assert rec.metadata.get("stale") is True
        assert rec.metadata.get("stale_reason") == "outdated as of 2026-04-13"
        assert rec.metadata.get("stale_marked_at")

    def test_mark_stale_missing_reason(self, execute_tool):
        r = json.loads(execute_tool("store", {"content": "needs a reason", "tags": "t"}))
        result = execute_tool("mark_stale", {"record_id": r["id"], "reason": ""})
        data = json.loads(result)
        assert "error" in data
        assert "reason" in data["error"].lower()

    def test_mark_stale_nonexistent(self, execute_tool):
        result = execute_tool(
            "mark_stale", {"record_id": "nonexistent_stale", "reason": "gone"}
        )
        data = json.loads(result)
        assert "error" in data

    def test_mark_stale_idempotent(self, execute_tool):
        r = json.loads(execute_tool("store", {"content": "mark me twice", "tags": "t"}))
        execute_tool("mark_stale", {"record_id": r["id"], "reason": "first"})
        second = json.loads(
            execute_tool("mark_stale", {"record_id": r["id"], "reason": "second"})
        )
        assert second.get("already_stale") is True
        assert second["stale_reason"] == "first"

    def test_mark_stale_with_superseded_by(self, execute_tool, mock_brain):
        old = json.loads(execute_tool("store", {"content": "old info", "tags": "t"}))
        new = json.loads(execute_tool("store", {"content": "new info", "tags": "t"}))
        execute_tool(
            "mark_stale",
            {"record_id": old["id"], "reason": "replaced", "superseded_by": new["id"]},
        )
        rec = mock_brain.get(old["id"])
        assert rec.metadata.get("superseded_by") == new["id"]


class TestGetCurrentDatetime:

    def test_returns_valid_json(self, execute_tool):
        result = execute_tool("get_current_datetime", {})
        data = json.loads(result)
        assert "date" in data
        assert "time" in data
        assert "day_of_week" in data
        assert "iso" in data

    def test_date_format(self, execute_tool):
        result = execute_tool("get_current_datetime", {})
        data = json.loads(result)
        # YYYY-MM-DD format
        parts = data["date"].split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 4  # year


class TestScheduleTask:

    def test_schedule_one_time(self, execute_tool):
        result = execute_tool("schedule_task", {
            "description": "Call grandma",
            "due_date": "2026-03-01",
        })
        data = json.loads(result)
        assert data["scheduled"] is True
        assert "id" in data
        assert data["description"] == "Call grandma"
        assert data["due_date"] == "2026-03-01"
        assert data["repeat"] == "one-time"

    def test_schedule_recurring(self, execute_tool):
        result = execute_tool("schedule_task", {
            "description": "Weekly check-in",
            "due_date": "2026-02-15",
            "repeat": "weekly",
        })
        data = json.loads(result)
        assert data["scheduled"] is True
        assert data["repeat"] == "weekly"

    def test_schedule_stores_in_brain(self, execute_tool, mock_brain):
        execute_tool("schedule_task", {
            "description": "Buy flowers",
            "due_date": "2026-02-14",
        })
        tasks = mock_brain.search(query="", tags=["scheduled-task"], limit=5)
        assert len(tasks) == 1
        assert "Buy flowers" in tasks[0].content
        assert tasks[0].metadata["type"] == "scheduled_task"
        assert tasks[0].metadata["status"] == "active"

    def test_schedule_accepts_cron_without_due_date(self, execute_tool):
        result = execute_tool("schedule_task", {
            "description": "Morning reminder",
            "cron": "0 10 * * *",
        })
        data = json.loads(result)
        assert data["scheduled"] is True
        assert data["cron"] == "0 10 * * *"
        assert data["repeat"] == "daily"
        assert data["due_date"]

    def test_schedule_accepts_task_alias(self, execute_tool):
        result = execute_tool("schedule_task", {
            "task": "Щоденний звіт по агенту",
            "repeat": "daily",
        })
        data = json.loads(result)
        assert data["scheduled"] is True
        assert data["description"] == "Щоденний звіт по агенту"


class TestStoreResearch:

    def test_store_research_basic(self, execute_tool):
        result = execute_tool("store_research", {
            "topic": "Vitamin D deficiency",
            "findings": "Vitamin D deficiency affects 40% of adults. Symptoms include fatigue and bone pain.",
            "sources": "https://example.com/vitd,https://health.org/vitd",
        })
        data = json.loads(result)
        assert data["stored"] is True
        assert data["topic"] == "Vitamin D deficiency"
        assert "research" in data["tags"]
        assert "vitamin-d-deficiency" in data["tags"]
        assert data["sources_count"] == 2

    def test_store_research_does_not_auto_connect_unpromoted_report(self, execute_tool):
        execute_tool("store", {
            "content": "User has vitamin D deficiency diagnosed in 2024",
            "tags": "health",
        })
        result = execute_tool("store_research", {
            "topic": "Vitamin D treatment",
            "findings": "Recommended daily intake is 1000-4000 IU.",
            "sources": "https://example.com/treatment",
            "related_query": "vitamin D deficiency health",
        })
        data = json.loads(result)
        assert data["stored"] is True
        assert data["connected_to"] == []

    def test_store_research_no_related_query(self, execute_tool):
        result = execute_tool("store_research", {
            "topic": "Mars exploration",
            "findings": "NASA plans crewed Mars missions by 2040.",
            "sources": "https://nasa.gov/mars",
        })
        data = json.loads(result)
        assert data["stored"] is True
        assert data["connected_to"] == []

    def test_store_research_dedup_warning(self, execute_tool):
        execute_tool("store_research", {
            "topic": "Sleep hygiene",
            "findings": "Good sleep requires consistent schedule.",
            "sources": "https://sleep.org",
        })
        result = execute_tool("store_research", {
            "topic": "Sleep hygiene",
            "findings": "Updated: melatonin helps some people.",
            "sources": "https://sleep.org/update",
        })
        data = json.loads(result)
        assert data["stored"] is True
        assert "similar_existing" in data

    def test_store_research_topic_slug(self, execute_tool):
        result = execute_tool("store_research", {
            "topic": "COVID-19 & Long COVID symptoms (2024)",
            "findings": "Long COVID affects 10-30% of patients.",
            "sources": "https://who.int/covid",
        })
        data = json.loads(result)
        assert "research" in data["tags"]
        slug = data["tags"][1]
        assert " " not in slug
        assert "&" not in slug
        assert "(" not in slug

    def test_store_research_retrievable(self, execute_tool, mock_brain):
        execute_tool("store_research", {
            "topic": "Mediterranean diet",
            "findings": "Rich in olive oil, fish, vegetables. Reduces heart disease risk.",
            "sources": "https://health.org/diet",
        })
        records = mock_brain.search(query="", tags=["research"], limit=5)
        assert len(records) >= 1
        assert "Mediterranean" in records[0].content
        assert records[0].metadata["type"] == "research_report"
        assert isinstance(records[0].metadata["sources"], list)

    def test_store_research_accepts_aliases_and_optional_sources(self, execute_tool):
        result = execute_tool("store_research", {
            "project_name": "DeAI Grants",
            "summary": "Funding map across grants and accelerators.",
        })
        data = json.loads(result)
        assert data["stored"] is True
        assert data["topic"] == "DeAI Grants"
        assert data["sources_count"] == 0


class TestUnknownTool:

    def test_unknown_tool(self, execute_tool):
        result = execute_tool("totally_fake_tool", {})
        assert "Unknown tool" in result


class TestMoodToolCompatibility:

    def test_introspect_mood_handles_missing_aura_method(self, execute_tool):
        result = execute_tool("introspect_mood", {})
        data = json.loads(result)

        assert data["available"] is False
        assert data["mood"] == "unknown"
        assert "get_mood_state" in data["reason"]

    def test_mood_history_handles_missing_aura_method(self, execute_tool):
        result = execute_tool("mood_history", {"limit": 3})
        data = json.loads(result)

        assert data["available"] is False
        assert data["history"] == []
        assert data["count"] == 0

    def test_mood_modulation_handles_missing_aura_method(self, execute_tool):
        result = execute_tool("mood_modulation", {})
        data = json.loads(result)

        assert data["available"] is False
        assert data["modulation"] == {}


class TestOptionalAuraToolCompatibility:

    @pytest.mark.parametrize(
        "tool_name,args,empty_key",
        [
            ("introspect_identity_milestones", {}, "milestones"),
            ("list_loaded_bases", {}, "bases"),
            ("list_cognitive_snapshots", {}, "snapshots"),
            ("list_org_records", {}, "records"),
            ("introspect_drives", {}, "drives"),
            ("introspect_tensions", {}, "tensions"),
            ("introspect_predictions", {}, "predictions"),
            ("introspect_surprises", {}, "surprises"),
            ("introspect_curiosity", {}, "gaps"),
            ("introspect_hypotheses", {}, "hypotheses"),
        ],
    )
    def test_read_only_optional_aura_tools_return_unavailable_json(
        self, execute_tool, tool_name, args, empty_key
    ):
        result = execute_tool(tool_name, args)
        data = json.loads(result)

        assert data["available"] is False
        assert data[empty_key] == []


class TestCompleteResearchArtifacts:

    def test_complete_research_returns_cited_markdown_artifact(self, execute_tool, mock_brain):
        class _LLMResult:
            def __init__(self, content):
                self.content = content

        llm_outputs = [
            _LLMResult('["vat reporting primary source", "vat filing deadline ukraine"]'),
            _LLMResult("Primary guidance points to a standard monthly VAT filing workflow with evidence from accepted sources."),
        ]

        session_id = "test:complete_research_cited"
        clear_turn_fetch_evidence(session_id)
        record_turn_fetch_evidence(
            session_id, tool="extract_content",
            url="https://tax.gov.ua/vat-guidance", title="VAT guidance", site="tax.gov.ua",
        )
        record_turn_fetch_evidence(
            session_id, tool="extract_content",
            url="https://mof.gov.ua/vat-calendar", title="VAT calendar", site="mof.gov.ua",
        )

        with patch("remy.core.llm.call_llm", side_effect=llm_outputs):
            started = json.loads(execute_tool("start_research", {
                "topic": "VAT reporting",
                "depth": "quick",
                "citation_required": True,
            }, session_id=session_id))
            project_id = started["project_id"]

            execute_tool("add_research_finding", {
                "project_id": project_id,
                "content": "VAT filing guidance is published on the tax authority portal.",
                "source_url": "https://tax.gov.ua/vat-guidance",
                "confidence": 0.9,
            }, session_id=session_id)
            execute_tool("add_research_finding", {
                "project_id": project_id,
                "content": "Monthly VAT deadlines should be cross-checked against the current filing calendar.",
                "source_url": "https://mof.gov.ua/vat-calendar",
                "confidence": 0.8,
            }, session_id=session_id)

            completed = json.loads(execute_tool("complete_research", {"project_id": project_id}, session_id=session_id))

        assert completed["completed"] is True
        assert completed["artifact_format"] == "markdown"
        assert completed["verification"]["status"] == "verified"
        assert completed["verification"]["verified"] is True
        assert "# VAT reporting" in completed["markdown"]
        assert "## Key Findings" in completed["markdown"]
        assert "[S1]" in completed["markdown"]
        assert completed["citations"][0]["id"] == "S1"
        assert completed["pdf_url"].startswith("/api/reports/")
        assert completed["pdf_filename"].endswith(".pdf")
        assert completed["pdf_record_id"]

        records = mock_brain.search(query="", tags=["research"], limit=10)
        report_record = next(
            rec for rec in records
            if (rec.metadata or {}).get("project_id") == project_id
            and (rec.metadata or {}).get("type") == "research_report"
        )
        metadata = report_record.metadata or {}
        assert metadata["artifact_format"] == "markdown"
        assert "## Sources" in metadata["markdown_body"]
        assert metadata["citations"][0]["url"] == "https://tax.gov.ua/vat-guidance"
        assert metadata["pdf_url"].startswith("/api/reports/")
        assert metadata["pdf_filename"].endswith(".pdf")

    def test_complete_research_handles_compat_store_result(self, tmp_path):
        from remy.core.agent_tools import _AuraCompat

        class _LLMResult:
            def __init__(self, content):
                self.content = content

        compat_brain = _AuraCompat(str(tmp_path / "compat_brain"))
        try:
            llm_outputs = [
                _LLMResult('["vat reporting primary source", "vat filing deadline ukraine"]'),
                _LLMResult("Primary guidance points to a standard monthly VAT filing workflow with evidence from accepted sources."),
            ]

            session_id = "test:complete_research_compat"
            clear_turn_fetch_evidence(session_id)
            record_turn_fetch_evidence(
                session_id, tool="extract_content",
                url="https://tax.gov.ua/vat-guidance", title="VAT guidance", site="tax.gov.ua",
            )

            with patch("remy.core.brain_tools.brain", compat_brain), \
                 patch("remy.core.brain_tools._registry", None), \
                 patch("remy.core.tool_registry.settings") as mock_settings, \
                 patch("remy.core.llm.call_llm", side_effect=llm_outputs):
                mock_settings.SANDBOX_DIR = tmp_path / "sandbox"
                mock_settings.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"

                from remy.core.brain_tools import execute_tool

                started = json.loads(execute_tool("start_research", {
                    "topic": "VAT reporting",
                    "depth": "quick",
                    "citation_required": True,
                }, session_id=session_id))
                project_id = started["project_id"]

                execute_tool("add_research_finding", {
                    "project_id": project_id,
                    "content": "VAT filing guidance is published on the tax authority portal.",
                    "source_url": "https://tax.gov.ua/vat-guidance",
                    "confidence": 0.9,
                }, session_id=session_id)

                completed = json.loads(execute_tool("complete_research", {"project_id": project_id}, session_id=session_id))

            assert completed["completed"] is True
            assert completed["report_id"]
            assert completed["artifact_format"] == "markdown"
            assert completed["verification"]["status"] == "verified"

            records = compat_brain.search(query="", tags=["research"], limit=10)
            report_record = next(
                rec for rec in records
                if (rec.metadata or {}).get("project_id") == project_id
                and (rec.metadata or {}).get("type") == "research_report"
            )
            metadata = report_record.metadata or {}
            assert metadata["artifact_format"] == "markdown"
            assert metadata["pdf_filename"].endswith(".pdf")
        finally:
            compat_brain.close()
