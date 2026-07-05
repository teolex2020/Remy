"""Tests for Working Memory Ring / Scratchpad (v2.3, Rec 14.3)."""

import json
import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def _isolate_scratchpad():
    from remy.core.scratchpad import clear_notes

    clear_notes()
    yield
    clear_notes()


# ============== Unit Tests: write_note ==============


class TestWriteNote:
    def test_write_returns_id(self):
        from remy.core.scratchpad import write_note

        result = write_note("Important finding about AI safety")
        assert "id" in result
        assert "content_preview" in result
        assert "AI safety" in result["content_preview"]

    def test_write_empty_rejected(self):
        from remy.core.scratchpad import write_note

        result = write_note("")
        assert "error" in result

    def test_write_whitespace_rejected(self):
        from remy.core.scratchpad import write_note

        result = write_note("   \n  ")
        assert "error" in result

    def test_write_stores_at_working_level(self):
        from remy.core.agent_tools import brain, brain_lock
        from remy.core.scratchpad import _SCRATCHPAD_TAG, write_note

        write_note("Test note for level check")
        with brain_lock:
            records = brain.search(tags=[_SCRATCHPAD_TAG], limit=5)
        assert len(records) >= 1


# ============== Unit Tests: read_notes ==============


class TestReadNotes:
    def test_empty_initially(self):
        from remy.core.scratchpad import read_notes

        notes = read_notes()
        assert notes == []

    def test_returns_written_notes(self):
        from remy.core.scratchpad import read_notes, write_note

        write_note("Note one")
        write_note("Note two")
        notes = read_notes()
        assert len(notes) == 2
        contents = " ".join(n["content"] for n in notes)
        assert "Note one" in contents
        assert "Note two" in contents

    def test_newest_first(self):
        import time

        from remy.core.scratchpad import read_notes, write_note

        write_note("First note")
        time.sleep(0.01)
        write_note("Second note")
        notes = read_notes()
        assert "Second note" in notes[0]["content"]
        assert "First note" in notes[1]["content"]

    def test_has_id_field(self):
        from remy.core.scratchpad import read_notes, write_note

        write_note("Note with ID")
        notes = read_notes()
        assert "id" in notes[0]


# ============== Unit Tests: clear_notes ==============


class TestClearNotes:
    def test_clear_empty(self):
        from remy.core.scratchpad import clear_notes

        count = clear_notes()
        assert count == 0

    def test_clear_deletes_all(self):
        from remy.core.scratchpad import clear_notes, read_notes, write_note

        write_note("Alpha note about cooking recipes")
        write_note("Beta note about science facts")
        write_note("Gamma note about history events")
        assert len(read_notes()) >= 3
        count = clear_notes()
        assert count >= 3
        assert len(read_notes()) == 0


# ============== Unit Tests: ring buffer eviction ==============


class TestRingBufferEviction:
    def test_evicts_oldest_when_full(self):
        from remy.core import scratchpad
        from remy.core.scratchpad import read_notes, write_note

        # Temporarily lower max
        original_max = scratchpad.MAX_SCRATCHPAD_NOTES
        scratchpad.MAX_SCRATCHPAD_NOTES = 5
        try:
            for i in range(7):
                write_note(f"Note {i}")
            notes = read_notes()
            assert len(notes) <= 5
            # Oldest (Note 0, Note 1) should be evicted
            contents = [n["content"] for n in notes]
            assert "Note 0" not in contents
        finally:
            scratchpad.MAX_SCRATCHPAD_NOTES = original_max


# ============== Unit Tests: get_scratchpad_context ==============


class TestGetScratchpadContext:
    def test_none_when_empty(self):
        from remy.core.scratchpad import get_scratchpad_context

        assert get_scratchpad_context() is None

    def test_includes_notes(self):
        from remy.core.scratchpad import get_scratchpad_context, write_note

        write_note("Key finding: transformers scale well")
        write_note("Paper: Attention Is All You Need")
        context = get_scratchpad_context()
        assert context is not None
        assert "SCRATCHPAD" in context
        assert "2 working notes" in context
        assert "transformers" in context
        assert "Attention" in context

    def test_note_count_in_header(self):
        from remy.core.scratchpad import get_scratchpad_context, write_note

        for i in range(4):
            write_note(f"Finding {i}")
        context = get_scratchpad_context()
        assert "working notes" in context

    def test_filter_only_targets_scratchpad_records(self):
        from remy.core.agent_tools import Level, brain, brain_lock
        from remy.core.scratchpad import (
            _SCRATCHPAD_TAG,
            filter_working_memory,
            get_scratchpad_metrics,
            write_note,
        )

        write_note("Kuzma note")
        with brain_lock:
            other = brain.store(
                content="Mission runtime trace",
                level=Level.WORKING,
                tags=["mission-trace"],
                metadata={"source": "agent-autonomous"},
            )

        with patch("remy.core.scratchpad.FILTER_MIN_SCORE", 0.99):
            result = filter_working_memory("unrelated query", session_id="s-test")
        assert result["filtered"] is True

        with brain_lock:
            untouched = brain.get(other.id)
            scratchpad_records = brain.search(tags=[_SCRATCHPAD_TAG], limit=20)

        assert untouched is not None
        assert "Mission runtime trace" in untouched.content
        assert any("Kuzma note" in rec.content for rec in scratchpad_records)
        metrics = get_scratchpad_metrics("s-test")
        assert metrics["scratchpad_working_total"] == 1
        assert metrics["scratchpad_working_bloat"] >= 0


# ============== Unit Tests: tool dispatch integration ==============


class TestScratchpadToolDispatch:
    def test_write_via_dispatch(self):
        from remy.core.tool_dispatch import execute_tool

        result = json.loads(
            execute_tool("scratchpad", {"action": "write", "content": "test note"}, session_id="s1")
        )
        assert "id" in result
        assert result["content_preview"] == "test note"

    def test_read_via_dispatch(self):
        from remy.core.tool_dispatch import execute_tool

        execute_tool("scratchpad", {"action": "write", "content": "note A"}, session_id="s1")
        result = json.loads(execute_tool("scratchpad", {"action": "read"}, session_id="s1"))
        assert result["count"] >= 1
        assert "note A" in result["notes"][0]["content"]

    def test_clear_via_dispatch(self):
        from remy.core.tool_dispatch import execute_tool

        execute_tool("scratchpad", {"action": "write", "content": "x"}, session_id="s1")
        result = json.loads(execute_tool("scratchpad", {"action": "clear"}, session_id="s1"))
        assert result["cleared"] is True
        assert result["deleted_count"] == 1

    def test_write_empty_error(self):
        from remy.core.tool_dispatch import execute_tool

        result = json.loads(
            execute_tool("scratchpad", {"action": "write", "content": ""}, session_id="s1")
        )
        assert "error" in result

    def test_default_action_is_read(self):
        from remy.core.tool_dispatch import execute_tool

        result = json.loads(execute_tool("scratchpad", {}, session_id="s1"))
        assert "notes" in result


# ============== Unit Tests: scratchpad in CORE_TOOL_NAMES ==============


class TestScratchpadInCore:
    def test_in_core_tools(self):
        from remy.core.tool_declarations import CORE_TOOL_NAMES

        assert "scratchpad" in CORE_TOOL_NAMES

    def test_declaration_exists(self):
        from remy.core.tool_declarations import BRAIN_TOOLS

        names = [t.name for t in BRAIN_TOOLS]
        assert "scratchpad" in names
