"""Tests for context deduplication in memory injection."""

from unittest.mock import patch


class TestExtractRecordIds:
    def test_extracts_ids(self):
        from remy.core.agent import _extract_record_ids

        text = "[id:abc123] Some content [id:def456] More content"
        ids = _extract_record_ids(text)
        assert ids == {"abc123", "def456"}

    def test_empty_text(self):
        from remy.core.agent import _extract_record_ids

        assert _extract_record_ids("") == set()

    def test_no_ids(self):
        from remy.core.agent import _extract_record_ids

        assert _extract_record_ids("Just plain text without any IDs") == set()

    def test_single_id(self):
        from remy.core.agent import _extract_record_ids

        ids = _extract_record_ids("[id:r_12345] record content here")
        assert ids == {"r_12345"}

    def test_uuid_style_ids(self):
        from remy.core.agent import _extract_record_ids

        ids = _extract_record_ids("[id:550e8400-e29b-41d4-a716-446655440000] content")
        assert ids == {"550e8400-e29b-41d4-a716-446655440000"}

    def test_mixed_content(self):
        from remy.core.agent import _extract_record_ids

        text = """Here is what you remember:
[id:rec1] [COG] [trust: 0.9 | 2d] User likes Python
[id:rec2] [CORE] [trust: 0.7 | 5d] User works at Acme
Some other text without IDs
[id:rec3] [COG] Another memory"""
        ids = _extract_record_ids(text)
        assert ids == {"rec1", "rec2", "rec3"}


def _make_hit(record_id: str, content: str, level: str = "DOMAIN") -> dict:
    return {
        "id": record_id,
        "content": f"[id:{record_id}] {content}",
        "level": level,
        "tags": [],
        "metadata": {},
    }


class TestInjectContextDedup:
    def test_removes_duplicate_lines(self):
        from langchain_core.messages import HumanMessage, SystemMessage
        from remy.core.agent import _inject_context

        sys_msg = SystemMessage(content="System prompt with [id:rec1] and [id:rec2] context")
        user_msg = HumanMessage(content="Tell me about my health data")

        state = {
            "messages": [sys_msg, user_msg],
            "session_id": "test",
            "channel": "desktop",
            "session_log": [],
            "enabled_tools": set(),
        }

        with patch("remy.core.agent_tools.brain"), patch(
            "remy.core.hybrid_search.search_exact_structured",
            return_value=[_make_hit("rec3", "User takes vitamin D")],
        ), patch("remy.core.hybrid_search.recall_cognitive_structured", return_value=[]):
            result = _inject_context(state)

        assert result is not None
        assert "vitamin D" in result.content

    def test_no_dedup_when_no_sys_ids(self):
        from langchain_core.messages import HumanMessage, SystemMessage
        from remy.core.agent import _inject_context

        sys_msg = SystemMessage(content="System prompt without any record references")
        user_msg = HumanMessage(content="Tell me about my health data")

        state = {
            "messages": [sys_msg, user_msg],
            "session_id": "test",
            "channel": "desktop",
            "session_log": [],
            "enabled_tools": set(),
        }

        with patch("remy.core.agent_tools.brain"), patch(
            "remy.core.hybrid_search.search_exact_structured",
            return_value=[_make_hit("rec4", "User blood pressure 120/80")],
        ), patch("remy.core.hybrid_search.recall_cognitive_structured", return_value=[]):
            result = _inject_context(state)

        assert result is not None
        assert "blood pressure" in result.content

    def test_all_duplicates_returns_none(self):
        from langchain_core.messages import HumanMessage, SystemMessage
        from remy.core.agent import _inject_context

        sys_msg = SystemMessage(content="[id:rec1] existing context")
        user_msg = HumanMessage(content="Tell me about my health data please")

        state = {
            "messages": [sys_msg, user_msg],
            "session_id": "test",
            "channel": "desktop",
            "session_log": [],
            "enabled_tools": set(),
        }

        with patch("remy.core.agent_tools.brain"), patch(
            "remy.core.hybrid_search.search_exact_structured", return_value=[]
        ), patch("remy.core.hybrid_search.recall_cognitive_structured", return_value=[]):
            result = _inject_context(state)

        assert result is None

    def test_preserves_lines_without_ids(self):
        from langchain_core.messages import HumanMessage, SystemMessage
        from remy.core.agent import _inject_context

        sys_msg = SystemMessage(content="[id:rec1] existing")
        user_msg = HumanMessage(content="Tell me about my health metrics")

        state = {
            "messages": [sys_msg, user_msg],
            "session_id": "test",
            "channel": "desktop",
            "session_log": [],
            "enabled_tools": set(),
        }

        with patch("remy.core.agent_tools.brain"), patch(
            "remy.core.hybrid_search.search_exact_structured",
            return_value=[{"id": "kb1", "content": "general context without ID"}],
        ), patch(
            "remy.core.hybrid_search.recall_cognitive_structured",
            return_value=[_make_hit("rec2", "New record about rec2 content")],
        ):
            result = _inject_context(state)

        assert result is not None
        assert "general context" in result.content
        assert "New record" in result.content
