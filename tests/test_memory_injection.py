"""
Tests for Proactive Memory Injection (RM-7).
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, SystemMessage

from remy.core.agent import _inject_context, AgentState

@pytest.fixture
def mock_brain():
    with patch("remy.core.agent_tools.brain") as mock:
        yield mock


def test_inject_context_success(mock_brain):
    """Test valid injection when relevant memory found."""
    state = {
        "messages": [HumanMessage(content="What did Maria like to do?")],
        "session_id": "test",
        "channel": "desktop",
        "session_log": []
    }

    with patch(
        "remy.core.hybrid_search.search_exact_structured",
        return_value=[{"id": "ex1", "content": "Grandmother Maria loved gardening."}],
    ), patch(
        "remy.core.hybrid_search.recall_cognitive_structured",
        return_value=[],
    ):
        msg = _inject_context(state)

    assert isinstance(msg, SystemMessage)
    assert "Relevant Memory Context" in msg.content
    assert "Grandmother Maria loved gardening" in msg.content
    assert "[Exact Memory]" in msg.content


def test_inject_context_no_memory(mock_brain):
    """Test no injection when nothing relevant found."""
    state = {
        "messages": [HumanMessage(content="Random noise here please"),],
        "session_id": "test",
        "channel": "desktop",
        "session_log": []
    }

    with patch("remy.core.hybrid_search.search_exact_structured", return_value=[]), patch(
        "remy.core.hybrid_search.recall_cognitive_structured", return_value=[]
    ):
        msg = _inject_context(state)
    assert msg is None


def test_inject_context_includes_exact_memory(mock_brain):
    """Test exact-memory results appear in injection."""
    state = {
        "messages": [HumanMessage(content="Where do I live?")],
        "session_id": "test",
        "channel": "desktop",
        "session_log": [],
    }

    with patch(
        "remy.core.hybrid_search.search_exact_structured",
        return_value=[
            {
                "id": "kb1",
                "content": "User Profile: Name: Oleksandr; Location: Velyka Dymerka",
            }
        ],
    ), patch(
        "remy.core.hybrid_search.recall_cognitive_structured",
        return_value=[],
    ):
        msg = _inject_context(state)

    assert isinstance(msg, SystemMessage)
    assert "[Exact Memory]" in msg.content
    assert "Velyka Dymerka" in msg.content


def test_inject_context_includes_cognitive_recall(mock_brain):
    state = {
        "messages": [HumanMessage(content="What drinks do I prefer lately?")],
        "session_id": "test",
        "channel": "desktop",
        "session_log": [],
    }

    with patch("remy.core.hybrid_search.search_exact_structured", return_value=[]), patch(
        "remy.core.hybrid_search.recall_cognitive_structured",
        return_value=[{"id": "c1", "content": "User tends to prefer tea over coffee."}],
    ):
        msg = _inject_context(state)

    assert isinstance(msg, SystemMessage)
    assert "[Cognitive Recall]" in msg.content
    assert "prefer tea" in msg.content

def test_inject_context_ignored_messages(mock_brain):
    """Test that short/system messages trigger no injection."""
    # System message last
    state = {"messages": [SystemMessage(content="sys")], "session_id": "t", "channel": "d", "session_log": []}
    assert _inject_context(state) is None

    # Short message
    state = {"messages": [HumanMessage(content="Hi")], "session_id": "t", "channel": "d", "session_log": []}
    assert _inject_context(state) is None
