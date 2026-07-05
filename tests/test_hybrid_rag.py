"""Tests for structured hybrid memory injection."""

from unittest.mock import patch

from langchain_core.messages import HumanMessage, SystemMessage

from remy.core.agent import _inject_context


def _state(user_text: str) -> dict:
    return {
        "messages": [HumanMessage(content=user_text)],
        "session_id": "test-session",
        "channel": "desktop",
        "session_log": [],
    }


class TestHybridRAG:
    def test_hybrid_injects_both_sources(self):
        with patch("remy.core.agent_tools.brain") as mock_brain, patch(
            "remy.core.hybrid_search.search_exact_structured"
        ) as mock_exact, patch(
            "remy.core.hybrid_search.recall_cognitive_structured"
        ) as mock_cognitive:
            mock_exact.return_value = [
                {"id": "kb1", "content": "Carpathian mountains are in Ukraine"},
            ]
            mock_cognitive.return_value = [
                {"id": "cg1", "content": "User likes hiking in the Carpathians"},
            ]

            result = _inject_context(_state("Tell me about mountains"))

            assert result is not None
            assert isinstance(result, SystemMessage)
            assert "[Exact Memory]" in result.content
            assert "[Cognitive Recall]" in result.content
            assert "hiking" in result.content
            assert "Carpathian mountains" in result.content

    def test_hybrid_cognitive_only_when_no_exact(self):
        with patch("remy.core.agent_tools.brain") as mock_brain, patch(
            "remy.core.hybrid_search.search_exact_structured", return_value=[]
        ), patch("remy.core.hybrid_search.recall_cognitive_structured") as mock_cognitive:
            mock_cognitive.return_value = [
                {"id": "cg2", "content": "User prefers tea over coffee"},
            ]

            result = _inject_context(_state("What drinks do I like?"))

            assert result is not None
            assert "[Cognitive Recall]" in result.content
            assert "[Exact Memory]" not in result.content

    def test_hybrid_exact_only_when_no_cognitive(self):
        with patch("remy.core.agent_tools.brain") as mock_brain, patch(
            "remy.core.hybrid_search.search_exact_structured"
        ) as mock_exact, patch(
            "remy.core.hybrid_search.recall_cognitive_structured", return_value=[]
        ):
            mock_exact.return_value = [
                {"id": "kb2", "content": "Python is a programming language"},
            ]

            result = _inject_context(_state("What is Python?"))

            assert result is not None
            assert "[Cognitive Recall]" not in result.content
            assert "[Exact Memory]" in result.content

    def test_hybrid_token_budget_cap(self):
        with patch("remy.core.agent_tools.brain") as mock_brain, patch(
            "remy.core.hybrid_search.search_exact_structured"
        ) as mock_exact, patch(
            "remy.core.hybrid_search.recall_cognitive_structured"
        ) as mock_cognitive:
            mock_exact.return_value = [{"id": "kb3", "content": "A " * 3000}]
            mock_cognitive.return_value = [{"id": "cg3", "content": "B " * 3000}]

            result = _inject_context(_state("Tell me everything"))

            assert result is not None
            assert len(result.content) <= 4850

    def test_hybrid_skips_short_messages(self):
        with patch("remy.core.agent_tools.brain") as mock_brain:
            result = _inject_context(_state("hi"))

            assert result is None
            mock_brain.assert_not_called()

    def test_hybrid_skips_empty_messages(self):
        state = {"messages": [], "session_id": "s", "channel": "desktop", "session_log": []}
        result = _inject_context(state)
        assert result is None

    def test_hybrid_handles_exact_errors(self):
        with patch("remy.core.agent_tools.brain") as mock_brain, patch(
            "remy.core.hybrid_search.search_exact_structured",
            side_effect=RuntimeError("Exact recall failed"),
        ), patch("remy.core.hybrid_search.recall_cognitive_structured") as mock_cognitive:
            mock_cognitive.return_value = [{"id": "cg4", "content": "Some cognitive data"}]

            result = _inject_context(_state("Tell me about something"))

            assert result is not None
            assert "[Cognitive Recall]" in result.content
            assert "[Exact Memory]" not in result.content

    def test_hybrid_returns_none_when_both_empty(self):
        with patch("remy.core.agent_tools.brain") as mock_brain, patch(
            "remy.core.hybrid_search.search_exact_structured", return_value=[]
        ), patch(
            "remy.core.hybrid_search.recall_cognitive_structured", return_value=[]
        ):
            result = _inject_context(_state("random question here"))
            assert result is None

    def test_hybrid_handles_list_dict_format(self):
        with patch("remy.core.agent_tools.brain") as mock_brain, patch(
            "remy.core.hybrid_search.search_exact_structured"
        ) as mock_exact, patch(
            "remy.core.hybrid_search.recall_cognitive_structured", return_value=[]
        ):
            mock_exact.return_value = [{"id": "kb4", "content": "Dict format data"}]

            result = _inject_context(_state("What about dict data?"))

            assert result is not None
            assert "Dict format data" in result.content

    def test_hybrid_exact_recall_uses_structured_parameters(self):
        with patch("remy.core.agent_tools.brain") as mock_brain, patch(
            "remy.core.hybrid_search.search_exact_structured"
        ) as mock_exact, patch(
            "remy.core.hybrid_search.recall_cognitive_structured", return_value=[]
        ):
            mock_exact.return_value = [{"id": "kb5", "content": "some data"}]

            _inject_context(_state("Some longer question here"))

            mock_exact.assert_called_once_with(
                mock_brain,
                "Some longer question here",
                top_k=3,
                lexical_limit=6,
            )
