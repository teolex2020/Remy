"""Tests for LangGraph tool wrappers — ensures BRAIN_TOOLS are properly bridged."""

import json
from unittest.mock import patch

import pytest
from langchain_core.tools import StructuredTool


@pytest.fixture
def mock_brain(tmp_path):
    """Real CognitiveMemory for integration testing."""
    from aura import Aura as CognitiveMemory
    b = CognitiveMemory(str(tmp_path / "test_brain"))
    yield b
    b.close()


@pytest.fixture
def tools(mock_brain, tmp_path):
    """Build LangChain tools with mocked brain and registry."""
    with patch("remy.core.brain_tools.brain", mock_brain), \
         patch("remy.core.brain_tools._registry", None), \
         patch("remy.core.tool_registry.settings") as mock_settings:
        mock_settings.SANDBOX_DIR = tmp_path / "sandbox"
        mock_settings.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"

        from remy.core.langgraph_tools import build_langchain_tools
        yield build_langchain_tools()


class TestToolGeneration:

    def test_all_brain_tools_converted(self, tools):
        """Every BRAIN_TOOLS declaration should produce a StructuredTool."""
        from remy.core.brain_tools import BRAIN_TOOLS
        assert len(tools) == len(BRAIN_TOOLS)

    def test_tools_are_structured_tools(self, tools):
        for tool in tools:
            assert isinstance(tool, StructuredTool)

    def test_tool_names_match(self, tools):
        from remy.core.brain_tools import BRAIN_TOOLS
        expected_names = {d.name for d in BRAIN_TOOLS}
        actual_names = {t.name for t in tools}
        assert actual_names == expected_names

    def test_tool_descriptions_not_empty(self, tools):
        for tool in tools:
            assert tool.description, f"Tool '{tool.name}' has no description"

    def test_recall_tool_has_query_param(self, tools):
        recall = next(t for t in tools if t.name == "recall")
        schema = recall.args_schema.model_json_schema()
        assert "query" in schema.get("properties", {})

    def test_store_tool_has_content_param(self, tools):
        store = next(t for t in tools if t.name == "store")
        schema = store.args_schema.model_json_schema()
        assert "content" in schema.get("properties", {})


class TestToolExecution:

    def test_get_current_datetime(self, tools):
        dt_tool = next(t for t in tools if t.name == "get_current_datetime")
        result = dt_tool.invoke({})
        data = json.loads(result)
        assert "date" in data
        assert "time" in data

    def test_store_and_recall(self, tools):
        store = next(t for t in tools if t.name == "store")
        recall = next(t for t in tools if t.name == "recall")

        result = store.invoke({"content": "Test memory from LangGraph", "tags": "test"})
        data = json.loads(result)
        assert data["stored"] is True

        result = recall.invoke({"query": "memory from LangGraph"})
        assert "LangGraph" in result

    def test_insights_tool(self, tools):
        insights = next(t for t in tools if t.name == "insights")
        result = insights.invoke({})
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_unknown_tool_not_present(self, tools):
        names = {t.name for t in tools}
        assert "nonexistent_tool" not in names


class TestDeepToDict:
    """Ensure Pydantic models from nested schemas are converted to plain dicts."""

    def test_plain_values_pass_through(self):
        from remy.core.langgraph_tools import _deep_to_dict
        assert _deep_to_dict("hello") == "hello"
        assert _deep_to_dict(42) == 42
        assert _deep_to_dict(True) is True

    def test_pydantic_model_converted(self):
        from pydantic import BaseModel
        from remy.core.langgraph_tools import _deep_to_dict

        class Section(BaseModel):
            type: str
            content: str

        section = Section(type="intro", content="Hello world")
        result = _deep_to_dict(section)
        assert isinstance(result, dict)
        assert result == {"type": "intro", "content": "Hello world"}

    def test_list_of_pydantic_models_converted(self):
        from pydantic import BaseModel
        from remy.core.langgraph_tools import _deep_to_dict

        class Item(BaseModel):
            name: str
            value: int

        items = [Item(name="a", value=1), Item(name="b", value=2)]
        result = _deep_to_dict(items)
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)
        assert result[0] == {"name": "a", "value": 1}

    def test_nested_pydantic_in_dict(self):
        from pydantic import BaseModel
        from remy.core.langgraph_tools import _deep_to_dict

        class Inner(BaseModel):
            x: int

        data = {"key": Inner(x=5), "plain": "text"}
        result = _deep_to_dict(data)
        assert result == {"key": {"x": 5}, "plain": "text"}


class TestSessionId:

    def test_set_and_get_session_id(self):
        from remy.core.langgraph_tools import set_session_id, get_session_id
        set_session_id("test-123")
        assert get_session_id() == "test-123"
        set_session_id(None)
        assert get_session_id() is None

    def test_session_id_used_in_tools(self, tools):
        """Tools that use session_id (like recall) should pick it up from module state."""
        from remy.core.langgraph_tools import set_session_id
        set_session_id("session-abc")

        recall = next(t for t in tools if t.name == "recall")
        # Just verify it doesn't error — session_id flows through
        result = recall.invoke({"query": "test"})
        assert isinstance(result, str)

        set_session_id(None)
