"""Tests for Selective Tool Schemas — core vs extended tool loading."""

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from remy.core.brain_tools import (
    BRAIN_TOOLS,
    CORE_TOOL_NAMES,
    EXTENDED_TOOL_NAMES,
    execute_tool,
)
from remy.core.langgraph_tools import get_all_tools, get_tools_by_names


# ============== FIXTURES ==============


@pytest.fixture
def mock_brain(tmp_path):
    from aura import Aura as CognitiveMemory
    b = CognitiveMemory(str(tmp_path / "test_brain"))
    yield b
    b.close()


@pytest.fixture(autouse=True)
def patch_brain_and_registry(mock_brain, tmp_path):
    """Patch brain and registry for all tests in this module."""
    with patch("remy.core.brain_tools.brain", mock_brain), \
         patch("remy.core.brain_tools._registry", None), \
         patch("remy.core.tool_registry.settings") as mock_settings, \
         patch("remy.core.langgraph_tools._cached_tools", None):
        mock_settings.SANDBOX_DIR = tmp_path / "sandbox"
        mock_settings.SANDBOX_TOOLS_DIR = tmp_path / "sandbox" / "tools"
        yield


# ============== TOOL CATEGORIES ==============


class TestToolCategories:

    def test_core_tools_contain_essentials(self):
        """CORE_TOOL_NAMES includes memory, utility, browser, profile, meta."""
        required = {
            "recall", "store", "search", "store_knowledge",
            "web_search", "get_current_datetime",
            "browse_page", "browser_act", "browser_close",
            "store_user_profile",
            "list_available_tools", "enable_tools",
        }
        assert required <= CORE_TOOL_NAMES  # All required tools must be in core

    def test_extended_tools_non_empty(self):
        """EXTENDED_TOOL_NAMES contains the remaining tools."""
        assert len(EXTENDED_TOOL_NAMES) > 0

    def test_no_overlap(self):
        """Core and extended sets are disjoint."""
        assert CORE_TOOL_NAMES & EXTENDED_TOOL_NAMES == set()

    def test_all_tools_covered(self):
        """Every BRAIN_TOOLS declaration is in either core or extended."""
        all_names = {t.name for t in BRAIN_TOOLS}
        categorized = CORE_TOOL_NAMES | EXTENDED_TOOL_NAMES
        orphans = all_names - categorized
        assert orphans == set(), f"Uncategorized tools: {orphans}"

    def test_no_phantom_core(self):
        """Every core tool name exists in BRAIN_TOOLS."""
        all_names = {t.name for t in BRAIN_TOOLS}
        phantoms = CORE_TOOL_NAMES - all_names
        assert phantoms == set(), f"Core names not in BRAIN_TOOLS: {phantoms}"

    def test_extended_derived_correctly(self):
        """EXTENDED is exactly BRAIN_TOOLS minus CORE."""
        all_names = {t.name for t in BRAIN_TOOLS}
        expected_extended = all_names - CORE_TOOL_NAMES
        assert EXTENDED_TOOL_NAMES == expected_extended


# ============== LIST_AVAILABLE_TOOLS ==============


class TestListAvailableTools:

    def test_returns_extended_only(self):
        """list_available_tools returns extended tool names and descriptions."""
        result = execute_tool("list_available_tools", {})
        data = json.loads(result)
        assert "available_tools" in data
        assert "count" in data
        names = {t["name"] for t in data["available_tools"]}
        # Should be extended tools only — no core tools
        assert names & CORE_TOOL_NAMES == set()
        assert names == EXTENDED_TOOL_NAMES

    def test_count_matches(self):
        """Count matches the number of returned tools."""
        result = execute_tool("list_available_tools", {})
        data = json.loads(result)
        assert data["count"] == len(data["available_tools"])

    def test_descriptions_present(self):
        """Each returned tool has a non-empty description."""
        result = execute_tool("list_available_tools", {})
        data = json.loads(result)
        for tool in data["available_tools"]:
            assert "description" in tool
            assert len(tool["description"]) > 0

    def test_descriptions_truncated(self):
        """Descriptions are truncated to 120 chars max."""
        result = execute_tool("list_available_tools", {})
        data = json.loads(result)
        for tool in data["available_tools"]:
            assert len(tool["description"]) <= 120


# ============== ENABLE_TOOLS ==============


class TestEnableTools:

    def test_enable_valid_tools(self):
        """enable_tools returns enabled list for valid extended tool names."""
        # Pick a known extended tool
        extended_name = next(iter(EXTENDED_TOOL_NAMES))
        result = execute_tool("enable_tools", {"tool_names": [extended_name]})
        data = json.loads(result)
        assert "enabled" in data
        assert extended_name in data["enabled"]

    def test_enable_empty_list(self):
        """enable_tools with empty list returns error."""
        result = execute_tool("enable_tools", {"tool_names": []})
        data = json.loads(result)
        assert "error" in data

    def test_enable_unknown_tool(self):
        """enable_tools reports unknown tool names."""
        result = execute_tool("enable_tools", {"tool_names": ["nonexistent_tool_xyz"]})
        data = json.loads(result)
        assert "unknown" in data
        assert "nonexistent_tool_xyz" in data["unknown"]

    def test_enable_core_tool_not_in_enabled(self):
        """Core tools are already loaded — they don't appear in 'enabled' list."""
        result = execute_tool("enable_tools", {"tool_names": ["recall"]})
        data = json.loads(result)
        # recall is core, not extended — should not be in enabled
        assert "recall" not in data.get("enabled", [])

    def test_enable_mixed_valid_invalid(self):
        """Mixed request: valid extended + unknown names."""
        extended_name = next(iter(EXTENDED_TOOL_NAMES))
        result = execute_tool("enable_tools", {
            "tool_names": [extended_name, "fake_tool_abc"]
        })
        data = json.loads(result)
        assert extended_name in data["enabled"]
        assert "fake_tool_abc" in data["unknown"]


# ============== GET_TOOLS_BY_NAMES ==============


class TestGetToolsByNames:

    def test_filters_correctly(self):
        """get_tools_by_names returns only tools with matching names."""
        tools = get_tools_by_names({"recall", "store"})
        names = {t.name for t in tools}
        assert names == {"recall", "store"}

    def test_empty_set_returns_empty(self):
        """Empty name set returns no tools."""
        tools = get_tools_by_names(set())
        assert tools == []

    def test_unknown_names_ignored(self):
        """Unknown names produce no results (no error)."""
        tools = get_tools_by_names({"nonexistent_tool_999"})
        assert tools == []

    def test_core_names_return_core_tools(self):
        """Core tool names return all core tools."""
        tools = get_tools_by_names(CORE_TOOL_NAMES)
        names = {t.name for t in tools}
        assert names == CORE_TOOL_NAMES


# ============== AGENT SELECTIVE LOADING ==============


class TestAgentSelectiveLoading:

    @pytest.fixture(autouse=True)
    def patch_agent_deps(self, tmp_path):
        with patch("remy.core.agent._compiled_graphs", {}), \
             patch("remy.core.agent._tool_call_count", {}):
            yield

    def test_call_model_autonomous_gets_all_tools(self):
        """Autonomous channel gets all tools bound."""
        from remy.core.agent import call_model

        state = {
            "messages": [HumanMessage(content="hello")],
            "channel": "autonomous",
            "session_id": "test-auto",
            "session_log": [],
            "enabled_tools": set(),
        }

        with patch("remy.core.llm.call_llm") as mock_llm, \
             patch("remy.core.agent._get_cached_system_instruction", return_value="sys"), \
             patch("remy.core.agent._inject_context", return_value=None), \
             patch("remy.core.agent._build_session_context", return_value=None):
            mock_llm.return_value = AIMessage(content="Hi")
            call_model(state)

            # Check tools passed to call_llm
            tools = mock_llm.call_args.kwargs.get("tools")
            assert tools is not None
            all_tools = get_all_tools()
            assert len(tools) == len(all_tools)

    def test_call_model_desktop_gets_core_only(self):
        """Desktop channel with no enabled_tools gets only core tools."""
        from remy.core.agent import call_model

        state = {
            "messages": [HumanMessage(content="hello")],
            "channel": "desktop",
            "session_id": "test-desktop",
            "session_log": [],
            "enabled_tools": set(),
        }

        with patch("remy.core.llm.call_llm") as mock_llm, \
             patch("remy.core.agent._get_cached_system_instruction", return_value="sys"), \
             patch("remy.core.agent._inject_context", return_value=None), \
             patch("remy.core.agent._build_session_context", return_value=None):
            mock_llm.return_value = AIMessage(content="Hi")
            call_model(state)

            tools = mock_llm.call_args.kwargs.get("tools")
            tool_names = {t.name for t in tools}
            assert tool_names == CORE_TOOL_NAMES

    def test_call_model_desktop_with_enabled_extends(self):
        """Desktop channel with enabled tools gets core + enabled."""
        from remy.core.agent import call_model

        extended_name = next(iter(EXTENDED_TOOL_NAMES))
        state = {
            "messages": [HumanMessage(content="hello")],
            "channel": "desktop",
            "session_id": "test-desktop-ext",
            "session_log": [],
            "enabled_tools": {extended_name},
        }

        with patch("remy.core.llm.call_llm") as mock_llm, \
             patch("remy.core.agent._get_cached_system_instruction", return_value="sys"), \
             patch("remy.core.agent._inject_context", return_value=None), \
             patch("remy.core.agent._build_session_context", return_value=None):
            mock_llm.return_value = AIMessage(content="Hi")
            call_model(state)

            tools = mock_llm.call_args.kwargs.get("tools")
            tool_names = {t.name for t in tools}
            assert extended_name in tool_names
            assert CORE_TOOL_NAMES.issubset(tool_names)

    def test_call_tools_tracks_enable(self):
        """call_tools detects enable_tools results and updates enabled_tools in state."""
        from remy.core.agent import call_tools

        extended_name = next(iter(EXTENDED_TOOL_NAMES))

        enable_result = json.dumps({"enabled": [extended_name]})
        ai_msg = AIMessage(
            content="",
            tool_calls=[{
                "id": "tc1",
                "name": "enable_tools",
                "args": {"tool_names": [extended_name]},
            }],
        )

        state = {
            "messages": [ai_msg],
            "channel": "desktop",
            "session_id": "test-enable",
            "session_log": [],
            "enabled_tools": set(),
        }

        with patch("remy.core.agent.get_all_tools") as mock_get:
            mock_tool = MagicMock()
            mock_tool.name = "enable_tools"
            mock_tool.invoke.return_value = enable_result
            mock_get.return_value = [mock_tool]

            result = call_tools(state)

        assert "enabled_tools" in result
        assert extended_name in result["enabled_tools"]

    def test_call_tools_no_enable_no_state_change(self):
        """call_tools without enable_tools does not set enabled_tools."""
        from remy.core.agent import call_tools

        ai_msg = AIMessage(
            content="",
            tool_calls=[{
                "id": "tc1",
                "name": "recall",
                "args": {"query": "test"},
            }],
        )

        state = {
            "messages": [ai_msg],
            "channel": "desktop",
            "session_id": "test-no-enable",
            "session_log": [],
            "enabled_tools": set(),
        }

        with patch("remy.core.agent.get_all_tools") as mock_get:
            mock_tool = MagicMock()
            mock_tool.name = "recall"
            mock_tool.invoke.return_value = "No memories found."
            mock_get.return_value = [mock_tool]

            result = call_tools(state)

        assert "enabled_tools" not in result

    def test_proactive_channel_gets_all_tools(self):
        """Proactive channel also gets all tools (same as autonomous)."""
        from remy.core.agent import call_model

        state = {
            "messages": [HumanMessage(content="hello")],
            "channel": "proactive",
            "session_id": "test-proactive",
            "session_log": [],
            "enabled_tools": set(),
        }

        with patch("remy.core.llm.call_llm") as mock_llm, \
             patch("remy.core.agent._get_cached_system_instruction", return_value="sys"), \
             patch("remy.core.agent._inject_context", return_value=None), \
             patch("remy.core.agent._build_session_context", return_value=None):
            mock_llm.return_value = AIMessage(content="Hi")
            call_model(state)

            tools = mock_llm.call_args.kwargs.get("tools")
            all_tools = get_all_tools()
            assert len(tools) == len(all_tools)
