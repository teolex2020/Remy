"""Test agent recursion limit handling."""

from unittest.mock import MagicMock, patch
import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from remy.core.agent import MAX_TOOL_ITERATIONS

def test_max_tool_iterations_handling(tmp_path):
    """Verify that hitting MAX_TOOL_ITERATIONS routes back to 'model' for wrap-up without KeyError."""
    
    # Mock settings and dependencies
    with patch("remy.core.agent.MAX_TOOL_ITERATIONS", 0), \
         patch("remy.core.agent._get_cached_system_instruction", return_value="System prompt"), \
         patch("remy.core.agent.get_all_tools", return_value=[]), \
         patch("remy.core.agent._sys_instruction_cache", {}), \
         patch("remy.core.agent._tool_call_count", {}), \
         patch("remy.core.agent._compiled_graphs", {}), \
         patch("remy.core.llm.call_llm") as mock_call_llm:

        # Mock LLM to return tool call then final wrap up
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"name": "test_tool", "args": {}, "id": "call_1"}]
        )
        final_msg = AIMessage(content="Final wrap up")

        # 1. First call: returns Tool Call
        # Agent sees tool call, checks limit (1 > 0), returns "model"
        # 2. Second call: returns Final Wrap Up
        mock_call_llm.side_effect = [tool_call_msg, final_msg]

        # Mock tool execution
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.invoke.return_value = "Tool result"
        
        with patch("remy.core.agent.get_all_tools", return_value=[mock_tool]):
            from remy.core.agent import build_agent_graph, AgentState
            
            # Rebuild graph to pick up the fix (and the mocked MAX_TOOL_ITERATIONS if it was used at build time, 
            # though here it's used at runtime in should_continue)
            graph = build_agent_graph("test_recursion")
            
            # Verify the edge exists in the compiled graph structure (if possible) or just run it
            
            messages = [SystemMessage(content="Start")]
            state = AgentState(
                messages=messages,
                session_id="test_session",
                channel="test_recursion",
                session_log=[]
            )
            
            # Run the graph
            # If the "model" edge is missing, this will raise KeyError: 'model'
            # Run the graph
            # If the "model" edge is missing, this will raise KeyError: 'model'
            try:
                # Increase recursion limit to be safe
                result = graph.invoke(state, {"recursion_limit": 20})
                
                # Verify we got the final message
                last_msg = result["messages"][-1]
                if not isinstance(last_msg, AIMessage):
                    pytest.fail(f"Expected AIMessage at end, got {type(last_msg).__name__} with content: {last_msg.content}")

                assert last_msg.content == "Final wrap up"
                
                # If we reached here without KeyError and got the final message, the fix is verified.
                
            except KeyError as e:
                pytest.fail(f"Graph raised KeyError: {e} - likely missing edge mapping")
