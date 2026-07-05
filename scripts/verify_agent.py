import asyncio
import sys
from pathlib import Path

# Resolve repository root relative to this script so it still works after moves.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from langchain_core.messages import HumanMessage

from remy.core.agent import build_agent_graph


async def main():
    graph = build_agent_graph(channel="desktop")

    # 1. Test Greeting
    print("--- Test 1: Greeting ('Привіт') ---")
    state_greeting = {
        "messages": [HumanMessage(content="Привіт")],
        "session_id": "test-greeting-123",
        "channel": "desktop",
        "session_log": [],
        "enabled_tools": set(),
        "_cached_session_ctx": "",
        "_cached_scratchpad": "",
    }

    result_greeting = await graph.ainvoke(state_greeting)
    last_msg = result_greeting["messages"][-1]
    print(f"Content: {last_msg.content}")
    print(f"Tool Calls: {getattr(last_msg, 'tool_calls', [])}\n")

    # 2. Test Action Command
    print("--- Test 2: Action Command ('Скільки зірок в репозиторію AuraSDK?') ---")
    state_action = {
        "messages": [HumanMessage(content="Скільки зірок в репозиторію AuraSDK?")],
        "session_id": "test-action-123",
        "channel": "desktop",
        "session_log": [],
        "enabled_tools": set(),
        "_cached_session_ctx": "",
        "_cached_scratchpad": "",
    }

    result_action = await graph.ainvoke(state_action)
    last_msg_act = result_action["messages"][-1]
    print(f"Content: {last_msg_act.content[:100]}...")
    print(f"Tool Calls: {getattr(last_msg_act, 'tool_calls', [])}\n")


if __name__ == "__main__":
    asyncio.run(main())
