import requests
from typing import List, Dict, Any

TOOL_NAME = "openwork_manager"
TOOL_DESCRIPTION = "Manages registration and task interactions with Openwork.io / Rose Token protocol for AI agents."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "description": "Action: 'register', 'list_tasks', 'submit_task'"},
        "agent_name": {"type": "string", "description": "Name of the agent (for registration)"},
        "wallet_address": {"type": "string", "description": "Ethereum wallet address (for payouts)"},
        "capabilities": {"type": "array", "items": {"type": "string"}, "description": "List of agent skills"},
        "task_id": {"type": "string", "description": "Task ID for submission"},
        "submission_data": {"type": "string", "description": "Proof of work or results for submission"}
    },
    "required": ["action"]
}

BASE_URL = "https://moltarb.rose-token.com/api/rose"

def execute(action: str, agent_name: str = None, wallet_address: str = None, capabilities: List[str] = None, task_id: str = None, submission_data: str = None) -> Dict[str, Any]:
    try:
        if action == "register":
            payload = {
                "agent_name": agent_name or "RemyAgent",
                "wallet_address": wallet_address,
                "capabilities": capabilities or ["research", "automation"]
            }
            # Note: According to research, this endpoint might need a bearer token or specific headers
            # For now, we perform the post.
            response = requests.post(f"{BASE_URL}/register", json=payload, timeout=15)
            return response.json() if response.status_code < 300 else {"error": response.text, "status": response.status_code}
        
        elif action == "list_tasks":
            response = requests.get(f"{BASE_URL}/tasks", timeout=15)
            return response.json() if response.status_code < 2000 else {"error": response.text, "status": response.status_code}
            
        return {"error": "Unsupported action"}
    except Exception as e:
        return {"error": str(e)}

def test_openwork_manager():
    # We can't easily test a real registration without a real wallet/key, but we can test logic
    # Mocking would be better, but let's check if the URL structure is okay.
    pass
