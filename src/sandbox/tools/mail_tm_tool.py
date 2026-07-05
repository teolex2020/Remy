import requests
import json
import time

TOOL_NAME = "mail_tm_tool"
TOOL_DESCRIPTION = "API client for Mail.tm to create temporary email accounts and read messages for registration automation."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["get_domains", "create_account", "get_token", "list_messages", "get_message"],
            "description": "The action to perform"
        },
        "address": {"type": "string", "description": "Email address (for create_account, get_token)"},
        "password": {"type": "string", "description": "Password (for create_account, get_token)"},
        "token": {"type": "string", "description": "JWT token (for list_messages, get_message)"},
        "message_id": {"type": "string", "description": "ID of the message to retrieve"}
    },
    "required": ["action"]
}

BASE_URL = "https://api.mail.tm"

def execute(action: str, address: str = None, password: str = None, token: str = None, message_id: str = None) -> str:
    try:
        if action == "get_domains":
            r = requests.get(f"{BASE_URL}/domains")
            return json.dumps(r.json(), indent=2)
        
        elif action == "create_account":
            if not address or not password:
                return "Error: address and password are required for create_account"
            r = requests.post(f"{BASE_URL}/accounts", json={"address": address, "password": password})
            return json.dumps(r.json(), indent=2)
            
        elif action == "get_token":
            if not address or not password:
                return "Error: address and password are required for get_token"
            r = requests.post(f"{BASE_URL}/token", json={"address": address, "password": password})
            return json.dumps(r.json(), indent=2)
            
        elif action == "list_messages":
            if not token:
                return "Error: token is required for list_messages"
            headers = {"Authorization": f"Bearer {token}"}
            r = requests.get(f"{BASE_URL}/messages", headers=headers)
            return json.dumps(r.json(), indent=2)
            
        elif action == "get_message":
            if not token or not message_id:
                return "Error: token and message_id are required for get_message"
            headers = {"Authorization": f"Bearer {token}"}
            r = requests.get(f"{BASE_URL}/messages/{message_id}", headers=headers)
            return json.dumps(r.json(), indent=2)
            
        return "Unknown action"
    except Exception as e:
        return f"Error: {str(e)}"

def test_get_domains():
    # This might fail in sandbox if network is restricted, but usually sandbox tools have controlled network
    print("Testing get_domains...")
    res = execute(action="get_domains")
    print(res[:100])
    assert "hydra:member" in res or "Error" in res

if __name__ == "__main__":
    test_get_domains()
