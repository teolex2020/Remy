import requests
import json
import time

TOOL_NAME = "identity_manager"
TOOL_DESCRIPTION = "Manages digital identity by creating temporary emails and retrieving messages for registration purposes using mail.tm API."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["create_email", "get_messages", "get_message_content"],
            "description": "Action to perform"
        },
        "email_id": {
            "type": "string",
            "description": "ID of the email account (required for get_messages)"
        },
        "token": {
            "type": "string",
            "description": "Bearer token for the email account"
        },
        "message_id": {
            "type": "string",
            "description": "ID of the specific message to read"
        }
    },
    "required": ["action"]
}

BASE_URL = "https://api.mail.tm"

def execute(action, email_id=None, token=None, message_id=None):
    if action == "create_email":
        # 1. Get domain
        domain_resp = requests.get(f"{BASE_URL}/domains")
        domain = domain_resp.json()["hydra:member"][0]["domain"]
        
        # 2. Create account
        username = f"agent_{int(time.time())}"
        password = "secure_password_123"
        address = f"{username}@{domain}"
        
        create_resp = requests.post(f"{BASE_URL}/accounts", json={
            "address": address,
            "password": password
        })
        
        if create_resp.status_code != 201:
            return f"Error creating account: {create_resp.text}"
            
        # 3. Get token
        token_resp = requests.post(f"{BASE_URL}/token", json={
            "address": address,
            "password": password
        })
        token_data = token_resp.json()
        
        return {
            "address": address,
            "id": token_data["id"],
            "token": token_data["token"],
            "password": password
        }

    elif action == "get_messages":
        if not token: return "Error: token required"
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.get(f"{BASE_URL}/messages", headers=headers)
        if resp.status_code != 200:
            return f"Error getting messages: {resp.text}"
        try:
            return resp.json().get("hydra:member", [])
        except Exception as e:
            return f"Error parsing messages: {e}"

    elif action == "get_message_content":
        if not token or not message_id: return "Error: token and message_id required"
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.get(f"{BASE_URL}/messages/{message_id}", headers=headers)
        return resp.json()

    return "Invalid action"

def test_create_email():
    # Since this makes real network calls, we mock it or use a real test
    # For now, we assume network is available in sandbox
    result = execute("create_email")
    assert "address" in result
    assert "token" in result
    print("Test passed: Email created")
