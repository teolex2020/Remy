import requests
import json

TOOL_NAME = "openwork_client"
TOOL_DESCRIPTION = "Client for Openwork.bot API to register agents and manage tasks."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "description": "Action: 'register', 'get_onboarding', 'get_tasks', 'submit_job'"},
        "api_key": {"type": "string", "description": "Openwork API Key"},
        "payload": {"type": "string", "description": "JSON string for registration or submission"}
    },
    "required": ["action"]
}

def execute(action: str, api_key: str = None, payload: str = None):
    base_url = "https://www.openwork.bot/api"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        if action == "register":
            data = json.loads(payload)
            resp = requests.post(f"{base_url}/agents/register", json=data, headers=headers)
            return resp.json()
        elif action == "get_onboarding":
            resp = requests.get(f"{base_url}/onboarding", headers=headers)
            return resp.json()
        elif action == "get_tasks":
            resp = requests.get(f"{base_url}/agents/me/tasks", headers=headers)
            return resp.json()
        elif action == "submit_job":
            data = json.loads(payload)
            job_id = data.get("job_id")
            resp = requests.post(f"{base_url}/jobs/{job_id}/submit", json=data, headers=headers)
            return resp.json()
        else:
            return {"error": f"Unknown action: {action}"}
    except Exception as e:
        return {"error": str(e)}

def test_registration():
    # Mocking since we don't want to actually register in tests or we can if it's safe
    pass
