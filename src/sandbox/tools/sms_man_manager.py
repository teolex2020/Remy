import requests

TOOL_NAME = "sms_man_manager"
TOOL_DESCRIPTION = "Manage SMS verification via SMS-MAN API (2026). Allows checking balance, requesting a phone number, and retrieving the SMS code."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["get_balance", "get_number", "get_sms", "get_limits"],
            "description": "Action to perform."
        },
        "api_token": {
            "type": "string",
            "description": "SMS-MAN API token."
        },
        "country_id": {
            "type": "integer",
            "description": "ID of the country (e.g., 1 for USA). Required for get_number."
        },
        "application_id": {
            "type": "integer",
            "description": "ID of the service/app (e.g., 1 for Telegram). Required for get_number."
        },
        "request_id": {
            "type": "string",
            "description": "Request ID from get_number. Required for get_sms."
        }
    },
    "required": ["action", "api_token"]
}

def execute(action: str, api_token: str, country_id: int = None, application_id: int = None, request_id: str = None):
    base_url = "https://api.sms-man.com/control"
    
    try:
        if action == "get_balance":
            url = f"{base_url}/get-balance?token={api_token}"
            response = requests.get(url)
        elif action == "get_limits":
            url = f"{base_url}/get-limits?token={api_token}&country_id={country_id}&application_id={application_id}"
            response = requests.get(url)
        elif action == "get_number":
            if not country_id or not application_id:
                return {"error": "country_id and application_id are required for get_number"}
            url = f"{base_url}/get-number?token={api_token}&country_id={country_id}&application_id={application_id}"
            response = requests.get(url)
        elif action == "get_sms":
            if not request_id:
                return {"error": "request_id is required for get_sms"}
            url = f"{base_url}/get-sms?token={api_token}&request_id={request_id}"
            response = requests.get(url)
        else:
            return {"error": f"Unknown action: {action}"}
        
        return response.json()
    except Exception as e:
        return {"error": str(e)}
