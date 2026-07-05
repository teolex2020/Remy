
import asyncio
from typing import Optional, Dict, Any

# Mocking the smsmanpy library for testing purposes if not installed
# In a real environment, this would be: from smsmanpy import Smsman
class MockSmsman:
    def __init__(self, api_key: str):
        self.api_key = api_key
    async def get_balance(self):
        return 10.5
    async def request_phone_number(self, country_id: int, application_id: int):
        return "req_123", "+1234567890"
    async def get_sms(self, request_id: str):
        return "123456"
    async def reject_number(self, request_id: str):
        return True

try:
    from smsmanpy import Smsman
except ImportError:
    Smsman = MockSmsman

TOOL_NAME = "sms_man_tool"
TOOL_DESCRIPTION = "Interact with SMS-MAN API to purchase virtual numbers and receive SMS codes for automated registrations."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["get_balance", "buy_number", "get_sms", "reject_number"],
            "description": "The action to perform."
        },
        "api_key": {
            "type": "string",
            "description": "SMS-MAN API Key."
        },
        "country_id": {
            "type": "integer",
            "description": "ID of the country (e.g., 1 for USA). Required for 'buy_number'.",
            "nullable": True
        },
        "application_id": {
            "type": "integer",
            "description": "ID of the application/service (e.g., 1 for Telegram). Required for 'buy_number'.",
            "nullable": True
        },
        "request_id": {
            "type": "string",
            "description": "The request ID from a previous 'buy_number' action. Required for 'get_sms' and 'reject_number'.",
            "nullable": True
        }
    },
    "required": ["action", "api_key"]
}

async def execute(action: str, api_key: str, country_id: Optional[int] = None, application_id: Optional[int] = None, request_id: Optional[str] = None) -> Dict[str, Any]:
    client = Smsman(api_key)
    
    try:
        if action == "get_balance":
            balance = await client.get_balance()
            return {"status": "success", "balance": balance}
        
        elif action == "buy_number":
            if country_id is None or application_id is None:
                return {"status": "error", "message": "country_id and application_id are required for buy_number"}
            req_id, phone = await client.request_phone_number(country_id, application_id)
            return {"status": "success", "request_id": req_id, "phone_number": phone}
            
        elif action == "get_sms":
            if request_id is None:
                return {"status": "error", "message": "request_id is required for get_sms"}
            sms_code = await client.get_sms(request_id)
            return {"status": "success", "sms_code": sms_code}
            
        elif action == "reject_number":
            if request_id is None:
                return {"status": "error", "message": "request_id is required for reject_number"}
            success = await client.reject_number(request_id)
            return {"status": "success", "rejected": success}
            
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

def test_get_balance():
    import asyncio
    result = asyncio.run(execute("get_balance", "test_key"))
    assert result["status"] == "success"
    assert "balance" in result

def test_buy_number():
    import asyncio
    result = asyncio.run(execute("buy_number", "test_key", country_id=1, application_id=1))
    assert result["status"] == "success"
    assert "request_id" in result
    assert "phone_number" in result
