
import requests

TOOL_NAME = "smspool_tool"
TOOL_DESCRIPTION = "API client for SMSPool.net to purchase virtual numbers and receive SMS codes for automated registration."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "description": "The action to perform: 'purchase', 'check_status', 'active_orders', 'balance'",
            "enum": ["purchase", "check_status", "cancel", "active_orders", "balance"]
        },
        "api_key": {
            "type": "string",
            "description": "SMSPool API Key"
        },
        "country": {
            "type": "string",
            "description": "Country ID or short name (required for 'purchase')"
        },
        "service": {
            "type": "string",
            "description": "Service ID or name (required for 'purchase')"
        },
        "order_id": {
            "type": "string",
            "description": "Order ID (required for 'check_status')"
        }
    },
    "required": ["action", "api_key"]
}

def execute(action: str, api_key: str, country: str = None, service: str = None, order_id: str = None) -> dict:
    base_url = "https://api.smspool.net"
    
    try:
        if action == "purchase":
            if not country or not service:
                return {"error": "Country and service are required for purchase"}
            response = requests.post(f"{base_url}/purchase/sms", data={
                "key": api_key,
                "country": country,
                "service": service
            })
        elif action == "check_status":
            if not order_id:
                return {"error": "Order ID is required for check_status"}
            response = requests.post(f"{base_url}/sms/check", data={
                "key": api_key,
                "orderid": order_id
            })
        elif action == "active_orders":
            response = requests.post(f"{base_url}/request/active", data={"key": api_key})
        elif action == "balance":
            response = requests.get(f"{base_url}/request/balance", params={"key": api_key})
        else:
            return {"error": f"Invalid action: {action}"}
        
        return response.json()
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

def test_purchase(monkeypatch):
    class MockResponse:
        def json(self):
            return {"success": 1, "number": "12025550123", "order_id": "ABC"}
    
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: MockResponse())
    
    result = execute(action="purchase", api_key="test", country="US", service="tinder")
    assert result["success"] == 1
    assert result["number"] == "12025550123"

def test_balance(monkeypatch):
    class MockResponse:
        def json(self):
            return {"balance": "10.00"}
    
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: MockResponse())
    
    result = execute(action="balance", api_key="test")
    assert result["balance"] == "10.00"
