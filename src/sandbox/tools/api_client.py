import requests
import json

TOOL_NAME = "api_client"
TOOL_DESCRIPTION = "A reliable HTTP client that handles JSON and headers correctly for API registrations."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "The URL to send the request to"},
        "method": {"type": "string", "description": "HTTP method (GET, POST, PUT, DELETE)"},
        "headers": {"type": "string", "description": "JSON string of headers"},
        "json_data": {"type": "string", "description": "JSON string of the body"}
    },
    "required": ["url", "method"]
}

def execute(url, method="POST", headers=None, json_data=None):
    try:
        header_dict = json.loads(headers) if headers else {}
        data_dict = json.loads(json_data) if json_data else None
        
        response = requests.request(
            method=method,
            url=url,
            headers=header_dict,
            json=data_dict,
            timeout=30
        )
        
        return {
            "status": response.status_code,
            "body": response.text,
            "url": url
        }
    except Exception as e:
        return {"error": str(e)}

def test_api_client():
    # Test with a mock bin or similar if needed, but for now we just verify logic
    pass
