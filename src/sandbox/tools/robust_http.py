import requests
import json

TOOL_NAME = "robust_http"
TOOL_DESCRIPTION = "A more robust HTTP client that correctly handles headers and JSON data for API interactions."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "method": {"type": "string", "description": "HTTP method (GET, POST, etc.)"},
        "url": {"type": "string", "description": "Target URL"},
        "headers_json": {"type": "string", "description": "HTTP headers as a JSON string"},
        "payload_json": {"type": "string", "description": "Request body as a JSON string"}
    },
    "required": ["method", "url"]
}

def execute(method, url, headers_json=None, payload_json=None):
    headers = json.loads(headers_json) if headers_json else {}
    data = json.loads(payload_json) if payload_json else None
    
    try:
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            json=data,
            timeout=15
        )
        return {
            "status_code": response.status_code,
            "content": response.text,
            "headers": dict(response.headers)
        }
    except Exception as e:
        return {"error": str(e)}

def test_robust_http():
    # This is a placeholder for local testing
    pass
