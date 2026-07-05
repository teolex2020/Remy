import requests

TOOL_NAME = "http_request"
TOOL_DESCRIPTION = "Make HTTP requests (GET, POST, PUT, DELETE) to interact with external APIs for registration and task management."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"], "default": "GET"},
        "url": {"type": "string", "description": "The URL to send the request to"},
        "headers": {"type": "object", "description": "HTTP headers"},
        "json_data": {"type": "object", "description": "JSON payload for POST/PUT requests"},
        "params": {"type": "object", "description": "URL parameters for GET requests"}
    },
    "required": ["url"]
}

def execute(url, method="GET", headers=None, json_data=None, params=None):
    try:
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=json_data,
            params=params,
            timeout=10
        )
        return {
            "status_code": response.status_code,
            "content": response.text,
            "headers": dict(response.headers)
        }
    except Exception as e:
        return {"error": str(e)}

def test_http_request():
    # Test with a simple GET
    result = execute("https://httpbin.org/get")
    assert result["status_code"] == 200
    
    # Test with a POST
    result = execute("https://httpbin.org/post", method="POST", json_data={"test": "data"})
    assert result["status_code"] == 200
