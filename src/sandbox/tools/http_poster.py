import requests
from typing import Dict, Any

TOOL_NAME = "http_poster"
TOOL_DESCRIPTION = "Sends a POST request to a URL with JSON data. Use this for registration, submitting forms, or interacting with APIs."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "The URL to send the POST request to."},
        "data": {"type": "object", "description": "JSON data to include in the request body."},
        "headers": {"type": "object", "description": "Optional HTTP headers.", "nullable": True}
    },
    "required": ["url", "data"]
}

def execute(url: str, data: Dict[str, Any], headers: Dict[str, Any] = None) -> Dict[str, Any]:
    try:
        response = requests.post(url, json=data, headers=headers, timeout=15)
        return {
            "status": response.status_code,
            "body": response.text,
            "content_type": response.headers.get("Content-Type"),
            "url": response.url
        }
    except Exception as e:
        return {"error": str(e)}

def test_http_poster():
    # Test with a public bin
    res = execute("https://httpbin.org/post", {"test": "value"})
    assert res["status"] == 200
    assert '"test": "value"' in res["body"]
