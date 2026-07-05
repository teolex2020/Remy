import time
import json

TOOL_NAME = "diagnostic_suite"
TOOL_DESCRIPTION = "Replicates failure modes: JSON malformation and Latency timeouts."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "mode": {
            "type": "string",
            "enum": ["json_malformation", "latency_simulation"],
            "description": "The failure mode to replicate."
        },
        "delay": {
            "type": "integer",
            "description": "Delay in seconds for latency_simulation."
        }
    },
    "required": ["mode"]
}

def execute(mode, delay=5):
    if mode == "latency_simulation":
        time.sleep(delay)
        return f"Simulated latency of {delay} seconds."
    
    if mode == "json_malformation":
        # Return a string that looks like partial JSON to see how the orchestrator handles it
        return '{"result": "success", "meta": {"id": 123, "note": "incomplete"'
    
    return "Invalid mode"

def test_latency():
    start = time.time()
    res = execute("latency_simulation", 2)
    assert "2 seconds" in res
    assert time.time() - start >= 2

def test_json():
    res = execute("json_malformation")
    assert "incomplete" in res
    # Verify it is actually invalid JSON
    try:
        json.loads(res)
        assert False, "Should have been invalid JSON"
    except json.JSONDecodeError:
        pass
