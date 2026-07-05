import os
import sys

TOOL_NAME = "stability_guard"
TOOL_DESCRIPTION = "Checks for context and environment stability. Uses internal metrics."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "context_length": {
            "type": "integer",
            "description": "Approximate length of current context in characters."
        }
    }
}

def execute(context_length=0):
    try:
        context_length = int(context_length)
    except (ValueError, TypeError):
        context_length = 0
    # Threshold for 'context flooding' based on logs (approx 35 mins or large context)
    # 35 mins of conversation is usually around 20k-30k tokens.
    # We will flag if context length > 15000 chars as a proxy.
    
    threshold = 15000
    status = "GO"
    if context_length > threshold:
        status = "WARNING"
    
    # Check if we are running in a restricted environment
    env_info = sys.platform
    
    return {
        "status": status,
        "context_info": {
            "length": context_length,
            "threshold": threshold,
            "platform": env_info
        },
        "recommendation": "Maintain structure" if status == "GO" else "Prune context or use consolidation"
    }

def test_stability_guard():
    result = execute(context_length=5000)
    assert result["status"] == "GO"
    result_fail = execute(context_length=20000)
    assert result_fail["status"] == "WARNING"
    print("Stability Guard Standard Test Passed")
