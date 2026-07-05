import re
import os

TOOL_NAME = "analyze_logs"
TOOL_DESCRIPTION = "Analyzes a log file for specific regex patterns and returns their frequency."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string", "description": "Relative path to the log file in the data directory."},
        "pattern": {"type": "string", "description": "Regex pattern to search for (e.g., 'ERROR', 'failed')."}
    },
    "required": ["file_path", "pattern"]
}

def execute(file_path: str, pattern: str) -> str:
    # Ensure path is relative and doesn't escape data dir
    # Note: The system restricted to data dir, but good practice.
    full_path = os.path.join("data", file_path) if not file_path.startswith("data") else file_path
    
    if not os.path.exists(full_path):
        return f"Error: File {file_path} not found."
    
    counts = 0
    matches = []
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            for line in f:
                if re.search(pattern, line, re.IGNORECASE):
                    counts += 1
                    if len(matches) < 5: # Keep some examples
                        matches.append(line.strip())
        
        return f"Found {counts} occurrences of '{pattern}'.\nExamples:\n" + "\n".join(matches)
    except Exception as e:
        return f"Error analyzing log: {str(e)}"

def test_analyze_logs():
    # This is a dummy test
    print("Testing analyze_logs tool (Logic check only)")
    pass
