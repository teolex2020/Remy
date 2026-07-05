import subprocess
import sys
import os

TOOL_NAME = "env_reporter"
TOOL_DESCRIPTION = "Reports environment info to a file."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "filename": {"type": "string", "description": "Target filename in data dir"}
    }
}

def execute(filename="env_report.txt"):
    try:
        # Get pip list
        result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
        packages = result.stdout
        
        # Get python version
        version = sys.version
        
        content = f"Python Version: {version}\n\nPackages:\n{packages}"
        
        # Write to file (relative to data dir)
        # Assuming we can write to the parent data directory
        filepath = os.path.join("..", filename)
        with open(filepath, "w") as f:
            f.write(content)
            
        return f"Report written to {filename}"
    except Exception as e:
        return f"Error: {str(e)}"

def test_env_reporter():
    res = execute("test_report.txt")
    assert "written" in res
