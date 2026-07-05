import subprocess
import sys

TOOL_NAME = "pkg_inspector"
TOOL_DESCRIPTION = "Inspects installed python packages and environment details."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Package name to search for (optional)"}
    }
}

def execute(query=None):
    try:
        # Get pip list
        result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
        packages = result.stdout
        
        if query:
            filtered = [line for line in packages.split('\n') if query.lower() in line.lower()]
            return "\n".join(filtered) if filtered else f"No packages found matching '{query}'"
        
        return packages
    except Exception as e:
        return f"Error: {str(e)}"

def test_pkg_inspector():
    # Simple test to check if it runs
    res = execute("pip")
    assert "pip" in res.lower()
