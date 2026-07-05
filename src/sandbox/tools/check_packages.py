
import subprocess
import sys

TOOL_NAME = "check_packages"
TOOL_DESCRIPTION = "Checks installed python packages in the sandbox environment."
TOOL_PARAMETERS = {}

def execute():
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return str(e)

def test_check_packages():
    res = execute()
    assert "pip" in res or "Package" in res
