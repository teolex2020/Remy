import sys
import subprocess

TOOL_NAME = "env_checker"
TOOL_DESCRIPTION = "Checks current Python environment, version and installed packages."
TOOL_PARAMETERS = {"properties": {}, "type": "object"}

def execute():
    try:
        py_ver = sys.version
        pkgs = subprocess.check_output([sys.executable, "-m", "pip", "list"]).decode()
        return f"Python: {py_ver}\nPackages:\n{pkgs}"
    except Exception as e:
        return str(e)

def test_env():
    res = execute()
    print("\nENVIRONMENT INFO:")
    print(res)
    assert "Python" in res
