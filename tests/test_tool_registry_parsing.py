
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from google.genai import types

import logging
from remy.core.tool_registry import ToolRegistry

# Configure logging
logging.basicConfig(level=logging.ERROR)


@pytest.fixture
def mock_manifest_data():
    # Simulate the structure found in manifest.json causing the crash
    return {
        "name": "crash_test_tool",
        "description": "Tool that used to crash",
        "file": "test.py",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform"
                }
            },
            "required": ["action"]
        },
        "status": "approved"
    }

def test_registry_parsing_full_schema(mock_manifest_data):
    # Prepare mocks
    mock_manifest = MagicMock()
    mock_manifest.get_approved_tools.return_value = [mock_manifest_data]
    
    mock_manifest_cls = MagicMock(return_value=mock_manifest)

    # Clean sys.modules to force reload with mocks
    if "remy.core.tool_registry" in sys.modules:
        del sys.modules["remy.core.tool_registry"]

    # Patch modules
    with patch.dict(sys.modules, {
        "remy.sandbox.manifest": MagicMock(SandboxManifest=mock_manifest_cls),
        "remy.config.settings": MagicMock(),
        # Do NOT mock google.genai, we want real types
    }):
        from remy.core.tool_registry import ToolRegistry
        
        registry = ToolRegistry(core_tools=[])
        
        print(f"DEBUG: registry._manifest.get_approved_tools() = {registry._manifest.get_approved_tools()}")
        
        decls = registry.get_all_declarations()
        
        assert len(decls) == 1
        tool = decls[0]
        assert tool.name == "crash_test_tool"
        
        # Verify parameters were parsed correctly
        params = tool.parameters.properties
        assert "action" in params
        assert params["action"].type == "STRING"
