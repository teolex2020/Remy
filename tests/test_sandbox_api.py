
import pytest
import sys
from unittest.mock import MagicMock, patch

@pytest.mark.asyncio
async def test_sandbox_api_endpoints():
    # Mock SandboxManifest
    mock_manifest_cls = MagicMock()
    mock_manifest = MagicMock()
    mock_manifest_cls.return_value = mock_manifest
    
    # Mock tool data
    mock_tool = {
        "name": "identity_manager",
        "description": "Manage identities",
        "status": "pending",
        "file": "identity_manager.py"
    }
    
    mock_manifest.summary.return_value = [mock_tool]
    mock_manifest.get_tool.return_value = mock_tool
    
    # Patch modules
    with patch.dict(sys.modules, {
        "remy.sandbox.manifest": MagicMock(SandboxManifest=mock_manifest_cls),
        "remy.config.settings": MagicMock(settings=MagicMock(SANDBOX_DIR=MagicMock())),
        "remy.core.agent_tools": MagicMock(),
    }):
        # Reload api to pick up mocks if needed, or just import
        if "remy.web.api" in sys.modules:
            del sys.modules["remy.web.api"]
            
        from remy.web.api import list_sandbox_tools, toggle_sandbox_tool
        
        # Test LIST
        tools_response = await list_sandbox_tools()
        assert len(tools_response["tools"]) == 1
        assert tools_response["tools"][0]["name"] == "identity_manager"
        
        # Test TOGGLE (pending -> approved)
        toggle_response = await toggle_sandbox_tool("identity_manager")
        assert toggle_response["status"] == "approved"
        assert toggle_response["name"] == "identity_manager"
        
        # Verify save was called
        mock_manifest.save.assert_called_once()
        assert mock_tool["status"] == "approved"
