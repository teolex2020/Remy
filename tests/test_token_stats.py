
import pytest
import os
import json
from unittest.mock import MagicMock, patch
from remy.core.usage_stats import UsageTracker, STATS_FILE
from remy.config.settings import settings

def test_usage_tracker_persistence(tmp_path):
    # Mock settings.DATA_DIR
    with patch("remy.core.usage_stats.settings") as mock_settings:
        mock_settings.DATA_DIR = tmp_path
        
        tracker = UsageTracker()
        tracker.record_usage("user", 100)
        tracker.record_usage("autonomy", 50)
        
        # Verify memory
        stats = tracker.get_stats()
        assert stats["user_tokens"] == 100
        assert stats["autonomy_tokens"] == 50
        assert stats["lifetime"]["user_tokens_reported"] == 100
        assert stats["lifetime"]["autonomy_tokens_reported"] == 50
        assert stats["session"]["user_tokens"] == 100
        assert stats["session"]["autonomy_tokens"] == 50
        
        # Verify disk
        file_path = tmp_path / STATS_FILE
        assert file_path.exists()
        data = json.loads(file_path.read_text())
        assert data["user_tokens"] == 100
        assert data["autonomy_tokens"] == 50
        assert data["user_tokens_reported"] == 100
        assert data["autonomy_tokens_reported"] == 50


def test_usage_tracker_breaks_down_reported_vs_estimated(tmp_path):
    with patch("remy.core.usage_stats.settings") as mock_settings:
        mock_settings.DATA_DIR = tmp_path

        tracker = UsageTracker()
        tracker.record_usage("user", 120, kind="reported")
        tracker.record_usage("autonomy", 80, kind="reported")
        tracker.record_usage("autonomy", 40, kind="estimated")

        stats = tracker.get_stats()
        assert stats["lifetime"]["user_tokens"] == 120
        assert stats["lifetime"]["user_tokens_reported"] == 120
        assert stats["lifetime"]["autonomy_tokens"] == 120
        assert stats["lifetime"]["autonomy_tokens_reported"] == 80
        assert stats["lifetime"]["autonomy_tokens_estimated"] == 40
        assert stats["session"]["total_tokens"] == 240

@pytest.mark.asyncio
async def test_api_stats_include_usage():
    import sys
    from unittest.mock import MagicMock
    
    # Mock agent_tools BEFORE importing api
    mock_agent_tools = MagicMock()
    mock_brain = MagicMock()
    mock_agent_tools.brain = mock_brain
    mock_brain.count.return_value = 100
    mock_brain.stats.return_value = {"records": 100}
    
    with patch.dict(sys.modules, {"remy.core.agent_tools": mock_agent_tools}):
        # Now import api, it will use the mock
        if "remy.web.api" in sys.modules:
            del sys.modules["remy.web.api"]
        from remy.web.api import get_brain_stats
        
        # Mock usage_tracker.get_stats
        with patch("remy.core.usage_stats.usage_tracker.get_stats") as mock_get:
            mock_get.return_value = {
                "user_tokens": 1000,
                "autonomy_tokens": 500,
                "last_updated": 12345.0
            }
            
            response = await get_brain_stats()
            assert "usage" in response
            assert response["usage"]["user_tokens"] == 1000
            assert response["usage"]["autonomy_tokens"] == 500
            assert response["usage"]["total_tokens"] == 1500
