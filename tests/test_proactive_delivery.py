"""
Tests for Proactive Research Delivery (RM-5).
"""
import pytest
import json
from unittest.mock import MagicMock, patch

from remy.core.autonomy import AutonomousLoop

@pytest.fixture
def mock_event_bus():
    # Patch where it is imported/used in autonomy.py
    with patch("remy.core.autonomy.event_bus") as mock:
        yield mock

def test_research_delivery_trigger(mock_event_bus):
    """Test that completion JSON triggers an event."""
    loop = AutonomousLoop()
    
    # Mock tool result from `complete_research`
    tool_result = json.dumps({
        "status": "research_complete",
        "topic": "Benefits of Sleep",
        "report": "Sleep is good."
    })
    
    loop._check_research_delivery(tool_result)
    
    # Check event published
    mock_event_bus.emit.assert_called_once()
    topic, data = mock_event_bus.emit.call_args[0]
    assert topic == "research.complete"
    assert data["topic"] == "Benefits of Sleep"
    assert "Sleep is good" in data["report"]
    assert "🔬" in data["message"]

def test_research_delivery_ignored(mock_event_bus):
    """Test that non-completion results are ignored."""
    loop = AutonomousLoop()
    
    # Normal tool result
    tool_result = "Added finding."
    
    loop._check_research_delivery(tool_result)
    
    mock_event_bus.emit.assert_not_called()
