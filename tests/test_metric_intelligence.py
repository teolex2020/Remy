"""Tests for generic metric/event intelligence tools."""
import pytest
import json
from unittest.mock import MagicMock, patch

from remy.core.brain_tools import _event_correlate, _metric_summary, _track_metric

@pytest.fixture
def mock_brain():
    with patch("remy.core.brain_tools.brain") as mock:
        yield mock

@pytest.fixture
def mock_llm():
    with patch("remy.core.llm.call_llm") as mock:
        mock.return_value.content = "Mocked LLM Analysis"
        yield mock

def test_track_metric_storage(mock_brain):
    """Test that track_metric validly stores data."""
    # Setup mock search to return empty (no history)
    mock_brain.search.return_value = []
    
    args = {
        "metric_type": "weight",
        "value": 75.5,
        "unit": "kg",
        "notes": "Morning weigh-in"
    }
    
    result = _track_metric(args)
    
    assert "Recorded weight: 75.5 kg" in result
    
    # Verify store call
    mock_brain.store.assert_called_once()
    call_kwargs = mock_brain.store.call_args[1]
    assert call_kwargs["content"] == "Metric: weight = 75.5 kg (Morning weigh-in)"
    assert "metric" in call_kwargs["tags"]
    assert call_kwargs["metadata"]["metric"] == "weight"
    assert call_kwargs["metadata"]["value"] == 75.5

def test_metric_trend(mock_brain):
    """Test trend calculation when history exists."""
    # Mock previous record
    mock_prev = MagicMock()
    mock_prev.metadata = {"value": 76.0, "timestamp": "2023-01-01T00:00:00"}
    
    # Current call will create a new one, but search mocks what's in DB
    # We simulate that the new one ISN'T in search yet, or is top of list? 
    # The code searches for tags=[metric_type].
    # Let's mock search returning [current (if stored?), previous]
    # Code: history = brain.search(...)
    # If we assume consecutive calls, the new one isn't in search result yet unless we say so.
    # The code sorts by timestamp.
    
    # Let's populate search with just the OLD one to keep it simple, 
    # wait, the code says:
    # "if len(sorted_hist) > 1 ... prev = sorted_hist[1]"
    # So it expects at least 2 records to calculate trend.
    # This implies the current record MUST be in the search results for the logic to work as written?
    # Actually looking at the code:
    # `rec = brain.store(...)` -> stores it.
    # `history = brain.search(...)` -> fetches it + old ones.
    # So yes, we need to return at least 2 records.
    
    mock_curr = MagicMock()
    mock_curr.metadata = {"value": 75.0, "timestamp": "2023-01-02T00:00:00"}
    
    mock_brain.search.return_value = [mock_curr, mock_prev]
    
    args = {"metric_type": "weight", "value": 75.0, "unit": "kg"}
    result = _track_metric(args)
    
    # 75.0 vs 76.0 -> -1.0 change
    assert "change: -1.00" in result

def test_metric_summary_empty(mock_brain):
    mock_brain.search.return_value = []
    result = _metric_summary({"period": "week"})
    assert "No tracked metrics or events found" in result

def test_metric_summary_aggregation(mock_brain):
    # Mock metrics
    m1 = MagicMock()
    m1.metadata = {"metric": "weight", "value": 80.0}
    m2 = MagicMock()
    m2.metadata = {"metric": "weight", "value": 82.0}
    
    # Mock events
    s1 = MagicMock()
    s1.content = "Release shipped"
    
    # brain.search is called for current metrics, legacy metrics, events, and legacy events.
    # We need side_effect
    def side_effect(query, tags, limit):
        if "metric" in tags:
            return [m1, m2]
        if "event" in tags:
            return [s1]
        return []
        
    mock_brain.search.side_effect = side_effect
    
    result = _metric_summary({"period": "week"})
    
    assert "Metric Summary" in result
    assert "weight: 2 entries, avg 81.0" in result
    assert "Release shipped" in result

def test_event_correlate(mock_brain, mock_llm):
    mock_brain.recall.return_value = ["Memory: Release shipped.", "Memory: Feedback increased."]
    
    result = _event_correlate({"event": "feedback spike"})
    
    assert "Correlation analysis for 'feedback spike'" in result
    assert "Mocked LLM Analysis" in result
