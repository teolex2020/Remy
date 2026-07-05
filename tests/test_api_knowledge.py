"""
Tests for Knowledge Dashboard API (RM-8).
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

# Import the endpoints directly
from remy.web.api import (
    get_research_projects,
    get_metric_data,
    get_extracted_facts,
    get_brain_stats
)

@pytest.fixture
def mock_brain():
    with patch("remy.web.api.brain") as mock:
        yield mock

@pytest.fixture
def mock_get_active_projects():
    with patch("remy.web.api.get_active_research_projects") as mock:
        yield mock

@pytest.mark.asyncio
async def test_get_research_projects(mock_brain, mock_get_active_projects):
    """Test retrieving active and completed research."""
    # Mock active
    mock_get_active_projects.return_value = [{"project_id": "p1", "topic": "Active Topic"}]
    
    # Mock completed
    mock_rec = MagicMock()
    mock_rec.content = "Completed Research Project: Old Topic"
    mock_rec.metadata = {
        "project_id": "p2",
        "completed_at": "2023-01-01",
        "report_preview": "Summary..."
    }
    mock_brain.search.return_value = [mock_rec]
    
    result = await get_research_projects()
    
    assert len(result["active"]) == 1
    assert result["active"][0]["topic"] == "Active Topic"
    
    assert len(result["completed"]) == 1
    assert result["completed"][0]["topic"] == "Old Topic"
    assert result["completed"][0]["status"] == "completed"

@pytest.mark.asyncio
async def test_get_metric_data(mock_brain):
    """Test retrieving tracked metrics."""
    mock_rec = MagicMock()
    mock_rec.metadata = {
        "metric": "weight",
        "value": 70.0,
        "unit": "kg",
        "timestamp": "2023-01-01",
        "notes": "morning"
    }
    mock_brain.search.return_value = [mock_rec]
    
    result = await get_metric_data()
    
    assert len(result["data"]) == 1
    assert result["data"][0]["metric"] == "weight"
    assert result["data"][0]["value"] == 70.0

@pytest.mark.asyncio
async def test_get_extracted_facts(mock_brain):
    """Test retrieving extracted facts."""
    mock_rec = MagicMock()
    mock_rec.content = "Vitamin C supports immune system."
    mock_rec.metadata = {
        "structure": {"subject": "Vitamin C", "predicate": "supports", "object": "immune system"},
        "source": "manual",
        "extracted_at": "2023-01-01"
    }
    mock_brain.search.return_value = [mock_rec]
    
    result = await get_extracted_facts()
    
    assert len(result["data"]) == 1
    assert result["data"][0]["content"] == "Vitamin C supports immune system."
    assert result["data"][0]["structure"]["subject"] == "Vitamin C"

@pytest.mark.asyncio
async def test_get_brain_stats():
    """Test retrieving brain stats."""
    result = await get_brain_stats()
    assert result["status"] == "online"
    assert result["memory_backend"] == "AuraMemory"
