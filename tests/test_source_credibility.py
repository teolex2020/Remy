"""
Tests for Source Credibility (RM-2).
"""
import pytest
from unittest.mock import MagicMock, patch

from remy.core.source_credibility import SourceCredibility, DEFAULT_SCORE

@pytest.fixture
def scorer():
    return SourceCredibility()

def test_default_scores(scorer):
    """Test standard domain scores."""
    assert scorer.get_score("https://www.mayoclinic.org/diseases") == 0.90
    assert scorer.get_score("https://en.wikipedia.org/wiki/Health") == 0.75
    assert scorer.get_score("https://www.tiktok.com/@user/video") == 0.20
    assert scorer.get_score("https://unknown-site.com/article") == DEFAULT_SCORE

def test_subdomain_handling(scorer):
    """Test that subdomains inherit parent scores if not explicitly defined."""
    # mail.google.com -> should match google.com if google.com was in list?
    # actually google.com isn't in default list, scholar.google.com is.
    
    # nytimes.com is 0.80
    assert scorer.get_score("https://cooking.nytimes.com/recipe") == 0.80

def test_url_parsing(scorer):
    """Test robust URL parsing."""
    assert scorer.get_score("mayoclinic.org") == 0.90
    assert scorer.get_score("http://mayoclinic.org") == 0.90
    assert scorer.get_score("www.mayoclinic.org") == 0.90

def test_user_overrides(scorer):
    """Test that brain records can override scores."""
    # Mock brain records
    mock_record = MagicMock()
    mock_record.metadata = {"domain": "example.com", "score": 0.99}
    
    scorer.load_overrides([mock_record])
    
    # Before override, it would be default
    # After override, it should be 0.99
    assert scorer.get_score("https://example.com/foo") == 0.99

def test_override_precedence(scorer):
    """Override should beat default cache."""
    # Wikipedia is 0.75 default
    assert scorer.get_score("wikipedia.org") == 0.75
    
    mock_record = MagicMock()
    mock_record.metadata = {"domain": "wikipedia.org", "score": 0.10}
    
    scorer.load_overrides([mock_record])
    
    assert scorer.get_score("wikipedia.org") == 0.10
