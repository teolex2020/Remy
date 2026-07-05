"""
Tests for Fact Extraction (RM-4).

Note: D-01 learning boundary tests are in test_d01_extract_facts_boundary.py.
These tests cover basic invocation and error handling only.
"""
import pytest
import json
from unittest.mock import MagicMock, patch

from remy.core.brain_tools import _extract_facts

_MOCK_RESPONSE = json.dumps([
    {"subject": "Vitamin C", "predicate": "supports", "object": "immune system", "context": "general health"},
    {"subject": "Oranges", "predicate": "contain", "object": "Vitamin C"},
])
_LLM_RETURN = MagicMock()
_LLM_RETURN.content = _MOCK_RESPONSE


@pytest.fixture
def mock_brain():
    with patch("remy.core.brain_tools.brain") as mock:
        yield mock


def test_extract_facts_success_no_fetch(mock_brain):
    """Without fetch evidence, extract_facts stores at WORKING level (D-01 boundary)."""
    from remy.core.agent_tools import Level

    with patch("remy.core.llm.call_llm", return_value=_LLM_RETURN):
        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=[]):
            args = {"text": "Oranges contain Vitamin C. Vitamin C supports the immune system."}
            result = _extract_facts(args, session_id="sess-test")

    # Stored 2 items but as unverified working notes
    assert mock_brain.store.call_count == 2
    assert "unverified" in result.lower() or "No fetch evidence" in result

    call1 = mock_brain.store.call_args_list[0][1]
    assert call1["level"] == Level.WORKING
    assert "extracted-fact" in call1["tags"]
    assert "quarantine-unverified" in call1["tags"]


def test_extract_facts_success_with_fetch(mock_brain):
    """With fetch evidence, extract_facts stores at DOMAIN level."""
    from remy.core.agent_tools import Level

    fetch_ev = [{"url": "https://example.com/nutrition", "title": "", "site": "", "tool": "extract_content"}]
    with patch("remy.core.llm.call_llm", return_value=_LLM_RETURN):
        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=fetch_ev):
            args = {"text": "Oranges contain Vitamin C. Vitamin C supports the immune system."}
            result = _extract_facts(args, session_id="sess-test")

    assert "Extracted and stored 2 facts" in result
    assert mock_brain.store.call_count == 2

    call1 = mock_brain.store.call_args_list[0][1]
    assert call1["level"] == Level.DOMAIN
    assert "extracted-fact" in call1["tags"]
    assert call1["metadata"]["structure"]["subject"] in ("Vitamin C", "Oranges")


def test_extract_facts_malformed_llm(mock_brain):
    """Test handling of bad LLM output."""
    bad_llm = MagicMock()
    bad_llm.content = "Not JSON"
    with patch("remy.core.llm.call_llm", return_value=bad_llm):
        result = _extract_facts({"text": "foo"})

    assert "Fact extraction failed" in result
    mock_brain.store.assert_not_called()
