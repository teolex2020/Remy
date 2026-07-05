"""
D-01 Learning Boundary Tests: extract_facts

Behavioral invariants:
  INV-1: No fetch evidence this turn в†’ store at Level.WORKING + quarantine-unverified tags.
          NEVER Level.DOMAIN from bare LLM extraction.
  INV-2: Fetch evidence present this turn в†’ store at Level.DOMAIN with source_url.
  INV-3: Unverified path returns a message explaining the downgrade (honest refusal).
  INV-4: Grounded path returns the normal "Extracted and stored N facts" message.
  INV-5: learning_channel and admission_class are always set in stored metadata.
"""
import json
import pytest
from unittest.mock import MagicMock, patch, call


# в”Ђв”Ђ helpers в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

_FACTS_RESPONSE = json.dumps([
    {"subject": "Vitamin C", "predicate": "supports", "object": "immune system", "context": "health"},
    {"subject": "Oranges", "predicate": "contain", "object": "Vitamin C"},
])

_LLM_RETURN = MagicMock()
_LLM_RETURN.content = _FACTS_RESPONSE


def _make_brain_mock():
    mock = MagicMock()
    mock.store.return_value = MagicMock(id="rec-test-1")
    return mock


# в”Ђв”Ђ fixtures в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

@pytest.fixture(autouse=True)
def patch_llm():
    """Patch call_llm at the source module so all lazy imports pick it up."""
    with patch("remy.core.llm.call_llm", return_value=_LLM_RETURN):
        yield


@pytest.fixture()
def brain_mock():
    """Patch brain in both call paths."""
    mock = _make_brain_mock()
    with patch("remy.core.tool_handlers.facts._get_brain", return_value=mock):
        with patch("remy.core.brain_tools.brain", mock):
            yield mock


# в”Ђв”Ђ INV-1: No fetch в†’ WORKING + quarantine в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

class TestNoFetchGrounding:
    """Without fetch evidence, extract_facts must NOT write DOMAIN."""

    def test_stores_at_working_level(self, brain_mock):
        """INV-1a: Level must be WORKING when no fetch evidence exists."""
        from remy.core.tool_handlers.facts import _extract_facts
        from remy.core.agent_tools import Level

        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=[]):
            _extract_facts({"text": "Oranges contain Vitamin C."}, channel="desktop", session_id="sess-nofetch")

        assert brain_mock.store.called
        for c in brain_mock.store.call_args_list:
            stored_level = c.kwargs.get("level") or (c.args[1] if len(c.args) > 1 else None)
            assert stored_level == Level.WORKING, (
                f"Expected Level.WORKING but got {stored_level!r}. "
                "LLM-only extraction must not reach DOMAIN."
            )

    def test_quarantine_tag_present(self, brain_mock):
        """INV-1b: quarantine-unverified tag must be present."""
        from remy.core.tool_handlers.facts import _extract_facts

        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=[]):
            _extract_facts({"text": "foo bar baz."}, channel="desktop", session_id="sess-nofetch")

        for c in brain_mock.store.call_args_list:
            tags = c.kwargs.get("tags") or []
            assert "quarantine-unverified" in tags, (
                f"quarantine-unverified missing from tags {tags!r}"
            )

    def test_unverified_extraction_tag_present(self, brain_mock):
        """INV-1c: unverified-extraction tag must be present."""
        from remy.core.tool_handlers.facts import _extract_facts

        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=[]):
            _extract_facts({"text": "foo."}, channel="desktop", session_id="sess-nofetch")

        for c in brain_mock.store.call_args_list:
            tags = c.kwargs.get("tags") or []
            assert "unverified-extraction" in tags

    def test_admission_class_is_unverified_claim(self, brain_mock):
        """INV-5a: admission_class must be 'unverified_claim' without grounding."""
        from remy.core.tool_handlers.facts import _extract_facts

        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=[]):
            _extract_facts({"text": "foo."}, channel="desktop", session_id="sess-nofetch")

        for c in brain_mock.store.call_args_list:
            meta = c.kwargs.get("metadata") or {}
            assert meta.get("admission_class") == "unverified_claim", (
                f"Expected admission_class=unverified_claim, got {meta.get('admission_class')!r}"
            )

    def test_learning_channel_is_unverified(self, brain_mock):
        """INV-5b: learning_channel must be 'unverified' without grounding."""
        from remy.core.tool_handlers.facts import _extract_facts

        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=[]):
            _extract_facts({"text": "foo."}, session_id="sess-nofetch")

        for c in brain_mock.store.call_args_list:
            meta = c.kwargs.get("metadata") or {}
            assert meta.get("learning_channel") == "unverified"

    def test_requires_grounding_flag_set(self, brain_mock):
        """INV-1d: requires_grounding=True must be set without fetch evidence."""
        from remy.core.tool_handlers.facts import _extract_facts

        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=[]):
            _extract_facts({"text": "foo."}, session_id="sess-nofetch")

        for c in brain_mock.store.call_args_list:
            meta = c.kwargs.get("metadata") or {}
            assert meta.get("requires_grounding") is True

    def test_no_domain_fact_tag(self, brain_mock):
        """INV-1e: 'fact' tag (DOMAIN marker) must NOT appear without grounding."""
        from remy.core.tool_handlers.facts import _extract_facts

        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=[]):
            _extract_facts({"text": "foo."}, session_id="sess-nofetch")

        for c in brain_mock.store.call_args_list:
            tags = c.kwargs.get("tags") or []
            assert "fact" not in tags, (
                f"'fact' tag must not appear without grounding; got {tags!r}"
            )


# в”Ђв”Ђ INV-3: Return message on unverified path в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

class TestUnverifiedReturnMessage:
    """Honest refusal message must inform caller about the downgrade."""

    def test_message_mentions_no_fetch_evidence(self, brain_mock):
        """INV-3: Return message must explain why DOMAIN was not written."""
        from remy.core.tool_handlers.facts import _extract_facts

        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=[]):
            result = _extract_facts({"text": "Oranges contain Vitamin C."}, session_id="sess-nofetch")

        assert "No fetch evidence" in result or "unverified" in result.lower(), (
            f"Expected downgrade explanation in result, got: {result!r}"
        )

    def test_message_mentions_extract_content(self, brain_mock):
        """INV-3: Message should suggest extract_content as the correct path."""
        from remy.core.tool_handlers.facts import _extract_facts

        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=[]):
            result = _extract_facts({"text": "Oranges contain Vitamin C."}, session_id="sess-nofetch")

        assert "extract_content" in result, (
            f"Expected mention of extract_content in result, got: {result!r}"
        )


# в”Ђв”Ђ INV-2: Fetch evidence present в†’ DOMAIN в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

class TestWithFetchGrounding:
    """With fetch evidence, extract_facts may write DOMAIN."""

    _FETCH_EVIDENCE = [
        {"url": "https://example.com/nutrition", "title": "Nutrition facts", "site": "example.com", "tool": "extract_content"}
    ]

    def test_stores_at_domain_level(self, brain_mock):
        """INV-2a: Level must be DOMAIN when fetch evidence exists."""
        from remy.core.tool_handlers.facts import _extract_facts
        from remy.core.agent_tools import Level

        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=self._FETCH_EVIDENCE), \
             patch("remy.core.ingestion.get_turn_fetch_evidence", return_value=self._FETCH_EVIDENCE):
            _extract_facts({"text": "Oranges contain Vitamin C."}, channel="desktop", session_id="sess-fetch")

        assert brain_mock.store.called
        for c in brain_mock.store.call_args_list:
            stored_level = c.kwargs.get("level") or (c.args[1] if len(c.args) > 1 else None)
            assert stored_level == Level.DOMAIN, (
                f"Expected Level.DOMAIN with fetch evidence, got {stored_level!r}"
            )

    def test_source_url_set_from_evidence(self, brain_mock):
        """INV-2b: source_url must be taken from fetch evidence."""
        from remy.core.tool_handlers.facts import _extract_facts

        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=self._FETCH_EVIDENCE), \
             patch("remy.core.ingestion.get_turn_fetch_evidence", return_value=self._FETCH_EVIDENCE):
            _extract_facts({"text": "Oranges contain Vitamin C."}, session_id="sess-fetch")

        for c in brain_mock.store.call_args_list:
            meta = c.kwargs.get("metadata") or {}
            assert meta.get("source_url") == "https://example.com/nutrition", (
                f"Expected source_url from fetch evidence, got {meta.get('source_url')!r}"
            )

    def test_admission_class_is_grounded_source_extract(self, brain_mock):
        """INV-5c: admission_class must be 'grounded_source_extract' with fetch evidence (Phase 2 canonical name)."""
        from remy.core.tool_handlers.facts import _extract_facts

        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=self._FETCH_EVIDENCE), \
             patch("remy.core.ingestion.get_turn_fetch_evidence", return_value=self._FETCH_EVIDENCE):
            _extract_facts({"text": "Oranges contain Vitamin C."}, session_id="sess-fetch")

        for c in brain_mock.store.call_args_list:
            meta = c.kwargs.get("metadata") or {}
            assert meta.get("admission_class") == "grounded_source_extract"

    def test_learning_channel_is_internet_evidence(self, brain_mock):
        """INV-5d: learning_channel must be 'internet_evidence' with fetch evidence."""
        from remy.core.tool_handlers.facts import _extract_facts

        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=self._FETCH_EVIDENCE), \
             patch("remy.core.ingestion.get_turn_fetch_evidence", return_value=self._FETCH_EVIDENCE):
            _extract_facts({"text": "Oranges contain Vitamin C."}, session_id="sess-fetch")

        for c in brain_mock.store.call_args_list:
            meta = c.kwargs.get("metadata") or {}
            assert meta.get("learning_channel") == "internet_evidence"

    def test_tool_verified_tag_present(self, brain_mock):
        """INV-2c: claim:tool-verified tag must be present."""
        from remy.core.tool_handlers.facts import _extract_facts

        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=self._FETCH_EVIDENCE), \
             patch("remy.core.ingestion.get_turn_fetch_evidence", return_value=self._FETCH_EVIDENCE):
            _extract_facts({"text": "Oranges contain Vitamin C."}, session_id="sess-fetch")

        for c in brain_mock.store.call_args_list:
            tags = c.kwargs.get("tags") or []
            assert "claim:tool-verified" in tags

    def test_no_quarantine_tag(self, brain_mock):
        """INV-2d: quarantine-unverified must NOT appear with fetch evidence."""
        from remy.core.tool_handlers.facts import _extract_facts

        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=self._FETCH_EVIDENCE), \
             patch("remy.core.ingestion.get_turn_fetch_evidence", return_value=self._FETCH_EVIDENCE):
            _extract_facts({"text": "Oranges contain Vitamin C."}, session_id="sess-fetch")

        for c in brain_mock.store.call_args_list:
            tags = c.kwargs.get("tags") or []
            assert "quarantine-unverified" not in tags


# в”Ђв”Ђ INV-4: Grounded path return message в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

class TestGroundedReturnMessage:
    def test_normal_success_message(self, brain_mock):
        """INV-4: Grounded path returns standard success message."""
        from remy.core.tool_handlers.facts import _extract_facts

        fetch_ev = [{"url": "https://example.com/x", "title": "", "site": "", "tool": "extract_content"}]
        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=fetch_ev), \
             patch("remy.core.ingestion.get_turn_fetch_evidence", return_value=fetch_ev):
            result = _extract_facts({"text": "Vitamin C supports immune system."}, session_id="sess-fetch")

        assert "Extracted and stored" in result
        assert "No fetch evidence" not in result


# в”Ђв”Ђ Brain_tools duplicate path в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

class TestBrainToolsDuplicatePath:
    """The _extract_facts in brain_tools.py must obey the same boundary."""

    @pytest.fixture()
    def brain_tools_brain_mock(self):
        mock = _make_brain_mock()
        with patch("remy.core.brain_tools.brain", mock):
            yield mock

    def test_no_fetch_stores_working(self, brain_tools_brain_mock):
        """brain_tools._extract_facts must also downgrade to WORKING without fetch."""
        from remy.core.brain_tools import _extract_facts as bt_extract_facts
        from remy.core.agent_tools import Level

        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=[]):
            bt_extract_facts({"text": "Some fact."}, channel="desktop", session_id="sess-bt-nofetch")

        for c in brain_tools_brain_mock.store.call_args_list:
            stored_level = c.kwargs.get("level") or (c.args[1] if len(c.args) > 1 else None)
            assert stored_level == Level.WORKING, (
                f"brain_tools path: expected WORKING without fetch, got {stored_level!r}"
            )

    def test_with_fetch_stores_domain(self, brain_tools_brain_mock):
        """brain_tools._extract_facts must store DOMAIN when fetch evidence exists."""
        from remy.core.brain_tools import _extract_facts as bt_extract_facts
        from remy.core.agent_tools import Level

        fetch_ev = [{"url": "https://example.com/z", "title": "", "site": "", "tool": "http_get"}]
        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=fetch_ev), \
             patch("remy.core.ingestion.get_turn_fetch_evidence", return_value=fetch_ev):
            bt_extract_facts({"text": "Some fact."}, channel="desktop", session_id="sess-bt-fetch")

        for c in brain_tools_brain_mock.store.call_args_list:
            stored_level = c.kwargs.get("level") or (c.args[1] if len(c.args) > 1 else None)
            assert stored_level == Level.DOMAIN, (
                f"brain_tools path: expected DOMAIN with fetch evidence, got {stored_level!r}"
            )

