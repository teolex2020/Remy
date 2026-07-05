"""
D-03 Store Memory Boundary Tests: store() chain / evidence transit

Behavioral invariants:
  INV-1: store(L3_DOMAIN) without fetch evidence this turn → downgraded to WORKING.
  INV-2: store(L3_DOMAIN) with fetch evidence this turn → stored at DOMAIN.
  INV-3: tag forgery: tool-verified tag without fetch evidence → treated as llm_unverified.
  INV-4: inference tag + llm_unverified provenance → quarantined (not unconditionally allowed).
  INV-5: user-direct channel + user attribution tags → DOMAIN allowed without fetch.
  INV-6: explicit admission_class in metadata → DOMAIN allowed (vetted internal path).
  INV-7: downgraded records get 'requires-grounding' tag.
  INV-8: downgraded records get 'downgraded-no-transit' tag.
"""
import json
import threading
from unittest.mock import MagicMock, patch


# ── shared helpers ────────────────────────────────────────────────────────────

_FETCH_EV = [{"url": "https://example.com/study", "title": "Study",
              "site": "example.com", "tool": "extract_content"}]

_REAL_LOCK = threading.Lock()


def _make_brain_mock():
    stored_rec = MagicMock()
    stored_rec.id = "rec-d03"
    mock = MagicMock()
    mock.store.return_value = stored_rec
    mock.search.return_value = []
    mock.update.return_value = None
    return mock


def _call_store(brain_mock, args: dict, session_id: str = "sess-d03",
                channel: str = "desktop_worker"):
    """Invoke _execute_tool_inner('store', …) with patched brain and brain_lock."""
    from remy.core.brain_tools import _execute_tool_inner

    with patch("remy.core.brain_tools.brain", brain_mock), \
         patch("remy.core.brain_tools.brain_lock", _REAL_LOCK), \
         patch("remy.core.tool_utils._check_duplicates", return_value=[]):
        return json.loads(_execute_tool_inner(
            "store", args, session_id=session_id, channel=channel
        ))


def _level_from_call(call):
    """Extract the level kwarg from a brain.store mock call."""
    return call.kwargs.get("level") or (call.args[1] if len(call.args) > 1 else None)


def _tags_from_call(call):
    """Extract the tags kwarg from a brain.store mock call."""
    return call.kwargs.get("tags") or (call.args[2] if len(call.args) > 2 else [])


# ── INV-1: No fetch evidence → DOMAIN downgraded to WORKING ──────────────────

class TestDomainDowngradeNoTransit:
    def test_no_fetch_domain_not_stored_at_domain(self):
        """INV-1: store(L3_DOMAIN) without fetch → NOT stored at DOMAIN."""
        brain_mock = _make_brain_mock()
        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=[]):
            _call_store(brain_mock, {
                "content": "Vitamin C boosts immunity",
                "level": "L3_DOMAIN",
                "tags": "health,fact",
            })
        from remy.core.agent_tools import Level
        level = _level_from_call(brain_mock.store.call_args)
        assert level != Level.DOMAIN, f"Expected not DOMAIN, got {level}"

    def test_no_fetch_domain_stored_at_working(self):
        """INV-1: Downgraded level must be WORKING."""
        brain_mock = _make_brain_mock()
        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=[]):
            _call_store(brain_mock, {
                "content": "Some claimed fact",
                "level": "L3_DOMAIN",
                "tags": "fact",
            })
        from remy.core.agent_tools import Level
        level = _level_from_call(brain_mock.store.call_args)
        assert level == Level.WORKING

    def test_no_fetch_still_stores_something(self):
        """INV-1: Downgraded store must still succeed (WORKING is valid)."""
        brain_mock = _make_brain_mock()
        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=[]):
            _call_store(brain_mock, {"content": "Some fact", "level": "L3_DOMAIN", "tags": "fact"})
        brain_mock.store.assert_called_once()


# ── INV-2: With fetch evidence → DOMAIN allowed ───────────────────────────────

class TestDomainAllowedWithTransit:
    def test_with_fetch_stored_at_domain(self):
        """INV-2: store(L3_DOMAIN) with fetch evidence → stored at DOMAIN."""
        brain_mock = _make_brain_mock()
        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=_FETCH_EV):
            _call_store(brain_mock, {
                "content": "Vitamin C boosts immunity",
                "level": "L3_DOMAIN",
                "tags": "health,fact,tool-verified",
            })
        from remy.core.agent_tools import Level
        level = _level_from_call(brain_mock.store.call_args)
        assert level == Level.DOMAIN

    def test_with_fetch_stores_once(self):
        """INV-2: Grounded DOMAIN store must call brain.store exactly once."""
        brain_mock = _make_brain_mock()
        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=_FETCH_EV):
            _call_store(brain_mock, {
                "content": "Study finding",
                "level": "L3_DOMAIN",
                "tags": "fact,tool-verified",
            })
        brain_mock.store.assert_called_once()


# ── INV-3: Tag forgery — tool-verified without fetch ─────────────────────────

class TestTagForgery:
    def test_tool_verified_tag_without_fetch_becomes_llm_unverified(self):
        """INV-3: tool-verified tag without fetch evidence → llm_unverified provenance."""
        from remy.core.brain_tools import _infer_claim_provenance
        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=[]):
            prov = _infer_claim_provenance(
                channel="desktop_worker",
                tags=["fact", "tool-verified"],
                session_id="sess-d03",
            )
        assert prov.kind == "llm_unverified"

    def test_tool_verified_tag_with_fetch_becomes_tool_verified(self):
        """INV-3: tool-verified tag WITH fetch evidence → tool_verified provenance."""
        from remy.core.brain_tools import _infer_claim_provenance
        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=_FETCH_EV):
            prov = _infer_claim_provenance(
                channel="desktop_worker",
                tags=["fact", "tool-verified"],
                session_id="sess-d03",
            )
        assert prov.kind == "tool_verified"

    def test_verified_external_tag_without_fetch_becomes_llm_unverified(self):
        """INV-3: verified-external tag without fetch → llm_unverified."""
        from remy.core.brain_tools import _infer_claim_provenance
        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=[]):
            prov = _infer_claim_provenance(
                channel="desktop_worker",
                tags=["fact", "verified-external"],
                session_id="sess-d03",
            )
        assert prov.kind == "llm_unverified"

    def test_web_search_cache_tag_without_fetch_becomes_llm_unverified(self):
        """INV-3: web-search-cache tag without fetch → llm_unverified."""
        from remy.core.brain_tools import _infer_claim_provenance
        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=[]):
            prov = _infer_claim_provenance(
                channel="desktop_worker",
                tags=["web-search-cache"],
                session_id="sess-d03",
            )
        assert prov.kind == "llm_unverified"


# ── INV-4: Inference tag bypass blocked ───────────────────────────────────────

class TestInferenceTagBypass:
    def test_inference_llm_unverified_is_quarantined(self):
        """INV-4: inference class + llm_unverified provenance → quarantine=True."""
        from remy.core.claim_provenance import decide_storage, ClaimProvenance
        prov = ClaimProvenance.llm_unverified(note="test")
        decision = decide_storage(
            requested_class="inference",
            provenance=prov,
        )
        assert decision.quarantine is True

    def test_inference_llm_unverified_no_factual_store(self):
        """INV-4: inference + llm_unverified → allow_factual_store=False."""
        from remy.core.claim_provenance import decide_storage, ClaimProvenance
        prov = ClaimProvenance.llm_unverified(note="test")
        decision = decide_storage(
            requested_class="inference",
            provenance=prov,
        )
        assert decision.allow_factual_store is False

    def test_proposal_llm_unverified_is_quarantined(self):
        """INV-4: proposal class + llm_unverified → quarantine=True."""
        from remy.core.claim_provenance import decide_storage, ClaimProvenance
        prov = ClaimProvenance.llm_unverified(note="test")
        decision = decide_storage(
            requested_class="proposal",
            provenance=prov,
        )
        assert decision.quarantine is True

    def test_inference_tool_verified_still_allowed(self):
        """INV-4: inference + tool_verified provenance → still allowed."""
        from remy.core.claim_provenance import decide_storage, ClaimProvenance
        prov = ClaimProvenance.tool_verified(tool="extract_content", locator="https://example.com")
        decision = decide_storage(
            requested_class="inference",
            provenance=prov,
        )
        assert decision.quarantine is False
        assert decision.allow_factual_store is True

    def test_inference_system_inferred_still_allowed(self):
        """INV-4: inference + system_inferred → still allowed."""
        from remy.core.claim_provenance import decide_storage, ClaimProvenance
        prov = ClaimProvenance.system_inferred(based_on=[], note="test")
        decision = decide_storage(
            requested_class="inference",
            provenance=prov,
        )
        assert decision.quarantine is False
        assert decision.allow_factual_store is True


# ── INV-5: User-direct channel + user tags → DOMAIN allowed ──────────────────

class TestUserDirectChannel:
    def test_user_direct_channel_with_user_tags_domain_allowed(self):
        """INV-5: user-profile tag in user-direct channel → DOMAIN without fetch."""
        brain_mock = _make_brain_mock()
        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=[]):
            _call_store(brain_mock, {
                "content": "User prefers vegetarian diet",
                "level": "L3_DOMAIN",
                "tags": "user-profile,from-user",
            }, channel="desktop")
        from remy.core.agent_tools import Level
        level = _level_from_call(brain_mock.store.call_args)
        assert level == Level.DOMAIN

    def test_worker_channel_without_user_tags_downgraded(self):
        """INV-5: non-user-direct channel without user tags → still downgraded."""
        brain_mock = _make_brain_mock()
        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=[]):
            _call_store(brain_mock, {
                "content": "Some data",
                "level": "L3_DOMAIN",
                "tags": "fact",
            }, channel="desktop_worker")
        from remy.core.agent_tools import Level
        level = _level_from_call(brain_mock.store.call_args)
        assert level == Level.WORKING


# ── INV-6: Explicit admission_class in metadata → vetted path ────────────────

class TestExplicitAdmissionClass:
    def test_canonical_admission_class_bypasses_transit_check(self):
        """INV-6 (Phase 2 tightened): canonical FACTUAL_SAFE admission_class in
        metadata → DOMAIN allowed (vetted internal path via ingestion API)."""
        brain_mock = _make_brain_mock()
        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=[]):
            _call_store(brain_mock, {
                "content": "Grounded finding from verified source",
                "level": "L3_DOMAIN",
                "tags": "research-finding",
                "metadata": {"admission_class": "grounded_external_fact",
                             "source_url": "https://example.com/study"},
            })
        from remy.core.agent_tools import Level
        level = _level_from_call(brain_mock.store.call_args)
        assert level == Level.DOMAIN

    def test_legacy_admission_class_does_not_bypass(self):
        """Phase 2: legacy non-canonical admission_class ('grounded_finding') is
        no longer a bypass — must route through ingestion module. Downgraded."""
        brain_mock = _make_brain_mock()
        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=[]):
            _call_store(brain_mock, {
                "content": "Legacy ad-hoc admission string",
                "level": "L3_DOMAIN",
                "tags": "research-finding",
                "metadata": {"admission_class": "grounded_finding",
                             "source_url": "https://example.com/study"},
            })
        from remy.core.agent_tools import Level
        level = _level_from_call(brain_mock.store.call_args)
        assert level == Level.WORKING
        tags = _tags_from_call(brain_mock.store.call_args)
        assert "downgraded-no-transit" in tags


# ── INV-7 & INV-8: Downgraded tags ───────────────────────────────────────────

class TestDowngradedTags:
    def test_downgraded_gets_requires_grounding_tag(self):
        """INV-7: Downgraded store must include requires-grounding tag."""
        brain_mock = _make_brain_mock()
        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=[]):
            _call_store(brain_mock, {
                "content": "Some fact without fetch",
                "level": "L3_DOMAIN",
                "tags": "fact",
            })
        tags = _tags_from_call(brain_mock.store.call_args)
        assert "requires-grounding" in tags

    def test_downgraded_gets_downgraded_no_transit_tag(self):
        """INV-8: Downgraded store must include downgraded-no-transit tag."""
        brain_mock = _make_brain_mock()
        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=[]):
            _call_store(brain_mock, {
                "content": "Some fact without fetch",
                "level": "L3_DOMAIN",
                "tags": "fact",
            })
        tags = _tags_from_call(brain_mock.store.call_args)
        assert "downgraded-no-transit" in tags

    def test_grounded_store_does_not_get_downgrade_tags(self):
        """INV-2 / INV-7–8: Grounded store must NOT get downgrade tags."""
        brain_mock = _make_brain_mock()
        with patch("remy.core.claim_provenance.get_turn_fetch_evidence", return_value=_FETCH_EV):
            _call_store(brain_mock, {
                "content": "Study finding with fetch",
                "level": "L3_DOMAIN",
                "tags": "fact,tool-verified",
            })
        tags = _tags_from_call(brain_mock.store.call_args)
        assert "requires-grounding" not in tags
        assert "downgraded-no-transit" not in tags
