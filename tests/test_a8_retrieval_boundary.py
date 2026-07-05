"""
Phase A.8 — ACL-Native Retrieval Boundary.

Behavior assertions on the factual retrieval path:

  - factual/citation/verify turns must not receive generated-report,
    research-project, research-finding, research, session-summary,
    scratchpad, quarantine-unverified, or claim:llm-unverified records
    as primary evidence
  - clean/verified records must pass through the evidence packet
  - the non-factual path is unaffected (forbidden classes still returned)
  - evidence items carry provenance + allowed_use annotations

No exact-text assertions. Only structural/behavioral.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from remy.core.hybrid_search import (
    _FACTUAL_FORBIDDEN_TAGS,
    _is_factual_forbidden,
    build_evidence_packet,
    search_exact_structured,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _fake_rec(record_id: str, tags: list[str], content: str = "some content", verified: bool = False):
    """Build a minimal fake PyO3-like record object."""
    r = SimpleNamespace()
    r.id = record_id
    r.content = content
    r.tags = tags
    r.level = SimpleNamespace(name="DOMAIN")
    r.strength = 0.6
    r.activation_count = 5
    r.metadata = {"source": "user" if verified else "agent-autonomous", "verified": verified}
    r.confidence = 0.7
    r.conflict_mass = 0
    r.subject = None
    r.outcome_polarity = None
    r.importance = None
    return r


def _make_brain(domain_records=None, semantic_records=None):
    """Minimal fake brain satisfying the search / recall_structured interface."""
    brain = MagicMock()
    brain.search.return_value = domain_records or []
    brain.recall_structured.return_value = semantic_records or []
    return brain


# ── Test 1: forbidden tags never enter evidence packet ────────────────────────


@pytest.mark.parametrize("forbidden_tag", sorted(_FACTUAL_FORBIDDEN_TAGS))
def test_forbidden_record_excluded_from_evidence_packet(forbidden_tag):
    """A record carrying any forbidden tag must not appear in build_evidence_packet()."""
    dirty = _fake_rec("dirty-1", [forbidden_tag], content="Research narrative blob")
    clean = _fake_rec("clean-1", ["user-confirmed"], content="Verified fact", verified=True)

    brain = _make_brain(domain_records=[dirty, clean])

    evidence = build_evidence_packet(brain, "find papers on X", top_k=10)
    evidence_ids = {e["id"] for e in evidence}

    assert "dirty-1" not in evidence_ids, (
        f"Record with tag '{forbidden_tag}' must not enter the evidence packet"
    )


def test_clean_record_passes_through_evidence_packet():
    """A record with no forbidden tags must be returned in the evidence packet."""
    clean = _fake_rec("clean-2", ["user-confirmed", "fact"], content="Verified fact", verified=True)
    brain = _make_brain(domain_records=[clean])

    evidence = build_evidence_packet(brain, "any factual query", top_k=5)
    evidence_ids = {e["id"] for e in evidence}

    assert "clean-2" in evidence_ids


# ── Test 2: quarantine + claim:llm-unverified never in primary substrate ──────


def test_quarantined_record_not_in_evidence():
    """quarantine-unverified must be excluded even if it has high activation_count."""
    quarantined = _fake_rec(
        "q-1",
        ["quarantine-unverified"],
        content="Plausible but unverified claim with 50 activations",
    )
    quarantined.activation_count = 50  # high signal — must still be excluded

    brain = _make_brain(domain_records=[quarantined])
    evidence = build_evidence_packet(brain, "verify claim about X", top_k=5)
    assert not any(e["id"] == "q-1" for e in evidence)


def test_llm_unverified_claim_not_in_evidence():
    brain = _make_brain(domain_records=[
        _fake_rec("llm-1", ["claim:llm-unverified"], content="LLM invented this"),
    ])
    evidence = build_evidence_packet(brain, "cite this claim", top_k=5)
    assert not any(e["id"] == "llm-1" for e in evidence)


# ── Test 3: non-factual path is unaffected ────────────────────────────────────


def test_search_exact_structured_still_returns_research_records():
    """The general recall path (non-factual) must NOT filter research/generated records.
    Phase A.8 only gates the factual path; general recall is unchanged.
    """
    research_rec = _fake_rec("rp-1", ["research-project"], content="Research project record")
    brain = _make_brain(domain_records=[research_rec])

    # general path — no forbidden-class filter
    results = search_exact_structured(brain, "any query", top_k=5)
    result_ids = {r["id"] for r in results}
    assert "rp-1" in result_ids, (
        "search_exact_structured must still return research records — "
        "forbidden filtering is factual-path-only"
    )


# ── Test 4: evidence items carry provenance annotations ───────────────────────


def test_evidence_packet_items_carry_provenance_fields():
    """Every item in the evidence packet must have verification_state,
    provenance, claim_status, and allowed_use fields."""
    rec = _fake_rec("prov-1", ["user-confirmed"], content="A grounded fact", verified=True)
    brain = _make_brain(domain_records=[rec])

    evidence = build_evidence_packet(brain, "grounded query", top_k=5)
    assert evidence, "should return at least one item"

    for item in evidence:
        assert "verification_state" in item, "missing verification_state"
        assert "provenance" in item, "missing provenance"
        assert "claim_status" in item, "missing claim_status"
        assert "allowed_use" in item, "missing allowed_use"


def test_verified_record_gets_cite_allowed_use():
    rec = _fake_rec("v-1", ["user-confirmed"], content="Verified fact", verified=True)
    brain = _make_brain(domain_records=[rec])

    evidence = build_evidence_packet(brain, "cite this", top_k=5)
    item = next((e for e in evidence if e["id"] == "v-1"), None)
    assert item is not None
    assert item["verification_state"] == "verified"
    assert item["allowed_use"] == "cite"


def test_unverified_agent_autonomous_record_excluded_from_evidence():
    """Phase A.9: an unverified agent-autonomous record with no explicit
    admission_class derives to working_state/reflection → forbidden on factual path.
    The A.8 test originally expected context_only; A.9 tightens this to exclusion."""
    rec = _fake_rec("u-1", ["fact"], content="Stated but unverified", verified=False)
    # _fake_rec sets source="agent-autonomous" which derives to "reflection" in A.9
    brain = _make_brain(domain_records=[rec])

    evidence = build_evidence_packet(brain, "any query", top_k=5)
    # A.9 admission gate: agent-autonomous + no explicit class → forbidden class → excluded
    assert not any(e["id"] == "u-1" for e in evidence), (
        "unverified agent-autonomous record must be excluded from evidence packet "
        "(A.9: derives to reflection/working_state which is FACTUAL_FORBIDDEN)"
    )


# ── Test 5: _is_factual_forbidden coverage ────────────────────────────────────


def test_is_factual_forbidden_false_for_safe_items():
    """Phase A.9: items are only safe if they carry explicit safe admission_class
    OR verified=True from a non-autonomous source. Tags alone are not sufficient
    (A.9 tightened this from A.8)."""
    safe_items = [
        # explicit safe class wins
        {"tags": ["user-confirmed"], "metadata": {"admission_class": "operator_asserted"}},
        {"tags": [], "metadata": {"admission_class": "grounded_external_fact", "verified": True}},
        {"tags": [], "metadata": {"verified": True, "source": "user"}},
    ]
    for item in safe_items:
        assert not _is_factual_forbidden(item), f"Should not be forbidden: {item}"


def test_is_factual_forbidden_true_for_all_forbidden_tags():
    for tag in _FACTUAL_FORBIDDEN_TAGS:
        item = {"tags": [tag]}
        assert _is_factual_forbidden(item), f"Should be forbidden: {tag}"


def test_is_factual_forbidden_true_if_mixed_with_safe_tags():
    item = {"tags": ["user-confirmed", "generated-report"]}
    assert _is_factual_forbidden(item), (
        "Mixed bag: one forbidden tag is enough to gate the record"
    )
