"""
Phase A.9 — Explicit Learning Boundary / Admission Classes.

Behavior assertions:

  1. derive_admission_class() correctly maps metadata + tags to admission class
  2. generated_analysis records never enter build_evidence_packet() as factual
  3. working_state / plan / reflection records excluded from factual recall
  4. grounded_external_fact / operator_asserted pass through evidence packet
  5. missing admission_class defaults safely (not silently promoted to factual)
  6. _is_factual_forbidden enforces admission_class from metadata, not just tags

No exact-text assertions. Only structural/behavioral.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from remy.core.memory_policy import (
    ADMISSION_CLASSES,
    FACTUAL_FORBIDDEN_ADMISSION_CLASSES,
    FACTUAL_SAFE_ADMISSION_CLASSES,
    derive_admission_class,
)
from remy.core.hybrid_search import (
    _FACTUAL_FORBIDDEN_TAGS,
    _is_factual_forbidden,
    build_evidence_packet,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _item(record_id: str, tags: list[str], admission_class: str | None = None,
          verified: bool = False, source: str = "agent-autonomous") -> dict:
    meta: dict = {"source": source, "verified": verified}
    if admission_class:
        meta["admission_class"] = admission_class
    return {"id": record_id, "tags": tags, "metadata": meta, "content": "some content", "score": 0.7}


def _make_brain(domain_records=None):
    brain = MagicMock()
    brain.search.return_value = domain_records or []
    brain.recall_structured.return_value = []
    return brain


# ── Test 1: derive_admission_class coverage ───────────────────────────────────


@pytest.mark.parametrize("tags,expected_class", [
    (["generated-report"],    "generated_analysis"),
    (["research-project"],    "research_artifact"),
    (["research-finding"],    "research_artifact"),
    (["research"],            "research_artifact"),
    (["research-summary"],    "research_artifact"),
    (["session-summary"],     "reflection"),
    (["scratchpad"],          "working_state"),
    (["quarantine-unverified"], "unverified_claim"),
    (["claim:llm-unverified"],  "unverified_claim"),
    (["autonomous-outcome"],  "reflection"),
    (["autonomous-plan"],     "plan"),
    (["session-reflection"],  "reflection"),
    (["todo-item"],           "working_state"),
    (["outcome-failure"],     "working_state"),
])
def test_derive_admission_class_from_tags(tags, expected_class):
    got = derive_admission_class({}, tags)
    assert got == expected_class, f"tags={tags}: expected {expected_class}, got {got}"


def test_explicit_admission_class_wins_over_tags():
    """Explicit metadata field takes priority over tag inference."""
    meta = {"admission_class": "operator_asserted"}
    got = derive_admission_class(meta, ["session-summary"])
    assert got == "operator_asserted"


def test_verified_user_source_defaults_to_grounded():
    meta = {"verified": True, "source": "user"}
    got = derive_admission_class(meta, [])
    assert got == "grounded_external_fact"


def test_autonomous_source_defaults_to_reflection():
    meta = {"source": "agent-autonomous", "verified": False}
    got = derive_admission_class(meta, [])
    assert got == "reflection"


def test_unknown_record_defaults_to_working_state():
    """No tags, no explicit class, no recognizable source → safe conservative default."""
    got = derive_admission_class({}, [])
    assert got == "working_state"
    assert got not in FACTUAL_SAFE_ADMISSION_CLASSES


# ── Test 2: generated_analysis never enters evidence packet ───────────────────


def test_generated_analysis_excluded_from_evidence_packet():
    """A record with admission_class=generated_analysis must not enter build_evidence_packet."""
    rec = SimpleNamespace()
    rec.id = "gen-1"
    rec.content = "LLM-synthesised research summary"
    rec.tags = []
    rec.level = SimpleNamespace(name="DOMAIN")
    rec.strength = 0.8
    rec.activation_count = 20
    rec.metadata = {"admission_class": "generated_analysis", "source": "agent-autonomous", "verified": False}
    rec.confidence = 0.7
    rec.conflict_mass = 0
    rec.subject = None
    rec.outcome_polarity = None
    rec.importance = None

    brain = _make_brain(domain_records=[rec])
    evidence = build_evidence_packet(brain, "cite this", top_k=5)
    assert not any(e["id"] == "gen-1" for e in evidence), (
        "generated_analysis record must not appear in evidence packet"
    )


# ── Test 3: working_state / plan / reflection excluded ────────────────────────


@pytest.mark.parametrize("forbidden_class", sorted(FACTUAL_FORBIDDEN_ADMISSION_CLASSES))
def test_forbidden_admission_class_excluded_from_evidence(forbidden_class):
    item = _item("test-1", [], admission_class=forbidden_class)
    assert _is_factual_forbidden(item), (
        f"admission_class={forbidden_class} must be forbidden on factual path"
    )


# ── Test 4: safe classes pass through ────────────────────────────────────────


@pytest.mark.parametrize("safe_class", sorted(FACTUAL_SAFE_ADMISSION_CLASSES))
def test_safe_admission_class_passes_through(safe_class):
    item = _item("safe-1", [], admission_class=safe_class, verified=True, source="user")
    assert not _is_factual_forbidden(item), (
        f"admission_class={safe_class} must NOT be forbidden"
    )


# ── Test 5: missing class defaults safely ────────────────────────────────────


def test_missing_admission_class_on_autonomous_record_is_forbidden():
    """An autonomous record with no explicit admission_class must derive to
    a non-factual class — it must not silently become factual substrate."""
    item = _item("anon-1", [], admission_class=None,
                 verified=False, source="agent-autonomous")
    # derive_admission_class returns "reflection" for agent-autonomous
    cls = derive_admission_class(item["metadata"], item["tags"])
    assert cls not in FACTUAL_SAFE_ADMISSION_CLASSES, (
        f"autonomous record with no explicit class derived to {cls!r} "
        f"which is incorrectly in FACTUAL_SAFE_ADMISSION_CLASSES"
    )


def test_missing_admission_class_empty_record_is_working_state():
    """Record with no metadata, no tags → defaults to working_state (not grounded)."""
    cls = derive_admission_class({}, [])
    assert cls == "working_state"
    assert cls in FACTUAL_FORBIDDEN_ADMISSION_CLASSES


# ── Test 6: _is_factual_forbidden uses admission_class not only tags ──────────


def test_is_factual_forbidden_respects_explicit_admission_class():
    """admission_class in metadata overrides tag-based inference."""
    # Tag says safe, but explicit class says forbidden
    item_tag_safe_class_forbidden = {
        "id": "x",
        "tags": ["user-confirmed"],  # no forbidden tag
        "metadata": {"admission_class": "generated_analysis"},
        "content": "x",
    }
    assert _is_factual_forbidden(item_tag_safe_class_forbidden), (
        "explicit admission_class=generated_analysis must trigger forbidden even with safe tags"
    )

    # Tag says forbidden but explicit class says safe
    item_tag_forbidden_class_safe = {
        "id": "y",
        "tags": ["session-summary"],  # would be forbidden by tag
        "metadata": {"admission_class": "operator_asserted"},
        "content": "y",
    }
    # Tag fires first in current impl — this is correct behaviour:
    # an operator_asserted record tagged session-summary is unusual enough
    # that the tag-gate should still apply. Verify the tag gate fires.
    assert _is_factual_forbidden(item_tag_forbidden_class_safe), (
        "tag gate fires before admission_class check — session-summary tag overrides"
    )


# ── Test 7: ADMISSION_CLASSES completeness ───────────────────────────────────


def test_safe_and_forbidden_classes_are_disjoint():
    overlap = FACTUAL_SAFE_ADMISSION_CLASSES & FACTUAL_FORBIDDEN_ADMISSION_CLASSES
    assert not overlap, f"Classes appear in both safe and forbidden sets: {overlap}"


def test_all_classes_in_master_set():
    for cls in FACTUAL_SAFE_ADMISSION_CLASSES | FACTUAL_FORBIDDEN_ADMISSION_CLASSES:
        assert cls in ADMISSION_CLASSES, f"{cls!r} not in ADMISSION_CLASSES master set"
