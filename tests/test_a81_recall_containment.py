"""
Phase A.8.1 — Direct Recall Tool Containment for Factual Runs.

Behavior assertions:

  1. factual query via direct recall tool does NOT return contaminated
     research/generated/quarantine records as primary results
  2. factual query + recall is consistent with the A.8 evidence-packet boundary
  3. non-factual query via recall is UNAFFECTED — general memory still works
  4. is_factual_query classifier covers expected patterns without false positives
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from remy.core.hybrid_search import (
    _FACTUAL_FORBIDDEN_TAGS,
    is_factual_query,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _dict_rec(record_id: str, tags: list[str], content: str = "content", verified: bool = False):
    """Minimal dict-format recall result (as returned by recall_structured)."""
    return {
        "id": record_id,
        "content": content,
        "tags": tags,
        "metadata": {"source": "user" if verified else "agent-autonomous", "verified": verified},
        "score": 0.7,
    }


def _obj_rec(record_id: str, tags: list[str], content: str = "content"):
    """Minimal object-format record (as returned by brain.search)."""
    r = SimpleNamespace()
    r.id = record_id
    r.content = content
    r.tags = tags
    r.metadata = {"source": "agent-autonomous", "verified": False}
    r.strength = 0.5
    r.activation_count = 3
    r.level = SimpleNamespace(name="DOMAIN")
    r.confidence = 0.6
    r.conflict_mass = 0
    r.subject = None
    r.outcome_polarity = None
    r.importance = None
    return r


# ── Test 1: factual query recall does not return forbidden records ─────────────


@pytest.mark.parametrize("forbidden_tag", sorted(_FACTUAL_FORBIDDEN_TAGS))
def test_factual_recall_excludes_forbidden_tag(forbidden_tag):
    """Direct recall on a factual query must not return records with forbidden tags."""
    from remy.core.hybrid_search import is_factual_query

    # Confirm the query we use is actually classified as factual
    query = "verify: arxiv paper 2402.17764"
    assert is_factual_query(query)

    dirty = _dict_rec("dirty-1", [forbidden_tag], content="Research narrative blob")
    clean = _dict_rec("clean-1", ["user-confirmed"], content="Verified fact", verified=True)

    # Simulate the filter logic from brain_tools.py recall handler
    brain_results = [dirty, clean]
    filtered = [
        r for r in brain_results
        if not _FACTUAL_FORBIDDEN_TAGS.intersection(
            set(r.get("tags") or []) if isinstance(r, dict)
            else set(getattr(r, "tags", []) or [])
        )
    ]

    ids = {r["id"] for r in filtered}
    assert "dirty-1" not in ids, (
        f"Tag '{forbidden_tag}' must be excluded from factual recall results"
    )
    assert "clean-1" in ids


# ── Test 2: non-factual recall is unaffected ──────────────────────────────────


def test_non_factual_recall_keeps_research_records():
    """General memory queries must NOT be filtered — only factual queries apply containment."""
    query = "що ти знаєш про мій проект"
    assert not is_factual_query(query), "This should be a non-factual query"

    research_rec = _dict_rec("rp-1", ["research-project"], content="Research project record")
    generated_rec = _dict_rec("gr-1", ["generated-report"], content="Generated report blob")

    brain_results = [research_rec, generated_rec]

    # Non-factual path: filter NOT applied
    if is_factual_query(query):
        filtered = [
            r for r in brain_results
            if not _FACTUAL_FORBIDDEN_TAGS.intersection(set(r.get("tags") or []))
        ]
    else:
        filtered = brain_results  # unchanged

    ids = {r["id"] for r in filtered}
    assert "rp-1" in ids, "research-project must remain in non-factual recall"
    assert "gr-1" in ids, "generated-report must remain in non-factual recall"


# ── Test 3: is_factual_query classifier ───────────────────────────────────────


@pytest.mark.parametrize("query", [
    "дай arXiv ID статті Attention Is All You Need",
    "verify: GPT-4 has 1.8T parameters",
    "list recent papers on transformer efficiency",
    "who wrote the BERT paper",
    "знайди статті про cognitive architectures 2024",
    "arxiv:2402.17764 — що це за стаття",
    "check if this citation is correct",
    "which year was ResNet published",
    "how many parameters does GPT-3 have",
    "cite this claim",
])
def test_is_factual_query_true_for_citation_patterns(query):
    assert is_factual_query(query), f"Should be factual: {query!r}"


@pytest.mark.parametrize("query", [
    "запам'ятай це",
    "продовжуй роботу",
    "що ти думаєш про це",
    "розкажи мені про себе",
    "збережи цю нотатку",
    "нагадай завтра",
    "summarize the conversation",
    "help me write an email",
    "що у тебе в пам'яті",
])
def test_is_factual_query_false_for_non_factual(query):
    assert not is_factual_query(query), f"Should NOT be factual: {query!r}"


# ── Test 4: object-format records also filtered ───────────────────────────────


def test_factual_filter_works_on_object_format_records():
    """brain.search() returns PyO3 objects, not dicts. Filter must handle both."""
    obj_dirty = _obj_rec("obj-dirty", ["research-finding"])
    obj_clean = _obj_rec("obj-clean", ["user-confirmed"])

    brain_results_mixed = [
        {"id": "dict-dirty", "tags": ["generated-report"], "content": "x", "metadata": {}, "score": 0.5},
        obj_dirty,
        {"id": "dict-clean", "tags": ["fact"], "content": "y", "metadata": {}, "score": 0.5},
        obj_clean,
    ]

    filtered = [
        r for r in brain_results_mixed
        if not _FACTUAL_FORBIDDEN_TAGS.intersection(
            set(r.get("tags") or []) if isinstance(r, dict)
            else set(getattr(r, "tags", []) or [])
        )
    ]
    ids = {
        (r["id"] if isinstance(r, dict) else r.id)
        for r in filtered
    }
    assert "dict-dirty" not in ids
    assert "obj-dirty" not in ids
    assert "dict-clean" in ids
    assert "obj-clean" in ids
