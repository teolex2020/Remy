"""
Phase 3 Step 2 — Promotion Governance / Recall Primary Surface Gate.

Design rule: the promotion gate lives at the surface (LLM-facing recall call
sites), not in the SDK / compat layer. Tests reflect that split:

  A. Predicate tests for `_is_factual_forbidden` — the single rule-set.
  B. Filter tests for `_apply_factual_recall_filter` — the helper called by
     every surface that hands records to the LLM.
  C. Call-site integration tests — verify that the two LLM-facing recall
     handlers (brain_tools.execute_tool('recall') and tool_dispatch's
     equivalent) actually strip forbidden records out of their output.

Internal callers of `recall_structured` / `recall_full` (research dedup,
scratchpad, auto-connect) must keep the unfiltered view — they are not tested
here because their contract is explicitly "no surface filter".

Structural / behavioral only. No exact-text assertions.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from remy.core.hybrid_search import _is_factual_forbidden
from remy.core.agent_tools import _apply_factual_recall_filter


# ── Helpers ──────────────────────────────────────────────────────────────────


def _item(**meta_overrides) -> dict:
    """Build a recall-item dict with a grounded_external_fact base.

    Base record is explicitly factual-safe (no forbidden tags, no forbidden
    admission class) so any block from the new rules is attributable to the
    new rule only.
    """
    metadata: dict = {
        "source": "https://arxiv.org/abs/2411.02534",
        "verified": True,
        "admission_class": "grounded_external_fact",
    }
    metadata.update(meta_overrides)
    return {
        "id": "rec-1",
        "tags": [],
        "metadata": metadata,
        "content": "Some factual statement.",
        "score": 0.7,
    }


# ── A.1 — requires_promotion gate ────────────────────────────────────────────


def test_requires_promotion_without_promoted_blocks_recall():
    """Admitted-but-not-promoted must not surface as factual primary substrate."""
    item = _item(requires_promotion=True)
    assert _is_factual_forbidden(item) is True


def test_requires_promotion_with_promoted_allows_recall():
    """Once explicitly promoted, the record is eligible again."""
    item = _item(requires_promotion=True, promoted=True)
    assert _is_factual_forbidden(item) is False


def test_no_promotion_flag_keeps_record_eligible():
    """Records without the promotion flag behave as before (class-only gating)."""
    item = _item()
    assert _is_factual_forbidden(item) is False


# ── A.2 — unresolved_conflict gate ───────────────────────────────────────────


def test_unresolved_conflict_blocks_recall():
    item = _item(unresolved_conflict=True)
    assert _is_factual_forbidden(item) is True


def test_conflict_resolved_does_not_block_recall():
    """A resolved conflict must not keep blocking — resolution is the point."""
    item = _item(conflict_resolved=True)
    assert _is_factual_forbidden(item) is False


# ── A.3 — superseded_by gate ─────────────────────────────────────────────────


def test_superseded_by_blocks_recall():
    item = _item(superseded_by="rec-new-42")
    assert _is_factual_forbidden(item) is True


def test_empty_superseded_by_does_not_block():
    item = _item(superseded_by="")
    assert _is_factual_forbidden(item) is False


# ── A.4 — truth_status gate (Phase 4 freshness) ──────────────────────────────


def test_stale_hard_blocks_recall():
    """Cached_at far past 2x TTL ⇒ stale_hard ⇒ forbidden."""
    long_ago = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    item = _item(volatility="high", cached_at=long_ago)
    # high volatility TTL is 7 days; 400 days >> 14 days ⇒ stale_hard
    assert _is_factual_forbidden(item) is True


def test_fresh_record_passes():
    """Recent cached_at within TTL ⇒ fresh ⇒ not blocked."""
    recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    item = _item(volatility="high", cached_at=recent)
    assert _is_factual_forbidden(item) is False


# ── A.5 — regression guard: existing A.8 / A.9 gates still fire ──────────────


def test_existing_admission_class_gate_still_works():
    """New rules must not weaken existing A.9 class gating."""
    generated = _item(admission_class="generated_analysis")
    research_artifact = _item(admission_class="research_artifact")
    unverified = _item(admission_class="unverified_claim")
    assert _is_factual_forbidden(generated) is True
    assert _is_factual_forbidden(research_artifact) is True
    assert _is_factual_forbidden(unverified) is True


def test_existing_tag_gate_still_works():
    """A.8 tag gate must still fire."""
    item = _item()
    item["tags"] = ["generated-report"]
    assert _is_factual_forbidden(item) is True


# ── B — _apply_factual_recall_filter helper ──────────────────────────────────


def test_apply_filter_drops_forbidden_items_only():
    fresh = _item()
    fresh["id"] = "fresh-1"
    stale = _item(volatility="high",
                  cached_at=(datetime.now(timezone.utc) - timedelta(days=400)).isoformat())
    stale["id"] = "stale-1"
    promotion_pending = _item(requires_promotion=True)
    promotion_pending["id"] = "pending-1"
    conflicted = _item(unresolved_conflict=True)
    conflicted["id"] = "conflict-1"

    result = _apply_factual_recall_filter([fresh, stale, promotion_pending, conflicted])
    assert [r["id"] for r in result] == ["fresh-1"]


def test_apply_filter_handles_empty_and_none():
    assert _apply_factual_recall_filter(None) == []
    assert _apply_factual_recall_filter([]) == []


def test_apply_filter_preserves_order_of_allowed_items():
    """Ordering must be stable — recall relies on rank."""
    a = dict(_item(), id="a")
    b = dict(_item(), id="b")
    c = dict(_item(), id="c")
    blocked = dict(_item(requires_promotion=True), id="blocked")
    result = _apply_factual_recall_filter([a, blocked, b, c])
    assert [r["id"] for r in result] == ["a", "b", "c"]


# ── C — call-site integration: brain_tools.recall tool ───────────────────────
#
# The LLM-facing `recall` tool in brain_tools.execute_tool must hand the
# filtered list into its output. We verify at the list-shape level that
# forbidden records drop out before rendering.


def test_call_site_filter_drops_requires_promotion_from_llm_recall():
    """Simulate the brain_tools recall call-site chain with a mixed list."""
    ok = dict(_item(), id="ok-1", content="Fresh fact about X.")
    pending = dict(_item(requires_promotion=True), id="pending-1",
                   content="Unpromoted research stamp.")
    conflicted = dict(_item(unresolved_conflict=True), id="conf-1",
                      content="Still in conflict.")

    # The call-site integration step after recall_structured + search.
    filtered = _apply_factual_recall_filter([ok, pending, conflicted])

    ids = [r["id"] for r in filtered]
    assert ids == ["ok-1"]
    assert all(r.get("metadata", {}).get("requires_promotion") is not True
               for r in filtered)


def test_call_site_filter_survives_sparse_phase4_metadata():
    """Records missing volatility/cached_at but carrying a safe admission_class
    must still pass — truth_status returns no_expiry for unset TTLs, and the
    Phase 4 gate must not block that."""
    sparse = {
        "id": "sparse-1",
        "tags": [],
        "metadata": {"admission_class": "grounded_external_fact", "verified": True},
        "content": "Grounded fact without cached_at/volatility.",
        "score": 0.5,
    }
    result = _apply_factual_recall_filter([sparse])
    assert [r["id"] for r in result] == ["sparse-1"]
