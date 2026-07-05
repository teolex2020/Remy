"""Live wiring test: provenance-ranked recall in the agent's memory-context path.

Proves that the hybrid_search.recall_provenance_ranked wrapper (now used by
agent.py's "Cognitive Recall" block) actually surfaces a lived-consequence
memory ahead of an equally-relevant model-generated one — using the REAL
AuraSDK, not a mock. Fail-soft fallback is also covered.
"""

import tempfile

import pytest

aura = pytest.importorskip("aura")
if not hasattr(aura.Aura, "recall_provenance_ranked"):
    pytest.skip(
        "installed AuraSDK lacks recall_provenance_ranked; wiring falls back safely",
        allow_module_level=True,
    )

from remy.core.agent_tools import _AuraCompat
from remy.core.hybrid_search import recall_provenance_ranked


def _store():
    return _AuraCompat(tempfile.mkdtemp())


def test_lived_consequence_outranks_model_generated_in_recall():
    store = _store()
    # Lived consequence on the topic (DECISIONS tier, consequence-support tag).
    store.capture_consequence(
        "fever management",
        "give acetaminophen",
        "temperature dropped, patient comfortable",
        1,
    )
    # Model-generated note on the same topic, slightly more keyword-dense.
    store.store(
        "fever management give acetaminophen common antipyretic option here",
        level=aura.Level.Working,
        source_type="generated",
    )

    hits = recall_provenance_ranked(
        store, "fever management acetaminophen", top_k=5, min_strength=0.0
    )
    assert hits, "expected at least one cognitive hit"
    # The top hit should be the lived consequence (provenance annotation present).
    top = hits[0]
    assert top.get("provenance") == "lived_consequence", (
        f"expected lived consequence on top, got {top.get('provenance')}: {top.get('content')!r}"
    )


def test_recall_provenance_ranked_fails_soft_on_bad_brain():
    class Broken:
        def recall_provenance_ranked(self, *a, **k):
            raise RuntimeError("boom")

        def recall_structured(self, *a, **k):
            return []

    # Must not raise; falls back to cognitive recall (empty here).
    hits = recall_provenance_ranked(Broken(), "anything", top_k=3)
    assert hits == []
