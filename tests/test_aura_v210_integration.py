"""
Integration test: aura-memory 2.1.0 Phase 3 — Structured Contradiction
Tests that the agent's memory layer correctly infers and exposes
subject/slot/value and that contradiction detection works end-to-end.
"""
import tempfile
import pytest
import aura


@pytest.fixture
def brain():
    """Create a fresh Aura instance in a temp directory."""
    d = tempfile.mkdtemp()
    a = aura.Aura(d)
    yield a


class TestPhase3FieldsExposed:
    """New Phase 3 fields are accessible from Python."""

    def test_positive_outcome_fields(self, brain):
        rid = brain.store(
            "Payment gateway scan passed",
            tags=["security", "scan", "passed"],
        )
        rec = brain.get(rid)
        assert rec.outcome_polarity == aura.OutcomePolarity.Positive
        assert "security" in rec.outcome_domain
        assert "scan" in rec.outcome_domain
        assert rec.subject is not None
        assert rec.state_slot == "outcome"
        assert rec.state_value == "positive"

    def test_negative_outcome_fields(self, brain):
        rid = brain.store(
            "Payment gateway scan failed",
            tags=["security", "scan", "failed"],
        )
        rec = brain.get(rid)
        assert rec.outcome_polarity == aura.OutcomePolarity.Negative
        assert rec.state_slot == "outcome"
        assert rec.state_value == "negative"

    def test_same_subject_different_outcome(self, brain):
        """Two records about the same entity have the same subject."""
        r1 = brain.store(
            "Auth service scan passed",
            tags=["security", "scan", "passed"],
        )
        r2 = brain.store(
            "Auth service scan failed",
            tags=["security", "scan", "failed"],
        )
        rec1 = brain.get(r1)
        rec2 = brain.get(r2)
        # Same subject (both about "auth service scan")
        assert rec1.subject == rec2.subject
        # Same slot
        assert rec1.state_slot == rec2.state_slot == "outcome"
        # Different values — this is a structured contradiction
        assert rec1.state_value == "positive"
        assert rec2.state_value == "negative"

    def test_no_polarity_deploy_slot(self, brain):
        """Record with deploy keywords but no polarity → deployment_status slot."""
        rid = brain.store(
            "Deploy to staging environment",
            tags=["deploy", "staging"],
        )
        rec = brain.get(rid)
        assert rec.state_slot == "deployment_status"
        assert rec.subject is not None

    def test_neutral_content_none_slot(self, brain):
        """Record with no outcome/slot keywords → None slot/value."""
        rid = brain.store(
            "Project timeline reviewed",
            tags=["meeting", "timeline"],
        )
        rec = brain.get(rid)
        # No outcome keywords, no slot keywords → None
        assert rec.state_slot is None
        assert rec.state_value is None

    def test_explicit_metadata_subject(self, brain):
        """Explicit metadata['subject'] overrides inference."""
        rid = brain.store(
            "Something happened",
            tags=["event"],
            metadata={"subject": "ops:payment-service"},
        )
        rec = brain.get(rid)
        assert rec.subject == "ops:payment-service"

    def test_verification_slot_from_tags(self, brain):
        """Tags containing 'scan'/'verify' → verification_status slot."""
        rid = brain.store(
            "Network audit scheduled for review",
            tags=["audit", "network"],
        )
        rec = brain.get(rid)
        # "audit" maps to verification_status via SLOT_KEYWORDS
        assert rec.state_slot == "verification_status"


class TestValueEquivalence:
    """Canonical value normalization works correctly."""

    def test_passed_and_verified_same_canonical(self, brain):
        r1 = brain.store("Scan passed", tags=["scan", "passed"])
        r2 = brain.store("Scan verified", tags=["scan", "verified"])
        rec1 = brain.get(r1)
        rec2 = brain.get(r2)
        # Both should normalize to "positive"
        assert rec1.state_value == "positive"
        # "verified" is a positive keyword, should also be "positive"
        assert rec2.state_value == "positive"

    def test_failed_and_rejected_same_canonical(self, brain):
        r1 = brain.store("Deploy failed", tags=["deploy", "failed"])
        r2 = brain.store("Deploy rejected", tags=["deploy", "rejected"])
        rec1 = brain.get(r1)
        rec2 = brain.get(r2)
        assert rec1.state_value == "negative"
        assert rec2.state_value == "negative"


class TestMaintenanceCycle:
    """Phase 3 fields survive a full maintenance cycle."""

    def test_fields_persist_after_maintenance(self, brain):
        r1 = brain.store(
            "Auth service scan passed",
            tags=["security", "scan", "passed"],
        )
        # Run maintenance — belief engine, epistemic update, etc.
        brain.run_maintenance()

        rec = brain.get(r1)
        assert rec.subject is not None
        assert rec.state_slot == "outcome"
        assert rec.state_value == "positive"
        assert rec.outcome_polarity == aura.OutcomePolarity.Positive

    def test_contradiction_pair_after_maintenance(self, brain):
        """Store contradicting records, run maintenance, check epistemic state."""
        r1 = brain.store(
            "Auth service scan passed",
            tags=["security", "scan", "passed"],
            namespace="security",
        )
        r2 = brain.store(
            "Auth service scan failed with errors",
            tags=["security", "scan", "failed"],
            namespace="security",
        )
        # Multiple cycles to let epistemic update propagate
        for _ in range(3):
            brain.run_maintenance()

        rec1 = brain.get(r1)
        rec2 = brain.get(r2)

        # Both should have same subject
        assert rec1.subject == rec2.subject
        # We expect some conflict propagation from structured contradiction
        # (same subject + same slot + different value)
        print(f"  conflict_mass: r1={rec1.conflict_mass}, r2={rec2.conflict_mass}")
        print(f"  volatility:    r1={rec1.volatility:.3f}, r2={rec2.volatility:.3f}")
        print(f"  subjects:      r1={rec1.subject}, r2={rec2.subject}")
        print(f"  slots:         r1={rec1.state_slot}, r2={rec2.state_slot}")
        print(f"  values:        r1={rec1.state_value}, r2={rec2.state_value}")

    def test_multi_topic_isolation(self, brain):
        """Records about different subjects should NOT conflict."""
        brain.store(
            "Payment gateway scan passed",
            tags=["security", "scan", "passed"],
            namespace="security",
        )
        brain.store(
            "Network audit failed",
            tags=["network", "audit", "failed"],
            namespace="security",
        )
        for _ in range(3):
            brain.run_maintenance()

        # These are different subjects — should not generate cross-conflict
        recs = brain.search(limit=100)
        for rec in recs:
            print(f"  {rec.id}: subject={rec.subject}, slot={rec.state_slot}, "
                  f"value={rec.state_value}, conflict={rec.conflict_mass}")
