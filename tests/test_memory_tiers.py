"""Tests for the two-tier cognitive/core memory architecture."""

from remy.core.agent_tools import (
    COGNITIVE_LEVELS,
    CORE_LEVELS,
    Level,
    tier_of,
)

# ============== tier_of() ==============


class TestTierOf:
    def test_working_is_cognitive(self):
        assert tier_of(Level.WORKING) == "cognitive"

    def test_decisions_is_cognitive(self):
        assert tier_of(Level.DECISIONS) == "cognitive"

    def test_domain_is_core(self):
        assert tier_of(Level.DOMAIN) == "core"

    def test_identity_is_core(self):
        assert tier_of(Level.IDENTITY) == "core"

    def test_string_working(self):
        assert tier_of("Level.Working") == "cognitive"

    def test_string_decisions(self):
        assert tier_of("Decisions") == "cognitive"

    def test_string_domain(self):
        assert tier_of("Level.Domain") == "core"

    def test_string_identity(self):
        assert tier_of("IDENTITY") == "core"

    def test_string_unknown_defaults_to_core(self):
        assert tier_of("something_else") == "core"


# ============== Constants ==============


class TestTierConstants:
    def test_cognitive_levels_tuple(self):
        assert len(COGNITIVE_LEVELS) == 2
        assert Level.WORKING in COGNITIVE_LEVELS
        assert Level.DECISIONS in COGNITIVE_LEVELS

    def test_core_levels_tuple(self):
        assert len(CORE_LEVELS) == 2
        assert Level.DOMAIN in CORE_LEVELS
        assert Level.IDENTITY in CORE_LEVELS

    def test_no_overlap(self):
        for lvl in COGNITIVE_LEVELS:
            assert lvl not in CORE_LEVELS


# ============== _AuraCompat tier methods ==============


class TestRecallCognitive:
    def test_returns_only_cognitive_records(self, brain):
        brain.store("session note", level=Level.WORKING, tags=["note"])
        brain.store("permanent fact", level=Level.DOMAIN, tags=["fact"])
        results = brain.recall_cognitive()
        all_content = " ".join(r.content for r in results)
        assert "session note" in all_content
        assert "permanent fact" not in all_content

    def test_empty_when_no_cognitive(self, brain):
        brain.store("core only", level=Level.DOMAIN, tags=["t"])
        results = brain.recall_cognitive()
        assert len(results) == 0

    def test_includes_decisions(self, brain):
        brain.store("decided X", level=Level.DECISIONS, tags=["decision"])
        results = brain.recall_cognitive()
        assert len(results) == 1
        assert "decided X" in results[0].content


class TestRecallCore:
    def test_returns_only_core_records(self, brain):
        brain.store("session note", level=Level.WORKING, tags=["note"])
        brain.store("permanent fact", level=Level.DOMAIN, tags=["fact"])
        results = brain.recall_core()
        all_content = " ".join(r.content for r in results)
        assert "permanent fact" in all_content
        assert "session note" not in all_content

    def test_includes_identity(self, brain):
        brain.store("user name is Teo", level=Level.IDENTITY, tags=["identity"])
        results = brain.recall_core()
        assert len(results) == 1
        assert "Teo" in results[0].content

    def test_empty_when_no_core(self, brain):
        brain.store("temp note", level=Level.WORKING, tags=["t"])
        results = brain.recall_core()
        assert len(results) == 0


class TestTierStats:
    def test_counts_by_tier(self, brain):
        brain.store(
            "Working memory about cooking recipes alpha",
            level=Level.WORKING,
            tags=["t"],
            deduplicate=False,
        )
        brain.store(
            "Working memory about science facts beta",
            level=Level.WORKING,
            tags=["t"],
            deduplicate=False,
        )
        brain.store(
            "Decision about project architecture gamma",
            level=Level.DECISIONS,
            tags=["t"],
            deduplicate=False,
        )
        brain.store(
            "Domain knowledge about history delta",
            level=Level.DOMAIN,
            tags=["t"],
            deduplicate=False,
        )
        brain.store(
            "Identity fact about user preferences epsilon",
            level=Level.IDENTITY,
            tags=["t"],
            deduplicate=False,
        )

        stats = brain.tier_stats()

        assert stats["cognitive"]["total"] == 3
        assert stats["cognitive"]["working"] == 2
        assert stats["cognitive"]["decisions"] == 1
        assert stats["core"]["total"] == 2
        assert stats["core"]["domain"] == 1
        assert stats["core"]["identity"] == 1
        assert stats["total"] == 5

    def test_empty_brain(self, brain):
        stats = brain.tier_stats()
        assert stats["cognitive"]["total"] == 0
        assert stats["core"]["total"] == 0
        assert stats["total"] == 0


class TestPromotionCandidates:
    def test_returns_list(self, brain):
        brain.store("note", level=Level.WORKING, tags=["t"])
        candidates = brain.promotion_candidates()
        assert isinstance(candidates, list)

    def test_low_activation_not_candidate(self, brain):
        brain.store("fresh note", level=Level.WORKING, tags=["t"])
        candidates = brain.promotion_candidates(min_activations=5, min_strength=0.7)
        assert len(candidates) == 0

    def test_core_records_excluded(self, brain):
        brain.store("domain fact", level=Level.DOMAIN, tags=["t"])
        candidates = brain.promotion_candidates(min_activations=0, min_strength=0.0)
        # Domain is core tier — should not appear in promotion candidates
        assert len(candidates) == 0
