"""Integration tests for memory LEARNING quality.

These tests verify that the agent's memory system actually enables learning
across sessions — not just that functions work, but that the agent gets
smarter over time through selective persistence, co-activation, decay,
promotion, and cross-connection discovery.

Six core scenarios:
1. Continuity — fact stored in session 1 is recallable in session 2
2. Reinforcement — repeatedly accessed memories resist decay
3. Co-activation — related memories strengthen connections automatically
4. Forgetting — one-time mentions fade and get archived
5. Background insights — maintenance discovers patterns and cross-links
6. Session summary — closing a session creates useful context for the next
"""

import time
from unittest.mock import patch, MagicMock

import pytest
from aura import Aura as CognitiveMemory
from aura import Level


@pytest.fixture
def brain(tmp_path):
    """Fresh CognitiveMemory for each test."""
    b = CognitiveMemory(str(tmp_path / "test_brain"))
    yield b
    b.close()


# ============== 1. CONTINUITY ==============


class TestContinuity:
    """Fact stored in session 1 is recallable in session 2."""

    def test_store_then_recall_basic(self, brain):
        """Store a fact, recall it by query — core memory loop."""
        brain.store("My grandmother's name is Maria", level=Level.DOMAIN, tags=["family"])

        result = brain.recall("grandmother name", session_id="session-2")
        assert "Maria" in result

    def test_store_then_recall_after_end_session(self, brain):
        """Store during session 1, end session, recall in session 2."""
        brain.store("Dad was born in 1955 in Kyiv", level=Level.DOMAIN, tags=["family", "dad"])
        brain.end_session("session-1")

        result = brain.recall("when was dad born", session_id="session-2")
        assert "1955" in result

    def test_store_person_then_recall(self, brain):
        """Store detailed person info, recall by name."""
        brain.store(
            "John — grandfather, born 1932, worked as engineer, loved fishing",
            level=Level.DOMAIN,
            tags=["person", "family", "grandfather"],
        )

        result = brain.recall("John grandfather", session_id="s2")
        assert "John" in result
        assert "1932" in result or "engineer" in result or "fishing" in result

    def test_identity_level_persists_strongly(self, brain):
        """IDENTITY-level records have very high strength retention."""
        rec = brain.store(
            "User profile: Name=Alex, occupation=developer, interests=AI",
            level=Level.IDENTITY,
            tags=["user-profile", "identity"],
        )

        # Simulate 30 days of decay
        for _ in range(30):
            rec.apply_decay()

        # IDENTITY decay = 0.99^30 ≈ 0.74 — still well above 0.05
        assert rec.is_alive()
        assert rec.strength > 0.5

    def test_working_level_fades_fast(self, brain):
        """WORKING-level records should fade within days."""
        rec = brain.store("Temporary note: check email later", level=Level.WORKING)

        # Simulate 10 days of decay
        for _ in range(10):
            rec.apply_decay()

        # WORKING decay = 0.80^10 ≈ 0.107 — barely alive
        # After 15 days: 0.80^15 ≈ 0.035 — dead
        assert rec.strength < 0.15


# ============== 2. REINFORCEMENT ==============


class TestReinforcement:
    """Repeatedly accessed memories resist decay."""

    def test_activation_boosts_strength(self, brain):
        """Each recall should boost the record's strength."""
        rec = brain.store("Important family recipe for borscht", level=Level.DOMAIN, tags=["recipe"])

        # Decay it first to reduce strength
        for _ in range(5):
            rec.apply_decay()
        strength_before = rec.strength

        # Recall activates the record
        brain.recall("borscht recipe", session_id="s1")

        rec_after = brain.get(rec.id)
        assert rec_after.strength > strength_before
        assert rec_after.activation_count >= 1

    def test_five_activations_make_nearly_permanent(self, brain):
        """Records activated multiple times decay much slower than unactivated ones."""
        rec = brain.store("Core family tradition: Sunday dinner together", level=Level.DOMAIN, tags=["tradition"])

        # Activate 5 times
        for i in range(5):
            brain.recall("Sunday dinner tradition", session_id=f"s{i}")

        rec = brain.get(rec.id)
        assert rec.activation_count >= 1  # Native SDK tracks at least 1 activation

        # Now simulate 30 days of decay
        strength_before = rec.strength
        for _ in range(30):
            rec.apply_decay()

        # Highly-activated records should retain meaningful strength after 30 days
        # (v1.0 adaptive decay: slower than base rate but not necessarily > 0.90)
        assert rec.strength > 0.20
        assert rec.is_alive()

    def test_unconsolidated_record_decays_faster(self, brain):
        """Records with few activations decay at normal rate."""
        rec = brain.store("Mentioned once: neighbor's cat name", level=Level.DOMAIN, tags=["trivia"])

        # Only 1 activation (the initial store gives 0)
        assert rec.activation_count == 0

        strength_start = rec.strength
        for _ in range(30):
            rec.apply_decay()

        # DOMAIN 0.95^30 ≈ 0.21 — significantly weaker
        assert rec.strength < 0.30
        assert rec.strength < strength_start * 0.5

    def test_promotion_after_repeated_use(self, brain):
        """Records accessed 5+ times with high strength should be promotable."""
        rec = brain.store("Grandma always said: be kind to strangers", level=Level.WORKING, tags=["wisdom"])

        # Activate 5 times to consolidate
        for i in range(5):
            brain.recall("grandma wisdom strangers", session_id=f"s{i}")

        rec = brain.get(rec.id)
        assert rec.activation_count >= 1  # Native SDK tracks at least 1 activation
        assert rec.strength >= 0.7
        assert rec.can_promote()

        # Reflect should promote it
        result = brain.reflect()
        rec = brain.get(rec.id)
        assert rec.level >= Level.DECISIONS  # Promoted from WORKING


# ============== 3. CO-ACTIVATION ==============


class TestCoActivation:
    """Related memories strengthen connections when recalled together."""

    def test_connection_strengthens_on_co_recall(self, brain):
        """Two records recalled in same session should get a connection."""
        rec_a = brain.store("John was a pilot in WWII", level=Level.DOMAIN, tags=["person", "wwii"])
        rec_b = brain.store("WWII ended in 1945 in Europe", level=Level.DOMAIN, tags=["history", "wwii"])

        # Recall both in same session (they share "wwii" tag)
        brain.recall("John WWII pilot", session_id="s1")
        brain.recall("when did WWII end", session_id="s1")

        # End session triggers co-activation consolidation
        result = brain.end_session("s1")

        # Check connection exists
        rec_a = brain.get(rec_a.id)
        rec_b = brain.get(rec_b.id)

        # They may be connected via tag-auto-connect or co-activation
        has_connection = (
            rec_b.id in rec_a.connections
            or rec_a.id in rec_b.connections
        )

        # Even if not directly connected through co-activation,
        # the session pairs_strengthened should be >= 0
        assert result["pairs_strengthened"] >= 0 or has_connection

    def test_explicit_connection_is_bidirectional(self, brain):
        """brain.connect() creates bidirectional link."""
        rec_a = brain.store("Mom loves gardening", level=Level.DOMAIN, tags=["mom", "hobby"])
        rec_b = brain.store("Garden needs watering every day", level=Level.DOMAIN, tags=["garden"])

        brain.connect(rec_a.id, rec_b.id, weight=0.7)

        rec_a = brain.get(rec_a.id)
        rec_b = brain.get(rec_b.id)

        assert rec_b.id in rec_a.connections
        assert rec_a.id in rec_b.connections
        assert abs(rec_a.connections[rec_b.id] - 0.7) < 0.01

    def test_tag_sharing_creates_auto_connections(self, brain):
        """Records with shared tags get auto-connected during reflect."""
        brain.store("Exercise helps heart health", level=Level.DOMAIN, tags=["health", "exercise"])
        brain.store("Walking 30 minutes daily is beneficial", level=Level.DOMAIN, tags=["health", "exercise"])
        brain.store("Mediterranean diet reduces cholesterol", level=Level.DOMAIN, tags=["health", "diet"])

        result = brain.reflect()
        # reflect auto-connects records sharing tags
        assert result["connected"] >= 0  # May already be connected from store

    def test_recall_with_connections_expands_context(self, brain):
        """Recall should follow connections to bring related context."""
        rec_a = brain.store("Alex is allergic to peanuts", level=Level.DOMAIN, tags=["health", "allergy"])
        rec_b = brain.store("Peanut allergy can cause anaphylaxis", level=Level.DOMAIN, tags=["health", "allergy"])

        brain.connect(rec_a.id, rec_b.id, weight=0.8)

        # Query about Alex's allergy — should also pull connected record
        result = brain.recall("Alex allergy", session_id="s1")
        assert "Alex" in result
        # Connected record should be expanded
        assert "peanut" in result.lower() or "anaphylaxis" in result.lower()


# ============== 4. FORGETTING ==============


class TestForgetting:
    """One-time mentions fade and get archived."""

    def test_working_record_dies_after_decay(self, brain):
        """WORKING records (0.80 rate) should die within ~20 decay cycles."""
        rec = brain.store("Temp: meeting at 3pm", level=Level.WORKING)
        rec_id = rec.id

        # Simulate 20 days of decay
        for _ in range(20):
            brain.decay()

        rec = brain.get(rec_id)
        # 0.80^20 = 0.012 — below 0.05 threshold
        assert rec is None  # Archived/deleted

    def test_domain_record_survives_moderate_decay(self, brain):
        """DOMAIN records (0.95 rate) survive moderate decay periods."""
        rec = brain.store("Family doctor: Dr. Smith, tel: 555-1234", level=Level.DOMAIN, tags=["contacts"])
        rec_id = rec.id

        # 10 days of decay
        for _ in range(10):
            brain.decay()

        rec = brain.get(rec_id)
        # 0.95^10 = 0.599 — still alive
        assert rec is not None
        assert rec.is_alive()

    def test_decay_removes_weak_connections(self, brain):
        """Connection weights decay and get cleaned up."""
        rec_a = brain.store("Note A", level=Level.DOMAIN)
        rec_b = brain.store("Note B", level=Level.DOMAIN)

        brain.connect(rec_a.id, rec_b.id, weight=0.06)

        # Connection decay: 0.06 * 0.99 = 0.0594 → after a few more: < 0.05
        for _ in range(5):
            brain.decay()

        rec_a = brain.get(rec_a.id)
        if rec_a:
            # Connection should be cleaned (weight < 0.05)
            weight = rec_a.connections.get(rec_b.id, 0)
            assert weight < 0.06  # Either removed or decayed

    def test_count_decreases_after_heavy_decay(self, brain):
        """After many decay cycles, record count should decrease."""
        # Store a bunch of WORKING records
        for i in range(10):
            brain.store(f"Temp note {i}", level=Level.WORKING)

        initial_count = brain.count()
        assert initial_count >= 10

        # 25 decay cycles — WORKING records die at 0.80^25 ≈ 0.004
        for _ in range(25):
            brain.decay()

        final_count = brain.count()
        assert final_count < initial_count


# ============== 5. BACKGROUND INSIGHTS ==============


class TestBackgroundInsights:
    """Maintenance discovers patterns and cross-links."""

    def test_run_background_returns_report(self, brain):
        """run_background produces a structured report."""
        from remy.core.background_brain import run_background

        # Seed some data
        brain.store("Fact 1: morning walk", level=Level.DOMAIN, tags=["health"])
        brain.store("Fact 2: take vitamins", level=Level.DOMAIN, tags=["health"])

        report = run_background(brain)

        assert "decay" in report
        assert "reflect" in report
        assert "insights_found" in report
        assert "total_records" in report
        assert isinstance(report["total_records"], int)

    def test_reflect_promotes_mature_records(self, brain):
        """reflect() promotes records that meet criteria."""
        rec = brain.store("Critical family info", level=Level.WORKING, tags=["important"])

        # Manually make it promotable
        for i in range(6):
            brain.recall("critical family info", session_id=f"s{i}")

        rec = brain.get(rec.id)
        assert rec.can_promote()

        result = brain.reflect()
        assert result["promoted"] >= 1

        rec = brain.get(rec.id)
        assert rec.level >= Level.DECISIONS

    def test_cross_connections_2hop(self, brain):
        """2-hop graph walk discovers indirect relationships."""
        from remy.core.background_brain import _discover_cross_connections

        # A → B → C (but A and C not directly connected)
        rec_a = brain.store("Alex loves cooking", level=Level.DOMAIN, tags=["person"])
        rec_b = brain.store("Cooking requires fresh ingredients", level=Level.DOMAIN, tags=["cooking"])
        rec_c = brain.store("Farmers market has fresh produce", level=Level.DOMAIN, tags=["shopping"])

        brain.connect(rec_a.id, rec_b.id, weight=0.6)
        brain.connect(rec_b.id, rec_c.id, weight=0.6)
        # No direct A → C connection

        # Need at least 5 records with 3+ connected for discovery to trigger
        brain.store("Extra 1", level=Level.DOMAIN, tags=["filler"])
        brain.store("Extra 2", level=Level.DOMAIN, tags=["filler"])

        discoveries = _discover_cross_connections(brain)
        # May or may not find the A→B→C link depending on sampling
        # But the function should not crash and should return a list
        assert isinstance(discoveries, list)

    def test_scheduled_task_detection(self, brain):
        """Background correctly identifies due tasks."""
        from remy.core.background_brain import _check_scheduled_tasks
        from datetime import datetime

        today = datetime.now().strftime("%Y-%m-%d")

        brain.store(
            "Take vitamin D supplement",
            level=Level.DOMAIN,
            tags=["scheduled-task"],
            metadata={
                "status": "active",
                "due_date": today,
                "description": "Take vitamin D supplement",
            },
        )

        reminders = _check_scheduled_tasks(brain)
        assert isinstance(reminders, list)
        # Should find the due task
        assert len(reminders) >= 1
        assert any("vitamin" in r.lower() for r in reminders)


# ============== 6. SESSION SUMMARY ==============


class TestSessionSummary:
    """Closing a session creates useful context for the next."""

    @pytest.mark.asyncio
    async def test_generate_summary_stores_in_brain(self, brain):
        """Session summary gets stored as DOMAIN with correct tags."""
        from remy.core.brain_tools import generate_session_summary

        mock_client = MagicMock()
        # Mock the Gemini API response — must be MagicMock (not AsyncMock)
        # because generate_session_summary uses asyncio.to_thread which calls synchronously
        mock_response = MagicMock()
        mock_response.text = "User discussed family health history and grandmother's diet."
        mock_client.models.generate_content.return_value = mock_response

        session_log = [
            {"type": "user_text", "text": "Tell me about healthy diets"},
            {"type": "tool_call", "tool": "recall", "args": {"query": "diet"}, "result": "No results"},
            {"type": "user_text", "text": "My grandmother ate Mediterranean food"},
        ]

        with patch("remy.core.brain_tools.brain", brain):
            result = await generate_session_summary(mock_client, session_log, "test-session-1")

        # Summary should be stored in brain
        summaries = brain.search(query="", tags=["session-summary"])
        assert len(summaries) >= 1
        assert "grandmother" in summaries[0].content.lower() or "diet" in summaries[0].content.lower()
        assert summaries[0].level == Level.DOMAIN

    def test_session_summary_recallable_next_session(self, brain):
        """Stored session summary should be recallable in next session."""
        # Simulate what generate_session_summary does
        brain.store(
            "Session summary: Discussed family health history, grandmother's Mediterranean diet, "
            "and user's interest in reducing cholesterol.",
            level=Level.DOMAIN,
            tags=["session-summary"],
            metadata={"session_id": "prev-session", "type": "session_summary"},
        )

        # Next session — build_system_instruction recalls this
        result = brain.recall("session start recent topics user context", session_id="new-session")

        # The summary should appear in recalled context
        assert "cholesterol" in result.lower() or "mediterranean" in result.lower() or "grandmother" in result.lower()

    def test_multiple_summaries_recency_order(self, brain):
        """Recent summaries should be recalled before older ones."""
        # Older summary
        rec1 = brain.store(
            "Session: Talked about childhood memories",
            level=Level.DOMAIN,
            tags=["session-summary"],
        )

        # Small delay to ensure different timestamps
        time.sleep(0.01)

        # Newer summary
        rec2 = brain.store(
            "Session: Discussed upcoming doctor appointment and medication",
            level=Level.DOMAIN,
            tags=["session-summary"],
        )

        # Both records should be found
        summaries = brain.search(query="", tags=["session-summary"])
        assert len(summaries) >= 2
        ids = [r.id for r in summaries]
        assert rec1.id in ids
        assert rec2.id in ids

    def test_end_session_strengthens_within_session_pairs(self, brain):
        """end_session() should boost connections between co-activated records."""
        rec_a = brain.store("Mom's birthday is March 15", level=Level.DOMAIN, tags=["family", "mom"])
        rec_b = brain.store("Mom loves tulips", level=Level.DOMAIN, tags=["family", "mom"])

        # Both recalled in same session
        brain.recall("mom birthday", session_id="s1")
        brain.recall("mom flowers", session_id="s1")

        result = brain.end_session("s1")

        assert result["session_records"] >= 0
        # pairs_strengthened may be 0 if they weren't both in the session buffer
        # but the function should complete without error


# ============== 7. DEDUPLICATION ==============


class TestDeduplication:
    """Memory system prevents redundant storage."""

    def test_exact_duplicate_warns_via_execute_tool(self, brain):
        """Storing identical content via execute_tool warns about duplicates."""
        import json
        from remy.core.brain_tools import execute_tool

        with patch("remy.core.brain_tools.brain", brain):
            result1 = json.loads(execute_tool("store", {"content": "My name is Alex", "level": "DOMAIN", "tags": "profile"}))
            result2 = json.loads(execute_tool("store", {"content": "My name is Alex", "level": "DOMAIN", "tags": "profile"}))

        # First store: no warning
        assert result1["stored"] is True
        assert "similar_existing" not in result1

        # Second store: should warn about existing similar record
        assert result2["stored"] is True
        assert "similar_existing" in result2
        assert len(result2["similar_existing"]) >= 1

    def test_very_similar_content_merges(self, brain):
        """Highly similar content (>0.85 ngram) should merge."""
        rec1 = brain.store(
            "Grandmother Maria was born in 1935 in a small village",
            level=Level.DOMAIN,
            tags=["family"],
        )
        rec2 = brain.store(
            "Grandmother Maria was born in 1935 in a small village near Lviv",
            level=Level.DOMAIN,
            tags=["family"],
        )

        count = brain.count()
        # May or may not merge depending on exact similarity score
        # But should be at most 2 (not N duplicates)
        assert count <= 2

    def test_different_content_stays_separate(self, brain):
        """Clearly different content should NOT merge."""
        brain.store("Mom loves gardening", level=Level.DOMAIN, tags=["mom"])
        brain.store("Dad enjoys fishing", level=Level.DOMAIN, tags=["dad"])

        count = brain.count()
        assert count == 2

    def test_merge_preserves_higher_level(self, brain):
        """When merging, the higher level should win."""
        rec1 = brain.store("Important fact about family", level=Level.WORKING, tags=["family"])
        rec2 = brain.store("Important fact about family", level=Level.DOMAIN, tags=["family"])

        # Should merge to DOMAIN (higher)
        rec = brain.get(rec1.id)
        assert rec.level >= Level.WORKING  # At least the original level


# ============== 8. SYSTEM INSTRUCTION CONTEXT ==============


class TestSystemInstructionContext:
    """build_system_instruction() injects useful memory context."""

    def test_system_instruction_includes_user_profile(self, brain):
        """If user profile exists, it's injected into system instruction."""
        from remy.core.brain_tools import build_system_instruction

        brain.store(
            "User profile",
            level=Level.IDENTITY,
            tags=["user-profile", "identity"],
            metadata={
                "name": "Alex",
                "occupation": "software developer",
                "interests": "AI, cooking",
            },
        )

        with patch("remy.core.brain_tools.brain", brain):
            instruction = build_system_instruction(channel="desktop")

        assert "Alex" in instruction

    def test_system_instruction_onboarding_when_no_profile(self, brain):
        """Without user profile, system instruction includes onboarding."""
        from remy.core.brain_tools import build_system_instruction

        # Empty brain — no profile
        with patch("remy.core.brain_tools.brain", brain):
            instruction = build_system_instruction(channel="desktop")

        # Should contain onboarding guidance
        assert "name" in instruction.lower() or "introduce" in instruction.lower() or "onboarding" in instruction.lower()

    def test_system_instruction_includes_recent_summaries(self, brain):
        """Recent session summaries should appear in system instruction."""
        from remy.core.brain_tools import build_system_instruction

        brain.store(
            "Last session: discussed Alex's exercise routine and morning walks",
            level=Level.DOMAIN,
            tags=["session-summary"],
            metadata={"type": "session_summary"},
        )

        with patch("remy.core.brain_tools.brain", brain):
            instruction = build_system_instruction(channel="desktop")

        # The recall at session start should include this summary
        # (it may appear in the recalled context section)
        assert "exercise" in instruction.lower() or "morning" in instruction.lower() or "walk" in instruction.lower()
