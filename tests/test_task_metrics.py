"""Tests for live task metrics tracking."""

from remy.core.task_metrics import (
    CycleOutcome,
    FamilyMetrics,
    TaskMetricsTracker,
    resolve_family,
)

# ============== FamilyMetrics ==============


class TestFamilyMetrics:
    def test_empty_metrics(self):
        fm = FamilyMetrics()
        assert fm.completion_rate == 0.0
        assert fm.blocked_rate == 0.0
        assert fm.retry_rate == 0.0
        assert fm.avg_duration_ms == 0.0

    def test_completion_rate(self):
        fm = FamilyMetrics(total_cycles=10, successes=7)
        assert fm.completion_rate == 0.7

    def test_blocked_rate(self):
        fm = FamilyMetrics(total_cycles=10, blocked_external=3)
        assert fm.blocked_rate == 0.3

    def test_retry_rate(self):
        fm = FamilyMetrics(total_cycles=10, failures=4)
        assert fm.retry_rate == 0.4

    def test_avg_duration(self):
        fm = FamilyMetrics(total_cycles=5, total_duration_ms=10000)
        assert fm.avg_duration_ms == 2000.0

    def test_to_summary(self):
        fm = FamilyMetrics(total_cycles=10, successes=6, failures=2, blocked_external=2)
        s = fm.to_summary()
        assert s["total_cycles"] == 10
        assert s["completion_rate"] == 0.6
        assert s["blocked_rate"] == 0.2
        assert s["retry_rate"] == 0.2


# ============== resolve_family ==============


class TestResolveFamily:
    def test_from_template(self):
        assert resolve_family({"goal_template": "signup_operator"}) == "signup_operator"

    def test_from_template_publisher(self):
        assert resolve_family({"goal_template": "publisher"}) == "publisher"

    def test_from_template_market_research(self):
        assert resolve_family({"goal_template": "market_research"}) == "market_research"

    def test_from_browser_worker_fallback(self):
        # Pack resolution is the single source of truth; worker param is vestigial
        assert resolve_family({"goal_template": "unknown"}, worker="browser_worker") == "general"

    def test_from_research_worker_fallback(self):
        assert resolve_family({"goal_template": "unknown"}, worker="research_worker") == "general"

    def test_inferred_from_description_signup(self):
        assert resolve_family({"description": "sign up on dev.to"}) == "signup_operator"

    def test_inferred_from_description_research(self):
        assert (
            resolve_family({"description": "competitive analysis of rivals"}) == "market_research"
        )

    def test_inferred_from_description_monitoring(self):
        assert resolve_family({"description": "monitor competitor pricing page"}) == "monitoring"

    def test_none_goal(self):
        assert resolve_family(None) == "general"

    def test_unknown_template(self):
        assert resolve_family({"goal_template": "custom_thing"}) == "general"


# ============== CycleOutcome ==============


class TestCycleOutcome:
    def test_basic(self):
        o = CycleOutcome(family="signup_operator", success=True, duration_ms=1500, tokens_used=500)
        assert o.family == "signup_operator"
        assert o.success is True
        assert o.blocked_external is False

    def test_blocked(self):
        o = CycleOutcome(family="signup_operator", success=False, blocked_external=True)
        assert o.blocked_external is True


# ============== TaskMetricsTracker ==============


class TestTaskMetricsTracker:
    def _make_tracker(self, tmp_path):
        return TaskMetricsTracker(path=tmp_path)

    def test_record_success(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.record(
            CycleOutcome(family="signup_operator", success=True, duration_ms=2000, tokens_used=100)
        )
        stats = tracker.get_family("signup_operator")
        assert stats["total_cycles"] == 1
        assert stats["successes"] == 1
        assert stats["completion_rate"] == 1.0

    def test_record_failure(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.record(
            CycleOutcome(family="publisher", success=False, duration_ms=3000, tokens_used=200)
        )
        stats = tracker.get_family("publisher")
        assert stats["total_cycles"] == 1
        assert stats["failures"] == 1
        assert stats["completion_rate"] == 0.0

    def test_record_blocked(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.record(CycleOutcome(family="signup_operator", success=False, blocked_external=True))
        stats = tracker.get_family("signup_operator")
        assert stats["blocked_external"] == 1
        assert stats["failures"] == 0

    def test_record_timeout(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.record(CycleOutcome(family="market_research", success=False, timeout=True))
        stats = tracker.get_family("market_research")
        assert stats["timeouts"] == 1

    def test_record_zero_tool(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.record(CycleOutcome(family="general", success=False, zero_tool=True))
        stats = tracker.get_family("general")
        assert stats["zero_tool_cycles"] == 1

    def test_multiple_records(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.record(
            CycleOutcome(family="signup_operator", success=True, duration_ms=1000, tokens_used=100)
        )
        tracker.record(
            CycleOutcome(family="signup_operator", success=False, duration_ms=2000, tokens_used=200)
        )
        tracker.record(
            CycleOutcome(family="signup_operator", success=True, duration_ms=1500, tokens_used=150)
        )
        stats = tracker.get_family("signup_operator")
        assert stats["total_cycles"] == 3
        assert stats["successes"] == 2
        assert stats["failures"] == 1
        assert stats["completion_rate"] == round(2 / 3, 3)

    def test_get_all(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.record(CycleOutcome(family="signup_operator", success=True))
        tracker.record(CycleOutcome(family="market_research", success=False))
        tracker.record(CycleOutcome(family="market_research", success=True))
        result = tracker.get_all()
        assert "signup_operator" in result["families"]
        assert "market_research" in result["families"]
        assert result["totals"]["total_cycles"] == 3
        assert result["totals"]["successes"] == 2

    def test_unknown_family_returns_empty(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        stats = tracker.get_family("nonexistent")
        assert stats["total_cycles"] == 0

    def test_unknown_family_mapped_to_general(self, tmp_path):
        tracker = self._make_tracker(tmp_path)
        tracker.record(CycleOutcome(family="custom_thing", success=True))
        stats = tracker.get_family("general")
        assert stats["total_cycles"] == 1

    def test_persistence(self, tmp_path):
        tracker1 = self._make_tracker(tmp_path)
        for _ in range(5):
            tracker1.record(
                CycleOutcome(family="publisher", success=True, duration_ms=1000, tokens_used=50)
            )
        tracker1.flush()

        # New tracker instance should load from disk
        tracker2 = self._make_tracker(tmp_path)
        stats = tracker2.get_family("publisher")
        assert stats["total_cycles"] == 5
        assert stats["successes"] == 5
