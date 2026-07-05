"""
Tests for PricingRegistry, UsageTracker cost extensions, and ResourceBudget cost limits.
"""

import json
import time

import pytest

from remy.core.pricing import PricingRegistry

# ============== PricingRegistry ==============


class TestPricingRegistry:
    """Tests for PricingRegistry (isolated with tmp_path)."""

    def _make_registry(self, tmp_path, defaults=None, overrides=None):
        """Create a PricingRegistry with custom config and data dirs."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        if defaults:
            (config_dir / "pricing.json").write_text(
                json.dumps({"models": defaults}), encoding="utf-8"
            )

        if overrides:
            (data_dir / "pricing.json").write_text(
                json.dumps({"models": overrides}), encoding="utf-8"
            )

        # Patch the default file path and create registry
        import remy.core.pricing as mod

        orig = mod._DEFAULT_PRICING_FILE
        mod._DEFAULT_PRICING_FILE = config_dir / "pricing.json"
        try:
            registry = PricingRegistry(data_dir=data_dir)
        finally:
            mod._DEFAULT_PRICING_FILE = orig
        return registry

    def test_exact_match(self, tmp_path):
        reg = self._make_registry(
            tmp_path,
            defaults={
                "gemini-2.0-flash": {
                    "input_cost_per_1m_tokens": 0.10,
                    "output_cost_per_1m_tokens": 0.40,
                }
            },
        )
        assert reg.get_price("gemini-2.0-flash") == (0.10, 0.40)

    def test_wildcard_match(self, tmp_path):
        reg = self._make_registry(
            tmp_path,
            defaults={
                "ollama/*": {"input_cost_per_1m_tokens": 0.0, "output_cost_per_1m_tokens": 0.0}
            },
        )
        assert reg.get_price("ollama/llama3") == (0.0, 0.0)

    def test_unknown_model_returns_zero(self, tmp_path):
        reg = self._make_registry(
            tmp_path,
            defaults={
                "gemini-2.0-flash": {
                    "input_cost_per_1m_tokens": 0.10,
                    "output_cost_per_1m_tokens": 0.40,
                }
            },
        )
        assert reg.get_price("unknown-model") == (0.0, 0.0)

    def test_exact_beats_wildcard(self, tmp_path):
        reg = self._make_registry(
            tmp_path,
            defaults={
                "ollama/*": {"input_cost_per_1m_tokens": 0.0, "output_cost_per_1m_tokens": 0.0},
                "ollama/special": {
                    "input_cost_per_1m_tokens": 1.0,
                    "output_cost_per_1m_tokens": 2.0,
                },
            },
        )
        assert reg.get_price("ollama/special") == (1.0, 2.0)

    def test_calculate_cost(self, tmp_path):
        reg = self._make_registry(
            tmp_path,
            defaults={
                "model-a": {"input_cost_per_1m_tokens": 1.0, "output_cost_per_1m_tokens": 2.0}
            },
        )
        cost = reg.calculate_cost("model-a", 1_000_000, 500_000)
        assert cost == pytest.approx(2.0)  # 1.0 + 0.5*2.0

    def test_calculate_cost_zero_tokens(self, tmp_path):
        reg = self._make_registry(
            tmp_path,
            defaults={
                "model-a": {"input_cost_per_1m_tokens": 1.0, "output_cost_per_1m_tokens": 2.0}
            },
        )
        assert reg.calculate_cost("model-a", 0, 0) == 0.0

    def test_calculate_cost_unknown_model(self, tmp_path):
        reg = self._make_registry(tmp_path, defaults={})
        assert reg.calculate_cost("no-such-model", 1000, 1000) == 0.0

    def test_override_takes_precedence(self, tmp_path):
        reg = self._make_registry(
            tmp_path,
            defaults={
                "model-a": {"input_cost_per_1m_tokens": 1.0, "output_cost_per_1m_tokens": 2.0}
            },
            overrides={
                "model-a": {"input_cost_per_1m_tokens": 5.0, "output_cost_per_1m_tokens": 10.0}
            },
        )
        assert reg.get_price("model-a") == (5.0, 10.0)

    def test_update_price_creates_user_file(self, tmp_path):
        reg = self._make_registry(
            tmp_path,
            defaults={
                "model-a": {"input_cost_per_1m_tokens": 1.0, "output_cost_per_1m_tokens": 2.0}
            },
        )
        reg.update_price("model-b", 3.0, 6.0)
        assert reg.get_price("model-b") == (3.0, 6.0)

        # Verify file was written
        user_file = tmp_path / "data" / "pricing.json"
        assert user_file.exists()
        data = json.loads(user_file.read_text(encoding="utf-8"))
        assert "model-b" in data["models"]

    def test_delete_price(self, tmp_path):
        reg = self._make_registry(
            tmp_path,
            defaults={
                "model-a": {"input_cost_per_1m_tokens": 1.0, "output_cost_per_1m_tokens": 2.0}
            },
            overrides={
                "model-a": {"input_cost_per_1m_tokens": 99.0, "output_cost_per_1m_tokens": 99.0}
            },
        )
        assert reg.get_price("model-a") == (99.0, 99.0)
        assert reg.delete_price("model-a") is True
        # Falls back to default
        assert reg.get_price("model-a") == (1.0, 2.0)

    def test_delete_nonexistent(self, tmp_path):
        reg = self._make_registry(tmp_path, defaults={})
        assert reg.delete_price("no-such") is False

    def test_get_all_prices_source_labels(self, tmp_path):
        reg = self._make_registry(
            tmp_path,
            defaults={
                "model-a": {"input_cost_per_1m_tokens": 1.0, "output_cost_per_1m_tokens": 2.0}
            },
            overrides={
                "model-b": {"input_cost_per_1m_tokens": 3.0, "output_cost_per_1m_tokens": 6.0}
            },
        )
        all_prices = reg.get_all_prices()
        assert all_prices["model-a"]["source"] == "default"
        assert all_prices["model-b"]["source"] == "override"

    def test_missing_default_file(self, tmp_path):
        # No defaults at all
        reg = self._make_registry(tmp_path, defaults=None)
        assert reg.get_price("anything") == (0.0, 0.0)

    def test_corrupted_json(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "pricing.json").write_text("NOT JSON", encoding="utf-8")

        import remy.core.pricing as mod

        orig = mod._DEFAULT_PRICING_FILE
        mod._DEFAULT_PRICING_FILE = config_dir / "pricing.json"
        try:
            reg = PricingRegistry(data_dir=tmp_path)
        finally:
            mod._DEFAULT_PRICING_FILE = orig
        assert reg.get_price("anything") == (0.0, 0.0)

    def test_reload(self, tmp_path):
        reg = self._make_registry(
            tmp_path,
            defaults={
                "model-a": {"input_cost_per_1m_tokens": 1.0, "output_cost_per_1m_tokens": 2.0}
            },
        )
        assert reg.get_price("model-a") == (1.0, 2.0)

        # Manually edit user file
        user_file = tmp_path / "data" / "pricing.json"
        user_file.write_text(
            json.dumps(
                {
                    "models": {
                        "model-a": {
                            "input_cost_per_1m_tokens": 99.0,
                            "output_cost_per_1m_tokens": 99.0,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        reg.reload()
        assert reg.get_price("model-a") == (99.0, 99.0)


# ============== UsageTracker cost extension ==============


class TestUsageTrackerCost:
    def test_record_usage_with_cost(self, tmp_path):
        from remy.core.usage_stats import UsageTracker

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("remy.core.usage_stats.settings.DATA_DIR", tmp_path)
            tracker = UsageTracker()
            tracker.record_usage_with_cost("user", 100, 0.005)
            tracker.record_usage_with_cost("autonomy", 200, 0.010)
            stats = tracker.get_stats()
            assert stats["user_tokens"] == 100
            assert stats["user_cost_usd"] == pytest.approx(0.005)
            assert stats["autonomy_tokens"] == 200
            assert stats["autonomy_cost_usd"] == pytest.approx(0.010)

    def test_backward_compat_record_usage(self, tmp_path):
        from remy.core.usage_stats import UsageTracker

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("remy.core.usage_stats.settings.DATA_DIR", tmp_path)
            tracker = UsageTracker()
            tracker.record_usage("user", 100)
            stats = tracker.get_stats()
            assert stats["user_tokens"] == 100
            assert stats["user_cost_usd"] == 0.0

    def test_load_old_format(self, tmp_path):
        # Old format without cost fields
        (tmp_path / "token_usage.json").write_text(
            json.dumps({"user_tokens": 500, "autonomy_tokens": 200, "last_updated": 0.0}),
            encoding="utf-8",
        )
        from remy.core.usage_stats import UsageTracker

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("remy.core.usage_stats.settings.DATA_DIR", tmp_path)
            tracker = UsageTracker()
            stats = tracker.get_stats()
            assert stats["user_tokens"] == 500
            assert stats["user_cost_usd"] == 0.0
            assert stats["autonomy_cost_usd"] == 0.0

    def test_cost_persistence(self, tmp_path):
        from remy.core.usage_stats import UsageTracker

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("remy.core.usage_stats.settings.DATA_DIR", tmp_path)
            tracker = UsageTracker()
            tracker.record_usage_with_cost("user", 100, 0.123)

            # Load fresh
            tracker2 = UsageTracker()
            stats = tracker2.get_stats()
            assert stats["user_cost_usd"] == pytest.approx(0.123)


# ============== ResourceBudget cost extension ==============


class TestResourceBudgetCost:
    def test_can_spend_cost_limit(self):
        from remy.core.autonomy_models import ResourceBudget

        budget = ResourceBudget(
            daily_limit=100_000,
            hourly_limit=20_000,
            session_limit=500_000,
            daily_cost_limit_usd=0.50,
        )
        budget.cost_today_usd = 0.50
        can, msg = budget.can_spend(100)
        assert can is False
        assert "cost limit" in msg.lower()

    def test_can_spend_no_cost_limit(self):
        from remy.core.autonomy_models import ResourceBudget

        budget = ResourceBudget(
            daily_limit=100_000,
            hourly_limit=20_000,
            session_limit=500_000,
            daily_cost_limit_usd=0.0,  # disabled
        )
        budget.cost_today_usd = 999.0  # huge cost but no limit
        can, _ = budget.can_spend(100)
        assert can is True

    def test_record_usage_with_cost(self):
        from remy.core.autonomy_models import ResourceBudget

        budget = ResourceBudget(
            daily_limit=100_000,
            hourly_limit=20_000,
            session_limit=500_000,
        )
        budget.record_usage(1000, cost_usd=0.05)
        assert budget.tokens_today == 1000
        assert budget.cost_today_usd == pytest.approx(0.05)
        assert budget.cost_this_session_usd == pytest.approx(0.05)
        assert budget.total_cost_lifetime_usd == pytest.approx(0.05)

    def test_daily_reset_clears_cost(self):
        from remy.core.autonomy_models import ResourceBudget

        budget = ResourceBudget(
            daily_limit=100_000,
            hourly_limit=20_000,
            session_limit=500_000,
            daily_cost_limit_usd=1.0,
        )
        budget.cost_today_usd = 0.99
        budget.last_day_reset = time.time() - 90_000  # >24h ago
        can, _ = budget.can_spend(100)
        # Daily reset should have cleared cost_today_usd
        assert budget.cost_today_usd == 0.0
        assert can is True

    def test_to_dict_includes_cost(self):
        from remy.core.autonomy_models import ResourceBudget

        budget = ResourceBudget(
            daily_limit=100_000,
            hourly_limit=20_000,
            session_limit=500_000,
            daily_cost_limit_usd=2.0,
        )
        budget.record_usage(500, cost_usd=0.01)
        d = budget.to_dict()
        assert "daily_cost_limit_usd" in d
        assert "cost_today_usd" in d
        assert "total_cost_lifetime_usd" in d
        assert d["cost_today_usd"] == pytest.approx(0.01)

    def test_backward_compat_record_usage(self):
        from remy.core.autonomy_models import ResourceBudget

        budget = ResourceBudget(
            daily_limit=100_000,
            hourly_limit=20_000,
            session_limit=500_000,
        )
        # Old-style call without cost
        budget.record_usage(1000)
        assert budget.tokens_today == 1000
        assert budget.cost_today_usd == 0.0
