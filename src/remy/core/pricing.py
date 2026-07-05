"""
Model pricing registry — two-tier JSON (defaults + user overrides).

Thread-safe singleton. Calculates USD cost from input/output token counts.
"""

import fnmatch
import json
import logging
import sys
import threading
from pathlib import Path

logger = logging.getLogger("Pricing")

if getattr(sys, "frozen", False):
    _DEFAULT_PRICING_FILE = Path(sys.executable).parent / "config" / "pricing.json"
else:
    _DEFAULT_PRICING_FILE = Path(__file__).parents[1] / "config" / "pricing.json"
_USER_PRICING_FILE_NAME = "pricing.json"


class PricingRegistry:
    """Thread-safe registry for model pricing.

    Loads config/pricing.json as defaults, merges data/pricing.json overrides.
    """

    def __init__(self, data_dir: Path | None = None):
        self._lock = threading.Lock()
        self._data_dir = data_dir
        self._defaults: dict[str, dict] = {}
        self._overrides: dict[str, dict] = {}
        self._merged: dict[str, dict] = {}
        self._load()

    def _user_pricing_path(self) -> Path:
        if self._data_dir:
            return self._data_dir / _USER_PRICING_FILE_NAME
        from remy.config.settings import settings

        return settings.DATA_DIR / _USER_PRICING_FILE_NAME

    def _load(self):
        self._defaults = self._read_file(_DEFAULT_PRICING_FILE)
        self._overrides = self._read_file(self._user_pricing_path())
        self._rebuild()

    def _read_file(self, path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("models", {})
        except Exception as e:
            logger.warning("Failed to load pricing %s: %s", path, e)
            return {}

    def _rebuild(self):
        merged = {**self._defaults, **self._overrides}
        # Filter out hidden models
        self._merged = {k: v for k, v in merged.items() if not v.get("_hidden")}

    def get_price(self, model_name: str) -> tuple[float, float]:
        """Return (input_cost_per_1m, output_cost_per_1m) for a model.

        Priority: exact match > wildcard > (0.0, 0.0).
        Looks up ALL models including hidden ones (for cost tracking).
        """
        with self._lock:
            # Check all sources (including hidden defaults) for cost tracking
            all_models = {**self._defaults, **self._overrides}
            entry = all_models.get(model_name)
            if entry and not entry.get("_hidden"):
                return (
                    entry.get("input_cost_per_1m_tokens", 0.0),
                    entry.get("output_cost_per_1m_tokens", 0.0),
                )
            # Hidden override — fall back to default pricing
            if entry and entry.get("_hidden") and model_name in self._defaults:
                default = self._defaults[model_name]
                return (
                    default.get("input_cost_per_1m_tokens", 0.0),
                    default.get("output_cost_per_1m_tokens", 0.0),
                )
            # Wildcard match
            for pattern, pentry in self._merged.items():
                if "*" in pattern or "?" in pattern:
                    if fnmatch.fnmatch(model_name, pattern):
                        return (
                            pentry.get("input_cost_per_1m_tokens", 0.0),
                            pentry.get("output_cost_per_1m_tokens", 0.0),
                        )
            return (0.0, 0.0)

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate USD cost for given model and token counts."""
        input_rate, output_rate = self.get_price(model)
        cost = (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate
        return round(cost, 8)

    def update_price(self, model: str, input_cost: float, output_cost: float):
        """Update price for a model. Saves to data/pricing.json."""
        with self._lock:
            self._overrides[model] = {
                "input_cost_per_1m_tokens": input_cost,
                "output_cost_per_1m_tokens": output_cost,
            }
            self._rebuild()
            self._save_overrides()

    def delete_price(self, model: str) -> bool:
        """Remove a model from the visible list.

        If it's a user override, deletes it.
        If it's a default model, hides it via a _hidden override.
        Returns True if model was found.
        """
        with self._lock:
            if model in self._overrides:
                if self._overrides[model].get("_hidden"):
                    return False  # already hidden
                del self._overrides[model]
                # If it also exists in defaults, hide it
                if model in self._defaults:
                    self._overrides[model] = {"_hidden": True}
                self._rebuild()
                self._save_overrides()
                return True
            elif model in self._defaults:
                self._overrides[model] = {"_hidden": True}
                self._rebuild()
                self._save_overrides()
                return True
            return False

    def get_all_prices(self) -> dict[str, dict]:
        """Return merged pricing table with source labels."""
        with self._lock:
            result = {}
            for model, entry in self._merged.items():
                result[model] = {
                    "input_cost_per_1m_tokens": entry.get("input_cost_per_1m_tokens", 0.0),
                    "output_cost_per_1m_tokens": entry.get("output_cost_per_1m_tokens", 0.0),
                    "source": "override" if model in self._overrides else "default",
                }
            return result

    def reload(self):
        """Reload pricing from disk."""
        with self._lock:
            self._load()

    def _save_overrides(self):
        from remy.core.file_utils import atomic_write

        data = {
            "_meta": {"note": "User pricing overrides"},
            "models": self._overrides,
        }
        try:
            atomic_write(self._user_pricing_path(), json.dumps(data, indent=2))
        except Exception as e:
            logger.warning("Failed to save user pricing: %s", e)


pricing_registry = PricingRegistry()
