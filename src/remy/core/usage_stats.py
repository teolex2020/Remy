"""
Token usage tracking module.
Persists global token counters for user vs. autonomous agent usage.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from remy.config.settings import settings

logger = logging.getLogger("UsageStats")

STATS_FILE = "token_usage.json"

@dataclass
class UsageStatsData:
    user_tokens: int = 0
    autonomy_tokens: int = 0
    user_tokens_reported: int = 0
    autonomy_tokens_reported: int = 0
    autonomy_tokens_estimated: int = 0
    user_cost_usd: float = 0.0
    autonomy_cost_usd: float = 0.0
    last_updated: float = 0.0

class UsageTracker:
    """Thread-safe tracker for token usage."""
    
    def __init__(self):
        self.path = settings.DATA_DIR / STATS_FILE
        self._lock = threading.Lock()
        self.data = self._load()
        self.session_started_at = time.time()
        self.session_user_tokens = 0
        self.session_autonomy_tokens = 0
        self.session_user_tokens_reported = 0
        self.session_autonomy_tokens_reported = 0
        self.session_autonomy_tokens_estimated = 0

    def _load(self) -> UsageStatsData:
        if not self.path.exists():
            return UsageStatsData()
        try:
            content = self.path.read_text(encoding="utf-8")
            if not content.strip():
                return UsageStatsData()
            data = json.loads(content)
            return UsageStatsData(
                user_tokens=data.get("user_tokens", 0),
                autonomy_tokens=data.get("autonomy_tokens", 0),
                user_tokens_reported=data.get("user_tokens_reported", data.get("user_tokens", 0)),
                autonomy_tokens_reported=data.get("autonomy_tokens_reported", 0),
                autonomy_tokens_estimated=data.get("autonomy_tokens_estimated", data.get("autonomy_tokens", 0)),
                user_cost_usd=data.get("user_cost_usd", 0.0),
                autonomy_cost_usd=data.get("autonomy_cost_usd", 0.0),
                last_updated=data.get("last_updated", 0.0)
            )
        except Exception as e:
            logger.warning(f"Failed to load usage stats: {e}")
            return UsageStatsData()

    def _save(self):
        try:
            data = asdict(self.data)
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to save usage stats: {e}")

    def record_usage(self, source: str, tokens: int, kind: str = "reported"):
        """
        Record token usage.
        source: 'user' or 'autonomy'
        kind: 'reported' or 'estimated'
        """
        if tokens <= 0:
            return
            
        with self._lock:
            if source == "user":
                self.data.user_tokens += tokens
                self.session_user_tokens += tokens
                if kind == "reported":
                    self.data.user_tokens_reported += tokens
                    self.session_user_tokens_reported += tokens
            elif source == "autonomy":
                self.data.autonomy_tokens += tokens
                self.session_autonomy_tokens += tokens
                if kind == "reported":
                    self.data.autonomy_tokens_reported += tokens
                    self.session_autonomy_tokens_reported += tokens
                else:
                    self.data.autonomy_tokens_estimated += tokens
                    self.session_autonomy_tokens_estimated += tokens
            
            self.data.last_updated = time.time()
            self._save()

    def record_usage_with_cost(self, source: str, tokens: int, cost_usd: float, kind: str = "reported"):
        """Record token usage with associated USD cost."""
        self.record_usage(source, tokens, kind)
        with self._lock:
            if source == "user":
                self.data.user_cost_usd += cost_usd
            elif source == "autonomy":
                self.data.autonomy_cost_usd += cost_usd
            self._save()

    def get_stats(self) -> dict:
        with self._lock:
            payload = asdict(self.data)
            payload["total_tokens"] = self.data.user_tokens + self.data.autonomy_tokens
            payload["lifetime"] = {
                "user_tokens": self.data.user_tokens,
                "autonomy_tokens": self.data.autonomy_tokens,
                "total_tokens": self.data.user_tokens + self.data.autonomy_tokens,
                "user_tokens_reported": self.data.user_tokens_reported,
                "autonomy_tokens_reported": self.data.autonomy_tokens_reported,
                "autonomy_tokens_estimated": self.data.autonomy_tokens_estimated,
            }
            payload["session"] = {
                "started_at": self.session_started_at,
                "user_tokens": self.session_user_tokens,
                "autonomy_tokens": self.session_autonomy_tokens,
                "total_tokens": self.session_user_tokens + self.session_autonomy_tokens,
                "user_tokens_reported": self.session_user_tokens_reported,
                "autonomy_tokens_reported": self.session_autonomy_tokens_reported,
                "autonomy_tokens_estimated": self.session_autonomy_tokens_estimated,
            }
            return payload

# Singleton instance
usage_tracker = UsageTracker()
