"""
Timing runtime for Remy v3.

Owns cycle timing markers so orchestration code does not compute elapsed
durations inline.
"""

from __future__ import annotations

import time


class TimingRuntime:
    """Provide timing helpers for cycle execution."""

    def start_cycle(self) -> float:
        return time.time()

    def elapsed_ms(self, started_at: float) -> int:
        return int((time.time() - started_at) * 1000)
