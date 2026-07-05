"""
Lifecycle runtime for Remy v3.

Owns session start/stop hooks so the autonomy loop stays focused on scheduling
and execution rather than direct lifecycle side effects.
"""

from __future__ import annotations


class LifecycleRuntime:
    """Encapsulate autonomy session lifecycle hooks."""

    def __init__(self, chief):
        self.chief = chief

    def start_session(self):
        self.chief.load_state()
        self.chief.goal_tracker.archive_stale()
        self.chief.audit.log_event(
            "autonomy_started",
            "v3 Autonomy loop started",
            actor="system",
        )

    def stop_session(self):
        self.chief.save_state()
