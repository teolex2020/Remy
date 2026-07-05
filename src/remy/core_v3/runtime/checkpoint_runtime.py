"""
Checkpoint Runtime for Remy v3.

Owns cycle checkpoint policy for early-return paths so the chief agent does not
decide inline when runtime state must be persisted.
"""

from __future__ import annotations


class CheckpointRuntime:
    """Persist runtime state at explicit checkpoint boundaries."""

    def early_exit(self, chief) -> None:
        chief.save_state()

    def complete_cycle(self, chief, *, plan) -> None:
        if chief.persistence_runtime is not None:
            chief.persistence_runtime.save_plan(plan)
        else:
            chief.plan_persistence.save(plan)
        chief.save_state()
