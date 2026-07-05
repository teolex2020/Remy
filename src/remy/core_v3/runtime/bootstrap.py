"""
Bootstrap facade for Remy v3.

Public API remains dict-shaped for compatibility, while actual service graph
construction lives in RuntimeContainer.
"""

from __future__ import annotations

import logging
from typing import Any

from .runtime_container import RuntimeContainer

log = logging.getLogger(__name__)


def create_v3_runtime():
    """Create and wire all v3 components."""
    container = RuntimeContainer.build()
    log.info("Remy v3 runtime initialized (Phase 8 — all components)")
    return container.as_dict()


def load_v2_state(runtime: dict[str, Any] | RuntimeContainer):
    """Load existing v2 state into v3 runtime."""
    runtime_dict = runtime.as_dict() if isinstance(runtime, RuntimeContainer) else runtime
    store = runtime_dict["store"]
    chief = runtime_dict["chief"]

    chief.load_state()
    if chief.all_missions():
        log.info("Resumed from v3 persisted state: %d missions", len(chief.all_missions()))
        runtime_dict["budget"].sync_from_v2()
        return chief.all_missions(), []

    missions = store.load_from_missions_json()
    for mission in missions:
        chief.accept_mission(mission)
        for task in store.mission_tasks(mission.id):
            chief.add_task(task)

    goals = store.load_goals_from_brain()
    for goal in goals:
        chief.add_goal(goal)

    runtime_dict["budget"].sync_from_v2()
    chief.save_state()

    log.info("V2 state loaded: %d missions, %d goals", len(missions), len(goals))
    return missions, goals
