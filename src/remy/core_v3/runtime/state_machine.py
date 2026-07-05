"""
Mission and task state machine for Remy v3.

Defines valid transitions and provides small helpers that keep
runtime state deterministic instead of prompt-driven.
"""

from __future__ import annotations

import logging
import time

from ..missions.mission_models import MissionStatus, TaskStatus

log = logging.getLogger(__name__)


VALID_TRANSITIONS: dict[MissionStatus, set[MissionStatus]] = {
    MissionStatus.INTAKE: {
        MissionStatus.PLANNING,
        MissionStatus.FAILED,
    },
    MissionStatus.PLANNING: {
        MissionStatus.ACTIVE,
        MissionStatus.PAUSED,
        MissionStatus.FAILED,
    },
    MissionStatus.ACTIVE: {
        MissionStatus.PAUSED,
        MissionStatus.BLOCKED,
        MissionStatus.COMPLETED,
        MissionStatus.FAILED,
        MissionStatus.ESCALATED,
    },
    MissionStatus.PAUSED: {
        MissionStatus.ACTIVE,
        MissionStatus.FAILED,
    },
    MissionStatus.BLOCKED: {
        MissionStatus.ACTIVE,
        MissionStatus.FAILED,
        MissionStatus.ESCALATED,
    },
    MissionStatus.ESCALATED: {
        MissionStatus.ACTIVE,
        MissionStatus.FAILED,
    },
    MissionStatus.COMPLETED: set(),
    MissionStatus.FAILED: set(),
}


TASK_VALID_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PENDING: {
        TaskStatus.ACTIVE,
        TaskStatus.WAITING,
        TaskStatus.BLOCKED,
        TaskStatus.BLOCKED_EXTERNAL,
        TaskStatus.BLOCKED_APPROVAL,
        TaskStatus.FAILED,
        TaskStatus.ABORTED,
        TaskStatus.SKIPPED,
    },
    TaskStatus.ACTIVE: {
        TaskStatus.RUNNING,
        TaskStatus.WAITING,
        TaskStatus.BLOCKED,
        TaskStatus.BLOCKED_EXTERNAL,
        TaskStatus.BLOCKED_APPROVAL,
        TaskStatus.FAILED,
        TaskStatus.ABORTED,
        TaskStatus.SKIPPED,
    },
    TaskStatus.RUNNING: {
        TaskStatus.COMPLETED,
        TaskStatus.PENDING,
        TaskStatus.WAITING,
        TaskStatus.BLOCKED,
        TaskStatus.BLOCKED_EXTERNAL,
        TaskStatus.BLOCKED_APPROVAL,
        TaskStatus.FAILED,
        TaskStatus.ABORTED,
    },
    TaskStatus.WAITING: {
        TaskStatus.PENDING,
        TaskStatus.ACTIVE,
        TaskStatus.BLOCKED,
        TaskStatus.BLOCKED_EXTERNAL,
        TaskStatus.BLOCKED_APPROVAL,
        TaskStatus.FAILED,
        TaskStatus.ABORTED,
    },
    TaskStatus.BLOCKED: {
        TaskStatus.PENDING,
        TaskStatus.ACTIVE,
        TaskStatus.FAILED,
        TaskStatus.ABORTED,
    },
    TaskStatus.BLOCKED_EXTERNAL: {
        TaskStatus.WAITING,
        TaskStatus.PENDING,
        TaskStatus.ACTIVE,
        TaskStatus.FAILED,
        TaskStatus.ABORTED,
    },
    TaskStatus.BLOCKED_APPROVAL: {
        TaskStatus.WAITING,
        TaskStatus.PENDING,
        TaskStatus.ACTIVE,
        TaskStatus.FAILED,
        TaskStatus.ABORTED,
    },
    TaskStatus.COMPLETED: set(),
    TaskStatus.FAILED: set(),
    TaskStatus.ABORTED: set(),
    TaskStatus.SKIPPED: set(),
}


class TransitionError(Exception):
    """Invalid state transition."""


def can_transition(from_status: MissionStatus, to_status: MissionStatus) -> bool:
    return to_status in VALID_TRANSITIONS.get(from_status, set())


def transition(mission, to_status: MissionStatus, reason: str = "") -> MissionStatus:
    from_status = mission.status
    if not can_transition(from_status, to_status):
        raise TransitionError(
            f"Cannot transition {mission.id} from {from_status.value} to {to_status.value}"
        )

    mission.status = to_status
    mission.updated_at = time.time()
    if to_status in (MissionStatus.COMPLETED, MissionStatus.FAILED):
        mission.completed_at = time.time()

    log.info(
        "Mission %s: %s -> %s (%s)",
        mission.id,
        from_status.value,
        to_status.value,
        reason or "no reason",
    )
    return to_status


def can_transition_task(from_status: TaskStatus, to_status: TaskStatus) -> bool:
    return to_status in TASK_VALID_TRANSITIONS.get(from_status, set())


def transition_task(task, to_status: TaskStatus, reason: str = "") -> TaskStatus:
    from_status = task.status
    if not can_transition_task(from_status, to_status):
        raise TransitionError(
            f"Cannot transition task {getattr(task, 'id', '<unknown>')} "
            f"from {from_status.value} to {to_status.value}"
        )

    task.status = to_status
    task.last_attempt_at = time.time()
    if to_status == TaskStatus.COMPLETED:
        task.completed_at = time.time()

    if hasattr(task, "waiting_reason"):
        task.waiting_reason = reason if to_status == TaskStatus.WAITING else ""
    if hasattr(task, "blocker_reason"):
        task.blocker_reason = (
            reason
            if to_status in (
                TaskStatus.BLOCKED,
                TaskStatus.BLOCKED_EXTERNAL,
                TaskStatus.BLOCKED_APPROVAL,
            )
            else ""
        )

    log.info(
        "Task %s: %s -> %s (%s)",
        getattr(task, "id", "<unknown>"),
        from_status.value,
        to_status.value,
        reason or "no reason",
    )
    return to_status
