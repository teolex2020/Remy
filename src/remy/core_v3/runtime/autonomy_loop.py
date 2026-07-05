"""
Autonomy Loop for Remy v3.

The main asyncio loop that drives autonomous mission execution.
Phase 2: Full lifecycle with maintenance phases, failure guards,
persistence, and v2 survival integration.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..agents.chief_agent import ChiefAgent, ChiefDecision, CycleResult
from .error_runtime import ErrorRuntime
from .guard_runtime import GuardRuntime
from .lifecycle_runtime import LifecycleRuntime
from .loop_runtime import LoopRuntime
from .maintenance_runtime import MaintenanceRuntime
from .scheduler_runtime import SchedulerRuntime

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loop phases (mirrors v2 tier structure)
# ---------------------------------------------------------------------------

class LoopPhase:
    MAINTENANCE = "maintenance"     # Background brain, survival check
    BUDGET_CHECK = "budget_check"
    GOAL_ARCHIVAL = "goal_archival"
    MISSION_EXECUTION = "mission_execution"
    STATE_PERSIST = "state_persist"


class AutonomyLoop:
    """Main autonomous execution loop.

    Cycle structure (mirrors v2 but structured):
    1. Maintenance (background brain, survival check) — every N cycles
    2. Budget check
    3. Failure guard
    4. Mission selection + Chief Agent cycle
    5. State persistence
    6. Sleep
    """

    def __init__(
        self,
        chief: ChiefAgent | None = None,
        cycle_interval_sec: float = 30.0,
        maintenance_interval_cycles: int = 10,
        max_consecutive_failures: int = 3,
        max_session_minutes: int = 480,
    ):
        self.chief = chief or ChiefAgent()
        self.cycle_interval_sec = cycle_interval_sec
        self.maintenance_interval = maintenance_interval_cycles
        self.max_consecutive_failures = max_consecutive_failures
        self.max_session_sec = max_session_minutes * 60

        import uuid
        self.session_id = f"v3-{uuid.uuid4().hex[:8]}"
        self._running = False
        self._cycle_count = 0
        self._consecutive_failures = 0
        self._maintenance_only = False
        self._session_start: float = 0.0
        self._last_maintenance: int = 0
        self._last_selection = None
        self._last_result = None
        self._last_scheduler_reason = ""
        self.scheduler_runtime = SchedulerRuntime(
            self.chief.mission_query_runtime,
            self.chief.projection_runtime,
            loop_runtime=None,
            evaluator=self.chief.evaluator,
        )
        self.loop_runtime = LoopRuntime(self.chief)
        self.scheduler_runtime.loop_runtime = self.loop_runtime
        self.maintenance_runtime = MaintenanceRuntime(self.chief)
        self.guard_runtime = GuardRuntime(self.chief)
        self.lifecycle_runtime = LifecycleRuntime(self.chief)
        self.error_runtime = ErrorRuntime()
        if getattr(self.chief, "ops_query_runtime", None) is not None:
            self.chief.ops_query_runtime.bind_autonomy(
                loop_runtime=self.loop_runtime,
                scheduler_runtime=self.scheduler_runtime,
                mission_query_runtime=self.chief.mission_query_runtime,
                projection_runtime=self.chief.projection_runtime,
            )

    # -------------------------------------------------------------------
    # Start / stop
    # -------------------------------------------------------------------

    async def start(self):
        """Start the autonomy loop."""
        if self._running:
            log.warning("Autonomy loop already running")
            return

        self._running = True
        self._session_start = time.time()
        log.info("v3 Autonomy loop started (interval=%.0fs)", self.cycle_interval_sec)
        self.lifecycle_runtime.start_session()

        try:
            while self._running:
                # Session time limit
                elapsed = time.time() - self._session_start
                if elapsed >= self.max_session_sec:
                    log.info("Session time limit reached (%.0f min)", elapsed / 60)
                    break

                await self._run_one_cycle()

                if self._running:
                    await asyncio.sleep(self.cycle_interval_sec)
        except asyncio.CancelledError:
            log.info("Autonomy loop cancelled")
        except Exception as e:
            log.exception("Autonomy loop fatal error: %s", e)
        finally:
            self._running = False
            self.lifecycle_runtime.stop_session()
            log.info("Autonomy loop stopped after %d cycles", self._cycle_count)

    @property
    def running(self) -> bool:
        """Compat property for v2 web toggle interface."""
        return self._running

    @running.setter
    def running(self, value: bool):
        self._running = value

    async def stop(self):
        """Stop the autonomy loop gracefully."""
        self._running = False

    async def run_single_cycle(self) -> CycleResult | None:
        """Run exactly one cycle (for testing or manual triggers)."""
        return await self._run_one_cycle()

    # -------------------------------------------------------------------
    # Main cycle
    # -------------------------------------------------------------------

    async def _run_one_cycle(self) -> CycleResult | None:
        """Execute one full autonomy cycle."""
        self._cycle_count += 1

        # Phase 1: Maintenance (every N cycles)
        if self._cycle_count - self._last_maintenance >= self.maintenance_interval:
            await self._run_maintenance()
            self._last_maintenance = self._cycle_count

        guard = self.guard_runtime.check(
            cycle_count=self._cycle_count,
            consecutive_failures=self._consecutive_failures,
            max_consecutive_failures=self.max_consecutive_failures,
            maintenance_only=self._maintenance_only,
        )
        if guard.consecutive_failures is not None:
            self._consecutive_failures = guard.consecutive_failures
        if not guard.proceed:
            if guard.reason == "budget_exhausted":
                log.warning("Cycle %d: budget exhausted, sleeping", self._cycle_count)
            elif guard.reason == "maintenance_only":
                log.info("Cycle %d: maintenance-only mode", self._cycle_count)
            elif guard.reason == "failure_cooldown":
                log.warning(
                    "Cycle %d: %d consecutive failures, cooling down",
                    self._cycle_count, self.max_consecutive_failures,
                )
            if guard.sleep_sec:
                await asyncio.sleep(guard.sleep_sec)
            return None

        # Phase 5: Mission selection + execution
        selection = self.scheduler_runtime.next_mission()
        self._last_selection = selection
        self._last_scheduler_reason = selection.reason
        if selection.mission is None:
            log.debug("Cycle %d: no runnable missions", self._cycle_count)
            return None

        mission = selection.mission

        log.info("Cycle %d: mission %s [%s]",
                 self._cycle_count, mission.id, mission.description[:50])

        try:
            result = await self.chief.run_cycle(mission)
        except Exception as e:
            log.exception("Cycle %d execution failed: %s", self._cycle_count, e)
            handled = self.error_runtime.handle_cycle_exception(
                mission_id=mission.id,
                error=e,
                consecutive_failures=self._consecutive_failures,
            )
            self._consecutive_failures = handled.consecutive_failures
            self._last_result = handled.cycle_result
            return handled.cycle_result

        # Phase 6: Handle result
        self._last_result = result
        self._consecutive_failures = self.loop_runtime.handle_result(
            result, self._consecutive_failures
        )
        return result

    # -------------------------------------------------------------------
    # Maintenance
    # -------------------------------------------------------------------

    async def _run_maintenance(self):
        """Run periodic maintenance tasks."""
        log.debug("Running maintenance (cycle %d)", self._cycle_count)
        report = self.maintenance_runtime.run()
        if report.get("aura_maintenance"):
            log.debug("Aura maintenance: %s", report["aura_maintenance"])

    # -------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    def status(self) -> dict[str, Any]:
        elapsed = time.time() - self._session_start if self._session_start else 0
        ops = getattr(self.chief, "ops_query_runtime", None)
        query = getattr(self.chief, "mission_query_runtime", None)
        mission = self._last_selection.mission if self._last_selection is not None else None
        mission_summary = None
        if mission is not None and query is not None:
            mission_summary = self.chief.projection_runtime.mission_summary(
                mission,
                plan=query.get_plan(mission.id),
            )
        approvals = ops.pending_approval_items(5) if ops is not None else []
        stuck_missions = ops.stuck_missions(5) if ops is not None else []
        scheduler_recent = ops.scheduler_decisions_recent(5) if ops is not None else []
        quality_debt = ops.quality_debt_by_specialist() if ops is not None else []
        evidence_debt_queue = (
            ops.evidence_debt_queue(5)
            if ops is not None and hasattr(ops, "evidence_debt_queue")
            else []
        )
        specialist_resolution = (
            self.chief.specialist_runtime.last_resolution()
            if getattr(self.chief, "specialist_runtime", None) is not None
            else {}
        )
        return {
            "running": self._running,
            "cycle_count": self._cycle_count,
            "session_elapsed_sec": int(elapsed),
            "consecutive_failures": self._consecutive_failures,
            "maintenance_only": self._maintenance_only,
            "active_missions": len(query.active_missions()) if query is not None else len(self.chief.active_missions()),
            "current_mission": mission_summary,
            "current_task": mission_summary.get("current_task") if mission_summary else None,
            "current_step": mission_summary.get("current_plan_step") if mission_summary else None,
            "last_cycle_result": {
                "decision": self._last_result.decision,
                "reason": self._last_result.reason,
                "mission_id": self._last_result.mission_id,
                "eval_verdict": self._last_result.eval_verdict,
                "cost_usd": self._last_result.cost_usd,
                "tokens_used": self._last_result.tokens_used,
            } if self._last_result is not None else None,
            "pending_approvals": ops.pending_approvals() if ops is not None else 0,
            "approval_queue": approvals,
            "scheduler_reason": self._last_scheduler_reason,
            "scheduler_selection": {
                "mission_id": self._last_selection.mission.id if self._last_selection and self._last_selection.mission else "",
                "score": self._last_selection.score if self._last_selection else 0.0,
                "reason": self._last_selection.reason if self._last_selection else "",
                "runnable_count": getattr(self._last_selection, "runnable_count", 0) if self._last_selection else 0,
                "details": dict(getattr(self._last_selection, "details", None) or {}) if self._last_selection else {},
            },
            "scheduler_decisions_recent": scheduler_recent,
            "stuck_missions_count": len(stuck_missions),
            "stuck_missions": stuck_missions,
            "specialist_resolution": specialist_resolution,
            "quality_debt_by_specialist": quality_debt,
            "evidence_debt_queue": evidence_debt_queue,
            "budget": ops.budget_summary() if ops is not None else self.chief.budget.summary(),
            "recorder": ops.execution_stats() if ops is not None else self.chief.recorder.stats(),
        }


async def run_autonomous_v3() -> dict[str, Any]:
    """Opt-in entrypoint for the v3 runtime."""
    from remy.core.combined_runner import run_autonomy_standalone

    await run_autonomy_standalone(version_override="v3")
    return {}
