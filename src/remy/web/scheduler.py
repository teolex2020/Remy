import asyncio
import logging
import os
from datetime import datetime, timedelta

from remy.core.background_brain import (
    _check_scheduled_tasks,
    run_background,
    send_notifications,
)
from remy.core.agent_tools import brain, brain_runtime_allows_access

logger = logging.getLogger("Scheduler")

# Delay before first maintenance run so startup is not immediately blocked.
_STARTUP_DELAY_SEC = max(int(os.environ.get("REMY_SCHEDULER_STARTUP_DELAY_SEC", "180") or "180"), 0)


class Scheduler:
    def __init__(self):
        self.running = False
        self.task = None
        # Start _last_full_run in the recent past so first maintenance fires
        # only after _STARTUP_DELAY_SEC, not immediately on boot.
        self._last_full_run = datetime.now()

    async def start(self):
        """Start the scheduler loop."""
        if self.running:
            return
        self.running = True
        self.task = asyncio.create_task(self._loop())
        # Fire on_start automations in background (non-blocking)
        asyncio.create_task(self._run_on_start_automations())
        asyncio.create_task(self._run_missed_automations())
        logger.info("Scheduler started.")

    async def stop(self):
        """Stop the scheduler loop."""
        if not self.running:
            return
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped.")

    async def _loop(self):
        """Main scheduler loop."""
        # Give the server time to fully start before any heavy work.
        await asyncio.sleep(_STARTUP_DELAY_SEC)

        while self.running:
            try:
                # 1. Check scheduled tasks (every minute)
                await self._run_task_check()

                # 2. Run full background maintenance (every hour)
                if datetime.now() - self._last_full_run > timedelta(hours=1):
                    await self._run_full_maintenance()

            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")

            # Wait for next minute
            await asyncio.sleep(60)

    async def _run_task_check(self):
        if not brain_runtime_allows_access():
            logger.info("Skipping task check because brain shutdown is in progress.")
            return
        from remy.core.microbrain import bind_project
        from remy.core.project_store import get_project_store

        for owner in get_project_store().list_projects(include_archived=False):
            with bind_project(owner.project_id):
                try:
                    reminders = await asyncio.to_thread(
                        _check_scheduled_tasks,
                        brain,
                    )

                    if reminders:
                        logger.debug(
                            "Found %d due task(s) in project %s.",
                            len(reminders),
                            owner.project_id,
                        )
                        report = {
                            "timestamp": datetime.now().isoformat(),
                            "task_reminders": reminders,
                            "owner_project_id": owner.project_id,
                            "brain_id": owner.brain_id,
                        }
                        await send_notifications(report, brain)

                except Exception as e:
                    logger.error(
                        "Task check failed for project %s: %s",
                        owner.project_id,
                        e,
                    )

        # Fire any scheduled pipelines / automations that are due
        await self._run_scheduled_pipelines()
        await self._run_automations()

    async def _run_scheduled_pipelines(self):
        from remy.core.microbrain import bind_project
        from remy.core.project_store import get_project_store

        for owner in get_project_store().list_projects(include_archived=False):
            with bind_project(owner.project_id):
                await self._run_scheduled_pipelines_for_project(owner)

    async def _run_scheduled_pipelines_for_project(self, owner):
        """Check all enabled scheduled pipelines and fire those that are due now."""
        try:
            from remy.web.routes.scheduled_pipeline_routes import _cron_is_due
            from remy.core.pipeline_runner import run_pipeline_steps
            from remy.core.project_store import project_data_root
            import json

            now = datetime.now()

            def _load_schedules():
                from remy.core.agent_tools import brain_lock
                with brain_lock:
                    try:
                        return brain.search(query="", tags=["scheduled-pipeline"], limit=200)
                    except Exception:
                        return []

            recs = await asyncio.to_thread(_load_schedules)

            for r in recs:
                meta = r.metadata or {}
                if meta.get("type") != "scheduled_pipeline":
                    continue
                if not meta.get("enabled", True):
                    continue

                cron = meta.get("cron", "")
                if not cron or not _cron_is_due(cron, now):
                    continue

                schedule_id = meta.get("schedule_id", "")
                pipeline_id = meta.get("pipeline_id", "")
                input_text = meta.get("input_text", "")
                name = meta.get("name", schedule_id)

                logger.info("Firing scheduled pipeline '%s' (cron=%s)", name, cron)

                pipeline_path = (
                    project_data_root(owner.project_id)
                    / "pipelines"
                    / f"{pipeline_id}.json"
                )
                if not pipeline_path.exists():
                    logger.warning("Scheduled pipeline '%s': file not found (%s)", name, pipeline_path)
                    continue

                attempt_id = ""
                run_record = None
                try:
                    from remy.core.run_envelope import (
                        RunCoordinator,
                        RunLimits,
                        finish_run,
                        start_run,
                    )

                    run_record = start_run(
                        kind="scheduled_pipeline",
                        source_id=schedule_id,
                        goal=input_text or name or "Scheduled pipeline",
                        idempotency_class="side_effecting",
                        owner_project_id=owner.project_id,
                        brain_id=owner.brain_id,
                        conversation_id=f"schedule:{schedule_id}",
                        channel="scheduler",
                        limits=RunLimits(
                            max_turns=250,
                            token_budget=250_000,
                            max_parallel_workers=1,
                        ),
                        metadata={
                            "schedule_id": schedule_id,
                            "pipeline_id": pipeline_id,
                            "name": name,
                            "scheduled_minute": now.strftime('%Y%m%d%H%M'),
                        },
                    )
                    attempt_id = run_record["attempt_id"]
                    coordinator = RunCoordinator(attempt_id)
                    with open(pipeline_path, "r", encoding="utf-8") as f:
                        pipeline_data = json.load(f)
                    steps = pipeline_data.get("steps", [])

                    last_output = ""
                    async for event in run_pipeline_steps(steps, input_text):
                        if event.get("type") == "step_done":
                            last_output = event.get("output", "")
                            coordinator.step(
                                str(event.get("label") or event.get("step") or "Pipeline step"),
                                signature=str(event.get("id") or event.get("step_id") or event.get("label") or ""),
                            )

                    logger.info("Scheduled pipeline '%s' finished. Output: %s…", name, last_output[:120])

                    # Update run metadata
                    def _update(sid=schedule_id, lo=last_output):
                        from remy.core.agent_tools import brain_lock
                        try:
                            with brain_lock:
                                recs2 = brain.search(query="", tags=["scheduled-pipeline"], limit=200)
                                for r2 in recs2:
                                    m2d = r2.metadata or {}
                                    if m2d.get("schedule_id") == sid:
                                        m2 = dict(m2d)
                                        m2["last_run_at"] = datetime.now().isoformat()
                                        m2["last_run_status"] = "ok"
                                        m2["run_count"] = m2.get("run_count", 0) + 1
                                        rid = getattr(r2, "id", None)
                                        if rid:
                                            try:
                                                brain.delete(rid)
                                            except Exception:
                                                pass
                                        summary = f"Scheduled pipeline: {m2['name']} | pipeline_id={m2['pipeline_id']} | cron={m2['cron']}"
                                        brain.store(summary, tags=["scheduled-pipeline"], metadata=m2)
                                        break
                        except Exception as ue:
                            logger.warning("Could not update run metadata for '%s': %s", sid, ue)

                    from remy.core.memory_write_queue import get_memory_write_queue
                    await get_memory_write_queue().run(_update)
                    finish_run(
                        attempt_id,
                        status="completed",
                        output_ref=f"scheduled-pipeline:{schedule_id}",
                        artifacts=[{
                            "kind": "scheduled_pipeline_output",
                            "ref": f"scheduled-pipeline:{schedule_id}",
                            "preview": last_output[:500],
                        }],
                    )

                except Exception as pe:
                    if attempt_id:
                        try:
                            from remy.core.run_envelope import finish_run

                            finish_run(attempt_id, status="failed", error=str(pe))
                        except Exception:
                            pass
                    logger.error("Scheduled pipeline '%s' failed: %s", name, pe)

        except Exception as e:
            logger.error("Scheduled pipelines check failed: %s", e)

    async def _run_automations(self):
        from remy.core.microbrain import bind_project
        from remy.core.project_store import get_project_store

        for owner in get_project_store().list_projects(include_archived=False):
            with bind_project(owner.project_id):
                await self._run_automations_for_project()

    async def _run_automations_for_project(self):
        """Check all automations with schedule trigger and fire those due now."""
        try:
            from remy.web.routes.automation_routes import cron_is_due, run_automation_record
            from remy.core.agent_tools import brain_lock

            now = datetime.now()

            def _load():
                with brain_lock:
                    try:
                        return brain.search(query="", tags=["automation"], limit=500)
                    except Exception:
                        return []

            recs = await asyncio.to_thread(_load)

            for r in recs:
                meta = r.metadata or {}
                if meta.get("type") != "automation":
                    continue
                if not meta.get("enabled", True):
                    continue
                trigger = meta.get("trigger", {})
                if trigger.get("type") == "on_start":
                    continue  # handled separately at startup
                cron = meta.get("cron", "")
                if not cron or not cron_is_due(cron, now):
                    continue

                name = meta.get("name", meta.get("automation_id", "?"))
                logger.info("Firing automation '%s'", name)

                result = await run_automation_record(brain, brain_lock, dict(meta))
                if result.get("ok"):
                    logger.info("Automation '%s' completed (%d steps)", name, result.get("steps_run", 0))
                else:
                    logger.error("Automation '%s' failed: %s", name, result.get("error", "unknown error"))

        except Exception as e:
            logger.error("Automations check failed: %s", e)

    async def _run_on_start_automations(self):
        """Fire all automations with trigger type 'on_start' once at startup."""
        await asyncio.sleep(10)  # let server fully initialize first
        from remy.core.microbrain import bind_project
        from remy.core.project_store import get_project_store

        for owner in get_project_store().list_projects(include_archived=False):
            with bind_project(owner.project_id):
                await self._run_on_start_automations_for_project()

    async def _run_on_start_automations_for_project(self):
        try:
            from remy.web.routes.automation_routes import run_automation_record
            from remy.core.agent_tools import brain_lock

            def _load():
                with brain_lock:
                    try:
                        return brain.search(query="", tags=["automation"], limit=500)
                    except Exception:
                        return []

            recs = await asyncio.to_thread(_load)
            for r in recs:
                meta = r.metadata or {}
                if meta.get("type") != "automation":
                    continue
                if not meta.get("enabled", True):
                    continue
                if meta.get("trigger", {}).get("type") != "on_start":
                    continue
                name = meta.get("name", "?")
                logger.info("Firing on_start automation '%s'", name)
                result = await run_automation_record(brain, brain_lock, dict(meta))
                if result.get("ok"):
                    logger.info("on_start automation '%s' done (%d steps)", name, result.get("steps_run", 0))
                else:
                    logger.error("on_start automation '%s' failed: %s", name, result.get("error", "unknown error"))
        except Exception as e:
            logger.error("on_start automations failed: %s", e)

    async def _run_missed_automations(self):
        """Catch up simple scheduled automations missed while Remy was not running."""
        await asyncio.sleep(15)
        from remy.core.microbrain import bind_project
        from remy.core.project_store import get_project_store

        for owner in get_project_store().list_projects(include_archived=False):
            with bind_project(owner.project_id):
                await self._run_missed_automations_for_project()

    async def _run_missed_automations_for_project(self):
        try:
            from remy.web.routes.automation_routes import latest_missed_scheduled_run, run_automation_record
            from remy.core.agent_tools import brain_lock

            now = datetime.now()

            def _load():
                with brain_lock:
                    try:
                        return brain.search(query="", tags=["automation"], limit=500)
                    except Exception:
                        return []

            recs = await asyncio.to_thread(_load)
            for r in recs:
                meta = dict(r.metadata or {})
                if meta.get("type") != "automation":
                    continue
                due_at = latest_missed_scheduled_run(meta, now)
                if not due_at:
                    continue
                name = meta.get("name", meta.get("automation_id", "?"))
                logger.info("Catching up missed automation '%s' scheduled for %s", name, due_at.isoformat())
                result = await run_automation_record(brain, brain_lock, meta)
                if result.get("ok"):
                    logger.info("Missed automation '%s' completed (%d steps)", name, result.get("steps_run", 0))
                else:
                    logger.error("Missed automation '%s' failed: %s", name, result.get("error", "unknown error"))
        except Exception as e:
            logger.error("Missed automations catch-up failed: %s", e)

    async def _run_full_maintenance(self):
        if not brain_runtime_allows_access():
            logger.info("Skipping full background maintenance because brain shutdown is in progress.")
            return
        from remy.core.microbrain import bind_project
        from remy.core.project_store import get_project_store

        for owner in get_project_store().list_projects(include_archived=False):
            with bind_project(owner.project_id):
                try:
                    logger.info(
                        "Running full background maintenance for project %s...",
                        owner.project_id,
                    )
                    report = await asyncio.to_thread(run_background, brain)

                    if (
                        report.get("insights_found", 0) > 0
                        or report.get("cross_connections", 0) > 0
                    ):
                        report["owner_project_id"] = owner.project_id
                        report["brain_id"] = owner.brain_id
                        await send_notifications(report, brain)

                    await self._run_brain_voice()
                except Exception as e:
                    logger.error(
                        "Full maintenance failed for project %s: %s",
                        owner.project_id,
                        e,
                    )
        self._last_full_run = datetime.now()
        logger.info("Full background maintenance complete.")

    async def _run_brain_voice(self):
        """Detect TCS-level events after maintenance and emit via event bus."""
        try:
            from remy.core.brain_voice import detect_and_record
            from remy.core.event_bus import event_bus
            from remy.core.microbrain import current_project_id
            from remy.core.project_store import get_project_store, local_brain_path

            owner = get_project_store().require_project(current_project_id())
            data_dir = str(local_brain_path(owner.project_id))
            new_events = await asyncio.to_thread(detect_and_record, data_dir)
            for ev in new_events:
                event_bus.emit(
                    "brain.voice",
                    {
                        **ev.to_dict(),
                        "owner_project_id": owner.project_id,
                        "brain_id": owner.brain_id,
                    },
                )
                logger.info("brain.voice emitted: kind=%s severity=%s", ev.kind, ev.severity)
        except Exception as e:
            logger.warning(f"Brain voice detection failed: {e}")
