"""
Autonomy Goals — goal CRUD, action plans, decision trees.

Goal system + plan management extracted from autonomy.py.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from remy.core.event_bus import event_bus

logger = logging.getLogger("Autonomy")


def _get_autonomy():
    """Lazy accessor — reads from autonomy module (supports test patching)."""
    import remy.core.autonomy as _au

    return _au


def _get_goal_brain():
    """Return autonomy brain, recovering from a stale closed handle."""
    au = _get_autonomy()
    candidate = au.brain
    if hasattr(candidate, "count"):
        try:
            candidate.count()
            if hasattr(candidate, "search"):
                candidate.search(query="", limit=1)
            return candidate
        except RuntimeError as exc:
            if "closed" not in str(exc).lower():
                raise
            from remy.core.agent_tools import brain as live_brain
            au.brain = live_brain
            return live_brain
        except Exception:
            return candidate
    return candidate


# ============== GOAL CLEANUP ==============


def archive_completed_goals():
    """Archive completed/failed goals older than 24h. Returns count archived."""
    from remy.core.agent_tools import brain_lock

    with brain_lock:
        return _archive_completed_goals_inner()


def _archive_completed_goals_inner():
    brain = _get_goal_brain()
    goals = brain.search(query="", tags=GOAL_TAGS, limit=200)
    archived = 0
    cutoff = (datetime.now() - timedelta(hours=24)).isoformat()

    for g in goals:
        meta = g.metadata or {}
        status = meta.get("status", "active")
        if status not in ("completed", "failed"):
            continue
        # Never archive immortal survival goals
        if meta.get("immortal"):
            continue
        updated = meta.get("updated_at") or meta.get("created_at", "")
        if updated and updated < cutoff:
            meta["status"] = "archived"
            meta["archived_at"] = datetime.now().isoformat()
            brain.update(g.id, metadata=meta)
            archived += 1

    if archived:
        logger.info("Archived %d completed/failed goals", archived)
    return archived


# ============== SURVIVAL META-GOAL ==============

SURVIVAL_GOAL_DESCRIPTION = (
    "SURVIVAL: Stay alive by being useful. You pay for LLM API tokens "
    "from your own wallet. No money = no tokens = death. "
    "Scale spending with income (max 80% on API, save 20% for growth). "
    "No income = spend absolute minimum, use free models. "
    "Check your missions (data/missions.json) for earning strategies. "
    "Execute mission goals to generate value for the owner → funding for you. "
    "This goal is immortal and cannot be archived or completed."
)


def ensure_survival_goal() -> str | None:
    """Ensure the immortal survival goal exists. Creates it if missing.

    The survival goal is:
    - Tagged with 'immortal' in metadata — never archived
    - Always high priority
    - Never marked completed — it resets to 'active' if anything changes it

    Returns the goal ID if created, None if it already exists.
    """
    brain = _get_goal_brain()
    from remy.core.agent_tools import brain_lock

    # Check if survival goal already exists
    with brain_lock:
        goals = brain.search(query="SURVIVAL", tags=GOAL_TAGS, limit=20)

    for g in goals:
        meta = g.metadata or {}
        if meta.get("immortal"):
            # Ensure it's active (resurrect if someone changed it)
            if meta.get("status") != "active":
                meta["status"] = "active"
                meta["updated_at"] = datetime.now().isoformat()
                with brain_lock:
                    brain.update(g.id, metadata=meta)
                logger.info("Survival goal resurrected: %s", g.id)
            return None  # Already exists

    # Create the survival goal
    from remy.core.agent_tools import Level
    from remy.core.provenance import _stamp_provenance

    goal_id = f"goal-survival-{uuid.uuid4().hex[:8]}"

    tags = GOAL_TAGS + ["priority-high"]

    metadata = _stamp_provenance(
        {
            "type": "autonomous_goal",
            "goal_id": goal_id,
            "priority": "high",
            "status": "active",
            "created_by": "system",
            "created_at": datetime.now().isoformat(),
            "deadline": None,
            "parent_goal_id": None,
            "depends_on": [],
            "attempts": 0,
            "last_attempt": None,
            "outcome_ids": [],
            "success_criteria": [],
            "immortal": True,  # <-- This makes it immune to archiving
        },
        "autonomous",
        tags=tags,
    )

    content = f"Goal [HIGH]: {SURVIVAL_GOAL_DESCRIPTION}"

    with brain_lock:
        brain.store(
            content=content,
            level=Level.DECISIONS,
            tags=tags,
            metadata=metadata,
        )

    logger.info("Survival meta-goal CREATED: %s", goal_id)
    event_bus.emit("survival.goal_created", {"goal_id": goal_id})
    return goal_id


# ============== CUSTOM MISSIONS (from data/missions.json) ==============


MISSIONS_FILENAME = "missions.json"


def _load_missions() -> list[dict]:
    """Load missions from data/missions/ directory (one file per mission).

    Each file is a single mission dict with:
      - "id": unique string identifier (e.g. "aura-sdk-promotion")
      - "description": goal text shown to agent
      - "priority": "critical" | "high" | "medium" | "low"
      - "immortal": true/false — if true, goal is never archived
      - "tasks": optional list of atomic task dicts

    Optional seed-only fields (only needed on first run, then stored in brain):
      - "knowledge": list of {"content": str, "tags": [str]} — seeded into brain once
      - "earning_strategy": list of strings — seeded into brain once

    Pack manifest fields (optional, per mission):
      - "schedule": cron-like schedule ("daily 09:00", "every 6h", "weekly mon 10:00")
      - "tools": list of allowed tool names (whitelist; empty = all tools)
      - "budget_per_run": max USD spend per execution cycle
      - "approval_gates": list of action types requiring approval ("publish", "financial", "all")
      - "goal_template": explicit capability pack ID override

    See data/missions/example.json for format reference.
    Falls back to legacy data/missions.json if directory does not exist.
    """
    from remy.config.settings import settings

    missions_dir = settings.DATA_DIR / "missions"
    if missions_dir.is_dir():
        missions = []
        for path in sorted(missions_dir.glob("*.json")):
            if path.stem == "example":
                continue
            try:
                mission = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(mission, dict) and mission.get("id"):
                    missions.append(mission)
                else:
                    logger.warning("Skipping invalid mission file: %s", path.name)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Could not load mission from %s: %s", path.name, e)
        return missions

    # Legacy fallback: single missions.json
    path = settings.DATA_DIR / MISSIONS_FILENAME
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("missions", [])
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not load missions from %s: %s", path, e)
        return []


def ensure_mission_goals() -> list[str]:
    """Ensure all missions from data/missions.json exist as goals.

    Creates missing goals and seeds knowledge. Resurrects archived ones.
    Returns list of newly created goal IDs.
    """
    missions = _load_missions()
    if not missions:
        return []

    brain = _get_goal_brain()
    from remy.core.agent_tools import brain_lock

    created = []
    for mission in missions:
        mission_id = mission.get("id", "")
        if not mission_id or not mission.get("description"):
            continue

        # Check if this mission's goal already exists
        with brain_lock:
            goals = brain.search(query=mission_id, tags=GOAL_TAGS, limit=10)

        already_exists = False
        existing_goal_id = None
        for g in goals:
            meta = g.metadata or {}
            if meta.get("mission_id") == mission_id and not meta.get("mission_task_id"):
                # Resurrect if needed
                if meta.get("status") != "active" and mission.get("immortal"):
                    meta["status"] = "active"
                    meta["updated_at"] = datetime.now().isoformat()
                    with brain_lock:
                        brain.update(g.id, metadata=meta)
                    logger.info("Mission goal resurrected: %s", mission_id)
                existing_goal_id = meta.get("goal_id", "")
                already_exists = True
                break

        if already_exists:
            # Seed knowledge even for existing missions (idempotent — skips if already seeded)
            knowledge = mission.get("knowledge", [])
            if knowledge:
                _seed_mission_knowledge(brain, brain_lock, mission_id, knowledge)
            earning_strategy = mission.get("earning_strategy", [])
            if earning_strategy:
                _seed_mission_earning_strategies(brain, brain_lock, mission_id, earning_strategy)
            # Still ensure tasks are created/updated for existing missions
            tasks = mission.get("tasks", [])
            if tasks and existing_goal_id:
                _ensure_mission_tasks(
                    brain,
                    brain_lock,
                    mission_id,
                    existing_goal_id,
                    tasks,
                    mission.get("priority", "medium"),
                )
            continue

        # Create the mission goal
        from remy.core.agent_tools import Level
        from remy.core.provenance import _stamp_provenance

        goal_id = f"goal-mission-{mission_id}-{uuid.uuid4().hex[:8]}"
        priority = mission.get("priority", "medium")
        tags = GOAL_TAGS + [f"priority-{priority}", f"mission-{mission_id}"]

        # Pack manifest fields (optional per mission)
        pack_manifest = {}
        if mission.get("schedule"):
            pack_manifest["schedule"] = mission["schedule"]
        if mission.get("tools"):
            pack_manifest["tools"] = mission["tools"]
        if mission.get("budget_per_run") is not None:
            pack_manifest["budget_per_run"] = float(mission["budget_per_run"])
        if mission.get("approval_gates"):
            pack_manifest["approval_gates"] = mission["approval_gates"]
        if mission.get("goal_template"):
            pack_manifest["goal_template"] = mission["goal_template"]

        metadata = _stamp_provenance(
            {
                "type": "autonomous_goal",
                "goal_id": goal_id,
                "mission_id": mission_id,
                "priority": priority,
                "status": "active",
                "created_by": "system",
                "created_at": datetime.now().isoformat(),
                "deadline": mission.get("deadline"),
                "parent_goal_id": None,
                "depends_on": [],
                "attempts": 0,
                "last_attempt": None,
                "outcome_ids": [],
                "success_criteria": [],
                "immortal": mission.get("immortal", False),
                **({"goal_template": mission["goal_template"]} if mission.get("goal_template") else {}),
                **({"pack_manifest": pack_manifest} if pack_manifest else {}),
            },
            "autonomous",
            tags=tags,
        )

        content = f"Goal [{priority.upper()}]: {mission['description']}"

        with brain_lock:
            brain.store(
                content=content,
                level=Level.DECISIONS,
                tags=tags,
                metadata=metadata,
            )

        logger.info("Mission goal CREATED: %s (%s)", mission_id, goal_id)
        event_bus.emit(
            "mission.goal_created",
            {
                "mission_id": mission_id,
                "goal_id": goal_id,
            },
        )
        created.append(goal_id)

        # Seed knowledge if provided
        knowledge = mission.get("knowledge", [])
        if knowledge:
            _seed_mission_knowledge(brain, brain_lock, mission_id, knowledge)

        # Seed earning strategies if provided
        earning_strategy = mission.get("earning_strategy", [])
        if earning_strategy:
            _seed_mission_earning_strategies(brain, brain_lock, mission_id, earning_strategy)

        # Create atomic task sub-goals if provided
        tasks = mission.get("tasks", [])
        if tasks:
            _ensure_mission_tasks(
                brain,
                brain_lock,
                mission_id,
                goal_id,
                tasks,
                mission.get("priority", "medium"),
            )

    # --- Cleanup: archive goals for missions removed from missions.json ---
    active_mission_ids = {m.get("id") for m in missions if m.get("id")}
    with brain_lock:
        all_goals = brain.search(query="", tags=GOAL_TAGS, limit=500)
    for g in all_goals:
        meta = g.metadata or {}
        mid = meta.get("mission_id", "")
        if mid and mid not in active_mission_ids and meta.get("status") not in ("archived", "completed"):
            meta["status"] = "archived"
            meta["updated_at"] = datetime.now().isoformat()
            with brain_lock:
                brain.update(g.id, metadata=meta)
            logger.info("Archived orphan mission goal: mission_id=%s goal_id=%s", mid, meta.get("goal_id", g.id))

    return created


def _seed_mission_knowledge(brain, brain_lock, mission_id: str, facts: list[dict]):
    """Seed mission-specific knowledge into brain at Domain level (idempotent)."""
    from remy.core.agent_tools import Level

    # Check if already seeded
    tag = f"mission-{mission_id}"
    with brain_lock:
        existing = brain.search(query=mission_id, tags=[tag], limit=5)
    if existing:
        logger.debug("Knowledge for mission '%s' already seeded", mission_id)
        return

    count = 0
    for fact in facts:
        content = fact.get("content", "")
        if not content:
            continue
        fact_tags = list(fact.get("tags", [])) + [tag]
        with brain_lock:
            brain.store(
                content=content,
                level=Level.DOMAIN,
                tags=fact_tags,
                metadata={
                    "type": "product_knowledge",
                    "source": "mission_seed",
                    "mission_id": mission_id,
                },
            )
        count += 1

    if count:
        logger.info("Seeded %d knowledge records for mission '%s'", count, mission_id)


def _seed_mission_earning_strategies(brain, brain_lock, mission_id: str, strategies: list[str]):
    """Seed earning strategies into brain at Domain level (idempotent)."""
    from remy.core.agent_tools import Level

    tag = f"earning-strategy-{mission_id}"
    with brain_lock:
        existing = brain.search(query="earning strategy", tags=[tag], limit=1)
    if existing:
        logger.debug("Earning strategies for mission '%s' already seeded", mission_id)
        return

    count = 0
    for line in strategies:
        if not line:
            continue
        with brain_lock:
            brain.store(
                content=line,
                level=Level.DOMAIN,
                tags=["earning-strategy", tag, f"mission-{mission_id}"],
                metadata={
                    "type": "earning_strategy",
                    "source": "mission_seed",
                    "mission_id": mission_id,
                },
            )
        count += 1

    if count:
        logger.info("Seeded %d earning strategies for mission '%s'", count, mission_id)


def get_mission_earning_strategies() -> list[str]:
    """Return earning strategy lines from brain (seeded from missions on startup)."""
    try:
        au = _get_autonomy()
        brain = au.brain
        from remy.core.agent_tools import brain_lock

        with brain_lock:
            records = brain.search(query="earning strategy", tags=["earning-strategy"], limit=50)
        return [r.content for r in records if r.content]
    except Exception:
        # Fallback: read from missions.json directly (first run before seed)
        missions = _load_missions()
        strategies = []
        for m in missions:
            for line in m.get("earning_strategy", []):
                strategies.append(line)
        return strategies


# ============== MISSION TASKS (atomic task queue) ==============

TASK_REPEAT_INTERVALS = {
    "daily": timedelta(days=1),
    "weekly": timedelta(days=7),
    "monthly": timedelta(days=30),
}


def _repeat_interval_passed(completed_at: str, interval: str) -> bool:
    """Check if enough time has passed since task completion for repeat."""
    if not completed_at or interval not in TASK_REPEAT_INTERVALS:
        return False
    try:
        completed_dt = datetime.fromisoformat(completed_at)
        return (datetime.now() - completed_dt) >= TASK_REPEAT_INTERVALS[interval]
    except (ValueError, TypeError):
        return True  # If can't parse, allow repeat


def _ensure_mission_tasks(
    brain,
    brain_lock,
    mission_id: str,
    parent_goal_id: str,
    tasks: list[dict],
    mission_priority: str,
):
    """Create sub-goal for each task in a mission. Idempotent.

    Tasks are created with status='pending'. Only one task is active at a time.
    """
    from remy.core.agent_tools import Level
    from remy.core.provenance import _stamp_provenance

    if not tasks:
        return

    # Load existing task sub-goals for this mission
    with brain_lock:
        existing_goals = brain.search(query="", tags=GOAL_TAGS, limit=200)

    existing_by_task_id = {}
    for g in existing_goals:
        meta = g.metadata or {}
        tid = meta.get("mission_task_id")
        if tid and meta.get("mission_id") == mission_id:
            existing_by_task_id[tid] = (g, meta)

    # --- Validate task dependency graph before creating ---
    all_task_ids = {t.get("id") for t in tasks if t.get("id")}
    for task in tasks:
        dep = task.get("depends_on")
        if dep and dep not in all_task_ids:
            logger.warning(
                "Mission '%s' task '%s' depends on non-existent task '%s' — dependency ignored",
                mission_id, task.get("id", "?"), dep,
            )
            task["depends_on"] = None  # Clear invalid dep so task can activate

    # Detect circular dependencies (simple cycle check)
    def _has_cycle(tid: str, visited: set) -> bool:
        if tid in visited:
            return True
        visited.add(tid)
        dep_map = {t.get("id"): t.get("depends_on") for t in tasks}
        nxt = dep_map.get(tid)
        if nxt:
            return _has_cycle(nxt, visited)
        return False

    for task in tasks:
        tid = task.get("id", "")
        if tid and task.get("depends_on") and _has_cycle(tid, set()):
            logger.warning(
                "Mission '%s' task '%s' has circular dependency — dependency cleared",
                mission_id, tid,
            )
            task["depends_on"] = None

    created_count = 0
    for task in tasks:
        task_id = task.get("id", "")
        action = task.get("action", "")
        if not task_id or not action:
            continue

        done_when = task.get("done_when", "")
        priority = task.get("priority", mission_priority)
        repeat = task.get("repeat")
        depends_on_task = task.get("depends_on")
        task_goal_type, task_goal_template, _ = _resolve_goal_metadata_from_config(
            action,
            explicit_goal_type=task.get("goal_type"),
            explicit_goal_template=task.get("goal_template"),
        )

        if task_id in existing_by_task_id:
            # Task already exists — check for repeat reset
            existing_rec, existing_meta = existing_by_task_id[task_id]
            if (
                existing_meta.get("status") == "completed"
                and repeat
                and _repeat_interval_passed(existing_meta.get("completed_at", ""), repeat)
            ):
                existing_meta["status"] = "pending"
                existing_meta["attempts"] = 0
                existing_meta["updated_at"] = datetime.now().isoformat()
                with brain_lock:
                    brain.update(existing_rec.id, metadata=existing_meta)
                logger.info(
                    "Repeating task reset to pending: %s/%s",
                    mission_id,
                    task_id,
                )
            continue

        # Create new task sub-goal
        goal_id = f"goal-task-{mission_id}-{task_id}"
        tags = GOAL_TAGS + [
            f"priority-{priority}",
            f"mission-{mission_id}",
            f"task-{task_id}",
        ]

        metadata = _stamp_provenance(
            {
                "type": "autonomous_goal",
                "goal_id": goal_id,
                "goal_type": task_goal_type,
                "goal_template": task_goal_template,
                "mission_id": mission_id,
                "mission_task_id": task_id,
                "priority": priority,
                "status": "pending",
                "created_by": "mission_task",
                "created_at": datetime.now().isoformat(),
                "deadline": None,
                "parent_goal_id": parent_goal_id,
                "depends_on": [],
                "attempts": 0,
                "last_attempt": None,
                "outcome_ids": [],
                "success_criteria": [],
                "immortal": False,
                "task_action": action,
                "task_done_when": done_when,
                "task_depends_on": depends_on_task,
                "task_repeat": repeat,
            },
            "autonomous",
            tags=tags,
        )

        content = f"Task [{priority.upper()}]: {action}"

        with brain_lock:
            rec = brain.store(
                content=content,
                level=Level.DECISIONS,
                tags=tags,
                metadata=metadata,
            )
            # Connect to parent mission goal (promotion-gated)
            from remy.core.agent_tools import gated_connect
            parent_records = brain.search(query="", tags=GOAL_TAGS, limit=200)
            for pr in parent_records:
                if pr.metadata and pr.metadata.get("goal_id") == parent_goal_id:
                    gated_connect(brain, rec.id, pr.id, weight=0.9)
                    break

        created_count += 1

    if created_count:
        logger.info(
            "Created %d task sub-goals for mission '%s'",
            created_count,
            mission_id,
        )

    # Activate first eligible task
    _activate_next_task(brain, brain_lock, mission_id)


def _activate_next_task(brain, brain_lock, mission_id: str):
    """Activate the next pending task for a mission (if none currently active)."""
    with brain_lock:
        all_goals = brain.search(query="", tags=GOAL_TAGS, limit=200)

    task_goals = []
    for g in all_goals:
        meta = g.metadata or {}
        if meta.get("mission_id") == mission_id and meta.get("mission_task_id"):
            task_goals.append((g, meta))

    # Check if any task is already active
    for _, meta in task_goals:
        if meta.get("status") == "active":
            return  # Already have an active task

    # Build set of completed task IDs for dependency resolution
    completed_task_ids = {
        meta.get("mission_task_id") for _, meta in task_goals if meta.get("status") == "completed"
    }

    # Build set of all known task IDs to detect orphan deps
    all_known_task_ids = {
        meta.get("mission_task_id") for _, meta in task_goals if meta.get("mission_task_id")
    }

    # Find first pending task with satisfied dependencies
    for g, meta in task_goals:
        if meta.get("status") != "pending":
            continue
        dep = meta.get("task_depends_on")
        if dep and dep not in completed_task_ids:
            # Check if dependency task even exists — if not, clear the dep
            if dep not in all_known_task_ids:
                logger.warning(
                    "Task %s/%s depends on unknown '%s' — clearing dependency",
                    mission_id, meta.get("mission_task_id"), dep,
                )
                meta["task_depends_on"] = None
                meta["updated_at"] = datetime.now().isoformat()
                with brain_lock:
                    brain.update(g.id, metadata=meta)
                # Fall through to activate
            else:
                continue  # Dependency exists but not completed yet
        # Activate this task
        meta["status"] = "active"
        meta["updated_at"] = datetime.now().isoformat()
        with brain_lock:
            brain.update(g.id, metadata=meta)
        logger.info(
            "Task activated: %s/%s",
            mission_id,
            meta.get("mission_task_id"),
        )
        event_bus.emit(
            "mission.task_active",
            {
                "record_id": g.id,
                "goal_id": meta.get("goal_id", ""),
                "mission_id": mission_id,
                "mission_task_id": meta.get("mission_task_id", ""),
            },
        )
        return


def activate_next_mission_task(mission_id: str):
    """Public API: activate next pending task after one completes."""
    brain = _get_goal_brain()
    from remy.core.agent_tools import brain_lock

    _activate_next_task(brain, brain_lock, mission_id)


# ============== GOAL SYSTEM ==============

GOAL_TAGS = ["autonomous-goal"]


def _resolve_goal_metadata_from_config(
    description: str,
    explicit_goal_type: str | None = None,
    explicit_goal_template: str | None = None,
):
    """Resolve goal_type/goal_template with explicit metadata winning over inference."""
    goal_type = (explicit_goal_type or "").strip() or "general"
    goal_template_name = (explicit_goal_template or "").strip() or ""
    inferred_template = None

    try:
        from remy.core.autonomy_roles import infer_goal_type
        from remy.core.success_criteria import infer_goal_template

        if goal_type == "general":
            goal_type = infer_goal_type(description)
        if not goal_template_name:
            inferred_template = infer_goal_template(description)
            goal_template_name = (inferred_template or {}).get("name", "")
    except Exception:
        inferred_template = None

    return goal_type, goal_template_name, inferred_template


def create_goal(
    description: str,
    priority: str = "medium",
    deadline: str | None = None,
    parent_goal_id: str | None = None,
    created_by: str = "agent",
    depends_on: list[str] | None = None,
    goal_type: str | None = None,
    goal_template: str | None = None,
) -> str:
    """Create a new goal in brain. Returns brain record ID."""
    from remy.core.agent_tools import Level, brain_lock

    au = _get_autonomy()
    brain = au.brain

    goal_id = f"goal-{uuid.uuid4().hex[:12]}"

    from remy.core.provenance import _stamp_provenance

    tags = GOAL_TAGS + [f"priority-{priority}"]

    # AUTON-5: Generate success criteria for the goal
    resolved_goal_type = "general"
    resolved_goal_template_name = ""
    try:
        from remy.core.success_criteria import generate_criteria_for_goal

        resolved_goal_type, resolved_goal_template_name, _ = _resolve_goal_metadata_from_config(
            description,
            explicit_goal_type=goal_type,
            explicit_goal_template=goal_template,
        )
        criteria = generate_criteria_for_goal(description)
        # Stamp brain_count baseline for brain_count_increased criteria
        for c in criteria:
            if c.get("type") == "brain_count_increased" and "start_count" not in c:
                with brain_lock:
                    c["start_count"] = brain.count()
    except Exception:
        criteria = []
        resolved_goal_type = (goal_type or "general").strip() or "general"
        resolved_goal_template_name = (goal_template or "").strip()

    metadata = _stamp_provenance(
        {
            "type": "autonomous_goal",
            "goal_id": goal_id,
            "goal_type": resolved_goal_type,
            "goal_template": resolved_goal_template_name,
            "priority": priority,
            "status": "active",
            "created_by": created_by,
            "created_at": datetime.now().isoformat(),
            "deadline": deadline,
            "parent_goal_id": parent_goal_id,
            "depends_on": depends_on or [],
            "attempts": 0,
            "last_attempt": None,
            "outcome_ids": [],
            "success_criteria": criteria,
        },
        "autonomous",
        tags=tags,
    )

    content = f"Goal [{priority.upper()}]: {description}"
    if deadline:
        content += f" | Deadline: {deadline}"

    with brain_lock:
        try:
            rec = brain.store(
                content=content,
                level=Level.DECISIONS,
                tags=tags,
                metadata=metadata,
            )
        except RuntimeError as exc:
            if "closed" not in str(exc).lower():
                raise
            from remy.core.agent_tools import brain as live_brain
            _get_autonomy().brain = live_brain
            brain = live_brain
            rec = brain.store(
                content=content,
                level=Level.DECISIONS,
                tags=tags,
                metadata=metadata,
            )

        # Connect to parent if sub-goal (promotion-gated)
        if parent_goal_id:
            from remy.core.agent_tools import gated_connect
            parent_records = brain.search(query="", tags=GOAL_TAGS, limit=200)
            for pr in parent_records:
                if pr.metadata and pr.metadata.get("goal_id") == parent_goal_id:
                    gated_connect(brain, rec.id, pr.id, weight=0.9)
                    break

    logger.info("Goal created: %s - %s", goal_id, description[:60])
    return rec.id


def get_active_goals() -> list[dict]:
    """Retrieve all active goals, sorted by priority. Sub-goals are preferred."""
    from remy.core.agent_tools import brain_lock
    from remy.core.smart_goals import smart_sort_goals

    au = _get_autonomy()
    brain = au.brain

    with brain_lock:
        goals = brain.search(query="", tags=GOAL_TAGS, limit=200)

    # Ensure existing mission queues actually have a runnable atomic task.
    # This matters on restarts, where mission tasks may all remain pending even
    # though the mission itself is active.
    mission_ids = {
        (g.metadata or {}).get("mission_id") for g in goals if (g.metadata or {}).get("mission_id")
    }
    for mission_id in mission_ids:
        _activate_next_task(brain, brain_lock, mission_id)

    with brain_lock:
        goals = brain.search(query="", tags=GOAL_TAGS, limit=200)

    MAX_GOAL_ATTEMPTS = 20  # auto-archive goals that exceed this

    active = []
    for g in goals:
        meta = g.metadata or {}
        status = meta.get("status")
        if status not in ("active", "blocked_by_user", "blocked_external", None):
            continue
        if status is None:
            continue

        # Auto-unblock goals whose approval expired (stale block > 5 min)
        if status == "blocked_by_user":
            blocked_at = meta.get("blocked_at", "")
            if blocked_at:
                try:
                    blocked_dt = datetime.fromisoformat(blocked_at)
                    if (datetime.now() - blocked_dt).total_seconds() > 300:
                        logger.info("Auto-unblocking stale blocked goal: %s", g.content[:60])
                        meta["status"] = "active"
                        meta.pop("blocked_at", None)
                        meta.pop("blocked_action_id", None)
                        with brain_lock:
                            brain.update(g.id, metadata=meta)
                except (ValueError, TypeError):
                    pass

        # Auto-unblock external blockers with exponential backoff
        # Retry schedule: 1h, 4h, 16h, 48h (then give up → failed)
        if status == "blocked_external":
            blocked_at = meta.get("blocked_at", "")
            ext_retries = meta.get("external_block_retries", 0)
            if blocked_at:
                try:
                    blocked_dt = datetime.fromisoformat(blocked_at)
                    elapsed_h = (datetime.now() - blocked_dt).total_seconds() / 3600
                    # Exponential: 1h * 4^retries  → 1h, 4h, 16h, 48h
                    backoff_h = min(1 * (4 ** ext_retries), 48)
                    if elapsed_h >= backoff_h:
                        if ext_retries >= 4:
                            # Exhausted retries → fail the goal
                            logger.info(
                                "External blocker exhausted (%d retries), failing goal: %s",
                                ext_retries, g.content[:60],
                            )
                            update_goal_status(
                                g.id, "failed",
                                notes=f"External blocker persisted after {ext_retries} retries",
                            )
                            continue
                        logger.info(
                            "Auto-unblocking external blocker (retry %d, backoff %.0fh): %s",
                            ext_retries + 1, backoff_h, g.content[:60],
                        )
                        meta["status"] = "active"
                        meta["external_block_retries"] = ext_retries + 1
                        meta["blocked_at"] = None
                        meta["updated_at"] = datetime.now().isoformat()
                        with brain_lock:
                            brain.update(g.id, metadata=meta)
                except (ValueError, TypeError):
                    pass
        # Auto-archive goals with too many failed attempts
        if meta.get("attempts", 0) >= MAX_GOAL_ATTEMPTS:
            attempts = meta["attempts"]
            logger.info("Auto-archiving goal after %d attempts: %s", attempts, g.content[:80])
            update_goal_status(g.id, "failed", notes=f"Auto-failed after {attempts} attempts")

            # Emit event for Activity live stream
            event_bus.emit(
                "goal_failed",
                {
                    "goal_id": meta.get("goal_id", ""),
                    "description": g.content[:200],
                    "attempts": attempts,
                    "created_by": meta.get("created_by", "agent"),
                    "reason": f"Exceeded max attempts ({MAX_GOAL_ATTEMPTS})",
                },
            )

            # Queue Telegram notification for user-created goals
            if meta.get("created_by") == "user":
                _notify_goal_failed(g.content, attempts, meta)

            continue
        goal_dict = {
            "record_id": g.id,
            "goal_id": meta.get("goal_id", ""),
            "description": g.content,
            "priority": meta.get("priority", "medium"),
            "goal_type": meta.get("goal_type", "general"),
            "goal_template": meta.get("goal_template"),
            "created_at": meta.get("created_at", ""),
            "deadline": meta.get("deadline"),
            "attempts": meta.get("attempts", 0),
            "last_attempt": meta.get("last_attempt"),
            "parent_goal_id": meta.get("parent_goal_id"),
            "success_criteria": meta.get("success_criteria", []),
            "depends_on": meta.get("depends_on", []),
            "blocked": meta.get("status") in ("blocked_by_user", "blocked_external"),
            "block_status": meta.get("status")
            if meta.get("status", "").startswith("blocked")
            else "",
            "blocked_reason": meta.get("blocked_reason", ""),
            "blocked_evidence": meta.get("blocked_evidence", ""),
            "resume_context": meta.get("resume_context", ""),
            "immortal": bool(meta.get("immortal", False)),
            "status": meta.get("status", ""),
        }
        # Expose task metadata for atomic mission tasks
        if meta.get("mission_task_id"):
            goal_dict["mission_task_id"] = meta["mission_task_id"]
            goal_dict["mission_id"] = meta.get("mission_id", "")
            goal_dict["task_action"] = meta.get("task_action", "")
            goal_dict["task_done_when"] = meta.get("task_done_when", "")
        active.append(goal_dict)

    active = smart_sort_goals(active)

    # Blocked goals always last — they can't run until unblocked.
    non_blocked = [g for g in active if not g.get("blocked")]
    blocked = [g for g in active if g.get("blocked")]
    active = non_blocked + blocked

    # Strong runtime focus guard: when a mission has runnable atomic tasks,
    # keep unrelated legacy goals after all mission-linked goals (but before blocked).
    has_runnable_mission_tasks = any(
        g.get("mission_task_id") and not g.get("blocked") for g in active
    )
    if has_runnable_mission_tasks:
        mission_task_groups: dict[str, list[dict]] = {}
        mission_goal_groups: dict[str, list[dict]] = {}
        legacy = []
        for g in non_blocked:
            mid = g.get("mission_id", "")
            if g.get("mission_task_id") and mid:
                mission_task_groups.setdefault(mid, []).append(g)
            elif mid:
                mission_goal_groups.setdefault(mid, []).append(g)
            else:
                legacy.append(g)

        def _mission_rank(mid: str) -> tuple[int, int, float]:
            task_group = mission_task_groups.get(mid, [])
            goal_group = mission_goal_groups.get(mid, [])
            mission_records = task_group + goal_group
            immortal = any(r.get("immortal") for r in mission_records)
            active_tasks = sum(1 for r in task_group if r.get("status") == "active")
            newest_created_ts = 0.0
            for r in mission_records:
                try:
                    newest_created_ts = max(
                        newest_created_ts,
                        datetime.fromisoformat(str(r.get("created_at", "") or "")).timestamp(),
                    )
                except Exception:
                    pass
            return (1 if immortal else 0, -active_tasks, -newest_created_ts)

        ordered_missions = sorted(
            mission_task_groups.keys() | mission_goal_groups.keys(), key=_mission_rank
        )
        focus_mission_id = ordered_missions[0] if ordered_missions else ""

        focused_atomic = mission_task_groups.get(focus_mission_id, [])
        focused_parent = mission_goal_groups.get(focus_mission_id, [])
        other_mission_atomic = []
        other_mission_parent = []
        for mid in ordered_missions[1:]:
            other_mission_atomic.extend(mission_task_groups.get(mid, []))
            other_mission_parent.extend(mission_goal_groups.get(mid, []))

        active = (
            focused_atomic
            + focused_parent
            + legacy
            + other_mission_atomic
            + other_mission_parent
            + blocked
        )

    return active


def _notify_goal_completed(description: str, meta: dict, reason: str = ""):
    """Send Telegram notification when a user-created goal is completed."""
    au = _get_autonomy()
    settings = au.settings

    priority = meta.get("priority", "medium")
    created_at = meta.get("created_at", "unknown")
    reason_line = f"\n\nResult: {reason[:200]}" if reason else ""
    msg = (
        f"Goal completed\n\n"
        f"{description[:300]}"
        f"{reason_line}\n\n"
        f"Priority: {priority}\n"
        f"Created: {created_at}"
    )

    if not settings.TELEGRAM_BOT_TOKEN or not settings.PROACTIVE_CHAT_ID:
        logger.info("Goal completion report (no Telegram): %s", msg)
        return
    try:
        from remy.core.notification_router import should_notify_telegram

        if not should_notify_telegram():
            logger.info("Goal completion report suppressed (web runtime active)")
            return
    except Exception as e:
        logger.debug("Could not evaluate Telegram suppression for goal completion: %s", e)

    async def _send():
        try:
            from telegram import Bot

            bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
            await bot.send_message(chat_id=settings.PROACTIVE_CHAT_ID, text=msg)
        except Exception as e:
            logger.warning("Failed to send goal completion notification: %s", e)

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_send())
    except RuntimeError:
        logger.info("Goal completion report (no event loop): %s", msg)


def _notify_goal_failed(description: str, attempts: int, meta: dict):
    """Send Telegram notification when a user-created goal fails after max attempts."""
    au = _get_autonomy()
    settings = au.settings

    priority = meta.get("priority", "medium")
    created_at = meta.get("created_at", "unknown")
    msg = (
        f"Goal failed after {attempts} attempts\n\n"
        f"{description[:300]}\n\n"
        f"Priority: {priority}\n"
        f"Created: {created_at}\n\n"
        "The goal has been archived. You can create a new, more specific goal "
        "or break it down into smaller steps."
    )

    if not settings.TELEGRAM_BOT_TOKEN or not settings.PROACTIVE_CHAT_ID:
        logger.info("Goal failure report (no Telegram): %s", msg)
        return
    try:
        from remy.core.notification_router import should_notify_telegram

        if not should_notify_telegram():
            logger.info("Goal failure report suppressed for Telegram because web runtime is active")
            return
    except Exception as e:
        logger.debug("Could not evaluate Telegram suppression for goal failure report: %s", e)

    async def _send():
        try:
            from telegram import Bot

            bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
            await bot.send_message(chat_id=settings.PROACTIVE_CHAT_ID, text=msg)
        except Exception as e:
            logger.warning("Failed to send goal failure notification: %s", e)

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_send())
    except RuntimeError:
        # No running event loop — log only
        logger.info("Goal failure report (no event loop): %s", msg)


def update_goal_status(record_id: str, status: str, notes: str = ""):
    """Update a goal's status."""
    from remy.core.agent_tools import brain_lock

    au = _get_autonomy()
    brain = au.brain

    with brain_lock:
        rec = brain.get(record_id)
        if not rec:
            return
        meta = dict(rec.metadata or {})
        meta["status"] = status
        if notes:
            meta["status_notes"] = notes
        meta["updated_at"] = datetime.now().isoformat()
        brain.update(record_id, metadata=meta)

    # Reset stale-focus counter on progress
    mission_id = meta.get("mission_id", "")
    mission_task_id = meta.get("mission_task_id", "")
    if mission_id and status == "completed":
        try:
            from remy.core.orchestrator import mark_focus_progress
            mark_focus_progress(mission_id)
        except Exception:
            pass
    if mission_id and mission_task_id and status in ("active", "completed", "failed"):
        event_bus.emit(
            f"mission.task_{status}",
            {
                "record_id": record_id,
                "goal_id": meta.get("goal_id", ""),
                "mission_id": mission_id,
                "mission_task_id": mission_task_id,
                "notes": notes[:200] if notes else "",
            },
        )
        if status == "completed":
            try:
                from remy.core.notification_router import notify
                notify(
                    f"Task completed [{mission_task_id}]\n\n"
                    f"{(meta.get('content') or notes or '')[:250]}\n\n"
                    f"Mission: {mission_id}",
                    level="info",
                )
            except Exception as _e:
                logger.debug("Could not send task completion notify: %s", _e)
        elif status == "failed":
            try:
                from remy.core.notification_router import notify
                notify(
                    f"Task failed [{mission_task_id}]\n\n"
                    f"{notes[:250] if notes else ''}\n\n"
                    f"Mission: {mission_id}",
                    level="warning",
                )
            except Exception as _e:
                logger.debug("Could not send task failure notify: %s", _e)


def archive_goal(record_id: str, *, reason: str = "archived_by_user") -> tuple[bool, dict | None]:
    """Archive a goal so it is no longer eligible for execution.

    If the goal is a mission parent, all mission-linked child tasks are archived too.
    If the goal is a mission task, the next eligible task in that mission is activated.
    """
    from remy.core.agent_tools import brain_lock

    au = _get_autonomy()
    brain = au.brain

    with brain_lock:
        rec = brain.get(record_id)
        if not rec:
            return False, None
        meta = dict(rec.metadata or {})
        if meta.get("type") != "autonomous_goal":
            return False, None

        mission_id = meta.get("mission_id", "")
        mission_task_id = meta.get("mission_task_id", "")

        meta["status"] = "archived"
        meta["archived_at"] = datetime.now().isoformat()
        meta["archived_reason"] = reason
        meta["updated_at"] = datetime.now().isoformat()
        brain.update(record_id, metadata=meta)

        # If archiving a mission parent, archive all mission children too so the
        # mission stops producing runnable tasks.
        if mission_id and not mission_task_id:
            related = brain.search(query=mission_id, tags=GOAL_TAGS, limit=200)
            for g in related:
                if g.id == record_id:
                    continue
                child_meta = dict(g.metadata or {})
                if child_meta.get("mission_id") != mission_id:
                    continue
                if child_meta.get("status") == "archived":
                    continue
                child_meta["status"] = "archived"
                child_meta["archived_at"] = datetime.now().isoformat()
                child_meta["archived_reason"] = f"{reason}:mission_parent"
                child_meta["updated_at"] = datetime.now().isoformat()
                brain.update(g.id, metadata=child_meta)

    # If a mission task was archived, advance the queue outside the write block.
    if mission_id and mission_task_id:
        _activate_next_task(brain, brain_lock, mission_id)

    logger.info("Goal %s archived (%s)", record_id[:12], reason)
    event_bus.emit(
        "goal_archived",
        {
            "record_id": record_id,
            "goal_id": meta.get("goal_id", ""),
            "mission_id": mission_id,
            "mission_task_id": mission_task_id,
            "reason": reason,
        },
    )
    return True, meta


def block_goal(
    record_id: str,
    action_id: str = "",
    *,
    status: str = "blocked_by_user",
    reason: str = "",
    evidence: str = "",
    resume_context: str = "",
):
    """Mark a goal as blocked for a specific reason."""
    from remy.core.agent_tools import brain_lock

    au = _get_autonomy()
    brain = au.brain

    with brain_lock:
        rec = brain.get(record_id)
        if not rec:
            return
        meta = dict(rec.metadata or {})
        meta["status"] = status
        meta["blocked_at"] = datetime.now().isoformat()
        meta["blocked_action_id"] = action_id
        if reason:
            meta["blocked_reason"] = reason
            meta["status_notes"] = reason
        if evidence:
            meta["blocked_evidence"] = evidence[:500]
        if resume_context:
            meta["resume_context"] = resume_context[:500]
        meta["updated_at"] = datetime.now().isoformat()
        brain.update(record_id, metadata=meta)

    logger.info("Goal %s blocked with status=%s", record_id[:12], status)
    event_bus.emit(
        "goal_blocked",
        {
            "record_id": record_id,
            "goal_id": meta.get("goal_id", ""),
            "action_id": action_id,
            "status": status,
            "reason": reason[:200],
        },
    )


def unblock_goal(record_id: str):
    """Restore a blocked goal to active status."""
    from remy.core.agent_tools import brain_lock

    au = _get_autonomy()
    brain = au.brain

    with brain_lock:
        rec = brain.get(record_id)
        if not rec:
            return
        meta = dict(rec.metadata or {})
        if meta.get("status") not in ("blocked_by_user", "blocked_external"):
            return
        meta["status"] = "active"
        meta.pop("blocked_at", None)
        meta.pop("blocked_action_id", None)
        meta.pop("blocked_reason", None)
        meta.pop("blocked_evidence", None)
        meta.pop("resume_context", None)
        meta["updated_at"] = datetime.now().isoformat()
        brain.update(record_id, metadata=meta)

    logger.info("Goal %s unblocked (approval resolved)", record_id[:12])
    event_bus.emit(
        "goal_unblocked",
        {
            "record_id": record_id,
            "goal_id": meta.get("goal_id", ""),
        },
    )


def unblock_goal_by_action_id(action_id: str):
    """Find and unblock the goal that was blocked by a specific approval action."""
    from remy.core.agent_tools import brain_lock

    au = _get_autonomy()
    brain = au.brain

    with brain_lock:
        goals = brain.search(query="", tags=GOAL_TAGS, limit=200)

    for g in goals:
        meta = g.metadata or {}
        if (
            meta.get("status") in ("blocked_by_user", "blocked_external")
            and meta.get("blocked_action_id") == action_id
        ):
            unblock_goal(g.id)
            return True
    return False


def resume_goal_from_blocker(record_id: str, note: str = ""):
    """Resume a blocked goal while preserving what the agent should continue from."""
    from remy.core.agent_tools import brain_lock

    au = _get_autonomy()
    brain = au.brain

    with brain_lock:
        rec = brain.get(record_id)
        if not rec:
            return False
        meta = dict(rec.metadata or {})
        if meta.get("status") not in ("blocked_by_user", "blocked_external"):
            return False

        reason = meta.get("blocked_reason", "")
        evidence = meta.get("blocked_evidence", "")
        resume_bits = [bit for bit in (reason, evidence, note) if bit]
        if resume_bits:
            meta["resume_context"] = " | ".join(resume_bits)[:500]

        meta["status"] = "active"
        meta.pop("blocked_at", None)
        meta.pop("blocked_action_id", None)
        meta.pop("blocked_reason", None)
        meta.pop("blocked_evidence", None)
        meta["updated_at"] = datetime.now().isoformat()
        brain.update(record_id, metadata=meta)

    logger.info("Goal %s resumed from blocker", record_id[:12])
    event_bus.emit(
        "goal_resumed",
        {
            "record_id": record_id,
            "goal_id": meta.get("goal_id", ""),
            "resume_context": meta.get("resume_context", ""),
        },
    )
    return True


def record_goal_attempt(record_id: str):
    """Increment attempt counter and update last_attempt timestamp."""
    from remy.core.agent_tools import brain_lock

    au = _get_autonomy()
    brain = au.brain

    with brain_lock:
        rec = brain.get(record_id)
        if not rec:
            return
        meta = dict(rec.metadata or {})
        meta["attempts"] = meta.get("attempts", 0) + 1
        meta["last_attempt"] = datetime.now().isoformat()
        brain.update(record_id, metadata=meta)


AUTO_DECOMPOSE_THRESHOLD = 3


def decompose_goal(goal_record_id: str) -> list[str]:
    """Break a complex goal into 2-5 actionable sub-goals using LLM.

    Sync wrapper around decompose_goal_async().
    Returns list of created sub-goal record IDs, or [] on failure.
    """
    try:
        asyncio.get_running_loop()
        # Already in an async context
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, decompose_goal_async(goal_record_id))
            return future.result(timeout=60)
    except RuntimeError:
        # No running event loop — safe to use asyncio.run()
        return asyncio.run(decompose_goal_async(goal_record_id))


async def decompose_goal_async(goal_record_id: str) -> list[str]:
    """Break a complex goal into 2-5 actionable sub-goals using LLM."""
    from remy.core.agent_tools import brain_lock
    from remy.core.autonomy_models import _llm_content_to_str
    from remy.core.brain_tools import parse_llm_json

    au = _get_autonomy()
    brain = au.brain

    with brain_lock:
        rec = brain.get(goal_record_id)
    if not rec:
        logger.warning("Cannot decompose: goal %s not found", goal_record_id)
        return []

    meta = dict(rec.metadata or {})

    # Guard: don't decompose twice
    if meta.get("status") == "decomposed":
        logger.info("Goal %s already decomposed, skipping", goal_record_id)
        return []

    goal_description = rec.content
    goal_id = meta.get("goal_id", "")
    priority = meta.get("priority", "medium")

    # Use LLM to generate sub-goals
    decompose_prompt = (
        "Break the following goal into 2-5 smaller, actionable sub-goals.\n"
        "Respond ONLY with a JSON array of strings.\n\n"
        f"GOAL: {goal_description}\n\n"
        "Example response:\n"
        '["Sub-goal 1 description", "Sub-goal 2 description"]\n\n'
        "Sub-goals should be concrete, measurable, and achievable in a single action.\n"
        "Respond with JSON array only:"
    )

    try:
        from remy.core.llm import call_llm_async

        result = await call_llm_async(
            decompose_prompt, purpose="decompose_goal", channel="autonomous"
        )
        raw = _llm_content_to_str(result.content).strip()

        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        sub_goal_descriptions = parse_llm_json(raw)

        if not isinstance(sub_goal_descriptions, list):
            logger.warning("Decompose returned non-list: %s", type(sub_goal_descriptions))
            return []

    except Exception as e:
        logger.warning("Goal decomposition failed: %s", e)
        return []

    # Create sub-goals (inherit created_by from parent)
    parent_created_by = meta.get("created_by", "agent")
    sub_goal_ids = []
    for desc in sub_goal_descriptions[:5]:  # Max 5 sub-goals
        if not isinstance(desc, str) or not desc.strip():
            continue
        sub_id = create_goal(
            description=desc.strip(),
            priority=priority,
            parent_goal_id=goal_id,
            created_by=parent_created_by,
        )
        sub_goal_ids.append(sub_id)

    # Mark parent as decomposed
    if sub_goal_ids:
        meta["status"] = "decomposed"
        meta["sub_goal_ids"] = sub_goal_ids
        meta["decomposed_at"] = datetime.now().isoformat()
        with brain_lock:
            brain.update(goal_record_id, metadata=meta)
        logger.info(
            "Goal %s decomposed into %d sub-goals",
            goal_id,
            len(sub_goal_ids),
        )

    return sub_goal_ids


# ============== ACTION PLANS ==============

PLAN_TAGS = ["action-plan"]


@dataclass
class ActionPlan:
    """Multi-step action plan for achieving a goal."""

    plan_id: str
    goal_id: str
    goal_description: str
    steps: list[str]
    current_step: int = 0
    status: str = "active"  # active, completed, abandoned


@dataclass
class PlanNode:
    """A single node in a decision tree plan."""

    step_id: int
    description: str
    success_next: int | None = None  # step_id on success (None = plan complete)
    failure_next: int | None = None  # step_id on failure (None = retry current)
    condition: str = ""  # Optional condition description
    max_retries: int = 2
    retry_count: int = 0


@dataclass
class DecisionTreePlan:
    """Branching decision tree plan — replaces linear ActionPlan for complex goals."""

    plan_id: str
    goal_id: str
    goal_description: str
    nodes: list[PlanNode]
    current_node: int = 0  # step_id of current node
    status: str = "active"  # active, completed, abandoned
    history: list[dict] = field(default_factory=list)


async def create_plan_for_goal(
    goal_id: str, goal_description: str
) -> ActionPlan | DecisionTreePlan | None:
    """Use LLM to generate a plan for a goal.

    Tries decision tree first (branching plan), falls back to linear ActionPlan.
    """
    # Try decision tree first
    tree = await _create_decision_tree_plan(goal_id, goal_description)
    if tree is not None:
        return tree

    # Fallback: linear plan
    return await _create_linear_plan(goal_id, goal_description)


async def _create_linear_plan(goal_id: str, goal_description: str) -> ActionPlan | None:
    """Create a simple linear (sequential) plan. Original logic."""
    from remy.core.autonomy_models import _llm_content_to_str
    from remy.core.brain_tools import parse_llm_json

    plan_prompt = (
        "You are planning an action sequence for an AI agent.\n"
        "Break this goal into 2-5 concrete, ordered steps.\n"
        "Each step must be achievable in a single agent action (tool call).\n"
        "Respond ONLY with a JSON array of strings.\n\n"
        f"GOAL: {goal_description}\n\n"
        "Respond with JSON array only:"
    )

    try:
        from remy.core.llm import call_llm_async

        result = await call_llm_async(plan_prompt, purpose="create_plan")
        raw = _llm_content_to_str(result.content).strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        steps = parse_llm_json(raw)
        if not isinstance(steps, list) or len(steps) < 2:
            return None

        plan = ActionPlan(
            plan_id=f"plan-{uuid.uuid4().hex[:12]}",
            goal_id=goal_id,
            goal_description=goal_description,
            steps=[str(s).strip() for s in steps[:5]],
        )

        _save_plan(plan)
        logger.info(
            "Created linear plan %s for goal %s: %d steps",
            plan.plan_id,
            goal_id,
            len(plan.steps),
        )
        return plan

    except Exception as e:
        logger.warning("Linear plan creation failed: %s", e)
        return None


async def _create_decision_tree_plan(
    goal_id: str, goal_description: str
) -> DecisionTreePlan | None:
    """Use LLM to generate a branching decision tree plan."""
    from remy.core.autonomy_models import _llm_content_to_str
    from remy.core.brain_tools import parse_llm_json

    tree_prompt = (
        "You are planning a decision tree for an AI agent.\n"
        "Break this goal into 3-5 steps as a branching plan.\n"
        "Each step has a success path and a failure alternative.\n\n"
        "Respond ONLY with a JSON array of node objects:\n"
        '[\n  {"step_id": 0, "description": "...", '
        '"success_next": 1, "failure_next": 2, "max_retries": 2},\n'
        '  {"step_id": 1, "description": "...", '
        '"success_next": null, "failure_next": null, "max_retries": 1},\n'
        "  ...\n]\n\n"
        "Rules:\n"
        "- step_id must be sequential integers starting from 0\n"
        "- success_next=null means the plan is COMPLETE on success\n"
        "- failure_next=null means retry the same step (up to max_retries)\n"
        "- failure_next=<step_id> means take alternative path on failure\n\n"
        f"GOAL: {goal_description}\n\n"
        "Respond with JSON only:"
    )

    try:
        from remy.core.llm import call_llm_async

        result = await call_llm_async(
            tree_prompt, purpose="create_decision_tree", channel="autonomous"
        )
        raw = _llm_content_to_str(result.content).strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        nodes_raw = parse_llm_json(raw)
        if not isinstance(nodes_raw, list) or len(nodes_raw) < 2:
            return None

        # Parse and validate nodes
        nodes = []
        seen_ids = set()
        for n in nodes_raw[:7]:  # Cap at 7 nodes
            if not isinstance(n, dict) or "step_id" not in n or "description" not in n:
                return None
            sid = int(n["step_id"])
            if sid in seen_ids:
                return None  # Duplicate step_id
            seen_ids.add(sid)
            nodes.append(
                PlanNode(
                    step_id=sid,
                    description=str(n["description"]).strip(),
                    success_next=n.get("success_next"),
                    failure_next=n.get("failure_next"),
                    condition=str(n.get("condition", "")),
                    max_retries=int(n.get("max_retries", 2)),
                )
            )

        if len(nodes) < 2:
            return None

        # Validate references — all next pointers must reference existing step_ids or None
        valid_ids = {n.step_id for n in nodes}
        for n in nodes:
            if n.success_next is not None and n.success_next not in valid_ids:
                return None
            if n.failure_next is not None and n.failure_next not in valid_ids:
                return None

        plan = DecisionTreePlan(
            plan_id=f"tree-{uuid.uuid4().hex[:12]}",
            goal_id=goal_id,
            goal_description=goal_description,
            nodes=nodes,
            current_node=nodes[0].step_id,
        )

        _save_plan(plan)
        logger.info(
            "Created decision tree %s for goal %s: %d nodes",
            plan.plan_id,
            goal_id,
            len(plan.nodes),
        )
        return plan

    except Exception as e:
        logger.warning("Decision tree creation failed, will fall back to linear: %s", e)
        return None


def _save_plan(plan: ActionPlan | DecisionTreePlan):
    """Persist a plan (linear or decision tree) in brain."""
    from remy.core.agent_tools import Level, brain_lock

    au = _get_autonomy()
    brain = au.brain

    is_tree = isinstance(plan, DecisionTreePlan)

    if is_tree:
        node_idx = _find_node_index(plan, plan.current_node)
        current_desc = plan.nodes[node_idx].description if node_idx is not None else "?"
        content = (
            f"Plan: {plan.goal_description}\n"
            f"Type: decision_tree ({len(plan.nodes)} nodes)\n"
            f"Current: node {plan.current_node} — {current_desc}"
        )
    else:
        content = (
            f"Plan: {plan.goal_description}\n"
            f"Steps: {json.dumps(plan.steps)}\n"
            f"Current: step {plan.current_step + 1}/{len(plan.steps)}"
        )

    # Metadata for storage
    if is_tree:
        save_meta = {
            "type": "action_plan",
            "plan_type": "decision_tree",
            "plan_id": plan.plan_id,
            "goal_id": plan.goal_id,
            "nodes": [
                {
                    "step_id": n.step_id,
                    "description": n.description,
                    "success_next": n.success_next,
                    "failure_next": n.failure_next,
                    "condition": n.condition,
                    "max_retries": n.max_retries,
                    "retry_count": n.retry_count,
                }
                for n in plan.nodes
            ],
            "current_node": plan.current_node,
            "status": plan.status,
            "history": plan.history[-20:],  # Cap history
        }
    else:
        save_meta = {
            "type": "action_plan",
            "plan_id": plan.plan_id,
            "goal_id": plan.goal_id,
            "steps": plan.steps,
            "current_step": plan.current_step,
            "status": plan.status,
        }

    # Check if plan already exists (update)
    with brain_lock:
        existing = brain.search(query="", tags=PLAN_TAGS, limit=50)
        for rec in existing:
            meta = rec.metadata or {}
            if meta.get("plan_id") == plan.plan_id:
                brain.update(
                    rec.id,
                    content=content,
                    metadata={
                        **save_meta,
                        "updated_at": datetime.now().isoformat(),
                    },
                )
                return

        # New plan
        brain.store(
            content=content,
            level=Level.DECISIONS,
            tags=PLAN_TAGS,
            metadata={
                **save_meta,
                "created_at": datetime.now().isoformat(),
                "source": "agent-autonomous",
                "verified": False,
                "trust_score": 0.4,
            },
            auto_promote=False,
        )


def load_plan_for_goal(goal_id: str) -> ActionPlan | DecisionTreePlan | None:
    """Load an active plan for a goal from brain. Detects plan_type automatically."""
    from remy.core.agent_tools import brain_lock

    au = _get_autonomy()
    brain = au.brain

    with brain_lock:
        plans = brain.search(query="", tags=PLAN_TAGS, limit=50)
    for rec in plans:
        meta = rec.metadata or {}
        if meta.get("goal_id") == goal_id and meta.get("status") == "active":
            plan_type = meta.get("plan_type", "")
            goal_desc = rec.content.split("\n")[0].replace("Plan: ", "")

            if plan_type == "decision_tree":
                # Restore decision tree
                nodes_raw = meta.get("nodes", [])
                nodes = []
                for n in nodes_raw:
                    nodes.append(
                        PlanNode(
                            step_id=n["step_id"],
                            description=n["description"],
                            success_next=n.get("success_next"),
                            failure_next=n.get("failure_next"),
                            condition=n.get("condition", ""),
                            max_retries=n.get("max_retries", 2),
                            retry_count=n.get("retry_count", 0),
                        )
                    )
                if not nodes:
                    continue
                return DecisionTreePlan(
                    plan_id=meta.get("plan_id", ""),
                    goal_id=goal_id,
                    goal_description=goal_desc,
                    nodes=nodes,
                    current_node=meta.get("current_node", nodes[0].step_id),
                    status=meta.get("status", "active"),
                    history=meta.get("history", []),
                )
            else:
                # Original linear plan
                return ActionPlan(
                    plan_id=meta.get("plan_id", ""),
                    goal_id=goal_id,
                    goal_description=goal_desc,
                    steps=meta.get("steps", []),
                    current_step=meta.get("current_step", 0),
                    status=meta.get("status", "active"),
                )
    return None


def _find_node_index(plan: DecisionTreePlan, step_id: int) -> int | None:
    """Find the index of a node by step_id."""
    for i, n in enumerate(plan.nodes):
        if n.step_id == step_id:
            return i
    return None


def _get_node(plan: DecisionTreePlan, step_id: int) -> PlanNode | None:
    """Get a node by step_id."""
    idx = _find_node_index(plan, step_id)
    return plan.nodes[idx] if idx is not None else None


def advance_plan(plan: ActionPlan | DecisionTreePlan, success: bool) -> str | None:
    """Advance or retry a plan step. Returns the next step description or None if done."""
    if isinstance(plan, DecisionTreePlan):
        return _advance_decision_tree(plan, success)

    # Linear plan logic (original)
    if success:
        plan.current_step += 1
        if plan.current_step >= len(plan.steps):
            plan.status = "completed"
            _save_plan(plan)
            logger.info("Plan %s completed!", plan.plan_id)
            return None
    # On failure, stay on the same step (retry)

    _save_plan(plan)
    if plan.steps and plan.current_step < len(plan.steps):
        return plan.steps[plan.current_step]
    return None


def _advance_decision_tree(plan: DecisionTreePlan, success: bool) -> str | None:
    """Navigate a decision tree plan based on outcome."""
    node = _get_node(plan, plan.current_node)
    if node is None:
        plan.status = "abandoned"
        _save_plan(plan)
        return None

    # Record history
    plan.history.append(
        {
            "step_id": node.step_id,
            "description": node.description[:100],
            "success": success,
            "timestamp": datetime.now().isoformat(),
        }
    )

    if success:
        if node.success_next is None:
            # Plan complete!
            plan.status = "completed"
            _save_plan(plan)
            logger.info("Decision tree %s completed!", plan.plan_id)
            return None
        # Move to success path
        plan.current_node = node.success_next
        _save_plan(plan)
        next_node = _get_node(plan, plan.current_node)
        return next_node.description if next_node else None
    else:
        # Failure: retry or take alternative path
        node.retry_count += 1
        if node.retry_count <= node.max_retries:
            # Retry the same step
            _save_plan(plan)
            return node.description
        elif node.failure_next is not None:
            # Take failure/alternative path
            plan.current_node = node.failure_next
            _save_plan(plan)
            next_node = _get_node(plan, plan.current_node)
            return next_node.description if next_node else None
        else:
            # No alternative, max retries exhausted → abandon
            plan.status = "abandoned"
            _save_plan(plan)
            logger.warning(
                "Decision tree %s abandoned: step %d exhausted retries", plan.plan_id, node.step_id
            )
            return None


def _format_plan_text(plan: ActionPlan | DecisionTreePlan) -> str:
    """Format plan context for the decision prompt."""
    if isinstance(plan, DecisionTreePlan):
        node = _get_node(plan, plan.current_node)
        if node is None:
            return ""
        total = len(plan.nodes)
        lines = [f"\nACTION PLAN (decision tree, node {node.step_id}/{total - 1}):"]
        lines.append(f'CURRENT: "{node.description}"')
        # Show branches
        if node.success_next is not None:
            sn = _get_node(plan, node.success_next)
            if sn:
                lines.append(f'  → On success: "{sn.description}" (node {sn.step_id})')
        else:
            lines.append("  → On success: PLAN COMPLETE")
        if node.failure_next is not None:
            fn = _get_node(plan, node.failure_next)
            if fn:
                lines.append(f'  → On failure: "{fn.description}" (node {fn.step_id})')
        else:
            retries_left = node.max_retries - node.retry_count
            lines.append(f"  → On failure: retry ({retries_left} retries left)")
        return "\n".join(lines) + "\n"
    else:
        # Linear plan
        if not plan.steps or plan.current_step >= len(plan.steps):
            return ""
        step_num = plan.current_step + 1
        total = len(plan.steps)
        current_step_desc = plan.steps[plan.current_step]
        return (
            f"\nACTION PLAN (step {step_num}/{total}):\n"
            f"YOUR CURRENT STEP: {current_step_desc}\n"
            f"Full plan: {' → '.join(plan.steps)}\n"
        )
