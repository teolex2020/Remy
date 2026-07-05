"""
Autonomous Agent Mode - goal-driven action loop with resource awareness.

Runs Remy as an autonomous agent that pursues its own goals, learns from outcomes,
and manages its own resource budget. Built on existing invoke_agent() + execute_tool().

Usage: remy --autonomous
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from remy.config.settings import settings
from remy.core.agent_tools import brain
from remy.core.brain_tools import (
    estimate_tokens,
    get_active_research_projects,
    parse_llm_json,
    tool_health,
)
from remy.core.event_bus import event_bus
from remy.core.notification_router import should_notify_telegram
from remy.core.autonomy_outcomes import (
    ActionRecord,
    recall_similar_outcomes,
    record_outcome,
)

logger = logging.getLogger("Autonomy")

# ============== BRAIN MONITOR ==============

def _write_brain_snapshot(session_id: str, event: str) -> None:
    """Write a brain state snapshot to brain_monitor.log.

    Called on session START and STOP so we can track daily changes.
    event: 'START' or 'STOP'
    """
    try:
        from remy.core.agent_tools import brain as _brain
        from remy.config.settings import settings as _settings
        import json as _json

        aura = _brain._aura
        stats = aura.stats()

        # Per-tag breakdown (top tags)
        all_records = _brain.list_records(limit=1000)
        tag_counts: dict[str, int] = {}
        level_counts: dict[str, int] = {}
        for r in all_records:
            level_name = str(getattr(r, "level", "unknown"))
            level_counts[level_name] = level_counts.get(level_name, 0) + 1
            for t in (getattr(r, "tags", None) or []):
                tag_counts[t] = tag_counts.get(t, 0) + 1

        top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:20]

        snapshot = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "session": session_id,
            "event": event,
            "total": stats.get("total_records", len(all_records)),
            "connections": stats.get("total_connections", 0),
            "levels": level_counts,
            "top_tags": dict(top_tags),
        }

        log_path = _settings.DATA_DIR / "logs" / "brain_monitor.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(snapshot, ensure_ascii=False) + "\n")

    except Exception as e:  # noqa: BLE001
        logger.warning("brain_monitor snapshot failed: %s", e)


def _llm_content_to_str(content) -> str:
    """Extract text from LLM result.content (may be str, list of blocks, or None)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, str):
                parts.append(c)
            elif hasattr(c, "text"):
                parts.append(c.text)
            elif isinstance(c, dict) and "text" in c:
                parts.append(c["text"])
            else:
                parts.append(str(c))
        return " ".join(parts)
    return str(content)


def _extract_usage_tokens(result) -> int:
    """Extract provider-reported total tokens from an LLM result if available."""
    try:
        meta = getattr(result, "response_metadata", None) or {}
        usage = meta.get("token_usage", {}) or {}
        total = int(usage.get("total_tokens", 0) or 0)
        return total if total > 0 else 0
    except Exception:
        return 0


def _extract_history_usage_tokens(messages) -> int:
    """Extract provider-reported tokens from the most recent AI message in history."""
    try:
        for msg in reversed(messages or []):
            total = _extract_usage_tokens(msg)
            if total > 0:
                return total
    except Exception:
        pass
    return 0


# _parse_llm_json moved to brain_tools.parse_llm_json (shared utility)
_parse_llm_json = parse_llm_json


# ============== AUTONOMY LOG (separate file) ==============


def _setup_autonomy_logger():
    """Set up a dedicated log file for autonomous mode actions."""
    from remy.core.logging_config import setup_autonomy_file_handler
    log_dir = settings.DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_autonomy_file_handler(logger, log_dir)


# ============== BUDGET PERSISTENCE ==============

BUDGET_FILE = "autonomy_budget.json"


def _get_budget_path() -> Path:
    return settings.DATA_DIR / BUDGET_FILE


def save_budget(budget: "ResourceBudget"):
    """Persist budget state to disk."""
    path = _get_budget_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "tokens_today": budget.tokens_today,
        "tokens_this_hour": budget.tokens_this_hour,
        "last_hour_reset": budget.last_hour_reset,
        "last_day_reset": budget.last_day_reset,
        "total_tokens_lifetime": budget.total_tokens_lifetime,
        "saved_at": time.time(),
    }
    from remy.core.file_utils import atomic_write
    atomic_write(path, json.dumps(data, indent=2))


def load_budget(budget: "ResourceBudget"):
    """Restore budget state from disk. Respects hourly/daily reset windows."""
    path = _get_budget_path()
    if not path.exists():
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read budget file, starting fresh")
        return

    now = time.time()

    # Restore lifetime total (always)
    budget.total_tokens_lifetime = data.get("total_tokens_lifetime", 0)

    # Restore daily counter if within same day window
    last_day = data.get("last_day_reset", 0)
    if now - last_day < 86400:
        budget.tokens_today = data.get("tokens_today", 0)
        budget.last_day_reset = last_day
    else:
        budget.last_day_reset = now

    # Restore hourly counter if within same hour window
    last_hour = data.get("last_hour_reset", 0)
    if now - last_hour < 3600:
        budget.tokens_this_hour = data.get("tokens_this_hour", 0)
        budget.last_hour_reset = last_hour
    else:
        budget.last_hour_reset = now

    logger.info(
        "Budget restored: %d today, %d this hour, %d lifetime",
        budget.tokens_today, budget.tokens_this_hour, budget.total_tokens_lifetime,
    )


# ============== GOAL CLEANUP ==============


def archive_completed_goals():
    """Compatibility wrapper to the extracted goal system."""
    from remy.core.autonomy_goals import archive_completed_goals as _archive_completed_goals

    return _archive_completed_goals()


def _archive_completed_goals_inner():
    goals = brain.search(query="", tags=GOAL_TAGS, limit=100)
    archived = 0
    cutoff = (datetime.now() - timedelta(hours=24)).isoformat()

    for g in goals:
        meta = g.metadata or {}
        status = meta.get("status", "active")
        if status not in ("completed", "failed"):
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


# ============== RESOURCE BUDGET ==============


@dataclass
class ResourceBudget:
    """Tracks token usage and enforces budget caps."""

    daily_limit: int
    hourly_limit: int
    session_limit: int
    tokens_today: int = 0
    tokens_this_hour: int = 0
    tokens_this_session: int = 0
    last_hour_reset: float = field(default_factory=time.time)
    last_day_reset: float = field(default_factory=time.time)
    total_tokens_lifetime: int = 0

    def can_spend(self, estimated_tokens: int = 1000) -> tuple[bool, str]:
        """Check if spending estimated_tokens is within budget."""
        now = time.time()

        # Reset hourly counter
        if now - self.last_hour_reset > 3600:
            self.tokens_this_hour = 0
            self.last_hour_reset = now

        # Reset daily counter
        if now - self.last_day_reset > 86400:
            self.tokens_today = 0
            self.last_day_reset = now

        if self.tokens_this_session + estimated_tokens > self.session_limit:
            return False, f"Session limit reached ({self.tokens_this_session}/{self.session_limit})"
        if self.tokens_this_hour + estimated_tokens > self.hourly_limit:
            return False, f"Hourly limit reached ({self.tokens_this_hour}/{self.hourly_limit})"
        if self.tokens_today + estimated_tokens > self.daily_limit:
            return False, f"Daily limit reached ({self.tokens_today}/{self.daily_limit})"
        return True, "ok"

    def record_usage(self, tokens: int):
        """Record token usage after an action."""
        self.tokens_today += tokens
        self.tokens_this_hour += tokens
        self.tokens_this_session += tokens
        self.total_tokens_lifetime += tokens

    def to_dict(self) -> dict:
        return {
            "daily_limit": self.daily_limit,
            "hourly_limit": self.hourly_limit,
            "session_limit": self.session_limit,
            "tokens_today": self.tokens_today,
            "tokens_this_hour": self.tokens_this_hour,
            "tokens_this_session": self.tokens_this_session,
            "total_tokens_lifetime": self.total_tokens_lifetime,
        }


# ============== GOAL SYSTEM ==============

GOAL_TAGS = ["autonomous-goal"]


def create_goal(
    description: str,
    priority: str = "medium",
    deadline: str | None = None,
    parent_goal_id: str | None = None,
    created_by: str = "agent",
) -> str:
    """Compatibility wrapper to the extracted goal system."""
    from remy.core.autonomy_goals import create_goal as _create_goal

    return _create_goal(
        description=description,
        priority=priority,
        deadline=deadline,
        parent_goal_id=parent_goal_id,
        created_by=created_by,
    )


def get_active_goals() -> list[dict]:
    """Compatibility wrapper to the extracted goal system."""
    from remy.core.autonomy_goals import get_active_goals as _get_active_goals

    return _get_active_goals()


def _notify_goal_failed(description: str, attempts: int, meta: dict):
    """Send Telegram notification when a user-created goal fails after max attempts.

    Runs async send in a fire-and-forget task if an event loop is running,
    otherwise logs the failure report.
    """
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
    if not should_notify_telegram():
        logger.info("Goal failure report (no Telegram): %s", msg)
        return

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
        # No running event loop - log only
        logger.info("Goal failure report (no event loop): %s", msg)


def _notify_goal_completed(description: str, meta: dict, reason: str = ""):
    """Send Telegram notification when a goal is completed."""
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
    if not should_notify_telegram():
        logger.info("Goal completion report suppressed (web runtime active)")
        return

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


def update_goal_status(record_id: str, status: str, notes: str = ""):
    """Compatibility wrapper to the extracted goal system."""
    from remy.core.autonomy_goals import update_goal_status as _update_goal_status

    _update_goal_status(record_id, status, notes)


def record_goal_attempt(record_id: str):
    """Compatibility wrapper to the extracted goal system."""
    from remy.core.autonomy_goals import record_goal_attempt as _record_goal_attempt

    _record_goal_attempt(record_id)


AUTO_DECOMPOSE_THRESHOLD = 3


def decompose_goal(goal_record_id: str) -> list[str]:
    """Break a complex goal into 2-5 actionable sub-goals using LLM.

    Returns list of created sub-goal record IDs.
    Guards against decomposing the same goal twice.
    """
    from remy.core.agent_tools import brain_lock
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
        from remy.core.llm import call_llm

        result = call_llm(decompose_prompt, purpose="decompose_goal")
        raw = _llm_content_to_str(result.content).strip()

        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        sub_goal_descriptions = _parse_llm_json(raw)

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
            goal_id, len(sub_goal_ids),
        )

    return sub_goal_ids


# ============== F1: AGENT ROLES (MULTI-AGENT DELEGATION) ==============


@dataclass
class AgentRole:
    """Defines a specialized agent role for autonomous delegation."""

    name: str
    description: str
    priority_tools: list[str]
    avoid_tools: list[str]
    instruction_suffix: str
    max_tool_iterations: int = 15


AGENT_ROLES: dict[str, AgentRole] = {
    "researcher": AgentRole(
        name="researcher",
        description="Deep investigation and knowledge gathering",
        priority_tools=[
            "recall", "recall_knowledge", "web_search", "start_research",
            "add_research_finding", "complete_research", "store_research", "extract_facts",
        ],
        avoid_tools=["write_file", "sandbox_create_tool"],
        instruction_suffix=(
            "ROLE: RESEARCHER\n"
            "You are in research mode. Gather, verify, and synthesize information.\n"
            "- Start with recall + recall_knowledge (free) before web_search (expensive)\n"
            "- Use start_research for multi-query investigations\n"
            "- Cross-reference findings against existing knowledge\n"
            "- Store conclusions via store_research or store_knowledge\n"
            "- Do NOT create files, tools, or goals - just gather knowledge\n"
        ),
        max_tool_iterations=10,
    ),
    "planner": AgentRole(
        name="planner",
        description="Goal creation, decomposition, and plan management",
        priority_tools=[
            "recall", "search", "create_subgoal", "complete_goal",
            "add_todo", "list_todos", "update_todo", "schedule_task",
            "delegate_task",
        ],
        avoid_tools=["web_search", "http_get", "write_file"],
        instruction_suffix=(
            "ROLE: PLANNER\n"
            "You are in planning mode. Organize, prioritize, and structure work.\n"
            "- Review existing goals and todos with recall + list_todos\n"
            "- Break complex goals into sub-goals with create_subgoal\n"
            "- Create action items with add_todo (category='agent')\n"
            "- Complete finished goals with complete_goal\n"
            "- Do NOT execute actions or do research - just plan\n"
        ),
        max_tool_iterations=8,
    ),
    "executor": AgentRole(
        name="executor",
        description="Execute concrete plan steps and external actions",
        priority_tools=[
            "read_file", "write_file", "http_get", "list_directory",
            "browse_page", "browser_act", "browser_close",
            "store", "update_record", "connect_records", "update_todo",
            "delegate_task",
        ],
        avoid_tools=["start_research", "create_subgoal"],
        instruction_suffix=(
            "ROLE: EXECUTOR\n"
            "You are in execution mode. Carry out concrete actions.\n"
            "- Follow the ACTION PLAN step precisely\n"
            "- Use read_file/write_file for file operations\n"
            "- Use browse_page/browser_act for web interactions (forms, JS-rendered pages)\n"
            "- Use http_get ONLY for simple API calls, NOT for web pages\n"
            "- Store results in brain after each action\n"
            "- Update todos as you complete steps\n"
            "- Report exact outcomes: what was done, what changed\n"
            "- Do NOT plan or research - execute the plan step\n"
        ),
        max_tool_iterations=12,
    ),
    "analyst": AgentRole(
        name="analyst",
        description="Data analysis, metric intelligence, pattern detection",
        priority_tools=[
            "metric_summary", "event_correlate", "extract_facts",
            "recall", "recall_knowledge", "search", "consolidate",
        ],
        avoid_tools=["web_search", "write_file", "http_get", "sandbox_create_tool"],
        instruction_suffix=(
            "ROLE: ANALYST\n"
            "You are in analysis mode. Find patterns and derive insights.\n"
            "- Use metric_summary and event_correlate for tracked metrics and events\n"
            "- Use extract_facts to structure knowledge from text\n"
            "- Use recall + search to cross-reference related information\n"
            "- Use consolidate to clean up redundant records\n"
            "- Produce clear, structured findings\n"
            "- Do NOT gather new data - analyze what exists\n"
        ),
        max_tool_iterations=8,
    ),
    "osint": AgentRole(
        name="osint",
        description="Open-source intelligence: market research, competitive analysis, lead discovery",
        priority_tools=[
            "recall", "recall_knowledge", "web_search", "extract_content",
            "http_get", "store", "search", "start_research",
            "add_research_finding", "complete_research", "store_research",
            "scratchpad",
        ],
        avoid_tools=["write_file", "sandbox_create_tool", "browse_page", "browser_act"],
        instruction_suffix=(
            "ROLE: OSINT INVESTIGATOR\n"
            "You are in intelligence-gathering mode. Find, verify, and synthesize external signals.\n"
            "- Start with recall + recall_knowledge before web_search\n"
            "- Use web_search and extract_content for competitive analysis and market research\n"
            "- Track source freshness and distinguish facts from inferences\n"
            "- Use start_research/add_research_finding/complete_research for multi-step investigations\n"
            "- Store high-signal findings in brain memory with cautious trust\n"
            "- Do NOT create files or execute browser automation unless explicitly re-routed\n"
        ),
        max_tool_iterations=10,
    ),
}


# ============== ACTION PLANS ==============

PLAN_TAGS = ["action-plan"]


PLAN_REVISION_THRESHOLD = 2  # consecutive failures on same step before revision


def _plan_consequence_policy_block(goal_description: str, failed_history: list[dict]) -> str:
    """Render lived consequence policy hints for failed plan steps."""
    try:
        from remy.core.consequence_gate import consult_policy_hint
    except Exception:
        return "- (consequence memory unavailable)"

    store = getattr(brain, "_aura", brain)
    lines: list[str] = []
    seen: set[str] = set()
    for entry in failed_history[-5:]:
        step = str(entry.get("step", "")).strip()
        if not step or step in seen:
            continue
        seen.add(step)
        hint = None
        try:
            for namespace in ("remy-autonomy", None):
                candidate = consult_policy_hint(store, goal_description, step, namespace=namespace)
                refutes = int(getattr(candidate, "refutes", 0) or 0)
                supports = int(getattr(candidate, "supports", 0) or 0)
                verdict = str(getattr(candidate, "verdict", "") or "inconclusive")
                if refutes > 0 or supports > 0 or verdict != "inconclusive":
                    hint = candidate
                    break
        except Exception:
            continue
        if hint is None:
            continue

        policy = getattr(hint, "hint", "verify_first")
        refutes = int(getattr(hint, "refutes", 0) or 0)
        supports = int(getattr(hint, "supports", 0) or 0)
        reason = str(getattr(hint, "reason", "") or "").strip()
        if policy not in {"avoid", "requires_evidence", "verify_first"} and refutes <= 0:
            continue
        if policy == "verify_first" and refutes <= 0 and not getattr(hint, "requires_evidence", False):
            continue

        suffix = f" reason: {reason}" if reason else ""
        lines.append(
            f"- {step}: policy={policy}, refutes={refutes}, supports={supports}.{suffix}"
        )

    return "\n".join(lines) if lines else "- (no long-term consequence hints for failed steps)"


def _store_plan_step_consequence(
    plan: "ActionPlan | DecisionTreePlan",
    step: str,
    *,
    success: bool,
    reason: str = "",
) -> None:
    """Persist autonomous plan step outcomes as lived consequence memory."""
    step = (step or "").strip()
    situation = getattr(plan, "goal_description", "") or ""
    if not step or not situation:
        return

    store = getattr(brain, "_aura", brain)
    capture = getattr(store, "capture_consequence", None)
    if capture is None:
        return

    consequence = "SUPPORTS" if success else "REFUTES"
    trust = 1 if success else -1
    try:
        capture(
            situation=situation,
            action=step[:240],
            consequence=consequence,
            trust=trust,
            scope=[
                "autonomous-plan-step",
                f"plan:{getattr(plan, 'plan_id', '')}",
                f"goal:{getattr(plan, 'goal_id', '')}",
                "result:success" if success else "result:failure",
            ],
            provenance=[
                "remy:autonomy_advance_plan",
                f"plan:{getattr(plan, 'plan_id', '')}",
            ],
            links={
                "goal_id": getattr(plan, "goal_id", ""),
                "plan_id": getattr(plan, "plan_id", ""),
                "reason": reason,
            },
            namespace="remy-autonomy",
        )
    except TypeError:
        try:
            capture(situation, step[:240], consequence, trust)
        except Exception:
            pass
    except Exception:
        pass

@dataclass
class ActionPlan:
    """Multi-step action plan for achieving a goal."""

    plan_id: str
    goal_id: str
    goal_description: str
    steps: list[str]
    current_step: int = 0
    status: str = "active"  # active, completed, abandoned
    consecutive_step_failures: int = 0
    failed_step_history: list[dict] = field(default_factory=list)


@dataclass
class PlanNode:
    """A single node in a decision tree plan."""

    step_id: int
    description: str
    success_next: int | None = None   # step_id on success (None = plan complete)
    failure_next: int | None = None   # step_id on failure (None = retry current)
    condition: str = ""               # Optional condition description
    max_retries: int = 2
    retry_count: int = 0


@dataclass
class DecisionTreePlan:
    """Branching decision tree plan - replaces linear ActionPlan for complex goals."""

    plan_id: str
    goal_id: str
    goal_description: str
    nodes: list[PlanNode]
    current_node: int = 0         # step_id of current node
    status: str = "active"        # active, completed, abandoned
    history: list[dict] = field(default_factory=list)


def create_plan_for_goal(goal_id: str, goal_description: str) -> ActionPlan | DecisionTreePlan | None:
    """Use LLM to generate a plan for a goal.

    Tries decision tree first (branching plan), falls back to linear ActionPlan.
    """
    # Try decision tree first
    tree = _create_decision_tree_plan(goal_id, goal_description)
    if tree is not None:
        return tree

    # Fallback: linear plan
    return _create_linear_plan(goal_id, goal_description)


def _revise_plan(old_plan: ActionPlan) -> ActionPlan | None:
    """Regenerate a plan when the current step keeps failing.

    Passes failed step history to LLM so it avoids repeating the same approach.
    """
    failed_steps = [
        f"- {e.get('step', '')} (failed: {e.get('reason', 'unknown')})"
        for e in old_plan.failed_step_history[-5:]
    ]
    failed_block = "\n".join(failed_steps) if failed_steps else "- (no details recorded)"
    consequence_block = _plan_consequence_policy_block(
        old_plan.goal_description,
        old_plan.failed_step_history,
    )

    plan_prompt = (
        "You are replanning an action sequence for an AI agent.\n"
        "The previous plan failed repeatedly. Create a NEW plan that avoids the same approaches.\n"
        "Use long-term consequence memory as a hard planning signal: avoid actions marked avoid, "
        "and add verification work before actions marked requires_evidence or verify_first.\n"
        "Break the goal into 2-5 concrete, ordered steps.\n"
        "Each step must be achievable in a single agent action (tool call).\n"
        "Respond ONLY with a JSON array of strings.\n\n"
        f"GOAL: {old_plan.goal_description}\n\n"
        f"PREVIOUSLY FAILED STEPS (do not repeat these approaches):\n{failed_block}\n\n"
        f"LONG-TERM CONSEQUENCE MEMORY:\n{consequence_block}\n\n"
        "Respond with JSON array only:"
    )

    try:
        from remy.core.llm import call_llm

        result = call_llm(plan_prompt, purpose="revise_plan")
        raw = _llm_content_to_str(result.content).strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        steps = _parse_llm_json(raw)
        if not isinstance(steps, list) or len(steps) < 2:
            return None

        new_plan = ActionPlan(
            plan_id=f"plan-{uuid.uuid4().hex[:12]}",
            goal_id=old_plan.goal_id,
            goal_description=old_plan.goal_description,
            steps=[str(s).strip() for s in steps[:5]],
            failed_step_history=old_plan.failed_step_history,  # carry history forward
        )
        _save_plan(new_plan)
        logger.info("Plan revised for goal %s: %d new steps", old_plan.goal_id, len(new_plan.steps))
        return new_plan
    except Exception as e:
        logger.warning("Plan revision failed: %s", e)
        return None


def _create_linear_plan(goal_id: str, goal_description: str) -> ActionPlan | None:
    """Create a simple linear (sequential) plan. Original logic."""
    plan_prompt = (
        "You are planning an action sequence for an AI agent.\n"
        "Break this goal into 2-5 concrete, ordered steps.\n"
        "Each step must be achievable in a single agent action (tool call).\n"
        "Respond ONLY with a JSON array of strings.\n\n"
        f"GOAL: {goal_description}\n\n"
        "Respond with JSON array only:"
    )

    try:
        from remy.core.llm import call_llm

        result = call_llm(plan_prompt, purpose="create_plan")
        raw = _llm_content_to_str(result.content).strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        steps = _parse_llm_json(raw)
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
            plan.plan_id, goal_id, len(plan.steps),
        )
        return plan

    except Exception as e:
        logger.warning("Linear plan creation failed: %s", e)
        return None


def _create_decision_tree_plan(goal_id: str, goal_description: str) -> DecisionTreePlan | None:
    """Use LLM to generate a branching decision tree plan."""
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
        from remy.core.llm import call_llm

        result = call_llm(tree_prompt, purpose="create_decision_tree")
        raw = _llm_content_to_str(result.content).strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        nodes_raw = _parse_llm_json(raw)
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
            nodes.append(PlanNode(
                step_id=sid,
                description=str(n["description"]).strip(),
                success_next=n.get("success_next"),
                failure_next=n.get("failure_next"),
                condition=str(n.get("condition", "")),
                max_retries=int(n.get("max_retries", 2)),
            ))

        if len(nodes) < 2:
            return None

        # Validate references - all next pointers must reference existing step_ids or None
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
            plan.plan_id, goal_id, len(plan.nodes),
        )
        return plan

    except Exception as e:
        logger.warning("Decision tree creation failed, will fall back to linear: %s", e)
        return None


def _save_plan(plan: ActionPlan | DecisionTreePlan):
    """Persist a plan (linear or decision tree) in brain."""
    from remy.core.agent_tools import Level
    from remy.core.agent_tools import brain_lock

    is_tree = isinstance(plan, DecisionTreePlan)

    if is_tree:
        node_idx = _find_node_index(plan, plan.current_node)
        current_desc = plan.nodes[node_idx].description if node_idx is not None else "?"
        content = (
            f"Plan: {plan.goal_description}\n"
            f"Type: decision_tree ({len(plan.nodes)} nodes)\n"
            f"Current: node {plan.current_node} - {current_desc}"
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
            "consecutive_step_failures": plan.consecutive_step_failures,
            "failed_step_history": plan.failed_step_history[-10:],
        }

    # Check if plan already exists (update)
    with brain_lock:
        existing = brain.search(query="", tags=PLAN_TAGS, limit=50)
        for rec in existing:
            meta = rec.metadata or {}
            if meta.get("plan_id") == plan.plan_id:
                brain.update(rec.id, content=content, metadata={
                    **save_meta,
                    "updated_at": datetime.now().isoformat(),
                })
                return

        # New plan
        brain.store(
            content=content,
            level=Level.DOMAIN,
            tags=PLAN_TAGS,
            metadata={
                **save_meta,
                "created_at": datetime.now().isoformat(),
                "source": "agent-autonomous",
                "verified": False,
                "trust_score": 0.4,
                "admission_class": "plan",
            },
        )


def load_plan_for_goal(goal_id: str) -> ActionPlan | DecisionTreePlan | None:
    """Load an active plan for a goal from brain. Detects plan_type automatically."""
    from remy.core.agent_tools import brain_lock
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
                    nodes.append(PlanNode(
                        step_id=n["step_id"],
                        description=n["description"],
                        success_next=n.get("success_next"),
                        failure_next=n.get("failure_next"),
                        condition=n.get("condition", ""),
                        max_retries=n.get("max_retries", 2),
                        retry_count=n.get("retry_count", 0),
                    ))
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
                    consecutive_step_failures=meta.get("consecutive_step_failures", 0),
                    failed_step_history=meta.get("failed_step_history", []),
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
    """Advance or retry a plan step. Returns the next step description or None if done.

    Handles both linear ActionPlan and branching DecisionTreePlan.
    """
    if isinstance(plan, DecisionTreePlan):
        return _advance_decision_tree(plan, success)

    # Linear plan logic (original)
    if success:
        current_step_desc = plan.steps[plan.current_step] if plan.current_step < len(plan.steps) else ""
        _store_plan_step_consequence(
            plan,
            current_step_desc,
            success=True,
            reason="step advanced successfully",
        )
        plan.current_step += 1
        plan.consecutive_step_failures = 0  # reset on success
        if plan.current_step >= len(plan.steps):
            plan.status = "completed"
            _save_plan(plan)
            logger.info("Plan %s completed!", plan.plan_id)
            return None
    else:
        # Track consecutive failures on this step
        plan.consecutive_step_failures += 1
        current_step_desc = plan.steps[plan.current_step] if plan.current_step < len(plan.steps) else ""
        if plan.consecutive_step_failures >= PLAN_REVISION_THRESHOLD:
            # Record the failed step and trigger revision
            plan.failed_step_history.append({
                "step": current_step_desc[:120],
                "reason": f"failed {plan.consecutive_step_failures} times",
                "timestamp": datetime.now().isoformat(),
            })
            _store_plan_step_consequence(
                plan,
                current_step_desc,
                success=False,
                reason=f"failed {plan.consecutive_step_failures} times",
            )
            logger.info(
                "Plan %s step %d failed %d times - triggering revision",
                plan.plan_id, plan.current_step, plan.consecutive_step_failures,
            )
            revised = _revise_plan(plan)
            if revised:
                return revised.steps[0] if revised.steps else None
            # Revision failed - mark plan abandoned so _decide_and_act creates a fresh one
            plan.status = "abandoned"
            _save_plan(plan)
            return None

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
    plan.history.append({
        "step_id": node.step_id,
        "description": node.description[:100],
        "success": success,
        "timestamp": datetime.now().isoformat(),
    })

    if success:
        _store_plan_step_consequence(
            plan,
            node.description,
            success=True,
            reason="decision-tree node advanced successfully",
        )
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
            _store_plan_step_consequence(
                plan,
                node.description,
                success=False,
                reason=f"decision-tree node exhausted {node.retry_count} retries",
            )
            plan.current_node = node.failure_next
            _save_plan(plan)
            next_node = _get_node(plan, plan.current_node)
            return next_node.description if next_node else None
        else:
            # No alternative, max retries exhausted - abandon
            _store_plan_step_consequence(
                plan,
                node.description,
                success=False,
                reason=f"decision-tree node exhausted {node.retry_count} retries",
            )
            plan.status = "abandoned"
            _save_plan(plan)
            logger.warning("Decision tree %s abandoned: step %d exhausted retries",
                           plan.plan_id, node.step_id)
            return None


def _format_plan_text(plan: ActionPlan | DecisionTreePlan) -> str:
    """Format plan context for the decision prompt."""
    if isinstance(plan, DecisionTreePlan):
        node = _get_node(plan, plan.current_node)
        if node is None:
            return ""
        total = len(plan.nodes)
        lines = [f"\nACTION PLAN (decision tree, node {node.step_id}/{total - 1}):"]
        lines.append(f"CURRENT: \"{node.description}\"")
        # Show branches
        if node.success_next is not None:
            sn = _get_node(plan, node.success_next)
            if sn:
                lines.append(f"  - On success: \"{sn.description}\" (node {sn.step_id})")
        else:
            lines.append("  - On success: PLAN COMPLETE")
        if node.failure_next is not None:
            fn = _get_node(plan, node.failure_next)
            if fn:
                lines.append(f"  - On failure: \"{fn.description}\" (node {fn.step_id})")
        else:
            retries_left = node.max_retries - node.retry_count
            lines.append(f"  - On failure: retry ({retries_left} retries left)")
        return "\n".join(lines) + "\n"
    else:
        # Linear plan
        if (not plan.steps or plan.current_step >= len(plan.steps)):
            return ""
        step_num = plan.current_step + 1
        total = len(plan.steps)
        current_step_desc = plan.steps[plan.current_step]
        return (
            f"\nACTION PLAN (step {step_num}/{total}):\n"
            f"YOUR CURRENT STEP: {current_step_desc}\n"
            f"Full plan: {' \u2192 '.join(plan.steps)}\n"
        )


# ============== AUTONOMOUS LOOP ==============

MAX_CONSECUTIVE_FAILURES = 3
_RESEARCH_PROGRESS_TOOLS = frozenset({
    "web_search",
    "extract_content",
    "http_get",
    "add_research_finding",
    "store_research",
    "store_knowledge",
    "extract_facts",
    "start_research",
    "complete_research",
})


def _summarize_research_activity(session_log: list[dict] | None) -> dict | None:
    if not session_log:
        return None

    research_calls = [
        entry for entry in session_log
        if isinstance(entry, dict)
        and entry.get("type") == "tool_call"
        and entry.get("tool") in _RESEARCH_PROGRESS_TOOLS
    ]
    if not research_calls:
        return None

    source_tools = {"web_search", "extract_content", "http_get"}
    storage_tools = {"add_research_finding", "store_research", "store_knowledge", "complete_research"}
    successful = [
        entry for entry in research_calls
        if entry.get("result") and "error" not in str(entry.get("result", "")).lower()[:120]
    ]
    last_call = research_calls[-1]
    last_tool = str(last_call.get("tool") or "")
    gathered = sum(1 for entry in research_calls if entry.get("tool") in source_tools)
    stored = sum(1 for entry in research_calls if entry.get("tool") in storage_tools)

    return {
        "tool": last_tool,
        "calls": len(research_calls),
        "successful_calls": len(successful),
        "gathered_sources": gathered,
        "stored_findings": stored,
        "summary": (
            f"Research: {len(research_calls)} calls, "
            f"{gathered} source steps, {stored} storage steps, last {last_tool}"
        ),
    }


class AutonomousLoop:
    """The main autonomous agent loop.

    Cycle:
    1. Check budget
    2. Run background brain maintenance (zero LLM cost)
    3. Get active goals
    4. Ask the agent "What should I do next?" with goal context
    5. Execute the agent's chosen action
    6. Record outcome
    7. Sleep and repeat
    """

    def __init__(self):
        try:
            _setup_autonomy_logger()
        except Exception:
            pass  # Logger setup is non-critical

        self.budget = ResourceBudget(
            daily_limit=settings.AUTONOMY_DAILY_TOKEN_LIMIT,
            hourly_limit=settings.AUTONOMY_HOURLY_TOKEN_LIMIT,
            session_limit=settings.AUTONOMY_SESSION_TOKEN_LIMIT,
        )
        try:
            load_budget(self.budget)
        except Exception:
            pass  # Budget file may not exist yet

        self.running = False
        self.session_id = f"auto-{uuid.uuid4().hex[:8]}"
        self.action_log: list[ActionRecord] = []
        self.consecutive_failures = 0
        self._last_action: ActionRecord | None = None
        self.consecutive_llm_failures = 0
        self.maintenance_only = False
        self._history: list = []
        self._session_log: list = []
        self._session_start_time: float = time.time()
        self._last_role_name: str = ""
        self._in_role_failures: int = 0

        # Proactive session tracking
        self._proactive_sessions_today: int = 0
        self._last_proactive_time: float = 0.0
        self._last_proactive_day: str = ""
        self._proactive_reminders_sent: set[str] = set()

        # Backoff tracking for error recovery
        self._backoff_failures: int = 0
        self._fast_recovery_idx: int = 0
        self._last_fast_recovery: float = 0.0

        # Runtime snapshot - updated each cycle for /api/autonomy/status
        self._current_goal: dict | None = None
        self._current_plan_step: dict | None = None  # {instruction, step_num, total_steps}
        self._current_task: dict | None = None  # mission task metadata
        self._last_cycle_result: dict | None = None  # {success, reason, decision}
        self._last_agent_response: dict | None = None  # {response, duration_ms, tokens_estimated}
        self._last_research_activity: dict | None = None
        self._last_preflight: dict | None = None
        self._last_loop_detection: dict | None = None
        self._last_plan_health: dict | None = None
        self._last_confidence_policy: dict | None = None
        self._loop_warning_text: str = ""
        try:
            from remy.core.loop_detection import LoopDetectionState
            self._loop_detection_state = LoopDetectionState()
        except Exception:
            self._loop_detection_state = None
        self._last_scheduler_reason: str = ""
        self._cycle_count: int = 0

    async def start(self):
        """Start the autonomous loop."""
        from remy.core.logging_config import ctx_channel, ctx_session_id
        ctx_session_id.set(self.session_id)
        ctx_channel.set("autonomous")

        self.running = True
        self._session_start_time = time.time()
        logger.info("Autonomous mode STARTED (session: %s, start_time=%.0f)", self.session_id, self._session_start_time)
        _write_brain_snapshot(self.session_id, "START")

        # Apply Aura 1.5.3 policy layer (persona, taxonomy, trust, archival rules)
        try:
            from remy.core.agent_tools import init_brain_policy
            init_brain_policy()
        except Exception as e:
            logger.warning("init_brain_policy failed: %s", e)

        # Archive old completed/failed goals
        archive_completed_goals()

        # Ensure critical persistent goals exist (survival + missions)
        try:
            from remy.core.autonomy_goals import ensure_survival_goal, ensure_mission_goals
            ensure_survival_goal()
            ensure_mission_goals()
        except Exception as e:
            logger.warning("Failed to ensure persistent goals: %s", e)

        # Seed spatial context (user location) from data/spatial_seed.json if not yet stored
        try:
            from remy.core.spatial_seed import ensure_spatial_context
            ensure_spatial_context()
        except Exception as e:
            logger.debug("spatial_seed skipped: %s", e)

        # Reset session timer immediately - no LLM calls on critical startup path.
        # If missions produced goals, use them. If truly empty, _seed_initial_goals
        # runs deferred inside the first cycle (not here) to keep startup fast.
        self._session_start_time = time.time()

        # Seed generic goals only if still none exist after mission setup.
        # Deferred: runs at start of first cycle via _maybe_seed_goals(), not here.
        self._needs_seed = not bool(get_active_goals())

        try:
            max_seconds = settings.AUTONOMY_MAX_SESSION_MINUTES * 60
            _first_cycle = True
            while self.running:
                # Session time limit
                elapsed = time.time() - self._session_start_time
                if elapsed >= max_seconds:
                    logger.info(
                        "Session time limit reached (%.0f min). Stopping.",
                        elapsed / 60,
                    )
                    break

                # Deferred seed: only if missions produced no goals.
                # Runs inside the loop (not before it) to keep startup fast.
                if _first_cycle and getattr(self, "_needs_seed", False):
                    self._seed_initial_goals()
                    self._needs_seed = False
                _first_cycle = False

                await self._cycle()
                await asyncio.sleep(self._compute_next_delay())
        except asyncio.CancelledError:
            logger.info("Autonomous loop cancelled")
        finally:
            await self._shutdown()

    async def stop(self):
        self.running = False

    def status(self) -> dict:
        """Return a real-time snapshot of the loop state for /api/autonomy/status."""
        # Stuck missions: goals with >= 3 consecutive failures and a mission_id
        stuck_missions: list[dict] = []
        try:
            active = get_active_goals()
            seen_missions: set[str] = set()
            for g in active:
                mid = g.get("mission_id")
                if mid and g.get("attempts", 0) >= 3 and mid not in seen_missions:
                    seen_missions.add(mid)
                    stuck_missions.append({
                        "mission_id": mid,
                        "goal_id": g["goal_id"],
                        "description": g["description"][:120],
                        "attempts": g["attempts"],
                    })
        except Exception:
            pass

        # Approval queue
        approval_queue: list[dict] = []
        try:
            from remy.core.approval_queue import approval_queue as _aq
            approval_queue = _aq.snapshot_pending()
        except Exception:
            pass

        # Build current_mission from current goal's mission_id
        current_mission = None
        if self._current_goal and self._current_goal.get("mission_id"):
            mid = self._current_goal["mission_id"]
            mission_data = None
            try:
                from remy.core.autonomy_goals import _load_missions
                for m in _load_missions():
                    if m.get("id") == mid:
                        mission_data = m
                        current_mission = {"id": mid, "description": m.get("description", mid)}
                        break
            except Exception:
                pass
            if not current_mission:
                current_mission = {"id": mid, "description": mid}
            try:
                from remy.core.activity_state import build_mission_activity_state

                mission_tasks = [
                    g for g in (active or [])
                    if g.get("mission_id") == mid and g.get("mission_task_id")
                ]
                mission_state = build_mission_activity_state(
                    mid,
                    mission_data=mission_data,
                    active_task_goal_records=mission_tasks,
                )
                if mission_state:
                    current_mission = mission_state
            except Exception:
                pass

        return {
            "running": self.running,
            "session_id": self.session_id,
            "cycle_count": self._cycle_count,
            "budget": self.budget.to_dict() if hasattr(self.budget, "to_dict") else None,
            "current_goal": self._current_goal,
            "current_mission": current_mission,
            "current_task": self._current_task,
            "current_step": self._current_plan_step,
            "last_cycle_result": self._last_cycle_result,
            "current_role": self._last_role_name,
            "last_agent_response": self._last_agent_response,
            "last_research_activity": self._last_research_activity,
            "last_preflight": self._last_preflight,
            "last_loop_detection": self._last_loop_detection,
            "last_plan_health": self._last_plan_health,
            "last_confidence_policy": self._last_confidence_policy,
            "scheduler_reason": self._last_scheduler_reason,
            "stuck_missions": stuck_missions,
            "stuck_missions_count": len(stuck_missions),
            "approval_queue": approval_queue,
            "pending_approvals": len(approval_queue),
            "quality_debt_by_specialist": [],
        }

    def _is_quiet_hours(self) -> bool:
        """Check if current time is within quiet hours."""
        hour = datetime.now().hour
        start = settings.AUTONOMY_QUIET_HOURS_START
        end = settings.AUTONOMY_QUIET_HOURS_END

        if start > end:  # e.g., 23:00 - 07:00 (crosses midnight)
            return hour >= start or hour < end
        else:  # e.g., 02:00 - 06:00
            return start <= hour < end

    async def _test_llm_health(self) -> bool:
        """Quick LLM ping to check if the API is back. Returns True if healthy."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model=settings.SUMMARY_MODEL,
                api_key=settings.GEMINI_API_KEY,
            )
            result = await asyncio.to_thread(llm.invoke, "Reply with OK")
            return bool(result and result.content)
        except Exception as e:
            logger.debug("LLM health check failed: %s", e)
            return False

    # ============== F1: ROLE SELECTION ==============

    def _select_role(self, goal: dict | None, plan: "ActionPlan | DecisionTreePlan | None") -> AgentRole:
        """Select agent role based on current goal and plan step. Pure Python, zero LLM."""
        if not goal:
            return AGENT_ROLES["planner"]

        desc_lower = goal["description"].lower()

        # If there's an active plan, analyze the current step
        step = None
        if plan and plan.status == "active":
            if isinstance(plan, DecisionTreePlan):
                node = _get_node(plan, plan.current_node)
                if node:
                    step = node.description.lower()
            elif plan.current_step < len(plan.steps):
                step = plan.steps[plan.current_step].lower()
        if step:
            if any(kw in step for kw in ("search", "research", "find out", "investigate", "learn", "query")):
                return AGENT_ROLES["researcher"]
            if any(kw in step for kw in ("plan", "break down", "organize", "create goal", "schedule", "prioritize")):
                return AGENT_ROLES["planner"]
            if any(kw in step for kw in ("analyze", "correlate", "summarize", "health", "pattern", "insight")):
                return AGENT_ROLES["analyst"]
            return AGENT_ROLES["executor"]

        # No plan: infer from goal description
        if self._is_research_goal(desc_lower):
            return AGENT_ROLES["researcher"]
        if any(kw in desc_lower for kw in ("analyze", "health", "correlate", "pattern", "insight", "review")):
            return AGENT_ROLES["analyst"]
        if any(kw in desc_lower for kw in ("organize", "plan", "schedule", "prioritize", "break down", "structure")):
            return AGENT_ROLES["planner"]
        return AGENT_ROLES["executor"]

    # ============== PROACTIVE SESSIONS ==============

    def _should_start_proactive_session(self) -> dict | None:
        """Check if we should start a proactive Telegram conversation.

        Pure Python, zero LLM calls. Returns trigger dict or None.
        """
        if not settings.AUTONOMY_PROACTIVE_SESSIONS_ENABLED:
            return None
        if not settings.TELEGRAM_BOT_TOKEN or not settings.PROACTIVE_CHAT_ID:
            return None
        if self._is_quiet_hours():
            return None

        now = datetime.now()
        now_ts = time.time()
        today_str = now.date().isoformat()

        # Daily reset
        if self._last_proactive_day != today_str:
            self._proactive_sessions_today = 0
            self._last_proactive_day = today_str
            self._proactive_reminders_sent.clear()

        # Max per day
        if self._proactive_sessions_today >= settings.AUTONOMY_PROACTIVE_MAX_PER_DAY:
            return None

        # Min interval
        if now_ts - self._last_proactive_time < settings.AUTONOMY_PROACTIVE_MIN_INTERVAL_SEC:
            return None

        # User recently active - don't interrupt
        try:
            from remy.core.agent_tools import brain_lock
            with brain_lock:
                summaries = brain.search(query="", tags=["session-summary"], limit=1)
            if summaries:
                meta = summaries[0].metadata or {}
                ts = meta.get("timestamp") or meta.get("created_at", "")
                if ts:
                    last_time = datetime.fromisoformat(ts)
                    if (now - last_time) < timedelta(minutes=30):
                        return None
        except Exception:
            pass

        # --- TRIGGER 0: Morning digest (replaces cold daily_digest text) ---
        # Fires once per day between DIGEST_HOUR and DIGEST_HOUR+1 if not yet sent today.
        try:
            from remy.core.daily_digest import DIGEST_HOUR, should_send_digest
            if should_send_digest() and now.hour == DIGEST_HOUR:
                # Collect overnight actions for context
                from remy.core.daily_digest import (
                    _collect_completed_today,
                    _collect_actions_today,
                    _collect_active_goals,
                    _collect_budget,
                )
                from remy.core.agent_tools import brain as _b, brain_lock as _bl
                completed = _collect_completed_today(_b, _bl)
                total_actions, success_actions = _collect_actions_today()
                active_goals = _collect_active_goals()
                balance, llm_cost = _collect_budget()

                ctx_parts = [f"Morning of {now.strftime('%d.%m.%Y')}"]
                if completed:
                    ctx_parts.append(f"Completed overnight: {', '.join(completed[:3])}")
                if total_actions:
                    ctx_parts.append(f"Actions taken: {total_actions} ({success_actions} successful)")
                if active_goals:
                    ctx_parts.append(f"Active goals: {', '.join(active_goals[:2])}")
                ctx_parts.append(f"Balance: ${balance} | LLM cost today: ${llm_cost}")

                # Mark daily_digest as sent so cold text version doesn't fire later
                try:
                    from remy.core.daily_digest import _mark_digest_sent
                    _mark_digest_sent()
                except Exception:
                    pass

                return {
                    "reason": "morning_digest",
                    "context": " | ".join(ctx_parts),
                    "priority": "high",
                }
        except Exception as _md_err:
            logger.debug("Morning digest trigger failed: %s", _md_err)

        # --- TRIGGER 1: Scheduled task due today ---
        try:
            with brain_lock:
                tasks = brain.search(query="", tags=["scheduled-task"], limit=20)
            for task in tasks:
                meta = task.metadata or {}
                if meta.get("status") != "active":
                    continue
                # Check both in-memory and persistent dedup
                if task.id in self._proactive_reminders_sent:
                    continue
                if meta.get("last_reminded_date") == today_str:
                    continue
                due_str = meta.get("due_date", "")
                if not due_str:
                    continue
                try:
                    due_date = datetime.fromisoformat(due_str)
                    if due_date.date() == now.date():
                        description = meta.get("description", task.content)
                        return {
                            "reason": "scheduled_task_due",
                            "context": f"Task due today: {description}",
                            "priority": "high",
                            "record_id": task.id,
                        }
                except Exception:
                    continue
        except Exception:
            pass

        # --- TRIGGER 2: Important memories at risk of decay ---
        try:
            with brain_lock:
                insights = brain.insights()
            for ins in insights:
                if ins.get("type") == "decay_risk":
                    records = ins.get("details", {}).get("records", [])
                    important = [r for r in records if r.get("level", 0) >= 3]
                    if important:
                        names = [r.get("content", "")[:60] for r in important[:2]]
                        return {
                            "reason": "decay_risk",
                            "context": f"Important memories fading: {'; '.join(names)}",
                            "priority": "medium",
                        }
        except Exception:
            pass

        # --- TRIGGER 3: No user interaction in >4 hours ---
        try:
            with brain_lock:
                summaries = brain.search(query="", tags=["session-summary"], limit=1)
            if summaries:
                meta = summaries[0].metadata or {}
                ts = meta.get("timestamp") or meta.get("created_at", "")
                if ts:
                    last_time = datetime.fromisoformat(ts)
                    hours_since = (now - last_time).total_seconds() / 3600
                    if hours_since > 4:
                        return {
                            "reason": "inactivity_checkin",
                            "context": f"No interaction for {hours_since:.0f} hours",
                            "priority": "low",
                        }
        except Exception:
            pass

        return None

    def _check_research_delivery(self, tool_result: str):
        """RM-5: Handle proactive delivery of research reports."""
        try:
            import json
            data = json.loads(tool_result)
            if data.get("status") == "research_complete":
                topic = data.get("topic", "Research")
                report = data.get("report", "")
                
                # Format for Telegram
                msg = f"\U0001f52c *Research Complete: {topic}*\n\n{report[:3000]}"
                if len(report) > 3000:
                    msg += "...\n\n(Full report saved to brain)"
                
                # Publish event
                event_bus.emit("research.complete", {
                    "topic": topic,
                    "report": report,
                    "message": msg
                })
                logger.info("Published research completion event for '%s'", topic)

                # Notify user (Telegram if not on web, otherwise event_bus only)
                try:
                    from remy.core.notification_router import notify
                    notify(
                        f"Research complete: {topic}\n\n{report[:1500]}"
                        + ("\n\n(Full report saved to brain)" if len(report) > 1500 else ""),
                        level="info",
                        event_type="research.complete",
                    )
                except Exception as _ne:
                    logger.debug("Could not send research completion notify: %s", _ne)
                
        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.error("Failed to handle research delivery: %s", e)

    async def _start_proactive_session(self, trigger: dict):
        """Generate and send a proactive opening message via Telegram.

        Uses invoke_agent() to generate contextual text, then sends via
        Telegram Bot API. The user's reply flows through TelegramBot.handle_message().
        """
        from remy.core.agent import invoke_agent
        from remy.core.proactive_context import get_proactive_context
        from remy.core.notification_router import is_user_active_on_web

        # Skip if user is currently chatting on web
        if is_user_active_on_web():
            logger.info("Proactive session skipped: user active on web")
            return

        # Budget check
        can_spend, reason_str = self.budget.can_spend(1500)
        if not can_spend:
            logger.info("Proactive session skipped: %s", reason_str)
            return

        proactive_context = get_proactive_context()

        is_morning = trigger.get("reason") == "morning_digest"
        if is_morning:
            proactive_prompt = (
                "=== MORNING DIGEST SESSION ===\n"
                "You are sending the user their morning briefing via Telegram.\n"
                "The user did NOT message you - YOU are reaching out proactively.\n\n"
                f"CONTEXT: {trigger['context']}\n\n"
                f"{proactive_context}\n"
                "INSTRUCTIONS:\n"
                "- Start with a warm good morning greeting (use their name from profile).\n"
                "- Mention any personal events today (birthdays etc.) FIRST if present.\n"
                "- Give a brief summary of what you accomplished overnight (2-3 bullet points max).\n"
                "- State what you plan to work on today.\n"
                "- Keep total message under 10 lines - this is Telegram.\n"
                "- End with one concrete question or offer to help with something specific.\n"
                "- Match the user's preferred language (check profile).\n"
            )
        else:
            proactive_prompt = (
                "=== PROACTIVE SESSION ===\n"
                "You are initiating a conversation with the user via Telegram.\n"
                "The user did NOT message you - YOU are reaching out proactively.\n\n"
                f"TRIGGER: {trigger['reason']}\n"
                f"CONTEXT: {trigger['context']}\n"
                f"PRIORITY: {trigger.get('priority', 'medium')}\n\n"
                f"{proactive_context}\n"
                "INSTRUCTIONS:\n"
                "- Open with a warm, natural greeting.\n"
                "- Be concise (2-4 sentences max). This is Telegram.\n"
                "- If it's a task reminder, ask if they've done it or need help.\n"
                "- If it's a memory fading, naturally bring up the topic.\n"
                "- If it's a check-in, ask how they're doing and reference what you know.\n"
                "- End with an open question that invites response.\n"
                "- Match the user's preferred language (check profile).\n"
            )

        try:
            proactive_session_id = f"proactive-{uuid.uuid4().hex[:8]}"

            start_time = time.time()
            response_text, proactive_history, _proactive_log = await invoke_agent(
                user_message=proactive_prompt,
                session_id=proactive_session_id,
                channel="proactive",
                session_log=[],
                history=[],
            )
            duration_ms = int((time.time() - start_time) * 1000)

            reported_tokens = _extract_history_usage_tokens(proactive_history)
            estimated_tokens = reported_tokens or estimate_tokens(proactive_prompt + response_text)
            self.budget.record_usage(estimated_tokens)
            save_budget(self.budget)
             
            # Global usage stats
            from remy.core.usage_stats import usage_tracker
            usage_tracker.record_usage(
                "autonomy",
                estimated_tokens,
                kind="reported" if reported_tokens else "estimated",
            )

            if not response_text or len(response_text.strip()) < 5:
                logger.warning("Proactive session: empty response from agent")
                return

            # Send via Telegram if token and chat_id are configured
            if settings.TELEGRAM_BOT_TOKEN and settings.PROACTIVE_CHAT_ID:
                from telegram import Bot
                bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
                await bot.send_message(
                    chat_id=settings.PROACTIVE_CHAT_ID,
                    text=response_text,
                )

            # Update tracking
            self._proactive_sessions_today += 1
            self._last_proactive_time = time.time()

            if trigger.get("record_id"):
                self._proactive_reminders_sent.add(trigger["record_id"])
                # Persist reminder date on the scheduled task so it survives restarts
                try:
                    from remy.core.agent_tools import brain_lock as _bl
                    with _bl:
                        _rec = brain.get(trigger["record_id"])
                        if _rec:
                            _meta = dict(_rec.metadata or {})
                            _meta["last_reminded_date"] = datetime.now().date().isoformat()
                            brain.update(trigger["record_id"], metadata=_meta)
                except Exception:
                    pass

            # Store in brain
            from remy.core.agent_tools import Level
            from remy.core.agent_tools import brain_lock
            with brain_lock:
                brain.store(
                    content=f"Proactive session: {trigger['reason']} - {response_text[:150]}",
                    level=Level.WORKING,
                    tags=["proactive-session", trigger["reason"]],
                    metadata={
                        "type": "proactive_session",
                        "trigger_reason": trigger["reason"],
                        "trigger_context": trigger["context"],
                        "session_id": proactive_session_id,
                        "timestamp": datetime.now().isoformat(),
                        "source": "agent-autonomous",
                        "verified": False,
                        "trust_score": 0.4,
                        "admission_class": "reflection",
                    },
                )

            logger.info(
                "Proactive session started: reason=%s, tokens=%d, duration=%dms",
                trigger["reason"], estimated_tokens, duration_ms,
            )

        except Exception as e:
            logger.error("Proactive session failed: %s", e)

    def _compute_next_delay(self) -> float:
        """Return adaptive sleep duration based on last action outcome and agent state."""
        base = float(settings.AUTONOMY_CYCLE_INTERVAL_SEC)
        min_delay = 30.0
        max_delay = 600.0

        action = self._last_action

        # No action this cycle - no goals or all blocked
        if action is None:
            delay = min(base * 2, max_delay)
            event_bus.emit("cycle_delay", {"delay_sec": delay, "reason": "no_action"})
            return delay

        # Pending approvals - back off significantly
        try:
            from remy.core.approval_queue import approval_queue as _aq
            if _aq.pending_count() > 0:
                delay = min(base * 3, max_delay)
                event_bus.emit("cycle_delay", {"delay_sec": delay, "reason": "awaiting_approval"})
                return delay
        except Exception:
            pass

        # Goal completed - momentum, speed up slightly
        if getattr(action, "verified", False):
            delay = max(base * 0.5, min_delay)
            event_bus.emit("cycle_delay", {"delay_sec": delay, "reason": "goal_completed"})
            return delay

        # Success but goal not done - normal pace
        if action.success:
            event_bus.emit("cycle_delay", {"delay_sec": base, "reason": "success"})
            return base

        # Failures - exponential backoff
        failures = self.consecutive_failures
        if failures <= 1:
            delay = base
        elif failures == 2:
            delay = min(base * 1.5, max_delay)
        else:
            delay = min(base * 2, max_delay)

        event_bus.emit("cycle_delay", {"delay_sec": delay, "reason": f"failure_{failures}"})
        return delay

    async def _cycle(self):
        """One cycle of autonomous decision-making."""
        # -1. Daily digest (fire-and-forget, non-blocking)
        try:
            from remy.core.daily_digest import maybe_send_digest
            from remy.core.agent_tools import brain as _brain, brain_lock as _bl
            maybe_send_digest(_brain, _bl)
        except Exception as _de:
            logger.debug("Daily digest check failed: %s", _de)

        # 0. Quiet hours check
        if self._is_quiet_hours():
            self._last_scheduler_reason = (
                f"Quiet hours ({settings.AUTONOMY_QUIET_HOURS_START:02d}:00"
                f"?{settings.AUTONOMY_QUIET_HOURS_END:02d}:00)"
            )
            event_bus.emit("quiet_hours", {
                "start": settings.AUTONOMY_QUIET_HOURS_START,
                "end": settings.AUTONOMY_QUIET_HOURS_END,
            })
            logger.info("Quiet hours active (%02d:00-%02d:00). Sleeping.",
                        settings.AUTONOMY_QUIET_HOURS_START,
                        settings.AUTONOMY_QUIET_HOURS_END)
            await asyncio.sleep(300)  # Check again in 5 minutes
            return

        # 0.5 Proactive session check
        # If a proactive session fires, skip the main _decide_and_act() this cycle
        # to prevent the agent from sending a duplicate greeting/plan.
        try:
            trigger = self._should_start_proactive_session()
            if trigger:
                await self._start_proactive_session(trigger)
                logger.info("Proactive session sent - skipping main cycle to avoid duplicate greeting.")
                return
        except Exception as e:
            logger.warning("Proactive session check failed: %s", e)

        # 1. LLM health - maintenance-only mode after 5 consecutive LLM failures
        if self.maintenance_only:
            if await self._test_llm_health():
                self.maintenance_only = False
                self.consecutive_llm_failures = 0
                self._last_scheduler_reason = "LLM API recovered"
                logger.info("LLM API recovered - exiting maintenance-only mode")
                event_bus.emit("llm_health", {"status": "recovered"})
            else:
                self._last_scheduler_reason = "LLM unavailable - maintenance-only mode"
                logger.info("Maintenance-only mode: LLM still unavailable, running background tasks only")
                try:
                    from remy.core.background_brain import run_background
                    run_background(brain)
                except Exception:
                    pass
                return

        # 1.5. Budget check
        can_spend, reason = self.budget.can_spend(2000)
        if not can_spend:
            self._last_scheduler_reason = f"Budget pause: {reason}"
            event_bus.emit("budget_warning", {"reason": reason})
            logger.info("Budget pause: %s", reason)
            await asyncio.sleep(300)
            return

        # 2. Background maintenance (zero LLM cost)
        try:
            from remy.core.background_brain import run_background
            run_background(brain)
        except Exception as e:
            logger.warning("Background maintenance failed: %s", e)

        # 3. Consecutive failure guard
        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            self._last_scheduler_reason = f"Too many consecutive failures ({self.consecutive_failures}) - pausing 10 min"
            logger.warning(
                "Too many consecutive failures (%d). Pausing 10 minutes.",
                self.consecutive_failures,
            )
            self.consecutive_failures = 0
            await asyncio.sleep(600)
            return

        self._last_scheduler_reason = "active"

        # 3.5. V16: Report wake signal to cognitive substrate
        try:
            from remy.core.v16_proposal import report_wake_signal, report_resource_signal
            last_activity = int(time.time() - self._last_activity_ts) if hasattr(self, '_last_activity_ts') else 0
            report_wake_signal(brain, "timer", last_activity)
            # Report resource budget state
            budget_info = self.budget.to_dict()
            total = budget_info.get("daily_limit", 0)
            used = budget_info.get("tokens_today", 0)
            ratio = max(0.0, (total - used) / total) if total > 0 else 1.0
            report_resource_signal(brain, ratio, used, total if total > 0 else None)
        except Exception as e:
            logger.debug("V16 wake signal failed: %s", e)

        # 4. Decide and act
        cycle_num = len(self.action_log) + 1
        event_bus.emit("cycle_start", {
            "cycle": cycle_num,
            "session_id": self.session_id,
            "budget": self.budget.to_dict(),
        })
        try:
            action = await self._decide_and_act()
            self._last_action = action  # track for adaptive timing
            if action:
                self.action_log.append(action)
                # Cap action_log to prevent unbounded growth in long-running sessions
                if len(self.action_log) > 200:
                    self.action_log = self.action_log[-200:]
                if action.success:
                    self.consecutive_failures = 0
                    self.consecutive_llm_failures = 0
                else:
                    self.consecutive_failures += 1

                # RM-5: Check for research completion in agent response
                if action.result and "research_complete" in action.result:
                    self._check_research_delivery(action.result)

        except Exception as e:
            logger.error("Autonomous cycle error: %s", e)
            self.consecutive_failures += 1
            self.consecutive_llm_failures += 1
            if self.consecutive_llm_failures >= 5 and not self.maintenance_only:
                self.maintenance_only = True
                logger.warning("5 consecutive LLM failures - entering maintenance-only mode")
                event_bus.emit("llm_health", {"status": "maintenance_only", "failures": self.consecutive_llm_failures})
        finally:
            # V16: Render proposals and log wake cycle (zero LLM cost)
            try:
                from remy.core.v16_proposal import render_proposals, log_wake_cycle
                proposals = render_proposals(brain, cycle_num, self.session_id)
                if proposals:
                    event_bus.emit("v16_proposals", {
                        "cycle": cycle_num,
                        "count": len(proposals),
                        "top": proposals[0] if proposals else None,
                    })
                action_desc = None
                action_result = None
                if hasattr(self, '_last_action') and self._last_action:
                    action_desc = getattr(self._last_action, 'description', None)
                    action_result = "success" if getattr(self._last_action, 'success', False) else "failure"
                log_wake_cycle(brain, cycle_num, self.session_id,
                               action_taken=action_desc, action_result=action_result)
            except Exception as v16e:
                logger.debug("V16 proposal/log failed: %s", v16e)

            event_bus.emit("cycle_end", {
                "cycle": cycle_num,
                "session_id": self.session_id,
                "budget": self.budget.to_dict(),
            })

    @staticmethod
    def _coerce_research_progress_evaluation(
        evaluation: dict, session_log: list[dict],
    ) -> dict:
        """Upgrade a failed evaluation to success when research tools made real progress.

        Research workers often collect findings but the LLM evaluator marks it as
        failure because the goal isn't "fully complete". This coerces the eval so
        partial research progress counts as success.
        """
        if evaluation.get("success"):
            return evaluation  # Already success - nothing to do

        RESEARCH_PROGRESS_TOOLS = frozenset({
            "web_search", "store_research", "add_research_finding",
            "store", "extract_content", "extract_facts", "store_knowledge",
        })

        tool_calls = [
            e for e in session_log
            if e.get("type") == "tool_call" and e.get("tool") in RESEARCH_PROGRESS_TOOLS
        ]

        if not tool_calls:
            return evaluation  # No research tools called

        # Check if any tool call had a non-error result
        real_results = [
            tc for tc in tool_calls
            if tc.get("result") and "error" not in str(tc.get("result", "")).lower()[:100]
        ]

        if real_results:
            logger.info(
                "Coercing research eval: %d research tool calls with results -> success",
                len(real_results),
            )
            return {
                **evaluation,
                "success": True,
                "confidence": max(evaluation.get("confidence", 0.4), 0.5),
                "reason": (
                    f"Research progress: {len(real_results)} findings collected. "
                    f"Original eval: {evaluation.get('reason', 'n/a')}"
                ),
            }

        return evaluation

    async def _evaluate_outcome(
        self, goal_description: str, agent_response: str,
        success_criteria: list | None = None, session_log: list | None = None,
    ) -> dict:
        """Evaluate the agent's response to determine success/failure.

        If success_criteria are provided, checks them first. If all are met,
        returns success immediately without an LLM call.

        Uses a lightweight LLM call (~200 tokens) to assess the outcome.
        Returns {"success": bool, "confidence": float, "reason": str, "goal_completed": bool}
        Falls back to {"success": True, "confidence": 0.5} if LLM call fails.
        """
        # Fast path: criteria-based evaluation (no LLM needed)
        if success_criteria:
            try:
                from remy.core.success_criteria import verify_criteria
                met_count, total, details = verify_criteria(success_criteria, session_log=session_log or [])
                if total > 0 and met_count == total:
                    types_str = ", ".join(d.get("type", d.get("criterion", {}).get("type", "?")) for d in details[:3])
                    return {
                        "success": True,
                        "confidence": 0.95,
                        "reason": f"All {total} success criteria met: {types_str}",
                        "goal_completed": True,
                    }
            except Exception as e:
                logger.debug("Criteria check failed, falling back to LLM: %s", e)

        eval_prompt = (
            "You are a STRICT evaluator of an AI agent's autonomous action. "
            "Your job is to be HONEST, not optimistic. Output ONLY a JSON object.\n\n"
            "HONEST FAILURE PROTOCOL:\n"
            "- If the agent just SAID it would do something but didn't actually do it -> success: false\n"
            "- If the agent used a tool but got an error or empty result -> success: false\n"
            "- If the agent repeated the same action as previous attempts -> success: false\n"
            "- If the response is vague filler without concrete outcome -> success: false\n"
            "- goal_completed: true ONLY if there is concrete evidence the goal is fully done "
            "(e.g., data was stored, file was written, research was synthesized). "
            "Planning to do something is NOT completion.\n\n"
            f"GOAL: {goal_description}\n\n"
            f"AGENT RESPONSE:\n{agent_response[:500]}\n\n"
            "Output exactly this JSON format (no extra text before or after):\n"
            '{"success": true, "confidence": 0.7, "reason": "brief explanation", "goal_completed": false}\n\n'
            "Fields:\n"
            "- success: true ONLY if there was measurable progress (tool called successfully, data changed)\n"
            "- confidence: 0.0-1.0 (be conservative - use 0.3-0.5 when uncertain)\n"
            "- reason: one sentence explaining what concretely happened\n"
            "- goal_completed: true ONLY with concrete evidence of full completion"
        )

        try:
            from remy.core.llm import call_llm_async

            result = await call_llm_async(eval_prompt, purpose="evaluate_outcome")
            raw = _llm_content_to_str(result.content).strip()

            if not raw or len(raw) < 5:
                logger.warning(
                    "Self-eval LLM returned empty/short. content type=%s, repr=%r",
                    type(result.content).__name__, result.content,
                )
                raise ValueError("LLM returned empty response")

            try:
                evaluation = _parse_llm_json(raw)
            except json.JSONDecodeError:
                # LLM returned text instead of JSON - use keyword heuristic
                logger.debug("Self-eval JSON parse failed, using heuristic on: %s", raw[:200])
                raw_lower = raw.lower()
                success = any(w in raw_lower for w in ("success", "progress", "completed", "achieved"))
                failed = any(w in raw_lower for w in ("fail", "error", "stuck", "unable", "cannot"))
                return {
                    "success": success and not failed,
                    "confidence": 0.4,
                    "reason": raw[:200],
                    "goal_completed": "completed" in raw_lower or "fully achieved" in raw_lower,
                }

            # Validate required fields
            success = bool(evaluation.get("success", False))
            confidence = float(evaluation.get("confidence", 0.5))
            goal_completed = bool(evaluation.get("goal_completed", False))

            # Honest failure guard: low confidence - downgrade success
            if confidence < 0.4 and success:
                logger.info("Low confidence (%.2f) - downgrading success to failure", confidence)
                success = False
            # Require high confidence for goal completion
            if goal_completed and confidence < 0.6:
                logger.info("Low confidence (%.2f) - rejecting goal_completed", confidence)
                goal_completed = False

            return {
                "success": success,
                "confidence": confidence,
                "reason": str(evaluation.get("reason", ""))[:200],
                "goal_completed": goal_completed,
            }

        except Exception as e:
            logger.warning("Self-evaluation failed, defaulting to failure: %s", e)
            return {
                "success": False,
                "confidence": 0.3,
                "reason": f"Evaluation unavailable: {e}",
                "goal_completed": False,
            }

    async def _decide_and_act(self) -> ActionRecord | None:
        """Ask the agent what to do, then do it."""

        cycle_num = len(self.action_log) + 1
        active_goals = get_active_goals()

        # Mission-first: prioritize runnable mission tasks over legacy goals
        from remy.core.orchestrator import focus_execution_goals

        active_goals = focus_execution_goals(active_goals)

        # Auto-generate goals if none exist
        if not active_goals:
            new_goals = self._generate_smart_goals()
            if new_goals:
                active_goals = get_active_goals()

        # Auto-decompose goals that keep failing (>= 3 attempts, not a sub-goal)
        if active_goals:
            top_goal = active_goals[0]
            if (
                top_goal["attempts"] >= AUTO_DECOMPOSE_THRESHOLD
                and not top_goal.get("parent_goal_id")
            ):
                logger.info(
                    "Auto-decomposing goal after %d failures: %s",
                    top_goal["attempts"], top_goal["description"][:60],
                )
                decompose_goal(top_goal["record_id"])
                active_goals = get_active_goals()

        # Auto-detect research goals and create research projects
        if active_goals:
            top_goal = active_goals[0]
            self._ensure_research_project(top_goal)

        if active_goals:
            top = active_goals[0]
            self._current_goal = {
                "goal_id": top["goal_id"],
                "description": top["description"][:200],
                "priority": top["priority"],
                "attempts": top["attempts"],
                "mission_id": top.get("mission_id"),
                "parent_goal_id": top.get("parent_goal_id"),
            }
            event_bus.emit("goal_selected", {
                "goal_id": top["goal_id"],
                "description": top["description"][:200],
                "priority": top["priority"],
                "attempts": top["attempts"],
                "mission_id": top.get("mission_id"),
            })
        else:
            self._current_goal = None

        # Load or create action plan for top goal
        current_plan = None
        if active_goals:
            top_goal = active_goals[0]
            current_plan = load_plan_for_goal(top_goal["goal_id"])
            # Discard abandoned plans - let a fresh one be created below
            if current_plan and getattr(current_plan, "status", "") == "abandoned":
                logger.info("Discarding abandoned plan for goal %s", top_goal["goal_id"])
                current_plan = None
            if not current_plan and top_goal["attempts"] >= 1:
                # Create a plan after first attempt
                current_plan = create_plan_for_goal(
                    top_goal["goal_id"], top_goal["description"],
                )

        past_outcomes = self._recent_outcomes_summary()
        # Scar guard: if the next planned step is a (goal, action) the world has
        # already REFUTED, surface a hard warning in the decision prompt so the
        # cycle does not loop on a lived failure (frequency of past attempts must
        # not bury a refutation). Reads the same (goal_description, step) pair
        # that _store_plan_step_consequence writes.
        scar_note = self._scar_warning_for_plan(top_goal, current_plan) if active_goals else ""
        if scar_note:
            past_outcomes = (past_outcomes + "\n\n" + scar_note) if past_outcomes else scar_note
        budget_info = self.budget.to_dict()
        strategy_hints = self._analyze_strategy_effectiveness()
        # F3: Append goal-type feedback to strategy hints
        goal_feedback = self._analyze_goal_type_feedback()
        if goal_feedback:
            strategy_hints = (strategy_hints + "\n" + goal_feedback) if strategy_hints else goal_feedback
        last_reflection = self._get_last_reflection()

        # Gather tool health status - merge in-memory circuit breaker + Aura persistent history
        from remy.core.brain_tools import tool_health as _tool_health
        health_report = _tool_health.get_health_report()
        try:
            from remy.core.agent_tools import brain as _brain
            aura_health = _brain.tool_health()  # persistent cross-session data
            if aura_health:
                # Aura data is supplementary - don't overwrite active circuit-open status
                for tool, status in aura_health.items():
                    if tool not in health_report:
                        health_report[tool] = f"{status} (historical)"
        except Exception:
            pass

        if current_plan and current_plan.status == "active":
            if isinstance(current_plan, DecisionTreePlan):
                node = _get_node(current_plan, current_plan.current_node)
                if node:
                    self._current_plan_step = {
                        "instruction": node.description[:200],
                        "step_num": current_plan.current_node,
                        "total_steps": len(current_plan.nodes),
                        "plan_type": "decision_tree",
                    }
                    event_bus.emit("plan_step", {
                        "step_num": current_plan.current_node,
                        "total_steps": len(current_plan.nodes),
                        "step_description": node.description[:200],
                        "plan_type": "decision_tree",
                    })
            elif (current_plan.steps
                    and current_plan.current_step < len(current_plan.steps)):
                step = current_plan.steps[current_plan.current_step]
                self._current_plan_step = {
                    "instruction": step[:200],
                    "step_num": current_plan.current_step + 1,
                    "total_steps": len(current_plan.steps),
                    "plan_type": "linear",
                }
                event_bus.emit("plan_step", {
                    "step_num": current_plan.current_step + 1,
                    "total_steps": len(current_plan.steps),
                    "step_description": step[:200],
                    "plan_type": "linear",
                })
        else:
            self._current_plan_step = None

        # Track current mission task if goal is a task sub-goal
        if active_goals:
            top_g = active_goals[0]
            if top_g.get("parent_goal_id"):
                self._current_task = {
                    "action": top_g["description"][:200],
                    "mission_id": top_g.get("mission_id"),
                    "goal_id": top_g["goal_id"],
                }
            else:
                self._current_task = None

        # F1: Select agent role based on goal + plan
        top_goal_dict = active_goals[0] if active_goals else None
        role = self._select_role(top_goal_dict, current_plan)
        self._last_role_name = role.name
        event_bus.emit("role_selected", {
            "role": role.name,
            "goal": top_goal_dict["description"][:100] if top_goal_dict else "none",
        })

        preflight_text = ""
        self._last_preflight = None
        if top_goal_dict:
            try:
                from remy.core.preflight import format_preflight_for_prompt, run_preflight

                remaining_budget = min(
                    max(0, self.budget.daily_limit - self.budget.tokens_today),
                    max(0, self.budget.hourly_limit - self.budget.tokens_this_hour),
                    max(0, self.budget.session_limit - self.budget.tokens_this_session),
                )
                preflight = run_preflight(
                    top_goal_dict.get("description", ""),
                    goal_attempts=int(top_goal_dict.get("attempts", 0) or 0),
                    budget_tokens_remaining=remaining_budget,
                    tool_health_report=health_report,
                )
                preflight_text = format_preflight_for_prompt(preflight)
                self._last_preflight = {
                    "can_proceed": preflight.can_proceed,
                    "predicted_success": preflight.predicted_success,
                    "difficulty": preflight.difficulty,
                    "suggestion": preflight.suggestion,
                    "warnings": list(preflight.warnings or []),
                    "budget_tokens_remaining": remaining_budget,
                }
                if preflight_text:
                    event_bus.emit("preflight_analysis", self._last_preflight)
            except Exception as _pf_err:
                logger.debug("preflight skipped: %s", _pf_err)

        decision_prompt = self._build_decision_prompt(
            active_goals, past_outcomes, budget_info,
            current_plan=current_plan,
            strategy_hints=strategy_hints,
            last_reflection=last_reflection,
            tool_health_report=health_report,
            role=role,
            preflight_text=preflight_text,
            loop_warning_text=self._loop_warning_text,
        )

        prompt_summary = f"Goals: {len(active_goals)} | Role: {role.name}"
        if current_plan:
            if isinstance(current_plan, DecisionTreePlan):
                prompt_summary += f" | Tree node {current_plan.current_node}/{len(current_plan.nodes)}"
            elif current_plan.steps:
                prompt_summary += f" | Plan step {current_plan.current_step + 1}/{len(current_plan.steps)}"
        prompt_summary += f" | Budget: {self.budget.tokens_today}/{self.budget.daily_limit}"
        event_bus.emit("thinking", {"summary": prompt_summary})

        start_time = time.time()
        from remy.core.orchestrator import dispatch_worker, format_execution_report

        response_text, new_history, new_log, worker_result = await dispatch_worker(
            goal=top_goal_dict,
            decision_prompt=decision_prompt,
            session_id=self.session_id,
            session_log=self._session_log,
            history=self._history,
            current_plan=current_plan,
        )
        duration_ms = int((time.time() - start_time) * 1000)

        # Normalize worker output into concise report
        response_text = format_execution_report(worker_result, response_text)

        event_bus.emit("agent_response", {
            "response": response_text[:500],
            "duration_ms": duration_ms,
            "tokens_estimated": estimate_tokens(decision_prompt + response_text),
        })
        self._last_agent_response = {
            "response": response_text[:500],
            "duration_ms": duration_ms,
            "tokens_estimated": estimate_tokens(decision_prompt + response_text),
        }

        reported_tokens = _extract_history_usage_tokens(new_history)
        estimated_tokens = reported_tokens or estimate_tokens(decision_prompt + response_text)
        self.budget.record_usage(estimated_tokens)
        save_budget(self.budget)
        
        # Track global usage
        from remy.core.usage_stats import usage_tracker
        usage_tracker.record_usage(
            "autonomy",
            estimated_tokens,
            kind="reported" if reported_tokens else "estimated",
        )

        self._history = new_history
        # Cap session_log to prevent unbounded growth in long-running sessions
        self._session_log = new_log[-50:] if len(new_log) > 50 else new_log
        try:
            if self._loop_detection_state is not None:
                from remy.core.loop_detection import (
                    detect_loop,
                    extract_fingerprint,
                    format_loop_warning_for_prompt,
                )

                fp = extract_fingerprint(self._session_log, cycle_num)
                loop_detection = detect_loop(self._loop_detection_state, fp)
                self._last_loop_detection = {
                    **loop_detection,
                    "repeated_tools": list(loop_detection.get("repeated_tools") or []),
                }
                self._loop_warning_text = format_loop_warning_for_prompt(loop_detection)
                if loop_detection.get("level") != "none":
                    event_bus.emit("loop_detection", self._last_loop_detection)
        except Exception as _loop_err:
            logger.debug("loop detection skipped: %s", _loop_err)

        goal_description = ""
        if active_goals:
            record_goal_attempt(active_goals[0]["record_id"])
            goal_description = active_goals[0].get("description", "")

        # Detect obvious failures before spending tokens on self-eval
        from remy.core.orchestrator import check_obvious_failure

        evaluation = check_obvious_failure(response_text)
        if not evaluation:
            evaluation = await self._evaluate_outcome(goal_description, response_text)

        # Coerce research progress: if research tools collected findings,
        # don't let the evaluator mark it as failure
        evaluation = self._coerce_research_progress_evaluation(
            evaluation, self._session_log,
        )

        self._last_cycle_result = {
            "success": evaluation["success"],
            "confidence": evaluation["confidence"],
            "reason": evaluation["reason"][:200],
            "goal_completed": evaluation["goal_completed"],
            "decision": "completed" if evaluation["goal_completed"] else ("success" if evaluation["success"] else "failed"),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self._cycle_count += 1
        event_bus.emit("evaluation", {
            "success": evaluation["success"],
            "confidence": evaluation["confidence"],
            "reason": evaluation["reason"],
            "goal_completed": evaluation["goal_completed"],
        })
        eval_tokens = estimate_tokens(goal_description + response_text[:500]) + 50
        estimated_tokens += eval_tokens
        self.budget.record_usage(eval_tokens)
        usage_tracker.record_usage("autonomy", eval_tokens, kind="estimated")

        # в”Ђв”Ђ Deterministic accountability check (zero-LLM, always runs) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
        # Catches "I did X" claims in response text with no matching tool call.
        try:
            from remy.core.autonomy_critique import check_action_accountability
            acct = check_action_accountability(response_text, self._session_log)
            acct_quality = acct.get("quality", 1.0)
            if acct_quality < 0.7:
                acct_issues = acct.get("issues", [])
                logger.warning(
                    "ACTION ACCOUNTABILITY: quality=%.2f issues=%s",
                    acct_quality, acct_issues[:2],
                )
                # Downgrade evaluation - agent claimed actions it didn't perform
                if acct_quality < 0.5 and evaluation.get("success"):
                    evaluation["success"] = False
                    evaluation["reason"] = (
                        f"Accountability check: unsubstantiated action claim(s) - "
                        + (acct_issues[0] if acct_issues else "claimed action without tool call")
                    )
        except Exception as _ac_err:
            logger.debug("accountability check skipped: %s", _ac_err)

        # в”Ђв”Ђ LLM self-critique (optional, token-cost, deeper analysis) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
        critique_enabled = getattr(settings, "AUTONOMY_SELF_CRITIQUE_ENABLED", False)
        if critique_enabled:
            try:
                from remy.core.autonomy_critique import critique_response, should_critique

                if should_critique(self._session_log):
                    decision_prompt_text = decision_prompt if isinstance(decision_prompt, str) else ""
                    critique = await critique_response(
                        goal_description,
                        decision_prompt_text,
                        response_text,
                        self._session_log,
                    )
                    quality = critique.get("quality", 1.0)
                    should_retry = critique.get("should_retry", False)

                    if should_retry and quality < 0.5:
                        # Retry the agent once
                        retry_text, retry_history, retry_log, retry_result = await dispatch_worker(
                            goal=top_goal_dict,
                            decision_prompt=decision_prompt,
                            session_id=self.session_id,
                            session_log=self._session_log,
                            history=self._history,
                            current_plan=current_plan,
                        )
                        response_text = format_execution_report(retry_result, retry_text)
                        self._history = retry_history
                        self._session_log = retry_log[-50:] if len(retry_log) > 50 else retry_log
                    elif quality < 0.3:
                        # Low quality, no retry - downgrade evaluation
                        evaluation["success"] = False
                        evaluation["reason"] = f"Self-critique downgrade (quality={quality:.2f}): {critique.get('issues', ['low quality'])[0] if critique.get('issues') else 'low quality'}"
            except Exception as _ce:
                logger.warning("Self-critique failed (non-blocking): %s", _ce)

        self._last_research_activity = _summarize_research_activity(self._session_log)
        try:
            from remy.core.turn_classification import classify_turn
            turn_class = classify_turn(self._session_log).value
        except Exception as _tc_err:
            logger.debug("turn classification skipped: %s", _tc_err)
            turn_class = "productive" if evaluation.get("success") else "idle"

        try:
            from remy.core.confidence_autonomy import (
                assess_action_confidence,
                infer_domain,
                record_domain_outcome,
            )

            domain = infer_domain(goal_description or response_text)
            record_domain_outcome(domain, bool(evaluation.get("success")))
            recent_successes = 0
            recent_failures = 0
            for prior in self.action_log[-10:]:
                prior_success = getattr(prior, "success", None)
                if prior_success is None and isinstance(prior, dict):
                    prior_success = prior.get("success")
                if prior_success:
                    recent_successes += 1
                else:
                    recent_failures += 1
            daily_limit = max(float(getattr(self.budget, "daily_limit", 0) or 0), 1.0)
            budget_pct = max(
                0.0,
                min(100.0, ((daily_limit - float(self.budget.tokens_today)) / daily_limit) * 100.0),
            )
            tool_health_issues = sum(
                1
                for status in (health_report or {}).values()
                if str(status).lower() not in {"ok", "healthy", "available"}
            )
            confidence, autonomy_action = assess_action_confidence(
                goal_description or response_text,
                budget_pct=budget_pct,
                tool_health_issues=tool_health_issues,
                recent_successes=recent_successes,
                recent_failures=recent_failures,
            )
            self._last_confidence_policy = {
                "domain": domain,
                "confidence": round(confidence, 3),
                "recommended_action": autonomy_action,
                "budget_pct": round(budget_pct, 1),
                "tool_health_issues": tool_health_issues,
                "recent_successes": recent_successes,
                "recent_failures": recent_failures,
                "enforced": False,
            }
            if autonomy_action != "execute_silent":
                event_bus.emit("confidence_policy", self._last_confidence_policy)
        except Exception as _conf_err:
            logger.debug("confidence autonomy skipped: %s", _conf_err)

        action = ActionRecord(
            action_id=uuid.uuid4().hex[:12],
            timestamp=datetime.now().isoformat(),
            goal_id=active_goals[0]["goal_id"] if active_goals else None,
            action_type="agent_invoke",
            description=response_text[:200],
            result=response_text[:500],
            success=evaluation["success"],
            tokens_used=estimated_tokens,
            duration_ms=duration_ms,
            turn_class=turn_class,
        )

        goal_record_id = active_goals[0]["record_id"] if active_goals else None
        record_outcome(action, goal_record_id)
        event_bus.emit("outcome", {
            "action_type": action.action_type,
            "success": action.success,
            "tokens_used": action.tokens_used,
            "duration_ms": action.duration_ms,
            "description": action.description[:200],
        })

        # в”Ђв”Ђ Record to execution_log + task_metrics в”Ђв”Ђ
        try:
            from remy.core.execution_log import record_cycle_execution
            from remy.core.task_metrics import (
                CycleOutcome,
                detect_memory_signals,
                resolve_family,
                task_metrics,
            )

            memory_assisted, retry_shaped = detect_memory_signals(self._session_log)
            record_cycle_execution(
                cycle_num=cycle_num,
                goal=top_goal_dict,
                worker_result=worker_result,
                session_log=self._session_log,
                evaluation=evaluation,
                duration_ms=duration_ms,
                tokens_used=estimated_tokens,
                cost_usd=0.0,
                turn_class=turn_class,
                verified=bool(evaluation.get("goal_completed")),
                repeated_failure=bool(
                    top_goal_dict and top_goal_dict.get("attempts", 0) >= 3
                    and not evaluation.get("success")
                ),
                memory_assisted=memory_assisted,
            )

            worker_status = getattr(worker_result, "status", "") or ""
            task_metrics.record(CycleOutcome(
                family=resolve_family(top_goal_dict),
                success=bool(evaluation.get("success")),
                blocked_external=worker_status == "blocked_external",
                zero_tool=not any(
                    e.get("type") == "tool_call"
                    for e in (self._session_log or [])
                    if isinstance(e, dict)
                ),
                timeout=worker_status == "timeout",
                duration_ms=duration_ms,
                tokens_used=estimated_tokens,
                memory_assisted=memory_assisted,
                retry_shaped=retry_shaped,
                verified=bool(evaluation.get("goal_completed")),
                repeated_failure=bool(
                    top_goal_dict and top_goal_dict.get("attempts", 0) >= 3
                    and not evaluation.get("success")
                ),
            ))
        except Exception as e:
            logger.warning("Metrics recording failed: %s", e)

        # Reset stale-focus counter on any successful step (not just goal completion)
        if evaluation["success"] and top_goal_dict and top_goal_dict.get("mission_id"):
            try:
                from remy.core.orchestrator import mark_focus_progress
                mark_focus_progress(top_goal_dict["mission_id"])
            except Exception:
                pass

        plan_advance_blocked = False
        if current_plan:
            try:
                from remy.core.plan_invalidation import (
                    abandon_plan,
                    insert_prerequisite,
                    process_step_result,
                )

                projected_failures = getattr(current_plan, "consecutive_step_failures", 0)
                if not evaluation["success"]:
                    projected_failures += 1
                health = process_step_result(
                    current_plan,
                    response_text,
                    bool(evaluation["success"]),
                    consecutive_failures=projected_failures,
                )
                self._last_plan_health = {
                    "plan_id": getattr(current_plan, "plan_id", ""),
                    "valid": health.valid,
                    "needs_update": health.needs_update,
                    "abandon": health.abandon,
                    "confidence": health.confidence,
                    "reason": health.reason,
                    "suggested_action": health.suggested_action,
                }
                if health.abandon:
                    abandon_plan(current_plan)
                    _save_plan(current_plan)
                    plan_advance_blocked = True
                elif health.suggested_action == "add_prerequisite":
                    prerequisite = f"Resolve prerequisite before retrying: {response_text[:160]}"
                    if insert_prerequisite(current_plan, prerequisite):
                        current_plan.consecutive_step_failures = 0
                        _save_plan(current_plan)
                        plan_advance_blocked = True
                if health.suggested_action != "continue":
                    event_bus.emit("plan_health", self._last_plan_health)
            except Exception as _ph_err:
                logger.debug("plan invalidation skipped: %s", _ph_err)

        # Advance plan on success/failure
        if current_plan:
            if not plan_advance_blocked:
                advance_plan(current_plan, evaluation["success"])
            # If plan just completed, treat it as goal completion signal
            if current_plan.status == "completed" and goal_record_id:
                if not evaluation["goal_completed"]:
                    evaluation["goal_completed"] = True
                    evaluation["reason"] = (
                        evaluation.get("reason", "") + " [plan completed all steps]"
                    ).strip()
                    logger.info("Plan completed -> marking goal complete: %s", goal_description[:80])

        # Auto-complete goal if evaluation says it's done
        if evaluation["goal_completed"] and goal_record_id:
            update_goal_status(
                goal_record_id, "completed",
                notes=f"Auto-completed: {evaluation['reason']}",
            )
            logger.info("Goal auto-completed: %s", goal_description[:80])
            if top_goal_dict and top_goal_dict.get("created_by") == "user":
                _notify_goal_completed(goal_description, top_goal_dict, evaluation.get("reason", ""))

        logger.info(
            "Action [%s] success=%s (%.0f%% confidence) tokens=%d: %s",
            action.action_id, evaluation["success"],
            evaluation["confidence"] * 100, estimated_tokens,
            evaluation["reason"][:80],
        )

        await self._notify_action(action)
        return action

    def _build_decision_prompt(
        self,
        goals: list[dict],
        past_outcomes: str,
        budget: dict,
        current_plan: ActionPlan | DecisionTreePlan | None = None,
        strategy_hints: str = "",
        last_reflection: str = "",
        tool_health_report: dict[str, str] | None = None,
        role: AgentRole | None = None,
        behavioral_rules: list[dict] | None = None,
        preflight_text: str = "",
        loop_warning_text: str = "",
    ) -> str:
        """Build the prompt that asks the agent what to do."""
        if goals:
            goal_lines = []
            completed_ids = {g.get("goal_id") or g.get("record_id") for g in goals if g.get("status") == "completed"}
            blocked_lines = []
            for g in goals[:5]:
                gid = g.get("goal_id") or g.get("record_id", "?")
                line = f"- [{g['priority'].upper()}] (id: {gid}) {g['description']}"
                if g.get("deadline"):
                    line += f" (deadline: {g['deadline']})"
                line += f" (attempts: {g['attempts']})"
                # Check if blocked by incomplete dependencies
                depends_on = g.get("depends_on", [])
                if depends_on:
                    unmet = [d for d in depends_on if d not in completed_ids]
                    if unmet:
                        line += f" [BLOCKED: waiting for {', '.join(unmet)}]"
                        blocked_lines.append(line)
                        continue
                goal_lines.append(line)
            if blocked_lines:
                goal_lines.append("\nGOAL MANAGEMENT - Blocked goals (skip until dependencies met):")
                goal_lines.extend(blocked_lines)
            goals_text = "\n".join(goal_lines)
        else:
            goals_text = "No active goals. Consider what to do next."

        # Plan context
        plan_text = ""
        if current_plan and current_plan.status == "active":
            plan_text = _format_plan_text(current_plan)

        # Reflection context
        reflection_text = ""
        if last_reflection:
            reflection_text = f"\nLESSONS FROM LAST SESSION:\n{last_reflection}\n"

        # Strategy hints
        strategy_text = ""
        if strategy_hints:
            strategy_text = f"\nSTRATEGY INSIGHTS:\n{strategy_hints}\n"

        # F5: Quality trend from eval metrics
        try:
            from remy.core.eval_metrics import get_metrics_summary
            _auto_metrics = get_metrics_summary(channel="autonomous", limit=10)
            if _auto_metrics.get("total_responses", 0) >= 3:
                _tool_success = _auto_metrics.get("tool_success_rate", 100)
                if _tool_success < 50:
                    strategy_text += (
                        f"\nQUALITY WARNING: Low tool success rate ({_tool_success}%). "
                        "Verify tool inputs carefully.\n"
                    )
        except Exception:
            pass

        # F6: Recent proactive sessions (prevent duplication)
        try:
            from remy.core.agent_tools import brain_lock
            with brain_lock:
                proactives = brain.search(query="", tags=["proactive-session"], limit=1)
            if proactives:
                meta = getattr(proactives[0], "metadata", None) or {}
                ts = meta.get("timestamp") or meta.get("created_at", "")
                if ts:
                    last_time = datetime.fromisoformat(ts)
                    # If proactive session was within last 1 hour
                    if (datetime.now() - last_time).total_seconds() < 3600:
                        content = proactives[0].content[:200]
                        strategy_text += (
                            f"\nIMPORTANT context: You just sent a proactive message to the user: \"{content}...\". "
                            "Do NOT repeat this greeting or plan. Assume the user has seen it.\n"
                        )
        except Exception:
            pass

        # Tool health
        health_text = ""
        if tool_health_report:
            lines = [f"- {tool}: {status}" for tool, status in tool_health_report.items()]
            health_text = "\nTOOL HEALTH:\n" + "\n".join(lines) + "\nAvoid unavailable tools. Use alternatives.\n"

        # Active research projects
        research_text = ""
        try:
            from remy.core.brain_tools import (
                get_active_research_projects,
                tool_health,
            )
            from remy.core.agent_tools import brain

            projects = get_active_research_projects()
            if projects:
                rlines = []
                for p in projects[:3]:
                    rlines.append(
                        f"- [{p['status'].upper()}] \"{p['topic']}\" - "
                        f"{p['queries_done']}/{p['queries_total']} queries done, "
                        f"{p['findings_count']} findings (project_id: {p['project_id']})"
                    )
                research_text = "\nACTIVE RESEARCH:\n" + "\n".join(rlines) + "\n"
        except Exception:
            pass

        # Active agent todos
        todos_text = ""
        try:
            from remy.core.agent_tools import brain as _brain, brain_lock
            with brain_lock:
                todo_records = _brain.search(query="", tags=["todo-item"], limit=20)
            agent_todos = []
            for r in todo_records:
                meta = getattr(r, "metadata", None) or {}
                if meta.get("type") != "todo_item":
                    continue
                if meta.get("status") in ("done", "archived"):
                    continue
                title = r.content.split(": ", 1)[-1].split(" | ")[0] if ": " in r.content else r.content
                status = meta.get("status", "pending")
                marker = " *IN PROGRESS*" if status == "in_progress" else ""
                agent_todos.append(f"- [{meta.get('priority', 'medium').upper()}] {title}{marker} (id: {r.id[:8]})")
            if agent_todos:
                todos_text = "\nACTIVE TODOS:\n" + "\n".join(agent_todos[:10]) + "\n"
                todos_text += "Use update_todo to mark steps done. Use add_todo (category='agent') for new steps.\n"
        except Exception:
            pass

        # Failed hypotheses - remind agent what already didn't work
        disproved_text = ""
        try:
            from remy.core.agent_tools import brain as _brain, brain_lock as _bl
            with _bl:
                disproved = _brain.search(query="", tags=["failed-hypothesis"], limit=5)
            if disproved:
                dlines = [f"- {r.content[:150]}" for r in disproved]
                disproved_text = (
                    "\nDISPROVED APPROACHES (do NOT retry these):\n"
                    + "\n".join(dlines) + "\n"
                )
        except Exception:
            pass

        # Mission anchoring: inject IDENTITY-level purpose from brain
        mission_text = ""
        try:
            from remy.core.brain_tools import get_user_profile_record
            _prof = get_user_profile_record()
            if _prof:
                meta = getattr(_prof, "metadata", None) or {}
                user_name = meta.get("name", "the user")
                mission_text = (
                    f"\nMISSION ANCHOR (never forget):\n"
                    f"Your primary purpose is to serve {user_name}. "
                    f"Every action must directly advance the user's goals or maintain your ability to do so. "
                    f"Do NOT drift into tangential research, self-improvement for its own sake, "
                    f"or reorganizing knowledge that nobody asked for. "
                    f"If no high-priority goal exists, pick the one most useful to {user_name}.\n"
                )
        except Exception:
            mission_text = (
                "\nMISSION ANCHOR: Focus on goals that directly serve the user. "
                "Avoid tangential tasks or self-improvement for its own sake.\n"
            )

        # F1: Role-specific section
        role_text = ""
        if role:
            role_text = (
                f"\n{role.instruction_suffix}\n"
                f"PRIORITY TOOLS: {', '.join(role.priority_tools)}\n"
                f"AVOID: {', '.join(role.avoid_tools)}\n"
            )

        # Behavioral rules (learned from past failures)
        rules_text = ""
        if behavioral_rules:
            from remy.core.autonomy_rules import format_rules_for_prompt
            rules_text = format_rules_for_prompt(behavioral_rules)

        # Budget forecast + savings report
        budget_forecast_text = ""
        savings_text = ""
        try:
            from remy.core.budget_negotiation import format_budget_forecast, savings_tracker
            budget_forecast_text = format_budget_forecast(goals)
            savings_text = savings_tracker.format_report()
        except Exception:
            pass

        return (
            "=== AUTONOMOUS MODE ===\n"
            "You are running autonomously. No human is present. Decide what to do.\n\n"
            f"{mission_text}\n"
            f"ACTIVE GOALS:\n{goals_text}\n"
            f"{plan_text}"
            f"{research_text}"
            f"{role_text}"
            f"\nRECENT OUTCOMES:\n{past_outcomes or 'No recent actions.'}\n"
            f"{reflection_text}"
            f"{strategy_text}"
            f"{disproved_text}"
            f"{rules_text}"
            f"{health_text}"
            f"{budget_forecast_text}"
            f"{savings_text}"
            f"{preflight_text}"
            f"{loop_warning_text}"
            f"\nBUDGET: {budget['tokens_today']}/{budget['daily_limit']} tokens today, "
            f"{budget['tokens_this_hour']}/{budget['hourly_limit']} this hour\n\n"
            "TOOL COSTS (estimated tokens per call):\n"
            "- recall (~50), search (~50), insights (~50), get_current_datetime (~20) - FREE (zero LLM)\n"
            "- store (~30), connect_records (~30), update_record (~30) - FREE (zero LLM)\n"
            "- web_search (~800) - EXPENSIVE (LLM + Google API). Check cache first via recall.\n"
            "- http_get (~100-500) - MODERATE (network I/O, no LLM)\n"
            "- start_research (~800) - EXPENSIVE (LLM generates query plan)\n"
            "- add_research_finding (~30) - FREE (storage only)\n"
            "- complete_research (~500) - MODERATE (LLM synthesizes report)\n"
            "- store_research (~50) - FREE (storage only)\n"
            "- read_file (~50), write_file (~30), list_directory (~30) - FREE\n\n"
            "INSTRUCTIONS:\n"
            "1. Follow the ACTION PLAN step if one exists, otherwise pick the best action.\n"
            "2. ALWAYS recall before web_search - cached results cost zero tokens.\n"
            "3. Be efficient with tokens. Prefer zero-cost tools when budget is tight.\n"
            "4. If a goal is completed, say so clearly.\n"
            "5. For research goals: use start_research - web_search + add_research_finding (one query per cycle) - complete_research.\n"
            "6. If ACTIVE RESEARCH exists, continue it before starting new research.\n"
            "\nWhat will you do?\n"
        )

    def _scar_warning_for_plan(self, top_goal: dict, plan) -> str:
        """Return a scar warning for the next planned (goal, action) pair, or "".

        Consults long-term consequence memory for a prior world refutation of the
        exact next step. Fail-soft: any error yields no warning, so the cycle is
        never broken by the check.
        """
        try:
            goal_desc = (top_goal or {}).get("description") or ""
            if not goal_desc or plan is None:
                return ""
            steps = getattr(plan, "steps", None) or []
            idx = getattr(plan, "current_step", 0)
            if not steps or idx >= len(steps):
                return ""
            action = str(steps[idx])[:240].strip()
            if not action:
                return ""
            from remy.core.consequence_gate import (
                consult_consequence_memory,
                render_scar_warning,
            )
            verdict = consult_consequence_memory(brain, goal_desc, action)
            if verdict.is_refuted:
                return render_scar_warning(verdict)
        except Exception:
            return ""
        return ""

    def _recent_outcomes_summary(self) -> str:
        """Summarize recent outcomes from in-memory log."""
        if not self.action_log:
            return ""
        recent = self.action_log[-5:]
        from remy.core.agent_tools import brain_lock
        # Build confidence map per action_type from brain
        confidence_map: dict[str, float] = {}
        try:
            with brain_lock:
                stored = brain.search(query="", tags=["autonomous-outcome"], limit=100)
            for r in stored:
                meta = r.metadata or {}
                atype = meta.get("action_type")
                conf = meta.get("confidence")
                if atype and conf is not None:
                    confidence_map[atype] = conf  # last stored = most up-to-date
        except Exception:
            pass

        lines = []
        for a in recent:
            status = "OK" if a.success else "FAIL"
            conf = confidence_map.get(a.action_type)
            conf_str = f" conf={conf:.0%}" if conf is not None else ""
            lines.append(f"- [{status}] {a.description[:80]} ({a.tokens_used} tokens{conf_str})")
        return "\n".join(lines)

    def _seed_initial_goals(self):
        """Create starter goals for first-time autonomous agent."""
        self._generate_smart_goals()

    def _generate_smart_goals(self) -> list[str]:
        """Generate context-aware goals from user data, time, and past sessions.

        Returns list of created goal record IDs.
        """
        # Gather context
        from remy.core.agent_tools import brain_lock
        from remy.core.brain_tools import get_user_profile_record
        _prof_rec = get_user_profile_record()
        with brain_lock:
            reflections = brain.search(query="", tags=["session-reflection"], limit=2)
        now = datetime.now()

        user_info = ""
        if _prof_rec:
            user_info = _prof_rec.content[:300]

        reflection_info = ""
        if reflections:
            reflection_info = "\n".join(r.content[:150] for r in reflections)

        context_prompt = (
            "You are an AI assistant generating autonomous goals for yourself.\n"
            "Based on the context below, suggest 2-3 specific, actionable goals.\n"
            "Respond ONLY with a JSON array of objects: "
            '[{"description": "...", "priority": "medium|high|low"}]\n\n'
            f"TIME: {now.strftime('%A %H:%M, %B %d')}\n"
            f"USER PROFILE: {user_info or 'Unknown user'}\n"
            f"PAST LESSONS: {reflection_info or 'No previous sessions'}\n\n"
            "Goals should be concrete and achievable. Respond with JSON only:"
        )

        try:
            from remy.core.llm import call_llm
            from concurrent.futures import ThreadPoolExecutor

            # Cap goal generation to 60s - default LLM retry chain can block 30+ min
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(call_llm, context_prompt, purpose="generate_goals")
                result = future.result(timeout=60)
            raw = _llm_content_to_str(result.content).strip()

            # Track usage
            reported_tokens = _extract_usage_tokens(result)
            est_tokens = reported_tokens or estimate_tokens(context_prompt + raw)
            self.budget.record_usage(est_tokens)
            save_budget(self.budget)
            from remy.core.usage_stats import usage_tracker
            usage_tracker.record_usage(
                "autonomy",
                est_tokens,
                kind="reported" if reported_tokens else "estimated",
            )

            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            goal_defs = _parse_llm_json(raw)
            if not isinstance(goal_defs, list):
                goal_defs = []

        except Exception as e:
            logger.warning("Smart goal generation failed, using defaults: %s", e)
            goal_defs = [
                {"description": "Review and organize stored knowledge", "priority": "medium"},
                {"description": "Research one interesting topic", "priority": "low"},
            ]

        created = []
        for gdef in goal_defs[:3]:
            desc = gdef.get("description", "") if isinstance(gdef, dict) else str(gdef)
            priority = gdef.get("priority", "medium") if isinstance(gdef, dict) else "medium"
            if desc:
                gid = create_goal(desc.strip(), priority=priority)
                created.append(gid)

        logger.info("Generated %d smart goals", len(created))
        return created

    def _analyze_strategy_effectiveness(self) -> str:
        """Analyze which action types succeed/fail most, with weekly trend. Pure Python, zero LLM cost."""
        from datetime import datetime, timedelta
        from remy.core.agent_tools import brain_lock

        now = datetime.now()
        this_week = now.strftime("week:%Y-W%W")
        last_week = (now - timedelta(weeks=1)).strftime("week:%Y-W%W")

        with brain_lock:
            outcomes_this = brain.search(query="", tags=["autonomous-outcome", this_week], limit=100)
            outcomes_last = brain.search(query="", tags=["autonomous-outcome", last_week], limit=100)
            outcomes_all  = brain.search(query="", tags=["autonomous-outcome"], limit=50)

        if not outcomes_all:
            return ""

        def _rate(records: list) -> dict[str, dict]:
            stats: dict[str, dict] = {}
            for rec in records:
                meta = rec.metadata or {}
                atype = meta.get("action_type", "unknown")
                success = meta.get("success", True)
                if atype not in stats:
                    stats[atype] = {"success": 0, "fail": 0}
                if success:
                    stats[atype]["success"] += 1
                else:
                    stats[atype]["fail"] += 1
            return stats

        stats_all  = _rate(outcomes_all)
        stats_this = _rate(outcomes_this)
        stats_last = _rate(outcomes_last)

        lines = []
        for atype, counts in sorted(stats_all.items()):
            total = counts["success"] + counts["fail"]
            if total < 2:
                continue
            rate = counts["success"] / total * 100
            emoji = "?" if rate >= 70 else "?" if rate >= 40 else "?"

            # Weekly trend
            trend = ""
            tw = stats_this.get(atype)
            lw = stats_last.get(atype)
            if tw and lw:
                tw_total = tw["success"] + tw["fail"]
                lw_total = lw["success"] + lw["fail"]
                if tw_total >= 2 and lw_total >= 2:
                    tw_rate = tw["success"] / tw_total * 100
                    lw_rate = lw["success"] / lw_total * 100
                    diff = tw_rate - lw_rate
                    if diff >= 10:
                        trend = " -"
                    elif diff <= -10:
                        trend = " -"

            lines.append(f"{emoji} {atype}: {rate:.0f}% success ({total} attempts){trend}")

        return "\n".join(lines)

    def _get_last_reflection(self) -> str:
        """Retrieve the last session reflection from brain."""
        from remy.core.agent_tools import brain_lock
        with brain_lock:
            reflections = brain.search(query="", tags=["session-reflection"], limit=1)
        if reflections:
            return reflections[0].content[:300]
        return ""

    def _analyze_goal_type_feedback(self) -> str:
        """Detect which goal-type keywords consistently fail. Zero LLM."""
        from remy.core.agent_tools import brain_lock
        with brain_lock:
            outcomes = brain.search(query="", tags=["autonomous-outcome"], limit=50)
        if not outcomes:
            return ""

        keyword_stats: dict[str, dict[str, int]] = {}
        _keywords = ("research", "organize", "learn", "health", "analyze", "write", "read", "plan")

        for rec in outcomes:
            meta = getattr(rec, "metadata", None) or {}
            success = meta.get("success", False)
            desc = rec.content.lower()
            for kw in _keywords:
                if kw in desc:
                    if kw not in keyword_stats:
                        keyword_stats[kw] = {"success": 0, "fail": 0}
                    if success:
                        keyword_stats[kw]["success"] += 1
                    else:
                        keyword_stats[kw]["fail"] += 1

        lines = []
        for kw, stats in sorted(keyword_stats.items()):
            total = stats["success"] + stats["fail"]
            if total >= 3:
                rate = stats["success"] / total
                if rate < 0.4:
                    lines.append(f"'{kw}' goals have low success ({rate:.0%}). Consider simpler approaches.")
        return "\n".join(lines)

    # ============== RESEARCH-AWARE GOALS (RM-3) ==============

    _RESEARCH_KEYWORDS = frozenset([
        "research", "investigate", "find out", "learn about", "study",
        "\u0434\u043e\u0441\u043b\u0456\u0434\u0436",
        "\u0434\u0456\u0437\u043d\u0430\u0442\u0438",
        "\u0432\u0438\u0432\u0447\u0438",
        "\u0437\u043d\u0430\u0439\u0434\u0438 \u0456\u043d\u0444\u043e\u0440\u043c\u0430\u0446\u0456\u044e",
    ])

    def _is_research_goal(self, description: str) -> bool:
        """Detect if a goal description implies research."""
        desc_lower = description.lower()
        return any(kw in desc_lower for kw in self._RESEARCH_KEYWORDS)

    def _ensure_research_project(self, goal: dict):
        """If goal is research-like and has no project yet, create one.

        Tags the goal metadata with is_research=True and project_id.
        Zero LLM cost if project already exists (just metadata check).
        """
        try:
            from remy.core.agent_tools import brain_lock
            with brain_lock:
                rec = brain.get(goal["record_id"])
            if not rec:
                return
            meta = dict(rec.metadata or {})

            # Already tagged or has a project
            if meta.get("is_research") is False:
                return
            if meta.get("research_project_id"):
                return

            # Check if it looks like research
            if not self._is_research_goal(goal["description"]):
                meta["is_research"] = False
                with brain_lock:
                    brain.update(goal["record_id"], metadata=meta)
                return

            # Check budget - start_research costs ~800 tokens
            can_spend, _ = self.budget.can_spend(800)
            if not can_spend:
                return

            # Extract topic from goal description
            desc = goal["description"]
            # Remove goal prefix like "Goal [HIGH]: "
            for prefix in ("Goal [HIGH]: ", "Goal [MEDIUM]: ", "Goal [LOW]: ", "Goal [CRITICAL]: "):
                if desc.startswith(prefix):
                    desc = desc[len(prefix):]
                    break

            from remy.core.brain_tools import _start_research
            import json as _json

            result = _json.loads(_start_research(
                {"topic": desc.strip(), "depth": "standard"},
                session_id=self.session_id,
            ))

            if result.get("created"):
                meta["is_research"] = True
                meta["research_project_id"] = result["project_id"]
                with brain_lock:
                    brain.update(goal["record_id"], metadata=meta)
                logger.info(
                    "Auto-created research project '%s' for goal: %s",
                    result["project_id"], desc[:60],
                )
                # Track budget for plan generation
                self.budget.record_usage(800)
                save_budget(self.budget)
                
                # Track global usage
                from remy.core.usage_stats import usage_tracker
                usage_tracker.record_usage("autonomy", 800, kind="estimated")
        except Exception as e:
            logger.warning("Research project auto-creation failed: %s", e)

    async def _generate_session_reflection(self) -> str | None:
        """Use LLM to reflect on this session's actions and extract lessons.

        Returns the reflection text, or None if LLM fails.
        """
        if not self.action_log:
            return None

        action_summary = "\n".join(
            f"- {'OK' if a.success else 'FAIL'}: {a.description[:100]} ({a.tokens_used} tokens)"
            for a in self.action_log[-10:]
        )

        reflect_prompt = (
            "You are an AI agent reflecting on your autonomous session.\n"
            "Analyze the actions below and extract 2-3 concise lessons.\n"
            "Focus on: what worked, what failed, and what to do differently.\n\n"
            f"ACTIONS THIS SESSION:\n{action_summary}\n\n"
            "Respond with 2-3 short bullet points (no JSON):\n"
        )

        try:
            from remy.core.llm import call_llm_async

            result = await call_llm_async(reflect_prompt, purpose="session_reflection")
            reflection = _llm_content_to_str(result.content).strip()[:500]

            # Track usage
            reported_tokens = _extract_usage_tokens(result)
            est_tokens = reported_tokens or estimate_tokens(reflect_prompt + reflection)
            self.budget.record_usage(est_tokens)
            save_budget(self.budget)
            from remy.core.usage_stats import usage_tracker
            usage_tracker.record_usage(
                "autonomy",
                est_tokens,
                kind="reported" if reported_tokens else "estimated",
            )

            # Store in brain
            from remy.core.agent_tools import Level
            from remy.core.agent_tools import brain_lock
            with brain_lock:
                brain.store(
                    content=reflection,
                    level=Level.DOMAIN,
                    tags=["session-reflection", "autonomous-session"],
                    metadata={
                        "type": "session_reflection",
                        "session_id": self.session_id,
                        "action_count": len(self.action_log),
                        "timestamp": datetime.now().isoformat(),
                        "source": "agent-autonomous",
                        "verified": False,
                        "trust_score": 0.4,
                        "admission_class": "reflection",
                    },
                )

            logger.info("Session reflection stored: %s", reflection[:100])
            event_bus.emit("reflection", {
                "session_id": self.session_id,
                "content": reflection[:300],
            })
            return reflection

        except Exception as e:
            logger.warning("Session reflection failed: %s", e)
            return None

    async def _notify_action(self, action: ActionRecord):
        """Send notification about autonomous action via notification_router."""
        if not settings.AUTONOMY_TELEGRAM_NOTIFICATIONS:
            return

        status = "done" if action.success else "FAILED"
        remaining = self.budget.daily_limit - self.budget.tokens_today
        msg = (
            f"[Autonomous] {status}: {action.description[:100]}\n"
            f"Tokens: {action.tokens_used} | Remaining today: {remaining}"
        )
        try:
            from remy.core.notification_router import notify
            notify(msg, level="info" if action.success else "warning",
                   event_type="autonomous.action",
                   event_data={"action_id": action.action_id, "success": action.success, "tokens_used": action.tokens_used},
                   parse_mode="")
        except Exception as e:
            logger.debug("Action notify failed: %s", e)

    async def _shutdown(self):
        """Cleanup on loop exit."""
        logger.info("Autonomous mode shutting down")
        _write_brain_snapshot(self.session_id, "STOP")
        logger.info(
            "Session stats: %d actions, %d tokens used",
            len(self.action_log), self.budget.tokens_this_session,
        )

        # Save budget state for next session
        save_budget(self.budget)

        # Generate session reflection (LLM-based) - timeout to prevent hanging
        try:
            await asyncio.wait_for(self._generate_session_reflection(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("Reflection generation timed out")
        except Exception as e:
            logger.warning("Reflection generation failed: %s", e)

        try:
            from remy.core.agent_tools import Level
            from remy.core.agent_tools import brain_lock
            summary = (
                f"Autonomous session {self.session_id}: "
                f"{len(self.action_log)} actions, "
                f"{self.budget.tokens_this_session} tokens, "
                f"{sum(1 for a in self.action_log if a.success)} successes, "
                f"{sum(1 for a in self.action_log if not a.success)} failures."
            )
            with brain_lock:
                brain.store(
                    content=summary,
                    level=Level.DOMAIN,
                    tags=["session-summary", "autonomous-session"],
                    metadata={
                        "type": "autonomous_session_summary",
                        "session_id": self.session_id,
                        "action_count": len(self.action_log),
                        "tokens_used": self.budget.tokens_this_session,
                        "source": "agent-autonomous",
                        "verified": False,
                        "trust_score": 0.4,
                        "admission_class": "reflection",
                    },
                )
        except Exception as e:
            logger.warning("Failed to store session summary: %s", e)


# ============== BLOCKER / RESUME DETECTION ==============


def _detect_external_blocker(goal: dict, session_log: list) -> dict | None:
    """Detect external blockers from session log (email verification, captcha, etc).

    Returns a dict with 'reason' and 'evidence' keys, or None if no blocker found.
    """
    _BLOCKER_PATTERNS = [
        ("email verification required", ["check your email", "verify your email", "confirm your email"]),
        ("captcha required", ["captcha", "i'm not a robot", "verify you are human"]),
        ("phone verification required", ["verify your phone", "enter your phone number", "sms code"]),
        ("account suspended", ["account suspended", "account banned", "account disabled"]),
        ("payment required", ["payment required", "upgrade your plan", "billing required"]),
    ]

    for entry in session_log:
        evidence = entry.get("evidence", {})
        text = (
            entry.get("answer", "") + " " +
            evidence.get("page_text_snippet", "") + " " +
            evidence.get("page_url", "")
        ).lower()

        for reason, keywords in _BLOCKER_PATTERNS:
            if any(kw in text for kw in keywords):
                return {"reason": reason, "evidence": evidence}

    return None


def _detect_resume_state_reset(goal: dict, session_log: list) -> str | None:
    """Detect if agent ended up at wrong page after resume (state reset).

    Returns "State reset detected: <reason>" string, or None if no reset detected.
    """
    resume_context = goal.get("resume_context", "")
    if not resume_context or not session_log:
        return None

    # Signals that suggest agent is at an earlier/wrong stage than expected
    _RESET_SIGNALS = [
        "signup", "create account", "register", "login", "sign in", "start over",
        "draft your post", "compose", "/compose/", "/signup", "/register",
    ]

    for entry in session_log:
        evidence = entry.get("evidence", {})
        text = (
            evidence.get("page_text_snippet", "") + " " +
            evidence.get("page_url", "")
        ).lower()

        for signal in _RESET_SIGNALS:
            if signal in text:
                return f"State reset detected: page indicates '{signal}' instead of expected resume state"

    return None


# ============== ENTRY POINT ==============


async def run_autonomous():
    """Entry point for --autonomous mode."""
    from remy.core.combined_runner import run_autonomy_standalone

    await run_autonomy_standalone()


# ============== BACKWARD-COMPAT EXPORTS ==============
# These functions were previously in autonomy.py but moved to orchestrator.py.
# Keep them importable from both locations for backward compatibility.

from remy.core.orchestrator import (
    _should_use_browser_worker,
    _should_use_research_worker,
    format_execution_report as _format_execution_report,
)

def _invoke_goal_worker(goal: dict, session_id: str | None = None, channel: str = "autonomous") -> str:
    """Backward-compat stub: invoke a worker for a goal."""
    from remy.core.orchestrator import dispatch_worker
    return dispatch_worker(goal, session_id=session_id, channel=channel)

# Backoff constants (used by heartbeat stability tests)
BACKOFF_BASE_SEC: float = 120.0           # 2 minutes base
BACKOFF_MAX_SEC: float = 900.0            # 15 minutes max
BACKOFF_FACTOR: float = 2.0
FAST_RECOVERY_DELAYS: list = [30, 60, 120]  # Fast recovery sequence
INVOKE_AGENT_TIMEOUT_SEC: int = 90           # LLM invoke timeout
