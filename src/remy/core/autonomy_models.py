"""
Autonomy Models — data classes, budget, logger, agent roles.

Stateless models + budget persistence + role definitions extracted from autonomy.py.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("Autonomy")


def _get_autonomy():
    """Lazy accessor — reads from autonomy module (supports test patching)."""
    import remy.core.autonomy as _au

    return _au


# ============== UTILITY ==============


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


# ============== AUTONOMY LOG (separate file) ==============


def _setup_autonomy_logger():
    """Set up a dedicated log file for autonomous mode actions."""
    au = _get_autonomy()
    settings = au.settings
    from remy.core.logging_config import setup_autonomy_file_handler

    log_dir = settings.DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_autonomy_file_handler(logger, log_dir)


# ============== BUDGET PERSISTENCE ==============

BUDGET_FILE = "autonomy_budget.json"


def _get_budget_path() -> Path:
    au = _get_autonomy()
    return au.settings.DATA_DIR / BUDGET_FILE


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
        "cost_today_usd": round(budget.cost_today_usd, 6),
        "total_cost_lifetime_usd": round(budget.total_cost_lifetime_usd, 6),
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

    # Restore lifetime totals (always)
    budget.total_tokens_lifetime = data.get("total_tokens_lifetime", 0)
    budget.total_cost_lifetime_usd = data.get("total_cost_lifetime_usd", 0.0)

    # Restore daily counter if within same day window
    last_day = data.get("last_day_reset", 0)
    if now - last_day < 86400:
        budget.tokens_today = data.get("tokens_today", 0)
        budget.cost_today_usd = data.get("cost_today_usd", 0.0)
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
        "Budget restored: %d today, %d this hour, %d lifetime, $%.4f today",
        budget.tokens_today,
        budget.tokens_this_hour,
        budget.total_tokens_lifetime,
        budget.cost_today_usd,
    )


# ============== RESOURCE BUDGET ==============


@dataclass
class ResourceBudget:
    """Tracks token usage and enforces budget caps."""

    daily_limit: int
    hourly_limit: int
    session_limit: int
    daily_cost_limit_usd: float = 0.0  # 0 = no USD limit
    tokens_today: int = 0
    tokens_this_hour: int = 0
    tokens_this_session: int = 0
    cost_today_usd: float = 0.0
    cost_this_session_usd: float = 0.0
    total_cost_lifetime_usd: float = 0.0
    last_hour_reset: float = field(default_factory=time.time)
    last_day_reset: float = field(default_factory=time.time)
    total_tokens_lifetime: int = 0

    def __post_init__(self):
        # Coerce to float — protects against MagicMock from test mocks
        try:
            self.daily_cost_limit_usd = float(self.daily_cost_limit_usd)
        except (TypeError, ValueError):
            self.daily_cost_limit_usd = 0.0

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
            self.cost_today_usd = 0.0
            self.last_day_reset = now

        if self.tokens_this_session + estimated_tokens > self.session_limit:
            return False, f"Session limit reached ({self.tokens_this_session}/{self.session_limit})"
        if self.tokens_this_hour + estimated_tokens > self.hourly_limit:
            return False, f"Hourly limit reached ({self.tokens_this_hour}/{self.hourly_limit})"
        if self.tokens_today + estimated_tokens > self.daily_limit:
            return False, f"Daily limit reached ({self.tokens_today}/{self.daily_limit})"

        # USD daily cost check
        if self.daily_cost_limit_usd > 0 and self.cost_today_usd >= self.daily_cost_limit_usd:
            return (
                False,
                f"Daily cost limit reached (${self.cost_today_usd:.4f}/${self.daily_cost_limit_usd:.2f})",
            )

        return True, "ok"

    def record_usage(self, tokens: int, cost_usd: float = 0.0):
        """Record token usage after an action."""
        self.tokens_today += tokens
        self.tokens_this_hour += tokens
        self.tokens_this_session += tokens
        self.total_tokens_lifetime += tokens
        self.cost_today_usd += cost_usd
        self.cost_this_session_usd += cost_usd
        self.total_cost_lifetime_usd += cost_usd

    def to_dict(self) -> dict:
        return {
            "daily_limit": self.daily_limit,
            "hourly_limit": self.hourly_limit,
            "session_limit": self.session_limit,
            "daily_cost_limit_usd": self.daily_cost_limit_usd,
            "tokens_today": self.tokens_today,
            "tokens_this_hour": self.tokens_this_hour,
            "tokens_this_session": self.tokens_this_session,
            "cost_today_usd": round(self.cost_today_usd, 6),
            "cost_this_session_usd": round(self.cost_this_session_usd, 6),
            "total_cost_lifetime_usd": round(self.total_cost_lifetime_usd, 6),
            "total_tokens_lifetime": self.total_tokens_lifetime,
        }


# ============== AGENT ROLES (MULTI-AGENT DELEGATION) ==============


@dataclass
class AgentRole:
    """Defines a specialized agent role for autonomous delegation."""

    name: str
    description: str
    priority_tools: list[str]
    avoid_tools: list[str]
    instruction_suffix: str
    max_tool_iterations: int = 15
    timeout_sec: int = 0  # 0 = use settings.WORKER_TIMEOUT_SEC


AGENT_ROLES: dict[str, AgentRole] = {
    "researcher": AgentRole(
        name="researcher",
        description="Deep investigation and knowledge gathering",
        priority_tools=[
            "recall",
            "web_search",
            "start_research",
            "add_research_finding",
            "complete_research",
            "store_research",
            "extract_facts",
        ],
        avoid_tools=["write_file", "sandbox_create_tool"],
        instruction_suffix=(
            "ROLE: RESEARCHER\n"
            "You are in research mode. Gather, verify, and synthesize information.\n"
            "- Start with recall (free) before web_search (expensive)\n"
            "- Use start_research for multi-query investigations\n"
            "- Cross-reference findings against existing knowledge\n"
            "- Store conclusions via store_research or store\n"
            "- Do NOT create files, tools, or goals — just gather knowledge\n"
        ),
        max_tool_iterations=10,
    ),
    "planner": AgentRole(
        name="planner",
        description="Goal creation, decomposition, and plan management",
        priority_tools=[
            "recall",
            "search",
            "create_subgoal",
            "complete_goal",
            "add_todo",
            "list_todos",
            "update_todo",
            "schedule_task",
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
            "- Do NOT execute actions or do research — just plan\n"
        ),
        max_tool_iterations=8,
    ),
    "executor": AgentRole(
        name="executor",
        description="Execute concrete plan steps and external actions",
        priority_tools=[
            "read_file",
            "write_file",
            "http_get",
            "list_directory",
            "browse_page",
            "browser_act",
            "browser_close",
            "store",
            "update_record",
            "connect_records",
            "update_todo",
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
            "- Do NOT plan or research — execute the plan step\n"
        ),
        max_tool_iterations=12,
    ),
    "analyst": AgentRole(
        name="analyst",
        description="Data analysis, metric intelligence, pattern detection",
        priority_tools=[
            "metric_summary",
            "event_correlate",
            "extract_facts",
            "recall",
            "search",
            "consolidate",
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
            "- Do NOT gather new data — analyze what exists\n"
        ),
        max_tool_iterations=8,
    ),
    "osint": AgentRole(
        name="osint",
        description="Open-source intelligence: market research, competitive analysis, lead discovery",
        priority_tools=[
            "recall",
            "web_search",
            "extract_content",
            "http_get",
            "store",
            "search",
            "start_research",
            "add_research_finding",
            "complete_research",
            "store_research",
            "scratchpad",
        ],
        avoid_tools=["write_file", "sandbox_create_tool", "browse_page", "browser_act"],
        instruction_suffix=(
            "ROLE: OSINT INVESTIGATOR\n"
            "You are an open-source intelligence specialist.\n"
            "- Gather competitive intelligence, market data, promotional opportunities\n"
            "- Use web_search for discovery, extract_content for deep page reading\n"
            "- Use start_research for multi-query investigations\n"
            "- Cross-reference from multiple sources before concluding\n"
            "- Use scratchpad for intermediate findings between tool calls\n"
            "- Score source credibility: official docs > reports > blogs > forums\n"
            "- Store conclusions with source URLs and confidence levels\n"
            "- COST DISCIPLINE: recall first (free), extract_content over http_get for articles\n"
            "- DEEP SEARCH: use site: operator for targeted results:\n"
            "  site:reddit.com for community discussions and real user opinions\n"
            "  site:news.ycombinator.com for tech community sentiment\n"
            "  site:github.com for repos, issues, and technical comparisons\n"
            "  site:dev.to OR site:medium.com for technical articles\n"
            "  filetype:pdf for whitepapers and reports\n"
            '  Combine: site:reddit.com "agent memory" alternatives\n'
        ),
        max_tool_iterations=20,
        timeout_sec=180,
    ),
}
