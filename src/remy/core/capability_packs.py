"""
Capability Packs — explicit, structured operating profiles for workers.

Each pack bundles:
- worker type (browser_worker, research_worker)
- allowed tools
- guardrails (hard rules the agent must follow)
- approval mode (when human approval is required)
- metrics family (for per-pack tracking in task_metrics)
- system prompt additions

Orchestrator resolves a pack for each goal cycle:
1. Explicit: goal metadata has `goal_template` matching a pack ID → use it
2. Inference: keyword-based fallback from goal description
3. Default: "general" pack (no special rules)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urlparse

from remy.config.settings import settings

logger = logging.getLogger("CapabilityPacks")


@dataclass(frozen=True)
class CapabilityPack:
    """One operational capability profile."""

    id: str
    label: str
    worker: str  # "browser_worker" | "research_worker" | "generic"
    metrics_family: str  # maps to task_metrics TASK_FAMILIES
    tools: tuple[str, ...]  # allowed tool names
    guardrails: tuple[str, ...]  # hard rules injected into worker prompt
    approval_mode: str  # "none" | "publish" | "financial" | "all_clicks"
    description: str = ""
    step_budget: int = 0  # max tool iterations; 0 = use role default
    timeout_sec: int = 0  # execution timeout; 0 = use role default
    modes: tuple[str, ...] = ()
    channels: tuple[str, ...] = ()
    research_mode: str = ""
    source_scope: str = ""
    citation_required: bool = False


# ============================================================
# Pack Definitions
# ============================================================

SIGNUP_OPERATOR = CapabilityPack(
    id="signup_operator",
    label="Signup Operator",
    worker="browser_worker",
    metrics_family="signup_operator",
    tools=(
        "browse_page",
        "browser_act",
        "browser_close",
        "store",
        "recall",
    ),
    guardrails=(
        "Do NOT enter payment information or credit card details.",
        "If blocked by captcha, email verification, SMS, or KYC — report blocked_external immediately.",
        "Verify typed email/username values match expected before clicking submit.",
        "If login fails 3+ times with the same error, stop and report blocked_external.",
        "Do not create accounts on financial or crypto services.",
    ),
    approval_mode="none",  # signup itself doesn't need approval (unless financial)
    description="Handles account creation and login flows on web services.",
    step_budget=15,  # signup flows are bounded — 15 steps is generous
    timeout_sec=90,  # browser flows need more than 60s default
)

PUBLISHER = CapabilityPack(
    id="publisher",
    label="Publisher",
    worker="browser_worker",
    metrics_family="publisher",
    tools=(
        "browse_page",
        "browser_act",
        "browser_close",
        "store",
        "recall",
    ),
    guardrails=(
        "ALWAYS stop at draft/queue state — never publish live without explicit approval.",
        "If the page shows 'published' or 'post is live', report this immediately as accidental publish.",
        "Do not edit or delete existing published content.",
        "Do not post comments or replies without draft review.",
        "Keep all generated content factual — do not fabricate claims or statistics.",
        "Cite sources for any comparison or competitive claims.",
    ),
    approval_mode="all_clicks",  # every click on publishing platforms needs approval
    description="Creates draft content on publishing platforms. Never publishes live without approval.",
    step_budget=12,  # publishing is focused — 12 steps max
    timeout_sec=90,
    modes=("comment", "post", "article"),
    channels=("x", "reddit", "devto", "generic"),
)

MARKET_RESEARCH = CapabilityPack(
    id="market_research",
    label="Market Research",
    worker="research_worker",
    metrics_family="market_research",
    tools=(
        "web_search",
        "extract_content",
        "http_get",
        "store",
        "recall",
        "start_research",
        "add_research_finding",
        "get_research_status",
    ),
    guardrails=(
        "Verify claims from at least 2 independent sources before storing.",
        "Tag contradictions explicitly — do not silently overwrite prior findings.",
        "Do not contact companies or individuals directly.",
        "Store raw source URLs with every finding.",
        "If a source is paywalled or inaccessible, note it and move on.",
    ),
    approval_mode="none",
    description="Competitive analysis, market research, OSINT gathering.",
    step_budget=20,  # research needs room to explore
    timeout_sec=180,
    research_mode="balanced",
    source_scope="web",
    citation_required=True,
)

MONITORING = CapabilityPack(
    id="monitoring",
    label="Monitoring",
    worker="monitoring_worker",
    metrics_family="monitoring",
    tools=(
        "web_search",
        "extract_content",
        "http_get",
        "store",
        "recall",
    ),
    guardrails=(
        "Compare current content against the stored snapshot — do not re-research from zero.",
        "Report only actual changes, not stylistic rewordings or formatting differences.",
        "Store change events with before/after summaries and source URL.",
        "If a page is unreachable, record 'unreachable' and move on — do not retry endlessly.",
        "Do NOT modify, comment on, or interact with monitored pages.",
    ),
    approval_mode="none",
    description="Tracks changes on competitor/product pages over time. Read-only observation.",
    step_budget=8,  # monitoring is simple: fetch + compare, few steps
    timeout_sec=120,
    research_mode="speed",
    source_scope="web",
    citation_required=True,
)

GENERAL = CapabilityPack(
    id="general",
    label="General",
    worker="generic",
    metrics_family="general",
    tools=(),  # no restriction — all tools available
    guardrails=(),
    approval_mode="none",
    description="Default pack for goals that don't match a specific capability.",
)


_PUBLISHER_MODE_KEYWORDS = {
    "comment": (
        "comment",
        "reply",
        "respond",
        "engage",
        "reaction",
    ),
    "article": (
        "article",
        "blog",
        "blog post",
        "dev.to",
        "devto",
        "writeup",
        "post article",
        "publish article",
        "longform",
    ),
    "post": (
        "post",
        "tweet",
        "thread",
        "status",
        "share post",
        "publish post",
        "social post",
    ),
}

_PUBLISHER_CHANNEL_DOMAINS = {
    "x.com": "x",
    "twitter.com": "x",
    "reddit.com": "reddit",
    "www.reddit.com": "reddit",
    "dev.to": "devto",
}

_PUBLISHER_CHANNEL_KEYWORDS = {
    "x": ("x.com", "twitter", "tweet", "thread", "x "),
    "reddit": ("reddit", "subreddit", "thread"),
    "devto": ("dev.to", "devto", "article"),
}

_PUBLISHER_PLAYBOOKS: dict[str, dict[str, tuple[str, ...]]] = {
    "x": {
        "comment": (
            "Keep it under 3 short sentences.",
            "Be specific and useful; avoid generic praise.",
            "Reference the exact post/topic before mentioning Remy or Aura.",
        ),
        "post": (
            "Lead with one concrete insight, not a slogan.",
            "Keep it scannable and avoid long paragraphs.",
            "End with a soft CTA or question, not a hard sell.",
        ),
        "article": (
            "Do not write a full article inside X; create a draft thread or summary instead.",
            "Keep each draft segment concise and evidence-backed.",
        ),
    },
    "reddit": {
        "comment": (
            "Match the thread tone and answer the thread, not your own agenda.",
            "Avoid marketing language or obvious promotion.",
            "Only mention Remy or Aura when directly relevant to the discussion.",
        ),
        "post": (
            "Write like a practitioner sharing specifics, not a brand announcement.",
            "State the setup, result, and tradeoff clearly.",
            "Avoid hype and unverifiable claims.",
        ),
        "article": (
            "Convert long-form ideas into a discussion-friendly text post.",
            "Use concrete examples and avoid copy that reads like a landing page.",
        ),
    },
    "devto": {
        "comment": (
            "Be brief, technical, and directly tied to the article.",
            "Add one useful observation or implementation detail.",
        ),
        "post": (
            "Prefer article mode on Dev.to unless the task explicitly asks for a short update.",
            "If drafting a short post, keep it technical and linkable.",
        ),
        "article": (
            "Use a clear technical structure: hook, problem, approach, results, tradeoffs.",
            "Name competitors or comparisons carefully and cite sources for factual claims.",
            "End with a concise takeaway and next step.",
        ),
    },
    "generic": {
        "comment": (
            "Be helpful, direct, and specific to the target content.",
            "Avoid promotion-first writing.",
        ),
        "post": (
            "Keep the draft focused on one idea and one audience.",
            "Prefer clarity over hype.",
        ),
        "article": (
            "Use a structured long-form draft with explicit sections and cited claims.",
            "Prefer concrete examples and evidence-backed comparisons.",
        ),
    },
}

# ============================================================
# Registry
# ============================================================

_PACKS: dict[str, CapabilityPack] = {
    p.id: p for p in (SIGNUP_OPERATOR, PUBLISHER, MARKET_RESEARCH, MONITORING, GENERAL)
}


def get_disabled_pack_ids() -> set[str]:
    """Return runtime-disabled pack IDs."""
    disabled = {str(item).strip() for item in (settings.PACKS_DISABLED or []) if str(item).strip()}
    disabled.discard(GENERAL.id)
    return disabled


def is_pack_enabled(pack_id: str) -> bool:
    """Check whether a pack is enabled for operator/runtime use."""
    return pack_id not in get_disabled_pack_ids()


def _pack_risk_profile(pack: CapabilityPack) -> str:
    if pack.approval_mode in {"all_clicks", "all"}:
        return "high"
    if pack.approval_mode in {"publish", "financial"}:
        return "medium"
    if pack.worker == "browser_worker":
        return "medium"
    return "low"


def _pack_budget_profile(pack: CapabilityPack) -> str:
    if pack.step_budget >= 15 or pack.timeout_sec >= 180:
        return "heavy"
    if pack.step_budget >= 8 or pack.timeout_sec >= 90:
        return "moderate"
    return "light"


def _goal_field(goal: dict | str | None, key: str, default: str = "") -> str:
    """Read a field from a goal-like object while preserving string compatibility."""
    if isinstance(goal, dict):
        value = goal.get(key, default)
    else:
        value = default
    return str(value or default)


def _goal_text(goal: dict | str | None) -> str:
    """Collapse goal text/metadata into one searchable lowercase string."""
    if isinstance(goal, str):
        return goal.lower()
    if not goal:
        return ""
    text_parts = [
        _goal_field(goal, "description"),
        _goal_field(goal, "task_action"),
        _goal_field(goal, "task_done_when"),
        _goal_field(goal, "target_url"),
        _goal_field(goal, "url"),
    ]
    return " ".join(part for part in text_parts if part).lower()


def get_pack(pack_id: str) -> CapabilityPack:
    """Get a capability pack by ID. Returns GENERAL if not found."""
    return _PACKS.get(pack_id, GENERAL)


def get_all_packs(include_disabled: bool = True) -> dict[str, CapabilityPack]:
    """Get registered packs, optionally excluding runtime-disabled packs."""
    packs = dict(_PACKS)
    if include_disabled:
        return packs
    disabled = get_disabled_pack_ids()
    return {pack_id: pack for pack_id, pack in packs.items() if pack_id not in disabled}


def resolve_pack(goal: dict | None) -> CapabilityPack:
    """Resolve the best capability pack for a goal.

    Priority:
    1. Explicit goal_template matching a pack ID
    2. Keyword inference from goal description
    3. GENERAL fallback
    """
    if not goal:
        return GENERAL

    # 1. Explicit match
    template = goal.get("goal_template", "")
    if template and template in _PACKS:
        pack = _PACKS[template]
        return pack if is_pack_enabled(pack.id) else GENERAL

    # 2. Keyword inference
    desc = (goal.get("description") or "").lower()

    if any(
        kw in desc for kw in ("signup", "sign up", "register", "create account", "login", "log in")
    ):
        return SIGNUP_OPERATOR if is_pack_enabled(SIGNUP_OPERATOR.id) else GENERAL

    if any(
        kw in desc
        for kw in ("publish", "post article", "draft article", "write article", "blog post")
    ):
        return PUBLISHER if is_pack_enabled(PUBLISHER.id) else GENERAL

    if any(
        kw in desc
        for kw in (
            "monitor",
            "track changes",
            "change detection",
            "watch page",
            "check for updates",
            "detect changes",
            "snapshot",
        )
    ):
        return MONITORING if is_pack_enabled(MONITORING.id) else GENERAL

    if any(
        kw in desc
        for kw in (
            "research",
            "competitive analysis",
            "market analysis",
            "osint",
            "competitors",
            "lead discovery",
        )
    ):
        return MARKET_RESEARCH if is_pack_enabled(MARKET_RESEARCH.id) else GENERAL

    # 3. Default
    return GENERAL


def format_guardrails_for_prompt(pack: CapabilityPack) -> str:
    """Format pack guardrails as prompt text."""
    if not pack.guardrails:
        return ""
    lines = [f"- {g}" for g in pack.guardrails]
    return f"\nGUARDRAILS ({pack.label}):\n" + "\n".join(lines) + "\n"


def infer_publisher_mode(goal: dict | str | None) -> str:
    """Infer publisher mode from explicit metadata or goal text."""
    if not goal:
        return "post"

    explicit = _goal_field(goal, "publisher_mode").strip().lower()
    if explicit in PUBLISHER.modes:
        return explicit

    haystack = _goal_text(goal)

    for mode, keywords in _PUBLISHER_MODE_KEYWORDS.items():
        if any(kw in haystack for kw in keywords):
            return mode
    return "post"


def infer_publisher_channel(goal: dict | str | None) -> str:
    """Infer target publishing channel from URL or goal text."""
    if not goal:
        return "generic"

    explicit = _goal_field(goal, "publisher_channel").strip().lower()
    if explicit in PUBLISHER.channels:
        return explicit

    target_url = _goal_field(goal, "target_url") or _goal_field(goal, "url")
    if target_url:
        try:
            hostname = (urlparse(target_url).hostname or "").lower()
        except Exception:
            hostname = ""
        if hostname in _PUBLISHER_CHANNEL_DOMAINS:
            return _PUBLISHER_CHANNEL_DOMAINS[hostname]
        for domain, channel in _PUBLISHER_CHANNEL_DOMAINS.items():
            if hostname.endswith("." + domain):
                return channel

    haystack = _goal_text(goal)
    for channel, keywords in _PUBLISHER_CHANNEL_KEYWORDS.items():
        if any(kw in haystack for kw in keywords):
            return channel
    return "generic"


def get_publisher_playbook(goal: dict | str | None) -> dict[str, object]:
    """Return structured publisher playbook info for the goal."""
    mode = infer_publisher_mode(goal)
    channel = infer_publisher_channel(goal)
    channel_rules = _PUBLISHER_PLAYBOOKS.get(channel, _PUBLISHER_PLAYBOOKS["generic"])
    rules = channel_rules.get(mode) or _PUBLISHER_PLAYBOOKS["generic"].get(mode, ())
    return {
        "mode": mode,
        "channel": channel,
        "rules": tuple(rules),
    }


def format_publisher_playbook_for_prompt(goal: dict | str | None) -> str:
    """Format publisher mode/channel playbook for prompt injection."""
    playbook = get_publisher_playbook(goal)
    rules = playbook["rules"]
    if not rules:
        return ""
    lines = [f"- {rule}" for rule in rules]
    return (
        "\nPUBLISHER MODE:\n"
        f"- Mode: {playbook['mode']}\n"
        f"- Channel: {playbook['channel']}\n"
        "PLAYBOOK:\n" + "\n".join(lines) + "\n"
    )


def pack_summary(pack: CapabilityPack) -> dict:
    """Compact dict for API/UI display."""
    return {
        "id": pack.id,
        "label": pack.label,
        "enabled": is_pack_enabled(pack.id),
        "worker": pack.worker,
        "metrics_family": pack.metrics_family,
        "approval_mode": pack.approval_mode,
        "guardrails_count": len(pack.guardrails),
        "tools_count": len(pack.tools),
        "modes": list(pack.modes),
        "channels": list(pack.channels),
        "description": pack.description,
        "step_budget": pack.step_budget,
        "timeout_sec": pack.timeout_sec,
        "research_mode": pack.research_mode,
        "source_scope": pack.source_scope,
        "citation_required": pack.citation_required,
        "risk_profile": _pack_risk_profile(pack),
        "budget_profile": _pack_budget_profile(pack),
        "tool_scope": "all tools" if not pack.tools else f"{len(pack.tools)} tools",
    }


_RESEARCH_MODES = ("speed", "balanced", "deep")
_SOURCE_SCOPES = ("web", "discussions", "papers", "domain")


def resolve_research_mode(goal: dict | None, pack: CapabilityPack | None = None) -> str:
    """Resolve research mode with explicit metadata first, then pack defaults."""
    explicit = str((goal or {}).get("research_mode", "") or "").strip().lower()
    if explicit in _RESEARCH_MODES:
        return explicit
    if pack and pack.research_mode in _RESEARCH_MODES:
        return pack.research_mode
    return "balanced"


def resolve_source_scope(goal: dict | None, pack: CapabilityPack | None = None) -> str:
    """Resolve source scope with explicit metadata first, then pack defaults."""
    explicit = str((goal or {}).get("source_scope", "") or "").strip().lower()
    if explicit in _SOURCE_SCOPES:
        return explicit
    if pack and pack.source_scope in _SOURCE_SCOPES:
        return pack.source_scope
    return "web"


def resolve_source_domains(goal: dict | None) -> list[str]:
    """Resolve and normalize source_domains metadata."""
    raw = (goal or {}).get("source_domains", [])
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",")]
    if not isinstance(raw, (list, tuple)):
        return []
    domains: list[str] = []
    for item in raw:
        value = str(item or "").strip().lower()
        if not value:
            continue
        if value.startswith("http://") or value.startswith("https://"):
            try:
                hostname = (urlparse(value).hostname or "").lower()
            except Exception:
                hostname = ""
            value = hostname or value
        if value.startswith("www."):
            value = value[4:]
        if value and value not in domains:
            domains.append(value)
    return domains


def resolve_citation_required(goal: dict | None, pack: CapabilityPack | None = None) -> bool:
    """Resolve citation-required behavior with explicit metadata first."""
    explicit = (goal or {}).get("citation_required", None)
    if isinstance(explicit, bool):
        return explicit
    if isinstance(explicit, str):
        lowered = explicit.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
    if pack:
        return bool(pack.citation_required)
    return False


def resolve_research_config(goal: dict | None) -> dict[str, object]:
    """Resolve research execution config for research/monitoring packs."""
    pack = resolve_pack(goal)
    mode = resolve_research_mode(goal, pack)
    scope = resolve_source_scope(goal, pack)
    domains = resolve_source_domains(goal)
    warnings: list[str] = []
    if scope == "domain" and not domains:
        warnings.append("domain scope requested without source_domains; falling back to web")
        scope = "web"
    return {
        "pack_id": pack.id,
        "research_mode": mode,
        "source_scope": scope,
        "source_domains": domains,
        "citation_required": resolve_citation_required(goal, pack),
        "warnings": warnings,
    }
