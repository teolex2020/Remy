"""
Specialist Agent Registry for Remy v3.

Defines the contract for specialist agents and manages their registration.
Phase 1: wraps v2 capability_packs as specialist definitions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Specialist contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpecialistProfile:
    """Definition of a specialist agent's capabilities and constraints.

    This is the v3 evolution of v2 CapabilityPack.
    """
    id: str                         # researcher, executor, analyst, ...
    label: str                      # Human-readable name
    description: str = ""

    # Execution constraints
    tools: tuple[str, ...] = ()     # Allowed tools
    guardrails: tuple[str, ...] = ()  # Hard rules
    step_budget: int = 10
    timeout_sec: int = 120
    approval_mode: str = "none"     # none, publish, financial, all_clicks

    # Specialization
    domains: tuple[str, ...] = ()   # research, browsing, analysis, coding, ...
    worker_type: str = "generic"    # Maps to v2 worker: browser_worker, research_worker, ...

    # Cost profile
    avg_cost_usd: float = 0.0
    avg_tokens: int = 0

    # Output contract
    expected_output: str = ""       # What this specialist should produce


# ---------------------------------------------------------------------------
# MVP specialists
# ---------------------------------------------------------------------------

RESEARCHER = SpecialistProfile(
    id="researcher",
    label="Researcher",
    description="Source discovery, query expansion, evidence gathering, synthesis",
    tools=("web_search", "extract_content", "http_get", "store", "recall",
           "start_research", "add_research_finding"),
    guardrails=("Cite all sources", "Track source credibility", "Flag contradictions"),
    step_budget=20,
    timeout_sec=180,
    domains=("research", "osint", "market_intelligence"),
    worker_type="research_worker",
    expected_output="Structured findings with sources, confidence, and contradictions",
)

EXECUTOR = SpecialistProfile(
    id="executor",
    label="Executor",
    description="Carry out concrete steps, operate tools, collect artifacts",
    tools=("browse_page", "browser_act", "browser_close", "store", "recall"),
    guardrails=("Verify actions before confirming", "Collect execution evidence"),
    step_budget=15,
    timeout_sec=90,
    domains=("browsing", "signup", "publishing"),
    worker_type="browser_worker",
    expected_output="Execution artifacts and completion evidence",
)

ANALYST = SpecialistProfile(
    id="analyst",
    label="Analyst",
    description="Pattern analysis, cross-source synthesis, recommendations",
    tools=("store", "recall", "web_search", "extract_content"),
    guardrails=("Score confidence on all claims", "Distinguish fact from inference"),
    step_budget=12,
    timeout_sec=120,
    domains=("analysis", "synthesis"),
    worker_type="generic",
    expected_output="Analysis report with confidence scores and recommendations",
)


# ---------------------------------------------------------------------------
# Extended specialists (Phase 3+)
# ---------------------------------------------------------------------------

OSINT_AGENT = SpecialistProfile(
    id="osint",
    label="OSINT / Market Intelligence",
    description="Competitor mapping, market signals, strategic profiling",
    tools=("web_search", "extract_content", "http_get", "store", "recall",
           "start_research", "add_research_finding"),
    guardrails=("Cite all sources", "Track source freshness", "Flag stale data"),
    step_budget=20,
    timeout_sec=180,
    domains=("osint", "market_intelligence", "competitive_analysis"),
    worker_type="research_worker",
    expected_output="Intelligence summary with sources and confidence",
)

BROWSER_OPERATOR = SpecialistProfile(
    id="browser_operator",
    label="Browser Operator",
    description="Web navigation, form interaction, flow completion",
    tools=("browse_page", "browser_act", "browser_close"),
    guardrails=(
        "Never enter payment info",
        "No CAPTCHA/KYC bypass",
        "Verify typed values match intent",
    ),
    step_budget=15,
    timeout_sec=90,
    approval_mode="all_clicks",
    domains=("browsing",),
    worker_type="browser_worker",
    expected_output="Flow completion evidence with screenshots",
)

CODER = SpecialistProfile(
    id="coder",
    label="Coder / Builder",
    description="Code changes, tool generation, debugging, automation",
    tools=("store", "recall"),
    guardrails=("Test before deploying", "No production writes without approval"),
    step_budget=15,
    timeout_sec=120,
    domains=("coding", "automation"),
    worker_type="generic",
    expected_output="Working code or tool with test evidence",
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class SpecialistRegistry:
    """Registry of available specialist agents.

    Phase 1: populated from hardcoded profiles + v2 capability packs.
    """

    def __init__(self):
        self._specialists: dict[str, SpecialistProfile] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register MVP + extended specialists."""
        for spec in (RESEARCHER, EXECUTOR, ANALYST, OSINT_AGENT,
                     BROWSER_OPERATOR, CODER):
            self._specialists[spec.id] = spec

    def get(self, specialist_id: str) -> SpecialistProfile | None:
        return self._specialists.get(specialist_id)

    def resolve(self, goal: dict | None = None) -> SpecialistProfile:
        """Resolve the best specialist for a goal.

        Phase 1: Delegates to v2 resolve_pack() and maps result.
        """
        try:
            from remy.core.capability_packs import resolve_pack
            pack = resolve_pack(goal)
            return self._from_v2_pack(pack)
        except ImportError:
            log.warning("v2 capability_packs not available, using default")
            return ANALYST

    def _from_v2_pack(self, pack) -> SpecialistProfile:
        """Map a v2 CapabilityPack to the closest v3 specialist."""
        pack_id = getattr(pack, "id", "")
        worker = getattr(pack, "worker", "generic")

        # Direct mapping
        mapping = {
            "market_research": "researcher",
            "monitoring": "researcher",
            "signup_operator": "executor",
            "publisher": "executor",
        }
        spec_id = mapping.get(pack_id, "analyst")

        spec = self._specialists.get(spec_id)
        if spec:
            return spec
        return ANALYST

    def register(self, profile: SpecialistProfile):
        self._specialists[profile.id] = profile

    def list_all(self) -> list[SpecialistProfile]:
        return list(self._specialists.values())

    def summary(self) -> list[dict[str, Any]]:
        return [
            {
                "id": s.id,
                "label": s.label,
                "domains": list(s.domains),
                "worker": s.worker_type,
                "budget": s.step_budget,
                "timeout": s.timeout_sec,
            }
            for s in self._specialists.values()
        ]
