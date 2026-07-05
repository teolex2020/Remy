"""
Mission intake runtime for Remy v3.

Owns deterministic intake-time classification:
- mission mode
- budget estimate
- risk assessment
- approval requirement
"""

from __future__ import annotations

import time

from ..missions.mission_models import BudgetEstimate, Mission, MissionMode, MissionStatus, RiskLevel


class IntakeRuntime:
    """Normalizes mission acceptance into explicit runtime policy."""

    def __init__(self, audit=None):
        self.audit = audit

    def accept(self, mission: Mission) -> Mission:
        mission.status = MissionStatus.INTAKE
        mission.updated_at = time.time()
        mission.mode = self.classify_mode(mission)
        mission.budget = self.estimate_budget(mission)
        mission.risk = self.assess_risk(mission)
        mission.requires_approval = mission.risk in (RiskLevel.HIGH, RiskLevel.CRITICAL)

        if self.audit is not None:
            self.audit.log_event(
                "mission_accepted",
                f"Accepted mission: {mission.description[:80]}",
                actor="chief",
                mission_id=mission.id,
                details={"mode": mission.mode.value, "risk": mission.risk.value},
            )
        return mission

    def classify_mode(self, mission: Mission) -> MissionMode:
        desc = mission.description.lower()
        if any(w in desc for w in ("research", "analyze", "investigate", "study")):
            return MissionMode.DEEP_RESEARCH
        if any(w in desc for w in ("monitor", "track", "watch", "alert")):
            return MissionMode.CONTINUOUS_MONITORING
        if any(w in desc for w in ("campaign", "promote", "launch")):
            return MissionMode.CAMPAIGN
        if any(w in desc for w in ("improve", "optimize", "fix")):
            return MissionMode.SELF_IMPROVEMENT
        return MissionMode.QUICK_TACTICAL

    def estimate_budget(self, mission: Mission) -> BudgetEstimate:
        mode_budgets = {
            MissionMode.QUICK_TACTICAL: BudgetEstimate(tokens=10_000, cost_usd=0.10, time_sec=120),
            MissionMode.DEEP_RESEARCH: BudgetEstimate(tokens=50_000, cost_usd=0.50, time_sec=600),
            MissionMode.CAMPAIGN: BudgetEstimate(tokens=100_000, cost_usd=1.00, time_sec=1800),
            MissionMode.CONTINUOUS_MONITORING: BudgetEstimate(tokens=5_000, cost_usd=0.05, time_sec=60),
            MissionMode.SELF_IMPROVEMENT: BudgetEstimate(tokens=20_000, cost_usd=0.20, time_sec=300),
        }
        return mode_budgets.get(mission.mode, BudgetEstimate())

    def assess_risk(self, mission: Mission) -> RiskLevel:
        desc = mission.description.lower()
        if any(w in desc for w in ("payment", "money", "transfer", "buy")):
            return RiskLevel.CRITICAL
        if any(w in desc for w in ("publish", "post", "signup", "register")):
            return RiskLevel.HIGH
        if any(w in desc for w in ("browse", "navigate", "scrape")):
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
