"""
Outcome Learner for Remy v3 Self-Improvement.

Learns from execution outcomes:
- What strategies work for which types of goals
- Which specialists perform best for which domains
- Common failure patterns and how to avoid them
- Successful tool sequences to replicate

Stores insights in Aura memory for future recall.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class Insight:
    """A learned insight from execution history."""
    id: str = ""
    category: str = ""        # strategy, failure_pattern, tool_sequence, specialist_fit
    description: str = ""
    confidence: float = 0.0   # 0.0–1.0
    supporting_evidence: int = 0
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def proposal(self) -> str:
        if self.category == "specialist_fit" and "low success rate" in self.description:
            return "Route fewer cycles through this specialist until the strategy or domain fit improves."
        if self.category == "failure_pattern":
            return "Add a runtime guard, fallback path, or explicit operator checkpoint for this blocker."
        if self.category == "cost_efficiency":
            return "Reduce cost exposure through tighter tool scope, cheaper models, or fewer redundant steps."
        if self.category == "quality_pattern":
            return "Increase evidence requirements and verification pressure before external claims are surfaced."
        return "Review this pattern and decide whether it should change routing, policy, or playbooks."

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "description": self.description,
            "confidence": round(self.confidence, 2),
            "supporting_evidence": self.supporting_evidence,
            "tags": list(self.tags),
            "proposal": self.proposal(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class OutcomeLearner:
    """Learns patterns from execution history and stores insights."""

    def __init__(self, min_evidence: int = 3):
        self.min_evidence = min_evidence
        self._insights: list[Insight] = []
        self._outcome_buffer: list[dict[str, Any]] = []
        self._buffer_max = 200
        self._stored_insight_ids: set[str] = set()
        self._stored_policy_ids: set[str] = set()

    def observe_outcome(
        self,
        goal_id: str,
        goal_description: str,
        specialist: str,
        status: str,
        tools_used: list[str] | None = None,
        blocker: str = "",
        duration_ms: int = 0,
        cost_usd: float = 0.0,
        unsupported_observed_claims: int = 0,
    ):
        """Record an execution outcome for learning."""
        self._outcome_buffer.append({
            "goal_id": goal_id,
            "description": goal_description[:100],
            "specialist": specialist,
            "status": status,
            "tools": tools_used or [],
            "blocker": blocker,
            "duration_ms": duration_ms,
            "cost_usd": cost_usd,
            "unsupported_observed_claims": unsupported_observed_claims,
            "timestamp": time.time(),
        })
        if len(self._outcome_buffer) > self._buffer_max:
            self._outcome_buffer = self._outcome_buffer[-self._buffer_max:]

    def analyze(self) -> list[Insight]:
        """Analyze outcome buffer and generate new insights."""
        new_insights = []

        new_insights.extend(self._analyze_specialist_fit())
        new_insights.extend(self._analyze_failure_patterns())
        new_insights.extend(self._analyze_cost_efficiency())
        new_insights.extend(self._analyze_factuality_patterns())

        # Merge with existing (update or add)
        for insight in new_insights:
            existing = self._find_insight(insight.category, insight.description)
            if existing:
                existing.supporting_evidence += 1
                existing.confidence = min(1.0, existing.confidence + 0.1)
                existing.updated_at = time.time()
            else:
                self._insights.append(insight)

        return new_insights

    def _analyze_specialist_fit(self) -> list[Insight]:
        """Which specialists work best for which goal types."""
        insights = []
        specialist_stats: dict[str, dict[str, int]] = {}

        for o in self._outcome_buffer:
            spec = o["specialist"]
            if not spec:
                continue
            if spec not in specialist_stats:
                specialist_stats[spec] = {"success": 0, "total": 0}
            specialist_stats[spec]["total"] += 1
            if o["status"] == "success":
                specialist_stats[spec]["success"] += 1

        for spec, stats in specialist_stats.items():
            if stats["total"] >= self.min_evidence:
                rate = stats["success"] / stats["total"]
                if rate >= 0.7:
                    insights.append(Insight(
                        id=f"fit_{spec}_high",
                        category="specialist_fit",
                        description=f"{spec} has high success rate ({rate:.0%})",
                        confidence=min(0.9, rate),
                        supporting_evidence=stats["total"],
                        tags=["specialist", spec],
                    ))
                elif rate <= 0.3:
                    insights.append(Insight(
                        id=f"fit_{spec}_low",
                        category="specialist_fit",
                        description=f"{spec} has low success rate ({rate:.0%}) — consider alternatives",
                        confidence=min(0.8, 1.0 - rate),
                        supporting_evidence=stats["total"],
                        tags=["specialist", spec, "warning"],
                    ))

        return insights

    def _analyze_failure_patterns(self) -> list[Insight]:
        """Find common failure patterns."""
        insights = []
        blocker_counts: dict[str, int] = {}
        total_failures = 0

        for o in self._outcome_buffer:
            if o["status"] in ("failure", "blocked"):
                total_failures += 1
                if o["blocker"]:
                    blocker_counts[o["blocker"]] = blocker_counts.get(o["blocker"], 0) + 1

        for blocker, count in blocker_counts.items():
            if count >= self.min_evidence:
                insights.append(Insight(
                    id=f"blocker_{blocker}",
                    category="failure_pattern",
                    description=f"Recurring blocker: {blocker} ({count} times)",
                    confidence=min(0.9, count / max(total_failures, 1)),
                    supporting_evidence=count,
                    tags=["blocker", blocker],
                ))

        return insights

    def _analyze_cost_efficiency(self) -> list[Insight]:
        """Find cost efficiency patterns."""
        insights = []
        specialist_costs: dict[str, list[float]] = {}

        for o in self._outcome_buffer:
            spec = o["specialist"]
            if spec and o["cost_usd"] > 0:
                specialist_costs.setdefault(spec, []).append(o["cost_usd"])

        for spec, costs in specialist_costs.items():
            if len(costs) >= self.min_evidence:
                avg = sum(costs) / len(costs)
                if avg > 0.05:  # Expensive
                    insights.append(Insight(
                        id=f"cost_{spec}",
                        category="cost_efficiency",
                        description=f"{spec} avg cost ${avg:.3f}/execution — optimize or limit",
                        confidence=0.7,
                        supporting_evidence=len(costs),
                        tags=["cost", spec],
                    ))

        return insights

    def _analyze_factuality_patterns(self) -> list[Insight]:
        """Find recurring unsupported observed-claim behavior."""
        insights = []
        specialist_counts: dict[str, dict[str, int]] = {}

        for outcome in self._outcome_buffer:
            claims = int(outcome.get("unsupported_observed_claims", 0) or 0)
            if claims <= 0:
                continue
            specialist = outcome["specialist"] or "unknown"
            if specialist not in specialist_counts:
                specialist_counts[specialist] = {"claims": 0, "cycles": 0}
            specialist_counts[specialist]["claims"] += claims
            specialist_counts[specialist]["cycles"] += 1

        for specialist, counts in specialist_counts.items():
            if counts["claims"] >= self.min_evidence:
                insights.append(Insight(
                    id=f"factuality_{specialist}",
                    category="quality_pattern",
                    description=(
                        f"{specialist} produced repeated unsupported observed claims "
                        f"({counts['claims']} across {counts['cycles']} cycle(s))"
                    ),
                    confidence=min(0.9, 0.4 + (counts["claims"] * 0.1)),
                    supporting_evidence=counts["claims"],
                    tags=["factuality", specialist, "verification"],
                ))

        return insights

    def _find_insight(self, category: str, description: str) -> Insight | None:
        for i in self._insights:
            if i.category == category and i.description == description:
                return i
        return None

    def get_insights(self, category: str = "") -> list[Insight]:
        """Get insights, optionally filtered by category."""
        if category:
            return [i for i in self._insights if i.category == category]
        return list(self._insights)

    def reviewable_insights(self, limit: int = 5) -> list[dict[str, Any]]:
        ranked = sorted(
            self._insights,
            key=lambda item: (item.confidence, item.supporting_evidence, item.updated_at),
            reverse=True,
        )
        return [item.to_dict() for item in ranked[:limit]]

    def _first_specific_tag(self, insight: Insight, ignored: set[str]) -> str:
        for tag in insight.tags:
            if tag and tag not in ignored:
                return tag
        return "unknown"

    def _policy_consequence_from_insight(self, insight: Insight) -> dict[str, Any] | None:
        """Translate a reviewable insight into a runtime consequence/policy unit."""
        scope = [
            f"insight:{insight.id}",
            f"category:{insight.category}",
            f"confidence:{insight.confidence:.2f}",
            f"evidence:{insight.supporting_evidence}",
        ]
        provenance = [
            "remy:outcome_learner",
            f"insight:{insight.id}",
        ]
        links = {"insight": insight.id}

        if insight.category == "specialist_fit":
            specialist = self._first_specific_tag(insight, {"specialist", "warning"})
            base = {
                "situation": f"specialist:{specialist}",
                "action": f"route_to:{specialist}",
                "scope": scope + [f"specialist:{specialist}"],
                "provenance": provenance,
                "links": links,
                "namespace": "remy-routing",
            }
            if "low success rate" in insight.description:
                return {
                    **base,
                    "consequence": "REFUTES",
                    "trust": -1,
                    "scope": base["scope"] + ["policy:avoid"],
                }
            if "high success rate" in insight.description:
                return {
                    **base,
                    "consequence": "SUPPORTS",
                    "trust": 1,
                    "scope": base["scope"] + ["policy:prefer"],
                }
            return None

        if insight.category == "quality_pattern":
            specialist = self._first_specific_tag(
                insight,
                {"factuality", "verification", "warning"},
            )
            return {
                "situation": f"specialist:{specialist}",
                "action": "answer_without_evidence",
                "consequence": "REFUTES",
                "trust": -1,
                "scope": scope + [
                    f"specialist:{specialist}",
                    "policy:requires_evidence",
                    "factuality-scar",
                ],
                "provenance": provenance,
                "links": links,
                "namespace": "remy-factuality",
            }

        if insight.category == "failure_pattern":
            blocker = self._first_specific_tag(insight, {"blocker", "warning"})
            return {
                "situation": f"blocker:{blocker}",
                "action": "repeat_without_guard",
                "consequence": "REFUTES",
                "trust": -1,
                "scope": scope + [
                    f"blocker:{blocker}",
                    "policy:avoid",
                    "policy:requires_evidence",
                ],
                "provenance": provenance,
                "links": links,
                "namespace": "remy-policy",
            }

        if insight.category == "cost_efficiency":
            specialist = self._first_specific_tag(insight, {"cost", "warning"})
            return {
                "situation": f"specialist:{specialist}",
                "action": "use_expensive_path",
                "consequence": "REFUTES",
                "trust": -1,
                "scope": scope + [
                    f"specialist:{specialist}",
                    "policy:verify_first",
                    "cost-scar",
                ],
                "provenance": provenance,
                "links": links,
                "namespace": "remy-policy",
            }

        return None

    def store_insights_to_memory(self):
        """Persist insights to Aura memory."""
        try:
            from ..memory.memory_api import get_memory, MemoryClass

            memory = get_memory()
            for insight in self._insights:
                if insight.confidence < 0.5:
                    continue
                if insight.id not in self._stored_insight_ids:
                    memory.store(
                        content=(
                            f"[INSIGHT] [{insight.category}] {insight.description} "
                            f"(confidence: {insight.confidence:.0%}, evidence: {insight.supporting_evidence})"
                        ),
                        tags=["insight", insight.category] + insight.tags,
                        metadata={
                            "insight_id": insight.id,
                            "category": insight.category,
                            "confidence": insight.confidence,
                        },
                        memory_class=MemoryClass.STRATEGIC,
                    )
                    self._stored_insight_ids.add(insight.id)
                if insight.id not in self._stored_policy_ids:
                    consequence = self._policy_consequence_from_insight(insight)
                    if consequence is not None:
                        memory.capture_consequence(**consequence)
                        self._stored_policy_ids.add(insight.id)
        except Exception as e:
            log.debug("Failed to store insights to memory: %s", e)

    def summary(self) -> dict[str, Any]:
        """Summary for observability."""
        return {
            "outcomes_observed": len(self._outcome_buffer),
            "insights_total": len(self._insights),
            "unsupported_observed_claims_total": sum(
                int(o.get("unsupported_observed_claims", 0) or 0)
                for o in self._outcome_buffer
            ),
            "insights_by_category": {
                cat: len([i for i in self._insights if i.category == cat])
                for cat in set(i.category for i in self._insights)
            },
            "top_insights": self.reviewable_insights(3),
        }
